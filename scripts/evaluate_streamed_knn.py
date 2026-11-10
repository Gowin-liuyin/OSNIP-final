#!/usr/bin/env python3
"""Independent fixed-set L2 KNN without loading the frozen decoder.

Tokenization, one CPU client key per prompt, and token-weighted prompt bootstrap
follow evaluate_paper_protocol.py. Vocabulary rows are streamed from safetensors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from diagnose_validation_retrieval import streamed_l2_topk
from evaluate_paper_protocol import knn_bootstrap
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256, read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.projection import iso_norm_project


@torch.no_grad()
def prepare_queries(token_rows, read_rows, encoder, *, key_dim, seed):
    """Match the existing CPU paper KNN client-key schedule exactly."""
    if not token_rows or any(not row for row in token_rows):
        raise ValueError('nonempty prompt token lists required')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    clean, encrypted = [], []
    for row in token_rows:
        h = torch.cat([read_rows(token, token+1).float() for token in row])
        key = torch.randn(key_dim, generator=rng, device='cpu')
        clean.append(h)
        encrypted.append(iso_norm_project(encoder(h[None], key), h[None])[0])
    return torch.cat(clean), torch.cat(encrypted)


def score_rows(rankings, token_rows):
    positions = sum(map(len,token_rows))
    if rankings.ndim != 2 or rankings.shape != (2*positions,10):
        raise ValueError('expected clean and encrypted global top10 rankings')
    rows = []
    offset = 0
    for index, ids in enumerate(token_rows):
        targets = torch.tensor(ids)
        row = {'index':index,'positions':len(ids)}
        for side, begin in [('clean',offset),('encrypted',positions+offset)]:
            ranked = rankings[begin:begin+len(ids)]
            row[side] = {f'top{k}':int((ranked[:,:k] == targets[:,None]).any(-1).sum()) for k in (1,5,10)}
        rows.append(row)
        offset += len(ids)
    counts = {side:{name:sum(row[side][name] for row in rows) for name in ('top1','top5','top10')}
              for side in ('clean','encrypted')}
    rates = {side:{name:count/positions for name,count in hits.items()} for side,hits in counts.items()}
    return rows,counts,rates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path',type=Path,default=ROOT/'model/Llama-3.2-1B')
    ap.add_argument('--knn-jsonl',type=Path,default=ROOT/'data/paper_attack_sets/knn.jsonl')
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-samples',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--bootstrap-samples',type=int,default=2000)
    ap.add_argument('--candidate-chunk',type=int,default=1024)
    ap.add_argument('--query-chunk',type=int,default=128)
    ap.add_argument('--keys-per-prompt',type=int,default=1,
        help='independent client keys per prompt; the paper fixes five for VMA and is silent for KNN')
    args = ap.parse_args()
    if min(args.max_samples,args.threads,args.candidate_chunk,args.query_chunk,args.keys_per_prompt)<1 or args.bootstrap_samples<2:
        raise ValueError('positive limits and at least two bootstrap draws required')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    config_path = args.checkpoint.parent/'config.json'
    config = json.loads(config_path.read_text())
    if Path(config['model_path']).resolve() != args.model_path.resolve():
        raise ValueError('checkpoint and model paths disagree')
    if config.get('architecture',config.get('encryptor_architecture','mlp_residual_v1')) != config.get('encryptor_architecture','mlp_residual_v1'):
        raise ValueError('ambiguous encoder architecture; provide its audited training config')
    bound = {str(p.resolve()):file_sha256(p) for p in [args.checkpoint,config_path,args.knn_jsonl,args.model_path/'config.json',args.model_path/'model.safetensors',args.model_path/'tokenizer.json']}
    texts = read_text_jsonl(args.knn_jsonl,limit=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    token_rows = [tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'] for text in texts]
    observed_rows = [ids for ids in token_rows for _ in range(args.keys_per_prompt)]
    encoder = build_encryptor_from_config(config).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), expected_config=config))
    with safe_open(args.model_path/'model.safetensors',framework='pt',device='cpu') as handle:
        table = handle.get_slice('model.embed_tokens.weight')
        vocabulary_size,width = table.get_shape()
        if width != config['width']:raise ValueError('model and encoder widths disagree')
        clean,encrypted = prepare_queries(observed_rows,lambda a,b:table[a:b],encoder,key_dim=config['key_dim'],seed=args.seed)
        print(json.dumps({'stage':'full_vocabulary_scan','samples':len(texts),'positions':len(clean),'vocabulary_size':vocabulary_size}),flush=True)
        rankings = streamed_l2_topk(torch.cat([clean,encrypted]),vocabulary_size,lambda a,b:table[a:b],
                                    candidate_chunk=args.candidate_chunk,query_chunk=args.query_chunk)
    rows,counts,rates = score_rows(rankings,observed_rows)
    if rates['clean']['top1'] < .99:
        raise ValueError(f"clean KNN positive control top-1 {rates['clean']['top1']} below 0.99; streamed attack pipeline invalid")
    if any(file_sha256(Path(path)) != digest for path,digest in bound.items()):
        raise ValueError('evaluation inputs changed during scan')
    report = {'format':'osnip_knn_report_v2','status':'completed_knn_only','paper_result':False,
        'security_accepted':False,'key_attack_performed':False,'threat_model':'passive_no_key_semi_honest_server',
        'model_path':str(args.model_path.resolve()),'checkpoint':str(args.checkpoint.resolve()),
        'checkpoint_sha256':bound[str(args.checkpoint.resolve())],'data_sha256':bound[str(args.knn_jsonl.resolve())],
        'samples':len(rows),'prompts':len(texts),'keys_per_prompt':args.keys_per_prompt,
        'positions':sum(map(len,observed_rows)),'counts':counts,'rates':rates,'rows':rows,
        'prompt_bootstrap':knn_bootstrap(rows,args.seed,args.bootstrap_samples),'bootstrap_samples':args.bootstrap_samples,
        'metric':'l2','full_vocabulary':True,'vocabulary_size':vocabulary_size,'seed':args.seed,'dtype':'float32','device':'cpu',
        'keys_per_prompt_reported':args.keys_per_prompt,
        'headline_aggregation':'token_micro over prompt-key observations; prompt-level bootstrap in prompt_bootstrap',
        'positive_controls_valid':rates['clean']['top1']>=.95,'elapsed_seconds':time.monotonic()-started,
        'tokenization':{'add_special_tokens':False,'truncation':False},'key_schedule':f'{args.keys_per_prompt} independent CPU-generator keys per prompt, shared across tokens',
        'bound_input_sha256':bound,
        'source_sha256':{name:file_sha256(ROOT/name) for name in ['scripts/evaluate_streamed_knn.py','scripts/diagnose_validation_retrieval.py','scripts/evaluate_paper_protocol.py','src/osnip_repro/attacks.py','src/osnip_repro/encoder.py','src/osnip_repro/projection.py','src/osnip_repro/checkpointing.py','src/osnip_repro/data.py']},
        'limitations':['KNN only; utility, VMA and joint acceptance remain separate.',
            'Float32 complete candidate coverage with unspecified tie ordering.',
            'Single training seed and client draw seed; fixed attack set has been used for prior exploration.',
            'CPU report cannot be reused as an identical MPS client-key draw history.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','samples','positions','rates','positive_controls_valid','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__': main()
