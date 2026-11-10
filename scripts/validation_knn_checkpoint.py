#!/usr/bin/env python3
"""Diagnose exact step1100 on fixed validation prompts before long training."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from osnip_repro.data import file_sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(rankings, token_rows, clean, encrypted, epsilon):
    import torch
    import torch.nn.functional as F
    from evaluate_streamed_knn import score_rows
    rows, counts, rates = score_rows(rankings, token_rows)
    target = torch.tensor([t for row in token_rows for t in row])
    cosine = F.cosine_similarity(clean, encrypted, dim=-1)
    count = len(target)
    per_token = []
    offset = 0
    for prompt, ids in enumerate(token_rows):
        for position, token in enumerate(ids):
            i = offset+position
            hits = (rankings[count+i] == token).nonzero().flatten()
            per_token.append({'prompt_index':prompt, 'token_position':position,
                'target_id':token, 'cosine':float(cosine[i]),
                'abs_cosine_le_epsilon':bool(cosine[i].abs() <= epsilon),
                'target_rank_if_top10':int(hits[0])+1 if len(hits) else None,
                'clean_top10_ids':rankings[i].tolist(),
                'encrypted_top10_ids':rankings[count+i].tolist()})
        offset += len(ids)
    conditional = {}
    for name, mask in [('abs_cosine_le_epsilon', cosine.abs() <= epsilon),
                       ('abs_cosine_gt_epsilon', cosine.abs() > epsilon),
                       ('signed_cosine_le_epsilon', cosine <= epsilon),
                       ('signed_cosine_gt_epsilon', cosine > epsilon),
                       ('negative_cosine', cosine <= 0),
                       ('positive_cosine', cosine > 0)]:
        n = int(mask.sum())
        conditional[name] = {'positions':n, 'rates':{f'top{k}':
            float((rankings[count:][mask,:k] == target[mask,None]).any(-1).float().mean()) if n else None
            for k in (1,5,10)}}
    return {'rows':rows, 'counts':counts, 'rates':rates, 'tokens':per_token,
            'conditional_recovery':conditional, 'mean_cosine':float(cosine.mean())}


def diagnose(checkpoint, output, *, expected_step=1100):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from diagnose_validation_retrieval import streamed_l2_topk
    from evaluate_paper_protocol import knn_bootstrap
    from osnip_repro.checkpointing import load_encoder_state_dict
    from osnip_repro.data import read_text_jsonl
    from osnip_repro.encoder import build_encryptor_from_config
    from osnip_repro.projection import encrypt_embeddings
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(2)
    config_path = checkpoint.parent/'config.json'
    config = json.loads(config_path.read_text())
    model = Path(config['model_path'])
    data = Path(config['valid_jsonl'])
    bound_paths = [checkpoint, config_path, data, model/'model.safetensors',
                   model/'config.json', model/'tokenizer.json', model/'tokenizer_config.json']
    bound = {str(p):file_sha256(p) for p in bound_paths}
    if bound[str(data)] != config['valid_sha256']:
        raise ValueError('validation data disagrees with checkpoint recipe')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('global_step') != expected_step:
        raise ValueError('diagnostic requires exact requested checkpoint step')
    validation = payload.get('metrics', {}).get('validation', {})
    validation_step = payload.get('metrics', {}).get('step')
    has_validation = all(isinstance(validation.get(k), (int,float)) and math.isfinite(validation[k])
                         for k in ['utility', 'privacy', 'ppl_ratio'])
    if payload['config']['valid_sha256'] != config['valid_sha256']:
        raise ValueError('checkpoint validation hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    texts = read_text_jsonl(data, limit=64)
    token_rows = tokenizer(texts, add_special_tokens=True, truncation=True,
                           max_length=config['max_length'], padding=False)['input_ids']
    if len(token_rows) != 64 or any(not row for row in token_rows):
        raise ValueError('expected 64 nonempty fixed validation prompts')
    encoder = build_encryptor_from_config(config).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(checkpoint), expected_config=config))
    wire_dtype = {'float16':torch.float16, 'float32':torch.float32, 'bfloat16':torch.bfloat16}[config['dtype']]
    clean, encrypted = [], []
    with safe_open(model/'model.safetensors', framework='pt', device='cpu') as handle:
        table = handle.get_slice('model.embed_tokens.weight')
        vocabulary_size, width = table.get_shape()
        if width != config['width']:
            raise ValueError('model/encoder width mismatch')
        with torch.inference_mode():
            for index, ids in enumerate(token_rows):
                h = torch.cat([table[t:t+1] for t in ids]).to(wire_dtype)[None]
                rng = torch.Generator(device='cpu').manual_seed(config['seed']+1000003+index)
                # First client key agrees with validation microbatch1 key schedule.
                key = torch.randn(1, config['key_dim'], generator=rng)
                clean.append(h[0].float())
                encrypted.append(encrypt_embeddings(encoder,h,key)[0].float())
        clean, encrypted = torch.cat(clean), torch.cat(encrypted)
        write_json(output.with_suffix('.progress.json'), {'state':'scanning_full_vocabulary',
            'step':expected_step, 'positions':len(clean), 'prompts':64,
            'vocabulary_size':vocabulary_size, 'security_accepted':False})
        rankings = streamed_l2_topk(torch.cat([clean,encrypted]), vocabulary_size,
            lambda a,b:table[a:b], candidate_chunk=2048, query_chunk=256)
    result = summarize(rankings, token_rows, clean, encrypted, config['epsilon'])
    control = result['rates']['clean']['top1'] >= .95
    report = {'format':'osnip_validation_checkpoint_knn_v1',
        'status':'completed' if control else 'invalid_positive_control',
        'scope':'validation_only_diagnostic_not_final_acceptance',
        'security_accepted':False, 'key_attack_performed':False,
        'checkpoint':str(checkpoint), 'checkpoint_sha256':bound[str(checkpoint)],
        'global_step':expected_step, 'data_jsonl':str(data), 'data_sha256':bound[str(data)],
        'samples':64, 'positions':len(clean), 'full_vocabulary':True,
        'vocabulary_size':vocabulary_size, 'metric':'l2',
        'device':'cpu', 'retrieval_dtype':'float32', 'client_wire_dtype':config['dtype'],
        'key_schedule':'validation first key per prompt, seed + 1000003 + prompt_index',
        'tokenization':{'add_special_tokens':True,'max_length':config['max_length'],
                        'truncation':True,'sample_selection':'first64_of_valid_jsonl'},
        'positive_controls_valid':control,
        'saved_validation_metrics':validation if has_validation else None,
        'validation_metrics_step':validation_step if has_validation else None,
        'validation_scope':'saved two-key validation on first128; KNN first key on first64',
        **result, 'prompt_bootstrap':knn_bootstrap(result['rows'],config['seed'],2000),
        'interpretation':('recovery_remains_high' if result['rates']['encrypted']['top10'] >= .95
                          else 'recovery_reduced_requires_full_evaluation'),
        'continuation_policy':'diagnostic only; runner holds for review by default before epoch8; no final safety threshold at step1100',
        'bound_input_sha256':bound,
        'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in
            [Path(__file__),ROOT/'scripts/diagnose_validation_retrieval.py',
             ROOT/'scripts/evaluate_streamed_knn.py',ROOT/'scripts/evaluate_paper_protocol.py',
             ROOT/'src/osnip_repro/projection.py',ROOT/'src/osnip_repro/encoder.py',
             ROOT/'src/osnip_repro/checkpointing.py']},
        'limitations':['CPU FP32 encryptor arithmetic may differ from MPS arithmetic.',
            'Validation diagnostics cannot replace independent KNN/VMA, WikiText, task or multi-seed acceptance.',
            'Recovery outcomes must not silently change the locked training recipe.',
            'Full candidate coverage; floating point ties have unspecified ordering.'],
        'elapsed_seconds':time.monotonic()-started}
    if any(file_sha256(Path(p)) != digest for p,digest in bound.items()):
        raise ValueError('diagnostic inputs changed during evaluation')
    write_json(output,report)
    write_json(output.with_suffix('.progress.json'), {'state':report['status'],
        'global_step':expected_step,'rates':report['rates'],'security_accepted':False})
    print(json.dumps({k:report[k] for k in ['status','global_step','samples','positions','rates','interpretation','elapsed_seconds']}),flush=True)
    if not control:
        raise ValueError('invalid clean positive control')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-step',type=int,default=1100)
    args=ap.parse_args()
    try:
        diagnose(args.checkpoint,args.output,expected_step=args.expected_step)
    except BaseException as exc:
        write_json(args.output.with_suffix('.failure.json'), {
            'state':'failed_no_automatic_retry', 'security_accepted':False,
            'error':type(exc).__name__+': '+str(exc)})
        raise

if __name__=='__main__':main()
