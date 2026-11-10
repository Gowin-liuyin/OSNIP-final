#!/usr/bin/env python3
"""CPU-only fresh privacy geometry control on frozen LM embeddings.

No decoder, utility measurement, attack, or security acceptance is performed.
Train and validation rows are selected independently from the pinned v2 splits.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from osnip_repro.data import read_text_jsonl, file_sha256, TextJsonlIndex
from osnip_repro.encoder import build_encryptor
from osnip_repro.losses import privacy_hinge, diversity_hinge
from osnip_repro.projection import iso_norm_project



class StreamingEmbeddingRows:
    """Read one prompt at a time with a bounded CPU embedding-row LRU."""

    def __init__(self, source, model_path, tokenizer, *, max_length, cache_rows=4096):
        if max_length < 1 or cache_rows < 1:
            raise ValueError('positive max_length and cache_rows required')
        self.index = TextJsonlIndex.build(source)
        self.model_path = Path(model_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_rows = cache_rows
        self.cache = OrderedDict()
        self.observed_indices = set()
        self.reads = 0

    def __len__(self):
        return len(self.index.offsets)

    def __getitem__(self, index):
        text = self.index.read_text(index)
        ids = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)['input_ids']
        if not ids:
            raise ValueError('empty tokenized training prompt')
        rows = []
        with safe_open(self.model_path/'model.safetensors', framework='pt', device='cpu') as handle:
            table = handle.get_slice('model.embed_tokens.weight')
            for token in ids:
                if token in self.cache:
                    row = self.cache.pop(token)
                else:
                    row = table[token:token+1].float().clone()
                self.cache[token] = row
                if len(self.cache) > self.cache_rows:
                    self.cache.popitem(last=False)
                rows.append(row)
        self.observed_indices.add(index)
        self.reads += 1
        return torch.cat(rows)

    def manifest(self):
        return {'pool_records': len(self), 'prompt_reads': self.reads,
                'unique_source_indices_seen': len(self.observed_indices),
                'entire_pool_seen': len(self.observed_indices) == len(self),
                'cache_capacity_rows': self.cache_rows, 'cached_rows': len(self.cache)}


def geometry_loss(encoder, h, k1, k2, *, epsilon, margin):
    raw1, raw2 = encoder(h, k1), encoder(h, k2)
    z1, z2 = iso_norm_project(raw1, h), iso_norm_project(raw2, h)
    privacy = .5 * (privacy_hinge(h, z1, epsilon) + privacy_hinge(h, z2, epsilon))
    diversity = diversity_hinge(z1, z2, margin)
    return privacy + diversity, z1, z2, privacy, diversity


@torch.no_grad()
def evaluate(encoder, rows, keys, *, epsilon, margin):
    privacy_sum = diversity_sum = cosine_sum = 0.
    satisfied = tokens = 0
    norm_error = 0.
    for h, pair in zip(rows, keys):
        _, z1, z2, privacy, diversity = geometry_loss(encoder, h, pair[0], pair[1], epsilon=epsilon, margin=margin)
        n = len(h)
        cosine = torch.stack([torch.nn.functional.cosine_similarity(h, z, dim=-1).abs() for z in (z1, z2)])
        tokens += 2*n
        satisfied += int((cosine <= epsilon).sum())
        cosine_sum += float(cosine.sum())
        privacy_sum += float(privacy)*2*n
        diversity_sum += float(diversity)*2*n
        norm_error = max(norm_error, max(float((z.norm(dim=-1)-h.norm(dim=-1)).abs().max()) for z in (z1,z2)))
    return {'privacy_hinge': privacy_sum/tokens, 'projected_diversity_hinge': diversity_sum/tokens,
            'mean_abs_cosine': cosine_sum/tokens, 'constraint_coverage': satisfied/tokens,
            'token_key_observations': tokens, 'iso_norm_max_abs_error': norm_error}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,default=ROOT/'model/Llama-3.2-1B')
    p.add_argument('--config',type=Path,default=ROOT/'configs/llama32_1b_question_disjoint_v2.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--train-samples',type=int,default=256,help='sampled-mode pool size; unused with --stream-training')
    p.add_argument('--stream-training',action='store_true',help='stream the entire pinned training split without a sample cap')
    p.add_argument('--cache-rows',type=int,default=4096)
    p.add_argument('--valid-samples',type=int,default=128)
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--architecture',default='mlp_residual_v1')
    p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--validation-seed',type=int,help='fix validation sample/key draws across different training seeds; defaults to training seed')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--evaluate-every',type=int,default=100)
    p.add_argument('--gradient-accumulation-steps',type=int,default=1)
    args=p.parse_args()
    if min(args.steps,args.train_samples,args.valid_samples,args.max_length,args.threads,args.evaluate_every,args.gradient_accumulation_steps,args.cache_rows)<1 or args.learning_rate<=0:
        raise ValueError('positive limits required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    if args.validation_seed is None: args.validation_seed=args.seed
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    fixed=json.loads(args.config.read_text()); data=fixed['data']
    if data.get('dataset_policy')!='question_disjoint_v2':raise ValueError('v2 data required')
    folder=Path(data['manifest_path']).parent
    rows_by_split={}; split_hashes={}; selection_indices={}
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    with safe_open(args.model_path/'model.safetensors',framework='pt',device='cpu') as f:
        table=f.get_slice('model.embed_tokens.weight')
        _,width=table.get_shape()
        for name,count in (('train',args.train_samples),('valid',args.valid_samples)):
            source=folder/(name+'.jsonl')
            split_hashes[name]=file_sha256(source)
            if split_hashes[name]!=data[name+'_sha256']:raise ValueError('split hash mismatch')
            if name=='train' and args.stream_training:
                rows_by_split[name]=StreamingEmbeddingRows(source,args.model_path,tokenizer,max_length=args.max_length,cache_rows=args.cache_rows)
                continue
            pool=read_text_jsonl(source)
            selected=random.Random(args.validation_seed+1 if name=='valid' else args.seed).sample(range(len(pool)),count)
            selection_indices[name]=selected
            ids=tokenizer([pool[i] for i in selected],truncation=True,max_length=args.max_length,padding=False)['input_ids']
            unique=sorted({token for row in ids for token in row})
            if any(not row for row in ids):raise ValueError('empty tokenized prompt')
            vectors=torch.cat([table[token:token+1].float() for token in unique])
            index={token:i for i,token in enumerate(unique)}
            rows_by_split[name]=[vectors[[index[token] for token in row]].clone() for row in ids]
    encoder=build_encryptor(width,key_dim=32,hidden_dim=256,architecture=args.architecture)
    initial={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=args.learning_rate,weight_decay=0.)
    rng=torch.Generator().manual_seed(args.seed+190734863)
    validation_keys=torch.randn(len(rows_by_split['valid']),2,32,generator=torch.Generator().manual_seed(args.validation_seed+1000003))
    epsilon=fixed['objective']['epsilon']; margin=fixed['objective']['diversity_margin']
    baseline=evaluate(encoder,rows_by_split['valid'],validation_keys,epsilon=epsilon,margin=margin)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run_config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    run_config.update(scope='client_geometry_optimization_only',split_sha256=split_hashes,
                      key_dim=32,hidden_dim=256,width=width,epsilon=epsilon,diversity_margin=margin,
                      diversity_space='projected_z',fresh_initialization=True,
                      key_schedule='independent_pair_per_prompt_shared_across_tokens',security_accepted=False,
                      training_pool_records=len(rows_by_split['train']),
                      training_pool_policy='full_split_streaming' if args.stream_training else 'sampled_subset_repeated',
                      sampled_source_indices=selection_indices,validation_key_seed=args.validation_seed+1000003)
    (args.output_dir/'config.json').write_text(json.dumps(run_config,indent=2)+'\n')
    (args.output_dir/'initial_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    order=[]; epoch=cursor=0; started=time.monotonic()
    with (args.output_dir/'progress.jsonl').open('w') as out:
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            privacy_sum = diversity_sum = 0.
            for _ in range(args.gradient_accumulation_steps):
                if cursor>=len(order):
                    order=list(range(len(rows_by_split['train'])))
                    random.Random(args.seed+epoch).shuffle(order);epoch+=1;cursor=0
                h=rows_by_split['train'][order[cursor]];cursor+=1
                keys=torch.randn(2,32,generator=rng)
                loss,_,_,privacy,diversity=geometry_loss(encoder,h,keys[0],keys[1],epsilon=epsilon,margin=margin)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                (loss/args.gradient_accumulation_steps).backward()
                privacy_sum+=float(privacy.detach())/args.gradient_accumulation_steps
                diversity_sum+=float(diversity.detach())/args.gradient_accumulation_steps
            grad=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
            optimizer.step()
            record={'step':step,'privacy':privacy_sum,'diversity':diversity_sum,
                    'gradient_norm_before_clip':float(grad),'elapsed_seconds':time.monotonic()-started}
            if step%args.evaluate_every==0 or step==args.steps:
                record['validation']=evaluate(encoder,rows_by_split['valid'],validation_keys,epsilon=epsilon,margin=margin)
                print(json.dumps(record),flush=True)
            out.write(json.dumps(record)+'\n');out.flush()
    final=record['validation']
    torch.save({'encoder':encoder.state_dict(),'config':run_config,'scope':'geometry_control_only'},args.output_dir/'encoder.pt')
    report={'format':'osnip_privacy_geometry_control_v1','status':'completed_diagnostic_only',
            'baseline_validation':baseline,'final_validation':final,'steps':args.steps,
            'parameters_changed':any(not torch.equal(initial[k],v) for k,v in encoder.state_dict().items()),
            'training_data_exposure':rows_by_split['train'].manifest() if args.stream_training else {
                'pool_records':len(rows_by_split['train']), 'prompt_reads':args.steps*args.gradient_accumulation_steps,
                'scope':'sampled_subset_not_full_split'},
            'source_hashes':{str(x.relative_to(ROOT)):file_sha256(x) for x in (Path(__file__),ROOT/'src/osnip_repro/encoder.py',ROOT/'src/osnip_repro/losses.py',ROOT/'src/osnip_repro/projection.py',ROOT/'src/osnip_repro/data.py')},
            'security_accepted':False,'key_attack_performed':False,
            'limitations':['No decoder, KL, task utility or attack was computed.',
                'Validation uses a small v2 subset; training pool and actual exposure are recorded separately.',
                'No decoder-backed Gate 3 or formal joint run was performed.',
                'This encoder checkpoint cannot resume into train_streaming.py.',
                'Loss decrease is an optimization diagnostic, not privacy acceptance.']}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
