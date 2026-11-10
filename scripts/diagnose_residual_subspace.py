#!/usr/bin/env python3
"""Geometry of the fixed output subspace of the residual MLP; no attacks.

For raw=h+d with d in S, the component of raw perpendicular to S is fixed.
At fixed raw norm rho*||h||, the smallest possible signed cosine is
((1-p)-sqrt(p)*sqrt(rho**2-(1-p)))/rho, where p=||P_S h||²/||h||².
This is a relaxed feasibility bound: tanh/key/utility constraints can only
restrict the reachable set further. It does not certify security.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256,read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config


def output_basis(weight, bias):
    matrix=torch.cat((weight.detach().double(),bias.detach().double()[:,None]),dim=-1)
    u,s,_=torch.linalg.svd(matrix,full_matrices=False)
    tolerance=max(matrix.shape)*torch.finfo(matrix.dtype).eps*float(s[0].detach())
    rank=int((s>tolerance).sum())
    if rank==0:
        return matrix[:,:0],s
    return u[:,:rank],s


def fixed_norm_cosine_lower_bound(projected_energy_fraction, raw_to_h_norm_ratio):
    p=projected_energy_fraction.double().clamp(0,1)
    rho=raw_to_h_norm_ratio.double()
    if (rho<=0).any() or not torch.isfinite(rho).all() or not torch.isfinite(p).all():
        raise ValueError('finite positive norm ratios and finite fractions required')
    b2=1-p
    if (rho.square()<b2-1e-6).any():
        raise ValueError('raw norm smaller than its fixed perpendicular component')
    return ((b2-p.sqrt()*(rho.square()-b2).clamp_min(0).sqrt())/rho).clamp(-1,1)


def describe(values):
    v=values.double().reshape(-1)
    return {'mean':float(v.mean()),'q10':float(v.quantile(.1)),
            'median':float(v.median()),'q90':float(v.quantile(.9)),
            'min':float(v.min()),'max':float(v.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-jsonl',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-samples',type=int,default=128)
    p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--keys-per-prompt',type=int,default=5)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261019)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new report path')
    if min(args.max_samples,args.max_length,args.keys_per_prompt,args.threads)<1:
        raise ValueError('positive limits required')
    torch.set_num_threads(args.threads)
    config=json.loads(args.config.read_text())
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    texts=read_text_jsonl(args.data_jsonl,limit=args.max_samples)
    token_rows=tokenizer(texts,truncation=True,max_length=args.max_length,padding=False)['input_ids']
    unique=sorted({token for row in token_rows for token in row})
    with safe_open(args.model_path/'model.safetensors',framework='pt',device='cpu') as f:
        table=f.get_slice('model.embed_tokens.weight')
        vectors=torch.cat([table[token:token+1].float() for token in unique])
    lookup={token:i for i,token in enumerate(unique)}
    h_rows=[vectors[[lookup[token] for token in row]] for row in token_rows]
    h=torch.cat(h_rows).double()
    h2=h.square().sum(-1)
    rng=torch.Generator().manual_seed(args.seed)
    keys=torch.randn(len(h_rows),args.keys_per_prompt,config['key_dim'],generator=rng)
    items=[]
    for checkpoint in args.checkpoint:
        digest=file_sha256(checkpoint)
        encoder=build_encryptor_from_config(config).eval()
        encoder.load_state_dict(load_encoder_state_dict(str(checkpoint), expected_config=config))
        if file_sha256(checkpoint)!=digest:
            raise RuntimeError('checkpoint changed; freeze it before diagnosis')
        basis,singular=output_basis(encoder.output_projection.weight,encoder.output_projection.bias)
        h_projected=(h@basis)@basis.T
        fraction=h_projected.square().sum(-1)/h2.clamp_min(1e-24)
        raw=[]
        with torch.inference_mode():
            for index,clean in enumerate(h_rows):
                raw.append(encoder(clean[None].expand(args.keys_per_prompt,-1,-1),keys[index]).double())
        raw=torch.cat(raw,dim=1)
        ratio=raw.norm(dim=-1)/h.norm(dim=-1)
        lower=fixed_norm_cosine_lower_bound(fraction[None].expand_as(ratio),ratio)
        cosine=(raw*h).sum(-1)/(raw.norm(dim=-1)*h.norm(dim=-1)).clamp_min(1e-24)
        residual=raw-h
        outside=residual-(residual@basis)@basis.T
        outside_fraction=float(outside.square().sum()/residual.square().sum().clamp_min(1e-24))
        eps=float(config['epsilon'])
        impossible=lower>eps
        actual=cosine.abs()<=eps
        assert not (actual & impossible).any(), 'feasibility bound contradicted observations'
        excess=(lower-cosine).max()
        if float(excess)>1e-5:
            raise RuntimeError('lower bound contradicted observed cosine')
        item={'checkpoint':str(checkpoint.resolve()),'sha256':digest,'output_subspace_rank':basis.shape[1],
              'embedding_width':h.shape[1],'singular_values':describe(singular),
              'h_energy_fraction_in_output_subspace':describe(fraction),
              'raw_to_h_norm_ratio':describe(ratio),'relaxed_min_signed_cosine_at_current_norm':describe(lower),
              'fraction_unreachable_at_current_norm':float(impossible.double().mean()),
              'fraction_reaches_actual_abs_cosine_constraint':float(actual.double().mean()),
              'fraction_relaxed_feasible_but_actual_fails':float((~impossible & ~actual).double().mean()),
              'residual_energy_outside_output_subspace_fraction':outside_fraction,
              'max_bound_minus_observed_cosine':float(excess)}
        items.append(item)
        print(json.dumps(item),flush=True)
    report={'format':'osnip_residual_subspace_diagnostic_v1','scope':'validation_client_geometry_only',
            'security_accepted':False,'key_attack_performed':False,'seed':args.seed,
            'prompts':len(h_rows),'tokens':len(h),'keys_per_prompt':args.keys_per_prompt,
            'max_length':args.max_length,'epsilon':config['epsilon'],
            'model_path':str(args.model_path.resolve()),'data_sha256':file_sha256(args.data_jsonl),
            'config_sha256':file_sha256(args.config),'model_manifest_sha256':file_sha256(args.model_path/'download_manifest.json'),
            'checkpoints':items,'source_hashes':{str(path.relative_to(ROOT)):file_sha256(path) for path in
                [Path(__file__),ROOT/'src/osnip_repro/encoder.py']},
            'limitations':['Output columns plus bias form a relaxed linear subspace; tanh and keys restrict it further.',
                           'Fixed-current-norm feasibility does not rule out solutions at other norms.',
                           'No Llama decoder, task utility, KNN or VMA is evaluated here.',
                           'Aggregate rates count token/key observations, not independent population samples.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
