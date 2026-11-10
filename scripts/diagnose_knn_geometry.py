#!/usr/bin/env python3
"""Compare learned OSNIP KNN against per-token cosine-matched random geometry."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from diagnose_validation_retrieval import streamed_l2_topk
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256,read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.projection import encrypt_embeddings


def matched_orthogonal(h,z,*,seed,target_cosine=None):
    """Preserve each h norm and match cos(h,z) with a random orthogonal direction."""
    if h.shape!=z.shape or h.ndim!=2 or not torch.isfinite(h).all() or not torch.isfinite(z).all():
        raise ValueError('h and z must be aligned finite matrices')
    hf,zf=h.double(),z.double()
    hn=hf.norm(dim=-1,keepdim=True)
    if (hn<=0).any():raise ValueError('clean embeddings must have nonzero norm')
    hu=hf/hn
    cos=F.cosine_similarity(hf,zf,dim=-1).clamp(-1+1e-12,1-1e-12)
    if target_cosine is not None:
        value=torch.as_tensor(target_cosine,dtype=torch.float64).expand_as(cos)
        if not torch.isfinite(value).all() or (value.abs()>=1).any():
            raise ValueError('target cosine must be finite and strictly between -1 and 1')
        cos=value
    rng=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(hf.shape,generator=rng,dtype=torch.float64)
    orth=noise-(noise*hu).sum(-1,keepdim=True)*hu
    on=orth.norm(dim=-1,keepdim=True)
    if (on<=1e-15).any():raise FloatingPointError('random orthogonal direction degenerated')
    orth=orth/on
    synthetic=hn*(cos[:,None]*hu+torch.sqrt((1-cos.square()).clamp_min(0))[:,None]*orth)
    return synthetic.to(h.dtype),cos.float()


def report_rankings(rankings,targets):
    n=len(targets)
    names=['clean','osnip','matched_random_orthogonal','random_orthogonal_cosine_0_3','random_orthogonal_cosine_0']
    out={}
    for i,name in enumerate(names):
        q=rankings[i*n:(i+1)*n]
        out[name]={f'top{k}':float((q[:,:k]==targets[:,None]).any(-1).float().mean()) for k in (1,5,10)}
    return out


def run(checkpoint,output,*,expected_step=1100,samples=64,seed=20261024):
    checkpoint,output=Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);started=time.monotonic()
    config_path=checkpoint.parent/'config.json';config=json.loads(config_path.read_text())
    model=Path(config['model_path']);data=Path(config['valid_jsonl'])
    paths=[checkpoint,config_path,data,model/'model.safetensors',model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json']
    bound={str(p):file_sha256(p) for p in paths}
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('global_step')!=expected_step:raise ValueError('unexpected checkpoint step')
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    texts=read_text_jsonl(data,limit=samples)
    token_rows=tokenizer(texts,add_special_tokens=True,truncation=True,max_length=config['max_length'],padding=False)['input_ids']
    if len(token_rows)!=samples or any(not x for x in token_rows):raise ValueError('invalid fixed validation sample')
    encoder=build_encryptor_from_config(config).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(checkpoint),expected_config=config))
    dtype={'float16':torch.float16,'float32':torch.float32,'bfloat16':torch.bfloat16}[config['dtype']]
    hs=[];zs=[]
    with safe_open(model/'model.safetensors',framework='pt',device='cpu') as handle:
        table=handle.get_slice('model.embed_tokens.weight');vocab,width=table.get_shape()
        if width!=config['width']:raise ValueError('embedding width mismatch')
        with torch.inference_mode():
            for i,row in enumerate(token_rows):
                h=torch.cat([table[t:t+1] for t in row]).to(dtype)[None]
                rng=torch.Generator(device='cpu').manual_seed(config['seed']+1000003+i)
                key=torch.randn(1,config['key_dim'],generator=rng)
                hs.append(h[0].float());zs.append(encrypt_embeddings(encoder,h,key)[0].float())
        h,z=torch.cat(hs),torch.cat(zs);target=torch.tensor([t for row in token_rows for t in row])
        synthetic,cos=matched_orthogonal(h,z,seed=seed)
        at_epsilon,_=matched_orthogonal(h,z,seed=seed+1,target_cosine=0.3)
        zero_cosine,_=matched_orthogonal(h,z,seed=seed+2,target_cosine=0.0)
        if not torch.allclose(torch.linalg.vector_norm(h.double(),dim=-1),torch.linalg.vector_norm(synthetic.double(),dim=-1),rtol=1e-6,atol=1e-5):
            raise AssertionError('synthetic norm mismatch')
        actual_cos=F.cosine_similarity(h.double(),z.double(),dim=-1)
        synth_cos=F.cosine_similarity(h.double(),synthetic.double(),dim=-1)
        if not torch.allclose(actual_cos,synth_cos,rtol=1e-6,atol=1e-6):raise AssertionError('synthetic cosine mismatch')
        queries=torch.cat([h,z,synthetic,at_epsilon,zero_cosine])
        rankings=streamed_l2_topk(queries,vocab,lambda a,b:table[a:b],candidate_chunk=2048,query_chunk=256)
    rates=report_rankings(rankings,target)
    controls={'norm_max_abs_error':float((h.double().norm(dim=-1)-synthetic.double().norm(dim=-1)).abs().max()),
        'cosine_max_abs_error':float((actual_cos-synth_cos).abs().max()),
        'matched_mean_cosine':float(cos.mean()),'learned_mean_cosine':float(actual_cos.mean()),
        'cosine_per_token_exactly_matched':True,
        'epsilon_control_cosine':0.3, 'zero_cosine_control_cosine':0.0,
        'all_vocabulary_candidates':True}
    result={'format':'osnip_knn_geometry_diagnostic_v1','status':'completed',
        'scope':'validation_only_geometric_control_not_security_acceptance','security_accepted':False,
        'key_attack_performed':False,'checkpoint':str(checkpoint),'checkpoint_sha256':bound[str(checkpoint)],
        'global_step':expected_step,'data_jsonl':str(data),'data_sha256':bound[str(data)],
        'samples':samples,'positions':len(target),'vocabulary_size':vocab,'full_vocabulary':True,
        'metric':'l2','seed':seed,'rates':rates,'controls':controls,
        'interpretation':{'matched_actual_angle':'generic_matched_geometry_also_preserves_recovery'
            if rates['matched_random_orthogonal']['top10']>=.95 else 'matched_random_recovery_reduced',
            'at_epsilon':'top10_recovery_high' if rates['random_orthogonal_cosine_0_3']['top10']>=.95
            else 'top10_recovery_reduced',
            'zero_cosine':'top10_recovery_high' if rates['random_orthogonal_cosine_0']['top10']>=.95
            else 'top10_recovery_reduced'},
        'limitations':['Random control matches per-token cosine and norm, not OSNIP tokenwise perturbation correlations.',
            'Validation prefix is diagnostic and not independent attack confirmation.',
            'Does not assess VMA, language-model utility, formal paper reproduction or key attacks.'],
        'bound_input_sha256':bound,'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in
            [Path(__file__),ROOT/'scripts/diagnose_validation_retrieval.py',ROOT/'src/osnip_repro/projection.py',ROOT/'src/osnip_repro/encoder.py']},
        'elapsed_seconds':time.monotonic()-started}
    if any(file_sha256(Path(p))!=digest for p,digest in bound.items()):raise ValueError('input changed during diagnostic')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','positions','vocabulary_size','rates','controls','interpretation','elapsed_seconds']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--expected-step',type=int,default=1100);ap.add_argument('--samples',type=int,default=64);ap.add_argument('--seed',type=int,default=20261024);a=ap.parse_args();run(a.checkpoint,a.output,expected_step=a.expected_step,samples=a.samples,seed=a.seed)
if __name__=='__main__':main()
