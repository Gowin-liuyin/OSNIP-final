#!/usr/bin/env python3
"""Short non-residual R_phi(h,k) probe; engineering diagnostic only."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from torch import nn
from train_streaming import run_objective, evaluate, make_epoch_order, next_batch
from osnip_repro.data import TextJsonlIndex, file_sha256
from osnip_repro.modeling import load_causal_lm

class PlainKeyMLP(nn.Module):
    def __init__(self,width,key_dim,hidden_dim,init_scale=1e-2):
        super().__init__(); self.width=int(width); self.key_dim=int(key_dim); self.hidden_dim=int(hidden_dim)
        self.input_projection=nn.Linear(width,hidden_dim)
        self.key_projection=nn.Linear(key_dim,hidden_dim,bias=False)
        self.output_projection=nn.Linear(hidden_dim,width)
        nn.init.xavier_uniform_(self.output_projection.weight,gain=init_scale); nn.init.zeros_(self.output_projection.bias)
    def _key(self,h,key):
        if key.ndim==1:return key.expand(*h.shape[:-1],-1)
        if key.shape==h.shape[:-1]+(self.key_dim,):return key
        if h.ndim>=3 and key.shape==h.shape[:-2]+(self.key_dim,):return key.unsqueeze(-2).expand(*h.shape[:-1],-1)
        raise ValueError('invalid key shape')
    def forward(self,h,key,attention_mask=None):
        return self.output_projection(torch.tanh(self.input_projection(h)+self.key_projection(self._key(h,key))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-path',type=Path,required=True);ap.add_argument('--train-jsonl',type=Path,required=True);ap.add_argument('--valid-jsonl',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--steps',type=int,default=100);ap.add_argument('--max-train-samples',type=int,default=800);ap.add_argument('--max-valid-samples',type=int,default=128);ap.add_argument('--device',default='mps');ap.add_argument('--dtype',default='float32');ap.add_argument('--seed',type=int,default=20261019);args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    tokenizer,model,device=load_causal_lm(str(args.model_path),device=args.device,dtype=args.dtype)
    train=TextJsonlIndex.build(args.train_jsonl,limit=args.max_train_samples);valid=TextJsonlIndex.build(args.valid_jsonl,limit=args.max_valid_samples)
    width=int(model.get_input_embeddings().embedding_dim); enc=PlainKeyMLP(width,32,256).to(device)
    opt=torch.optim.AdamW(enc.parameters(),lr=1e-5,weight_decay=0.0); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,(step+1)/100))
    ns=argparse.Namespace(model_path=args.model_path,train_jsonl=args.train_jsonl,valid_jsonl=args.valid_jsonl,output_dir=args.output_dir,steps=args.steps,batch_size=1,gradient_accumulation_steps=8,max_length=128,max_valid_samples=args.max_valid_samples,key_dim=32,hidden_dim=256,encryptor_architecture='plain_mlp_probe',epsilon=.3,diversity_margin=1.,diversity_space='projected_z',lambda_util=1.,lambda_priv=1.,lambda_div=1.,warmup_steps=100,tau_low=.05,tau_high=.2,learning_rate=1e-5,weight_decay=0.,device=args.device,dtype=args.dtype,seed=args.seed,constraint_mode='gated',grad_clip=1.)
    gen=torch.Generator(device='cpu').manual_seed(args.seed+190734863);order=make_epoch_order(len(train.offsets),args.seed,0);cursor=0;started=time.time();progress=[]
    for step in range(args.steps):
        enc.train();opt.zero_grad(set_to_none=True);acc={}
        for _ in range(8):
            if cursor>=len(order):cursor=0;order=make_epoch_order(len(train.offsets),args.seed,step//len(order)+1)
            texts,cursor=next_batch(train,order,cursor,1);losses,stats=run_objective(model,enc,tokenizer,texts,ns,device,training=True,step=step+1,key_generator=gen);(losses['total']/8).backward()
            for k,v in stats.items():acc[k]=acc.get(k,0)+v/8
        norm=float(torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0));opt.step();sched.step();row={'step':step+1,'cursor':cursor,'lr':float(opt.param_groups[0]['lr']),'gradient_norm_before_clip':norm,'elapsed_seconds':time.time()-started};row.update(acc)
        if (step+1)%100==0 or step+1==args.steps:
            row['validation']=evaluate(model,enc,tokenizer,valid,ns,device,step+1)
        progress.append(row);(args.output_dir/'train_progress.jsonl').write_text('\n'.join(json.dumps(x) for x in progress)+'\n')
    torch.save(enc.state_dict(),args.output_dir/'encryptor.pt');report={'format':'osnip_plain_mlp_probe_report_v1','status':'completed_with_limits','paper_result':False,'security_accepted':False,'key_attack_performed':False,'architecture':'plain_mlp_no_residual','steps_completed':args.steps,'train_records':len(train.offsets),'valid_records':len(valid.offsets),'last_validation':progress[-1].get('validation'),'model_path':str(args.model_path.resolve()),'train_sha256':file_sha256(args.train_jsonl),'valid_sha256':file_sha256(args.valid_jsonl),'source_hashes':{'scripts/run_plain_mlp_probe.py':file_sha256(Path(__file__))},'limitations':['Short 100-step engineering probe; no security or convergence claim.','No formal KNN/VMA and no key attack.','This is a structural probe, not the undisclosed author architecture.']};(args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
