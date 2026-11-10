#!/usr/bin/env python3
"""Controlled raw_r/projected_z continuation from one frozen checkpoint.

Keep optimizer, scheduler, client RNG, data cursor and LM fixed; discard inherited
best-selection scores because the two arms measure different diversity losses.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch
from osnip_repro.data import file_sha256


def neutral_selection_checkpoint(payload):
    if payload.get('format') != 'osnip_training_checkpoint_v1':
        raise ValueError('full training checkpoint required')
    result = copy.deepcopy(payload)
    result['extra']['best_utility'] = math.inf
    result['extra']['best_pareto'] = math.inf
    return result


def build_commands(config, *, staged, output_root, additional_steps, validation_samples, interval, device, dtype):
    total = int(config['_source_global_step']) + additional_steps
    shared = [sys.executable, str(ROOT/'scripts/train_streaming.py'),
              '--model-path', config['model_path'], '--train-jsonl', config['train_jsonl'],
              '--valid-jsonl', config['valid_jsonl'], '--steps', str(total),
              '--max-valid-samples', str(validation_samples), '--checkpoint-every',str(interval),
              '--evaluate-every',str(interval), '--device',device,'--dtype',dtype]
    # Forward every scalar recipe field the resume guard treats as immutable so
    # both arms match the frozen parent recipe instead of trainer defaults.
    for field in ('batch_size','gradient_accumulation_steps','max_length','key_dim','hidden_dim',
                  'epsilon','diversity_margin','lambda_util','lambda_priv','lambda_div',
                  'warmup_steps','constraint_ramp_steps','loss_warmup_steps',
                  'tau_low','tau_high','learning_rate','weight_decay','seed',
                  'constraint_mode','grad_clip','encryptor_architecture','gate_scope',
                  'gradient_balance','alpha_init','alpha_ramp_steps',
                  'lambda_knn','knn_negative_count','knn_topk','knn_margin','lambda_hidden',
                  'max_train_samples'):
        if config.get(field) is not None:
            shared += ['--'+field.replace('_','-'),str(config[field])]
    if config.get('freeze_alpha'):
        shared += ['--freeze-alpha']
    if config.get('hidden_layers'):
        shared += ['--hidden-layers',*[str(value) for value in config['hidden_layers']]]
    return {space: shared + ['--diversity-space',space,'--resume',str(staged[space]),
                             '--output-dir',str(output_root/space)]
            for space in ('raw_r','projected_z')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--additional-steps',type=int,default=200)
    p.add_argument('--max-valid-samples',type=int,default=None,
                   help='defaults to the parent recipe value; the resume guard refuses divergence')
    p.add_argument('--interval',type=int,default=100)
    p.add_argument('--device',default='mps')
    p.add_argument('--dtype',default='float32')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if min(args.additional_steps,args.interval)<1 or (args.max_valid_samples or 1)<1:
        raise ValueError('positive steps, validation samples and interval required')
    source=args.source_checkpoint.resolve()
    digest=file_sha256(source)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if file_sha256(source)!=digest:
        raise RuntimeError('source changed during loading; use a frozen checkpoint')
    config=copy.deepcopy(payload['config'])
    if args.device!=config['device'] or args.dtype!=config['dtype']:
        raise ValueError('paired continuation must retain source device/dtype for optimizer/RNG fidelity')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError('choose a new paired output directory')
    args.output_root.mkdir(parents=True,exist_ok=True)
    stage_dir=args.output_root/'shared_start'
    staged={space: stage_dir/space/'start.pt' for space in ('raw_r','projected_z')}
    config['_source_global_step']=int(payload['global_step'])
    validation_samples=args.max_valid_samples or int(config.get('max_valid_samples') or 128)
    commands=build_commands(config,staged=staged,output_root=args.output_root,
                            additional_steps=args.additional_steps,validation_samples=validation_samples,
                            interval=args.interval,device=args.device,dtype=args.dtype)
    plan={'format':'osnip_diversity_space_pair_v1','scope':'exploratory_paired_validation_continuation',
          'source_checkpoint':str(source),'source_checkpoint_sha256':digest,
          'source_step':payload['global_step'],'additional_steps':args.additional_steps,
          'source_epoch':payload['extra']['epoch'],'source_cursor':payload['extra']['cursor'],
          'execute':args.execute,'security_accepted':False,'commands':commands,
          'selection_policy':'reset inherited best_utility and best_pareto; select separately within each arm',
          'invariants':['identical encoder/optimizer/scheduler/RNG/client keys and data order at fork',
                        'source device/dtype and all optimizer/objective settings retained except diversity_space',
                        'same validation sample budget and step intervals in both arms'],
          'limitations':['Not a fresh training seed or confirmatory test.','No key attack.','Run sequentially after releasing other full-model jobs.']}
    (args.output_root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if not args.execute:
        return
    # No previous best files are staged: otherwise train_streaming copies metrics
    # selected under a different objective into a new arm.
    stage_dir.mkdir(parents=True, exist_ok=True)
    for space, staged_path in staged.items():
        # Each arm resumes from its own staged checkpoint whose embedded config
        # records that arm's diversity_space: the resume guard treats the field
        # as immutable, so one shared stage could never fork both arms.
        arm_dir = staged_path.parent
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_payload = neutral_selection_checkpoint(payload)
        arm_payload['config'] = dict(arm_payload['config'], diversity_space=space,
                                     phase_transition={'source': str(source), 'source_sha256': digest,
                                                       'changed_fields': ['diversity_space']})
        torch.save(arm_payload, staged_path)
        (arm_dir/'config.json').write_text(json.dumps(arm_payload['config'],indent=2)+'\n')
        (arm_dir/'staging_manifest.json').write_text(json.dumps({
            'source_sha256':digest,'staged_sha256':file_sha256(staged_path),
            'changes':['extra.best_utility=inf','extra.best_pareto=inf','config.diversity_space='+space],
            'encoder_optimizer_scheduler_rng_cursor_unchanged':True},indent=2)+'\n')
    for space,command in commands.items():
        subprocess.run(command,check=True)
        report=json.loads((args.output_root/space/'report.json').read_text())
        if report['steps_completed']!=payload['global_step']+args.additional_steps:
            raise RuntimeError('arm failed to finish the planned training budget')
    comparisons={space:json.loads((args.output_root/space/'report.json').read_text()) for space in commands}
    (args.output_root/'comparison.json').write_text(json.dumps({
        'scope':'validation_only','security_accepted':False,
        'source_checkpoint_sha256':digest,'arms':comparisons},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
