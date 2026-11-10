#!/usr/bin/env python3
"""Generate or execute the formal question-disjoint v2 training matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from osnip_repro.checkpointing import require_locked_recipe


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',type=Path,default=ROOT/'model/Llama-3.2-1B')
    p.add_argument('--config',type=Path,default=ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json')
    p.add_argument('--data-manifest',type=Path,default=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json')
    p.add_argument('--output-root',type=Path,default=ROOT/'reports/llama32_1b/question_disjoint_v2_matrix')
    p.add_argument('--pilot-report',type=Path,default=ROOT/'reports/llama32_1b/question_disjoint_v2_pilot/report.json')
    p.add_argument('--execute',action='store_true')
    return p.parse_args()


def build_plan(args):
    cfg=json.loads(args.config.read_text()); require_locked_recipe(cfg); data=cfg['data']; defaults=cfg['formal_training_defaults']
    if data.get('dataset_policy')!='question_disjoint_v2' or data.get('fresh_training_required') is not True:
        raise ValueError('matrix requires question-disjoint v2 fresh-training policy')
    if Path(data['manifest_path']).resolve()!=args.data_manifest.resolve():
        raise ValueError('config and data manifest differ')
    train=args.data_manifest.parent/'train.jsonl'; valid=args.data_manifest.parent/'valid.jsonl'
    variants=cfg['matrix_variants']; requested_architectures=cfg['matrix_architectures']; seeds=cfg['matrix_seeds']
    identity=cfg.get('architecture_identity',{})
    aliases=identity.get('aliases',{})
    if aliases:
        digest=hashlib.sha256((ROOT/'src/osnip_repro/encoder.py').read_bytes()).hexdigest()
        if digest!=identity.get('encoder_source_sha256'):
            raise ValueError('architecture alias audit is stale after encoder source change')
        if any(alias==canonical or canonical in aliases or canonical not in requested_architectures
               for alias,canonical in aliases.items()):
            raise ValueError('architecture aliases require a distinct canonical label in the selected matrix')
    architectures=list(dict.fromkeys(aliases.get(name,name) for name in requested_architectures))
    commands=[]
    for variant in variants:
      for arch in architectures:
       for seed in seeds:
        out=args.output_root/f"{variant['name']}__{arch}_seed{seed}"
        if variant['name']=='two_phase_diagnostic':
          cmd=[sys.executable,str(ROOT/'scripts/run_two_phase_diagnostic.py'),'--execute',
               '--model-path',str(args.model_path),'--train-jsonl',str(train),'--valid-jsonl',str(valid),
               '--output-dir',str(out),'--phase1-steps',str(defaults['two_phase_phase1_steps']),
               '--total-steps',str(defaults['steps']),'--batch-size',str(defaults['batch_size']),
               '--gradient-accumulation-steps',str(defaults['gradient_accumulation_steps']),
               '--max-length',str(defaults['max_length']),'--max-valid-samples','128',
               '--key-dim',str(defaults['key_dim']),'--hidden-dim',str(defaults['hidden_dim']),
               '--tau-low',str(defaults['tau_low']),'--tau-high',str(defaults['tau_high']),
               '--learning-rate',str(defaults['learning_rate']),'--warmup-steps',str(defaults['warmup_steps']),
               '--epsilon',str(defaults['epsilon']),'--diversity-margin',str(defaults['diversity_margin']),
               '--grad-clip',str(defaults['grad_clip']),'--checkpoint-every',str(defaults['checkpoint_every']),
               '--evaluate-every',str(defaults['evaluate_every']),'--encryptor-architecture',arch,
               '--seed',str(seed),'--device',str(defaults['device']),'--dtype',str(defaults['dtype']),
               '--diversity-space',str(defaults['diversity_space'])]
        else:
          cmd=[sys.executable,str(ROOT/'scripts/train_streaming.py'),
               '--model-path',str(args.model_path),'--train-jsonl',str(train),'--valid-jsonl',str(valid),
               '--output-dir',str(out),'--steps',str(defaults['steps']),'--batch-size',str(defaults['batch_size']),
               '--gradient-accumulation-steps',str(defaults['gradient_accumulation_steps']),
               '--max-length',str(defaults['max_length']),'--max-valid-samples','128',
               '--key-dim',str(defaults['key_dim']),'--hidden-dim',str(defaults['hidden_dim']),
               '--learning-rate',str(defaults['learning_rate']),'--warmup-steps',str(defaults['warmup_steps']),
               '--tau-low',str(defaults['tau_low']),'--tau-high',str(defaults['tau_high']),
               '--epsilon',str(defaults['epsilon']),'--diversity-margin',str(defaults['diversity_margin']),
               '--diversity-space',str(defaults['diversity_space']),'--grad-clip',str(defaults['grad_clip']),
               '--checkpoint-every',str(defaults['checkpoint_every']),'--evaluate-every',str(defaults['evaluate_every']),
               '--loss-warmup-steps',str(defaults['loss_warmup_steps']),
               '--constraint-ramp-steps',str(defaults['constraint_ramp_steps']),
               '--gate-scope',str(defaults['gate_scope']),'--weight-decay',str(defaults['weight_decay']),
               '--gradient-balance',str(defaults['gradient_balance']),
               '--alpha-init',str(defaults['alpha_init']),'--alpha-ramp-steps',str(defaults['alpha_ramp_steps']),
               '--constraint-mode',str(variant['constraint_mode']),'--lambda-util',str(variant['lambda_util']),
               '--lambda-priv',str(variant['lambda_priv']),'--lambda-div',str(variant['lambda_div']),
               '--encryptor-architecture',arch,'--seed',str(seed),'--device',str(defaults['device']),'--dtype',str(defaults['dtype'])]
        commands.append({'variant':variant['name'],'architecture':arch,'seed':seed,'output_dir':str(out),'command':cmd})
    gate0=args.output_root/'gate0'
    plan={'format':'osnip_question_disjoint_matrix_plan_v1','config':str(args.config.resolve()),
          'data_manifest':str(args.data_manifest.resolve()),'train_jsonl':str(train.resolve()),'valid_jsonl':str(valid.resolve()),
          'output_root':str(args.output_root.resolve()),'pilot_report':str(args.pilot_report.resolve()),
          'command_count':len(commands),'variants':[v['name'] for v in variants],'architectures':architectures,'seeds':seeds,
          'requested_architectures':requested_architectures,'architecture_aliases':aliases,
          'architecture_identity':identity,'unique_architecture_count':len(architectures),
          'formal_steps':defaults['steps'],'fresh_initialization_required':True,'resume_allowed':False,
          'gate0_command':[sys.executable,str(ROOT/'scripts/run_gate0.py'),'--model-path',str(args.model_path),'--output-dir',str(gate0),'--config',str(args.config),'--data-manifest',str(args.data_manifest)],
          'commands':commands,'execute':False,'security_accepted':False,'status':'plan_only_until_gate1_lock'}
    return plan



def trainer_commands(item):
    """Expand the diagnostic wrapper into the two actual trainer invocations."""
    command = item['command']
    if item['variant'] != 'two_phase_diagnostic':
        return [command]
    phase1_steps = command[command.index('--phase1-steps')+1]
    total_steps = command[command.index('--total-steps')+1]
    output = Path(item['output_dir'])
    shared = [command[0], str(ROOT/'scripts/train_streaming.py')]
    index = 2
    while index < len(command):
        flag = command[index]
        if flag == '--execute':
            index += 1
            continue
        value = command[index+1]
        if flag not in ('--phase1-steps','--total-steps','--output-dir'):
            shared.extend((flag,value))
        index += 2
    shared += ['--constraint-mode','constant','--lambda-priv','1.0','--lambda-div','1.0']
    phase1_dir = output/'phase1_privacy_injection'
    # A23 moved the phase-2 resume origin to the staged transition checkpoint.
    phase2_resume = output/'phase2_start'/'start.pt'
    return [shared+['--output-dir',str(phase1_dir),'--steps',phase1_steps,'--lambda-util','0.0'],
            shared+['--output-dir',str(output),'--steps',total_steps,'--lambda-util','1.0',
                    '--resume',str(phase2_resume)]]


def verify_completed_command(command, fixed):
    """Refuse partial or foreign results before skipping any matrix cell."""
    import math
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_question_disjoint_evaluation import verify_training_lineage
    options = dict(zip(command[2::2],command[3::2]))
    run = Path(options['--output-dir'])
    config = json.loads((run/'config.json').read_text())
    report = json.loads((run/'report.json').read_text())
    if (report.get('status') != 'completed_with_limits' or report.get('security_accepted') is not False
            or report.get('steps_completed') != int(options['--steps'])):
        raise ValueError('existing training report is incomplete or has unexpected status: '+str(run))
    path_fields = {'model_path','train_jsonl','valid_jsonl','output_dir','resume'}
    for flag, value in options.items():
        name = flag[2:].replace('-','_')
        actual = config.get(name)
        if name in path_fields:
            matches = actual is not None and Path(actual).resolve() == Path(value).resolve()
        elif isinstance(actual,bool):
            matches = False
        elif isinstance(actual,(int,float)):
            matches = math.isfinite(actual) and actual == float(value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError('existing training setting differs: '+name+' at '+str(run))
    if '--resume' not in options and config.get('resume') is not None:
        raise ValueError('expected fresh initialization at '+str(run))
    if config.get('max_train_samples') is not None:
        raise ValueError('formal matrix requires the full training pool at '+str(run))
    # Variant cells intentionally override lambda/constraint fields, so compose
    # the per-cell pinned recipe from the trainer command before fingerprinting.
    import copy
    from osnip_repro.checkpointing import NUMERIC_RECIPE_FIELDS, RECIPE_FIELDS
    fixed_cell = copy.deepcopy(fixed)
    cell_training = fixed_cell.setdefault('training', {})
    for flag, value in options.items():
        name = flag[2:].replace('-', '_')
        if name not in RECIPE_FIELDS:
            continue
        if name in NUMERIC_RECIPE_FIELDS:
            try:
                cell_training[name] = int(value)
            except ValueError:
                cell_training[name] = float(value)
        else:
            cell_training[name] = value
    return verify_training_lineage(run,config,report,fixed=fixed_cell,model_path=Path(options['--model-path']))



def verify_pilot_prerequisites(args, fixed):
    """Bind the completed pilot and its gates to the selected v2 inputs."""
    from run_question_disjoint_pilot import build_plan as build_pilot_plan
    run = args.pilot_report.parent
    plan = build_pilot_plan(model_path=args.model_path,config_path=args.config,
                            manifest_path=args.data_manifest,output_dir=run)
    pilot = json.loads(args.pilot_report.read_text())
    if pilot.get('status')!='completed_with_limits':
        raise ValueError('v2 pilot is not completed')
    lineage = verify_completed_command(plan['command'],fixed)
    for label,command,filename in [('gate0',plan['gate0_command'],'gate0_report.json'),
                                    ('gate1',plan['gate1_command'],'gate1_report.json')]:
        folder = Path(command[command.index('--output-dir')+1])
        path = folder/filename
        report = json.loads(path.read_text())
        if report.get('status')!='ready' or report.get('model_path') is None or Path(report['model_path']).resolve()!=args.model_path.resolve():
            raise ValueError('v2 pilot '+label+' is not ready for the selected model')
        if label=='gate1':
            recorded = report.get('command',[])
            for flag in ('--train-jsonl','--valid-jsonl'):
                expected = command[command.index(flag)+1]
                if flag not in recorded or Path(recorded[recorded.index(flag)+1]).resolve()!=Path(expected).resolve():
                    raise ValueError('v2 pilot gate1 uses different training inputs')
    return {'pilot':str(args.pilot_report.resolve()),'lineage':lineage,'gates_verified':True}


def main():
    args=parse_args(); plan=build_plan(args); args.output_root.mkdir(parents=True,exist_ok=True)
    path=args.output_root/'execution_plan.json'
    if path.exists(): raise FileExistsError(path)
    plan['execute']=args.execute; path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'plan':str(path.resolve()),'commands':plan['command_count'],'execute':args.execute},ensure_ascii=False,indent=2))
    if not args.execute: return
    cfg=json.loads(args.config.read_text());
    if cfg['formal_training_defaults'].get('status')!='locked_after_gate1':
        raise RuntimeError('formal hyperparameters are provisional; set status=locked_after_gate1 only after v2 Gate 1 review')
    verify_pilot_prerequisites(args,cfg)
    gate0=Path(plan['gate0_command'][plan['gate0_command'].index('--output-dir')+1]); subprocess.run(plan['gate0_command'],check=True)
    if json.loads((gate0/'gate0_report.json').read_text()).get('status')!='ready': raise RuntimeError('v2 Gate 0 failed')
    for item in plan['commands']:
      out=Path(item['output_dir'])
      if (out/'report.json').exists():
        for command in trainer_commands(item): verify_completed_command(command,cfg)
        continue
      if out.exists() and any(out.iterdir()): raise FileExistsError(out)
      subprocess.run(item['command'],check=True)
      for command in trainer_commands(item): verify_completed_command(command,cfg)

if __name__=='__main__': main()
