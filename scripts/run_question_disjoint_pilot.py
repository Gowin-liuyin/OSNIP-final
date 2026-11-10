#!/usr/bin/env python3
"""Plan or run a fresh-initialization pilot on the audited question-disjoint v2 data."""
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
DEFAULT_CONFIG = ROOT / 'configs/llama32_1b_question_disjoint_v2.json'
DEFAULT_MANIFEST = ROOT / 'data/osnip_65k_question_disjoint_v2/data_manifest.json'


def build_plan(*, model_path: Path, config_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    require_locked_recipe(config)
    data = config['data']
    if data.get('dataset_policy') != 'question_disjoint_v2' or data.get('fresh_training_required') is not True:
        raise ValueError('pilot requires the question-disjoint v2 fresh-training config')
    if Path(data['manifest_path']).resolve() != manifest_path.resolve():
        raise ValueError('config and selected data manifest differ')
    pilot = config['gate1_pilot']
    train_path, valid_path = manifest_path.parent / 'train.jsonl', manifest_path.parent / 'valid.jsonl'
    command = [sys.executable, str(ROOT / 'scripts/train_streaming.py'),
               '--model-path', str(model_path), '--train-jsonl', str(train_path), '--valid-jsonl', str(valid_path),
               '--output-dir', str(output_dir), '--steps', str(pilot['steps']),
               '--batch-size', str(pilot['batch_size']), '--gradient-accumulation-steps', str(pilot['gradient_accumulation_steps']),
               '--max-length', str(pilot['max_length']), '--max-valid-samples', str(pilot['max_valid_samples']),
               '--key-dim', str(pilot['key_dim']), '--hidden-dim', str(pilot['hidden_dim']),
               '--encryptor-architecture', pilot['encryptor_architecture'], '--epsilon', str(pilot['epsilon']),
               '--diversity-margin', str(pilot['diversity_margin']), '--diversity-space', pilot['diversity_space'],
               '--lambda-util', str(pilot['lambda_util']), '--lambda-priv', str(pilot['lambda_priv']), '--lambda-div', str(pilot['lambda_div']),
               '--warmup-steps', str(pilot['warmup_steps']), '--tau-low', str(pilot['tau_low']), '--tau-high', str(pilot['tau_high']),
               '--learning-rate', str(pilot['learning_rate']), '--weight-decay', str(pilot['weight_decay']),
               '--device', pilot['device'], '--dtype', pilot['dtype'], '--seed', str(pilot['seed']),
               '--constraint-mode', pilot['constraint_mode'], '--grad-clip', str(pilot['grad_clip']),
               '--checkpoint-every', str(pilot['checkpoint_every']), '--evaluate-every', str(pilot['evaluate_every'])]
    gate0_command = [sys.executable, str(ROOT / 'scripts/run_gate0.py'),
                     '--model-path', str(model_path), '--output-dir', str(output_dir.parent / (output_dir.name + '_gate0')),
                     '--config', str(config_path), '--data-manifest', str(manifest_path)]
    gate1_command = [sys.executable, str(ROOT / 'scripts/run_gate1.py'),
                     '--model-path', str(model_path), '--train-jsonl', str(train_path), '--valid-jsonl', str(valid_path),
                     '--output-dir', str(output_dir.parent / (output_dir.name + '_gate1')),
                     '--steps', '2', '--max-train-samples', '32', '--max-valid-samples', '8',
                     '--batch-size', '1', '--gradient-accumulation-steps', '1', '--max-length', '32',
                     '--device', pilot['device'], '--dtype', pilot['dtype'], '--seed', str(pilot['seed']),
                     '--diversity-space', pilot['diversity_space'],
                     '--encryptor-architecture', pilot['encryptor_architecture']]
    utility_control = config.get('pre_pilot_utility_control')
    utility_command = None
    if utility_control:
        output = output_dir.parent/(output_dir.name+'_geometry_control_utility.json')
        utility_command = [sys.executable,str(ROOT/'scripts/evaluate_no_key.py'),
            '--model-path',str(model_path),'--checkpoint',utility_control['checkpoint'],
            '--data-jsonl',utility_control['data_jsonl'],'--output',str(output),
            '--max-samples',str(utility_control['max_samples']),'--max-length',str(utility_control['max_length']),
            '--batch-size',str(utility_control['batch_size']),'--seed',str(utility_control['seed']),
            '--max-ppl-ratio',str(utility_control['max_ppl_ratio']),'--max-mean-kl',str(utility_control['max_mean_kl']),
            '--device',pilot['device'],'--dtype',pilot['dtype'],'--skip-vocabulary-probe']
    utility_checks=[]
    if utility_control:utility_checks.append({'control':utility_control,'command':utility_command})
    for control in config.get('additional_pre_pilot_utility_controls',[]):
        if not utility_command:raise ValueError('additional utility controls require primary utility control settings')
        extra=list(utility_command)
        if any(control.get(k)!=utility_control.get(k) for k in ('data_jsonl','data_sha256','max_samples','max_length','batch_size','seed','max_ppl_ratio','max_mean_kl')):
            raise ValueError('additional utility controls must use the same evaluation protocol')
        extra[extra.index('--checkpoint')+1]=control['checkpoint']
        suffix=control['output_suffix']
        if Path(suffix).name!=suffix or not suffix.endswith('.json'):raise ValueError('invalid utility report suffix')
        extra[extra.index('--output')+1]=str(output_dir.parent/(output_dir.name+suffix))
        utility_checks.append({'control':control,'command':extra})
    return {
        'format': 'osnip_question_disjoint_pilot_plan_v1',
        'scope': pilot['scope'], 'initialization': 'fresh_encoder_no_resume',
        'config': str(config_path.resolve()), 'data_manifest': str(manifest_path.resolve()),
        'train_sha256': data['train_sha256'], 'valid_sha256': data['valid_sha256'],
        'excluded_confirmatory_sciq_sha256': data['exclusion_sha256']['sciq_confirmatory'],
        'output_dir': str(output_dir.resolve()), 'execute': False, 'security_accepted': False,
        'gate0_command': gate0_command, 'gate1_command': gate1_command, 'command': command,
        'pre_pilot_utility_control':utility_control,'pre_pilot_utility_command':utility_command,
        'pre_pilot_utility_checks':utility_checks,
        'limitations': config['limitations'],
    }



def verify_utility_control_inputs(control):
    """Bind the diagnostic to fixed encoder/config/data bytes before loading a LM."""
    checkpoint = Path(control['checkpoint'])
    for path,expected in [(checkpoint,control['checkpoint_sha256']),
                          (checkpoint.parent/'config.json',control['checkpoint_config_sha256']),
                          (Path(control['data_jsonl']),control['data_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('geometry utility control fingerprint mismatch: '+str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'model/Llama-3.2-1B')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/llama32_1b/question_disjoint_v2_pilot')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = build_plan(model_path=args.model_path, config_path=args.config, manifest_path=args.data_manifest, output_dir=args.output_dir)
    plan_path = args.output_dir.parent / (args.output_dir.name + '_plan.json')
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing.get('format') != 'osnip_question_disjoint_pilot_plan_v1' or existing.get('output_dir') != str(args.output_dir.resolve()):
            raise FileExistsError(f'refusing to overwrite unrelated plan: {plan_path}')
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan['execute'] = args.execute
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'plan': str(plan_path.resolve()), 'execute': args.execute, 'gate0_command': plan['gate0_command'], 'gate1_command': plan['gate1_command'], 'command': plan['command']}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'pilot output is non-empty: {args.output_dir}')
    for path in (args.model_path, args.data_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    for split in ('train.jsonl', 'valid.jsonl'):
        if not (args.data_manifest.parent / split).is_file():
            raise FileNotFoundError(args.data_manifest.parent / split)
    gate0_dir = args.output_dir.parent / (args.output_dir.name + '_gate0')
    gate1_dir = args.output_dir.parent / (args.output_dir.name + '_gate1')
    for path in (gate0_dir, gate1_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'preflight output is non-empty: {path}')
    for check in plan['pre_pilot_utility_checks']:
        verify_utility_control_inputs(check['control'])
        utility_output=Path(check['command'][check['command'].index('--output')+1])
        if utility_output.exists(): raise FileExistsError(utility_output)
    subprocess.run(plan['gate0_command'], check=True)
    gate0 = json.loads((gate0_dir / 'gate0_report.json').read_text())
    if gate0.get('status') != 'ready':
        raise RuntimeError('Gate 0 did not pass; Gate 1 and pilot were not started')
    subprocess.run(plan['gate1_command'], check=True)
    gate1 = json.loads((gate1_dir / 'gate1_report.json').read_text())
    if gate1.get('status') != 'ready':
        raise RuntimeError('Gate 1 did not pass; 1,000-step pilot was not started')
    for check in plan['pre_pilot_utility_checks']:
        verify_utility_control_inputs(check['control'])
        subprocess.run(check['command'], check=True)
    subprocess.run(plan['command'], check=True)


if __name__ == '__main__':
    main()
