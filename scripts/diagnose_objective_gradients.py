#!/usr/bin/env python3
"""Validation-only local gradient diagnosis; no optimizer step or key attack."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
COMPONENTS = ('utility', 'privacy', 'diversity')


def component_gradients(losses, parameters):
    import torch
    parameters = tuple(p for p in parameters if p.requires_grad)
    result = {}
    for name in (*COMPONENTS, 'total'):
        grads = torch.autograd.grad(losses[name], parameters, retain_graph=name != 'total', allow_unused=True)
        result[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).to('cpu').to(torch.float64)
                                  for p, g in zip(parameters, grads)])
        if not torch.isfinite(result[name]).all():
            raise ValueError('nonfinite gradient: ' + name)
    return result


def gradient_summary(gradients, weighted, *, grad_clip):
    import torch
    if grad_clip <= 0:
        raise ValueError('positive clip required')
    combined = sum(weighted.values())
    error = float((combined - gradients['total']).norm())
    reference = float(gradients['total'].norm())
    if error > 2e-5 * max(reference, 1e-8) + 1e-9:
        raise ValueError('weighted components disagree with autograd total')
    norms = {name: float(g.norm()) for name, g in gradients.items()}
    pairs = {}
    for i, a in enumerate(COMPONENTS):
        for b in COMPONENTS[i + 1:]:
            denom = norms[a] * norms[b]
            pairs[a + ':' + b] = float(gradients[a] @ gradients[b]) / denom if denom > 0 else None
    clip_scale = min(1., grad_clip / (reference + 1e-6))
    return {'gradient_norms': norms, 'component_cosines': pairs,
            'weighted_component_norms': {k: float(g.norm()) for k, g in weighted.items()},
            'total_gradient_reconstruction_absolute_error': error,
            'hypothetical_sgd_clip_scale': clip_scale,
            'first_order_loss_change_per_unit_sgd_lr':
                {k: -float(g @ combined) for k, g in gradients.items() if k != 'total'},
            'first_order_loss_change_per_unit_clipped_sgd_lr':
                {k: -clip_scale * float(g @ combined) for k, g in gradients.items() if k != 'total'}}


def diagnose_one(model, encoder, tokenizer, text, args, device, *, step, key_seed):
    from train_streaming import run_objective
    losses, stats = run_objective(model, encoder, tokenizer, [text], args, device,
                                  training=True, step=step, key_seed=key_seed)
    gradients = component_gradients(losses, encoder.parameters())
    weights = {'utility': args.lambda_util, 'privacy': stats['privacy_weight'], 'diversity': stats['diversity_weight']}
    weights = {k: v * stats.get('loss_weight', 1.0) for k, v in weights.items()}
    weighted = {k: gradients[k] * weights[k] for k in COMPONENTS}
    report = {'key_seed': key_seed, 'stats': stats, 'weights': weights,
              **gradient_summary(gradients, weighted, grad_clip=args.grad_clip)}
    return report, gradients, weighted


def command_for_pid(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True, check=False)
    return shlex.split(result.stdout.strip()) if result.stdout.strip() else None


def validate_wait_identity(tokens, script, flag, expected_path):
    if not any(Path(token).name == script for token in tokens):
        raise ValueError('wait PID identity mismatch: ' + script)
    try:
        selected = Path(tokens[tokens.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError('wait PID lacks expected path') from exc
    if not selected.is_absolute():
        selected = ROOT / selected
    if selected.resolve() != expected_path.resolve():
        raise ValueError('wait PID output path mismatch')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--training-run', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--max-samples', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--after-training-pid', type=int)
    ap.add_argument('--after-scorer-pid', type=int)
    ap.add_argument('--scorer-report', type=Path)
    ap.add_argument('--poll-seconds', type=int, default=30)
    args = ap.parse_args()
    if min(args.max_samples, args.poll_seconds) < 1:
        raise ValueError('positive sample count and interval required')
    waits = (args.after_training_pid, args.after_scorer_pid, args.scorer_report)
    if any(v is not None for v in waits) and not all(v is not None for v in waits):
        raise ValueError('serial wait requires both PIDs and the scorer report')
    if any(pid is not None and pid <= 0 for pid in waits[:2]):
        raise ValueError('positive PIDs required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {'security_accepted': False, 'key_attack_performed': False,
             'checkpoint': str(args.checkpoint.resolve()), 'training_pid': args.after_training_pid,
             'scorer_pid': args.after_scorer_pid}
    def save(status):
        state.update(status=status, observed_at=datetime.datetime.now().astimezone().isoformat())
        (args.output_dir / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
    try:
        config = json.loads((args.training_run / 'config.json').read_text())
        if args.after_training_pid is not None:
            # Validate both live identities now and on every poll, then check completion artifacts.
            save('waiting_for_training_and_scorer')
            while True:
                trainer = command_for_pid(args.after_training_pid)
                scorer = command_for_pid(args.after_scorer_pid)
                if trainer:
                    validate_wait_identity(trainer, 'train_streaming.py', '--output-dir', args.training_run)
                if scorer:
                    validate_wait_identity(scorer, 'verify_summary_scorer.py', '--output', args.scorer_report)
                if not trainer and not scorer:
                    break
                time.sleep(args.poll_seconds)
            training = json.loads((args.training_run / 'report.json').read_text())
            scorer_result = json.loads(args.scorer_report.read_text())
            if training.get('status') != 'completed_with_limits' or training.get('steps_completed') != config['steps']:
                raise ValueError('expected pilot budget did not finish')
            if scorer_result.get('status') != 'paired_scorer_verified':
                raise ValueError('scorer did not verify successfully')
        save('loading_frozen_checkpoint')
        import torch
        from osnip_repro.data import file_sha256, read_text_jsonl
        from osnip_repro.encoder import build_encryptor_from_config
        from osnip_repro.modeling import load_causal_lm
        torch.set_num_threads(2)
        checkpoint_hash = file_sha256(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError('checkpoint changed during loading')
        if payload.get('format') != 'osnip_training_checkpoint_v1' or payload.get('config') != config:
            raise ValueError('checkpoint/config binding mismatch')
        for name, digest in config['source_hashes'].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError('training source drift: ' + name)
        data = Path(config['valid_jsonl'])
        if file_sha256(data) != config['valid_sha256']:
            raise ValueError('validation data changed')
        if args.max_samples != config['gradient_accumulation_steps'] or config['batch_size'] != 1:
            raise ValueError('this diagnostic requires one complete batch-size-one accumulation group')
        texts = read_text_jsonl(data, limit=args.max_samples)
        if len(texts) != args.max_samples:
            raise ValueError('short validation input')
        tokenizer, model, device = load_causal_lm(config['model_path'], device=args.device, dtype=config['dtype'])
        encoder = build_encryptor_from_config(config).to(device).train()
        encoder.load_state_dict(payload['encoder'])
        step = int(payload['global_step']) + 1
        del payload
        objective_args = argparse.Namespace(**config)
        rows, averaged, weighted_average = [], {}, {}
        save('measuring_validation_gradients')
        for i, text in enumerate(texts):
            row, gradients, weighted = diagnose_one(model, encoder, tokenizer, text, objective_args, device,
                                                     step=step, key_seed=args.seed + i)
            rows.append({'prompt_index': i, **row})
            for name, gradient in gradients.items():
                averaged[name] = averaged.get(name, 0) + gradient / len(texts)
            for name, gradient in weighted.items():
                weighted_average[name] = weighted_average.get(name, 0) + gradient / len(texts)
            print(json.dumps({'prompt_completed': i + 1, 'prompts': len(texts)}), flush=True)
        result = {'format': 'osnip_validation_objective_gradients_v1', 'status': 'diagnostic_completed',
                  'scope': 'exploratory_validation_only', 'security_accepted': False, 'key_attack_performed': False,
                  'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
                  'checkpoint_step': step - 1, 'gate_step': step, 'device': str(device), 'dtype': config['dtype'],
                  'config_sha256': file_sha256(args.training_run / 'config.json'),
                  'data_jsonl': str(data), 'data_sha256': file_sha256(data), 'max_length': config['max_length'],
                  'model_path': config['model_path'],
                  'model_manifest_sha256': file_sha256(Path(config['model_path']) / 'download_manifest.json'),
                  'source_hashes': {**config['source_hashes'], str(Path(__file__).relative_to(ROOT)): file_sha256(Path(__file__))},
                  'prompts': len(rows), 'rows': rows,
                  'accumulated_gradient': gradient_summary(averaged, weighted_average, grad_clip=config['grad_clip']),
                  'limitations': ['Fixed validation prompts/keys are not actual next training microbatches.',
                                  'Weights are computed per microbatch before gradient accumulation, as in the trainer.',
                                  'Directional derivatives describe a local SGD direction, not AdamW with checkpoint moments.',
                                  'No parameters are updated. Clipping direction analysis does not establish causality.',
                                  'This diagnostic does not measure KNN, VMA, task utility or model capacity limits.']}
        if file_sha256(args.checkpoint) != checkpoint_hash or file_sha256(data) != config['valid_sha256']:
            raise ValueError('input artifacts changed during diagnostic')
        (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        save('diagnostic_completed')
    except Exception as exc:
        state.update(exception=type(exc).__name__, detail=str(exc))
        save('failed')
        raise


if __name__ == '__main__':
    main()
