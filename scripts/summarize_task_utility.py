#!/usr/bin/env python3
"""Paired prompt bootstrap and source diagnostics from a saved task evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def summarize(rows: list[dict], *, seed: int, samples: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    summary = {}
    for source in ['all', *sorted({row['source'] for row in rows})]:
        selected = rows if source == 'all' else [row for row in rows if row['source'] == source]
        clean = torch.tensor([row['clean_correct'] for row in selected], dtype=torch.float64)
        encrypted = torch.tensor([row['encrypted_correct'] for row in selected], dtype=torch.float64)
        indices = torch.randint(len(selected), (samples, len(selected)), generator=generator)
        clean_bootstrap = clean[indices].sum(-1)
        encrypted_bootstrap = encrypted[indices].sum(-1)
        valid = clean_bootstrap > 0
        quantiles = torch.tensor([.025, .975], dtype=torch.float64)
        retention_ci = (torch.quantile(encrypted_bootstrap[valid] / clean_bootstrap[valid], quantiles).tolist()
                        if valid.any() else None)
        summary[source] = {
            'samples': len(selected), 'clean_correct': int(clean.sum()), 'encrypted_correct': int(encrypted.sum()),
            'clean_accuracy': float(clean.mean()), 'encrypted_accuracy': float(encrypted.mean()),
            'retention': float(encrypted.sum() / clean.sum()) if clean.sum() > 0 else None,
            'retention_bootstrap_95ci': retention_ci,
            'bootstrap_resamples_with_zero_clean_correct': int((~valid).sum()),
            'paired_accuracy_delta_bootstrap_95ci': torch.quantile((encrypted[indices] - clean[indices]).mean(-1), quantiles).tolist(),
            'correct_label_counts': {label: sum(row['correct'] == label for row in selected)
                                     for label in sorted({row['correct'] for row in selected})},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--bootstrap-samples', type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_samples < 2:
        raise ValueError('bootstrap-samples must be >= 2')
    if args.output.exists():
        raise FileExistsError(args.output)
    task = json.loads(args.input.read_text())
    if not task['rows']:
        raise ValueError('task report has no scored observations')
    result = {'format': 'osnip_task_utility_stratified_diagnostic_v2',
              'checkpoint_sha256': task['checkpoint_sha256'], 'choice_policy': task.get('choice_policy'),
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'bootstrap_samples': args.bootstrap_samples, 'bootstrap_seed': args.seed,
              'security_accepted': False, 'key_attack_performed': False,
              'by_source': summarize(task['rows'], seed=args.seed, samples=args.bootstrap_samples),
              'limitations': ['Prompt bootstrap is conditional on this checkpoint and single client key seed.',
                              'A point estimate at 0.95 does not establish a population retention bound.',
                              'Aggregate retention can mask losses on an individual source.']}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output.resolve()), 'by_source': result['by_source']}, indent=2))


if __name__ == '__main__':
    main()
