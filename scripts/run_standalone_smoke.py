#!/usr/bin/env python3
"""Run a tiny independent OSNIP objective/gradient smoke; not a paper result."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch import nn
from osnip_repro.encoder import ARCHITECTURES, KeyConditionedEncryptor
from osnip_repro.losses import osnip_loss
from osnip_repro.projection import iso_norm_project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('outputs/standalone-smoke/report.json'))
    ap.add_argument('--steps', type=int, default=8)
    args = ap.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive')
    width, vocab, key_dim = 32, 97, 16
    results = {}
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        torch.manual_seed(20261017 + architecture_index)
        predictor = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, vocab)).eval()
        for p in predictor.parameters():
            p.requires_grad_(False)
        enc = KeyConditionedEncryptor(width, key_dim=key_dim, hidden_dim=64, architecture=architecture)
        opt = torch.optim.AdamW(enc.parameters(), lr=3e-3)
        h = torch.randn(12, width)
        k1 = torch.randn(key_dim)
        k2 = torch.randn(key_dim)
        clean_logits = predictor(h).detach()
        history = []
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            z1 = iso_norm_project(enc(h, k1), h)
            z2 = iso_norm_project(enc(h, k2), h)
            losses = osnip_loss(clean_logits, predictor(z1), h, z1, epsilon=0.3, lambda_priv=1.0, z_other=z2, diversity_margin=0.5, lambda_div=0.1)
            losses['total'].backward()
            opt.step()
            history.append({'step': step + 1, **{name: float(value.detach()) for name, value in losses.items()}})
        results[architecture] = {
            'architecture': architecture,
            'steps': args.steps,
            'finite': all(torch.isfinite(value).all().item() for value in enc.parameters()),
            'output_shape': list(z1.shape),
            'key_sensitive': not torch.allclose(enc(h, k1), enc(h, k2)),
            'projected_norm_error': float((z1.norm(dim=-1) - h.norm(dim=-1)).abs().max().detach()),
            'history': history,
        }
    report = {
        'format': 'osnip_standalone_architecture_smoke_v2',
        'status': 'engineering_smoke',
        'paper_result': False,
        'architectures': list(ARCHITECTURES),
        'results': results,
        'all_passed': all(row['finite'] and row['key_sensitive'] and row['projected_norm_error'] < 1e-5 for row in results.values()),
        'width': width,
        'vocab': vocab,
        'key_dim': key_dim,
        'server_frozen': True,
        'limitations': ['toy predictor and synthetic vectors', 'Qwen/Llama utility and attack benchmarks are not measured here', 'encryptor architectures are configurable engineering variants, not the undisclosed author implementation'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'all_passed': report['all_passed'], 'architectures': list(results), 'output': str(args.output.resolve())}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
