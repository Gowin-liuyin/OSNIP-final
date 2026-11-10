#!/usr/bin/env python3
"""Client-side geometry diagnostic; no LM forward, recovery, or key inference."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256, read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.projection import iso_norm_project


def energy_components(vectors: torch.Tensor) -> dict[str, float | None]:
    """Exact unweighted token energy split, with zero-energy fractions undefined."""
    if vectors.ndim != 2 or vectors.shape[0] < 2 or not torch.isfinite(vectors).all():
        raise ValueError('need at least two finite token vectors')
    values = vectors.double()
    common = values.mean(dim=0, keepdim=True)
    total = float(values.square().sum())
    shared = float(common.square().sum()) * len(values)
    centered = float((values - common).square().sum())
    return {'total_energy': total, 'shared_energy': shared, 'centered_energy': centered,
            'shared_fraction': shared / total if total > 0 else None,
            'centered_fraction': centered / total if total > 0 else None}


def summarize_geometry(h: torch.Tensor, raw: torch.Tensor, *, epsilon: float,
                       margin: float) -> dict:
    """raw is [independent keys, tokens, width]; all keys belong to the client."""
    if raw.ndim != 3 or raw.shape[1:] != h.shape or raw.shape[0] < 2:
        raise ValueError('need aligned embeddings and at least two client keys')
    if h.ndim != 2 or h.shape[0] < 2 or not torch.isfinite(h).all() or not torch.isfinite(raw).all():
        raise ValueError('need finite observations with at least two tokens')
    reference = h.unsqueeze(0).expand_as(raw)
    projected = iso_norm_project(raw, reference)
    residual = raw - reference
    h_centered = h - h.mean(dim=0, keepdim=True)
    h_variation = float(h_centered.double().square().sum())
    raw_centered = raw - raw.mean(dim=1, keepdim=True)
    variation_error = float((raw_centered - h_centered).double().square().sum())
    cosine = F.cosine_similarity(reference, projected, dim=-1)
    h_norm = h.norm(dim=-1).clamp_min(1e-12)
    raw_norm = raw.norm(dim=-1).clamp_min(1e-12)
    per_key = [energy_components(row) for row in residual]
    total_energy = sum(row['total_energy'] for row in per_key)
    shared_energy = sum(row['shared_energy'] for row in per_key)
    key_pairs = list(itertools.combinations(range(len(raw)), 2))
    raw_distance = torch.stack([(raw[a] - raw[b]).norm(dim=-1) for a, b in key_pairs])
    projected_distance = torch.stack([(projected[a] - projected[b]).norm(dim=-1) for a, b in key_pairs])
    projected_components = [energy_components(row) for row in projected - reference]
    projected_total = sum(row['total_energy'] for row in projected_components)
    return {
        'tokens': len(h), 'keys': len(raw),
        'raw_residual_total_energy': total_energy,
        'raw_residual_shared_energy': shared_energy,
        'raw_residual_shared_fraction': shared_energy / total_energy if total_energy else None,
        'projected_residual_total_energy': projected_total,
        'projected_residual_shared_energy': sum(row['shared_energy'] for row in projected_components),
        'centered_raw_change_relative_to_h_energy': variation_error / (len(raw) * h_variation) if h_variation else None,
        'mean_cosine': float(cosine.mean()),
        'privacy_hinge': float((cosine.abs() - epsilon).clamp_min(0).mean()),
        'fraction_abs_cosine_le_epsilon': float((cosine.abs() <= epsilon).float().mean()),
        'relative_projected_perturbation': float(((projected-reference).norm(dim=-1)/h_norm).mean()),
        'mean_raw_to_h_norm_ratio': float((raw_norm/h_norm).mean()),
        'max_relative_projection_norm_error': float(((projected.norm(dim=-1)-h_norm).abs()/h_norm).max()),
        'raw_key_pair_mean_distance': float(raw_distance.mean()),
        'projected_key_pair_mean_distance': float(projected_distance.mean()),
        'raw_key_pair_margin_satisfied_fraction': float((raw_distance >= margin).float().mean()),
        'projected_key_pair_margin_satisfied_fraction': float((projected_distance >= margin).float().mean()),
    }



def analytical_surrogate_control() -> dict:
    """Synthetic counterexample to treating orthogonality as a recovery bound."""
    h = F.pad(torch.eye(32), (0, 2048 - 32))
    offset = torch.zeros(2048)
    offset[-1] = 5.
    raw = torch.stack((h + offset, h - offset))
    z = iso_norm_project(raw, h.unsqueeze(0).expand_as(raw))
    # Equal-norm vocabulary retains the original token's largest dot product.
    nearest = torch.cdist(z.reshape(-1, 2048), h).argmin(dim=-1)
    targets = torch.arange(32).repeat(2)
    # A fixed homogeneous decoder ignores the added null direction and scale.
    def decoder(values):
        semantic = values[..., :32]
        return 5. * F.normalize(semantic, dim=-1)
    clean_logp = decoder(h).log_softmax(-1)
    encrypted_logp = decoder(z).log_softmax(-1)
    kl = (clean_logp.exp() * (clean_logp - encrypted_logp)).sum(-1)
    result = summarize_geometry(h, raw, epsilon=.3, margin=1.)
    return {'scope': 'analytical_synthetic_control_not_llama', 'vocabulary_size': 32,
            'width': 2048, 'max_utility_kl': float(kl.abs().max()),
            'knn_top1': float((nearest == targets).float().mean()),
            'geometry': result,
            'interpretation': 'Iso-norm, cosine hinge, key diversity and even exact toy utility preservation do not by themselves bound token recovery. This does not refute the paper existence theorem or measure Llama security.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--data-jsonl', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--max-samples', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--keys-per-prompt', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_samples, args.threads) < 1 or min(args.max_length, args.keys_per_prompt) < 2:
        parser.error('positive samples/threads and at least two tokens/keys required')
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    ids = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)['input_ids']
    if any(len(row) < 2 for row in ids):
        raise ValueError('all prompts must have at least two active tokens')
    unique = sorted({token for row in ids for token in row})
    # Slice only observed rows; never load the full LM or allocate the vocabulary table.
    with safe_open(args.model_path / 'model.safetensors', framework='pt', device='cpu') as handle:
        table = handle.get_slice('model.embed_tokens.weight')
        embedding_rows = torch.cat([table[token:token+1].float() for token in unique])
    lookup = {token: index for index, token in enumerate(unique)}
    embeddings = [embedding_rows[[lookup[token] for token in row]] for row in ids]
    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    keys = torch.randn(len(ids), args.keys_per_prompt, int(config['key_dim']), generator=rng)
    report = {
        'format': 'osnip_client_geometry_diagnostic_v1',
        'scope': 'validation_client_geometry_only', 'security_accepted': False,
        'model_path': str(args.model_path.resolve()),
        'model_manifest_sha256': file_sha256(args.model_path / 'download_manifest.json'),
        'data_jsonl': str(args.data_jsonl.resolve()), 'data_sha256': file_sha256(args.data_jsonl),
        'config_sha256': file_sha256(args.config), 'seed': args.seed,
        'max_length': args.max_length, 'prompts': len(ids), 'tokens': sum(map(len, ids)),
        'keys_per_prompt': args.keys_per_prompt, 'unique_embedding_rows_loaded': len(unique),
        'weight_tensor': 'model.embed_tokens.weight', 'device': 'cpu', 'dtype': 'float32',
        'source_hashes': {str(path.relative_to(ROOT)): file_sha256(path) for path in
                          [Path(__file__), ROOT/'src/osnip_repro/encoder.py', ROOT/'src/osnip_repro/projection.py']},
        'limitations': ['Geometry is not token recovery or a privacy acceptance test.',
                        'Shared component is a descriptive per-prompt token mean, not an inferred key.',
                        'Same known client keys and validation prompts are used across checkpoints.',
                        'Energy fractions are ratios of summed energies; remaining means are token weighted.'],
        'analytical_surrogate_control': analytical_surrogate_control(),
        'checkpoints': [],
    }
    for checkpoint in args.checkpoint:
        before = file_sha256(checkpoint)
        encoder = build_encryptor_from_config(config).eval()
        encoder.load_state_dict(load_encoder_state_dict(str(checkpoint), map_location='cpu', expected_config=config))
        if file_sha256(checkpoint) != before:
            raise RuntimeError('checkpoint changed while loading; use a frozen snapshot')
        rows = []
        with torch.inference_mode():
            for index, h in enumerate(embeddings):
                raw = encoder(h.unsqueeze(0).expand(args.keys_per_prompt, -1, -1), keys[index])
                row = summarize_geometry(h, raw, epsilon=float(config['epsilon']),
                                         margin=float(config['diversity_margin']))
                row['prompt_index'] = index
                rows.append(row)
        token_count = sum(row['tokens'] for row in rows)
        summary = {}
        for name in rows[0]:
            if name in {'tokens', 'keys', 'prompt_index'} or name.endswith('_energy') or name == 'raw_residual_shared_fraction':
                continue
            if name == 'max_relative_projection_norm_error':
                summary[name] = max(row[name] for row in rows)
            elif all(row[name] is not None for row in rows):
                summary[name] = sum(row[name] * row['tokens'] for row in rows) / token_count
            else:
                summary[name] = None
        for prefix in ('raw_residual', 'projected_residual'):
            total = sum(row[f'{prefix}_total_energy'] for row in rows)
            shared = sum(row[f'{prefix}_shared_energy'] for row in rows)
            summary[f'{prefix}_shared_energy_fraction'] = shared / total if total else None
        # This quantity is a per-prompt normalized ratio, unlike the pooled energy fractions.
        summary['centered_raw_change_relative_to_h_energy'] = (
            sum(row['centered_raw_change_relative_to_h_energy'] * row['tokens'] for row in rows)/token_count
            if all(row['centered_raw_change_relative_to_h_energy'] is not None for row in rows) else None)
        item = {'checkpoint': str(checkpoint.resolve()), 'sha256': before, 'summary': summary, 'observations': rows}
        report['checkpoints'].append(item)
        print(json.dumps({'checkpoint': str(checkpoint), 'summary': summary}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
