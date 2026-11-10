#!/usr/bin/env python3
"""Estimate OSNIP's local semantic coverage and Theorem 2.5 condition.

The diagnostic samples directions on the input-embedding sphere, replaces one
causal input embedding at a time, and measures the frozen model's KL change at
that position. It is a theory/geometry diagnostic, not a security result and
never uses an encryptor key or an inversion attack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F

from osnip_repro.data import read_text_jsonl
from osnip_repro.modeling import forward_logits, input_embeddings, load_causal_lm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spherical_complement_bound(width: int, epsilon: float) -> float:
    if width < 2:
        raise ValueError("embedding width must be at least 2")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1")
    return min(1.0, 2.0 * math.exp(-((width - 2) * epsilon * epsilon) / 2.0))


def per_example_kl(clean_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
    if clean_logits.shape[-1] != perturbed_logits.shape[-1]:
        raise ValueError("clean and perturbed logits must have the same vocabulary width")
    clean_prob = F.softmax(clean_logits.float(), dim=-1)
    perturbed_log_prob = F.log_softmax(perturbed_logits.float(), dim=-1)
    clean_log_prob = F.log_softmax(clean_logits.float(), dim=-1)
    return (clean_prob * (clean_log_prob - perturbed_log_prob)).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--data-jsonl", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--directions", type=int, default=64)
    ap.add_argument("--direction-chunk", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261018)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.max_samples, args.max_positions, args.directions, args.direction_chunk, args.max_length) < 1:
        raise ValueError("sample, position, direction, chunk and length limits must be positive")
    if args.delta < 0:
        raise ValueError("delta must be non-negative")
    if not args.data_jsonl.is_file():
        raise FileNotFoundError(args.data_jsonl)
    torch.manual_seed(args.seed)
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 7919)
    width = int(model.get_input_embeddings().embedding_dim)
    bound = spherical_complement_bound(width, args.epsilon)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.max_length)
            input_ids = encoded["input_ids"].to(device)
            if input_ids.shape[1] < 1:
                continue
            mask = torch.ones_like(input_ids)
            h = input_embeddings(model, input_ids)
            clean_logits = forward_logits(model, h, mask)
            position_limit = min(int(input_ids.shape[1]), args.max_positions)
            for position in range(position_limit):
                reference = h[0, position]
                radius = reference.float().norm().clamp_min(1e-12)
                reference_unit = reference.float() / radius
                directions = torch.randn((args.directions, width), generator=generator)
                directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                semantic_hits = orthogonal_hits = intersection_hits = 0
                kl_values: list[float] = []
                cosine_values: list[float] = []
                for start in range(0, args.directions, args.direction_chunk):
                    chunk = directions[start:start + args.direction_chunk].to(device)
                    replacements = chunk.to(dtype=h.dtype) * radius.to(dtype=h.dtype)
                    perturbed = h.expand(replacements.shape[0], -1, -1).clone()
                    perturbed[:, position, :] = replacements
                    chunk_mask = mask.expand(replacements.shape[0], -1)
                    logits = forward_logits(model, perturbed, chunk_mask)
                    clean_at_position = clean_logits[:, position, :].expand(replacements.shape[0], -1)
                    kl = per_example_kl(clean_at_position, logits[:, position, :])
                    cosine = torch.abs(torch.matmul(chunk.float(), reference_unit))
                    semantic = kl <= args.delta
                    orthogonal = cosine <= args.epsilon
                    semantic_hits += int(semantic.sum())
                    orthogonal_hits += int(orthogonal.sum())
                    intersection_hits += int((semantic & orthogonal).sum())
                    kl_values.extend(float(value) for value in kl.cpu())
                    cosine_values.extend(float(value) for value in cosine.cpu())
                alpha = semantic_hits / args.directions
                orthogonal_rate = orthogonal_hits / args.directions
                intersection_rate = intersection_hits / args.directions
                rows.append({
                    "sample": sample_index,
                    "position": position,
                    "token_id": int(input_ids[0, position]),
                    "width": width,
                    "directions": args.directions,
                    "delta": args.delta,
                    "epsilon": args.epsilon,
                    "alpha_delta": alpha,
                    "orthogonal_rate": orthogonal_rate,
                    "intersection_rate": intersection_rate,
                    "theorem_bound": bound,
                    "theorem_condition": alpha > bound,
                    "mean_kl": sum(kl_values) / max(len(kl_values), 1),
                    "max_kl": max(kl_values, default=0.0),
                    "mean_abs_cosine": sum(cosine_values) / max(len(cosine_values), 1),
                })

    mean = lambda key: sum(float(row[key]) for row in rows) / max(len(rows), 1)
    report = {
        "format": "osnip_semantic_coverage_diagnostic_v1",
        "status": "completed_with_limits",
        "paper_result": False,
        "theoretical_diagnostic": True,
        "security_accepted": False,
        "key_attack_performed": False,
        "model_path": str(args.model_path.resolve()),
        "data_jsonl": str(args.data_jsonl.resolve()),
        "samples": len(texts),
        "positions": len(rows),
        "width": width,
        "directions_per_position": args.directions,
        "delta": args.delta,
        "epsilon": args.epsilon,
        "theorem_bound": bound,
        "mean_alpha_delta": mean("alpha_delta"),
        "mean_orthogonal_rate": mean("orthogonal_rate"),
        "mean_intersection_rate": mean("intersection_rate"),
        "theorem_condition_rate": sum(bool(row["theorem_condition"]) for row in rows) / max(len(rows), 1),
        "rows": rows,
        "checkpoint": None,
        "data_sha256": sha256(args.data_jsonl),
        "source_sha256": {"scripts/evaluate_semantic_coverage.py": sha256(Path(__file__))},
        "limitations": [
            "Monte Carlo estimate over local input-embedding directions",
            "one-token-at-a-time causal replacement diagnostic",
            "not a proof of OSNIP security and not an attack result",
            "no key attack is performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "positions": len(rows),
        "mean_alpha_delta": report["mean_alpha_delta"],
        "mean_intersection_rate": report["mean_intersection_rate"],
        "theorem_condition_rate": report["theorem_condition_rate"],
        "theorem_bound": bound,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
