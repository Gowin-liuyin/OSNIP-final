#!/usr/bin/env python3
"""Attribute the KNN gap: per-position signed cosine vs Top-k hit rate.

The paper's Figure 6a/D.2 shows KNN ASR collapsing once cos(h,z) < ~0.3.
This diagnostic reproduces that curve for the evaluated checkpoint and,
unlike the previous version, keeps negative cosines in their own bucket
instead of merging them into [0.0, 0.1). That distinction matters because
the paper's KNN hinge is on |cos| while the attack threshold is on the
signed cosine: at signed cos <= 0.2 the real-vocabulary ASR is zero, at
cos ~0.25 it is already ~24% at Top-10 (reports/diagnostics/knn_cos_probe.json).

Usage (remote):
    PYTHONPATH=src .venv/bin/python scripts/diagnose_knn_coverage_buckets.py \
        --model-path <model> --checkpoint <best_pareto.pt> \
        --knn-jsonl data/paper_attack_sets/knn.jsonl \
        --output queue2/eval/knn_coverage_buckets.json --device cuda --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.attacks import vocabulary_topk
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256, read_text_jsonl
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import encrypt_embeddings

# (low, high, label); low/high are signed cosine bounds, the first bucket is open below.
BUCKETS = ((None, 0.0, "<0.00"), (0.0, 0.1, "0.00-0.10"), (0.1, 0.2, "0.10-0.20"),
           (0.2, 0.3, "0.20-0.30"), (0.3, 0.4, "0.30-0.40"), (0.4, 0.5, "0.40-0.50"),
           (0.5, 0.6, "0.50-0.60"), (0.6, 0.8, "0.60-0.80"), (0.8, 1.01, "0.80-1.01"))


def bucket_of(value: float):
    for low, high, label in BUCKETS:
        if (low is None or value >= low) and value < high:
            return label
    return BUCKETS[-1][2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--knn-jsonl", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    if Path(str(config.get("model_path", ""))).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint config and --model-path disagree")
    epsilon = float(config.get("epsilon", 0.3))
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    vocab = model.get_input_embeddings().weight.detach()
    texts = read_text_jsonl(args.knn_jsonl, limit=args.max_samples)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    buckets = {label: {"positions": 0, "top1": 0, "top5": 0, "top10": 0} for _, _, label in BUCKETS}
    signs = {"positive": {"positions": 0, "top1": 0, "top5": 0, "top10": 0},
             "nonpositive": {"positions": 0, "top1": 0, "top5": 0, "top10": 0}}
    total = 0
    abs_covered = signed_covered = positive = 0
    cosines = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if not ids:
                continue
            id_tensor = torch.tensor([ids], device=device)
            h = input_embeddings(model, id_tensor)
            key = torch.randn(config["key_dim"], generator=rng, device="cpu").to(device)
            z = encrypt_embeddings(encoder, h, key)
            cos = torch.nn.functional.cosine_similarity(h[0].float(), z[0].float(), dim=-1).cpu()
            rankings = vocabulary_topk(z[0].float(), vocab.float(), metric="l2", k=10, chunk=2048).cpu()
            targets = torch.tensor(ids)
            for position in range(len(ids)):
                value = float(cos[position])
                total += 1
                cosines.append(value)
                abs_covered += int(abs(value) <= epsilon)
                signed_covered += int(value <= epsilon)
                positive += int(value > 0)
                hit = {f"top{k}": int((rankings[position, :k] == targets[position]).any()) for k in (1, 5, 10)}
                row = buckets[bucket_of(value)]
                row["positions"] += 1
                for k in (1, 5, 10):
                    row[f"top{k}"] += hit[f"top{k}"]
                sign_row = signs["positive" if value > 0 else "nonpositive"]
                sign_row["positions"] += 1
                for k in (1, 5, 10):
                    sign_row[f"top{k}"] += hit[f"top{k}"]

    def rates(row):
        n = max(row["positions"], 1)
        return {**{k: v for k, v in row.items() if k != "positions"},
                **{f"top{k}_rate": row[f"top{k}"] / n for k in (1, 5, 10)}}

    cosines_tensor = torch.tensor(cosines, dtype=torch.float64)
    report = {
        "format": "osnip_knn_coverage_buckets_v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model_path": str(args.model_path.resolve()),
        "knn_jsonl": str(args.knn_jsonl.resolve()),
        "data_sha256": file_sha256(args.knn_jsonl),
        "seed": args.seed, "dtype": args.dtype, "device": str(device),
        "epsilon": epsilon,
        "positions": total,
        "coverage_abs_cos_le_epsilon": abs_covered / max(total, 1),
        "coverage_signed_cos_le_epsilon": signed_covered / max(total, 1),
        "positive_cosine_fraction": positive / max(total, 1),
        "mean_signed_cosine": float(cosines_tensor.mean()),
        "mean_abs_cosine": float(cosines_tensor.abs().mean()),
        "signed_cosine_quantiles": {q: float(cosines_tensor.quantile(float(q)))
                                    for q in ("0.1", "0.5", "0.9")},
        "buckets": {label: rates(row) for label, row in buckets.items()},
        "asr_by_sign": {name: rates(row) for name, row in signs.items()},
        "interpretation": ("ASR concentrated in the positive-cosine tail means a training operating-point "
                           "outcome, not an attack bug; the paper's 2% 3B result requires essentially no mass "
                           "above signed cos ~0.2"),
        "source_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/diagnose_knn_coverage_buckets.py", "src/osnip_repro/attacks.py",
            "src/osnip_repro/encoder.py", "src/osnip_repro/projection.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"positions": total, "mean_signed_cosine": report["mean_signed_cosine"],
                      "positive_fraction": report["positive_cosine_fraction"],
                      "coverage_abs": report["coverage_abs_cos_le_epsilon"],
                      "coverage_signed": report["coverage_signed_cos_le_epsilon"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
