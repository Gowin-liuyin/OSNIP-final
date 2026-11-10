#!/usr/bin/env python3
"""Evaluate no-key vocabulary matching against clean and OSNIP embeddings.

This is an embedding-level attack. It is a required positive-control probe,
but it is not the paper's full autoregressive hidden-state attack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from osnip_repro.data import read_text_jsonl, tokenize_texts
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import encrypt_embeddings


def _topk_hits(query: torch.Tensor, embedding: torch.Tensor, ids: torch.Tensor, *, metric: str, topk: tuple[int, ...], chunk: int) -> dict[int, int]:
    from osnip_repro.attacks import vocabulary_topk, topk_counts
    rankings = vocabulary_topk(query, embedding, metric=metric, k=max(topk), chunk=chunk)
    return topk_counts(rankings, ids, topk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-positions", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-chunk", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if min(args.max_samples, args.max_length, args.max_positions, args.batch_size, args.vocab_chunk) < 1:
        raise ValueError("positive sample/length/position/batch/chunk limits required")
    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if Path(str(config.get("model_path", ""))).resolve() != Path(args.model_path).resolve():
        raise ValueError("checkpoint config and --model-path disagree; refusing cross-run evaluation")
    tokenizer, model, device = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device)
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    encoder.eval()
    vocab = model.get_input_embeddings().weight.detach()
    if vocab.shape[1] != config.get("width"):
        raise ValueError("model embedding width disagrees with checkpoint config")
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    topk = (1, 5, 10)
    totals = {"clean": {m: {k: 0 for k in topk} for m in ("l2", "l1")}, "encrypted": {m: {k: 0 for k in topk} for m in ("l2", "l1")}}
    total = 0
    cosine_sum = 0.0
    positions_left = args.max_positions
    for start in range(0, len(texts), args.batch_size):
        batch = tokenize_texts(tokenizer, texts[start : start + args.batch_size], max_length=args.max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"].bool()
        with torch.no_grad():
            h = input_embeddings(model, ids)
            key = torch.randn(ids.shape[0], config["key_dim"]).to(device)
            z = encrypt_embeddings(encoder, h, key)
        active = mask.reshape(-1)
        h_rows = h.reshape(-1, h.shape[-1])[active]
        z_rows = z.reshape(-1, z.shape[-1])[active]
        id_rows = ids.reshape(-1)[active]
        if h_rows.shape[0] > positions_left:
            h_rows, z_rows, id_rows = h_rows[:positions_left], z_rows[:positions_left], id_rows[:positions_left]
        positions_left -= h_rows.shape[0]
        if h_rows.numel() == 0:
            continue
        for name, query in (("clean", h_rows), ("encrypted", z_rows)):
            for metric in ("l2", "l1"):
                hits = _topk_hits(query, vocab, id_rows, metric=metric, topk=topk, chunk=args.vocab_chunk)
                for k, value in hits.items():
                    totals[name][metric][k] += value
        cosine_sum += float(torch.nn.functional.cosine_similarity(h_rows.float(), z_rows.float(), dim=-1).sum())
        total += h_rows.shape[0]
        if positions_left <= 0:
            break
    report = {
        "format": "osnip_no_key_embedding_attack_report_v1",
        "status": "completed_with_limits",
        "attack_scope": "embedding_level_no_key_positive_control_probe",
        "paper_result": False,
        "samples": len(texts),
        "positions": total,
        "metrics": {name: {metric: {f"top{k}": totals[name][metric][k] / max(total, 1) for k in topk} for metric in ("l2", "l1")} for name in totals},
        "mean_cosine": cosine_sum / max(total, 1),
        "limitations": ["embedding-level matching only", "does not implement the paper's autoregressive hidden-state vocabulary attack", "no independent holdout or confidence interval"],
    }
    report["positive_controls_valid"] = all(report["metrics"]["clean"][m]["top1"] >= 0.95 for m in ("l2", "l1"))
    report["model_path"] = str(Path(args.model_path).resolve())
    report["data_jsonl"] = str(Path(args.data_jsonl).resolve())
    report["checkpoint"] = str(args.checkpoint)
    report["seed"] = args.seed
    import hashlib
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report["data_sha256"] = hashlib.sha256(Path(args.data_jsonl).read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
