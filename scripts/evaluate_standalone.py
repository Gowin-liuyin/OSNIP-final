#!/usr/bin/env python3
"""Evaluate clean versus OSNIP-encrypted causal-LM logits on JSONL text."""
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
from osnip_repro.modeling import causal_ppl, forward_logits, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    tokenizer, model, device = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device)
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    encoder.eval()
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    clean_nll, encrypted_nll, count = 0.0, 0.0, 0
    cosines: list[float] = []
    for start in range(0, len(texts), args.batch_size):
        batch = tokenize_texts(tokenizer, texts[start : start + args.batch_size], max_length=args.max_length)
        batch = {key: value.to(device) for key, value in batch.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        with torch.no_grad():
            h = input_embeddings(model, ids)
            clean_logits = forward_logits(model, h, mask)
            key = torch.randn(ids.shape[0], config["key_dim"], device=device)
            z = iso_norm_project(encoder(h, key, attention_mask=mask), h)
            encrypted_logits = forward_logits(model, z, mask)
            clean_nll += float(causal_ppl(clean_logits, ids, mask)) * int(mask[:, 1:].sum())
            encrypted_nll += float(causal_ppl(encrypted_logits, ids, mask)) * int(mask[:, 1:].sum())
            count += int(mask[:, 1:].sum())
            cosines.extend(torch.nn.functional.cosine_similarity(h.float(), z.float(), dim=-1)[mask.bool()].cpu().tolist())
    report = {
        "format": "osnip_standalone_evaluation_report_v1",
        "status": "completed_with_limits",
        "paper_result": False,
        "tokens": count,
        "clean_ppl": float(torch.exp(torch.tensor(clean_nll / max(count, 1)))),
        "encrypted_ppl": float(torch.exp(torch.tensor(encrypted_nll / max(count, 1)))),
        "mean_cosine": sum(cosines) / max(len(cosines), 1),
        "limitations": ["single local model/data split", "no KNN or vocabulary-matching attack in this evaluator", "no independent holdout"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
