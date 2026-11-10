#!/usr/bin/env python3
"""Batch no-key autoregressive vocabulary matching on JSONL prompts.

This is an engineering reproduction of the public hidden-state matching
protocol. It never gives the candidate attack the client key or target IDs.
Full-vocabulary mode is required for an interpretable privacy result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from osnip_repro.data import read_text_jsonl
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import encrypt_embeddings
from osnip_repro.vma import autoregressive_match, hidden_at_layer, model_candidates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empty_metrics() -> dict[str, int]:
    return {"top1": 0, "top5": 0, "top10": 0, "positions": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--data-jsonl", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=2)
    ap.add_argument("--max-positions", type=int, default=1)
    ap.add_argument("--keys-per-prompt", type=int, default=2)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--candidate-chunk", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    if min(args.max_samples, args.max_positions, args.keys_per_prompt, args.candidate_chunk) < 1:
        ap.error("sample, position, key and chunk limits must be positive")
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    if args.layer < 1 or args.layer > model.config.num_hidden_layers:
        raise ValueError("layer must be a decoder layer, 1..num_hidden_layers")
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    candidate_forward = model_candidates(model, layer=args.layer)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    metrics = {"clean": empty_metrics(), "encrypted": empty_metrics()}
    rows = []
    with torch.no_grad():
        for sample_index, text in enumerate(texts):
            token_ids = tokenizer.encode(text, add_special_tokens=False)[:args.max_positions]
            if not token_ids:
                continue
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            h = input_embeddings(model, ids)
            clean_target = hidden_at_layer(model, layer=args.layer, inputs_embeds=h, attention_mask=mask)[0]
            clean_result = autoregressive_match(clean_target, candidate_forward,
                                                vocab_size=model.config.vocab_size,
                                                chunk=args.candidate_chunk, topk=10)
            clean_hits = [int((clean_result["rankings"][i] == token_ids[i]).any()) for i in range(len(token_ids))]
            clean_top1 = [int(clean_result["predictions"][i] == token_ids[i]) for i in range(len(token_ids))]
            clean_top5 = [int((clean_result["rankings"][i, :5] == token_ids[i]).any()) for i in range(len(token_ids))]
            for k, values in ((1, clean_top1), (5, clean_top5), (10, clean_hits)):
                metrics["clean"][f"top{k}"] += sum(values)
            metrics["clean"]["positions"] += len(token_ids)
            rows.append({"sample": sample_index, "key_index": -1, "side": "clean",
                         "positions": len(token_ids), "predictions": clean_result["predictions"],
                         "target_ids_for_scoring_only": token_ids,
                         "top1_hits": sum(clean_top1), "top5_hits": sum(clean_top5),
                         "top10_hits": sum(clean_hits),
                         "searched_candidate_sizes": clean_result["searched_candidate_sizes"]})
            for key_index in range(args.keys_per_prompt):
                key = torch.randn(config["key_dim"], generator=rng).to(device)
                z = encrypt_embeddings(encoder, h, key)
                encrypted_target = hidden_at_layer(model, layer=args.layer, inputs_embeds=z, attention_mask=mask)[0]
                result = autoregressive_match(encrypted_target, candidate_forward,
                                              vocab_size=model.config.vocab_size,
                                              chunk=args.candidate_chunk, topk=10)
                hits = {
                    "top1": [int(result["predictions"][i] == token_ids[i]) for i in range(len(token_ids))],
                    "top5": [int((result["rankings"][i, :5] == token_ids[i]).any()) for i in range(len(token_ids))],
                    "top10": [int((result["rankings"][i] == token_ids[i]).any()) for i in range(len(token_ids))],
                }
                for name, values in hits.items():
                    metrics["encrypted"][name] += sum(values)
                metrics["encrypted"]["positions"] += len(token_ids)
                rows.append({"sample": sample_index, "key_index": key_index, "side": "encrypted",
                             "positions": len(token_ids), "predictions": result["predictions"],
                             "target_ids_for_scoring_only": token_ids,
                             "top1_hits": sum(hits["top1"]), "top5_hits": sum(hits["top5"]),
                             "top10_hits": sum(hits["top10"]),
                             "searched_candidate_sizes": result["searched_candidate_sizes"]})
    if metrics["clean"]["positions"] == 0 or metrics["encrypted"]["positions"] == 0:
        raise ValueError("no attackable positions")
    rates = {side: {key: value / max(metrics[side]["positions"], 1)
                    for key, value in metrics[side].items() if key != "positions"}
             for side in metrics}
    report = {
        "format": "osnip_vocabulary_matching_batch_v1",
        "status": "full_vocabulary_diagnostic",
        "paper_result": False,
        "security_accepted": False,
        "attack_scope": "passive_no_key_autoregressive_hidden_state_matching",
        "key_attack_performed": False,
        "model_path": str(args.model_path.resolve()), "data_jsonl": str(args.data_jsonl.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "layer": args.layer,
        "samples": len(texts), "max_positions": args.max_positions, "keys_per_prompt": args.keys_per_prompt,
        "counts": metrics, "rates": rates, "rows": rows,
        "positive_control_valid": all(rates["clean"][key] >= 0.95 for key in ("top1", "top5", "top10")),
        "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(args.data_jsonl),
        "source_sha256": {"scripts/evaluate_vocabulary_matching_batch.py": sha256(Path(__file__)),
                          "src/osnip_repro/vma.py": sha256(ROOT / "src/osnip_repro/vma.py")},
        "limitations": ["local Qwen2.5-0.5B proxy, not the paper's Qwen3-14B/32B", "small prompt/key sample",
                        "full-vocabulary CPU diagnostic; no confidence interval", "does not evaluate key attacks"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "rates": rates,
                      "positive_control_valid": report["positive_control_valid"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
