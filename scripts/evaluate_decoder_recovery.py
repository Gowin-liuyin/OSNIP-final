#!/usr/bin/env python3
"""Measure OSNIP's claimed shallow-obfuscation/deep-recovery trajectory."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import file_sha256, read_text_jsonl, tokenize_texts
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import forward_logits_and_hidden_states, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--data-jsonl", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, action="append", required=True)
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--keys-per-prompt", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 5, 10, 15, 16])
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def row_metrics(clean_logits: torch.Tensor, encrypted_logits: torch.Tensor,
                clean_hidden: tuple[torch.Tensor, ...], encrypted_hidden: tuple[torch.Tensor, ...],
                input_layers: tuple[int, ...], input_ids: torch.Tensor, mask: torch.Tensor) -> dict:
    lengths = mask.sum(dim=1).detach().cpu().tolist()
    valid = mask[:, 1:].bool() & mask[:, :-1].bool()
    clean_logp = F.log_softmax(clean_logits[:, :-1].float(), dim=-1)
    encrypted_logp = F.log_softmax(encrypted_logits[:, :-1].float(), dim=-1)
    clean_prob = clean_logp.exp()
    token_kl = (clean_prob * (clean_logp - encrypted_logp)).sum(-1)
    tokens = int(valid.sum().detach().cpu())
    if tokens == 0:
        raise ValueError("decoder recovery requires at least one valid causal target")
    kl_sum = float(token_kl[valid].sum().detach().cpu())
    labels = input_ids[:, 1:, None]
    clean_nll_sum = float((-clean_logp.gather(-1, labels).squeeze(-1))[valid].sum().detach().cpu())
    encrypted_nll_sum = float((-encrypted_logp.gather(-1, labels).squeeze(-1))[valid].sum().detach().cpu())
    values = {}
    for layer, clean, encrypted in zip(input_layers, clean_hidden, encrypted_hidden):
        cosine = F.cosine_similarity(clean.float(), encrypted.float(), dim=-1)
        # Layer statistics must cover exactly the causal valid positions that
        # the KL/NLL denominators use; position 0 and padding predecessors are
        # not causal targets and would mislabel the causal_token_weighted claim.
        flat = cosine[:, :-1][valid]
        values[str(layer)] = {
            "mean_cosine": float(flat.mean().detach().cpu()),
            "mean_abs_cosine": float(flat.abs().mean().detach().cpu()),
            "min_cosine": float(flat.min().detach().cpu()),
            "max_cosine": float(flat.max().detach().cpu()),
            "tokens": int(flat.numel()),
        }
    return {"kl": kl_sum / tokens, "kl_sum": kl_sum, "clean_nll_sum": clean_nll_sum,
            "encrypted_nll_sum": encrypted_nll_sum, "layers": values, "tokens": tokens,
            "lengths": [int(x) for x in lengths]}


def summarize_rows(rows: list[dict], layers: list[int]) -> dict:
    tokens = sum(row["tokens"] for row in rows)
    if tokens == 0:
        raise ValueError("no causal tokens in decoder recovery observations")
    clean_nll = sum(row["clean_nll_sum"] for row in rows) / tokens
    encrypted_nll = sum(row["encrypted_nll_sum"] for row in rows) / tokens
    summary = {"kl": sum(row["kl_sum"] for row in rows) / tokens,
               "aggregation": "causal_token_weighted", "causal_tokens": tokens,
               "clean_ppl": math.exp(clean_nll), "encrypted_ppl": math.exp(encrypted_nll),
               "ppl_ratio": math.exp(encrypted_nll - clean_nll), "layers": {}}
    for layer in layers:
        layer_rows = [row["layers"][str(layer)] for row in rows]
        count = sum(row["tokens"] for row in layer_rows)
        summary["layers"][str(layer)] = {
            key: sum(item[key] * item["tokens"] for item in layer_rows) / count
            for key in ("mean_cosine", "mean_abs_cosine")}
    return summary


def main() -> None:
    args = parse_args()
    if min(args.max_samples, args.keys_per_prompt, args.max_length) < 1:
        raise ValueError("sample, key and length limits must be positive")
    if len(set(args.layers)) != len(args.layers) or any(layer < 0 for layer in args.layers):
        raise ValueError("layers must be unique and nonnegative")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    config = json.loads(args.config.read_text())
    if Path(str(config.get("model_path", ""))).resolve() != args.model_path.resolve():
        raise ValueError("config and --model-path disagree; refusing cross-run diagnostic")
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    reports = []
    for checkpoint in args.checkpoint:
        rng = torch.Generator(device="cpu").manual_seed(args.seed)
        before = file_sha256(checkpoint)
        encoder = build_encryptor_from_config(config).to(device).eval()
        encoder.load_state_dict(load_encoder_state_dict(str(checkpoint), map_location=device, expected_config=config))
        rows = []
        with torch.inference_mode():
            for prompt_index, text in enumerate(texts):
                batch = tokenize_texts(tokenizer, [text], max_length=args.max_length)
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device).bool()
                h = input_embeddings(model, input_ids)
                clean_logits, clean_hidden = forward_logits_and_hidden_states(model, h, mask, tuple(args.layers))
                for key_index in range(args.keys_per_prompt):
                    key = torch.randn(config["key_dim"], generator=rng).to(device)
                    z = iso_norm_project(encoder(h, key, attention_mask=mask), h)
                    encrypted_logits, encrypted_hidden = forward_logits_and_hidden_states(model, z, mask, tuple(args.layers))
                    metrics = row_metrics(clean_logits, encrypted_logits, clean_hidden, encrypted_hidden, tuple(args.layers), input_ids, mask)
                    rows.append({"prompt": prompt_index, "key": key_index, **metrics})
        if file_sha256(checkpoint) != before:
            raise RuntimeError(f"checkpoint changed while loading: {checkpoint}")
        summary = summarize_rows(rows, args.layers)
        reports.append({"checkpoint": str(checkpoint.resolve()), "sha256": before, "effective_alpha_scale": float(encoder.alpha_scale.detach().cpu()) if hasattr(encoder, "alpha_scale") else None, "summary": summary, "rows": rows})
        print(json.dumps({"checkpoint": str(checkpoint), "summary": summary}, ensure_ascii=False), flush=True)
    output = {
        "format": "osnip_decoder_recovery_diagnostic_v2",
        "security_accepted": False,
        "key_attack_performed": False,
        "threat_model": "passive_no_key_semi_honest_server",
        "model_path": str(args.model_path.resolve()),
        "data_jsonl": str(args.data_jsonl.resolve()),
        "data_sha256": file_sha256(args.data_jsonl),
        "config_sha256": file_sha256(args.config),
        "source_sha256": {"scripts/evaluate_decoder_recovery.py": file_sha256(Path(__file__)),
                           "src/osnip_repro/checkpointing.py": file_sha256(ROOT / "src/osnip_repro/checkpointing.py"),
                           "src/osnip_repro/encoder.py": file_sha256(ROOT / "src/osnip_repro/encoder.py")},
        "common_key_draws_across_checkpoints": True,
        "layers": args.layers,
        "samples": len(texts),
        "keys_per_prompt": args.keys_per_prompt,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "reports": reports,
        "limitations": [
            "Diagnostic hidden-state trajectory; no KNN/VMA security acceptance.",
            "Fixed prompts and finite key draws; no key inference or key attack.",
            "The decoder is evaluated frozen and the encryptor is client-side.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
