#!/usr/bin/env python3
"""Evaluate clean/encrypted PPL with non-overlapping targets in sliding windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers
from osnip_repro.data import read_text_jsonl
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm, sliding_window_nll
from osnip_repro.projection import iso_norm_project, encrypt_embeddings


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--data-jsonl", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--paper-clean-ppl", type=float, default=None,
                    help="paper Table 2 clean PPL for this model (3B 9.63, 1B 8.63); adds a paper-baseline ratio")
    args = ap.parse_args()
    if args.max_length < 2 or args.stride < 1 or args.stride >= args.max_length:
        ap.error("require max-length >= 2 and 1 <= stride < max-length")
    if args.paper_clean_ppl is not None and args.paper_clean_ppl <= 0:
        ap.error("--paper-clean-ppl must be positive")
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    clean_nll = encrypted_nll = 0.0
    clean_tokens = encrypted_tokens = 0
    cosine_sum = cosine_count = 0
    lengths = []
    started = time.monotonic()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            h = input_embeddings(model, ids)
            key = torch.randn(config["key_dim"], generator=rng).to(device)
            z = encrypt_embeddings(encoder, h, key, attention_mask=torch.ones_like(ids))
            clean_value, clean_count = sliding_window_nll(model, h, ids, max_length=args.max_length, stride=args.stride)
            encrypted_value, encrypted_count = sliding_window_nll(model, z, ids, max_length=args.max_length, stride=args.stride)
            clean_nll += clean_value
            encrypted_nll += encrypted_value
            clean_tokens += clean_count
            encrypted_tokens += encrypted_count
            cosine_sum += float(torch.nn.functional.cosine_similarity(h.float(), z.float(), dim=-1).sum())
            cosine_count += int(h.shape[1])
            lengths.append(int(ids.shape[1]))
            progress = {"stage": "sliding_ppl", "samples_completed": len(lengths),
                        "samples_target": len(texts), "target_tokens": clean_tokens,
                        "clean_ppl": math.exp(clean_nll / clean_tokens),
                        "encrypted_ppl": math.exp(encrypted_nll / encrypted_tokens),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "security_accepted": False}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
    if not clean_tokens or clean_tokens != encrypted_tokens:
        raise ValueError("clean/encrypted sliding-window target counts disagree")
    clean_mean = clean_nll / clean_tokens
    encrypted_mean = encrypted_nll / encrypted_tokens
    clean_ppl = math.exp(clean_mean)
    encrypted_ppl = math.exp(encrypted_mean)
    paper_ratio = (encrypted_ppl / args.paper_clean_ppl
                   if args.paper_clean_ppl and args.paper_clean_ppl > 0 else None)
    report = {
        "format": "osnip_sliding_window_ppl_v1",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "status": "completed_with_limits",
        "paper_result": False,
        "security_accepted": False,
        "model_path": str(args.model_path.resolve()), "data_jsonl": str(args.data_jsonl.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "samples": len(lengths),
        "sequence_token_lengths": lengths, "target_tokens": clean_tokens,
        "max_length": args.max_length, "stride": args.stride,
        "seed": args.seed, "device": str(device), "dtype": args.dtype,
        "key_schedule": "one CPU-generator key per text, moved to the model device",
        "clean_ppl": clean_ppl, "encrypted_ppl": encrypted_ppl,
        "ppl_ratio": math.exp(encrypted_mean - clean_mean),
        "paper_clean_ppl": args.paper_clean_ppl,
        "clean_ppl_vs_paper_clean": (clean_ppl / args.paper_clean_ppl
                                     if args.paper_clean_ppl and args.paper_clean_ppl > 0 else None),
        "ppl_ratio_vs_paper_clean": paper_ratio,
        "mean_cosine": cosine_sum / max(cosine_count, 1),
        "checkpoint_sha256": sha256(args.checkpoint), "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").is_file() else None, "data_sha256": sha256(args.data_jsonl),
        "source_sha256": {"scripts/evaluate_sliding_ppl.py": sha256(Path(__file__)),
                          "src/osnip_repro/modeling.py": sha256(ROOT / "src/osnip_repro/modeling.py")},
        "limitations": ["PPL utility only; no privacy attack in this report", "one client RNG seed",
                        "project ratio uses the local clean baseline; paper_ratio uses the paper's reported clean PPL when supplied"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("output" ,) if k in report} | {
        "report": str(args.output.resolve()), "samples": report["samples"],
        "target_tokens": report["target_tokens"], "clean_ppl": report["clean_ppl"],
        "encrypted_ppl": report["encrypted_ppl"], "ppl_ratio": report["ppl_ratio"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
