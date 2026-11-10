#!/usr/bin/env python3
"""Validate a local causal LM before starting OSNIP training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers
from osnip_repro.modeling import forward_logits, input_embeddings, load_causal_lm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--text", default="The capital of France is Paris.")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    if not args.model_path.is_dir():
        report = {
            "format": "osnip_model_environment_manifest_v1",
            "status": "failed",
            "model_path": str(args.model_path.resolve()),
            "failure": {"type": "missing_model_directory", "detail": f"missing: {args.model_path}"},
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(args.output.resolve()), "status": "failed", "reason": report["failure"]}, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask", torch.ones_like(ids)).to(device)
    with torch.no_grad():
        direct = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        embeddings = input_embeddings(model, ids)
        via_embeddings = forward_logits(model, embeddings, mask)
    max_difference = float((direct - via_embeddings).abs().max())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    probe = embeddings.detach().clone().requires_grad_(True)
    probe_logits = forward_logits(model, probe, mask)
    probe_loss = probe_logits[..., : min(1024, probe_logits.shape[-1])].float().square().mean()
    probe_loss.backward()
    finite = all(torch.isfinite(t).all() for t in (direct, via_embeddings, probe, probe.grad))
    model_files = {path.name: sha256(path) for path in sorted(args.model_path.iterdir()) if path.is_file()}
    report = {
        "format": "osnip_model_environment_manifest_v1",
        "status": "ready" if finite and not trainable and max_difference <= args.tolerance else "failed",
        "model_path": str(args.model_path.resolve()),
        "model_config": model.config.to_dict(),
        "tokenizer": {"class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id},
        "device": str(device), "dtype": str(next(model.parameters()).dtype),
        "direct_vs_inputs_embeds_max_abs_difference": max_difference,
        "tolerance": args.tolerance,
        "all_model_parameters_frozen": not trainable,
        "trainable_parameter_names": trainable,
        "input_gradient_finite": bool(torch.isfinite(probe.grad).all()),
        "logits_finite": finite,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "mps_available": torch.backends.mps.is_available()},
        "model_file_sha256": model_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "max_difference": max_difference, "input_gradient_finite": report["input_gradient_finite"]}, ensure_ascii=False, indent=2))
    if report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
