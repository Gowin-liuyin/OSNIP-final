#!/usr/bin/env python3
"""Run a short objective smoke and emit an auditable OSNIP Gate 1 report."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.encoder import ARCHITECTURES, build_encryptor_from_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--train-jsonl", type=Path, default=ROOT / "data/osnip_65k/train.jsonl")
    ap.add_argument("--valid-jsonl", type=Path, default=ROOT / "data/osnip_65k/valid.jsonl")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=32)
    ap.add_argument("--max-valid-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--evaluate-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--encryptor-architecture", choices=ARCHITECTURES, default="mlp_residual_v1")
    ap.add_argument("--diversity-space", choices=("projected_z", "raw_r"), default="projected_z")
    return ap.parse_args()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "gate1_report.json"
    if args.steps < 1 or args.max_train_samples < 1 or args.max_valid_samples < 1:
        raise ValueError("steps and sample limits must be positive")
    run_dir = args.output_dir / "training_smoke"
    if report_path.exists() or run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Gate 1 output exists; choose a new output directory")
    command = [
        sys.executable, str(ROOT / "scripts/train_streaming.py"),
        "--model-path", str(args.model_path),
        "--train-jsonl", str(args.train_jsonl),
        "--valid-jsonl", str(args.valid_jsonl),
        "--output-dir", str(run_dir),
        "--steps", str(args.steps),
        "--max-train-samples", str(args.max_train_samples),
        "--max-valid-samples", str(args.max_valid_samples),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--max-length", str(args.max_length),
        "--learning-rate", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--checkpoint-every", str(args.checkpoint_every),
        "--evaluate-every", str(args.evaluate_every),
        "--constraint-mode", "gated",
        "--lambda-util", "1.0", "--lambda-priv", "1.0", "--lambda-div", "1.0",
        "--encryptor-architecture", args.encryptor_architecture,
        "--diversity-space", args.diversity_space,
        "--seed", str(args.seed), "--device", args.device, "--dtype", args.dtype,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    training_report_path = run_dir / "report.json"
    checks: dict[str, dict[str, object]] = {}
    if not training_report_path.is_file():
        checks["training"] = {"passed": False, "detail": "training report was not produced", "exit_code": completed.returncode, "stderr_tail": completed.stderr[-2000:]}
        report = {"format": "osnip_gate1_report_v1", "status": "failed", "model_path": str(args.model_path.resolve()), "command": command, "checks": checks}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "failed", "output": str(report_path.resolve()), "checks": checks}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    training = json.loads(training_report_path.read_text())
    validation = training.get("last_validation", {})
    finite_fields = ("utility", "privacy", "diversity", "clean_ppl", "encrypted_ppl", "ppl_ratio", "mean_cosine_key1", "mean_cosine_key2")
    finite_ok = all(finite(validation.get(field)) for field in finite_fields)
    checks["training"] = {"passed": completed.returncode == 0 and training.get("status") == "completed_with_limits", "exit_code": completed.returncode, "steps_completed": training.get("steps_completed")}
    checks["finite_objective"] = {"passed": finite_ok, "fields": {field: validation.get(field) for field in finite_fields}}
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    dtype_name = str(config.get("dtype", "float32"))
    embedding_dtype = getattr(torch, dtype_name)
    relative_tolerance = max(1e-5, 2 * torch.finfo(embedding_dtype).eps)
    checks["iso_norm_projection"] = {
        "passed": finite(validation.get("norm_max_relative_error"))
            and validation["norm_max_relative_error"] <= relative_tolerance
            and finite(validation.get("projection_fp32_max_relative_error"))
            and validation["projection_fp32_max_relative_error"] <= 1e-5,
        "max_abs_error": validation.get("norm_max_abs_error"),
        "max_relative_error": validation.get("norm_max_relative_error"),
        "relative_tolerance": relative_tolerance,
        "fp32_projection_max_relative_error": validation.get("projection_fp32_max_relative_error"),
        "fp32_relative_tolerance": 1e-5,
        "dtype": dtype_name,
    }
    checks["distinct_key_draws"] = {"passed": validation.get("keys_distinct_fraction") == 1.0}
    checks["key_schedule"] = {"passed": config.get("key_schedule") == "independent_key_per_example_shared_across_tokens", "observed": config.get("key_schedule")}
    required = ("last.pt", "best_utility.pt", "best_pareto.pt", "source_hashes.json", "model_manifest.json", "environment.json", "initial_encoder.pt", "initial_validation.json")
    checks["checkpoint_artifacts"] = {"passed": all((run_dir / name).is_file() for name in required), "missing": [name for name in required if not (run_dir / name).is_file()]}
    checkpoint_ok = False
    checkpoint_failure = None
    try:
        payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        tensors = payload.get("encoder", {})
        checkpoint_ok = (payload.get("format") == "osnip_training_checkpoint_v1" and bool(tensors)
            and all(isinstance(value, torch.Tensor) for value in tensors.values())
            and all(torch.isfinite(value).all().item() for value in tensors.values()))
    except Exception as exc:
        checkpoint_failure = f"{type(exc).__name__}: {exc}"
    checks["checkpoint_finite"] = {"passed": checkpoint_ok, "failure": checkpoint_failure}
    if checkpoint_ok:
        initial = torch.load(run_dir / "initial_encoder.pt", map_location="cpu", weights_only=False)
        changed = any(not torch.equal(initial[name], tensor) for name, tensor in tensors.items())
        checks["optimizer_changed_parameters"] = {"passed": changed}
        restored = build_encryptor_from_config(config).eval()
        restored.load_state_dict(tensors)
        exported = build_encryptor_from_config(config).eval()
        exported.load_state_dict(torch.load(run_dir / "encryptor.pt", map_location="cpu", weights_only=False))
        generator = torch.Generator().manual_seed(72109)
        # Probe in the restored module's own dtype: fp16/bf16 checkpoints carry
        # half-precision tensors and would otherwise crash the roundtrip matmuls.
        probe_dtype = next(restored.parameters()).dtype
        h = torch.randn(2, 3, config["width"], generator=generator, dtype=probe_dtype)
        keys = torch.randn(2, config["key_dim"], generator=generator, dtype=probe_dtype)
        with torch.no_grad():
            actual = restored(h, keys)
            expected = exported(h, keys)
            per_example = torch.stack([restored(h[i], keys[i]) for i in range(2)])
        checks["checkpoint_output_roundtrip"] = {"passed": torch.equal(actual, expected), "max_abs_difference": float((actual - expected).abs().max())}
        checks["per_example_key_behavior"] = {"passed": torch.allclose(actual, per_example, atol=1e-6, rtol=1e-5)}

    status = "ready" if all(bool(check.get("passed")) for check in checks.values()) else "failed"
    report = {"format": "osnip_gate1_report_v1", "status": status, "model_path": str(args.model_path.resolve()), "training_run": str(run_dir.resolve()), "command": command, "checks": checks, "limitations": ["Gate 1 verifies engineering objective smoke only", "not a utility or privacy acceptance result", "no key attack is in scope"]}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "output": str(report_path.resolve()), "checks": checks}, ensure_ascii=False, indent=2))
    if status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
