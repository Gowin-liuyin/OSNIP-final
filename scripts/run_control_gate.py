#!/usr/bin/env python3
"""Run one auditable OSNIP utility-only or privacy-only control gate."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.encoder import ARCHITECTURES
from osnip_repro.validation import control_acceptance


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("utility_only", "privacy_only"), required=True)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--train-jsonl", type=Path, default=ROOT / "data/osnip_65k/train.jsonl")
    ap.add_argument("--valid-jsonl", type=Path, default=ROOT / "data/osnip_65k/valid.jsonl")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-valid-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--evaluate-every", type=int, default=100)
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
    if args.steps < 1 or args.max_valid_samples < 1:
        raise ValueError("steps and max-valid-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "training"
    if (args.output_dir / f"{args.mode}_gate_report.json").exists() or run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("control output exists; choose a new output directory")
    if args.mode == "utility_only":
        constraint_mode, lambda_util, lambda_priv, lambda_div = "off", "1.0", "0.0", "0.0"
    else:
        constraint_mode, lambda_util, lambda_priv, lambda_div = "constant", "0.0", "1.0", "1.0"
    command = [
        sys.executable, str(ROOT / "scripts/train_streaming.py"),
        "--model-path", str(args.model_path), "--train-jsonl", str(args.train_jsonl), "--valid-jsonl", str(args.valid_jsonl),
        "--output-dir", str(run_dir), "--steps", str(args.steps), "--max-valid-samples", str(args.max_valid_samples),
        "--batch-size", str(args.batch_size), "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--max-length", str(args.max_length), "--learning-rate", str(args.learning_rate), "--warmup-steps", str(args.warmup_steps),
        "--checkpoint-every", str(args.checkpoint_every), "--evaluate-every", str(args.evaluate_every),
        "--constraint-mode", constraint_mode, "--lambda-util", lambda_util, "--lambda-priv", lambda_priv, "--lambda-div", lambda_div,
        "--encryptor-architecture", args.encryptor_architecture,
        "--diversity-space", args.diversity_space,
        "--seed", str(args.seed), "--device", args.device, "--dtype", args.dtype,
    ]
    if args.max_train_samples is not None:
        command += ["--max-train-samples", str(args.max_train_samples)]
    completed = subprocess.run(command, capture_output=True, text=True)
    training_path = run_dir / "report.json"
    checks: dict[str, object] = {}
    if not training_path.is_file():
        checks["training"] = {"passed": False, "exit_code": completed.returncode, "stderr_tail": completed.stderr[-2000:]}
    else:
        training = json.loads(training_path.read_text())
        validation = training.get("last_validation", {})
        common_fields = ("utility", "privacy", "diversity", "ppl_ratio", "mean_cosine_key1", "mean_cosine_key2")
        checks["training"] = {"passed": completed.returncode == 0 and training.get("status") == "completed_with_limits", "exit_code": completed.returncode, "steps_completed": training.get("steps_completed")}
        checks["finite_metrics"] = {"passed": all(finite(validation.get(field)) for field in common_fields), "metrics": {field: validation.get(field) for field in common_fields}}
        checks["control_objective"] = control_acceptance(args.mode, training.get("baseline_validation", {}), validation)
    status = "ready" if checks and all(bool(value.get("passed")) for value in checks.values() if isinstance(value, dict)) else "failed"
    report = {"format": "osnip_control_gate_report_v1", "status": status, "mode": args.mode, "model_path": str(args.model_path.resolve()), "training_run": str(run_dir.resolve()), "command": command, "checks": checks, "security_accepted": False, "limitations": ["control gate is diagnostic and does not establish privacy or security", "full 65k data and formal seeds are required for final conclusions", "no key attack is in scope"]}
    output = args.output_dir / f"{args.mode}_gate_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "mode": args.mode, "output": str(output.resolve()), "checks": checks}, ensure_ascii=False, indent=2))
    if status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
