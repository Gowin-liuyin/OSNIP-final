#!/usr/bin/env python3
"""Run the diagnostic privacy-injection then utility-recovery OSNIP recipe."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.encoder import ARCHITECTURES
TRAINER = ROOT / "scripts/train_streaming.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--train-jsonl", required=True, type=Path)
    ap.add_argument("--valid-jsonl", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--phase1-steps", type=int, default=3000)
    ap.add_argument("--total-steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-valid-samples", type=int, default=256)
    ap.add_argument("--key-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--tau-low", type=float, default=0.01)
    ap.add_argument("--tau-high", type=float, default=0.10)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--diversity-margin", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--evaluate-every", type=int, default=100)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--encryptor-architecture", choices=ARCHITECTURES, default="mlp_residual_v1")
    ap.add_argument("--diversity-space", choices=("projected_z", "raw_r"), default="projected_z")
    ap.add_argument("--execute", action="store_true")
    return ap.parse_args()


def trainer_command(args: argparse.Namespace, output: Path, steps: int, *, resume: Path | None, lambda_util: float) -> list[str]:
    command = [
        sys.executable, str(TRAINER),
        "--model-path", str(args.model_path),
        "--train-jsonl", str(args.train_jsonl),
        "--valid-jsonl", str(args.valid_jsonl),
        "--output-dir", str(output),
        "--steps", str(steps),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--max-length", str(args.max_length),
        "--max-valid-samples", str(args.max_valid_samples),
        "--key-dim", str(args.key_dim), "--hidden-dim", str(args.hidden_dim),
        "--learning-rate", str(args.learning_rate),
        "--warmup-steps", str(args.warmup_steps),
        "--tau-low", str(args.tau_low), "--tau-high", str(args.tau_high),
        "--epsilon", str(args.epsilon),
        "--diversity-margin", str(args.diversity_margin),
        "--grad-clip", str(args.grad_clip),
        "--checkpoint-every", str(args.checkpoint_every),
        "--evaluate-every", str(args.evaluate_every),
        "--constraint-mode", "constant",
        "--lambda-util", str(lambda_util),
        "--lambda-priv", "1.0",
        "--lambda-div", "1.0",
        "--encryptor-architecture", args.encryptor_architecture,
        "--diversity-space", args.diversity_space,
        "--seed", str(args.seed),
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    if args.max_train_samples is not None:
        command += ["--max-train-samples", str(args.max_train_samples)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def stage_phase_transition(source: Path, stage_dir: Path, *, lambda_util: float) -> Path:
    """Copy the phase-1 checkpoint with an honestly recorded recipe change.

    Phase 2 intentionally changes lambda_util, which the resume recipe guard
    treats as immutable. Staging a checkpoint whose embedded config records the
    transition keeps both the guard and the artifact provenance truthful.
    """
    import copy

    import torch

    from osnip_repro.data import file_sha256

    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format") != "osnip_training_checkpoint_v1":
        raise ValueError("phase transition requires a full training checkpoint")
    staged_config = copy.deepcopy(payload["config"])
    staged_config["lambda_util"] = lambda_util
    staged_config["phase_transition"] = {
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "changed_fields": ["lambda_util"],
    }
    payload["config"] = staged_config
    staged = stage_dir / "start.pt"
    torch.save(payload, staged)
    (stage_dir / "config.json").write_text(json.dumps(staged_config, indent=2) + "\n")
    (stage_dir / "staging_manifest.json").write_text(json.dumps({
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "staged_sha256": file_sha256(staged),
        "changes": ["config.lambda_util rewritten for phase 2; weights/optimizer/scheduler/RNG/cursor untouched"],
    }, indent=2) + "\n")
    return staged


def main() -> None:
    args = parse_args()
    if args.phase1_steps < 1 or args.total_steps <= args.phase1_steps:
        raise ValueError("total-steps must be greater than positive phase1-steps")
    phase1_dir = args.output_dir / "phase1_privacy_injection"
    stage_dir = args.output_dir / "phase2_start"
    phase1 = trainer_command(args, phase1_dir, args.phase1_steps, resume=None, lambda_util=0.0)
    phase2 = trainer_command(args, args.output_dir, args.total_steps, resume=stage_dir / "start.pt", lambda_util=1.0)
    plan = {
        "format": "osnip_two_phase_diagnostic_plan_v1",
        "phase1": {"purpose": "privacy injection", "command": phase1, "output_dir": str(phase1_dir.resolve())},
        "phase2": {"purpose": "utility recovery with privacy retained", "command": phase2, "output_dir": str(args.output_dir.resolve())},
        "total_steps": args.total_steps,
        "phase1_steps": args.phase1_steps,
        "security_accepted": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "two_phase_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"execute": args.execute, "phase1_steps": args.phase1_steps, "total_steps": args.total_steps, "plan": str((args.output_dir / 'two_phase_plan.json').resolve())}, ensure_ascii=False, indent=2))
    if not args.execute:
        print("phase1", " ".join(phase1))
        print("phase2", " ".join(phase2))
        return
    subprocess.run(phase1, check=True)
    stage_phase_transition(phase1_dir / "last.pt", stage_dir, lambda_util=1.0)
    subprocess.run(phase2, check=True)


if __name__ == "__main__":
    main()
