#!/usr/bin/env python3
"""Generate or execute the pre-registered Llama-3.2-1B OSNIP ablations."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--train-jsonl", type=Path, default=ROOT / "data/osnip_65k/train.jsonl")
    ap.add_argument("--valid-jsonl", type=Path, default=ROOT / "data/osnip_65k/valid.jsonl")
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/llama32_1b_ablation_matrix.json")
    ap.add_argument("--output-root", type=Path, default=ROOT / "reports/llama32_1b")
    ap.add_argument("--gate0-output-dir", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/llama32_1b_reproduction.json")
    ap.add_argument("--data-manifest", type=Path, default=ROOT / "data/osnip_65k/data_manifest.json")
    ap.add_argument("--attack-manifest", type=Path, default=ROOT / "data/paper_attack_sets/attack_manifest.json")
    ap.add_argument("--execute", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text())
    model_config_path = args.model_path / "config.json"
    gate0_output_dir = args.gate0_output_dir or (args.output_root / "gate0")
    gate0_command = [
        sys.executable, str(ROOT / "scripts/run_gate0.py"),
        "--model-path", str(args.model_path),
        "--output-dir", str(gate0_output_dir),
        "--config", str(args.config),
        "--data-manifest", str(args.data_manifest),
        "--attack-manifest", str(args.attack_manifest),
        "--device", str(matrix.get("device", "cpu")),
        "--dtype", str(matrix.get("dtype", "float32")),
    ]
    preflight = {
        "model_path": str(args.model_path.resolve()),
        "model_directory": args.model_path.is_dir(),
        "config_file": model_config_path.is_file(),
        "data_manifest": args.data_manifest.is_file(),
        "attack_manifest": args.attack_manifest.is_file(),
        "gate0_output_dir": str(gate0_output_dir.resolve()),
        "gate0_command": gate0_command,
        "execute_allowed": False,
    }
    if args.execute:
        if not args.model_path.is_dir():
            raise FileNotFoundError(
                f"model directory is missing: {args.model_path}; download and validate "
                "meta-llama/Llama-3.2-1B before --execute"
            )
        if not model_config_path.is_file():
            raise FileNotFoundError(f"model config is missing: {model_config_path}")
        for required_manifest in (args.data_manifest, args.attack_manifest, args.train_jsonl, args.valid_jsonl):
            if not required_manifest.is_file():
                raise FileNotFoundError(f"required reproduction input is missing: {required_manifest}")
        model_config = json.loads(model_config_path.read_text())
        if model_config.get("model_type") != "llama" or "LlamaForCausalLM" not in (model_config.get("architectures") or []):
            raise ValueError(f"--execute requires a LlamaForCausalLM directory, got {model_config.get('model_type')!r}")
        preflight["execute_allowed"] = True
    architectures = matrix.get("encryptor_variants", ["mlp_residual_v1"])
    commands: list[dict[str, object]] = []
    defaults = matrix["training_defaults"]
    for variant in matrix["variants"]:
        for architecture in architectures:
            for seed in matrix["seeds"]:
                output = args.output_root / f"{variant['name']}__{architecture}_seed{seed}"
                if variant["name"] == "two_phase_diagnostic":
                    command = [
                        sys.executable, str(ROOT / "scripts/run_two_phase_diagnostic.py"), "--execute",
                        "--model-path", str(args.model_path),
                        "--train-jsonl", str(args.train_jsonl),
                        "--valid-jsonl", str(args.valid_jsonl),
                        "--output-dir", str(output),
                        "--phase1-steps", str(defaults.get("two_phase_phase1_steps", 3000)),
                        "--total-steps", str(defaults["steps"]),
                        "--batch-size", str(defaults["batch_size"]),
                        "--gradient-accumulation-steps", str(defaults["gradient_accumulation_steps"]),
                        "--max-length", str(defaults["max_length"]),
                        "--learning-rate", str(defaults["learning_rate"]),
                        "--warmup-steps", str(defaults["warmup_steps"]),
                        "--epsilon", str(defaults["epsilon"]),
                        "--diversity-margin", str(defaults["diversity_margin"]),
                        "--diversity-space", str(defaults.get("diversity_space", "projected_z")),
                        "--grad-clip", str(defaults["grad_clip"]),
                        "--checkpoint-every", str(defaults["checkpoint_every"]),
                        "--evaluate-every", str(defaults["evaluate_every"]),
                        "--encryptor-architecture", architecture,
                        "--seed", str(seed), "--device", matrix["device"], "--dtype", matrix["dtype"],
                    ]
                else:
                    command = [
                        sys.executable, str(ROOT / "scripts/train_streaming.py"),
                        "--model-path", str(args.model_path),
                        "--train-jsonl", str(args.train_jsonl),
                        "--valid-jsonl", str(args.valid_jsonl),
                        "--output-dir", str(output),
                        "--steps", str(defaults["steps"]),
                        "--batch-size", str(defaults["batch_size"]),
                        "--gradient-accumulation-steps", str(defaults["gradient_accumulation_steps"]),
                        "--max-length", str(defaults["max_length"]),
                        "--learning-rate", str(defaults["learning_rate"]),
                        "--warmup-steps", str(defaults["warmup_steps"]),
                        "--epsilon", str(defaults["epsilon"]),
                        "--diversity-margin", str(defaults["diversity_margin"]),
                        "--diversity-space", str(defaults.get("diversity_space", "projected_z")),
                        "--grad-clip", str(defaults["grad_clip"]),
                        "--checkpoint-every", str(defaults["checkpoint_every"]),
                        "--evaluate-every", str(defaults["evaluate_every"]),
                        "--constraint-mode", str(variant["constraint_mode"]),
                        "--lambda-util", str(variant["lambda_util"]),
                        "--lambda-priv", str(variant["lambda_priv"]),
                        "--lambda-div", str(variant["lambda_div"]),
                        "--encryptor-architecture", architecture,
                        "--seed", str(seed), "--device", matrix["device"], "--dtype", matrix["dtype"],
                    ]
                commands.append({"variant": variant["name"], "architecture": architecture, "seed": seed, "output_dir": str(output), "command": command})
    plan = {
        "format": "osnip_ablation_execution_plan_v2",
        "matrix": str(args.matrix.resolve()),
        "execute": args.execute,
        "architectures": architectures,
        "commands": commands,
        "command_count": len(commands),
        "preflight": preflight,
        "gate0": {"command": gate0_command, "required_before_training": True, "output_dir": str(gate0_output_dir.resolve())},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "execution_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"commands": len(commands), "architectures": architectures, "plan": str(plan_path.resolve()), "execute": args.execute}, ensure_ascii=False, indent=2))
    if not args.execute:
        print("gate0", " ".join(gate0_command))
        for item in commands:
            print(" ".join(item["command"]))
        return
    subprocess.run(gate0_command, check=True)
    gate0_report_path = gate0_output_dir / "gate0_report.json"
    gate0_report = json.loads(gate0_report_path.read_text()) if gate0_report_path.is_file() else {}
    if gate0_report.get("status") != "ready":
        raise RuntimeError(f"Gate 0 did not pass: {gate0_report_path}")
    for item in commands:
        output = Path(item["output_dir"])
        if (output / "report.json").is_file():
            print(json.dumps({"status": "skip_existing", "output": str(output)}, ensure_ascii=False))
            continue
        print(json.dumps({"status": "running", "variant": item["variant"], "architecture": item["architecture"], "seed": item["seed"]}, ensure_ascii=False), flush=True)
        subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
