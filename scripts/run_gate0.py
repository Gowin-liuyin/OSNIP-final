#!/usr/bin/env python3
"""Run the model and fixed-input Gate 0 preflight as one auditable command."""
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
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/llama32_1b_question_disjoint_v2.json")
    ap.add_argument("--data-manifest", type=Path, default=ROOT / "data/osnip_65k_question_disjoint_v2/data_manifest.json")
    ap.add_argument("--attack-manifest", type=Path, default=ROOT / "data/paper_attack_sets/attack_manifest.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model_environment.json", "input_validation.json", "gate0_report.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"choose a new Gate 0 directory; existing artifact: {args.output_dir / name}")
    model_output = args.output_dir / "model_environment.json"
    input_output = args.output_dir / "input_validation.json"
    model_cmd = [sys.executable, str(ROOT / "scripts/validate_model_environment.py"), "--model-path", str(args.model_path), "--output", str(model_output), "--device", args.device, "--dtype", args.dtype, "--tolerance", str(args.tolerance)]
    input_cmd = [sys.executable, str(ROOT / "scripts/validate_reproduction_inputs.py"), "--model-path", str(args.model_path), "--config", str(args.config), "--data-manifest", str(args.data_manifest), "--attack-manifest", str(args.attack_manifest), "--output", str(input_output)]
    results: dict[str, dict[str, object]] = {}
    for name, command, report_path in (("model_environment", model_cmd, model_output), ("fixed_inputs", input_cmd, input_output)):
        completed = subprocess.run(command, capture_output=True, text=True)
        report = json.loads(report_path.read_text()) if report_path.is_file() else {"status": "failed", "failure": "validator produced no report", "stdout": completed.stdout, "stderr": completed.stderr}
        results[name] = {"status": report.get("status", "failed"), "report": str(report_path.resolve()), "exit_code": completed.returncode, "stderr_tail": completed.stderr[-2000:] if completed.returncode else "", "failure": report.get("failure"), "failed_checks": report.get("checks", []) and [check for check in report["checks"] if check.get("status") == "failed"]}
    status = "ready" if all(item["status"] == "ready" and item["exit_code"] == 0 for item in results.values()) else "failed"
    report = {"format": "osnip_gate0_report_v1", "status": status, "model_path": str(args.model_path.resolve()), "checks": results, "next_gate": "gate1_objective_smoke" if status == "ready" else "obtain_and_validate_model_or_repair_fixed_inputs"}
    output = args.output_dir / "gate0_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "output": str(output.resolve()), "checks": results}, ensure_ascii=False, indent=2))
    if status != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
