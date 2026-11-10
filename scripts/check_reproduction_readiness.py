#!/usr/bin/env python3
"""Check whether the locked OSNIP reproduction inputs are ready for Gate 0."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, default=ROOT / "model/Llama-3.2-1B")
    ap.add_argument("--data-manifest", type=Path, default=ROOT / "data/osnip_65k/data_manifest.json")
    ap.add_argument("--attack-manifest", type=Path, default=ROOT / "data/paper_attack_sets/attack_manifest.json")
    ap.add_argument("--summary-manifest", type=Path, default=ROOT / "data/osnip_sources/cnn_dailymail_test.manifest.json")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def file_check(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.is_file()}


def main() -> None:
    args = parse_args()
    checks: dict[str, Any] = {}
    gate0_blockers: list[str] = []
    formal_blockers: list[str] = []
    model_config = args.model_path / "config.json"
    checks["model_directory"] = {"path": str(args.model_path.resolve()), "exists": args.model_path.is_dir()}
    checks["model_config"] = file_check(model_config)
    if not args.model_path.is_dir():
        gate0_blockers.append("missing Llama-3.2-1B model directory")
    if not model_config.is_file():
        gate0_blockers.append("missing model config.json")
    else:
        config = json.loads(model_config.read_text())
        architecture_ok = config.get("model_type") == "llama" and "LlamaForCausalLM" in (config.get("architectures") or [])
        checks["model_architecture"] = {"passed": architecture_ok, "model_type": config.get("model_type"), "architectures": config.get("architectures", [])}
        if not architecture_ok:
            gate0_blockers.append("model is not LlamaForCausalLM")

    for name, path in (("data_manifest", args.data_manifest), ("attack_manifest", args.attack_manifest), ("summary_manifest", args.summary_manifest)):
        checks[name] = file_check(path)
        if not path.is_file():
            (gate0_blockers if name in ("data_manifest", "attack_manifest") else formal_blockers).append(f"missing {name}")
        else:
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                checks[name]["valid_json"] = False
                (gate0_blockers if name in ("data_manifest", "attack_manifest") else formal_blockers).append(f"invalid JSON in {name}")
            else:
                checks[name]["valid_json"] = True
                checks[name]["status"] = payload.get("status")
                checks[name]["records"] = payload.get("records_written", (payload.get("construction") or {}).get("train_total_actual"))
                # validate_reproduction_inputs accepts both locked statuses for
                # question-disjoint data; a missing status is never "ready".
                if payload.get("status") not in ("ready", "ready_for_fresh_training_only"):
                    (gate0_blockers if name in ("data_manifest", "attack_manifest") else formal_blockers).append(f"{name} is not ready")

    bertscore_available = importlib.util.find_spec("bert_score") is not None
    checks["bert_score_dependency"] = {"passed": bertscore_available, "required_for_formal_bertscore": True}
    if not bertscore_available:
        formal_blockers.append("bert-score package is not installed")
    gate0_allowed = not gate0_blockers
    formal_evaluation_ready = gate0_allowed and not formal_blockers
    status = "ready_for_gate0" if gate0_allowed else "blocked"
    report = {
        "format": "osnip_reproduction_readiness_report_v2",
        "status": status,
        "gate0_allowed": gate0_allowed,
        "formal_evaluation_ready": formal_evaluation_ready,
        "security_accepted": False,
        "checks": checks,
        "gate0_blockers": gate0_blockers,
        "formal_evaluation_blockers": formal_blockers,
        "next_step": "run_gate0.py" if gate0_allowed else "resolve Gate 0 blockers before Gate 0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "gate0_allowed": report["gate0_allowed"], "formal_evaluation_ready": report["formal_evaluation_ready"], "gate0_blockers": gate0_blockers, "formal_evaluation_blockers": formal_blockers, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
