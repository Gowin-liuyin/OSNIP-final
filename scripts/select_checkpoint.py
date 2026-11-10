#!/usr/bin/env python3
"""Select a recorded OSNIP checkpoint without silently choosing the last step."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from osnip_repro.checkpointing import load_encoder_state_dict, recipe_fingerprint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--criterion", choices=("utility", "pareto", "last"), default="utility",
                    help="last selects the paper-final checkpoint (last.pt at the prescribed step)")
    ap.add_argument("--output", type=Path)
    return ap.parse_args()


def run_config_of(run_dir: Path) -> dict[str, Any]:
    for name in ("config.json", "train_config.json"):
        path = run_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def checkpoint_summary(path: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    """Summarize one candidate and refuse artifacts that cannot be trusted."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "osnip_training_checkpoint_v1":
        raise ValueError(f"{path.name}: unsupported checkpoint format {payload.get('format')!r}")
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{path.name}: invalid global_step {step!r}")
    metrics = payload.get("metrics", {})
    validation = metrics.get("validation", {})
    # Legacy payloads predate validation-embedded steps; their metrics step is
    # saved alongside global_step, so falling back keeps old runs selectable.
    recorded_step = validation.get("step", metrics.get("step"))
    validation_fresh = recorded_step == step
    config = payload.get("config", {})
    if run_config and recipe_fingerprint(config) != recipe_fingerprint(run_config):
        raise ValueError(f"{path.name}: embedded recipe differs from the run config")
    snapshot = {
        "best_utility.pt": "best_utility_encoder.pt",
        "best_pareto.pt": "best_pareto_encoder.pt",
        "last.pt": "encryptor.pt",
    }.get(path.name, "encryptor.pt")
    encoder_path = path.parent / snapshot
    if not encoder_path.is_file():
        raise ValueError(f"{path.name}: missing paired encoder artifact {snapshot}")
    load_encoder_state_dict(str(encoder_path))
    embedded = payload.get("encoder")
    if isinstance(embedded, dict):
        for name, tensor in embedded.items():
            if torch.is_tensor(tensor) and not torch.isfinite(tensor.float()).all():
                raise ValueError(f"{path.name}: non-finite embedded encoder tensor {name}")
    return {"path": str(path.resolve()), "encoder_state": str(encoder_path.resolve()),
            "global_step": step, "validation": validation, "validation_fresh": validation_fresh,
            "config": config}


def main() -> None:
    args = parse_args()
    run_config = run_config_of(args.run_dir)
    candidates = {name: args.run_dir / name for name in ("best_utility.pt", "best_pareto.pt", "last.pt")}
    if args.criterion == "last":
        if not candidates["last.pt"].is_file():
            raise ValueError("last.pt is missing or carries stale validation; cannot select the paper-final checkpoint")
        available = [candidates["last.pt"]]
    else:
        available = [path for path in candidates.values() if path.is_file()]
    if not available:
        raise FileNotFoundError(f"no OSNIP checkpoints found in {args.run_dir}")
    if args.criterion == "last" and not run_config:
        raise ValueError("last selection requires the run config's final step")
    summaries = [checkpoint_summary(path, run_config) for path in available]
    metric = "utility" if args.criterion == "utility" else "pareto_score"
    fresh = [row for row in summaries if row["validation_fresh"]]
    if not fresh:
        raise ValueError("no candidate carries validation from its own global_step; refusing stale selection")
    if args.criterion == "last":
        # The paper reports one model at the end of the published schedule;
        # select last.pt explicitly instead of a validation-minimum snapshot.
        selected = next((row for row in fresh if Path(row["path"]).name == "last.pt"), None)
        if selected is None:
            raise ValueError("last.pt is missing or carries stale validation; cannot select the paper-final checkpoint")
        for metric in ("utility", "privacy", "ppl_ratio"):
            value = selected["validation"].get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"last.pt validation metric {metric} missing or non-finite")
        expected_steps = run_config.get("steps", run_config.get("formal_steps"))
        if expected_steps is not None and int(selected["global_step"]) != int(expected_steps):
            raise ValueError(f"last.pt is at step {selected['global_step']}, not the configured final step {expected_steps}")
        report = {"format": "osnip_checkpoint_selection_v1", "criterion": "last", "metric": "final_step",
                  "selected": selected, "candidates": summaries,
                  "skipped_stale_candidates": [row["path"] for row in summaries if not row["validation_fresh"]]}
        output = args.output or args.run_dir / "selected_last.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"criterion": "last", "selected": selected["path"], "step": selected["global_step"],
                          "output": str(output.resolve())}, ensure_ascii=False, indent=2))
        return
    for row in summaries:
        value = row["validation"].get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{Path(row['path']).name}: selection metric {metric} missing or non-finite")
    selected = min(fresh, key=lambda row: float(row["validation"][metric]))
    report = {"format": "osnip_checkpoint_selection_v1", "criterion": args.criterion, "metric": metric, "selected": selected, "candidates": summaries,
              "skipped_stale_candidates": [row["path"] for row in summaries if not row["validation_fresh"]]}
    output = args.output or args.run_dir / f"selected_{args.criterion}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"criterion": args.criterion, "selected": selected["path"], "step": selected["global_step"], "output": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
