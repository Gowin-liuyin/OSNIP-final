#!/usr/bin/env python3
"""Run the complete OSNIP utility/privacy evaluation suite for one training run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--criterion", choices=("utility", "pareto", "last"), default="pareto",
                    help="last evaluates the paper-final last.pt at the prescribed step")
    ap.add_argument("--test-jsonl", type=Path, default=ROOT / "data/osnip_65k/test.jsonl")
    ap.add_argument("--task-test-jsonl", type=Path, default=None,
                    help="official test-split task set; defaults to --test-jsonl for legacy runs")
    ap.add_argument("--ppl-jsonl", type=Path, default=ROOT / "data/wikitext2/test.jsonl")
    ap.add_argument("--arc-easy", type=Path, default=ROOT / "data/osnip_sources/raw/arc_easy.jsonl")
    ap.add_argument("--arc-challenge", type=Path, default=ROOT / "data/osnip_sources/raw/arc_challenge.jsonl")
    ap.add_argument("--sciq", type=Path, default=ROOT / "data/osnip_sources/raw/sciq.jsonl")
    ap.add_argument("--summarization-jsonl", type=Path)
    ap.add_argument("--max-summarization-samples", type=int, default=1000)
    ap.add_argument("--summary-max-input-length", type=int, default=512)
    ap.add_argument("--summary-max-new-tokens", type=int, default=128)
    ap.add_argument("--summary-num-beams", type=int, default=1)
    ap.add_argument("--skip-bertscore", action="store_true")
    ap.add_argument("--require-bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--knn-jsonl", type=Path, default=ROOT / "data/paper_attack_sets/knn.jsonl")
    ap.add_argument("--vma-jsonl", type=Path, default=ROOT / "data/paper_attack_sets/vma.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 5, 10, 15])
    ap.add_argument("--keys-per-prompt", type=int, default=5)
    ap.add_argument("--max-task-total", type=int, default=None)
    ap.add_argument("--task-source-split", choices=("train", "validation", "test"), default=None,
                    help="restrict task raw records to an official HF split (paper ARC-Easy uses test)")
    ap.add_argument("--max-utility-samples", type=int, default=None)
    ap.add_argument("--utility-batch-size", type=int, default=1)
    ap.add_argument("--max-ppl-samples", type=int, default=None)
    ap.add_argument("--max-knn-samples", type=int, default=1000)
    ap.add_argument("--max-vma-prompts", type=int, default=200)
    ap.add_argument("--vma-max-tokens", type=int, default=512,
                    help="VMA observation length; C.2 is silent, D.3's 16 is visualization only")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--ppl-max-length", type=int, default=512)
    ap.add_argument("--ppl-stride", type=int, default=256)
    ap.add_argument("--paper-clean-ppl", type=float, default=None,
                    help="paper Table 2 clean PPL (3B 9.63, 1B 8.63) for a paper-baseline ratio")
    ap.add_argument("--vocab-chunk", type=int, default=2048)
    ap.add_argument("--candidate-chunk", type=int, default=256)
    ap.add_argument("--candidate-mode", choices=("full", "proposal"), default="full")
    ap.add_argument("--candidate-k", type=int, default=256)
    ap.add_argument("--skip-input-vocabulary-probe", action="store_true",
                    help="skip the diagnostic input-vocabulary probe in utility evaluation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def command(args: argparse.Namespace, script: str, output: Path, extra: list[str]) -> list[str]:
    base = [PYTHON, str(ROOT / "scripts" / script), "--model-path", str(args.model_path), "--checkpoint", str(args.checkpoint), "--output", str(output), "--device", args.device, "--dtype", args.dtype]
    return base + extra


def main() -> None:
    args = parse_args()
    if args.require_bertscore and (args.skip_bertscore or args.summarization_jsonl is None):
        raise ValueError("required BERTScore needs summarization data and cannot be skipped")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(args.run_dir)
    args.checkpoint = None  # type: ignore[attr-defined]
    selection_path = args.run_dir / f"selected_{args.criterion}.json"
    select = [PYTHON, str(ROOT / "scripts/select_checkpoint.py"), "--run-dir", str(args.run_dir), "--criterion", args.criterion, "--output", str(selection_path)]
    if selection_path.is_file() and not args.overwrite and not (args.execute and args.criterion == "last"):
        selection = json.loads(selection_path.read_text())
    else:
        if not args.execute:
            # Dry-run placeholder follows the requested criterion and is
            # labelled so downstream replays never mistake it for a selection.
            placeholder_name = "last.pt" if args.criterion == "last" else f"best_{args.criterion}.pt"
            selection = {"selected": {"path": str(args.run_dir / placeholder_name)},
                         "placeholder": True}
        else:
            subprocess.run(select, check=True)
            selection = json.loads(selection_path.read_text())
    checkpoint = Path(selection["selected"]["path"]).resolve()
    if checkpoint.parent != args.run_dir.resolve():
        raise ValueError("checkpoint selection points outside the requested run directory")
    if args.criterion == "last" and checkpoint.name != "last.pt":
        raise ValueError("criterion=last must select last.pt at the paper-final step")
    if args.execute and not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
    args.checkpoint = checkpoint  # type: ignore[attr-defined]
    outputs = {
        "utility": args.run_dir / "utility_test.json",
        "task": args.run_dir / "task_utility_test.json",
        "ppl": args.run_dir / "sliding_ppl_test.json",
        "privacy": args.run_dir / "paper_protocol_test.json",
    }
    if args.summarization_jsonl is not None:
        outputs["summarization"] = args.run_dir / "summarization_utility_test.json"
    derived_outputs = (
        args.run_dir / "utility_report.json",
        args.run_dir / "knn_report.json",
        args.run_dir / "vma_report.json",
        args.run_dir / "confidence_intervals.json",
        outputs["privacy"].with_suffix(".progress.json"),
        outputs["privacy"].with_suffix(".knn.json"),
        outputs["privacy"].with_suffix(".vma_rows.jsonl"),
    )
    if args.overwrite:
        for stale in (*outputs.values(), *derived_outputs):
            if stale.is_file():
                stale.unlink()
    if args.execute and not args.overwrite:
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for output in outputs.values():
            if not output.is_file():
                continue
            existing = json.loads(output.read_text())
            if existing.get("checkpoint_sha256") != digest:
                raise ValueError(
                    f"existing output {output.name} belongs to a different checkpoint; "
                    "use --overwrite to re-evaluate the newly selected checkpoint")
    commands: dict[str, list[str]] = {}
    if args.utility_batch_size < 1:
        raise ValueError("utility-batch-size must be positive")
    utility_extra = ["--data-jsonl", str(args.test_jsonl), "--max-length", str(args.max_length), "--max-samples", str(args.max_utility_samples or 1000), "--batch-size", str(args.utility_batch_size), "--vocab-chunk", str(args.vocab_chunk), "--seed", str(args.seed)]
    if args.skip_input_vocabulary_probe:
        utility_extra.append("--skip-vocabulary-probe")
    commands["utility"] = command(args, "evaluate_no_key.py", outputs["utility"], utility_extra)
    task_extra = ["--test-jsonl", str(args.task_test_jsonl or args.test_jsonl), "--arc-easy", str(args.arc_easy), "--arc-challenge", str(args.arc_challenge), "--sciq", str(args.sciq), "--max-length", str(args.max_length), "--seed", str(args.seed)]
    if args.task_source_split is not None:
        task_extra += ["--source-split", args.task_source_split]
    if args.max_task_total is not None:
        task_extra += ["--max-total", str(args.max_task_total)]
    commands["task"] = command(args, "evaluate_task_utility.py", outputs["task"], task_extra)
    ppl_extra = ["--data-jsonl", str(args.ppl_jsonl), "--max-length", str(args.ppl_max_length), "--stride", str(args.ppl_stride), "--seed", str(args.seed)]
    if args.paper_clean_ppl is not None:
        ppl_extra += ["--paper-clean-ppl", str(args.paper_clean_ppl)]
    if args.max_ppl_samples is not None:
        ppl_extra += ["--max-samples", str(args.max_ppl_samples)]
    commands["ppl"] = command(args, "evaluate_sliding_ppl.py", outputs["ppl"], ppl_extra)
    privacy_extra = ["--knn-jsonl", str(args.knn_jsonl), "--vma-jsonl", str(args.vma_jsonl), "--layers", *[str(layer) for layer in args.layers], "--keys-per-prompt", str(args.keys_per_prompt), "--max-knn-samples", str(args.max_knn_samples), "--max-vma-prompts", str(args.max_vma_prompts), "--max-tokens", str(args.vma_max_tokens), "--vocab-chunk", str(args.vocab_chunk), "--candidate-chunk", str(args.candidate_chunk), "--candidate-mode", args.candidate_mode, "--candidate-k", str(args.candidate_k), "--seed", str(args.seed)]
    if not args.overwrite:
        # Prefer this run's own derived full-row report (identical binding),
        # then any independent bound v2 report; the suite's v1 summary carries
        # no rows and can never support deterministic key replay.
        knn_candidates = (outputs["privacy"].with_suffix(".knn.json"), args.run_dir / "knn_report.json")
        for candidate in knn_candidates:
            if not candidate.is_file():
                continue
            knn_existing = json.loads(candidate.read_text())
            if knn_existing.get("format") == "osnip_knn_report_v2" and knn_existing.get("rows"):
                privacy_extra += ["--knn-report", str(candidate)]
                break
        if outputs["privacy"].with_suffix(".vma_rows.jsonl").is_file():
            privacy_extra.append("--resume-vma")
    commands["privacy"] = command(args, "evaluate_paper_protocol.py", outputs["privacy"], privacy_extra)
    if args.summarization_jsonl is not None:
        summary_extra = [
            "--data-jsonl", str(args.summarization_jsonl),
            "--max-samples", str(args.max_summarization_samples),
            "--max-input-length", str(args.summary_max_input_length),
            "--max-new-tokens", str(args.summary_max_new_tokens),
            "--num-beams", str(args.summary_num_beams),
            "--bertscore-model", args.bertscore_model,
            "--seed", str(args.seed),
        ]
        if args.skip_bertscore:
            summary_extra.append("--skip-bertscore")
        if args.require_bertscore:
            summary_extra.append("--require-bertscore")
        commands["summarization"] = command(args, "evaluate_summarization_utility.py", outputs["summarization"], summary_extra)
    plan = {"format": "osnip_evaluation_suite_plan_v2", "run_dir": str(args.run_dir.resolve()), "criterion": args.criterion, "checkpoint": str(checkpoint.resolve()), "checkpoint_placeholder": bool(selection.get("placeholder", False)), "commands": commands, "outputs": {name: str(path.resolve()) for name, path in outputs.items()}, "execute": args.execute, "security_accepted": False}
    plan_path = args.run_dir / "evaluation_suite_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint), "plan": str(plan_path.resolve()), "execute": args.execute, "outputs": {name: str(path) for name, path in outputs.items()}}, ensure_ascii=False, indent=2))
    if not args.execute:
        for name, cmd in commands.items():
            print(name, " ".join(cmd))
        return
    for name, cmd in commands.items():
        output = outputs[name]
        if output.is_file():
            if not args.overwrite:
                print(json.dumps({"status": "skip_existing", "name": name, "output": str(output)}, ensure_ascii=False))
                continue
            output.unlink()
        subprocess.run(cmd, check=True)
    utility_report = json.loads(outputs["utility"].read_text())
    task_report = json.loads(outputs["task"].read_text())
    ppl_report = json.loads(outputs["ppl"].read_text())
    privacy_report = json.loads(outputs["privacy"].read_text())
    joint_report = {"format": "osnip_utility_report_v2", "joint": utility_report, "task": task_report, "sliding_ppl": ppl_report}
    if "summarization" in outputs:
        joint_report["summarization"] = json.loads(outputs["summarization"].read_text())
    (args.run_dir / "utility_report.json").write_text(json.dumps(joint_report, ensure_ascii=False, indent=2) + "\n")
    (args.run_dir / "knn_report.json").write_text(json.dumps({"format": "osnip_knn_report_v1", **privacy_report.get("knn", {}), "checkpoint": privacy_report.get("checkpoint"), "data_sha256": privacy_report.get("data_sha256", {}).get("knn")}, ensure_ascii=False, indent=2) + "\n")
    (args.run_dir / "vma_report.json").write_text(json.dumps({"format": "osnip_vma_report_v1", **privacy_report.get("vma", {}), "stopword_policy": privacy_report.get("stopword_policy"), "checkpoint": privacy_report.get("checkpoint"), "data_sha256": privacy_report.get("data_sha256", {}).get("vma")}, ensure_ascii=False, indent=2) + "\n")
    confidence = {str(layer): summary.get("prompt_paired", {}) for layer, summary in privacy_report.get("vma", {}).get("summary", {}).items()}
    (args.run_dir / "confidence_intervals.json").write_text(json.dumps({"format": "osnip_confidence_interval_report_v1", "vma": confidence}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
