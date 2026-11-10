#!/usr/bin/env python3
"""Evaluate clean/encrypted multiple-choice utility on held-out ARC and SciQ rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers
import torch.nn.functional as F

from osnip_repro.data import _iter_json_values
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import forward_logits, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project, encrypt_embeddings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--test-jsonl", required=True, type=Path)
    ap.add_argument("--arc-easy", required=True, type=Path)
    ap.add_argument("--arc-challenge", required=True, type=Path)
    ap.add_argument("--sciq", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--choice-seed", type=int, default=42, help="independent reproducible SciQ choice permutation")
    ap.add_argument("--min-clean-accuracy", type=float, default=0.25, help="reject a vacuous clean floor; paper 1B ARC-Easy clean is 0.603")
    ap.add_argument("--min-per-source-retention", type=float, default=0.80, help="a single benchmark collapse must not be masked by the pooled retention")
    ap.add_argument("--source-split", choices=("train", "validation", "test"), default=None,
                    help="restrict raw records to an official HF split; the paper's ARC-Easy row is the test split")
    ap.add_argument("--seed", type=int, default=20261018)
    return ap.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    return list(_iter_json_values(path))


def arc_task(record: dict[str, Any]) -> tuple[str, str, list[str], int] | None:
    choices = record.get("choices", {})
    labels = [str(value).strip() for value in choices.get("label", [])]
    values = [str(value).strip() for value in choices.get("text", [])]
    if not labels or len(labels) != len(values):
        return None
    answer = str(record.get("answerKey", "")).strip()
    if answer not in labels:
        return None
    stem = str(record.get("question", "")).strip()
    options = "\n".join(f"{label}: {value}" for label, value in zip(labels, values)) + "\nAnswer:"
    return stem, options, labels, labels.index(answer)


def sciq_task(record: dict[str, Any], *, seed: int = 42) -> tuple[str, str, list[str], int] | None:
    correct_answer = str(record.get("correct_answer", "")).strip()
    if not correct_answer:
        # Without a non-empty gold answer, order.index(0) would silently treat
        # the first distractor as correct.
        return None
    values = [correct_answer, record.get("distractor1"), record.get("distractor2"), record.get("distractor3")]
    values = [str(value).strip() for value in values if value is not None and str(value).strip()]
    if len(values) < 2:
        return None
    order = list(range(len(values)))
    random.Random(seed).shuffle(order)
    values = [values[index] for index in order]
    correct_index = order.index(0)
    labels = [chr(ord("A") + index) for index in range(len(values))]
    stem_pieces = [str(record.get("question", "")).strip()]
    support = record.get("support", record.get("context", ""))
    if support:
        stem_pieces.append(f"Context: {str(support).strip()}")
    stem = "\n".join(piece for piece in stem_pieces if piece)
    options = "\n".join(f"{label}: {value}" for label, value in zip(labels, values)) + "\nAnswer:"
    return stem, options, labels, correct_index


# Counts prompts whose stem exceeded the token budget and was head-truncated;
# reported so truncation can never silently change the scored question.
PROMPT_TRUNCATIONS: dict[str, int] = {"count": 0}


def encode_prompt_answer(tokenizer: Any, stem: str, options: str, answer: str, max_length: int, prompt_token_budget: int | None = None) -> tuple[list[int], list[int], bool] | None:
    """Encode a task prompt while always preserving the options and Answer cue.

    The previous head-truncation kept the question stem and silently dropped the
    option list and the trailing "Answer:" position, making the scored prompt
    unsolvable. Only the stem/context is truncated here.
    """
    answer_ids = list(tokenizer(" " + answer, add_special_tokens=False)["input_ids"])
    option_ids = list(tokenizer(options, add_special_tokens=False)["input_ids"])
    if not answer_ids or len(answer_ids) >= max_length:
        return None
    keep_prompt = max_length - len(answer_ids) if prompt_token_budget is None else prompt_token_budget
    stem_budget = keep_prompt - len(option_ids)
    if stem_budget < 0:
        return None
    stem_ids = list(tokenizer(stem, add_special_tokens=False)["input_ids"])
    truncated = len(stem_ids) > stem_budget
    if truncated:
        stem_ids = stem_ids[:stem_budget]
    return stem_ids + option_ids, answer_ids, truncated


def answer_nll(model: Any, tokenizer: Any, encoder: torch.nn.Module, stem: str, options: str, answer: str, *, key: torch.Tensor | None, max_length: int, prompt_token_budget: int | None = None, device: torch.device) -> float | None:
    encoded = encode_prompt_answer(tokenizer, stem, options, answer, max_length, prompt_token_budget)
    if encoded is None:
        return None
    prompt_ids, answer_ids, _ = encoded
    all_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if prompt_tensor.shape[1] == 0:
        return None
    answer_tensor = torch.tensor([answer_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        prompt_embeddings = input_embeddings(model, prompt_tensor)
        if key is not None:
            # The client encrypts the user prompt. Candidate answer tokens are
            # scored by the server after that encrypted prefix; they are not
            # part of the client observation.
            prompt_embeddings = encrypt_embeddings(encoder, prompt_embeddings, key)
        answer_embeddings = input_embeddings(model, answer_tensor)
        embeddings = torch.cat((prompt_embeddings, answer_embeddings), dim=1)
        mask = torch.ones_like(all_ids)
        logits = forward_logits(model, embeddings, mask).float()
    first_logit = len(prompt_ids) - 1
    target_logits = logits[:, first_logit:all_ids.shape[1] - 1]
    target_ids = all_ids[:, len(prompt_ids):]
    if target_logits.shape[1] != target_ids.shape[1] or target_ids.shape[1] == 0:
        return None
    return float(F.cross_entropy(target_logits.reshape(-1, target_logits.shape[-1]), target_ids.reshape(-1), reduction="sum"))



def choice_nlls(model: Any, tokenizer: Any, encoder: torch.nn.Module, stem: str, options: str,
                labels: list[str], *, key: torch.Tensor | None, max_length: int,
                prompt_token_budget: int, device: torch.device) -> list[float]:
    """Score single-token labels with one causal prompt forward.

    The answer is not part of the client observation. For single-token answers,
    the last prompt logit is exactly the score used by ``answer_nll``. General
    multi-token answers retain the existing teacher-forced implementation.
    """
    encoded = [encode_prompt_answer(tokenizer, stem, options, label, max_length, prompt_token_budget)
               for label in labels]
    if any(item is None for item in encoded):
        return []
    if all(len(item[1]) == 1 for item in encoded):
        prompt_ids = encoded[0][0]
        if not prompt_ids:
            return []
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            h = input_embeddings(model, ids)
            if key is not None:
                h = encrypt_embeddings(encoder, h, key)
            logits = forward_logits(model, h, torch.ones_like(ids))[0, -1].float()
            targets = torch.tensor([item[1][0] for item in encoded], device=device)
            return (-F.log_softmax(logits, dim=-1)[targets]).cpu().tolist()
    scores = [answer_nll(model, tokenizer, encoder, stem, options, label, key=key,
                         max_length=max_length, prompt_token_budget=prompt_token_budget,
                         device=device) for label in labels]
    return [] if any(value is None for value in scores) else scores


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    if args.max_length < 4 or (args.max_total is not None and args.max_total < 1) or (args.max_per_source is not None and args.max_per_source < 1):
        raise ValueError("invalid sample or max-length limits")
    if not 0 <= args.min_clean_accuracy <= 1 or not 0 <= args.min_per_source_retention <= 1:
        raise ValueError("task gate thresholds must be in [0, 1]")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))

    raw_paths = {"arc_easy": args.arc_easy, "arc_challenge": args.arc_challenge, "sciq": args.sciq}
    raw = {source: load_records(path) for source, path in raw_paths.items()}
    test_rows = list(_iter_json_values(args.test_jsonl))
    source_seen: dict[str, int] = {source: 0 for source in raw}
    selected: list[tuple[str, dict[str, Any]]] = []
    for row in test_rows:
        source = str(row.get("source", ""))
        if source not in raw or (args.max_per_source is not None and source_seen[source] >= args.max_per_source):
            continue
        index = row.get("source_index")
        if not isinstance(index, int) or not 0 <= index < len(raw[source]):
            continue
        record = raw[source][index]
        if args.source_split is not None and str(record.get("_source_split", "")) != args.source_split:
            continue
        selected.append((source, record))
        source_seen[source] += 1
        if args.max_total is not None and len(selected) >= args.max_total:
            break
    if not selected:
        raise ValueError("test split contains no supported ARC/SciQ records")

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    clean_correct = encrypted_correct = agreement = 0
    clean_nll = encrypted_nll = 0.0
    scored = 0
    started = time.monotonic()
    with torch.no_grad():
        for task_index, (source, record) in enumerate(selected):
            task = arc_task(record) if source.startswith("arc") else sciq_task(record, seed=args.choice_seed + task_index)
            if task is None:
                continue
            stem, options, labels, correct_index = task
            key = torch.randn(config["key_dim"], generator=rng).to(device)
            answer_lengths = [len(tokenizer(" " + label, add_special_tokens=False)["input_ids"]) for label in labels]
            max_answer_length = max(answer_lengths, default=0)
            prompt_token_budget = args.max_length - max_answer_length
            if prompt_token_budget < 1:
                continue
            option_length = len(tokenizer(options, add_special_tokens=False)["input_ids"])
            stem_length = len(tokenizer(stem, add_special_tokens=False)["input_ids"])
            if option_length > prompt_token_budget:
                # Options plus Answer cue alone exceed the budget; the row cannot
                # be scored without deleting the choices, so skip it explicitly.
                PROMPT_TRUNCATIONS["count"] += 1
                continue
            if stem_length > prompt_token_budget - option_length:
                # Counted once per task, not once per candidate encoding.
                PROMPT_TRUNCATIONS["count"] += 1
            clean_scores = choice_nlls(model, tokenizer, encoder, stem, options, labels,
                                       key=None, max_length=args.max_length,
                                       prompt_token_budget=prompt_token_budget, device=device)
            encrypted_scores = choice_nlls(model, tokenizer, encoder, stem, options, labels,
                                           key=key, max_length=args.max_length,
                                           prompt_token_budget=prompt_token_budget, device=device)
            if not clean_scores or not encrypted_scores:
                continue
            clean_prediction = min(range(len(labels)), key=clean_scores.__getitem__)
            encrypted_prediction = min(range(len(labels)), key=encrypted_scores.__getitem__)
            clean_hit = clean_prediction == correct_index
            encrypted_hit = encrypted_prediction == correct_index
            clean_correct += int(clean_hit)
            encrypted_correct += int(encrypted_hit)
            agreement += int(clean_prediction == encrypted_prediction)
            clean_nll += clean_scores[correct_index]
            encrypted_nll += encrypted_scores[correct_index]
            scored += 1
            rows.append({"index": task_index, "source": source, "choices": labels, "correct": labels[correct_index], "clean_prediction": labels[clean_prediction], "encrypted_prediction": labels[encrypted_prediction], "clean_correct": clean_hit, "encrypted_correct": encrypted_hit, "clean_correct_answer_nll": clean_scores[correct_index], "encrypted_correct_answer_nll": encrypted_scores[correct_index]})
            if scored % 25 == 0 or task_index + 1 == len(selected):
                print(json.dumps({"stage": "task", "scored": scored, "selected": len(selected),
                                  "elapsed_seconds": round(time.monotonic() - started, 2),
                                  "clean_correct": clean_correct,
                                  "encrypted_correct": encrypted_correct}), flush=True)

    if not scored:
        raise ValueError("no task examples could be scored within max-length")
    clean_accuracy = clean_correct / scored
    encrypted_accuracy = encrypted_correct / scored
    per_source = {source: {
        "samples": sum(row["source"] == source for row in rows),
        "clean_correct": sum(row["source"] == source and row["clean_correct"] for row in rows),
        "encrypted_correct": sum(row["source"] == source and row["encrypted_correct"] for row in rows),
        "retention": (sum(row["source"] == source and row["encrypted_correct"] for row in rows)
                      / sum(row["source"] == source and row["clean_correct"] for row in rows)
                      if sum(row["source"] == source and row["clean_correct"] for row in rows) else None),
    } for source in raw}
    per_source_gate = {source: (row["retention"] is not None and row["retention"] >= args.min_per_source_retention)
                       for source, row in per_source.items()}
    report = {
        "format": "osnip_task_utility_report_v1",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "status": "completed_with_limits",
        "paper_result": False,
        "security_accepted": False,
        "model_path": str(args.model_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "test_jsonl": str(args.test_jsonl.resolve()),
        "sources": {name: str(path.resolve()) for name, path in raw_paths.items()},
        "records_selected": len(selected),
        "records_scored": scored,
        "max_length": args.max_length,
        "prompt_truncated_count": PROMPT_TRUNCATIONS["count"],
        "choice_policy": {"arc": "retain_original_labels", "sciq": "shuffle_with_independent_rng",
                          "seed": args.choice_seed, "per_example_seed": "choice_seed + task_index"},
        "source_counts": {source: sum(row["source"] == source for row in rows) for source in raw},
        "source_split_filter": args.source_split,
        "per_source": per_source,
        "per_source_gate": per_source_gate,
        "clean_accuracy": clean_accuracy,
        "encrypted_accuracy": encrypted_accuracy,
        "accuracy_retention_vs_clean": (encrypted_correct / clean_correct) if clean_correct > 0 else 0.0,
        "clean_encrypted_decision_agreement": agreement / scored,
        "clean_correct_answer_nll": clean_nll / scored,
        "encrypted_correct_answer_nll": encrypted_nll / scored,
        "correct_answer_nll_ratio": math.exp((encrypted_nll - clean_nll) / scored),
        "task_gate": {"passed": (clean_accuracy >= args.min_clean_accuracy and clean_correct > 0
                                 and encrypted_accuracy / max(clean_accuracy, 1e-12) >= 0.95
                                 and all(per_source_gate.values())),
                      "min_retention": 0.95,
                      "min_clean_accuracy": args.min_clean_accuracy,
                      "min_per_source_retention": args.min_per_source_retention},
        "rows": rows,
        "seed": args.seed,
        "device": str(device), "dtype": args.dtype,
        "key_schedule": "one CPU-generator key per task, moved to the model device and shared across candidate choices",
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").is_file() else None,
        "test_sha256": sha256(args.test_jsonl),
        "source_sha256": {"scripts/evaluate_task_utility.py": sha256(Path(__file__)),
                          **{f"raw/{name}": sha256(path) for name, path in raw_paths.items()}},
        "limitations": ["multiple-choice utility on held-out ARC/SciQ only", "not a substitute for every paper task benchmark", "clean/encrypted scores use the same per-example key across candidate choices", "no key attack is performed", "over-long stems are head-truncated; the option list and trailing 'Answer:' cue are always preserved (count in prompt_truncated_count)"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records_scored": scored, "clean_accuracy": clean_accuracy, "encrypted_accuracy": encrypted_accuracy, "retention": report["accuracy_retention_vs_clean"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
