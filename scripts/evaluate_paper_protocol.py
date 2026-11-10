#!/usr/bin/env python3
"""Run the paper-shaped no-key KNN/VMA evaluation on fixed attack sets.

KNN is evaluated with the public input embedding vocabulary. VMA is evaluated
layer by layer with full-vocabulary autoregressive matching and own-prefix
feedback. The attack receives neither client keys nor target IDs; IDs are used
only after ranking to score ASR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers

from osnip_repro.attacks import topk_counts, vocabulary_topk
from osnip_repro.data import read_text_jsonl
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.evaluation_journal import EvaluationJournal
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project, encrypt_embeddings
from osnip_repro.vma import autoregressive_match, hidden_at_layer, model_candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stopwords() -> tuple[set[str], str]:
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        return set(nltk_stopwords.words("english")), "nltk.corpus.stopwords"
    except LookupError:
        vendored = ROOT / "references/nltk_stopwords_english.txt"
        if vendored.is_file():
            return {line.strip() for line in vendored.read_text(encoding="utf-8").splitlines() if line.strip()}, "references/nltk_stopwords_english.txt"
        raise RuntimeError("NLTK English stopwords corpus is missing and the vendored snapshot is unavailable")
    except ImportError:
        vendored = ROOT / "references/nltk_stopwords_english.txt"
        if vendored.is_file():
            return {line.strip() for line in vendored.read_text(encoding="utf-8").splitlines() if line.strip()}, "references/nltk_stopwords_english.txt"
        raise RuntimeError("nltk is required for Clean ASR; install the repository requirements")


def word_stopword_mask(tokenizer: Any, text: str, ids: list[int], words: set[str]) -> list[bool]:
    """Whole-word stopword mask aligned to token positions.

    Piece-level matching mislabels fragments of content words as stop words
    (shrinking the Clean denominator) and never excludes multi-piece stop
    words (pulling Clean toward Total). Map each token offset back to its
    whitespace-delimited source word and test that word instead.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) < len(ids):
        raise ValueError("offset mapping shorter than tokenized ids; cannot align stop words")
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    result: list[bool] = []
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        # Byte-level BPE (Llama/Qwen) includes the leading space in a
        # word-initial token's offset, e.g. " the" -> (14, 18). Matching the
        # raw offset against a \S+ span misses the word entirely and marks all
        # such stopwords as content, which collapses Clean ASR toward Total ASR.
        while start < end and text[start].isspace():
            start += 1
        word = ""
        if start < end:
            for span_start, span_end in spans:
                if span_start <= start < span_end:
                    word = re.sub(r"^[^a-z]+|[^a-z]+$", "", text[span_start:span_end].lower())
                    break
        result.append(bool(word and word in words))
    # Observations may be truncated to max_tokens; the mask must match the
    # evaluated positions exactly or Clean denominators over-count.
    return result[:len(ids)]


def bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    generator = torch.Generator().manual_seed(seed)
    data = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(0, len(values), (samples, len(values)), generator=generator)
    estimates = data[indices].mean(dim=1).sort().values
    lo = float(estimates[max(0, int(0.025 * samples) - 1)])
    hi = float(estimates[min(samples - 1, int(0.975 * samples))])
    return [lo, hi]


def rate_summary(values: list[float], seed: int, samples: int = 2000) -> dict[str, Any]:
    mean = sum(values) / max(len(values), 1)
    return {"mean": mean, "bootstrap_95ci": bootstrap_ci(values, seed, samples), "n_prompts": len(values)}



def knn_bootstrap(rows: list[dict[str, Any]], seed: int, samples: int) -> dict[str, Any]:
    """Resample complete prompts and recompute token-weighted hit ratios."""
    if not rows or samples < 2:
        raise ValueError("nonempty rows and bootstrap-samples >= 2 are required")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(rows), (samples, len(rows)), generator=generator)
    denominators = torch.tensor([row["positions"] for row in rows], dtype=torch.float64)[indices].sum(-1)
    result = {}
    for side in ("clean", "encrypted"):
        result[side] = {}
        for k in ("top1", "top5", "top10"):
            values = torch.tensor([row[side][k] for row in rows], dtype=torch.float64)
            estimates = values[indices].sum(-1) / denominators
            result[side][k] = {"bootstrap_95ci": torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
                               "n_prompts": len(rows)}
    return result


def require_clean_positive_control(rates: dict[str, Any], threshold: float = 0.99) -> bool:
    """The clean branch must invert near-perfectly or the attack pipeline is invalid.

    The paper's non-private KNN ASR is 1.000; on tied Llama embeddings every
    clean token self-matches at distance 0. A 0.95 floor would let a broken
    tokenizer/vocabulary pipeline pass, so the formal path requires 0.99.
    """
    valid = bool(rates.get("clean", {}).get("top1", 0.0) >= threshold)
    if not valid:
        raise ValueError(f"clean KNN positive control top-1 {rates.get('clean', {}).get('top1')} "
                         f"below {threshold}; attack pipeline invalid")
    return valid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--knn-jsonl", required=True, type=Path)
    ap.add_argument("--vma-jsonl", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 5, 10, 15])
    ap.add_argument("--keys-per-prompt", type=int, default=5)
    ap.add_argument("--max-knn-samples", type=int, default=1000)
    ap.add_argument("--max-vma-prompts", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--vocab-chunk", type=int, default=2048)
    ap.add_argument("--candidate-chunk", type=int, default=256)
    ap.add_argument("--candidate-mode", choices=("full", "proposal"), default="full", help="full is required for formal security; proposal is a reduced-candidate engineering smoke")
    ap.add_argument("--candidate-k", type=int, default=256, help="public next-token proposal size in proposal mode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--knn-report", type=Path, help="reuse a bound complete KNN report, replaying its client key draws")
    ap.add_argument("--resume-vma", action="store_true")
    ap.add_argument("--knn-only", action="store_true", help="finish and save full KNN independently; no VMA/security claim")
    ap.add_argument("--progress-every", type=int, default=25)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    if args.bootstrap_samples < 2:
        raise ValueError("bootstrap-samples must be >= 2")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if min(args.keys_per_prompt, args.max_knn_samples, args.max_vma_prompts, args.max_tokens, args.vocab_chunk, args.candidate_chunk, args.candidate_k) < 1:
        raise ValueError("sample, key, token, candidate and chunk limits must be positive")
    if any(layer < 1 for layer in args.layers):
        raise ValueError("VMA layers must be decoder layers >= 1")
    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    if any(layer > model.config.num_hidden_layers for layer in args.layers):
        raise ValueError(f"requested VMA layer exceeds model depth {model.config.num_hidden_layers}")
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    words, stopword_source = stopwords()
    knn_texts = read_text_jsonl(args.knn_jsonl, limit=args.max_knn_samples)
    vma_texts = read_text_jsonl(args.vma_jsonl, limit=args.max_vma_prompts)
    vocabulary = model.get_input_embeddings().weight.detach()

    knn_counts = {side: {f"top{k}": 0 for k in (1, 5, 10)} for side in ("clean", "encrypted")}
    knn_positions = 0
    client_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    # Independent stream so VMA key draws never depend on KNN-stage parameters
    # (sample counts, jsonl length); keeps the 5-keys-per-instance seed contract.
    # Both streams are CPU generators moved to the model device: a CUDA generator
    # with the same seed produces a different stream, which would make the
    # reported 3B keys non-comparable with the CPU/streamed/training paths.
    vma_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    started = time.monotonic()
    knn_rows = []
    reused_knn = None
    if args.knn_report is not None:
        reused_knn = json.loads(args.knn_report.read_text())
        expected = {"checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(args.knn_jsonl),
                    "samples": len(knn_texts), "seed": args.seed, "dtype": args.dtype,
                    "device": str(device), "vocabulary_size": len(vocabulary),
                    "full_vocabulary": True, "metric": "l2",
                    "key_schedule": "one CPU-generator key per prompt, moved to the model device and shared across tokens"}
        if any(reused_knn.get(k) != v for k, v in expected.items()):
            raise ValueError("reused KNN report does not match checkpoint/data/key/device/protocol")
        if reused_knn.get("source_sha256", {}).get("src/osnip_repro/attacks.py") != sha256(ROOT / "src/osnip_repro/attacks.py"):
            raise ValueError("reused KNN retrieval source changed")
        if len(reused_knn.get("rows", [])) != len(knn_texts):
            raise ValueError("reused KNN report is incomplete")
    with torch.no_grad():
        for sample_index, text in enumerate(knn_texts):
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if ids.shape[1] == 0:
                continue
            h = input_embeddings(model, ids)
            key = torch.randn(config["key_dim"], generator=client_rng).to(device)
            if reused_knn is None:
                z = encrypt_embeddings(encoder, h, key)
                targets = ids[0]
                cosine = torch.nn.functional.cosine_similarity(h[0].float(), z[0].float(), dim=-1)
                row = {"index": sample_index, "positions": int(ids.shape[1]),
                       "mean_signed_cosine": float(cosine.mean()),
                       "mean_abs_cosine": float(cosine.abs().mean()),
                       "positive_cosine_fraction": float((cosine > 0).float().mean())}
                for side, query in (("clean", h[0]), ("encrypted", z[0])):
                    rankings = vocabulary_topk(query.float(), vocabulary, metric="l2", k=10, chunk=args.vocab_chunk)
                    row[side] = {f"top{k}": value for k, value in topk_counts(rankings, targets).items()}
            else:
                row = reused_knn["rows"][sample_index]
                if row["index"] != sample_index or row["positions"] != int(ids.shape[1]):
                    raise ValueError("reused KNN row does not match tokenized input")
            for side in ("clean", "encrypted"):
                for k, value in row[side].items():
                    knn_counts[side][k] += value
            knn_positions += int(ids.shape[1])
            knn_rows.append(row)
            if len(knn_rows) % args.progress_every == 0 or sample_index + 1 == len(knn_texts):
                progress = {"stage": "knn", "samples_completed": len(knn_rows),
                            "samples_target": len(knn_texts), "positions": knn_positions,
                            "elapsed_seconds": round(time.monotonic() - started, 2),
                            "counts": knn_counts, "security_accepted": False}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    if knn_positions == 0:
        raise ValueError("KNN attack set produced no tokens")
    knn_rates = {side: {key: value / knn_positions for key, value in values.items()} for side, values in knn_counts.items()}
    positive_controls_valid = require_clean_positive_control(knn_rates)
    cosine_rows = [row for row in knn_rows if "mean_signed_cosine" in row]
    knn_cosine = None
    if cosine_rows and len(cosine_rows) == len(knn_rows):
        weight = sum(row["positions"] for row in cosine_rows)
        knn_cosine = {
            "mean_signed_cosine": sum(row["mean_signed_cosine"] * row["positions"] for row in cosine_rows) / weight,
            "mean_abs_cosine": sum(row["mean_abs_cosine"] * row["positions"] for row in cosine_rows) / weight,
            "positive_cosine_fraction": sum(row["positive_cosine_fraction"] * row["positions"] for row in cosine_rows) / weight,
            "interpretation": ("real-vocabulary probe: Top-10 ASR is ~0 at signed cos<=0.2, ~24% at 0.25, "
                               "~34% at 0.3; the headline rate is governed by the positive-cosine tail, "
                               "not by the |cos|<=epsilon hinge alone"),
            "bucket_diagnostic": "scripts/diagnose_knn_coverage_buckets.py",
        }

    knn_report = {
        "format": "osnip_knn_report_v2", "status": "completed_knn_only",
        "paper_result": False, "security_accepted": False, "key_attack_performed": False,
        "threat_model": "passive_no_key_semi_honest_server",
        "model_path": str(args.model_path.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(args.knn_jsonl),
        "samples": len(knn_rows), "prompts": len(knn_rows), "positions": knn_positions,
        "keys_per_prompt": 1,
        "counts": knn_counts, "rates": knn_rates, "rows": knn_rows,
        "cosine_diagnostics": knn_cosine,
        "positive_controls_valid": positive_controls_valid,
        "prompt_bootstrap": knn_bootstrap(knn_rows, args.seed, args.bootstrap_samples),
        "bootstrap_samples": args.bootstrap_samples,
        "metric": "l2", "full_vocabulary": True, "vocabulary_size": len(vocabulary),
        "key_schedule": "one CPU-generator key per prompt, moved to the model device and shared across tokens",
        "seed": args.seed, "dtype": args.dtype, "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {"scripts/evaluate_paper_protocol.py": sha256(Path(__file__)),
                          "src/osnip_repro/attacks.py": sha256(ROOT / "src/osnip_repro/attacks.py"),
                          "src/osnip_repro/projection.py": sha256(ROOT / "src/osnip_repro/projection.py")},
        "limitations": ["KNN only; VMA and joint security acceptance remain separate"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    knn_path = args.output if args.knn_only else args.output.with_suffix(".knn.json")
    if args.knn_report is None or knn_path.resolve() != args.knn_report.resolve():
        knn_path.write_text(json.dumps(knn_report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": "knn_complete", "output": str(knn_path.resolve()), "rates": knn_rates}), flush=True)
    if args.knn_only:
        return

    candidate_forwards = {layer: model_candidates(model, layer=layer) for layer in args.layers}
    proposal_history: list[list[int]] = []
    fallback_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if fallback_token is None:
        fallback_token = 0

    @torch.no_grad()
    def propose(prefix: tuple[int, ...]) -> torch.Tensor:
        context = list(prefix) if prefix else [int(fallback_token)]
        context_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids=context_ids, use_cache=False).logits[0, -1]
        candidates = torch.topk(logits, k=min(args.candidate_k, logits.numel())).indices.detach().cpu()
        proposal_history.append(candidates.tolist())
        return candidates

    journal = EvaluationJournal(args.output.with_suffix(".vma_rows.jsonl"), {
        "checkpoint_sha256": knn_report["checkpoint_sha256"],
        "knn_sha256": sha256(args.knn_jsonl), "vma_sha256": sha256(args.vma_jsonl),
        "config_sha256": sha256(config_path), "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").exists() else None,
        "seed": args.seed, "device": str(device), "dtype": args.dtype,
        "key_generator": "cpu", "keys_per_prompt": args.keys_per_prompt, "layers": args.layers,
        "max_tokens": args.max_tokens, "max_knn_samples": args.max_knn_samples,
        "max_vma_prompts": args.max_vma_prompts, "candidate_mode": args.candidate_mode,
        "candidate_k": args.candidate_k, "candidate_chunk": args.candidate_chunk,
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/evaluate_paper_protocol.py", "src/osnip_repro/vma.py",
            "src/osnip_repro/evaluation_journal.py", "src/osnip_repro/projection.py")},
        "stopwords_sha256": hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest(),
    }, resume=args.resume_vma,
       bounds={"prompts": len(vma_texts), "layers": tuple(args.layers),
               "key_min": -1, "keys_per_prompt": args.keys_per_prompt})
    active_observation = {}

    def vma_progress(values: dict[str, int]) -> None:
        progress = {"stage": "vma", **active_observation, **values,
                    "completed_observations": len(journal.rows),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "security_accepted": False}
        args.output.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress), flush=True)

    def match_vma(observed: torch.Tensor, candidate_forward: Any, target_ids: list[int]) -> tuple[dict[str, Any], float]:
        proposal_history.clear()
        result = autoregressive_match(observed, candidate_forward, vocab_size=model.config.vocab_size, chunk=args.candidate_chunk, topk=10, candidate_ids_fn=propose if args.candidate_mode == "proposal" else None, progress=vma_progress)
        if args.candidate_mode == "full":
            return result, 1.0
        coverage = [target_ids[index] in proposal_history[index] for index in range(min(len(target_ids), len(proposal_history)))]
        return result, sum(coverage) / max(len(coverage), 1)

    def score_observation(observed: torch.Tensor, target_ids: list[int], stop_mask: list[bool],
                          prompt: int, layer: int, key: int) -> dict[str, Any]:
        row = journal.get(prompt, layer, key)
        content_count = sum(not value for value in stop_mask)
        if row is None:
            active_observation.update(prompt=prompt, layer=layer, key=key)
            result, coverage = match_vma(observed, candidate_forwards[layer], target_ids)
            hits = [int(a == b) for a, b in zip(result["predictions"], target_ids)]
            prefix_correct = [result["predictions"][:i] == target_ids[:i] for i in range(len(target_ids))]
            row = {"prompt": prompt, "layer": layer, "key": key,
                   "positions": len(target_ids), "content_positions": content_count,
                   "total_top1_hits": sum(hits),
                   "prefix_correct_conditional_hits": sum(hit for hit, ok in zip(hits, prefix_correct) if ok),
                   "prefix_correct_positions": sum(prefix_correct),
                   "clean_top1_hits": sum(hit for hit, stop in zip(hits, stop_mask) if not stop),
                   "candidate_coverage": coverage}
            journal.append(row)
        if row["positions"] != len(target_ids) or row["content_positions"] != content_count:
            raise ValueError("VMA journal token/stopword counts mismatch")
        return row

    vma_prompt_rows: dict[str, dict[int, list[float]]] = {"clean_total": {layer: [] for layer in args.layers}, "clean_clean": {layer: [] for layer in args.layers}}
    encrypted_prompt_rows: dict[int, dict[str, list[float]]] = {layer: {"total": [], "clean": []} for layer in args.layers}
    vma_counts: dict[int, dict[str, dict[str, int]]] = {layer: {"clean": {"total": 0, "clean": 0, "positions": 0, "content_positions": 0}, "encrypted": {"total": 0, "clean": 0, "positions": 0, "content_positions": 0}} for layer in args.layers}
    vma_candidate_coverages: dict[int, dict[str, list[float]]] = {layer: {"clean": [], "encrypted": []} for layer in args.layers}
    vma_rows: list[dict[str, Any]] = []
    vma_clean_rows: list[dict[str, Any]] = []
    vma_truncated = 0
    with torch.no_grad():
        for prompt_index, text in enumerate(vma_texts):
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if ids.shape[1] == 0:
                continue
            if ids.shape[1] > args.max_tokens:
                ids = ids[:, :args.max_tokens]
                vma_truncated += 1
            target_ids = ids[0].tolist()
            mask = torch.ones_like(ids)
            stop_mask = word_stopword_mask(tokenizer, text, target_ids, words)
            h = input_embeddings(model, ids)
            clean_states = {layer: hidden_at_layer(model, layer=layer, inputs_embeds=h, attention_mask=mask)[0] for layer in args.layers}
            encrypted_states: list[dict[int, torch.Tensor]] = []
            for key_index in range(args.keys_per_prompt):
                key = torch.randn(config["key_dim"], generator=vma_rng).to(device)
                z = encrypt_embeddings(encoder, h, key)
                encrypted_states.append({layer: hidden_at_layer(model, layer=layer, inputs_embeds=z, attention_mask=mask)[0] for layer in args.layers})
            for layer in args.layers:
                clean_row = score_observation(clean_states[layer], target_ids, stop_mask, prompt_index, layer, -1)
                vma_clean_rows.append(clean_row)
                vma_candidate_coverages[layer]["clean"].append(clean_row["candidate_coverage"])
                clean_total = clean_row["total_top1_hits"]
                clean_content = clean_row["clean_top1_hits"]
                content_count = clean_row["content_positions"]
                vma_counts[layer]["clean"]["total"] += clean_total
                vma_counts[layer]["clean"]["clean"] += clean_content
                vma_counts[layer]["clean"]["positions"] += len(target_ids)
                vma_counts[layer]["clean"]["content_positions"] += content_count
                vma_prompt_rows["clean_total"][layer].append(clean_total / max(len(target_ids), 1))
                vma_prompt_rows["clean_clean"][layer].append(clean_content / max(content_count, 1))
                prompt_encrypted_total: list[float] = []
                prompt_encrypted_clean: list[float] = []
                for key_index, states in enumerate(encrypted_states):
                    row = score_observation(states[layer], target_ids, stop_mask, prompt_index, layer, key_index)
                    vma_candidate_coverages[layer]["encrypted"].append(row["candidate_coverage"])
                    total = row["total_top1_hits"]
                    content = row["clean_top1_hits"]
                    vma_counts[layer]["encrypted"]["total"] += total
                    vma_counts[layer]["encrypted"]["clean"] += content
                    vma_counts[layer]["encrypted"]["positions"] += len(target_ids)
                    vma_counts[layer]["encrypted"]["content_positions"] += content_count
                    prompt_encrypted_total.append(total / max(len(target_ids), 1))
                    prompt_encrypted_clean.append(content / max(content_count, 1))
                    vma_rows.append(row)

                encrypted_prompt_rows[layer]["total"].append(sum(prompt_encrypted_total) / max(len(prompt_encrypted_total), 1))
                encrypted_prompt_rows[layer]["clean"].append(sum(prompt_encrypted_clean) / max(len(prompt_encrypted_clean), 1))

    vma_summary: dict[str, Any] = {}
    for layer in args.layers:
        clean_pos = vma_counts[layer]["clean"]["positions"]
        encrypted_pos = vma_counts[layer]["encrypted"]["positions"]
        vma_summary[str(layer)] = {
            "clean_positive_control": {"total_asr": vma_counts[layer]["clean"]["total"] / max(clean_pos, 1), "clean_asr": vma_counts[layer]["clean"]["clean"] / max(vma_counts[layer]["clean"]["content_positions"], 1)},
            "encrypted": {"total_asr": vma_counts[layer]["encrypted"]["total"] / max(encrypted_pos, 1), "clean_asr": vma_counts[layer]["encrypted"]["clean"] / max(vma_counts[layer]["encrypted"]["content_positions"], 1)},
            "candidate_coverage": {
                "clean_mean": sum(vma_candidate_coverages[layer]["clean"]) / max(len(vma_candidate_coverages[layer]["clean"]), 1),
                "encrypted_mean": sum(vma_candidate_coverages[layer]["encrypted"]) / max(len(vma_candidate_coverages[layer]["encrypted"]), 1),
            },
            "prompt_paired": {
                "clean_total": rate_summary(vma_prompt_rows["clean_total"][layer], args.seed + layer, args.bootstrap_samples),
                "clean_clean": rate_summary(vma_prompt_rows["clean_clean"][layer], args.seed + 1000 + layer, args.bootstrap_samples),
                "encrypted_total": rate_summary(encrypted_prompt_rows[layer]["total"], args.seed + 2000 + layer, args.bootstrap_samples),
                "encrypted_clean": rate_summary(encrypted_prompt_rows[layer]["clean"], args.seed + 3000 + layer, args.bootstrap_samples),
                "paired_difference_clean_minus_encrypted": {
                    "total": rate_summary([a - b for a, b in zip(vma_prompt_rows["clean_total"][layer], encrypted_prompt_rows[layer]["total"])], args.seed + 4000 + layer, args.bootstrap_samples),
                    "clean": rate_summary([a - b for a, b in zip(vma_prompt_rows["clean_clean"][layer], encrypted_prompt_rows[layer]["clean"])], args.seed + 5000 + layer, args.bootstrap_samples),
                },
            },
        }
    for layer in args.layers:
        stats = vma_counts[layer]["clean"]
        if stats["positions"] >= 100:
            exclusion = 1.0 - stats["content_positions"] / max(stats["positions"], 1)
            if exclusion < 0.05:
                raise ValueError(
                    f"VMA stopword mask excludes only {exclusion:.1%} of tokens at layer {layer}; "
                    "Clean ASR would collapse toward Total ASR")
            control = vma_summary[str(layer)]["clean_positive_control"]
            if control["total_asr"] < 0.5:
                raise ValueError(
                    f"VMA clean positive control total ASR {control['total_asr']:.3f} is below 0.5 "
                    f"at layer {layer}; the attack pipeline is invalid")
    report = {
        "format": "osnip_paper_protocol_evaluation_v1",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "status": "paper_protocol_diagnostic",
        "paper_result": False,
        "security_accepted": False,
        "threat_model": "passive_no_key_semi_honest_server",
        "key_attack_performed": False,
        "model_path": str(args.model_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "knn_jsonl": str(args.knn_jsonl.resolve()),
        "vma_jsonl": str(args.vma_jsonl.resolve()),
        "knn": {"samples": len(knn_rows), "prompts": len(knn_rows), "keys_per_prompt": 1, "positions": knn_positions, "counts": knn_counts, "rates": knn_rates, "rows": knn_rows, "cosine_diagnostics": knn_cosine, "metric": "l2", "full_vocabulary": True, "vocabulary_size": len(vocabulary), "prompt_bootstrap": knn_report["prompt_bootstrap"], "key_schedule": knn_report["key_schedule"]},
        "vma": {"prompts": len(vma_texts), "keys_per_prompt": args.keys_per_prompt, "layers": args.layers, "summary": vma_summary, "rows": vma_rows, "clean_rows": vma_clean_rows, "max_tokens": args.max_tokens, "prompt_truncated_count": vma_truncated, "distance_metric": "l1", "prefix_policy": "own_top1_predictions", "key_schedule": "five CPU-generator keys per prompt, moved to the model device and shared across tokens", "stopwords_sha256": hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest(), "candidate_mode": args.candidate_mode, "candidate_k": args.candidate_k, "journal_binding_sha256": journal.binding, "prefix_cache": True, "full_vocabulary": args.candidate_mode == "full",
                "layer_indexing": "1-indexed: layer L = HF hidden_states[L]; final norm included when L equals decoder depth",
                "stopword_policy": "whole-word via tokenizer offsets against the NLTK english corpus",
                "headline_aggregation": "token_micro; prompt_macro in summary[*].prompt_paired"},
        "stopword_policy": {"source": stopword_source, "matching": "whole-word via tokenizer offsets; pieces inherit their source word's verdict"},
        "seeds": {"sampling": 42, "key": args.seed, "bootstrap": args.seed},
        "source_sha256": {"scripts/evaluate_paper_protocol.py": sha256(Path(__file__)), "src/osnip_repro/vma.py": sha256(ROOT / "src/osnip_repro/vma.py"), "references/nltk_stopwords_english.txt": sha256(ROOT / "references/nltk_stopwords_english.txt")},
        "data_sha256": {"knn": sha256(args.knn_jsonl), "vma": sha256(args.vma_jsonl)},
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").is_file() else None,
        "limitations": (["proposal candidate pool; not comparable to full-vocabulary ASR"] if args.candidate_mode == "proposal" else []) + ["objective-aligned configurable encryptor, not the undisclosed author implementation", "full-vocabulary VMA can be computationally expensive on CPU", "confidence intervals are bootstrap prompt-level intervals", "task utility and sliding PPL are separate reports", "no key attack is performed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "knn": knn_rates, "vma_layers": args.layers, "prompts": len(vma_texts), "candidate_mode": args.candidate_mode, "security_accepted": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
