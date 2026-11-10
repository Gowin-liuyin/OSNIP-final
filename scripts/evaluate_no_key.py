#!/usr/bin/env python3
"""Joint utility and no-key input-vocabulary probe on identical observations.

This diagnostic is not the paper's autoregressive hidden-state attack and
cannot certify privacy, even when every diagnostic screen passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers
from osnip_repro.attacks import topk_counts, vocabulary_topk
from osnip_repro.data import read_text_jsonl, tokenize_texts
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.losses import utility_kl
from osnip_repro.modeling import causal_ppl, forward_logits, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project, encrypt_embeddings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-jsonl", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-chunk", type=int, default=2048)
    parser.add_argument("--skip-vocabulary-probe", action="store_true",
                        help="compute utility only; the formal KNN/VMA report remains separate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-ppl-ratio", type=float, default=2.0,
                        help="engineering utility gate; report only, not a paper universal threshold")
    parser.add_argument("--max-mean-kl", type=float, default=1.0,
                        help="engineering utility gate; report only, not a paper universal threshold")
    args = parser.parse_args()
    if min(args.max_samples, args.batch_size, args.vocab_chunk) <= 0 or args.max_length < 2:
        parser.error("positive counts and max-length >= 2 are required")
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and evaluation server models disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    if model.get_input_embeddings().embedding_dim != config["width"]:
        raise ValueError("checkpoint and model embedding widths disagree")
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    # An independent client RNG keeps observations invariant to model loading
    # and encoder construction. No key or encoder is passed into retrieval.
    client_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    counts = {side: {metric: {k: 0 for k in (1, 5, 10)} for metric in ("l2", "l1")} for side in ("clean", "encrypted")}
    clean_nll = encrypted_nll = cosine_sum = kl_sum = relative_sum = 0.0
    token_count = row_count = 0
    max_norm_error = 0.0
    observation_digest = hashlib.sha256()
    vocabulary = model.get_input_embeddings().weight.detach()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenize_texts(tokenizer, texts[start:start + args.batch_size], max_length=args.max_length)
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            # This evaluator uses right padding so every nonpadding target has
            # a nonpadding predecessor; left padding needs another NLL mask.
            if ((~mask[:, :-1].bool()) & mask[:, 1:].bool()).any():
                raise ValueError("right padding is required for this evaluator")
            h = input_embeddings(model, ids)
            client_key = torch.randn(ids.shape[0], config["key_dim"], generator=client_rng).to(device)
            z = encrypt_embeddings(encoder, h, client_key, attention_mask=mask)
            clean_logits = forward_logits(model, h, mask)
            encrypted_logits = forward_logits(model, z, mask)
            valid = mask.bool()
            causal_valid = mask[:, 1:].bool()
            h_rows, z_rows, targets = h[valid], z[valid], ids[valid]
            rows = targets.numel()
            predictions = int(causal_valid.sum())
            clean_causal = clean_logits[:, :-1][causal_valid]
            encrypted_causal = encrypted_logits[:, :-1][causal_valid]
            if not all(torch.isfinite(t).all() for t in (h_rows, z_rows, clean_causal, encrypted_causal)):
                raise ValueError("non-finite observation or logits; evaluation is invalid")
            clean_nll += float(causal_ppl(clean_logits, ids, mask)) * predictions
            encrypted_nll += float(causal_ppl(encrypted_logits, ids, mask)) * predictions
            kl_sum += float(utility_kl(clean_causal, encrypted_causal)) * predictions
            cosine_sum += float(torch.nn.functional.cosine_similarity(h_rows.float(), z_rows.float(), dim=-1).sum())
            relative_sum += float(((z_rows.float() - h_rows.float()).norm(dim=-1) / h_rows.float().norm(dim=-1).clamp_min(1e-12)).sum())
            max_norm_error = max(max_norm_error, float((z_rows.float().norm(dim=-1) - h_rows.float().norm(dim=-1)).abs().max()))
            observation_digest.update(targets.cpu().contiguous().numpy().tobytes())
            observation_digest.update(z_rows.float().cpu().contiguous().numpy().tobytes())
            if not args.skip_vocabulary_probe:
                for side, observation in (("clean", h_rows), ("encrypted", z_rows)):
                    for metric in ("l2", "l1"):
                        rankings = vocabulary_topk(observation, vocabulary, metric=metric, k=10, chunk=args.vocab_chunk)
                        for k, hits in topk_counts(rankings, targets).items():
                            counts[side][metric][k] += hits
            token_count += predictions
            row_count += rows
    if token_count == 0 or row_count == 0:
        raise ValueError("no valid causal targets or attack observations")
    metrics = {side: {metric: {f"top{k}": hits / row_count for k, hits in ks.items()} for metric, ks in mm.items()} for side, mm in counts.items()}
    clean_mean_nll, encrypted_mean_nll = clean_nll / token_count, encrypted_nll / token_count
    if max(clean_mean_nll, encrypted_mean_nll) > 700:
        raise ValueError("NLL is too large for a finite perplexity report")
    controls_valid = (None if args.skip_vocabulary_probe else
                      all(mm["top1"] >= 0.95 for mm in metrics["clean"].values()))
    utility = {
        "clean_nll": clean_mean_nll, "encrypted_nll": encrypted_mean_nll,
        "clean_ppl": math.exp(clean_mean_nll), "encrypted_ppl": math.exp(encrypted_mean_nll),
        "ppl_ratio": math.exp(encrypted_mean_nll - clean_mean_nll), "mean_logit_kl": kl_sum / max(token_count, 1),
    }
    utility_failed = (utility["ppl_ratio"] > args.max_ppl_ratio or
                      utility["mean_logit_kl"] > args.max_mean_kl)
    privacy_failed = (False if args.skip_vocabulary_probe else
                      any(metrics["encrypted"][metric]["top1"] >= 0.95 for metric in ("l1", "l2")))
    sources = [Path(__file__).resolve(), *sorted((ROOT / "src/osnip_repro").glob("*.py"))]
    model_files = sorted(path for path in args.model_path.iterdir() if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model", ".txt"))
    report = {
        "format": "osnip_joint_no_key_probe_v1",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "paper_result": False,
        "security_accepted": False,
        "status": (("utility_failed" if utility_failed else "utility_evaluated_privacy_not_run")
                   if args.skip_vocabulary_probe else
                   "invalid_positive_control" if not controls_valid else
                   "utility_and_privacy_failed" if utility_failed and privacy_failed else
                   "utility_failed" if utility_failed else
                   "privacy_probe_failed" if privacy_failed else
                   "incomplete_security_evaluation"),
        "threat_model": "passive_server_knows_model_and_vocabulary_but_no_client_key",
        "attack_scope": ("not_evaluated" if args.skip_vocabulary_probe else
                         "input_embedding_full_vocabulary_L1_L2_only"),
        "vocabulary_probe_performed": not args.skip_vocabulary_probe,
        "key_attack_performed": False,
        "same_observations_for_utility_and_attack": None if args.skip_vocabulary_probe else True,
        "key_schedule": "one CPU-generator key per example, moved to the model device",
        "positive_controls_valid": controls_valid,
        "samples": len(texts), "positions": row_count, "causal_targets": token_count,
        "utility": {**utility, "gate": {
            "passed": not utility_failed,
            "max_ppl_ratio": args.max_ppl_ratio,
            "max_mean_kl": args.max_mean_kl,
            "is_engineering_threshold": True,
        }},
        "representation": {"mean_cosine": cosine_sum / row_count, "relative_perturbation": relative_sum / row_count, "max_absolute_norm_error": max_norm_error},
        "metrics": None if args.skip_vocabulary_probe else metrics,
        "counts": (None if args.skip_vocabulary_probe else
                   {side: {metric: {f"top{k}": n for k, n in ks.items()} for metric, ks in mm.items()} for side, mm in counts.items()}),
        "configuration": {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_sha256": sha256(args.checkpoint), "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").is_file() else None, "checkpoint_config_sha256": sha256(config_path),
        "data_sha256": sha256(args.data_jsonl), "observations_sha256": observation_digest.hexdigest(),
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sources},
        "model_file_sha256": {path.name: sha256(path) for path in model_files},
        "limitations": [
            "not the paper autoregressive hidden-state vocabulary-matching attack",
            "one proxy model, one training seed and one client RNG seed; small validation prefixes only",
            "engineering utility thresholds are explicit but are not universal paper acceptance criteria",
            "validation has been used for exploration; final held-out test and task utility are pending",
            "no security certificate, even if the probe later passes",
        ] + (["input-vocabulary probe skipped; null counts/metrics are not attack results; use evaluate_paper_protocol.py for formal KNN/VMA"]
             if args.skip_vocabulary_probe else []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "status": report["status"], "positions": row_count, "utility": report["utility"], "metrics": report["metrics"], "vocabulary_probe_performed": report["vocabulary_probe_performed"]}, indent=2))


if __name__ == "__main__":
    main()
