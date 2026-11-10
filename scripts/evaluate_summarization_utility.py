#!/usr/bin/env python3
"""Evaluate clean/encrypted abstractive summarization utility.

The input JSONL must contain an article and reference summary. Accepted field
aliases are ``article``/``highlights`` and ``text``/``summary``. The same
frozen causal LM and client encryptor generate clean and encrypted summaries;
no key or encryptor is exposed to the server-side metric code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import transformers
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.data import _iter_json_values
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project, encrypt_embeddings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_fields(record: dict[str, Any]) -> tuple[str, str]:
    article = record.get("article", record.get("text"))
    reference = record.get("highlights", record.get("summary"))
    if not isinstance(article, str) or not article.strip():
        raise ValueError("summary record needs a non-empty article or text field")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("summary record needs a non-empty highlights or summary field")
    return article.strip(), reference.strip()


def load_records(path: Path, limit: int | None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for record in _iter_json_values(path):
        rows.append(record_fields(record))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"no summary records found in {path}")
    return rows


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token_left == token_right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    if not candidate_tokens or not reference_tokens:
        return 0.0
    length = lcs_length(candidate_tokens, reference_tokens)
    precision = length / len(candidate_tokens)
    recall = length / len(reference_tokens)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--data-jsonl", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--max-input-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--key-dim", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--skip-bertscore", action="store_true")
    ap.add_argument("--require-bertscore", action="store_true")
    return ap.parse_args()


def generate_summary(model: Any, tokenizer: Any, embeddings: torch.Tensor, mask: torch.Tensor,
                     *, max_new_tokens: int, num_beams: int) -> str:
    # The server receives only the representation and padding mask. With a
    # decoder-only model and inputs_embeds alone, generate returns continuation
    # IDs; sending the original input_ids would expose the protected prompt.
    generated = model.generate(
        inputs_embeds=embeddings,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def resolve_scorer_snapshot(model_name: str) -> tuple[dict, dict]:
    """Load a byte-locked local scorer; evaluation never downloads weights."""
    if model_name != "roberta-large":
        raise ValueError("formal paired BERTScore currently requires the fixed roberta-large scorer")
    manifest_path = ROOT / "model/roberta-large/download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("download and verify the fixed local BERTScore snapshot first")
    manifest = json.loads(manifest_path.read_text())
    directory = Path(manifest["local_dir"]).resolve()
    required = {"config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"}
    files = manifest.get("files", {})
    if manifest.get("model_id") != model_name or not required.issubset(files):
        raise ValueError("incomplete or mismatched BERTScore model manifest")
    for name, metadata in files.items():
        if Path(name).name != name or sha256(directory/name) != metadata["sha256"]:
            raise ValueError("BERTScore snapshot bytes differ from the locked manifest")
    from bert_score.utils import model2layers
    return ({"model_type": str(directory), "num_layers": model2layers[model_name]},
            {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
             "directory": str(directory), "files": files, "model_id": model_name,
             "source": manifest.get("mirror_model_id", manifest.get("repo_id")),
             "revision": manifest.get("mirror_revision", manifest.get("revision")),
             "hf_mirror_equivalence_independently_verified": False})


def compute_paired_bertscore(clean: list[str], encrypted: list[str], references: list[str],
                             model_name: str, device: torch.device) -> dict[str, Any]:
    """Score both branches in one call so tokenizer/model/scorer settings agree."""
    if not references or len(clean) != len(references) or len(encrypted) != len(references):
        raise ValueError("paired BERTScore requires aligned nonempty generations and references")
    try:
        from bert_score import score
    except ImportError:
        return {"available": False, "model": model_name, "reason": "bert-score is not installed"}
    scorer_options, provenance = resolve_scorer_snapshot(model_name)
    (precision, recall, f1), scorer_hash = score(
        clean + encrypted, references + references, **scorer_options, lang="en",
        device=str(device), verbose=False, return_hash=True,
    )
    # Detect source replacement while the scorer was active.
    if resolve_scorer_snapshot(model_name)[1] != provenance:
        raise ValueError("BERTScore snapshot changed while scoring")
    if not all(values.numel() == 2 * len(references) and torch.isfinite(values).all()
               for values in (precision, recall, f1)):
        raise ValueError("BERTScore returned non-finite or unaligned per-example scores")
    n = len(references)
    branches = {side: {name: float(values[begin:end].mean())
                      for name, values in (("precision", precision), ("recall", recall), ("f1", f1))}
                for side, begin, end in (("clean", 0, n), ("encrypted", n, 2*n))}
    clean_f1 = branches["clean"]["f1"]
    return {"available": True, "model": model_name, "scorer_hash": scorer_hash, "provenance": provenance,
            **branches, "f1_retention": branches["encrypted"]["f1"] / clean_f1 if clean_f1 > 0 else None,
            "clean_positive": clean_f1 > 0,
            "rows": [{"index": index, "clean_f1": float(f1[index]), "encrypted_f1": float(f1[n+index])}
                     for index in range(n)]}


def main() -> None:
    args = parse_args()
    if args.require_bertscore and args.skip_bertscore:
        raise ValueError("required BERTScore cannot be skipped")
    if min(args.max_samples, args.max_input_length, args.max_new_tokens, args.num_beams) < 1:
        raise ValueError("sample, length, generation and beam limits must be positive")
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    config_path = args.checkpoint.parent / "config.json"
    bound_inputs = {str(path.resolve()): sha256(path) for path in (args.checkpoint, config_path, args.data_jsonl)}
    config = json.loads(config_path.read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    records = load_records(args.data_jsonl, args.max_samples)
    torch.manual_seed(args.seed)
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    key_dim = int(config["key_dim"] if args.key_dim is None else args.key_dim)
    if key_dim != int(config["key_dim"]):
        raise ValueError("--key-dim must agree with checkpoint config")
    key_rng = torch.Generator(device="cpu").manual_seed(args.seed + 15485863)
    clean_summaries: list[str] = []
    encrypted_summaries: list[str] = []
    references: list[str] = []
    rows: list[dict[str, Any]] = []
    truncated = 0
    with torch.no_grad():
        for index, (article, reference) in enumerate(records):
            # Keep the trailing "Summary:" generation cue: truncate the article,
            # never the end of the assembled prompt. Right truncation of the full
            # prompt silently deleted the cue and the article tail.
            prefix = "Summarize the following article.\n\n"
            cue = "\n\nSummary:"
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            cue_ids = tokenizer(cue, add_special_tokens=False)["input_ids"]
            special_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            budget = args.max_input_length - len(prefix_ids) - len(cue_ids) - len(special_ids)
            if budget < 1:
                raise ValueError("max-input-length leaves no room for the article text")
            article_ids = tokenizer(article, add_special_tokens=False, truncation=True,
                                    max_length=budget)["input_ids"]
            full_article_ids = tokenizer(article, add_special_tokens=False)["input_ids"]
            if len(full_article_ids) > len(article_ids):
                truncated += 1
            input_ids = torch.tensor([special_ids + prefix_ids + article_ids + cue_ids], device=device)
            mask = torch.ones_like(input_ids)
            h = input_embeddings(model, input_ids)
            key = torch.randn((1, key_dim), generator=key_rng).to(device)
            z = encrypt_embeddings(encoder, h, key, attention_mask=mask)
            clean = generate_summary(model, tokenizer, h, mask, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            encrypted = generate_summary(model, tokenizer, z, mask, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            clean_summaries.append(clean)
            encrypted_summaries.append(encrypted)
            references.append(reference)
            rows.append({
                "index": index,
                "reference": reference,
                "clean_summary": clean,
                "encrypted_summary": encrypted,
                "clean_rouge_l": rouge_l_f1(clean, reference),
                "encrypted_rouge_l": rouge_l_f1(encrypted, reference),
            })
    clean_rouge = sum(row["clean_rouge_l"] for row in rows) / max(len(rows), 1)
    encrypted_rouge = sum(row["encrypted_rouge_l"] for row in rows) / max(len(rows), 1)
    # Release the full server and last GPU observations before the scorer loads.
    del model, encoder, h, z, key
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    bertscore = ({"available": False, "skipped": True, "model": args.bertscore_model}
                 if args.skip_bertscore else
                 compute_paired_bertscore(clean_summaries, encrypted_summaries, references, args.bertscore_model, device))
    if args.require_bertscore and not bertscore.get("available"):
        raise RuntimeError("formal summarization requires completed clean/encrypted BERTScore")
    if any(sha256(Path(path)) != digest for path, digest in bound_inputs.items()):
        raise ValueError("summarization inputs changed during evaluation")
    report = {
        "format": "osnip_summarization_utility_report_v2",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "status": "completed_with_limits",
        "paper_result": False,
        "security_accepted": False,
        "threat_model": "passive_no_key_semi_honest_server",
        "key_attack_performed": False,
        "model_path": str(args.model_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "data_jsonl": str(args.data_jsonl.resolve()),
        "records": len(rows),
        "encryptor_architecture": config.get("encryptor_architecture", "mlp_residual_v1"),
        "clean": {"rouge_l": clean_rouge},
        "encrypted": {"rouge_l": encrypted_rouge},
        "retention": {"rouge_l": encrypted_rouge / clean_rouge if clean_rouge > 0 else None,
                      "bertscore_f1": bertscore.get("f1_retention")},
        "positive_controls": {"clean_rouge_l_positive": clean_rouge > 0,
                              "clean_bertscore_f1_positive": bertscore.get("clean_positive")},
        "bertscore": bertscore,
        "bertscore_clean": {"available": bertscore.get("available", False), **bertscore.get("clean", {})},
        "bertscore_encrypted": {"available": bertscore.get("available", False), **bertscore.get("encrypted", {})},
        "seed": args.seed, "key_seed": args.seed + 15485863,
        "key_schedule": "one independent CPU generator key per prompt shared across tokens",
        "server_input": "inputs_embeds_and_attention_mask_only_no_original_input_ids",
        "device": str(device), "dtype": args.dtype,
        "generation": {"max_input_length": args.max_input_length, "max_new_tokens": args.max_new_tokens,
                       "num_beams": args.num_beams, "article_truncated_count": truncated,
                       "truncation_policy": "truncate article only; always keep the trailing 'Summary:' cue"},
        "rows": rows,
        "checkpoint_sha256": bound_inputs[str(args.checkpoint.resolve())],
        "model_manifest_sha256": sha256(args.checkpoint.parent / "model_manifest.json") if (args.checkpoint.parent / "model_manifest.json").is_file() else None,
        "checkpoint_config_sha256": bound_inputs[str(config_path.resolve())],
        "bound_input_sha256": bound_inputs,
        "model_file_sha256": {path.name: sha256(path) for path in sorted(args.model_path.iterdir())
                              if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model", ".txt")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in [Path(__file__).resolve(), *sorted((ROOT / "src/osnip_repro").glob("*.py"))]},
        "data_sha256": sha256(args.data_jsonl),
        "limitations": [
            "CNN/DailyMail data must be supplied as a local JSONL snapshot",
            "ROUGE-L is implemented locally; BERTScore requires optional bert-score dependency",
            "this report measures utility and is not a privacy result",
            "articles longer than the input budget are truncated; the 'Summary:' cue is always retained and the count is reported",
            "no key attack is performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(rows), "clean_rouge_l": clean_rouge, "encrypted_rouge_l": encrypted_rouge, "bertscore_available": bertscore.get("available", False)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
