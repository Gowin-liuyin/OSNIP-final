#!/usr/bin/env python3
"""Construct the paper-style KNN and VMA attack pools from raw sources."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osnip_repro.data import _iter_json_values, file_sha256, normalize_text, text_sha256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpaca", required=True, type=Path)
    ap.add_argument("--arc-easy", required=True, type=Path)
    ap.add_argument("--arc-challenge", required=True, type=Path)
    ap.add_argument("--sciq", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--knn-reserved", type=Path, help="reserved Alpaca instruction+input JSONL emitted by prepare_osnip_datasets.py")
    ap.add_argument("--wikitext", type=Path)
    ap.add_argument("--knn-samples", type=int, default=1000)
    ap.add_argument("--vma-prompts", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=512)
    ap.add_argument("--wikitext-cap", type=int, default=150000)
    ap.add_argument("--pool-cap", type=int, default=300000, help="Appendix C.2 merged candidate-pool cap")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def record_text(record: dict[str, Any], source: str, *, include_answer: bool = False, shuffle_options: bool = False, option_rng: random.Random | None = None) -> str:
    if source in {"alpaca", "alpaca_knn_holdout"}:
        if source == "alpaca_knn_holdout":
            return str(record.get("text", "")).strip()
        pieces = [str(record.get("instruction", "")).strip(), str(record.get("input", "")).strip()]
        return "\n\n".join(piece for piece in pieces if piece)
    if source.startswith("arc"):
        choices = record.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        values = choices.get("text", []) if isinstance(choices, dict) else []
        options = "\n".join(f"{label}: {value}" for label, value in zip(labels, values))
        pieces = [str(record.get("question", "")).strip(), options]
        if include_answer:
            pieces.append(f"Answer: {record.get('answerKey', '')}")
        return "\n".join(piece for piece in pieces if piece)
    if source == "sciq":
        options = [record.get("correct_answer"), record.get("distractor1"), record.get("distractor2"), record.get("distractor3")]
        options = [str(value).strip() for value in options if value is not None and str(value).strip()]
        if shuffle_options:
            if option_rng is None:
                raise ValueError("option_rng is required when shuffle_options is enabled")
            option_rng.shuffle(options)
        option_text = " | ".join(options)
        pieces = [str(record.get("question", "")).strip(), f"Options: {option_text}" if option_text else ""]
        return "\n".join(piece for piece in pieces if piece)
    if source == "wikitext":
        return str(record.get("text", "")).strip()
    raise ValueError(source)


def load_texts(path: Path, source: str, min_chars: int, max_chars: int, *, include_answer: bool = False, cap: int | None = None, shuffle_options: bool = False, seed: int = 42) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(_iter_json_values(path)):
        text = record_text(record, source, include_answer=include_answer, shuffle_options=shuffle_options, option_rng=random.Random(seed + index))
        length = len(normalize_text(text))
        if min_chars <= length <= max_chars:
            source_index = record.get("source_index", index)
            rows.append({"text": text, "source": source, "source_index": int(source_index), "text_sha256": text_sha256(text)})
            if cap is not None and len(rows) >= cap:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def main() -> None:
    args = parse_args()
    if args.knn_samples < 1 or args.vma_prompts < 1 or args.min_chars < 0 or args.max_chars < args.min_chars:
        raise ValueError("invalid sample or character limits")
    paths = {"alpaca": args.alpaca, "arc_easy": args.arc_easy, "arc_challenge": args.arc_challenge, "sciq": args.sciq}
    if args.wikitext:
        paths["wikitext"] = args.wikitext
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    knn_rng = random.Random(args.seed)
    # Independent VMA stream: changing the KNN sample size or holdout mode
    # must not silently change the 200 VMA prompts selected by the same seed.
    vma_rng = random.Random(args.seed + 1000)
    alpaca = load_texts(args.alpaca, "alpaca", args.min_chars, args.max_chars)
    if args.knn_reserved:
        if not args.knn_reserved.is_file():
            raise FileNotFoundError(args.knn_reserved)
        knn_pool = load_texts(args.knn_reserved, "alpaca_knn_holdout", args.min_chars, args.max_chars)
    else:
        knn_pool = list(alpaca)
    if len(knn_pool) < args.knn_samples:
        raise ValueError(f"KNN pool has {len(knn_pool)} records, fewer than requested {args.knn_samples}")
    knn_rng.shuffle(knn_pool)
    knn = knn_pool[:args.knn_samples]
    vma_pool: list[dict[str, Any]] = []
    for name in ("alpaca", "arc_easy", "arc_challenge", "sciq"):
        vma_pool.extend(load_texts(paths[name], name, args.min_chars, args.max_chars, shuffle_options=(name == "sciq"), seed=args.seed))
    if args.wikitext:
        vma_pool.extend(load_texts(args.wikitext, "wikitext", max(args.min_chars, 51), args.max_chars, cap=args.wikitext_cap))
    pool_before_cap = len(vma_pool)
    pool_cap_applied = pool_before_cap > args.pool_cap
    if pool_cap_applied:
        # Appendix C.2 caps the merged pool at 300,000. Shuffle first so the
        # cap does not silently drop the last source (WikiText) entirely.
        vma_rng.shuffle(vma_pool)
        vma_pool = vma_pool[:args.pool_cap]
    unique: dict[str, dict[str, Any]] = {}
    for row in vma_pool:
        unique.setdefault(row["text_sha256"], row)
    vma_pool = list(unique.values())
    if len(vma_pool) < args.vma_prompts:
        raise ValueError(f"VMA pool has {len(vma_pool)} records, fewer than requested {args.vma_prompts}")
    vma_rng.shuffle(vma_pool)
    vma = vma_pool[:args.vma_prompts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"knn": knn, "vma": vma}
    output_hashes = {name: write_jsonl(args.output_dir / f"{name}.jsonl", rows) for name, rows in outputs.items()}
    manifest = {
        "format": "osnip_paper_attack_manifest_v1",
        "status": "ready",
        "seed": args.seed,
        "rng_streams": {"knn": args.seed, "vma": args.seed + 1000, "sciq_option": "seed + source_index"},
        "filters": {"min_chars": args.min_chars, "max_chars": args.max_chars, "wikitext_min_chars": max(args.min_chars, 51),
                    "pool_cap": args.pool_cap, "pool_size_before_cap": pool_before_cap, "pool_cap_applied": pool_cap_applied},
        "knn": {"source": "alpaca_instruction_plus_input", "records": len(knn), "file_sha256": output_hashes["knn"], "reserved_holdout": args.knn_reserved is not None, "reserved_file": None if args.knn_reserved is None else str(args.knn_reserved.resolve()), "reserved_file_sha256": None if args.knn_reserved is None else file_sha256(args.knn_reserved)},
        "vma": {"sources": sorted(set(row["source"] for row in vma)), "records": len(vma), "file_sha256": output_hashes["vma"], "keys_per_prompt": 5},
        "source_files": {name: {"path": str(path.resolve()), "file_sha256": file_sha256(path)} for name, path in paths.items()},
        "source_counts": {split: {source: sum(row["source"] == source for row in rows) for source in sorted(set(row["source"] for row in rows))} for split, rows in outputs.items()},
        "sciq_option_policy": "deterministic per-record shuffle with seed + source_index for VMA pool",
        "stopword_policy": "whole-word Clean ASR via tokenizer offsets against the vendored NLTK English corpus; pieces inherit their source word",
    }
    (args.output_dir / "attack_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ready", "output_dir": str(args.output_dir.resolve()), "counts": {name: len(rows) for name, rows in outputs.items()}, "source_counts": manifest["source_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
