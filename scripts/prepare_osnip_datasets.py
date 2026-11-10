#!/usr/bin/env python3
"""Build disjoint, hashed OSNIP train/validation/test JSONL manifests.

The script intentionally accepts local JSON/JSONL files. Downloading is kept
outside the reproduction step so source revisions and file hashes are explicit.
Each output record retains its source and source index for auditability.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import _iter_json_values, _text_from_record, file_sha256, normalize_text, text_sha256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpaca", required=True, type=Path)
    ap.add_argument("--arc-easy", required=True, type=Path)
    ap.add_argument("--arc-challenge", required=True, type=Path)
    ap.add_argument("--sciq", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--train-total", type=int, default=65000)
    ap.add_argument("--general-fraction", type=float, default=0.8)
    ap.add_argument("--valid-count", type=int, default=1000)
    ap.add_argument("--test-count", type=int, default=1000)
    ap.add_argument("--knn-reserve", type=int, default=1000, help="reserve Alpaca instruction+input records for the independent KNN attack set")
    ap.add_argument("--knn-min-chars", type=int, default=20, help="minimum normalized character length for KNN-reserved Alpaca text")
    ap.add_argument("--knn-max-chars", type=int, default=512, help="maximum normalized character length for KNN-reserved Alpaca text")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--revision", action="append", default=[], metavar="SOURCE=REVISION")
    ap.add_argument("--allow-incomplete", action="store_true", help="permit fewer than requested records for smoke tests")
    ap.add_argument("--allow-unpinned", action="store_true", help="allow missing source revisions for local smoke tests")
    ap.add_argument("--paper-literal", action="store_true",
                    help="PMLR Table 9/C.1 frame: 65,151 rows, 80%% Alpaca target, and no reserved Alpaca holdout "
                         "(the KNN set is then sampled from the full Alpaca pool, as in Appendix C.1)")
    ap.add_argument("--sciq-answer-cue", action="store_true",
                    help="shuffle SciQ options and append an explicit Answer cue; the legacy v2 pin keeps this off")
    ap.add_argument("--exclude-official-test", action="store_true",
                    help="exclude raw ARC/SciQ _source_split=test rows from training; --paper-literal implies this")
    return ap.parse_args()


def source_text(record: dict[str, Any], source: str, index: int = 0, *, sciq_answer_cue: bool = False,
                sciq_seed: int = 42) -> str:
    if source.startswith("arc"):
        question = str(record.get("question", "")).strip()
        choices = record.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        values = choices.get("text", []) if isinstance(choices, dict) else []
        options = "\n".join(f"{label}: {value}" for label, value in zip(labels, values))
        answer = str(record.get("answerKey", record.get("answer", ""))).strip()
        pieces = [piece for piece in (question, options, f"Answer: {answer}" if answer else "") if piece]
        if pieces:
            return "\n".join(pieces)
    if source == "sciq":
        pieces = [str(record.get("question", "")).strip()]
        support = record.get("support", record.get("context", ""))
        if support:
            pieces.append(f"Context: {str(support).strip()}")
        options = [record.get("correct_answer"), record.get("distractor1"), record.get("distractor2"), record.get("distractor3")]
        options = [str(value).strip() for value in options if value is not None and str(value).strip()]
        if options:
            correct = str(record.get("correct_answer", "")).strip()
            if sciq_answer_cue and not correct:
                # Never fabricate a gold answer from the first distractor.
                pieces.append("Options: " + " | ".join(options))
            elif sciq_answer_cue:
                order = list(range(len(options)))
                random.Random(sciq_seed + index).shuffle(order)
                # Paper-literal line: shuffled options and an explicit Answer cue.
                # The legacy v2 line kept the correct answer first and no cue;
                # that pin is preserved by leaving the flag off.
                pieces.append("Options: " + " | ".join(options[i] for i in order))
                pieces.append(f"Answer: {correct}")
            else:
                pieces.append("Options: " + " | ".join(options))
        text = "\n".join(piece for piece in pieces if piece)
        if text:
            return text
    return _text_from_record(record)


def load_source(path: Path, source: str, min_chars: int, *, sciq_answer_cue: bool = False,
                sciq_seed: int = 42, exclude_official_test: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(_iter_json_values(path)):
        if (exclude_official_test and source in {"arc_easy", "arc_challenge", "sciq"}
                and record.get("_source_split") == "test"):
            continue
        text = source_text(record, source, index, sciq_answer_cue=sciq_answer_cue, sciq_seed=sciq_seed)
        if len(normalize_text(text)) < min_chars:
            continue
        attack_text = ("\n\n".join(piece for piece in (str(record.get("instruction", "")).strip(), str(record.get("input", "")).strip()) if piece) if source == "alpaca" else text)
        rows.append({
            "text": text,
            "attack_text": attack_text,
            "source": source,
            "source_index": index,
            "text_sha256": text_sha256(text),
            "attack_text_sha256": text_sha256(attack_text),
        })
    if not rows:
        raise ValueError(f"no usable records in {path}")
    return rows


def parse_revisions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"revision must be SOURCE=REVISION: {value}")
        source, revision = value.split("=", 1)
        if not source or not revision:
            raise ValueError(f"revision must be SOURCE=REVISION: {value}")
        result[source] = revision
    return result


def take_records(pool: list[dict[str, Any]], count: int, rng: random.Random, *, allow_incomplete: bool) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("record counts must be non-negative")
    if len(pool) < count and not allow_incomplete:
        raise ValueError(f"requested {count} records but only {len(pool)} are available for {pool[0]['source'] if pool else 'source'}")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:count]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def main() -> None:
    args = parse_args()
    if args.paper_literal:
        # Table 9's 65,151 rows and C.1's full-Alpaca KNN sample. The public
        # Alpaca pool (52,002 after dedup) is 119 rows short of 80%, so the
        # actual fraction is recorded in the manifest (about 79.82/20.18);
        # do not duplicate rows to manufacture an exact 80/20.
        args.train_total = 65151
        args.knn_reserve = 0
    if args.train_total < 1 or args.valid_count < 0 or args.test_count < 0 or args.knn_reserve < 0:
        raise ValueError("train-total must be positive and split counts/reserved KNN count cannot be negative")
    if args.knn_min_chars < 0 or args.knn_max_chars < args.knn_min_chars:
        raise ValueError("invalid KNN character limits")
    if not 0 < args.general_fraction < 1:
        raise ValueError("general-fraction must be between 0 and 1")
    paths = {
        "alpaca": args.alpaca,
        "arc_easy": args.arc_easy,
        "arc_challenge": args.arc_challenge,
        "sciq": args.sciq,
    }
    revisions = parse_revisions(args.revision)
    missing_revisions = sorted(set(paths) - set(revisions))
    if missing_revisions and not args.allow_unpinned:
        raise ValueError("missing pinned revisions for: " + ", ".join(missing_revisions) + "; use --allow-unpinned only for smoke tests")
    rng = random.Random(args.seed)

    loaded: dict[str, list[dict[str, Any]]] = {
        source: load_source(path, source, args.min_chars,
                            sciq_answer_cue=(args.paper_literal or args.sciq_answer_cue),
                            sciq_seed=args.seed,
                            exclude_official_test=(args.paper_literal or args.exclude_official_test))
        for source, path in paths.items()
    }
    # Deduplicate by normalized text globally. Keeping the first source makes
    # cross-source contamination visible in the manifest instead of silently
    # assigning the same example to multiple pools.
    seen: set[str] = set()
    duplicate_counts: Counter[str] = Counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    for source, rows in loaded.items():
        unique: list[dict[str, Any]] = []
        for row in rows:
            key = row["text_sha256"]
            if key in seen:
                duplicate_counts[source] += 1
                continue
            seen.add(key)
            unique.append(row)
        pools[source] = unique

    reserve_pool = [
        row for row in pools["alpaca"]
        if args.knn_min_chars <= len(normalize_text(row["attack_text"])) <= args.knn_max_chars
    ]
    if len(reserve_pool) < args.knn_reserve and not args.allow_incomplete:
        raise ValueError(
            f"requested {args.knn_reserve} Alpaca KNN holdout records after "
            f"{args.knn_min_chars}-{args.knn_max_chars} character filtering, "
            f"but only {len(reserve_pool)} are available"
        )
    reserve_rng = random.Random(args.seed + 7001)
    reserve_rng.shuffle(reserve_pool)
    knn_reserved = reserve_pool[:min(args.knn_reserve, len(reserve_pool))]
    reserved_hashes = {row["text_sha256"] for row in knn_reserved}
    pools["alpaca"] = [row for row in pools["alpaca"] if row["text_sha256"] not in reserved_hashes]

    general_target = round(args.train_total * args.general_fraction)
    science_target = args.train_total - general_target
    science_names = ("arc_easy", "arc_challenge", "sciq")
    requested_science_targets = {
        name: science_target // len(science_names) + (1 if i < science_target % len(science_names) else 0)
        for i, name in enumerate(science_names)
    }
    # Reserve mixed holdouts before training selection. Alpaca has only two
    # spare rows beyond the 52k training target, so holdouts are drawn from
    # the science sources with a fixed 40/20/40 Easy/Challenge/SciQ policy.
    def allocate_holdout(total: int, capacities: dict[str, int]) -> dict[str, int]:
        weights = {"arc_easy": 0.4, "arc_challenge": 0.2, "sciq": 0.4}
        targets = {name: min(int(total * weights[name]), capacities[name]) for name in science_names}
        remaining = total - sum(targets.values())
        while remaining > 0:
            candidates = [name for name in science_names if targets[name] < capacities[name]]
            if not candidates:
                raise ValueError("not enough science records for requested validation/test holdout")
            source = max(candidates, key=lambda name: (weights[name] * total - targets[name], -science_names.index(name)))
            targets[source] += 1
            remaining -= 1
        return targets

    available_for_holdout = {name: len(pools[name]) for name in science_names}
    valid_targets = allocate_holdout(args.valid_count, available_for_holdout)
    for name, count in valid_targets.items():
        valid_targets[name] = min(count, len(pools[name]))
        available_for_holdout[name] -= valid_targets[name]
    test_targets = allocate_holdout(args.test_count, available_for_holdout)
    valid: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    remaining_pools: dict[str, list[dict[str, Any]]] = {name: list(rows) for name, rows in pools.items()}
    for source in science_names:
        rng.shuffle(remaining_pools[source])
        valid.extend(remaining_pools[source][:valid_targets[source]])
        remaining_pools[source] = remaining_pools[source][valid_targets[source]:]
        test.extend(remaining_pools[source][:test_targets[source]])
        remaining_pools[source] = remaining_pools[source][test_targets[source]:]

    selected: list[dict[str, Any]] = []
    alpaca_train_target = min(general_target, len(remaining_pools["alpaca"]))
    selected.extend(take_records(remaining_pools["alpaca"], alpaca_train_target, rng, allow_incomplete=args.allow_incomplete))
    science_pools = {name: remaining_pools[name] for name in science_names}
    science_target_with_reserve = args.train_total - alpaca_train_target
    science_targets = {name: min(requested_science_targets[name], len(science_pools[name])) for name in science_names}
    remaining_science = science_target_with_reserve - sum(science_targets.values())
    while remaining_science > 0:
        candidates = [name for name in science_names if science_targets[name] < len(science_pools[name])]
        if not candidates:
            if args.allow_incomplete:
                break
            raise ValueError(f"science sources contain only {sum(len(science_pools[name]) for name in science_names)} unique records after holdouts, fewer than requested {science_target}")
        source = min(candidates, key=lambda name: (science_targets[name], science_names.index(name)))
        science_targets[source] += 1
        remaining_science -= 1
    for source in science_names:
        selected.extend(take_records(science_pools[source], science_targets[source], rng, allow_incomplete=args.allow_incomplete))
    if len(selected) < args.train_total and not args.allow_incomplete:
        raise RuntimeError(f"only selected {len(selected)} of requested {args.train_total} training records")

    train = list(selected)
    rng.shuffle(train)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reserve_rows = [{"text": row["attack_text"], "source": "alpaca_knn_holdout", "source_index": row["source_index"], "text_sha256": row["attack_text_sha256"]} for row in knn_reserved]
    reserve_path = args.output_dir / "knn_reserved.jsonl"
    reserve_hash = write_jsonl(reserve_path, reserve_rows)
    output_paths = {name: args.output_dir / f"{name}.jsonl" for name in ("train", "valid", "test")}
    output_hashes = {name: write_jsonl(output_paths[name], rows) for name, rows in (("train", train), ("valid", valid), ("test", test))}
    split_hashes = {name: [row["text_sha256"] for row in rows] for name, rows in (("train", train), ("valid", valid), ("test", test))}
    overlaps = {
        f"{left}__{right}": sorted(set(split_hashes[left]) & set(split_hashes[right]))
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
        if set(split_hashes[left]) & set(split_hashes[right])
    }
    manifest = {
        "format": "osnip_dataset_manifest_v1",
        "status": "ready" if not overlaps and len(train) == args.train_total else "incomplete",
        "construction": {
            "train_total_requested": args.train_total,
            "train_total_actual": len(train),
            "general_fraction": args.general_fraction,
            "general_target": general_target,
            "general_train_actual": alpaca_train_target,
            "general_fraction_actual": alpaca_train_target / max(len(train), 1),
            "knn_reserve_requested": args.knn_reserve,
            "knn_reserve_actual": len(knn_reserved),
            "knn_reserve_file_sha256": reserve_hash,
            "science_target_requested": science_target,
            "science_target_with_reserve": science_target_with_reserve,
            "science_targets_requested": requested_science_targets,
            "science_targets_actual": science_targets,
            "valid_count_requested": args.valid_count,
            "test_count_requested": args.test_count,
            "holdout_policy": {"sources": ["arc_easy", "arc_challenge", "sciq"], "fractions": {"arc_easy": 0.4, "arc_challenge": 0.2, "sciq": 0.4}, "valid_actual": valid_targets, "test_actual": test_targets},
            "min_chars": args.min_chars,
            "knn_filter": {"min_chars": args.knn_min_chars, "max_chars": args.knn_max_chars, "field": "attack_text"},
            "seed": args.seed,
            "deduplicate_normalized_text": True,
            "paper_literal": args.paper_literal,
            "sciq_answer_cue": bool(args.paper_literal or args.sciq_answer_cue),
            "exclude_official_test": bool(args.paper_literal or args.exclude_official_test),
        },
        "sources": {
            source: {
                "path": str(path.resolve()),
                "file_sha256": file_sha256(path),
                "revision": revisions.get(source, "UNPINNED_LOCAL_SOURCE"),
                "records_read": len(loaded[source]),
                "records_after_global_dedup": len(pools[source]),
                "duplicate_records_dropped": duplicate_counts[source],
            }
            for source, path in paths.items()
        },
        "splits": {
            name: {"path": str(path.resolve()), "records": len(rows), "file_sha256": output_hashes[name], "text_sha256": split_hashes[name]}
            for name, (path, rows) in zip(output_paths, ((output_paths["train"], train), (output_paths["valid"], valid), (output_paths["test"], test)))
        },
        "overlaps": overlaps,
        "source_counts": {name: dict(Counter(row["source"] for row in rows)) for name, rows in (("train", train), ("valid", valid), ("test", test))},
        "knn_holdout": {"path": str(reserve_path.resolve()), "records": len(reserve_rows), "file_sha256": reserve_hash, "attack_text_hashes": [row["text_sha256"] for row in reserve_rows]},
        "allow_incomplete": args.allow_incomplete,
        "allow_unpinned": args.allow_unpinned,
    }
    manifest_path = args.output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir.resolve()), "counts": {name: len(rows) for name, rows in (("train", train), ("valid", valid), ("test", test))}, "manifest": str(manifest_path.resolve())}, ensure_ascii=False, indent=2))
    if overlaps or (len(train) != args.train_total and not args.allow_incomplete):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
