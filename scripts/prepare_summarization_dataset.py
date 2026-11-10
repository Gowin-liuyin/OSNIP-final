#!/usr/bin/env python3
"""Normalize a local CNN/DailyMail-style snapshot for OSNIP evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import _iter_json_values, file_sha256, normalize_text, text_sha256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--source-id", default="local")
    ap.add_argument("--source-revision", default="unlocked")
    ap.add_argument("--source-split", default="unknown",
                    help="use 'test' for the paper Table 3 protocol; the aggregate requires source.split=test")
    ap.add_argument("--limit", type=int)
    return ap.parse_args()


def fields(record: dict[str, Any]) -> tuple[str, str]:
    article = record.get("article", record.get("text"))
    summary = record.get("highlights", record.get("summary"))
    if not isinstance(article, str) or not normalize_text(article):
        raise ValueError("record has no non-empty article/text field")
    if not isinstance(summary, str) or not normalize_text(summary):
        raise ValueError("record has no non-empty highlights/summary field")
    return article.strip(), summary.strip()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    input_records = 0
    dropped_duplicates = 0
    for record in _iter_json_values(args.input):
        input_records += 1
        article, summary = fields(record)
        identity = text_sha256(article + "\n\n" + summary)
        if identity in seen:
            dropped_duplicates += 1
            continue
        seen.add(identity)
        rows.append({
            "article": article,
            "highlights": summary,
            "source_index": input_records - 1,
            "record_sha256": identity,
        })
        if args.limit is not None and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("no valid summarization records found")
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "format": "osnip_summarization_dataset_manifest_v1",
        "status": "ready",
        "source": {"id": args.source_id, "revision": args.source_revision, "split": args.source_split},
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "input_records_seen": input_records,
        "records_written": len(rows),
        "duplicates_dropped": dropped_duplicates,
        "limit": args.limit,
        "fields": {"article": "article", "reference": "highlights"},
        "normalized_text_identity": "sha256(collapse_whitespace(article + blank_line + highlights))",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ready", "records": len(rows), "output": str(args.output.resolve()), "manifest": str(manifest_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
