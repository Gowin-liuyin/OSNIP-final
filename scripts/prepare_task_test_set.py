#!/usr/bin/env python3
"""Build the official test-split task set from fetched raw sources.

The paper's ARC-Easy row is the official test split, not a custom mixed
pool. evaluate_task_utility.py indexes raw records by (source, source_index);
this writer records only rows whose `_source_split` is `test`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import _iter_json_values, file_sha256


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arc-easy", required=True, type=Path)
    ap.add_argument("--arc-challenge", required=True, type=Path)
    ap.add_argument("--sciq", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = {"arc_easy": args.arc_easy, "arc_challenge": args.arc_challenge, "sciq": args.sciq}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {}
    for source, path in paths.items():
        count = 0
        for index, record in enumerate(_iter_json_values(path)):
            if record.get("_source_split") != "test":
                continue
            rows.append({"source": source, "source_index": index})
            count += 1
        counts[source] = count
        if count == 0:
            raise ValueError(f"{source} has no rows tagged _source_split=test")
    output = args.output_dir / "task_test.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "format": "osnip_task_test_manifest_v1",
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "records": len(rows),
        "source_counts": counts,
        "sources": {name: {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
                    for name, path in paths.items()},
        "split": "test",
    }
    (args.output_dir / "task_test.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "source_counts": counts, "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
