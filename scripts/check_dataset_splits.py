#!/usr/bin/env python3
"""Check JSONL split disjointness and write a reproducible dataset manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import read_text_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--valid", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    paths = {name: path.resolve() for name, path in (("train", args.train), ("valid", args.valid), ("test", args.test))}
    rows = {name: read_text_jsonl(path) for name, path in paths.items()}
    hashes = {name: [text_hash(text) for text in texts] for name, texts in rows.items()}
    overlaps = {}
    names = tuple(paths)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(set(hashes[left]) & set(hashes[right]))
            if shared:
                overlaps[f"{left}__{right}"] = shared
    report = {
        "format": "osnip_dataset_split_manifest_v1",
        "status": "overlap_detected" if overlaps else "disjoint",
        "splits": {
            name: {"path": str(paths[name]), "records": len(rows[name]),
                   "file_sha256": file_sha256(paths[name]), "text_sha256": hashes[name]}
            for name in names
        },
        "overlaps": overlaps,
        "normalization": "collapse runs of whitespace before text hashing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
                      "records": {name: len(rows[name]) for name in names},
                      "overlap_pairs": list(overlaps)}, ensure_ascii=False, indent=2))
    if overlaps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
