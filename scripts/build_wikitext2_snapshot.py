#!/usr/bin/env python3
"""Build a pinned WikiText-2 snapshot JSONL from the Hugging Face hub.

WikiText-2-raw-v1 stores each split as one line per paragraph; the test split
contains 62 article blocks introduced by `` = Title = `` headings. This builder
groups lines into articles (the unit the paper's sliding-window PPL and prior
runs used) and records file SHA-256 plus article counts in a manifest.

Usage (remote with mirror):
    HF_ENDPOINT=https://hf-mirror.com python scripts/build_wikitext2_snapshot.py \
        --output-dir data/wikitext2 --revision b08601e04326c79dfdd32d625aee71d232d685c3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_articles(lines: list[str]) -> list[str]:
    articles: list[str] = []
    current: list[str] = []
    for line in lines:
        is_heading = line.startswith(" = ") and line.rstrip().endswith(" =") and not line.startswith(" = = ")
        if is_heading and current:
            articles.append("\n".join(current))
            current = []
        if line.strip():
            current.append(line.rstrip("\n"))
    if current:
        articles.append("\n".join(current))
    return [article for article in articles if article.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    parser.add_argument("--splits", default="test")
    args = parser.parse_args()

    from datasets import load_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "osnip_wikitext2_snapshot_v1",
        "dataset": args.dataset,
        "config": args.config,
        "revision": args.revision,
        "splits": {},
    }
    dataset = load_dataset(args.dataset, args.config, revision=args.revision)
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        lines = dataset[split]["text"]
        articles = split_articles(list(lines))
        target = args.output_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8") as stream:
            for article in articles:
                stream.write(json.dumps({"text": article}, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "path": str(target),
            "file": target.name,
            "articles": len(articles),
            "records": len(articles),
            "source_lines": len(lines),
            "sha256": sha256(target),
        }
        print(f"{split}: {len(articles)} articles -> {target}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()