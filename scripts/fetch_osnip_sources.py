#!/usr/bin/env python3
"""Fetch the public OSNIP training sources and record immutable revisions.

Alpaca is copied from the checked-in reference snapshot by default. ARC and
SciQ are loaded from Hugging Face at explicit commit revisions. The output
files are raw source records; ``prepare_osnip_datasets.py`` performs the
normalized-text conversion and split construction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.data import file_sha256

ALPACA_REFERENCE = ROOT / "references/online/stanford_alpaca-alpaca_data.json"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
ARC_REPO = "allenai/ai2_arc"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
SCIQ_REPO = "allenai/sciq"
SCIQ_REVISION = "2c94ad3e1aafab77146f384e23536f97a4849815"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--alpaca", type=Path, default=ALPACA_REFERENCE)
    ap.add_argument("--download-alpaca", action="store_true", help="download Alpaca from its pinned public URL instead of the local reference snapshot")
    ap.add_argument("--arc-revision", default=ARC_REVISION)
    ap.add_argument("--sciq-revision", default=SCIQ_REVISION)
    ap.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    ap.add_argument("--wikitext-cap", type=int, default=150000)
    ap.add_argument("--skip-wikitext", action="store_true",
                    help="skip the WikiText-103 snapshot; it feeds PPL evaluation only and "
                         "prepare_osnip_datasets.py never reads it, so a slow mirror does not "
                         "need to block the training corpus")
    return ap.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def load_hf_source(repo: str, config: str | None, revision: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset
    dataset = load_dataset(repo, config, revision=revision) if config else load_dataset(repo, revision=revision)
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split, table in dataset.items():
        split_counts[split] = len(table)
        for row in table:
            item = dict(row)
            item["_source_split"] = split
            rows.append(item)
    return rows, split_counts


def main() -> None:
    args = parse_args()
    if not args.alpaca.is_file():
        raise FileNotFoundError(args.alpaca)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    alpaca_out = raw_dir / "alpaca.json"
    if args.download_alpaca:
        from urllib.request import Request, urlopen
        request = Request(ALPACA_URL, headers={"User-Agent": "OSNIP-reproduction"})
        with urlopen(request, timeout=60) as response:
            alpaca_out.write_bytes(response.read())
        alpaca_revision = "github-main-blob-3779ddbc040543ab1834ef216c983d6fcc06cc9a"
        alpaca_origin = ALPACA_URL
    else:
        shutil.copy2(args.alpaca, alpaca_out)
        alpaca_revision = "local-reference-" + file_sha256(args.alpaca)
        alpaca_origin = str(args.alpaca.resolve())
    alpaca_rows = json.loads(alpaca_out.read_text(encoding="utf-8"))
    if not isinstance(alpaca_rows, list):
        raise ValueError("Alpaca source must be a JSON list")

    arc_easy_rows, arc_easy_splits = load_hf_source(ARC_REPO, "ARC-Easy", args.arc_revision)
    arc_challenge_rows, arc_challenge_splits = load_hf_source(ARC_REPO, "ARC-Challenge", args.arc_revision)
    sciq_rows, sciq_splits = load_hf_source(SCIQ_REPO, None, args.sciq_revision)
    if args.skip_wikitext:
        wiki_rows: list[dict[str, Any]] = []
        wiki_splits: dict[str, int] = {}
    else:
        from datasets import load_dataset
        wiki_table = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, revision=args.wikitext_revision, split="train")
        wiki_rows = []
        for row in wiki_table:
            text = str(row.get("text", "")).strip()
            if len(" ".join(text.split())) > 50:
                wiki_rows.append({"text": text})
                if len(wiki_rows) >= args.wikitext_cap:
                    break
        wiki_splits = {"train": len(wiki_table), "filtered_train": len(wiki_rows)}
    sources = {
        "alpaca": (alpaca_rows, {"all": len(alpaca_rows)}, alpaca_revision, alpaca_origin, alpaca_out),
        "arc_easy": (arc_easy_rows, arc_easy_splits, args.arc_revision, f"hf://datasets/{ARC_REPO}", raw_dir / "arc_easy.jsonl"),
        "arc_challenge": (arc_challenge_rows, arc_challenge_splits, args.arc_revision, f"hf://datasets/{ARC_REPO}", raw_dir / "arc_challenge.jsonl"),
        "sciq": (sciq_rows, sciq_splits, args.sciq_revision, f"hf://datasets/{SCIQ_REPO}", raw_dir / "sciq.jsonl"),
        "wikitext": (wiki_rows, wiki_splits, args.wikitext_revision, f"hf://datasets/{WIKITEXT_REPO}", raw_dir / "wikitext.jsonl"),
    }
    if args.skip_wikitext:
        sources.pop("wikitext")
    manifest_sources: dict[str, Any] = {}
    for source, (rows, split_counts, revision, origin, path) in sources.items():
        if source == "alpaca":
            output_hash = file_sha256(path)
        else:
            output_hash = write_rows(path, rows)
        manifest_sources[source] = {
            "origin": origin,
            "revision": revision,
            "records": len(rows),
            "split_counts": split_counts,
            "path": str(path.resolve()),
            "file_sha256": output_hash,
        }
    manifest = {
        "format": "osnip_raw_sources_manifest_v1",
        "status": "ready",
        "sources": manifest_sources,
        "next_step": "run prepare_osnip_datasets.py with the four raw files",
    }
    manifest_path = args.output_dir / "raw_sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ready", "output_dir": str(args.output_dir.resolve()), "manifest": str(manifest_path.resolve()), "counts": {name: item["records"] for name, item in manifest_sources.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
