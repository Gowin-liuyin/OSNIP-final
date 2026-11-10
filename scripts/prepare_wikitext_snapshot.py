#!/usr/bin/env python3
"""Import an existing pinned WikiText-2 snapshot into standalone OSNIP storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    upstream = json.loads((args.source_dir / 'manifest.json').read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('choose an empty WikiText snapshot directory')
    files = {}
    for split in ('train', 'valid', 'test'):
        source = args.source_dir / f'{split}.jsonl'
        actual = sha256(source)
        if actual != upstream['splits'][split]['sha256']:
            raise ValueError(f'upstream split hash mismatch: {split}')
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows) != upstream['splits'][split]['documents']:
            raise ValueError(f'upstream count mismatch: {split}')
        files[split] = {'file': source.name, 'records': len(rows), 'sha256': actual}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for info in files.values():
        shutil.copy2(args.source_dir / info['file'], args.output_dir / info['file'])
    shutil.copy2(args.source_dir / 'manifest.json', args.output_dir / 'upstream_manifest.json')
    report = {
        'format': 'osnip_wikitext2_snapshot_v1', 'status': 'ready', 'dataset': 'WikiText-2',
        'upstream_repository': upstream['repository'], 'upstream_revision': upstream['revision'],
        'upstream_manifest_sha256': sha256(args.source_dir / 'manifest.json'), 'splits': files,
        'preprocessing': 'article JSONL from pinned pytorch/examples mirror; preserves <unk> and token spacing',
        'protocol_boundary': 'article-wise sliding windows, not a concatenated wikitext-2-raw-v1 benchmark',
        'notes': ['underlying Wikipedia attribution/license obligations apply', 'no AloePri code is needed to read this standalone snapshot'],
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
