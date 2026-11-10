#!/usr/bin/env python3
"""Audit question leakage and reserve unused SciQ without mutating active data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from osnip_repro.data import _iter_json_values, file_sha256, text_sha256
from prepare_osnip_datasets import source_text


def question_hash(record):
    question = record.get('question')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('raw source record must have a nonempty question')
    return text_sha256(question.casefold())


def source_identity(row, raw):
    source = row.get('source')
    if source not in raw:
        return None
    index = row.get('source_index')
    if not isinstance(index, int) or not 0 <= index < len(raw[source]):
        raise ValueError(f'invalid raw source index for {source}')
    return source, index, question_hash(raw[source][index])


def audit_and_select(raw, splits, exclusions, *, count, seed):
    if count < 1:
        raise ValueError('positive holdout count required')
    identities = {name: [source_identity(row, raw) for row in rows] for name, rows in splits.items()}
    questions = {name: {item[2] for item in rows if item is not None} for name, rows in identities.items()}
    overlap = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            shared = questions[left] & questions[right]
            overlap[f'{left}__{right}'] = {
                'distinct_questions': len(shared), 'question_sha256': sorted(shared),
                'left_rows': [i for i, item in enumerate(identities[left]) if item is not None and item[2] in shared],
                'right_rows': [i for i, item in enumerate(identities[right]) if item is not None and item[2] in shared]}
    used_ids = set()
    used_questions = set()
    used_text = set()
    for rows in [*splits.values(), *exclusions.values()]:
        for row in rows:
            for field in ('text', 'attack_text'):
                if isinstance(row.get(field), str) and row[field].strip():
                    used_text.add(text_sha256(row[field]))
            item = source_identity(row, raw)
            if item is not None:
                source, index, qhash = item
                used_ids.add((source, index))
                used_questions.add(qhash)
                used_text.add(text_sha256(source_text(raw[source][index], source)))
    candidates = []
    seen_questions = set(used_questions)
    seen_text = set(used_text)
    rejected = Counter()
    for index, record in enumerate(raw['sciq']):
        if ('sciq', index) in used_ids:
            rejected['source_index_already_used'] += 1
            continue
        qhash = question_hash(record)
        text = source_text(record, 'sciq')
        identity = text_sha256(text)
        if qhash in seen_questions:
            rejected['question_already_used_or_duplicate'] += 1
            continue
        if identity in seen_text:
            rejected['text_already_used_or_duplicate'] += 1
            continue
        seen_questions.add(qhash)
        seen_text.add(identity)
        candidates.append({'source': 'sciq', 'source_index': index, 'text': text,
                           'text_sha256': identity, 'question_sha256': qhash,
                           'upstream_split': record.get('_source_split', 'unknown')})
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f'need {count} unused unique questions but only {len(candidates)} remain')
    reserved = candidates[:count]
    train_valid_questions = questions['train'] | questions['valid']
    clean_test = []
    dropped = []
    test_seen = set(train_valid_questions)
    for index, row in enumerate(splits['test']):
        item = identities['test'][index]
        if item is not None and item[2] in test_seen:
            dropped.append(index)
            continue
        if item is not None:
            test_seen.add(item[2])
        clean_test.append(row)
    return {'question_overlap': overlap, 'candidate_pool_count': len(candidates),
            'candidate_rejections': dict(rejected), 'reserved': reserved,
            'clean_test': clean_test, 'dropped_test_indices': dropped,
            'blocked_distinct_question_count': len(used_questions)}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows))
    return {'path': str(path.resolve()), 'records': len(rows), 'sha256': file_sha256(path),
            'source_counts': dict(Counter(row['source'] for row in rows))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-manifest', type=Path, default=ROOT/'data/osnip_sources/raw_sources_manifest.json')
    p.add_argument('--split-dir', type=Path, default=ROOT/'data/osnip_65k')
    p.add_argument('--knn-jsonl', type=Path, default=ROOT/'data/paper_attack_sets/knn.jsonl')
    p.add_argument('--vma-jsonl', type=Path, default=ROOT/'data/paper_attack_sets/vma.jsonl')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261019)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('choose a new output directory')
    raw_manifest = json.loads(args.raw_manifest.read_text())
    raw = {}
    source_evidence = {}
    for name in ('arc_easy', 'arc_challenge', 'sciq'):
        spec = raw_manifest['sources'][name]
        path = Path(spec['path'])
        if file_sha256(path) != spec['file_sha256']:
            raise ValueError(f'raw source hash mismatch: {name}')
        raw[name] = list(_iter_json_values(path))
        source_evidence[name] = spec
    paths = {name: args.split_dir/f'{name}.jsonl' for name in ('train', 'valid', 'test')}
    exclusion_paths = {'knn_reserved': args.split_dir/'knn_reserved.jsonl',
                       'knn': args.knn_jsonl, 'vma': args.vma_jsonl}
    splits = {name: list(_iter_json_values(path)) for name, path in paths.items()}
    exclusions = {name: list(_iter_json_values(path)) for name, path in exclusion_paths.items()}
    input_hashes = {name: {'path': str(path.resolve()), 'sha256': file_sha256(path)}
                    for name, path in {**paths, **exclusion_paths}.items()}
    result = audit_and_select(raw, splits, exclusions, count=args.count, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reserved = write_rows(args.output_dir/'sciq_confirmatory.jsonl', result.pop('reserved'))
    clean_test = write_rows(args.output_dir/'test_question_disjoint.jsonl', result.pop('clean_test'))
    manifest = {'format': 'osnip_question_disjoint_confirmatory_manifest_v1',
                'status': 'reserved_not_evaluated', 'seed': args.seed, 'security_accepted': False,
                'input_hashes': input_hashes, 'raw_sources': source_evidence,
                'raw_manifest_sha256': file_sha256(args.raw_manifest),
                'script_sha256': file_sha256(Path(__file__)),
                'question_normalization': 'casefold then collapse whitespace then SHA-256',
                'sciq_confirmatory': reserved, 'existing_test_question_disjoint': clean_test,
                **result,
                'usage_policy': 'Use fresh SciQ once after locking candidate/config/thresholds. It is additional confirmation, not the paper mixed benchmark.',
                'limitations': ['Question equality does not detect semantic paraphrases or pretraining contamination.',
                                'Question-disjoint old test subset has already been exposed through prior full-test evaluation.',
                                'SciQ confirmation has a different source distribution and does not replace mixed ARC/SciQ, KNN, VMA or other required utility tasks.',
                                'Old train/valid/test and reports are preserved; future retraining should group by question before splitting.']}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in ('status', 'sciq_confirmatory',
        'existing_test_question_disjoint', 'candidate_pool_count', 'dropped_test_indices')},indent=2))
    print(json.dumps({key: value['distinct_questions'] for key,value in result['question_overlap'].items()},indent=2))


if __name__ == '__main__':
    main()
