#!/usr/bin/env python3
"""Build a separate 65k pool with question-disjoint science holdouts.

Preserves the legacy snapshot and ongoing experiments. Fixed attack sets and
reserved SciQ confirmation are excluded. Fresh training from initialization is
required; this construction does not make old exposed tests fresh again.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from osnip_repro.data import _iter_json_values, file_sha256, text_sha256
from prepare_osnip_datasets import load_source, write_jsonl
from prepare_confirmatory_holdout import question_hash


def enrich_sources(raw_paths, *, exclude_official_test=False):
    pools = {}
    for name, path in raw_paths.items():
        raw = list(_iter_json_values(path))
        rows = load_source(path, name, 1, exclude_official_test=exclude_official_test)
        for row in rows:
            if name != 'alpaca':
                row['question_sha256'] = question_hash(raw[row['source_index']])
        pools[name] = rows
    return pools


def excluded_identities(rows, pools):
    by_index = {(row['source'], row['source_index']): row for source in pools.values() for row in source}
    indices, questions, texts, attack_texts = set(), set(), set(), set()
    for row in rows:
        source = row.get('source')
        if source == 'alpaca_knn_holdout':
            source = 'alpaca'
        identity = (source, row.get('source_index'))
        if source in pools and identity not in by_index:
            raise ValueError(f'invalid exclusion source identity: {identity}')
        original = by_index.get(identity)
        if original is not None:
            indices.add(identity)
            if original.get('question_sha256'):
                questions.add(original['question_sha256'])
            texts.add(original['text_sha256'])
            attack_texts.add(original['attack_text_sha256'])
        if isinstance(row.get('text'), str):
            texts.add(text_sha256(row['text']))
            attack_texts.add(text_sha256(row['text']))
    return indices, questions, texts, attack_texts


def construct(pools, *, excluded, legacy_valid, legacy_test, train_total, valid_count, test_count, seed):
    if min(train_total, valid_count, test_count) < 1:
        raise ValueError('positive split sizes required')
    blocked_ids, blocked_q, blocked_text, blocked_attack = excluded_identities(excluded, pools)
    seen_text, seen_q, seen_alpaca_attack = set(blocked_text), set(blocked_q), set(blocked_attack)
    available = {name: [] for name in pools}
    dropped = Counter()
    for name, rows in pools.items():
        for row in rows:
            q = row.get('question_sha256')
            if (name, row['source_index']) in blocked_ids:
                dropped['fixed_exclusion_index'] += 1
            elif row['text_sha256'] in seen_text:
                dropped['duplicate_or_excluded_text'] += 1
            elif q and q in seen_q:
                dropped['duplicate_or_excluded_question'] += 1
            elif name == 'alpaca' and row['attack_text_sha256'] in seen_alpaca_attack:
                dropped['duplicate_or_excluded_alpaca_instruction_input'] += 1
            else:
                available[name].append(row)
                seen_text.add(row['text_sha256'])
                if q:
                    seen_q.add(q)
                if name == 'alpaca':
                    seen_alpaca_attack.add(row['attack_text_sha256'])
    science = ('arc_easy', 'arc_challenge', 'sciq')
    remaining = {name: {row['question_sha256']: row for row in available[name]} for name in science}
    # Prefer original held-out questions even if their canonical source changed.
    question_lookup = {row['question_sha256']: row for name in science for row in available[name]}
    raw_lookup = {(row['source'],row['source_index']): row for rows in pools.values() for row in rows}
    legacy_test_questions = {raw_lookup[(row['source'],row['source_index'])]['question_sha256'] for row in legacy_test}
    legacy_valid_questions = {raw_lookup[(row['source'],row['source_index'])]['question_sha256'] for row in legacy_valid}
    used = set()
    rng = random.Random(seed)
    def holdout(legacy, count, forbidden):
        chosen = []
        for row in legacy:
            q = raw_lookup[(row['source'],row['source_index'])]['question_sha256']
            candidate = question_lookup.get(q)
            if candidate is not None and q not in used and q not in forbidden and len(chosen) < count:
                chosen.append(candidate)
                used.add(q)
        weights = {'arc_easy': .4, 'arc_challenge': .2, 'sciq': .4}
        refill = {name: [row for row in available[name] if row['question_sha256'] not in used and row['question_sha256'] not in forbidden] for name in science}
        for rows in refill.values():
            rng.shuffle(rows)
        while len(chosen) < count:
            candidates = [name for name in science if refill[name]]
            if not candidates:
                raise ValueError('not enough distinct questions for holdouts')
            counts = Counter(row['source'] for row in chosen)
            name = max(candidates, key=lambda name: (weights[name]*count-counts[name], -science.index(name)))
            candidate = refill[name].pop()
            q = candidate['question_sha256']
            if q in used:
                continue
            chosen.append(candidate)
            used.add(q)
        return chosen
    valid = holdout(legacy_valid, valid_count, legacy_test_questions - legacy_valid_questions)
    test = holdout(legacy_test, test_count, set())
    for name in science:
        remaining[name] = {q: row for q, row in remaining[name].items() if q not in used}
    general = list(available['alpaca'])
    rng.shuffle(general)
    general = general[:min(round(train_total*.8),len(general))]
    science_needed = train_total-len(general)
    science_rows = {name: list(remaining[name].values()) for name in science}
    for rows in science_rows.values():
        rng.shuffle(rows)
    chosen_science = []
    counts = Counter()
    while len(chosen_science)<science_needed:
        candidates = [name for name in science if science_rows[name]]
        if not candidates:
            raise ValueError(f'insufficient unique science questions: need {science_needed}, have {len(chosen_science)}')
        name = min(candidates,key=lambda name:(counts[name],science.index(name)))
        chosen_science.append(science_rows[name].pop())
        counts[name]+=1
    train = general+chosen_science
    rng.shuffle(train)
    splits = {'train':train,'valid':valid,'test':test}
    qsets = {name:{row['question_sha256'] for row in rows if 'question_sha256' in row} for name,rows in splits.items()}
    textsets = {name:{row['text_sha256'] for row in rows} for name,rows in splits.items()}
    for name,rows in splits.items():
        assert len(textsets[name])==len(rows)
        assert not (qsets[name]&blocked_q)
        assert not (textsets[name]&blocked_text)
        assert len(qsets[name])==sum('question_sha256' in row for row in rows)
    for left,right in (('train','valid'),('train','test'),('valid','test')):
        assert not (qsets[left]&qsets[right])
        assert not (textsets[left]&textsets[right])
    return splits, {'dropped':dict(dropped),'general_fraction_actual':len(general)/len(train),
                    'general_records':len(general),'science_records':len(chosen_science),
                    'overlaps':{},'question_overlaps':{}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-manifest',type=Path,default=ROOT/'data/osnip_sources/raw_sources_manifest.json')
    p.add_argument('--legacy-split-dir',type=Path,default=ROOT/'data/osnip_65k')
    p.add_argument('--attack-dir',type=Path,default=ROOT/'data/paper_attack_sets')
    p.add_argument('--confirmatory-manifest',type=Path,default=ROOT/'data/confirmatory_20261019/manifest.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--exclude-official-test',action='store_true',help='exclude ARC/SciQ official test rows from training')
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('choose a new dataset directory')
    raw_manifest=json.loads(args.raw_manifest.read_text())
    paths={name:Path(raw_manifest['sources'][name]['path']) for name in ('alpaca','arc_easy','arc_challenge','sciq')}
    for name,path in paths.items():
        if file_sha256(path)!=raw_manifest['sources'][name]['file_sha256']:
            raise ValueError('source bytes changed: '+name)
    confirmation=json.loads(args.confirmatory_manifest.read_text())
    confirm=confirmation['sciq_confirmatory']
    if file_sha256(confirm['path'])!=confirm['sha256']:
        raise ValueError('confirmatory bytes changed')
    exclusion_paths={'knn_reserved':args.legacy_split_dir/'knn_reserved.jsonl',
                     'knn':args.attack_dir/'knn.jsonl','vma':args.attack_dir/'vma.jsonl',
                     'sciq_confirmatory':Path(confirm['path'])}
    excluded=[row for path in exclusion_paths.values() for row in _iter_json_values(path)]
    valid=list(_iter_json_values(args.legacy_split_dir/'valid.jsonl'))
    test=list(_iter_json_values(args.legacy_split_dir/'test.jsonl'))
    pools=enrich_sources(paths, exclude_official_test=args.exclude_official_test)
    splits,construction=construct(pools,excluded=excluded,legacy_valid=valid,legacy_test=test,
                                train_total=65000,valid_count=1000,test_count=1000,seed=args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output={}
    for name,rows in splits.items():
        path=args.output_dir/f'{name}.jsonl'
        digest=write_jsonl(path,rows)
        output[name]={'path':str(path.resolve()),'records':len(rows),'file_sha256':digest,
                      'source_counts':dict(Counter(row['source'] for row in rows))}
    manifest={'format':'osnip_question_disjoint_dataset_manifest_v1','status':'ready_for_fresh_training_only',
              'security_accepted':False,'seed':args.seed,'splits':output,'construction':construction,
              'sources':{name:raw_manifest['sources'][name] for name in paths},
              'exclusions':{name:{'path':str(path.resolve()),'file_sha256':file_sha256(path)} for name,path in exclusion_paths.items()},
              'legacy_splits':{name:{'path':str((args.legacy_split_dir/f'{name}.jsonl').resolve()),
                                    'sha256':file_sha256(args.legacy_split_dir/f'{name}.jsonl')} for name in ('train','valid','test')},
              'question_normalization':'casefold then collapse whitespace then SHA-256',
              'script_sha256':file_sha256(Path(__file__)),
              'limitations':['New data require training from initialization; old checkpoint optimizer/cursor cannot resume into this pool.',
                             'Historical holdout questions have been exposed; this construction does not make them blind again.',
                             'Whole normalized question/text matching does not exclude paraphrases or pretraining contamination.',
                             'Actual general/science ratio is reported and differs from paper 80/20 due fixed independent reserves.',
                             'Old pinned reproduction config is not valid for these new dataset hashes.']}
    (args.output_dir/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'splits':output,'construction':construction},indent=2))


if __name__=='__main__':
    main()
