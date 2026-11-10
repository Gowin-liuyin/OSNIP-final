#!/usr/bin/env python3
"""Recompute identities from raw sources for the separate fresh-training pool."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from osnip_repro.data import _iter_json_values,file_sha256
from prepare_osnip_datasets import source_text


def normalized_hash(value, *, casefold=False):
    if casefold:
        value=value.casefold()
    return hashlib.sha256(' '.join(value.split()).encode()).hexdigest()


def verify_dataset(manifest, *, expected_counts=None):
    if manifest.get('format')!='osnip_question_disjoint_dataset_manifest_v1':
        raise ValueError('expected new question-disjoint manifest')
    sciq_answer_cue=bool((manifest.get('construction') or {}).get('sciq_answer_cue',False))
    raw={}
    for name,spec in manifest['sources'].items():
        if file_sha256(spec['path'])!=spec['file_sha256']:
            raise ValueError(f'raw source hash mismatch: {name}')
        raw[name]=list(_iter_json_values(Path(spec['path'])))
    def identity(row):
        source=row['source']
        if source=='alpaca_knn_holdout':
            source='alpaca'
        if source not in raw:
            return None
        index=row.get('source_index')
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(raw[source]):
            raise ValueError('invalid raw source index')
        original=raw[source][index]
        question=normalized_hash(original['question'],casefold=True) if source!='alpaca' else None
        text=normalized_hash(source_text(original,source,index,sciq_answer_cue=sciq_answer_cue))
        attack=normalized_hash('\n\n'.join(str(original.get(field,'')).strip() for field in ('instruction','input') if str(original.get(field,'')).strip())) if source=='alpaca' else text
        return (source,index),question,text,attack
    excluded_ids,excluded_q,excluded_text,excluded_attack=set(),set(),set(),set()
    for name,spec in manifest['exclusions'].items():
        if file_sha256(spec['path'])!=spec['file_sha256']:
            raise ValueError(f'exclusion hash mismatch: {name}')
        for row in _iter_json_values(Path(spec['path'])):
            actual=identity(row)
            if actual:
                index,q,text,attack=actual
                excluded_ids.add(index)
                if q:
                    excluded_q.add(q)
                excluded_text.add(text)
                excluded_attack.add(attack)
            if isinstance(row.get('text'),str):
                h=normalized_hash(row['text'])
                excluded_text.add(h)
                excluded_attack.add(h)
    expected_counts=expected_counts or {'train':65000,'valid':1000,'test':1000}
    if set(manifest['splits'])!=set(expected_counts):
        raise ValueError('must include train/valid/test')
    question_sets,text_sets,identity_sets,source_counts={},{},{},{}
    for name,spec in manifest['splits'].items():
        if file_sha256(spec['path'])!=spec['file_sha256']:
            raise ValueError(f'split hash mismatch: {name}')
        rows=list(_iter_json_values(Path(spec['path'])))
        if len(rows)!=expected_counts[name] or spec['records']!=len(rows):
            raise ValueError('wrong split record count')
        ids,questions,texts,alpaca_attack=set(),set(),set(),set()
        counts=Counter()
        for row in rows:
            actual=identity(row)
            if actual is None:
                raise ValueError('unknown source in split')
            index,q,text,attack=actual
            original=raw[index[0]][index[1]]
            if row.get('text')!=source_text(original,index[0],index[1],sciq_answer_cue=sciq_answer_cue) or row.get('text_sha256')!=text or row.get('attack_text_sha256')!=attack:
                raise ValueError('record content/hash disagrees with raw source')
            if q and row.get('question_sha256')!=q:
                raise ValueError('question hash disagrees with raw source')
            if index in ids or text in texts or (q and q in questions):
                raise ValueError('duplicate source/text/question within split')
            if index in excluded_ids or text in excluded_text or (q and q in excluded_q):
                raise ValueError('record overlaps fixed exclusions')
            if index[0]=='alpaca':
                if attack in alpaca_attack or attack in excluded_attack:
                    raise ValueError('Alpaca instruction/input duplicates or overlaps exclusions')
                alpaca_attack.add(attack)
            ids.add(index)
            texts.add(text)
            if q:
                questions.add(q)
            counts[index[0]]+=1
        if dict(counts)!=spec['source_counts']:
            raise ValueError('source counts differ from manifest')
        identity_sets[name],question_sets[name],text_sets[name],source_counts[name]=ids,questions,texts,dict(counts)
    for left,right in itertools.combinations(expected_counts,2):
        if identity_sets[left]&identity_sets[right] or question_sets[left]&question_sets[right] or text_sets[left]&text_sets[right]:
            raise ValueError(f'cross split leakage: {left}/{right}')
    general=source_counts['train'].get('alpaca',0)
    if general!=manifest['construction']['general_records'] or abs(general/expected_counts['train']-manifest['construction']['general_fraction_actual'])>1e-12:
        raise ValueError('general/science mixture inconsistent')
    return {'split_counts':expected_counts,'source_counts':source_counts,
            'source_identity_overlap':0,'normalized_question_overlap':0,'normalized_text_overlap':0,
            'fixed_exclusion_overlap':0,'actual_general_fraction':general/expected_counts['train'],
            'fresh_training_required':True,'security_accepted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest=json.loads(args.manifest.read_text())
    result=verify_dataset(manifest)
    result.update({'format':'osnip_question_disjoint_data_verification_v1','status':'verified_data_only',
                   'manifest_sha256':file_sha256(args.manifest),'verifier_sha256':file_sha256(Path(__file__)),
                   'limitations':['No model training or security validation.',
                                  'Exact normalized questions do not detect semantic paraphrases.',
                                  'Previously exposed holdout questions remain exposed.']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
