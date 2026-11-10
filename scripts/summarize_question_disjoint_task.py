#!/usr/bin/env python3
"""Filter an existing task evaluation without changing its scoring/key draws."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from osnip_repro.data import _iter_json_values,file_sha256
from evaluate_task_utility import arc_task,sciq_task
from summarize_task_utility import summarize


def filter_existing_scores(task, test_rows, raw, dropped_indices):
    if task['records_selected']!=len(test_rows) or task['records_scored']!=len(test_rows):
        raise ValueError('requires a complete unfiltered supported-source task evaluation')
    rows=task['rows']
    if len(rows)!=len(test_rows) or sorted(row['index'] for row in rows)!=list(range(len(test_rows))):
        raise ValueError('scored indices are missing, duplicated or not aligned to the test file')
    policy=task.get('choice_policy',{})
    if policy.get('sciq')!='shuffle_with_independent_rng' or policy.get('per_example_seed')!='choice_seed + task_index':
        raise ValueError('requires corrected shuffled-choice task protocol')
    selected=[]
    for row in rows:
        index=row['index']
        record=test_rows[index]
        source=record['source']
        if source not in raw or row['source']!=source:
            raise ValueError('source mismatch')
        source_index=record['source_index']
        original=raw[source][source_index]
        expected=arc_task(original) if source.startswith('arc') else sciq_task(original,seed=policy['seed']+index)
        if expected is None:
            raise ValueError('saved choices/labels do not match the pinned source and original task index')
        _stem,_options,expected_labels,expected_correct=expected
        if row['choices']!=expected_labels or row['correct']!=expected_labels[expected_correct]:
            raise ValueError('saved choices/labels do not match the pinned source and original task index')
        if row['clean_correct']!=(row['clean_prediction']==row['correct']) or row['encrypted_correct']!=(row['encrypted_prediction']==row['correct']):
            raise ValueError('saved correctness disagrees with predictions')
        if index not in dropped_indices:
            selected.append({**row,'source_index':source_index})
    if not selected:
        raise ValueError('no question-disjoint rows remain')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-report',type=Path,required=True)
    p.add_argument('--holdout-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--bootstrap-samples',type=int,default=5000)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_samples<2:
        raise ValueError('at least two bootstrap samples required')
    task=json.loads(args.task_report.read_text())
    manifest=json.loads(args.holdout_manifest.read_text())
    test=manifest['input_hashes']['test']
    if file_sha256(test['path'])!=test['sha256'] or task['test_sha256']!=test['sha256']:
        raise ValueError('test file/report/holdout manifest hash mismatch')
    raw={}
    for source,spec in manifest['raw_sources'].items():
        if file_sha256(spec['path'])!=spec['file_sha256']:
            raise ValueError('raw source hash mismatch')
        raw[source]=list(_iter_json_values(Path(spec['path'])))
    clean_spec=manifest['existing_test_question_disjoint']
    if file_sha256(clean_spec['path'])!=clean_spec['sha256']:
        raise ValueError('clean test subset changed')
    test_rows=list(_iter_json_values(Path(test['path'])))
    filtered=[row for i,row in enumerate(test_rows) if i not in set(manifest['dropped_test_indices'])]
    if filtered!=list(_iter_json_values(Path(clean_spec['path']))):
        raise ValueError('clean subset disagrees with exclusion indices')
    rows=filter_existing_scores(task,test_rows,raw,set(manifest['dropped_test_indices']))
    result={'format':'osnip_question_disjoint_task_diagnostic_v1','scope':'retrospective_subset_of_existing_task_scores',
            'security_accepted':False,'key_attack_performed':False,
            'checkpoint_sha256':task['checkpoint_sha256'],'task_report_sha256':file_sha256(args.task_report),
            'holdout_manifest_sha256':file_sha256(args.holdout_manifest),'script_sha256':file_sha256(Path(__file__)),
            'bootstrap_samples':args.bootstrap_samples,'bootstrap_seed':args.seed,
            'original_rows':len(task['rows']),'removed_rows':len(task['rows'])-len(rows),'rows':rows,
            'choice_policy':task['choice_policy'],'by_source':summarize(rows,seed=args.seed,samples=args.bootstrap_samples),
            'limitations':['Retrospective subset remains exposed and is not fresh confirmatory data.',
                           'Original task indices, shuffled choices and client-key assignments are preserved; no new model evaluation.',
                           'Question identity detects exact normalized duplicates, not semantic paraphrases.',
                           'Filtered utility does not replace full privacy or benchmark acceptance.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'removed_rows':result['removed_rows'],'by_source':result['by_source']},indent=2))


if __name__=='__main__':
    main()
