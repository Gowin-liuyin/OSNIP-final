import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('question_training',scripts/'prepare_question_disjoint_training.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pools_fixture():
    pools={name:[] for name in ('alpaca','arc_easy','arc_challenge','sciq')}
    for source in pools:
        for i in range(12):
            row={'source':source,'source_index':i,'text':f'{source} text {i}',
                 'text_sha256':f'{source}:text:{i}','attack_text_sha256':f'{source}:attack:{i}'}
            if source!='alpaca':
                row['question_sha256']=f'{source}:question:{i}'
            pools[source].append(row)
    # Different full strings/source labels but exactly the same question.
    pools['arc_challenge'][0]['question_sha256']=pools['arc_easy'][0]['question_sha256']
    pools['alpaca'][1]['attack_text_sha256']=pools['alpaca'][0]['attack_text_sha256']
    return pools


def test_new_splits_exclude_attack_confirmation_and_cross_source_questions():
    pools=pools_fixture(); original=copy.deepcopy(pools)
    excluded=[{'source':'sciq','source_index':0},
              {'source':'alpaca_knn_holdout','source_index':2}]
    valid=[{'source':'arc_easy','source_index':0}]
    test=[{'source':'arc_challenge','source_index':0},{'source':'sciq','source_index':1}]
    params=dict(excluded=excluded,legacy_valid=valid,legacy_test=test,
                train_total=20,valid_count=3,test_count=3,seed=27)
    splits,meta=module.construct(pools,**params)
    assert pools==original
    assert {name:len(rows) for name,rows in splits.items()}=={'train':20,'valid':3,'test':3}
    questions={name:{r['question_sha256'] for r in rows if 'question_sha256' in r} for name,rows in splits.items()}
    assert not (questions['train']&questions['valid'] or questions['train']&questions['test'] or questions['valid']&questions['test'])
    ids={(row['source'],row['source_index']) for rows in splits.values() for row in rows}
    assert ('sciq',0) not in ids and ('alpaca',2) not in ids
    all_q=[r['question_sha256'] for rows in splits.values() for r in rows if 'question_sha256' in r]
    assert all_q.count(pools['arc_easy'][0]['question_sha256'])==1
    assert pools['sciq'][1]['question_sha256'] in questions['test']
    assert ('alpaca',1) not in ids
    assert module.construct(pools,**params)==(splits,meta)


def test_new_splits_fail_on_insufficient_capacity_or_invalid_exclusion():
    pools=pools_fixture()
    with pytest.raises(ValueError,match='insufficient'):
        module.construct(pools,excluded=[],legacy_valid=[],legacy_test=[],
                         train_total=100,valid_count=1,test_count=1,seed=4)
    with pytest.raises(ValueError,match='source identity'):
        module.excluded_identities([{'source':'sciq','source_index':99}],pools)


def test_independent_verifier_recomputes_raw_content_and_rejects_rehashed_tampering(tmp_path):
    import json
    verifier_spec=importlib.util.spec_from_file_location('question_training_verifier',scripts/'verify_question_disjoint_training.py')
    verifier=importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    paths={}
    for source in ('alpaca','arc_easy','arc_challenge','sciq'):
        rows=[]
        for i in range(15):
            if source=='alpaca':
                rows.append({'instruction':f'Instruction {i}','input':'','output':f'answer {i}'})
            elif source.startswith('arc'):
                rows.append({'question':f'{source} question {i}',
                    'choices':{'label':['A','B'],'text':['correct','wrong']},'answerKey':'A'})
            else:
                rows.append({'question':f'SciQ question {i}','correct_answer':'correct',
                    'distractor1':'wrong1','distractor2':'wrong2','distractor3':'wrong3'})
        path=tmp_path/f'{source}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        paths[source]=path
    pools=module.enrich_sources(paths)
    excluded=[{'source':'sciq','source_index':0}]
    splits,construction=module.construct(pools,excluded=excluded,legacy_valid=[],legacy_test=[],
                                      train_total=20,valid_count=3,test_count=3,seed=18)
    exclusion_path=tmp_path/'exclude.jsonl'
    exclusion_path.write_text(json.dumps(excluded[0])+'\n')
    manifest={'format':'osnip_question_disjoint_dataset_manifest_v1',
        'sources':{name:{'path':str(path),'file_sha256':module.file_sha256(path)} for name,path in paths.items()},
        'exclusions':{'reserve':{'path':str(exclusion_path),'file_sha256':module.file_sha256(exclusion_path)}},
        'splits':{},'construction':construction}
    for name,rows in splits.items():
        path=tmp_path/f'{name}.jsonl'
        digest=module.write_jsonl(path,rows)
        from collections import Counter
        manifest['splits'][name]={'path':str(path),'file_sha256':digest,'records':len(rows),
                                 'source_counts':dict(Counter(row['source'] for row in rows))}
    expected={'train':20,'valid':3,'test':3}
    assert verifier.verify_dataset(manifest,expected_counts=expected)['fixed_exclusion_overlap']==0
    # Updating the outer file hash must not hide a fabricated question identity.
    changed=copy.deepcopy(splits['valid'])
    changed[0]['question_sha256']='forged'
    path=Path(manifest['splits']['valid']['path'])
    manifest['splits']['valid']['file_sha256']=module.write_jsonl(path,changed)
    with pytest.raises(ValueError,match='question hash'):
        verifier.verify_dataset(manifest,expected_counts=expected)
    manifest['splits']['valid']['file_sha256']=module.write_jsonl(path,splits['valid'])
    # An exclusion copy must fail even with self-consistent source data/hashes.
    changed=copy.deepcopy(splits['valid'])
    changed[0]=pools['sciq'][0]
    manifest['splits']['valid']['file_sha256']=module.write_jsonl(path,changed)
    with pytest.raises(ValueError,match='exclusions'):
        verifier.verify_dataset(manifest,expected_counts=expected)
