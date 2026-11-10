import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('privacy_integrity_aggregate', ROOT/'scripts/aggregate_reproduction_report.py')
aggregate = importlib.util.module_from_spec(spec); spec.loader.exec_module(aggregate)


@pytest.fixture
def report(tmp_path):
    for name in ('knn','vma'):
        (tmp_path/(name+'.jsonl')).write_text(''.join(json.dumps({'text':f'{name} prompt {i}'})+'\n' for i in range(2)))
    counts = {side:{k:6 if side=='clean' else 0 for k in ('top1','top5','top10')} for side in ('clean','encrypted')}
    def observation(prompt, layer, key):
        return {'prompt':prompt,'layer':layer,'key':key,'positions':3,'content_positions':2,
                'total_top1_hits':3 if key==-1 else 0,'clean_top1_hits':2 if key==-1 else 0,'candidate_coverage':1.}
    return {'format':'osnip_paper_protocol_evaluation_v1','key_attack_performed':False,
            'knn_jsonl':str(tmp_path/'knn.jsonl'),'vma_jsonl':str(tmp_path/'vma.jsonl'),
            'knn':{'samples':2,'positions':6,'counts':counts,'metric':'l2','full_vocabulary':True,'vocabulary_size':16,
                   'rates':{s:{k:v/6 for k,v in values.items()} for s,values in counts.items()},
                   'rows':[{'index':i,'positions':3,'clean':{k:3 for k in ('top1','top5','top10')},
                            'encrypted':{k:0 for k in ('top1','top5','top10')}} for i in range(2)]},
            'vma':{'prompts':2,'keys_per_prompt':2,'layers':[1,2],'max_tokens':8,'full_vocabulary':True,'candidate_mode':'full',
                   'summary':{str(layer):{'clean_positive_control':{'total_asr':1.,'clean_asr':1.},
                                          'encrypted':{'total_asr':0.,'clean_asr':0.}} for layer in (1,2)},
                   'clean_rows':[observation(i,l,-1) for i in range(2) for l in (1,2)],
                   'rows':[observation(i,l,k) for i in range(2) for l in (1,2) for k in range(2)]}}


def test_complete_cartesian_coverage_and_recomputed_counts(report):
    result=aggregate.verify_privacy_observations(report)
    assert result == {'passed':True,'errors':[],'knn_rows_verified':2,
                      'vma_clean_observations_verified':4,'vma_encrypted_observations_verified':8}


@pytest.mark.parametrize('mutation', ['missing_vma', 'duplicate_vma', 'missing_clean', 'wrong_key', 'layer_gap',
    'bad_hits', 'vma_rate', 'knn_rate', 'missing_knn', 'duplicate_knn', 'wrong_positions', 'reduced_candidates',
    'zero_coverage', 'boolean_count', 'inconsistent_tokens', 'nonfinite_asr', 'impossible_content_hits'])
def test_invalid_or_incomplete_observations_are_rejected(report,mutation):
    bad=copy.deepcopy(report)
    if mutation=='missing_vma': bad['vma']['rows'].pop()
    elif mutation=='duplicate_vma': bad['vma']['rows'][-1]=bad['vma']['rows'][0]
    elif mutation=='missing_clean': del bad['vma']['clean_rows']
    elif mutation=='wrong_key': bad['vma']['rows'][0]['key']=-1
    elif mutation=='layer_gap': del bad['vma']['summary']['2']
    elif mutation=='bad_hits': bad['vma']['rows'][0]['total_top1_hits']=-1
    elif mutation=='vma_rate': bad['vma']['summary']['1']['encrypted']['total_asr']=.1
    elif mutation=='knn_rate': bad['knn']['rates']['encrypted']['top10']=.1
    elif mutation=='missing_knn': del bad['knn']['rows']
    elif mutation=='duplicate_knn': bad['knn']['rows'][1]['index']=0
    elif mutation=='wrong_positions': bad['knn']['positions']=5
    elif mutation=='reduced_candidates': bad['vma']['candidate_mode']='proposal'
    elif mutation=='zero_coverage': bad['vma']['rows'][0]['candidate_coverage']=0.
    elif mutation=='boolean_count': bad['vma']['rows'][0]['prompt']=False
    elif mutation=='inconsistent_tokens': bad['vma']['rows'][0]['positions']=4
    elif mutation=='nonfinite_asr': bad['vma']['summary']['1']['encrypted']['total_asr']=float('nan')
    elif mutation=='impossible_content_hits': bad['vma']['rows'][0].update(content_positions=3,total_top1_hits=1,clean_top1_hits=0)
    result=aggregate.verify_privacy_observations(bad)
    assert not result['passed'] and result['errors']


def test_distinct_attack_inputs_and_actual_record_count_are_required(report):
    path=Path(report['vma_jsonl'])
    path.write_text(json.dumps({'text':'same'})+'\n'+json.dumps({'text':'same'})+'\n')
    assert not aggregate.verify_privacy_observations(report)['passed']
    path.write_text(json.dumps({'text':'one'})+'\n')
    assert not aggregate.verify_privacy_observations(report)['passed']


def test_unequal_length_prompts_use_token_weighted_rates(report):
    # Add one content token to every observation of prompt 1, not just one key/layer.
    for row in report['vma']['clean_rows']+report['vma']['rows']:
        if row['prompt']==1:
            row['positions']+=1; row['content_positions']+=1
            if row['key']==-1:
                row['total_top1_hits']+=1; row['clean_top1_hits']+=1
            else:
                row['total_top1_hits']=1; row['clean_top1_hits']=1
    for layer in report['vma']['summary'].values():
        layer['encrypted']={'total_asr':1/7,'clean_asr':1/5}
    assert aggregate.verify_privacy_observations(report)['passed']
    report['vma']['summary']['1']['encrypted']['total_asr']=.5*(0+1/4)
    assert not aggregate.verify_privacy_observations(report)['passed']
