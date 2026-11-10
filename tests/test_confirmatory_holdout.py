import importlib.util
import sys
from pathlib import Path

import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('confirmatory',scripts/'prepare_confirmatory_holdout.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def science(question,answer='yes'):
    return {'question':question,'correct_answer':answer,'distractor1':'no','distractor2':'maybe','distractor3':'later'}


def test_question_leakage_detected_across_format_source_and_case():
    raw={'arc_easy':[{'question':'Shared Question','choices':{'label':['A'],'text':['x']},'answerKey':'A'}],
         'arc_challenge':[], 'sciq':[science('  SHARED   question '),science('Fresh 1'),science('Fresh 2'),science('Fresh 3')]}
    row=lambda s,i:{'source':s,'source_index':i,'text':str((s,i))}
    splits={'train':[row('arc_easy',0)],'valid':[], 'test':[row('sciq',0),row('sciq',1)]}
    result=module.audit_and_select(raw,splits,{},count=2,seed=9)
    assert result['question_overlap']['train__test']['distinct_questions']==1
    assert result['dropped_test_indices']==[0]
    assert len(result['clean_test'])==1
    assert {r['source_index'] for r in result['reserved']}=={2,3}
    assert result['reserved']==module.audit_and_select(raw,splits,{},count=2,seed=9)['reserved']


def test_reserve_excludes_attack_records_and_repeated_questions():
    raw={'arc_easy':[],'arc_challenge':[], 'sciq':[science('seen'),science('Seen','other'),science('fresh')]}
    splits={'train':[],'valid':[],'test':[]}
    excluded={'vma':[{'source':'sciq','source_index':0,'text':'unrelated serialization'}]}
    result=module.audit_and_select(raw,splits,excluded,count=1,seed=1)
    assert result['reserved'][0]['source_index']==2
    with pytest.raises(ValueError,match='only 1'):
        module.audit_and_select(raw,splits,excluded,count=2,seed=1)
    with pytest.raises(ValueError,match='source index'):
        module.audit_and_select(raw,splits,{'vma':[{'source':'sciq','source_index':99}]},count=1,seed=1)
