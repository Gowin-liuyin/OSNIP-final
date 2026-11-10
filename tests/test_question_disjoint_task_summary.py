import importlib.util
import sys
from pathlib import Path

import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('filtered_task',scripts/'summarize_question_disjoint_task.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_task():
    raw={'sciq':[{'question':f'q{i}','correct_answer':'yes','distractor1':'no','distractor2':'later','distractor3':'maybe'} for i in range(3)]}
    test=[{'source':'sciq','source_index':i} for i in range(3)]
    rows=[]
    for i in range(3):
        _,_,labels,answer=module.sciq_task(raw['sciq'][i],seed=42+i)
        rows.append({'index':i,'source':'sciq','choices':labels,'correct':labels[answer],
                     'clean_prediction':labels[answer],'encrypted_prediction':labels[answer],
                     'clean_correct':True,'encrypted_correct':True})
    task={'records_selected':3,'records_scored':3,'rows':rows,'choice_policy':{
        'sciq':'shuffle_with_independent_rng','per_example_seed':'choice_seed + task_index','seed':42}}
    return task,test,raw


def test_filtered_task_keeps_original_choice_and_key_indices():
    task,test,raw=fixture_task()
    result=module.filter_existing_scores(task,test,raw,{0})
    assert [row['index'] for row in result]==[1,2]
    assert result[0]['correct']==task['rows'][1]['correct']
    assert result[1]['source_index']==2


def test_filtered_task_rejects_mismatched_labels_or_incomplete_rows():
    task,test,raw=fixture_task()
    task['rows'][1]['correct']='invalid'
    with pytest.raises(ValueError,match='choices/labels'):
        module.filter_existing_scores(task,test,raw,{0})
    task,test,raw=fixture_task()
    task['rows'][1]['index']=0
    with pytest.raises(ValueError,match='indices'):
        module.filter_existing_scores(task,test,raw,{0})
