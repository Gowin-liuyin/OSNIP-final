import torch
import pytest
from scripts.validation_knn_checkpoint import summarize, diagnose
from scripts.diagnose_validation_retrieval import streamed_l2_topk


def test_token_evidence_and_conditional_counts_agree_with_full_distance_reference():
    torch.manual_seed(56)
    table=torch.randn(19,6)
    ids=[[1,7],[3,8,11]]
    clean=table[torch.tensor([1,7,3,8,11])]
    encrypted=clean.clone()
    encrypted[2:]=table[torch.tensor([6,12,14])]
    ranked=streamed_l2_topk(torch.cat([clean,encrypted]),19,lambda a,b:table[a:b],
                            candidate_chunk=4,query_chunk=3)
    result=summarize(ranked,ids,clean,encrypted,.3)
    assert result['rates']['clean']['top1']==1
    assert result['rates']['encrypted']['top1']==pytest.approx(2/5)
    assert [t['target_id'] for t in result['tokens']]==[1,7,3,8,11]
    assert [t['token_position'] for t in result['tokens']]==[0,1,0,1,2]
    for t in result['tokens']:
        assert t['target_rank_if_top10']==(t['encrypted_top10_ids'].index(t['target_id'])+1
            if t['target_id'] in t['encrypted_top10_ids'] else None)
    assert sum(result['conditional_recovery'][k]['positions'] for k in
               ('abs_cosine_le_epsilon', 'abs_cosine_gt_epsilon')) == 5
    assert sum(result['conditional_recovery'][k]['positions'] for k in
               ('signed_cosine_le_epsilon', 'signed_cosine_gt_epsilon')) == 5
    assert sum(result['conditional_recovery'][k]['positions'] for k in
               ('negative_cosine', 'positive_cosine')) == 5


def test_diagnostic_refuses_overwriting_existing_evidence(tmp_path):
    output=tmp_path/'report.json';output.write_text('existing')
    with pytest.raises(FileExistsError):diagnose(tmp_path/'missing.pt',output)
    assert output.read_text()=='existing'
