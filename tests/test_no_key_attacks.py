import pytest
import torch

from osnip_repro.attacks import vocabulary_topk, topk_counts


@pytest.mark.parametrize("metric,p", [("l2", 2), ("l1", 1)])
def test_full_vocabulary_ranking_matches_independent_distance(metric, p):
    g = torch.Generator().manual_seed(12)
    candidates = torch.randn(27, 7, generator=g)
    ids = torch.tensor([0, 5, 12, 26])
    query = candidates[ids] + 0.001 * torch.randn(4, 7, generator=g)
    expected = torch.cdist(query.double(), candidates.double(), p=p).topk(10, largest=False).indices
    ranked = vocabulary_topk(query, candidates, metric=metric, k=10, chunk=4)
    assert torch.equal(ranked, expected)
    assert topk_counts(ranked, ids) == {1: 4, 5: 4, 10: 4}


def test_attack_rejects_invalid_observations():
    with pytest.raises(ValueError, match="non-finite"):
        vocabulary_topk(torch.tensor([[float("nan")]]), torch.ones(12, 1), metric="l2")


def test_topk_reranks_near_ties_exactly_and_breaks_ties_by_lowest_id():
    torch.manual_seed(3)
    base = torch.randn(64) * 0.2
    near = base.clone(); near[0] += 1e-3
    nearer = base.clone(); nearer[0] += 5e-4
    candidates = torch.stack([base, near, torch.full((64,), -7.0), nearer])
    rankings = vocabulary_topk(base[None], candidates, metric="l2", k=3, chunk=2)
    assert rankings[0].tolist() == [0, 3, 1]
    duplicated = torch.stack([near, near.clone(), base])
    tied = vocabulary_topk(base[None], duplicated, metric="l2", k=2, chunk=4)
    assert tied[0].tolist() == [2, 0]
    l1 = vocabulary_topk(base[None], candidates, metric="l1", k=2, chunk=2)
    assert l1[0].tolist() == [0, 3]
