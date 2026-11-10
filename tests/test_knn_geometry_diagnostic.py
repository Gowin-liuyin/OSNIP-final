import torch
import torch.nn.functional as F
import pytest
from scripts.diagnose_knn_geometry import matched_orthogonal


def test_random_orthogonal_control_matches_each_learned_cosine_and_norm():
    torch.manual_seed(81)
    h=torch.randn(257,96)
    noise=torch.randn_like(h)
    z=h+1.7*noise
    synthetic,c=matched_orthogonal(h,z,seed=20261024)
    assert torch.allclose(h.norm(dim=-1),synthetic.norm(dim=-1),rtol=1e-5,atol=1e-5)
    actual=F.cosine_similarity(h.double(),z.double(),dim=-1)
    matched=F.cosine_similarity(h.double(),synthetic.double(),dim=-1)
    assert torch.allclose(actual,matched,rtol=1e-6,atol=1e-6)
    second,_=matched_orthogonal(h,z,seed=20261024)
    assert torch.equal(synthetic,second)
    other,_=matched_orthogonal(h,z,seed=20261025)
    assert not torch.equal(synthetic,other)


def test_matched_orthogonal_rejects_zero_and_misaligned_inputs():
    with pytest.raises(ValueError):matched_orthogonal(torch.ones(2,3),torch.ones(3,3),seed=1)
    with pytest.raises(ValueError):matched_orthogonal(torch.zeros(2,3),torch.ones(2,3),seed=1)
