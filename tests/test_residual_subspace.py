import importlib.util
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('subspace_diagnostic',Path(__file__).resolve().parents[1]/'scripts/diagnose_residual_subspace.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_output_basis_handles_dependent_columns_and_bias():
    weight=torch.tensor([[1.,2.],[0.,0.],[0.,0.]])
    bias=torch.tensor([0.,1.,0.])
    basis,_=module.output_basis(weight,bias)
    assert basis.shape==(3,2)
    expected=torch.diag(torch.tensor([1.,1.,0.],dtype=torch.float64))
    assert torch.allclose(basis@basis.T,expected)


def test_fixed_norm_bound_is_attained_by_opposite_projected_component():
    # ||h||=1, subspace is x-axis; raw perpendicular component remains .8.
    h=torch.tensor([.6,.8],dtype=torch.float64)
    raw=torch.tensor([-1.2,.8],dtype=torch.float64)
    rho=raw.norm()
    bound=module.fixed_norm_cosine_lower_bound(torch.tensor(.36),rho)
    cosine=(h@raw)/(h.norm()*raw.norm())
    assert float(bound)==pytest.approx(float(cosine),abs=1e-7)


def test_fixed_norm_bound_contains_random_attainable_residuals():
    rng=torch.Generator().manual_seed(18)
    h=torch.randn(100,7,generator=rng,dtype=torch.float64)
    raw=h.clone()
    raw[:,:3]+=torch.randn(100,3,generator=rng,dtype=torch.float64)
    fraction=h[:,:3].square().sum(-1)/h.square().sum(-1)
    bound=module.fixed_norm_cosine_lower_bound(fraction,raw.norm(dim=-1)/h.norm(dim=-1))
    cosine=torch.nn.functional.cosine_similarity(h,raw,dim=-1)
    assert (bound<=cosine+1e-12).all()


def test_fixed_norm_bound_exposes_zero_projection_norm_cost():
    fraction=torch.zeros(2)
    bounds=module.fixed_norm_cosine_lower_bound(fraction,torch.tensor([1.,4.]))
    assert bounds.tolist()==[1.,.25]
    with pytest.raises(ValueError,match='perpendicular'):
        module.fixed_norm_cosine_lower_bound(torch.tensor([0.]),torch.tensor([.5]))
