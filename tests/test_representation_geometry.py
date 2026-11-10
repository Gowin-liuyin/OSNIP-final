"""Analytical controls for client-side geometry; does not test attack success."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('geometry_diagnostic', Path(__file__).resolve().parents[1]/'scripts/diagnose_representation_geometry.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def test_shared_and_centered_energy_match_analytical_decomposition():
    values = torch.tensor([[4., 1.], [2., 1.]])
    result = geometry.energy_components(values)
    assert result['total_energy'] == 22
    assert result['shared_energy'] == 20
    assert result['centered_energy'] == 2
    assert result['shared_fraction'] == pytest.approx(20/22)
    assert geometry.energy_components(torch.zeros(2, 3))['shared_fraction'] is None


def test_shared_offset_can_dominate_energy_while_preserving_token_differences():
    h = torch.tensor([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    offsets = torch.tensor([[0., 0., 5.], [0., 0., -5.]])
    raw = h[None] + offsets[:, None]
    result = geometry.summarize_geometry(h, raw, epsilon=.3, margin=1.)
    assert result['raw_residual_shared_fraction'] == pytest.approx(1.)
    assert result['centered_raw_change_relative_to_h_energy'] == pytest.approx(0., abs=1e-12)
    assert result['fraction_abs_cosine_le_epsilon'] == 1.
    assert result['max_relative_projection_norm_error'] < 1e-6
    assert result['raw_key_pair_mean_distance'] == 10.
    assert result['projected_key_pair_mean_distance'] == pytest.approx(10/(26**.5))


def test_projection_can_remove_raw_key_diversity():
    h = torch.eye(3)
    raw = torch.stack((2*h, 5*h))
    result = geometry.summarize_geometry(h, raw, epsilon=.3, margin=1.)
    assert result['raw_key_pair_margin_satisfied_fraction'] == 1.
    assert result['projected_key_pair_margin_satisfied_fraction'] == 0.
    assert result['projected_key_pair_mean_distance'] == 0.


def test_geometry_rejects_nonfinite_and_too_few_tokens():
    with pytest.raises(ValueError):
        geometry.energy_components(torch.ones(1, 3))
    with pytest.raises(ValueError):
        geometry.summarize_geometry(torch.ones(2, 3), torch.full((2, 2, 3), float('nan')), epsilon=.3, margin=1.)


def test_all_geometry_surrogates_and_toy_utility_can_pass_with_full_recovery():
    result = geometry.analytical_surrogate_control()
    assert result['knn_top1'] == 1.
    assert result['max_utility_kl'] < 1e-6
    assert result['geometry']['privacy_hinge'] == 0.
    assert result['geometry']['projected_key_pair_margin_satisfied_fraction'] == 1.
    assert result['geometry']['max_relative_projection_norm_error'] < 1e-6
