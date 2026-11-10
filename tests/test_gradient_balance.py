import pytest

import argparse

import torch

from scripts.train_streaming import backward_balanced


def test_backward_balanced_normalizes_component_gradients():
    encoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(1.0)
    x = torch.tensor([[2.0]])
    y = encoder(x)
    losses = {"utility": (y - 4.0).pow(2).mean(), "privacy": (y + 2.0).pow(2).mean()}
    norms = backward_balanced(losses, encoder, component_weights={"utility": 1.0, "privacy": 1.0}, accumulation_steps=1)
    assert norms["utility"] > 0 and norms["privacy"] > 0
    assert encoder.weight.grad is not None
    assert torch.isfinite(encoder.weight.grad).all()


def test_aggregate_refuses_empty_representation_weights_instead_of_zeroing_privacy():
    from osnip_repro.validation import aggregate_validation_stats
    rows = [{"causal_tokens": 10, "representation_tokens": 0, "examples": 1,
             "clean_nll_sum": 5.0, "encrypted_nll_sum": 5.0,
             "utility": 0.1, "privacy": 9.0, "privacy_key1": 9.0,
             "norm_max_abs_error": 0.0}]
    with pytest.raises(ValueError, match="no representation_tokens"):
        aggregate_validation_stats(rows)
    healthy = [{"causal_tokens": 10, "representation_tokens": 8, "examples": 1,
                "clean_nll_sum": 5.0, "encrypted_nll_sum": 5.0,
                "utility": 0.1, "privacy": 0.5, "privacy_key1": 0.5,
                "norm_max_abs_error": 0.0}]
    result = aggregate_validation_stats(healthy)
    assert result["privacy"] == pytest.approx(0.5)


def test_aggregate_refuses_ragged_batch_schemas():
    from osnip_repro.validation import aggregate_validation_stats
    base = {"causal_tokens": 10, "representation_tokens": 8, "examples": 1,
            "clean_nll_sum": 5.0, "encrypted_nll_sum": 5.0,
            "utility": 0.1, "privacy": 0.5, "norm_max_abs_error": 0.0}
    late = dict(base, extra_late_metric=2.0)
    with pytest.raises(ValueError, match="inconsistent validation row schema"):
        aggregate_validation_stats([base, late])
    result = aggregate_validation_stats([late, dict(late)])
    assert result["extra_late_metric"] == pytest.approx(2.0)
