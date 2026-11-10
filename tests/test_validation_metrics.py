import math

import pytest

from osnip_repro.validation import aggregate_validation_stats, control_acceptance


def test_corpus_perplexity_weights_nll_by_valid_targets():
    # One easy token and three difficult ones: arithmetic mean of PPL is wrong.
    rows = [dict(clean_nll_sum=0., encrypted_nll_sum=1., causal_tokens=1, representation_tokens=2, examples=1,
                 norm_max_abs_error=0., utility=1., privacy=.5, clean_ppl=1., encrypted_ppl=math.e),
            dict(clean_nll_sum=6., encrypted_nll_sum=9., causal_tokens=3, representation_tokens=4, examples=1,
                 norm_max_abs_error=1e-7, utility=2., privacy=.2, clean_ppl=math.exp(2), encrypted_ppl=math.exp(3))]
    result = aggregate_validation_stats(rows)
    assert result['clean_ppl'] == pytest.approx(math.exp(6/4))
    assert result['encrypted_ppl'] == pytest.approx(math.exp(10/4))
    assert result['ppl_ratio'] == pytest.approx(math.e)
    assert result['utility'] == pytest.approx(7/4)
    assert result['privacy'] == pytest.approx(.3)
    assert result['norm_max_abs_error'] == 1e-7


def test_finite_but_failed_controls_cannot_pass_behavior_gate():
    assert not control_acceptance('utility_only', {'utility': .01}, {'utility': .05, 'ppl_ratio': 1.0})['passed']
    baseline = {'mean_abs_cosine_key1': .99, 'mean_abs_cosine_key2': .99}
    final = {'mean_abs_cosine_key1': .8, 'mean_abs_cosine_key2': .8, 'privacy_key1': .5, 'privacy_key2': .5}
    assert not control_acceptance('privacy_only', baseline, final)['passed']
    final.update(mean_abs_cosine_key1=.28, mean_abs_cosine_key2=.27, privacy_key1=.02, privacy_key2=.01)
    assert control_acceptance('privacy_only', baseline, final)['passed']
    final['mean_abs_cosine_key2'] = .9
    assert not control_acceptance('privacy_only', baseline, final)['passed']
