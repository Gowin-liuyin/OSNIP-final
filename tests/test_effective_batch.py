"""Compare bounded-memory batch gradients against retained reference graphs."""
import argparse
import copy

import pytest
import torch
from torch import nn

from scripts.train_streaming import backward_effective_batch, run_objective
from osnip_repro.curriculum import constraint_weight
from osnip_repro.encoder import KeyConditionedEncryptor


class FrozenPredictor(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embedding = nn.Embedding(13, 8, dtype=dtype)
        self.head = nn.Linear(8, 13, dtype=dtype)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **kwargs):
        # An actual frozen nonlinear predictor with gradients to embeddings.
        return argparse.Namespace(logits=self.head(torch.tanh(inputs_embeds)))


class Tokenizer:
    def __call__(self, texts, **kwargs):
        rows = [[int(i) for i in text.split()] for text in texts]
        size = max(map(len, rows))
        return {'input_ids': torch.tensor([r + [0]*(size-len(r)) for r in rows]),
                'attention_mask': torch.tensor([[1]*len(r)+[0]*(size-len(r)) for r in rows])}


def recipe():
    return argparse.Namespace(max_length=8, key_dim=2, lambda_util=1., lambda_priv=1.,
        lambda_div=2., lambda_hidden=0., lambda_knn=0., constraint_mode='gated',
        warmup_steps=200, constraint_ramp_steps=1000, loss_warmup_steps=10,
        tau_low=.00001, tau_high=.1, epsilon=.3, diversity_margin=1.,
        diversity_space='raw_r', gradient_balance='none')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_replayed_batch_matches_retained_graph_gradients_and_rng(dtype):
    torch.manual_seed(4)
    model = FrozenPredictor(dtype).eval()
    actual = KeyConditionedEncryptor(8, key_dim=2, hidden_dim=4)
    reference = copy.deepcopy(actual)
    args = recipe()
    batches = [['1 2 3'], ['4 5 6 7 8']]
    generator = torch.Generator().manual_seed(99)
    terms = []
    for texts in batches:
        terms.append(run_objective(model, reference, Tokenizer(), texts, args,
            torch.device('cpu'), training=True, step=5, key_generator=generator,
            constraint_weights=(0., 0.)))
    total_causal = sum(s['causal_tokens'] for _, s in terms)
    total_rep = sum(s['representation_tokens'] for _, s in terms)
    util = sum(loss['utility'] * stat['causal_tokens']/total_causal for loss, stat in terms)
    weights = [constraint_weight(5, float(util.detach()), base=base, warmup_steps=1000,
                                tau_low=args.tau_low, tau_high=args.tau_high) for base in (1., 2.)]
    expected = sum(.5 * (loss['utility']*stat['causal_tokens']/total_causal
                   + (weights[0]*loss['privacy'] + weights[1]*loss['diversity'])
                   * stat['representation_tokens']/total_rep) for loss, stat in terms)
    expected.backward()
    replay_generator = torch.Generator().manual_seed(99)
    report = backward_effective_batch(model, actual, Tokenizer(), batches, args,
        torch.device('cpu'), step=5, key_generator=replay_generator)
    assert torch.equal(generator.get_state(), replay_generator.get_state())
    assert report['effective_batch_examples'] == 2
    assert report['gate_utility'] == pytest.approx(float(util.detach()), rel=1e-6)
    assert report['privacy_weight'] == pytest.approx(weights[0])
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
        assert torch.isfinite(a.grad).all()
    assert all(p.grad is None for p in model.parameters())
    assert report['projection_fp32_max_relative_error'] < 1e-5


def test_separate_schedules_and_validation_unscaled():
    torch.manual_seed(4)
    model = FrozenPredictor(torch.float32)
    enc = KeyConditionedEncryptor(8, key_dim=2, hidden_dim=4)
    args = recipe()
    args.tau_low, args.tau_high = 0.5, 1.
    loss, stats = run_objective(model, enc, Tokenizer(), ['1 2 3'], args,
        torch.device('cpu'), training=True, step=5, key_seed=99)
    assert stats['loss_weight'] == .5
    assert stats['privacy_weight'] == .005
    assert stats['diversity_weight'] == .01
    torch.testing.assert_close(loss['total'], .5 * (loss['utility'] + .005*loss['privacy'] + .01*loss['diversity']))
    _, validation = run_objective(model, enc, Tokenizer(), ['1 2 3'], args,
        torch.device('cpu'), training=False, step=0, key_seed=99)
    assert validation['loss_weight'] == 1.


def test_shared_encryption_preserves_backbone_dtype_and_trainable_gradient():
    from osnip_repro.projection import encrypt_embeddings
    torch.manual_seed(7)
    encoder = KeyConditionedEncryptor(8, key_dim=2, hidden_dim=4)
    h = torch.randn(2, 4, 8).half()
    key = torch.randn(2, 2)
    z = encrypt_embeddings(encoder, h, key)
    assert z.dtype == h.dtype
    error = (z.float().norm(dim=-1)-h.float().norm(dim=-1)).abs()/h.float().norm(dim=-1)
    assert error.max() < 2*torch.finfo(torch.float16).eps
    (z.float().square()*torch.arange(1, 9)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
