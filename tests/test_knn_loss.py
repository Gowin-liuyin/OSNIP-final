import torch

from osnip_repro.losses import vocabulary_margin_loss


def test_vocabulary_margin_loss_is_finite_and_deterministic():
    weight = torch.eye(8)
    z = weight[:3].clone()
    targets = torch.tensor([0, 1, 2])
    first = vocabulary_margin_loss(z, targets, weight, negative_count=4, topk=2, margin=0.1)
    second = vocabulary_margin_loss(z, targets, weight, negative_count=4, topk=2, margin=0.1)
    assert torch.isfinite(first)
    assert torch.equal(first, second)


def test_sampled_margin_penalizes_leakage_and_gradient_improves_margin():
    weight = torch.eye(8)
    targets = torch.tensor([0])
    exposed = weight[:1].clone()
    hidden = -exposed
    assert vocabulary_margin_loss(exposed, targets, weight, negative_count=7, topk=2, margin=0.1) > 1
    assert vocabulary_margin_loss(hidden, targets, weight, negative_count=7, topk=2, margin=0.1) == 0
    z = (exposed + 0.2 * torch.ones_like(exposed)).requires_grad_()
    before = vocabulary_margin_loss(z, targets, weight, negative_count=7, topk=2, margin=0.1)
    grad, = torch.autograd.grad(before, z)
    after = vocabulary_margin_loss(z.detach() - 0.1 * grad, targets, weight, negative_count=7, topk=2, margin=0.1)
    assert after < before


def test_full_negative_sample_matches_exact_cosine_margin_with_no_duplicates():
    import torch.nn.functional as F
    weight = torch.eye(8)
    targets = torch.tensor([0])
    # Distinct similarities make a repeated/missing negative change the kth rank.
    z = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
    similarities = F.normalize(z, dim=-1)[0]
    expected = (similarities[0] + 2.0 - similarities[1:].topk(4).values[-1]).clamp_min(0)
    actual = vocabulary_margin_loss(z, targets, weight, negative_count=7, topk=4, margin=2.0)
    torch.testing.assert_close(actual, expected)
