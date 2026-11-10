import torch

from osnip_repro.curriculum import utility_safety_gate, warmup_weight
from osnip_repro.encoder import KeyConditionedEncryptor
from osnip_repro.losses import diversity_hinge, osnip_two_key_loss, privacy_hinge, utility_kl
from osnip_repro.projection import iso_norm_project


def test_encoder_shape_and_key_sensitivity():
    torch.manual_seed(1)
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=12)
    h = torch.randn(3, 8)
    z1 = model(h, torch.ones(4))
    z2 = model(h, -torch.ones(4))
    assert z1.shape == h.shape
    assert not torch.allclose(z1, z2)


def test_paper_losses_and_projection():
    h = torch.randn(5, 8)
    z = h * 2
    assert utility_kl(torch.randn(5, 11), torch.randn(5, 11)).item() >= 0
    assert privacy_hinge(h, h, 1.0).item() < 1e-6
    assert diversity_hinge(h, h, 1.0).item() > 0.0
    projected = iso_norm_project(z, h)
    assert torch.allclose(projected.norm(dim=-1), h.norm(dim=-1), atol=1e-5)


def test_curriculum_bounds():
    assert warmup_weight(0, 100) == 0.0
    assert warmup_weight(100, 100) == 1.0
    assert utility_safety_gate(0.01, 0.01, 0.1) == 1.0
    assert utility_safety_gate(0.1, 0.01, 0.1) == 0.0


def test_two_key_objective_constrains_both_keys():
    clean = torch.randn(4, 9)
    h = torch.randn(4, 8)
    z1, z2 = h.clone(), -h.clone()
    result = osnip_two_key_loss(clean, clean, clean, h, z1, z2, epsilon=0.3, lambda_priv=1.0)
    assert result["utility"].item() < 1e-6
    assert result["privacy"].item() > 0.0
    assert result["privacy_key1"].item() > 0.0
    assert result["privacy_key2"].item() > 0.0


def test_causal_utility_excludes_sequence_first_position():
    from osnip_repro.modeling import causal_nll_stats
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    logits = torch.full((1, 3, 5), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    nll, count = causal_nll_stats(logits, ids, mask)
    assert count == 2
    assert nll.item() < 1e-5


def test_encoder_supports_independent_key_per_batch_example():
    torch.manual_seed(3)
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=12)
    h = torch.randn(2, 5, 8)
    keys = torch.randn(2, 4)
    batched = model(h, keys)
    separate = torch.stack([model(h[i], keys[i]) for i in range(2)])
    assert torch.allclose(batched, separate)


def test_all_encryptor_architectures_preserve_contract_and_projection():
    from osnip_repro.encoder import ARCHITECTURES

    torch.manual_seed(11)
    h = torch.randn(2, 4, 8)
    keys = torch.randn(2, 4)
    for architecture in ARCHITECTURES:
        model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=12, architecture=architecture)
        z = model(h, keys)
        assert z.shape == h.shape, architecture
        assert torch.isfinite(z).all(), architecture
        assert not torch.allclose(z[0], model(h[0], torch.ones(4))), architecture
        projected = iso_norm_project(z, h)
        assert torch.allclose(projected.norm(dim=-1), h.norm(dim=-1), atol=1e-5), architecture


def test_full_width_residual_uses_model_width_for_effective_hidden():
    from osnip_repro.encoder import KeyConditionedEncryptor
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=2, architecture="full_width_residual_v1")
    assert model.effective_hidden_dim == 8
    h = torch.randn(2, 3, 8)
    key = torch.randn(2, 4)
    z = model(h, key)
    assert z.shape == h.shape
    assert torch.isfinite(z).all()


def test_contextual_residual_uses_sequence_context_and_mask():
    from osnip_repro.encoder import KeyConditionedEncryptor
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=32, architecture="contextual_residual_v1")
    h = torch.randn(2, 4, 8)
    key = torch.randn(2, 4)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    z = model(h, key, attention_mask=mask)
    assert z.shape == h.shape
    assert torch.isfinite(z).all()


def test_orthogonal_direction_alpha_scale_is_runtime_configurable():
    from osnip_repro.encoder import KeyConditionedEncryptor
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=2, architecture="orthogonal_direction_v1", alpha_init=1.0)
    h = torch.randn(2, 3, 8)
    key = torch.randn(2, 4)
    model.set_alpha_scale(0.0)
    z = model(h, key)
    assert torch.allclose(z, h)
    model.set_alpha_scale(1.0)
    z2 = model(h, key)
    assert not torch.allclose(z2, h)


def test_orthogonal_direction_starts_near_orthogonalized_direction():
    from osnip_repro.encoder import KeyConditionedEncryptor
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=2, architecture="orthogonal_direction_v1", alpha_init=0.5)
    h = torch.randn(2, 3, 8)
    key = torch.randn(2, 4)
    z = model(h, key)
    cosine = torch.nn.functional.cosine_similarity(h, z, dim=-1)
    assert z.shape == h.shape
    assert torch.isfinite(z).all()
    assert cosine.mean() > 0.85


def test_tokenwise_keys_are_supported_and_change_token_outputs():
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=12, architecture="tokenwise_key_v1")
    h = torch.randn(2, 4, 8)
    shared = torch.randn(2, 4)
    tokenwise = shared[:, None, :].expand(2, 4, 4).clone()
    tokenwise[:, 1] = -tokenwise[:, 1]
    z_shared = model(h, shared)
    z_tokenwise = model(h, tokenwise)
    assert torch.allclose(z_shared[:, (0, 2, 3)], z_tokenwise[:, (0, 2, 3)])
    assert not torch.allclose(z_shared[:, 1], z_tokenwise[:, 1])


def test_encryptor_config_roundtrip_selects_architecture():
    from osnip_repro.encoder import build_encryptor_from_config

    config = {"width": 8, "key_dim": 4, "hidden_dim": 12, "encryptor_architecture": "key_film_residual_v1"}
    model = build_encryptor_from_config(config)
    assert model.architecture == config["encryptor_architecture"]
    restored = build_encryptor_from_config(config)
    restored.load_state_dict(model.state_dict())
    h, key = torch.randn(2, 8), torch.randn(2, 4)
    assert torch.allclose(model(h, key), restored(h, key))


def test_two_key_loss_can_measure_raw_representation_diversity():
    clean = torch.randn(3, 5)
    h = torch.randn(3, 4)
    z1 = h.clone()
    z2 = h.clone()
    raw1 = h.clone()
    raw2 = h * 10.0
    projected = osnip_two_key_loss(clean, clean, clean, h, z1, z2, epsilon=0.3, lambda_priv=0.0, diversity_margin=1.0, lambda_div=1.0)
    raw = osnip_two_key_loss(clean, clean, clean, h, z1, z2, epsilon=0.3, lambda_priv=0.0, diversity_margin=1.0, lambda_div=1.0, diversity_z1=raw1, diversity_z2=raw2)
    assert projected["diversity"].item() > raw["diversity"].item()


def test_contextual_valid_outputs_ignore_masked_padding_but_use_context():
    torch.manual_seed(83)
    model = KeyConditionedEncryptor(8, key_dim=4, hidden_dim=32,
                                    architecture='contextual_residual_v1').eval()
    h, key = torch.randn(1, 4, 8), torch.randn(1, 4)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    baseline = model(h, key, attention_mask=mask)
    changed_pad = h.clone()
    changed_pad[:, 3] = torch.randn(8) * 100
    torch.testing.assert_close(model(changed_pad, key, attention_mask=mask)[:, :3], baseline[:, :3])
    changed_context = h.clone()
    changed_context[:, 1] = torch.randn(8) * 10
    assert not torch.allclose(model(changed_context, key, attention_mask=mask)[:, 0], baseline[:, 0], atol=1e-7)
