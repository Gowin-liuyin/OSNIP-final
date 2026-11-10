from types import SimpleNamespace

import pytest

from scripts.train_streaming import RESUME_DEFAULTS, RESUME_IMMUTABLE, validate_resume_recipe


def base_args(**overrides):
    values = {name: RESUME_DEFAULTS.get(name) for name in RESUME_IMMUTABLE}
    values.update({
        "seed": 42, "max_length": 128, "batch_size": 1, "gradient_accumulation_steps": 64,
        "max_train_samples": None, "lambda_util": 1.0, "lambda_priv": 1.0, "lambda_div": 2.0,
        "epsilon": 0.3, "diversity_margin": 1.0, "diversity_space": "projected_z",
        "constraint_mode": "gated", "dtype": "bfloat16", "gate_scope": "effective_batch",
        "loss_warmup_steps": 10, "constraint_ramp_steps": 1000, "warmup_steps": 200,
        "tau_low": 0.05, "tau_high": 0.3, "learning_rate": 1e-4, "weight_decay": 0.0,
        "grad_clip": 1.0, "gradient_balance": "none", "key_dim": 32, "hidden_dim": 256,
        "alpha_init": 0.05, "alpha_ramp_steps": 0, "freeze_alpha": False,
        "encryptor_architecture": "mlp_residual_v1", "max_valid_samples": 256,
        "lambda_knn": 0.0, "knn_negative_count": 64, "knn_topk": 10, "knn_margin": 0.05,
        "lambda_hidden": 0.0, "hidden_layers": [],
    })
    values.update(overrides)
    return SimpleNamespace(**values)


def prior_from(args):
    return {name: getattr(args, name) for name in RESUME_IMMUTABLE}


def test_identical_recipe_resumes():
    args = base_args()
    validate_resume_recipe(prior_from(args), args)


@pytest.mark.parametrize("field,value", [
    ("learning_rate", 1e-5),
    ("weight_decay", 0.01),
    ("grad_clip", 0.5),
    ("gradient_balance", "component_norm"),
    ("key_dim", 64),
    ("hidden_dim", 512),
    ("alpha_init", 0.1),
    ("encryptor_architecture", "low_rank_residual_v1"),
    ("max_valid_samples", 512),
    ("hidden_layers", [1, 5]),
    ("tau_low", 0.01),
    ("tau_high", 0.2),
    ("loss_warmup_steps", 0),
    ("constraint_ramp_steps", 500),
    ("batch_size", 2),
    ("gradient_accumulation_steps", 128),
    ("gate_scope", "microbatch"),
    ("diversity_space", "raw_r"),
])
def test_optimizer_and_structure_fields_are_immutable(field, value):
    args = base_args()
    prior = prior_from(args)
    prior[field] = value
    with pytest.raises(ValueError, match=f"resume recipe changed: {field}"):
        validate_resume_recipe(prior, args)


def test_missing_legacy_fields_fall_back_to_defaults():
    args = base_args(constraint_ramp_steps=None, gate_scope="microbatch", loss_warmup_steps=10)
    prior = prior_from(args)
    for name in RESUME_DEFAULTS:
        prior.pop(name)
    validate_resume_recipe(prior, args)


def test_missing_legacy_field_that_differs_from_default_is_refused():
    args = base_args(constraint_ramp_steps=1000)
    prior = prior_from(args)
    prior.pop("constraint_ramp_steps")
    with pytest.raises(ValueError, match="resume recipe changed: constraint_ramp_steps"):
        validate_resume_recipe(prior, args)

def test_phase_transition_staging_records_lambda_change(tmp_path):
    import torch
    from scripts.run_two_phase_diagnostic import stage_phase_transition
    source = tmp_path / "phase1" / "last.pt"
    source.parent.mkdir()
    base = base_args(lambda_util=0.0)
    config = {name: getattr(base, name) for name in RESUME_IMMUTABLE}
    config["model_path"] = "/tmp/fake"
    torch.save({"format": "osnip_training_checkpoint_v1", "global_step": 5,
                "encoder": {"w": torch.zeros(2)}, "config": config}, source)
    staged = stage_phase_transition(source, tmp_path / "stage", lambda_util=1.0)
    payload = torch.load(staged, map_location="cpu", weights_only=False)
    assert payload["config"]["lambda_util"] == 1.0
    assert payload["config"]["phase_transition"]["changed_fields"] == ["lambda_util"]
    assert torch.equal(payload["encoder"]["w"], torch.zeros(2))
    validate_resume_recipe(payload["config"], base_args(lambda_util=1.0))
    with pytest.raises(ValueError, match="resume recipe changed: lambda_util"):
        validate_resume_recipe(payload["config"], base_args(lambda_util=0.0))
