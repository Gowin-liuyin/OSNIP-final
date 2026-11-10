import pytest

import torch

from osnip_repro.checkpointing import (
    RECIPE_FIELDS,
    flatten_recipe,
    load_encoder_state_dict,
    recipe_fingerprint,
)
from scripts.run_question_disjoint_evaluation import verify_training_lineage

RECIPE = {
    "learning_rate": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0, "gradient_balance": "none",
    "batch_size": 1, "gradient_accumulation_steps": 64, "warmup_steps": 200,
    "constraint_ramp_steps": 1000, "loss_warmup_steps": 0, "constraint_mode": "gated",
    "gate_scope": "effective_batch", "tau_low": 0.05, "tau_high": 0.3,
    "lambda_util": 1.0, "lambda_priv": 1.0, "lambda_div": 2.0,
    "epsilon": 0.3, "diversity_margin": 1.0, "diversity_space": "projected_z",
    "key_dim": 32, "hidden_dim": 256, "alpha_init": 0.05, "alpha_ramp_steps": 0,
    "encryptor_architecture": "mlp_residual_v1", "max_length": 128, "dtype": "bfloat16",
}


def run_config(**overrides):
    cfg = dict(RECIPE)
    cfg.update({
        "seed": 7,
        "train_sha256": "a" * 64,
        "valid_sha256": "b" * 64,
        "model_path": "/tmp/fake-model",
    })
    cfg.update(overrides)
    return cfg


def fixed_config(**recipe_overrides):
    recipe = dict(RECIPE)
    recipe.update(recipe_overrides)
    return {
        "data": {"train_sha256": "a" * 64, "valid_sha256": "b" * 64},
        "objective": {name: recipe[name] for name in
                      ("epsilon", "diversity_margin", "diversity_space",
                       "lambda_util", "lambda_priv", "lambda_div")},
        "training": {name: value for name, value in recipe.items() if name not in
                     ("epsilon", "diversity_margin", "diversity_space",
                      "lambda_util", "lambda_priv", "lambda_div")},
    }


def test_flatten_recipe_reads_nested_blocks_and_prefers_top_level():
    flat = flatten_recipe({"objective": {"epsilon": 0.1}, "training": {"grad_clip": 2.0},
                           "grad_clip": 1.0, "nested": {"ignored": True}})
    assert flat["epsilon"] == 0.1
    assert flat["grad_clip"] == 1.0
    assert "nested" not in flat


def test_fingerprint_tracks_recipe_fields_only():
    base = run_config()
    same = run_config(device="cuda", checkpoint_every=50)
    assert recipe_fingerprint(base) == recipe_fingerprint(same)
    assert recipe_fingerprint(base) != recipe_fingerprint(run_config(tau_high=0.2))
    missing = dict(base)
    missing.pop("grad_clip")
    assert recipe_fingerprint(base) != recipe_fingerprint(missing)


def test_encoder_loader_refuses_payload_with_different_recipe(tmp_path):
    good = run_config()
    payload = {"format": "osnip_training_checkpoint_v1", "global_step": 3,
               "encoder": {"w": torch.zeros(2)}, "config": good}
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)
    load_encoder_state_dict(str(path), expected_config=good)
    drifted = run_config(learning_rate=1e-5)
    try:
        load_encoder_state_dict(str(path), expected_config=drifted)
        raise AssertionError("recipe drift must be refused")
    except ValueError as exc:
        assert "recipe fingerprint mismatch" in str(exc)


def write_lineage(tmp_path, recorded):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"format": "osnip_training_checkpoint_v1", "global_step": 0,
                "config": recorded}, run_dir / "last.pt")
    return run_dir


def test_lineage_refuses_unlocked_pinned_config(tmp_path):
    fixed = fixed_config()
    fixed["training"].pop("gate_scope")
    cfg = run_config(model_path=str(tmp_path))
    run_dir = write_lineage(tmp_path, dict(cfg, resume=None))
    try:
        verify_training_lineage(run_dir, cfg, {"steps_completed": 0},
                                fixed=fixed, model_path=tmp_path)
        raise AssertionError("unlocked pinned config must be refused")
    except ValueError as exc:
        assert "does not lock recipe fields" in str(exc)


def test_lineage_refuses_run_config_drift_from_pinned(tmp_path):
    cfg = run_config(model_path=str(tmp_path), tau_high=0.15)
    run_dir = write_lineage(tmp_path, dict(cfg, resume=None))
    try:
        verify_training_lineage(run_dir, cfg, {"steps_completed": 0},
                                fixed=fixed_config(), model_path=tmp_path)
        raise AssertionError("drifted run recipe must be refused")
    except ValueError as exc:
        assert "differs from the pinned v2 recipe" in str(exc)


def test_lineage_refuses_checkpoint_from_other_recipe(tmp_path):
    cfg = run_config(model_path=str(tmp_path))
    recorded = run_config(model_path=str(tmp_path), gradient_accumulation_steps=8)
    run_dir = write_lineage(tmp_path, dict(recorded, resume=None))
    try:
        verify_training_lineage(run_dir, cfg, {"steps_completed": 0},
                                fixed=fixed_config(), model_path=tmp_path)
        raise AssertionError("foreign-recipe checkpoint must be refused")
    except ValueError as exc:
        assert "resume lineage recipe differs" in str(exc)


def test_lineage_accepts_consistent_fresh_run(tmp_path):
    cfg = run_config(model_path=str(tmp_path))
    run_dir = write_lineage(tmp_path, dict(cfg, resume=None))
    lineage = verify_training_lineage(run_dir, cfg, {"steps_completed": 0},
                                      fixed=fixed_config(), model_path=tmp_path)
    assert len(lineage) == 1
    assert RECIPE_FIELDS

def test_require_locked_recipe_refuses_placeholders():
    from osnip_repro.checkpointing import require_locked_recipe
    require_locked_recipe(fixed_config())
    unlocked = fixed_config()
    unlocked["training"]["learning_rate"] = "TUNE_ON_GATE_1_AND_FREEZE_BEFORE_MAIN_RUN"
    with pytest.raises(ValueError, match="unlocked recipe placeholders: learning_rate"):
        require_locked_recipe(unlocked)
    enums = fixed_config()
    enums["training"]["constraint_mode"] = "gated"
    enums["training"]["dtype"] = "bfloat16"
    require_locked_recipe(enums)
