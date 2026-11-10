"""The 3B launcher must emit the PMLR Table 9 recipe and refuse old values."""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

import scripts.run_reproduction_3b as launcher

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/llama32_3b_instruct_reproduction.json"


def config() -> dict:
    return json.loads(CONFIG.read_text())


def test_config_is_valid_and_emits_table9_command():
    cfg = config()
    launcher.validate_recipe(cfg)
    paths = launcher.resolve_paths(cfg, Path("/fake/model"), ROOT)
    command = launcher.trainer_command(cfg, paths, ROOT / "run")
    pairs = dict(zip(command[::2], command[1::2]))
    assert pairs["--steps"] == "2032"
    assert pairs["--gradient-accumulation-steps"] == "256"
    assert pairs["--learning-rate"] == "1e-05"
    assert pairs["--tau-high"] == "0.15" and pairs["--tau-low"] == "0.05"
    assert pairs["--loss-warmup-steps"] == "10"
    assert pairs["--diversity-space"] == "raw_r"
    assert pairs["--lambda-div"] == "2.0"
    assert pairs["--seed"] == "42"


def test_evaluation_uses_final_checkpoint_and_layer20():
    cfg = config()
    paths = launcher.resolve_paths(cfg, Path("/fake/model"), ROOT)
    command = launcher.evaluation_command(cfg, paths, ROOT / "run")
    pairs = dict(zip(command[::2], command[1::2]))
    assert pairs["--criterion"] == "last"
    assert pairs["--task-source-split"] == "test"
    assert Path(command[command.index("--task-test-jsonl") + 1]).name == "task_test.jsonl"
    assert pairs["--paper-clean-ppl"] == "9.63"
    assert "--skip-input-vocabulary-probe" in command
    assert command[command.index("--layers") + 1:command.index("--layers") + 6] == ["1", "5", "10", "15", "20"]
    assert pairs["--keys-per-prompt"] == "5"


@pytest.mark.parametrize("section,field,value", [
    ("training", "learning_rate", 1e-4),
    ("training", "steps", 8125),
    ("training", "gradient_accumulation_steps", 64),
    ("objective", "tau_high", 0.3),
    ("objective", "loss_warmup_steps", 0),
    ("objective", "diversity_space", "projected_z"),
    ("training", "seed", 43),
    ("objective", "lambda_div", 1.0),
    ("training", "warmup_steps", 100),
    ("objective", "constraint_ramp_steps", 500),
    ("objective", "gate_scope", "microbatch"),
    ("objective", "lambda_util", 0.5),
])
def test_old_recipe_values_are_refused(section, field, value):
    cfg = config()
    cfg[section][field] = value
    with pytest.raises(ValueError):
        launcher.validate_recipe(cfg)


def test_non_paper_budget_is_refused():
    cfg = config()
    cfg["training_rows"] = 1000
    cfg["training"].update(steps=1000, batch_size=1, gradient_accumulation_steps=1,
                           effective_batch=1, max_length=64)
    with pytest.raises(ValueError):
        launcher.validate_recipe(cfg)


def run_main(argv):
    with mock.patch.object(sys, "argv", ["run_reproduction_3b", *argv]):
        launcher.main()


def test_execute_accepts_its_own_dry_run_plan_and_resumes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append(command))
    base = ["--model-path", "/fake/model", "--output-dir", str(tmp_path)]
    run_main(base)
    run_main(base + ["--execute", "--skip-evaluation"])
    assert calls and "--resume" not in calls[0]
    (tmp_path / "train").mkdir(exist_ok=True)
    (tmp_path / "train" / "last.pt").write_bytes(b"checkpoint")
    run_main(base + ["--execute", "--skip-evaluation"])
    assert calls[-1][-2:] == ["--resume", str(tmp_path / "train" / "last.pt")]


def test_execute_refuses_a_plan_from_a_different_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: None)
    run_main(["--model-path", "/fake/model", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="differs"):
        run_main(["--model-path", "/other/model", "--output-dir", str(tmp_path),
                  "--execute", "--skip-evaluation"])


def test_execute_rejects_changed_model_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: None)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 3072, "intermediate_size": 8192,
        "num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8,
        "vocab_size": 128256}))
    (model / "tokenizer.json").write_text("first")
    base = ["--model-path", str(model), "--output-dir", str(tmp_path / "out")]
    run_main(base)
    (model / "tokenizer.json").write_text("second")
    with pytest.raises(ValueError, match="model files changed"):
        run_main(base + ["--execute", "--skip-evaluation"])
