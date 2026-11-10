"""Lock the reproduction recipe to PMLR 306 Appendix D.6 / Table 9.

The published paper gives BS/GPU 64, 8 epochs, 65,151 samples and Figure 8
plots ~2,000 optimizer steps for the 1B/3B models. Only four data-parallel
GPUs reconcile those numbers: 64*4 = 256 effective batch, and
8*65151/256 = 2036 steps. The package has 65,000 training rows, so its exact
8-epoch exposure is 8*65000/256 = 2032 steps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def test_effective_batch_and_steps_match_table9_figure_axis():
    data = 65151
    epochs = 8
    steps_paper = data * epochs / (64 * 4)
    assert math.ceil(steps_paper) == 2036
    assert 1900 <= steps_paper <= 2100  # Figure 8a/8b x-axis ends at 2,000
    steps_package = 65000 * epochs / 256
    assert math.ceil(steps_package) == 2032


def test_llama32_3b_formal_config_is_paper_conformant():
    cfg = load("llama32_3b_instruct_reproduction.json")
    obj = cfg["objective"]
    tr = cfg["training"]
    assert tr["epochs"] == 8
    assert tr["learning_rate"] == 1e-5
    assert tr["batch_size"] * tr["gradient_accumulation_steps"] == 256
    assert tr["effective_batch"] == 256
    assert tr["steps"] == 2032
    assert tr["seed"] == 42
    assert tr["warmup_steps"] == 200
    assert obj["constraint_ramp_steps"] == 1000
    assert obj["loss_warmup_steps"] == 10
    assert obj["tau_low"] == 0.05 and obj["tau_high"] == 0.15
    assert obj["lambda_priv"] == 1.0 and obj["lambda_div"] == 2.0
    assert obj["epsilon"] < 0.4
    assert obj["diversity_margin"] > 0.5
    assert obj["diversity_space"] == "raw_r"


def test_llama32_1b_v2_formal_block_is_paper_conformant():
    cfg = load("llama32_1b_question_disjoint_v2.json")
    tr = cfg["training"]
    assert tr["learning_rate"] == 1e-5
    assert tr["batch_size"] * tr["gradient_accumulation_steps"] == 256
    assert tr["formal_steps"] == 2032
    assert tr["seeds"] == [42]
    assert tr["warmup_steps"] == 200
    assert tr["constraint_ramp_steps"] == 1000
    assert tr["loss_warmup_steps"] == 10
    assert tr["tau_low"] == 0.05 and tr["tau_high"] == 0.15
    assert tr["lambda_priv"] == 1.0 and tr["lambda_div"] == 2.0
    assert cfg["objective"]["epsilon"] < 0.4
    assert cfg["objective"]["diversity_margin"] > 0.4
    assert tr["diversity_space"] == "raw_r"
    assert cfg["objective"]["diversity_space"] == "raw_r"


def test_paper_literal_3b_config_uses_65151_rows_and_2036_steps():
    cfg = load("llama32_3b_instruct_paper_literal.json")
    assert cfg["training_rows"] == 65151
    assert cfg["training"]["steps"] == 2036
    assert cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"] == 256
    assert cfg["training"]["learning_rate"] == 1e-5
    assert cfg["objective"]["tau_high"] == 0.15
    assert cfg["objective"]["diversity_space"] == "raw_r"
    from scripts.run_reproduction_3b import validate_recipe
    validate_recipe(cfg)
