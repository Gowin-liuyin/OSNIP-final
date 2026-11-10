import json

import pytest
import torch

from scripts.select_checkpoint import main as select_main

ENCODER_NAMES = {
    "best_utility.pt": "best_utility_encoder.pt",
    "best_pareto.pt": "best_pareto_encoder.pt",
    "last.pt": "encryptor.pt",
}
RECIPE = {"learning_rate": 1e-4, "tau_high": 0.3, "epsilon": 0.3}


def write_run(tmp_path, *, candidates=("best_utility.pt", "last.pt"), with_encoder=True,
              fmt="osnip_training_checkpoint_v1", step_mismatch=False, metric=None,
              payload_recipe=None):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(RECIPE))
    for index, name in enumerate(candidates):
        step = 100 + index
        if with_encoder:
            torch.save({"w": torch.zeros(2)}, run / ENCODER_NAMES[name])
        validation = {"step": step + (1 if step_mismatch else 0),
                      "utility": 0.2 + 0.1 * index, "pareto_score": 0.6 - 0.1 * index,
                      "privacy": 0.1, "ppl_ratio": 1.1}
        if metric == "missing" and index == 0:
            validation.pop("utility")
        if metric == "nan" and index == 0:
            validation["utility"] = float("nan")
        torch.save({"format": fmt, "global_step": step, "metrics": {"validation": validation},
                    "config": dict(payload_recipe or RECIPE)}, run / name)
    return run


def run_selector(run, criterion="utility"):
    import sys
    from unittest import mock
    with mock.patch.object(sys, "argv", ["select_checkpoint", "--run-dir", str(run), "--criterion", criterion]):
        select_main()
    return json.loads((run / f"selected_{criterion}.json").read_text())


def test_selects_lowest_validation_metric(tmp_path):
    run = write_run(tmp_path)
    report = run_selector(run)
    assert report["selected"]["path"].endswith("best_utility.pt")
    assert report["candidates"][0]["encoder_state"].endswith("best_utility_encoder.pt")
    pareto = run_selector(run, "pareto")
    assert pareto["selected"]["path"].endswith("last.pt")


def test_last_criterion_selects_the_paper_final_checkpoint(tmp_path):
    run = write_run(tmp_path)
    report = run_selector(run, "last")
    assert report["selected"]["path"].endswith("last.pt")
    assert report["metric"] == "final_step"


def test_last_criterion_refuses_a_run_without_last_checkpoint(tmp_path):
    run = write_run(tmp_path, candidates=("best_utility.pt",))
    with pytest.raises(ValueError, match="last.pt is missing"):
        run_selector(run, "last")


def test_last_criterion_refuses_wrong_final_step(tmp_path):
    run = write_run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    config["steps"] = 999
    (run / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configured final step"):
        run_selector(run, "last")


def test_last_criterion_refuses_nonfinite_validation(tmp_path):
    run = write_run(tmp_path)
    payload = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    payload["metrics"]["validation"]["privacy"] = float("nan")
    torch.save(payload, run / "last.pt")
    with pytest.raises(ValueError, match="non-finite"):
        run_selector(run, "last")


@pytest.mark.parametrize("kwargs,match", [
    ({"with_encoder": False}, "missing paired encoder artifact"),
    ({"metric": "missing"}, "missing or non-finite"),
    ({"metric": "nan"}, "missing or non-finite"),
    ({"fmt": "other_format_v9"}, "unsupported checkpoint format"),
    ({"step_mismatch": True}, "refusing stale selection"),
    ({"payload_recipe": {"learning_rate": 1e-5}}, "embedded recipe differs"),
])
def test_refuses_incomplete_or_foreign_candidates(tmp_path, kwargs, match):
    run = write_run(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        run_selector(run)

def test_stale_candidate_is_skipped_while_fresh_one_is_selected(tmp_path):
    run = write_run(tmp_path)
    import torch as _torch
    payload = _torch.load(run / "best_utility.pt", map_location="cpu", weights_only=False)
    payload["metrics"]["validation"]["step"] += 7
    _torch.save(payload, run / "best_utility.pt")
    report = run_selector(run)
    assert report["selected"]["path"].endswith("last.pt")
    assert len(report["skipped_stale_candidates"]) == 1
    assert report["skipped_stale_candidates"][0].endswith("best_utility.pt")
