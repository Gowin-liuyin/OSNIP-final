import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from scripts.run_evaluation_suite import main as suite_main


def run_plan(tmp_path, criterion="utility", extra_files=None):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    for name, payload in (extra_files or {}).items():
        (run / name).write_text(json.dumps(payload))
    argv = ["run_evaluation_suite", "--model-path", str(tmp_path / "model"),
            "--run-dir", str(run), "--criterion", criterion]
    with mock.patch.object(sys, "argv", argv):
        suite_main()
    return json.loads((run / "evaluation_suite_plan.json").read_text())


def test_dry_run_reuses_derived_full_row_knn_report(tmp_path):
    derived = {"format": "osnip_knn_report_v2", "rows": [{"index": 0, "positions": 2,
               "clean": {"top1": 2}, "encrypted": {"top1": 0}}]}
    plan = run_plan(tmp_path, extra_files={"paper_protocol_test.knn.json": derived})
    privacy = plan["commands"]["privacy"]
    assert "--knn-report" in privacy
    assert privacy[privacy.index("--knn-report") + 1].endswith("paper_protocol_test.knn.json")


def test_dry_run_ignores_rowless_v1_summary(tmp_path):
    plan = run_plan(tmp_path, extra_files={"knn_report.json": {"format": "osnip_knn_report_v1"}})
    assert "--knn-report" not in plan["commands"]["privacy"]


def test_dry_run_placeholder_follows_criterion_and_is_labelled(tmp_path):
    plan = run_plan(tmp_path, criterion="utility")
    assert plan["checkpoint"].endswith("best_utility.pt")
    assert plan["checkpoint_placeholder"] is True
    pareto = run_plan(tmp_path / "second", criterion="pareto")
    assert pareto["checkpoint"].endswith("best_pareto.pt")
    final = run_plan(tmp_path / "third", criterion="last")
    assert final["checkpoint"].endswith("last.pt")


def test_execute_rejects_a_selection_outside_the_run_directory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    stale = tmp_path / "old" / "last.pt"
    stale.parent.mkdir()
    stale.write_bytes(b"old")
    (run / "selected_pareto.json").write_text(json.dumps({"selected": {"path": str(stale)}}))
    argv = ["run_evaluation_suite", "--model-path", str(tmp_path / "model"),
            "--run-dir", str(run), "--criterion", "pareto", "--execute"]
    with mock.patch.object(sys, "argv", argv), pytest.raises(ValueError, match="outside"):
        suite_main()


def test_execute_rejects_existing_outputs_from_a_different_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "best_utility.pt"
    checkpoint.write_bytes(b"current")
    (run / "selected_utility.json").write_text(json.dumps({"selected": {"path": str(checkpoint)}}))
    (run / "utility_test.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    argv = ["run_evaluation_suite", "--model-path", str(tmp_path / "model"),
            "--run-dir", str(run), "--criterion", "utility", "--execute"]
    with mock.patch.object(sys, "argv", argv), pytest.raises(ValueError, match="different checkpoint"):
        suite_main()
