import json
import sys
from unittest import mock

import pytest

from scripts.check_reproduction_readiness import main as readiness_main


def run_readiness(tmp_path, data_status):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    data = tmp_path / "data_manifest.json"
    payload = {"records_written": 65000}
    if data_status is not None:
        payload["status"] = data_status
    data.write_text(json.dumps(payload))
    attack = tmp_path / "attack_manifest.json"
    attack.write_text(json.dumps({"status": "ready", "records_written": 1000}))
    out = tmp_path / "out.json"
    argv = ["check_reproduction_readiness", "--model-path", str(model),
            "--data-manifest", str(data), "--attack-manifest", str(attack),
            "--summary-manifest", str(tmp_path / "missing.json"), "--output", str(out)]
    with mock.patch.object(sys, "argv", argv):
        readiness_main()
    return json.loads(out.read_text())


def test_question_disjoint_locked_status_passes_gate0(tmp_path):
    report = run_readiness(tmp_path, "ready_for_fresh_training_only")
    assert report["status"] == "ready_for_gate0"


def test_ready_status_passes_gate0(tmp_path):
    report = run_readiness(tmp_path, "ready")
    assert report["status"] == "ready_for_gate0"


def test_missing_status_is_a_blocker(tmp_path):
    report = run_readiness(tmp_path, None)
    assert report["status"] == "blocked"