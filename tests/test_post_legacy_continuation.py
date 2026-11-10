import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('post_legacy', ROOT / 'scripts/continue_after_legacy_run.py')
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def test_continuation_requires_exact_successful_legacy_run(tmp_path):
    report_path = tmp_path / 'report.json'
    expected = tmp_path / 'run'
    expected.mkdir()
    (expected / 'last.pt').write_bytes(b'checkpoint')
    report = {'status': 'completed_with_limits', 'steps_completed': 10000,
              'artifacts': {'last.pt': str(expected / 'last.pt')}}
    report_path.write_text(json.dumps(report))
    assert continuation.validate_run(report_path, expected_output=expected)['steps_completed'] == 10000
    report['steps_completed'] = 9999
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='exactly 10,000'):
        continuation.validate_run(report_path, expected_output=expected)


def test_continuation_rejects_missing_report(tmp_path):
    with pytest.raises(RuntimeError, match='without final report'):
        continuation.validate_run(tmp_path / 'missing.json')
