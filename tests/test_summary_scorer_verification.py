import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scorer_verification',ROOT/'scripts/verify_summary_scorer.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


def test_serial_scorer_wait_checks_actual_trainer_identity_and_output():
    expected=ROOT/'reports/llama32_1b/question_disjoint_v2_pilot'
    scorer.validate_training_identity(f'python {ROOT}/scripts/train_streaming.py --output-dir {expected}',expected)
    with pytest.raises(ValueError,match='not the expected trainer'):
        scorer.validate_training_identity(f'python evaluate_no_key.py --output-dir {expected}',expected)
    with pytest.raises(ValueError,match='output path differs'):
        scorer.validate_training_identity('python scripts/train_streaming.py --output-dir reports/other',expected)
    with pytest.raises(ValueError,match='no output path'):
        scorer.validate_training_identity('python scripts/train_streaming.py',expected)


def test_scorer_monitor_exposes_portable_process_inspection():
    import os
    from scripts import verify_summary_scorer as scorer
    assert scorer.process_command(os.getpid())
    assert scorer.command_tokens(r"python C:\a\b.py --x 1")[1] == r"C:\a\b.py"
