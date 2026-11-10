"""Regression for matrix flags being rejected or lost at the two-phase CLI."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('two_phase', ROOT / 'scripts/run_two_phase_diagnostic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_two_phase_cli_forwards_nondefault_parameters_in_both_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(ROOT / 'scripts/run_two_phase_diagnostic.py'),
        '--model-path', str(tmp_path / 'model'), '--train-jsonl', str(tmp_path / 'train.jsonl'),
        '--valid-jsonl', str(tmp_path / 'valid.jsonl'), '--output-dir', str(tmp_path / 'run'),
        '--seed', '123', '--key-dim', '17', '--hidden-dim', '513', '--tau-low', '.05', '--tau-high', '.2'])
    args = module.parse_args()
    source = tmp_path / 'phase1' / 'last.pt'
    for resume, weight in ((None, 0.), (source, 1.)):
        command = module.trainer_command(args, tmp_path / 'phase', 3000 if resume is None else 10000,
                                         resume=resume, lambda_util=weight)
        for flag, value in [('--key-dim', '17'), ('--hidden-dim', '513'), ('--tau-low', '0.05'), ('--tau-high', '0.2')]:
            assert command[command.index(flag) + 1] == value
        assert ('--resume' in command) == (resume is not None)
        if resume is not None:
            assert command[command.index('--resume') + 1] == str(source)
