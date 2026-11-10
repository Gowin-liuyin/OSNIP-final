"""Run the diagnostic report path with an exact small frozen model."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('no_key_reporting', ROOT / 'scripts/evaluate_no_key.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.mark.parametrize('skip_probe,flip_encoder,expected_status', [
    (True, False, 'utility_evaluated_privacy_not_run'),
    (True, True, 'utility_failed'),
    (False, False, 'privacy_probe_failed'),
])
def test_completed_report_distinguishes_skipped_attack_from_zero_recovery(
        tmp_path, monkeypatch, skip_probe, flip_encoder, expected_status):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}')
    checkpoint = tmp_path / 'encoder.pt'
    checkpoint.write_bytes(b'fixed diagnostic encoder')
    (tmp_path / 'config.json').write_text(json.dumps({
        'model_path': str(model_dir), 'width': 14, 'key_dim': 2,
    }))
    data = tmp_path / 'input.jsonl'
    data.write_text('{"text":"one fixed prompt"}\n')
    output = tmp_path / 'report.json'

    class Encoder(torch.nn.Module):
        def forward(self, h, key, attention_mask=None):
            return -h if flip_encoder else h

    class FrozenModel:
        embedding = torch.nn.Embedding.from_pretrained(torch.eye(14))

        def get_input_embeddings(self):
            return self.embedding

    model = FrozenModel()
    monkeypatch.setattr(evaluator, 'load_causal_lm', lambda *a, **kw: (None, model, torch.device('cpu')))
    monkeypatch.setattr(evaluator, 'build_encryptor_from_config', lambda config: Encoder())
    monkeypatch.setattr(evaluator, 'load_encoder_state_dict', lambda *a, **kw: {})
    monkeypatch.setattr(evaluator, 'tokenize_texts', lambda *a, **kw: {
        'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long),
    })
    monkeypatch.setattr(evaluator, 'forward_logits', lambda model, h, mask: h * 10)
    if skip_probe:
        def forbidden_attack(*args, **kwargs):
            pytest.fail('a skipped probe must never execute vocabulary retrieval')
        monkeypatch.setattr(evaluator, 'vocabulary_topk', forbidden_attack)
    argv = ['evaluate_no_key.py', '--model-path', str(model_dir), '--checkpoint', str(checkpoint),
            '--data-jsonl', str(data), '--output', str(output), '--max-mean-kl', '0.1']
    if skip_probe:
        argv.append('--skip-vocabulary-probe')
    monkeypatch.setattr(sys, 'argv', argv)
    evaluator.main()
    report = json.loads(output.read_text())
    assert report['status'] == expected_status
    assert report['utility']['gate']['passed'] is (not flip_encoder)
    assert report['security_accepted'] is False
    assert report['key_attack_performed'] is False
    assert report['vocabulary_probe_performed'] is (not skip_probe)
    assert report['causal_targets'] == 2
    if skip_probe:
        assert report['metrics'] is None and report['counts'] is None
        assert report['positive_controls_valid'] is None
        assert report['same_observations_for_utility_and_attack'] is None
        assert report['attack_scope'] == 'not_evaluated'
    else:
        assert report['positive_controls_valid'] is True
        assert report['same_observations_for_utility_and_attack'] is True
        assert report['metrics']['clean']['l2']['top1'] == 1.0
        assert report['metrics']['encrypted']['l2']['top1'] == 1.0
        assert not any('probe skipped' in note for note in report['limitations'])
