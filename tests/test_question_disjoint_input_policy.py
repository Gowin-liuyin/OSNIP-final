"""Prevent legacy, substituted or incompletely pinned inputs entering v2."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import verify_question_disjoint_training as verifier

spec = importlib.util.spec_from_file_location('input_policy', SCRIPTS / 'validate_reproduction_inputs.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    attack_dir = tmp_path / 'attacks'
    data_dir.mkdir()
    attack_dir.mkdir()
    exclusions = {}
    for name in ('knn_reserved', 'knn', 'vma', 'sciq_confirmatory'):
        path = attack_dir / (name + '.jsonl')
        row = {'text': 'a sufficiently long reserved prompt'}
        path.write_text((json.dumps(row) + '\n') * (200 if name == 'vma' else 1000))
        exclusions[name] = {'path': str(path), 'file_sha256': policy.sha256(path)}
    manifest = {
        'status': 'ready_for_fresh_training_only',
        'exclusions': exclusions,
        'sources': {name: {'revision': name + '-revision'} for name in ('alpaca', 'arc_easy', 'arc_challenge', 'sciq')},
        'splits': {name: {'path': str(data_dir / (name + '.jsonl')), 'file_sha256': 'f' * 64} for name in ('train', 'valid', 'test')},
        'construction': {'general_fraction_actual': .7833230769230769},
    }
    config = {
        'data': {
            'dataset_policy': 'question_disjoint_v2', 'fresh_training_required': True,
            'manifest_sha256': 'a' * 64, 'raw_sources_manifest_sha256': 'b' * 64, 'attack_manifest_sha256': 'c' * 64,
            'exclusion_sha256': {name: spec['file_sha256'] for name, spec in exclusions.items()},
            'source_revisions': {name: spec['revision'] for name, spec in manifest['sources'].items()},
            'knn': {'min_chars': 20, 'max_chars': 512},
            'general_fraction': manifest['construction']['general_fraction_actual'], 'general_fraction_is_paper_exact': False,
            'science_fraction': 1.-manifest['construction']['general_fraction_actual'],
            **{name + '_sha256': spec['file_sha256'] for name, spec in manifest['splits'].items()},
        },
        'objective': {'diversity_space': 'raw_r'},
        'training': {'diversity_space': 'raw_r'},
    }
    # These tests isolate the configuration binding; raw-content verification has
    # independent real-source/tampering tests in test_question_disjoint_training.
    calls = []
    monkeypatch.setattr(verifier, 'verify_dataset', lambda m: calls.append(m) or {'verified': True})
    return config, manifest, {'attack_dir': attack_dir, 'data_dir': data_dir}, calls


def test_policy_accepts_pinned_inputs_and_invokes_raw_verifier(inputs):
    config, manifest, paths, calls = inputs
    assert policy.validate_question_disjoint_policy(config, manifest, **paths) == {'verified': True}
    assert calls == [manifest]


@pytest.mark.parametrize('mutation,match', [
    (lambda c, m: c['data'].pop('dataset_policy'), 'explicit question_disjoint_v2'),
    (lambda c, m: c['data'].update(fresh_training_required=False), 'fresh_training_required'),
    (lambda c, m: m.update(status='ready'), 'not ready for fresh'),
    (lambda c, m: c['data'].pop('manifest_sha256'), 'fingerprint is not pinned'),
    (lambda c, m: c['data']['exclusion_sha256'].pop('sciq_confirmatory'), 'all four'),
    (lambda c, m: c['data']['exclusion_sha256'].update(knn='0' * 64), 'fingerprint mismatch'),
    (lambda c, m: m['exclusions']['knn'].update(path='/other/knn.jsonl'), 'evaluated attack file'),
    (lambda c, m: c['objective'].update(diversity_space='projected_z'), 'raw_r'),
    (lambda c, m: c['training'].update(diversity_space='projected_z'), 'raw_r'),
    (lambda c, m: c['data']['source_revisions'].pop('sciq'), 'source revision'),
    (lambda c, m: m['sources']['sciq'].update(revision='changed'), 'source revision'),
    (lambda c, m: c['data'].pop('test_sha256'), 'split fingerprint'),
    (lambda c, m: m['splits']['train'].update(path='/other/train.jsonl'), 'split path differs'),
    (lambda c, m: m['splits'].pop('test'), 'exactly train/valid/test'),
    (lambda c, m: c['data'].update(general_fraction=.8), 'actual general fraction'),
    (lambda c, m: c['data'].update(general_fraction_is_paper_exact=True), 'paper-ratio deviation'),
    (lambda c, m: c['data'].update(science_fraction=.2), 'actual non-Alpaca fraction'),
])
def test_policy_rejects_incompatible_or_unpinned_inputs_before_raw_verification(inputs, mutation, match):
    config, manifest, paths, calls = inputs
    mutation(config, manifest)
    with pytest.raises(ValueError, match=match):
        policy.validate_question_disjoint_policy(config, manifest, **paths)
    assert calls == []


def test_policy_checks_actual_exclusion_count_and_character_filter(inputs):
    config, manifest, paths, calls = inputs
    path = Path(manifest['exclusions']['knn']['path'])
    path.write_text(json.dumps({'text': 'a sufficiently long reserved prompt'}) + '\n')
    with pytest.raises(ValueError, match='wrong record count'):
        policy.validate_question_disjoint_policy(config, manifest, **paths)
    path.write_text((json.dumps({'text': 'short'}) + '\n') * 1000)
    with pytest.raises(ValueError, match='character filter'):
        policy.validate_question_disjoint_policy(config, manifest, **paths)
    assert calls == []
