import importlib.util
import json
import torch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_evaluation', ROOT / 'scripts/run_question_disjoint_evaluation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    fixed_path = ROOT / 'configs/llama32_1b_question_disjoint_v2.json'
    manifest_path = ROOT / 'data/osnip_65k_question_disjoint_v2/data_manifest.json'
    fixed = json.loads(fixed_path.read_text())
    config = {'train_sha256': fixed['data']['train_sha256'], 'valid_sha256': fixed['data']['valid_sha256'],
              'diversity_space': 'raw_r', 'steps': fixed['training']['formal_steps'], 'resume': None,
              'seed': 20261019, 'encryptor_architecture': 'mlp_residual_v1',
              'model_path': str((ROOT / 'model/Llama-3.2-1B').resolve())}
    from osnip_repro.checkpointing import flatten_recipe
    config.update({name: value for name, value in flatten_recipe(fixed).items()
                   if name not in config})
    (run / 'config.json').write_text(json.dumps(config))
    report = {'status': 'completed_with_limits', 'security_accepted': False,
              'steps_completed': fixed['training']['formal_steps'], 'resumed_from': None,
              'model_path': str((ROOT / 'model/Llama-3.2-1B').resolve())}
    (run / 'report.json').write_text(json.dumps(report))
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': config['steps'], 'config': config}, run / 'last.pt')
    return run, fixed_path, manifest_path


requires_fixed_snapshots = pytest.mark.skipif(
    any(not (ROOT / path).is_file() for path in (
        'data/osnip_65k_question_disjoint_v2/data_manifest.json',
        'data/osnip_65k_question_disjoint_v2/train.jsonl',
        'data/osnip_65k_question_disjoint_v2/valid.jsonl',
        'data/osnip_65k_question_disjoint_v2/test.jsonl',
        'data/osnip_sources/cnn_dailymail_test.manifest.json',
        'data/osnip_sources/cnn_dailymail_test.jsonl',
    )),
    reason='requires the pinned question-disjoint and summarization snapshots',
)


@requires_fixed_snapshots
def test_v2_evaluation_binds_new_test_and_full_formal_attacks(tmp_path):
    run, config, manifest = fixture_run(tmp_path)
    plan = evaluation.build_plan(model_path=ROOT / 'model/Llama-3.2-1B', run_dir=run,
                                 config_path=config, manifest_path=manifest)
    command = plan['command']
    assert command[command.index('--test-jsonl') + 1] == str(ROOT / 'data/osnip_65k_question_disjoint_v2/test.jsonl')
    assert command[command.index('--knn-jsonl') + 1] == str(ROOT / 'data/paper_attack_sets/knn.jsonl')
    assert command[command.index('--vma-jsonl') + 1] == str(ROOT / 'data/paper_attack_sets/vma.jsonl')
    assert plan['full_vocabulary'] and plan['knn_samples'] == 1000 and plan['vma_prompts'] == 200
    assert plan['confirmatory_sciq_used'] is False and plan['key_attacks_performed'] is False
    assert command[command.index('--summarization-jsonl') + 1] == str(ROOT / 'data/osnip_sources/cnn_dailymail_test.jsonl')
    assert command[command.index('--max-summarization-samples') + 1] == '999'
    assert '--require-bertscore' in command and '--skip-bertscore' not in command
    assert command[command.index('--utility-batch-size') + 1] == '1'
    assert plan['summarization']['records'] == 999


@requires_fixed_snapshots
def test_v2_evaluation_refuses_legacy_data_run_and_existing_artifacts(tmp_path):
    run, config_path, manifest = fixture_run(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    config['train_sha256'] = '0' * 64
    (run / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='not trained on the pinned v2'):
        evaluation.build_plan(model_path=ROOT / 'model/Llama-3.2-1B', run_dir=run,
                              config_path=config_path, manifest_path=manifest)
    config['train_sha256'] = json.loads(config_path.read_text())['data']['train_sha256']
    (run / 'config.json').write_text(json.dumps(config))
    (run / 'selected_pareto.json').write_text('{}')
    with pytest.raises(FileExistsError, match='evaluation artifacts already exist'):
        evaluation.build_plan(model_path=ROOT / 'model/Llama-3.2-1B', run_dir=run,
                              config_path=config_path, manifest_path=manifest)


@pytest.mark.parametrize('config_patch,report_patch,match', [
    ({'steps': 1000}, {}, 'exactly '),
    ({'resume': 'legacy/last.pt'}, {'resumed_from': 'legacy/last.pt'}, 'ancestry disagree'),
])
@requires_fixed_snapshots
def test_v2_evaluation_rejects_pilot_or_resumed_runs(tmp_path, config_patch, report_patch, match):
    run, config_path, manifest = fixture_run(tmp_path)
    config = json.loads((run / 'config.json').read_text()); config.update(config_patch)
    (run / 'config.json').write_text(json.dumps(config))
    report = json.loads((run / 'report.json').read_text()); report.update(report_patch)
    if config_patch.get('steps'):
        report['steps_completed'] = config_patch['steps']
    (run / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        evaluation.build_plan(model_path=ROOT / 'model/Llama-3.2-1B', run_dir=run,
                              config_path=config_path, manifest_path=manifest)


def test_v2_lineage_accepts_same_v2_two_phase_but_rejects_legacy_and_cycles(tmp_path):
    run, config_path, manifest = fixture_run(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    training = json.loads((run / 'report.json').read_text())
    fixed = json.loads(config_path.read_text())
    model = ROOT / 'model/Llama-3.2-1B'
    source = tmp_path / 'phase1.pt'
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': 1500, 'config': config}, source)
    resumed = dict(config, resume=str(source.resolve()))
    training['resumed_from'] = resumed['resume']
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': config['steps'], 'config': resumed}, run / 'last.pt')
    lineage = evaluation.verify_training_lineage(run, resumed, training, fixed=fixed, model_path=model)
    assert [item['global_step'] for item in lineage] == [config['steps'], 1500]
    assert lineage[-1]['resume'] is None
    legacy = dict(config, train_sha256='0' * 64)
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': 1500, 'config': legacy}, source)
    with pytest.raises(ValueError, match='different data'):
        evaluation.verify_training_lineage(run, resumed, training, fixed=fixed, model_path=model)
    cyclic = dict(config, resume=str(source.resolve()))
    torch.save({'format': 'osnip_training_checkpoint_v1', 'global_step': 1500, 'config': cyclic}, source)
    with pytest.raises(ValueError, match='cyclic'):
        evaluation.verify_training_lineage(run, resumed, training, fixed=fixed, model_path=model)
