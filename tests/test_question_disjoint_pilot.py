import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_pilot', ROOT / 'scripts/run_question_disjoint_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_v2_pilot_command_is_explicit_fresh_projected_z_without_resume(tmp_path):
    config_path = ROOT / 'configs/llama32_1b_question_disjoint_v2.json'
    manifest = ROOT / 'data/osnip_65k_question_disjoint_v2/data_manifest.json'
    plan = pilot.build_plan(model_path=ROOT / 'model/Llama-3.2-1B', config_path=config_path,
                            manifest_path=manifest, output_dir=tmp_path / 'run')
    command = plan['command']
    assert str(ROOT / 'data/osnip_65k_question_disjoint_v2/train.jsonl') == command[command.index('--train-jsonl') + 1]
    assert str(ROOT / 'data/osnip_65k_question_disjoint_v2/valid.jsonl') == command[command.index('--valid-jsonl') + 1]
    assert '--resume' not in command and '--reset-optimizer' not in command
    assert command[command.index('--diversity-space') + 1] == 'projected_z'
    assert plan['initialization'] == 'fresh_encoder_no_resume'
    assert '--data-manifest' in plan['gate0_command']
    assert '--train-jsonl' in plan['gate1_command'] and '--valid-jsonl' in plan['gate1_command']
    assert plan['gate0_command'][plan['gate0_command'].index('--config') + 1] == str(config_path)
    assert plan['security_accepted'] is False


def test_v2_pilot_rejects_manifest_substitution(tmp_path):
    with pytest.raises(ValueError, match='manifest differ'):
        pilot.build_plan(model_path=ROOT / 'model/Llama-3.2-1B',
                         config_path=ROOT / 'configs/llama32_1b_question_disjoint_v2.json',
                         manifest_path=tmp_path / 'other.json', output_dir=tmp_path / 'run')


def test_v2_pilot_evaluates_frozen_geometry_control_without_resuming_it(tmp_path):
    plan=pilot.build_plan(model_path=ROOT/'model/Llama-3.2-1B',config_path=ROOT/'configs/llama32_1b_question_disjoint_v2.json',
                          manifest_path=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json',output_dir=tmp_path/'pilot')
    utility=plan['pre_pilot_utility_command']
    assert utility[utility.index('--checkpoint')+1]==plan['pre_pilot_utility_control']['checkpoint']
    assert '--skip-vocabulary-probe' in utility
    assert utility[utility.index('--max-mean-kl')+1]=='0.1'
    assert '--resume' not in plan['command']
    assert plan['pre_pilot_utility_control']['utility_failure_blocks_pilot'] is False


def test_v2_control_preflight_rejects_changed_encoder_bytes(tmp_path):
    import hashlib
    checkpoint=tmp_path/'encoder.pt';checkpoint.write_bytes(b'frozen')
    config=tmp_path/'config.json';config.write_text('{}')
    data=tmp_path/'valid.jsonl';data.write_text('validation')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    control={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'checkpoint_config_sha256':sha(config),
             'data_jsonl':str(data),'data_sha256':sha(data)}
    pilot.verify_utility_control_inputs(control)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        pilot.verify_utility_control_inputs(control)


def test_three_seed_utility_controls_share_evaluation_protocol_and_stay_serial(tmp_path):
    plan=pilot.build_plan(model_path=ROOT/'model/Llama-3.2-1B',config_path=ROOT/'configs/llama32_1b_question_disjoint_v2.json',
                          manifest_path=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json',output_dir=tmp_path/'pilot')
    checks=plan['pre_pilot_utility_checks']
    assert len(checks)==3
    if any(not Path(check['control']['checkpoint']).is_file() for check in checks):
        pytest.skip("historical geometry-control checkpoints are not shipped in this package")
    outputs=[]
    for check in checks:
        pilot.verify_utility_control_inputs(check['control'])
        command=check['command']
        assert command[command.index('--checkpoint')+1]==check['control']['checkpoint']
        assert command[command.index('--seed')+1]=='20261021'
        outputs.append(command[command.index('--output')+1])
    assert len(set(outputs))==3
    assert '--resume' not in plan['command']
