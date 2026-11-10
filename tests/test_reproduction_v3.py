import copy
import pytest
from scripts.run_reproduction_v3 import gate, verify_bound, write_json


def report():
    return {'status':'completed_with_limits','steps_executed_this_run':100,
        'baseline_validation':{'privacy':.7},
        'last_validation':{'utility':.1,'privacy':.6,'diversity':.2,'ppl_ratio':1.05,
            'projection_fp32_max_relative_error':1e-7,'norm_max_relative_error':1e-4}}


def test_continuation_gate_does_not_substitute_utility_for_privacy_progress():
    value=report()
    assert gate(value,value,require_privacy_progress=True)['passed']
    value['last_validation']['privacy']=.7
    assert not gate(value,value,require_privacy_progress=True)['passed']
    value['last_validation']['utility']=float('nan')
    assert not gate(value)['passed']
    assert gate(report())['security_accepted'] is False


def test_input_drift_fails_before_reproduction_launch(tmp_path, monkeypatch):
    import scripts.run_reproduction_v3 as runner
    from osnip_repro.data import file_sha256
    source=tmp_path/'source.py';source.write_text('original')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    plan={'bound_inputs':{'source.py':file_sha256(source)}}
    verify_bound(plan)
    source.write_text('changed')
    with pytest.raises(ValueError,match='input changed'):verify_bound(plan)


def test_status_writer_never_replaces_evidence_with_nonfinite_json(tmp_path):
    import json
    path=tmp_path/'status.json'
    write_json(path,{'state':'running','security_accepted':False})
    with pytest.raises(ValueError):write_json(path,{'metric':float('nan')})
    assert json.loads(path.read_text())['state']=='running'


def diagnostic_fixture(run_dir):
    import json
    from osnip_repro.data import file_sha256
    run_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=run_dir/'last.pt';checkpoint.write_bytes(b'fixed1100')
    valid=run_dir/'valid.jsonl';valid.write_text('{"text":"held out"}\n')
    config={'valid_jsonl':str(valid.resolve()),'valid_sha256':file_sha256(valid)}
    (run_dir/'config.json').write_text(json.dumps(config))
    return {'status':'completed','global_step':1100,'samples':64,
        'data_jsonl':str(valid.resolve()),'data_sha256':file_sha256(valid),
        'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':file_sha256(checkpoint),
        'full_vocabulary':True,'vocabulary_size':128256,'positive_controls_valid':True,
        'rates':{side:{f'top{k}':1.0 for k in (1,5,10)} for side in ('clean','encrypted')},
        'bound_input_sha256':{str(checkpoint):file_sha256(checkpoint),str(valid):file_sha256(valid)}}


def test_knn_checkpoint_gate_rejects_wrong_step_partial_vocab_and_checkpoint_drift(tmp_path):
    from scripts.run_reproduction_v3 import validation_knn_gate
    value=diagnostic_fixture(tmp_path)
    assert validation_knn_gate(value,tmp_path)['passed']
    assert validation_knn_gate(value,tmp_path)['security_accepted'] is False
    for key,bad in [('global_step',1150),('full_vocabulary',False),('samples',2),
                    ('data_sha256','wrong'),('positive_controls_valid',False)]:
        changed=copy.deepcopy(value);changed[key]=bad
        assert not validation_knn_gate(changed,tmp_path)['passed']
    (tmp_path/'last.pt').write_bytes(b'drift')
    assert not validation_knn_gate(value,tmp_path)['passed']


def test_runner_runs_knn_before_epoch8_and_holds_even_when_recovery_is_high(tmp_path,monkeypatch):
    import json
    import scripts.run_reproduction_v3 as runner
    stages=[{'name':name,'directory':str(tmp_path/name),'command':[name],
             'timeout_seconds':10} for name in ['pilot100','ramp1100','epoch8']]
    ramp=tmp_path/'ramp1100'
    diagnostic=diagnostic_fixture(ramp)
    calls=[]
    class Child:
        pid=123456
        def __init__(self,command,**kwargs):
            calls.append(command)
            if command[0] in ['pilot100','ramp1100','epoch8']:
                path=tmp_path/command[0];path.mkdir(exist_ok=True)
                runner.write_json(path/'report.json',report())
            else:
                assert 'validation_knn_checkpoint.py' in command[1]
                runner.write_json(ramp/'validation_knn_step1100.json',diagnostic)
        def poll(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    plan={'output':str(tmp_path),'python':'python','bound_inputs':{},
          'validation_knn_checkpoint':{'enabled':True,'timeout_seconds':10,'continue_after_review':False},
          'seeds':[{'seed':42,'stages':stages}]}
    runner.execute(plan)
    assert [c[0] for c in calls]==['pilot100','ramp1100','python']
    status=json.loads((tmp_path/'execution_status.json').read_text())
    assert status['state']=='held_after_validation_knn'
    assert status['security_accepted'] is False
    assert status['automatic_continuation_enabled'] is False
    assert not (tmp_path/'epoch8').exists()


def test_old_recipe_cannot_silently_skip_new_checkpoint(tmp_path):
    from scripts.run_reproduction_v3 import execute
    with pytest.raises(ValueError,match='explicitly bind'):
        execute({'output':str(tmp_path)})
    assert not (tmp_path/'execution_status.json').exists()


def test_v3_recipe_flags_match_relocked_values():
    from pathlib import Path
    from scripts.run_reproduction_v3 import recipe_common_flags
    flags = recipe_common_flags(Path('/m'), Path('/d'))
    pairs = dict(zip(flags[::2], flags[1::2]))
    assert pairs['--learning-rate'] == '1e-5'
    assert pairs['--tau-high'] == '.15' and pairs['--tau-low'] == '.05'
    assert pairs['--loss-warmup-steps'] == '10'
    assert pairs['--max-valid-samples'] == '256'
    assert pairs['--dtype'] == 'bfloat16'
    assert pairs['--diversity-space'] == 'raw_r'
    assert pairs['--gradient-accumulation-steps'] == '256' and pairs['--gate-scope'] == 'effective_batch'
