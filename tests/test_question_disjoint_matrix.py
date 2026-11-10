import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v2_matrix',ROOT/'scripts/run_question_disjoint_matrix.py')
matrix=importlib.util.module_from_spec(spec); spec.loader.exec_module(matrix)


def test_v2_matrix_includes_full_matched_controls(tmp_path):
    args=matrix.parse_args
    class A: pass
    a=A(); a.model_path=ROOT/'model/Llama-3.2-1B'; a.config=ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json'; a.data_manifest=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json'; a.output_root=tmp_path/'matrix'; a.pilot_report=tmp_path/'pilot.json'; a.execute=False
    plan=matrix.build_plan(a)
    assert plan['command_count']==60
    assert len(plan['requested_architectures'])==5 and plan['unique_architecture_count']==4
    assert plan['architecture_aliases']=={'tokenwise_key_v1':'mlp_residual_v1'}
    assert 'tokenwise_key_v1' not in plan['architectures']
    assert plan['variants']==['utility_only','privacy_only','gated_default','constant_balanced','two_phase_diagnostic']
    gate0=plan['gate0_command']
    assert gate0[gate0.index('--config')+1]==str(a.config)
    assert plan['fresh_initialization_required'] and not plan['resume_allowed']
    assert all('--resume' not in item['command'] for item in plan['commands'] if item['variant'] != 'two_phase_diagnostic')
    for item in plan['commands']:
        cmd=item['command']
        assert str(ROOT/'data/osnip_65k_question_disjoint_v2/train.jsonl') == cmd[cmd.index('--train-jsonl')+1]
        assert str(ROOT/'data/osnip_65k_question_disjoint_v2/valid.jsonl') == cmd[cmd.index('--valid-jsonl')+1]
        assert cmd[cmd.index('--device')+1]=='mps'
        assert cmd[cmd.index('--dtype')+1]=='float32'
        if item['variant'] in ('utility_only','privacy_only'):
            weights=tuple(float(cmd[cmd.index(flag)+1]) for flag in ('--lambda-util','--lambda-priv','--lambda-div'))
            assert weights==((1.,0.,0.) if item['variant']=='utility_only' else (0.,1.,1.))
            assert cmd[cmd.index('--constraint-mode')+1]==('off' if item['variant']=='utility_only' else 'constant')
            assert cmd[cmd.index('--steps')+1]=='10000'
            assert cmd[cmd.index('--diversity-space')+1]=='projected_z'


def test_v2_matrix_rejects_execution_before_hyperparameter_lock(tmp_path):
    class A: pass
    a=A(); a.model_path=ROOT/'model/Llama-3.2-1B'; a.config=ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json'; a.data_manifest=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json'; a.output_root=tmp_path/'matrix'; a.pilot_report=tmp_path/'pilot.json'; a.execute=True
    plan=matrix.build_plan(a)
    cfg=json.loads(a.config.read_text())
    assert cfg['formal_training_defaults']['status']=='provisional_until_v2_gate1_passes'
    assert plan['status']=='plan_only_until_gate1_lock'


@pytest.mark.parametrize('patch,match',[
    ({'status':'failed'},'incomplete'),
    ({'steps_completed':999},'incomplete'),
    ({'security_accepted':True},'unexpected status'),
])
def test_matrix_rejects_partial_existing_report(tmp_path,patch,match):
    run=tmp_path/'run';run.mkdir()
    (run/'config.json').write_text('{}')
    report={'status':'completed_with_limits','steps_completed':10000,'security_accepted':False}
    report.update(patch)
    (run/'report.json').write_text(json.dumps(report))
    command=['python','train_streaming.py','--output-dir',str(run),'--steps','10000']
    with pytest.raises(ValueError,match=match):
        matrix.verify_completed_command(command,{})


def test_matrix_verifies_settings_and_fresh_checkpoint_before_skip(tmp_path):
    import torch
    fixed=json.loads((ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json').read_text())
    class A: pass
    a=A();a.model_path=ROOT/'model/Llama-3.2-1B';a.config=ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json';a.data_manifest=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json';a.output_root=tmp_path/'matrix';a.pilot_report=tmp_path/'pilot.json';a.execute=False
    item=matrix.build_plan(a)['commands'][0]
    command=matrix.trainer_commands(item)[0]
    options=dict(zip(command[2::2],command[3::2]))
    config={}
    for flag,value in options.items():
        try: value=json.loads(value)
        except json.JSONDecodeError: pass
        config[flag[2:].replace('-','_')]=value
    config.update(resume=None,train_sha256=fixed['data']['train_sha256'],valid_sha256=fixed['data']['valid_sha256'])
    run=Path(item['output_dir']);run.mkdir(parents=True)
    report={'status':'completed_with_limits','steps_completed':10000,'security_accepted':False,'resumed_from':None}
    (run/'config.json').write_text(json.dumps(config));(run/'report.json').write_text(json.dumps(report))
    torch.save({'format':'osnip_training_checkpoint_v1','global_step':10000,'config':config},run/'last.pt')
    assert len(matrix.verify_completed_command(command,fixed))==1
    config['seed']+=1
    (run/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='setting differs: seed'):
        matrix.verify_completed_command(command,fixed)


def test_matrix_expands_two_phase_with_correct_weights_and_origin(tmp_path):
    item={'variant':'two_phase_diagnostic','output_dir':str(tmp_path/'run'),
          'command':['python','wrapper','--execute','--model-path','model','--output-dir',str(tmp_path/'run'),
                     '--phase1-steps','3000','--total-steps','10000','--seed','17']}
    phase1,phase2=matrix.trainer_commands(item)
    a=dict(zip(phase1[2::2],phase1[3::2]));b=dict(zip(phase2[2::2],phase2[3::2]))
    assert a['--steps']=='3000' and b['--steps']=='10000'
    assert a['--lambda-util']=='0.0' and b['--lambda-util']=='1.0'
    assert '--resume' not in a
    # A23: phase 2 resumes from the staged transition checkpoint, not phase 1.
    assert b['--resume']==str(Path(a['--output-dir']).parent/'phase2_start'/'start.pt')
    assert a['--seed']==b['--seed']=='17'


def test_matrix_pilot_prerequisite_checks_actual_run_and_gate_inputs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    config=ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json'
    fixed=json.loads(config.read_text())
    run=tmp_path/'pilot';run.mkdir()
    report=run/'report.json';report.write_text(json.dumps({'status':'completed_with_limits'}))
    args=SimpleNamespace(pilot_report=report,model_path=ROOT/'model/Llama-3.2-1B',config=config,
                         data_manifest=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json')
    verified=[]
    def verify(command,locked):
        verified.append(command)
        assert command[command.index('--steps')+1]=='1000'
        assert command[command.index('--diversity-space')+1]=='projected_z'
        return [{'global_step':1000,'resume':None}]
    monkeypatch.setattr(matrix,'verify_completed_command',verify)
    gate0=tmp_path/'pilot_gate0';gate1=tmp_path/'pilot_gate1';gate0.mkdir();gate1.mkdir()
    base={'status':'ready','model_path':str(args.model_path)}
    (gate0/'gate0_report.json').write_text(json.dumps(base))
    gate1_report={**base,'command':['python','trainer','--train-jsonl',str(args.data_manifest.parent/'train.jsonl'),
                                  '--valid-jsonl',str(args.data_manifest.parent/'valid.jsonl')]}
    (gate1/'gate1_report.json').write_text(json.dumps(gate1_report))
    assert matrix.verify_pilot_prerequisites(args,fixed)['gates_verified']
    assert verified
    gate1_report['command'][-1]='legacy/valid.jsonl'
    (gate1/'gate1_report.json').write_text(json.dumps(gate1_report))
    with pytest.raises(ValueError,match='different training inputs'):
        matrix.verify_pilot_prerequisites(args,fixed)
    (gate0/'gate0_report.json').write_text(json.dumps({**base,'status':'failed'}))
    with pytest.raises(ValueError,match='gate0 is not ready'):
        matrix.verify_pilot_prerequisites(args,fixed)
    report.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='pilot is not completed'):
        matrix.verify_pilot_prerequisites(args,fixed)


def test_matrix_refuses_stale_alias_audit_or_invalid_canonical_label(tmp_path):
    from types import SimpleNamespace
    cfg=json.loads((ROOT/'configs/llama32_1b_question_disjoint_matrix_v2.json').read_text())
    path=tmp_path/'matrix.json'
    args=SimpleNamespace(model_path=ROOT/'model/Llama-3.2-1B',config=path,
                         data_manifest=ROOT/'data/osnip_65k_question_disjoint_v2/data_manifest.json',
                         output_root=tmp_path/'output',pilot_report=tmp_path/'pilot/report.json')
    original=cfg['architecture_identity']['encoder_source_sha256']
    cfg['architecture_identity']['encoder_source_sha256']='0'*64
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='alias audit is stale'):
        matrix.build_plan(args)
    cfg['architecture_identity']['encoder_source_sha256']=original
    cfg['architecture_identity']['aliases']={'tokenwise_key_v1':'not_a_selected_architecture'}
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='canonical label'):
        matrix.build_plan(args)


def test_verify_completed_command_composes_per_cell_recipe(tmp_path):
    import torch
    from osnip_repro.checkpointing import RECIPE_FIELDS
    recipe = {"learning_rate": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0, "gradient_balance": "none",
              "batch_size": 1, "gradient_accumulation_steps": 64, "warmup_steps": 200,
              "constraint_ramp_steps": 1000, "loss_warmup_steps": 0, "constraint_mode": "gated",
              "gate_scope": "effective_batch", "tau_low": 0.05, "tau_high": 0.3,
              "lambda_util": 1.0, "lambda_priv": 1.0, "lambda_div": 2.0,
              "epsilon": 0.3, "diversity_margin": 1.0, "diversity_space": "projected_z",
              "key_dim": 32, "hidden_dim": 256, "alpha_init": 0.05, "alpha_ramp_steps": 0,
              "encryptor_architecture": "mlp_residual_v1", "max_length": 128, "dtype": "bfloat16"}
    run = tmp_path / "cell"
    run.mkdir()
    config = dict(recipe, lambda_priv=0.0, seed=5, train_sha256="a" * 64, valid_sha256="b" * 64,
                  model_path=str(tmp_path / "model"), resume=None, output_dir=str(run), steps=10)
    (run / "config.json").write_text(json.dumps(config))
    (run / "report.json").write_text(json.dumps(
        {"status": "completed_with_limits", "security_accepted": False,
         "steps_completed": 10, "resumed_from": None}))
    torch.save({"format": "osnip_training_checkpoint_v1", "global_step": 10,
                "config": config}, run / "last.pt")
    fixed = {"data": {"train_sha256": "a" * 64, "valid_sha256": "b" * 64},
             "training": dict(recipe)}
    command = ["python", "train_streaming.py", "--output-dir", str(run), "--steps", "10",
               "--lambda-priv", "0.0", "--model-path", str(tmp_path / "model")]
    matrix.verify_completed_command(command, fixed)
