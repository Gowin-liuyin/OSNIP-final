import json
import subprocess
import sys
from pathlib import Path

import pytest



def bind_fixture_provenance(run: Path):
    """Attach test reports to real small local files without loading a model."""
    import hashlib
    checkpoint=run/'best_pareto.pt'
    test=run/'test.jsonl';test.write_text('held-out test data')
    ppl_data=run/'ppl.jsonl';ppl_data.write_text('ppl data')
    knn=run/'knn.jsonl';knn.write_text(''.join(json.dumps({'text':f'knn fixture prompt {i}'})+'\n' for i in range(1000)))
    vma=run/'vma.jsonl';vma.write_text(''.join(json.dumps({'text':f'vma fixture prompt {i}'})+'\n' for i in range(200)))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    config=json.loads((run/'config.json').read_text());config['model_path']=str(run/'model')
    config.update(train_sha256=sha(test),valid_sha256=sha(ppl_data),diversity_space='projected_z',
                  constraint_mode='gated',lambda_util=1.,lambda_priv=1.,lambda_div=1.,key_dim=32,
                  hidden_dim=256,max_length=128,learning_rate=1e-5,steps=10000,epsilon=.3,diversity_margin=1.)
    (run/'config.json').write_text(json.dumps(config))
    import torch
    torch.save({"format": "osnip_training_checkpoint_v1", "global_step": 100,
                "encoder": {"w": torch.zeros(2)}, "config": config,
                "metrics": {"validation": {}}, "extra": {}}, checkpoint)
    for name in ['utility_test.json','task_utility_test.json','sliding_ppl_test.json','paper_protocol_test.json']:
        path=run/name;report=json.loads(path.read_text())
        report.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),model_path=config['model_path'])
        root=Path(__file__).resolve().parents[1]
        source_keys={'utility_test.json':('scripts/evaluate_no_key.py','src/osnip_repro/modeling.py'),
                     'task_utility_test.json':('scripts/evaluate_task_utility.py',),
                     'sliding_ppl_test.json':('scripts/evaluate_sliding_ppl.py','src/osnip_repro/modeling.py'),
                     'paper_protocol_test.json':('scripts/evaluate_paper_protocol.py','src/osnip_repro/vma.py')}[name]
        report['source_sha256']={key: sha(root/key) for key in source_keys}
        if name=='utility_test.json':
            report.update(configuration={'checkpoint':str(checkpoint),'model_path':config['model_path'],'data_jsonl':str(test)},
                          checkpoint_config_sha256=sha(run/'config.json'),data_sha256=sha(test))
        elif name=='task_utility_test.json':
            report.update(records_scored=1000,test_jsonl=str(test),test_sha256=sha(test),task_gate={'passed':True},
                          clean_accuracy=1.0,accuracy_retention_vs_clean=1.0,
                          rows=[{'index':i,'source':'arc','choices':['A','B'],'correct':'A',
                                 'clean_prediction':'A','encrypted_prediction':'A',
                                 'clean_correct':True,'encrypted_correct':True} for i in range(1000)])
        elif name=='sliding_ppl_test.json':
            report.update(data_jsonl=str(ppl_data),data_sha256=sha(ppl_data))
        else:
            report.update(knn_jsonl=str(knn),vma_jsonl=str(vma),data_sha256={'knn':sha(knn),'vma':sha(vma)})
            report.update(format='osnip_paper_protocol_evaluation_v1', key_attack_performed=False)
            report['knn'].update(positions=3000, metric='l2', full_vocabulary=True, vocabulary_size=32,
                rows=[{'index':i, 'positions':3, 'clean':{k:3 for k in ('top1','top5','top10')},
                       'encrypted':{k:0 for k in ('top1','top5','top10')}} for i in range(1000)],
                counts={'clean':{k:3000 for k in ('top1','top5','top10')},
                        'encrypted':{k:0 for k in ('top1','top5','top10')}},
                rates={'clean':{k:1. for k in ('top1','top5','top10')},
                       'encrypted':{k:0. for k in ('top1','top5','top10')}})
            layers=[int(k) for k in report['vma']['summary']]
            def observation(prompt, layer, key):
                return {'prompt':prompt,'layer':layer,'key':key,'positions':3,'content_positions':2,
                        'total_top1_hits':3 if key==-1 else 0,'clean_top1_hits':2 if key==-1 else 0,
                        'candidate_coverage':1.}
            report['vma'].update(layers=layers, candidate_mode='full', max_tokens=512,
                clean_rows=[observation(i,layer,-1) for i in range(200) for layer in layers],
                rows=[observation(i,layer,key) for i in range(200) for layer in layers for key in range(5)])
        path.write_text(json.dumps(report))
    root=Path(__file__).resolve().parents[1]
    policy=json.loads((root/'configs/llama32_1b_summary_evaluation.json').read_text())
    n=policy['records']
    scorer_dir=run/'scorer';scorer_dir.mkdir(exist_ok=True)
    scorer_files={}
    for name in ['config.json','model.safetensors','tokenizer.json','vocab.json','merges.txt']:
        file=scorer_dir/name;file.write_bytes(('test-only '+name).encode())
        scorer_files[name]={'sha256':sha(file),'bytes':file.stat().st_size}
    scorer_manifest=scorer_dir/'download_manifest.json'
    scorer_manifest.write_text(json.dumps({'local_dir':str(scorer_dir.resolve()),'model_id':'roberta-large','files':scorer_files}))
    scorer_provenance={'manifest':str(scorer_manifest),'manifest_sha256':sha(scorer_manifest),
                       'directory':str(scorer_dir.resolve()),'files':scorer_files}
    summary={'format':'osnip_summarization_utility_report_v2','status':'completed_with_limits',
             'server_input':'inputs_embeds_and_attention_mask_only_no_original_input_ids','key_attack_performed':False,'security_accepted':False,
             'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
             'checkpoint_config_sha256':sha(run/'config.json'),'model_path':config['model_path'],
             'data_jsonl':str((root / policy['data_jsonl']).resolve()),'data_sha256':policy['data_sha256'],'records':n,
             'clean':{'rouge_l':.3},'encrypted':{'rouge_l':.3},
             'retention':{'rouge_l':1.,'bertscore_f1':1.},
             'generation':{k:policy[k] for k in ['max_input_length','max_new_tokens','num_beams']},
             'rows':[{'index':i,'clean_rouge_l':.3,'encrypted_rouge_l':.3} for i in range(n)],
             'bertscore':{'available':True,'model':policy['bertscore_model'],'scorer_hash':'fixture-scorer','provenance':scorer_provenance,
                          'clean':{'f1':.8},'encrypted':{'f1':.8},
                          'rows':[{'index':i,'clean_f1':.8,'encrypted_f1':.8} for i in range(n)]}}
    (run/'summarization_utility_test.json').write_text(json.dumps(summary))


requires_summary_snapshot = pytest.mark.skipif(
    any(not (Path(__file__).resolve().parents[1] / f'data/osnip_sources/{name}').is_file()
        for name in ('cnn_dailymail_test.manifest.json', 'cnn_dailymail_test.jsonl')),
    reason='requires the pinned CNN/DailyMail summarization snapshot',
)


@requires_summary_snapshot
def test_aggregator_reads_seed_from_config_and_applies_strict_gates(tmp_path: Path):
    run_root = tmp_path / "runs"
    run = run_root / "seed_run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"seed": 20261018, "encryptor_architecture": "key_film_residual_v1"}))
    (run / "report.json").write_text(json.dumps({"status": "completed_with_limits"}))
    (run / "utility_test.json").write_text(json.dumps({"samples": 1000, "utility": {"ppl_ratio": 1.0, "mean_logit_kl": 0.01}}))
    (run / "task_utility_test.json").write_text(json.dumps({"accuracy_retention_vs_clean": 1.0, "task_gate": {"passed": True}}))
    (run / "sliding_ppl_test.json").write_text(json.dumps({"samples": 100, "ppl_ratio": 1.0}))
    (run / "paper_protocol_test.json").write_text(json.dumps({
        "security_accepted": False,
        "vma": {
            "prompts": 200,
            "keys_per_prompt": 5,
            "full_vocabulary": True,
            "summary": {str(layer): {
                "clean_positive_control": {"total_asr": 1.0, "clean_asr": 1.0},
                "encrypted": {"total_asr": 0.0, "clean_asr": 0.0},
            } for layer in (1, 5, 10, 15)},
        },
        "knn": {
            "samples": 1000,
            "rates": {
                "clean": {"top1": 1.0},
                "encrypted": {"top10": 0.0},
            },
        },
    }))
    bind_fixture_provenance(run)
    output = tmp_path / "aggregate"
    script = Path(__file__).resolve().parents[1] / "scripts/aggregate_reproduction_report.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run-root", str(run_root), "--output-dir", str(output), "--min-seeds", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "final_reproduction_report.json").read_text())
    assert json.loads(result.stdout)["seeds"] == [20261018]
    assert report["status"] == "security_accepted"
    assert report["runs"][0]["encryptor_architecture"] == "key_film_residual_v1"

def test_aggregator_prioritizes_utility_failure_when_both_gates_fail(tmp_path: Path):
    run_root = tmp_path / "runs"
    run = run_root / "seed_run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"seed": 1}))
    (run / "report.json").write_text(json.dumps({"status": "completed_with_limits"}))
    (run / "utility_test.json").write_text(json.dumps({"samples": 1000, "utility": {"ppl_ratio": 9.0, "mean_logit_kl": 9.0}}))
    (run / "task_utility_test.json").write_text(json.dumps({"accuracy_retention_vs_clean": 0.0}))
    (run / "sliding_ppl_test.json").write_text(json.dumps({"samples": 100, "ppl_ratio": 1.0}))
    (run / "paper_protocol_test.json").write_text(json.dumps({
        "vma": {
            "prompts": 200,
            "keys_per_prompt": 5,
            "full_vocabulary": True,
            "summary": {str(layer): {
                "clean_positive_control": {"total_asr": 1.0, "clean_asr": 1.0},
                "encrypted": {"total_asr": 0.0, "clean_asr": 0.0},
            } for layer in (1, 5, 10, 15)},
        },
        "knn": {"samples": 1000, "rates": {"clean": {"top1": 1.0}, "encrypted": {"top10": 0.0}}},
    }))
    output = tmp_path / "aggregate"
    script = Path(__file__).resolve().parents[1] / "scripts/aggregate_reproduction_report.py"
    subprocess.run([sys.executable, str(script), "--run-root", str(run_root), "--output-dir", str(output), "--min-seeds", "1"], check=True)
    report = json.loads((output / "final_reproduction_report.json").read_text())
    assert report["status"] == "utility_failed"


def test_semantic_coverage_diagnostic_helpers_are_finite():
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from evaluate_semantic_coverage import per_example_kl, spherical_complement_bound

    logits = __import__("torch").randn(3, 7)
    assert __import__("torch").allclose(per_example_kl(logits, logits), __import__("torch").zeros(3), atol=1e-6)
    bound = spherical_complement_bound(896, 0.3)
    assert 0.0 <= bound <= 1.0


def test_reproduction_and_ablation_configs_share_architectures():
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    reproduction = json.loads((configs_dir / "llama32_1b_reproduction.json").read_text())
    matrix = json.loads((configs_dir / "llama32_1b_ablation_matrix.json").read_text())
    assert reproduction["training"]["encryptor_variants"] == matrix["encryptor_variants"]
    assert reproduction["training"]["diversity_space"] == "projected_z"


def test_summarization_utility_helpers():
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from evaluate_summarization_utility import record_fields, rouge_l_f1

    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert rouge_l_f1("a b", "c d") == 0.0
    assert record_fields({"article": "article", "highlights": "summary"}) == ("article", "summary")
    assert record_fields({"text": "article", "summary": "summary"}) == ("article", "summary")


def test_summarization_dataset_preparation_writes_manifest(tmp_path: Path):
    import subprocess
    import sys

    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"article": "one", "highlights": "summary"}) + "\n" + json.dumps({"text": "two", "summary": "summary two"}) + "\n")
    output = tmp_path / "prepared.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_summarization_dataset.py"
    subprocess.run([sys.executable, str(script), "--input", str(source), "--output", str(output)], check=True)
    manifest = json.loads((tmp_path / "prepared.jsonl.manifest.json").read_text())
    assert manifest["status"] == "ready"
    assert manifest["records_written"] == 2
    assert len(output.read_text().splitlines()) == 2


def test_ppl_complete_snapshot_verification_checks_hashes_and_full_target_coverage():
    import importlib.util
    import hashlib
    script = Path(__file__).resolve().parents[1] / "scripts/aggregate_reproduction_report.py"
    spec = importlib.util.spec_from_file_location("aggregate_ppl_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = script.parents[1] / "data/wikitext2/test.jsonl"
    if not data.is_file():
        __import__("pytest").skip("pinned WikiText snapshot not downloaded")
    n = len(data.read_text().splitlines())
    report = {"samples": n, "sequence_token_lengths": [2] * n, "target_tokens": n,
              "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    assert module.complete_locked_ppl_snapshot(report)
    assert not module.complete_locked_ppl_snapshot({**report, "samples": n - 1})
    assert not module.complete_locked_ppl_snapshot({**report, "target_tokens": n - 1})
    assert not module.complete_locked_ppl_snapshot({**report, "data_sha256": "wrong"})


def test_aggregator_rejects_bad_sliding_ppl_and_missing_required_vma_layers(tmp_path):
    run = tmp_path / 'single_run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'seed': 1}))
    (run / 'report.json').write_text(json.dumps({'status': 'completed_with_limits'}))
    (run / 'utility_test.json').write_text(json.dumps({'samples': 1000, 'utility': {'ppl_ratio': 1., 'mean_logit_kl': .01}}))
    (run / 'task_utility_test.json').write_text(json.dumps({'accuracy_retention_vs_clean': 1., 'task_gate': {'passed': True}}))
    (run / 'sliding_ppl_test.json').write_text(json.dumps({'samples': 100, 'ppl_ratio': 3.}))
    privacy = {
        'knn': {'samples': 1000, 'rates': {'clean': {'top1': 1.}, 'encrypted': {'top10': 0.}}},
        'vma': {'prompts': 200, 'keys_per_prompt': 5, 'full_vocabulary': True, 'summary': {
            '1': {'clean_positive_control': {'total_asr': 1., 'clean_asr': 1.},
                  'encrypted': {'total_asr': 0., 'clean_asr': 0.}}}}}
    (run / 'paper_protocol_test.json').write_text(json.dumps(privacy))
    script = Path(__file__).resolve().parents[1] / 'scripts/aggregate_reproduction_report.py'
    output = tmp_path / 'aggregate'
    command = [sys.executable, str(script), '--run-dir', str(run), '--output-dir', str(output), '--min-seeds', '1']
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads((output / 'final_reproduction_report.json').read_text())
    assert not report['checks']['sliding_ppl_gate']['passed']
    assert not report['checks']['vma_layer_coverage']['passed']
    assert report['status'] == 'utility_failed'
    (run / 'sliding_ppl_test.json').write_text(json.dumps({'samples': 100, 'ppl_ratio': 1.}))
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads((output / 'final_reproduction_report.json').read_text())
    assert report['checks']['sliding_ppl_gate']['passed']
    assert report['status'] != 'security_accepted'
    assert not report['security_accepted']


def test_suite_plan_reuses_independent_knn_and_requests_bound_vma_resume(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'selected_pareto.json').write_text(json.dumps({'selected': {'path': str(run / 'best_pareto.pt')}}))
    (run / 'knn_report.json').write_text(json.dumps({'format': 'osnip_knn_report_v2', 'rows': [{'index': 0}]}))
    (run / 'paper_protocol_test.vma_rows.jsonl').write_text('{}\n')
    script = Path(__file__).resolve().parents[1] / 'scripts/run_evaluation_suite.py'
    subprocess.run([sys.executable, str(script), '--model-path', str(tmp_path / 'model'), '--run-dir', str(run),
                    '--skip-input-vocabulary-probe'], check=True, capture_output=True)
    plan = json.loads((run / 'evaluation_suite_plan.json').read_text())
    assert '--knn-report' in plan['commands']['privacy']
    assert '--resume-vma' in plan['commands']['privacy']
    assert '--skip-vocabulary-probe' in plan['commands']['utility']
    assert plan['commands']['privacy'][plan['commands']['privacy'].index('--candidate-mode') + 1] == 'full'
    assert not plan['execute']


@requires_summary_snapshot
def test_aggregator_refuses_mixed_checkpoints_and_reduced_task_reports(tmp_path):
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/aggregate_reproduction_report.py'
    spec=importlib.util.spec_from_file_location('provenance_aggregate',script)
    aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)
    # Generate the already-verified passing fixture with concrete input bindings.
    test_aggregator_reads_seed_from_config_and_applies_strict_gates(tmp_path)
    run=tmp_path/'runs/seed_run'
    paths={name:run/filename for name,filename in [('utility','utility_test.json'),('task','task_utility_test.json'),('ppl','sliding_ppl_test.json'),('privacy','paper_protocol_test.json')]}
    reports={name:json.loads(path.read_text()) for name,path in paths.items()}
    config=json.loads((run/'config.json').read_text())
    assert aggregate.report_provenance(run,config,**reports)['passed']
    reports['privacy']['checkpoint_sha256']='0'*64
    provenance=aggregate.report_provenance(run,config,**reports)
    assert not provenance['passed'] and any('privacy: checkpoint' in e for e in provenance['errors'])
    paths['privacy'].write_text(json.dumps(reports['privacy']))
    command=[sys.executable,str(script),'--run-dir',str(run),'--output-dir',str(tmp_path/'aggregate'),'--min-seeds','1']
    subprocess.run(command,check=True,capture_output=True)
    output=json.loads((tmp_path/'aggregate/final_reproduction_report.json').read_text())
    assert not output['security_accepted'] and not output['all_evaluations_complete']
    assert not output['checks']['artifact_provenance']['passed']
    reports['privacy']['checkpoint_sha256']=reports['utility']['checkpoint_sha256']
    paths['privacy'].write_text(json.dumps(reports['privacy']))
    reports['task']['records_scored']=16
    paths['task'].write_text(json.dumps(reports['task']))
    subprocess.run(command,check=True,capture_output=True)
    output=json.loads((tmp_path/'aggregate/final_reproduction_report.json').read_text())
    assert not output['security_accepted'] and not output['checks']['task_sample_count']['passed']
    reports['task']['records_scored']=1000
    paths['task'].write_text(json.dumps(reports['task']))
    # Local data replacement invalidates report provenance even if metric values still pass.
    Path(reports['task']['test_jsonl']).write_text('replaced split')
    assert not aggregate.report_provenance(run,config,**reports)['passed']
    assert not aggregate.report_provenance(run,{**config,'model_path':'different_model'},**reports)['passed']


def test_aggregate_condition_identity_does_not_pool_changed_objectives_or_architectures():
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/aggregate_reproduction_report.py'
    spec=importlib.util.spec_from_file_location('aggregate_conditions',script)
    aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)
    config={name:1 for name in aggregate.CONDITION_FIELDS}
    config.update(model_path='model',encryptor_architecture='mlp_residual_v1',seed=1)
    identity=aggregate.condition_identity(config)
    assert identity['complete']
    assert identity==aggregate.condition_identity({**config,'seed':2,'output_dir':'other'})
    assert identity!=aggregate.condition_identity({**config,'resume':'phase1/last.pt'})
    assert identity!=aggregate.condition_identity({**config,'lambda_priv':2})
    assert identity!=aggregate.condition_identity({**config,'encryptor_architecture':'key_film_residual_v1'})
    assert identity!=aggregate.condition_identity({**config,'train_sha256':'other'})
    del config['learning_rate']
    assert not aggregate.condition_identity(config)['complete']


@requires_summary_snapshot
def test_summary_gate_refuses_missing_scorer_short_rows_and_mixed_checkpoints(tmp_path):
    import copy
    import importlib.util
    spec=importlib.util.spec_from_file_location('summary_aggregate_test',Path(__file__).resolve().parents[1]/'scripts/aggregate_reproduction_report.py')
    aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)
    test_aggregator_reads_seed_from_config_and_applies_strict_gates(tmp_path)
    run=tmp_path/'runs/seed_run'
    config=json.loads((run/'config.json').read_text());utility=json.loads((run/'utility_test.json').read_text())
    summary=json.loads((run/'summarization_utility_test.json').read_text())
    assert aggregate.summarize_summary_evaluation(run,config,utility,summary)['passed']
    assert not aggregate.summarize_summary_evaluation(run,config,utility,None)['complete']
    bad=copy.deepcopy(summary);bad['bertscore']['available']=False
    assert not aggregate.summarize_summary_evaluation(run,config,utility,bad)['complete']
    bad=copy.deepcopy(summary);bad['rows']=bad['rows'][:-1]
    assert not aggregate.summarize_summary_evaluation(run,config,utility,bad)['complete']
    bad=copy.deepcopy(summary);bad['checkpoint_sha256']='0'*64
    assert not aggregate.summarize_summary_evaluation(run,config,utility,bad)['complete']
    bad=copy.deepcopy(summary);bad['encrypted']['rouge_l']=.01;bad['retention']['rouge_l']=.01/.3
    for row in bad['rows']:row['encrypted_rouge_l']=.01
    gate=aggregate.summarize_summary_evaluation(run,config,utility,bad)
    assert gate['complete'] and not gate['passed']
    bad=copy.deepcopy(summary);bad['retention']['rouge_l']=100
    assert not aggregate.summarize_summary_evaluation(run,config,utility,bad)['complete']


@requires_summary_snapshot
def test_summary_gate_rejects_clear_prompt_server_contract(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('summary_contract_aggregate',Path(__file__).resolve().parents[1]/'scripts/aggregate_reproduction_report.py')
    aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)
    test_aggregator_reads_seed_from_config_and_applies_strict_gates(tmp_path)
    run=tmp_path/'runs/seed_run'
    config=json.loads((run/'config.json').read_text());utility=json.loads((run/'utility_test.json').read_text())
    summary=json.loads((run/'summarization_utility_test.json').read_text())
    del summary['server_input']
    report=aggregate.summarize_summary_evaluation(run,config,utility,summary)
    assert not report['complete'] and any('prompt token IDs' in error for error in report['errors'])


def test_aggregate_separates_new_diagnostic_training_recipes():
    from scripts.aggregate_reproduction_report import condition_identity, EXTENSION_CONDITION_DEFAULTS
    base = {'model_path': 'model'}
    original = condition_identity(base)
    assert original == condition_identity({**base, **EXTENSION_CONDITION_DEFAULTS})
    for field, value in {'lambda_hidden': 1.0, 'hidden_layers': [10, 15],
                         'gradient_balance': 'component_norm', 'lambda_knn': 1.0,
                         'knn_negative_count': 128, 'knn_topk': 5, 'knn_margin': 0.1,
                         'alpha_init': 3.2, 'alpha_ramp_steps': 1000,
                         'freeze_alpha': True, 'reset_optimizer': True,
                         'grad_clip': 0.5, 'dtype': 'float16'}.items():
        assert original != condition_identity({**base, field: value}), field


def test_knn_fallback_requires_checkpoint_binding(tmp_path):
    from scripts.aggregate_reproduction_report import bound_knn_fallback
    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_bytes(b"checkpoint-bytes")
    import hashlib
    digest = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    legacy: list[str] = []
    (run / "knn_report.json").write_text(json.dumps({"checkpoint_sha256": digest, "rates": {}}))
    assert bound_knn_fallback(run, legacy) == {}
    assert legacy == ["knn:knn_report.json(rowless-ignored)"]
    legacy = []
    (run / "knn_report.json").write_text(json.dumps(
        {"checkpoint_sha256": digest, "rates": {}, "rows": [{"index": 0}]}))
    assert bound_knn_fallback(run, legacy)["rates"] == {}
    assert legacy == ["knn:knn_report.json(bound)"]
    legacy = []
    (run / "knn_report.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64, "rates": {"x": 1}}))
    assert bound_knn_fallback(run, legacy) == {}
    assert legacy == ["knn:knn_report.json(unbound-ignored)"]


def test_aggregate_finite_rejects_json_booleans():
    import scripts.aggregate_reproduction_report as aggregate
    assert aggregate.finite(1.0) is True
    assert aggregate.finite(True) is False
    assert aggregate.finite(False) is False
    assert aggregate.finite(None) is False
