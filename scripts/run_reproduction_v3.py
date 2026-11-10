#!/usr/bin/env python3
"""Execute a byte-bound OSNIP recipe serially; never infer security from training."""
from __future__ import annotations
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from osnip_repro.data import file_sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_bound(plan):
    for name, digest in plan['bound_inputs'].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('execution input changed: '+name)


def gate(report, previous=None, *, require_privacy_progress=False):
    values = report.get('last_validation', {})
    checks = {'completed': report.get('status') == 'completed_with_limits',
              'actual_updates': report.get('steps_executed_this_run', 0) > 0,
              'finite': all(isinstance(values.get(k), (int,float)) and math.isfinite(values[k])
                            for k in ('utility','privacy','diversity','ppl_ratio')),
              'utility_stable': values.get('utility', math.inf) <= .16,
              'ppl_stable': values.get('ppl_ratio', math.inf) <= 1.35,
              'projection': values.get('projection_fp32_max_relative_error', math.inf) <= 1e-5
                            and values.get('norm_max_relative_error', math.inf) <= 2*2**-10}
    if require_privacy_progress:
        baseline = (previous or {}).get('baseline_validation', {})
        checks['privacy_progress'] = values.get('privacy', math.inf) <= .99*baseline.get('privacy', 0.)
    return {'passed': all(checks.values()), 'checks': checks,
            'meaning': 'validation-only continuation gate, not privacy/utility acceptance',
            'security_accepted': False}


def validation_knn_command(python, ramp_directory):
    ramp_directory = Path(ramp_directory)
    return [str(python),str(ROOT/'scripts/validation_knn_checkpoint.py'),
            '--checkpoint',str(ramp_directory/'last.pt'),
            '--output',str(ramp_directory/'validation_knn_step1100.json')]


def validation_knn_gate(report, ramp_directory):
    """Validate diagnostic provenance; never accept safety from a pilot KNN."""
    checkpoint = Path(ramp_directory)/'last.pt'
    config = json.loads((Path(ramp_directory)/'config.json').read_text())
    values = report.get('rates', {})
    metrics = [values.get(side, {}).get(k) for side in ('clean','encrypted')
               for k in ('top1','top5','top10')]
    checks = {
        'completed': report.get('status') == 'completed',
        'step1100': report.get('global_step') == 1100,
        'fixed_validation64': report.get('samples') == 64
            and report.get('data_jsonl') == str(Path(config['valid_jsonl']).resolve())
            and report.get('data_sha256') == config['valid_sha256'],
        'checkpoint_bound': report.get('checkpoint') == str(checkpoint.resolve())
            and report.get('checkpoint_sha256') == file_sha256(checkpoint),
        'full_vocabulary': report.get('full_vocabulary') is True
            and report.get('vocabulary_size', 0) > 10,
        'positive_control': report.get('positive_controls_valid') is True
            and values.get('clean', {}).get('top1', 0) >= .95,
        'finite_rates': all(isinstance(x,(int,float)) and math.isfinite(x)
                            and 0 <= x <= 1 for x in metrics),
        'bound_inputs_unchanged': bool(report.get('bound_input_sha256'))
            and all(Path(p).is_file() and file_sha256(p) == digest
                    for p,digest in report.get('bound_input_sha256', {}).items()),
    }
    return {'passed':all(checks.values()),'checks':checks,
            'security_accepted':False,'meaning':'diagnostic validity, not final safety acceptance',
            'rates':values,'interpretation':report.get('interpretation')}


def recipe_common_flags(model: Path, data: Path) -> list[str]:
    """Locked paper recipe from PMLR 306 Appendix D.6 / Table 9.

    Table 9 gives BS/GPU 64; the Figure 8 step axis (~2,000 updates for
    1B/3B) implies 4 data-parallel GPUs, i.e. effective batch 256. On one
    device that is emulated with gradient accumulation 256. LR 1e-5, tau
    [0.05,0.15], 10-step loss warm-up, 200-step LR warm-up, 1,000-step
    constraint ramp, lambda1/lambda2 = 1/2, seed 42, raw-R Eq.19 diversity.
    """
    return ['--model-path',str(model),'--train-jsonl',str(data/'train.jsonl'),
        '--valid-jsonl',str(data/'valid.jsonl'),'--batch-size','1',
        '--gradient-accumulation-steps','256','--gate-scope','effective_batch',
        '--max-length','128','--max-valid-samples','256',
        '--encryptor-architecture','mlp_residual_v1','--key-dim','32','--hidden-dim','256',
        '--diversity-space','raw_r','--lambda-util','1','--lambda-priv','1','--lambda-div','2',
        '--epsilon','.3','--diversity-margin','1','--learning-rate','1e-5',
        '--warmup-steps','200','--constraint-ramp-steps','1000','--loss-warmup-steps','10',
        '--tau-low','.05','--tau-high','.15','--device','mps','--dtype','bfloat16',
        '--checkpoint-every','25','--evaluate-every','100','--constraint-mode','gated']


def build_plan(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('execution directory must be inside OSNIP')
    model = ROOT/'model/Llama-3.2-1B'
    data = ROOT/'data/osnip_65k_question_disjoint_v2'
    common = recipe_common_flags(model, data)
    seeds = []
    for seed in [42,43,44]:
        stages = []
        resume = None
        for name, steps in [('pilot100',100),('ramp1100',1100),('epoch8',2032)]:
            run = output/f'seed{seed}'/name
            cmd = [sys.executable,str(ROOT/'scripts/train_streaming.py'),*common,
                   '--seed',str(seed),'--steps',str(steps),'--output-dir',str(run)]
            if resume is not None: cmd += ['--resume',str(resume)]
            stages.append({'name':name,'steps':steps,'directory':str(run),'command':cmd,
                           'timeout_seconds':10*24*3600})
            resume = run/'last.pt'
        final = Path(stages[-1]['directory'])
        evaluate = [sys.executable,str(ROOT/'scripts/run_evaluation_suite.py'),
            '--model-path',str(model),'--run-dir',str(final),'--criterion','last',
            '--test-jsonl',str(data/'test.jsonl'),'--max-utility-samples','1000',
            '--max-task-total','1000','--utility-batch-size','1',
            '--ppl-jsonl',str(ROOT/'data/wikitext2/test.jsonl'),
            '--summarization-jsonl',str(ROOT/'data/osnip_sources/cnn_dailymail_test.jsonl'),
            '--max-summarization-samples','999','--require-bertscore',
            '--layers','1','5','10','15','--keys-per-prompt','5',
            '--max-knn-samples','1000','--max-vma-prompts','200','--candidate-mode','full',
            '--candidate-chunk','32','--vocab-chunk','1024',
            '--device','mps','--dtype','bfloat16','--seed','42',
            '--skip-input-vocabulary-probe','--execute']
        seeds.append({'seed':seed,'stages':stages,'evaluation_command':evaluate,
                      'evaluation_timeout_seconds':14*24*3600})
    paths = set(ROOT.rglob('src/osnip_repro/*.py')) | set((ROOT/'scripts').glob('*.py'))
    paths |= {data/name for name in ['train.jsonl','valid.jsonl','test.jsonl','data_manifest.json']}
    paths |= {model/name for name in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']}
    paths |= set((ROOT/'data/osnip_sources/raw').glob('*.json*'))
    paths |= {ROOT/'data/osnip_sources/cnn_dailymail_test.jsonl', ROOT/'data/wikitext2/test.jsonl',
              ROOT/'data/paper_attack_sets/knn.jsonl', ROOT/'data/paper_attack_sets/vma.jsonl',
              ROOT/'references/nltk_stopwords_english.txt'}
    for manifest in [ROOT/'model/roberta-large/download_manifest.json']:
        paths.add(manifest)
        meta = json.loads(manifest.read_text())
        paths |= {Path(meta['local_dir'])/name for name in meta['files']}
    return {'format':'osnip_reproduction_execution_v3','output':str(output),
        'python':sys.executable,'seeds':seeds,'train_records':65000,'samples_per_update':256,
        'formal_steps':2032,'formal_samples_seen':520192,'epochs':8,
        'paper_reference':{'source':'PMLR 306 Appendix D.6 / Table 9',
            'bs_per_gpu':64,'paper_gpus':4,'effective_batch':256,'paper_data':65151,
            'paper_steps':2036,'derivation':'8*65151/(64*4)=2036; package 65,000 rows gives 8*65000/256=2032',
            'learning_rate':1e-5,'tau_low':0.05,'tau_high':0.15,'loss_warmup_steps':10,
            'optimizer_warmup_steps':200,'constraint_ramp_steps':1000,'lambda_priv':1.0,
            'lambda_div':2.0,'seed':42,'epsilon_bound':'<0.4','diversity_margin_bound':'>0.5'},
        'bound_inputs':{str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(paths)},
        'security_accepted':False,'key_attack_performed':False,
        'validation_knn_checkpoint':{'enabled':True,'step':1100,'samples':64,
            'data_split':'valid','candidate_mode':'full','timeout_seconds':6*3600,
            'continue_after_review':False},
        'scope':'public-objective reproduction on question-disjoint v2; paper deviations disclosed',
        'deviations':['65000 instead of 65151 training rows; 78.3323% Alpaca instead of about 80%',
            'author encryptor architecture and exact templates undisclosed; engineering MLP used',
            'Gen Len=10 meaning not resolved; training uses teacher-forced max128 causal KL',
            'loss warm-up is 10 steps per PMLR D.6; the earlier "paper specifies none" note was based on arXiv v1, which omits Appendix D.6',
            'effective batch 256 is emulated on one device with gradient accumulation and a detached global gate; the paper used 4 data-parallel GPUs',
            'formal diversity is raw_r Eq.19; projected_z is a disclosed sensitivity arm',
            'epsilon=0.3 and diversity margin=1.0 are choices inside the published <0.4 / >0.5 bounds, not disclosed exact values',
            '999 unique summary samples; only ARC/SciQ task evaluator currently available (not paper ten-task suite)',
            'existing test/attack sets historically exposed; results are confirmation, not newly blind validation'],
        'acceptance':{'kind':'project_engineering_gate_distinct_from_paper',
            'max_knn_top10':.1,'max_ppl_ratio':1.25,'max_kl':.1,'min_task_retention':.95,
            'paper_llama_ppl_ratio':11.27/8.63,'paper_knn_top10':.0661}}


def execute(plan):
    policy = plan.get('validation_knn_checkpoint')
    if not policy or policy.get('enabled') is not True:
        raise ValueError('plan must explicitly bind the step1100 validation KNN checkpoint')
    output = Path(plan['output'])
    status = {'format':'osnip_execution_status_v1','runner_pid':os.getpid(),
              'security_accepted':False,'key_attack_performed':False}
    def update(**values):
        status.update(values)
        status['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(output/'execution_status.json',status)
    def run(command, directory, timeout, progress_directory=None):
        verify_bound(plan)
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'process.log').open('w') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start = time.monotonic()
            try:
                while True:
                    code = child.poll()
                    progress = Path(progress_directory or directory)/'train_progress.jsonl'
                    last = None
                    if progress.is_file():
                        with progress.open('rb') as stream:
                            stream.seek(max(0,progress.stat().st_size-65536))
                            lines = stream.read().splitlines()
                        if lines:
                            try: last=json.loads(lines[-1])
                            except json.JSONDecodeError: pass
                    update(child_pid=child.pid,child_exit_code=code,command=command,
                           process_log=str(directory/'process.log'),elapsed_seconds=time.monotonic()-start,
                           latest_step=None if last is None else last.get('step'),
                           samples_seen=None if last is None else last.get('samples_seen'),
                           latest_progress_age_seconds=None if not progress.is_file() else time.time()-progress.stat().st_mtime)
                    if code is not None: break
                    if time.monotonic()-start > timeout:
                        os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL); child.wait()
                        raise TimeoutError('hard timeout; latest checkpoint retained')
                    time.sleep(30)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                raise
            if code != 0: raise RuntimeError('child failed; see '+str(directory/'process.log'))
        verify_bound(plan)
    try:
        for item in plan['seeds']:
            first = None
            for stage in item['stages']:
                run_dir = Path(stage['directory'])
                if (run_dir/'report.json').exists():
                    raise FileExistsError('refusing ambiguous repeated execution: '+str(run_dir))
                update(state='training',seed=item['seed'],stage=stage['name'],run_directory=str(run_dir))
                # process.log resides beside trainer outputs so output refusal remains effective.
                run(stage['command'],run_dir.parent/(stage['name']+'_process'),stage['timeout_seconds'],run_dir)
                report=json.loads((run_dir/'report.json').read_text())
                if first is None: first=report
                if stage['name'] != 'epoch8':
                    decision=gate(report,first,require_privacy_progress=stage['name']=='ramp1100')
                    write_json(run_dir/'continuation_gate.json',decision)
                    if not decision['passed']:
                        update(state='held_at_validation_gate',decision=decision)
                        return
                if stage['name'] == 'ramp1100':
                    update(state='validation_knn',stage='validation_knn_step1100')
                    run(validation_knn_command(plan['python'],run_dir),
                        run_dir.parent/'validation_knn_step1100_process',policy['timeout_seconds'])
                    diagnostic=json.loads((run_dir/'validation_knn_step1100.json').read_text())
                    decision=validation_knn_gate(diagnostic,run_dir)
                    write_json(run_dir/'validation_knn_gate.json',decision)
                    if not decision['passed'] or policy.get('continue_after_review') is not True:
                        update(state='held_after_validation_knn' if decision['passed']
                               else 'held_at_validation_knn_gate',decision=decision,
                               automatic_continuation_enabled=False)
                        return
            update(state='evaluation',stage='full_same_checkpoint_evaluation')
            run(item['evaluation_command'],run_dir.parent/'evaluation_process',item['evaluation_timeout_seconds'])
        aggregate=[sys.executable,str(ROOT/'scripts/aggregate_reproduction_report.py'),
            '--output-dir',str(output/'aggregate')]
        for item in plan['seeds']:
            aggregate += ['--run-dir',item['stages'][-1]['directory']]
        update(state='aggregating',stage='joint_acceptance')
        run(aggregate,output/'aggregate_process',3600)
        update(state='queue_finished_review_aggregate_required',stage='complete')
    except BaseException as exc:
        update(state='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan_path=args.output.resolve()/'execution_plan.json'
    if args.execute:
        plan=json.loads(plan_path.read_text())
        if Path(plan['output']) != args.output.resolve(): raise ValueError('plan output mismatch')
        if (args.output/'execution_status.json').exists(): raise FileExistsError('execution already started')
        execute(plan)
    else:
        if plan_path.exists(): raise FileExistsError(plan_path)
        write_json(plan_path,build_plan(args.output))
        print(plan_path)

if __name__=='__main__': main()
