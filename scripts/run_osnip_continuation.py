#!/usr/bin/env python3
"""Resume one frozen OSNIP seed to its full budget, then evaluate that checkpoint."""
from __future__ import annotations
import argparse,datetime,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from osnip_repro.data import file_sha256


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def verify_inputs(manifest):
    for name,digest in manifest['bound_inputs'].items():
        p=ROOT/name
        if not p.is_file() or file_sha256(p)!=digest:
            raise ValueError('continuation input changed: '+name)
    checkpoint=Path(manifest['parent_checkpoint'])
    if file_sha256(checkpoint)!=manifest['parent_checkpoint_sha256']:
        raise ValueError('parent checkpoint changed')


def run_child(command,log_path,timeout,status_cb,progress_path=None):
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        try:
            while True:
                code=child.poll()
                progress={}
                if progress_path and Path(progress_path).is_file():
                    try:
                        lines=Path(progress_path).read_text().splitlines()
                        if lines:progress=json.loads(lines[-1])
                    except (OSError,json.JSONDecodeError):pass
                status_cb(child_pid=child.pid,child_exit_code=code,elapsed_seconds=time.monotonic()-started,
                          latest_step=progress.get('step'),samples_seen=progress.get('samples_seen'),
                          latest_progress_age_seconds=(None if not progress_path or not Path(progress_path).exists()
                            else time.time()-Path(progress_path).stat().st_mtime),
                          latest_metrics={k:progress.get(k) for k in ['utility','privacy','ppl_ratio','mean_cosine_key1','mean_cosine_key2'] if k in progress})
                if code is not None:break
                if time.monotonic()-started>timeout:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise TimeoutError('hard timeout; saved artifacts retained')
                time.sleep(30)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if code!=0:raise RuntimeError('child failed; inspect '+str(log_path))


def execute(manifest_path):
    manifest_path=Path(manifest_path).resolve();m=json.loads(manifest_path.read_text())
    out=Path(m['output']).resolve();status_path=out/'continuation_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') not in ['created']:
        raise FileExistsError('continuation already started; refusing automatic retry')
    run_dir=Path(m['run_directory'])
    if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError('run directory is not empty')
    state={'format':'osnip_single_seed_continuation_status_v1','runner_pid':os.getpid(),
           'security_accepted':False,'key_attack_performed':False,'automatic_retry':False}
    def update(**values):
        state.update(values);state['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(status_path,state)
    try:
        verify_inputs(m)
        update(state='training',seed=m['seed'],stage='epoch8_resume',run_directory=str(run_dir),
               parent_checkpoint=m['parent_checkpoint'],parent_checkpoint_sha256=m['parent_checkpoint_sha256'],
               command=m['training_command'],process_log=m['training_log'])
        run_child(m['training_command'],m['training_log'],m['training_timeout_seconds'],update,
                  run_dir/'train_progress.jsonl')
        verify_inputs(m)
        report=json.loads((run_dir/'report.json').read_text())
        if report.get('status')!='completed_with_limits' or report.get('steps_completed')!=m['target_steps'] or report.get('security_accepted') is not False:
            raise ValueError('training report failed completion/budget/security-state checks')
        update(state='evaluation',stage='full_same_checkpoint_evaluation',training_report=str(run_dir/'report.json'),
               command=m['evaluation_command'],process_log=m['evaluation_log'])
        run_child(m['evaluation_command'],m['evaluation_log'],m['evaluation_timeout_seconds'],update)
        verify_inputs(m)
        update(state='evaluation_completed_review_required',stage='seed42_single_seed_screen_complete',
               evaluation_report=str(run_dir/'utility_report.json'),
               security_accepted=False,multi_seed_complete=False)
    except BaseException as exc:
        update(state='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),
               security_accepted=False)
        raise


def create_manifest(output,parent_plan):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('output must remain inside OSNIP')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    plan_path=Path(parent_plan).resolve();plan=json.loads(plan_path.read_text())
    item=next(x for x in plan['seeds'] if x['seed']==42)
    stage=next(x for x in item['stages'] if x['name']=='epoch8')
    ramp=next(x for x in item['stages'] if x['name']=='ramp1100')
    parent=Path(ramp['directory'])/'last.pt'
    import torch
    payload=torch.load(parent,map_location='cpu',weights_only=False)
    if payload.get('global_step')!=1100:raise ValueError('expected v4 seed42 global_step=1100 checkpoint')
    gate=json.loads((Path(ramp['directory'])/'validation_knn_gate.json').read_text())
    if not gate.get('passed'):raise ValueError('step1100 diagnostic gate is invalid')
    run_dir=output/'seed42/epoch8'
    if run_dir.exists() and any(run_dir.iterdir()):raise FileExistsError(run_dir)
    cmd=list(stage['command'])
    for i,value in enumerate(cmd):
        if value=='--output-dir':cmd[i+1]=str(run_dir)
        if value=='--resume':cmd[i+1]=str(parent)
    if str(parent) not in cmd or str(run_dir) not in cmd or '--steps' not in cmd:
        raise ValueError('could not safely rebind planned command')
    eval_cmd=list(item['evaluation_command'])
    for i,value in enumerate(eval_cmd):
        if value=='--run-dir':eval_cmd[i+1]=str(run_dir)
    if '--run-dir' not in eval_cmd:raise ValueError('evaluation command missing run directory')
    bound={name:file_sha256(ROOT/name) for name in plan['bound_inputs']}
    for name in ['scripts/run_osnip_continuation.py','scripts/train_streaming.py','scripts/run_evaluation_suite.py',
                 'src/osnip_repro/checkpointing.py','src/osnip_repro/curriculum.py','src/osnip_repro/data.py',
                 'src/osnip_repro/encoder.py','src/osnip_repro/losses.py','src/osnip_repro/modeling.py','src/osnip_repro/projection.py']:
        bound[name]=file_sha256(ROOT/name)
    m={'format':'osnip_single_seed_continuation_v1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'output':str(output),'seed':42,'parent_plan':str(plan_path),'parent_plan_sha256':file_sha256(plan_path),
       'parent_checkpoint':str(parent),'parent_checkpoint_sha256':file_sha256(parent),
       'parent_global_step':1100,'target_steps':2032,'additional_updates':932,
       'run_directory':str(run_dir),'training_command':cmd,'evaluation_command':eval_cmd,
       'training_log':str(output/'seed42/epoch8_process/process.log'),
       'evaluation_log':str(output/'seed42/evaluation_process/process.log'),
       'training_timeout_seconds':10*24*3600,'evaluation_timeout_seconds':14*24*3600,
       'bound_inputs':bound,'locked_recipe':{k:payload['config'].get(k) for k in
           ['seed','batch_size','gradient_accumulation_steps','gate_scope','max_length','max_valid_samples',
            'encryptor_architecture','key_dim','hidden_dim','diversity_space','lambda_util','lambda_priv','lambda_div',
            'lambda_knn','epsilon','diversity_margin','learning_rate','warmup_steps','constraint_ramp_steps',
            'loss_warmup_steps','tau_low','tau_high','device','dtype','constraint_mode','train_sha256','valid_sha256']},
       'validation_knn_gate':gate,'security_accepted':False,'key_attack_performed':False,
       'automatic_retry':False,'multi_seed_complete':False}
    write_json(output/'continuation_manifest.json',m)
    write_json(output/'continuation_status.json',{'format':'osnip_single_seed_continuation_status_v1',
        'state':'created','security_accepted':False,'key_attack_performed':False,
        'automatic_retry':False,'runner_pid':None})
    return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--parent-plan',type=Path,required=True);ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();path=a.output.resolve()/'continuation_manifest.json'
    if a.execute:
        if not path.is_file():raise FileNotFoundError(path)
        execute(path)
    else:
        if path.exists():raise FileExistsError(path)
        m=create_manifest(a.output,a.parent_plan);print(json.dumps({'manifest':str(path),'checkpoint':m['parent_checkpoint'],
            'step_target':m['target_steps'],'additional_updates':m['additional_updates'],
            'training_command':m['training_command'],'evaluation_command':m['evaluation_command']},indent=2))
if __name__=='__main__':main()
