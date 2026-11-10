#!/usr/bin/env python3
"""Verify the local summary scorer, optionally after an identified pilot exits."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from evaluate_summarization_utility import compute_paired_bertscore, resolve_scorer_snapshot, sha256
from osnip_repro.process_inspection import command_tokens, process_command


def validate_training_identity(command: str, expected_run: Path) -> None:
    tokens=command_tokens(command)
    if not any(Path(token).name=='train_streaming.py' for token in tokens):
        raise ValueError('monitored process is not the expected trainer')
    try:
        selected=Path(tokens[tokens.index('--output-dir')+1])
    except (ValueError,IndexError) as exc:
        raise ValueError('monitored trainer has no output path') from exc
    if not selected.is_absolute():selected=ROOT/selected
    if selected.resolve()!=expected_run.resolve():
        raise ValueError('monitored trainer output path differs')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--wait-for-training-pid',type=int)
    ap.add_argument('--expected-run',type=Path)
    ap.add_argument('--expected-steps',type=int,default=1000)
    ap.add_argument('--poll-seconds',type=int,default=30)
    ap.add_argument('--snapshot-only',action='store_true',help='inspect tokenizer/config/weights; never load the scorer decoder')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.poll_seconds<1 or args.expected_steps<1:raise ValueError('positive wait interval and steps required')
    if args.wait_for_training_pid is not None and (args.wait_for_training_pid<1 or args.expected_run is None):
        raise ValueError('a positive trainer PID and expected run are both required')
    state_path=args.output.with_suffix('.state.json')
    if state_path.exists():raise FileExistsError(state_path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    state={'format':'osnip_summary_scorer_verification_state_v1','status':'starting',
           'pid':args.wait_for_training_pid,'expected_run':str(args.expected_run.resolve()) if args.expected_run else None,
           'snapshot_only':args.snapshot_only,'security_accepted':False}
    def save(status):
        state.update(status=status,observed_at=datetime.datetime.now().astimezone().isoformat())
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    try:
        if args.wait_for_training_pid is not None:
            save('waiting_for_identified_training_process')
            while True:
                command=process_command(args.wait_for_training_pid)
                if command is None:break
                validate_training_identity(command,args.expected_run)
                time.sleep(args.poll_seconds)
            training=json.loads((args.expected_run/'report.json').read_text())
            if training.get('status')!='completed_with_limits' or training.get('steps_completed')!=args.expected_steps:
                raise ValueError('pilot did not complete the expected training budget')
            state['completed_training_report_sha256']=sha256(args.expected_run/'report.json')
        save('verifying_local_snapshot')
        from transformers import AutoConfig,AutoTokenizer
        from safetensors import safe_open
        options,provenance=resolve_scorer_snapshot('roberta-large')
        directory=Path(options['model_type'])
        config=AutoConfig.from_pretrained(directory,local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=False)
        if config.model_type!='roberta' or config.num_hidden_layers!=24 or config.hidden_size!=1024:
            raise ValueError('scorer snapshot is not the intended RoBERTa-large architecture')
        token_ids=tokenizer('A fixed tokenizer verification sentence.',add_special_tokens=True)['input_ids']
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as handle:
            keys=list(handle.keys())
            names=[name for name in keys if name.endswith('embeddings.word_embeddings.weight')]
            if len(names)!=1:raise ValueError('ambiguous scorer embedding table')
            embedding_shape=list(handle.get_slice(names[0]).get_shape())
        if embedding_shape != [config.vocab_size,1024] or not token_ids:
            raise ValueError('scorer tokenizer/config/embedding metadata disagree')
        result={'format':'osnip_summary_scorer_verification_v1','status':'snapshot_verified_score_pending',
                'observed_at':datetime.datetime.now().astimezone().isoformat(),'security_accepted':False,
                'key_attack_performed':False,'provenance':provenance,'architecture':config.model_type,
                'layers':config.num_hidden_layers,'width':config.hidden_size,'embedding_shape':embedding_shape,
                'tokenizer_ids':token_ids,'bertscore':None,
                'limitations':['Scorer smoke is infrastructure verification, not an OSNIP result.',
                               'ModelScope mirror equivalence to a particular HF revision is unproven.']}
        if not args.snapshot_only:
            save('running_cpu_paired_scorer_smoke')
            import torch
            torch.set_num_threads(2)
            clean=['A small cat sleeps on the mat.','The train arrives at noon.']
            encrypted=[clean[0],'Blue oceans surround the island.']
            score=compute_paired_bertscore(clean,encrypted,clean,'roberta-large',torch.device('cpu'))
            if (not score.get('available') or score['clean']['f1']<.99
                    or not score['rows'][1]['encrypted_f1']<score['rows'][1]['clean_f1']):
                raise ValueError('scorer clean/disrupted controls failed')
            result.update(status='paired_scorer_verified',bertscore=score)
        if resolve_scorer_snapshot('roberta-large')[1]!=provenance:
            raise ValueError('scorer snapshot changed during verification')
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        save(result['status'])
    except Exception as exc:
        state.update(exception=type(exc).__name__,detail=str(exc));save('failed');raise


if __name__=='__main__':main()
