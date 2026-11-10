#!/usr/bin/env python3
"""Wait for one identified legacy run, then start the pinned v2 pilot serially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from osnip_repro.process_inspection import command_tokens, process_command
EXPECTED_OUTPUT = (ROOT / 'reports/llama32_1b/gated_tau02_extend10k').resolve()
EXPECTED_REPORT = EXPECTED_OUTPUT / 'report.json'
PILOT = ROOT / 'scripts/run_question_disjoint_pilot.py'


def validate_run(report_path: Path, *, expected_output: Path = EXPECTED_OUTPUT) -> dict:
    if not report_path.is_file():
        raise RuntimeError(f'legacy run ended without final report: {report_path}')
    report = json.loads(report_path.read_text())
    if report.get('status') != 'completed_with_limits' or report.get('steps_completed') != 10000:
        raise RuntimeError('legacy run did not complete exactly 10,000 steps')
    if Path(report.get('artifacts', {}).get('last.pt', '')).resolve() != (expected_output / 'last.pt'):
        raise RuntimeError('legacy report artifact path does not match the monitored run')
    if not (expected_output / 'last.pt').is_file() or not report_path.is_file():
        raise RuntimeError('legacy run final artifacts are incomplete')
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--poll-seconds', type=int, default=30)
    ap.add_argument('--state-file', type=Path, default=ROOT / 'reports/llama32_1b/post_legacy_v2_continuation.json')
    ap.add_argument('--execute', action='store_true', help='after verified completion, execute Gate 0 -> Gate 1 -> v2 pilot')
    args = ap.parse_args()
    if args.pid < 1 or args.poll_seconds < 1:
        raise ValueError('pid and poll interval must be positive')
    if args.state_file.exists():
        raise FileExistsError(f'refusing to overwrite continuation state: {args.state_file}')

    state = {'format': 'osnip_post_legacy_continuation_v1', 'pid': args.pid,
             'expected_output': str(EXPECTED_OUTPUT), 'execute': args.execute,
             'status': 'waiting_for_identified_process', 'security_accepted': False}
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(state, ensure_ascii=False), flush=True)

    while True:
        command = process_command(args.pid)
        if command is None:
            break
        tokens = command_tokens(command)
        try:
            output_arg = tokens[tokens.index('--output-dir') + 1]
            command_output = Path(output_arg)
            if not command_output.is_absolute():
                command_output = (ROOT / command_output).resolve()
        except (ValueError, IndexError):
            command_output = Path('/__invalid_training_output__')
        if 'scripts/train_streaming.py' not in command or command_output.resolve() != EXPECTED_OUTPUT:
            state.update(status='stopped_pid_identity_mismatch', observed_command=command)
            args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
            raise RuntimeError('monitored PID no longer identifies the expected legacy training run')
        time.sleep(args.poll_seconds)

    report = validate_run(EXPECTED_REPORT, expected_output=EXPECTED_OUTPUT)
    state.update(status='legacy_run_completed', steps_completed=report['steps_completed'],
                 completed_run_report=str(EXPECTED_REPORT))
    args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    command = [sys.executable, str(PILOT), '--model-path', str(ROOT / 'model/Llama-3.2-1B'),
               '--config', str(ROOT / 'configs/llama32_1b_question_disjoint_v2.json'),
               '--data-manifest', str(ROOT / 'data/osnip_65k_question_disjoint_v2/data_manifest.json'),
               '--output-dir', str(ROOT / 'reports/llama32_1b/question_disjoint_v2_pilot'), '--execute']
    state['status'] = 'starting_v2_gates_and_pilot'
    state['pilot_command'] = command
    args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    subprocess.run(command, check=True)
    state['status'] = 'v2_pilot_completed'
    state['pilot_report'] = str(ROOT / 'reports/llama32_1b/question_disjoint_v2_pilot/report.json')
    args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
