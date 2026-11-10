#!/usr/bin/env python3
"""Run the full held-out evaluation bound to the question-disjoint v2 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from verify_question_disjoint_training import verify_dataset
from osnip_repro.checkpointing import RECIPE_FIELDS, flatten_recipe, recipe_fingerprint, require_locked_recipe

DEFAULT_CONFIG = ROOT / 'configs/llama32_1b_question_disjoint_v2.json'
DEFAULT_MANIFEST = ROOT / 'data/osnip_65k_question_disjoint_v2/data_manifest.json'
DEFAULT_SUMMARY_POLICY = ROOT / 'configs/llama32_1b_summary_evaluation.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



def verify_training_lineage(run_dir: Path, config: dict, training: dict, *, fixed: dict, model_path: Path) -> list[dict]:
    """Check frozen training checkpoints back to a v2 fresh initialization.

    A same-dataset phase transition is allowed. A missing, cyclic or overwritten
    ancestry cannot establish fresh provenance and must be audited separately.
    This checks artifact consistency, not adversarially forged metadata.
    """
    import torch

    identity = ("train_sha256", "valid_sha256", "seed", "encryptor_architecture", "diversity_space")
    expected = {name: config.get(name) for name in identity}
    if any(expected[name] != fixed["data"][name] for name in ("train_sha256", "valid_sha256")):
        raise ValueError("training lineage origin must use the pinned v2 data")
    if any(expected[name] is None for name in identity):
        raise ValueError("training lineage identity fields are missing")
    missing = [name for name in RECIPE_FIELDS if name not in flatten_recipe(fixed)]
    if missing:
        raise ValueError("pinned v2 config does not lock recipe fields: " + ", ".join(missing))
    require_locked_recipe(fixed)
    if recipe_fingerprint(fixed) != recipe_fingerprint(config):
        raise ValueError("run config recipe differs from the pinned v2 recipe")
    if config.get("model_path") is None or Path(config["model_path"]).resolve() != model_path.resolve():
        raise ValueError("training config uses a different model path")
    if training.get("resumed_from") != config.get("resume"):
        raise ValueError("training report/config resume provenance disagrees")
    lineage, seen = [], set()
    path = run_dir / "last.pt"
    ceiling = int(training["steps_completed"]) + 1
    while True:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError("cyclic or overwritten resume ancestry; fresh v2 origin is unproven")
        seen.add(resolved)
        if not resolved.is_file():
            raise ValueError("missing resume lineage checkpoint: " + str(resolved))
        before = sha256(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if sha256(resolved) != before:
            raise ValueError("resume lineage checkpoint changed while reading")
        if payload.get("format") != "osnip_training_checkpoint_v1":
            raise ValueError("unsupported resume lineage checkpoint format")
        recorded = payload.get("config", {})
        if any(recorded.get(name) != value for name, value in expected.items()):
            raise ValueError("resume lineage uses different data, seed, architecture or diversity space")
        if recipe_fingerprint(recorded) != recipe_fingerprint(config):
            raise ValueError("resume lineage recipe differs from the run config")
        if recorded.get("model_path") is None or Path(recorded["model_path"]).resolve() != model_path.resolve():
            raise ValueError("resume lineage model identity differs")
        step = payload.get("global_step")
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < ceiling:
            raise ValueError("resume lineage steps are not strictly increasing")
        if not lineage and step != training["steps_completed"]:
            raise ValueError("last checkpoint does not match completed training steps")
        if not lineage and recorded.get("resume") != config.get("resume"):
            raise ValueError("last checkpoint and run config ancestry disagree")
        lineage.append({"checkpoint": str(resolved), "sha256": before, "global_step": step,
                        "resume": recorded.get("resume")})
        if "resume" not in recorded:
            raise ValueError("checkpoint lacks explicit fresh/resume origin")
        if recorded["resume"] is None:
            break
        path = Path(recorded["resume"])
        if not path.is_absolute():
            raise ValueError("resume lineage must record absolute checkpoint paths")
        ceiling = step
    return lineage


def build_plan(*, model_path: Path, run_dir: Path, config_path: Path, manifest_path: Path,
               device: str = 'mps', dtype: str = 'bfloat16', utility_batch_size: int = 1,
               summary_policy_path: Path = DEFAULT_SUMMARY_POLICY,
               task_test_jsonl: Path | None = None) -> dict:
    if utility_batch_size < 1:
        raise ValueError('utility batch size must be positive')
    fixed = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    train_config_path = run_dir / 'config.json'
    report_path = run_dir / 'report.json'
    if not train_config_path.is_file() or not report_path.is_file():
        raise FileNotFoundError('completed training report and config.json are required')
    train_config = json.loads(train_config_path.read_text())
    training = json.loads(report_path.read_text())
    if fixed['data'].get('dataset_policy') != 'question_disjoint_v2' or fixed['data'].get('fresh_training_required') is not True:
        raise ValueError('expected a fresh-training v2 reproduction config')
    if Path(fixed['data']['manifest_path']).resolve() != manifest_path.resolve():
        raise ValueError('selected v2 manifest differs from pinned config')
    if manifest.get('status') != 'ready_for_fresh_training_only':
        raise ValueError('v2 manifest is not ready')
    verify_dataset(manifest)
    for split in ('train', 'valid', 'test'):
        path = manifest_path.parent / f'{split}.jsonl'
        expected = fixed['data'][f'{split}_sha256']
        if manifest['splits'][split]['file_sha256'] != expected or sha256(path) != expected:
            raise ValueError(f'v2 {split} data fingerprint mismatch')
    if sha256(manifest_path) != fixed['data']['manifest_sha256']:
        raise ValueError('v2 manifest fingerprint mismatch')
    if train_config.get('train_sha256') != fixed['data']['train_sha256'] or train_config.get('valid_sha256') != fixed['data']['valid_sha256']:
        raise ValueError('training run was not trained on the pinned v2 train/valid splits')
    expected_steps = fixed['training']['formal_steps']
    if train_config.get('steps') != expected_steps or training.get('steps_completed') != expected_steps:
        raise ValueError(f'formal v2 evaluation requires exactly {expected_steps} completed training steps')
    if training.get('status') != 'completed_with_limits' or training.get('security_accepted') is not False:
        raise ValueError('training run is incomplete or has an unexpected status')
    if Path(training.get('model_path', '')).resolve() != model_path.resolve():
        raise ValueError('training report uses a different model path')
    if train_config.get('diversity_space') != 'raw_r':
        raise ValueError('v2 main evaluation requires raw_r Eq.19 diversity')

    lineage = verify_training_lineage(run_dir, train_config, training, fixed=fixed, model_path=model_path)

    summary = json.loads(summary_policy_path.read_text())
    summary_manifest = Path(summary['manifest'])
    summary_data = Path(summary['data_jsonl'])
    if sha256(summary_manifest) != summary['manifest_sha256'] or sha256(summary_data) != summary['data_sha256']:
        raise ValueError('summarization snapshot fingerprint mismatch')
    summary_source = json.loads(summary_manifest.read_text())
    if (summary_source.get('status') != 'ready' or summary_source.get('output_sha256') != summary['data_sha256']
            or sum(bool(line.strip()) for line in summary_data.open()) != summary['records']
            or summary_source.get('records_written') != summary['records'] or summary.get('require_bertscore') is not True):
        raise ValueError('formal summarization policy requires complete snapshot and BERTScore')

    outputs = ('selected_pareto.json', 'utility_test.json', 'task_utility_test.json',
               'sliding_ppl_test.json', 'paper_protocol_test.json', 'summarization_utility_test.json')
    existing = [name for name in outputs if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(f'evaluation artifacts already exist; choose a clean completed run: {existing}')
    suite = [sys.executable, str(ROOT / 'scripts/run_evaluation_suite.py'),
             '--model-path', str(model_path), '--run-dir', str(run_dir), '--criterion', 'last',
             '--test-jsonl', str(manifest_path.parent / 'test.jsonl'),
             '--ppl-jsonl', str(ROOT / 'data/wikitext2/test.jsonl'),
             '--arc-easy', str(ROOT / 'data/osnip_sources/raw/arc_easy.jsonl'),
             '--arc-challenge', str(ROOT / 'data/osnip_sources/raw/arc_challenge.jsonl'),
             '--sciq', str(ROOT / 'data/osnip_sources/raw/sciq.jsonl'),
             '--summarization-jsonl', str(summary_data), '--max-summarization-samples', str(summary['records']),
             '--summary-max-input-length', str(summary['max_input_length']),
             '--summary-max-new-tokens', str(summary['max_new_tokens']), '--summary-num-beams', str(summary['num_beams']),
             '--bertscore-model', summary['bertscore_model'], '--require-bertscore',
             '--knn-jsonl', str(ROOT / 'data/paper_attack_sets/knn.jsonl'),
             '--vma-jsonl', str(ROOT / 'data/paper_attack_sets/vma.jsonl'),
             '--layers', '1', '5', '10', '15', '--keys-per-prompt', '5',
             '--max-utility-samples', '1000', '--max-knn-samples', '1000', '--max-vma-prompts', '200',
             '--utility-batch-size', str(utility_batch_size), '--max-length', '128',
             '--ppl-max-length', '512', '--ppl-stride', '256', '--paper-clean-ppl', '8.63',
             '--task-source-split', 'test', '--candidate-mode', 'full',
             '--vocab-chunk', '8192', '--candidate-chunk', '256', '--skip-input-vocabulary-probe',
             '--device', device, '--dtype', dtype, '--execute']
    if task_test_jsonl is not None:
        suite += ['--task-test-jsonl', str(task_test_jsonl)]
    return {'format': 'osnip_question_disjoint_evaluation_plan_v1',
            'run_dir': str(run_dir.resolve()), 'model_path': str(model_path.resolve()),
            'test_split': str((manifest_path.parent / 'test.jsonl').resolve()),
            'task_test_split': str(task_test_jsonl.resolve()) if task_test_jsonl is not None else None,
            'test_sha256': fixed['data']['test_sha256'], 'checkpoint_criterion': 'last',
            'training_lineage': lineage, 'fresh_v2_origin_verified': True,
            'full_vocabulary': True, 'knn_samples': 1000, 'vma_prompts': 200,
            'keys_per_prompt': 5, 'vma_layers': [1, 5, 10, 15],
            'summarization_policy': str(summary_policy_path.resolve()),
            'summarization_policy_sha256': sha256(summary_policy_path), 'summarization': summary,
            'confirmatory_sciq_used': False, 'key_attacks_performed': False,
            'security_accepted': False, 'execute': False, 'command': suite,
            'required_after_evaluation': 'aggregate independent fresh seeds and evaluate the reserved SciQ confirmatory set once after candidate/protocol lock'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path', type=Path, default=ROOT / 'model/Llama-3.2-1B')
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--data-manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--utility-batch-size', type=int, default=1)
    ap.add_argument('--summary-policy', type=Path, default=DEFAULT_SUMMARY_POLICY)
    ap.add_argument('--task-test-jsonl', type=Path, default=None,
                    help='official test-split task set from prepare_task_test_set.py')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    plan = build_plan(model_path=args.model_path, run_dir=args.run_dir, config_path=args.config,
                      manifest_path=args.data_manifest, device=args.device, dtype=args.dtype,
                      utility_batch_size=args.utility_batch_size, summary_policy_path=args.summary_policy,
                      task_test_jsonl=args.task_test_jsonl)
    plan_path = args.run_dir / 'question_disjoint_evaluation_plan.json'
    if plan_path.exists():
        raise FileExistsError(plan_path)
    plan['execute'] = args.execute
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'plan': str(plan_path.resolve()), 'execute': args.execute,
                      'test_split': plan['test_split'], 'command': plan['command']}, ensure_ascii=False, indent=2))
    if args.execute:
        subprocess.run(plan['command'], check=True)


if __name__ == '__main__':
    main()
