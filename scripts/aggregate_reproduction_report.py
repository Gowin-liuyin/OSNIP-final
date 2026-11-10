#!/usr/bin/env python3
"""Aggregate OSNIP run/evaluation artifacts under strict acceptance gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from osnip_repro.checkpointing import recipe_fingerprint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--run-root", type=Path)
    scope.add_argument("--run-dir", action="append", type=Path, help="explicit run(s); avoid mixing exploratory variants")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--max-ppl-ratio", type=float, default=1.25)
    ap.add_argument("--max-paper-ppl-ratio", type=float, default=None,
                    help="optional gate on ppl_ratio_vs_paper_clean (paper Table 2 3B/1B about 1.306)")
    ap.add_argument("--max-mean-kl", type=float, default=0.10)
    ap.add_argument("--max-knn-top10", type=float, default=0.10)
    ap.add_argument("--max-vma-total-asr", type=float, default=0.10)
    ap.add_argument("--max-vma-clean-asr", type=float, default=0.10)
    ap.add_argument("--min-positive-control-asr", type=float, default=0.95)
    ap.add_argument("--min-task-retention", type=float, default=0.95)
    ap.add_argument("--min-utility-samples", type=int, default=1000)
    ap.add_argument("--min-task-samples", type=int, default=1000)
    ap.add_argument("--min-ppl-samples", type=int, default=100)
    ap.add_argument("--min-knn-samples", type=int, default=1000)
    ap.add_argument("--min-vma-prompts", type=int, default=200)
    ap.add_argument("--required-vma-layers", type=int, nargs="+", default=[1, 5, 10, 15])
    ap.add_argument("--required-keys-per-prompt", type=int, default=5)
    ap.add_argument("--expected-steps", type=int)
    ap.add_argument("--expected-effective-batch", type=int)
    ap.add_argument("--expected-seed", type=int)
    ap.add_argument("--expected-diversity-space", choices=("raw_r", "projected_z"))
    ap.add_argument("--require-last-checkpoint", action="store_true")
    ap.add_argument("--expected-vma-max-tokens", type=int)
    ap.add_argument("--expected-ppl-max-length", type=int)
    ap.add_argument("--expected-ppl-stride", type=int)
    ap.add_argument("--expected-paper-clean-ppl", type=float)
    ap.add_argument("--expected-task-source-split", choices=("train", "validation", "test"))
    ap.add_argument("--expected-task-max-length", type=int)
    ap.add_argument("--expected-utility-max-length", type=int)
    return ap.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def max_finite(values: list[float]) -> float | None:
    return max(values) if values else None


def min_finite(values: list[float]) -> float | None:
    return min(values) if values else None



def complete_locked_ppl_snapshot(report: dict[str, Any] | None) -> bool:
    """Verify complete coverage of the repository's pinned article snapshot.

    A 62-article full corpus must not be confused with a short 62-row prefix.
    This exception requires the fixed manifest/data hashes and every causal
    target in every article; arbitrary user-supplied manifests cannot waive the
    normal sample floor.
    """
    if not report:
        return False
    root = Path(__file__).resolve().parents[1]
    locked = read_json(root / "configs/llama32_1b_reproduction.json") or {}
    identity = locked.get("data", {})
    manifest_path = root / "data/wikitext2/manifest.json"
    expected_manifest = identity.get("wikitext2_manifest_sha256")
    expected_data = identity.get("wikitext2_test_sha256")
    if not expected_manifest or not expected_data or not manifest_path.is_file():
        return False
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest:
        return False
    manifest = read_json(manifest_path) or {}
    snapshot = manifest.get("splits", {}).get("test", {})
    data_path = manifest_path.parent / snapshot.get("file", "")
    if not data_path.is_file() or hashlib.sha256(data_path.read_bytes()).hexdigest() != expected_data:
        return False
    lengths = report.get("sequence_token_lengths", [])
    samples = report.get("samples")
    return bool(
        report.get("data_sha256") == expected_data == snapshot.get("sha256")
        and samples == snapshot.get("records") and isinstance(samples, int) and samples > 0
        and isinstance(lengths, list) and len(lengths) == samples
        and all(isinstance(length, int) and length >= 2 for length in lengths)
        and report.get("target_tokens") == sum(length - 1 for length in lengths)
    )




CONDITION_FIELDS = ('model_path','train_sha256','valid_sha256','encryptor_architecture',
    'diversity_space','constraint_mode','lambda_util','lambda_priv','lambda_div',
    'key_dim','hidden_dim','max_length','learning_rate','steps','epsilon','diversity_margin')
OPTIONAL_CONDITION_FIELDS = ('batch_size','gradient_accumulation_steps','weight_decay',
    'warmup_steps','tau_low','tau_high','max_train_samples','source_hashes')
EVAL_CONDITION_FIELDS = ('test_sha256','task_source_split','task_max_length','choice_seed',
    'task_min_clean_accuracy','task_min_per_source_retention',
    'utility_data_sha256','utility_max_length',
    'ppl_max_length','ppl_stride','ppl_data_sha256','paper_clean_ppl',
    'knn_data_sha256','vma_data_sha256',
    'vma_layers','vma_keys_per_prompt','vma_max_tokens','summary_data_sha256','summary_max_new_tokens')

# Historical runs predate these switches; absence means the original default.
EXTENSION_CONDITION_DEFAULTS = {
    'lambda_hidden': 0.0, 'hidden_layers': [], 'gradient_balance': 'none',
    'lambda_knn': 0.0, 'knn_negative_count': 64, 'knn_topk': 10, 'knn_margin': 0.05,
    'alpha_init': 0.05, 'alpha_ramp_steps': 0, 'freeze_alpha': False,
    'grad_clip': 1.0, 'dtype': 'float32', 'reset_optimizer': False,
    'loss_warmup_steps': 10, 'constraint_ramp_steps': 1000, 'gate_scope': 'effective_batch',
}


def condition_identity(config: dict, run_dir: Path | None = None, evaluation: dict | None = None) -> dict:
    missing = [name for name in CONDITION_FIELDS if config.get(name) is None]
    values = {name:config.get(name) for name in CONDITION_FIELDS+OPTIONAL_CONDITION_FIELDS}
    values.update({name: config.get(name, default) for name, default in EXTENSION_CONDITION_DEFAULTS.items()})
    if evaluation is not None:
        # Evaluation protocol is part of the reproduction condition: pooling a
        # 128-token task run with a 256-token run, or a different KNN/VMA set,
        # must split the aggregate instead of averaging incompatible numbers.
        values['evaluation_protocol'] = {name: evaluation.get(name, '<absent>')
                                          for name in EVAL_CONDITION_FIELDS}
    if isinstance(values['model_path'],str):values['model_path']=str(Path(values['model_path']).resolve())
    phase_plan = read_json(run_dir/'two_phase_plan.json') if run_dir is not None else None
    values['training_recipe'] = ({'kind':'two_phase_diagnostic','phase1_steps':phase_plan.get('phase1_steps'),
                                   'total_steps':phase_plan.get('total_steps')}
                                  if phase_plan else {'kind':'resumed_or_phase_transition' if config.get('resume') else 'fresh'})
    return {'complete':not missing,'missing':missing,'values':values}


def bound_knn_fallback(run_dir: Path, legacy_sources: list[str]) -> dict[str, Any]:
    """Accept a standalone KNN report only when bound to this run's checkpoint."""
    fallback = read_json(run_dir / "knn_report.json") or {}
    if not fallback:
        return {}
    digest = fallback.get("checkpoint_sha256")
    candidates: list[str] = []
    last = run_dir / "last.pt"
    if last.is_file():
        candidates.append(hashlib.sha256(last.read_bytes()).hexdigest())
    selected = read_json(run_dir / "selected_pareto.json") or read_json(run_dir / "selected_utility.json") or {}
    selected_path = (selected.get("selected") or {}).get("path")
    if isinstance(selected_path, str) and Path(selected_path).is_file():
        candidates.append(hashlib.sha256(Path(selected_path).read_bytes()).hexdigest())
    if digest not in candidates:
        legacy_sources.append("knn:knn_report.json(unbound-ignored)")
        return {}
    if not fallback.get("rows"):
        # Rowless summaries cannot support observation-integrity checks; the
        # counts they carry would feed sample gates without any evidence rows.
        legacy_sources.append("knn:knn_report.json(rowless-ignored)")
        return {}
    legacy_sources.append("knn:knn_report.json(bound)")
    return fallback


def report_provenance(run_dir: Path, config: dict, utility: dict | None, task: dict | None,
                      ppl: dict | None, privacy: dict | None, *, require_last: bool = False) -> dict:
    """Require the same existing encoder and server model in all final reports.

    This checks local artifact consistency, not adversarially forged reports.
    Different client draws are permitted; encoder/model/data identities are not.
    """
    errors = []
    sources = {'utility':utility,'task':task,'ppl':ppl,'privacy':privacy}
    if any(not isinstance(r,dict) for r in sources.values()):
        return {'passed':False,'errors':['all four evaluation reports are required']}
    utility_options = utility.get('configuration',{})
    selected_last = read_json(run_dir/'selected_last.json')
    selected = selected_last or read_json(run_dir/'selected_pareto.json')
    selected_path = (selected or {}).get('selected',{}).get('path')
    if require_last:
        if not selected_last:
            errors.append('paper-final selection requires selected_last.json')
        if not isinstance(selected_path, str) or Path(selected_path).name != 'last.pt':
            errors.append('paper-final selection must be last.pt')
    reference = selected_path or utility_options.get('checkpoint')
    checkpoint = Path(reference).resolve() if isinstance(reference,str) and reference else None
    if checkpoint is None or not checkpoint.is_file():
        return {'passed':False,'errors':['selected or utility checkpoint path is missing']}
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_model = config.get('model_path')
    for name,report in sources.items():
        options = report.get('configuration',{}) if name=='utility' else report
        recorded_path = options.get('checkpoint')
        if (not isinstance(recorded_path,str) or Path(recorded_path).resolve()!=checkpoint
                or report.get('checkpoint_sha256')!=digest):
            errors.append(name+': checkpoint identity differs or is missing')
        model = options.get('model_path')
        if (not isinstance(model,str) or not isinstance(expected_model,str)
                or Path(model).resolve()!=Path(expected_model).resolve()):
            errors.append(name+': server model identity differs or is missing')
    manifests = {report.get('model_manifest_sha256') for report in sources.values()
                 if report.get('model_manifest_sha256') is not None}
    if len(manifests) > 1:
        errors.append('evaluation reports disagree on the model manifest')
    run_manifest = run_dir / 'model_manifest.json'
    if run_manifest.is_file():
        expected_manifest = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
        for name, report in sources.items():
            recorded = report.get('model_manifest_sha256')
            if recorded is not None and recorded != expected_manifest:
                errors.append(f"{name}: model manifest differs from the run manifest")
    evaluator_sources = {
        'utility': ('scripts/evaluate_no_key.py', 'src/osnip_repro/modeling.py'),
        'task': ('scripts/evaluate_task_utility.py',),
        'ppl': ('scripts/evaluate_sliding_ppl.py', 'src/osnip_repro/modeling.py'),
        'privacy': ('scripts/evaluate_paper_protocol.py', 'src/osnip_repro/vma.py'),
    }
    for name, keys in evaluator_sources.items():
        recorded = sources[name].get('source_sha256') or {}
        for key in keys:
            path = ROOT / key
            if not path.is_file() or recorded.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
                errors.append(f"{name}: evaluator source {key} is missing or stale")
    task_rows = task.get('rows')
    if not isinstance(task_rows, list) or not task_rows:
        errors.append('task: per-row evidence is missing')
    else:
        if len(task_rows) != task.get('records_scored'):
            errors.append('task: row count differs from records_scored')
        if any(row.get('clean_correct') != (row.get('clean_prediction') == row.get('correct'))
               or row.get('encrypted_correct') != (row.get('encrypted_prediction') == row.get('correct'))
               for row in task_rows):
            errors.append('task: row correctness disagrees with predictions')
        clean_hits = sum(row.get('clean_correct') is True for row in task_rows)
        encrypted_hits = sum(row.get('encrypted_correct') is True for row in task_rows)
        clean_accuracy = clean_hits / len(task_rows)
        retention = encrypted_hits / clean_hits if clean_hits else 0.0
        if not finite(task.get('clean_accuracy')) or abs(float(task['clean_accuracy']) - clean_accuracy) > 1e-9:
            errors.append('task: clean accuracy differs from row evidence')
        if not finite(task.get('accuracy_retention_vs_clean')) or abs(float(task['accuracy_retention_vs_clean']) - retention) > 1e-9:
            errors.append('task: retention differs from row evidence')
    for source, path in (task.get('sources') or {}).items():
        if not isinstance(path, str) or not Path(path).is_file():
            errors.append(f"task: raw source path is missing: {source}")
        elif (task.get('source_sha256') or {}).get(f"raw/{source}") != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            errors.append(f"task: raw source bytes differ: {source}")
    config_path = run_dir/'config.json'
    if not config_path.is_file():config_path=run_dir/'train_config.json'
    if (not config_path.is_file() or utility.get('checkpoint_config_sha256')
            != hashlib.sha256(config_path.read_bytes()).hexdigest()):
        errors.append('utility: checkpoint config fingerprint differs or is missing')
    run_config = read_json(config_path) or {}
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except Exception as exc:
        errors.append(f'checkpoint payload unreadable: {type(exc).__name__}: {exc}')
        payload = None
    payload_config = payload.get('config', {}) if isinstance(payload, dict) else {}
    if payload_config and recipe_fingerprint(payload_config) != recipe_fingerprint(run_config):
        errors.append('checkpoint payload recipe differs from run config')
    if require_last and isinstance(payload, dict):
        expected_step = run_config.get('steps', run_config.get('formal_steps'))
        if expected_step is not None and payload.get('global_step') != int(expected_step):
            errors.append('last.pt is not at the configured final step')
    bindings = [('utility',utility_options.get('data_jsonl'),utility.get('data_sha256')),
                ('task',task.get('test_jsonl'),task.get('test_sha256')),
                ('ppl',ppl.get('data_jsonl'),ppl.get('data_sha256')),
                ('knn',privacy.get('knn_jsonl'),privacy.get('data_sha256',{}).get('knn')),
                ('vma',privacy.get('vma_jsonl'),privacy.get('data_sha256',{}).get('vma'))]
    for name,path,expected in bindings:
        if (not isinstance(path,str) or not Path(path).is_file()
                or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected):
            errors.append(name+': evaluation data fingerprint differs or is missing')
    # Utility KL and task accuracy can use different carriers (the task set is
    # the official test split; the KL set is the run's held-out text). Each is
    # bound above and both hashes enter the condition identity, so do not force
    # them to be byte-identical.
    return {'passed':not errors,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'errors':errors}


def verify_privacy_observations(report: dict | None) -> dict:
    """Check complete row coverage and recompute rates; this is not proof against forged reports."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def integer(value, minimum=0):
        return type(value) is int and value >= minimum

    def rate_matches(recorded, hits, count):
        return (type(recorded) in (int, float) and math.isfinite(recorded)
                and 0 <= recorded <= 1 and count > 0
                and math.isclose(recorded, hits / count, rel_tol=1e-9, abs_tol=1e-12))

    def distinct_inputs(path, count):
        texts = []
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                text = json.loads(line)['text']
                require(isinstance(text, str) and bool(text.strip()), 'attack inputs must contain nonempty text')
                texts.append(text)
                if len(texts) == count:
                    break
        require(len(texts) == count, 'attack data has fewer rows than declared')
        require(len(set(texts)) == count, 'attack prompts must be distinct')

    try:
        require(isinstance(report, dict), 'privacy report is required')
        require(report.get('format') == 'osnip_paper_protocol_evaluation_v1'
                and report.get('key_attack_performed') is False, 'invalid privacy report scope')
        knn = report['knn']
        rows, n = knn.get('rows'), knn.get('samples')
        require(integer(n, 1) and isinstance(rows, list) and len(rows) == n, 'complete KNN rows are required')
        require(knn.get('full_vocabulary') is True and knn.get('metric') == 'l2'
                and integer(knn.get('vocabulary_size'), 10), 'full-vocabulary L2 KNN metadata is required')
        counts = {side: {k: 0 for k in ('top1', 'top5', 'top10')} for side in ('clean', 'encrypted')}
        positions = 0
        for i, row in enumerate(rows):
            require(type(row.get('index')) is int and row['index'] == i
                    and integer(row.get('positions'), 1), 'KNN row index/positions mismatch')
            positions += row['positions']
            for side in counts:
                values = [row[side][k] for k in counts[side]]
                require(all(integer(v) for v in values) and 0 <= values[0] <= values[1] <= values[2] <= row['positions'],
                        'KNN counts must be bounded and monotonic')
                for k in counts[side]:
                    counts[side][k] += row[side][k]
        require(type(knn.get('positions')) is int and knn['positions'] == positions, 'KNN position total mismatch')
        for side in counts:
            for k, hits in counts[side].items():
                require(type(knn['counts'][side][k]) is int and knn['counts'][side][k] == hits
                        and rate_matches(knn['rates'][side][k], hits, positions), 'KNN summary differs from rows')
        distinct_inputs(report['knn_jsonl'], n)

        vma = report['vma']
        prompts, keys, layers = vma.get('prompts'), vma.get('keys_per_prompt'), vma.get('layers')
        require(integer(prompts, 1) and integer(keys, 1) and isinstance(layers, list) and bool(layers)
                and all(integer(layer, 1) for layer in layers) and len(set(layers)) == len(layers), 'invalid VMA dimensions')
        require(vma.get('full_vocabulary') is True and vma.get('candidate_mode') == 'full', 'full-vocabulary VMA is required')
        require(isinstance(vma.get('summary'), dict) and set(vma['summary']) == {str(layer) for layer in layers},
                'VMA summary layer coverage mismatch')
        require(integer(vma.get('max_tokens'), 1), 'VMA token limit is required')
        token_counts = {}
        side_counts = {}
        for side, field, expected_keys in [('clean_positive_control', 'clean_rows', [-1]),
                                            ('encrypted', 'rows', list(range(keys)))]:
            observations = vma.get(field)
            require(isinstance(observations, list) and len(observations) == prompts * len(layers) * len(expected_keys),
                    'complete VMA ' + field + ' are required')
            seen = set()
            totals = {layer: {'positions': 0, 'content_positions': 0, 'total_top1_hits': 0, 'clean_top1_hits': 0} for layer in layers}
            for row in observations:
                identity = (row['prompt'], row['layer'], row['key'])
                require(all(type(v) is int for v in identity) and 0 <= identity[0] < prompts
                        and identity[1] in layers and identity[2] in expected_keys and identity not in seen,
                        'duplicate or out-of-range VMA observation')
                seen.add(identity)
                fields = ('positions', 'content_positions', 'total_top1_hits', 'clean_top1_hits')
                require(all(integer(row.get(k)) for k in fields) and 0 < row['positions'] <= vma['max_tokens']
                        and row['content_positions'] <= row['positions']
                        and row['clean_top1_hits'] <= row['content_positions']
                        and row['clean_top1_hits'] <= row['total_top1_hits'] <= row['positions']
                        and row['total_top1_hits'] - row['clean_top1_hits'] <= row['positions'] - row['content_positions'],
                        'VMA counts are outside valid bounds')
                require(type(row.get('candidate_coverage')) in (int, float) and row['candidate_coverage'] == 1.,
                        'full VMA observation must cover all candidates')
                lengths = (row['positions'], row['content_positions'])
                require(token_counts.setdefault(identity[0], lengths) == lengths, 'VMA prompt token counts disagree across keys/layers')
                for k in fields:
                    totals[identity[1]][k] += row[k]
            # Unique, in-range identities plus the exact product size establish full Cartesian coverage.
            for layer, values in totals.items():
                summary = vma['summary'][str(layer)][side]
                require(rate_matches(summary.get('total_asr'), values['total_top1_hits'], values['positions'])
                        and rate_matches(summary.get('clean_asr'), values['clean_top1_hits'], values['content_positions']),
                        'VMA summary differs from observation counts')
            side_counts[side] = len(observations)
        distinct_inputs(report['vma_jsonl'], prompts)
        return {'passed': True, 'errors': [], 'knn_rows_verified': n,
                'vma_clean_observations_verified': side_counts['clean_positive_control'],
                'vma_encrypted_observations_verified': side_counts['encrypted']}
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return {'passed': False, 'errors': [str(exc)]}


def summarize_summary_evaluation(run_dir: Path, config: dict, utility: dict | None,
                                 summary: dict | None) -> dict:
    """Verify complete paired summary evidence against the fixed local snapshot."""
    root = Path(__file__).resolve().parents[1]
    policy = read_json(root / 'configs/llama32_1b_summary_evaluation.json') or {}
    errors = []
    if not summary:
        return {'complete': False, 'passed': False, 'errors': ['summarization report is required']}
    def policy_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    manifest_path = policy_path(policy.get('manifest', '/__missing_manifest__'))
    data_path = policy_path(policy.get('data_jsonl', '/__missing_data__'))
    for name, path, expected in [('manifest', manifest_path, policy.get('manifest_sha256')),
                                  ('data', data_path, policy.get('data_sha256'))]:
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('summarization '+name+' fingerprint mismatch')
    source = read_json(manifest_path) or {}
    source_split = ((source.get("source") or {}).get("split") if isinstance(source, dict) else None)
    if policy.get("require_test_split", True) and source_split != "test":
        errors.append("summarization manifest source split is not the paper's test split")
    expected_records = policy.get('records')
    config_path=run_dir/'config.json'
    if not config_path.is_file():config_path=run_dir/'train_config.json'
    rows = summary.get('rows', [])
    if (summary.get('status') != 'completed_with_limits'
            or summary.get('format') != 'osnip_summarization_utility_report_v2'
            or summary.get('records') != expected_records or source.get('records_written') != expected_records
            or not isinstance(rows, list) or len(rows) != expected_records
            or not all(isinstance(row,dict) for row in rows)
            or [row.get('index') for row in rows] != list(range(expected_records or 0))):
        errors.append('complete ordered summarization rows are required')
    if summary.get('data_sha256') != policy.get('data_sha256') or summary.get('data_jsonl') != str(data_path.resolve()):
        errors.append('summary report does not use the fixed data snapshot')
    options = (utility or {}).get('configuration', {})
    checkpoint = Path(options.get('checkpoint', '/__missing_checkpoint__'))
    if (not checkpoint.is_file() or summary.get('checkpoint') != str(checkpoint.resolve())
            or summary.get('checkpoint_sha256') != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            or summary.get('model_path') != config.get('model_path')
            or not config_path.is_file()
            or summary.get('checkpoint_config_sha256') != hashlib.sha256(config_path.read_bytes()).hexdigest()):
        errors.append('summarization checkpoint/model/config binding differs')
    if summary.get('server_input') != 'inputs_embeds_and_attention_mask_only_no_original_input_ids':
        errors.append('server generation must receive no original prompt token IDs')
    clean_rouge = (summary.get('clean') or {}).get('rouge_l')
    if not finite(clean_rouge) or float(clean_rouge) < float(policy.get('min_clean_rouge_l', 0.05)):
        errors.append('clean ROUGE-L is missing or below the absolute floor')
    clean_bertscore = (summary.get('bertscore_clean') or {}).get('f1')
    if clean_bertscore is not None and (not finite(clean_bertscore)
                                        or float(clean_bertscore) < float(policy.get('min_clean_bertscore_f1', 0.5))):
        errors.append('clean BERTScore is below the absolute floor')
    evaluator_source = (summary.get('source_sha256') or {}).get('scripts/evaluate_summarization_utility.py')
    expected_source = hashlib.sha256((root / 'scripts/evaluate_summarization_utility.py').read_bytes()).hexdigest()
    if evaluator_source != expected_source:
        errors.append('summarization evaluator source is stale')
    if summary.get('key_attack_performed') is not False or summary.get('security_accepted') is not False:
        errors.append('summary utility report has an invalid security scope')
    generation = summary.get('generation', {})
    if any(generation.get(k) != policy.get(k) for k in ('max_input_length','max_new_tokens','num_beams')):
        errors.append('summarization generation settings differ from policy')
    bert = summary.get('bertscore', {})
    bert_rows = bert.get('rows', [])
    if (bert.get('available') is not True or bert.get('model') != policy.get('bertscore_model')
            or not bert.get('scorer_hash') or not isinstance(bert_rows, list) or len(bert_rows) != expected_records
            or not all(isinstance(row,dict) for row in bert_rows)
            or [row.get('index') for row in bert_rows] != list(range(expected_records or 0))):
        errors.append('complete paired BERTScore is required')
    scorer = bert.get('provenance', {})
    scorer_manifest_path = Path(scorer.get('manifest', '/__missing_scorer__'))
    scorer_manifest = read_json(scorer_manifest_path) or {}
    scorer_directory = Path(scorer.get('directory', '/__missing_scorer_directory__'))
    scorer_files = scorer.get('files', {})
    required_scorer_files = {'config.json','model.safetensors','tokenizer.json','vocab.json','merges.txt'}
    if (not scorer_manifest_path.is_file()
            or hashlib.sha256(scorer_manifest_path.read_bytes()).hexdigest() != scorer.get('manifest_sha256')
            or scorer_manifest.get('local_dir') != str(scorer_directory.resolve())
            or scorer_manifest.get('model_id') != policy.get('bertscore_model')
            or scorer_manifest.get('files') != scorer_files or not required_scorer_files.issubset(scorer_files)):
        errors.append('locked BERTScore model bytes and manifest are required')
    else:
        for name, metadata in scorer_files.items():
            file=scorer_directory/name
            if (Path(name).name != name or not file.is_file()
                    or hashlib.sha256(file.read_bytes()).hexdigest()!=metadata.get('sha256')):
                errors.append('BERTScore model file fingerprint mismatch: '+name)
    pairs = [('rouge_l', summary.get('clean',{}).get('rouge_l'), summary.get('encrypted',{}).get('rouge_l')),
             ('bertscore_f1', bert.get('clean',{}).get('f1'), bert.get('encrypted',{}).get('f1'))]
    retentions = {}
    for name, clean, encrypted in pairs:
        if not finite(clean) or not finite(encrypted) or clean <= 0:
            errors.append(name+': finite scores and positive clean control are required')
            continue
        retention = float(encrypted)/float(clean)
        reported = summary.get('retention',{}).get(name)
        if not finite(reported) or not math.isclose(retention, reported, rel_tol=1e-6, abs_tol=1e-8):
            errors.append(name+': retention disagrees with actual scores')
        retentions[name] = retention
    if isinstance(rows,list) and rows and all(isinstance(row,dict) for row in rows):
        for side in ('clean','encrypted'):
            values=[row.get(side+'_rouge_l') for row in rows]
            mean=summary.get(side,{}).get('rouge_l')
            if (not all(finite(x) and 0 <= x <= 1 for x in values) or not finite(mean)
                    or not math.isclose(sum(values)/len(values),mean,rel_tol=1e-6,abs_tol=1e-8)):
                errors.append(side+': ROUGE summary differs from per-example rows')
    if isinstance(bert_rows,list) and bert_rows and all(isinstance(row,dict) for row in bert_rows):
        for side in ('clean','encrypted'):
            values=[row.get(side+'_f1') for row in bert_rows]
            mean=bert.get(side,{}).get('f1')
            if (not all(finite(x) for x in values) or not finite(mean)
                    or not math.isclose(sum(values)/len(values),mean,rel_tol=1e-6,abs_tol=1e-8)):
                errors.append(side+': BERTScore summary differs from per-example rows')
    passed = not errors and retentions.get('rouge_l',-1) >= policy.get('min_rouge_l_retention', .85) and retentions.get('bertscore_f1',-1) >= policy.get('min_bertscore_f1_retention', .95)
    return {'complete':not errors,'passed':passed,'errors':errors,'records':summary.get('records'),
            'retention':retentions,'policy':str(root/'configs/llama32_1b_summary_evaluation.json'),
            'scorer_manifest_sha256':scorer.get('manifest_sha256'),
            'limitations':['Project utility gates, not paper-universal thresholds.',
                           'Mirror equivalence to an exact Hugging Face revision is not proven.']}


def main() -> None:
    args = parse_args()
    if args.min_seeds < 1:
        raise ValueError("min-seeds must be positive")
    if args.run_dir:
        run_dirs = list(dict.fromkeys(path.resolve() for path in args.run_dir))
        if any(not path.is_dir() for path in run_dirs):
            raise FileNotFoundError("explicit evaluation run does not exist")
    else:
        run_dirs = sorted(path for path in args.run_root.iterdir() if path.is_dir()) if args.run_root.is_dir() else []
    seed_records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        train = read_json(run_dir / "report.json")
        config = read_json(run_dir / "config.json") or read_json(run_dir / "train_config.json") or {}
        legacy_sources: list[str] = []
        utility = read_json(run_dir / "utility_test.json")
        if utility is None:
            utility = read_json(run_dir / "joint-test.json")
            if utility is not None:
                legacy_sources.append("utility:joint-test.json")
        task = read_json(run_dir / "task_utility_test.json")
        ppl = read_json(run_dir / "sliding_ppl_test.json")
        privacy = read_json(run_dir / "paper_protocol_test.json")
        if privacy is None:
            privacy = read_json(run_dir / "paper-protocol.json")
            if privacy is not None:
                legacy_sources.append("privacy:paper-protocol.json")
        if train is None:
            continue
        summarization_report = read_json(run_dir / "summarization_utility_test.json")
        # Do not fall back to validation-split metrics: the utility gate must
        # be based on the held-out utility report, not training-time validation.
        utility_values = utility.get("utility", {}) if utility else {}
        knn_block = (privacy or {}).get("knn", {})
        if not knn_block:
            knn_block = bound_knn_fallback(run_dir, legacy_sources)
        knn = knn_block.get("rates", {})
        vma_layers = (privacy or {}).get("vma", {}).get("summary", {})
        encrypted_total_asrs: list[float] = []
        encrypted_clean_asrs: list[float] = []
        clean_total_asrs: list[float] = []
        clean_clean_asrs: list[float] = []
        for summary in vma_layers.values():
            encrypted = summary.get("encrypted", {})
            positive = summary.get("clean_positive_control", {})
            for value, target in (
                (encrypted.get("total_asr"), encrypted_total_asrs),
                (encrypted.get("clean_asr"), encrypted_clean_asrs),
                (positive.get("total_asr"), clean_total_asrs),
                (positive.get("clean_asr"), clean_clean_asrs),
            ):
                if finite(value):
                    target.append(float(value))
        full_vma_protocol = bool((privacy or {}).get("vma", {}).get("full_vocabulary", False))
        evaluation_protocol = {
            "test_sha256": (task or {}).get("test_sha256"),
            "task_source_split": (task or {}).get("source_split_filter"),
            "task_max_length": (task or {}).get("max_length"),
            "choice_seed": ((task or {}).get("choice_policy") or {}).get("seed"),
            "task_min_clean_accuracy": ((task or {}).get("task_gate") or {}).get("min_clean_accuracy"),
            "task_min_per_source_retention": ((task or {}).get("task_gate") or {}).get("min_per_source_retention"),
            "utility_data_sha256": (utility or {}).get("data_sha256"),
            "utility_max_length": ((utility or {}).get("configuration") or {}).get("max_length"),
            "ppl_max_length": (ppl or {}).get("max_length"),
            "ppl_stride": (ppl or {}).get("stride"),
            "ppl_data_sha256": (ppl or {}).get("data_sha256"),
            "paper_clean_ppl": (ppl or {}).get("paper_clean_ppl"),
            "knn_data_sha256": ((privacy or {}).get("data_sha256") or {}).get("knn"),
            "vma_data_sha256": ((privacy or {}).get("data_sha256") or {}).get("vma"),
            "vma_layers": tuple(((privacy or {}).get("vma") or {}).get("layers") or ()),
            "vma_keys_per_prompt": ((privacy or {}).get("vma") or {}).get("keys_per_prompt"),
            "vma_max_tokens": ((privacy or {}).get("vma") or {}).get("max_tokens"),
            "summary_data_sha256": (summarization_report or {}).get("data_sha256"),
            "summary_max_new_tokens": (((summarization_report or {}).get("generation") or {}).get("max_new_tokens")),
        }
        record = {
            "run": str(run_dir.resolve()),
            "seed": train.get("seed", train.get("configuration", {}).get("seed", config.get("seed"))),
            "encryptor_architecture": train.get("encryptor_architecture", config.get("encryptor_architecture", "mlp_residual_v1")),
            "train_status": train.get("status"),
            "condition_identity": condition_identity(config,run_dir,evaluation=evaluation_protocol),
            "evaluation_protocol": evaluation_protocol,
            "utility": utility_values,
            "knn": knn,
            "vma_encrypted_total_asr_max": max_finite(encrypted_total_asrs),
            "vma_encrypted_clean_asr_max": max_finite(encrypted_clean_asrs),
            "vma_clean_positive_total_asr_min": min_finite(clean_total_asrs),
            "vma_clean_positive_clean_asr_min": min_finite(clean_clean_asrs),
            "test_utility_present": utility is not None,
            "utility_samples": utility.get("samples") if utility else None,
            "ppl_present": ppl is not None,
            "ppl_samples": ppl.get("samples") if ppl else None,
            "ppl_complete_locked_snapshot": complete_locked_ppl_snapshot(ppl),
            "sliding_ppl_ratio": ppl.get("ppl_ratio") if ppl else None,
            "sliding_ppl_paper_ratio": ppl.get("ppl_ratio_vs_paper_clean") if ppl else None,
            "task_utility_present": task is not None,
            "task_samples": task.get("records_scored") if task else None,
            "artifact_provenance": report_provenance(run_dir,config,utility,task,ppl,privacy,
                                                      require_last=args.require_last_checkpoint),
            "summarization": summarize_summary_evaluation(run_dir, config, utility, summarization_report),
            "task_accuracy_retention": task.get("accuracy_retention_vs_clean") if task else None,
            "task_gate_passed": bool((task or {}).get("task_gate", {}).get("passed", False)),
            "task_per_source_gate": (task or {}).get("per_source_gate"),
            "test_privacy_present": privacy is not None,
            "privacy_observations": verify_privacy_observations(privacy),
            "knn_samples": knn_block.get("prompts", knn_block.get("samples")) if knn_block else None,
            "knn_positions": knn_block.get("positions") if knn_block else None,
            "vma_prompts": (privacy or {}).get("vma", {}).get("prompts"),
            "vma_keys_per_prompt": (privacy or {}).get("vma", {}).get("keys_per_prompt"),
            "full_vma_protocol": full_vma_protocol,
            "vma_valid_layers": [str(layer) for layer, summary in vma_layers.items()
                                 if all(finite(summary.get(side, {}).get(metric))
                                        for side in ("clean_positive_control", "encrypted")
                                        for metric in ("total_asr", "clean_asr"))],
            "security_accepted_in_source": bool((privacy or {}).get("security_accepted", False)),
            "legacy_fallback_sources": legacy_sources,
        }
        seed_records.append(record)

    unique_seeds = sorted({row["seed"] for row in seed_records if row["seed"] is not None})
    checks: dict[str, Any] = {}
    checks["minimum_seed_count"] = {"passed": len(unique_seeds) >= args.min_seeds, "observed": len(unique_seeds), "required": args.min_seeds}
    signatures = {json.dumps(row['condition_identity']['values'],sort_keys=True) for row in seed_records}
    checks["same_reproduction_condition"] = {"passed": bool(seed_records) and len(signatures)==1 and all(row['condition_identity']['complete'] for row in seed_records), "distinct_conditions":len(signatures), "missing_fields":[row['condition_identity']['missing'] for row in seed_records]}
    checks["distinct_training_seeds"] = {"passed":bool(seed_records) and len(unique_seeds)==len(seed_records), "runs":len(seed_records), "unique_seeds":len(unique_seeds)}
    checks["completed_training"] = {"passed":bool(seed_records) and all(row['train_status']=='completed_with_limits' for row in seed_records)}
    checks["test_utility_present"] = {"passed": bool(seed_records) and all(row["test_utility_present"] for row in seed_records)}
    checks["utility_sample_count"] = {"passed": bool(seed_records) and all(isinstance(row["utility_samples"], int) and row["utility_samples"] >= args.min_utility_samples for row in seed_records), "minimum": args.min_utility_samples, "observed": [row["utility_samples"] for row in seed_records]}
    checks["ppl_sample_count"] = {"passed": bool(seed_records) and all(row["ppl_present"] and ((isinstance(row["ppl_samples"], int) and row["ppl_samples"] >= args.min_ppl_samples) or row["ppl_complete_locked_snapshot"]) for row in seed_records), "minimum": args.min_ppl_samples, "observed": [row["ppl_samples"] for row in seed_records], "complete_locked_snapshot": [row["ppl_complete_locked_snapshot"] for row in seed_records]}
    checks["task_utility_present"] = {"passed": bool(seed_records) and all(row["task_utility_present"] for row in seed_records)}
    checks["task_sample_count"] = {"passed": bool(seed_records) and all(type(row["task_samples"]) is int and row["task_samples"] >= args.min_task_samples for row in seed_records), "minimum": args.min_task_samples, "observed": [row["task_samples"] for row in seed_records]}
    checks["artifact_provenance"] = {"passed": bool(seed_records) and all(row["artifact_provenance"]["passed"] for row in seed_records), "per_run": [row["artifact_provenance"] for row in seed_records]}
    checks["test_privacy_present"] = {"passed": bool(seed_records) and all(row["test_privacy_present"] for row in seed_records)}
    checks["privacy_observation_integrity"] = {"passed": bool(seed_records) and all(row["privacy_observations"]["passed"] for row in seed_records), "per_run": [row["privacy_observations"] for row in seed_records]}
    checks["knn_sample_count"] = {"passed": bool(seed_records) and all(isinstance(row["knn_samples"], int) and row["knn_samples"] >= args.min_knn_samples for row in seed_records), "minimum": args.min_knn_samples, "observed": [row["knn_samples"] for row in seed_records]}
    checks["vma_prompt_count"] = {"passed": bool(seed_records) and all(isinstance(row["vma_prompts"], int) and row["vma_prompts"] >= args.min_vma_prompts for row in seed_records), "minimum": args.min_vma_prompts, "observed": [row["vma_prompts"] for row in seed_records]}
    checks["vma_key_count"] = {"passed": bool(seed_records) and all(isinstance(row["vma_keys_per_prompt"], int) and row["vma_keys_per_prompt"] == args.required_keys_per_prompt for row in seed_records), "required": args.required_keys_per_prompt, "observed": [row["vma_keys_per_prompt"] for row in seed_records]}
    checks["full_vma_protocol"] = {"passed": bool(seed_records) and all(row["full_vma_protocol"] for row in seed_records)}

    checks["vma_layer_coverage"] = {"passed": bool(seed_records) and all(
        {str(layer) for layer in args.required_vma_layers}.issubset(row["vma_valid_layers"])
        for row in seed_records), "required": args.required_vma_layers,
        "observed": [row["vma_valid_layers"] for row in seed_records]}

    checks["summarization_evaluation_complete"] = {"passed": bool(seed_records) and all(row["summarization"]["complete"] for row in seed_records), "per_run": [row["summarization"] for row in seed_records]}
    checks["same_summary_scorer"] = {"passed": bool(seed_records) and len({row["summarization"].get("scorer_manifest_sha256") for row in seed_records})==1 and all(row["summarization"].get("scorer_manifest_sha256") for row in seed_records)}
    checks["summarization_utility_gate"] = {"passed": bool(seed_records) and all(row["summarization"]["passed"] for row in seed_records)}

    utility_pass: list[bool] = []
    task_utility_pass: list[bool] = []
    knn_pass: list[bool] = []
    vma_pass: list[bool] = []
    vma_positive_pass: list[bool] = []
    positive_controls: list[bool] = []
    for row in seed_records:
        utility = row["utility"]
        utility_pass.append(
            finite(utility.get("ppl_ratio"))
            and float(utility["ppl_ratio"]) <= args.max_ppl_ratio
            and finite(utility.get("mean_logit_kl"))
            and float(utility["mean_logit_kl"]) <= args.max_mean_kl
        )
        task_utility_pass.append(
            row["task_utility_present"]
            and row["task_gate_passed"]
            and finite(row.get("task_accuracy_retention"))
            and float(row["task_accuracy_retention"]) >= args.min_task_retention
        )
        clean_knn = row["knn"].get("clean", {}) if isinstance(row["knn"], dict) else {}
        encrypted_knn = row["knn"].get("encrypted", {}) if isinstance(row["knn"], dict) else {}
        clean_knn_ok = finite(clean_knn.get("top1")) and float(clean_knn["top1"]) >= args.min_positive_control_asr
        encrypted_knn_ok = finite(encrypted_knn.get("top10")) and float(encrypted_knn["top10"]) <= args.max_knn_top10
        vma_pass.append(
            finite(row["vma_encrypted_total_asr_max"])
            and float(row["vma_encrypted_total_asr_max"]) <= args.max_vma_total_asr
            and finite(row["vma_encrypted_clean_asr_max"])
            and float(row["vma_encrypted_clean_asr_max"]) <= args.max_vma_clean_asr
        )
        vma_positive_pass.append(
            finite(row["vma_clean_positive_total_asr_min"])
            and float(row["vma_clean_positive_total_asr_min"]) >= args.min_positive_control_asr
            and finite(row["vma_clean_positive_clean_asr_min"])
            and float(row["vma_clean_positive_clean_asr_min"]) >= args.min_positive_control_asr
        )
        positive_controls.append(clean_knn_ok and vma_positive_pass[-1])
        knn_pass.append(clean_knn_ok and encrypted_knn_ok)

    ppl_metric_pass = [finite(row["sliding_ppl_ratio"]) and 0 < float(row["sliding_ppl_ratio"]) <= args.max_ppl_ratio for row in seed_records]
    checks["sliding_ppl_gate"] = {"passed": bool(ppl_metric_pass) and all(ppl_metric_pass), "per_seed": ppl_metric_pass, "max_ppl_ratio": args.max_ppl_ratio}
    if args.max_paper_ppl_ratio is not None:
        paper_ppl_pass = [finite(row.get("sliding_ppl_paper_ratio"))
                          and float(row["sliding_ppl_paper_ratio"]) <= args.max_paper_ppl_ratio for row in seed_records]
        checks["paper_ppl_gate"] = {"passed": bool(seed_records) and all(paper_ppl_pass),
                                    "per_seed": paper_ppl_pass, "max_paper_ppl_ratio": args.max_paper_ppl_ratio}
    checks["utility_gate"] = {"passed": bool(utility_pass) and all(utility_pass), "per_seed": utility_pass, "max_ppl_ratio": args.max_ppl_ratio, "max_mean_logit_kl": args.max_mean_kl}
    checks["task_utility_gate"] = {"passed": bool(task_utility_pass) and all(task_utility_pass), "per_seed": task_utility_pass, "min_accuracy_retention": args.min_task_retention}
    checks["knn_positive_controls"] = {"passed": bool(seed_records) and all(
        finite(row["knn"].get("clean", {}).get("top1"))
        and float(row["knn"]["clean"]["top1"]) >= args.min_positive_control_asr for row in seed_records),
        "min_asr": args.min_positive_control_asr}
    checks["knn_gate"] = {"passed": bool(knn_pass) and all(knn_pass), "per_seed": knn_pass, "max_top10": args.max_knn_top10}
    checks["vma_gate"] = {"passed": bool(vma_pass) and all(vma_pass) and checks["full_vma_protocol"]["passed"], "per_seed": vma_pass, "max_total_asr": args.max_vma_total_asr, "max_clean_asr": args.max_vma_clean_asr}
    checks["vma_positive_controls"] = {"passed": bool(vma_positive_pass) and all(vma_positive_pass), "per_seed": vma_positive_pass, "min_asr": args.min_positive_control_asr}
    checks["positive_controls"] = {"passed": bool(positive_controls) and all(positive_controls), "per_seed": positive_controls, "min_asr": args.min_positive_control_asr}

    lock_errors: list[str] = []
    for row in seed_records:
        values = row["condition_identity"]["values"]
        ev = row.get("evaluation_protocol") or {}
        if args.expected_steps is not None and values.get("steps") != args.expected_steps:
            lock_errors.append(f"steps {values.get('steps')} != {args.expected_steps}")
        if args.expected_effective_batch is not None:
            effective = int(values.get("batch_size") or 0) * int(values.get("gradient_accumulation_steps") or 0)
            if effective != args.expected_effective_batch:
                lock_errors.append(f"effective batch {effective} != {args.expected_effective_batch}")
        if args.expected_seed is not None and row.get("seed") != args.expected_seed:
            lock_errors.append(f"seed {row.get('seed')} != {args.expected_seed}")
        if args.expected_diversity_space is not None and values.get("diversity_space") != args.expected_diversity_space:
            lock_errors.append(f"diversity_space {values.get('diversity_space')} != {args.expected_diversity_space}")
        if args.expected_vma_max_tokens is not None and ev.get("vma_max_tokens") != args.expected_vma_max_tokens:
            lock_errors.append(f"vma_max_tokens {ev.get('vma_max_tokens')} != {args.expected_vma_max_tokens}")
        if args.expected_ppl_max_length is not None and ev.get("ppl_max_length") != args.expected_ppl_max_length:
            lock_errors.append(f"ppl_max_length {ev.get('ppl_max_length')} != {args.expected_ppl_max_length}")
        if args.expected_ppl_stride is not None and ev.get("ppl_stride") != args.expected_ppl_stride:
            lock_errors.append(f"ppl_stride {ev.get('ppl_stride')} != {args.expected_ppl_stride}")
        if args.expected_paper_clean_ppl is not None:
            value = ev.get("paper_clean_ppl")
            if not finite(value) or abs(float(value) - args.expected_paper_clean_ppl) > 1e-9:
                lock_errors.append(f"paper_clean_ppl {value} != {args.expected_paper_clean_ppl}")
        if args.expected_task_source_split is not None and ev.get("task_source_split") != args.expected_task_source_split:
            lock_errors.append(f"task_source_split {ev.get('task_source_split')} != {args.expected_task_source_split}")
        if args.expected_task_max_length is not None and ev.get("task_max_length") != args.expected_task_max_length:
            lock_errors.append(f"task_max_length {ev.get('task_max_length')} != {args.expected_task_max_length}")
        if args.expected_utility_max_length is not None and ev.get("utility_max_length") != args.expected_utility_max_length:
            lock_errors.append(f"utility_max_length {ev.get('utility_max_length')} != {args.expected_utility_max_length}")
    checks["recipe_lock"] = {"passed": bool(seed_records) and not lock_errors, "errors": lock_errors,
                             "expected": {"steps": args.expected_steps, "effective_batch": args.expected_effective_batch,
                                          "seed": args.expected_seed, "diversity_space": args.expected_diversity_space,
                                          "vma_max_tokens": args.expected_vma_max_tokens,
                                          "ppl_max_length": args.expected_ppl_max_length, "ppl_stride": args.expected_ppl_stride,
                                          "paper_clean_ppl": args.expected_paper_clean_ppl,
                                          "task_source_split": args.expected_task_source_split,
                                          "task_max_length": args.expected_task_max_length,
                                          "utility_max_length": args.expected_utility_max_length}}

    all_passed = bool(seed_records) and all(item.get("passed", False) for item in checks.values())
    privacy_failed = checks["positive_controls"]["passed"] and (not checks["knn_gate"]["passed"] or not checks["vma_gate"]["passed"])
    utility_failed = (not checks["utility_gate"]["passed"] or not checks["task_utility_gate"]["passed"] or not checks["sliding_ppl_gate"]["passed"]
                      or (checks["summarization_evaluation_complete"]["passed"] and not checks["summarization_utility_gate"]["passed"]))
    status = "security_accepted" if all_passed else (
        "inconclusive" if not seed_records or not checks["minimum_seed_count"]["passed"] else
        "utility_failed" if utility_failed else
        "privacy_failed" if privacy_failed else
        "incomplete_security_evaluation"
    )
    report = {
        "format": "osnip_final_reproduction_report_v2",
        "status": status,
        "security_accepted": status == "security_accepted",
        "knn_privacy_failed": checks["knn_positive_controls"]["passed"] and not checks["knn_gate"]["passed"],
        "all_evaluations_complete": all(checks[name]["passed"] for name in (
            "completed_training", "same_reproduction_condition", "distinct_training_seeds", "test_utility_present", "utility_sample_count", "task_utility_present", "task_sample_count", "artifact_provenance", "ppl_sample_count",
            "test_privacy_present", "privacy_observation_integrity", "knn_sample_count", "vma_prompt_count", "vma_key_count",
            "full_vma_protocol", "vma_layer_coverage", "summarization_evaluation_complete", "same_summary_scorer")),
        "threat_model": "passive_no_key_semi_honest_server",
        "key_attack_performed": False,
        "run_root": str(args.run_root.resolve()) if args.run_root is not None else None,
        "seeds": unique_seeds,
        "checks": checks,
        "runs": seed_records,
        "thresholds": {
            "max_ppl_ratio": args.max_ppl_ratio,
            "max_mean_kl": args.max_mean_kl,
            "max_knn_top10": args.max_knn_top10,
            "max_vma_total_asr": args.max_vma_total_asr,
            "max_vma_clean_asr": args.max_vma_clean_asr,
            "min_positive_control_asr": args.min_positive_control_asr,
            "min_task_retention": args.min_task_retention,
            "min_utility_samples": args.min_utility_samples,
            "min_task_samples": args.min_task_samples,
            "min_ppl_samples": args.min_ppl_samples,
            "min_knn_samples": args.min_knn_samples,
            "min_vma_prompts": args.min_vma_prompts,
            "required_keys_per_prompt": args.required_keys_per_prompt,
            "required_vma_layers": args.required_vma_layers,
        },
        "limitations": ["aggregation cannot repair missing or invalid source artifacts", "security acceptance requires test reports and all configured seeds", "both VMA Total ASR and Clean ASR are required", "no key attack is in scope"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_reproduction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown = ["# OSNIP Llama-3.2-1B final reproduction report", "", f"Status: `{status}`", f"Security accepted: `{report['security_accepted']}`", "", "## Gates", ""]
    for name, check in checks.items():
        markdown.append(f"- `{name}`: `{check.get('passed')}`")
    markdown += ["", "## Scope", "", "The report covers the passive no-key semi-honest server. Key attacks are excluded.", ""]
    (args.output_dir / "final_reproduction_report.md").write_text("\n".join(markdown))
    print(json.dumps({"status": status, "security_accepted": report["security_accepted"], "seeds": unique_seeds, "output": str((args.output_dir / "final_reproduction_report.json").resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
