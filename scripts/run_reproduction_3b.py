#!/usr/bin/env python3
"""Build or execute the paper-formal Llama-3.2-3B-Instruct OSNIP run.

The recipe is read from configs/llama32_3b_instruct_reproduction.json and is
checked against PMLR 306 Appendix D.6 / Table 9 before any command is emitted.
This is the missing launcher that makes the 3B recipe independently executable
from the package.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osnip_repro.checkpointing import require_locked_recipe
from osnip_repro.data import file_sha256

DEFAULT_CONFIG = ROOT / "configs/llama32_3b_instruct_reproduction.json"
CONFIG_FIELDS = ("model_path", "train_jsonl", "valid_jsonl", "test_jsonl", "knn_jsonl",
                 "vma_jsonl", "wikitext2_jsonl")

# PMLR Table 9 epsilon/delta are bounds per backbone: 1B <0.4/>0.4, 3B <0.35/>0.5.
BACKBONES = {
    "Llama-3.2-1B": {"width": 2048, "layers": 16, "epsilon_max": 0.4, "delta_min": 0.4},
    "Llama-3.2-3B-Instruct": {"width": 3072, "layers": 28, "epsilon_max": 0.35, "delta_min": 0.5},
}


def backbone_spec(config: dict) -> tuple[str, dict]:
    name = str(config.get("model", ""))
    for key, spec in BACKBONES.items():
        if key in name:
            if int(config.get("width", 0)) != spec["width"] or int(config.get("layers", 0)) != spec["layers"]:
                raise ValueError(f"config width/layers differ from {key}")
            return key, {**spec, "model": key}
    raise ValueError(f"launcher supports only {sorted(BACKBONES)} backbones")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model-path", type=Path, help="override config model_path_local")
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true")
    return ap.parse_args()


def resolve_paths(config: dict, model_override: Path | None, data_root: Path) -> dict[str, Path]:
    model = model_override or Path(config.get("model_path_local", ""))
    data = config["data"]
    return {
        "model_path": model,
        "train_jsonl": data_root / data["train"],
        "valid_jsonl": data_root / data["valid"],
        "test_jsonl": data_root / data["test"],
        "task_test_jsonl": data_root / data["task_test"],
        "knn_jsonl": data_root / data["knn"],
        "vma_jsonl": data_root / data["vma"],
        "wikitext2_jsonl": data_root / data["wikitext2"],
    }


def validate_recipe(config: dict) -> None:
    require_locked_recipe(config)
    objective, training = config["objective"], config["training"]
    _, spec = backbone_spec(config)
    effective = int(training["batch_size"]) * int(training["gradient_accumulation_steps"])
    if effective != int(training["effective_batch"]) or effective != 256:
        raise ValueError(f"paper effective batch must be 256: {effective}")
    if int(training["epochs"]) != 8:
        raise ValueError("paper Table 9 epochs must be 8")
    if int(training["max_length"]) != 128:
        raise ValueError("training max_length must stay 128 until Gen Len 10 is defined")
    expected_steps = math.ceil(int(config.get("training_rows", 65000)) * int(training["epochs"]) / effective)
    if int(training["steps"]) != expected_steps:
        raise ValueError(f"steps {training['steps']} != ceil(rows*epochs/effective_batch) = {expected_steps}")
    if int(config.get("training_rows", 65000)) not in (65000, 65151):
        raise ValueError("training_rows must be the package's 65,000 or the paper's 65,151")
    if training["learning_rate"] != 1e-5:
        raise ValueError("paper Table 9 LR must be 1e-5")
    if training["seed"] != 42:
        raise ValueError("paper run seed must be 42")
    if int(training["warmup_steps"]) != 200:
        raise ValueError("optimizer LR warm-up must be 200 steps")
    if int(objective["constraint_ramp_steps"]) != 1000:
        raise ValueError("constraint ramp must be 1000 steps")
    if objective.get("gate_scope") != "effective_batch" or objective.get("constraint_mode") != "gated":
        raise ValueError("paper curriculum requires gated constraints over the effective batch")
    if objective["lambda_util"] != 1.0:
        raise ValueError("paper Table 9 fixes the utility coefficient at 1.0")
    if float(objective.get("lambda_knn", 0.0)) != 0.0 or float(objective.get("lambda_hidden", 0.0)) != 0.0:
        raise ValueError("paper Eq.27 has no optional KNN or hidden-state losses")
    if objective["lambda_priv"] != 1.0 or objective["lambda_div"] != 2.0:
        raise ValueError("paper Table 9 lambda1/lambda2 must be 1.0/2.0")
    if objective["tau_high"] != 0.15 or objective["tau_low"] != 0.05:
        raise ValueError("paper D.6 tau interval must be [0.05, 0.15]")
    if objective["loss_warmup_steps"] != 10:
        raise ValueError("paper D.6 loss warm-up must be 10")
    if not 0 <= objective["epsilon"] < spec["epsilon_max"]:
        raise ValueError(f"epsilon must satisfy 0 <= epsilon < {spec['epsilon_max']} for {spec['model']}")
    if objective["diversity_margin"] <= spec["delta_min"]:
        raise ValueError(f"diversity margin must be > {spec['delta_min']} for {spec['model']}")
    if objective["diversity_space"] != "raw_r":
        raise ValueError("paper Eq.19 diversity must be evaluated on the raw R_phi pair")


def trainer_command(config: dict, paths: dict[str, Path], output: Path) -> list[str]:
    obj, tr = config["objective"], config["training"]
    return [sys.executable, str(ROOT / "scripts/train_streaming.py"),
            "--model-path", str(paths["model_path"]),
            "--train-jsonl", str(paths["train_jsonl"]),
            "--valid-jsonl", str(paths["valid_jsonl"]),
            "--output-dir", str(output),
            "--steps", str(tr["steps"]),
            "--batch-size", str(tr["batch_size"]),
            "--gradient-accumulation-steps", str(tr["gradient_accumulation_steps"]),
            "--max-length", str(tr["max_length"]),
            "--max-valid-samples", str(tr["max_valid_samples"]),
            "--key-dim", str(tr["key_dim"]),
            "--hidden-dim", str(tr["hidden_dim"]),
            "--alpha-init", str(tr.get("alpha_init", 0.05)),
            "--alpha-ramp-steps", str(tr.get("alpha_ramp_steps", 0)),
            "--encryptor-architecture", str(tr["encryptor_architecture"]),
            "--epsilon", str(obj["epsilon"]),
            "--diversity-margin", str(obj["diversity_margin"]),
            "--diversity-space", str(obj["diversity_space"]),
            "--lambda-util", str(obj["lambda_util"]),
            "--lambda-priv", str(obj["lambda_priv"]),
            "--lambda-div", str(obj["lambda_div"]),
            "--lambda-knn", str(obj.get("lambda_knn", 0.0)),
            "--lambda-hidden", str(obj.get("lambda_hidden", 0.0)),
            "--warmup-steps", str(tr["warmup_steps"]),
            "--constraint-ramp-steps", str(obj["constraint_ramp_steps"]),
            "--loss-warmup-steps", str(obj["loss_warmup_steps"]),
            "--tau-low", str(obj["tau_low"]),
            "--tau-high", str(obj["tau_high"]),
            "--learning-rate", str(tr["learning_rate"]),
            "--weight-decay", str(tr["weight_decay"]),
            "--grad-clip", str(tr["grad_clip"]),
            "--device", str(tr["device"]),
            "--dtype", str(tr["dtype"]),
            "--seed", str(tr["seed"]),
            "--constraint-mode", str(obj["constraint_mode"]),
            "--gate-scope", str(obj["gate_scope"]),
            "--gradient-balance", str(tr["gradient_balance"]),
            "--checkpoint-every", str(tr["checkpoint_every"]),
            "--evaluate-every", str(tr["evaluate_every"])]


def evaluation_command(config: dict, paths: dict[str, Path], run_dir: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts/run_evaluation_suite.py"),
               "--model-path", str(paths["model_path"]), "--run-dir", str(run_dir),
               "--criterion", "last", "--test-jsonl", str(paths["test_jsonl"]),
               "--ppl-jsonl", str(paths["wikitext2_jsonl"]),
               "--paper-clean-ppl", str(config.get("paper_clean_ppl", 9.63)),
               "--knn-jsonl", str(paths["knn_jsonl"]), "--vma-jsonl", str(paths["vma_jsonl"]),
               "--layers", *[str(layer) for layer in config.get("vma_layers", [1, 5, 10, 15, 20])],
               "--keys-per-prompt", "5", "--max-knn-samples", "1000",
               "--max-vma-prompts", "200", "--candidate-mode", "full", "--candidate-chunk", "64",
               "--vma-max-tokens", str(config.get("vma_max_tokens", 16)),
               "--task-source-split", "test",
               "--device", config["training"]["device"], "--dtype", config["training"]["dtype"],
               "--seed", "42", "--skip-input-vocabulary-probe", "--execute"]
    command += ["--task-test-jsonl", str(paths["task_test_jsonl"])]
    summary = ROOT / "data/osnip_sources/cnn_dailymail_test.jsonl"
    if summary.is_file():
        command += ["--summarization-jsonl", str(summary), "--max-summarization-samples", "999",
                    "--require-bertscore"]
    return command


def hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items() if path.is_file()}


def model_file_hashes(model_path: Path) -> dict[str, str]:
    """Bind the model weights/tokenizer, not just the directory path."""
    if not model_path.is_dir():
        return {}
    names = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "generation_config.json")
    files = [model_path / name for name in names if (model_path / name).is_file()]
    files += sorted(model_path.glob("*.safetensors"))
    files += sorted(model_path.glob("pytorch_model*.bin"))
    return {str(path): file_sha256(path) for path in files}


def validate_model_identity(config: dict, model_path: Path) -> None:
    model_config = model_path / "config.json"
    if not model_config.is_file():
        return
    from transformers import AutoConfig
    loaded = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if int(loaded.hidden_size) != int(config["width"]) or int(loaded.num_hidden_layers) != int(config["layers"]):
        raise ValueError("model width/layers differ from the locked config")


def validate_data_identity(config: dict, paths: dict[str, Path]) -> None:
    for name in ("train", "valid", "test"):
        pinned = config.get("data", {}).get(f"{name}_sha256")
        path = paths.get(f"{name}_jsonl")
        if pinned and path is not None and path.is_file() and file_sha256(path) != pinned:
            raise ValueError(f"{name} data bytes differ from the pinned config hash")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    validate_recipe(config)
    paths = resolve_paths(config, args.model_path, args.data_root)
    validate_model_identity(config, paths["model_path"])
    validate_data_identity(config, paths)
    output = args.output_dir.resolve()
    run_dir = output / "train"
    train = trainer_command(config, paths, run_dir)
    evaluate = evaluation_command(config, paths, run_dir)
    plan = {
        "format": "osnip_reproduction_3b_plan_v1",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "model": config["model"],
        "model_path": str(paths["model_path"]),
        "run_dir": str(run_dir),
        "train_command": train,
        "evaluation_command": evaluate,
        "paper_reference": {
            "source": "PMLR 306 Appendix D.6 / Table 9",
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "bs_per_gpu": 64, "paper_gpus": 4, "effective_batch": config["training"]["effective_batch"],
            "paper_data": 65151, "paper_steps": 2036,
            "package_data": config.get("training_rows", 65000), "package_steps": config["training"]["steps"],
            "tau": [config["objective"]["tau_low"], config["objective"]["tau_high"]],
            "loss_warmup_steps": config["objective"]["loss_warmup_steps"],
            "lambda_priv": config["objective"]["lambda_priv"], "lambda_div": config["objective"]["lambda_div"],
            "seed": config["training"]["seed"],
        },
        "deviations": config.get("notes", []),
        "input_sha256": hashes(paths),
        "model_sha256": model_file_hashes(paths["model_path"]),
        "security_accepted": False,
        "execute": args.execute,
    }
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "execution_plan.json"
    if args.execute:
        if plan_path.exists():
            # A dry run writes the plan first; --execute must accept that exact
            # plan (the queue flow depends on it) but refuse a different recipe.
            prior = json.loads(plan_path.read_text())
            if (prior.get("train_command") != train
                    or prior.get("evaluation_command") != evaluate
                    or prior.get("config_sha256") != plan["config_sha256"]):
                raise ValueError("existing execution plan differs from the current locked recipe; regenerate it")
            if prior.get("input_sha256") != plan["input_sha256"]:
                raise ValueError("model or data bytes changed since the execution plan was written")
            if prior.get("model_sha256") != plan["model_sha256"]:
                raise ValueError("model files changed since the execution plan was written")
        else:
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        resume = run_dir / "last.pt"
        if resume.is_file():
            train = train + ["--resume", str(resume)]
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory has no last.pt to resume and is non-empty: {run_dir}")
        subprocess.run(train, check=True, cwd=ROOT)
        if not args.skip_evaluation:
            subprocess.run(evaluate, check=True, cwd=ROOT)
    else:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"plan": str(plan_path), "steps": config["training"]["steps"],
                          "effective_batch": config["training"]["effective_batch"],
                          "learning_rate": config["training"]["learning_rate"],
                          "tau_high": config["objective"]["tau_high"],
                          "diversity_space": config["objective"]["diversity_space"]}, indent=2))


if __name__ == "__main__":
    main()
