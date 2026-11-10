#!/usr/bin/env python3
"""Train the independent OSNIP baseline against a frozen causal LM."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from osnip_repro.curriculum import constraint_weight
from osnip_repro.data import read_text_jsonl, tokenize_texts
from osnip_repro.encoder import ARCHITECTURES, KeyConditionedEncryptor, build_encryptor
from osnip_repro.losses import osnip_two_key_loss, utility_kl
from osnip_repro.modeling import forward_logits, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encryptor-architecture", choices=ARCHITECTURES, default="mlp_residual_v1")
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--diversity-margin", type=float, default=1.0)
    parser.add_argument("--diversity-space", choices=("projected_z", "raw_r"), default="projected_z")
    parser.add_argument("--lambda-util", type=float, default=1.0)
    parser.add_argument("--lambda-priv", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--tau-low", type=float, default=0.01)
    parser.add_argument("--tau-high", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--constraint-mode", choices=("gated", "constant", "off"), default="gated")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if min(args.max_samples, args.steps, args.batch_size) < 1 or args.max_length < 2:
        raise ValueError("positive sample/step/batch counts and max_length >= 2 are required")
    if args.learning_rate <= 0 or min(args.lambda_util, args.lambda_priv, args.lambda_div) < 0:
        raise ValueError("learning-rate must be positive and loss weights must be non-negative")
    if (args.output_dir / "encryptor.pt").exists():
        raise FileExistsError("output directory already contains a checkpoint; choose a new output directory")
    torch.manual_seed(args.seed)
    tokenizer, model, device = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    texts = read_text_jsonl(args.data_jsonl, limit=args.max_samples)
    batches = []
    for start in range(0, len(texts), args.batch_size):
        batch = tokenize_texts(tokenizer, texts[start : start + args.batch_size], max_length=args.max_length)
        batches.append({key: value.to(device) for key, value in batch.items()})
    width = int(model.get_input_embeddings().embedding_dim)
    encoder = build_encryptor(width, key_dim=args.key_dim, hidden_dim=args.hidden_dim, architecture=args.encryptor_architecture).to(device)
    if args.initial_checkpoint is not None:
        initial_config = json.loads((args.initial_checkpoint.parent / "config.json").read_text())
        for field, value in (("width", width), ("key_dim", args.key_dim), ("hidden_dim", args.hidden_dim), ("encryptor_architecture", args.encryptor_architecture)):
            if initial_config[field] != value:
                raise ValueError(f"initial checkpoint {field} disagrees with training configuration")
        if Path(initial_config["model_path"]).resolve() != Path(args.model_path).resolve():
            raise ValueError("initial checkpoint belongs to a different server model")
        if initial_config.get("encryptor_architecture", "mlp_residual_v1") != args.encryptor_architecture:
            raise ValueError("initial checkpoint belongs to a different encryptor architecture")
        encoder.load_state_dict(torch.load(args.initial_checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []

    if args.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    batch_order = torch.randperm(len(batches)).tolist()
    for step in range(args.steps):
        if step and step % len(batches) == 0:
            batch_order = torch.randperm(len(batches)).tolist()
        batch = batches[batch_order[step % len(batches)]]
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"].bool()
        with torch.no_grad():
            h = input_embeddings(model, input_ids).detach()
            clean_logits = forward_logits(model, h, mask).detach()
        # Sample one independent user/session key per example, then share it
        # across that example's token positions. A single key for the entire
        # minibatch would not match the paper's per-user conditioning.
        key1 = torch.randn(input_ids.shape[0], args.key_dim, device=device)
        key2 = torch.randn(input_ids.shape[0], args.key_dim, device=device)
        optimizer.zero_grad(set_to_none=True)
        raw_r1 = encoder(h, key1)
        raw_r2 = encoder(h, key2)
        z1 = iso_norm_project(raw_r1, h)
        z2 = iso_norm_project(raw_r2, h)
        encrypted_logits_1 = forward_logits(model, z1, mask)
        encrypted_logits_2 = forward_logits(model, z2, mask)
        # Causal utility compares predictions with valid next-token targets.
        # The last logit has no target here. Representation constraints
        # still cover every non-padding embedding.
        causal_active = (mask[:, 1:] & mask[:, :-1]).reshape(-1)
        clean_active = clean_logits[:, :-1].reshape(-1, clean_logits.shape[-1])[causal_active]
        encrypted_active_1 = encrypted_logits_1[:, :-1].reshape(-1, encrypted_logits_1.shape[-1])[causal_active]
        encrypted_active_2 = encrypted_logits_2[:, :-1].reshape(-1, encrypted_logits_2.shape[-1])[causal_active]
        representation_active = mask.reshape(-1)
        h_active = h.reshape(-1, h.shape[-1])[representation_active]
        z_active = z1.reshape(-1, z1.shape[-1])[representation_active]
        z_other_active = z2.reshape(-1, z2.shape[-1])[representation_active]
        raw_r_active = raw_r1.reshape(-1, raw_r1.shape[-1])[representation_active]
        raw_r_other_active = raw_r2.reshape(-1, raw_r2.shape[-1])[representation_active]
        utility_value = float((utility_kl(clean_active, encrypted_active_1) + utility_kl(clean_active, encrypted_active_2)).mul(0.5).detach())
        if args.constraint_mode == "off":
            privacy_weight = 0.0
            diversity_weight = 0.0
        elif args.constraint_mode == "constant":
            privacy_weight = args.lambda_priv
            diversity_weight = args.lambda_div
        else:
            privacy_weight = constraint_weight(step, utility_value, base=args.lambda_priv, warmup_steps=args.warmup_steps, tau_low=args.tau_low, tau_high=args.tau_high)
            diversity_weight = constraint_weight(step, utility_value, base=args.lambda_div, warmup_steps=args.warmup_steps, tau_low=args.tau_low, tau_high=args.tau_high)
        losses = osnip_two_key_loss(
            clean_active,
            encrypted_active_1,
            encrypted_active_2,
            h_active,
            z_active,
            z_other_active,
            epsilon=args.epsilon,
            lambda_priv=privacy_weight,
            diversity_margin=args.diversity_margin,
            lambda_div=diversity_weight,
            diversity_z1=raw_r_active if args.diversity_space == "raw_r" else None,
            diversity_z2=raw_r_other_active if args.diversity_space == "raw_r" else None,
        )
        losses["total"] = args.lambda_util * losses["utility"] + privacy_weight * losses["privacy"] + diversity_weight * losses["diversity"]
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip))
        optimizer.step()
        history.append({
            "step": step + 1,
            **{name: float(value.detach()) for name, value in losses.items()},
            "privacy_weight": privacy_weight,
            "diversity_weight": diversity_weight,
            "mean_cosine_key1": float(torch.nn.functional.cosine_similarity(h_active.float(), z_active.float(), dim=-1).mean().detach()),
            "mean_cosine_key2": float(torch.nn.functional.cosine_similarity(h_active.float(), z_other_active.float(), dim=-1).mean().detach()),
            "relative_perturbation_key1": float(((z_active.float() - h_active.float()).norm(dim=-1) / h_active.float().norm(dim=-1).clamp_min(1e-12)).mean().detach()),
            "relative_perturbation_key2": float(((z_other_active.float() - h_active.float()).norm(dim=-1) / h_active.float().norm(dim=-1).clamp_min(1e-12)).mean().detach()),
            "gradient_norm_before_clip": grad_norm,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.output_dir / "encryptor.pt")
    config = {name: str(value.resolve()) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config["model_path"] = str(Path(args.model_path).resolve())
    config["data_jsonl"] = str(Path(args.data_jsonl).resolve())
    config.update({"width": width, "model_class": model.__class__.__name__, "status": "engineering_training",
                   "key_schedule": "independent_key_per_example_shared_across_tokens", "diversity_space": args.diversity_space})
    config["data_sha256"] = hashlib.sha256(Path(args.data_jsonl).read_bytes()).hexdigest()
    config["checkpoint_sha256"] = hashlib.sha256((args.output_dir / "encryptor.pt").read_bytes()).hexdigest()
    config["initial_checkpoint_sha256"] = None if args.initial_checkpoint is None else hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
    config["optimizer_resumed"] = False
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    report = {
        "format": "osnip_standalone_training_report_v1",
        "status": "completed_with_limits",
        "paper_result": False,
        "model_path": args.model_path,
        "samples": len(texts),
        "steps": args.steps,
        "width": width,
        "history": history,
        "constraint_mode": args.constraint_mode,
        "active_constraint_steps": sum(row["privacy_weight"] > 0 or row["diversity_weight"] > 0 for row in history),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "data_sha256": config["data_sha256"],
        "security_accepted": False,
        "limitations": ["single local model/data split", "architecture and hyperparameters are a configurable baseline", "attack evaluation and independent holdout are not included", "this is an engineering baseline, not the author implementation"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
