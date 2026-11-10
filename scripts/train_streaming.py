#!/usr/bin/env python3
"""Train OSNIP's objective-aligned encryptor with bounded host-memory use.

The frozen LLM is loaded once; JSONL examples are indexed by byte offset and
only the current batch is tokenized. Checkpoints include optimizer/scheduler,
RNG state, data cursor, source hashes, and validation metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from osnip_repro.checkpointing import load_training_checkpoint, recipe_fingerprint, save_training_checkpoint, validate_resume_budget
from osnip_repro.curriculum import constraint_weight, warmup_weight
from osnip_repro.data import TextJsonlIndex, file_sha256, tokenize_texts
from osnip_repro.encoder import ARCHITECTURES, KeyConditionedEncryptor, build_encryptor
from osnip_repro.losses import osnip_two_key_loss, utility_kl, vocabulary_margin_loss
from osnip_repro.modeling import causal_nll_stats, forward_hidden_states, forward_logits, forward_logits_and_hidden_states, input_embeddings, load_causal_lm
from osnip_repro.projection import iso_norm_project
from osnip_repro.validation import aggregate_validation_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--train-jsonl", required=True, type=Path)
    ap.add_argument("--valid-jsonl", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--gate-scope", choices=("microbatch", "effective_batch"), default="effective_batch",
                    help="effective_batch replays keys after a no-grad batch-utility pass; paper Eq.21 uses the batch utility")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-valid-samples", type=int, default=256)
    ap.add_argument("--key-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--alpha-init", type=float, default=0.05,
                    help="initial orthogonal-direction amplitude; unused by residual architectures")
    ap.add_argument("--freeze-alpha", action="store_true",
                    help="freeze orthogonal-direction amplitude after initialization; diagnostic only")
    ap.add_argument("--alpha-ramp-steps", type=int, default=0,
                    help="linearly ramp orthogonal-direction amplitude scale; 0 disables the ramp")
    ap.add_argument("--encryptor-architecture", choices=ARCHITECTURES, default="mlp_residual_v1")
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--diversity-margin", type=float, default=1.0)
    ap.add_argument("--diversity-space", choices=("projected_z", "raw_r"), default="raw_r",
                    help="raw_r is the literal Eq.19 R_phi pair; projected_z is the deployed-space sensitivity")
    ap.add_argument("--lambda-util", type=float, default=1.0)
    ap.add_argument("--lambda-priv", type=float, default=1.0)
    ap.add_argument("--lambda-div", type=float, default=2.0)
    ap.add_argument("--lambda-knn", type=float, default=0.0,
                    help="optional sampled vocabulary-ranking loss; 0 keeps the paper-shaped objective")
    ap.add_argument("--knn-negative-count", type=int, default=64)
    ap.add_argument("--knn-topk", type=int, default=10)
    ap.add_argument("--knn-margin", type=float, default=0.05)
    ap.add_argument("--lambda-hidden", type=float, default=0.0,
                    help="optional layerwise hidden-state recovery weight; 0 keeps the paper-shaped objective unchanged")
    ap.add_argument("--hidden-layers", type=int, nargs="*", default=[],
                    help="decoder layers for optional recovery loss, e.g. --hidden-layers 1 5 10 15")
    ap.add_argument("--warmup-steps", type=int, default=200,
                    help="optimizer learning-rate warm-up steps; PMLR D.6 gives 200")
    ap.add_argument("--constraint-ramp-steps", type=int, default=1000,
                    help="privacy/diversity constraint ramp steps; PMLR D.6 gives 1000 (do not couple to LR warm-up)")
    ap.add_argument("--loss-warmup-steps", type=int, default=10,
                    help="global objective loss warm-up steps; PMLR Appendix D.6 gives 10")
    ap.add_argument("--tau-low", type=float, default=0.05)
    ap.add_argument("--tau-high", type=float, default=0.15)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--constraint-mode", choices=("gated", "constant", "off"), default="gated")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--gradient-balance", choices=("none", "component_norm"), default="none",
                    help="optional per-loss gradient normalization; none keeps the paper-shaped backward pass")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--evaluate-every", type=int, default=100)
    ap.add_argument("--token-id-map", type=Path,
                    help="optional [vocab]-long tensor (e.g. AloePri perm_vocab) applied to token "
                         "ids right after tokenization so embeddings and CE targets use the "
                         "server-side mapped vocabulary")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="resume encoder/data/RNG state but initialize a fresh optimizer and scheduler")
    return ap.parse_args()


RESUME_IMMUTABLE = (
    "seed", "max_length", "batch_size", "gradient_accumulation_steps",
    "max_train_samples", "lambda_util", "lambda_priv", "lambda_div", "epsilon",
    "diversity_margin", "diversity_space", "constraint_mode", "dtype", "gate_scope",
    "loss_warmup_steps", "constraint_ramp_steps", "warmup_steps", "tau_low", "tau_high",
    "learning_rate", "weight_decay", "grad_clip", "gradient_balance",
    "key_dim", "hidden_dim", "alpha_init", "alpha_ramp_steps", "freeze_alpha",
    "encryptor_architecture", "max_valid_samples",
    "lambda_knn", "knn_negative_count", "knn_topk", "knn_margin",
    "lambda_hidden", "hidden_layers",
)
RESUME_DEFAULTS = {"loss_warmup_steps": 10, "constraint_ramp_steps": None, "gate_scope": "microbatch"}


def validate_resume_recipe(prior: dict, args) -> None:
    """Refuse to resume when any experiment-defining field differs from the checkpoint.

    ``prior`` is the config payload embedded in the training checkpoint. Fields
    absent from older payloads fall back to ``RESUME_DEFAULTS``. Device and
    checkpoint cadence stay mutable: they are engineering choices, while model
    identity is bound separately by the run manifests.
    """
    for name in RESUME_IMMUTABLE:
        if prior.get(name, RESUME_DEFAULTS.get(name)) != getattr(args, name):
            raise ValueError("resume recipe changed: " + name)


def model_file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError("model directory is missing")
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if not files:
        raise ValueError("model directory is empty")
    return files


def load_token_map(path: Path, vocab_size: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensor = payload.get("perm_vocab") if isinstance(payload, dict) else payload
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("token map must be a one-dimensional integer tensor")
    tensor = tensor.long()
    expected = torch.arange(vocab_size, dtype=torch.long)
    if tensor.numel() != vocab_size or not torch.equal(torch.sort(tensor).values, expected):
        raise ValueError("token map must be a vocabulary permutation")
    if isinstance(payload, dict) and "inv_perm_vocab" in payload:
        inverse = payload["inv_perm_vocab"]
        if not isinstance(inverse, torch.Tensor) or inverse.ndim != 1 or inverse.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or inverse.numel() != vocab_size or not torch.equal(inverse.long()[tensor], expected):
            raise ValueError("token map inverse does not match")
    return tensor


def token_map_commitment(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    values = tensor.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(b"osnip-token-map-v1\0" + len(values).to_bytes(8, "little") + values.tobytes()).hexdigest()


def validate_resume_binding(prior: dict, model_files: dict, token_sha: str | None, commitment: str | None) -> None:
    if not prior.get("model_files_sha256") or prior["model_files_sha256"] != model_files:
        raise ValueError("resume checkpoint model contents are missing or changed")
    if "token_map_commitment_sha256" not in prior or prior["token_map_commitment_sha256"] != commitment or prior.get("token_id_map_sha256") != token_sha:
        raise ValueError("resume checkpoint token map is missing or changed")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_hashes() -> dict[str, str]:
    paths = [ROOT / "src/osnip_repro", ROOT / "scripts/train_streaming.py", ROOT / "scripts/prepare_osnip_datasets.py"]
    result: dict[str, str] = {}
    for path in paths:
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for candidate in candidates:
            result[str(candidate.relative_to(ROOT))] = file_sha256(candidate)
    return result


def make_epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def next_batch(index: TextJsonlIndex, order: list[int], cursor: int, batch_size: int) -> tuple[list[str], int]:
    indices = order[cursor:cursor + batch_size]
    if not indices:
        raise RuntimeError("empty training batch")
    return index.read_texts(indices), cursor + len(indices)


def pack_valid_rows(values: torch.Tensor, lengths: list[int], *, causal: bool) -> torch.Tensor:
    """Pack unpadded rows without MPS boolean advanced indexing.

    Boolean indexing on MPS dispatches through ``nonzero`` and synchronizes a
    scalar for every batch.  Validation over 1,000 examples can therefore
    spend most of its time waiting on Metal.  Lengths are transferred once
    per batch and the remaining operations are ordinary differentiable views.
    """
    rows: list[torch.Tensor] = []
    for row, length in enumerate(lengths):
        end = max(0, int(length) - (1 if causal else 0))
        if end:
            rows.append(values[row, :end])
    if not rows:
        raise ValueError("batch has no valid tokens")
    return torch.cat(rows, dim=0)


def active_tensors(lengths: list[int], clean_logits: torch.Tensor, encrypted_logits: torch.Tensor, h: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, ...]:
    def select(logits: torch.Tensor) -> torch.Tensor:
        return pack_valid_rows(logits[:, :-1], lengths, causal=True)
    return (
        select(clean_logits), select(encrypted_logits),
        pack_valid_rows(h, lengths, causal=False),
        pack_valid_rows(z1, lengths, causal=False),
        pack_valid_rows(z2, lengths, causal=False),
    )


def run_objective(model: Any, encoder: torch.nn.Module, tokenizer: Any, texts: list[str], args: argparse.Namespace, device: torch.device, *, training: bool, step: int, key_seed: int | None = None, key_generator: torch.Generator | None = None, constraint_weights: tuple[float, float] | None = None) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if hasattr(encoder, "set_alpha_scale"):
        ramp_steps = int(getattr(args, "alpha_ramp_steps", 0))
        encoder.set_alpha_scale(min(1.0, float(step) / ramp_steps) if ramp_steps > 0 else 1.0)
    batch = tokenize_texts(tokenizer, texts, max_length=args.max_length)
    token_id_map = getattr(args, "token_id_map_tensor", None)
    if token_id_map is not None:
        batch = dict(batch)
        batch["input_ids"] = token_id_map[batch["input_ids"]]
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device).bool()
    # One small host sync per batch replaces the per-token MPS nonzero/item
    # synchronization caused by boolean advanced indexing.
    lengths = [int(value) for value in mask.sum(dim=1).detach().to(device="cpu").tolist()]
    hidden_lambda = float(getattr(args, "lambda_hidden", 0.0))
    hidden_layers = tuple(getattr(args, "hidden_layers", ()))
    lambda_knn = float(getattr(args, "lambda_knn", 0.0))
    knn_negative_count = int(getattr(args, "knn_negative_count", 64))
    knn_topk = int(getattr(args, "knn_topk", 10))
    knn_margin = float(getattr(args, "knn_margin", 0.05))
    with torch.no_grad():
        h = input_embeddings(model, input_ids).detach()
        if hidden_lambda > 0 and hidden_layers:
            clean_logits, clean_hidden_values = forward_logits_and_hidden_states(model, h, mask, hidden_layers)
            clean_logits = clean_logits.detach()
            clean_hidden = tuple(value.detach() for value in clean_hidden_values)
        else:
            clean_logits = forward_logits(model, h, mask).detach()
            clean_hidden = ()
    if key_seed is None:
        if key_generator is None:
            raise ValueError("training objective requires an explicit independent key generator")
        key1 = torch.randn((input_ids.shape[0], args.key_dim), generator=key_generator).to(device)
        key2 = torch.randn((input_ids.shape[0], args.key_dim), generator=key_generator).to(device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(key_seed)
        key1 = torch.randn((input_ids.shape[0], args.key_dim), generator=generator).to(device)
        key2 = torch.randn((input_ids.shape[0], args.key_dim), generator=generator).to(device)
    # Keep the trainable encryptor and the frozen backbone at their own dtypes.
    # MPS rejects mixed-dtype Linear accumulators when a float32 encryptor
    # consumes float16 Llama embeddings.  The projection is computed in the
    # encryptor dtype, then encrypted embeddings are cast back before the
    # frozen model forward.
    encoder_dtype = next(encoder.parameters()).dtype
    h_for_encoder = h.to(dtype=encoder_dtype)
    key1 = key1.to(dtype=encoder_dtype)
    key2 = key2.to(dtype=encoder_dtype)
    raw_r1 = encoder(h_for_encoder, key1, attention_mask=mask)
    raw_r2 = encoder(h_for_encoder, key2, attention_mask=mask)
    z1 = iso_norm_project(raw_r1, h_for_encoder).to(dtype=h.dtype)
    z2 = iso_norm_project(raw_r2, h_for_encoder).to(dtype=h.dtype)
    if hidden_lambda > 0 and hidden_layers:
        encrypted_logits_1, encrypted_hidden_1 = forward_logits_and_hidden_states(model, z1, mask, hidden_layers)
        encrypted_logits_2, encrypted_hidden_2 = forward_logits_and_hidden_states(model, z2, mask, hidden_layers)
    else:
        encrypted_logits_1 = forward_logits(model, z1, mask)
        encrypted_logits_2 = forward_logits(model, z2, mask)
        encrypted_hidden_1 = encrypted_hidden_2 = ()
    clean_active, encrypted_active_1, h_active, z_active, z_other_active = active_tensors(lengths, clean_logits, encrypted_logits_1, h, z1, z2)
    raw_r1_active = pack_valid_rows(raw_r1, lengths, causal=False)
    raw_r2_active = pack_valid_rows(raw_r2, lengths, causal=False)
    encrypted_active_2 = pack_valid_rows(encrypted_logits_2[:, :-1], lengths, causal=True)
    if constraint_weights is not None:
        privacy_weight, diversity_weight = constraint_weights
    elif args.constraint_mode == "off":
        privacy_weight = diversity_weight = 0.0
    elif args.constraint_mode == "constant" or not training:
        # Validation never backpropagates through the gate.  Keeping the
        # configured base weight avoids one MPS scalar synchronization per
        # validation example while preserving the recorded objective metrics.
        privacy_weight, diversity_weight = args.lambda_priv, args.lambda_div
    else:
        utility_value = float((utility_kl(clean_active, encrypted_active_1) + utility_kl(clean_active, encrypted_active_2)).mul(0.5).detach())
        configured_constraint_ramp = getattr(args, "constraint_ramp_steps", None)
        constraint_ramp_steps = configured_constraint_ramp if configured_constraint_ramp is not None else args.warmup_steps
        privacy_weight = constraint_weight(step, utility_value, base=args.lambda_priv, warmup_steps=constraint_ramp_steps, tau_low=args.tau_low, tau_high=args.tau_high)
        diversity_weight = constraint_weight(step, utility_value, base=args.lambda_div, warmup_steps=constraint_ramp_steps, tau_low=args.tau_low, tau_high=args.tau_high)
    losses = osnip_two_key_loss(clean_active, encrypted_active_1, encrypted_active_2, h_active, z_active, z_other_active, epsilon=args.epsilon, lambda_priv=privacy_weight, diversity_margin=args.diversity_margin, lambda_div=diversity_weight, diversity_z1=raw_r1_active if args.diversity_space == "raw_r" else None, diversity_z2=raw_r2_active if args.diversity_space == "raw_r" else None)
    if hidden_lambda > 0 and hidden_layers:
        def recovery_loss(clean_values: tuple[torch.Tensor, ...], encrypted_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            values = []
            for clean_value, encrypted_value in zip(clean_values, encrypted_values):
                clean_active_value = pack_valid_rows(clean_value, lengths, causal=False).float()
                encrypted_active_value = pack_valid_rows(encrypted_value, lengths, causal=False).float()
                values.append((1.0 - torch.nn.functional.cosine_similarity(clean_active_value, encrypted_active_value, dim=-1)).mean())
            return torch.stack(values).mean()
        losses["hidden_recovery"] = 0.5 * (
            recovery_loss(clean_hidden, encrypted_hidden_1) + recovery_loss(clean_hidden, encrypted_hidden_2)
        )
    else:
        losses["hidden_recovery"] = z1.new_zeros(())
    if lambda_knn > 0:
        target_active = pack_valid_rows(input_ids, lengths, causal=False).long()
        embedding_weight = model.get_input_embeddings().weight.detach()
        losses["knn_privacy"] = 0.5 * (
            vocabulary_margin_loss(z_active, target_active, embedding_weight,
                                   negative_count=knn_negative_count, topk=knn_topk, margin=knn_margin)
            + vocabulary_margin_loss(z_other_active, target_active, embedding_weight,
                                     negative_count=knn_negative_count, topk=knn_topk, margin=knn_margin)
        )
    else:
        losses["knn_privacy"] = z1.new_zeros(())
    loss_weight = warmup_weight(step, getattr(args, "loss_warmup_steps", 0)) if training else 1.0
    losses["loss_weight"] = z1.new_tensor(loss_weight)
    losses["total"] = loss_weight * (args.lambda_util * losses["utility"] + privacy_weight * losses["privacy"]
                       + diversity_weight * losses["diversity"] + hidden_lambda * losses["hidden_recovery"]
                       + lambda_knn * losses["knn_privacy"])
    with torch.no_grad():
        causal_count = sum(max(length - 1, 0) for length in lengths)
        clean_nll, clean_count = causal_nll_stats(clean_logits, input_ids, mask, valid_count=causal_count)
        encrypted_nll, encrypted_count = causal_nll_stats(encrypted_logits_1, input_ids, mask, valid_count=causal_count)
        metric_tensor = torch.stack((
            losses["utility"], losses["utility_key1"], losses["utility_key2"],
            losses["privacy_key1"], losses["privacy_key2"], losses["privacy"], losses["diversity"], losses["hidden_recovery"], losses["knn_privacy"],
            clean_nll, encrypted_nll,
            torch.maximum((z1.float().norm(dim=-1) - h.float().norm(dim=-1)).abs().max(), (z2.float().norm(dim=-1) - h.float().norm(dim=-1)).abs().max()),
            (key1 != key2).any(dim=-1).float().mean(),
            torch.nn.functional.cosine_similarity(h_active.float(), z_active.float(), dim=-1).abs().mean(),
            torch.nn.functional.cosine_similarity(h_active.float(), z_other_active.float(), dim=-1).abs().mean(),
            torch.nn.functional.cosine_similarity(h_active.float(), z_active.float(), dim=-1).mean(),
            torch.nn.functional.cosine_similarity(h_active.float(), z_other_active.float(), dim=-1).mean(),
            ((z_active.float() - h_active.float()).norm(dim=-1) / h_active.float().norm(dim=-1).clamp_min(1e-12)).mean(),
            ((z_other_active.float() - h_active.float()).norm(dim=-1) / h_active.float().norm(dim=-1).clamp_min(1e-12)).mean(),
        )).detach().to(device="cpu").tolist()
        utility, utility_key1, utility_key2, privacy_key1, privacy_key2, privacy, diversity, hidden_recovery, knn_privacy, clean_nll_value, encrypted_nll_value, norm_error, distinct, abs_cos1, abs_cos2, cos1, cos2, rel1, rel2 = (float(value) for value in metric_tensor)
        h_norm = h.float().norm(dim=-1).clamp_min(1e-12)
        relative_norm_error = torch.maximum(
            ((z1.float().norm(dim=-1) - h_norm).abs() / h_norm).max(),
            ((z2.float().norm(dim=-1) - h_norm).abs() / h_norm).max())
        raw_projection_error = torch.maximum(
            ((iso_norm_project(raw_r1, h_for_encoder).float().norm(dim=-1) - h_norm).abs() / h_norm).max(),
            ((iso_norm_project(raw_r2, h_for_encoder).float().norm(dim=-1) - h_norm).abs() / h_norm).max())
        stats = {
            "norm_max_relative_error": float(relative_norm_error.detach().cpu()),
            "projection_fp32_max_relative_error": float(raw_projection_error.detach().cpu()),
            "utility": utility, "utility_key1": utility_key1, "utility_key2": utility_key2,
            "privacy_key1": privacy_key1, "privacy_key2": privacy_key2, "privacy": privacy, "diversity": diversity, "hidden_recovery": hidden_recovery, "knn_privacy": knn_privacy,
            "privacy_weight": float(privacy_weight), "diversity_weight": float(diversity_weight), "loss_weight": float(loss_weight),
            "clean_nll_sum": clean_nll_value, "encrypted_nll_sum": encrypted_nll_value,
            "causal_tokens": clean_count, "representation_tokens": sum(lengths), "examples": int(input_ids.shape[0]),
            "norm_max_abs_error": norm_error, "keys_distinct_fraction": distinct,
            "mean_abs_cosine_key1": abs_cos1, "mean_abs_cosine_key2": abs_cos2,
            "clean_ppl": math.exp(clean_nll_value / max(clean_count, 1)),
            "encrypted_ppl": math.exp(encrypted_nll_value / max(encrypted_count, 1)),
            "mean_cosine_key1": cos1, "mean_cosine_key2": cos2,
            "relative_perturbation_key1": rel1, "relative_perturbation_key2": rel2,
        }
    return losses, stats


@torch.no_grad()
def evaluate(model: Any, encoder: torch.nn.Module, tokenizer: Any, index: TextJsonlIndex, args: argparse.Namespace, device: torch.device, step: int) -> dict[str, float]:
    encoder.eval()
    count = min(len(index.offsets), args.max_valid_samples or len(index.offsets))
    rows: list[dict[str, float]] = []
    for start in range(0, count, args.batch_size):
        texts = index.read_texts(list(range(start, min(start + args.batch_size, count))))
        # Common validation keys across checkpoints: step only affects the gate,
        # never the held-out perturbation draws used for checkpoint selection.
        _, stats = run_objective(model, encoder, tokenizer, texts, args, device, training=False, step=step, key_seed=args.seed + 1000003 + start)
        rows.append(stats)
    return aggregate_validation_stats(rows)


def checkpoint_config(args: argparse.Namespace, width: int, train_index: TextJsonlIndex, valid_index: TextJsonlIndex) -> dict[str, Any]:
    config = {key: (str(value.resolve()) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "token_id_map_tensor"}
    config.update({
        "format": "osnip_streaming_training_config_v1",
        "model_path": str(args.model_path.resolve()),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "valid_jsonl": str(args.valid_jsonl.resolve()),
        "train_sha256": file_sha256(args.train_jsonl),
        "valid_sha256": file_sha256(args.valid_jsonl),
        "width": width,
        "key_schedule": "independent_key_per_example_shared_across_tokens",
        "key_rng_seed": args.seed + 190734863,
        "validation_key_seed": args.seed + 1000003,
        "validation_protocol": "fixed_keys_token_weighted_v2",
        "diversity_space": args.diversity_space,
        "encryptor_architecture": args.encryptor_architecture,
        "streaming_index": train_index.manifest(),
        "validation_index": valid_index.manifest(),
        "source_hashes": source_hashes(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
        "token_id_map": None if args.token_id_map is None else str(args.token_id_map.resolve()),
        "token_id_map_sha256": None if args.token_id_map is None else file_sha256(args.token_id_map),
    })
    config["recipe_fingerprint"] = recipe_fingerprint(config)
    return config



def write_run_manifests(output_dir: Path, model_path: Path, tokenizer: Any, model: Any, device: torch.device, train_jsonl: Path, valid_jsonl: Path) -> None:
    model_files = {str(path.relative_to(model_path)): file_sha256(path) for path in sorted(model_path.rglob("*")) if path.is_file()}
    model_manifest = {
        "format": "osnip_model_manifest_v1",
        "model_path": str(model_path.resolve()),
        "model_config": model.config.to_dict(),
        "tokenizer": {"class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id},
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "model_file_sha256": model_files,
    }
    environment = {
        "format": "osnip_environment_manifest_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "train_jsonl": str(train_jsonl.resolve()),
        "train_sha256": file_sha256(train_jsonl),
        "valid_jsonl": str(valid_jsonl.resolve()),
        "valid_sha256": file_sha256(valid_jsonl),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "source_hashes.json").write_text(json.dumps({"format": "osnip_source_hash_manifest_v1", "hashes": source_hashes()}, ensure_ascii=False, indent=2) + "\n")
    data_manifest = train_jsonl.parent / "data_manifest.json"
    if data_manifest.is_file():
        shutil.copy2(data_manifest, output_dir / "data_manifest.json")


def backward_effective_batch(model, encoder, tokenizer, batches, args, device, *, step, key_generator):
    """Bounded-memory equivalent of pooled-token batch loss with a detached gate.

    First pass obtains exact denominators and global utility. Each second-pass
    microbatch reuses its CPU key-generator state; neither model uses dropout.
    No optimizer update occurs between passes. This is slower than a resident
    batch but does not retain 64 Llama computation graphs.
    """
    if args.gradient_balance != "none":
        raise ValueError("effective_batch requires ordinary weighted gradients")
    preview, states = [], []
    with torch.no_grad():
        for texts in batches:
            states.append(key_generator.get_state().clone())
            _, stats = run_objective(model, encoder, tokenizer, texts, args, device,
                                     training=True, step=step, key_generator=key_generator,
                                     constraint_weights=(0.0, 0.0))
            preview.append(stats)
    advanced_state = key_generator.get_state().clone()
    totals = aggregate_validation_stats(preview)
    if args.constraint_mode == "off":
        weights = (0.0, 0.0)
    elif args.constraint_mode == "constant":
        weights = (args.lambda_priv, args.lambda_div)
    else:
        ramp = getattr(args, "constraint_ramp_steps", None)
        ramp = args.warmup_steps if ramp is None else ramp
        weights = tuple(constraint_weight(step, totals["utility"], base=base,
                         warmup_steps=ramp, tau_low=args.tau_low, tau_high=args.tau_high)
                        for base in (args.lambda_priv, args.lambda_div))
    rows = []
    for texts, state in zip(batches, states):
        key_generator.set_state(state)
        losses, stats = run_objective(model, encoder, tokenizer, texts, args, device,
                                     training=True, step=step, key_generator=key_generator,
                                     constraint_weights=weights)
        utility_fraction = stats["causal_tokens"] / totals["causal_tokens"]
        geometry_fraction = stats["representation_tokens"] / totals["representation_tokens"]
        objective = stats["loss_weight"] * (
            args.lambda_util * utility_fraction * losses["utility"]
            + geometry_fraction * (weights[0] * losses["privacy"] + weights[1] * losses["diversity"]
                + args.lambda_hidden * losses["hidden_recovery"] + args.lambda_knn * losses["knn_privacy"]))
        objective.backward()
        rows.append(stats)
    key_generator.set_state(advanced_state)
    result = aggregate_validation_stats(rows)
    result["effective_batch_examples"] = int(totals["examples"])
    result["gate_utility"] = totals["utility"]
    return result


def backward_balanced(losses: dict[str, torch.Tensor], encoder: torch.nn.Module,
                      *, component_weights: dict[str, float], accumulation_steps: int) -> dict[str, float]:
    """Accumulate a weighted, per-component unit-norm gradient."""
    params = tuple(parameter for parameter in encoder.parameters() if parameter.requires_grad)
    components = [(name, float(weight)) for name, weight in component_weights.items()
                  if name in losses and float(weight) != 0.0]
    component_norms: dict[str, float] = {}
    combined = [None for _ in params]
    for index, (name, weight) in enumerate(components):
        gradients = torch.autograd.grad(losses[name], params, retain_graph=index < len(components) - 1,
                                        allow_unused=True)
        squared = None
        for gradient in gradients:
            if gradient is not None:
                value = gradient.detach().float().pow(2).sum()
                squared = value if squared is None else squared + value
        norm = torch.sqrt(squared if squared is not None else torch.zeros((), device=next(encoder.parameters()).device))
        norm_value = float(norm.detach().cpu())
        component_norms[name] = norm_value
        scale = weight / max(norm_value, 1e-12)
        for position, gradient in enumerate(gradients):
            if gradient is None:
                continue
            contribution = gradient.detach() * scale / accumulation_steps
            combined[position] = contribution if combined[position] is None else combined[position] + contribution
    for parameter, gradient in zip(params, combined):
        if gradient is not None:
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
    return component_norms


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation_steps, args.max_length, args.checkpoint_every, args.evaluate_every) < 1:
        raise ValueError("steps, batch, accumulation, max_length, and intervals must be positive")
    if args.warmup_steps < 0 or (args.constraint_ramp_steps is not None and args.constraint_ramp_steps < 0) or args.loss_warmup_steps < 0:
        raise ValueError("warmup, constraint-ramp, and loss-warmup steps must be nonnegative")
    if args.learning_rate <= 0 or args.grad_clip <= 0:
        raise ValueError("learning-rate and grad-clip must be positive")
    if not 0 <= args.epsilon < 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 1 (paper Eq.15)")
    if args.diversity_margin <= 0:
        raise ValueError("diversity-margin must be positive (paper Eq.19)")
    if args.lambda_knn < 0 or args.knn_negative_count < 1 or args.knn_topk < 1 or args.knn_topk > args.knn_negative_count or args.knn_margin < 0:
        raise ValueError("invalid sampled vocabulary-ranking settings")
    if args.alpha_init <= 0 or args.alpha_ramp_steps < 0:
        raise ValueError("alpha-init must be positive and alpha-ramp-steps nonnegative")
    if args.lambda_hidden < 0 or any(layer < 0 for layer in args.hidden_layers):
        raise ValueError("lambda-hidden and hidden-layers must be nonnegative")
    if args.lambda_hidden > 0 and not args.hidden_layers:
        raise ValueError("hidden-layers are required when lambda-hidden is positive")
    if len(set(args.hidden_layers)) != len(args.hidden_layers):
        raise ValueError("hidden-layers must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise FileExistsError("output directory is non-empty; choose a new directory or use --resume")
    if args.gate_scope == "effective_batch" and args.gradient_balance != "none":
        raise ValueError("effective_batch requires gradient-balance none")
    if args.gate_scope == "effective_batch" and args.encryptor_architecture == "contextual_residual_v1":
        raise ValueError("two-pass effective batch currently audited only for deterministic tokenwise encryptors")
    if args.resume is not None:
        validate_resume_budget(str(args.resume), args.steps)
        prior = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        validate_resume_recipe(prior, args)
    model_files = model_file_hashes(args.model_path)
    token_sha = None if args.token_id_map is None else file_sha256(args.token_id_map)
    train_sha = file_sha256(args.train_jsonl)
    valid_sha = file_sha256(args.valid_jsonl)
    if args.token_id_map is not None:
        model_config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
        args.token_id_map_tensor = load_token_map(args.token_id_map, int(model_config["vocab_size"]))
        if file_sha256(args.token_id_map) != token_sha:
            raise ValueError("token map changed while loading")
    else:
        args.token_id_map_tensor = None
    commitment = token_map_commitment(args.token_id_map_tensor)
    if args.resume is not None:
        validate_resume_binding(prior, model_files, token_sha, commitment)
        if prior.get("train_sha256") != train_sha or prior.get("valid_sha256") != valid_sha:
            raise ValueError("resume checkpoint data contents changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    if model_file_hashes(args.model_path) != model_files:
        raise ValueError("model files changed while loading")
    train_index = TextJsonlIndex.build(args.train_jsonl, limit=args.max_train_samples)
    valid_index = TextJsonlIndex.build(args.valid_jsonl, limit=args.max_valid_samples)
    width = int(model.get_input_embeddings().embedding_dim)
    encoder = build_encryptor(width, key_dim=args.key_dim, hidden_dim=args.hidden_dim, architecture=args.encryptor_architecture, alpha_init=args.alpha_init).to(device)
    if args.freeze_alpha and args.encryptor_architecture == "orthogonal_direction_v1":
        for parameter in encoder.alpha_projection.parameters():
            parameter.requires_grad_(False)
    if args.alpha_ramp_steps > 0:
        encoder.set_alpha_scale(0.0)
    optimizer = torch.optim.AdamW((parameter for parameter in encoder.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(args.warmup_steps, 1)))
    start_step = 0
    epoch = 0
    cursor = 0
    best_utility = math.inf
    best_pareto = math.inf
    key_generator = torch.Generator(device="cpu").manual_seed(args.seed + 190734863)
    initial_encoder_state = {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()}
    config = checkpoint_config(args, width, train_index, valid_index)
    config["model_files_sha256"] = model_files
    config["token_map_commitment_sha256"] = commitment
    if config["train_sha256"] != train_sha or config["valid_sha256"] != valid_sha:
        raise ValueError("training data changed while indexing")
    if config["token_id_map_sha256"] != token_sha:
        raise ValueError("token map changed before training")
    config["recipe_fingerprint"] = recipe_fingerprint(config)
    if args.resume is not None:
        payload = load_training_checkpoint(
            str(args.resume),
            encoder=encoder,
            optimizer=None if args.reset_optimizer else optimizer,
            scheduler=None if args.reset_optimizer else scheduler,
            map_location=device,
        )
        start_step = int(payload["global_step"])
        extra = payload.get("extra", {})
        epoch = int(extra.get("epoch", 0))
        cursor = int(extra.get("cursor", 0))
        best_utility = float(extra.get("best_utility", math.inf))
        best_pareto = float(extra.get("best_pareto", math.inf))
        key_rng_state = extra.get("key_rng_state")
        if key_rng_state is not None:
            if not isinstance(key_rng_state, torch.Tensor):
                raise ValueError("resume checkpoint key_rng_state is not a tensor")
            # Device-aware torch.load can move this CPU generator state to MPS.
            key_generator.set_state(key_rng_state.detach().to(device="cpu", dtype=torch.uint8))
        if payload.get("config", {}).get("train_sha256") != config["train_sha256"]:
            raise ValueError("resume checkpoint belongs to a different training data file")
        if payload.get("config", {}).get("valid_sha256") != config["valid_sha256"]:
            raise ValueError("resume checkpoint belongs to a different validation data file")
        if payload.get("config", {}).get("model_path") != config["model_path"]:
            raise ValueError("resume checkpoint belongs to a different model")
        if payload.get("config", {}).get("encryptor_architecture", "mlp_residual_v1") != config["encryptor_architecture"]:
            raise ValueError("resume checkpoint belongs to a different encryptor architecture")
    write_run_manifests(args.output_dir, args.model_path, tokenizer, model, device, args.train_jsonl, args.valid_jsonl)
    (args.output_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    # Existing evaluation scripts consume config.json; keep the same locked
    # configuration under that compatibility name.
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if args.resume is None:
        torch.save(initial_encoder_state, args.output_dir / "initial_encoder.pt")
    else:
        # Carry existing best snapshots to a new resume directory; never claim
        # artifacts from a previous directory exist locally when they do not.
        resume_dir = args.resume.parent
        for name in ("best_utility.pt", "best_utility_encoder.pt", "best_pareto.pt", "best_pareto_encoder.pt", "initial_encoder.pt"):
            source = resume_dir / name
            target = args.output_dir / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
        if not (args.output_dir / "best_utility.pt").is_file():
            best_utility = math.inf
        if not (args.output_dir / "best_pareto.pt").is_file():
            best_pareto = math.inf
    baseline_validation = evaluate(model, encoder, tokenizer, valid_index, args, device, start_step)
    baseline_path = args.output_dir / ("initial_validation.json" if args.resume is None else f"resume_validation_step{start_step}.json")
    baseline_path.write_text(json.dumps({"global_step": start_step, "protocol": "fixed_keys_token_weighted_v2", "validation": baseline_validation}, ensure_ascii=False, indent=2) + "\n")
    order = make_epoch_order(len(train_index.offsets), args.seed, epoch)
    if cursor >= len(order):
        epoch += cursor // len(order)
        cursor = cursor % len(order)
        order = make_epoch_order(len(train_index.offsets), args.seed, epoch)
    progress_path = args.output_dir / "train_progress.jsonl"
    mode = "a" if args.resume is not None else "w"
    progress = progress_path.open(mode, encoding="utf-8")
    last_validation: dict[str, float] = {}
    started = time.time()
    try:
        for step in range(start_step, args.steps):
            encoder.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            batches = []
            for _ in range(args.gradient_accumulation_steps):
                if cursor >= len(order):
                    epoch += 1
                    cursor = 0
                    order = make_epoch_order(len(train_index.offsets), args.seed, epoch)
                texts, cursor = next_batch(train_index, order, cursor, args.batch_size)
                batches.append(texts)
            if args.gate_scope == "effective_batch":
                accumulated = backward_effective_batch(model, encoder, tokenizer, batches, args, device,
                                                       step=step + 1, key_generator=key_generator)
            else:
                for texts in batches:
                    losses, stats = run_objective(model, encoder, tokenizer, texts, args, device, training=True,
                                                  step=step + 1, key_generator=key_generator)
                    if args.gradient_balance == "none":
                        (losses["total"] / args.gradient_accumulation_steps).backward()
                    else:
                        component_weights = {
                            "utility": args.lambda_util * stats["loss_weight"],
                            "privacy": stats["privacy_weight"] * stats["loss_weight"],
                            "diversity": stats["diversity_weight"] * stats["loss_weight"],
                            "hidden_recovery": args.lambda_hidden * stats["loss_weight"],
                            "knn_privacy": args.lambda_knn * stats["loss_weight"],
                        }
                        backward_balanced(losses, encoder, component_weights=component_weights,
                                          accumulation_steps=args.gradient_accumulation_steps)
                    for key, value in stats.items():
                        accumulated[key] = accumulated.get(key, 0.0) + value / args.gradient_accumulation_steps
            grad_norm = float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("nonfinite gradient norm; optimizer update refused")
            optimizer.step()
            scheduler.step()
            record: dict[str, Any] = {"step": step + 1, "epoch": epoch, "cursor": cursor, "lr": float(optimizer.param_groups[0]["lr"]), "gradient_norm_before_clip": grad_norm, "elapsed_seconds": time.time() - started}
            record.update(accumulated)
            record["samples_seen"] = epoch * len(order) + cursor
            record["epochs_completed"] = record["samples_seen"] / len(order)
            if device.type == "mps":
                record["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
                record["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            best_utility_updated = False
            best_pareto_updated = False
            if (step + 1) % args.evaluate_every == 0 or step + 1 == args.steps:
                last_validation = {**evaluate(model, encoder, tokenizer, valid_index, args, device, step + 1),
                                     "step": step + 1}
                record["validation"] = last_validation
                if last_validation["utility"] < best_utility:
                    best_utility = last_validation["utility"]
                    best_utility_updated = True
                    record["best_utility_updated"] = True
                if last_validation["pareto_score"] < best_pareto:
                    best_pareto = last_validation["pareto_score"]
                    best_pareto_updated = True
                    record["best_pareto_updated"] = True
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
            if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps or best_utility_updated or best_pareto_updated:
                checkpoint_metrics = {"step": step + 1, "train": accumulated, "validation": last_validation}
                extra = {"epoch": epoch, "cursor": cursor, "best_utility": best_utility, "best_pareto": best_pareto, "key_rng_state": key_generator.get_state()}
                save_training_checkpoint(str(args.output_dir / "last.pt"), encoder=encoder, optimizer=optimizer, scheduler=scheduler, global_step=step + 1, config=config, metrics=checkpoint_metrics, extra=extra)
                if best_utility_updated:
                    save_training_checkpoint(str(args.output_dir / "best_utility.pt"), encoder=encoder, optimizer=optimizer, scheduler=scheduler, global_step=step + 1, config=config, metrics=checkpoint_metrics, extra=extra)
                    torch.save(encoder.state_dict(), args.output_dir / "best_utility_encoder.pt")
                if best_pareto_updated:
                    save_training_checkpoint(str(args.output_dir / "best_pareto.pt"), encoder=encoder, optimizer=optimizer, scheduler=scheduler, global_step=step + 1, config=config, metrics=checkpoint_metrics, extra=extra)
                    torch.save(encoder.state_dict(), args.output_dir / "best_pareto_encoder.pt")
    finally:
        progress.close()
    if (model_file_hashes(args.model_path) != model_files or
        file_sha256(args.train_jsonl) != config["train_sha256"] or
        file_sha256(args.valid_jsonl) != config["valid_sha256"] or
        (args.token_id_map is not None and file_sha256(args.token_id_map) != token_sha)):
        raise ValueError("training inputs changed during training")
    torch.save(encoder.state_dict(), args.output_dir / "encryptor.pt")
    torch.save({"format": "osnip_encoder_binding_v1", "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "model_files_sha256": model_files, "token_map_commitment_sha256": commitment,
                "train_sha256": config["train_sha256"], "valid_sha256": config["valid_sha256"],
                "training_seed": args.seed, "global_step": args.steps}, args.output_dir / "training_binding.pt")
    report = {
        "format": "osnip_streaming_training_report_v1",
        "status": "completed_with_limits",
        "paper_result": False,
        "security_accepted": False,
        "model_path": str(args.model_path.resolve()),
        "seed": args.seed,
        "encryptor_architecture": args.encryptor_architecture,
        "train_records": len(train_index.offsets),
        "valid_records": len(valid_index.offsets),
        "steps_completed": args.steps,
        "steps_executed_this_run": args.steps - start_step,
        "resumed_from": None if args.resume is None else str(args.resume.resolve()),
        "validation_protocol": "fixed_keys_token_weighted_v2",
        "baseline_validation": baseline_validation,
        "baseline_step": start_step,
        "last_validation": last_validation,
        "best_utility": best_utility,
        "best_pareto": best_pareto,
        "artifacts": {name: str((args.output_dir / name).resolve()) for name in ("last.pt", "best_utility.pt", "best_pareto.pt", "encryptor.pt", "training_binding.pt", "best_utility_encoder.pt", "best_pareto_encoder.pt", "train_progress.jsonl", "train_config.json", "config.json", "model_manifest.json", "environment.json", "source_hashes.json")},
        "limitations": ["objective-aligned configurable encryptor baseline", "no KNN/VMA security acceptance in trainer", "paper's undisclosed architecture and full protocol require separate evaluation"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
