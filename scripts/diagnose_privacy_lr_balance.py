"""Controlled probe: can the joint OSNIP objective reach the privacy region at a
given encryptor learning rate?

The paper (Figure 8) shows the privacy hinge converging to zero within ~1200
steps while utility KL temporarily rises to ~0.1. Prior diagnostics
(diagnostics/geometry-and-gradient-20261017.json) measured a utility gradient
norm ~13x the privacy gradient norm, and the executed pilot used
learning_rate=1e-5 (configs still carry TUNE_* placeholders). With AdamW a
per-parameter step is bounded by ~lr, so 1000 steps at 1e-5 cannot move the
encryptor far enough to push cos(h,z) from ~1.0 to <=epsilon.

This probe isolates that mechanism on a tiny random causal LM (no pretrained
weights needed): same encoder/loss/projection code paths as training, constant
constraint weights (no curriculum gate), and a sweep over learning rate and
lambda_priv plus a privacy-only control. It reports the cos/privacy/utility
trajectory so the stall hypothesis can be confirmed or rejected numerically.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/diagnose_privacy_lr_balance.py \
        --steps 300 --output reports/diagnostics/privacy_lr_balance.json
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from osnip_repro.encoder import build_encryptor
from osnip_repro.losses import osnip_two_key_loss, privacy_hinge, utility_kl
from osnip_repro.modeling import forward_logits, input_embeddings
from osnip_repro.projection import encrypt_embeddings


def tiny_model(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def component_gradient_norms(encoder, pieces) -> dict[str, float]:
    norms = {}
    for name, loss in pieces.items():
        encoder.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        norms[name] = float(torch.norm(
            torch.stack([p.grad.norm() for p in encoder.parameters() if p.grad is not None]))
        )
    encoder.zero_grad(set_to_none=True)
    return norms


def run_cell(lr: float, lambda_priv: float, lambda_div: float, privacy_only: bool,
             steps: int, seed: int) -> dict:
    model = tiny_model(seed)
    width = model.config.hidden_size
    encoder = build_encryptor(width, key_dim=32, hidden_dim=64, architecture="mlp_residual_v1")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.0)

    gen = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(0, model.config.vocab_size, (8, 24), generator=gen)
    mask = torch.ones_like(ids)
    key_gen = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        h = input_embeddings(model, ids)
        clean_logits = forward_logits(model, h, mask)
    shifted_clean = clean_logits[:, :-1]

    trajectory = []
    grad_norms = None
    for step in range(steps + 1):
        k1 = torch.randn(32, generator=key_gen)
        k2 = torch.randn(32, generator=key_gen)
        z1 = encrypt_embeddings(encoder, h, k1)
        z2 = encrypt_embeddings(encoder, h, k2)
        with torch.no_grad():
            cos1 = torch.nn.functional.cosine_similarity(
                h.reshape(-1, width).float(), z1.reshape(-1, width).float(), dim=-1).mean()
        if step % 50 == 0 or step == steps:
            enc1 = forward_logits(model, z1, mask)[:, :-1]
            enc2 = forward_logits(model, z2, mask)[:, :-1]
            trajectory.append({
                "step": step,
                "cos_key1": float(cos1),
                "privacy_hinge": float(0.5 * (privacy_hinge(h, z1, 0.3) + privacy_hinge(h, z2, 0.3))),
                "utility_kl": float(0.5 * (utility_kl(shifted_clean, enc1) + utility_kl(shifted_clean, enc2))),
            })
        if step == steps:
            break
        enc1 = forward_logits(model, z1, mask)[:, :-1]
        enc2 = forward_logits(model, z2, mask)[:, :-1]
        parts = osnip_two_key_loss(shifted_clean, enc1, enc2, h, z1, z2,
                                   epsilon=0.3, lambda_priv=lambda_priv,
                                   diversity_margin=1.0, lambda_div=lambda_div)
        if privacy_only:
            total = parts["privacy"] * lambda_priv + parts["diversity"] * lambda_div
        else:
            total = parts["total"]
        if step == 0:
            grad_norms = component_gradient_norms(
                encoder, {"utility": parts["utility"], "privacy": lambda_priv * parts["privacy"],
                          "diversity": lambda_div * parts["diversity"]})
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
    return {"lr": lr, "lambda_priv": lambda_priv, "lambda_div": lambda_div,
            "privacy_only": privacy_only, "gradient_norms_step0": grad_norms,
            "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    cells = []
    for lr, lambda_priv in product((1e-5, 1e-4, 1e-3), (1.0, 10.0)):
        cells.append(run_cell(lr, lambda_priv, 1.0, False, args.steps, args.seed))
    for lr in (1e-5, 1e-3):
        cells.append(run_cell(lr, 1.0, 0.0, True, args.steps, args.seed))

    for cell in cells:
        first, last = cell["trajectory"][0], cell["trajectory"][-1]
        print(f"lr={cell['lr']:.0e} lambda_priv={cell['lambda_priv']:.0f} "
              f"privacy_only={cell['privacy_only']}: cos {first['cos_key1']:.3f} -> "
              f"{last['cos_key1']:.3f}, privacy {first['privacy_hinge']:.3f} -> "
              f"{last['privacy_hinge']:.3f}, utility {first['utility_kl']:.4f} -> "
              f"{last['utility_kl']:.4f}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"steps": args.steps, "seed": args.seed, "cells": cells},
                      handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
