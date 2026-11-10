"""Sensitivity probe: does the utility/privacy token-position choice matter?

Training uses utility KL on the L-1 causal positions but the privacy hinge on
all L representation positions (scripts/train_streaming.py). The paper does
not specify either set. This probe quantifies the choice on a tiny random
causal LM with the repository's own losses:

  A (current): utility [:, :-1], privacy [:, :]
  B:           utility [:, :-1], privacy [:, :-1]
  C:           utility [:, : ],  privacy [:, :]

Per seed it reports loss values, per-component gradient norms and the cosine
between variant total-loss gradients at initialization, then runs 50 AdamW
steps per variant and compares the resulting mean cos(h, z). If gradient
directions and end geometry agree closely, the choice is an immaterial
engineering detail; otherwise it must be preregistered as a protocol factor.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/diagnose_token_mask_sensitivity.py \
        --steps 50 --output reports/diagnostics/token_mask_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from osnip_repro.encoder import build_encryptor
from osnip_repro.losses import diversity_hinge, privacy_hinge, utility_kl
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


def variant_losses(shift: str, model, encoder, h, keys, epsilon=0.3, margin=1.0):
    z1 = encrypt_embeddings(encoder, h, keys[0])
    z2 = encrypt_embeddings(encoder, h, keys[1])
    clean = forward_logits(model, h, torch.ones(h.shape[:2], dtype=torch.long))
    enc1 = forward_logits(model, z1, torch.ones(h.shape[:2], dtype=torch.long))
    enc2 = forward_logits(model, z2, torch.ones(h.shape[:2], dtype=torch.long))
    utility_slice = (slice(None), slice(None, -1)) if shift in ("a", "b") else (slice(None), slice(None))
    privacy_z1, privacy_z2 = (z1, z2) if shift != "b" else (z1[:, :-1], z2[:, :-1])
    privacy_h = h if shift != "b" else h[:, :-1]
    utility = 0.5 * (utility_kl(clean[utility_slice], enc1[utility_slice])
                     + utility_kl(clean[utility_slice], enc2[utility_slice]))
    privacy = 0.5 * (privacy_hinge(privacy_h, privacy_z1, epsilon)
                     + privacy_hinge(privacy_h, privacy_z2, epsilon))
    diversity = diversity_hinge(z1, z2, margin)
    return utility, privacy, diversity


def gradient_cosine(encoder, losses_a, losses_b) -> float:
    grads = []
    for losses in (losses_a, losses_b):
        encoder.zero_grad(set_to_none=True)
        total = losses[0] + losses[1] + 2.0 * losses[2]
        total.backward(retain_graph=True)
        grads.append(torch.cat([p.grad.reshape(-1) for p in encoder.parameters() if p.grad is not None]))
    encoder.zero_grad(set_to_none=True)
    return float(torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0))


def run_variant(shift: str, steps: int, seed: int) -> dict:
    model = tiny_model(seed)
    width = model.config.hidden_size
    encoder = build_encryptor(width, key_dim=32, hidden_dim=64)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 11)
    ids = torch.randint(0, model.config.vocab_size, (8, 24), generator=gen)
    with torch.no_grad():
        h = input_embeddings(model, ids)
    keys = (torch.randn(32, generator=gen), torch.randn(32, generator=gen))
    init = variant_losses(shift, model, encoder, h, keys)
    other = variant_losses("a" if shift != "a" else "b", model, encoder, h, keys)
    init_info = {"utility": float(init[0]), "privacy": float(init[1]), "diversity": float(init[2]),
                 "grad_cosine_vs_comparison": gradient_cosine(encoder, init, other)}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        utility, privacy, diversity = variant_losses(shift, model, encoder, h, keys)
        (utility + privacy + 2.0 * diversity).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        z = encrypt_embeddings(encoder, h, keys[0])
        cos = float(torch.nn.functional.cosine_similarity(
            h.reshape(-1, width).float(), z.reshape(-1, width).float(), dim=-1).mean())
    return {**init_info, "mean_cosine_after": cos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    results = {shift: [run_variant(shift, args.steps, seed) for seed in args.seeds]
               for shift in ("a", "b", "c")}
    summary = {}
    for shift, rows in results.items():
        summary[shift] = {key: sum(row[key] for row in rows) / len(rows)
                          for key in ("utility", "privacy", "diversity",
                                      "grad_cosine_vs_comparison", "mean_cosine_after")}
        print(f"variant {shift}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary[shift].items()))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"steps": args.steps, "seeds": args.seeds, "summary": summary,
                       "per_seed": results}, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()