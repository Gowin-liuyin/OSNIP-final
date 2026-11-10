"""Observations for the four attack targets.

plaintext: the server sees the original token embedding.
aloepri: the server sees that embedding plus AloePri's alpha_e noise.
osnip: the server sees the iso-norm projected z from the encryptor.
a+b: the server sees that z after the same AloePri noise.

ISA and KNN both consume these vectors. ISA also needs the model that the
server can differentiate through. This module does not load that model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

TARGETS = ("plaintext", "aloepri", "osnip", "a+b")


def require_checkpoint(target: str, checkpoint: Any) -> None:
    if target in {"osnip", "a+b"} and checkpoint is None:
        raise ValueError(f"{target} needs --checkpoint. Refusing to invent an OSNIP ciphertext.")


def observe(
    embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    target: str,
    *,
    seed: int = 42,
    alpha_e: float = 1.0,
    encoder: Any = None,
    key_dim: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the per-token vector the attacker sees, and how it was built."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if embeddings.ndim != 2 or token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError("embeddings must be [vocab, width] and token ids a nonempty vector")
    clean = embeddings[token_ids]
    info: dict[str, Any] = {"target": target, "alpha_e": 0.0, "osnip": False}
    if target == "plaintext":
        return clean, info
    if target == "aloepri":
        return _noise(clean, embeddings, alpha_e, seed), {**info, "alpha_e": float(alpha_e)}
    if encoder is None or key_dim is None:
        raise ValueError(f"{target} needs a loaded OSNIP encryptor")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randn(int(key_dim), generator=generator).to(clean.device, dtype=clean.dtype)
    from osnip_repro.projection import encrypt_embeddings

    z = encrypt_embeddings(encoder, clean.unsqueeze(0), key)[0]
    info.update(osnip=True, key_dim=int(key_dim), keys_per_prompt=1, key_generator="cpu")
    if target == "osnip":
        return z, info
    noised = _noise(z, embeddings, alpha_e, seed + 1)
    return noised, {**info, "alpha_e": float(alpha_e), "composition": "osnip z, then AloePri embedding noise"}


def _noise(rows: torch.Tensor, table: torch.Tensor, alpha_e: float, seed: int) -> torch.Tensor:
    if alpha_e < 0:
        raise ValueError("alpha_e must be nonnegative")
    sigma = float(table.detach().float().std())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(rows.shape, generator=generator).to(rows.device, dtype=rows.dtype)
    return rows + float(alpha_e) * sigma * noise
