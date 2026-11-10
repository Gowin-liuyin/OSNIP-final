from __future__ import annotations

import torch


def iso_norm_project(z: torch.Tensor, h: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project z onto the hypersphere with radius ||h||, per vector."""
    z_norm = z.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    h_norm = h.float().norm(dim=-1, keepdim=True)
    return (z.float() * (h_norm / z_norm)).to(dtype=z.dtype)


def encrypt_embeddings(encoder, h: torch.Tensor, key: torch.Tensor,
                       attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """FP32 client projection followed by the frozen backbone's wire dtype."""
    parameter = next(encoder.parameters(), None)
    client_dtype = parameter.dtype if parameter is not None else h.dtype
    client_h = h.to(dtype=client_dtype)
    client_key = key.to(device=h.device, dtype=client_dtype)
    raw = (encoder(client_h, client_key) if attention_mask is None
           else encoder(client_h, client_key, attention_mask=attention_mask))
    return iso_norm_project(raw, client_h).to(dtype=h.dtype)
