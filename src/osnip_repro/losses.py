from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def utility_kl(clean_logits: torch.Tensor, encrypted_logits: torch.Tensor) -> torch.Tensor:
    """D_KL(p_clean || p_encrypted), averaged over all non-vocabulary axes."""
    clean_prob = F.softmax(clean_logits.float(), dim=-1)
    encrypted_log_prob = F.log_softmax(encrypted_logits.float(), dim=-1)
    token_kl = F.kl_div(encrypted_log_prob, clean_prob, reduction="none").sum(dim=-1)
    return token_kl.mean()


def privacy_hinge(h: torch.Tensor, z: torch.Tensor, epsilon: float) -> torch.Tensor:
    """max(0, |cos(h,z)| - epsilon), averaged over vectors."""
    h_flat = h.float().reshape(-1, h.shape[-1])
    z_flat = z.float().reshape(-1, z.shape[-1])
    cosine = F.cosine_similarity(h_flat, z_flat, dim=-1)
    return (cosine.abs() - float(epsilon)).clamp_min(0.0).mean()


def diversity_hinge(z1: torch.Tensor, z2: torch.Tensor, margin: float) -> torch.Tensor:
    """max(0, margin - ||z1-z2||_2), averaged over vectors."""
    distance = (z1.float() - z2.float()).reshape(-1, z1.shape[-1]).norm(dim=-1)
    return (float(margin) - distance).clamp_min(0.0).mean()


def osnip_loss(
    clean_logits: torch.Tensor,
    encrypted_logits: torch.Tensor,
    h: torch.Tensor,
    z: torch.Tensor,
    *,
    epsilon: float,
    lambda_priv: float,
    z_other: torch.Tensor | None = None,
    diversity_margin: float = 1.0,
    lambda_div: float = 0.0,
) -> dict[str, torch.Tensor]:
    utility = utility_kl(clean_logits, encrypted_logits)
    privacy = privacy_hinge(h, z, epsilon)
    diversity = z.new_zeros(()) if z_other is None else diversity_hinge(z, z_other, diversity_margin)
    total = utility + float(lambda_priv) * privacy + float(lambda_div) * diversity
    return {"total": total, "utility": utility, "privacy": privacy, "diversity": diversity}


def osnip_two_key_loss(
    clean_logits: torch.Tensor,
    encrypted_logits_1: torch.Tensor,
    encrypted_logits_2: torch.Tensor,
    h: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    epsilon: float,
    lambda_priv: float,
    diversity_margin: float = 1.0,
    lambda_div: float = 0.0,
    diversity_z1: torch.Tensor | None = None,
    diversity_z2: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """OSNIP objective averaged over two key draws.

    The previous baseline only constrained the first key draw and used the
    second draw for diversity. This version matches the expectation over
    randomized keys more closely: both draws must preserve utility and satisfy
    the privacy hinge, while the pair is additionally separated.
    """
    utility_1 = utility_kl(clean_logits, encrypted_logits_1)
    utility_2 = utility_kl(clean_logits, encrypted_logits_2)
    privacy_1 = privacy_hinge(h, z1, epsilon)
    privacy_2 = privacy_hinge(h, z2, epsilon)
    utility = 0.5 * (utility_1 + utility_2)
    privacy = 0.5 * (privacy_1 + privacy_2)
    diversity = diversity_hinge(z1 if diversity_z1 is None else diversity_z1, z2 if diversity_z2 is None else diversity_z2, diversity_margin)
    total = utility + float(lambda_priv) * privacy + float(lambda_div) * diversity
    return {
        "total": total,
        "utility": utility,
        "utility_key1": utility_1,
        "utility_key2": utility_2,
        "privacy": privacy,
        "privacy_key1": privacy_1,
        "privacy_key2": privacy_2,
        "diversity": diversity,
    }


def vocabulary_margin_loss(z: torch.Tensor, target_ids: torch.Tensor, embedding_weight: torch.Tensor,
                           *, negative_count: int = 64, topk: int = 10, margin: float = 0.05) -> torch.Tensor:
    """Sampled cosine-margin extension, distinct from formal full-vocab L2 KNN.

    Minimize target similarity relative to the kth distinct negative. This
    optional training surrogate cannot provide security acceptance on its own.
    """
    if z.ndim != 2 or target_ids.ndim != 1 or z.shape[0] != target_ids.shape[0]:
        raise ValueError("z must be [tokens,width] and target_ids must align")
    if negative_count < topk or topk < 1 or margin < 0:
        raise ValueError("require negative_count >= topk >= 1 and nonnegative margin")
    vocab_size = embedding_weight.shape[0]
    if vocab_size < 2 or negative_count > vocab_size - 1:
        raise ValueError("negative_count must fit the distinct non-target vocabulary")
    offsets = torch.arange(negative_count, device=target_ids.device, dtype=torch.long)
    base = target_ids[:, None]
    # A coprime stride cycles through the compressed non-target vocabulary
    # without replacement. Expanding at the target index excludes the target.
    stride = 12345
    while math.gcd(stride, vocab_size - 1) != 1:
        stride -= 1
    compressed = (base * 1103515245 + offsets[None, :] * stride + 1013904223) % (vocab_size - 1)
    negatives = compressed + (compressed >= base).long()
    z_unit = torch.nn.functional.normalize(z.float(), dim=-1)
    target_unit = torch.nn.functional.normalize(embedding_weight[target_ids].float(), dim=-1)
    negative_unit = torch.nn.functional.normalize(embedding_weight[negatives].float(), dim=-1)
    target_similarity = (z_unit * target_unit).sum(-1, keepdim=True)
    negative_similarity = torch.einsum("nd,nkd->nk", z_unit, negative_unit)
    top_negative = negative_similarity.topk(k=topk, dim=-1).values[:, -1:]
    return (target_similarity + float(margin) - top_negative).clamp_min(0.0).mean()
