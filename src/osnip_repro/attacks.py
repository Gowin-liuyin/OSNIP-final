"""Exact chunked vocabulary retrieval for no-key attack controls."""
from __future__ import annotations

import torch


@torch.no_grad()
def _lex_topk(values: torch.Tensor, ids: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k by (value ascending, id ascending) so exact ties keep lowest ids."""
    if ids.ndim == values.ndim - 1:
        ids = ids.unsqueeze(0).expand_as(values)
    by_id = torch.sort(ids, dim=-1, stable=True).indices
    ordered_ids = ids.gather(-1, by_id)
    ordered_values = values.gather(-1, by_id)
    take = torch.sort(ordered_values, dim=-1, stable=True).indices[..., :k]
    return ordered_values.gather(-1, take), ordered_ids.gather(-1, take)


@torch.no_grad()
def vocabulary_topk(query: torch.Tensor, candidates: torch.Tensor, *, metric: str, k: int = 10, chunk: int = 2048) -> torch.Tensor:
    if query.ndim != 2 or candidates.ndim != 2 or query.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidates must be [rows, same_width]")
    if k < 1 or k > candidates.shape[0] or chunk < 1:
        raise ValueError("invalid top-k or chunk size")
    if metric not in {"l2", "l1"}:
        raise ValueError(metric)
    if not torch.isfinite(query).all():
        raise ValueError("attack input contains non-finite values")
    # Keep the search on the caller's device and dtype. The previous version
    # copied every query and candidate block to CPU float64, which made the
    # formal Llama vocabulary scan effectively unusable. Chunked global top-k
    # merging remains exact for the selected dtype and avoids materializing the
    # full [queries, vocabulary] distance matrix.
    q = query.detach()
    e = candidates.detach().to(device=q.device, dtype=q.dtype)
    if not torch.is_floating_point(q) or q.dtype in (torch.float16, torch.bfloat16):
        q = q.float()
        e = e.float()
    best_values = torch.full((q.shape[0], k), float("inf"), dtype=q.dtype, device=q.device)
    best_ids = torch.full((q.shape[0], k), -1, dtype=torch.long, device=q.device)
    candidate_norms = e.square().sum(-1) if metric == "l2" else None
    for start in range(0, e.shape[0], chunk):
        part = e[start:start + chunk]
        if not torch.isfinite(part).all():
            raise ValueError("attack input contains non-finite values")
        if metric == "l2":
            distance = (q.square().sum(-1, keepdim=True) + candidate_norms[start:start + chunk].unsqueeze(0) - 2 * q @ part.T).clamp_min(0)
        else:
            distance = torch.cdist(q, part, p=1)
        local_k = min(k, len(part))
        values, ids = _lex_topk(distance, torch.arange(len(part), device=distance.device) + start, local_k)
        merged = torch.cat((best_values, values), dim=-1)
        merged_ids = torch.cat((best_ids, ids), dim=-1)
        best_values, best_ids = _lex_topk(merged, merged_ids, k)
    # The expanded squared-distance form loses precision when a candidate is
    # nearly equal to the query, so re-rank the retained union exactly in
    # float64 (cheap: only k candidates per query) and break ties by id.
    retained = e[best_ids.clamp_min(0)].double()
    exact = ((q.double().unsqueeze(1) - retained) ** 2).sum(-1) if metric == "l2" \
        else (q.double().unsqueeze(1) - retained).abs().sum(-1)
    # Lexicographic (distance, id): id-ascending first, then a stable sort by
    # distance keeps the lowest id among exact ties.
    by_id = torch.sort(best_ids, dim=-1).indices
    ids_sorted = best_ids.gather(-1, by_id)
    exact_sorted = exact.gather(-1, by_id)
    ranked = ids_sorted.gather(-1, torch.sort(exact_sorted, dim=-1, stable=True).indices)
    return ranked.cpu()


def topk_counts(rankings: torch.Tensor, target_ids: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, int]:
    target_ids = target_ids.detach().cpu()
    if rankings.shape[0] != target_ids.numel():
        raise ValueError("one target is required per query")
    return {k: int((rankings[:, :k] == target_ids[:, None]).any(-1).sum()) for k in ks}
