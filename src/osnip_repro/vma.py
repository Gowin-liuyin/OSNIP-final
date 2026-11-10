"""No-key autoregressive vocabulary matching.

Protocol reference: leopardracer/vma-external, vocab_matching_attack.ipynb.
This independently written implementation uses public-model forwards, searches
all supplied vocabulary IDs, and feeds its OWN predictions back. It does not
use the target token sequence, encryptor, key, or future observations.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


class _StopAtLayer(Exception):
    pass


def _cache_key_value_pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (keys, values) from a filled DynamicCache.

    transformers 4.x iteration yields (keys, values) pairs while 5.x yields
    (keys, values, sliding_window) triples; taking the first two entries of
    each layer keeps the prefix-cache path working across both majors.
    """
    layers = getattr(cache, 'layers', None)
    if layers is not None and all(hasattr(layer, 'keys') and hasattr(layer, 'values') for layer in layers):
        return [(layer.keys, layer.values) for layer in layers]
    return [tuple(entry)[:2] for entry in cache]


def _candidate_cache(expanded: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Build a DynamicCache pre-filled with the shared prefix K/V.

    transformers 5.x accepts ``ddp_cache_data``; 4.45/4.x does not and must be
    populated through ``update`` layer by layer. The 4.x update path keeps the
    same expanded views and does not mutate the reusable prefix cache.
    """
    from transformers.cache_utils import DynamicCache

    try:
        return DynamicCache(ddp_cache_data=expanded or None)
    except TypeError:
        cache = DynamicCache()
        for layer_index, (keys, values) in enumerate(expanded):
            cache.update(keys, values, layer_index)
        return cache


@torch.no_grad()
def hidden_at_layer(model: Any, *, layer: int, input_ids: torch.Tensor | None = None,
                    inputs_embeds: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                    past_key_values: Any = None, use_cache: bool = False) -> torch.Tensor:
    """HF convention: 0=input; L=after L decoder blocks (last L includes norm).

    Stop computation at the observed layer; no model head or later layers are
    evaluated. The hook is removed in finally, including on model failure.
    Supports standard Llama/Qwen decoders with .model.layers and .model.norm.
    """
    base = model.model
    if not hasattr(base, 'layers') or not hasattr(base, 'norm'):
        raise ValueError('VMA requires a Llama/Qwen-style decoder')
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError('provide exactly one of input_ids and inputs_embeds')
    if not 0 <= layer <= len(base.layers):
        raise ValueError('layer is outside model depth')
    if layer == 0:
        return model.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
    captured: list[torch.Tensor] = []

    def stop(_module: Any, _inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        captured.append(state)
        raise _StopAtLayer

    hook = base.layers[layer - 1].register_forward_hook(stop)
    try:
        try:
            base(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                 past_key_values=past_key_values, use_cache=use_cache)
        except _StopAtLayer:
            pass
    finally:
        hook.remove()
    if not captured:
        raise RuntimeError('requested decoder layer was not executed')
    state = captured[0]
    return base.norm(state) if layer == len(base.layers) else state


def model_candidates(model: Any, *, layer: int, use_prefix_cache: bool = True) -> Callable[[Sequence[int], torch.Tensor], torch.Tensor]:
    """Public-model-only oracle; cache only the attacker's own predicted prefix.

    Each candidate chunk gets a separate cache object. Expanding prefix K/V
    views avoids re-evaluating the prefix while candidate appends cannot mutate
    the reusable prefix cache. No target IDs, client key or encryptor is used.
    ``use_prefix_cache=False`` retains the independent full-forward reference.
    """
    from transformers.cache_utils import DynamicCache

    device = model.get_input_embeddings().weight.device
    saved_prefix: tuple[int, ...] | None = None
    prefix_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def forward(prefix: Sequence[int], candidates: torch.Tensor) -> torch.Tensor:
        nonlocal saved_prefix, prefix_data
        prefix = tuple(prefix)
        candidate_rows = candidates.to(device).reshape(-1, 1)
        if not use_prefix_cache or layer == 0:
            prefix_rows = torch.tensor(prefix, device=device, dtype=torch.long).reshape(1, -1).expand(len(candidates), -1)
            ids = torch.cat((prefix_rows, candidate_rows), dim=-1)
            return hidden_at_layer(model, layer=layer, input_ids=ids,
                                   attention_mask=torch.ones_like(ids))[:, -1]
        if prefix != saved_prefix:
            prefix_data = []
            if prefix:
                prefix_ids = torch.tensor([prefix], device=device, dtype=torch.long)
                cache = DynamicCache()
                hidden_at_layer(model, layer=layer, input_ids=prefix_ids,
                                attention_mask=torch.ones_like(prefix_ids),
                                past_key_values=cache, use_cache=True)
                prefix_data = _cache_key_value_pairs(cache)
            saved_prefix = prefix
        # DynamicCache.update concatenates into a new tensor. The views in this
        # fresh cache therefore never accumulate tokens from a different chunk.
        expanded = [(k.expand(len(candidates), -1, -1, -1),
                     v.expand(len(candidates), -1, -1, -1)) for k, v in prefix_data]
        candidate_cache = _candidate_cache(expanded)
        mask = torch.ones((len(candidates), len(prefix) + 1), dtype=torch.long, device=device)
        return hidden_at_layer(model, layer=layer, input_ids=candidate_rows,
                               attention_mask=mask, past_key_values=candidate_cache,
                               use_cache=True)[:, -1]

    return forward


@torch.no_grad()
def autoregressive_match(observed: torch.Tensor, candidate_forward: Callable[[Sequence[int], torch.Tensor], torch.Tensor],
                         *, vocab_size: int, chunk: int = 256, topk: int = 10,
                         candidate_ids_fn: Callable[[Sequence[int]], torch.Tensor] | None = None,
                         progress: Callable[[dict[str, int]], None] | None = None) -> dict[str, Any]:
    """Exact full-vocabulary L1 ranks; predicted top1 is the next prefix token.

    No epsilon early stop: all vocab entries are searched at every position.
    A supplied reduced vocabulary must be labelled as such by the evaluator.
    Deterministic ties select the lowest token ID.
    """
    if observed.ndim != 2 or not torch.isfinite(observed).all() or observed.shape[0] == 0:
        raise ValueError('observed must be nonempty finite [positions,width]')
    if not 1 <= topk <= vocab_size or chunk < 1:
        raise ValueError('invalid vocab_size, topk or chunk')
    prefix: list[int] = []
    ranks, errors, searched_sizes = [], [], []
    for observation in observed.detach().cpu().double():
        best_values = torch.empty(0, dtype=torch.float64)
        best_ids = torch.empty(0, dtype=torch.long)
        candidates = (torch.arange(vocab_size, dtype=torch.long) if candidate_ids_fn is None else candidate_ids_fn(tuple(prefix)).detach().cpu().long())
        if candidates.ndim != 1 or candidates.numel() == 0 or candidates.min() < 0 or candidates.max() >= vocab_size:
            raise ValueError('candidate_ids_fn returned invalid candidates')
        candidates = torch.unique(candidates, sorted=True)
        for start in range(0, len(candidates), chunk):
            ids = candidates[start:start + chunk]
            states = candidate_forward(tuple(prefix), ids).detach().cpu().double()
            if states.shape != (len(ids), observed.shape[1]) or not torch.isfinite(states).all():
                raise ValueError('candidate hidden states have invalid shape or values')
            distance = (states - observation).abs().sum(-1)
            values = torch.cat((best_values, distance))
            combined_ids = torch.cat((best_ids, ids))
            # ID sort first, then stable distance sort defines ties globally.
            by_id = combined_ids.argsort(stable=True)
            order = by_id[values[by_id].argsort(stable=True)[:topk]]
            best_values, best_ids = values[order], combined_ids[order]
        # Keep a rectangular result even when a reduced proposal pool has
        # fewer than the requested top-k entries. Missing ranks are explicit
        # sentinels and cannot count as token hits.
        if best_ids.numel() < topk:
            pad = topk - best_ids.numel()
            best_ids = torch.cat((best_ids, torch.full((pad,), -1, dtype=torch.long)))
            best_values = torch.cat((best_values, torch.full((pad,), float("inf"), dtype=torch.float64)))
        ranks.append(best_ids[:topk])
        errors.append(float(best_values[0]))
        searched_sizes.append(len(candidates))
        prefix.append(int(best_ids[0]))
        if progress is not None:
            progress({"positions_completed": len(prefix), "positions_target": len(observed),
                      "candidates_searched": len(candidates)})
    return {'predictions': prefix, 'rankings': torch.stack(ranks), 'l1_errors': errors,
            'vocabulary_size': vocab_size, 'searched_candidate_sizes': searched_sizes,
            'all_candidates_searched': candidate_ids_fn is None,
            'prefix_policy': 'own_top1_predictions', 'metric': 'l1'}
