#!/usr/bin/env python3
"""Micro-batch sweep for a frozen backbone: peak VRAM and seconds per update.

CLAUDE.md requires picking the micro-batch from measured peak memory before a full
run. The model is loaded once and every candidate micro-batch is timed in the same
process, so a 30B-class backbone pays the load cost only once.

The loop shape mirrors ``train_streaming.py``: frozen backbone, a small trainable head
on the input embeddings, cross-entropy through the full vocabulary, backward through
the whole graph (the backbone is frozen but ``z`` needs its gradient).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python scripts/smoke_backbone_vram.py --model-path /root/autodl-tmp/models/deepseek-coder-33b-base \
        --batch-sizes 2 4 8 16 --max-length 128 --steps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.osnip_repro.modeling import load_causal_lm  # noqa: E402

EFFECTIVE_BATCH = 256


def measure(model, embedding, head, optimizer, *, batch_size: int, max_length: int,
            steps: int, warmup: int, vocab: int, device: torch.device, seed: int) -> dict:
    generator = torch.Generator("cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch_size, max_length), generator=generator).to(device)
    attention_mask = torch.ones_like(input_ids)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    timings: list[float] = []
    loss_value = float("nan")
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        began = time.time()
        with torch.no_grad():
            hidden = embedding(input_ids)
        encrypted = head(hidden)
        logits = model(inputs_embeds=encrypted, attention_mask=attention_mask, use_cache=False).logits
        loss = nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        loss_value = float(loss.detach())
        if step >= warmup:
            timings.append(time.time() - began)

    peak = torch.cuda.max_memory_allocated()
    del input_ids, attention_mask
    return {
        "micro_batch": batch_size,
        "gradient_accumulation_steps": EFFECTIVE_BATCH // batch_size,
        "vram_before_gib": round(baseline / 2**30, 2),
        "peak_vram_gib": round(peak / 2**30, 2),
        "seconds_per_micro_step": round(sum(timings) / len(timings), 3),
        "seconds_per_update": round(sum(timings) / len(timings) * (EFFECTIVE_BATCH // batch_size), 1),
        "loss": loss_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for size in args.batch_sizes:
        if EFFECTIVE_BATCH % size:
            raise SystemExit(f"micro-batch {size} does not divide {EFFECTIVE_BATCH}")

    start = time.time()
    tokenizer, model, device = load_causal_lm(args.model_path, device=args.device, dtype=args.dtype)
    load_seconds = round(time.time() - start, 1)

    embedding = model.get_input_embeddings()
    width = int(embedding.embedding_dim)
    vocab = int(embedding.num_embeddings)
    head_dtype = next(model.parameters()).dtype
    head = nn.Sequential(nn.Linear(width, args.head_width), nn.Tanh(), nn.Linear(args.head_width, width))
    head = head.to(device=device, dtype=head_dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

    total = torch.cuda.get_device_properties(0).total_memory
    results, failures = [], []
    for size in args.batch_sizes:
        try:
            record = measure(model, embedding, head, optimizer, batch_size=size,
                             max_length=args.max_length, steps=args.steps, warmup=args.warmup,
                             vocab=vocab, device=device, seed=args.seed)
            record["peak_fraction"] = round(record["peak_vram_gib"] * 2**30 / total, 3)
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            failures.append({"micro_batch": size, "error": "out_of_memory", "detail": str(error)[:200]})
            print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    safe = [r for r in results if r["peak_fraction"] <= 0.85]
    fastest = min(safe, key=lambda r: r["seconds_per_update"]) if safe else None
    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "gpu_total_gib": round(total / 2**30, 2),
        "load_seconds": load_seconds,
        "results": results,
        "failures": failures,
        "recommended_micro_batch": fastest["micro_batch"] if fastest else None,
        "recommended_seconds_per_update": fastest["seconds_per_update"] if fastest else None,
    }
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del tokenizer, model


if __name__ == "__main__":
    main()
