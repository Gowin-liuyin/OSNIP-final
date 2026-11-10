#!/usr/bin/env python3
"""Run the public vocabulary-matching attack against clean/encrypted states.

The attack follows the public vma-external protocol: it compares candidate
hidden states at a chosen layer and feeds its own top-1 token back. For the
large Qwen vocabulary this script can use a public next-token proposal pool;
that mode is reported as reduced-candidate and is not treated as full-vocab
ASR. No target IDs or client key are passed to the attack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from osnip_repro.data import read_text_jsonl
from osnip_repro.checkpointing import load_encoder_state_dict
from osnip_repro.encoder import build_encryptor_from_config
from osnip_repro.modeling import input_embeddings, load_causal_lm
from osnip_repro.projection import encrypt_embeddings
from osnip_repro.vma import autoregressive_match, hidden_at_layer, model_candidates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--text", default="Explain why the sky appears blue.")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--max-positions", type=int, default=2)
    ap.add_argument("--candidate-mode", choices=("full", "proposal"), default="full")
    ap.add_argument("--public-prefix", default="")
    ap.add_argument("--candidate-k", type=int, default=32)
    ap.add_argument("--candidate-chunk", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if min(args.max_positions, args.candidate_k, args.candidate_chunk) < 1:
        ap.error("max-positions, candidate-k and candidate-chunk must be positive")
    if args.output.exists():
        raise FileExistsError(f"choose a new report path: {args.output}")
    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    if Path(config["model_path"]).resolve() != args.model_path.resolve():
        raise ValueError("checkpoint and model paths disagree")
    tokenizer, model, device = load_causal_lm(str(args.model_path), device=args.device, dtype=args.dtype)
    if args.layer < 1 or args.layer > model.config.num_hidden_layers:
        raise ValueError("layer must be a decoder layer, 1..num_hidden_layers")
    encoder = build_encryptor_from_config(config).to(device).eval()
    encoder.load_state_dict(load_encoder_state_dict(str(args.checkpoint), map_location=device, expected_config=config))
    # Public framing, if any, is explicitly supplied, never inferred from labels.
    public_prefix = tokenizer.encode(args.public_prefix, add_special_tokens=False)
    content_ids = tokenizer.encode(args.text, add_special_tokens=False)[:args.max_positions]
    if not content_ids:
        raise ValueError("text produced no attackable content tokens")
    if args.candidate_mode == "proposal" and not public_prefix:
        raise ValueError("proposal mode requires an explicit public prefix")
    actual_ids = public_prefix + content_ids
    ids = torch.tensor([actual_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        h = input_embeddings(model, ids)
        key_rng = torch.Generator(device="cpu").manual_seed(args.seed)
        key = torch.randn(config["key_dim"], generator=key_rng).to(device)
        z = encrypt_embeddings(encoder, h, key)
        clean_target = hidden_at_layer(model, layer=args.layer, inputs_embeds=h, attention_mask=mask)[0, len(public_prefix):]
        encrypted_target = hidden_at_layer(model, layer=args.layer, inputs_embeds=z, attention_mask=mask)[0, len(public_prefix):]
    base_forward = model_candidates(model, layer=args.layer)
    candidate_forward = lambda prefix, candidates: base_forward((*public_prefix, *prefix), candidates)
    proposal_history = []
    @torch.no_grad()
    def propose(prefix):
        prefix_ids = torch.tensor([[*public_prefix, *prefix]], dtype=torch.long, device=device)
        logits = model(input_ids=prefix_ids, use_cache=False).logits[0, -1]
        proposed = torch.topk(logits, k=min(args.candidate_k, logits.numel())).indices.cpu()
        proposal_history.append(proposed.tolist())
        return proposed
    results = {}
    for name, target in (("clean", clean_target), ("encrypted", encrypted_target)):
        proposal_history.clear()
        result = autoregressive_match(target, candidate_forward, vocab_size=model.config.vocab_size,
                                      chunk=args.candidate_chunk, topk=10,
                                      candidate_ids_fn=propose if args.candidate_mode == "proposal" else None)
        true_ids = content_ids
        rankings = result["rankings"]
        coverage = [token in proposal_history[i] for i, token in enumerate(true_ids)] if args.candidate_mode == "proposal" else [True] * len(true_ids)
        results[name] = {
            "predictions": result["predictions"],
            "target_ids_for_scoring_only": true_ids[:len(rankings)],
            "top1_hits": sum(int(result["predictions"][i] == true_ids[i]) for i in range(len(rankings))),
            "top10_hits": sum(int((rankings[i] == true_ids[i]).any()) for i in range(len(rankings))),
            "candidate_coverage": sum(coverage) / max(len(coverage), 1),
            "searched_candidate_sizes": result["searched_candidate_sizes"],
            "l1_errors": result["l1_errors"],
        }
    report = {
        "format": "osnip_vocabulary_matching_attack_v1",
        "status": "full_vocabulary_diagnostic" if args.candidate_mode == "full" else "reduced_candidate_probe",
        "paper_result": False,
        "security_accepted": False,
        "attack_scope": "autoregressive_hidden_state_layerwise_vocabulary_matching",
        "key_attack_performed": False,
        "model_path": str(args.model_path.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "layer": args.layer, "text": args.text, "public_prefix_tokens": len(public_prefix),
        "attack_positions": int(clean_target.shape[0]), "candidate_mode": args.candidate_mode, "candidate_k": args.candidate_k,
        "vocabulary_size": model.config.vocab_size, "results": results,
        "checkpoint_sha256": sha256(args.checkpoint), "data_not_applicable": True,
        "positive_control_valid": results["clean"]["top1_hits"] == len(content_ids),
        "source_sha256": {"scripts/evaluate_vocabulary_matching.py": sha256(Path(__file__)), "src/osnip_repro/vma.py": sha256(ROOT / "src/osnip_repro/vma.py")},
        "limitations": (["reduced candidate pool; not comparable to full-vocabulary ASR"] if args.candidate_mode == "proposal" else []) + [
            "one prompt, one key, one model and one layer; not a security certificate",
            "input embeddings for the optional public prefix are also encrypted in the observation; candidate model uses its clean public framing",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "status": report["status"], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
