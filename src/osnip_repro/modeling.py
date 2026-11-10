from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

_SPLIT_PREFIX = "model.language_model."


def is_split_text_checkpoint(model_path: str) -> bool:
    """True for checkpoints whose text tower lives under ``model.language_model.*``.

    ``AutoModelForCausalLM`` maps ``model_type=qwen3_5`` to ``Qwen3_5ForCausalLM``,
    which expects ``model.*`` keys, but multimodal Qwen3.5/3.8 checkpoints store the
    language model under ``model.language_model.*`` next to a vision tower. Loading
    them through the auto class leaves every weight at its random initialisation.
    """
    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.is_file() or not (path / "model.safetensors.index.json").is_file():
        return False
    config = json.loads(config_path.read_text())
    return config.get("model_type") == "qwen3_5" and any(
        "ConditionalGeneration" in arch for arch in config.get("architectures") or ())


def checkpoint_dtype(model_path: str) -> torch.dtype:
    config = json.loads((Path(model_path) / "config.json").read_text())
    name = (config.get("text_config") or config).get("dtype") or "bfloat16"
    return getattr(torch, name)


def load_split_text_backbone(model_path: str, torch_dtype: torch.dtype) -> Any:
    """Materialise only the text tower, dropping the vision tower and MTP head."""
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    path = Path(model_path)
    config = Qwen3_5TextConfig(**json.loads((path / "config.json").read_text())["text_config"])
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    known = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
    required = {name for name, _ in model.named_parameters()}
    remaining = set(required)
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(str(path / shard), device="cpu")
        remapped = {}
        for key, value in tensors.items():
            if key.startswith(_SPLIT_PREFIX):
                name = "model." + key[len(_SPLIT_PREFIX):]
            elif key.startswith("lm_head."):
                name = key
            else:
                continue
            if name in known:
                remapped[name] = value.to(torch_dtype)
        remaining.difference_update(remapped)
        model.load_state_dict(remapped, strict=False, assign=True)
        del tensors, remapped
    if remaining:
        raise RuntimeError(f"text backbone is missing {len(remaining)} tensors, e.g. {sorted(remaining)[:3]}")
    return model


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    value = getattr(torch, name)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def load_causal_lm(model_path: str, *, device: str = "cpu", dtype: str = "auto") -> tuple[Any, Any, torch.device]:
    target = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_split_text_checkpoint(model_path):
        torch_dtype = checkpoint_dtype(model_path) if dtype == "auto" else resolve_dtype(dtype, target)
        model = load_split_text_backbone(model_path, torch_dtype)
    else:
        torch_dtype = resolve_dtype(dtype, target)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            **({"dtype": torch_dtype} if int(transformers.__version__.split(".")[0]) >= 5
               else {"torch_dtype": torch_dtype}))
    model.to(target).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, target


def input_embeddings(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(input_ids)


def forward_logits(model: Any, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
    return output.logits


def forward_logits_and_hidden_states(model: Any, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                                     layers: tuple[int, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run one model forward and return logits plus selected hidden states."""
    if not layers:
        raise ValueError("at least one hidden layer is required")
    if any(layer < 0 for layer in layers):
        raise ValueError("hidden layers must be nonnegative")
    output = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                   use_cache=False, output_hidden_states=True)
    hidden = output.hidden_states
    if hidden is None:
        raise RuntimeError("model did not return hidden states")
    max_layer = len(hidden) - 1
    if any(layer > max_layer for layer in layers):
        raise ValueError(f"hidden layer exceeds model depth: max={max_layer}")
    return output.logits, tuple(hidden[layer] for layer in layers)


def forward_hidden_states(model: Any, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                          layers: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Return selected decoder hidden states while preserving gradients.

    Layer 0 is the supplied embedding input. Positive indices follow the
    Hugging Face ``hidden_states`` convention and refer to outputs after the
    corresponding decoder block. This helper is used only by the optional
    layerwise recovery diagnostic; the default OSNIP objective is unchanged.
    """
    if not layers:
        return ()
    if any(layer < 0 for layer in layers):
        raise ValueError("hidden layers must be nonnegative")
    output = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                   use_cache=False, output_hidden_states=True)
    hidden = output.hidden_states
    if hidden is None:
        raise RuntimeError("model did not return hidden states")
    max_layer = len(hidden) - 1
    if any(layer > max_layer for layer in layers):
        raise ValueError(f"hidden layer exceeds model depth: max={max_layer}")
    return tuple(hidden[layer] for layer in layers)


def causal_nll_stats(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, valid_count: int | None = None) -> tuple[torch.Tensor, int]:
    """Return summed causal NLL and the number of valid target tokens.

    A target requires both itself and its predecessor to be non-padding.
    The first real token has no preceding real context in this sequence.
    """
    shifted_logits = logits[:, :-1].float()
    shifted_ids = input_ids[:, 1:]
    shifted_mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    token_loss = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_ids.reshape(-1),
        reduction="none",
    ).reshape_as(shifted_ids)
    valid = int(valid_count) if valid_count is not None else int(shifted_mask.sum())
    return (token_loss * shifted_mask).sum(), valid


def causal_ppl(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    nll, count = causal_nll_stats(logits, input_ids, attention_mask)
    return nll / max(count, 1)


@torch.no_grad()
def selected_target_nll(model: Any, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                        positions: torch.Tensor, targets: torch.Tensor, *, head_chunk: int = 64) -> torch.Tensor:
    """Exact target NLL without retaining every position's full-vocabulary logits.

    Llama/Qwen expose the same decoder and output head used by their causal LM
    forward. Chunk only the selected decoder rows through that head. The fallback
    keeps support for small test models without a decoder/head interface.
    """
    if head_chunk < 1:
        raise ValueError("head_chunk must be positive")
    if hasattr(model, "model") and callable(getattr(model, "get_output_embeddings", None)):
        hidden = model.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                             use_cache=False).last_hidden_state
        selected = hidden.reshape(-1, hidden.shape[-1])[positions]
        total = torch.zeros((), dtype=torch.float32, device=embeddings.device)
        head = model.get_output_embeddings()
        for start in range(0, len(targets), head_chunk):
            logits = head(selected[start:start + head_chunk]).float()
            total += torch.nn.functional.cross_entropy(logits, targets[start:start + head_chunk], reduction="sum")
        return total
    logits = forward_logits(model, embeddings, attention_mask)
    selected_logits = logits.reshape(-1, logits.shape[-1])[positions].float()
    return torch.nn.functional.cross_entropy(selected_logits, targets, reduction="sum")


@torch.no_grad()
def sliding_window_nll(model: Any, embeddings: torch.Tensor, input_ids: torch.Tensor,
                       *, max_length: int, stride: int, window_batch_size: int = 2) -> tuple[float, int]:
    """Evaluate one or more unpadded sequences with non-overlapping targets.

    Windows overlap as required for long-context perplexity, while each target
    token is counted exactly once. ``embeddings`` must already represent the
    clean or encrypted input sequence and have shape ``[batch, length, width]``.
    """
    if embeddings.ndim != 3 or input_ids.ndim != 2 or embeddings.shape[:2] != input_ids.shape:
        raise ValueError("embeddings and input_ids must agree in [batch, length]")
    if max_length < 2 or stride < 1 or stride >= max_length or window_batch_size < 1:
        raise ValueError("require max_length >= 2, 1 <= stride < max_length, and positive window_batch_size")
    total_nll: torch.Tensor | None = None
    total_count = 0
    for row in range(input_ids.shape[0]):
        length = int(input_ids.shape[1])
        if length < 2:
            continue
        previous_end = 0
        end = min(max_length, length)
        windows: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
        while True:
            begin = max(0, end - max_length)
            window_embeddings = embeddings[row:row + 1, begin:end]
            window_ids = input_ids[row:row + 1, begin:end]
            target_begin = max(previous_end, 1)
            if target_begin < end:
                windows.append((window_embeddings, window_ids, target_begin - begin, end - begin))
            previous_end = end
            if end >= length:
                break
            end = min(end + stride, length)
        for batch_start in range(0, len(windows), window_batch_size):
            batch_windows = windows[batch_start:batch_start + window_batch_size]
            batch_length = max(window[3] for window in batch_windows)
            batch_embeddings = embeddings.new_zeros((len(batch_windows), batch_length, embeddings.shape[-1]))
            batch_ids = input_ids.new_zeros((len(batch_windows), batch_length))
            batch_mask = input_ids.new_zeros((len(batch_windows), batch_length))
            for batch_row, (window_embeddings, window_ids, _local_begin, window_length) in enumerate(batch_windows):
                batch_embeddings[batch_row, :window_length] = window_embeddings[0]
                batch_ids[batch_row, :window_length] = window_ids[0]
                batch_mask[batch_row, :window_length] = 1
            positions, targets = [], []
            for batch_row, (_window_embeddings, window_ids, local_begin, window_length) in enumerate(batch_windows):
                positions.append(torch.arange(local_begin - 1, window_length - 1,
                                               device=input_ids.device) + batch_row * batch_length)
                targets.append(window_ids[0, local_begin:])
            positions = torch.cat(positions)
            targets = torch.cat(targets)
            loss = selected_target_nll(model, batch_embeddings, batch_mask, positions, targets)
            total_nll = loss if total_nll is None else total_nll + loss
            total_count += int(targets.numel())
    if total_nll is None:
        return 0.0, total_count
    return float(total_nll.detach().cpu()), total_count
