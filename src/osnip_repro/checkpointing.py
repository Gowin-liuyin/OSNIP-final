from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

import numpy as np
import torch

# Fields whose values define the experiment. Any difference between a training
# checkpoint's embedded config and the config a consumer expects means the
# artifact does not belong to that run, even when tensor shapes agree.
RECIPE_FIELDS = (
    "learning_rate", "weight_decay", "grad_clip", "gradient_balance",
    "batch_size", "gradient_accumulation_steps", "warmup_steps",
    "constraint_ramp_steps", "loss_warmup_steps", "constraint_mode", "gate_scope",
    "tau_low", "tau_high", "lambda_util", "lambda_priv", "lambda_div",
    "epsilon", "diversity_margin", "diversity_space",
    "key_dim", "hidden_dim", "alpha_init", "alpha_ramp_steps", "encryptor_architecture",
    "max_length", "dtype",
)


def flatten_recipe(config: dict[str, Any]) -> dict[str, Any]:
    """Collect recipe fields from flat run configs and nested pinned configs.

    Pinned experiment configs group values under ``objective``/``training``/
    ``data`` blocks while training configs are flat argparse namespaces; top
    level wins on collision so a flat run config always describes itself.
    """
    flat: dict[str, Any] = {}
    for block in ("objective", "training", "data"):
        section = config.get(block)
        if isinstance(section, dict):
            flat.update(section)
    flat.update({key: value for key, value in config.items()
                 if not isinstance(value, dict)})
    return flat


NUMERIC_RECIPE_FIELDS = (
    "learning_rate", "weight_decay", "grad_clip",
    "batch_size", "gradient_accumulation_steps", "warmup_steps",
    "constraint_ramp_steps", "loss_warmup_steps", "tau_low", "tau_high",
    "lambda_util", "lambda_priv", "lambda_div", "epsilon", "diversity_margin",
    "key_dim", "hidden_dim", "alpha_init", "alpha_ramp_steps", "max_length",
)


def require_locked_recipe(config: dict[str, Any]) -> None:
    """Refuse configs whose numeric recipe fields are placeholder strings.

    Pinned plans previously shipped ``TUNE_...`` strings that silently became
    whatever a later command line chose; a locked recipe must carry numbers.
    Enum-like fields (modes, spaces, dtypes) are legitimately strings and are
    not part of this check.
    """
    flat = flatten_recipe(config)
    unlocked = [name for name in NUMERIC_RECIPE_FIELDS if isinstance(flat.get(name), str)]
    if unlocked:
        raise ValueError("unlocked recipe placeholders: " + ", ".join(unlocked))


def recipe_fingerprint(config: dict[str, Any]) -> str:
    """Stable digest of the experiment-defining fields of a run config.

    Computed on demand so checkpoints saved before this fingerprint existed
    remain verifiable. Absent fields are encoded explicitly, so a config that
    lacks a field can never collide with one that sets it.
    """
    flat = flatten_recipe(config)
    payload = {name: flat.get(name, "<absent>") for name in RECIPE_FIELDS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Normalize serialized RNG state after device-aware checkpoint loading.

    ``torch.load(..., map_location="mps")`` can move every serialized tensor,
    including the CPU RNG state, onto MPS. PyTorch's RNG setters require a
    CPU ``torch.uint8`` tensor, so restore must normalize the device and dtype.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.uint8)


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(item) for item in state["torch_cuda"]])
    if torch.backends.mps.is_available() and "torch_mps" in state:
        torch.mps.set_rng_state(_cpu_byte_tensor(state["torch_mps"]))


def save_training_checkpoint(
    path: str,
    *,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    global_step: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": "osnip_training_checkpoint_v1",
        "global_step": int(global_step),
        "encoder": encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "config": config,
        "metrics": metrics,
        "extra": {} if extra is None else extra,
        "rng_state": capture_rng_state(),
    }
    tmp_path = str(path) + ".tmp"
    torch.save(payload, tmp_path)
    import os as _os
    _os.replace(tmp_path, path)


def load_training_checkpoint(
    path: str,
    *,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "osnip_training_checkpoint_v1":
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
    encoder.load_state_dict(_encoder_state(payload))
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload


def validate_resume_budget(path: str, target_steps: int) -> int:
    """Reject an empty resume before loading the LLM or writing run artifacts."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "osnip_training_checkpoint_v1":
        raise ValueError("resume requires an OSNIP training checkpoint")
    step = int(payload["global_step"])
    if step < 0 or target_steps <= step:
        raise ValueError(f"resume target steps ({target_steps}) must exceed checkpoint global_step ({step})")
    return step


def _encoder_state(payload: Any, *, expected_config: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Resolve legacy runtime state only when checkpoint metadata proves it."""
    full = isinstance(payload, dict) and isinstance(payload.get("encoder"), dict)
    config = payload.get("config", {}) if full else (expected_config or {})
    if full and expected_config is not None:
        defaults = {"encryptor_architecture": "mlp_residual_v1", "key_dim": 32,
                    "hidden_dim": 256, "alpha_ramp_steps": 0}
        for name in ("width", "model_path", *defaults):
            if name not in config and name not in expected_config:
                continue
            if config.get(name, defaults.get(name)) != expected_config.get(name, defaults.get(name)):
                raise ValueError(f"checkpoint/config mismatch for {name}")
        if recipe_fingerprint(config) != recipe_fingerprint(expected_config):
            raise ValueError("checkpoint/config recipe fingerprint mismatch")
    state = dict(payload["encoder"] if full else payload) if isinstance(payload, dict) else None
    if not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                            for key, value in state.items()):
        raise ValueError("encoder artifact must be a nonempty tensor state dict")
    orthogonal = config.get("encryptor_architecture") == "orthogonal_direction_v1" or "alpha_projection.weight" in state
    if orthogonal:
        if "alpha_scale" not in state:
            if "alpha_ramp_steps" not in config:
                raise ValueError("legacy orthogonal artifact lacks ramp metadata; supply its original config")
            ramp_steps = int(config["alpha_ramp_steps"])
            if ramp_steps < 0:
                raise ValueError("negative alpha ramp in checkpoint")
            if ramp_steps and not full:
                raise ValueError("legacy bare ramp artifact has no saved scale/step; use its full training checkpoint")
            step = int(payload["global_step"]) if full else 0
            if step < 0:
                raise ValueError("negative checkpoint global_step")
            scale = min(1.0, step / ramp_steps) if ramp_steps else 1.0
            state["alpha_scale"] = torch.tensor(scale)
        value = state["alpha_scale"]
        if value.ndim != 0 or not math.isfinite(float(value.detach().cpu())) or float(value.detach().cpu()) < 0:
            raise ValueError("invalid alpha scale in encoder artifact")
    return state


def load_encoder_state_dict(path: str, *, map_location: str | torch.device = "cpu",
                            expected_config: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Load a bare encoder or full checkpoint, including its effective amplitude.

    Historical full ramp checkpoints derive the missing scale from their own
    global_step and config. Historical bare ramp exports cannot prove a step
    and are rejected. New bare exports carry the scale in their state dict.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return _encoder_state(payload, expected_config=expected_config)
