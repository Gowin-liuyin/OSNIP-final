from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OSNIPConfig:
    """Paper-facing objective parameters; architecture remains configurable."""

    width: int
    key_dim: int = 32
    hidden_dim: int = 256
    epsilon: float = 0.3
    diversity_margin: float = 1.0
    lambda_priv: float = 1.0
    lambda_div: float = 1.0
    warmup_steps: int = 1_000
    tau_low: float = 0.01
    tau_high: float = 0.10
    use_iso_norm_projection: bool = True
