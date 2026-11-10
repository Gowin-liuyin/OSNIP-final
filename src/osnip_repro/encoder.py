from __future__ import annotations

import math
import torch
from torch import nn


ARCHITECTURES = (
    "mlp_residual_v1",
    "full_width_residual_v1",
    "orthogonal_direction_v1",
    "contextual_residual_v1",
    "mlp_residual_wide_v1",
    "low_rank_residual_v1",
    "key_film_residual_v1",
    "tokenwise_key_v1",
)


class KeyConditionedEncryptor(nn.Module):
    """Key-conditioned residual encryptor variants under one h -> z contract.

    The paper does not disclose the complete client network. Each architecture
    is therefore named and logged as an objective-aligned engineering variant.
    A session key is shared across tokens by default; explicitly aligned
    ``[batch, tokens, key_dim]`` keys are also accepted for tokenwise tests.
    """

    def __init__(self, width: int, key_dim: int = 32, hidden_dim: int = 256, init_scale: float = 1e-2, architecture: str = "mlp_residual_v1", alpha_init: float = 0.05) -> None:
        super().__init__()
        if min(width, key_dim, hidden_dim) <= 0:
            raise ValueError("width, key_dim, and hidden_dim must be positive")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown encryptor architecture: {architecture}; choose one of {ARCHITECTURES}")
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        self.width = int(width)
        self.key_dim = int(key_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.architecture = architecture
        self.alpha_init = float(alpha_init)
        effective_hidden = width if architecture == "full_width_residual_v1" else (hidden_dim * 2 if architecture == "mlp_residual_wide_v1" else hidden_dim)
        self.effective_hidden_dim = effective_hidden

        if architecture in {"mlp_residual_v1", "full_width_residual_v1", "mlp_residual_wide_v1", "tokenwise_key_v1"}:
            self.input_projection = nn.Linear(width, effective_hidden)
            self.key_projection = nn.Linear(key_dim, effective_hidden, bias=False)
            self.output_projection = nn.Linear(effective_hidden, width)
            self._init_output(self.output_projection)
        elif architecture == "low_rank_residual_v1":
            rank = min(effective_hidden, max(8, effective_hidden // 4))
            self.rank = rank
            self.input_projection = nn.Linear(width, rank, bias=False)
            self.key_projection = nn.Linear(key_dim, rank, bias=False)
            self.output_projection = nn.Linear(rank, width)
            self._init_output(self.output_projection)
        elif architecture == "key_film_residual_v1":
            self.input_projection = nn.Linear(width, effective_hidden)
            self.key_film = nn.Linear(key_dim, effective_hidden * 2, bias=False)
            self.output_projection = nn.Linear(effective_hidden, width)
            self._init_output(self.output_projection)
        elif architecture == "contextual_residual_v1":
            self.input_projection = nn.Linear(width, effective_hidden)
            self.key_projection = nn.Linear(key_dim, effective_hidden, bias=False)
            heads = max(1, min(8, effective_hidden // 32))
            while effective_hidden % heads != 0 and heads > 1:
                heads -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=effective_hidden, nhead=heads, dim_feedforward=effective_hidden * 2,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.output_projection = nn.Linear(effective_hidden, width)
            self._init_output(self.output_projection)
        elif architecture == "orthogonal_direction_v1":
            self.register_buffer("alpha_scale", torch.tensor(1.0))
            # Use a full-width directional basis. The scalar amplitude starts
            # near zero, so initialization remains close to identity while
            # the learned direction is explicitly orthogonal to h.
            effective_hidden = width
            self.effective_hidden_dim = effective_hidden
            self.input_projection = nn.Linear(width, effective_hidden)
            self.key_projection = nn.Linear(key_dim, effective_hidden, bias=False)
            self.direction_projection = nn.Linear(effective_hidden, width)
            self.alpha_projection = nn.Linear(effective_hidden, 1)
            nn.init.xavier_uniform_(self.direction_projection.weight, gain=1e-2)
            nn.init.zeros_(self.direction_projection.bias)
            nn.init.zeros_(self.alpha_projection.weight)
            nn.init.constant_(self.alpha_projection.bias, math.log(math.expm1(self.alpha_init)))

    def set_alpha_scale(self, scale: float) -> None:
        if scale < 0 or not math.isfinite(float(scale)):
            raise ValueError("alpha scale must be finite and nonnegative")
        if self.architecture == "orthogonal_direction_v1":
            self.alpha_scale.fill_(float(scale))

    def _init_output(self, layer: nn.Linear) -> None:
        nn.init.xavier_uniform_(layer.weight, gain=self.init_scale)
        nn.init.zeros_(layer.bias)

    def _expand_key(self, h: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if key.ndim == 1:
            if key.shape[0] != self.key_dim:
                raise ValueError("key width disagrees with key_dim")
            return key.expand(*h.shape[:-1], -1)
        if key.shape == h.shape[:-1] + (self.key_dim,):
            return key
        if h.ndim >= 3 and key.shape == h.shape[:-2] + (self.key_dim,):
            return key.unsqueeze(-2).expand(*h.shape[:-1], -1)
        raise ValueError("key must be [key_dim], [batch,key_dim], or align with h")

    def forward(self, h: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if h.ndim < 2 or h.shape[-1] != self.width:
            raise ValueError("h must have shape [..., width]")
        expanded_key = self._expand_key(h, key)
        if self.architecture in {"mlp_residual_v1", "full_width_residual_v1", "mlp_residual_wide_v1", "tokenwise_key_v1"}:
            hidden = torch.tanh(self.input_projection(h) + self.key_projection(expanded_key))
        elif self.architecture == "low_rank_residual_v1":
            hidden = torch.tanh(self.input_projection(h) + self.key_projection(expanded_key))
        elif self.architecture == "key_film_residual_v1":
            base = self.input_projection(h)
            gamma, beta = self.key_film(expanded_key).chunk(2, dim=-1)
            hidden = torch.tanh(base * (1.0 + 0.1 * torch.tanh(gamma)) + beta)
            return h + self.output_projection(hidden)
        elif self.architecture == "contextual_residual_v1":
            squeeze_sequence = h.ndim == 2
            sequence_h = h.unsqueeze(1) if squeeze_sequence else h
            sequence_key = expanded_key.unsqueeze(1) if squeeze_sequence else expanded_key
            hidden = torch.tanh(self.input_projection(sequence_h) + self.key_projection(sequence_key))
            padding_mask = None if attention_mask is None else ~attention_mask.bool()
            if padding_mask is not None and squeeze_sequence:
                padding_mask = padding_mask.unsqueeze(1)
            hidden = self.context_encoder(hidden, src_key_padding_mask=padding_mask)
            output = sequence_h + self.output_projection(hidden)
            return output[:, 0] if squeeze_sequence else output
        elif self.architecture == "orthogonal_direction_v1":
            hidden = torch.tanh(self.input_projection(h) + self.key_projection(expanded_key))
            direction = self.direction_projection(hidden).float()
            h_float = h.float()
            h_norm = h_float.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            unit_h = h_float / h_norm
            direction = direction - (direction * unit_h).sum(dim=-1, keepdim=True) * unit_h
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            alpha = torch.nn.functional.softplus(self.alpha_projection(hidden).float()) * self.alpha_scale.float()
            return (h_float + alpha * h_norm * direction).to(dtype=h.dtype)
        if self.architecture in {"mlp_residual_v1", "full_width_residual_v1", "mlp_residual_wide_v1", "tokenwise_key_v1", "low_rank_residual_v1"}:
            return h + self.output_projection(hidden)
        raise RuntimeError(f"unhandled encryptor architecture: {self.architecture}")


def build_encryptor(width: int, *, key_dim: int = 32, hidden_dim: int = 256, init_scale: float = 1e-2, architecture: str = "mlp_residual_v1", alpha_init: float = 0.05) -> KeyConditionedEncryptor:
    return KeyConditionedEncryptor(width, key_dim=key_dim, hidden_dim=hidden_dim, init_scale=init_scale, architecture=architecture, alpha_init=alpha_init)


def build_encryptor_from_config(config: dict, *, width: int | None = None) -> KeyConditionedEncryptor:
    """Construct an encryptor exactly as recorded by a training run.

    Older artifacts predate configurable architectures, so they intentionally
    fall back to the original ``mlp_residual_v1`` contract.
    """
    model_width = int(config["width"] if width is None else width)
    return build_encryptor(
        model_width,
        key_dim=int(config.get("key_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        init_scale=float(config.get("init_scale", 1e-2)),
        architecture=str(config.get("encryptor_architecture", "mlp_residual_v1")),
        alpha_init=float(config.get("alpha_init", 0.05)),
    )
