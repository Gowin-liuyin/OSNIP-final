from __future__ import annotations


def warmup_weight(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(warmup_steps)))


def utility_safety_gate(utility_loss: float, tau_low: float, tau_high: float) -> float:
    if tau_high <= tau_low:
        raise ValueError("tau_high must be greater than tau_low")
    return min(1.0, max(0.0, (float(tau_high) - float(utility_loss)) / (float(tau_high) - float(tau_low))))


def constraint_weight(step: int, utility_loss: float, *, base: float, warmup_steps: int, tau_low: float, tau_high: float) -> float:
    return float(base) * warmup_weight(step, warmup_steps) * utility_safety_gate(utility_loss, tau_low, tau_high)
