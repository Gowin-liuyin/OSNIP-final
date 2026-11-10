"""Held-out metric aggregation shared by training and behavioral gates."""
from __future__ import annotations

import math
from collections.abc import Iterable


def aggregate_validation_stats(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        raise ValueError("validation requires at least one batch")
    tokens = sum(row["causal_tokens"] for row in rows)
    if tokens <= 0:
        raise ValueError("validation requires valid causal targets")
    sums = {key: sum(row[key] for row in rows) for key in (
        "clean_nll_sum", "encrypted_nll_sum", "causal_tokens", "representation_tokens", "examples")}
    result = dict(sums)
    skipped = set(sums) | {"clean_ppl", "encrypted_ppl", "norm_max_abs_error"}
    metric_keys = set().union(*(row.keys() for row in rows)) - skipped
    for key in metric_keys:
        if any(key not in row for row in rows):
            raise ValueError(f"inconsistent validation row schema: {key} missing in some batches")
    for key in metric_keys:
        weight = "causal_tokens" if key.startswith("utility") else "examples" if key.endswith("weight") or key == "keys_distinct_fraction" else "representation_tokens"
        denominator = sums[weight]
        if denominator <= 0:
            raise ValueError(f"validation batches carry no {weight}; cannot aggregate {key}")
        result[key] = (max(row[key] for row in rows) if "max_relative_error" in key
                       else sum(row[key] * row[weight] for row in rows) / denominator)
    result["norm_max_abs_error"] = max(row["norm_max_abs_error"] for row in rows)
    result["clean_nll"] = sums["clean_nll_sum"] / tokens
    result["encrypted_nll"] = sums["encrypted_nll_sum"] / tokens
    for key in ("clean", "encrypted"):
        try:
            result[f"{key}_ppl"] = math.exp(result[f"{key}_nll"])
        except OverflowError:
            result[f"{key}_ppl"] = math.inf
    try:
        result["ppl_ratio"] = math.exp(result["encrypted_nll"] - result["clean_nll"])
    except OverflowError:
        result["ppl_ratio"] = math.inf
    result["pareto_score"] = result["utility"] + result["privacy"]
    return result


def control_acceptance(mode: str, baseline: dict[str, float], final: dict[str, float], *,
                       epsilon: float = 0.3, max_kl: float = 0.10,
                       max_ppl_ratio: float = 1.25, privacy_tolerance: float = 0.05) -> dict:
    """Separate numerical completion from evidence that a control learned."""
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if mode == "utility_only":
        initial = baseline.get("utility")
        observed = final.get("utility")
        # A near-identity initialization already below 1e-6 may be retained;
        # otherwise demand a measured improvement on the same validation keys.
        improved = finite(initial) and finite(observed) and (
            observed <= initial * 0.95 or initial <= 1e-6 and observed <= 1e-6)
        return {
            "passed": improved and finite(final.get("ppl_ratio")) and final["ppl_ratio"] <= max_ppl_ratio and observed <= max_kl,
            "initial_kl": initial, "final_kl": observed,
            "kl_improved": improved, "max_kl": max_kl,
            "ppl_ratio": final.get("ppl_ratio"), "max_ppl_ratio": max_ppl_ratio,
            "meaning": "fixed-key held-out KL must improve and utility bounds must hold",
        }
    if mode != "privacy_only":
        raise ValueError(f"unknown control mode: {mode}")
    per_key = {}
    for key in (1, 2):
        cosine = final.get(f"mean_abs_cosine_key{key}")
        hinge = final.get(f"privacy_key{key}")
        initial = baseline.get(f"mean_abs_cosine_key{key}")
        per_key[str(key)] = {
            "passed": all(finite(x) for x in (cosine, hinge, initial)) and cosine < initial and cosine <= epsilon + privacy_tolerance and hinge <= privacy_tolerance,
            "initial_abs_cosine": initial, "final_abs_cosine": cosine, "final_hinge": hinge,
        }
    return {"passed": all(row["passed"] for row in per_key.values()), "keys": per_key,
            "epsilon": epsilon, "privacy_tolerance": privacy_tolerance,
            "meaning": "both held-out key draws must approach the cosine band; this is not an attack result"}
