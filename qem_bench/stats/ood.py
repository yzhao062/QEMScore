"""Both open OOD-degradation definitions, kept side by side."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def raw_normalized_ood_degradation(
    method_mae: Mapping[str, float],
    raw_mae: Mapping[str, float],
    *,
    tau: float,
    source_split: str = "S0",
) -> dict:
    """Compute raw-normalized improvements and degradation with a required floor.

    Per-split degradation remains signed and uncapped. Only the composite over
    non-source splits is clipped to [0, 1].
    """

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a finite, positive frozen MAE floor")
    if set(method_mae) != set(raw_mae):
        raise ValueError("method_mae and raw_mae must contain identical splits")
    if source_split not in method_mae:
        raise ValueError(f"source split {source_split!r} is missing")
    if len(method_mae) < 2:
        raise ValueError("at least one OOD split is required")
    improvements = {}
    for split in method_mae:
        method_value = float(method_mae[split])
        raw_value = float(raw_mae[split])
        if not np.isfinite(method_value) or not np.isfinite(raw_value):
            raise ValueError("MAE values must be finite")
        improvements[split] = (raw_value - method_value) / max(raw_value, tau)
    source_improvement = improvements[source_split]
    degradation = {
        split: source_improvement - improvement
        for split, improvement in improvements.items()
        if split != source_split
    }
    composite = float(np.mean([np.clip(value, 0.0, 1.0) for value in degradation.values()]))
    return {
        "formula": "raw-normalized-with-floor",
        "tau": float(tau),
        "source_split": source_split,
        "improvement": improvements,
        "degradation": degradation,
        "composite": composite,
    }


def method_s0_relative_degradation_rejected(
    method_mae: Mapping[str, float], *, source_split: str = "S0"
) -> dict:
    """Compute the unstable method-S0-relative alternative retained for comparison."""

    if source_split not in method_mae:
        raise ValueError(f"source split {source_split!r} is missing")
    source = float(method_mae[source_split])
    if not np.isfinite(source) or source <= 0:
        raise ValueError("the rejected formula requires strictly positive source MAE")
    degradation = {}
    for split, value in method_mae.items():
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("MAE values must be finite")
        if split != source_split:
            degradation[split] = (value - source) / source
    return {
        "formula": "rejected-method-S0-relative",
        "source_split": source_split,
        "degradation": degradation,
    }
