"""Accuracy and harm metrics.

The headline harm endpoint is per-item excess absolute loss over the raw estimate,
h = max(0, |m - y| - |r - y|), reported as distributions and totals. Overcorrection
rate (the share of items with h > 0) is a descriptive companion. Predictions are not
clipped before scoring; a physicality-violation rate reports range breaches.
"""

from __future__ import annotations

import numpy as np

PHYS_BOUND = 1.0 + 1e-9  # Z-type Pauli expectations live in [-1, 1]


def excess_absolute_loss(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.abs(m - y) - np.abs(r - y))


def method_metrics(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> dict:
    err = m - y
    h = excess_absolute_loss(m, r, y)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "signed_bias": float(np.mean(err)),
        "excess_loss_total": float(np.sum(h)),
        "excess_loss_mean": float(np.mean(h)),
        "excess_loss_max": float(np.max(h)),
        "overcorrection_rate": float(np.mean(h > 0)),
        "physicality_violation_rate": float(np.mean(np.abs(m) > PHYS_BOUND)),
        "n_items": int(len(y)),
    }
