"""Per-cell accuracy and harm metrics with an explicit pooled diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

PHYS_BOUND = 1.0 + 1e-9

DEFAULT_CELL_GROUPING = "six-part"
CELL_GROUPINGS: dict[str, tuple[str, ...]] = {
    "six-part": (
        "method",
        "split",
        "family",
        "noise_family",
        "severity",
        "observable",
    ),
    "observable-excluded": (
        "method",
        "split",
        "family",
        "noise_family",
        "severity",
    ),
}


def excess_absolute_loss(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return per-item harm relative to the paired raw estimate."""

    return np.maximum(0.0, np.abs(m - y) - np.abs(r - y))


def pooled_method_metrics(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> dict:
    """Compute flat item-pooled metrics for diagnostics only."""

    m, r, y = _validated_arrays(m, r, y)
    return _metrics_from_arrays(m, r, y)


def method_metrics(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> dict:
    """Backward-compatible alias for the explicitly pooled diagnostic."""

    return pooled_method_metrics(m, r, y)


def build_cell_records(
    method: str,
    items: Sequence[Mapping[str, object]],
    predictions: Sequence[float] | np.ndarray,
    raw_predictions: Sequence[float] | np.ndarray,
    *,
    artifact_id: str,
    grouping: str = DEFAULT_CELL_GROUPING,
) -> list[dict]:
    """Build one traceable metric record per declared cell.

    ``six-part`` implements the default six-field reporting cell contract.
    ``observable-excluded`` keeps sibling observables from the same counts draw
    in one cell. The caller must select the grouping explicitly when departing
    from the default.
    """

    if grouping not in CELL_GROUPINGS:
        raise ValueError(
            f"unknown cell grouping {grouping!r}; expected one of {sorted(CELL_GROUPINGS)}"
        )
    if not method:
        raise ValueError("method must be nonempty")
    if not artifact_id:
        raise ValueError("artifact_id must be nonempty")

    predictions = np.asarray(predictions, dtype=float)
    raw_predictions = np.asarray(raw_predictions, dtype=float)
    if predictions.ndim != 1 or raw_predictions.ndim != 1:
        raise ValueError("predictions and raw_predictions must be one-dimensional")
    if len(items) != len(predictions) or len(items) != len(raw_predictions):
        raise ValueError("items and prediction arrays must have equal lengths")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(raw_predictions)):
        raise ValueError("predictions must be finite")

    fields = CELL_GROUPINGS[grouping]
    grouped: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for item, prediction, raw_prediction in zip(
        items, predictions, raw_predictions, strict=True
    ):
        ideal = float(item["ideal_expectation"])
        item_values = {
            "item_id": str(item["item_id"]),
            "circuit_id": str(item["measurement_group"]),
            "split": str(item["split"]),
            "family": str(item["family"]),
            "stratum": str(item["stratum"]),
            "noise_family": str(item["noise_family"]),
            "severity": str(item["severity"]),
            "observable": str(item["observable"]),
            "prediction": float(prediction),
            "raw_prediction": float(raw_prediction),
            "target": ideal,
            "absolute_error": float(abs(prediction - ideal)),
            "squared_error": float((prediction - ideal) ** 2),
            "signed_error": float(prediction - ideal),
            "excess_absolute_loss": float(
                max(0.0, abs(prediction - ideal) - abs(raw_prediction - ideal))
            ),
            "physicality_violation": bool(abs(prediction) > PHYS_BOUND),
        }
        values = {
            "method": method,
            "split": item_values["split"],
            "family": item_values["family"],
            "noise_family": item_values["noise_family"],
            "severity": item_values["severity"],
            "observable": item_values["observable"],
        }
        grouped[tuple(values[field] for field in fields)].append(item_values)

    records: list[dict] = []
    for key_values in sorted(grouped, key=lambda value: tuple(map(str, value))):
        rows = grouped[key_values]
        key = dict(zip(fields, key_values, strict=True))
        cell_payload = json.dumps(
            {"artifact_id": artifact_id, "grouping": grouping, "key": key},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cell_id = "sha256:" + hashlib.sha256(cell_payload).hexdigest()
        m = np.asarray([row["prediction"] for row in rows], dtype=float)
        r = np.asarray([row["raw_prediction"] for row in rows], dtype=float)
        y = np.asarray([row["target"] for row in rows], dtype=float)
        records.append(
            {
                "cell_id": cell_id,
                "artifact_id": artifact_id,
                "grouping": grouping,
                "key_fields": list(fields),
                "key": key,
                "metrics": _metrics_from_arrays(m, r, y),
                "circuit_ids": sorted({row["circuit_id"] for row in rows}),
                "item_ids": [row["item_id"] for row in rows],
                "items": rows,
            }
        )
    return records


def headline_metrics(cell_records: Sequence[Mapping[str, object]]) -> dict:
    """Return flat macro headline values while retaining cell totals as totals."""

    if not cell_records:
        raise ValueError("at least one cell record is required")
    metrics = [record["metrics"] for record in cell_records]

    def macro(name: str) -> float:
        return float(np.mean([float(metric[name]) for metric in metrics]))

    return {
        "mae": macro("mae"),
        "rmse": macro("rmse"),
        "signed_bias": macro("signed_bias"),
        "excess_loss_total": float(
            sum(float(metric["excess_loss_total"]) for metric in metrics)
        ),
        "excess_loss_mean": macro("excess_loss_mean"),
        "excess_loss_max": float(
            max(float(metric["excess_loss_max"]) for metric in metrics)
        ),
        "overcorrection_rate": macro("overcorrection_rate"),
        "physicality_violation_rate": macro("physicality_violation_rate"),
        "n_items": int(sum(int(metric["n_items"]) for metric in metrics)),
        "n_cells": len(cell_records),
    }


def _validated_arrays(
    m: np.ndarray, r: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (m, r, y))
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("metric inputs must be one-dimensional")
    if not arrays[0].size or len({len(value) for value in arrays}) != 1:
        raise ValueError("metric inputs must be nonempty and have equal lengths")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("metric inputs must be finite")
    return arrays


def _metrics_from_arrays(m: np.ndarray, r: np.ndarray, y: np.ndarray) -> dict:
    m, r, y = _validated_arrays(m, r, y)
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


__all__ = [
    "CELL_GROUPINGS",
    "DEFAULT_CELL_GROUPING",
    "PHYS_BOUND",
    "build_cell_records",
    "excess_absolute_loss",
    "headline_metrics",
    "method_metrics",
    "pooled_method_metrics",
]
