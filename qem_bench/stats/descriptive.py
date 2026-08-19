"""Macro descriptive summaries across declared metric cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def macro_mean_iqr(
    cell_records: Sequence[Mapping[str, object]],
) -> dict[str, dict]:
    """Summarize every scalar cell metric by macro mean and interquartile range."""

    if not cell_records:
        raise ValueError("at least one cell record is required")
    metric_names = tuple(cell_records[0]["metrics"])
    summaries: dict[str, dict] = {}
    for name in metric_names:
        if name == "n_items":
            continue
        values = np.asarray(
            [float(record["metrics"][name]) for record in cell_records], dtype=float
        )
        summaries[name] = {
            "mean": float(np.mean(values)),
            "q1": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q3": float(np.quantile(values, 0.75)),
            "n_cells": len(values),
        }
    return summaries
