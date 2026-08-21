"""Macro descriptive summaries across declared metric cells."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

TYPE7_QUANTILE_METHOD = "Hyndman-Fan type 7 linear interpolation"


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
        values = [float(record["metrics"][name]) for record in cell_records]
        ordered = sorted(0.0 if value == 0.0 else value for value in values)
        summaries[name] = {
            "mean": math.fsum(values) / len(values),
            "q1": _type7_quantile(ordered, 1, 4),
            "median": _type7_quantile(ordered, 1, 2),
            "q3": _type7_quantile(ordered, 3, 4),
            "n_cells": len(values),
        }
    return summaries


def _type7_quantile(
    ordered: Sequence[float], numerator: int, denominator: int
) -> float:
    scaled_rank = (len(ordered) - 1) * numerator
    lower_index, remainder = divmod(scaled_rank, denominator)
    lower = ordered[lower_index]
    if remainder == 0:
        return lower
    upper = ordered[lower_index + 1]
    upper_weight = remainder / denominator
    return math.fsum(((1.0 - upper_weight) * lower, upper_weight * upper))
