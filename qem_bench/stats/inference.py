"""Paired cell inference and rank summaries."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PlannedComparisonFamily:
    """A predeclared and immutable Holm family."""

    name: str
    metric_family: str
    comparisons: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.metric_family:
            raise ValueError("comparison-family name and metric_family must be nonempty")
        if not self.comparisons:
            raise ValueError("at least one planned comparison is required")
        if len(set(self.comparisons)) != len(self.comparisons):
            raise ValueError("planned comparisons must be unique")
        if any(len(pair) != 2 or not pair[0] or not pair[1] for pair in self.comparisons):
            raise ValueError("each comparison must contain two nonempty method names")


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("p_values must be a nonempty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for position, index in enumerate(order):
        running = max(running, (m - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilcoxon_rank_sums(differences: Sequence[float]) -> tuple[float, float]:
    """Compute Demsar-style signed rank sums, splitting zero ranks evenly."""

    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("differences must be nonempty and one-dimensional")
    if np.any(~np.isfinite(differences)):
        raise ValueError("differences must be finite")
    ranks = stats.rankdata(np.abs(differences), method="average")
    zero_half = float(np.sum(ranks[differences == 0])) / 2.0
    positive = float(np.sum(ranks[differences > 0])) + zero_half
    negative = float(np.sum(ranks[differences < 0])) + zero_half
    return positive, negative


def paired_wilcoxon_holm(
    cell_records: Sequence[Mapping[str, object]],
    *,
    metric: str,
    family: PlannedComparisonFamily,
) -> list[dict]:
    """Run only the comparisons frozen in ``family`` and apply Holm correction."""

    if family.metric_family != metric:
        raise ValueError(
            f"planned family is for {family.metric_family!r}, not requested metric {metric!r}"
        )
    indexed: dict[str, dict[tuple[object, ...], float]] = {}
    cell_ids: dict[str, dict[tuple[object, ...], str]] = {}
    for record in cell_records:
        method = str(record["key"]["method"])
        pair_key = _paired_cell_key(record)
        if pair_key in indexed.setdefault(method, {}):
            raise ValueError(f"duplicate paired cell for method {method!r}: {pair_key!r}")
        indexed[method][pair_key] = float(record["metrics"][metric])
        cell_ids.setdefault(method, {})[pair_key] = str(record["cell_id"])

    results: list[dict] = []
    raw_p_values: list[float] = []
    for left, right in family.comparisons:
        if left not in indexed or right not in indexed:
            raise ValueError(f"planned comparison {left!r} vs {right!r} is unavailable")
        keys = sorted(set(indexed[left]) & set(indexed[right]), key=repr)
        if not keys:
            raise ValueError(f"planned comparison {left!r} vs {right!r} has no paired cells")
        if set(indexed[left]) != set(indexed[right]):
            raise ValueError(f"planned comparison {left!r} vs {right!r} is not fully paired")
        differences = np.asarray(
            [indexed[left][key] - indexed[right][key] for key in keys], dtype=float
        )
        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="zsplit",
                method="auto",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        positive, negative = wilcoxon_rank_sums(differences)
        denominator = positive + negative
        rank_biserial = (positive - negative) / denominator if denominator else 0.0
        raw_p_values.append(p_value)
        results.append(
            {
                "left": left,
                "right": right,
                "metric": metric,
                "n_pairs": len(keys),
                "statistic": statistic,
                "p_value": p_value,
                "median_difference": float(np.median(differences)),
                "rank_biserial": float(rank_biserial),
                "left_cell_ids": [cell_ids[left][key] for key in keys],
                "right_cell_ids": [cell_ids[right][key] for key in keys],
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for result, adjusted_p in zip(results, adjusted, strict=True):
        result["holm_adjusted_p"] = float(adjusted_p)
        result["comparison_family"] = family.name
    return results


def mean_ranks(
    cell_records: Sequence[Mapping[str, object]],
    *,
    metric: str,
    lower_is_better: bool = True,
) -> dict[str, float]:
    """Compute mean method ranks over fully paired cells."""

    methods = sorted({str(record["key"]["method"]) for record in cell_records})
    if len(methods) < 2:
        raise ValueError("mean ranks require at least two methods")
    indexed: dict[tuple[object, ...], dict[str, float]] = {}
    for record in cell_records:
        key = _paired_cell_key(record)
        method = str(record["key"]["method"])
        indexed.setdefault(key, {})[method] = float(record["metrics"][metric])
    incomplete = [key for key, values in indexed.items() if set(values) != set(methods)]
    if incomplete:
        raise ValueError(f"mean ranks require complete pairing; {len(incomplete)} cells differ")
    rank_rows = []
    for key in sorted(indexed, key=repr):
        values = np.asarray([indexed[key][method] for method in methods], dtype=float)
        if not lower_is_better:
            values = -values
        rank_rows.append(stats.rankdata(values, method="average"))
    averages = np.mean(np.asarray(rank_rows), axis=0)
    return {method: float(rank) for method, rank in zip(methods, averages, strict=True)}


def friedman_rank_test(
    scores: Sequence[Sequence[float]], *, lower_is_better: bool = True
) -> dict:
    """Return mean ranks, Friedman chi-square, and Iman-Davenport F."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 2 or np.any(~np.isfinite(scores)):
        raise ValueError("scores must be a finite N by k matrix with N,k >= 2")
    ranked = np.asarray(
        [
            stats.rankdata(row if lower_is_better else -row, method="average")
            for row in scores
        ]
    )
    n, k = ranked.shape
    average_ranks = np.mean(ranked, axis=0)
    chi_square = (12 * n / (k * (k + 1))) * (
        float(np.sum(average_ranks**2)) - k * (k + 1) ** 2 / 4
    )
    denominator = n * (k - 1) - chi_square
    iman_davenport = (
        ((n - 1) * chi_square / denominator) if denominator > 0 else float("inf")
    )
    return {
        "average_ranks": average_ranks,
        "friedman_chi_square": float(chi_square),
        "iman_davenport_f": float(iman_davenport),
        "n_blocks": n,
        "n_methods": k,
    }


def bias_by_stratum(cell_records: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    """Summarize signed cell bias and its spread for each reported stratum."""

    grouped: dict[str, list[float]] = {}
    for record in cell_records:
        rows = record.get("items", ())
        strata = {str(row["stratum"]) for row in rows}
        if len(strata) != 1:
            raise ValueError("a cell record must belong to exactly one stratum")
        grouped.setdefault(next(iter(strata)), []).append(
            float(record["metrics"]["signed_bias"])
        )
    summaries = {}
    for stratum, biases in sorted(grouped.items()):
        values = np.asarray(biases, dtype=float)
        summaries[stratum] = {
            "macro_bias": float(np.mean(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.ptp(values)),
            "n_cells": len(values),
        }
    return summaries


def _paired_cell_key(record: Mapping[str, object]) -> tuple[object, ...]:
    key = record["key"]
    fields = [field for field in record["key_fields"] if field != "method"]
    return (record["artifact_id"], *(key[field] for field in fields))
