"""Circuit-blocked hierarchical bootstrap for macro cell statistics."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_circuits: int
    seed: int
    metric: str
    paired: bool

    @property
    def width(self) -> float:
        return self.upper - self.lower


def circuit_blocked_bootstrap(
    cell_records: Sequence[Mapping[str, object]],
    *,
    metric: str = "mae",
    reference_records: Sequence[Mapping[str, object]] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int,
) -> BootstrapInterval:
    """Bootstrap physical circuits within their source pools.

    A sampled circuit contributes all of its severity, observable, and cell rows.
    For paired method differences, the same circuit draws and item identities are
    used for both methods.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rows, grouping = _flatten(cell_records)
    reference_by_item = None
    if reference_records is not None:
        reference_rows, reference_grouping = _flatten(reference_records)
        if reference_grouping != grouping:
            raise ValueError("paired bootstrap requires identical cell groupings")
        reference_by_item = {
            (row["artifact_id"], row["item_id"]): row for row in reference_rows
        }
        left_ids = {(row["artifact_id"], row["item_id"]) for row in rows}
        if left_ids != set(reference_by_item):
            raise ValueError("paired bootstrap requires identical artifact/item pairs")

    circuits: dict[tuple[object, ...], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        stratum_key = (str(row["bootstrap_stratum_id"]),)
        circuits[stratum_key][str(row["circuit_id"])].append(row)
    n_circuits = sum(len(values) for values in circuits.values())
    if n_circuits < 2:
        raise ValueError("bootstrap requires at least two circuits")

    estimate = _macro_statistic(rows, metric)
    if reference_by_item is not None:
        estimate -= _macro_statistic(list(reference_by_item.values()), metric)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    strata = [
        (key, tuple(sorted(by_circuit)))
        for key, by_circuit in sorted(circuits.items(), key=lambda pair: repr(pair[0]))
    ]
    for sample_index in range(n_resamples):
        sampled_rows: list[dict] = []
        for stratum_key, circuit_ids in strata:
            draws = rng.integers(0, len(circuit_ids), size=len(circuit_ids))
            by_circuit = circuits[stratum_key]
            for draw_number, draw in enumerate(draws):
                circuit_rows = by_circuit[circuit_ids[int(draw)]]
                for row in circuit_rows:
                    copied = dict(row)
                    copied["bootstrap_copy"] = draw_number
                    sampled_rows.append(copied)
        statistic = _macro_statistic(sampled_rows, metric)
        if reference_by_item is not None:
            sampled_reference = []
            for row in sampled_rows:
                reference = dict(reference_by_item[(row["artifact_id"], row["item_id"])])
                reference["bootstrap_copy"] = row["bootstrap_copy"]
                sampled_reference.append(reference)
            statistic -= _macro_statistic(sampled_reference, metric)
        samples[sample_index] = statistic

    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n_resamples=n_resamples,
        n_circuits=n_circuits,
        seed=int(seed),
        metric=metric,
        paired=reference_records is not None,
    )


def _flatten(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict], str]:
    if not records:
        raise ValueError("at least one cell record is required")
    groupings = {str(record["grouping"]) for record in records}
    if len(groupings) != 1:
        raise ValueError("all records must use one cell grouping")
    flattened = []
    for record in records:
        key = record["key"]
        cell_key = tuple((field, key[field]) for field in record["key_fields"])
        for item in record["items"]:
            row = dict(item)
            row["artifact_id"] = str(item.get("artifact_id", record["artifact_id"]))
            row["cell_key"] = cell_key
            flattened.append(row)
    return flattened, next(iter(groupings))


def _macro_statistic(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        copy = row.get("bootstrap_copy")
        grouped[(row["cell_key"], copy)].append(row)
    # Different bootstrap copies of the same circuit belong to the same statistical
    # cell. Remove the copy discriminator after it has preserved multiplicity.
    by_cell: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for (cell_key, _copy), values in grouped.items():
        by_cell[cell_key].extend(values)
    values = [_item_metric(cell_rows, metric) for cell_rows in by_cell.values()]
    return float(np.mean(values))


def _item_metric(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    if metric == "mae":
        return float(np.mean([row["absolute_error"] for row in rows]))
    if metric == "rmse":
        return float(np.sqrt(np.mean([row["squared_error"] for row in rows])))
    if metric == "signed_bias":
        return float(np.mean([row["signed_error"] for row in rows]))
    if metric == "excess_loss_mean":
        return float(np.mean([row["excess_absolute_loss"] for row in rows]))
    if metric == "excess_loss_total":
        return float(np.sum([row["excess_absolute_loss"] for row in rows]))
    if metric == "overcorrection_rate":
        return float(np.mean([row["excess_absolute_loss"] > 0 for row in rows]))
    if metric == "physicality_violation_rate":
        return float(np.mean([row["physicality_violation"] for row in rows]))
    raise ValueError(f"unsupported bootstrap metric {metric!r}")
