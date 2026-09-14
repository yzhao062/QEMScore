"""Source-validation incremental-value gate (the GD-11 panel option).

The full method must have control/full macro MAE > 1.05 against *both*
feature-only and noisy-only ridge controls in at least two continuous families.
Each paired 95% circuit-block interval for control MAE minus full MAE must
also lie strictly above zero. The interval establishes a positive difference,
not a 5% population benefit. Intervals are conditional on the fitted/selected
models, pointwise, and do not include training or model-selection uncertainty.

This is a claim gate, not an equivalence test: failure does not prove that a
method ignores measurements. Missing evidence is distinct from failure.
For non-ridge full methods, the linear controls do not isolate measurement use
from model capacity; a pass still needs the separate model-matched shuffle audit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math

from qemscore.datasets.schema import FAMILY_STRATA
from qemscore.stats.bootstrap import circuit_blocked_bootstrap

CONTROLS = ("feat-only", "noisy-only")
_CELL_FIELDS = ("split_id", "family", "noise_family", "severity", "observable")


def evaluate_incremental_value(
    validation_items: Sequence[Mapping[str, object]],
    predictions_by_method: Mapping[str, Mapping[str, float]],
    *,
    full_method: str = "ridge",
    ratio_margin: float = 1.05,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int,
) -> dict:
    """Evaluate paired predictions keyed by validation ``item_id``.

    Pass only source validation rows from one split-v2 split/partition; never
    filter a combined train/test stream here. Invalid roles, identities, or
    partial prediction maps raise ValueError. No rows (including legacy-v1's
    absent validation), missing methods, fewer than two continuous families,
    or insufficient circuit replication return ``status='not_evaluable'``.
    Random Clifford rows are excluded before targets/predictions are read.

    Macro cells are (split ID, family, noise family, severity, observable).
    Physical circuits are sampled within their declared circuit pools; all
    their severities, observables, shots, and replicates travel together.
    Each pool needs at least two circuits, since a singleton cannot estimate
    circuit sampling uncertainty. The required family count is fixed at two.
    """
    # Normalize before the guards. A NumPy scalar is an ordinary input here, and
    # comparing against one yields numpy.bool_, which is not a Python bool and
    # which strict JSON serialization refuses. Converting the arithmetic operands
    # themselves keeps every derived flag native, rather than converting only the
    # copy that reaches the result.
    ratio_margin = float(ratio_margin)
    confidence = float(confidence)
    if not math.isfinite(ratio_margin) or ratio_margin <= 1:
        raise ValueError("ratio_margin must be finite and greater than one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # These two travel into the result as metadata, so an integer too large to
    # render as decimal would make the returned dictionary unserializable.
    for name, value in (("n_resamples", n_resamples), ("seed", seed)):
        try:
            str(value)
        except ValueError as error:
            raise ValueError(f"{name} must be convertible to decimal") from error
    if not isinstance(full_method, str) or not full_method or full_method in CONTROLS:
        raise ValueError("full_method must be a nonempty name distinct from both controls")

    result = {
        "status": "not_evaluable",
        "reason": None,
        "full_method": full_method,
        "controls": list(CONTROLS),
        "evaluation_role": "source_validation",
        "ratio_margin": float(ratio_margin),
        "required_families": 2,
        "confidence": float(confidence),
        "n_resamples": n_resamples,
        "seed": seed,
        "interval_metric": "control_macro_mae_minus_full_macro_mae",
        "interval_scope": "pointwise; conditional on fitted and selected models",
        "excluded_items": 0,
        "families": {},
        "passing_families": [],
    }
    rows_by_family = defaultdict(list)
    identities = set()
    scopes = set()
    circuit_strata = defaultdict(set)
    for item in validation_items:
        if item.get("split") != "validation" or item.get("domain") != "source":
            raise ValueError("gate requires source validation rows only")
        if item.get("dataset_schema_version") != "split-v2":
            raise ValueError("source validation requires split-v2 metadata")
        for field in ("item_id", "split_id", "partition_id", "family", "stratum"):
            _identity(item, field)
        scopes.add((item["split_id"], item["partition_id"]))
        item_id = item["item_id"]
        if item_id in identities:
            raise ValueError("validation item IDs must be unique")
        identities.add(item_id)
        family = item["family"]
        if family not in FAMILY_STRATA or item["stratum"] != FAMILY_STRATA[family]:
            raise ValueError("family and stratum must match the dataset contract")
        if item["stratum"] != "continuous_regression":
            result["excluded_items"] += 1
            continue
        circuit = _identity(item, "circuit_id")
        # The canonical pool is authoritative. An optional alias may repeat it but
        # must never redefine it: a caller that widened the pools this way could
        # turn insufficient circuit replication into a passing scientific claim.
        pool = _identity(item, "circuit_pool_id")
        if ("bootstrap_stratum_id" in item
                and _identity(item, "bootstrap_stratum_id") != pool):
            raise ValueError("bootstrap_stratum_id must match circuit_pool_id")
        circuit_strata[circuit].add((family, pool))
        for field in _CELL_FIELDS:
            _identity(item, field)
        target = float(item["ideal_expectation"])
        if not math.isfinite(target):
            raise ValueError("validation targets must be finite")
        rows_by_family[family].append(dict(item, bootstrap_stratum_id=pool))
    if len(scopes) > 1:
        raise ValueError("evaluate one split and partition at a time")
    if any(len(strata) != 1 for strata in circuit_strata.values()):
        raise ValueError("physical circuits must not span families or bootstrap strata")
    if not rows_by_family:
        result["reason"] = "no_continuous_source_validation"
        return result
    required = (full_method, *CONTROLS)
    missing = [method for method in required if method not in predictions_by_method]
    if missing:
        result["reason"] = "missing_methods: " + ", ".join(missing)
        return result
    continuous_ids = {item["item_id"] for rows in rows_by_family.values() for item in rows}
    for method in required:
        predictions = predictions_by_method[method]
        if not isinstance(predictions, Mapping):
            raise ValueError("predictions must be keyed by validation item ID")
        if not continuous_ids <= predictions.keys() or predictions.keys() - identities:
            raise ValueError("prediction item IDs must match source validation rows")
        if any(not math.isfinite(float(predictions[key])) for key in continuous_ids):
            raise ValueError("predictions must be finite")

    for family, rows in sorted(rows_by_family.items()):
        rows = sorted(rows, key=lambda item: item["item_id"])
        pools = defaultdict(set)
        for item in rows:
            pools[item["bootstrap_stratum_id"]].add(item["circuit_id"])
        records = {
            method: _records(rows, predictions_by_method[method]) for method in required
        }
        maes = {method: _macro_mae(values) for method, values in records.items()}
        family_result = {
            "status": "not_evaluable",
            "reason": None,
            "n_items": len(rows),
            "n_circuits": sum(map(len, pools.values())),
            "circuits_by_pool": {pool: len(ids) for pool, ids in sorted(pools.items())},
            "n_cells": len(records[full_method]),
            "full_mae": maes[full_method],
            "comparisons": {},
        }
        result["families"][family] = family_result
        if any(len(ids) < 2 for ids in pools.values()):
            family_result["reason"] = "fewer_than_two_circuits_in_a_pool"
            continue
        for control in CONTROLS:
            interval = circuit_blocked_bootstrap(
                records[control], reference_records=records[full_method],
                confidence=confidence, n_resamples=n_resamples, seed=seed,
            )
            # Finite macro means do not guarantee a finite interval: resampling can
            # draw one large-error circuit repeatedly and overflow the reduction,
            # after which the quantiles are NaN. Checking only the lower bound
            # would then attach a passing decision to a malformed interval.
            if not all(math.isfinite(value) for value in (
                interval.estimate, interval.lower, interval.upper
            )):
                raise ValueError("incremental-value interval must be finite")
            full_mae = maes[full_method]
            control_mae = maes[control]
            # Cross multiplication defines zero-error cases without JSON infinity.
            margin_met = bool(control_mae > ratio_margin * full_mae)
            positive_interval = bool(interval.lower > 0)
            # An exact zero is not the only denominator that has no representable
            # ratio: a tiny positive full MAE overflows to infinity, which then
            # fails strict JSON serialization. Report both as undefined.
            ratio = control_mae / full_mae if full_mae else float("inf")
            ratio_defined = full_mae > 0 and math.isfinite(ratio)
            family_result["comparisons"][control] = {
                "control_mae": control_mae,
                "control_over_full_mae": ratio if ratio_defined else None,
                "ratio_defined": ratio_defined,
                "ratio_undefined_reason": (
                    None if ratio_defined
                    else ("zero_full_mae" if not full_mae else "ratio_overflow")
                ),
                "margin_met": margin_met,
                "difference_interval": asdict(interval),
                "positive_interval": positive_interval,
                "passed": margin_met and positive_interval,
            }
        passed = all(value["passed"] for value in family_result["comparisons"].values())
        family_result["status"] = "passed" if passed else "failed"
        if passed:
            result["passing_families"].append(family)

    passing = len(result["passing_families"])
    unavailable = sum(value["status"] == "not_evaluable"
                      for value in result["families"].values())
    if len(rows_by_family) < 2:
        result["reason"] = "fewer_than_two_continuous_families"
    elif passing >= 2:
        result.update(status="passed", reason="two_families_meet_both_comparisons")
    elif passing + unavailable >= 2:
        result["reason"] = "insufficient_circuit_replication"
    else:
        result.update(status="failed", reason="fewer_than_two_families_meet_both_comparisons")
    return result


def _identity(item: Mapping[str, object], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"validation rows require nonempty {field}")
    return value


def _records(rows: list[dict], predictions: Mapping[str, float]) -> list[dict]:
    cells = defaultdict(list)
    for item in rows:
        cells[tuple(item[field] for field in _CELL_FIELDS)].append({
            "item_id": item["item_id"],
            "circuit_id": item["circuit_id"],
            "bootstrap_stratum_id": item["bootstrap_stratum_id"],
            "absolute_error": _finite_absolute_error(
                predictions[item["item_id"]], item["ideal_expectation"]
            ),
        })
    return [{
        "artifact_id": "source-validation",
        "grouping": "incremental-value-v1",
        "key_fields": list(_CELL_FIELDS),
        "key": dict(zip(_CELL_FIELDS, key, strict=True)),
        "items": values,
    } for key, values in sorted(cells.items())]


def _finite_absolute_error(prediction: object, target: object) -> float:
    """Absolute error, refusing a difference that leaves binary64.

    Finite operands are not enough: a target near the negative limit against a
    prediction near the positive one overflows the subtraction. Every derived
    error is checked here, before the replication refusal reads a mean, so no
    return path can carry a nonfinite value out of this module.
    """
    error = abs(float(prediction) - float(target))
    if not math.isfinite(error):
        raise ValueError("derived absolute errors must be finite")
    return error


def _macro_mae(records: list[dict]) -> float:
    """Equal-weight mean over cells, refusing any input arithmetic cannot represent.

    ``math.fsum`` raises ``OverflowError`` on an intermediate that leaves binary64,
    which would surface as an unhandled numerical error rather than as a decision.
    A macro mean that cannot be formed is not evidence, so it is rejected here
    instead of being carried into a comparison.
    """
    try:
        value = math.fsum(
            math.fsum(row["absolute_error"] for row in record["items"])
            / len(record["items"])
            for record in records
        ) / len(records)
    except OverflowError as error:
        raise ValueError("macro MAE overflowed binary64") from error
    if not math.isfinite(value):
        raise ValueError("macro MAE must be finite")
    return value
