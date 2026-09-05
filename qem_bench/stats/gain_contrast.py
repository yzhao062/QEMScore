"""Untouched-test relative gain and its cross-regime contrast.

The gate in :mod:`qem_bench.stats.incremental_value` compares a full method with
its controls on source validation and returns a pass or a failure. That verdict
cannot carry a headline. Promoting a change of verdict between two regimes
compares two significance categories rather than estimating the difference
between two effects, which is the distinction Gelman and Stern describe. This
module estimates the difference instead.

Within one family and one regime the relative gain is

    g = 1 - MAE_full / MAE_control

on untouched test rows, where each MAE is the equal-weight mean over the
(noise family, severity, observable) cells. The primary endpoint is the contrast

    Delta = g(contrast regime) - g(baseline regime).

Both are recomputed inside every bootstrap draw rather than derived from the
point estimates. Physical circuits are resampled with replacement inside a
regime's pool and carry all of their rows, so severities and observables of one
circuit never separate. The two predictors share each draw, which makes the
ratio paired within a regime. The two regimes draw from independent streams,
because changing the regime changes the pool descriptor and therefore the
parameter draws: they are independent samples rather than matched pairs.

What the interval does not carry: it conditions on the fitted and selected
models, so it excludes training and model-selection variation, and it is
pointwise. A positive contrast establishes an increase in signed relative gain,
which can occur between two negative gains; positive benefit at an endpoint
needs that endpoint's own interval.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import numpy as np

from qem_bench.datasets.schema import FAMILY_STRATA

CELL_FIELDS = ("noise_family", "severity", "observable")
# Draws are evaluated in blocks. The whole index matrix is small, but expanding
# it against the per-cell tables at once is not, so the work is chunked.
_DRAW_CHUNK = 512


def evaluate_gain_contrast(
    regimes: Mapping[str, Mapping[str, object]],
    *,
    full_method: str,
    control_method: str,
    baseline_regime: str,
    contrast_regime: str,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    root_seed: int,
) -> dict:
    """Estimate per-family gains in two regimes and the contrast between them.

    ``regimes`` maps a regime label to ``{"items": rows, "predictions_by_method":
    {method: {item_id: prediction}}}``. Rows are untouched test rows of one
    split-v2 dataset; pass one dataset per regime and never a merged stream.
    Invalid roles, identities, or partial prediction maps raise ValueError. A
    family that cannot support the estimate is reported as ``not_evaluable``
    rather than dropped, so a missing comparison can never read as a success.
    """

    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    for name, method in (("full_method", full_method),
                         ("control_method", control_method)):
        if not isinstance(method, str) or not method:
            raise ValueError(f"{name} must be a nonempty name")
    if full_method == control_method:
        raise ValueError("full_method and control_method must differ")
    if baseline_regime == contrast_regime:
        raise ValueError("the two regimes must differ")
    for label in (baseline_regime, contrast_regime):
        if label not in regimes:
            raise ValueError(f"regimes must contain {label!r}")
    if set(regimes) != {baseline_regime, contrast_regime}:
        raise ValueError("pass exactly the two regimes being contrasted")

    # The stream mapping is recorded rather than implicit: a reader has to be
    # able to regenerate the same draws from the root seed alone.
    ordered = (baseline_regime, contrast_regime)
    result = {
        "schema_version": "qem-bench-gain-contrast-v1",
        "status": "not_evaluable",
        "reason": None,
        "full_method": full_method,
        "control_method": control_method,
        "evaluation_role": "untouched_test",
        "baseline_regime": baseline_regime,
        "contrast_regime": contrast_regime,
        "gain_metric": "1 - full_macro_mae / control_macro_mae",
        "contrast_metric": "gain_contrast_regime_minus_gain_baseline_regime",
        "interval_scope": (
            "pointwise; conditional on fitted and selected models; "
            "paired between predictors within a regime and independent across regimes"
        ),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "root_seed": root_seed,
        "cell_fields": list(CELL_FIELDS),
        "stream_mapping": {},
        "families": {},
        "families_with_positive_contrast": [],
    }

    tables = {}
    families_per_regime = {}
    for label in ordered:
        tables[label] = _regime_tables(
            regimes[label], full_method=full_method, control_method=control_method,
            label=label,
        )
        families_per_regime[label] = set(tables[label])

    # The union rather than the intersection. A family that one regime never
    # produced is a comparison this run could not make, and reporting only the
    # families both regimes happen to hold would let a single-family result read
    # as positive in every family.
    families = sorted(families_per_regime[ordered[0]] | families_per_regime[ordered[1]])
    if not families:
        result["reason"] = "no_family_in_either_regime"
        return result

    alpha = 1.0 - confidence
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    for family_index, family in enumerate(families):
        family_result = {
            "status": "not_evaluable",
            "reason": None,
            "regimes": {},
            "contrast": None,
        }
        result["families"][family] = family_result

        draws = {}
        blocked = None
        for regime_index, label in enumerate(ordered):
            table = tables[label].get(family)
            seed = _stream_seed(root_seed, regime_index, family_index)
            if table is None:
                result["stream_mapping"][f"{label}/{family}"] = seed
                family_result["regimes"][label] = None
                blocked = blocked or "family_missing_from_a_regime"
                continue
            result["stream_mapping"][f"{label}/{family}"] = seed
            summary = {
                "n_items": table["n_items"],
                "n_circuits": table["n_circuits"],
                "n_cells": table["n_cells"],
                "full_mae": table["full_mae"],
                "control_mae": table["control_mae"],
                "absolute_reduction": table["control_mae"] - table["full_mae"],
                "gain": None,
                "gain_interval": None,
                "seed": seed,
            }
            family_result["regimes"][label] = summary
            if table["n_circuits"] < 2:
                blocked = blocked or "fewer_than_two_circuits"
                continue
            if table["control_mae"] <= 0:
                blocked = blocked or "zero_control_mae"
                continue
            # A control error small enough to overflow the ratio is positive and
            # finite, so the guard above passes it. Assigning the resulting
            # infinity would leave a nonfinite number in a result that has to
            # serialize under strict JSON.
            gain = 1.0 - table["full_mae"] / table["control_mae"]
            if not math.isfinite(gain):
                blocked = blocked or "nonfinite_point_ratio"
                continue
            summary["gain"] = gain
            draws[label] = _bootstrap_gains(table, seed=seed, n_resamples=n_resamples)
            if draws[label] is None:
                blocked = blocked or "nonfinite_resampled_ratio"
                continue
            lower, upper = (float(value) for value in np.quantile(draws[label], quantiles))
            summary["gain_interval"] = _interval(summary["gain"], lower, upper)

        if blocked is not None or len(draws) != 2:
            family_result["reason"] = blocked or "regime_missing_for_family"
            continue

        contrast_draws = draws[contrast_regime] - draws[baseline_regime]
        if not np.all(np.isfinite(contrast_draws)):
            family_result["reason"] = "nonfinite_contrast_draw"
            continue
        estimate = (family_result["regimes"][contrast_regime]["gain"]
                    - family_result["regimes"][baseline_regime]["gain"])
        lower, upper = (float(value) for value in np.quantile(contrast_draws, quantiles))
        family_result["contrast"] = _interval(estimate, lower, upper)
        family_result["status"] = "evaluated"
        if lower > 0:
            result["families_with_positive_contrast"].append(family)

    evaluated = [name for name, value in result["families"].items()
                 if value["status"] == "evaluated"]
    positive = result["families_with_positive_contrast"]
    if len(evaluated) < len(families):
        result["reason"] = "not_every_family_is_evaluable"
    elif len(positive) == len(evaluated):
        result.update(status="positive_in_every_family",
                      reason="every_evaluated_family_has_a_strictly_positive_contrast")
    else:
        result.update(status="evaluated",
                      reason="not_every_evaluated_family_has_a_strictly_positive_contrast")
    return result


def _stream_seed(root_seed: int, regime_index: int, family_index: int) -> int:
    """Derive one reproducible stream per regime and family.

    Independence across regimes is a design requirement rather than an
    optimization: their circuit pools are different samples, so pairing draws
    across them would assert a correspondence the data does not have.
    """
    entropy = np.random.SeedSequence([root_seed, regime_index, family_index])
    return int(entropy.generate_state(1, dtype=np.uint32)[0])


def _interval(estimate: float, lower: float, upper: float) -> dict:
    for value in (estimate, lower, upper):
        if not math.isfinite(value):
            raise ValueError("gain intervals must be finite")
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "excludes_zero_above": bool(lower > 0),
        "excludes_zero_below": bool(upper < 0),
    }


def _regime_tables(
    regime: Mapping[str, object], *, full_method: str, control_method: str, label: str,
) -> dict[str, dict]:
    """Build per-family cell tables of summed errors and counts per circuit.

    The bootstrap needs sums per (circuit, cell) rather than rows, because a
    resample is a multiset of circuits and every draw re-forms the same cell
    means. Building them once turns each draw into array arithmetic.
    """

    items = regime.get("items")
    predictions_by_method = regime.get("predictions_by_method")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(f"regime {label!r} requires a sequence of items")
    if not isinstance(predictions_by_method, Mapping):
        raise ValueError(f"regime {label!r} requires predictions keyed by method")
    for method in (full_method, control_method):
        if method not in predictions_by_method:
            raise ValueError(f"regime {label!r} lacks predictions for {method!r}")
        if not isinstance(predictions_by_method[method], Mapping):
            raise ValueError("predictions must be keyed by test item ID")

    rows_by_family: dict[str, list[dict]] = defaultdict(list)
    identities: set[str] = set()
    circuit_families: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.get("split") != "test":
            raise ValueError("the gain contrast requires untouched test rows only")
        if item.get("dataset_schema_version") != "split-v2":
            raise ValueError("untouched test rows require split-v2 metadata")
        item_id = _identity(item, "item_id")
        if item_id in identities:
            raise ValueError("test item IDs must be unique within a regime")
        identities.add(item_id)
        family = _identity(item, "family")
        if family not in FAMILY_STRATA or item.get("stratum") != FAMILY_STRATA[family]:
            raise ValueError("family and stratum must match the dataset contract")
        if item["stratum"] != "continuous_regression":
            continue
        circuit = _identity(item, "circuit_id")
        circuit_families[circuit].add(family)
        for field in CELL_FIELDS:
            _identity(item, field)
        target = float(item["ideal_expectation"])
        if not math.isfinite(target):
            raise ValueError("test targets must be finite")
        rows_by_family[family].append(item)
    if any(len(names) != 1 for names in circuit_families.values()):
        raise ValueError("physical circuits must not span families")

    continuous_ids = {item["item_id"] for rows in rows_by_family.values() for item in rows}
    for method in (full_method, control_method):
        predictions = predictions_by_method[method]
        if not continuous_ids <= predictions.keys() or predictions.keys() - identities:
            raise ValueError("prediction item IDs must match the regime's test rows")
        if any(not math.isfinite(float(predictions[key])) for key in continuous_ids):
            raise ValueError("predictions must be finite")

    tables = {}
    for family, rows in rows_by_family.items():
        circuits = sorted({item["circuit_id"] for item in rows})
        cells = sorted({tuple(item[field] for field in CELL_FIELDS) for item in rows})
        circuit_index = {name: index for index, name in enumerate(circuits)}
        cell_index = {key: index for index, key in enumerate(cells)}
        shape = (len(circuits), len(cells))
        errors = {method: np.zeros(shape, dtype=float)
                  for method in (full_method, control_method)}
        counts = np.zeros(shape, dtype=float)
        for item in rows:
            row = circuit_index[item["circuit_id"]]
            column = cell_index[tuple(item[field] for field in CELL_FIELDS)]
            counts[row, column] += 1.0
            for method in (full_method, control_method):
                errors[method][row, column] += _finite_absolute_error(
                    predictions_by_method[method][item["item_id"]],
                    item["ideal_expectation"],
                )
        tables[family] = {
            "n_items": len(rows),
            "n_circuits": len(circuits),
            "n_cells": len(cells),
            "counts": counts,
            "full_errors": errors[full_method],
            "control_errors": errors[control_method],
            "full_mae": _macro_mae(errors[full_method], counts),
            "control_mae": _macro_mae(errors[control_method], counts),
        }
    return tables


def _bootstrap_gains(table: Mapping[str, object], *, seed: int, n_resamples: int):
    """Resample circuits and recompute the ratio on every draw.

    Returns ``None`` when any draw cannot produce a finite gain, which happens
    when a resample gives the control an exactly zero macro error. That is a
    property of the draw rather than of the estimate, so it makes the family
    not evaluable instead of raising.
    """

    counts = table["counts"]
    full_errors = table["full_errors"]
    control_errors = table["control_errors"]
    n_circuits = counts.shape[0]
    rng = np.random.default_rng(seed)
    gains = np.empty(n_resamples, dtype=float)
    filled = 0
    while filled < n_resamples:
        block = min(_DRAW_CHUNK, n_resamples - filled)
        draws = rng.integers(0, n_circuits, size=(block, n_circuits))
        # Each draw sums the drawn circuits' per-cell totals. A circuit drawn
        # twice contributes twice, which is what sampling with replacement means
        # for a cell mean.
        drawn_counts = counts[draws].sum(axis=1)
        drawn_full = full_errors[draws].sum(axis=1)
        drawn_control = control_errors[draws].sum(axis=1)
        if not np.all(drawn_counts > 0):
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            full_macro = (drawn_full / drawn_counts).mean(axis=1)
            control_macro = (drawn_control / drawn_counts).mean(axis=1)
            block_gains = 1.0 - full_macro / control_macro
        if not np.all(np.isfinite(block_gains)):
            return None
        gains[filled:filled + block] = block_gains
        filled += block
    return gains


def _identity(item: Mapping[str, object], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"test rows require nonempty {field}")
    return value


def _finite_absolute_error(prediction: object, target: object) -> float:
    error = abs(float(prediction) - float(target))
    if not math.isfinite(error):
        raise ValueError("derived absolute errors must be finite")
    return error


def _macro_mae(errors: np.ndarray, counts: np.ndarray) -> float:
    """Equal-weight mean over cells of each cell's mean absolute error.

    Errors and counts are pooled over circuits before each cell mean is formed,
    which is what a bootstrap draw does. Dividing per circuit and then averaging
    would weight a circuit contributing one row to a cell the same as one
    contributing three, so the point estimate would sit outside the distribution
    of the draws whenever a cell's rows are unevenly distributed over circuits.
    """
    cell_errors = errors.sum(axis=0)
    cell_counts = counts.sum(axis=0)
    if not np.all(cell_counts > 0):
        raise ValueError("every cell must hold at least one row")
    value = float((cell_errors / cell_counts).mean())
    if not math.isfinite(value):
        raise ValueError("macro MAE must be finite")
    return value
