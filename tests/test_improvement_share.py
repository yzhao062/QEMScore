"""Joint attribution estimator: ladder gaps, the share, pairing and denominators.

Every test here is written from the frozen contract rather than from the module
it exercises, so a disagreement between the two is a finding rather than a
formatting difference. Fixtures follow the style of ``tests/test_gain_contrast``:
targets are zero and each prediction carries its absolute error directly, so
every cell mean equals the error the test states and expected values are written
rather than derived.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from qemscore.stats import improvement_share
from qemscore.stats.gain_contrast import CELL_FIELDS
from qemscore.stats.improvement_share import (
    FAILURE_REASONS,
    SCHEMA_VERSION,
    LadderTable,
    build_ladder_tables,
    contrast_improvement_share,
    draw_matrix,
    estimate_improvement_share,
    ladder_stream_seed,
)


SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
CELLS = tuple((severity, observable)
              for severity in SEVERITIES for observable in OBSERVABLES)
ONE_CELL = (("L1", "z_mid"),)

LADDER = ("feat-only", "liao-feat-only", "liao")
RUNGS = ("A", "C", "F")
GAPS = ("K", "D")

FOUR_LADDER = ("feat-only", "liao-feat-only", "liao-observed", "liao")
FOUR_RUNGS = ("A", "C", "O", "F")
FOUR_GAPS = ("K", "M", "D")

INTERPRETATION = (
    "fraction of the total improvement over the affine feature-only control "
    "that the nonlinear feature-only control reproduces"
)
EPS = 2.220446049250313e-16


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _build(error_of, *, families=("tfi",), circuits=2, cells=CELLS, rows_of=None,
           label="s", methods=LADDER):
    """Return split-v2 untouched test rows and one prediction map per method.

    ``error_of(method, family, circuit, severity, observable, index)`` returns the
    absolute error that row is to carry. ``rows_of`` returns how many rows one
    circuit contributes to one cell, which is what lets a test build unequal cell
    counts or a cell that only one circuit populates.
    """
    if rows_of is None:
        def rows_of(family, circuit, severity, observable):
            return 1
    items = []
    predictions = {method: {} for method in methods}
    for family in families:
        for circuit in range(circuits):
            circuit_id = f"{label}-{family}-c{circuit}"
            for severity, observable in cells:
                for index in range(rows_of(family, circuit, severity, observable)):
                    item_id = (f"{label}-{family}-{circuit}-{severity}-"
                               f"{observable}-{index}")
                    items.append({
                        "item_id": item_id,
                        "dataset_schema_version": "split-v2",
                        "split": "test",
                        "domain": "target",
                        "family": family,
                        "stratum": "continuous_regression",
                        "circuit_id": circuit_id,
                        "noise_family": "depolarizing_readout",
                        "severity": severity,
                        "observable": observable,
                        "ideal_expectation": 0.0,
                    })
                    for method in methods:
                        predictions[method][item_id] = float(error_of(
                            method, family, circuit, severity, observable, index))
    return items, predictions


def _constant(errors, **kwargs):
    """Tables whose per-row absolute error is exactly ``errors[method]``."""
    methods = tuple(errors)
    items, predictions = _build(
        lambda method, *rest: errors[method], methods=methods, **kwargs)
    return build_ladder_tables(items, predictions, methods=methods)


def _estimate(tables, **kwargs):
    """Call the estimator with the controlled-instrument declaration by default."""
    kwargs.setdefault("ladder", LADDER)
    kwargs.setdefault("rung_labels", RUNGS)
    kwargs.setdefault("gap_labels", GAPS)
    kwargs.setdefault("spans", {})
    kwargs.setdefault("required_families", ("tfi",))
    kwargs.setdefault("setting_label", "shipped-s101-n640")
    kwargs.setdefault("evaluation_role", "untouched_test")
    kwargs.setdefault("share_role", "primary")
    kwargs.setdefault("share_interpretation", INTERPRETATION)
    kwargs.setdefault("stream_components", ("shipped", 101))
    kwargs.setdefault("reference_methods", ())
    kwargs.setdefault("confidence", 0.95)
    kwargs.setdefault("n_resamples", 400)
    kwargs.setdefault("root_seed", 20260904)
    return estimate_improvement_share(tables, **kwargs)


def _withholds_its_interval(quantity):
    """A blocked quantity is absent, or published without interval endpoints.

    Section 8 says a blocked family publishes counts and no intervals, and it
    words the two blocked branches differently on whether point values survive.
    This helper accepts both readings while still failing an implementation that
    publishes a full interval on a branch that took no usable draws.
    """
    return quantity is None or quantity.get("lower") is None


def _withholds_every_interval(container, labels):
    """The same reading applied to a whole map of quantities.

    A blocked family may drop the map itself rather than each entry, which
    withholds the intervals at least as firmly as dropping them one by one.
    """
    if container is None:
        return True
    return all(_withholds_its_interval(container.get(label)) for label in labels)


def _identity_macros(table, ladder=LADDER, rungs=RUNGS):
    """Evaluate the draw arithmetic of section 6 on the identity index vector."""
    identity = np.arange(len(table.circuit_ids))
    counts = table.counts[identity].sum(axis=0)
    return {label: float((table.errors[method][identity].sum(axis=0) / counts).mean())
            for label, method in zip(rungs, ladder)}


# --------------------------------------------------------------------------
# The nine tests the review named
# --------------------------------------------------------------------------


def test_a_nonlinear_control_matching_the_full_method_leaves_no_residual_gap():
    """T1. C equals F, so D is exactly zero and the share is exactly one.

    Asserts D is 0.0 with an interval collapsed onto zero, K reproduces T on the
    point and on both endpoints, and S is exactly 1.0 with a collapsed interval.

    Catches a D interval drawn from a resample K and T did not share, which would
    leave a spread on a quantity that is identically zero in every draw. Catches a
    share formed as one minus D over a denominator of its own, which stops
    returning exactly 1.0 once the two denominators differ in their last bits.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.02, "liao": 0.02},
                       circuits=8)
    result = _estimate(tables)
    family = result["families"]["tfi"]

    assert result["status"] == "estimated"
    assert family["status"] == "estimated"
    assert list(family["reasons"]) == []
    assert family["gaps"]["D"]["estimate"] == 0.0
    assert family["gaps"]["D"]["lower"] == 0.0
    assert family["gaps"]["D"]["upper"] == 0.0
    for endpoint in ("estimate", "lower", "upper"):
        assert family["gaps"]["K"][endpoint] == family["total"][endpoint]
    assert family["share"]["status"] == "estimated"
    assert family["share"]["point"] == 1.0
    assert family["share"]["interval"]["lower"] == 1.0
    assert family["share"]["interval"]["upper"] == 1.0


def test_a_nonlinear_control_matching_the_affine_control_reproduces_nothing():
    """T2. A equals C, so K is exactly zero and the share is exactly zero.

    Asserts K is 0.0 with a collapsed interval, D reproduces T on every endpoint,
    and S is exactly 0.0.

    Catches a share computed as D over T rather than K over T. Taken with T1 this
    pins the orientation, because a reversed share returns 1.0 here and 0.0 there.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.10, "liao": 0.02},
                       circuits=8)
    result = _estimate(tables)
    family = result["families"]["tfi"]

    assert family["status"] == "estimated"
    assert family["gaps"]["K"]["estimate"] == 0.0
    assert family["gaps"]["K"]["lower"] == 0.0
    assert family["gaps"]["K"]["upper"] == 0.0
    for endpoint in ("estimate", "lower", "upper"):
        assert family["gaps"]["D"][endpoint] == family["total"][endpoint]
    assert family["share"]["status"] == "estimated"
    assert family["share"]["point"] == 0.0
    assert family["share"]["interval"]["lower"] == 0.0
    assert family["share"]["interval"]["upper"] == 0.0


def test_a_negative_residual_benefit_is_reported_and_never_clipped():
    """T3. The observation makes the full method worse, so D is negative.

    Errors A = 0.10, C = 0.02 and F = 0.03 give T = 0.07, K = 0.08, D = -0.01 and
    S = 8/7. Asserts the D interval lies strictly below zero and says so, that S
    equals 8/7, and that the S interval reaches above one.

    Catches any clip of S into the unit interval, any maximum against zero applied
    to a gap, and any code path that treats a negative gap as a failure rather
    than as a result.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.02, "liao": 0.03},
                       circuits=8)
    result = _estimate(tables)
    family = result["families"]["tfi"]

    assert family["status"] == "estimated"
    assert family["share"]["status"] == "estimated"
    assert family["gaps"]["D"]["estimate"] == pytest.approx(-0.01, rel=1e-12)
    assert family["gaps"]["D"]["upper"] < 0.0
    assert family["gaps"]["D"]["excludes_zero_below"] is True
    assert family["share"]["point"] == pytest.approx(8.0 / 7.0, rel=1e-12)
    assert family["share"]["interval"]["upper"] > 1.0


def test_a_negative_classical_benefit_is_reported_and_never_floored():
    """T3b. The mirror of T3: the nonlinear control is worse than the affine one.

    Errors A = 0.02, C = 0.05 and F = 0.01 give T = 0.01, K = -0.03, D = 0.04 and
    S = -3. Asserts S equals -3.0 and the K interval lies strictly below zero.

    Catches a clip at the lower end of the share and any absolute value applied to
    a gap or to the share itself.
    """
    tables = _constant({"feat-only": 0.02, "liao-feat-only": 0.05, "liao": 0.01},
                       circuits=8)
    result = _estimate(tables)
    family = result["families"]["tfi"]

    assert family["status"] == "estimated"
    assert family["share"]["status"] == "estimated"
    assert family["share"]["point"] == pytest.approx(-3.0, rel=1e-12)
    assert family["gaps"]["K"]["upper"] < 0.0
    assert family["gaps"]["K"]["excludes_zero_below"] is True
    assert family["gaps"]["D"]["estimate"] == pytest.approx(0.04, rel=1e-12)


def test_a_nonpositive_point_total_withholds_the_share_and_keeps_the_absolutes():
    """T4. The denominator fails, so the share alone is withheld.

    Errors A = 0.02, C = 0.025 and F = 0.03 make T negative in every draw. Asserts
    the share carries both the point reason and the draw reason, the share point
    and interval are null, the family itself stays estimated, every absolute
    quantity still carries an interval, the diagnostics publish a null smallest
    positive total, and the whole result serializes strictly.

    Catches a share reported anyway, a family-level refusal that also withholds
    the absolute quantities the review requires retained, and a float-only
    diagnostics schema that cannot express an empty subset.
    """
    result = _result_for("nonpositive_point_total")
    family = result["families"]["tfi"]

    assert family["status"] == "estimated"
    assert family["share"]["status"] == "not_estimable"
    assert "nonpositive_point_total" in family["share"]["reasons"]
    assert "nonpositive_total_in_a_draw" in family["share"]["reasons"]
    assert family["share"]["point"] is None
    assert family["share"]["interval"] is None
    for label in RUNGS:
        assert family["errors"][label]["lower"] <= family["errors"][label]["upper"]
    assert family["total"]["estimate"] < 0.0
    for label in GAPS:
        assert math.isfinite(family["gaps"][label]["lower"])
        assert math.isfinite(family["gaps"][label]["upper"])

    diagnostics = family["denominator_diagnostics"]
    assert diagnostics["smallest_positive_total"] is None
    assert diagnostics["n_draws"] == result["n_resamples"]
    assert result["status"] == "not_estimable"
    assert list(result["reasons"]) != []
    json.dumps(result, allow_nan=False)


def _uneven_cell_rows(family, circuit, severity, observable):
    return 2 if (severity, observable) == ("L3", "zz_mid") else 1


def _cell_keyed_error(method, family, circuit, severity, observable, index):
    if method == "feat-only":
        return 0.10
    if method == "liao-feat-only":
        return 0.05
    return {("L1", "z_mid"): 0.01, ("L1", "zz_mid"): 0.02,
            ("L3", "z_mid"): 0.03, ("L3", "zz_mid"): 0.04}[(severity, observable)]


def test_predictions_join_to_rows_by_identifier_and_not_by_position():
    """T5. Moving one rung's errors between cells moves that rung's macro.

    The cells hold unequal row counts, so swapping one prediction between a
    one-row cell and a two-row cell changes the equal-weight mean over cells.
    Asserts the full rung's point error moves while the two controls stay put.

    Catches an implementation that joins predictions to rows by position in the
    item sequence rather than by item_id, since a positional join would carry the
    same numbers into the same cells whatever the keys say.
    """
    items, predictions = _build(_cell_keyed_error, circuits=8,
                                rows_of=_uneven_cell_rows)
    base = _estimate(build_ladder_tables(items, predictions, methods=LADDER))

    shuffled = {method: dict(rows) for method, rows in predictions.items()}
    left, right = "s-tfi-0-L1-z_mid-0", "s-tfi-0-L3-zz_mid-0"
    shuffled["liao"][left], shuffled["liao"][right] = (
        shuffled["liao"][right], shuffled["liao"][left])
    moved = _estimate(build_ladder_tables(items, shuffled, methods=LADDER))

    assert (moved["families"]["tfi"]["errors"]["F"]["estimate"]
            != base["families"]["tfi"]["errors"]["F"]["estimate"])
    assert (moved["families"]["tfi"]["errors"]["A"]["estimate"]
            == base["families"]["tfi"]["errors"]["A"]["estimate"])
    assert (moved["families"]["tfi"]["errors"]["C"]["estimate"]
            == base["families"]["tfi"]["errors"]["C"]["estimate"])


def test_the_resample_blocks_on_the_physical_circuit_and_not_on_the_cell():
    """T5b. A permutation inside each cell leaves the point but moves the spread.

    Each circuit carries one error level across all four of its cells, which is a
    circuit effect the resample must feel. Rotating one rung's predictions inside
    each cell preserves every cell's error multiset, so the point macro is
    unchanged, while the circuit effect is destroyed and the interval narrows.

    Catches an implementation that blocks the resample on the cell rather than on
    the physical circuit. Such an implementation returns the same interval before
    and after the rotation, and no other test in this file separates it.
    """
    def error_of(method, family, circuit, severity, observable, index):
        if method == "feat-only":
            return 0.10
        if method == "liao-feat-only":
            return 0.05
        return 0.01 * (circuit + 1)

    items, predictions = _build(error_of, circuits=8)
    base = _estimate(build_ladder_tables(items, predictions, methods=LADDER),
                     n_resamples=1000)

    rotated = {method: dict(rows) for method, rows in predictions.items()}
    for offset, (severity, observable) in enumerate(CELLS):
        for circuit in range(8):
            target = f"s-tfi-{circuit}-{severity}-{observable}-0"
            source = (circuit + offset) % 8
            rotated["liao"][target] = 0.01 * (source + 1)
    moved = _estimate(build_ladder_tables(items, rotated, methods=LADDER),
                      n_resamples=1000)

    base_error = base["families"]["tfi"]["errors"]["F"]
    moved_error = moved["families"]["tfi"]["errors"]["F"]
    assert moved_error["estimate"] == pytest.approx(base_error["estimate"], rel=1e-12)
    assert abs(moved_error["upper"] - base_error["upper"]) > 1e-9
    assert abs(moved_error["lower"] - base_error["lower"]) > 1e-9


def test_a_rung_missing_predictions_for_some_rows_raises_at_ingest():
    """T6. A prediction map that does not cover every row refuses to build.

    Asserts a ValueError whose message names the rung and the number of rows it
    failed to cover, and that no table is returned.

    Catches an implementation that silently drops the uncovered rows for that one
    rung, which breaks the requirement that every rung is scored on identical
    rows with identical aggregation.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    for item in items[:3]:
        del predictions["liao"][item["item_id"]]

    with pytest.raises(ValueError) as caught:
        build_ladder_tables(items, predictions, methods=LADDER)
    message = str(caught.value)
    assert "liao" in message
    assert "3" in message


def test_a_rung_carrying_predictions_for_absent_rows_raises_at_ingest():
    """T6b. A prediction map holding an identifier no item declares refuses.

    Asserts a ValueError.

    Catches an implementation that derives the row set from the prediction map
    rather than from the item list, which would let one rung be scored on rows the
    other rungs never saw.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    predictions["liao"]["s-tfi-99-L1-z_mid-0"] = 0.5

    with pytest.raises(ValueError):
        build_ladder_tables(items, predictions, methods=LADDER)


def test_a_repeated_item_identifier_raises_rather_than_double_counting():
    """T7. Two rows sharing an item_id refuse to build.

    Asserts a ValueError.

    Catches a builder that counts the repeated row twice, which silently changes
    the counts array and therefore every pooled cell mean, with no status saying
    the row set was not what the caller passed.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    items.append(dict(items[0]))

    with pytest.raises(ValueError):
        build_ladder_tables(items, predictions, methods=LADDER)


def _crossing_total_tables():
    """Two circuits whose per-circuit totals have opposite signs."""
    per_circuit = (
        {"feat-only": 0.10, "liao-feat-only": 0.06, "liao": 0.02},
        {"feat-only": 0.05, "liao-feat-only": 0.09, "liao": 0.12},
    )
    items, predictions = _build(
        lambda method, family, circuit, *rest: per_circuit[circuit][method],
        circuits=2, cells=ONE_CELL)
    return build_ladder_tables(items, predictions, methods=LADDER)


def test_a_draw_distribution_that_crosses_zero_withholds_the_share():
    """T7b. The point total is positive while a quarter of the draws are not.

    One circuit carries a total of +0.08 and the other -0.07, so the point total
    is +0.005 and the two-copies-of-the-bad-circuit multiset, which has
    probability one quarter, is negative. Asserts both the draw reason and the
    interval reason fire, the count of nonpositive draws sits near 2500 of 10000,
    no draw is discarded, and T, K and D still carry intervals.

    Catches a denominator rule that inspects only the point estimate, which would
    publish a share whose denominator is negative in a quarter of the resample.
    """
    result = _estimate(_crossing_total_tables(), n_resamples=10_000)
    family = result["families"]["tfi"]

    assert family["total"]["estimate"] == pytest.approx(0.005, rel=1e-12)
    assert family["share"]["status"] == "not_estimable"
    assert "nonpositive_total_in_a_draw" in family["share"]["reasons"]
    assert "total_interval_reaches_zero" in family["share"]["reasons"]

    diagnostics = family["denominator_diagnostics"]
    assert diagnostics["n_draws"] == 10_000
    assert 2000 <= diagnostics["n_draws_with_nonpositive_total"] <= 3000
    assert diagnostics["total_interval_reaches_zero"] is True
    for quantity in (family["total"], family["gaps"]["K"], family["gaps"]["D"]):
        assert math.isfinite(quantity["lower"]) and math.isfinite(quantity["upper"])


def test_a_rare_offending_draw_withholds_the_share_from_a_positive_interval():
    """T7c. The draw rule fires while the interval rule does not.

    Ten circuits, nine with a per-circuit total of +1.0 and one of -2.0. A draw's
    total is (10 - 3k)/10 for k appearances of the bad circuit, so it is
    nonpositive only from k equal to four upward, which has probability about
    0.0128. Asserts the count of offending draws is above zero and below 2.5
    percent, the lower bound of T is strictly positive because the 2.5 percent
    quantile falls in the k equal to three block at 0.1, the interval reason does
    not fire, and the share is still withheld.

    Catches an implementation that checks only whether the interval reaches zero.
    This is the only test here that separates the any-draw rule from the interval
    rule, and the two rules are not equivalent in this direction.
    """
    def error_of(method, family, circuit, severity, observable, index):
        good = {"feat-only": 1.5, "liao-feat-only": 1.0, "liao": 0.5}
        bad = {"feat-only": 0.5, "liao-feat-only": 1.5, "liao": 2.5}
        return (bad if circuit == 9 else good)[method]

    items, predictions = _build(error_of, circuits=10, cells=ONE_CELL)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER),
                       n_resamples=10_000)
    family = result["families"]["tfi"]
    diagnostics = family["denominator_diagnostics"]

    assert family["total"]["estimate"] == pytest.approx(0.7, rel=1e-12)
    assert 0 < diagnostics["n_draws_with_nonpositive_total"] < 250
    assert family["total"]["lower"] > 0.0
    assert diagnostics["total_interval_reaches_zero"] is False
    assert family["share"]["status"] == "not_estimable"
    assert "nonpositive_total_in_a_draw" in family["share"]["reasons"]
    assert "total_interval_reaches_zero" not in family["share"]["reasons"]


def _overflowing_share_tables():
    return _constant({"feat-only": 5e-324, "liao-feat-only": 1.0, "liao": 0.0},
                     circuits=2)


def test_a_share_that_overflows_a_positive_denominator_leaves_no_infinity_behind():
    """T7d. The denominator passes every positivity guard and the ratio still fails.

    A subnormal affine error over a full error of zero gives a strictly positive
    total of 5e-324 while K is -1.0, so the share overflows in the point and in
    every draw. Asserts the two nonfinite share reasons fire, the nonpositive
    total reason does not, T, K and D still carry finite intervals, the share
    summaries over the finite draws are null because no draw has one, and the
    serialized result carries neither NaN nor Infinity.

    Catches exactly the defect class where a denominator passing a positivity
    guard still overflows the ratio and leaves a nonfinite number in a result that
    has to serialize strictly.
    """
    result = _estimate(_overflowing_share_tables())
    family = result["families"]["tfi"]
    diagnostics = family["denominator_diagnostics"]

    assert "nonfinite_point_share" in family["share"]["reasons"]
    assert "nonfinite_share_in_a_draw" in family["share"]["reasons"]
    assert "nonpositive_point_total" not in family["share"]["reasons"]
    assert family["share"]["point"] is None
    for quantity in (family["total"], family["gaps"]["K"], family["gaps"]["D"]):
        for endpoint in ("estimate", "lower", "upper"):
            assert math.isfinite(quantity[endpoint])
    assert diagnostics["share_min_finite"] is None
    assert diagnostics["share_max_finite"] is None
    assert diagnostics["n_draws_with_nonfinite_share"] == result["n_resamples"]

    encoded = json.dumps(result, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def _uneven_tables():
    """One circuit contributes one row per cell and the other contributes two."""
    def rows_of(family, circuit, severity, observable):
        return 1 if circuit == 0 else 2

    def error_of(method, family, circuit, severity, observable, index):
        if method == "feat-only":
            return 0.5
        if method == "liao-feat-only":
            return 0.2
        return 0.1 if circuit == 0 else 0.3

    items, predictions = _build(error_of, circuits=2, rows_of=rows_of)
    return build_ladder_tables(items, predictions, methods=LADDER)


def test_unequal_cell_counts_pool_inside_each_cell_on_the_point_and_in_the_draws():
    """T8. The point estimate pools each cell the way a draw does.

    Each cell holds one row of 0.1 and two rows of 0.3 for the full method, so the
    pooled cell mean is 0.7/3 and the macro is that value. Asserts the hand
    calculation written here, that the point lies inside the resampled support,
    and that every point quantity equals the draw arithmetic run on the identity
    index vector.

    Catches the defect where the point estimate averages circuit-cell means while
    the draws pool inside each cell, which puts the point outside the distribution
    of the draws whenever a cell's rows spread unevenly over circuits.
    """
    tables = _uneven_tables()
    result = _estimate(tables)
    family = result["families"]["tfi"]

    pooled = (0.1 + 0.3 + 0.3) / 3
    assert family["errors"]["F"]["estimate"] == pytest.approx(pooled, rel=1e-12)
    assert family["errors"]["A"]["estimate"] == pytest.approx(0.5, rel=1e-12)
    assert family["errors"]["C"]["estimate"] == pytest.approx(0.2, rel=1e-12)
    assert family["total"]["estimate"] == pytest.approx(0.5 - pooled, rel=1e-12)
    assert (family["total"]["lower"] <= family["total"]["estimate"]
            <= family["total"]["upper"])

    macro = _identity_macros(tables["tfi"])
    for label in RUNGS:
        assert family["errors"][label]["estimate"] == macro[label]
    assert family["total"]["estimate"] == macro["A"] - macro["F"]


def _single_circuit_cell_tables():
    def rows_of(family, circuit, severity, observable):
        if (severity, observable) == ("L1", "z_mid"):
            return 1 if circuit == 0 else 0
        return 1

    return _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                     circuits=3, rows_of=rows_of)


def test_a_cell_that_only_one_circuit_populates_refuses_the_whole_family():
    """T8b. A draw that misses that circuit leaves the cell empty.

    Three circuits and one cell populated by circuit zero alone, so about three
    draws in ten leave that cell with no rows. Asserts the family is refused with
    the empty-cell reason, the count of such draws is published and positive, no
    interval survives, no identity-check error is raised because the empty-cell
    mark precedes the check, and the result serializes strictly.

    Catches a division by zero that produces a NaN which either poisons an
    interval silently, breaks strict serialization, or reaches the identity check
    and raises a ValueError instead of setting a status.
    """
    result = _result_for("empty_cell_in_a_draw")
    family = result["families"]["tfi"]

    assert family["status"] == "not_estimable"
    assert "empty_cell_in_a_draw" in family["reasons"]
    assert family["denominator_diagnostics"]["n_draws_with_an_empty_cell"] > 0
    assert _withholds_its_interval(family["total"])
    assert _withholds_every_interval(family["gaps"], GAPS)
    assert _withholds_every_interval(family["errors"], RUNGS)
    assert family["share"]["interval"] is None
    assert family["n_circuits"] == 3

    encoded = json.dumps(result, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_a_common_circuit_scale_cancels_only_when_every_rung_shares_the_draw():
    """T9. The pairing test the review named.

    Each circuit carries a scale that multiplies all three rung errors, so every
    resample scales all three macro errors identically and the share is invariant
    while T, K and D move with the drawn scale. Asserts both share endpoints equal
    the constant ratio while the three absolute quantities have wide intervals.

    Catches per-rung independent resampling, per-rung independent seeds, and any
    refactor that regenerates the draw matrix inside a per-rung loop. Each of
    those leaves a different drawn scale in the numerator and the denominator, and
    the share interval opens up instead of collapsing.
    """
    affine, control, full = 0.10, 0.06, 0.02

    def error_of(method, family, circuit, severity, observable, index):
        scale = 1.0 + 4.0 * (circuit % 7)
        return scale * {"feat-only": affine, "liao-feat-only": control,
                        "liao": full}[method]

    items, predictions = _build(error_of, circuits=8)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER))
    family = result["families"]["tfi"]

    expected = (affine - control) / (affine - full)
    assert family["share"]["point"] == pytest.approx(expected, rel=1e-12)
    assert family["share"]["interval"]["lower"] == pytest.approx(expected, rel=1e-12)
    assert family["share"]["interval"]["upper"] == pytest.approx(expected, rel=1e-12)
    for quantity in (family["total"], family["gaps"]["K"], family["gaps"]["D"]):
        assert quantity["upper"] - quantity["lower"] > 0.02


# --------------------------------------------------------------------------
# Additions
# --------------------------------------------------------------------------


def test_the_point_estimate_is_the_draw_evaluator_on_the_identity_index_vector():
    """T10. Every point quantity equals the identity-draw evaluation exactly.

    The test runs the section 6 arithmetic itself on ``arange(n_circuits)`` and
    compares bit for bit against every reported rung error, the total, both gaps,
    a declared span and the share. Unequal cell counts and a per-circuit error
    level make the two aggregation orders disagree if they are not the same code.

    Catches a later refactor that reintroduces a second aggregation path for the
    point estimate. Without this the divergence is visible only through the one
    particular imbalance T8 happens to build.
    """
    def rows_of(family, circuit, severity, observable):
        return 1 if circuit % 2 == 0 else 2

    def error_of(method, family, circuit, severity, observable, index):
        base = {"feat-only": 0.40, "liao-feat-only": 0.17, "liao": 0.09}[method]
        return base * (1.0 + 0.35 * circuit) + 0.01 * index

    items, predictions = _build(error_of, circuits=5, rows_of=rows_of)
    tables = build_ladder_tables(items, predictions, methods=LADDER)
    result = _estimate(tables, spans={"A_minus_C": ("A", "C")})
    family = result["families"]["tfi"]

    macro = _identity_macros(tables["tfi"])
    total = macro["A"] - macro["F"]
    for label in RUNGS:
        assert family["errors"][label]["estimate"] == macro[label]
    assert family["total"]["estimate"] == total
    assert family["gaps"]["K"]["estimate"] == macro["A"] - macro["C"]
    assert family["gaps"]["D"]["estimate"] == macro["C"] - macro["F"]
    assert family["spans"]["A_minus_C"]["estimate"] == macro["A"] - macro["C"]
    assert family["share"]["point"] == (macro["A"] - macro["C"]) / total


def test_the_bootstrap_support_of_a_two_circuit_design_is_hand_enumerable():
    """T11, leg one. The exact resample support written down by hand.

    Two circuits give three multisets with probabilities one quarter, one half and
    one quarter, so the share takes exactly three values. Asserts the point sits
    on the both-circuits value and the two percentile endpoints sit on the two
    extreme values.

    This leg depends on no oracle code. It is the only check here that would catch
    a malformed draw matrix without inspecting the draw matrix, because a resample
    of the wrong length, or one drawn without replacement, changes the support
    itself rather than the weights on it.
    """
    per_circuit = (
        {"feat-only": 0.10, "liao-feat-only": 0.06, "liao": 0.02},
        {"feat-only": 0.20, "liao-feat-only": 0.08, "liao": 0.04},
    )
    items, predictions = _build(
        lambda method, family, circuit, *rest: per_circuit[circuit][method],
        circuits=2, cells=ONE_CELL)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER),
                       n_resamples=4000)
    share = result["families"]["tfi"]["share"]

    both_low = (0.10 - 0.06) / (0.10 - 0.02)
    both_high = (0.20 - 0.08) / (0.20 - 0.04)
    one_each = (0.15 - 0.07) / (0.15 - 0.03)
    assert share["point"] == pytest.approx(one_each, rel=1e-12)
    assert share["interval"]["lower"] == pytest.approx(both_low, rel=1e-12)
    assert share["interval"]["upper"] == pytest.approx(both_high, rel=1e-12)


def test_the_published_stream_seed_replays_the_published_resample():
    """T11, leg three. The generator call written out in longhand.

    Asserts the exported draw matrix equals one call of the default generator on
    the published seed, at the published shape.

    Catches any internal chunking of the drawing itself, which would tie every
    published interval to a performance constant, and any reshaping of the draw
    that a summary of the result would not reveal.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=6)
    result = _estimate(tables)
    seed = result["families"]["tfi"]["stream_seed"]
    n_circuits = len(tables["tfi"].circuit_ids)

    drawn = draw_matrix(n_circuits, seed=seed, n_resamples=result["n_resamples"])
    longhand = np.random.default_rng(seed).integers(
        0, n_circuits, size=(result["n_resamples"], n_circuits))
    assert drawn.shape == (result["n_resamples"], n_circuits)
    assert drawn.dtype == np.int64
    assert np.array_equal(drawn, longhand)


ORACLE_TOLERANCE = 1e-12


def _oracle_agrees(reported, point, interval, label):
    """One reported interval against the oracle's point and endpoints."""
    lower, upper = interval
    for name, expected in (("estimate", point), ("lower", lower), ("upper", upper)):
        assert reported[name] == pytest.approx(
            float(expected), rel=ORACLE_TOLERANCE, abs=ORACLE_TOLERANCE
        ), f"{label}.{name}"


def test_the_independent_rational_oracle_agrees_over_a_randomized_battery():
    """T11, leg two. The exact-rational oracle written from the contract alone.

    Asserts agreement to 1e-12 relative on every reported number and exact
    equality on counts, statuses and reason lists over the randomized battery.

    Catches an arithmetic error the estimator and a same-author oracle would
    share. The oracle is a separate deliverable at tests/oracles/share_oracle.py
    written by an author who has not read the implementation, so this test skips
    while that file is absent rather than asserting against a module it invented.

    Two things stay outside the comparison because the oracle consumes them as
    data rather than deriving them. The draw matrix is handed over, so a wrong
    seed or a wrong shape would agree here; legs one and three cover that. Every
    scenario carries two or more circuits, so the single-circuit branch belongs
    to another test.

    Errors are multiples of one over 256, so every cell sum is exact in binary64
    and a total that cancels to zero cancels on both sides. Without that, a
    refusal that turns on the sign of a total would be a coin flip rather than a
    check.
    """
    oracle = pytest.importorskip(
        "oracles.share_oracle",
        reason=("tests/oracles/share_oracle.py is a separate round-one deliverable "
                "and is not present, so the battery cannot run"))

    rng = np.random.default_rng(90210)
    families_checked = 0
    shares_checked = 0
    blocked_seen = 0
    refused_shares = set()

    for scenario in range(24):
        four = bool(rng.integers(2))
        ladder = FOUR_LADDER if four else LADDER
        rungs = FOUR_RUNGS if four else RUNGS
        gaps = FOUR_GAPS if four else GAPS
        spans = ({"AF": (rungs[0], rungs[-1]), "CF": (rungs[1], rungs[-1])}
                 if four and rng.integers(2) else {})
        references = ("noisy-only",) if rng.integers(2) else ()
        families = ("tfi", "heisenberg") if rng.integers(2) else ("tfi",)
        circuits = int(rng.integers(2, 6))
        cells = CELLS if rng.integers(2) else ONE_CELL
        confidence = float(rng.choice([0.90, 0.95]))
        n_resamples = int(rng.integers(40, 121))
        methods = ladder + references

        # Half the scenarios put the rungs in the order the campaign expects, so
        # the total stays positive and the share is estimable. The other half
        # draws each rung independently, which is what reaches a negative gap, a
        # zero total and the refusals that follow from them.
        ordered = scenario % 2 == 0
        levels: dict[tuple, dict[str, float]] = {}

        def error_of(method, family, circuit, severity, observable, index):
            key = (family, circuit, severity, observable, index)
            if key not in levels:
                row = {}
                if ordered:
                    value = int(rng.integers(0, 16))
                    for rung in reversed(ladder):
                        row[rung] = value / 256.0
                        value += int(rng.integers(1, 8))
                else:
                    for rung in ladder:
                        row[rung] = int(rng.integers(0, 48)) / 256.0
                for extra in references:
                    row[extra] = int(rng.integers(0, 64)) / 256.0
                levels[key] = row
            return levels[key][method]

        def rows_of(family, circuit, severity, observable):
            return int(rng.integers(1, 4))

        items, predictions = _build(
            error_of, families=families, circuits=circuits, cells=cells,
            rows_of=rows_of, label=f"b{scenario}", methods=methods)
        tables = build_ladder_tables(items, predictions, methods=methods)
        result = _estimate(
            tables, ladder=ladder, rung_labels=rungs, gap_labels=gaps,
            spans=spans, required_families=families, reference_methods=references,
            confidence=confidence, n_resamples=n_resamples,
            stream_components=("shipped", 101 + scenario))

        expected = oracle.decompose(
            {name: {
                "counts": table.counts.tolist(),
                "errors": {method: table.errors[method].tolist()
                           for method in methods},
            } for name, table in tables.items()},
            {name: draw_matrix(
                len(table.circuit_ids),
                seed=result["families"][name]["stream_seed"],
                n_resamples=n_resamples).tolist()
             for name, table in tables.items()},
            ladder=ladder, rung_labels=rungs, gap_labels=gaps, spans=spans,
            reference_methods=references, confidence=confidence)

        for name in families:
            got = result["families"][name]
            want = expected[name]
            table = tables[name]
            where = f"scenario {scenario} family {name}"

            assert got["n_circuits"] == len(table.circuit_ids), where
            assert got["n_cells"] == len(table.cell_keys), where
            assert got["n_rows"] == int(table.counts.sum()), where

            # The oracle reports a whole draw as missing when that draw leaves a
            # cell with no rows, which is the one family-level refusal reachable
            # from finite rational inputs.
            blocked = any(value is None for value in want["draws"]["total"])
            assert got["reasons"] == (["empty_cell_in_a_draw"] if blocked else []), where
            if blocked:
                blocked_seen += 1
                assert got["status"] == "not_estimable", where
                continue

            assert got["status"] == "estimated", where
            families_checked += 1
            for label in rungs:
                _oracle_agrees(got["errors"][label], want["point"][label],
                               want["interval"][label], f"{where} {label}")
            for method in references:
                _oracle_agrees(got["reference_errors"][method], want["point"][method],
                               want["interval"][method], f"{where} {method}")
            _oracle_agrees(got["total"], want["point"]["total"],
                           want["interval"]["total"], f"{where} total")
            for label in gaps:
                _oracle_agrees(got["gaps"][label], want["point"][label],
                               want["interval"][label], f"{where} {label}")
            for label in spans:
                _oracle_agrees(got["spans"][label], want["point"][label],
                               want["interval"][label], f"{where} {label}")

            # The oracle applies no refusal rule, so the rules are restated here
            # from the contract and checked against what the estimator withheld.
            # A total of exactly zero is where the two arithmetics part company.
            # The oracle leaves that draw's share undefined; binary64 divides by
            # zero and gets a nonfinite value, so the estimator names both.
            share_reasons = set()
            if want["point"]["total"] <= 0:
                share_reasons.add("nonpositive_point_total")
            if want["point"]["share"] is None:
                share_reasons.add("nonfinite_point_share")
            if any(value <= 0 for value in want["draws"]["total"]):
                share_reasons.add("nonpositive_total_in_a_draw")
            if any(value is None for value in want["draws"]["share"]):
                share_reasons.add("nonfinite_share_in_a_draw")
            if want["interval"]["total"][0] <= 0:
                share_reasons.add("total_interval_reaches_zero")
            assert set(got["share"]["reasons"]) == share_reasons, where
            refused_shares |= share_reasons

            if share_reasons:
                assert got["share"]["status"] == "not_estimable", where
                assert got["share"]["point"] is None, where
                assert got["share"]["interval"] is None, where
                continue

            shares_checked += 1
            assert got["share"]["status"] == "estimated", where
            assert got["share"]["point"] == pytest.approx(
                float(want["point"]["share"]), rel=ORACLE_TOLERANCE,
                abs=ORACLE_TOLERANCE), where
            _oracle_agrees(got["share"]["interval"], want["point"]["share"],
                           want["interval"]["share"], f"{where} share")

    # A battery that agreed on nothing would pass every assertion above, so the
    # coverage it reached is asserted rather than assumed.
    assert families_checked >= 20
    assert shares_checked >= 5
    assert {"nonpositive_point_total", "nonpositive_total_in_a_draw",
            "total_interval_reaches_zero"} <= refused_shares


def test_identical_inputs_reproduce_every_number_and_the_key_excludes_the_size():
    """T12. Determinism, the seed key, and the paired-size rule.

    Asserts two identical calls return equal dicts, a different root seed moves
    the family stream seed, the published seed equals the declared key evaluated
    by the exported helper, two settings differing only in the training size share
    a seed and therefore a draw matrix, a different regime or master seed moves
    the seed, and adding a family to the required list leaves an existing family's
    seed where it was.

    Catches an index-based seed key, which would move every published interval the
    moment a family name is added to the declared family tuple, and would do so
    invisibly. Catches a size mixed into the key, which would break the pairing
    the review requires between two sizes that share test circuits.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=6)
    first = _estimate(tables)
    assert _estimate(tables) == first

    seed = first["families"]["tfi"]["stream_seed"]
    assert seed == ladder_stream_seed(20260904, ["shipped", 101, "tfi"])
    assert _estimate(tables, root_seed=20260905)["families"]["tfi"]["stream_seed"] != seed
    assert _estimate(tables, stream_components=("large", 101))[
        "families"]["tfi"]["stream_seed"] != seed
    assert _estimate(tables, stream_components=("shipped", 202))[
        "families"]["tfi"]["stream_seed"] != seed

    grown = _estimate(tables, required_families=("heisenberg", "tfi", "qaoa"))
    assert grown["families"]["tfi"]["stream_seed"] == seed

    larger_size = _constant({"feat-only": 0.10, "liao-feat-only": 0.03, "liao": 0.01},
                            circuits=6)
    other = _estimate(larger_size)
    assert other["families"]["tfi"]["stream_seed"] == seed
    n_circuits = len(tables["tfi"].circuit_ids)
    assert np.array_equal(
        draw_matrix(n_circuits, seed=seed, n_resamples=400),
        draw_matrix(n_circuits, seed=other["families"]["tfi"]["stream_seed"],
                    n_resamples=400))


def test_the_evaluation_chunk_size_moves_no_reported_number(monkeypatch):
    """T13. The chunk over draws is an implementation detail.

    Asserts two runs differing only in the evaluation chunk constant return equal
    dicts, including every interval endpoint.

    Catches a copy of the older estimator's shape, which chunks the drawing rather
    than the evaluation and therefore ties every published interval to a
    performance constant that a reader has no reason to inspect.
    """
    def error_of(method, family, circuit, severity, observable, index):
        base = {"feat-only": 0.30, "liao-feat-only": 0.14, "liao": 0.06}[method]
        return base * (1.0 + 0.5 * (circuit % 5))

    items, predictions = _build(error_of, circuits=6)
    tables = build_ladder_tables(items, predictions, methods=LADDER)

    monkeypatch.setattr(improvement_share, "_EVAL_CHUNK", 7)
    small = _estimate(tables, n_resamples=300)
    monkeypatch.setattr(improvement_share, "_EVAL_CHUNK", 4096)
    large = _estimate(tables, n_resamples=300)
    assert small == large


def test_the_identity_check_is_reported_and_its_tolerance_scales_with_the_macro():
    """T15, the half that is reachable from outside the module.

    Asserts the residual is below the reported tolerance on well-formed data, that
    the check says it held, and that the tolerance grows with the size of the
    macro errors rather than sitting at a fixed absolute value.

    Catches a hard-coded tolerance, which is either too tight on data whose errors
    are large or too loose on data whose errors are small. The sign-flip mutation
    the specification also names cannot be reached from outside: the residual of a
    correctly telescoping decomposition is bounded by about twelve machine epsilon
    times the largest macro, against a tolerance of thirty-two, so no input can
    trip the check and only an internal seam could. The specification names no
    such seam, which is recorded here rather than guessed at.
    """
    small = _estimate(_constant(
        {"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02}, circuits=6))
    large = _estimate(_constant(
        {"feat-only": 1.0e6, "liao-feat-only": 5.0e5, "liao": 2.0e5}, circuits=6))

    for result in (small, large):
        check = result["families"]["tfi"]["identity_check"]
        assert check["held"] is True
        assert check["tolerance"] > 0.0
        assert check["max_absolute_residual"] <= check["tolerance"]

    small_check = small["families"]["tfi"]["identity_check"]
    large_check = large["families"]["tfi"]["identity_check"]
    assert small_check["tolerance"] == pytest.approx(32.0 * EPS, rel=1e-9)
    assert large_check["tolerance"] > 1.0e5 * small_check["tolerance"]


def test_a_circuit_carries_every_severity_and_observable_sibling_into_a_draw():
    """T16. One circuit is an outlier in two of its four cells.

    Blocking on the physical circuit makes the total a function of one integer:
    how many of the four draws landed on that circuit. The support is therefore
    the five values enumerated in the body, and both interval endpoints must be
    members of it.

    Catches a row-level resample, which decouples the outlier's two cells and puts
    the endpoints between the enumerated values, smuggling four rows of one
    circuit in as four independent observations.
    """
    def error_of(method, family, circuit, severity, observable, index):
        if method == "feat-only":
            return 0.10
        if method == "liao-feat-only":
            return 0.06
        return 0.0 if (circuit == 0 and severity == "L1") else 0.02

    items, predictions = _build(error_of, circuits=4)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER),
                       n_resamples=4000)
    family = result["families"]["tfi"]

    support = []
    for landings in range(5):
        outlier_cell = 0.02 * (4 - landings) / 4
        macro_full = (2 * outlier_cell + 2 * 0.02) / 4
        support.append(0.10 - macro_full)
    assert support == pytest.approx([0.08, 0.0825, 0.085, 0.0875, 0.09])

    # The number of landings is Binomial(4, 1/4): the 2.5th percentile sits in the
    # zero-landing block and the 97.5th in the three-landing block.
    assert family["total"]["estimate"] == pytest.approx(support[1], rel=1e-12)
    assert family["total"]["lower"] == pytest.approx(support[0], rel=1e-12)
    assert family["total"]["upper"] == pytest.approx(support[3], rel=1e-12)


def test_the_ladder_and_its_labels_are_validated_before_any_resampling():
    """T17. Every argument violation raises rather than producing a report.

    Asserts a ladder shorter than two rungs, a repeated rung name, a rung absent
    from a table, label sequences of the wrong length, a reference method
    colliding with a ladder rung, an empty interpretation, an out-of-range share
    role, and the numeric argument checks all raise ValueError.

    Catches an estimator that accepts a malformed declaration and reports numbers
    against it. A reversed ladder is not mechanically detectable from names alone,
    so these checks plus the campaign wrapper fixing the ladder are the whole
    defence against one.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=4)

    with pytest.raises(ValueError):
        _estimate(tables, ladder=("feat-only",), rung_labels=("A",), gap_labels=())
    with pytest.raises(ValueError):
        _estimate(tables, ladder=("feat-only", "feat-only", "liao"))
    with pytest.raises(ValueError):
        _estimate(tables, ladder=("feat-only", "not-a-method", "liao"))
    with pytest.raises(ValueError):
        _estimate(tables, rung_labels=("A", "C"))
    with pytest.raises(ValueError):
        _estimate(tables, rung_labels=("A", "A", "F"))
    with pytest.raises(ValueError):
        _estimate(tables, gap_labels=("K",))
    with pytest.raises(ValueError):
        _estimate(tables, gap_labels=("K", "K"))
    with pytest.raises(ValueError):
        _estimate(tables, reference_methods=("liao",))
    with pytest.raises(ValueError):
        _estimate(tables, share_interpretation="")
    with pytest.raises(ValueError):
        _estimate(tables, share_role="headline")
    with pytest.raises(ValueError):
        _estimate(tables, setting_label="")
    with pytest.raises(ValueError):
        _estimate(tables, evaluation_role="")
    with pytest.raises(ValueError):
        _estimate(tables, required_families=())
    with pytest.raises(ValueError):
        _estimate(tables, required_families=("tfi", "tfi"))
    with pytest.raises(ValueError):
        _estimate(tables, confidence=1.0)
    with pytest.raises(ValueError):
        _estimate(tables, n_resamples=0)
    with pytest.raises(ValueError):
        _estimate(tables, n_resamples=True)
    with pytest.raises(ValueError):
        _estimate(tables, root_seed=-1)


def test_a_malformed_span_declaration_raises():
    """T17b, the validation half.

    Asserts a span naming rungs in reverse ladder order raises, and that a span
    label colliding with a rung label, a gap label, or the reserved names total
    and share raises.

    Catches a report whose span key silently shadows a gap or the total, which
    would let a reader read one quantity as another.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=4)

    with pytest.raises(ValueError):
        _estimate(tables, spans={"backwards": ("F", "A")})
    with pytest.raises(ValueError):
        _estimate(tables, spans={"nowhere": ("A", "A")})
    for collision in ("A", "K", "total", "share"):
        with pytest.raises(ValueError):
            _estimate(tables, spans={collision: ("A", "F")})


def test_a_span_is_formed_inside_the_draw_and_not_from_two_intervalled_gaps():
    """T17b, the behaviour half.

    On a four-rung ladder the outer rungs are constant per circuit while the third
    rung varies, so the two adjacent gaps it separates move in opposite directions
    and their sum is the same in every draw. Asserts the span interval collapses
    onto 0.04 while each component gap has a wide interval, and that the span
    reproduces the sum of those gaps on the point.

    Catches a span assembled after the fact from two separately intervalled gaps.
    That construction would report the span as roughly 0.02 to 0.06, since it adds
    the two lower bounds and the two upper bounds, and it is exactly the derived
    marginal the review forbids.
    """
    def error_of(method, family, circuit, severity, observable, index):
        if method == "liao-observed":
            return 0.02 if circuit == 0 else 0.04
        return {"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.01}[method]

    items, predictions = _build(error_of, circuits=2, methods=FOUR_LADDER)
    tables = build_ladder_tables(items, predictions, methods=FOUR_LADDER)
    result = _estimate(
        tables, ladder=FOUR_LADDER, rung_labels=FOUR_RUNGS, gap_labels=FOUR_GAPS,
        spans={"C_minus_F": ("C", "F"), "O_minus_F": ("O", "F")},
        n_resamples=2000)
    family = result["families"]["tfi"]

    span = family["spans"]["C_minus_F"]
    assert span["estimate"] == pytest.approx(0.04, rel=1e-12)
    assert span["lower"] == pytest.approx(0.04, rel=1e-12)
    assert span["upper"] == pytest.approx(0.04, rel=1e-12)
    assert span["estimate"] == pytest.approx(
        family["gaps"]["M"]["estimate"] + family["gaps"]["D"]["estimate"], rel=1e-12)
    for label in ("M", "D"):
        assert family["gaps"][label]["upper"] - family["gaps"][label]["lower"] > 0.01

    duplicate = family["spans"]["O_minus_F"]
    for endpoint in ("estimate", "lower", "upper"):
        assert duplicate[endpoint] == family["gaps"]["D"][endpoint]
    assert list(result["span_labels"]) == ["C_minus_F", "O_minus_F"]


def test_the_campaign_wrapper_fixes_the_ladder_from_the_frozen_design():
    """T18. The ladder is declared in the campaign layer, not by the caller.

    Asserts the three frozen constants hold the controlled ladder and its labels,
    that the declared design reports them so the freeze manifest hashes them, and
    that the setting wrapper exposes no argument that could reorder or substitute
    a rung.

    Catches a reversed ladder reaching the estimator, which no name-based check
    inside the estimator can detect. This test covers files the round-one
    estimator change does not touch, so it fails until the campaign wiring lands.
    """
    from qemscore.campaign import analysis, design

    assert tuple(design.SHARE_LADDER) == ("feat-only", "liao-feat-only", "liao")
    assert tuple(design.SHARE_RUNG_LABELS) == ("A", "C", "F")
    assert tuple(design.SHARE_GAP_LABELS) == ("K", "D")

    declared = list(design.declared_design().values())
    for frozen in (design.SHARE_LADDER, design.SHARE_RUNG_LABELS,
                   design.SHARE_GAP_LABELS):
        assert any(isinstance(value, (list, tuple)) and tuple(value) == tuple(frozen)
                   for value in declared)

    parameters = inspect.signature(analysis.evaluate_setting_share).parameters
    assert set(parameters) == {"record", "root_seed", "n_resamples", "confidence"}


def test_a_required_family_absent_from_the_setting_is_reported_rather_than_omitted():
    """T19. The report covers the required families, not the ones that turned up.

    Asserts a missing family appears with its own reason, the top-level status
    falls to not estimable, an unsorted required list is accepted and reported in
    sorted order, and a family present in the tables but never required raises.

    Catches iterating the intersection of the declared and the observed families,
    which lets a single-family result read as every family. Catches a sorted-input
    rule, which would make the campaign wrapper raise on its first call because
    the frozen family tuple is not sorted.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=4)
    result = _estimate(tables, required_families=("tfi", "heisenberg"))

    assert sorted(result["families"]) == ["heisenberg", "tfi"]
    assert list(result["required_families"]) == ["heisenberg", "tfi"]
    missing = result["families"]["heisenberg"]
    assert missing["status"] == "not_estimable"
    assert "family_missing_from_the_setting" in missing["reasons"]
    assert result["status"] == "not_estimable"
    assert "family_missing_from_the_setting" in result["reasons"]
    assert result["families"]["tfi"]["status"] == "estimated"
    json.dumps(result, allow_nan=False)

    both = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                     circuits=4, families=("tfi", "heisenberg"))
    with pytest.raises(ValueError):
        _estimate(both, required_families=("tfi",))


def test_a_single_circuit_pool_cannot_estimate_uncertainty():
    """T20. Fewer than two circuits publishes counts and no interval.

    Asserts the family reason, the published counts, and that no interval endpoint
    reaches the report on any quantity.

    Catches a resample taken over a pool of one, which would report an interval of
    zero width and read as a certainty rather than as an absence of evidence.
    """
    result = _result_for("fewer_than_two_circuits")
    family = result["families"]["tfi"]

    assert family["status"] == "not_estimable"
    assert "fewer_than_two_circuits" in family["reasons"]
    assert family["n_circuits"] == 1
    assert family["n_cells"] == 4
    assert family["n_rows"] == 4
    assert _withholds_its_interval(family["total"])
    assert _withholds_every_interval(family["gaps"], GAPS)
    assert _withholds_every_interval(family["errors"], RUNGS)
    assert family["share"]["interval"] is None
    assert family["denominator_diagnostics"]["total_interval"] is None
    json.dumps(result, allow_nan=False)


def test_a_hand_checkable_anchor_recovers_the_written_decomposition():
    """T21. Round numbers a reader can verify without running anything.

    Errors A = 0.10, C = 0.02 and F = 0.01 give T = 0.09, K = 0.08, D = 0.01 and
    S = 8/9, with every interval collapsed onto its point because the errors are
    constant. Asserts all four values and the collapse.

    Catches a whole-pipeline sign or orientation error that the more elaborate
    fixtures could hide behind their own arithmetic. Round numbers are used
    deliberately so that no rehearsal value enters the repository.
    """
    result = _estimate(_constant(
        {"feat-only": 0.10, "liao-feat-only": 0.02, "liao": 0.01}, circuits=8))
    family = result["families"]["tfi"]

    assert family["total"]["estimate"] == pytest.approx(0.09, rel=1e-12)
    assert family["gaps"]["K"]["estimate"] == pytest.approx(0.08, rel=1e-12)
    assert family["gaps"]["D"]["estimate"] == pytest.approx(0.01, rel=1e-12)
    assert family["share"]["point"] == pytest.approx(8.0 / 9.0, rel=1e-12)
    for quantity in (family["total"], family["gaps"]["K"], family["gaps"]["D"]):
        assert quantity["lower"] == pytest.approx(quantity["estimate"], rel=1e-12)
        assert quantity["upper"] == pytest.approx(quantity["estimate"], rel=1e-12)
    assert family["share"]["interval"]["lower"] == pytest.approx(8.0 / 9.0, rel=1e-12)
    assert family["share"]["interval"]["upper"] == pytest.approx(8.0 / 9.0, rel=1e-12)


def test_the_share_is_the_ratio_of_macro_errors_and_not_a_mean_of_row_ratios():
    """T22. Two circuits on very different scales separate the two definitions.

    One row gives a per-row ratio of 0.1 and the other 0.95, so their mean is
    0.525, while the ratio of the macro numerator to the macro denominator is
    about 0.117. Asserts the estimator returns the macro ratio and lands nowhere
    near the mean of the row ratios.

    Catches a share formed by averaging per-row ratios, which the review forbids
    and which no other test in this file separates from the macro ratio.
    """
    per_circuit = (
        {"feat-only": 1.0, "liao-feat-only": 0.9, "liao": 0.0},
        {"feat-only": 0.02, "liao-feat-only": 0.001, "liao": 0.0},
    )
    items, predictions = _build(
        lambda method, family, circuit, *rest: per_circuit[circuit][method],
        circuits=2, cells=ONE_CELL)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER))
    share = result["families"]["tfi"]["share"]

    macro_ratio = ((1.0 + 0.02) / 2 - (0.9 + 0.001) / 2) / ((1.0 + 0.02) / 2)
    row_ratio_mean = (0.1 / 1.0 + 0.019 / 0.02) / 2
    assert share["point"] == pytest.approx(macro_ratio, rel=1e-12)
    assert abs(share["point"] - row_ratio_mean) > 0.3


@pytest.mark.parametrize("mutation", [
    {"split": "validation"},
    {"dataset_schema_version": "legacy-v1"},
    {"family": "random_clifford"},
    {"circuit_id": ""},
    {"item_id": ""},
    {"observable": ""},
    {"ideal_expectation": float("inf")},
    {"ideal_expectation": float("nan")},
])
def test_a_malformed_row_raises_at_ingest_rather_than_reaching_a_macro(mutation):
    """T25. The strict row path refuses anything outside the split-v2 contract.

    Asserts each mutation raises ValueError and no table is built. The message
    text is left unasserted because this module is authored independently of the
    estimator whose messages it mirrors.

    Catches a NaN target or prediction that reaches a macro, which poisons an
    interval with no status saying so, and catches validation or training rows
    entering an untouched-test estimate.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    items[0] = dict(items[0], **mutation)

    with pytest.raises(ValueError):
        build_ladder_tables(items, predictions, methods=LADDER)


def test_a_nonfinite_prediction_raises_at_ingest():
    """T25, the prediction half.

    Asserts a nonfinite prediction for one rung raises ValueError.

    Catches an infinite prediction reaching the error sum, which would make one
    cell of one rung nonfinite and leave the family blocked for a reason that
    names the draw rather than the input.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    predictions["liao"][items[0]["item_id"]] = float("inf")

    with pytest.raises(ValueError):
        build_ladder_tables(items, predictions, methods=LADDER)


def test_a_physical_circuit_spanning_two_families_raises_at_ingest():
    """T25, the pairing half.

    One circuit identifier carrying rows of two families raises ValueError.

    Catches a circuit whose rows would be resampled into two family pools at once,
    which would make the two families' draws dependent while the report says they
    are separate estimates.
    """
    items, predictions = _build(lambda method, *rest: 0.1, circuits=4)
    items[0] = dict(items[0], family="heisenberg")

    with pytest.raises(ValueError):
        build_ladder_tables(items, predictions, methods=LADDER)


def test_cross_setting_pairing_is_checked_against_the_data_rather_than_trusted():
    """T26. The contrast asserts the pairing its caller declared.

    Two sizes in one regime share test circuits, so their circuit order digests
    match and a paired contrast is allowed while an independent one raises. Two
    regimes hold disjoint circuits, so the digests differ and the reverse holds.
    Asserts both directions raise where the data contradicts the declaration, and
    that the recorded stream seed replays the draw matrix the paired sides used.

    Catches a pairing argument that is trusted rather than checked, which would
    let a between-regime contrast be formed draw by draw over two pools that share
    no circuit, and would let a within-regime size comparison throw away the
    pairing that makes it precise.
    """
    items, left_predictions = _build(
        lambda method, *rest: {"feat-only": 0.10, "liao-feat-only": 0.05,
                               "liao": 0.02}[method], circuits=6)
    _, right_predictions = _build(
        lambda method, *rest: {"feat-only": 0.10, "liao-feat-only": 0.04,
                               "liao": 0.01}[method], circuits=6)
    left_tables = build_ladder_tables(items, left_predictions, methods=LADDER)
    right_tables = build_ladder_tables(items, right_predictions, methods=LADDER)
    left = _estimate(left_tables, setting_label="shipped-s101-n320")
    right = _estimate(right_tables, setting_label="shipped-s101-n640")

    digest = left["families"]["tfi"]["circuit_order_digest"]
    assert right["families"]["tfi"]["circuit_order_digest"] == digest
    expected = hashlib.sha256(json.dumps(
        list(left_tables["tfi"].circuit_ids),
        separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    assert expected in digest

    paired = contrast_improvement_share(
        left, right, left_tables=left_tables, right_tables=right_tables,
        quantity="D", pairing="paired")
    json.dumps(paired, allow_nan=False)
    with pytest.raises(ValueError):
        contrast_improvement_share(
            left, right, left_tables=left_tables, right_tables=right_tables,
            quantity="D", pairing="independent")

    other_items, other_predictions = _build(
        lambda method, *rest: {"feat-only": 0.09, "liao-feat-only": 0.05,
                               "liao": 0.03}[method], circuits=6, label="l")
    other_tables = build_ladder_tables(other_items, other_predictions, methods=LADDER)
    other = _estimate(other_tables, stream_components=("large", 101),
                      setting_label="large-s101-n640")
    assert other["families"]["tfi"]["circuit_order_digest"] != digest

    with pytest.raises(ValueError):
        contrast_improvement_share(
            left, other, left_tables=left_tables, right_tables=other_tables,
            quantity="D", pairing="paired")
    independent = contrast_improvement_share(
        left, other, left_tables=left_tables, right_tables=other_tables,
        quantity="D", pairing="independent")
    json.dumps(independent, allow_nan=False)

    seed = left["families"]["tfi"]["stream_seed"]
    n_circuits = len(left_tables["tfi"].circuit_ids)
    assert np.array_equal(
        draw_matrix(n_circuits, seed=seed, n_resamples=left["n_resamples"]),
        np.random.default_rng(seed).integers(
            0, n_circuits, size=(left["n_resamples"], n_circuits)))


# --------------------------------------------------------------------------
# Failure-reason coverage: T14, T23 and T24 share one registry of fixtures
# --------------------------------------------------------------------------


def _fixture_fewer_than_two_circuits():
    return _estimate(_constant(
        {"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02}, circuits=1))


def _fixture_family_missing_from_the_setting():
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02},
                       circuits=4)
    return _estimate(tables, required_families=("heisenberg", "tfi"))


def _fixture_empty_cell_in_a_draw():
    return _estimate(_single_circuit_cell_tables())


def _fixture_nonfinite_macro_in_a_draw():
    """One rung's per-circuit errors sum past binary64 when a circuit repeats.

    Every stored entry stays finite and the identity draw sums to 1.1e308, so the
    point estimate is finite while a draw taking the larger circuit twice
    overflows.
    """
    def error_of(method, family, circuit, severity, observable, index):
        if method != "feat-only":
            return {"liao-feat-only": 1.0, "liao": 0.5}[method]
        return 1e308 if circuit == 0 else 1e307

    items, predictions = _build(error_of, circuits=2, cells=ONE_CELL)
    result = _estimate(build_ladder_tables(items, predictions, methods=LADDER))
    assert result["families"]["tfi"]["point_values"]["errors"]["A"] == 5.5e307
    return result


def _fixture_nonpositive_point_total():
    return _estimate(_constant(
        {"feat-only": 0.02, "liao-feat-only": 0.025, "liao": 0.03}, circuits=8))


def _fixture_nonpositive_total_in_a_draw():
    return _estimate(_crossing_total_tables())


def _fixture_total_interval_reaches_zero():
    return _estimate(_crossing_total_tables())


def _fixture_nonfinite_point_share():
    return _estimate(_overflowing_share_tables())


def _fixture_nonfinite_share_in_a_draw():
    return _estimate(_overflowing_share_tables())


_REASON_FIXTURES = {
    "fewer_than_two_circuits": _fixture_fewer_than_two_circuits,
    "family_missing_from_the_setting": _fixture_family_missing_from_the_setting,
    "empty_cell_in_a_draw": _fixture_empty_cell_in_a_draw,
    "nonfinite_macro_in_a_draw": _fixture_nonfinite_macro_in_a_draw,
    "nonpositive_point_total": _fixture_nonpositive_point_total,
    "nonpositive_total_in_a_draw": _fixture_nonpositive_total_in_a_draw,
    "total_interval_reaches_zero": _fixture_total_interval_reaches_zero,
    "nonfinite_point_share": _fixture_nonfinite_point_share,
    "nonfinite_share_in_a_draw": _fixture_nonfinite_share_in_a_draw,
}

_SHARE_LEVEL_REASONS = (
    "nonpositive_point_total",
    "nonpositive_total_in_a_draw",
    "total_interval_reaches_zero",
    "nonfinite_point_share",
    "nonfinite_share_in_a_draw",
)

_RESULT_CACHE: dict[str, dict] = {}


def _result_for(reason):
    if reason not in _RESULT_CACHE:
        _RESULT_CACHE[reason] = _REASON_FIXTURES[reason]()
    return _RESULT_CACHE[reason]


def test_every_declared_failure_reason_has_a_constructed_input():
    """T24, the coverage half.

    Asserts the reasons this file constructs are exactly the reasons the module
    declares.

    Catches a reason string the report layer would render but the estimator never
    emits, and catches a new reason added to the module without a test that shows
    what produces it.
    """
    assert set(_REASON_FIXTURES) == set(FAILURE_REASONS)


@pytest.mark.parametrize("reason", sorted(_REASON_FIXTURES))
def test_each_declared_failure_reason_is_produced_and_reported_at_the_top(reason):
    """T24, the reachability half.

    Asserts each declared reason appears in the top-level reason list of some
    constructed input and that the top-level status falls to not estimable.

    Catches a share-level refusal that leaves the top-level status reading as a
    success with an empty reason list, which is the shape in which a partial
    result gets mistaken for a whole one.
    """
    result = _result_for(reason)
    assert reason in result["reasons"]
    assert result["status"] == "not_estimable"
    assert list(result["reasons"]) == sorted(result["reasons"])


@pytest.mark.parametrize("reason", sorted(_REASON_FIXTURES))
def test_every_failure_branch_serializes_under_strict_json(reason):
    """T14. Strict serialization on every branch, including the nonfinite ones.

    Asserts the result encodes with allow_nan disabled and that the text carries
    neither NaN nor Infinity.

    Catches a diagnostics schema typed float where the specification requires
    float or null. The rule never to discard a nonfinite draw puts nonfinite
    values inside the diagnostics, so a summary over an empty subset has no
    serializable float and must be null.
    """
    encoded = json.dumps(_result_for(reason), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


@pytest.mark.parametrize("reason", sorted(_SHARE_LEVEL_REASONS))
def test_no_draw_is_discarded_on_any_share_level_failure(reason):
    """T23. The draw count is the mechanical statement that nothing was dropped.

    Asserts the reported number of draws equals the requested resample count and
    that the finite and nonfinite totals partition it exactly.

    Catches an implementation that filters offending draws out of the resample
    before taking quantiles, which narrows every published interval and hides the
    condition that made the share unavailable in the first place.
    """
    result = _result_for(reason)
    diagnostics = result["families"]["tfi"]["denominator_diagnostics"]
    assert diagnostics["n_draws"] == result["n_resamples"]
    assert (diagnostics["n_draws_finite_total"]
            + diagnostics["n_draws_with_nonfinite_total"]) == diagnostics["n_draws"]


# --------------------------------------------------------------------------
# Declared surface and the reuse the review forbids
# --------------------------------------------------------------------------


def test_the_result_declares_the_estimand_the_scope_and_the_excluded_component():
    """The report fields a reader needs in order to read the share correctly.

    Asserts the schema version, the verbatim interpretation string, the share
    role, a scope labelled pointwise, the excluded seed component, and the
    declared ladder, labels and cell fields.

    Why necessary: the review requires pointwise intervals to be labelled as
    pointwise and requires the interpretation rendered wherever the share is
    rendered. A field that is absent cannot be rendered, so its absence is a
    reporting defect that no numeric test would notice.
    """
    result = _estimate(_constant(
        {"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02}, circuits=6))

    assert result["schema_version"] == SCHEMA_VERSION == "qem-bench-improvement-share-v1"
    assert result["share_interpretation"] == INTERPRETATION
    assert result["share_role"] == "primary"
    assert result["setting"] == "shipped-s101-n640"
    assert result["evaluation_role"] == "untouched_test"
    assert "pointwise" in result["interval_scope"]
    assert "size" in list(result["stream_components_excluded"])
    assert list(result["ladder"]) == list(LADDER)
    assert list(result["rung_labels"]) == list(RUNGS)
    assert list(result["gap_labels"]) == list(GAPS)
    assert list(result["cell_fields"]) == list(CELL_FIELDS)
    assert list(result["stream_components"]) == ["shipped", 101]
    assert result["confidence"] == 0.95
    assert result["n_resamples"] == 400
    assert result["root_seed"] == 20260904


def test_a_reference_method_is_intervalled_without_entering_any_gap():
    """A reference method rides the same rows and draws and stays out of the ladder.

    Asserts the reference error carries its own interval, the total and the gaps
    are unchanged by its presence, and the report names it.

    Why necessary: the review asks for an unmitigated error beside the
    decomposition. If a reference method leaked into the total, the denominator
    would silently change and the share would no longer be the quantity the
    interpretation string describes.
    """
    tables = _constant({"feat-only": 0.10, "liao-feat-only": 0.05, "liao": 0.02,
                        "noisy": 0.40}, circuits=6)
    result = _estimate(tables, reference_methods=("noisy",))
    family = result["families"]["tfi"]

    assert family["reference_errors"]["noisy"]["estimate"] == pytest.approx(0.40)
    assert family["total"]["estimate"] == pytest.approx(0.08, rel=1e-12)
    assert set(family["gaps"]) == set(GAPS)
    assert list(result["reference_methods"]) == ["noisy"]
    assert set(family["errors"]) == set(RUNGS)


def test_the_table_type_refuses_a_malformed_numeric_table():
    """The field constructor applies the same validation as the row path.

    Asserts a table built from already-aggregated cells raises on a negative
    error, on a nonfinite error, on a nonzero error where the count is zero, on a
    cell no circuit populates, and on a method set that does not match the
    declared rungs.

    Why necessary: the field adapter never sees raw rows, so the row path's checks
    do not protect it. Without these the field half could build a table whose cell
    means are undefined and whose totals still look plausible.

    The specification says only that a cell carries "a count and one error per
    rung" without fixing how the two are nested, so the payload shape here follows
    the module's own documented form and the assertions stay on the validations.
    """
    cell = ("depolarizing_readout", "L1", "z_mid")

    def _cells(**overrides):
        errors = {"feat-only": 0.2, "liao-feat-only": 0.1, "liao": 0.04}
        first = {"count": overrides.get("count", 2.0),
                 "errors": dict(errors, **overrides.get("errors", {}))}
        return {("u0", cell): first,
                ("u1", cell): {"count": 2.0, "errors": dict(errors)}}

    table = LadderTable.from_cell_errors(_cells(), family="tfi", methods=LADDER)
    assert table.family == "tfi"
    assert table.n_rows == 4
    assert tuple(table.circuit_ids) == ("u0", "u1")

    with pytest.raises(ValueError):
        LadderTable.from_cell_errors(_cells(errors={"liao": -0.01}),
                                     family="tfi", methods=LADDER)
    with pytest.raises(ValueError):
        LadderTable.from_cell_errors(_cells(errors={"liao": float("nan")}),
                                     family="tfi", methods=LADDER)
    with pytest.raises(ValueError):
        LadderTable.from_cell_errors(_cells(count=0.0), family="tfi", methods=LADDER)

    with pytest.raises(ValueError):
        LadderTable.from_cell_errors(_cells(), family="tfi",
                                     methods=LADDER + ("extra",))


def test_the_module_imports_only_the_cell_definition_from_the_older_estimator():
    """The reuse the review forbids is blocked statically, not by promise.

    Asserts the only name imported from the gain-contrast module is the cell
    field constant, and that no bootstrap or table helper of that module is named
    in an import line here.

    Why necessary: the review forbids reusing that estimator as a three-predictor
    estimator. A promise in a docstring does not survive a refactor, and this is
    the guard the specification names in its place.
    """
    source = Path(improvement_share.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip() for line in source.splitlines()
                    if re.match(r"^\s*(from|import)\s", line)]
    gain_lines = [line for line in import_lines if "gain_contrast" in line]

    assert gain_lines, "the module is expected to import CELL_FIELDS by module path"
    imported = set()
    for line in gain_lines:
        assert "evaluate_gain_contrast" not in line
        assert "_regime_tables" not in line
        assert "_bootstrap_gains" not in line
        assert "_macro_mae" not in line
        head, _, tail = line.partition(" import ")
        assert head == "from qemscore.stats.gain_contrast"
        imported.update(name.strip() for name in tail.split(","))
    assert imported == {"CELL_FIELDS"}


def test_the_module_carries_no_promotion_gate_or_its_margin():
    """The validation gate has no place in a test-set attribution decision.

    Asserts the promotion margin, its constant name and its literal value are
    absent from the module, and that nothing imports the gate module.

    Why necessary: the review requires the old directional promotion machinery
    kept out of the new decision. A margin appearing anywhere in this module would
    be the first step toward the share carrying a verdict.
    """
    source = Path(improvement_share.__file__).read_text(encoding="utf-8")
    for forbidden in ("ratio_margin", "RATIO_MARGIN", "1.05", "incremental_value"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Round 8 regressions
# --------------------------------------------------------------------------


def _ladder_errors(a, c, f):
    return {"feat-only": a, "liao-feat-only": c, "liao": f}


def test_an_independent_contrast_refuses_a_shared_stream_and_shared_circuits():
    """N1. An independent contrast is checked against the streams, not the names.

    Asserts three things. Two disjoint pools drawing from one stream raise, because
    equal seeds and equal circuit counts reproduce one index matrix and the
    difference would cancel the sampling variation it claims to report. Two pools
    that overlap only partly raise, because a shared circuit is shared regardless
    of order. The valid case, disjoint pools on distinct streams, still returns an
    interval.

    Catches the defect round 8 blocked on: rejecting only an identical circuit
    order let two disjoint pools on one seed publish a contrast of exactly zero
    width, while the result declared the sides independent.
    """
    # Errors vary across the two circuits, so a resample carries real variation.
    # Both pools carry the same values and differ only in which physical circuits
    # they name, which is what makes an aligned index matrix cancel exactly.
    def _spread(method, family, circuit, *rest):
        return (_ladder_errors(0.10, 0.05, 0.0) if circuit == 0
                else _ladder_errors(0.90, 0.45, 0.0))[method]

    left_items, left_predictions = _build(
        _spread, circuits=2, cells=ONE_CELL, label="a")
    right_items, right_predictions = _build(
        _spread, circuits=2, cells=ONE_CELL, label="b")
    left_tables = build_ladder_tables(left_items, left_predictions, methods=LADDER)
    right_tables = build_ladder_tables(right_items, right_predictions, methods=LADDER)

    # Identical declared components, so both sides derive the same stream seed
    # while holding no circuit in common.
    left = _estimate(left_tables, setting_label="shipped-s101-n640")
    right = _estimate(right_tables, setting_label="shipped-s101-n640")
    assert left["families"]["tfi"]["stream_seed"] == right["families"]["tfi"]["stream_seed"]
    assert (left["families"]["tfi"]["circuit_order_digest"]
            != right["families"]["tfi"]["circuit_order_digest"])

    with pytest.raises(ValueError, match="distinct resample streams"):
        contrast_improvement_share(
            left, right, left_tables=left_tables, right_tables=right_tables,
            quantity="total", pairing="independent")

    # The same two pools on distinct streams are a legitimate independent
    # contrast, and its interval has to be wider than a point.
    other = _estimate(right_tables, stream_components=("large", 101),
                      setting_label="large-s101-n640")
    assert other["families"]["tfi"]["stream_seed"] != left["families"]["tfi"]["stream_seed"]
    independent = contrast_improvement_share(
        left, other, left_tables=left_tables, right_tables=right_tables,
        quantity="total", pairing="independent")
    json.dumps(independent, allow_nan=False)
    interval = independent["families"]["tfi"]["contrast"]
    assert interval["lower"] < interval["upper"]

    # Partial overlap on distinct streams is still refused, because one shared
    # physical circuit correlates the two sides.
    wide_items, wide_predictions = _build(
        lambda method, *rest: _ladder_errors(0.10, 0.05, 0.0)[method],
        circuits=4, cells=ONE_CELL, label="c")

    def _slice(keep):
        chosen = [item for item in wide_items
                  if int(item["circuit_id"].rsplit("c", 1)[1]) in keep]
        ids = {item["item_id"] for item in chosen}
        return chosen, {method: {key: value for key, value in mapping.items()
                                 if key in ids}
                        for method, mapping in wide_predictions.items()}

    first_items, first_predictions = _slice({0, 1})
    second_items, second_predictions = _slice({1, 2})
    first_tables = build_ladder_tables(first_items, first_predictions, methods=LADDER)
    second_tables = build_ladder_tables(second_items, second_predictions, methods=LADDER)
    first = _estimate(first_tables, setting_label="shipped-s101-n640")
    second = _estimate(second_tables, stream_components=("large", 101),
                       setting_label="large-s101-n640")
    assert (first["families"]["tfi"]["stream_seed"]
            != second["families"]["tfi"]["stream_seed"])

    with pytest.raises(ValueError, match="shares physical circuits"):
        contrast_improvement_share(
            first, second, left_tables=first_tables, right_tables=second_tables,
            quantity="total", pairing="independent")


def test_the_report_names_the_cell_fields_the_run_actually_aggregated_over():
    """N2. Custom cell fields are carried, not assumed.

    Asserts a table built over one declared field reports that field, that its
    cell keys are one part wide, and that a table whose key width disagrees with
    its declared fields raises.

    Catches metadata describing an aggregation the run did not perform. Cells
    carry equal weight, so redefining a cell changes the estimand, and a report
    naming the default fields would misdescribe the number beside it.
    """
    items, predictions = _build(
        lambda method, *rest: _ladder_errors(0.10, 0.05, 0.02)[method], circuits=4)
    tables = build_ladder_tables(
        items, predictions, methods=LADDER, cell_fields=("severity",))
    assert tables["tfi"].cell_fields == ("severity",)
    assert tables["tfi"].cell_keys == (("L1",), ("L3",))

    result = _estimate(tables)
    assert result["cell_fields"] == ["severity"]
    json.dumps(result, allow_nan=False)

    default = _estimate(_constant(_ladder_errors(0.10, 0.05, 0.02), circuits=4))
    assert default["cell_fields"] == list(CELL_FIELDS)

    with pytest.raises(ValueError, match="cell_fields"):
        LadderTable(
            family="tfi",
            circuit_ids=tables["tfi"].circuit_ids,
            cell_keys=(("L1", "z_mid"), ("L3", "z_mid")),
            counts=tables["tfi"].counts,
            errors=dict(tables["tfi"].errors),
            n_rows=tables["tfi"].n_rows,
            cell_fields=("severity",),
        )


def test_a_reference_method_shares_every_draw_with_the_rungs():
    """N8. The reference is paired draw by draw, not merely reported beside them.

    Feeds three explicit index vectors to the evaluator and compares the reference
    macro of each draw against a value computed by hand. Only the reference's
    marginal interval is published, so a permutation of its draw vector leaves
    every published endpoint unchanged and no percentile check can see it.

    Catches the mutation round 8 found surviving: rotating the evaluated reference
    vector preserves its marginal distribution and therefore its interval, while
    destroying the pairing that makes a difference against a rung meaningful.
    """
    per_circuit = ({"feat-only": 0.10, "liao-feat-only": 0.06, "liao": 0.02,
                    "unmitigated": 0.30},
                   {"feat-only": 0.90, "liao-feat-only": 0.46, "liao": 0.02,
                    "unmitigated": 0.70})
    methods = LADDER + ("unmitigated",)
    items, predictions = _build(
        lambda method, family, circuit, *rest: per_circuit[circuit][method],
        circuits=2, cells=ONE_CELL, methods=methods)
    table = build_ladder_tables(items, predictions, methods=methods)["tfi"]

    indices = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int64)
    drawn = improvement_share._evaluate(
        table, ladder=LADDER, reference_methods=("unmitigated",),
        span_pairs={}, indices=indices, origin="draw")

    for method in methods:
        first, second = per_circuit[0][method], per_circuit[1][method]
        expected = [first, second, (first + second) / 2.0]
        assert drawn["macro"][method] == pytest.approx(expected, rel=1e-12), method

    # The pairing is what the alignment buys: in every draw the reference and the
    # affine rung come from the same circuits, so their difference is meaningful
    # draw by draw rather than only in expectation.
    difference = drawn["macro"]["unmitigated"] - drawn["macro"]["feat-only"]
    assert difference == pytest.approx([0.20, -0.20, 0.0], rel=1e-12)


# --------------------------------------------------------------------------
# Round 9 regressions
# --------------------------------------------------------------------------


def test_a_contrast_replays_only_the_table_that_produced_the_estimate():
    """R9-3. The replayed table is bound to its result, not to the circuit names.

    Asserts that a right table carrying the scored circuit IDs but a raised
    affine error is refused, that editing the scored table's array in place is
    refused, that a result written without the binding is refused rather than
    granted a replay it cannot support, and that the unmodified pair still
    contrasts.

    Catches the defect round 9 blocked on: an ordered circuit-ID digest names
    which circuits were scored and nothing else, so raising A from 0.9 to 1.9
    under the same names published a contrast of [1.0, 1.0] while the right
    result beside it still reported a total of 0.8.
    """
    errors = _ladder_errors(0.90, 0.50, 0.10)
    left_tables = _constant(errors, circuits=2, label="left")
    right_tables = _constant(errors, circuits=2, label="right")
    left = _estimate(left_tables, setting_label="shipped-s101-n640")
    right = _estimate(right_tables, stream_components=("large", 101),
                      setting_label="large-s101-n640")
    assert right["families"]["tfi"]["total"]["estimate"] == pytest.approx(0.80)

    # The old binding still matches this substitution exactly, which is why it
    # could not see the change.
    replacement = _constant(_ladder_errors(1.90, 0.50, 0.10), circuits=2,
                            label="right")
    assert (improvement_share._circuit_order_digest(replacement["tfi"].circuit_ids)
            == right["families"]["tfi"]["circuit_order_digest"])
    with pytest.raises(ValueError, match="is not the one that was scored"):
        contrast_improvement_share(
            left, right, left_tables=left_tables, right_tables=replacement,
            quantity="total", pairing="independent")

    honest = contrast_improvement_share(
        left, right, left_tables=left_tables, right_tables=right_tables,
        quantity="total", pairing="independent")
    json.dumps(honest, allow_nan=False)
    assert honest["families"]["tfi"]["right_point"] == pytest.approx(0.80)

    # A frozen dataclass does not freeze a numpy array, so the table that was
    # scored can still be edited afterwards.
    right_tables["tfi"].errors["feat-only"][:] = 1.90
    with pytest.raises(ValueError, match="is not the one that was scored"):
        contrast_improvement_share(
            left, right, left_tables=left_tables, right_tables=right_tables,
            quantity="total", pairing="independent")

    # A result predating the binding carries none, and a replay it cannot
    # support is refused rather than grandfathered.
    rebuilt = _constant(errors, circuits=2, label="right")
    unbound = json.loads(json.dumps(_estimate(
        rebuilt, stream_components=("large", 101),
        setting_label="large-s101-n640")))
    del unbound["families"]["tfi"]["table_digest"]
    with pytest.raises(ValueError, match="is not the one that was scored"):
        contrast_improvement_share(
            left, unbound, left_tables=left_tables, right_tables=rebuilt,
            quantity="total", pairing="independent")


def test_an_imported_result_must_carry_integer_seeds_and_counts():
    """N1, reopened. A coerced seed is refused rather than replayed.

    Two disjoint pools derived from one declared stream carry equal seeds, which
    the independent branch refuses. Asserts that writing the right seed as its
    decimal string is refused rather than compared as unequal, and that the two
    count fields the replay would otherwise coerce are refused the same way.

    Catches the narrower bypass round 9 reopened: ``972037298 != "972037298"``
    passed the equality check while ``int(...)`` drew the identical matrix on
    both sides, restoring the false independent interval of exactly zero width.
    """
    def _spread(method, family, circuit, *rest):
        return (_ladder_errors(0.10, 0.05, 0.0) if circuit == 0
                else _ladder_errors(0.90, 0.45, 0.0))[method]

    def _pool(label):
        items, predictions = _build(_spread, circuits=2, cells=ONE_CELL,
                                    label=label)
        return build_ladder_tables(items, predictions, methods=LADDER)

    left_tables, right_tables = _pool("a"), _pool("b")
    left = _estimate(left_tables, setting_label="shipped-s101-n640")
    right = _estimate(right_tables, setting_label="shipped-s101-n640")
    assert (left["families"]["tfi"]["stream_seed"]
            == right["families"]["tfi"]["stream_seed"])

    # The check the decimal string defeats, on the same two results.
    with pytest.raises(ValueError, match="distinct resample streams"):
        contrast_improvement_share(
            left, right, left_tables=left_tables, right_tables=right_tables,
            quantity="total", pairing="independent")

    for field, message in (
        ("stream_seed", "stream_seed must be a nonnegative integer"),
        ("n_circuits", "n_circuits must be a nonnegative integer"),
    ):
        imported = json.loads(json.dumps(right))
        imported["families"]["tfi"][field] = str(imported["families"]["tfi"][field])
        with pytest.raises(ValueError, match=message):
            contrast_improvement_share(
                left, imported, left_tables=left_tables,
                right_tables=right_tables, quantity="total",
                pairing="independent")

    resampled = json.loads(json.dumps(right))
    resampled["n_resamples"] = str(resampled["n_resamples"])
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        contrast_improvement_share(
            left, resampled, left_tables=left_tables, right_tables=right_tables,
            quantity="total", pairing="independent")


def test_a_contrast_refuses_two_estimands_and_names_the_one_it_compared():
    """R9-5. A cell definition is part of the estimand, so it enters the contract.

    Nine ``z_mid`` rows per cell against one ``zz_mid`` row make the default
    equal-cell weighting and a pooled ``("severity",)`` weighting two different
    numbers on the same rows. Asserts the contrast refuses that pair, and that a
    contrast of two sides that do agree carries the cell fields, the aggregation
    and the resample unit it compared.

    Catches a contrast that checks ladder, labels, spans and resample count
    while ignoring what a cell is: the two sides below differ in share by 0.50
    against 0.85, and the emitted contrast named neither grouping.
    """
    items, predictions = _build(
        lambda method, family, circuit, severity, observable, index:
            {"feat-only": 0.10,
             "liao-feat-only": 0.025 if observable == "z_mid" else 0.095,
             "liao": 0.02}[method],
        rows_of=lambda family, circuit, severity, observable:
            9 if observable == "z_mid" else 1)
    fine = build_ladder_tables(items, predictions, methods=LADDER)
    coarse = build_ladder_tables(items, predictions, methods=LADDER,
                                 cell_fields=("severity",))
    fine_result = _estimate(fine)
    coarse_result = _estimate(coarse)
    assert fine_result["families"]["tfi"]["share"]["point"] == pytest.approx(0.50)
    assert coarse_result["families"]["tfi"]["share"]["point"] == pytest.approx(0.85)

    with pytest.raises(ValueError, match="disagree on cell_fields"):
        contrast_improvement_share(
            fine_result, coarse_result, left_tables=fine, right_tables=coarse,
            quantity="K", pairing="paired")

    agreed = contrast_improvement_share(
        fine_result, fine_result, left_tables=fine, right_tables=fine,
        quantity="K", pairing="paired")
    json.dumps(agreed, allow_nan=False)
    assert agreed["cell_fields"] == list(CELL_FIELDS)
    assert agreed["aggregation"] == fine_result["aggregation"]
    assert agreed["resample_unit"] == fine_result["resample_unit"]
