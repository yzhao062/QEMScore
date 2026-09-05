"""Primary-endpoint estimator: known gains, pairing, streams, blocks, denominators."""

import json
import math

import numpy as np
import pytest

from qem_bench.stats.gain_contrast import evaluate_gain_contrast


SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")


def _regime(errors, *, families=("tfi", "heisenberg"), circuits=8, label="r"):
    """Build one regime whose per-row absolute errors are exactly prescribed.

    ``errors`` maps a method to its absolute error, so a test can state the gain
    it expects instead of deriving one. Targets are zero and predictions carry
    the error directly, which keeps every cell mean equal to that error.
    """
    rows, predictions = [], {method: {} for method in errors}
    for family in families:
        for circuit in range(circuits):
            for severity in SEVERITIES:
                for observable in OBSERVABLES:
                    item_id = f"{label}-{family}-{circuit}-{severity}-{observable}"
                    rows.append({
                        "item_id": item_id,
                        "dataset_schema_version": "split-v2",
                        "split": "test", "domain": "target",
                        "family": family, "stratum": "continuous_regression",
                        "circuit_id": f"{label}-{family}-c{circuit}",
                        "noise_family": "depolarizing_readout",
                        "severity": severity, "observable": observable,
                        "ideal_expectation": 0.0,
                    })
                    for method, error in errors.items():
                        predictions[method][item_id] = error
    return {"items": rows, "predictions_by_method": predictions}


def _evaluate(shipped, large, **kwargs):
    return evaluate_gain_contrast(
        {"shipped": shipped, "large": large},
        full_method="ridge", control_method="feat-only",
        baseline_regime="shipped", contrast_regime="large",
        n_resamples=kwargs.pop("n_resamples", 400), root_seed=20260904, **kwargs,
    )


def test_gain_and_contrast_recover_prescribed_values():
    # Errors are constant per row, so both gains are exact and so is the contrast.
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    result = _evaluate(shipped, large)

    assert result["status"] == "positive_in_every_family"
    assert sorted(result["families"]) == ["heisenberg", "tfi"]
    for family in result["families"].values():
        assert family["status"] == "evaluated"
        assert family["regimes"]["shipped"]["gain"] == pytest.approx(0.0)
        assert family["regimes"]["large"]["gain"] == pytest.approx(0.2)
        assert family["regimes"]["large"]["absolute_reduction"] == pytest.approx(0.02)
        assert family["contrast"]["estimate"] == pytest.approx(0.2)
        # Constant errors leave no resampling spread, so the interval collapses
        # onto the estimate and still excludes zero from above.
        assert family["contrast"]["lower"] == pytest.approx(0.2)
        assert family["contrast"]["upper"] == pytest.approx(0.2)
        assert family["contrast"]["excludes_zero_above"] is True
    assert result["families_with_positive_contrast"] == ["heisenberg", "tfi"]


def test_a_positive_contrast_can_sit_between_two_negative_gains():
    """The review's counterexample: promotion licenses a rise, not a benefit."""
    shipped = _regime({"ridge": 0.12, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.11, "feat-only": 0.10}, label="l")
    result = _evaluate(shipped, large)

    family = result["families"]["tfi"]
    assert family["regimes"]["shipped"]["gain"] == pytest.approx(-0.2)
    assert family["regimes"]["large"]["gain"] == pytest.approx(-0.1)
    assert family["contrast"]["estimate"] == pytest.approx(0.1)
    assert family["contrast"]["excludes_zero_above"] is True
    # Both endpoint intervals are strictly negative, so nothing here supports a
    # claim of positive benefit at either endpoint.
    for label in ("shipped", "large"):
        assert family["regimes"][label]["gain_interval"]["excludes_zero_below"] is True
    assert result["status"] == "positive_in_every_family"


def test_predictors_share_each_draw_within_a_regime():
    """Pairing is what keeps a common circuit effect out of the ratio.

    Both predictors carry the same per-circuit error scale, so every resample
    scales both macro errors identically and the ratio is invariant. An
    implementation that resampled the two predictors independently would leave
    that scale in the numerator only, and the interval would open up.
    """
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.05, "feat-only": 0.10}, label="l")
    for regime, ratio in ((shipped, 1.0), (large, 0.5)):
        for index, row in enumerate(regime["items"]):
            scale = 1.0 + 4.0 * ((index // (len(SEVERITIES) * len(OBSERVABLES))) % 7)
            regime["predictions_by_method"]["feat-only"][row["item_id"]] = 0.10 * scale
            regime["predictions_by_method"]["ridge"][row["item_id"]] = 0.10 * scale * ratio

    result = _evaluate(shipped, large)
    for family in result["families"].values():
        assert family["regimes"]["shipped"]["gain"] == pytest.approx(0.0)
        assert family["regimes"]["large"]["gain"] == pytest.approx(0.5)
        interval = family["contrast"]
        assert interval["lower"] == pytest.approx(0.5)
        assert interval["upper"] == pytest.approx(0.5)


def test_regimes_draw_from_independent_recorded_streams():
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    result = _evaluate(shipped, large)

    seeds = result["stream_mapping"]
    assert set(seeds) == {"shipped/tfi", "shipped/heisenberg",
                          "large/tfi", "large/heisenberg"}
    assert len(set(seeds.values())) == 4, seeds
    for family in result["families"].values():
        assert family["regimes"]["shipped"]["seed"] != family["regimes"]["large"]["seed"]
    # The mapping is a function of the root seed alone, so the same inputs
    # reproduce it and a different root seed moves every stream.
    assert _evaluate(shipped, large)["stream_mapping"] == seeds
    moved = evaluate_gain_contrast(
        {"shipped": shipped, "large": large},
        full_method="ridge", control_method="feat-only",
        baseline_regime="shipped", contrast_regime="large",
        n_resamples=400, root_seed=20260905,
    )["stream_mapping"]
    assert set(moved.values()).isdisjoint(seeds.values())


def test_a_circuit_carries_every_severity_and_observable_into_a_draw():
    """One circuit is an outlier in both of its lower-severity cells.

    Blocking on the physical circuit makes the gain a function of one integer:
    how many of the four draws landed on that circuit. The support is therefore
    the five values enumerated below, and the interval endpoints must be members
    of it. Resampling cells independently would decouple the outlier's two cells
    and put the endpoints between those values, which is how a row-level
    resample smuggles four rows of one circuit in as four observations.
    """
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, circuits=4, label="s")
    large = _regime({"ridge": 0.10, "feat-only": 0.10}, circuits=4, label="l")
    for row in large["items"]:
        if row["circuit_id"] == "l-tfi-c0" and row["severity"] == "L1":
            large["predictions_by_method"]["ridge"][row["item_id"]] = 0.0

    result = _evaluate(shipped, large, n_resamples=4000)
    family = result["families"]["tfi"]
    assert family["regimes"]["large"]["n_circuits"] == 4
    assert family["regimes"]["large"]["n_items"] == 16
    assert family["regimes"]["large"]["n_cells"] == 4

    # With k of the four draws on the outlier, its two lower-severity cells each
    # average 0.10 * (4 - k) / 4 while the other two stay at 0.10.
    support = []
    for k in range(5):
        outlier_cell = 0.10 * (4 - k) / 4
        macro = (2 * outlier_cell + 2 * 0.10) / 4
        support.append(1.0 - macro / 0.10)
    assert support == pytest.approx([0.0, 0.125, 0.25, 0.375, 0.5])
    # k is Binomial(4, 1/4): the 2.5th percentile sits on k=0 and the 97.5th on
    # k=3, and both order statistics fall strictly inside their blocks.
    interval = family["regimes"]["large"]["gain_interval"]
    assert interval["estimate"] == pytest.approx(0.125)
    assert interval["lower"] == pytest.approx(0.0)
    assert interval["upper"] == pytest.approx(0.375)
    assert family["contrast"]["lower"] == pytest.approx(0.0)
    assert family["contrast"]["upper"] == pytest.approx(0.375)


def test_an_unavailable_denominator_is_reported_rather_than_dropped():
    shipped = _regime({"ridge": 0.10, "feat-only": 0.0}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    result = _evaluate(shipped, large)

    assert result["status"] == "not_evaluable"
    assert result["reason"] == "not_every_family_is_evaluable"
    for family in result["families"].values():
        assert family["status"] == "not_evaluable"
        assert family["reason"] == "zero_control_mae"
        assert family["contrast"] is None
        assert family["regimes"]["shipped"]["gain"] is None
        # The regime that could be summarized still reports its numbers.
        assert family["regimes"]["large"]["control_mae"] == pytest.approx(0.10)
    assert result["families_with_positive_contrast"] == []


def test_a_single_circuit_pool_cannot_estimate_uncertainty():
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, circuits=1, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, circuits=1, label="l")
    result = _evaluate(shipped, large)
    assert result["status"] == "not_evaluable"
    for family in result["families"].values():
        assert family["reason"] == "fewer_than_two_circuits"


def test_partial_positivity_is_not_promoted():
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    for row in large["items"]:
        if row["family"] == "heisenberg":
            large["predictions_by_method"]["ridge"][row["item_id"]] = 0.10
    result = _evaluate(shipped, large)

    assert result["status"] == "evaluated"
    assert result["families_with_positive_contrast"] == ["tfi"]
    assert result["families"]["heisenberg"]["contrast"]["excludes_zero_above"] is False


def test_the_result_serializes_under_strict_json():
    import json

    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    encoded = json.dumps(_evaluate(shipped, large), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


@pytest.mark.parametrize("mutation,message", [
    ({"split": "validation"}, "untouched test rows only"),
    ({"dataset_schema_version": "legacy-v1"}, "split-v2 metadata"),
    ({"family": "random_clifford"}, "family and stratum must match"),
    ({"circuit_id": ""}, "nonempty circuit_id"),
    ({"ideal_expectation": float("inf")}, "targets must be finite"),
])
def test_malformed_rows_raise_rather_than_returning_a_verdict(mutation, message):
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    shipped["items"][0] = dict(shipped["items"][0], **mutation)
    with pytest.raises(ValueError, match=message):
        _evaluate(shipped, large)


def test_arguments_are_checked_before_any_resampling():
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, label="l")
    with pytest.raises(ValueError, match="must differ"):
        evaluate_gain_contrast(
            {"shipped": shipped, "large": large}, full_method="ridge",
            control_method="ridge", baseline_regime="shipped",
            contrast_regime="large", root_seed=1, n_resamples=10)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_gain_contrast(
            {"shipped": shipped, "large": large}, full_method="ridge",
            control_method="feat-only", baseline_regime="shipped",
            contrast_regime="large", root_seed=1, n_resamples=0)
    with pytest.raises(ValueError, match="exactly the two regimes"):
        evaluate_gain_contrast(
            {"shipped": shipped, "large": large, "extra": shipped},
            full_method="ridge", control_method="feat-only",
            baseline_regime="shipped", contrast_regime="large",
            root_seed=1, n_resamples=10)


def test_a_known_spread_reproduces_a_hand_computed_quantile():
    """Two circuits with different errors give an enumerable draw distribution.

    With two circuits the resample is one of three multisets, with probabilities
    1/4, 1/2, 1/4. The gain takes three values accordingly, so the 2.5th and
    97.5th percentiles must land on the extreme ones.
    """
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, circuits=2,
                      families=("tfi", "heisenberg"), label="s")
    large = _regime({"ridge": 0.10, "feat-only": 0.10}, circuits=2,
                    families=("tfi", "heisenberg"), label="l")
    for row in large["items"]:
        if row["circuit_id"].endswith("c0"):
            large["predictions_by_method"]["ridge"][row["item_id"]] = 0.02
        else:
            large["predictions_by_method"]["ridge"][row["item_id"]] = 0.06

    result = _evaluate(shipped, large, n_resamples=4000)
    family = result["families"]["tfi"]
    both = 1.0 - ((0.02 + 0.06) / 2) / 0.10
    only_low = 1.0 - 0.02 / 0.10
    only_high = 1.0 - 0.06 / 0.10
    assert family["regimes"]["large"]["gain"] == pytest.approx(both)
    interval = family["regimes"]["large"]["gain_interval"]
    assert interval["lower"] == pytest.approx(only_high)
    assert interval["upper"] == pytest.approx(only_low)
    assert math.isclose(family["contrast"]["estimate"], both, rel_tol=1e-12)


def test_a_family_missing_from_one_regime_is_reported_rather_than_omitted():
    """Iterating the intersection would let one family read as every family."""
    shipped = _regime({"ridge": 0.10, "feat-only": 0.10}, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, families=("tfi",), label="l")
    result = _evaluate(shipped, large)

    assert sorted(result["families"]) == ["heisenberg", "tfi"]
    heisenberg = result["families"]["heisenberg"]
    assert heisenberg["status"] == "not_evaluable"
    assert heisenberg["reason"] == "family_missing_from_a_regime"
    assert heisenberg["regimes"]["large"] is None
    # The regime that did produce the family still reports its numbers.
    assert heisenberg["regimes"]["shipped"]["control_mae"] == pytest.approx(0.10)
    assert result["families"]["tfi"]["status"] == "evaluated"
    # One evaluated family out of two cannot be positive in every family.
    assert result["status"] == "not_evaluable"
    assert result["reason"] == "not_every_family_is_evaluable"
    assert result["families_with_positive_contrast"] == ["tfi"]
    json.dumps(result, allow_nan=False)


def test_a_control_error_too_small_to_divide_by_leaves_no_infinity_behind():
    """The positive-denominator guard passes a value the ratio still overflows."""
    shipped = _regime({"ridge": 1.0, "feat-only": 5e-324}, circuits=2, label="s")
    large = _regime({"ridge": 0.08, "feat-only": 0.10}, circuits=2, label="l")
    result = _evaluate(shipped, large)

    for family in result["families"].values():
        assert family["status"] == "not_evaluable"
        assert family["reason"] == "nonfinite_point_ratio"
        assert family["regimes"]["shipped"]["gain"] is None
        assert family["regimes"]["shipped"]["gain_interval"] is None
    json.dumps(result, allow_nan=False)


def _uneven_regime(label):
    """One cell row for the first circuit and two for the second, per cell.

    A point estimate that averages circuit-cell means weights those circuits
    equally; a bootstrap draw pools their errors and counts. The two disagree
    unless the point estimate pools as well.
    """
    rows, predictions = [], {"ridge": {}, "feat-only": {}}
    for severity in SEVERITIES:
        for observable in OBSERVABLES:
            for circuit, errors in (("a", (0.1,)), ("b", (0.3, 0.3))):
                for index, error in enumerate(errors):
                    item_id = f"{label}-{circuit}-{severity}-{observable}-{index}"
                    rows.append({
                        "item_id": item_id,
                        "dataset_schema_version": "split-v2",
                        "split": "test", "domain": "target",
                        "family": "tfi", "stratum": "continuous_regression",
                        "circuit_id": f"{label}-c{circuit}",
                        "noise_family": "depolarizing_readout",
                        "severity": severity, "observable": observable,
                        "ideal_expectation": 0.0,
                    })
                    predictions["ridge"][item_id] = error
                    predictions["feat-only"][item_id] = 0.1
    return {"items": rows, "predictions_by_method": predictions}


def test_the_point_estimate_pools_each_cell_the_way_a_draw_does():
    shipped = _uneven_regime("s")
    large = _uneven_regime("l")
    result = evaluate_gain_contrast(
        {"shipped": shipped, "large": large},
        full_method="ridge", control_method="feat-only",
        baseline_regime="shipped", contrast_regime="large",
        n_resamples=200, root_seed=20260904,
    )

    family = result["families"]["tfi"]
    pooled = (0.1 + 0.3 + 0.3) / 3
    assert family["regimes"]["shipped"]["full_mae"] == pytest.approx(pooled)
    assert family["regimes"]["shipped"]["gain"] == pytest.approx(1.0 - pooled / 0.1)
    # A draw that happens to take each circuit once reproduces the point value,
    # so the estimate has to lie inside the resampled support.
    interval = family["regimes"]["shipped"]["gain_interval"]
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
