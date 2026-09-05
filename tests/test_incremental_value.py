"""Independent gate outcomes, pairing, and circuit-block uncertainty guards."""

import copy
import json

import pytest

from qem_bench.stats.incremental_value import evaluate_incremental_value


def _case(families=("tfi", "heisenberg"), circuits=4):
    rows = []
    for family in families:
        for circuit in range(circuits):
            rows.append({
                "item_id": f"{family}-{circuit}",
                "dataset_schema_version": "split-v2",
                "split_id": "S0", "partition_id": "headline",
                "split": "validation", "domain": "source",
                "family": family, "stratum": "continuous_regression",
                "circuit_id": f"{family}-c{circuit}",
                "circuit_pool_id": family,
                "noise_family": "depolarizing_readout", "severity": "L1",
                "observable": "z_mid", "ideal_expectation": 0.0,
            })
    predictions = {
        method: {row["item_id"]: value for row in rows}
        for method, value in (("ridge", 0.1), ("feat-only", 0.2), ("noisy-only", 0.3))
    }
    return rows, predictions


def _gate(rows, predictions, **kwargs):
    return evaluate_incremental_value(rows, predictions, seed=13, n_resamples=200, **kwargs)


def test_two_families_need_both_controls_and_positive_paired_intervals():
    rows, predictions = _case()
    result = _gate(rows, predictions)
    assert result["status"] == "passed"
    assert result["passing_families"] == ["heisenberg", "tfi"]
    for family in result["families"].values():
        assert family["full_mae"] == pytest.approx(0.1)
        for control, difference in (("feat-only", 0.1), ("noisy-only", 0.2)):
            interval = family["comparisons"][control]["difference_interval"]
            assert (interval["lower"], interval["upper"]) == pytest.approx((difference, difference))
            assert interval["n_circuits"] == 4
    predictions["noisy-only"]["tfi-0"] = 0
    predictions["noisy-only"]["tfi-1"] = 0
    predictions["noisy-only"]["tfi-2"] = 0
    predictions["noisy-only"]["tfi-3"] = 0
    assert _gate(rows, predictions)["status"] == "failed"


@pytest.mark.parametrize("control", ["feat-only", "noisy-only"])
def test_margin_is_strict_and_separate_from_positive_difference(control):
    rows, predictions = _case()
    predictions["ridge"] = dict.fromkeys(predictions["ridge"], 1.0)
    predictions["feat-only"] = dict.fromkeys(predictions["ridge"], 2.0)
    predictions["noisy-only"] = dict.fromkeys(predictions["ridge"], 2.0)
    predictions[control] = dict.fromkeys(predictions["ridge"], 1.05)
    result = _gate(rows, predictions)
    comparison = result["families"]["tfi"]["comparisons"][control]
    assert comparison["positive_interval"] is True
    assert comparison["margin_met"] is False
    assert result["status"] == "failed"


def test_positive_point_margin_does_not_replace_uncertainty():
    rows, predictions = _case(circuits=2)
    for row in rows:
        key = row["item_id"]
        predictions["ridge"][key] = 0.2
        predictions["feat-only"][key] = 0.0 if key.endswith("0") else 0.8
    result = _gate(rows, predictions)
    comparison = result["families"]["tfi"]["comparisons"]["feat-only"]
    assert comparison["margin_met"] is True
    assert comparison["difference_interval"]["lower"] == pytest.approx(-0.2)
    assert comparison["positive_interval"] is False
    assert result["status"] == "failed"


def test_pairs_keep_circuit_variation_and_all_severity_siblings_together():
    rows, _ = _case(circuits=4)
    rows = [dict(row, item_id=f"{row['item_id']}-{severity}", severity=severity)
            for row in rows for severity in ("L1", "L2")]
    predictions = {method: {} for method in ("ridge", "feat-only", "noisy-only")}
    for row in rows:
        circuit = int(row["circuit_id"][-1])
        full = 1 + circuit
        delta = 0.1 + (1 if row["severity"] == "L1" else -1) * circuit / 5
        predictions["ridge"][row["item_id"]] = full
        predictions["feat-only"][row["item_id"]] = full + delta
        predictions["noisy-only"][row["item_id"]] = full + delta
    result = _gate(rows, predictions)
    for family in result["families"].values():
        interval = family["comparisons"]["feat-only"]["difference_interval"]
        assert interval["n_circuits"] == 4
        assert (interval["lower"], interval["upper"]) == pytest.approx((0.1, 0.1))


def test_macro_cells_do_not_pool_unequal_item_counts():
    rows, _ = _case(circuits=4)
    rows += [dict(row, item_id=row["item_id"] + "-L2", severity="L2")
             for row in rows if row["item_id"].endswith(("0", "1"))]
    predictions = {method: {row["item_id"]: (0.0 if row["severity"] == "L1" else 0.6)
                           for row in rows}
                   for method in ("ridge", "feat-only", "noisy-only")}
    result = _gate(rows, predictions)
    assert result["families"]["tfi"]["full_mae"] == pytest.approx(0.3)
    assert result["families"]["tfi"]["full_mae"] != pytest.approx(0.2)


def test_clifford_cannot_supply_second_family_and_its_labels_are_not_read():
    rows, predictions = _case(families=("tfi",))
    rows.append({"item_id": "rc", "dataset_schema_version": "split-v2",
                 "split_id": "S0", "partition_id": "headline",
                 "split": "validation", "domain": "source",
                 "family": "random_clifford", "stratum": "clifford_control"})
    result = _gate(rows, predictions)
    assert result["status"] == "not_evaluable"
    assert result["reason"] == "fewer_than_two_continuous_families"
    assert result["excluded_items"] == 1
    assert set(result["families"]) == {"tfi"}


@pytest.mark.parametrize("role,domain", [("train", "source"), ("test", "target"),
                                        ("test", "source"), ("validation", "target")])
def test_forbidden_roles_rejected_before_reading_labels(role, domain):
    rows, predictions = _case()
    rows[0].update(split=role, domain=domain)
    del rows[0]["ideal_expectation"]
    with pytest.raises(ValueError, match="source validation rows only"):
        _gate(rows, predictions)


def test_missing_validation_methods_and_circuit_replication_are_not_passes():
    assert _gate([], {})["status"] == "not_evaluable"
    rows, predictions = _case()
    del predictions["noisy-only"]
    assert _gate(rows, predictions)["reason"] == "missing_methods: noisy-only"
    rows, predictions = _case(circuits=1)
    assert _gate(rows, predictions)["status"] == "not_evaluable"
    rows, predictions = _case(circuits=2)
    for row in rows:
        row["circuit_pool_id"] = row["circuit_id"]
    assert _gate(rows, predictions)["status"] == "not_evaluable"


def test_two_passing_families_suffice_with_a_third_unavailable_and_ood_source_rows():
    rows, predictions = _case(families=("tfi", "heisenberg", "qaoa"))
    for row in rows:
        row["split_id"] = "S1"
        if row["family"] == "qaoa":
            row["circuit_pool_id"] = row["circuit_id"]
    result = _gate(rows, predictions)
    assert result["status"] == "passed"
    assert result["families"]["qaoa"]["status"] == "not_evaluable"
    assert result["passing_families"] == ["heisenberg", "tfi"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "duplicate", "strata", "family", "scope"])
def test_malformed_pairing_is_rejected(mutation):
    rows, predictions = _case()
    if mutation == "missing":
        del predictions["ridge"][rows[0]["item_id"]]
    elif mutation == "extra":
        predictions["ridge"]["target-test-item"] = 0.1
    elif mutation == "nan":
        predictions["ridge"][rows[0]["item_id"]] = float("nan")
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "strata":
        rows[1]["circuit_id"] = rows[0]["circuit_id"]
        rows[1]["circuit_pool_id"] = "another-pool"
    elif mutation == "family":
        rows[0]["stratum"] = "clifford_control"
    elif mutation == "scope":
        rows[0]["split_id"] = "S4"
    with pytest.raises(ValueError):
        _gate(rows, predictions)


def test_a_conflicting_pool_alias_cannot_widen_singleton_pools():
    """The canonical pool decides replication; an alias may repeat it, never widen it.

    Each circuit here sits in its own pool, so no pool can estimate circuit
    sampling uncertainty and the gate must decline to evaluate. Relabelling the
    rows with a family-wide alias would merge four singleton pools into two
    populated ones and turn that refusal into a pass, which is how insufficient
    replication becomes a positive scientific claim.
    """
    rows, predictions = _case()
    for row in rows:
        row["circuit_pool_id"] = row["circuit_id"]
    assert _gate(rows, predictions)["status"] == "not_evaluable"

    widened = copy.deepcopy(rows)
    for row in widened:
        row["bootstrap_stratum_id"] = row["family"]
    with pytest.raises(ValueError, match="bootstrap_stratum_id must match"):
        _gate(widened, predictions)

    agreeing = copy.deepcopy(rows)
    for row in agreeing:
        row["bootstrap_stratum_id"] = row["circuit_pool_id"]
    assert _gate(agreeing, predictions)["status"] == "not_evaluable"

    missing = copy.deepcopy(rows)
    for row in missing:
        row["bootstrap_stratum_id"] = row.pop("circuit_pool_id")
    with pytest.raises(ValueError):
        _gate(missing, predictions)


@pytest.mark.parametrize("conflict_index", range(8))
def test_one_conflicting_alias_anywhere_in_the_batch_is_refused(conflict_index):
    """The alias check must read every row, not the first one.

    Changing all rows together accepts an implementation that inspects only the
    first, so each case here leaves the alias absent or agreeing everywhere else
    and plants the single disagreement at a different index. The fixture holds
    four TFI rows then four Heisenberg rows, so indices 0, 3, 4 and 7 are family
    endpoints and 1, 2, 5 and 6 are interior: covering every row rejects a check
    that runs only at the boundaries of each family as well as one that runs only
    on the first row overall.
    """
    rows, predictions = _case()
    for row in rows:
        row["circuit_pool_id"] = row["family"]
    rows[conflict_index]["bootstrap_stratum_id"] = "widened"
    with pytest.raises(ValueError, match="bootstrap_stratum_id must match"):
        _gate(rows, predictions)


def test_one_family_may_agree_while_another_conflicts():
    """A per-family alias must not let a disagreement in a sibling family through."""
    rows, predictions = _case()
    for row in rows:
        row["circuit_pool_id"] = row["family"]
        if row["family"] == "tfi":
            row["bootstrap_stratum_id"] = row["circuit_pool_id"]
    assert _gate(rows, predictions)["status"] == "passed"

    conflicting = copy.deepcopy(rows)
    conflicting[-1]["bootstrap_stratum_id"] = "widened"
    with pytest.raises(ValueError, match="bootstrap_stratum_id must match"):
        _gate(conflicting, predictions)


def test_a_later_alias_only_row_still_requires_the_canonical_pool():
    """Dropping the canonical field on one late row must not be excused by its alias."""
    rows, predictions = _case()
    for row in rows:
        row["circuit_pool_id"] = row["family"]
    rows[-1]["bootstrap_stratum_id"] = rows[-1].pop("circuit_pool_id")
    with pytest.raises(ValueError, match="circuit_pool_id"):
        _gate(rows, predictions)


@pytest.mark.parametrize(
    "control_error,margin,defined,ratio,reason",
    [
        # A losing and a tying control must not be excused by the tiny denominator:
        # a weakening that declared every full MAE below 1e-300 to meet the margin
        # would pass if only the winning control were exercised.
        (0.0, False, True, 0.0, None),
        (1e-320, False, True, 1.0, None),
        (0.2, True, False, None, "ratio_overflow"),
    ],
)
def test_a_tiny_positive_full_error_keeps_the_margin_honest_and_stays_json_safe(
    control_error, margin, defined, ratio, reason
):
    """A denominator can be positive and still have no representable ratio.

    The exact-zero branch does not cover 1e-320: dividing by it overflows to
    infinity, which strict JSON serialization rejects. The margin decision is
    separate from the ratio and is asserted here for a losing, a tying, and a
    winning control against the same denominator.
    """
    rows, predictions = _case()
    predictions["ridge"] = dict.fromkeys(predictions["ridge"], 1e-320)
    for control in ("feat-only", "noisy-only"):
        predictions[control] = dict.fromkeys(predictions[control], control_error)
    result = _gate(rows, predictions)
    assert result["status"] == ("passed" if margin else "failed")
    for family in ("tfi", "heisenberg"):
        for control in ("feat-only", "noisy-only"):
            comparison = result["families"][family]["comparisons"][control]
            assert comparison["margin_met"] is margin
            assert comparison["ratio_defined"] is defined
            assert comparison["ratio_undefined_reason"] == reason
            if defined:
                assert comparison["control_over_full_mae"] == pytest.approx(ratio)
            else:
                assert comparison["control_over_full_mae"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape", ["overflowing-difference", "singleton-pools",
                                   "overflowing-macro-sum", "resampled-interval"])
def test_finite_inputs_that_derive_nonfinite_evidence_are_refused(shape):
    """Finite operands are not a finite result, and a broken number is not evidence.

    Each shape reaches a different arithmetic path: the subtraction itself, the
    refusal path that reads a mean before declining, the fsum reduction, and the
    bootstrap's resampled mean. None may return a value, because every one of
    them would attach a decision to a quantity that cannot be represented.
    """
    rows, predictions = _case()
    if shape == "singleton-pools":
        for row in rows:
            row["circuit_pool_id"] = row["circuit_id"]
    if shape == "resampled-interval":
        # Macro means stay finite; only repeated resampling of the large-error
        # circuit overflows the reduction and drives the quantiles to NaN.
        predictions["ridge"] = dict.fromkeys(predictions["ridge"], 0.1)
        for control in ("feat-only", "noisy-only"):
            predictions[control] = {
                row["item_id"]: (8e307 if row["item_id"].endswith("-0") else 0.2)
                for row in rows
            }
    else:
        for row in rows:
            row["ideal_expectation"] = -1e308
        magnitude = 1e308 if shape != "overflowing-macro-sum" else 1e307
        predictions["ridge"] = dict.fromkeys(predictions["ridge"], magnitude)
    with pytest.raises(ValueError):
        _gate(rows, predictions)


def test_item_key_pairing_ignores_input_order_and_accepts_any_full_method():
    rows, predictions = _case()
    # Vary the targets before the snapshot. With constant targets and constant
    # per-method predictions, an implementation that paired predictions to rows
    # positionally rather than by item_id would produce the same errors and pass.
    # Distinct per-row targets make the reversal detect that mispairing: the
    # positional variant scores 0.2 here instead of the known 0.1.
    for index, row in enumerate(rows):
        row["ideal_expectation"] = index / 10
        for method, error in (
            ("ridge", 0.1), ("feat-only", 0.2), ("noisy-only", 0.3)
        ):
            predictions[method][row["item_id"]] = (
                row["ideal_expectation"] + error
            )
    original = copy.deepcopy((rows, predictions))
    predictions["liao"] = predictions.pop("ridge")
    for method in predictions:
        predictions[method] = dict(reversed(list(predictions[method].items())))
    result = _gate(list(reversed(rows)), predictions, full_method="liao")
    assert result["status"] == "passed"
    assert result["families"]["tfi"]["full_mae"] == pytest.approx(0.1)
    assert rows == original[0]
    assert predictions["liao"] == original[1]["ridge"]


@pytest.mark.parametrize("control_error,expected", [(0.0, "failed"), (0.2, "passed")])
def test_zero_full_error_has_explicit_ratio_and_finite_json(control_error, expected):
    rows, predictions = _case()
    predictions["ridge"] = dict.fromkeys(predictions["ridge"], 0.0)
    for control in ("feat-only", "noisy-only"):
        predictions[control] = dict.fromkeys(predictions[control], control_error)
    result = _gate(rows, predictions)
    assert result["status"] == expected
    # The margin decision is independent of the ratio and must be asserted on its
    # own: an implementation that divided instead of cross-multiplying would call
    # a zero-versus-zero tie a met margin, and the overall status would still hide
    # it behind the interval requirement.
    for family in ("tfi", "heisenberg"):
        for control in ("feat-only", "noisy-only"):
            comparison = result["families"][family]["comparisons"][control]
            assert comparison["ratio_defined"] is False
            assert comparison["control_over_full_mae"] is None
            assert comparison["ratio_undefined_reason"] == "zero_full_mae"
            assert comparison["margin_met"] is (control_error > 0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"ratio_margin": 1.0}, {"ratio_margin": float("nan")},
                                   {"confidence": 1.0}, {"n_resamples": 0},
                                   {"seed": -1}, {"full_method": "feat-only"}])
def test_invalid_gate_contract_rejected(kwargs):
    rows, predictions = _case()
    arguments = dict(seed=13, n_resamples=200)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_incremental_value(rows, predictions, **arguments)
