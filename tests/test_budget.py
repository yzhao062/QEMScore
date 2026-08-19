"""Closed-form circuit-evaluation budget tests."""

import pytest

from qem_bench.budget import (
    ROSTER,
    Budget,
    BudgetInputs,
    Method,
    TierMode,
    TierPair,
    check_constraints,
    feasibility_report,
    method_budget,
    minimum_tier_constant,
)


def _buckets(method, inputs):
    budget = method_budget(method, inputs)
    return budget.B_train, budget.B_extra, budget.B_pred, budget.total


def test_declared_roster_formulas():
    inputs = BudgetInputs(10, 4, 100, 3, 7, groups=2)
    expected = {
        Method.RAW: (0, 0, 800, 800),
        Method.RIDGE: (2_000, 0, 800, 2_800),
        Method.LEARNED_REGRESSORS: (2_000, 0, 800, 2_800),
        Method.LOCAL_DIGITAL_ZNE: (0, 1_600, 800, 2_400),
        Method.CDR: (5_600, 0, 800, 6_400),
        Method.VNCDR: (16_800, 1_600, 800, 19_200),
        Method.LIAO: (2_000, 0, 800, 2_800),
    }
    assert set(expected) == set(ROSTER)
    for method, buckets in expected.items():
        assert _buckets(method, inputs) == buckets


@pytest.mark.parametrize(
    ("method", "inputs", "expected"),
    [
        (Method.CDR, BudgetInputs(0, 1, 8_192, 1, 70), 581_632),
        (Method.VNCDR, BudgetInputs(0, 1, 8_192, 3, 100), 2_482_176),
        (Method.CDR, BudgetInputs(0, 32, 2_048, 1, 10), 720_896),
        (Method.VNCDR, BudgetInputs(0, 32, 2_048, 3, 10), 2_162_688),
        (Method.CDR, BudgetInputs(0, 100, 8_192, 1, 70), 58_163_200),
        (Method.VNCDR, BudgetInputs(0, 100, 8_192, 3, 100), 248_217_600),
    ],
)
def test_dev_plan_section_3_1_values(method, inputs, expected):
    assert method_budget(method, inputs).total == expected


def test_tier_modes_differ_and_report_the_flip():
    inputs = BudgetInputs(70, 70, 100, 3, 10)
    tier = TierPair.from_constant(10_000)
    split = check_constraints(Method.RIDGE, inputs, tier, TierMode.SPLIT_CAPS)
    combined = check_constraints(Method.RIDGE, inputs, tier, TierMode.COMBINED_CAP)
    assert split.feasible
    assert not combined.feasible

    row = feasibility_report(tier, [Method.RIDGE], inputs)[0]
    assert row.flips
    assert row.split_caps.status == "feasible"
    assert row.combined_cap.status == "budget-infeasible"


def test_amortized_cost_is_reporting_only_and_never_binds():
    inputs = BudgetInputs(0, 100, 8_192, 1, 70)
    budget = method_budget(Method.CDR, inputs)
    assert budget.amortized_per_test.total == 581_632
    assert budget.amortized_per_test.total <= 1_000_000
    row = feasibility_report(TierPair.declared("H"), [Method.CDR], inputs)[0]
    assert not row.split_caps.feasible
    assert not row.combined_cap.feasible


def test_method_and_statistical_constraints_are_checked_together():
    inputs = BudgetInputs(0, 20, 2_048, 1, 0)
    row = feasibility_report(
        TierPair.declared("H"), [Method.RAW], inputs, required_cells=145
    )[0]
    assert row.statistical_base_evals == 5_939_200
    assert not row.split_caps.statistical_feasible
    assert not row.combined_cap.statistical_feasible
    assert row.split_caps.status == "budget-infeasible"

    required = minimum_tier_constant(Method.RAW, inputs, required_cells=145)
    assert required.split_joint == 5_939_200
    assert required.combined_joint == 5_939_200


@pytest.mark.parametrize(
    "field",
    [
        "n_train_circuits",
        "n_test_circuits",
        "shots",
        "n_scales",
        "m",
        "groups",
    ],
)
@pytest.mark.parametrize(
    ("value", "problem"),
    [(True, "a non-boolean integer"), (1.5, "an integer")],
)
def test_budget_count_fields_require_non_boolean_integers(field, value, problem):
    inputs = BudgetInputs(1, 1, 10, 1, 1)._replace(**{field: value})
    with pytest.raises(ValueError, match=rf"{field} must be {problem}"):
        method_budget(Method.RIDGE, inputs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_budget_count_fields_reject_nonfinite_values(value):
    inputs = BudgetInputs(value, 1, 10, 1, 1)
    with pytest.raises(ValueError, match="n_train_circuits must be finite"):
        method_budget(Method.RIDGE, inputs)


@pytest.mark.parametrize(
    ("field", "value", "bound"),
    [
        ("n_train_circuits", -1, "nonnegative"),
        ("n_test_circuits", 0, "positive"),
        ("shots", 0, "positive"),
        ("n_scales", 0, "positive"),
        ("m", -1, "nonnegative"),
        ("groups", 0, "positive"),
    ],
)
def test_budget_count_fields_enforce_bounds(field, value, bound):
    inputs = BudgetInputs(1, 1, 10, 1, 1)._replace(**{field: value})
    with pytest.raises(ValueError, match=rf"{field} must be {bound}"):
        method_budget(Method.RAW, inputs)


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        (True, "a non-boolean integer"),
        (1.5, "an integer"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (0, "positive"),
        (-1, "positive"),
    ],
)
@pytest.mark.parametrize(
    "check",
    [
        lambda inputs, tier, value: check_constraints(
            Method.RAW, inputs, tier, TierMode.SPLIT_CAPS, required_cells=value
        ),
        lambda inputs, tier, value: feasibility_report(
            tier, [Method.RAW], inputs, required_cells=value
        ),
        lambda inputs, tier, value: minimum_tier_constant(
            Method.RAW, inputs, required_cells=value
        ),
    ],
)
def test_required_cells_is_a_positive_non_boolean_integer(value, problem, check):
    with pytest.raises(ValueError, match=rf"required_cells must be {problem}"):
        check(BudgetInputs(1, 1, 10, 1, 1), TierPair.from_constant(100), value)


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        (True, "a non-boolean integer"),
        (1.5, "an integer"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-1, "nonnegative"),
    ],
)
def test_constant_tier_cap_is_a_nonnegative_non_boolean_integer(value, problem):
    with pytest.raises(ValueError, match=rf"cap must be {problem}"):
        TierPair.from_constant(value)


@pytest.mark.parametrize("field", ["training_cap", "test_cap", "combined_cap"])
def test_direct_tier_pair_caps_are_validated_before_constraints(field):
    tier = TierPair(100, 100, 100)._replace(**{field: -1})
    with pytest.raises(ValueError, match=rf"{field} must be nonnegative"):
        check_constraints(
            Method.RAW,
            BudgetInputs(1, 1, 10, 1, 1),
            tier,
            TierMode.SPLIT_CAPS,
        )


@pytest.mark.parametrize(
    ("value", "problem"),
    [(True, "a non-boolean integer"), (1.5, "an integer"), (0, "positive")],
)
def test_budget_scaling_factor_is_a_positive_non_boolean_integer(value, problem):
    with pytest.raises(ValueError, match=rf"scale factor must be {problem}"):
        Budget(1, 1, 1, 1).scaled(value)
