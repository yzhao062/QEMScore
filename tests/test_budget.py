"""Closed-form circuit-evaluation budget tests."""

import pytest

from qem_bench.budget import (
    PAPER_TIER_MODE,
    ROSTER,
    TIERS,
    Budget,
    BudgetInputs,
    Method,
    TierMode,
    TierPair,
    campaign_budget,
    check_constraints,
    feasibility_report,
    method_budget,
    minimum_tier_constant,
)
from tools.campaign_cost import estimate_core_hours, main as campaign_cost_main


_T1_INPUTS = {
    Method.RAW: BudgetInputs(0, 100, 8_192, 1, 0),
    Method.RIDGE: BudgetInputs(100, 100, 8_192, 1, 0),
    Method.LEARNED_REGRESSORS: BudgetInputs(100, 100, 8_192, 1, 0),
    Method.LOCAL_DIGITAL_ZNE: BudgetInputs(0, 100, 8_192, 3, 0),
    Method.CDR: BudgetInputs(0, 100, 8_192, 1, 70),
    Method.VNCDR: BudgetInputs(0, 100, 8_192, 3, 100),
    Method.LIAO: BudgetInputs(100, 100, 8_192, 1, 0),
}

_T1_MINIMUMS = {
    Method.RAW: (819_200, 819_200),
    Method.RIDGE: (819_200, 1_638_400),
    Method.LEARNED_REGRESSORS: (819_200, 1_638_400),
    Method.LOCAL_DIGITAL_ZNE: (2_457_600, 2_457_600),
    Method.CDR: (57_344_000, 58_163_200),
    Method.VNCDR: (245_760_000, 248_217_600),
    Method.LIAO: (819_200, 1_638_400),
}


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


def test_rebased_declared_tiers_are_symmetric_and_use_combined_paper_mode():
    assert TIERS == {"L": 2_500_000, "M": 25_000_000, "H": 250_000_000}
    assert PAPER_TIER_MODE is TierMode.COMBINED_CAP
    for name, cap in TIERS.items():
        assert TierPair.declared(name) == TierPair(cap, cap, cap)


@pytest.mark.parametrize(
    ("ladder", "expected"),
    [
        (
            {"L": 10_000, "M": 100_000, "H": 1_000_000},
            {"L": set(), "M": set(), "H": {Method.RAW}},
        ),
        (
            TIERS,
            {
                "L": {
                    Method.RAW,
                    Method.RIDGE,
                    Method.LEARNED_REGRESSORS,
                    Method.LOCAL_DIGITAL_ZNE,
                    Method.LIAO,
                },
                "M": {
                    Method.RAW,
                    Method.RIDGE,
                    Method.LEARNED_REGRESSORS,
                    Method.LOCAL_DIGITAL_ZNE,
                    Method.LIAO,
                },
                "H": set(Method),
            },
        ),
    ],
)
def test_t1_combined_cap_feasibility_under_old_and_new_ladders(ladder, expected):
    for tier, cap in ladder.items():
        funded = {
            method
            for method, inputs in _T1_INPUTS.items()
            if feasibility_report(TierPair.from_constant(cap), [method], inputs)[
                0
            ].combined_cap.feasible
        }
        assert funded == expected[tier]


def test_t1_minimum_symmetric_tier_constants_are_computed_for_both_readings():
    for method, inputs in _T1_INPUTS.items():
        requirement = minimum_tier_constant(method, inputs)
        split, combined = _T1_MINIMUMS[method]
        assert requirement.split_joint == split
        assert requirement.combined_joint == combined


def test_t1_declared_new_ladder_has_no_reading_flips():
    for cap in TIERS.values():
        for method, inputs in _T1_INPUTS.items():
            row = feasibility_report(TierPair.from_constant(cap), [method], inputs)[0]
            assert not row.flips


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
    old_row = feasibility_report(TierPair.from_constant(1_000_000), [Method.CDR], inputs)[
        0
    ]
    assert not old_row.split_caps.feasible
    assert not old_row.combined_cap.feasible
    new_row = feasibility_report(TierPair.declared("H"), [Method.CDR], inputs)[0]
    assert new_row.split_caps.feasible
    assert new_row.combined_cap.feasible


def test_method_and_statistical_constraints_are_checked_together():
    inputs = BudgetInputs(0, 20, 2_048, 1, 0)
    row = feasibility_report(
        TierPair.from_constant(1_000_000),
        [Method.RAW],
        inputs,
        required_cells=145,
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


def test_campaign_budget_names_the_runner_counting_unit():
    budget = campaign_budget("h", 12, 100, [Method.RAW, Method.VNCDR])
    assert budget.tier == "H"
    assert budget.width == 12
    assert budget.paired_budget_cell_count == 100
    assert budget.methods == (Method.RAW, Method.VNCDR)
    assert budget.per_method_paired_budget_cell_cap == 250_000_000
    assert budget.total_circuit_evaluations == 50_000_000_000

    two_paired_budget_cells = campaign_budget("L", 12, 2, [Method.RAW])
    assert two_paired_budget_cells.per_method_paired_budget_cell_cap == 2_500_000
    assert two_paired_budget_cells.total_circuit_evaluations == 5_000_000


@pytest.mark.parametrize(
    ("argument", "value", "problem"),
    [
        ("width", True, "a non-boolean integer"),
        ("width", 0, "positive"),
        ("paired_budget_cell_count", 1.5, "an integer"),
        ("paired_budget_cell_count", 0, "positive"),
    ],
)
def test_campaign_counts_are_positive_non_boolean_integers(argument, value, problem):
    kwargs = {"tier": "H", "width": 10, "paired_budget_cell_count": 100}
    kwargs[argument] = value
    with pytest.raises(ValueError, match=rf"{argument} must be {problem}"):
        campaign_budget(**kwargs)


@pytest.mark.parametrize("methods", [[], "raw", [Method.RAW, "raw"]])
def test_campaign_method_set_must_be_nonempty_iterable_without_duplicates(methods):
    with pytest.raises(ValueError, match="methods must"):
        campaign_budget("L", 8, 1, methods)


def test_supplied_calibrations_reproduce_width_paired_budget_cell_tradeoff():
    at_12 = campaign_budget("H", 12, 100, [Method.RAW])
    at_10 = campaign_budget("H", 10, 1_370, [Method.RAW])
    hours_12 = estimate_core_hours(at_12.total_circuit_evaluations, 2_082)
    hours_10 = estimate_core_hours(at_10.total_circuit_evaluations, 28_567)
    assert hours_10 == pytest.approx(hours_12, rel=0.002)


def test_campaign_cost_cli_reports_supplied_calibration(capsys):
    assert (
        campaign_cost_main(
            [
                "--tier",
                "L",
                "--width",
                "10",
                "--paired-budget-cells",
                "2",
                "--methods",
                "raw",
                "vncdr",
                "--calibration",
                "10=28567",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "tier: L (combined-cap)" in output
    assert (
        "paired budget cells (paired-budget-cell-v1 descriptors): 2" in output
    )
    assert "total circuit evaluations: 10,000,000" in output
    assert "calibration: n=10, 28,567 shots/second" in output


def test_9f07300_public_calls_remain_compatible():
    assert Method("raw") is Method.RAW
    inputs = BudgetInputs(1, 1, 10, 1, 0)
    budget = method_budget(Method.RAW, inputs)
    mode = TierMode("combined-cap")
    pair = TierPair.from_constant(TIERS["L"])
    assert check_constraints(Method.RAW, inputs, pair, mode).feasible
    assert feasibility_report(pair, [Method.RAW], inputs)[0].per_cell == budget
    assert minimum_tier_constant(Method.RAW, inputs).combined_joint == budget.total
