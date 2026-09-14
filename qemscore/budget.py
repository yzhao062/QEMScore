"""Closed-form circuit-evaluation budgets for the declared method roster.

Inputs describe one paired statistical cell. ``groups`` is the number of
physical measurement groups per circuit. Sibling observable rows sharing one
counts draw do not increase it. Exact simulator labels are excluded.
"""

import math
from enum import Enum
from typing import Iterable, NamedTuple


TIERS = {"L": 2_500_000, "M": 25_000_000, "H": 250_000_000}


class Method(str, Enum):
    RAW = "raw"
    RIDGE = "ridge"
    LEARNED_REGRESSORS = "learned-regressors"
    LOCAL_DIGITAL_ZNE = "local-digital-zne"
    CDR = "cdr"
    VNCDR = "vncdr"
    LIAO = "liao-style"


ROSTER = tuple(Method)


class TierMode(str, Enum):
    SPLIT_CAPS = "split-caps"
    COMBINED_CAP = "combined-cap"


PAPER_TIER_MODE = TierMode.COMBINED_CAP


class BudgetInputs(NamedTuple):
    n_train_circuits: int
    n_test_circuits: int
    shots: int
    n_scales: int
    m: int
    groups: int = 1


class AmortizedBudget(NamedTuple):
    """Reporting-only values per test circuit."""

    B_train: float
    B_extra: float
    B_pred: float

    @property
    def total(self) -> float:
        return sum(self)


class Budget(NamedTuple):
    """Binding split totals and their reporting-only projection."""

    B_train: int
    B_extra: int
    B_pred: int
    n_test_circuits: int

    @property
    def total(self) -> int:
        return self.B_train + self.B_extra + self.B_pred

    @property
    def amortized_per_test(self) -> AmortizedBudget:
        n = self.n_test_circuits
        return AmortizedBudget(self.B_train / n, self.B_extra / n, self.B_pred / n)

    def scaled(self, factor: int) -> "Budget":
        _validate_integer("scale factor", factor, minimum=1)
        return Budget(
            self.B_train * factor,
            self.B_extra * factor,
            self.B_pred * factor,
            self.n_test_circuits * factor,
        )


class TierPair(NamedTuple):
    """Explicit caps for both supported readings of a tier pair."""

    training_cap: int
    test_cap: int
    combined_cap: int

    @classmethod
    def from_constant(cls, cap: int) -> "TierPair":
        _validate_integer("cap", cap, minimum=0)
        return cls(cap, cap, cap)

    @classmethod
    def declared(cls, name: str) -> "TierPair":
        try:
            return cls.from_constant(TIERS[name.upper()])
        except KeyError as exc:
            raise ValueError(f"unknown tier {name!r}") from exc


class ConstraintCheck(NamedTuple):
    mode: TierMode
    method_feasible: bool
    statistical_feasible: bool

    @property
    def feasible(self) -> bool:
        return self.method_feasible and self.statistical_feasible

    @property
    def status(self) -> str:
        return "feasible" if self.feasible else "budget-infeasible"


class FeasibilityRow(NamedTuple):
    method: Method
    per_cell: Budget
    required: Budget
    required_cells: int
    statistical_base_evals: int
    split_caps: ConstraintCheck
    combined_cap: ConstraintCheck

    @property
    def flips(self) -> bool:
        return self.split_caps.feasible != self.combined_cap.feasible


class TierRequirement(NamedTuple):
    """Smallest symmetric tier constant H under each reading."""

    split_method: int
    combined_method: int
    statistical_base: int
    split_joint: int
    combined_joint: int


class CampaignBudget(NamedTuple):
    """Combined-cap allocation over ``paired-budget-cell-v1`` descriptors."""

    tier: str
    width: int
    paired_budget_cell_count: int
    methods: tuple[Method, ...]
    per_method_paired_budget_cell_cap: int
    total_circuit_evaluations: int


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-boolean integer")
    if isinstance(value, int):
        if value < minimum:
            bound = "nonnegative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be {bound}")
        return
    try:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    except TypeError:
        pass
    raise ValueError(f"{name} must be an integer")


def _validate(inputs: BudgetInputs) -> None:
    _validate_integer("n_train_circuits", inputs.n_train_circuits, minimum=0)
    _validate_integer("n_test_circuits", inputs.n_test_circuits, minimum=1)
    _validate_integer("shots", inputs.shots, minimum=1)
    _validate_integer("n_scales", inputs.n_scales, minimum=1)
    _validate_integer("m", inputs.m, minimum=0)
    _validate_integer("groups", inputs.groups, minimum=1)


def _validate_tier_pair(tier_pair: TierPair) -> None:
    _validate_integer("training_cap", tier_pair.training_cap, minimum=0)
    _validate_integer("test_cap", tier_pair.test_cap, minimum=0)
    _validate_integer("combined_cap", tier_pair.combined_cap, minimum=0)


def _validate_required_cells(required_cells: int) -> None:
    _validate_integer("required_cells", required_cells, minimum=1)


def method_budget(method: Method | str, inputs: BudgetInputs) -> Budget:
    """Return binding split totals for one cell."""

    _validate(inputs)
    method = Method(method)
    train = inputs.n_train_circuits * inputs.groups * inputs.shots
    test = inputs.n_test_circuits * inputs.groups * inputs.shots
    k = inputs.n_scales
    targets = inputs.n_test_circuits * inputs.groups
    if method is Method.RAW:
        buckets = 0, 0, test
    elif method in (Method.RIDGE, Method.LEARNED_REGRESSORS, Method.LIAO):
        buckets = train, 0, test
    elif method is Method.LOCAL_DIGITAL_ZNE:
        buckets = 0, (k - 1) * test, test
    elif method is Method.CDR:
        buckets = targets * inputs.m * inputs.shots, 0, test
    else:
        buckets = (
            targets * inputs.m * k * inputs.shots,
            targets * (k - 1) * inputs.shots,
            test,
        )
    return Budget(*buckets, inputs.n_test_circuits)


def check_constraints(
    method: Method | str,
    inputs: BudgetInputs,
    tier_pair: TierPair,
    mode: TierMode | str,
    *,
    required_cells: int = 1,
) -> ConstraintCheck:
    """Check method and required-cell base costs using binding totals only."""

    _validate(inputs)
    _validate_tier_pair(tier_pair)
    _validate_required_cells(required_cells)
    mode = TierMode(mode)
    required = method_budget(method, inputs).scaled(required_cells)
    statistical = required_cells * inputs.n_test_circuits * inputs.groups * inputs.shots
    if mode is TierMode.SPLIT_CAPS:
        method_ok = required.B_train <= tier_pair.training_cap and (
            required.B_extra + required.B_pred <= tier_pair.test_cap
        )
        statistical_ok = statistical <= tier_pair.test_cap
    else:
        method_ok = required.total <= tier_pair.combined_cap
        statistical_ok = statistical <= tier_pair.combined_cap
    return ConstraintCheck(mode, method_ok, statistical_ok)


def feasibility_report(
    tier_pair: TierPair,
    roster: Iterable[Method | str],
    inputs: BudgetInputs,
    *,
    required_cells: int = 1,
) -> tuple[FeasibilityRow, ...]:
    """Report both tier readings side by side for each requested method."""

    _validate(inputs)
    _validate_tier_pair(tier_pair)
    _validate_required_cells(required_cells)
    statistical = required_cells * inputs.n_test_circuits * inputs.groups * inputs.shots
    rows = []
    for value in roster:
        method = Method(value)
        per_cell = method_budget(method, inputs)
        checks = [
            check_constraints(method, inputs, tier_pair, mode, required_cells=required_cells)
            for mode in TierMode
        ]
        rows.append(
            FeasibilityRow(
                method,
                per_cell,
                per_cell.scaled(required_cells),
                required_cells,
                statistical,
                *checks,
            )
        )
    return tuple(rows)


def minimum_tier_constant(
    method: Method | str,
    inputs: BudgetInputs,
    *,
    required_cells: int = 1,
) -> TierRequirement:
    """Invert feasibility for symmetric split caps H and one total cap H."""

    _validate_required_cells(required_cells)
    required = method_budget(method, inputs).scaled(required_cells)
    statistical = required_cells * inputs.n_test_circuits * inputs.groups * inputs.shots
    split_method = max(required.B_train, required.B_extra + required.B_pred)
    combined_method = required.total
    return TierRequirement(
        split_method,
        combined_method,
        statistical,
        max(split_method, statistical),
        max(combined_method, statistical),
    )


def campaign_budget(
    tier: str,
    width: int,
    paired_budget_cell_count: int,
    methods: Iterable[Method | str] = ROSTER,
) -> CampaignBudget:
    """Allocate the tier cap by method and ``paired-budget-cell-v1`` descriptor.

    The paper allocates the combined tier cap to every selected method in every
    paired budget cell. ``paired_budget_cell_count`` is the number of runner
    ``paired-budget-cell-v1`` descriptors, not the number of six-part report
    cells. The tier is not a campaign-total cap. Width is recorded explicitly
    but does not enter the circuit-evaluation formulas. Runtime conversion is
    machine-specific and belongs outside this module.
    """

    _validate_integer("width", width, minimum=1)
    _validate_integer(
        "paired_budget_cell_count", paired_budget_cell_count, minimum=1
    )
    try:
        tier_name = tier.upper()
    except AttributeError as exc:
        raise ValueError("tier must be a string") from exc
    tier_pair = TierPair.declared(tier_name)
    if isinstance(methods, (str, bytes)):
        raise ValueError("methods must be a nonempty iterable of methods")
    try:
        normalized_methods = tuple(Method(value) for value in methods)
    except TypeError as exc:
        raise ValueError("methods must be a nonempty iterable of methods") from exc
    if not normalized_methods:
        raise ValueError("methods must be a nonempty iterable of methods")
    if len(set(normalized_methods)) != len(normalized_methods):
        raise ValueError("methods must not contain duplicates")
    total = tier_pair.combined_cap * paired_budget_cell_count * len(normalized_methods)
    return CampaignBudget(
        tier_name,
        width,
        paired_budget_cell_count,
        normalized_methods,
        tier_pair.combined_cap,
        total,
    )
