"""Estimate campaign wall time from aggregate throughput measurements."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.budget import Method, PAPER_TIER_MODE, ROSTER, campaign_budget


def _calibration(value: str) -> tuple[int, float]:
    width_text, separator, rate_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "calibration must have the form WIDTH=AGGREGATE_SHOTS_PER_SECOND"
        )
    try:
        width = int(width_text)
        rate = float(rate_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "calibration must have the form WIDTH=AGGREGATE_SHOTS_PER_SECOND"
        ) from exc
    if width <= 0:
        raise argparse.ArgumentTypeError("calibration width must be positive")
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError(
            "calibration aggregate shots per second must be finite and positive"
        )
    return width, rate


def _methods(values: Sequence[str]) -> tuple[Method, ...]:
    tokens = [token.strip().lower() for value in values for token in value.split(",")]
    if tokens == ["all"]:
        return ROSTER
    if not tokens or any(not token or token == "all" for token in tokens):
        raise ValueError("methods must be 'all' or a nonempty list of method names")
    aliases = {method.name.lower(): method for method in Method}
    aliases.update({method.value: method for method in Method})
    try:
        methods = tuple(aliases[token] for token in tokens)
    except KeyError as exc:
        choices = ", ".join(method.value for method in Method)
        raise ValueError(f"unknown method {exc.args[0]!r}; choose from {choices}") from exc
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    return methods


def estimate_wall_hours(
    total_circuit_evaluations: int,
    aggregate_shots_per_second: float,
) -> float:
    """Convert circuit evaluations to wall-hours at fixed aggregate throughput."""

    if isinstance(total_circuit_evaluations, bool) or not isinstance(
        total_circuit_evaluations, int
    ):
        raise ValueError("total_circuit_evaluations must be an integer")
    if total_circuit_evaluations < 0:
        raise ValueError("total_circuit_evaluations must be nonnegative")
    if (
        not math.isfinite(aggregate_shots_per_second)
        or aggregate_shots_per_second <= 0
    ):
        raise ValueError("aggregate_shots_per_second must be finite and positive")
    return total_circuit_evaluations / aggregate_shots_per_second / 3_600


def estimate_core_hours(
    total_circuit_evaluations: int,
    per_core_shots_per_second: float,
) -> float:
    """Convert circuit evaluations to core-hours using a per-core throughput."""

    if not math.isfinite(per_core_shots_per_second) or per_core_shots_per_second <= 0:
        raise ValueError("per_core_shots_per_second must be finite and positive")
    return estimate_wall_hours(total_circuit_evaluations, per_core_shots_per_second)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a combined-cap campaign allocation and convert it to wall-hours "
            "using aggregate throughput measured for the full worker allocation."
        )
    )
    parser.add_argument("--tier", required=True, help="declared tier: L, M, or H")
    parser.add_argument("--width", required=True, type=int, help="circuit width in qubits")
    parser.add_argument(
        "--paired-budget-cells",
        required=True,
        type=int,
        help="number of paired-budget-cell-v1 descriptors",
    )
    parser.add_argument(
        "--methods",
        required=True,
        nargs="+",
        metavar="METHOD",
        help="method names, comma-separated names, or 'all'",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        action="append",
        type=_calibration,
        metavar="WIDTH=AGGREGATE_SHOTS_PER_SECOND",
        help=(
            "aggregate throughput of the full execution configuration; repeat to "
            "supply multiple widths"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    calibrations: dict[int, float] = {}
    for width, rate in args.calibration:
        if width in calibrations and calibrations[width] != rate:
            parser.error(f"conflicting throughput calibrations for width {width}")
        calibrations[width] = rate
    if args.width not in calibrations:
        parser.error(f"no throughput calibration supplied for width {args.width}")
    try:
        methods = _methods(args.methods)
        budget = campaign_budget(
            args.tier, args.width, args.paired_budget_cells, methods
        )
    except ValueError as exc:
        parser.error(str(exc))
    rate = calibrations[args.width]
    wall_hours = estimate_wall_hours(budget.total_circuit_evaluations, rate)
    print(f"tier: {budget.tier} ({PAPER_TIER_MODE.value})")
    print(f"width: {budget.width}")
    print(
        "paired budget cells (paired-budget-cell-v1 descriptors): "
        f"{budget.paired_budget_cell_count:,}"
    )
    print(f"methods: {', '.join(method.value for method in budget.methods)}")
    print(
        "combined cap per method and paired-budget-cell-v1 descriptor: "
        f"{budget.per_method_paired_budget_cell_cap:,}"
    )
    print(f"total circuit evaluations: {budget.total_circuit_evaluations:,}")
    print(
        f"calibration: n={budget.width}, {rate:,.12g} shots/second "
        "(aggregate across the full worker allocation)"
    )
    print("reported quantity: estimated wall-hours")
    print(
        "parallelism assumption: the calibration includes the full worker allocation; "
        "the estimate applies no additional parallel speedup"
    )
    print(f"estimated wall-hours: {wall_hours:,.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
