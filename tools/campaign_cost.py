"""Estimate campaign wall time from aggregate throughput measurements."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.budget import (
    CampaignBudget,
    Method,
    PAPER_TIER_MODE,
    ROSTER,
    campaign_budget,
)
from qem_bench.runner.run import validate_run_artifact


_RUNNER_METHOD_NAMES = {
    Method.RAW: "raw",
    Method.RIDGE: "ridge",
    Method.LEARNED_REGRESSORS: "learned-regressors",
    Method.LOCAL_DIGITAL_ZNE: "zne",
    Method.CDR: "cdr",
    Method.VNCDR: "vncdr",
    Method.LIAO: "liao",
}
_ROSTER_METHOD_BY_RUNNER_NAME = {
    runner_name: method for method, runner_name in _RUNNER_METHOD_NAMES.items()
}


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


def _realized_evaluations(
    runs: Sequence[Mapping[str, object]], reservation: CampaignBudget
) -> int:
    expected_methods = set(reservation.methods)
    represented_methods: set[Method] = set()
    cells: dict[str, Mapping[str, object]] = {}
    coverage: set[tuple[str, Method]] = set()
    realized_evaluations = 0

    for run in runs:
        if run["dataset_schema_version"] != "split-v2":
            raise ValueError(
                "realized-cost input requires split-v2 runner artifacts with "
                "validated budget metadata"
            )
        methods = run["methods"]
        runner_names = set(methods) & set(_ROSTER_METHOD_BY_RUNNER_NAME)
        represented_methods.update(
            _ROSTER_METHOD_BY_RUNNER_NAME[name] for name in runner_names
        )
        for name in runner_names:
            realized_evaluations += methods[name]["ledger"]["total"]

        for budget_cell_id, cell in run["budget"].items():
            descriptor = cell["pairing"]["descriptor"]
            if budget_cell_id in cells and cells[budget_cell_id] != descriptor:
                raise ValueError(
                    f"runner artifacts disagree on paired budget cell "
                    f"{budget_cell_id!r}"
                )
            cells[budget_cell_id] = descriptor
            if cell["tier"] != reservation.tier:
                raise ValueError(
                    f"runner artifact tier {cell['tier']!r} does not match "
                    f"reservation tier {reservation.tier!r}"
                )
            if descriptor["n_qubits"] != reservation.width:
                raise ValueError(
                    f"runner artifact width {descriptor['n_qubits']} does not match "
                    f"reservation width {reservation.width}"
                )
            for name in runner_names:
                method = _ROSTER_METHOD_BY_RUNNER_NAME[name]
                record = cell["methods"][name]
                if record["budget_method"] != method.value:
                    raise ValueError(
                        f"runner method {name!r} does not identify budget method "
                        f"{method.value!r}"
                    )
                key = (budget_cell_id, method)
                if key in coverage:
                    raise ValueError(
                        f"runner artifacts duplicate realized ledger coverage for "
                        f"{method.value!r} in paired budget cell {budget_cell_id!r}"
                    )
                coverage.add(key)

    if represented_methods != expected_methods:
        expected = sorted(method.value for method in expected_methods)
        actual = sorted(method.value for method in represented_methods)
        raise ValueError(
            "runner artifact roster does not match reservation methods: "
            f"expected={expected}, actual={actual}"
        )
    if len(cells) != reservation.paired_budget_cell_count:
        raise ValueError(
            f"runner artifacts contain {len(cells)} paired budget cells, but the "
            f"reservation declares {reservation.paired_budget_cell_count}"
        )
    expected_coverage = {
        (budget_cell_id, method)
        for budget_cell_id in cells
        for method in expected_methods
    }
    if coverage != expected_coverage:
        raise ValueError(
            "runner artifacts must cover every reserved method and paired budget "
            "cell exactly once"
        )
    return realized_evaluations


def _load_validated_runs(
    manifest: Mapping[str, object], base_dir: Path
) -> list[dict]:
    if manifest.get("schema_version") != "qem-bench-report-manifest-v1":
        raise ValueError("unsupported results manifest schema_version")
    entries = manifest.get("runs")
    if not isinstance(entries, list):
        raise ValueError("results manifest runs must be an array")
    resolved_base = base_dir.resolve()
    runs = []
    artifact_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("results manifest run entries must be objects")
        declared = entry.get("results")
        if not isinstance(declared, str) or not declared:
            raise ValueError("results manifest run results must be nonempty paths")
        relative_path = Path(declared)
        if relative_path.is_absolute() or relative_path.drive:
            raise ValueError("results manifest run paths must be relative")
        display_path = relative_path.as_posix()
        try:
            resolved_path = (resolved_base / relative_path).resolve()
            resolved_path.relative_to(resolved_base)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"results manifest run path escapes its directory: {display_path}"
            ) from exc
        try:
            payload = resolved_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"invalid runner result at {display_path}: unable to read result"
            ) from exc
        try:
            run = validate_run_artifact(json.loads(payload))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(f"invalid runner result at {display_path}: {exc}") from exc
        artifact_id = str(run["artifact_id"])
        if artifact_id in artifact_ids:
            raise ValueError(
                f"results manifest contains duplicate run artifact_id {artifact_id!r}"
            )
        artifact_ids.add(artifact_id)
        runs.append(run)
    if not runs:
        raise ValueError("results manifest contains no runs")
    return runs


def realized_evaluations_from_manifest(
    results_manifest: str | Path, reservation: CampaignBudget
) -> int:
    """Return validated realized cost for one matching campaign reservation."""

    manifest_path = Path(results_manifest)
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except OSError as exc:
        raise ValueError(f"unable to read results manifest {manifest_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"results manifest {manifest_path} is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("results manifest must be an object")
    runs = _load_validated_runs(manifest, manifest_path.resolve().parent)
    return _realized_evaluations(runs, reservation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a reserved combined-cap campaign allocation and convert it to "
            "wall-hours using aggregate throughput measured for the full worker "
            "allocation. Optionally report validated realized runner-ledger cost "
            "separately."
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
    parser.add_argument(
        "--results-manifest",
        type=Path,
        help=(
            "optional qem-bench-report-manifest-v1 whose validated split-v2 runner "
            "ledgers exactly cover the reserved roster and paired budget cells"
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
        realized_evaluations = (
            None
            if args.results_manifest is None
            else realized_evaluations_from_manifest(args.results_manifest, budget)
        )
    except ValueError as exc:
        parser.error(str(exc))
    rate = calibrations[args.width]
    reserved_evaluations = budget.total_circuit_evaluations
    reserved_wall_hours = estimate_wall_hours(reserved_evaluations, rate)
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
    print(f"reserved total circuit evaluations: {reserved_evaluations:,}")
    print(
        f"calibration: n={budget.width}, {rate:,.12g} shots/second "
        "(aggregate across the full worker allocation)"
    )
    print("reported reserved quantity: estimated reserved-allocation wall-hours")
    print(
        "parallelism assumption: the calibration includes the full worker allocation; "
        "the estimate applies no additional parallel speedup"
    )
    print(f"estimated reserved-allocation wall-hours: {reserved_wall_hours:,.3f}")
    if realized_evaluations is not None:
        realized_wall_hours = estimate_wall_hours(realized_evaluations, rate)
        print(f"realized circuit evaluations: {realized_evaluations:,}")
        print(f"estimated realized wall-hours: {realized_wall_hours:,.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
