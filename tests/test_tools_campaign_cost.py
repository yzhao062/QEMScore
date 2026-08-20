import copy
import json

import pytest

from qem_bench.budget import (
    ROSTER,
    BudgetInputs,
    Method,
    campaign_budget,
    method_budget,
)
from qem_bench.datasets.generate import generate
from qem_bench.runner.run import run
from tools.campaign_cost import estimate_core_hours, estimate_wall_hours, main


_T1_INPUTS = {
    Method.RAW: BudgetInputs(0, 100, 8_192, 1, 0),
    Method.RIDGE: BudgetInputs(100, 100, 8_192, 1, 0),
    Method.LEARNED_REGRESSORS: BudgetInputs(100, 100, 8_192, 1, 0),
    Method.LOCAL_DIGITAL_ZNE: BudgetInputs(0, 100, 8_192, 3, 0),
    Method.CDR: BudgetInputs(0, 100, 8_192, 1, 70),
    Method.VNCDR: BudgetInputs(0, 100, 8_192, 3, 100),
    Method.LIAO: BudgetInputs(100, 100, 8_192, 1, 0),
}
_RUNNER_ROSTER = (
    (Method.RAW, "raw"),
    (Method.RIDGE, "ridge"),
    (Method.LOCAL_DIGITAL_ZNE, "zne"),
    (Method.LIAO, "liao"),
)


@pytest.fixture(scope="module")
def split_results_manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign-ledgers")
    generate("s0-t0-micro", root / "data")
    results = run(root / "data", root / "run", budget_tier="H")
    manifest_path = root / "results-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": "run/results.json"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, results


def _split_campaign_args(manifest_path, results):
    budget_cells = list(results["budget"].values())
    assert {cell["tier"] for cell in budget_cells} == {"H"}
    widths = {
        cell["pairing"]["descriptor"]["n_qubits"] for cell in budget_cells
    }
    assert len(widths) == 1
    return [
        "--tier",
        "H",
        "--width",
        str(next(iter(widths))),
        "--paired-budget-cells",
        str(len(budget_cells)),
        "--methods",
        *(method.value for method, _ in _RUNNER_ROSTER),
        "--calibration",
        f"{next(iter(widths))}=100",
        "--results-manifest",
        str(manifest_path),
    ]


def test_aggregate_throughput_reports_wall_hours_not_core_hours():
    total_evaluations = 2_500_000
    worker_count = 4
    per_worker_rate = 25.0
    aggregate_rate = worker_count * per_worker_rate

    wall_hours = estimate_wall_hours(total_evaluations, aggregate_rate)
    core_hours = estimate_core_hours(total_evaluations, per_worker_rate)

    assert wall_hours == pytest.approx(6.944444444444445)
    assert core_hours == pytest.approx(27.77777777777778)
    assert core_hours == pytest.approx(worker_count * wall_hours)


def test_campaign_output_states_quantity_and_parallelism_assumption(capsys):
    assert main(
        [
            "--tier",
            "L",
            "--width",
            "8",
            "--paired-budget-cells",
            "1",
            "--methods",
            "raw",
            "--calibration",
            "8=100",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert (
        "paired budget cells (paired-budget-cell-v1 descriptors): 1" in output
    )
    assert (
        "combined cap per method and paired-budget-cell-v1 descriptor: "
        "2,500,000" in output
    )
    assert "reserved total circuit evaluations: 2,500,000" in output
    assert (
        "reported reserved quantity: estimated reserved-allocation wall-hours"
        in output
    )
    assert "calibration includes the full worker allocation" in output
    assert "applies no additional parallel speedup" in output
    assert "estimated reserved-allocation wall-hours: 6.944" in output
    assert not any(
        line.startswith("total circuit evaluations:") for line in output.splitlines()
    )
    assert "realized circuit evaluations:" not in output
    assert "estimated realized wall-hours:" not in output
    assert "estimated core-hours" not in output


def test_t1_reserved_allocation_is_distinct_from_modeled_required_cost(capsys):
    assert set(_T1_INPUTS) == set(ROSTER)
    reserved_evaluations = campaign_budget(
        "H", 12, 1, ROSTER
    ).total_circuit_evaluations
    modeled_required_evaluations = sum(
        method_budget(method, _T1_INPUTS[method]).total for method in ROSTER
    )

    assert reserved_evaluations == 1_750_000_000
    assert modeled_required_evaluations == 314_572_800
    assert reserved_evaluations / modeled_required_evaluations == pytest.approx(
        5.563100179036458
    )
    assert main(
        [
            "--tier",
            "H",
            "--width",
            "12",
            "--paired-budget-cells",
            "1",
            "--methods",
            "all",
            "--calibration",
            "12=100",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "reserved total circuit evaluations: 1,750,000,000" in output
    assert "realized circuit evaluations:" not in output


def test_validated_runner_ledgers_report_realized_cost_separately(
    split_results_manifest, capsys
):
    manifest_path, results = split_results_manifest
    expected_realized = sum(
        results["methods"][runner_name]["ledger"]["total"]
        for _, runner_name in _RUNNER_ROSTER
    )

    assert main(_split_campaign_args(manifest_path, results)) == 0

    output = capsys.readouterr().out
    assert "reserved total circuit evaluations: 1,000,000,000" in output
    assert f"realized circuit evaluations: {expected_realized:,}" in output
    assert (
        f"estimated realized wall-hours: {expected_realized / 100 / 3_600:,.3f}"
        in output
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("tier", "runner artifact tier 'H' does not match reservation tier 'L'"),
        ("width", "runner artifact width 3 does not match reservation width 4"),
        ("cells", "runner artifacts contain 1 paired budget cells"),
        ("methods", "runner artifact roster does not match reservation methods"),
    ],
)
def test_runner_artifacts_must_match_every_reservation_axis(
    split_results_manifest, capsys, field, message
):
    manifest_path, results = split_results_manifest
    args = _split_campaign_args(manifest_path, results)
    if field == "tier":
        args[args.index("--tier") + 1] = "L"
    elif field == "width":
        args[args.index("--width") + 1] = "4"
        args[args.index("--calibration") + 1] = "4=100"
    elif field == "cells":
        args[args.index("--paired-budget-cells") + 1] = "2"
    else:
        args.remove(Method.LIAO.value)

    with pytest.raises(SystemExit, match="2"):
        main(args)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_malformed_runner_ledger_is_refused_without_reservation_fallback(
    split_results_manifest, capsys
):
    manifest_path, results = split_results_manifest
    tampered = copy.deepcopy(results)
    tampered["methods"]["raw"]["ledger"]["total"] += 1
    bad_results_path = manifest_path.parent / "bad-results.json"
    bad_results_path.write_text(json.dumps(tampered), encoding="utf-8")
    bad_manifest_path = manifest_path.parent / "bad-results-manifest.json"
    bad_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": bad_results_path.name}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main(_split_campaign_args(bad_manifest_path, results))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ledger does not match its buckets" in captured.err
