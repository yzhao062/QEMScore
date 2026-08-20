import pytest

from tools.campaign_cost import estimate_core_hours, estimate_wall_hours, main


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
    assert "total circuit evaluations: 2,500,000" in output
    assert "reported quantity: estimated wall-hours" in output
    assert "calibration includes the full worker allocation" in output
    assert "applies no additional parallel speedup" in output
    assert "estimated wall-hours: 6.944" in output
    assert "estimated core-hours" not in output
