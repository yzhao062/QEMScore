"""Cell metrics, statistical oracles, OOD formulas, and report smoke tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qem_bench.datasets.generate import generate
from qem_bench.reports import generate_report
from qem_bench.reports.generate import _load_runs
from qem_bench.runner.metrics import (
    CELL_GROUPINGS,
    DEFAULT_CELL_GROUPING,
    build_cell_records,
    headline_metrics,
    pooled_method_metrics,
)
from qem_bench.runner.run import (
    _dataset_item_stream_hashes,
    _run_artifact_id,
    run,
    validate_run_artifact,
)
from qem_bench.stats import (
    PlannedComparisonFamily,
    circuit_blocked_bootstrap,
    critical_difference,
    critical_difference_resolution,
    friedman_rank_test,
    holm_adjust,
    method_s0_relative_degradation_rejected,
    macro_mean_iqr,
    raw_normalized_ood_degradation,
    wilcoxon_holm_power_analysis,
    wilcoxon_rank_sums,
)
from qem_bench.stats.plots import critical_difference_diagram


def _item(index, *, circuit, severity="L1", observable="z_mid"):
    return {
        "item_id": f"item-{index}",
        "measurement_group": circuit,
        "split": "test",
        "family": "tfi",
        "stratum": "continuous_regression",
        "noise_family": "depolarizing_readout",
        "severity": severity,
        "observable": observable,
        "ideal_expectation": 0.0,
    }


def _rebuild_run_derived_results(result):
    items = result["test_items"]
    raw_predictions = np.asarray(result["methods"]["raw"]["predictions"])
    targets = np.asarray([item["ideal_expectation"] for item in items])
    for name, spec in result["methods"].items():
        predictions = np.asarray(spec["predictions"])
        records_by_grouping = {
            grouping: build_cell_records(
                name,
                items,
                predictions,
                raw_predictions,
                artifact_id=result["artifact_id"],
                grouping=grouping,
            )
            for grouping in CELL_GROUPINGS
        }
        default_records = records_by_grouping[DEFAULT_CELL_GROUPING]
        spec.update(
            {
                "cell_grouping": DEFAULT_CELL_GROUPING,
                "cell_records": default_records,
                "macro": macro_mean_iqr(default_records),
                "metrics": headline_metrics(default_records),
                "pooled_diagnostic": pooled_method_metrics(
                    predictions, raw_predictions, targets
                ),
                "cell_grouping_sensitivity": {
                    grouping: {
                        "n_cells": len(records),
                        "macro": macro_mean_iqr(records),
                        "headline_metrics": headline_metrics(records),
                    }
                    for grouping, records in records_by_grouping.items()
                },
            }
        )
    ridge_mae = result["methods"]["ridge"]["metrics"]["mae"]
    feature_only_mae = result["methods"]["feat-only"]["metrics"]["mae"]
    result["surrogate_alarm"].update(
        {
            "ridge_mae": ridge_mae,
            "feature_only_mae": feature_only_mae,
            "triggered": bool(feature_only_mae <= ridge_mae * 1.05),
        }
    )


def _expected_run_artifact_id(result):
    return _run_artifact_id(
        result["dataset_hash"],
        result["preset"],
        result["dataset_item_stream_hashes"],
        result["test_items"],
        result["methods"],
        dataset_environment_contract=result["dataset_environment_contract"],
        environment_contract=result["environment_contract"],
    )


def test_pooled_diagnostic_differs_from_macro_on_unequal_cells():
    items = [_item(0, circuit="c0", severity="L1")]
    items.extend(_item(i, circuit=f"c{i}", severity="L2") for i in range(1, 4))
    predictions = np.array([0.0, 1.0, 1.0, 1.0])
    raw = np.zeros(4)
    records = build_cell_records(
        "method", items, predictions, raw, artifact_id="artifact", grouping="six-part"
    )
    assert len(records) == 2
    assert headline_metrics(records)["mae"] == pytest.approx(0.5)
    assert pooled_method_metrics(predictions, raw, np.zeros(4))["mae"] == pytest.approx(
        0.75
    )
    assert all(record["artifact_id"] == "artifact" for record in records)
    assert all(record["items"] for record in records)


def test_cell_grouping_parameter_keeps_both_declared_options():
    items = [
        _item(0, circuit="c0", observable="z_mid"),
        _item(1, circuit="c0", observable="zz_mid"),
    ]
    predictions = np.array([0.1, 0.3])
    six_part = build_cell_records(
        "m", items, predictions, predictions, artifact_id="a", grouping="six-part"
    )
    observable_excluded = build_cell_records(
        "m",
        items,
        predictions,
        predictions,
        artifact_id="a",
        grouping="observable-excluded",
    )
    assert len(six_part) == 2
    assert len(observable_excluded) == 1
    assert len(observable_excluded[0]["circuit_ids"]) == 1


def test_split_artifact_identity_uses_per_cell_item_stream_hashes():
    manifest = {
        "dataset_schema_version": "split-v2",
        "cells": [
            {"cell_id": "S0/source/0", "item_stream_hash": "a" * 64},
            {"cell_id": "S0/target/0", "item_stream_hash": "b" * 64},
        ],
    }
    assert _dataset_item_stream_hashes(manifest) == {
        "S0/source/0": "a" * 64,
        "S0/target/0": "b" * 64,
    }


def test_demsar_2006_published_wilcoxon_oracle():
    # Demsar 2006, Table 2 and page 7: R+ = 93 and R- = 12.
    c45 = np.array(
        [.763, .599, .954, .628, .882, .936, .661, .583, .775, 1, .940, .619, .972, .957]
    )
    tuned = np.array(
        [.768, .591, .971, .661, .888, .931, .668, .583, .838, 1, .962, .666, .981, .978]
    )
    positive, negative = wilcoxon_rank_sums(tuned - c45)
    assert positive == pytest.approx(93.0)
    assert negative == pytest.approx(12.0)


def test_demsar_2006_published_friedman_oracle():
    # Demsar 2006, Table 6 and pages 13-14.
    published_ranks = np.array(
        [
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [4, 1, 2, 3],
            [4, 1, 3, 2],
            [4, 2, 3, 1],
            [1, 2.5, 4, 2.5],
            [3, 2, 4, 1],
            [2.5, 2.5, 4, 1],
            [4, 3, 2, 1],
            [2.5, 2.5, 2.5, 2.5],
            [4, 2.5, 1, 2.5],
            [3, 2, 4, 1],
            [4, 1, 2, 3],
            [3, 1, 4, 2],
        ]
    )
    result = friedman_rank_test(published_ranks, lower_is_better=True)
    assert result["average_ranks"] == pytest.approx(
        [3.143, 2.000, 2.893, 1.964], abs=5e-4
    )
    assert result["friedman_chi_square"] == pytest.approx(9.28, abs=0.01)
    assert result["iman_davenport_f"] == pytest.approx(3.69, abs=0.01)


def test_holm_hand_computed_oracle():
    # Sorted raw p-values .01, .03, .04 become .03, .06, .06 after monotonicity.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cd_resolution_and_planned_wilcoxon_power_are_distinct():
    resolution = critical_difference_resolution(9, rank_separation=1.0)
    assert resolution.label == "critical-difference resolution"
    assert resolution.q_alpha == 3.102
    assert resolution.minimum_paired_cells == 145
    assert critical_difference(145, 9) <= 1.0

    family = PlannedComparisonFamily(
        "headline-mae-v1",
        "mae",
        (("learned", "raw"), ("learned", "zne"), ("reject", "parent")),
    )
    power = wilcoxon_holm_power_analysis(family)
    assert power.planned_comparisons == 3
    assert power.minimum_paired_blocks == 47
    assert power.estimated_power >= 0.80


def test_circuit_blocking_widens_interval_when_observables_per_circuit_rise():
    independent_values = np.array([0.0, 1.0] * 12)
    independent_items = [
        _item(i, circuit=f"c{i}", observable="z_mid") for i in range(24)
    ]
    clustered_values = np.repeat([0.0, 0.0, 1.0, 1.0], 6)
    clustered_items = [
        _item(i, circuit=f"c{i // 6}", observable=f"z{i % 6}") for i in range(24)
    ]
    independent = build_cell_records(
        "m",
        independent_items,
        independent_values,
        np.zeros(24),
        artifact_id="independent",
        grouping="observable-excluded",
    )
    clustered = build_cell_records(
        "m",
        clustered_items,
        clustered_values,
        np.zeros(24),
        artifact_id="clustered",
        grouping="observable-excluded",
    )
    narrow = circuit_blocked_bootstrap(
        independent, metric="mae", n_resamples=2_000, seed=10
    )
    wide = circuit_blocked_bootstrap(clustered, metric="mae", n_resamples=2_000, seed=10)
    assert wide.width > narrow.width
    assert wide.n_circuits == 4
    assert narrow.n_circuits == 24


def test_ood_counterexample_keeps_rejected_formula_visible():
    method = {"S0": 0.001, "S1": 0.010}
    raw = {"S0": 0.080, "S1": 0.100}
    recommended = raw_normalized_ood_degradation(method, raw, tau=0.02)
    rejected = method_s0_relative_degradation_rejected(method)
    assert recommended["improvement"] == pytest.approx({"S0": 0.9875, "S1": 0.9})
    assert recommended["degradation"]["S1"] == pytest.approx(0.0875)
    assert recommended["composite"] == pytest.approx(0.0875)
    assert rejected["degradation"]["S1"] == pytest.approx(9.0)


def test_reports_render_tex_pdf_and_trace_real_cells(tmp_path):
    data = tmp_path / "data"
    results_dir = tmp_path / "results"
    generate("t0-micro", data)
    results = run(data, results_dir)
    assert results["dataset_item_stream_hashes"] == {
        "legacy-v1": results["dataset_hash"]
    }
    manifest = tmp_path / "report-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": "results/results.json"}],
            }
        ),
        encoding="utf-8",
    )
    outputs = generate_report(manifest, tmp_path / "report", bootstrap_resamples=100)
    tex = outputs["table1"].read_text(encoding="utf-8")
    assert "Local digital ZNE" in tex
    assert "Nominal total" in tex and "Realized total" in tex
    assert outputs["figure1"].read_bytes().startswith(b"%PDF")
    trace = json.loads(outputs["trace"].read_text(encoding="utf-8"))
    assert trace["artifacts"] == [results["artifact_id"]]
    assert trace["table1"]["raw"]["cell_ids"]
    assert trace["source_manifest"] == manifest.name
    assert trace["source_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert str(manifest.resolve()) not in json.dumps(trace)


def test_checked_report_trace_authenticates_manifest_bytes():
    report_dir = Path(__file__).parents[1] / "examples" / "report-walking-skeleton"
    manifest_bytes = (report_dir / "results-manifest.json").read_bytes()
    trace = json.loads((report_dir / "report-trace.json").read_text(encoding="utf-8"))

    assert trace["source_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )


def test_report_loader_rejects_tampered_metrics_but_accepts_valid_run(tmp_path):
    data = tmp_path / "data"
    results_dir = tmp_path / "results"
    generate("t0-micro", data)
    results = run(data, results_dir)
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": "results/results.json"}],
    }

    assert validate_run_artifact(results) is results
    assert _load_runs(manifest, tmp_path)[0]["artifact_id"] == results["artifact_id"]
    assert results["analysis_contract"]["metric_schema"] == (
        "qem-bench-cell-metrics-v1"
    )
    assert len(results["methods"]["raw"]["predictions"]) == results["n_test_items"]

    tampered = copy.deepcopy(results)
    tampered["methods"]["raw"]["metrics"]["mae"] = 999.0
    tampered["methods"]["raw"]["cell_records"][0]["metrics"]["mae"] = 999.0
    (results_dir / "results.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="derived result mismatch"):
        _load_runs(manifest, tmp_path)


def test_run_identity_binds_test_targets_and_is_stable(tmp_path):
    data = tmp_path / "data"
    generate("t0-micro", data)
    first = run(data, tmp_path / "first")
    second = run(data, tmp_path / "second")

    assert first["artifact_id"] == second["artifact_id"]
    assert first["test_items"] == second["test_items"]
    assert validate_run_artifact(first) is first

    tampered = copy.deepcopy(first)
    target_item = next(
        item
        for item in tampered["test_items"]
        if item["ideal_expectation"] == pytest.approx(0.966655505395)
    )
    target_item["ideal_expectation"] = 0.841655505395
    _rebuild_run_derived_results(tampered)

    assert tampered["methods"]["raw"]["metrics"] != first["methods"]["raw"]["metrics"]
    assert _expected_run_artifact_id(tampered) != first["artifact_id"]
    assert tampered["artifact_id"] == first["artifact_id"]
    with pytest.raises(ValueError, match="artifact_id"):
        validate_run_artifact(tampered)


def test_report_renders_zero_measurement_control_in_separate_log_column(tmp_path):
    data = tmp_path / "data"
    results_dir = tmp_path / "results"
    generate("t0-micro", data)
    run(data, results_dir)
    manifest = tmp_path / "report-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": "results/results.json"}],
            }
        ),
        encoding="utf-8",
    )

    outputs = generate_report(
        manifest,
        tmp_path / "report",
        methods=("raw", "feat-only"),
        bootstrap_resamples=10,
    )
    trace = json.loads(outputs["trace"].read_text(encoding="utf-8"))
    assert trace["table1"]["feat-only"]["ledger"]["test_budget_ratio"] is None
    point = next(
        point
        for point in trace["figure1"]["points"]
        if point["method"] == "feat-only"
    )
    assert point["circuit_evals_per_mitigated_expectation"] == 0.0
    assert point["plot_x"] > 0.0
    assert point["zero_cost_control"] is True
    assert "undefined" in outputs["table1"].read_text(encoding="utf-8")


def test_critical_difference_diagram_renders_without_display(tmp_path):
    path = critical_difference_diagram(
        {"raw": 2.5, "ridge": 1.5, "zne": 2.0}, 20, tmp_path / "cd.pdf"
    )
    assert path.read_bytes().startswith(b"%PDF")
