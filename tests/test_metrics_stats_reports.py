"""Cell metrics, statistical oracles, OOD formulas, and report smoke tests."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qem_bench.datasets.generate import generate
from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split
from qem_bench.reports import generate_report
from qem_bench.reports.generate import _load_runs, _merge_cell_records
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
from qem_bench.validation import validate_split_artifact


def _item(
    index,
    *,
    circuit,
    severity="L1",
    observable="z_mid",
    measurement_group=None,
):
    return {
        "item_id": f"item-{index}",
        "measurement_group": measurement_group or circuit,
        "circuit_id": circuit,
        "circuit_pool_id": "pool-tfi-test",
        "split": "test",
        "family": "tfi",
        "instance": index,
        "stratum": "continuous_regression",
        "noise_family": "depolarizing_readout",
        "severity": severity,
        "observable": observable,
        "ideal_expectation": 0.0,
    }


def _legacy_item(*, circuit_seed=17, j=0.5):
    return {
        "item_id": "tfi-0003-z_mid",
        "measurement_group": "tfi-0003-g0",
        "split": "test",
        "family": "tfi",
        "instance": 3,
        "n_qubits": 4,
        "circuit_seed": circuit_seed,
        "steps": 2,
        "j": j,
        "h": 0.75,
        "dt": 0.1,
        "stratum": "continuous_regression",
        "noise_family": "depolarizing_readout",
        "severity": "L1",
        "shots": 256,
        "sampler_seed": 101,
        "observable": "z_mid",
        "ideal_expectation": 0.0,
    }


def _legacy_qaoa_item(edges):
    item = _legacy_item()
    for field in ("steps", "j", "h", "dt"):
        item.pop(field)
    item.update(
        {
            "family": "qaoa",
            "n_qubits": 3,
            "p": 1,
            "graph_class": "path",
            "edges": edges,
            "edge_probability": None,
            "gammas": [0.25],
            "betas": [0.5],
        }
    )
    return item


def _derived_circuit_id(item):
    records = build_cell_records(
        "method",
        [item],
        np.array([0.0]),
        np.zeros(1),
        artifact_id="identity-probe",
        grouping="six-part",
    )
    return records[0]["circuit_ids"][0]


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
        role_assignment=result.get("role_assignment"),
        budget=result.get("budget"),
    )


def _resign_run_artifact(result):
    result["artifact_id"] = _expected_run_artifact_id(result)
    _rebuild_run_derived_results(result)


@pytest.mark.protocol("QEM-P001")
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


@pytest.mark.protocol("QEM-P002")
def test_bootstrap_blocks_split_physical_circuits_across_severities(tmp_path):
    base_spec = SPLIT_PRESETS["s0-t0-micro"]
    fixed_axes = {
        axis: list(values) for axis, values in base_spec.fixed_axes.items()
    }
    fixed_axes["noise_strength"] = ["L1", "L2"]
    data = tmp_path / "split"
    generate_split(replace(base_spec, fixed_axes=fixed_axes), data)
    generated, manifest = validate_split_artifact(data)
    items = [item for item in generated if item["split"] == "test"]

    assert manifest["dataset_schema_version"] == "split-v2"
    assert len({item["measurement_group"] for item in items}) == 4
    assert len({item["circuit_id"] for item in items}) == 2
    assert {item["severity"] for item in items} == {"L1", "L2"}
    records = build_cell_records(
        "method",
        items,
        np.linspace(-0.5, 0.5, len(items)),
        np.zeros(len(items)),
        artifact_id="artifact-split",
        grouping="six-part",
    )
    interval = circuit_blocked_bootstrap(
        records, metric="mae", n_resamples=100, seed=20260819
    )

    assert interval.n_circuits == 2
    assert {
        item["circuit_id"] for record in records for item in record["items"]
    } == {item["circuit_id"] for item in items}


def test_report_estimand_is_invariant_to_artifact_partition():
    items = [_item(index, circuit=f"physical-{index}") for index in range(4)]
    predictions = np.array([0.0, 1.0, 1.0, 1.0])
    one_artifact = build_cell_records(
        "method",
        items,
        predictions,
        np.zeros(4),
        artifact_id="artifact-one",
        grouping="six-part",
    )
    two_artifacts = build_cell_records(
        "method",
        items[:1],
        predictions[:1],
        np.zeros(1),
        artifact_id="artifact-a",
        grouping="six-part",
    ) + build_cell_records(
        "method",
        items[1:],
        predictions[1:],
        np.zeros(3),
        artifact_id="artifact-b",
        grouping="six-part",
    )

    assert macro_mean_iqr(one_artifact)["mae"]["mean"] == pytest.approx(0.75)
    assert macro_mean_iqr(two_artifacts)["mae"]["mean"] == pytest.approx(0.5)

    merged_one = _merge_cell_records(one_artifact)
    merged_two = _merge_cell_records(two_artifacts)
    one_mean = macro_mean_iqr(merged_one)["mae"]["mean"]
    two_mean = macro_mean_iqr(merged_two)["mae"]["mean"]
    one_interval = circuit_blocked_bootstrap(
        merged_one, metric="mae", n_resamples=100, seed=20260819
    )
    two_interval = circuit_blocked_bootstrap(
        merged_two, metric="mae", n_resamples=100, seed=20260819
    )

    assert len(merged_one) == len(merged_two) == 1
    assert one_mean == pytest.approx(0.75)
    assert two_mean == pytest.approx(one_mean)
    assert one_interval.estimate == pytest.approx(one_mean)
    assert two_interval.estimate == pytest.approx(one_mean)
    assert (one_interval.lower, one_interval.upper) == (0.25, 1.0)
    assert (two_interval.lower, two_interval.upper) == (
        one_interval.lower,
        one_interval.upper,
    )
    assert len(set(merged_two[0]["item_ids"])) == 4
    assert len(set(merged_two[0]["circuit_ids"])) == 4


def test_bootstrap_counts_cross_artifact_physical_circuits_once():
    first_severity = [
        _item(index, circuit=f"physical-{index}", severity="L1")
        for index in range(2)
    ]
    second_severity = [
        _item(index + 2, circuit=f"physical-{index}", severity="L2")
        for index in range(2)
    ]
    records = build_cell_records(
        "method",
        first_severity,
        np.array([0.0, 1.0]),
        np.zeros(2),
        artifact_id="artifact-l1",
        grouping="six-part",
    ) + build_cell_records(
        "method",
        second_severity,
        np.array([0.25, 0.75]),
        np.zeros(2),
        artifact_id="artifact-l2",
        grouping="six-part",
    )

    interval = circuit_blocked_bootstrap(
        _merge_cell_records(records),
        metric="mae",
        n_resamples=100,
        seed=20260819,
    )

    assert interval.n_circuits == 2


def test_report_merge_namespaces_source_local_identifiers():
    records = []
    for artifact_id, prediction in (("artifact-a", 0.0), ("artifact-b", 1.0)):
        records.extend(
            build_cell_records(
                "method",
                [_item(0, circuit="physical-local")],
                np.array([prediction]),
                np.zeros(1),
                artifact_id=artifact_id,
                grouping="six-part",
            )
        )

    merged = _merge_cell_records(records)

    assert len(merged) == 1
    assert len(set(merged[0]["item_ids"])) == 2
    assert merged[0]["circuit_ids"] == ["physical-local"]
    assert {item["artifact_id"] for item in merged[0]["items"]} == {
        "artifact-a",
        "artifact-b",
    }


def test_legacy_circuit_digest_prevents_cross_dataset_collisions_and_refuses_gaps():
    first = _legacy_item(circuit_seed=17, j=0.5)
    second = _legacy_item(circuit_seed=29, j=0.8)
    assert f"{first['family']}:{first['instance']}" == (
        f"{second['family']}:{second['instance']}"
    )

    records = build_cell_records(
        "method",
        [first],
        np.array([0.0]),
        np.zeros(1),
        artifact_id="legacy-dataset-a",
        grouping="six-part",
    ) + build_cell_records(
        "method",
        [second],
        np.array([1.0]),
        np.zeros(1),
        artifact_id="legacy-dataset-b",
        grouping="six-part",
    )
    same_circuit_new_execution = dict(first)
    same_circuit_new_execution.update(
        {
            "item_id": "other-item",
            "measurement_group": "other-group",
            "noise_family": "amplitude_phase_damping",
            "severity": "L2",
            "shots": 4096,
            "sampler_seed": 999,
        }
    )
    same_circuit_records = build_cell_records(
        "method",
        [same_circuit_new_execution],
        np.array([0.0]),
        np.zeros(1),
        artifact_id="legacy-dataset-c",
        grouping="six-part",
    )
    merged = _merge_cell_records(records)
    interval = circuit_blocked_bootstrap(
        merged, metric="mae", n_resamples=100, seed=20260819
    )

    assert records[0]["circuit_ids"] == same_circuit_records[0]["circuit_ids"]
    assert len(merged[0]["circuit_ids"]) == 2
    assert interval.n_circuits == 2

    incomplete = dict(first)
    incomplete.pop("h")
    with pytest.raises(
        ValueError,
        match=r"cannot determine physical circuit.*missing circuit fields: \['h'\]",
    ):
        build_cell_records(
            "method",
            [incomplete],
            np.array([0.0]),
            np.zeros(1),
            artifact_id="legacy-incomplete",
            grouping="six-part",
        )

    null_parameter = dict(first)
    null_parameter["h"] = None
    with pytest.raises(
        ValueError,
        match=r"cannot determine physical circuit.*missing circuit fields: \['h'\]",
    ):
        build_cell_records(
            "method",
            [null_parameter],
            np.array([0.0]),
            np.zeros(1),
            artifact_id="legacy-null",
            grouping="six-part",
        )


def test_legacy_circuit_digest_canonicalizes_numeric_encodings_and_qaoa_edges():
    def probe(j):
        item = _legacy_item(circuit_seed=1, j=j)
        item["h"] = 0.5
        return _derived_circuit_id(item)

    digests = {
        label: probe(value)
        for label, value in (
            ("1", 1),
            ("1.0", 1.0),
            ("0.0", 0.0),
            ("-0.0", -0.0),
        )
    }
    unit_digest = (
        "circuit-049f094f1fcef3aa43a8d1a2070ad6371e21c34e2f88ac2c285e3675c84fefc9"
    )
    zero_digest = (
        "circuit-07b0c47b6857e06a90e13c2af7849fa59d79f5fba6dd46cdc6000835c1d047c3"
    )
    assert digests == {
        "1": unit_digest,
        "1.0": unit_digest,
        "0.0": zero_digest,
        "-0.0": zero_digest,
    }
    assert _derived_circuit_id(
        _legacy_qaoa_item([[0, 1], [1, 2]])
    ) == _derived_circuit_id(_legacy_qaoa_item([[2, 1], [1, 0]]))


def test_legacy_and_split_routes_derive_the_same_physical_circuit_id(tmp_path):
    data = tmp_path / "shared-circuit-identity"
    generate("s0-t0-micro", data)
    items, _ = validate_split_artifact(data)
    split_item = items[0]
    legacy_projection = dict(split_item)
    legacy_projection.pop("circuit_id")

    assert _derived_circuit_id(legacy_projection) == split_item["circuit_id"]


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


@pytest.fixture(scope="module")
def report_schema_version_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("report-schema-version-runs")
    legacy_data = root / "legacy-data"
    generate("t0-micro", legacy_data)
    run(legacy_data, root / "legacy-run")

    split_data = root / "split-data"
    split_spec = replace(SPLIT_PRESETS["s0-t0-micro"], budget_tier="H")
    generate_split(split_spec, split_data)
    run(split_data, root / "split-run")
    return root


def test_report_manifest_rejects_mixed_dataset_schema_versions(
    report_schema_version_runs,
):
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [
            {"results": "legacy-run/results.json"},
            {"results": "split-run/results.json"},
        ],
    }

    with pytest.raises(ValueError) as error:
        _load_runs(manifest, report_schema_version_runs)

    assert str(error.value) == (
        "report manifest cannot mix dataset schema versions; "
        "found ['legacy-v1', 'split-v2']"
    )


def test_report_manifest_accepts_one_dataset_schema_version(
    report_schema_version_runs,
):
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": "legacy-run/results.json"}],
    }

    loaded = _load_runs(manifest, report_schema_version_runs)

    assert [item["dataset_schema_version"] for item in loaded] == ["legacy-v1"]


def test_report_manifest_rejects_zero_runs(tmp_path):
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [],
    }

    with pytest.raises(ValueError, match="^report manifest contains no runs$"):
        _load_runs(manifest, tmp_path)


def test_report_manifest_rejects_missing_dataset_schema_version(
    report_schema_version_runs,
):
    source = report_schema_version_runs / "legacy-run" / "results.json"
    unversioned = json.loads(source.read_text(encoding="utf-8"))
    unversioned.pop("dataset_schema_version")
    path = report_schema_version_runs / "missing-version.json"
    path.write_text(json.dumps(unversioned), encoding="utf-8")
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": path.name}],
    }

    with pytest.raises(ValueError) as error:
        _load_runs(manifest, report_schema_version_runs)

    assert str(error.value) == (
        "invalid runner result at missing-version.json: unversioned "
        "qem-bench-run-v2 artifact requires explicit migration; "
        "dataset_schema_version is required"
    )


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


@pytest.mark.protocol("QEM-P005")
def test_run_validator_rejects_resigned_invalid_ledgers(tmp_path):
    data = tmp_path / "data"
    generate("t0-micro", data)
    valid = run(data, tmp_path / "results")
    mutations = (
        (
            lambda spec: spec.__setitem__("ledger", []),
            "ledger must be an object",
        ),
        (
            lambda spec: spec["ledger"].__setitem__("B_train", -1),
            "ledger B_train must be a nonnegative integer",
        ),
        (
            lambda spec: spec["ledger"].pop("B_extra"),
            "ledger B_extra must be a nonnegative integer",
        ),
        (
            lambda spec: spec["ledger"].__setitem__("unexpected", 0),
            "ledger does not match its buckets",
        ),
        (
            lambda spec: spec["ledger"].__setitem__(
                "B_pred", float(spec["ledger"]["B_pred"])
            ),
            "ledger B_pred must be a nonnegative integer",
        ),
        (
            lambda spec: spec["ledger"].__setitem__("B_pred", True),
            "ledger B_pred must be a nonnegative integer",
        ),
    )
    derived_fields = (
        "total",
        "amortized_per_test_item",
        "nominal_test_total",
        "realized_test_total",
        "nominal_total",
        "test_budget_ratio",
        "circuit_evals_per_mitigated_expectation",
    )

    for mutate, message in mutations:
        tampered = copy.deepcopy(valid)
        mutate(tampered["methods"]["raw"])
        _resign_run_artifact(tampered)
        with pytest.raises(ValueError, match=message):
            validate_run_artifact(tampered)

    for field in derived_fields:
        tampered = copy.deepcopy(valid)
        tampered["methods"]["raw"]["ledger"][field] = -1
        _resign_run_artifact(tampered)
        with pytest.raises(ValueError, match="ledger does not match its buckets"):
            validate_run_artifact(tampered)

    nonfinite = copy.deepcopy(valid)
    nonfinite["methods"]["raw"]["ledger"]["test_budget_ratio"] = float("nan")
    with pytest.raises(ValueError, match="ledger is not canonical JSON"):
        validate_run_artifact(nonfinite)


def test_report_errors_keep_relative_portable_run_paths(monkeypatch, tmp_path):
    work = tmp_path / "path-leak"
    work.mkdir()
    (work / "bad.json").write_text(
        json.dumps({"schema_version": "unsupported"}), encoding="utf-8"
    )
    (work / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": "bad.json"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(work)

    with pytest.raises(ValueError) as error:
        generate_report("manifest.json", "report", bootstrap_resamples=1)

    message = str(error.value)
    assert message == (
        "invalid runner result at bad.json: unsupported runner result schema_version"
    )
    assert str(work.resolve()) not in message
    assert re.search(r"[A-Za-z]:[\\/]", message) is None

    (work / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "qem-bench-report-manifest-v1",
                "runs": [{"results": "missing.json"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as missing_error:
        generate_report("manifest.json", "report", bootstrap_resamples=1)
    missing_message = str(missing_error.value)
    assert missing_message == (
        "invalid runner result at missing.json: unable to read result"
    )
    assert str(work.resolve()) not in missing_message
    assert re.search(r"[A-Za-z]:[\\/]", missing_message) is None


def test_report_manifest_rejects_absolute_and_escaping_run_paths(tmp_path):
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": str((tmp_path / "absolute.json").resolve())}],
    }
    with pytest.raises(
        ValueError, match="report manifest run results paths must be relative"
    ):
        _load_runs(manifest, tmp_path)

    manifest["runs"] = [{"results": "../outside.json"}]
    with pytest.raises(ValueError) as error:
        _load_runs(manifest, tmp_path / "manifest-dir")
    assert str(error.value) == (
        "report run result path escapes manifest directory: ../outside.json"
    )


def test_report_manifest_rejects_duplicate_run_artifact_ids(monkeypatch, tmp_path):
    artifact = {"artifact_id": "sha256:duplicate"}
    for filename in ("first.json", "second.json"):
        (tmp_path / filename).write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(
        "qem_bench.reports.generate.validate_run_artifact", lambda value: value
    )
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [
            {"results": "first.json"},
            {"results": "second.json"},
        ],
    }

    with pytest.raises(ValueError, match="duplicate run artifact_id"):
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
