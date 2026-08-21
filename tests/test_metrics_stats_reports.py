"""Cell metrics, statistical oracles, OOD formulas, and report smoke tests."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qem_bench.circuits.tfi import TFIParams, build_tfi_circuit
from qem_bench.datasets.generate import (
    LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
    LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
    PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD,
    UNKNOWN_IDENTITY_ENCODING_PROFILE,
    generate,
    validate_physical_identity_encoding_profiles,
)
from qem_bench.datasets.schema import FAMILY_STRATA
from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split
from qem_bench.datasets.splits import ROLES, SplitSpec, resolve_split_spec
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
    _load,
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
from qem_bench.validation import (
    _validate_split_tfi_profile_rows,
    validate_split_artifact,
)


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


def _write_legacy_run_variant(
    source,
    destination,
    *,
    dataset_hash,
    preset,
    family=None,
    identity_profiles=None,
):
    result = json.loads(source.read_text(encoding="utf-8"))
    result["dataset_hash"] = dataset_hash
    result["dataset_item_stream_hashes"] = {"legacy-v1": dataset_hash}
    result["preset"] = preset
    if family is not None:
        for item in result["test_items"]:
            item["family"] = family
    if identity_profiles is not None:
        result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = copy.deepcopy(
            identity_profiles
        )
        result["methods"]["raw"]["config"]["run_artifact_identity"][
            PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD
        ] = copy.deepcopy(identity_profiles)
    _resign_run_artifact(result)
    assert validate_run_artifact(result) is result
    destination.write_text(json.dumps(result), encoding="utf-8")


def _write_historical_run_variant(
    source, destination, *, dataset_hash, preset
):
    result = json.loads(source.read_text(encoding="utf-8"))
    result["dataset_hash"] = dataset_hash
    result["dataset_item_stream_hashes"] = {"legacy-v1": dataset_hash}
    result["preset"] = preset
    result.pop(PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD)
    result["methods"]["raw"]["config"]["run_artifact_identity"].pop(
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD
    )
    _resign_run_artifact(result)
    assert validate_run_artifact(result) is result
    destination.write_text(json.dumps(result), encoding="utf-8")


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


@pytest.mark.parametrize(
    "stratum_source",
    ("bootstrap_stratum_id", "circuit_pool_id", "family"),
)
def test_bootstrap_rejects_cross_artifact_circuits_split_by_every_stratum_source(
    stratum_source,
):
    def source_items(source, artifact_index):
        items = [
            _item(0, circuit="physical-shared"),
            _item(artifact_index + 1, circuit=f"physical-{artifact_index}"),
        ]
        for item in items:
            if stratum_source == "family":
                item.pop("circuit_pool_id")
            item[stratum_source] = source
        return items

    def merged_records(second_source):
        records = []
        for artifact_index, source in enumerate(("source-a", second_source)):
            items = source_items(source, artifact_index)
            records.extend(
                build_cell_records(
                    "method",
                    items,
                    np.zeros(len(items)),
                    np.zeros(len(items)),
                    artifact_id=f"artifact-{artifact_index}",
                    grouping="six-part",
                )
            )
        return _merge_cell_records(records)

    positive = circuit_blocked_bootstrap(
        merged_records("source-a"),
        metric="mae",
        n_resamples=10,
        seed=20260819,
    )
    assert positive.n_circuits == 3

    with pytest.raises(
        ValueError,
        match=(
            r"physical circuits span bootstrap strata: "
            r"\{'physical-shared': \['source-a', 'source-b'\]\}"
        ),
    ):
        circuit_blocked_bootstrap(
            merged_records("source-b"),
            metric="mae",
            n_resamples=10,
            seed=20260819,
        )


@pytest.mark.parametrize("changed_identity", ("circuit_id", "bootstrap_stratum_id"))
def test_paired_bootstrap_rejects_mismatched_block_identities(changed_identity):
    items = [_item(index, circuit=f"physical-{index}") for index in range(2)]
    left = build_cell_records(
        "left",
        items,
        np.zeros(2),
        np.zeros(2),
        artifact_id="artifact",
        grouping="six-part",
    )
    matching_reference = build_cell_records(
        "reference",
        items,
        np.ones(2),
        np.zeros(2),
        artifact_id="artifact",
        grouping="six-part",
    )
    positive = circuit_blocked_bootstrap(
        left,
        reference_records=matching_reference,
        metric="mae",
        n_resamples=10,
        seed=20260819,
    )
    assert positive.paired is True
    assert positive.n_circuits == 2

    changed_items = copy.deepcopy(items)
    if changed_identity == "circuit_id":
        changed_items[0]["circuit_id"] = "physical-other"
    else:
        changed_items[0]["bootstrap_stratum_id"] = "other-stratum"
    changed_reference = build_cell_records(
        "reference",
        changed_items,
        np.ones(2),
        np.zeros(2),
        artifact_id="artifact",
        grouping="six-part",
    )

    with pytest.raises(
        ValueError,
        match="paired bootstrap requires identical circuit and stratum identities",
    ):
        circuit_blocked_bootstrap(
            left,
            reference_records=changed_reference,
            metric="mae",
            n_resamples=10,
            seed=20260819,
        )


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
        "circuit-fb751055544664e69b5cf3e425c352e2a968ecce207dbd73cf5c82bf0455362f"
    )
    zero_digest = (
        "circuit-a4f60181e1284fd121508f5684d984da61e9b366762e33227210b36d6f16cec5"
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


def test_legacy_manifest_declares_the_selected_serializer_profile(
    report_schema_version_runs,
    tmp_path,
):
    rounded_manifest = json.loads(
        (report_schema_version_runs / "legacy-data" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rounded_manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == {
        "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
    }
    assert rounded_manifest["dataset_hash"] == (
        "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"
    )

    exact_manifest = generate(
        copy.deepcopy(rounded_manifest["config"]),
        tmp_path / "exact-legacy-data",
    )
    assert exact_manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == {
        "tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
    }


def test_declared_identity_profile_domain_is_closed_for_every_family():
    candidates = (
        LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
        LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
        UNKNOWN_IDENTITY_ENCODING_PROFILE,
        "future-profile",
    )
    accepted = 0
    refused = 0
    for family in sorted(FAMILY_STRATA):
        for profile in candidates:
            expected = profile == LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE or (
                family == "tfi"
                and profile == LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
            )
            value = {family: profile}
            if expected:
                assert validate_physical_identity_encoding_profiles(
                    value, families={family}
                ) == value
                accepted += 1
            else:
                with pytest.raises(ValueError, match="identity encoding profile"):
                    validate_physical_identity_encoding_profiles(
                        value, families={family}
                    )
                refused += 1

    with pytest.raises(ValueError, match="unknown families"):
        validate_physical_identity_encoding_profiles(
            {"future-family": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
            families={"future-family"},
        )
    with pytest.raises(ValueError, match="family names must be strings"):
        validate_physical_identity_encoding_profiles(
            {1: LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
            families={"tfi"},
        )
    assert accepted == len(FAMILY_STRATA) + 1
    assert refused == len(FAMILY_STRATA) * 3 - 1


@pytest.mark.parametrize(
    ("profiles", "message"),
    (
        (None, "profiles must be an object"),
        ([], "profiles must be an object"),
        ("legacy-binary64-v1", "profiles must be an object"),
        (1, "profiles must be an object"),
        (True, "profiles must be an object"),
        ({}, "profile families do not match dataset families"),
        (
            {"future-family": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
            "profiles contain unknown families",
        ),
        ({"tfi": "future-profile"}, "unknown physical identity encoding profile"),
        ({"tfi": None}, "unknown physical identity encoding profile"),
        (
            {"tfi": UNKNOWN_IDENTITY_ENCODING_PROFILE},
            "unknown physical identity encoding profile",
        ),
        (
            {"qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
            "profile families do not match dataset families",
        ),
        (
            {
                "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
                "qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
            },
            "profile families do not match dataset families",
        ),
        (
            {"tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
            "identity fields do not match the declared serializer profile",
        ),
    ),
    ids=(
        "null-map",
        "array-map",
        "string-map",
        "integer-map",
        "boolean-map",
        "missing-family",
        "unknown-family",
        "unknown-profile",
        "non-string-profile",
        "declared-unknown-profile",
        "foreign-known-family",
        "extra-known-family",
        "wrong-allowed-profile",
    ),
)
def test_legacy_dataset_loader_refuses_every_invalid_profile_class(
    report_schema_version_runs,
    tmp_path,
    profiles,
    message,
):
    source = report_schema_version_runs / "legacy-data"
    data = tmp_path / "data"
    shutil.copytree(source, data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = profiles
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(data, tmp_path / "run")


def test_legacy_dataset_loader_refuses_coordinated_tfi_profile_relabels(
    report_schema_version_runs,
    tmp_path,
):
    rounded_data = report_schema_version_runs / "legacy-data"
    rounded_manifest = json.loads(
        (rounded_data / "manifest.json").read_text(encoding="utf-8")
    )
    exact_data = tmp_path / "exact-data"
    generate(copy.deepcopy(rounded_manifest["config"]), exact_data)

    _, loaded_rounded = _load(rounded_data)
    _, loaded_exact = _load(exact_data)
    assert loaded_rounded[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == {
        "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
    }
    assert loaded_exact[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == {
        "tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
    }

    relabels = (
        (
            rounded_data,
            "custom",
            LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
        ),
        (
            exact_data,
            "t0-micro",
            LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
        ),
    )
    for index, (source, preset, profile) in enumerate(relabels):
        data = tmp_path / f"relabel-{index}"
        shutil.copytree(source, data)
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["preset"] = preset
        manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = {"tfi": profile}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="identity fields do not match the declared serializer profile",
        ):
            _load(data)


def test_split_tfi_profile_replay_covers_every_role_pool_and_report_boundary(
    monkeypatch,
    tmp_path,
):
    generate_module = importlib.import_module("qem_bench.datasets.generate")
    original = generate_module._sample_and_build_circuit
    spec = replace(SPLIT_PRESETS["s0-t0-micro"], budget_tier="H")

    exact_data = tmp_path / "exact-data"
    generate_split(spec, exact_data)

    def rounded_sample(cfg, rng, instance, circuit_seed):
        params, circuit = original(cfg, rng, instance, circuit_seed)
        if isinstance(params, TFIParams):
            params = replace(params, j=round(params.j, 12), h=round(params.h, 12))
            circuit = build_tfi_circuit(params)
        return params, circuit

    rounded_data = tmp_path / "rounded-data"
    with monkeypatch.context() as generation_patch:
        generation_patch.setattr(
            generate_module,
            "_sample_and_build_circuit",
            rounded_sample,
        )
        generate_split(spec, rounded_data)
    rounded_manifest_path = rounded_data / "manifest.json"
    rounded_manifest = json.loads(
        rounded_manifest_path.read_text(encoding="utf-8")
    )
    rounded_manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = {
        "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
    }
    rounded_manifest_path.write_text(
        json.dumps(rounded_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    exact_items, exact_manifest = validate_split_artifact(exact_data)
    rounded_items, rounded_manifest = validate_split_artifact(rounded_data)
    resolution = resolve_split_spec(spec)
    expected_draws = {
        (pool.split, pool.circuit_pool_id, instance)
        for pool in resolution.circuit_pools
        if pool.family == "tfi"
        for instance in range(pool.n_instances)
    }
    for rows in (exact_items, rounded_items):
        observed_draws = {
            (item["split"], item["circuit_pool_id"], item["instance"])
            for item in rows
            if item["family"] == "tfi"
        }
        assert observed_draws == expected_draws
    assert {role for role, _, _ in expected_draws} == set(ROLES)

    for pool in resolution.circuit_pools:
        if pool.family != "tfi":
            continue
        altered = copy.deepcopy(exact_items)
        target_rows = [
            item
            for item in altered
            if item["circuit_pool_id"] == pool.circuit_pool_id
            and item["instance"] == 0
        ]
        assert target_rows
        rounded_j = round(target_rows[0]["j"], 12)
        assert rounded_j != target_rows[0]["j"]
        for item in target_rows:
            item["j"] = rounded_j
        with pytest.raises(
            ValueError,
            match=(
                r"split TFI identity fields do not match the declared "
                rf"serializer profile .* for pool '{pool.circuit_pool_id}', instance 0"
            ),
        ):
            _validate_split_tfi_profile_rows(
                altered,
                exact_manifest,
                resolution,
            )

    relabels = (
        (exact_data, LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE),
        (rounded_data, LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE),
    )
    for index, (source, profile) in enumerate(relabels):
        data = tmp_path / f"relabel-{index}"
        shutil.copytree(source, data)
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = {"tfi": profile}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(
            ValueError,
            match="split TFI identity fields do not match the declared serializer profile",
        ):
            validate_split_artifact(data)

    exact_run = run(exact_data, tmp_path / "exact-run")
    rounded_run = run(rounded_data, tmp_path / "rounded-run")
    assert validate_run_artifact(exact_run) is exact_run
    assert validate_run_artifact(rounded_run) is rounded_run
    report_manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [
            {"results": "exact-run/results.json"},
            {"results": "rounded-run/results.json"},
        ],
    }
    with pytest.raises(
        ValueError,
        match="report manifest cannot merge physical identity encoding profiles",
    ):
        _load_runs(report_manifest, tmp_path)


def test_disjoint_family_s3_run_projects_profiles_to_test_families(tmp_path):
    spec = SplitSpec(
        split_id="S3",
        source_domain={"circuit_family": ["tfi"]},
        target_domain={"circuit_family": ["qaoa"]},
        fixed_axes={
            "noise_family": ["depolarizing_readout"],
            "noise_strength": ["L1"],
            "family_native_depth": [1],
            "observable_class": ["z_mid"],
            "shots": [16],
        },
        n_qubits=[3],
        role_counts={"train": 1, "validation": 1, "test": 1},
        family_parameters={
            "tfi": {"dt": 0.2},
            "qaoa": {"graph_classes": ["path"]},
        },
        allowed_couplings=("family_native_parameters",),
        budget_tier="H",
    )
    data = tmp_path / "data"
    manifest = generate_split(spec, data)
    result = run(data, tmp_path / "run")

    assert manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == {
        "qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
        "tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
    }
    expected_run_profiles = {
        "qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
    }
    assert {item["family"] for item in result["test_items"]} == {"qaoa"}
    assert result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == expected_run_profiles
    assert result["methods"]["raw"]["config"]["run_artifact_identity"][
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD
    ] == expected_run_profiles
    assert validate_run_artifact(result) is result

    report_manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": "run/results.json"}],
    }
    loaded = _load_runs(report_manifest, tmp_path)
    assert [run["artifact_id"] for run in loaded] == [result["artifact_id"]]


def test_runner_binds_declared_profiles_and_refuses_resigned_profile_drift(
    report_schema_version_runs,
):
    source = report_schema_version_runs / "legacy-run" / "results.json"
    valid = json.loads(source.read_text(encoding="utf-8"))
    assert validate_run_artifact(valid) is valid

    drifted = copy.deepcopy(valid)
    drifted[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = {
        "tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
    }
    _resign_run_artifact(drifted)
    with pytest.raises(
        ValueError,
        match="methods.raw.config.run_artifact_identity",
    ):
        validate_run_artifact(drifted)

    invalid_profiles = (
        {},
        {"future-family": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
        {"tfi": "future-profile"},
        {"qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE},
    )
    refused = 0
    for profiles in invalid_profiles:
        invalid = copy.deepcopy(valid)
        invalid[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = profiles
        invalid["methods"]["raw"]["config"]["run_artifact_identity"][
            PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD
        ] = copy.deepcopy(profiles)
        _resign_run_artifact(invalid)
        with pytest.raises(ValueError, match="identity encoding profile"):
            validate_run_artifact(invalid)
        refused += 1
    assert refused == len(invalid_profiles)


def test_missing_dataset_profile_becomes_signed_unknown(
    report_schema_version_runs,
    tmp_path,
):
    source = report_schema_version_runs / "legacy-data"
    data = tmp_path / "historical-data"
    shutil.copytree(source, data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run(data, tmp_path / "historical-run")
    expected = {"tfi": UNKNOWN_IDENTITY_ENCODING_PROFILE}
    assert result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] == expected
    assert result["methods"]["raw"]["config"]["run_artifact_identity"][
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD
    ] == expected
    assert validate_run_artifact(result) is result


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


def test_report_manifest_rejects_mixed_legacy_tfi_identity_encodings(
    report_schema_version_runs,
    tmp_path,
):
    source = report_schema_version_runs / "legacy-run" / "results.json"
    _write_legacy_run_variant(
        source,
        tmp_path / "rounded.json",
        dataset_hash="1" * 64,
        preset="rounded-profile",
        identity_profiles={
            "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
        },
    )
    _write_legacy_run_variant(
        source,
        tmp_path / "exact.json",
        dataset_hash="2" * 64,
        preset="exact-profile",
        identity_profiles={
            "tfi": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
        },
    )
    manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": "rounded.json"}, {"results": "exact.json"}],
    }

    with pytest.raises(ValueError) as error:
        _load_runs(manifest, tmp_path)

    assert str(error.value) == (
        "report manifest cannot merge physical identity encoding profiles for "
        "family 'tfi'; found ['legacy-binary64-v1', "
        "'legacy-tfi-rounded-12-v1']"
    )


def test_report_manifest_accepts_each_legacy_tfi_identity_profile_and_ignores_others(
    report_schema_version_runs,
    tmp_path,
):
    source = report_schema_version_runs / "legacy-run" / "results.json"
    profiles = {
        "rounded": (
            LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
            ("1" * 64, "2" * 64),
        ),
        "exact": (
            LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
            ("3" * 64, "4" * 64),
        ),
    }

    for profile_name, (profile, hashes) in profiles.items():
        paths = []
        for index, dataset_hash in enumerate(hashes):
            path = tmp_path / f"{profile_name}-{index}.json"
            _write_legacy_run_variant(
                source,
                path,
                dataset_hash=dataset_hash,
                preset=f"{profile_name}-{index}",
                identity_profiles={"tfi": profile},
            )
            paths.append(path)
        manifest = {
            "schema_version": "qem-bench-report-manifest-v1",
            "runs": [{"results": path.name} for path in paths],
        }
        assert len(_load_runs(manifest, tmp_path)) == 2

    for index, dataset_hash in enumerate(("5" * 64, "6" * 64)):
        _write_legacy_run_variant(
            source,
            tmp_path / f"qaoa-{index}.json",
            dataset_hash=dataset_hash,
            preset=f"qaoa-{index}",
            family="qaoa",
            identity_profiles={
                "qaoa": LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
            },
        )
    non_tfi_manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": f"qaoa-{index}.json"} for index in range(2)],
    }
    assert len(_load_runs(non_tfi_manifest, tmp_path)) == 2


@pytest.mark.parametrize("partner_profile", ("unknown", "declared"))
def test_report_manifest_refuses_to_merge_missing_profiles(
    report_schema_version_runs,
    tmp_path,
    partner_profile,
):
    source = report_schema_version_runs / "legacy-run" / "results.json"
    first = tmp_path / "historical-0.json"
    _write_historical_run_variant(
        source,
        first,
        dataset_hash="7" * 64,
        preset="historical-0",
    )
    single_manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": first.name}],
    }
    assert len(_load_runs(single_manifest, tmp_path)) == 1

    partner = tmp_path / f"{partner_profile}.json"
    if partner_profile == "unknown":
        _write_historical_run_variant(
            source,
            partner,
            dataset_hash="8" * 64,
            preset="historical-1",
        )
        expected_profiles = "['unknown', 'unknown']"
    else:
        _write_legacy_run_variant(
            source,
            partner,
            dataset_hash="9" * 64,
            preset="declared",
            identity_profiles={
                "tfi": LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
            },
        )
        expected_profiles = "['legacy-tfi-rounded-12-v1', 'unknown']"
    combined_manifest = {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [{"results": first.name}, {"results": partner.name}],
    }

    with pytest.raises(ValueError) as error:
        _load_runs(combined_manifest, tmp_path)

    assert str(error.value) == (
        "report manifest cannot merge physical identity encoding profiles for "
        f"family 'tfi'; found {expected_profiles}"
    )


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
