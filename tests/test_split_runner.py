from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from qem_bench.baselines.zne import ZNEMitigator
from qem_bench.budget import Method, TIERS
from qem_bench.datasets.generate import generate, group_shots
from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split
from qem_bench.runner.run import (
    MethodOutput,
    MethodRegistration,
    _budget_preflight,
    _paired_budget_cell_evals,
    _run_artifact_id,
    _validation_candidate_record,
    register_method,
    registered_methods,
    run,
    unregister_method,
    validate_run_artifact,
)
from qem_bench.validation import split_spec_hash


def _only_budget_cell(results):
    assert len(results["budget"]) == 1
    return next(iter(results["budget"].values()))


def _budget_row(role, width, group, shots):
    return {
        "dataset_schema_version": "split-v2",
        "split_id": "S0",
        "split_axis": "circuit_instance",
        "partition_id": "headline",
        "axis_values": {
            "circuit_family": "tfi",
            "family_native_depth": 1,
            "noise_family": "depolarizing_readout",
            "noise_strength": "L1",
            "observable_class": ["z_mid"],
            "shots": shots,
        },
        "n_qubits": width,
        "replicate": 0,
        "stratum": "continuous_regression",
        "split": role,
        "cell_id": f"cell-{role}-{width}-{group}",
        "measurement_group": group,
        "shots": shots,
    }


def test_runner_revalidates_and_names_source_only_validation_rule(tmp_path):
    data = tmp_path / "data"
    generate("s0-t0-micro", data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_spec"]["role_domains"]["validation"] = "target"
    manifest["split_spec_hash"] = split_spec_hash(manifest["split_spec"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="role_domains must be train=source, validation=source, test=target",
    ):
        run(data, tmp_path / "run", budget_tier="H")


def test_budget_preflight_hard_fails_before_method_execution(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    out = tmp_path / "run"
    generate("s0-t0-micro", data)
    monkeypatch.setitem(TIERS, "L", 100)

    def unexpected_execution(*args, **kwargs):
        del args, kwargs
        raise AssertionError("budget preflight must precede method execution")

    monkeypatch.setattr(ZNEMitigator, "predict", unexpected_execution)
    with pytest.raises(
        ValueError,
        match=(
            r"budget-infeasible at tier L \(combined-cap\): method 'raw' "
            r"requires 1024 circuit evaluations, cap 100, shortfall 924"
        ),
    ):
        run(data, out, budget_tier="L")

    assert not (out / "results.json").exists()


def test_legacy_v1_result_is_identical_to_pre_split_runner(tmp_path):
    data = tmp_path / "data"
    manifest = generate("t0-micro", data)
    results = run(data, tmp_path / "run")
    compatibility_methods = {
        name: spec for name, spec in results["methods"].items() if name != "liao"
    }
    rounded_floating_result = {
        name: {
            "predictions": [
                round(float(value), 12) for value in spec["predictions"]
            ],
            "metrics": {
                key: round(float(value), 12)
                for key, value in spec["metrics"].items()
                if type(value) is float
            },
        }
        for name, spec in compatibility_methods.items()
    }
    digest = hashlib.sha256(
        json.dumps(
            rounded_floating_result, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    assert manifest["dataset_hash"] == (
        "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"
    )
    assert manifest["dataset_schema_version"] == "legacy-v1"
    assert results["schema_version"] == "qem-bench-run-v2"
    assert results["analysis_contract"]["metric_schema"] == (
        "qem-bench-cell-metrics-v1"
    )
    assert len({item["circuit_id"] for item in results["test_items"]}) == 6
    assert all(
        item["circuit_id"].startswith("circuit-")
        and len(item["circuit_id"]) == len("circuit-") + 64
        for item in results["test_items"]
    )
    assert list(results["methods"]) == [
        "raw",
        "ridge",
        "zne",
        "liao",
        "feat-only",
        "noisy-only",
        "shrinkage",
        "shuf-noisy",
    ]
    assert {name: spec["role"] for name, spec in results["methods"].items()} == {
        "raw": "baseline",
        "ridge": "learned",
        "zne": "qem-baseline",
        "liao": "competitor",
        "feat-only": "control",
        "noisy-only": "control",
        "shrinkage": "control",
        "shuf-noisy": "diagnostic",
    }
    assert {
        name: {
            key: value
            for key, value in spec["config"].items()
            if key != "run_artifact_identity"
        }
        for name, spec in results["methods"].items()
    } == {
        "raw": {},
        "ridge": {"best_alpha": 0.1},
        "zne": {
            "scale_factors": [1, 3, 5],
            "extrapolator": "richardson",
            "scale_one": "reused stored noisy_expectation",
            "fold_order": "optimization-level-1 transpile, then global fold",
            "sharing": "one folded execution per measurement group and scale",
            "seed_stream": "SeedSequence(master_seed, spawn_key=(2,))",
        },
        "liao": {
            "model": "random_forest",
            "tree_algorithm": "CART",
            "n_estimators": 100,
            "criterion": "squared_error",
            "min_samples_split": 2,
            "max_features": 1,
            "random_state": 7,
            "feature_fidelity": {
                "status": "restricted-feature-ablation",
                "published_liao_encoding_reproduced": False,
                "omitted_published_fields": [
                    "native-gate count vector",
                    "angle bins",
                    "sparse Pauli-observable encoding",
                ],
            },
        },
        "feat-only": {},
        "noisy-only": {},
        "shrinkage": {},
        "shuf-noisy": {},
    }
    assert {
        name: tuple(
            spec["ledger"][field] for field in ("B_train", "B_extra", "B_pred")
        )
        for name, spec in results["methods"].items()
    } == {
        "raw": (0, 0, 3072),
        "ridge": (5120, 0, 3072),
        "zne": (0, 6144, 3072),
        "liao": (5120, 0, 3072),
        "feat-only": (0, 0, 0),
        "noisy-only": (5120, 0, 3072),
        "shrinkage": (0, 0, 0),
        "shuf-noisy": (5120, 0, 3072),
    }
    assert {
        name: (spec["metrics"]["n_items"], spec["metrics"]["n_cells"])
        for name, spec in results["methods"].items()
    } == {name: (12, 4) for name in results["methods"]}
    assert digest == (
        "2508da04ee376a27894523d26b879ac41d510f04344a50f231cdab6e8368c210"
    )
    assert results["methods"]["liao"]["config"]["feature_fidelity"] == {
        "status": "restricted-feature-ablation",
        "published_liao_encoding_reproduced": False,
        "omitted_published_fields": [
            "native-gate count vector",
            "angle bins",
            "sparse Pauli-observable encoding",
        ],
    }
    assert "role_assignment" not in results
    assert "budget" not in results


def test_unversioned_run_artifact_requires_explicit_migration(tmp_path):
    data = tmp_path / "data"
    generate("t0-micro", data)
    historical = copy.deepcopy(run(data, tmp_path / "run"))
    historical.pop("dataset_schema_version")
    historical.pop("dataset_manifest_sha256")
    historical["methods"]["raw"]["config"].pop("run_artifact_identity")

    with pytest.raises(
        ValueError,
        match=(
            "unversioned qem-bench-run-v2 artifact requires explicit migration; "
            "dataset_schema_version is required"
        ),
    ):
        validate_run_artifact(historical)


def test_artifact_tier_and_phased_method_registration_contract(tmp_path):
    data = tmp_path / "data"
    spec = replace(SPLIT_PRESETS["s0-t0-micro"], budget_tier="H")
    generate_split(spec, data)
    observed: dict[str, list[str]] = {}

    class RegisteredRaw:
        def fit(self, items, *, manifest, split_v2):
            del manifest
            assert split_v2
            observed["fit"] = [item["split"] for item in items]
            return self

        def select(self, items):
            observed["select"] = [item["split"] for item in items]
            return self

        def predict(self, items, *, manifest):
            observed["manifest"] = sorted(manifest)
            observed["predict"] = [item["split"] for item in items]
            return MethodOutput(
                [item["noisy_expectation"] for item in items],
                0,
                0,
                group_shots(items),
                {"implementation": "registered-test-method"},
            )

    registration = MethodRegistration(
        "registered-raw", "competitor", RegisteredRaw, Method.RAW
    )
    register_method(registration)
    try:
        results = run(data, tmp_path / "run")
    finally:
        unregister_method(registration.name)

    assert set(observed["fit"]) == {"train"}
    assert set(observed["select"]) == {"validation"}
    assert set(observed["predict"]) == {"test"}
    assert observed["manifest"] == ["dataset_schema_version", "master_seed"]
    budget_cell = _only_budget_cell(results)
    assert budget_cell["tier"] == "H"
    assert budget_cell["cap"] == TIERS["H"]
    assert budget_cell["methods"]["registered-raw"]["status"] == "feasible"
    assert "liao" in {registration.name for registration in registered_methods()}
    assert results["methods"]["liao"]["role"] == "competitor"
    assert results["methods"]["liao"]["config"]["feature_fidelity"][
        "published_liao_encoding_reproduced"
    ] is False
    assert budget_cell["methods"]["liao"]["budget_method"] == "liao-style"
    assert results["dataset_schema_version"] == "split-v2"
    assert results["dataset_manifest_sha256"].startswith("sha256:")
    assert validate_run_artifact(results) is results

    forged = copy.deepcopy(results)
    forged.pop("role_assignment")
    forged.pop("n_validation_items")
    forged["artifact_id"] = _run_artifact_id(
        forged["dataset_hash"],
        forged["preset"],
        forged["dataset_item_stream_hashes"],
        forged["test_items"],
        forged["methods"],
        dataset_environment_contract=forged["dataset_environment_contract"],
        environment_contract=forged["environment_contract"],
        budget=forged["budget"],
    )
    with pytest.raises(ValueError, match="split-v2 run artifact has invalid fields"):
        validate_run_artifact(forged)

    extra_top = copy.deepcopy(results)
    extra_top["unauthenticated"] = True
    with pytest.raises(ValueError, match="invalid fields"):
        validate_run_artifact(extra_top)

    extra_method = copy.deepcopy(results)
    extra_method["methods"]["raw"]["unauthenticated"] = True
    with pytest.raises(ValueError, match="method 'raw' has invalid fields"):
        validate_run_artifact(extra_method)


def test_predict_boundary_refuses_test_labels_but_runner_still_scores(tmp_path):
    data = tmp_path / "data"
    generate("s0-t0-micro", data)
    observed = {}

    class LabelProbe:
        def fit(self, items, *, manifest, split_v2):
            del items, manifest
            assert split_v2
            return self

        def select(self, items):
            del items
            return self

        def predict(self, items, *, manifest):
            observed["manifest_keys"] = set(manifest)
            observed["item_keys"] = set().union(*(set(item) for item in items))
            try:
                _ = items[0]["ideal_expectation"]
            except KeyError:
                observed["label_access_refused"] = True
            return MethodOutput(np.zeros(len(items)), 0, 0, 0, {"probe": "malicious"})

    registration = MethodRegistration(
        "label-probe", "malicious", LabelProbe, None
    )
    register_method(registration)
    try:
        results = run(data, tmp_path / "run", budget_tier="H")
    finally:
        unregister_method(registration.name)

    assert observed["label_access_refused"]
    assert observed["manifest_keys"] == {"dataset_schema_version", "master_seed"}
    assert {
        "ideal_expectation",
        "label_method",
        "circuit_sidecar",
        "observable_sidecar",
        "counts_sidecar",
    }.isdisjoint(observed["item_keys"])
    targets = np.asarray(
        [item["ideal_expectation"] for item in results["test_items"]]
    )
    assert results["methods"]["label-probe"]["metrics"]["mae"] == pytest.approx(
        float(np.mean(np.abs(targets)))
    )
    assert validate_run_artifact(results) is results


def test_budget_cap_is_applied_per_paired_statistical_cell():
    train = [
        _budget_row("train", 3, "source-3", 1_300_000),
        _budget_row("train", 5, "source-5", 1_300_000),
    ]
    test = [
        _budget_row("test", 3, "test-3", 1_300_000),
        _budget_row("test", 5, "test-5", 1_300_000),
    ]
    costs_by_cell = _paired_budget_cell_evals(train, [], test)
    assert len(costs_by_cell) == 2
    raw = MethodRegistration("raw-probe", "baseline", lambda: None, Method.RAW)
    accepted = {
        cell_id: _budget_preflight(
            "L", [raw], costs.source_evals, costs.test_evals
        )
        for cell_id, costs in costs_by_cell.items()
    }
    assert sum(
        cell["methods"]["raw-probe"]["modeled_ledger"]["total"]
        for cell in accepted.values()
    ) == 2_600_000
    assert all(cell["status"] == "feasible" for cell in accepted.values())

    rejected_cost = next(
        iter(
            _paired_budget_cell_evals(
                [_budget_row("train", 3, "source", 2_500_001)],
                [],
                [_budget_row("test", 3, "test", 2_500_001)],
            ).values()
        )
    )
    with pytest.raises(
        ValueError,
        match=r"requires 2500001 circuit evaluations, cap 2500000, shortfall 1",
    ):
        _budget_preflight(
            "L", [raw], rejected_cost.source_evals, rejected_cost.test_evals
        )


def test_ridge_selection_blocks_use_physical_circuit_ids():
    class Estimator:
        def predict(self, matrix):
            del matrix
            return np.asarray([0.0, 2.0])

    class Model:
        def _matrix(self, items):
            return items

    items = [
        {
            "family": "tfi",
            "instance": 0,
            "circuit_id": "circuit-a",
            "ideal_expectation": 0.0,
            "noisy_expectation": 0.0,
        },
        {
            "family": "tfi",
            "instance": 0,
            "circuit_id": "circuit-b",
            "ideal_expectation": 0.0,
            "noisy_expectation": 0.0,
        },
    ]
    record = _validation_candidate_record(1.0, Estimator(), Model(), items)
    assert record["standard_error"] == pytest.approx(1.0)
