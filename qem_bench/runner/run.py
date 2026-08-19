"""Benchmark runner: validate the dataset, train on the train split, score every
method and control on the test split, and report accuracy, harm, and the
three-bucket circuit-evaluation ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from qem_bench.baselines.controls import (
    FeatureOnlyControl,
    NoisyOnlyControl,
    ShrinkageControl,
    ShuffledNoisyControl,
)
from qem_bench.baselines.ridge import RidgeMitigator
from qem_bench.baselines.zne import SCALE_FACTORS, ZNEMitigator
from qem_bench.datasets.generate import dataset_hash, group_shots
from qem_bench.datasets.schema import (
    FAMILY_STRATA,
    FEATURE_SPEC_VERSION,
    FEATURES,
    validate_groups,
    validate_item,
)
from qem_bench.noise import SEVERITY_GRIDS
from qem_bench.reproducibility import (
    environment_contract,
    validate_environment_contract,
)
from qem_bench.runner.metrics import (
    CELL_GROUPINGS,
    DEFAULT_CELL_GROUPING,
    build_cell_records,
    headline_metrics,
    pooled_method_metrics,
)
from qem_bench.stats.descriptive import macro_mean_iqr
from qem_bench.validation import (
    LEGACY_SCHEMA_VERSION,
    SPLIT_SCHEMA_VERSION,
    validate_split_artifact,
)

SUPPORTED_LABEL_METHODS = frozenset({"statevector", "stim"})
METRIC_SCHEMA = "qem-bench-cell-metrics-v1"
SURROGATE_ALARM_NOTE = (
    "triggered means the learned model shows no measured incremental value "
    "from the noisy measurement on this slice; treat its score as a "
    "plumbing checksum, not mitigation"
)
IDENTITY_ITEM_FIELDS = (
    "item_id",
    "measurement_group",
    "split",
    "family",
    "stratum",
    "noise_family",
    "severity",
    "observable",
    "ideal_expectation",
)


def _adapt_legacy_v1_rows(items: list[dict]) -> None:
    """Restore fields omitted by the two historical legacy serializers."""
    for item in items:
        if "stratum" in item:
            continue
        family = item.get("family")
        if family not in FAMILY_STRATA:
            raise ValueError(f"unknown circuit family {family!r}")
        item["stratum"] = FAMILY_STRATA[family]


def _load_legacy_v1(
    items: list[dict], manifest: dict
) -> tuple[list[dict], dict]:
    """Validate the isolated legacy-v1 train/test artifact shape."""
    try:
        validate_environment_contract(manifest["environment_contract"])
    except KeyError as exc:
        raise ValueError("legacy-v1 manifest requires environment_contract") from exc
    _adapt_legacy_v1_rows(items)

    current_spec = {"version": FEATURE_SPEC_VERSION, "features": list(FEATURES)}
    if manifest.get("feature_spec") != current_spec:
        raise ValueError("dataset feature_spec does not match the installed code")

    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("dataset contains duplicate item_id values")

    for item in items:
        validate_item(item)
    validate_groups(items)

    row_noise_families = {item["noise_family"] for item in items}
    if len(row_noise_families) != 1:
        raise ValueError(
            "item rows must contain exactly one noise_family; "
            f"found {sorted(row_noise_families)!r}"
        )
    row_noise_family = next(iter(row_noise_families))
    if row_noise_family not in SEVERITY_GRIDS:
        raise ValueError(f"unknown row noise_family {row_noise_family!r}")

    config = manifest.get("config")
    manifest_noise_family = (
        config.get("noise_family") if isinstance(config, dict) else None
    )
    if manifest_noise_family != row_noise_family:
        raise ValueError(
            "manifest config.noise_family does not match item rows: "
            f"manifest={manifest_noise_family!r}, rows={row_noise_family!r}"
        )

    valid_severities = SEVERITY_GRIDS[row_noise_family]
    invalid_severities = sorted(
        {
            item["severity"]
            for item in items
            if item["severity"] not in valid_severities
        }
    )
    if invalid_severities:
        raise ValueError(
            f"item row severities are invalid for {row_noise_family}: "
            f"{invalid_severities!r}"
        )
    if manifest.get("severity_grid") != valid_severities:
        raise ValueError(
            "manifest severity_grid does not match the installed registry for "
            f"{row_noise_family}"
        )
    if manifest.get("severity_grids") != SEVERITY_GRIDS:
        raise ValueError("manifest severity_grids does not match the installed registry")

    ledger = manifest.get("generation_ledger", {})
    if ledger.get("train_circuit_evals") != group_shots(items, "train") or ledger.get(
        "test_circuit_evals"
    ) != group_shots(items, "test"):
        raise ValueError("manifest generation ledger does not match item rows")
    row_methods = {item["label_method"] for item in items}
    bad_methods = row_methods - SUPPORTED_LABEL_METHODS
    if bad_methods:
        raise ValueError(f"unsupported label_method values in items: {sorted(bad_methods)}")
    label_counts = {
        method: sum(item["label_method"] == method for item in items)
        for method in sorted(SUPPORTED_LABEL_METHODS)
    }
    for method, expected in label_counts.items():
        key = f"label_evals_{method}"
        if ledger.get(key, 0) != expected:
            raise ValueError(f"manifest {key} does not match item rows")

    strata = {item["stratum"] for item in items}
    if len(strata) != 1:
        raise ValueError(
            "the Clifford control stratum is excluded from the "
            "continuous-regression headline; runner requires one stratum per dataset"
        )

    counts = manifest.get("counts", {})
    expected_counts = {
        "items": len(items),
        "train_items": sum(1 for it in items if it["split"] == "train"),
        "test_items": sum(1 for it in items if it["split"] == "test"),
        "instances": len({it["instance"] for it in items}),
        "measurement_groups": len({it["measurement_group"] for it in items}),
    }
    if counts != expected_counts:
        raise ValueError(
            f"manifest counts do not match item rows: manifest={counts}, rows={expected_counts}"
        )
    return items, manifest


def _load(data_dir: Path) -> tuple[list[dict], dict]:
    """Dispatch dataset loading by the explicit artifact schema version."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("dataset_schema_version")
    if version is None:
        raise ValueError(
            "dataset_schema_version is required; migrate unversioned artifacts explicitly"
        )
    if version == SPLIT_SCHEMA_VERSION:
        validate_split_artifact(data_dir)
        raise ValueError(
            "split-v2 runner loading is not implemented; this runner accepts legacy-v1 "
            "artifacts only"
        )
    if version != LEGACY_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset_schema_version {version!r}")

    items = [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("dataset contains duplicate item_id values")

    items.sort(key=lambda item: item["item_id"])
    actual_hash = dataset_hash(items)
    expected_hash = manifest.get("dataset_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"dataset hash mismatch: manifest={expected_hash}, items={actual_hash}"
        )

    return _load_legacy_v1(items, manifest)


def circuit_evaluation_ratio(
    realized_test_total: int, nominal_test_total: int
) -> float | None:
    """Return the realized-to-nominal test cost ratio when it is defined."""

    if nominal_test_total == 0:
        return None
    return realized_test_total / nominal_test_total


def _ledger(train_evals: int, extra_evals: int, pred_evals: int, n_test: int) -> dict:
    total = train_evals + extra_evals + pred_evals
    nominal_test_total = pred_evals
    realized_test_total = extra_evals + pred_evals
    return {
        "B_train": train_evals,
        "B_extra": extra_evals,
        "B_pred": pred_evals,
        "total": total,
        "amortized_per_test_item": total / n_test,
        "nominal_test_total": nominal_test_total,
        "realized_test_total": realized_test_total,
        "nominal_total": train_evals + nominal_test_total,
        "test_budget_ratio": circuit_evaluation_ratio(
            realized_test_total, nominal_test_total
        ),
        "circuit_evals_per_mitigated_expectation": realized_test_total / n_test,
    }


def _dataset_item_stream_hashes(manifest: dict) -> dict[str, str]:
    """Return the validated item-stream identities represented by a dataset."""

    version = manifest["dataset_schema_version"]
    if version == SPLIT_SCHEMA_VERSION:
        return {
            str(cell["cell_id"]): str(cell["item_stream_hash"])
            for cell in manifest["cells"]
        }
    if version == LEGACY_SCHEMA_VERSION:
        return {LEGACY_SCHEMA_VERSION: str(manifest["dataset_hash"])}
    raise ValueError(f"unsupported dataset_schema_version {version!r}")


def _identity_test_items(items: list[dict]) -> list[dict]:
    """Return the ordered test-item fields that determine reported results."""

    return [
        {field: item[field] for field in IDENTITY_ITEM_FIELDS}
        for item in sorted(items, key=lambda item: item["item_id"])
    ]


def _run_artifact_id(
    dataset_hash: str,
    preset: str,
    item_stream_hashes: dict[str, str],
    test_items: list[dict],
    methods: dict[str, dict],
    *,
    dataset_environment_contract: Mapping[str, object],
    environment_contract: Mapping[str, object],
) -> str:
    payload = {
        "schema": "qem-bench-run-v2",
        "dataset_hash": dataset_hash,
        "preset": preset,
        "item_stream_hashes": dict(sorted(item_stream_hashes.items())),
        "test_items": test_items,
        "analysis_contract": _analysis_contract(),
        "dataset_environment_contract": dataset_environment_contract,
        "environment_contract": environment_contract,
        "methods": {
            name: {
                "predictions": [float(value) for value in spec["predictions"]],
                "ledger": spec["ledger"],
                "role": spec["role"],
                "config": spec["config"],
            }
            for name, spec in sorted(methods.items())
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _analysis_contract() -> dict:
    return {
        "metric_schema": METRIC_SCHEMA,
        "cell_groupings": {
            name: list(fields) for name, fields in CELL_GROUPINGS.items()
        },
        "default_cell_grouping": DEFAULT_CELL_GROUPING,
    }


def _prediction_vector(method: str, spec: Mapping[str, object], n_test: int) -> np.ndarray:
    try:
        predictions = np.asarray(spec["predictions"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"run artifact method {method!r} has invalid predictions") from exc
    if predictions.ndim != 1 or len(predictions) != n_test:
        raise ValueError(
            f"run artifact method {method!r} predictions do not match n_test_items"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"run artifact method {method!r} predictions must be finite")
    return predictions


def _canonical_test_items(run_result: Mapping[str, object]) -> list[dict]:
    test_items = run_result.get("test_items")
    if not isinstance(test_items, list) or not test_items:
        raise ValueError("run artifact requires test_items")
    if not all(isinstance(item, dict) for item in test_items):
        raise ValueError("run artifact test_items must contain objects")

    try:
        items = _identity_test_items(test_items)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run artifact has invalid test item metadata") from exc
    if items != test_items or any(
        set(item) != set(IDENTITY_ITEM_FIELDS) for item in test_items
    ):
        raise ValueError("run artifact test_items must be canonical identity projections")

    n_test = run_result.get("n_test_items")
    if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test <= 0:
        raise ValueError("run artifact has invalid n_test_items")
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != n_test or len(item_ids) != len(set(item_ids)):
        raise ValueError("run artifact test_items do not match n_test_items")

    string_fields = IDENTITY_ITEM_FIELDS[:-1]
    for item in items:
        if any(
            not isinstance(item[field], str) or not item[field]
            for field in string_fields
        ):
            raise ValueError("run artifact test item identity fields must be nonempty strings")
        target = item["ideal_expectation"]
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not np.isfinite(target)
        ):
            raise ValueError("run artifact test item targets must be finite numbers")
        if item["split"] != "test":
            raise ValueError("run artifact test_items must belong to the test split")
    return items


def _require_run_match(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"run artifact derived result mismatch: {label}")


def _surrogate_alarm(methods: Mapping[str, Mapping[str, object]]) -> dict:
    ridge_mae = methods["ridge"]["metrics"]["mae"]
    feat_mae = methods["feat-only"]["metrics"]["mae"]
    return {
        "ridge_mae": ridge_mae,
        "feature_only_mae": feat_mae,
        "triggered": bool(feat_mae <= ridge_mae * 1.05),
        "note": SURROGATE_ALARM_NOTE,
    }


def validate_run_artifact(run_result: dict) -> dict:
    """Validate a serialized run identity and recompute every reported result."""

    if run_result.get("schema_version") != "qem-bench-run-v2":
        raise ValueError("unsupported runner result schema_version")
    _require_run_match(
        "analysis_contract", run_result.get("analysis_contract"), _analysis_contract()
    )
    methods = run_result.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("run artifact requires methods")
    if not isinstance(methods.get("raw"), dict):
        raise ValueError("run artifact requires a raw method")
    for name, spec in methods.items():
        if not isinstance(spec, dict):
            raise ValueError(f"run artifact method {name!r} must be an object")
        for field in ("predictions", "ledger", "role", "config"):
            if field not in spec:
                raise ValueError(f"run artifact method {name!r} lacks {field}")

    items = _canonical_test_items(run_result)
    try:
        dataset_environment_contract = validate_environment_contract(
            run_result["dataset_environment_contract"]
        )
        run_environment_contract = validate_environment_contract(
            run_result["environment_contract"]
        )
        expected_artifact_id = _run_artifact_id(
            str(run_result["dataset_hash"]),
            str(run_result["preset"]),
            dict(run_result["dataset_item_stream_hashes"]),
            items,
            methods,
            dataset_environment_contract=dataset_environment_contract,
            environment_contract=run_environment_contract,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run artifact identity inputs are invalid") from exc
    _require_run_match(
        "artifact_id", run_result.get("artifact_id"), expected_artifact_id
    )

    n_test = len(items)
    raw_predictions = _prediction_vector("raw", methods["raw"], n_test)
    targets = np.asarray([item["ideal_expectation"] for item in items], dtype=float)
    artifact_id = str(run_result["artifact_id"])
    for name, spec in methods.items():
        predictions = _prediction_vector(name, spec, n_test)
        records_by_grouping = {
            grouping: build_cell_records(
                name,
                items,
                predictions,
                raw_predictions,
                artifact_id=artifact_id,
                grouping=grouping,
            )
            for grouping in CELL_GROUPINGS
        }
        default_records = records_by_grouping[DEFAULT_CELL_GROUPING]
        expected_values = {
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
        for field, expected in expected_values.items():
            _require_run_match(f"methods.{name}.{field}", spec.get(field), expected)

    _require_run_match(
        "surrogate_alarm", run_result.get("surrogate_alarm"), _surrogate_alarm(methods)
    )
    return run_result


def run(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items, manifest = _load(data_dir)
    dataset_environment_contract = validate_environment_contract(
        manifest["environment_contract"]
    )
    run_environment_contract = environment_contract()
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    if not train or not test:
        raise ValueError("dataset must contain both train and test items")

    r = np.array([it["noisy_expectation"] for it in test])
    y = np.array([it["ideal_expectation"] for it in test])
    n_test = len(test)
    train_group_evals = group_shots(items, "train")
    test_group_evals = group_shots(items, "test")

    ridge = RidgeMitigator().fit(train)
    zne = ZNEMitigator().fit(train)
    zne_seed_stream = np.random.SeedSequence(
        int(manifest["master_seed"]), spawn_key=(2,)
    )
    zne_predictions, zne_extra_per_group = zne.predict(
        test, seed_stream=zne_seed_stream
    )
    zne_extra_evals = int(np.sum(zne_extra_per_group, dtype=np.int64))

    # Required surrogate and leakage controls (frozen design). Cost semantics:
    # feature-only and shrinkage consume no noisy measurements at all (their only
    # inputs are structure features and exact train labels, which are logged
    # separately); noisy-only and shuffled-noisy consume the same measurement groups
    # as the primary ridge. shuffled-noisy is a diagnostic, never a deployable method.
    controls = {
        "feat-only": (FeatureOnlyControl().fit(train), _ledger(0, 0, 0, n_test), "control"),
        "noisy-only": (
            NoisyOnlyControl().fit(train),
            _ledger(train_group_evals, 0, test_group_evals, n_test),
            "control",
        ),
        "shrinkage": (ShrinkageControl().fit(train), _ledger(0, 0, 0, n_test), "control"),
        "shuf-noisy": (
            ShuffledNoisyControl().fit(train),
            _ledger(train_group_evals, 0, test_group_evals, n_test),
            "diagnostic",
        ),
    }

    methods: dict[str, dict] = {
        "raw": {
            "predictions": r,
            "ledger": _ledger(0, 0, test_group_evals, n_test),
            "role": "baseline",
            "config": {},
        },
        "ridge": {
            "predictions": ridge.predict(test),
            "ledger": _ledger(train_group_evals, 0, test_group_evals, n_test),
            "role": "learned",
            "config": {"best_alpha": ridge.best_alpha_},
        },
        "zne": {
            "predictions": zne_predictions,
            "ledger": _ledger(0, zne_extra_evals, test_group_evals, n_test),
            "role": "qem-baseline",
            "config": {
                "scale_factors": list(SCALE_FACTORS),
                "extrapolator": zne.extrapolator,
                "scale_one": "reused stored noisy_expectation",
                "fold_order": "optimization-level-1 transpile, then global fold",
                "sharing": "one folded execution per measurement group and scale",
                "seed_stream": "SeedSequence(master_seed, spawn_key=(2,))",
            },
        },
    }
    for name, (model, ledger, role) in controls.items():
        methods[name] = {
            "predictions": model.predict(test),
            "ledger": ledger,
            "role": role,
            "config": {},
        }

    item_stream_hashes = _dataset_item_stream_hashes(manifest)
    test_items = _identity_test_items(test)
    artifact_id = _run_artifact_id(
        manifest["dataset_hash"],
        manifest["preset"],
        item_stream_hashes,
        test_items,
        methods,
        dataset_environment_contract=dataset_environment_contract,
        environment_contract=run_environment_contract,
    )

    results: dict = {
        "schema_version": "qem-bench-run-v2",
        "artifact_id": artifact_id,
        "analysis_contract": _analysis_contract(),
        "dataset_environment_contract": dataset_environment_contract,
        "environment_contract": run_environment_contract,
        "dataset_hash": manifest["dataset_hash"],
        "dataset_item_stream_hashes": item_stream_hashes,
        "test_items": test_items,
        "preset": manifest["preset"],
        "feature_spec": manifest["feature_spec"],
        "stratum": items[0]["stratum"],
        "n_train_items": len(train),
        "n_test_items": n_test,
        "label_evals": {
            method: int(
                manifest["generation_ledger"].get(f"label_evals_{method}", 0)
            )
            for method in sorted(SUPPORTED_LABEL_METHODS)
        },
        "methods": {},
    }
    results["label_evals_statevector"] = results["label_evals"]["statevector"]
    results["label_evals_stim"] = results["label_evals"]["stim"]
    for name, spec in methods.items():
        predictions = np.asarray(spec["predictions"], dtype=float)
        records_by_grouping = {
            grouping: build_cell_records(
                name,
                test,
                predictions,
                r,
                artifact_id=artifact_id,
                grouping=grouping,
            )
            for grouping in CELL_GROUPINGS
        }
        default_records = records_by_grouping[DEFAULT_CELL_GROUPING]
        results["methods"][name] = {
            "predictions": predictions.tolist(),
            "cell_grouping": DEFAULT_CELL_GROUPING,
            "cell_records": default_records,
            "macro": macro_mean_iqr(default_records),
            "metrics": headline_metrics(default_records),
            "pooled_diagnostic": pooled_method_metrics(predictions, r, y),
            "cell_grouping_sensitivity": {
                grouping: {
                    "n_cells": len(records),
                    "macro": macro_mean_iqr(records),
                    "headline_metrics": headline_metrics(records),
                }
                for grouping, records in records_by_grouping.items()
            },
            "ledger": spec["ledger"],
            "role": spec["role"],
            "config": spec["config"],
        }

    # Surrogate alarm (frozen design): if removing the noisy measurement leaves
    # accuracy intact, the model class is emulating the simulator on this slice.
    results["surrogate_alarm"] = _surrogate_alarm(results["methods"])

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    cols = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("signed_bias", "Bias"),
        ("excess_loss_total", "ExcessLoss(sum)"),
        ("overcorrection_rate", "OCR"),
        ("physicality_violation_rate", "PhysViol"),
    ]
    print(f"\nartifact {results['artifact_id'][7:19]}  "
          f"dataset {results['dataset_hash'][:12]}  preset {results['preset']}  "
          f"test items {results['n_test_items']}")
    header = f"{'method':<11}{'role':<13}" + "".join(f"{label:>16}" for _, label in cols)
    print(header)
    for name, spec in results["methods"].items():
        met = spec["metrics"]
        row = f"{name:<11}{spec['role']:<13}" + "".join(
            f"{met[key]:>16.6f}" for key, _ in cols
        )
        print(row)
    print(f"\n{'method':<11}{'B_train':>12}{'B_extra':>12}{'B_pred':>12}{'nominal':>12}"
          f"{'realized':>12}{'ratio':>9}{'eval/EV':>11}")
    for name, spec in results["methods"].items():
        led = spec["ledger"]
        ratio = led["test_budget_ratio"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.2f}"
        print(f"{name:<11}{led['B_train']:>12}{led['B_extra']:>12}{led['B_pred']:>12}"
              f"{led['nominal_total']:>12}{led['total']:>12}{ratio_text:>9}"
              f"{led['circuit_evals_per_mitigated_expectation']:>11.1f}")
    label_summary = ", ".join(
        f"{method}={count}" for method, count in results["label_evals"].items()
    )
    print(f"\nexact labels ({label_summary}; logged separately)")
    alarm = results["surrogate_alarm"]
    state = "TRIGGERED" if alarm["triggered"] else "clear"
    print(f"surrogate alarm: {state} (ridge MAE {alarm['ridge_mae']:.6f} vs "
          f"feature-only {alarm['feature_only_mae']:.6f})")
