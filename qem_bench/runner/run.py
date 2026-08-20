"""Benchmark runner with validated roles and binding circuit-evaluation caps.

Methods fit on train, select on source validation for split-v2, and report only
on test. Every tiered run is preflighted against the budget model before method
execution, then its realized three-bucket ledger is checked against that model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qem_bench.baselines.controls import (
    FeatureOnlyControl,
    NoisyOnlyControl,
    ShrinkageControl,
    ShuffledNoisyControl,
)
from qem_bench.baselines.liao import (
    LiaoMitigator,
    LiaoRandomForestMitigator,
)
from qem_bench.baselines.ridge import ALPHA_GRID, RidgeMitigator
from qem_bench.baselines.zne import SCALE_FACTORS, ZNEMitigator
from qem_bench.budget import (
    BudgetInputs,
    Method,
    TIERS,
    TierMode,
    TierPair,
    check_constraints,
    method_budget,
    minimum_tier_constant,
)
from qem_bench.datasets.generate import (
    PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD,
    UNKNOWN_IDENTITY_ENCODING_PROFILE,
    dataset_hash,
    group_shots,
    legacy_physical_identity_encoding_profiles,
    validate_physical_identity_encoding_profiles,
)
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
    normalized_bootstrap_ids,
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
    "circuit_id",
    "bootstrap_stratum_id",
    "ideal_expectation",
)
PREDICTION_ITEM_FIELDS = frozenset(
    {
        "item_id",
        "split",
        "family",
        "instance",
        "n_qubits",
        "circuit_seed",
        "observable",
        "pauli_label",
        "obs_locality",
        "noise_family",
        "severity",
        "shots",
        "measurement_group",
        "noisy_expectation",
        "noisy_stderr",
        "two_qubit_gates",
        "transpiled_depth",
        "steps",
        "j",
        "h",
        "jx",
        "jy",
        "jz",
        "dt",
        "p",
        "graph_class",
        "edges",
        "edge_probability",
        "gammas",
        "betas",
        "depth",
        "non_clifford_count",
        "theta",
    }
)
PREDICTION_MANIFEST_FIELDS = frozenset(
    {"dataset_schema_version", "master_seed"}
)
BUDGET_MODE = TierMode.COMBINED_CAP
BUDGET_ENFORCEMENT = "hard-fail"
RUN_IDENTITY_CONFIG_KEY = "run_artifact_identity"
RUN_IDENTITY_FIELDS = frozenset(
    {
        "dataset_schema_version",
        "dataset_manifest_sha256",
        "feature_spec",
        "stratum",
        "n_train_items",
        "n_validation_items",
        "n_test_items",
        "label_evals",
    }
)
RUN_PROFILE_FIELDS = frozenset({PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD})
RUN_METHOD_FIELDS = frozenset(
    {
        "predictions",
        "cell_grouping",
        "cell_records",
        "macro",
        "metrics",
        "pooled_diagnostic",
        "cell_grouping_sensitivity",
        "ledger",
        "role",
        "config",
    }
)
RUN_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "analysis_contract",
        "dataset_schema_version",
        "dataset_manifest_sha256",
        "dataset_environment_contract",
        "environment_contract",
        "dataset_hash",
        "dataset_item_stream_hashes",
        "test_items",
        "preset",
        "feature_spec",
        "stratum",
        "n_train_items",
        "n_test_items",
        "label_evals",
        "methods",
        "label_evals_statevector",
        "label_evals_stim",
        "surrogate_alarm",
    }
)
RUN_SPLIT_FIELDS = frozenset({"n_validation_items", "role_assignment", "budget"})
_BUDGET_INPUT_FIELDS = (
    "n_train_circuits",
    "n_test_circuits",
    "shots",
    "n_scales",
    "m",
    "groups",
)


@dataclass(frozen=True)
class MethodOutput:
    """One registered method's predictions and independently realized ledger."""

    predictions: Sequence[float]
    B_train: int
    B_extra: int
    B_pred: int
    config: Mapping[str, object]


@dataclass(frozen=True)
class MethodRegistration:
    """Runner plugin contract for one prediction method.

    ``factory`` returns a fresh object with three ordered methods:
    ``fit(train_items, manifest=..., split_v2=...)``,
    ``select(validation_items)``, and
    ``predict(test_items, manifest=...) -> MethodOutput``. The runner alone
    assigns role lists to these phases. ``budget_method=None`` is reserved for
    methods that realize an all-zero circuit-evaluation ledger.
    """

    name: str
    role: str
    factory: Callable[[], object]
    budget_method: Method | str | None
    n_scales: int = 1
    m: int = 0


@dataclass(frozen=True)
class _PairedBudgetCellCosts:
    source_evals: int
    test_evals: int
    pairing_descriptor: Mapping[str, object]
    source_cell_ids: tuple[str, ...]
    test_cell_ids: tuple[str, ...]


_METHOD_REGISTRY: dict[str, MethodRegistration] = {}


def register_method(
    registration: MethodRegistration, *, replace: bool = False
) -> None:
    """Register a method without modifying the runner's execution loop."""

    if not isinstance(registration, MethodRegistration):
        raise TypeError("registration must be a MethodRegistration")
    if not registration.name or not registration.role:
        raise ValueError("registered method name and role must be nonempty")
    if registration.budget_method is not None:
        Method(registration.budget_method)
    method_budget(
        Method.RAW,
        BudgetInputs(0, 1, 1, registration.n_scales, registration.m, 1),
    )
    if registration.name in _METHOD_REGISTRY and not replace:
        raise ValueError(f"runner method {registration.name!r} is already registered")
    _METHOD_REGISTRY[registration.name] = registration


def registered_methods() -> tuple[MethodRegistration, ...]:
    """Return the active method registrations in deterministic run order."""

    return tuple(_METHOD_REGISTRY.values())


def unregister_method(name: str) -> MethodRegistration:
    """Remove and return one registration, primarily for isolated callers/tests."""

    try:
        return _METHOD_REGISTRY.pop(name)
    except KeyError as exc:
        raise ValueError(f"runner method {name!r} is not registered") from exc


def _adapt_legacy_v1_rows(items: list[dict]) -> None:
    """Restore fields omitted by the two historical legacy serializers."""
    for item in items:
        if "stratum" in item:
            continue
        family = item.get("family")
        if family not in FAMILY_STRATA:
            raise ValueError(f"unknown circuit family {family!r}")
        item["stratum"] = FAMILY_STRATA[family]


def _normalize_dataset_identity_encoding_profiles(
    items: list[dict], manifest: dict
) -> None:
    families = {str(item["family"]) for item in items}
    declared = PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD in manifest
    if declared:
        profiles = validate_physical_identity_encoding_profiles(
            manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD],
            families=families,
        )
    else:
        profiles = {
            family: UNKNOWN_IDENTITY_ENCODING_PROFILE
            for family in sorted(families)
        }

    if declared and manifest["dataset_schema_version"] == LEGACY_SCHEMA_VERSION:
        expected = legacy_physical_identity_encoding_profiles(
            manifest.get("preset"), manifest.get("config")
        )
        if profiles != expected:
            raise ValueError(
                "legacy dataset physical identity encoding profiles do not match "
                f"the generator serializer: declared={profiles}, expected={expected}"
            )
    manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = profiles


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
        # Legacy loader: the version dispatcher upstream already rejected any
        # other version, and the six shipped legacy-v1 examples carry no
        # dataset_schema_version key at all, so read it from the constant.
        validate_item(item, schema_version=LEGACY_SCHEMA_VERSION)
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
    _normalize_dataset_identity_encoding_profiles(items, manifest)
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
        items, manifest = validate_split_artifact(data_dir)
        _normalize_dataset_identity_encoding_profiles(items, manifest)
        return items, manifest
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


def _validate_ledger(method: str, value: object, n_test: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"run artifact method {method!r} ledger must be an object")
    for field in ("B_train", "B_extra", "B_pred"):
        field_value = value.get(field)
        if type(field_value) is not int or field_value < 0:
            raise ValueError(
                f"run artifact method {method!r} ledger {field} "
                "must be a nonnegative integer"
            )
    expected = _ledger(
        value["B_train"], value["B_extra"], value["B_pred"], n_test
    )
    try:
        actual_json = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected_json = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"run artifact method {method!r} ledger is not canonical JSON"
        ) from exc
    if actual_json != expected_json:
        raise ValueError(
            f"run artifact method {method!r} ledger does not match its buckets"
        )


def _item_ids_hash(items: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        sorted(str(item["item_id"]) for item in items),
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _role_assignment(
    train: list[dict], validation: list[dict], test: list[dict]
) -> dict[str, object]:
    roles = {"train": train, "validation": validation, "test": test}
    item_ids = {
        role: {str(item["item_id"]) for item in role_items}
        for role, role_items in roles.items()
    }
    overlaps = {
        "train_validation": len(item_ids["train"] & item_ids["validation"]),
        "train_test": len(item_ids["train"] & item_ids["test"]),
        "validation_test": len(item_ids["validation"] & item_ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"role assignment leakage: item_id overlap {overlaps}")
    return {
        "item_counts": {role: len(role_items) for role, role_items in roles.items()},
        "item_ids_sha256": {
            role: _item_ids_hash(role_items) for role, role_items in roles.items()
        },
        "fit_role": "train",
        "selection_role": "validation",
        "report_role": "test",
        "test_items_used_for_selection": 0,
        "pairwise_item_id_overlap": overlaps,
    }


def _role_protocol(assignment: Mapping[str, object]) -> dict[str, object]:
    hashes = assignment["item_ids_sha256"]
    counts = assignment["item_counts"]
    return {
        "fit_role": "train",
        "fit_items": counts["train"],
        "fit_item_ids_sha256": hashes["train"],
        "selection_role": "validation",
        "selection_items": counts["validation"],
        "selection_item_ids_sha256": hashes["validation"],
        "report_role": "test",
        "test_items_used_for_selection": 0,
    }


def _ridge_training_items(
    model: RidgeMitigator, train_items: list[dict]
) -> list[dict]:
    if not isinstance(model, ShuffledNoisyControl):
        return train_items
    rng = np.random.default_rng(model.shuffle_seed)
    shuffled = [dict(item) for item in train_items]
    column = np.asarray([item["noisy_expectation"] for item in shuffled], dtype=float)
    rng.shuffle(column)
    for item, value in zip(shuffled, column):
        item["noisy_expectation"] = float(value)
    return shuffled


def _validation_candidate_record(
    alpha: float,
    estimator: Pipeline,
    model: RidgeMitigator,
    validation_items: list[dict],
) -> dict[str, object]:
    predictions = estimator.predict(model._matrix(validation_items))
    targets = np.asarray(
        [item["ideal_expectation"] for item in validation_items], dtype=float
    )
    raw = np.asarray(
        [item["noisy_expectation"] for item in validation_items], dtype=float
    )
    errors = np.abs(predictions - targets)
    by_circuit: dict[str, list[float]] = {}
    for item, error in zip(validation_items, errors):
        key = item.get("circuit_id", item.get("measurement_group"))
        if not isinstance(key, str) or not key:
            raise ValueError(
                "validation items require circuit_id or measurement_group"
            )
        by_circuit.setdefault(key, []).append(float(error))
    circuit_mae = np.asarray(
        [np.mean(values) for _, values in sorted(by_circuit.items())], dtype=float
    )
    standard_error = (
        float(np.std(circuit_mae, ddof=1) / np.sqrt(len(circuit_mae)))
        if len(circuit_mae) > 1
        else 0.0
    )
    return {
        "alpha": float(alpha),
        "macro_mae": float(np.mean(circuit_mae)),
        "standard_error": standard_error,
        "excess_absolute_loss_total": float(
            np.sum(errors - np.abs(raw - targets), dtype=float)
        ),
        "estimator": estimator,
    }


class _RawRunnerMethod:
    def fit(
        self, train_items: list[dict], *, manifest: Mapping[str, object], split_v2: bool
    ) -> "_RawRunnerMethod":
        del train_items, manifest, split_v2
        return self

    def select(self, validation_items: list[dict]) -> "_RawRunnerMethod":
        del validation_items
        return self

    def predict(
        self, test_items: list[dict], *, manifest: Mapping[str, object]
    ) -> MethodOutput:
        del manifest
        return MethodOutput(
            [item["noisy_expectation"] for item in test_items],
            0,
            0,
            group_shots(test_items),
            {},
        )


class _RidgeRunnerMethod:
    def __init__(
        self,
        model_factory: Callable[[], RidgeMitigator],
        *,
        source_measurements: bool,
        test_measurements: bool,
        legacy_reports_alpha: bool,
    ) -> None:
        self._model = model_factory()
        self._source_measurements = source_measurements
        self._test_measurements = test_measurements
        self._legacy_reports_alpha = legacy_reports_alpha
        self._split_v2 = False
        self._train_items: list[dict] = []
        self._validation_items: list[dict] = []
        self._candidates: list[dict[str, object]] = []
        self._selected: Pipeline | None = None
        self._selection: dict[str, object] | None = None

    def fit(
        self, train_items: list[dict], *, manifest: Mapping[str, object], split_v2: bool
    ) -> "_RidgeRunnerMethod":
        del manifest
        self._split_v2 = split_v2
        self._train_items = list(train_items)
        if not split_v2:
            self._model.fit(train_items)
            return self

        fit_items = _ridge_training_items(self._model, train_items)
        x_train = self._model._matrix(fit_items)
        y_train = np.asarray(
            [item["ideal_expectation"] for item in fit_items], dtype=float
        )
        for alpha in ALPHA_GRID:
            estimator = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "ridge",
                        Ridge(alpha=alpha, random_state=self._model.random_state),
                    ),
                ]
            )
            estimator.fit(x_train, y_train)
            self._candidates.append({"alpha": float(alpha), "estimator": estimator})
        return self

    def select(self, validation_items: list[dict]) -> "_RidgeRunnerMethod":
        if not self._split_v2:
            return self
        if not validation_items:
            raise ValueError("split-v2 method selection requires validation items")
        self._validation_items = list(validation_items)
        records = [
            _validation_candidate_record(
                float(candidate["alpha"]),
                candidate["estimator"],
                self._model,
                validation_items,
            )
            for candidate in self._candidates
        ]
        best = min(records, key=lambda value: (value["macro_mae"], -value["alpha"]))
        threshold = float(best["macro_mae"] + best["standard_error"])
        eligible = [
            record for record in records if float(record["macro_mae"]) <= threshold
        ]
        selected = min(
            eligible,
            key=lambda value: (
                value["excess_absolute_loss_total"],
                -value["alpha"],
            ),
        )
        self._selected = selected["estimator"]
        selected_alpha = float(selected["alpha"])
        self._selection = {
            "rule": "validation macro-MAE one-standard-error; then lower excess "
            "absolute loss, equal cost, then larger alpha",
            "one_standard_error_threshold": threshold,
            "selected_alpha": selected_alpha,
            "candidates": [
                {
                    key: value
                    for key, value in record.items()
                    if key != "estimator"
                }
                | {"selected": float(record["alpha"]) == selected_alpha}
                for record in records
            ],
        }
        return self

    def predict(
        self, test_items: list[dict], *, manifest: Mapping[str, object]
    ) -> MethodOutput:
        del manifest
        if self._split_v2:
            if self._selected is None or self._selection is None:
                raise RuntimeError("select validation data before prediction")
            predictions = self._selected.predict(self._model._matrix(test_items))
            config: dict[str, object] = {
                "best_alpha": self._selection["selected_alpha"],
                "validation_selection": self._selection,
            }
        else:
            predictions = self._model.predict(test_items)
            config = (
                {"best_alpha": self._model.best_alpha_}
                if self._legacy_reports_alpha
                else {}
            )
        source_items = self._train_items + self._validation_items
        return MethodOutput(
            predictions,
            group_shots(source_items) if self._source_measurements else 0,
            0,
            group_shots(test_items) if self._test_measurements else 0,
            config,
        )


class _ZNERunnerMethod:
    def __init__(self) -> None:
        self._model = ZNEMitigator()
        self._split_v2 = False

    def fit(
        self, train_items: list[dict], *, manifest: Mapping[str, object], split_v2: bool
    ) -> "_ZNERunnerMethod":
        del manifest
        self._split_v2 = split_v2
        self._model.fit(train_items)
        return self

    def select(self, validation_items: list[dict]) -> "_ZNERunnerMethod":
        del validation_items
        return self

    def predict(
        self, test_items: list[dict], *, manifest: Mapping[str, object]
    ) -> MethodOutput:
        seed_stream = np.random.SeedSequence(int(manifest["master_seed"]), spawn_key=(2,))
        predictions, extra_per_group = self._model.predict(
            test_items, seed_stream=seed_stream
        )
        config: dict[str, object] = {
            "scale_factors": list(SCALE_FACTORS),
            "extrapolator": self._model.extrapolator,
            "scale_one": "reused stored noisy_expectation",
            "fold_order": "optimization-level-1 transpile, then global fold",
            "sharing": "one folded execution per measurement group and scale",
            "seed_stream": "SeedSequence(master_seed, spawn_key=(2,))",
        }
        if self._split_v2:
            config["validation_selection"] = {
                "rule": "predeclared fixed B1 protocol",
                "selected_extrapolator": self._model.extrapolator,
            }
        return MethodOutput(
            predictions,
            0,
            int(np.sum(extra_per_group, dtype=np.int64)),
            group_shots(test_items),
            config,
        )


class _LiaoRunnerMethod:
    def __init__(self) -> None:
        self._split_v2 = False
        self._random_state = 0
        self._train_items: list[dict] = []
        self._validation_items: list[dict] = []
        self._model: LiaoMitigator | LiaoRandomForestMitigator | None = None

    def fit(
        self, train_items: list[dict], *, manifest: Mapping[str, object], split_v2: bool
    ) -> "_LiaoRunnerMethod":
        self._split_v2 = split_v2
        self._random_state = int(manifest["master_seed"])
        self._train_items = list(train_items)
        if not split_v2:
            self._model = LiaoRandomForestMitigator(
                random_state=self._random_state
            ).fit(self._train_items)
        return self

    def select(self, validation_items: list[dict]) -> "_LiaoRunnerMethod":
        if not self._split_v2:
            return self
        if not validation_items:
            raise ValueError("split-v2 Liao selection requires validation items")
        self._validation_items = list(validation_items)
        self._model = LiaoMitigator(random_state=self._random_state).fit(
            self._train_items, self._validation_items
        )
        return self

    def predict(
        self, test_items: list[dict], *, manifest: Mapping[str, object]
    ) -> MethodOutput:
        del manifest
        if self._model is None:
            raise RuntimeError("fit and select source data before Liao prediction")
        config = dict(self._model.config_)
        config["feature_fidelity"] = {
            "status": "restricted-feature-ablation",
            "published_liao_encoding_reproduced": False,
            "omitted_published_fields": [
                "native-gate count vector",
                "angle bins",
                "sparse Pauli-observable encoding",
            ],
        }
        source_items = self._train_items + self._validation_items
        return MethodOutput(
            self._model.predict(test_items),
            group_shots(source_items),
            0,
            group_shots(test_items),
            config,
        )


class _ShrinkageRunnerMethod:
    def __init__(self) -> None:
        self._model = ShrinkageControl()

    def fit(
        self, train_items: list[dict], *, manifest: Mapping[str, object], split_v2: bool
    ) -> "_ShrinkageRunnerMethod":
        del manifest, split_v2
        self._model.fit(train_items)
        return self

    def select(self, validation_items: list[dict]) -> "_ShrinkageRunnerMethod":
        del validation_items
        return self

    def predict(
        self, test_items: list[dict], *, manifest: Mapping[str, object]
    ) -> MethodOutput:
        del manifest
        return MethodOutput(self._model.predict(test_items), 0, 0, 0, {})


def _register_builtin_methods() -> None:
    registrations = (
        MethodRegistration("raw", "baseline", _RawRunnerMethod, Method.RAW),
        MethodRegistration(
            "ridge",
            "learned",
            lambda: _RidgeRunnerMethod(
                RidgeMitigator,
                source_measurements=True,
                test_measurements=True,
                legacy_reports_alpha=True,
            ),
            Method.RIDGE,
        ),
        MethodRegistration(
            "zne",
            "qem-baseline",
            _ZNERunnerMethod,
            Method.LOCAL_DIGITAL_ZNE,
            n_scales=len(SCALE_FACTORS),
        ),
        MethodRegistration(
            "liao",
            "competitor",
            _LiaoRunnerMethod,
            Method.LIAO,
        ),
        MethodRegistration(
            "feat-only",
            "control",
            lambda: _RidgeRunnerMethod(
                FeatureOnlyControl,
                source_measurements=False,
                test_measurements=False,
                legacy_reports_alpha=False,
            ),
            None,
        ),
        MethodRegistration(
            "noisy-only",
            "control",
            lambda: _RidgeRunnerMethod(
                NoisyOnlyControl,
                source_measurements=True,
                test_measurements=True,
                legacy_reports_alpha=False,
            ),
            Method.RIDGE,
        ),
        MethodRegistration(
            "shrinkage", "control", _ShrinkageRunnerMethod, None
        ),
        MethodRegistration(
            "shuf-noisy",
            "diagnostic",
            lambda: _RidgeRunnerMethod(
                ShuffledNoisyControl,
                source_measurements=True,
                test_measurements=True,
                legacy_reports_alpha=False,
            ),
            Method.RIDGE,
        ),
    )
    for registration in registrations:
        register_method(registration)


def _budget_pairing_descriptor(item: Mapping[str, object]) -> dict[str, object]:
    if item.get("dataset_schema_version") != SPLIT_SCHEMA_VERSION:
        return {
            "contract": "legacy-v1-single-cell",
            "stratum": item.get("stratum"),
        }
    axis_values = item.get("axis_values")
    split_axis = item.get("split_axis")
    if not isinstance(axis_values, Mapping) or not isinstance(split_axis, str):
        raise ValueError("split-v2 budget accounting requires axis_values and split_axis")
    return {
        "contract": "paired-budget-cell-v1",
        "dataset_schema_version": SPLIT_SCHEMA_VERSION,
        "split_id": item.get("split_id"),
        "split_axis": split_axis,
        "partition_id": item.get("partition_id"),
        "fixed_axis_values": {
            str(key): copy.deepcopy(value)
            for key, value in sorted(axis_values.items())
            if key != split_axis
        },
        "n_qubits": item.get("n_qubits"),
        "replicate": item.get("replicate"),
        "stratum": item.get("stratum"),
    }


def _budget_cell_id_from_descriptor(descriptor: Mapping[str, object]) -> str:
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "budget-cell-" + hashlib.sha256(encoded).hexdigest()


def _budget_cell_id(item: Mapping[str, object]) -> str:
    return _budget_cell_id_from_descriptor(_budget_pairing_descriptor(item))


def _paired_budget_cell_evals(
    train: list[dict], validation: list[dict], test: list[dict]
) -> dict[str, _PairedBudgetCellCosts]:
    rows_by_cell: dict[str, dict[str, list[dict]]] = {}
    descriptors: dict[str, dict[str, object]] = {}
    for role, role_items in (
        ("source", [*train, *validation]),
        ("test", test),
    ):
        for item in role_items:
            budget_cell_id = _budget_cell_id(item)
            descriptor = _budget_pairing_descriptor(item)
            previous = descriptors.setdefault(budget_cell_id, descriptor)
            if previous != descriptor:
                raise ValueError("budget cell identifier collision")
            rows_by_cell.setdefault(
                budget_cell_id, {"source": [], "test": []}
            )[role].append(item)

    costs: dict[str, _PairedBudgetCellCosts] = {}
    for budget_cell_id, role_rows in sorted(rows_by_cell.items()):
        if not role_rows["source"] or not role_rows["test"]:
            raise ValueError(
                f"budget cell {budget_cell_id} does not pair source and test rows"
            )
        costs[budget_cell_id] = _PairedBudgetCellCosts(
            source_evals=group_shots(role_rows["source"]),
            test_evals=group_shots(role_rows["test"]),
            pairing_descriptor=copy.deepcopy(descriptors[budget_cell_id]),
            source_cell_ids=tuple(
                sorted({str(item["cell_id"]) for item in role_rows["source"]})
            ),
            test_cell_ids=tuple(
                sorted({str(item["cell_id"]) for item in role_rows["test"]})
            ),
        )
    return costs


def _budget_inputs(
    registration: MethodRegistration, source_evals: int, test_evals: int
) -> BudgetInputs:
    return BudgetInputs(
        source_evals,
        test_evals,
        1,
        registration.n_scales,
        registration.m,
        1,
    )


def _budget_record(
    registration: MethodRegistration,
    source_evals: int,
    test_evals: int,
    tier_pair: TierPair,
) -> dict[str, object]:
    if registration.budget_method is None:
        return {
            "budget_method": None,
            "inputs": None,
            "modeled_ledger": {"B_train": 0, "B_extra": 0, "B_pred": 0, "total": 0},
            "required_combined_cap": 0,
            "shortfall": 0,
            "method_feasible": True,
            "statistical_feasible": True,
            "status": "feasible",
        }

    method = Method(registration.budget_method)
    inputs = _budget_inputs(registration, source_evals, test_evals)
    modeled = method_budget(method, inputs)
    check = check_constraints(method, inputs, tier_pair, BUDGET_MODE)
    required = minimum_tier_constant(method, inputs).combined_joint
    cap = tier_pair.combined_cap
    return {
        "budget_method": method.value,
        "inputs": {field: getattr(inputs, field) for field in _BUDGET_INPUT_FIELDS},
        "modeled_ledger": {
            "B_train": modeled.B_train,
            "B_extra": modeled.B_extra,
            "B_pred": modeled.B_pred,
            "total": modeled.total,
        },
        "required_combined_cap": required,
        "shortfall": max(0, required - cap),
        "method_feasible": check.method_feasible,
        "statistical_feasible": check.statistical_feasible,
        "status": check.status,
    }


def _budget_preflight(
    tier: str,
    registrations: Sequence[MethodRegistration],
    source_evals: int,
    test_evals: int,
) -> dict[str, object]:
    tier_pair = TierPair.declared(tier)
    cap = TIERS[tier]
    if tier_pair.combined_cap != cap:
        raise ValueError(f"tier {tier!r} does not resolve consistently")
    method_records = {
        registration.name: _budget_record(
            registration, source_evals, test_evals, tier_pair
        )
        for registration in registrations
    }
    violations = [
        (name, record)
        for name, record in method_records.items()
        if record["status"] != "feasible"
    ]
    if violations:
        details = "; ".join(
            f"method {name!r} requires {record['required_combined_cap']} circuit "
            f"evaluations, cap {cap}, shortfall {record['shortfall']}"
            for name, record in violations
        )
        raise ValueError(
            f"budget-infeasible at tier {tier} ({BUDGET_MODE.value}): {details}"
        )
    return {
        "tier": tier,
        "mode": BUDGET_MODE.value,
        "cap": cap,
        "enforcement": BUDGET_ENFORCEMENT,
        "status": "feasible",
        "normalization": (
            "one paired statistical cell represented as unit-shot BudgetInputs; "
            "validation is charged to B_train"
        ),
        "source_base_evals": source_evals,
        "test_base_evals": test_evals,
        "methods": method_records,
    }


def _prediction_items(test_items: Sequence[Mapping[str, object]]) -> list[dict]:
    return [
        {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if key in PREDICTION_ITEM_FIELDS and key != "ideal_expectation"
        }
        for item in test_items
    ]


def _prediction_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: copy.deepcopy(manifest[key])
        for key in PREDICTION_MANIFEST_FIELDS
    }


def _execute_registered_methods(
    registrations: Sequence[MethodRegistration],
    train: list[dict],
    validation: list[dict],
    test: list[dict],
    manifest: dict,
    assignment: Mapping[str, object] | None,
) -> dict[str, dict]:
    split_v2 = manifest["dataset_schema_version"] == SPLIT_SCHEMA_VERSION
    methods: dict[str, dict] = {}
    for registration in registrations:
        protocol = registration.factory()
        try:
            fitted = protocol.fit(train, manifest=manifest, split_v2=split_v2)
            if fitted is not protocol:
                raise ValueError("fit must return the registered method instance")
            if split_v2:
                selected = protocol.select(validation)
                if selected is not protocol:
                    raise ValueError("select must return the registered method instance")
            prediction_items = _prediction_items(test)
            output = protocol.predict(
                prediction_items,
                manifest=_prediction_manifest(manifest),
            )
        except AttributeError as exc:
            raise TypeError(
                f"runner method {registration.name!r} does not implement the phased "
                "fit/select/predict contract"
            ) from exc
        if not isinstance(output, MethodOutput):
            raise TypeError(
                f"runner method {registration.name!r} predict must return MethodOutput"
            )
        buckets = (output.B_train, output.B_extra, output.B_pred)
        if any(type(value) is not int or value < 0 for value in buckets):
            raise ValueError(
                f"runner method {registration.name!r} ledger buckets must be "
                "nonnegative integers"
            )
        predictions = np.asarray(output.predictions, dtype=float)
        if predictions.shape != (len(test),) or not np.all(np.isfinite(predictions)):
            raise ValueError(
                f"runner method {registration.name!r} must return one finite "
                "prediction per test item"
            )
        config = dict(output.config)
        if RUN_IDENTITY_CONFIG_KEY in config:
            raise ValueError(
                f"runner method {registration.name!r} config uses reserved key "
                f"{RUN_IDENTITY_CONFIG_KEY!r}"
            )
        if assignment is not None:
            if "role_protocol" in config:
                raise ValueError(
                    f"runner method {registration.name!r} config uses reserved key "
                    "'role_protocol'"
                )
            config["role_protocol"] = _role_protocol(assignment)
        methods[registration.name] = {
            "predictions": predictions,
            "ledger": _ledger(*buckets, len(test)),
            "role": registration.role,
            "config": config,
        }
    return methods


def _check_realized_budget(
    methods: Mapping[str, Mapping[str, object]],
    budget: Mapping[str, Mapping[str, object]],
) -> None:
    modeled_by_method = {
        name: {"B_train": 0, "B_extra": 0, "B_pred": 0, "total": 0}
        for name in methods
    }
    for cell in budget.values():
        for name, record in cell["methods"].items():
            for field in modeled_by_method[name]:
                modeled_by_method[name][field] += record["modeled_ledger"][field]
    for name, expected in modeled_by_method.items():
        actual_ledger = methods[name]["ledger"]
        actual = {
            key: actual_ledger[key] for key in ("B_train", "B_extra", "B_pred", "total")
        }
        if actual != expected:
            raise ValueError(
                f"budget ledger mismatch for method {name!r}: "
                f"modeled={expected}, realized={actual}"
            )


_register_builtin_methods()


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

    projected = []
    for item in sorted(items, key=lambda item: item["item_id"]):
        row = {
            field: item[field]
            for field in IDENTITY_ITEM_FIELDS
            if field not in {"circuit_id", "bootstrap_stratum_id"}
        }
        # The bootstrap blocks on the physical circuit, so the identity the run
        # ID binds has to carry which circuit and stratum each row belongs to.
        # Resolving it here, through the metrics helper, keeps the artifact
        # self-describing and keeps one implementation of the legacy digest.
        circuit_id, stratum_id = normalized_bootstrap_ids(item)
        row["circuit_id"] = circuit_id
        row["bootstrap_stratum_id"] = stratum_id
        projected.append(row)
    return projected


def _run_artifact_id(
    dataset_hash: str,
    preset: str | None,
    item_stream_hashes: dict[str, str],
    test_items: list[dict],
    methods: dict[str, dict],
    *,
    dataset_environment_contract: Mapping[str, object],
    environment_contract: Mapping[str, object],
    role_assignment: Mapping[str, object] | None = None,
    budget: Mapping[str, object] | None = None,
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
    if role_assignment is not None:
        payload["role_assignment"] = role_assignment
    if budget is not None:
        payload["budget"] = budget
    raw = methods.get("raw")
    raw_config = raw.get("config") if isinstance(raw, Mapping) else None
    identity_metadata = (
        raw_config.get(RUN_IDENTITY_CONFIG_KEY)
        if isinstance(raw_config, Mapping)
        else None
    )
    if identity_metadata is not None:
        if not isinstance(identity_metadata, Mapping):
            raise ValueError("run artifact identity metadata must be an object")
        payload.update(copy.deepcopy(dict(identity_metadata)))
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


def _validate_role_assignment(
    run_result: Mapping[str, object], methods: Mapping[str, Mapping[str, object]]
) -> Mapping[str, object] | None:
    assignment = run_result.get("role_assignment")
    if assignment is None:
        return None
    if not isinstance(assignment, dict):
        raise ValueError("run artifact role_assignment must be an object")
    required = {
        "item_counts",
        "item_ids_sha256",
        "fit_role",
        "selection_role",
        "report_role",
        "test_items_used_for_selection",
        "pairwise_item_id_overlap",
    }
    if set(assignment) != required:
        raise ValueError("run artifact role_assignment has invalid fields")
    counts = assignment["item_counts"]
    hashes = assignment["item_ids_sha256"]
    overlaps = assignment["pairwise_item_id_overlap"]
    roles = {"train", "validation", "test"}
    if not isinstance(counts, dict) or set(counts) != roles:
        raise ValueError("run artifact role_assignment has invalid item_counts")
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError("run artifact role item counts must be positive integers")
    if not isinstance(hashes, dict) or set(hashes) != roles or any(
        not isinstance(value, str) or not value.startswith("sha256:")
        for value in hashes.values()
    ):
        raise ValueError("run artifact role_assignment has invalid item hashes")
    expected_overlaps = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    if overlaps != expected_overlaps:
        raise ValueError(
            "run artifact role assignment violates pairwise role disjointness"
        )
    expected_roles = {
        "fit_role": "train",
        "selection_role": "validation",
        "report_role": "test",
        "test_items_used_for_selection": 0,
    }
    for field, expected in expected_roles.items():
        if assignment[field] != expected:
            raise ValueError(f"run artifact role_assignment violates {field}")
    _require_run_match("n_train_items", run_result.get("n_train_items"), counts["train"])
    _require_run_match(
        "n_validation_items", run_result.get("n_validation_items"), counts["validation"]
    )
    _require_run_match("n_test_items", run_result.get("n_test_items"), counts["test"])
    test_items = run_result.get("test_items")
    if not isinstance(test_items, list):
        raise ValueError("run artifact requires test_items")
    _require_run_match("test role item hash", hashes["test"], _item_ids_hash(test_items))
    expected_protocol = _role_protocol(assignment)
    for name, spec in methods.items():
        config = spec.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"run artifact method {name!r} config must be an object")
        _require_run_match(
            f"methods.{name}.config.role_protocol",
            config.get("role_protocol"),
            expected_protocol,
        )
    return assignment


def _validate_budget_report(
    budget: object,
    methods: Mapping[str, Mapping[str, object]],
    item_stream_hashes: Mapping[str, str],
) -> Mapping[str, object] | None:
    if budget is None:
        return None
    if not isinstance(budget, dict) or not budget:
        raise ValueError("run artifact budget must contain paired cells")
    aggregate = {
        name: {"B_train": 0, "B_extra": 0, "B_pred": 0, "total": 0}
        for name in methods
    }
    tiers: set[str] = set()
    assigned_cell_ids: set[str] = set()
    expected_cell_fields = {
        "tier",
        "mode",
        "cap",
        "enforcement",
        "status",
        "normalization",
        "source_base_evals",
        "test_base_evals",
        "methods",
        "pairing",
    }
    for budget_cell_id, cell in sorted(budget.items()):
        digest = budget_cell_id.removeprefix("budget-cell-")
        if (
            not budget_cell_id.startswith("budget-cell-")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("run artifact budget has an invalid budget_cell_id")
        if not isinstance(cell, dict) or set(cell) != expected_cell_fields:
            raise ValueError(
                f"run artifact budget cell {budget_cell_id!r} has invalid fields"
            )
        pairing = cell["pairing"]
        if not isinstance(pairing, dict) or set(pairing) != {
            "descriptor",
            "source_cell_ids",
            "test_cell_ids",
        }:
            raise ValueError(
                f"run artifact budget cell {budget_cell_id!r} has invalid pairing"
            )
        descriptor = pairing["descriptor"]
        descriptor_fields = {
            "contract",
            "dataset_schema_version",
            "split_id",
            "split_axis",
            "partition_id",
            "fixed_axis_values",
            "n_qubits",
            "replicate",
            "stratum",
        }
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != descriptor_fields
            or descriptor.get("contract") != "paired-budget-cell-v1"
            or descriptor.get("dataset_schema_version") != SPLIT_SCHEMA_VERSION
            or not isinstance(descriptor.get("fixed_axis_values"), dict)
            or not isinstance(descriptor.get("split_id"), str)
            or not isinstance(descriptor.get("split_axis"), str)
            or not isinstance(descriptor.get("partition_id"), str)
            or type(descriptor.get("n_qubits")) is not int
            or descriptor.get("n_qubits", 0) <= 0
            or type(descriptor.get("replicate")) is not int
            or descriptor.get("replicate", -1) < 0
            or descriptor.get("stratum") not in set(FAMILY_STRATA.values())
        ):
            raise ValueError(
                f"run artifact budget cell {budget_cell_id!r} has invalid descriptor"
            )
        try:
            expected_budget_cell_id = _budget_cell_id_from_descriptor(descriptor)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"run artifact budget cell {budget_cell_id!r} descriptor is invalid"
            ) from exc
        _require_run_match(
            f"budget.{budget_cell_id}.pairing descriptor hash",
            budget_cell_id,
            expected_budget_cell_id,
        )
        source_cell_ids = pairing["source_cell_ids"]
        test_cell_ids = pairing["test_cell_ids"]
        if (
            not isinstance(source_cell_ids, list)
            or not source_cell_ids
            or not isinstance(test_cell_ids, list)
            or not test_cell_ids
            or any(not isinstance(value, str) for value in source_cell_ids)
            or any(not isinstance(value, str) for value in test_cell_ids)
            or len(source_cell_ids) != len(set(source_cell_ids))
            or len(test_cell_ids) != len(set(test_cell_ids))
            or source_cell_ids != sorted(source_cell_ids)
            or test_cell_ids != sorted(test_cell_ids)
            or set(source_cell_ids) & set(test_cell_ids)
        ):
            raise ValueError(
                f"run artifact budget cell {budget_cell_id!r} has invalid role cells"
            )
        role_cell_ids = set(source_cell_ids) | set(test_cell_ids)
        if role_cell_ids - set(item_stream_hashes) or role_cell_ids & assigned_cell_ids:
            raise ValueError(
                "run artifact budget pairing does not partition dataset cells"
            )
        assigned_cell_ids.update(role_cell_ids)
        tier = cell.get("tier")
        if not isinstance(tier, str) or tier != tier.upper():
            raise ValueError("run artifact budget tier must be an uppercase string")
        tier_pair = TierPair.declared(tier)
        tiers.add(tier)
        expected_top = {
            "mode": BUDGET_MODE.value,
            "cap": TIERS[tier],
            "enforcement": BUDGET_ENFORCEMENT,
            "status": "feasible",
            "normalization": (
                "one paired statistical cell represented as unit-shot "
                "BudgetInputs; validation is charged to B_train"
            ),
        }
        for field, expected in expected_top.items():
            _require_run_match(
                f"budget.{budget_cell_id}.{field}", cell.get(field), expected
            )
        source_evals = cell.get("source_base_evals")
        test_evals = cell.get("test_base_evals")
        if type(source_evals) is not int or source_evals < 0:
            raise ValueError(
                "run artifact budget source_base_evals must be nonnegative"
            )
        if type(test_evals) is not int or test_evals <= 0:
            raise ValueError("run artifact budget test_base_evals must be positive")
        records = cell.get("methods")
        if not isinstance(records, dict) or set(records) != set(methods):
            raise ValueError("run artifact budget methods do not match run methods")

        for name, record in records.items():
            if not isinstance(record, dict):
                raise ValueError(
                    f"run artifact budget method {name!r} must be an object"
                )
            budget_method = record.get("budget_method")
            if budget_method is None:
                expected_record = {
                    "budget_method": None,
                    "inputs": None,
                    "modeled_ledger": {
                        "B_train": 0,
                        "B_extra": 0,
                        "B_pred": 0,
                        "total": 0,
                    },
                    "required_combined_cap": 0,
                    "shortfall": 0,
                    "method_feasible": True,
                    "statistical_feasible": True,
                    "status": "feasible",
                }
            else:
                inputs_payload = record.get("inputs")
                if not isinstance(inputs_payload, dict) or set(inputs_payload) != set(
                    _BUDGET_INPUT_FIELDS
                ):
                    raise ValueError(
                        f"run artifact budget method {name!r} has invalid inputs"
                    )
                inputs = BudgetInputs(
                    **{
                        field: inputs_payload[field]
                        for field in _BUDGET_INPUT_FIELDS
                    }
                )
                if (
                    inputs.n_train_circuits != source_evals
                    or inputs.n_test_circuits != test_evals
                    or inputs.shots != 1
                    or inputs.groups != 1
                ):
                    raise ValueError(
                        f"run artifact budget method {name!r} violates paired-cell "
                        "unit-shot normalization"
                    )
                method = Method(budget_method)
                modeled = method_budget(method, inputs)
                check = check_constraints(method, inputs, tier_pair, BUDGET_MODE)
                required_cap = minimum_tier_constant(method, inputs).combined_joint
                expected_record = {
                    "budget_method": method.value,
                    "inputs": inputs_payload,
                    "modeled_ledger": {
                        "B_train": modeled.B_train,
                        "B_extra": modeled.B_extra,
                        "B_pred": modeled.B_pred,
                        "total": modeled.total,
                    },
                    "required_combined_cap": required_cap,
                    "shortfall": max(0, required_cap - tier_pair.combined_cap),
                    "method_feasible": check.method_feasible,
                    "statistical_feasible": check.statistical_feasible,
                    "status": check.status,
                }
            _require_run_match(
                f"budget.{budget_cell_id}.methods.{name}",
                record,
                expected_record,
            )
            if expected_record["status"] != "feasible":
                raise ValueError(
                    f"run artifact records infeasible budget method {name!r}"
                )
            for field, value in expected_record["modeled_ledger"].items():
                aggregate[name][field] += value
    if len(tiers) != 1:
        raise ValueError("run artifact budget cells must use one tier")
    if assigned_cell_ids != set(item_stream_hashes):
        raise ValueError("run artifact budget does not cover every dataset cell")
    for name, expected in aggregate.items():
        actual_ledger = methods[name]["ledger"]
        actual = {
            key: actual_ledger[key]
            for key in ("B_train", "B_extra", "B_pred", "total")
        }
        _require_run_match(
            f"budget campaign realized ledger for {name}", actual, expected
        )
    return budget


def _is_lower_sha256(value: object, *, prefix: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:") if prefix else value
    if prefix and not value.startswith("sha256:"):
        return False
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def run_identity_encoding_profiles(
    run_result: Mapping[str, object],
) -> dict[str, str]:
    """Return declared profiles or explicit unknowns for historical artifacts."""

    items = run_result.get("test_items")
    if not isinstance(items, list) or not items or any(
        not isinstance(item, Mapping) or not isinstance(item.get("family"), str)
        for item in items
    ):
        raise ValueError(
            "run artifact identity encoding profiles require test item families"
        )
    families = {str(item["family"]) for item in items}
    if PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD not in run_result:
        return {
            family: UNKNOWN_IDENTITY_ENCODING_PROFILE
            for family in sorted(families)
        }
    return validate_physical_identity_encoding_profiles(
        run_result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD],
        families=families,
        allow_unknown=True,
    )


def _validate_run_contract_fields(
    run_result: Mapping[str, object], methods: Mapping[str, Mapping[str, object]]
) -> str:
    version = run_result.get("dataset_schema_version")
    if version is None:
        raise ValueError(
            "unversioned qem-bench-run-v2 artifact requires explicit migration; "
            "dataset_schema_version is required"
        )
    profile_declared = PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD in run_result
    profile_fields = RUN_PROFILE_FIELDS if profile_declared else frozenset()
    if version == SPLIT_SCHEMA_VERSION:
        expected_fields = RUN_COMMON_FIELDS | RUN_SPLIT_FIELDS | profile_fields
    elif version == LEGACY_SCHEMA_VERSION:
        expected_fields = RUN_COMMON_FIELDS | profile_fields
    else:
        raise ValueError(f"unsupported run dataset_schema_version {version!r}")
    if set(run_result) != expected_fields:
        missing = sorted(expected_fields - set(run_result))
        extra = sorted(set(run_result) - expected_fields)
        raise ValueError(
            f"{version} run artifact has invalid fields; "
            f"missing={missing}, extra={extra}"
        )

    if profile_declared:
        value = run_result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD]
        if not isinstance(value, Mapping):
            raise ValueError(
                "run artifact physical identity encoding profiles must be an object"
            )
        validate_physical_identity_encoding_profiles(
            value,
            families=set(value),
            allow_unknown=True,
        )

    for name, spec in methods.items():
        if set(spec) != RUN_METHOD_FIELDS:
            missing = sorted(RUN_METHOD_FIELDS - set(spec))
            extra = sorted(set(spec) - RUN_METHOD_FIELDS)
            raise ValueError(
                f"run artifact method {name!r} has invalid fields; "
                f"missing={missing}, extra={extra}"
            )
        if not isinstance(spec.get("config"), dict):
            raise ValueError(f"run artifact method {name!r} config must be an object")

    dataset_hash_value = run_result.get("dataset_hash")
    if not _is_lower_sha256(dataset_hash_value):
        raise ValueError("run artifact dataset_hash must be a lowercase SHA-256")
    stream_hashes = run_result.get("dataset_item_stream_hashes")
    if not isinstance(stream_hashes, dict) or not stream_hashes:
        raise ValueError("run artifact dataset_item_stream_hashes must be an object")
    if version == LEGACY_SCHEMA_VERSION:
        expected_streams = {LEGACY_SCHEMA_VERSION: dataset_hash_value}
        if stream_hashes != expected_streams:
            raise ValueError(
                "legacy-v1 run artifact must carry only its legacy item stream"
            )
    elif any(
        not isinstance(key, str)
        or not key.startswith("cell-")
        or not _is_lower_sha256(value)
        for key, value in stream_hashes.items()
    ):
        raise ValueError("split-v2 run artifact has invalid cell stream hashes")

    expected_feature_spec = {
        "version": FEATURE_SPEC_VERSION,
        "features": list(FEATURES),
    }
    _require_run_match(
        "feature_spec", run_result.get("feature_spec"), expected_feature_spec
    )
    stratum = run_result.get("stratum")
    if stratum not in set(FAMILY_STRATA.values()):
        raise ValueError("run artifact has an invalid stratum")
    n_train = run_result.get("n_train_items")
    if type(n_train) is not int or n_train <= 0:
        raise ValueError("run artifact n_train_items must be a positive integer")
    if version == SPLIT_SCHEMA_VERSION:
        n_validation = run_result.get("n_validation_items")
        if type(n_validation) is not int or n_validation <= 0:
            raise ValueError(
                "split-v2 run artifact n_validation_items must be positive"
            )
    else:
        n_validation = 0

    label_evals = run_result.get("label_evals")
    if (
        not isinstance(label_evals, dict)
        or set(label_evals) != SUPPORTED_LABEL_METHODS
        or any(type(value) is not int or value < 0 for value in label_evals.values())
    ):
        raise ValueError("run artifact label_evals has invalid fields or values")
    for method in sorted(SUPPORTED_LABEL_METHODS):
        _require_run_match(
            f"label_evals_{method}",
            run_result.get(f"label_evals_{method}"),
            label_evals[method],
        )

    raw_config = methods["raw"]["config"]
    identity_metadata = raw_config.get(RUN_IDENTITY_CONFIG_KEY)
    expected_identity_fields = RUN_IDENTITY_FIELDS | profile_fields
    if (
        not isinstance(identity_metadata, dict)
        or set(identity_metadata) != expected_identity_fields
    ):
        raise ValueError("run artifact lacks closed identity metadata")
    for name, spec in methods.items():
        if name != "raw" and RUN_IDENTITY_CONFIG_KEY in spec["config"]:
            raise ValueError(
                f"run artifact method {name!r} uses reserved identity metadata"
            )
    expected_identity = {
        "dataset_schema_version": version,
        "dataset_manifest_sha256": run_result["dataset_manifest_sha256"],
        "feature_spec": run_result["feature_spec"],
        "stratum": stratum,
        "n_train_items": n_train,
        "n_validation_items": n_validation,
        "n_test_items": run_result.get("n_test_items"),
        "label_evals": label_evals,
    }
    if profile_declared:
        expected_identity[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD] = copy.deepcopy(
            run_result[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD]
        )
    _require_run_match(
        "methods.raw.config.run_artifact_identity",
        identity_metadata,
        expected_identity,
    )
    if not _is_lower_sha256(run_result.get("dataset_manifest_sha256"), prefix=True):
        raise ValueError(
            "run artifact dataset_manifest_sha256 must be a prefixed SHA-256"
        )
    return version


def validate_run_artifact(run_result: dict) -> dict:
    """Validate a serialized run identity and recompute every reported result."""

    if run_result.get("schema_version") != "qem-bench-run-v2":
        raise ValueError("unsupported runner result schema_version")
    methods = run_result.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("run artifact requires methods")
    if not isinstance(methods.get("raw"), dict):
        raise ValueError("run artifact requires a raw method")
    for name, spec in methods.items():
        if not isinstance(spec, dict):
            raise ValueError(f"run artifact method {name!r} must be an object")
    dataset_schema_version = _validate_run_contract_fields(run_result, methods)
    _require_run_match(
        "analysis_contract", run_result.get("analysis_contract"), _analysis_contract()
    )

    items = _canonical_test_items(run_result)
    run_identity_encoding_profiles(run_result)
    n_test = len(items)
    for name, spec in methods.items():
        _validate_ledger(name, spec["ledger"], n_test)
    if dataset_schema_version == SPLIT_SCHEMA_VERSION:
        assignment = _validate_role_assignment(run_result, methods)
        if assignment is None:
            raise ValueError("split-v2 run artifact requires role_assignment")
        budget = _validate_budget_report(
            run_result.get("budget"),
            methods,
            run_result["dataset_item_stream_hashes"],
        )
        if budget is None:
            raise ValueError("split-v2 run artifact requires an enforced budget")
    else:
        assignment = None
        budget = None
    try:
        dataset_environment_contract = validate_environment_contract(
            run_result["dataset_environment_contract"]
        )
        run_environment_contract = validate_environment_contract(
            run_result["environment_contract"]
        )
        expected_artifact_id = _run_artifact_id(
            str(run_result["dataset_hash"]),
            run_result["preset"],
            dict(run_result["dataset_item_stream_hashes"]),
            items,
            methods,
            dataset_environment_contract=dataset_environment_contract,
            environment_contract=run_environment_contract,
            role_assignment=assignment,
            budget=budget,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run artifact identity inputs are invalid") from exc
    _require_run_match(
        "artifact_id", run_result.get("artifact_id"), expected_artifact_id
    )

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


def _resolve_budget_tier(
    manifest: Mapping[str, object], configured_tier: str | None
) -> str | None:
    version = manifest["dataset_schema_version"]
    if version == LEGACY_SCHEMA_VERSION and configured_tier is not None:
        raise ValueError("legacy-v1 runs do not carry split budget contracts")
    declared_tier: object = None
    if version == SPLIT_SCHEMA_VERSION:
        split_spec = manifest.get("split_spec")
        if not isinstance(split_spec, dict):
            raise ValueError("split-v2 manifest requires split_spec")
        declared_tier = split_spec.get("budget_tier")
    if declared_tier is not None and not isinstance(declared_tier, str):
        raise ValueError("artifact budget_tier must be a string when declared")
    declared = declared_tier.upper() if isinstance(declared_tier, str) else None
    configured = configured_tier.upper() if configured_tier is not None else None
    if declared is not None and configured is not None and declared != configured:
        raise ValueError(
            f"configured budget tier {configured!r} conflicts with artifact tier "
            f"{declared!r}"
        )
    tier = declared or configured
    if version == SPLIT_SCHEMA_VERSION and tier is None:
        raise ValueError(
            "split-v2 run requires a budget tier from split_spec.budget_tier or --tier"
        )
    if tier is not None:
        TierPair.declared(tier)
    return tier


def run(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    budget_tier: str | None = None,
) -> dict:
    data_dir = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    dataset_manifest_sha256 = (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    items, manifest = _load(data_dir)
    dataset_environment_contract = validate_environment_contract(
        manifest["environment_contract"]
    )
    run_environment_contract = environment_contract()
    split_v2 = manifest["dataset_schema_version"] == SPLIT_SCHEMA_VERSION
    train = [it for it in items if it["split"] == "train"]
    validation = [it for it in items if it["split"] == "validation"]
    test = [it for it in items if it["split"] == "test"]
    if not train or not test:
        raise ValueError("dataset must contain both train and test items")
    if split_v2 and not validation:
        raise ValueError("split-v2 dataset must contain validation items")
    strata = {item["stratum"] for item in items}
    if len(strata) != 1:
        raise ValueError(
            "the Clifford control stratum is excluded from the "
            "continuous-regression headline; runner requires one stratum per dataset"
        )

    assignment = _role_assignment(train, validation, test) if split_v2 else None
    registrations = registered_methods()
    tier = _resolve_budget_tier(manifest, budget_tier)
    if tier is not None:
        costs_by_cell = _paired_budget_cell_evals(train, validation, test)
        budget = {
            budget_cell_id: {
                **_budget_preflight(
                    tier,
                    registrations,
                    costs.source_evals,
                    costs.test_evals,
                ),
                "pairing": {
                    "descriptor": costs.pairing_descriptor,
                    "source_cell_ids": list(costs.source_cell_ids),
                    "test_cell_ids": list(costs.test_cell_ids),
                },
            }
            for budget_cell_id, costs in sorted(costs_by_cell.items())
        }
    else:
        budget = None
    methods = _execute_registered_methods(
        registrations, train, validation, test, manifest, assignment
    )
    if budget is not None:
        _check_realized_budget(methods, budget)

    r = np.asarray(methods["raw"]["predictions"], dtype=float)
    y = np.array([it["ideal_expectation"] for it in test])
    n_test = len(test)

    if split_v2:
        raw_label_evals = manifest["generation_ledger"]["label_evals_by_method"]
    else:
        raw_label_evals = {
            method: int(
                manifest["generation_ledger"].get(f"label_evals_{method}", 0)
            )
            for method in sorted(SUPPORTED_LABEL_METHODS)
        }
    label_evals = {
        method: int(raw_label_evals.get(method, 0))
        for method in sorted(SUPPORTED_LABEL_METHODS)
    }
    run_identity = {
        "dataset_schema_version": manifest["dataset_schema_version"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD: copy.deepcopy(
            manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD]
        ),
        "feature_spec": manifest["feature_spec"],
        "stratum": items[0]["stratum"],
        "n_train_items": len(train),
        "n_validation_items": len(validation) if split_v2 else 0,
        "n_test_items": n_test,
        "label_evals": label_evals,
    }
    raw_config = methods["raw"]["config"]
    if RUN_IDENTITY_CONFIG_KEY in raw_config:
        raise ValueError(
            f"raw method config uses reserved key {RUN_IDENTITY_CONFIG_KEY!r}"
        )
    raw_config[RUN_IDENTITY_CONFIG_KEY] = copy.deepcopy(run_identity)

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
        role_assignment=assignment,
        budget=budget,
    )

    results: dict = {
        "schema_version": "qem-bench-run-v2",
        "artifact_id": artifact_id,
        "analysis_contract": _analysis_contract(),
        "dataset_schema_version": manifest["dataset_schema_version"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_environment_contract": dataset_environment_contract,
        "environment_contract": run_environment_contract,
        "dataset_hash": manifest["dataset_hash"],
        "dataset_item_stream_hashes": item_stream_hashes,
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD: copy.deepcopy(
            manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD]
        ),
        "test_items": test_items,
        "preset": manifest["preset"],
        "feature_spec": manifest["feature_spec"],
        "stratum": items[0]["stratum"],
        "n_train_items": len(train),
        "n_test_items": n_test,
        "label_evals": label_evals,
        "methods": {},
    }
    if split_v2:
        results["n_validation_items"] = len(validation)
        results["role_assignment"] = assignment
    if budget is not None:
        results["budget"] = budget
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
    if "role_assignment" in results:
        counts = results["role_assignment"]["item_counts"]
        print(
            "roles "
            + ", ".join(
                f"{role}={counts[role]}" for role in ("train", "validation", "test")
            )
            + "  selection=validation-only  test-used-for-selection=0"
        )
    if "budget" in results:
        budget_cells = list(results["budget"].values())
        tiers = {cell["tier"] for cell in budget_cells}
        caps = {cell["cap"] for cell in budget_cells}
        statuses = {cell["status"] for cell in budget_cells}
        modes = {cell["mode"] for cell in budget_cells}
        enforcements = {cell["enforcement"] for cell in budget_cells}
        print(
            f"budget cells={len(budget_cells)} tier={next(iter(tiers))} "
            f"mode={next(iter(modes))} cap-per-method-cell={next(iter(caps))} "
            f"status={next(iter(statuses))} "
            f"enforcement={next(iter(enforcements))}"
        )
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
