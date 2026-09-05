"""Structural assertions, frozen fits, and the campaign's primary endpoint.

Plan review round 2 raised one blocking High: the plan named an endpoint whose
analysis program did not exist. `tools/measure_incremental_value.py` evaluates
source-validation gates, records validation predictions, runs the roster, and
records test MAEs. It computes neither untouched-test gain intervals nor the
cross-regime contrast, and `circuit_blocked_bootstrap` cannot produce them: its
paired path subtracts macro statistics rather than forming a ratio of two fitted
predictors' errors and then a difference between independent datasets.

This module is that missing program, in three separable pieces.

`assert_campaign_structure` runs before any fitting. It checks realized physical
circuit counts per family and role, the training-prefix nesting that lets two
sizes share one seed's draws, and the shared validation and test identities that
make the sizes comparable. Checking these after fitting would mean discovering a
mis-generated dataset from a number rather than from a structure.

`build_setting_record` fits the four scoring arms and the two training-shuffle
diagnostics, then writes one self-contained record per setting: every selected
configuration, every item-keyed prediction, the gate, the shuffle intervals, and
a projection of the untouched test rows, all bound to the dataset hash and the
code revision that produced them.

`evaluate_campaign` reads only those records. Every reported statistic and table
is regenerated from them, so a number in the manuscript can be traced to a file
rather than to a run that has since been overwritten. A setting whose record is
absent is reported as a failed setting and supplies no replication; it is never
silently dropped, because dropping one after seeing its numbers is the failure
mode the freeze exists to prevent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
import math

import numpy as np

from qem_bench.baselines.controls import shuffle_noisy_items
from qem_bench.baselines.liao import LiaoMitigator
from qem_bench.campaign.design import (
    ARMS,
    BASELINE_REGIME,
    BOOTSTRAP_RESAMPLES,
    CONFIDENCE,
    CONTRAST_REGIME,
    DROPPED_FEATURE,
    FAMILIES,
    FIXED_MODEL_SHUFFLE_SEEDS,
    PRIMARY_ARM,
    PRIMARY_SIZE,
    RATIO_MARGIN,
    REQUIRED_FAMILIES,
    ROOT_SEED,
    SEEDS,
    TRAINING_SHUFFLE_ARMS,
    TRAINING_SHUFFLE_SEED,
    campaign_setting_keys,
    declared_design,
    expected_circuits,
    is_frozen_setting,
    setting_key,
)
from qem_bench.runner.metrics import build_cell_records, headline_metrics
from qem_bench.runner.run import (
    _prediction_items,
    _prediction_manifest,
    registered_methods,
)
from qem_bench.stats.bootstrap import circuit_blocked_bootstrap
from qem_bench.stats.gain_contrast import evaluate_gain_contrast
from qem_bench.stats.incremental_value import CONTROLS, evaluate_incremental_value

RECORD_SCHEMA_VERSION = "qem-bench-campaign-record-v1"
REPORT_SCHEMA_VERSION = "qem-bench-campaign-report-v1"
ROLES = ("train", "validation", "test")
# The fixed-model permutation preserves these groups, so a permuted value stays
# a plausible reading of the same cell rather than of a different noise level.
FIXED_MODEL_SHUFFLE_STRATA = ("family", "noise_family", "severity", "observable")
# Test rows travel inside the record so the analysis is reproducible from the
# record alone. Only the fields the contrast reads are retained.
TEST_ROW_FIELDS = (
    "item_id",
    "dataset_schema_version",
    "split",
    "family",
    "stratum",
    "circuit_id",
    "instance",
    "noise_family",
    "severity",
    "observable",
    "ideal_expectation",
)
# Validation rows travel too, because the gate, the training-shuffle intervals
# and the fixed-model summaries all have to regenerate from the record without a
# refit and without the dataset directory. These are the fields the gate, the
# cell records and the circuit-blocked bootstrap read.
VALIDATION_ROW_FIELDS = (
    "item_id",
    "dataset_schema_version",
    "split_id",
    "split_axis",
    "partition_id",
    "split",
    "domain",
    "family",
    "stratum",
    "circuit_id",
    "circuit_pool_id",
    "instance",
    "noise_family",
    "severity",
    "observable",
    "shots",
    "noisy_expectation",
    "ideal_expectation",
)


# --------------------------------------------------------------------------
# Structure, asserted before anything is fitted
# --------------------------------------------------------------------------


def assert_campaign_structure(
    settings: Sequence[Mapping[str, object]],
    *,
    expected_by_size: Mapping[int, Mapping[str, int]] | None = None,
) -> dict:
    """Check realized identities and roles across the settings supplied.

    ``settings`` is a sequence of ``{"regime", "seed", "size", "items"}``. Each
    setting is checked on its own; then sizes sharing a regime and seed are
    checked for training-prefix nesting and identical validation and test
    circuits, and regimes sharing a seed and size are checked for disjoint
    circuits. Any violation raises ValueError. The returned report records what
    was checked, including which settings the supplied subset did not cover.

    ``expected_by_size`` overrides the frozen per-family circuit counts, which a
    rehearsal needs and the campaign never uses. Overriding it marks the report
    as not frozen, and that mark travels into every record and report built from
    the same run, so rehearsal numbers cannot be published as campaign scores.
    """

    if not settings:
        raise ValueError("at least one setting is required")
    per_setting: dict[tuple[str, int, int], dict] = {}
    for setting in settings:
        for field in ("regime", "seed", "size", "items"):
            if field not in setting:
                raise ValueError(f"each setting requires {field!r}")
        key = (str(setting["regime"]), int(setting["seed"]), int(setting["size"]))
        if key in per_setting:
            raise ValueError(f"setting {setting_key(*key)} was supplied twice")
        override = None if expected_by_size is None else expected_by_size.get(key[2])
        per_setting[key] = _assert_setting(*key, setting["items"], expected=override)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "declared_design": declared_design(),
        "design_frozen": all(
            value["design_frozen"] for value in per_setting.values()),
        "asserted_circuits": {
            setting_key(*key): value["expected"]
            for key, value in per_setting.items()
        },
        "settings": {
            setting_key(*key): value["counts"] for key, value in per_setting.items()
        },
        "cross_size_checks": [],
        "cross_regime_checks": [],
        "settings_not_supplied": [
            key for key in campaign_setting_keys()
            if key not in {setting_key(*name) for name in per_setting}
        ],
    }

    by_regime_seed: dict[tuple[str, int], list[int]] = defaultdict(list)
    by_seed_size: dict[tuple[int, int], list[str]] = defaultdict(list)
    for regime, seed, size in per_setting:
        by_regime_seed[(regime, seed)].append(size)
        by_seed_size[(seed, size)].append(regime)

    for (regime, seed), sizes in sorted(by_regime_seed.items()):
        ordered = sorted(sizes)
        for smaller, larger in zip(ordered, ordered[1:], strict=False):
            small = per_setting[(regime, seed, smaller)]["identities"]
            large = per_setting[(regime, seed, larger)]["identities"]
            for family in sorted(small):
                _assert_training_prefix(
                    small[family]["train"], large[family]["train"],
                    label=f"{regime}/seed {seed}: n{smaller} inside n{larger}, {family}",
                )
                for role in ("validation", "test"):
                    if small[family][role] != large[family][role]:
                        raise ValueError(
                            f"{regime}/seed {seed}: {role} circuits differ between "
                            f"n{smaller} and n{larger} in {family}; both sizes must "
                            "score the same circuits"
                        )
            report["cross_size_checks"].append({
                "regime": regime,
                "seed": seed,
                "sizes": [smaller, larger],
                "training_prefix_nested": True,
                "validation_and_test_identical": True,
            })

    for (seed, size), regimes in sorted(by_seed_size.items()):
        ordered = sorted(regimes)
        for first, second in zip(ordered, ordered[1:], strict=False):
            # Every physical circuit on each side, across families and roles. A
            # role-by-role comparison would pass a shipped training circuit that
            # reappears as a large test circuit, which is the same dependence the
            # check exists to rule out.
            shared = _all_circuits(
                per_setting[(first, seed, size)]["identities"]
            ) & _all_circuits(per_setting[(second, seed, size)]["identities"])
            if shared:
                raise ValueError(
                    f"seed {seed}, n{size}: {first} and {second} share "
                    f"{len(shared)} physical circuits in any family or role; the "
                    "two regimes have to be independent samples"
                )
            report["cross_regime_checks"].append({
                "seed": seed,
                "size": size,
                "regimes": [first, second],
                "circuits_disjoint": True,
            })
    return report


def _all_circuits(identities: Mapping[str, Mapping[str, Mapping[int, str]]]) -> set[str]:
    """Every physical circuit of one setting, whatever family or role holds it."""
    return {
        circuit
        for by_role in identities.values()
        for numbering in by_role.values()
        for circuit in numbering.values()
    }


def _assert_setting(
    regime: str,
    seed: int,
    size: int,
    items: object,
    *,
    expected: Mapping[str, int] | None = None,
) -> dict:
    """Check one setting's realized counts, roles, and instance numbering."""

    key = setting_key(regime, seed, size)
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(f"{key}: items must be a sequence of rows")
    if not items:
        raise ValueError(f"{key}: the dataset is empty")

    frozen = expected_circuits(size) if is_frozen_setting(seed, size) else None
    if expected is None:
        if frozen is None:
            raise ValueError(
                f"{key}: seed {seed} and size {size} are outside the frozen design, "
                "so the expected circuit counts have to be supplied"
            )
        expected = frozen
    else:
        missing = [role for role in ROLES if role not in expected]
        if missing:
            raise ValueError(f"{key}: expected counts omit {missing}")
        expected = {role: int(expected[role]) for role in ROLES}
    circuit_roles: dict[str, set[tuple[str, str]]] = defaultdict(set)
    instances: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    rows: dict[tuple[str, str], int] = defaultdict(int)
    for item in items:
        if item.get("dataset_schema_version") != "split-v2":
            raise ValueError(f"{key}: every row requires split-v2 metadata")
        if item.get("split_id") != "S0":
            raise ValueError(f"{key}: the campaign scores S0 rows only")
        role = str(item["split"])
        if role not in ROLES:
            raise ValueError(f"{key}: unexpected role {role!r}")
        family = str(item["family"])
        if family not in FAMILIES:
            raise ValueError(f"{key}: unexpected circuit family {family!r}")
        circuit = str(item["circuit_id"])
        instance = int(item["instance"])
        circuit_roles[circuit].add((family, role))
        numbering = instances[(family, role)]
        if numbering.setdefault(instance, circuit) != circuit:
            raise ValueError(
                f"{key}: instance {instance} of {family}/{role} names two circuits"
            )
        rows[(family, role)] += 1

    spanning = [circuit for circuit, names in circuit_roles.items() if len(names) != 1]
    if spanning:
        raise ValueError(
            f"{key}: {len(spanning)} physical circuits span more than one family or "
            "role; roles have to be disjoint and families cannot share a circuit"
        )

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for family in FAMILIES:
        counts[family] = {}
        for role in ROLES:
            numbering = instances[(family, role)]
            realized = len(set(numbering.values()))
            if realized != expected[role]:
                raise ValueError(
                    f"{key}: {family}/{role} realized {realized} physical circuits, "
                    f"expected {expected[role]}"
                )
            if sorted(numbering) != list(range(expected[role])):
                raise ValueError(
                    f"{key}: {family}/{role} instance numbering is not 0 through "
                    f"{expected[role] - 1}"
                )
            if len(numbering) != realized:
                raise ValueError(
                    f"{key}: {family}/{role} reuses one physical circuit across "
                    "two instances"
                )
            counts[family][role] = {
                "circuits": realized,
                "items": rows[(family, role)],
            }
    return {
        "counts": counts,
        "expected": dict(expected),
        "design_frozen": expected == frozen,
        "frozen_setting": is_frozen_setting(seed, size),
        "identities": {
            family: {role: dict(instances[(family, role)]) for role in ROLES}
            for family in FAMILIES
        },
    }


def _assert_training_prefix(
    small: Mapping[int, str], large: Mapping[int, str], *, label: str,
) -> None:
    """The smaller training pool has to be the larger pool's leading instances."""
    if len(small) > len(large):
        raise ValueError(f"{label}: the smaller size has more training circuits")
    for instance, circuit in sorted(small.items()):
        if large.get(instance) != circuit:
            raise ValueError(
                f"{label}: training instance {instance} differs between the sizes, so "
                "the smaller pool is not a prefix of the larger one"
            )


# --------------------------------------------------------------------------
# One setting's frozen fits
# --------------------------------------------------------------------------


def build_setting_record(
    items: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    regime: str,
    seed: int,
    size: int,
    data_path: str,
    code_revision: str,
    expected: Mapping[str, int] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    gate_seed: int = ROOT_SEED,
    training_shuffle_seed: int = TRAINING_SHUFFLE_SEED,
    fixed_model_shuffle_seeds: Sequence[int] = FIXED_MODEL_SHUFFLE_SEEDS,
) -> dict:
    """Fit one setting's arms and return its complete, self-contained record.

    Selection reads source validation only, and prediction rows never carry
    ``ideal_expectation``, so a test label cannot reach fitting or selection.
    The record retains every selected configuration and every item-keyed
    prediction, so the analysis never refits. ``expected`` overrides the frozen
    per-family circuit counts for a rehearsal, and the record says so.
    """

    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("a record has to name the code revision that produced it")
    structure = _assert_setting(regime, seed, size, items, expected=expected)
    master_seed = int(manifest["master_seed"])
    if master_seed != seed:
        raise ValueError(
            f"{setting_key(regime, seed, size)}: the artifact was generated with "
            f"master seed {master_seed}, not {seed}"
        )

    roles = {role: [row for row in items if row["split"] == role] for role in ROLES}
    prediction_manifest = _prediction_manifest(manifest)
    prediction_rows = {
        role: _prediction_items(roles[role]) for role in ("validation", "test")
    }

    predictions: dict[str, dict[str, dict[str, float]]] = {"validation": {}, "test": {}}
    configs: dict[str, object] = {}
    fitted: dict[str, object] = {}
    for arm in ARMS.values():
        for method in (arm["full"], arm["control"]):
            if method in configs:
                continue
            model = _fit_arm(
                method, roles["train"], roles["validation"], manifest, master_seed)
            fitted[method] = model
            configs[method] = _arm_config(method, model)
            for role in ("validation", "test"):
                predictions[role][method] = _keyed_predictions(
                    roles[role],
                    _arm_predict(
                        method, model, prediction_rows[role], prediction_manifest),
                )

    # The gate names both of its controls, and a missing one makes it report
    # `not_evaluable` rather than fail, so the record would carry an empty gate
    # that still looked like a gate. Fit whichever control the arms did not.
    for method in CONTROLS:
        if method in configs:
            continue
        model = _fit_arm(
            method, roles["train"], roles["validation"], manifest, master_seed)
        configs[method] = _arm_config(method, model)
        predictions["validation"][method] = _keyed_predictions(
            roles["validation"],
            _arm_predict(
                method, model, prediction_rows["validation"], prediction_manifest),
        )

    shuffled_train = shuffle_noisy_items(roles["train"], seed=training_shuffle_seed)
    for source, shuffled_name in TRAINING_SHUFFLE_ARMS.items():
        model = _fit_arm(
            source, shuffled_train, roles["validation"], manifest, master_seed)
        configs[shuffled_name] = _arm_config(source, model)
        predictions["validation"][shuffled_name] = _keyed_predictions(
            roles["validation"],
            _arm_predict(
                source, model, prediction_rows["validation"], prediction_manifest),
        )

    dataset_hash = str(manifest["dataset_hash"])
    # Everything the plan froze has to hold for a record to claim it is one. A
    # cheaper resample count or a different diagnostic seed makes the run a
    # rehearsal, and a rehearsal never reaches a publication path.
    settings_frozen = (
        structure["design_frozen"]
        and structure["frozen_setting"]
        and n_resamples == BOOTSTRAP_RESAMPLES
        and confidence == CONFIDENCE
        and gate_seed == ROOT_SEED
        and training_shuffle_seed == TRAINING_SHUFFLE_SEED
        and tuple(int(value) for value in fixed_model_shuffle_seeds)
        == FIXED_MODEL_SHUFFLE_SEEDS
    )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "setting": setting_key(regime, seed, size),
        "regime": regime,
        "seed": seed,
        "size": size,
        "data_path": str(data_path),
        "dataset_hash": dataset_hash,
        "split_spec_hash": str(manifest["split_spec_hash"]),
        "master_seed": master_seed,
        "code_revision": code_revision,
        "declared_design": declared_design(),
        "design_frozen": settings_frozen,
        "asserted_circuits": structure["expected"],
        "n_resamples": n_resamples,
        "confidence": confidence,
        "gate_seed": gate_seed,
        "counts": structure["counts"],
        "selection_role": "source_validation",
        "evaluation_role": "untouched_test",
        "label_isolation": (
            "prediction rows omit ideal_expectation, so a test label cannot enter "
            "fitting or selection"
        ),
        "selected_configs": configs,
        "validation_predictions": predictions["validation"],
        "test_predictions": predictions["test"],
        "test_items": [
            {field: row[field] for field in TEST_ROW_FIELDS} for row in roles["test"]
        ],
        "validation_items": [
            {field: row[field] for field in VALIDATION_ROW_FIELDS}
            for row in roles["validation"]
        ],
        "gates": {
            arm["full"]: evaluate_incremental_value(
                roles["validation"],
                predictions["validation"],
                full_method=arm["full"],
                ratio_margin=RATIO_MARGIN,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=gate_seed,
            )
            for arm in ARMS.values()
        },
        "training_shuffle": {
            "seed": training_shuffle_seed,
            "permutation_strata": (
                "global across families, severities, and observables, then refit"
            ),
            "scope": (
                "paired source-validation diagnostic; it shares the rows selection "
                "used, so its interval is conditional"
            ),
            "differences": _training_shuffle_differences(
                roles["validation"],
                predictions["validation"],
                dataset_hash=dataset_hash,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=gate_seed,
            ),
        },
        "fixed_model_shuffle": _fixed_model_shuffle(
            roles["validation"],
            prediction_rows["validation"],
            prediction_manifest,
            fitted=fitted,
            dataset_hash=dataset_hash,
            shuffle_seeds=tuple(int(value) for value in fixed_model_shuffle_seeds),
        ),
    }


def _fit_arm(
    method: str,
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    master_seed: int,
) -> object:
    """Fit and select one arm, keeping each paired arm on one code path.

    The two ridge arms differ only in their model factory and the two Liao arms
    only in the dropped column, so "the same candidate family" is a structural
    property of how they are built rather than an assertion about them.
    """
    if method in TRAINING_SHUFFLE_ARMS.values():
        raise ValueError("fit the source arm on shuffled training rows instead")
    if method.startswith("liao"):
        drop = (
            (DROPPED_FEATURE,)
            if method == ARMS["capacity_matched"]["control"]
            else ()
        )
        return LiaoMitigator(random_state=master_seed, drop_features=drop).fit(
            list(train), list(validation))
    registration = next(
        (value for value in registered_methods() if value.name == method), None)
    if registration is None:
        raise ValueError(f"no registered method named {method!r}")
    model = registration.factory().fit(list(train), manifest=manifest, split_v2=True)
    model.select(list(validation))
    return model


def _arm_predict(
    method: str,
    model: object,
    rows: Sequence[Mapping[str, object]],
    prediction_manifest: Mapping[str, object],
) -> Sequence[float]:
    if method.startswith("liao"):
        return model.predict(list(rows))
    return model.predict(list(rows), manifest=prediction_manifest).predictions


def _arm_config(method: str, model: object) -> dict:
    """Record what the arm resolved to, so a reader never has to refit to see it."""
    if method.startswith("liao"):
        return dict(model.config_)
    selection = getattr(model, "_selection", None)
    if selection is None:
        raise RuntimeError("select validation rows before recording a configuration")
    return {
        "best_alpha": selection["selected_alpha"],
        "validation_selection": selection,
    }


def _keyed_predictions(
    rows: Sequence[Mapping[str, object]], values: Sequence[float],
) -> dict[str, float]:
    keyed = {}
    for row, value in zip(rows, values, strict=True):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("predictions have to be finite")
        keyed[str(row["item_id"])] = number
    return keyed


def _family_mae(
    rows: Sequence[Mapping[str, object]],
    values: Sequence[float],
    *,
    artifact_id: str,
) -> dict[str, float]:
    """Equal-weight macro MAE per family over the declared cells."""
    array = np.asarray(list(values), dtype=float)
    result = {}
    for family in sorted({str(row["family"]) for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if str(row["family"]) == family
        ]
        subset = [rows[index] for index in indices]
        records = build_cell_records(
            "campaign",
            subset,
            array[indices],
            [row["noisy_expectation"] for row in subset],
            artifact_id=artifact_id,
        )
        result[family] = float(headline_metrics(records)["mae"])
    return result


def _training_shuffle_differences(
    validation: Sequence[Mapping[str, object]],
    validation_predictions: Mapping[str, Mapping[str, float]],
    *,
    dataset_hash: str,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict:
    """Paired shuffled-minus-unshuffled intervals, per source arm and family."""
    differences: dict[str, dict[str, dict]] = {}
    for source, shuffled_name in TRAINING_SHUFFLE_ARMS.items():
        differences[source] = {}
        for family in sorted({str(row["family"]) for row in validation}):
            subset = [row for row in validation if str(row["family"]) == family]
            records = {
                name: build_cell_records(
                    name,
                    subset,
                    [validation_predictions[name][str(row["item_id"])]
                     for row in subset],
                    [row["noisy_expectation"] for row in subset],
                    artifact_id=dataset_hash,
                )
                for name in (source, shuffled_name)
            }
            differences[source][family] = asdict(circuit_blocked_bootstrap(
                records[shuffled_name],
                reference_records=records[source],
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            ))
    return differences


def _fixed_model_shuffle(
    validation: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    prediction_manifest: Mapping[str, object],
    *,
    fitted: Mapping[str, object],
    dataset_hash: str,
    shuffle_seeds: Sequence[int],
) -> dict:
    """Re-score the selected models on permuted measurements, without refitting.

    The permutation preserves the cell strata, so it asks what the fitted model
    does when the measurement no longer belongs to the circuit. It scores the
    model off the joint distribution of its inputs, which Hooker, Mentch, and
    Zhou describe as the limitation of unrestricted input permutation, so it is
    a diagnostic rather than an estimate of measurement benefit.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(prediction_rows):
        groups[tuple(row[field] for field in FIXED_MODEL_SHUFFLE_STRATA)].append(index)

    order = [str(row["item_id"]) for row in validation]
    retained: dict[str, dict[str, list[float]]] = {
        method: {} for method in TRAINING_SHUFFLE_ARMS}
    for shuffle_seed in shuffle_seeds:
        permuted = list(prediction_rows)
        for indices in groups.values():
            changed = shuffle_noisy_items(
                [prediction_rows[index] for index in indices], seed=shuffle_seed)
            for index, row in zip(indices, changed, strict=True):
                permuted[index] = row
        for method in retained:
            values = _arm_predict(
                method, fitted[method], permuted, prediction_manifest)
            retained[method][str(shuffle_seed)] = [float(value) for value in values]

    return {
        "seeds": list(shuffle_seeds),
        "permutation_strata": list(FIXED_MODEL_SHUFFLE_STRATA),
        "scope": (
            "selected models re-scored on permuted measurements, with no refit; "
            "off-manifold, so a diagnostic rather than an effect estimate"
        ),
        # Every permutation's predictions, not the scores derived from them. A
        # summary alone cannot be re-derived without refitting the model, so the
        # diagnostic could not be regenerated from the record.
        "item_order": order,
        "predictions": retained,
        "mae": _fixed_model_shuffle_mae(validation, retained, order,
                                        dataset_hash=dataset_hash),
    }


def _fixed_model_shuffle_mae(
    validation: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    order: Sequence[str],
    *,
    dataset_hash: str,
) -> dict:
    """Summarize retained permutation predictions, per method and family."""
    if [str(row["item_id"]) for row in validation] != list(order):
        raise ValueError("permutation predictions must follow the validation order")
    summary: dict[str, dict[str, dict]] = {}
    for method, by_seed in predictions.items():
        scores: dict[str, list[float]] = defaultdict(list)
        for _, values in sorted(by_seed.items(), key=lambda pair: int(pair[0])):
            for family, mae in _family_mae(
                    validation, values, artifact_id=dataset_hash).items():
                scores[family].append(mae)
        summary[method] = {
            family: {
                "mean": float(np.mean(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "per_seed": [float(value) for value in values],
            }
            for family, values in sorted(scores.items())
        }
    return summary


# --------------------------------------------------------------------------
# The campaign endpoint and diagnostics, regenerated from the records alone
# --------------------------------------------------------------------------


def regenerate_setting_diagnostics(
    record: Mapping[str, object],
    *,
    n_resamples: int | None = None,
    confidence: float | None = None,
) -> dict:
    """Recompute one setting's gates and both shuffle diagnostics from its record.

    No model is refitted and the dataset directory is never read. Left to their
    defaults, the resample count and confidence come from the record, so the
    result has to equal what `build_setting_record` stored; a test asserts
    exactly that, which is what makes the retained maps sufficient rather than
    merely present.
    """
    validation = list(record["validation_items"])
    predictions = record["validation_predictions"]
    dataset_hash = str(record["dataset_hash"])
    n_resamples = int(record["n_resamples"] if n_resamples is None else n_resamples)
    confidence = float(record["confidence"] if confidence is None else confidence)
    gate_seed = int(record["gate_seed"])
    shuffle = record["fixed_model_shuffle"]
    return {
        "gates": {
            arm["full"]: evaluate_incremental_value(
                validation,
                predictions,
                full_method=arm["full"],
                ratio_margin=RATIO_MARGIN,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=gate_seed,
            )
            for arm in ARMS.values()
        },
        "training_shuffle_differences": _training_shuffle_differences(
            validation,
            predictions,
            dataset_hash=dataset_hash,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=gate_seed,
        ),
        "fixed_model_shuffle_mae": _fixed_model_shuffle_mae(
            validation,
            shuffle["predictions"],
            shuffle["item_order"],
            dataset_hash=dataset_hash,
        ),
    }


def evaluate_campaign(
    records: Iterable[Mapping[str, object]],
    *,
    rosters: Mapping[str, Mapping[str, object]] | None = None,
    baseline_regime: str = BASELINE_REGIME,
    contrast_regime: str = CONTRAST_REGIME,
    root_seed: int = ROOT_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    required_families: int = REQUIRED_FAMILIES,
    primary_arm: str = PRIMARY_ARM,
    primary_size: int = PRIMARY_SIZE,
) -> dict:
    """Estimate every contrast from the records and apply the promotion rule.

    Nothing is fitted here. A setting whose record is missing is reported as a
    failed setting, and a failed setting can never count as a replication.

    ``rosters`` binds each setting to the run artifact that produced its
    descriptive results, zero-noise extrapolation among them. The endpoint arms
    are fitted by `build_setting_record` and never by the roster, so a missing
    roster is reported rather than fatal; a roster built from a different dataset
    is fatal, because that is a number attached to the wrong setting.
    """

    if primary_arm not in ARMS:
        raise ValueError(f"unknown arm {primary_arm!r}")
    indexed: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for record in records:
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"records have to carry schema_version {RECORD_SCHEMA_VERSION!r}")
        key = (str(record["regime"]), int(record["seed"]), int(record["size"]))
        if setting_key(*key) != record["setting"]:
            raise ValueError(f"record {record['setting']!r} disagrees with its fields")
        if key in indexed:
            raise ValueError(f"setting {record['setting']} appears twice")
        indexed[key] = record

    supplied = {setting_key(*key) for key in indexed}
    frozen = (
        all(bool(value.get("design_frozen")) for value in indexed.values())
        and n_resamples == BOOTSTRAP_RESAMPLES
        and confidence == CONFIDENCE
        and root_seed == ROOT_SEED
        and required_families == REQUIRED_FAMILIES
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "declared_design": declared_design(),
        "design_frozen": frozen,
        "baseline_regime": baseline_regime,
        "contrast_regime": contrast_regime,
        "root_seed": root_seed,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "required_families": required_families,
        "primary_arm": primary_arm,
        "primary_size": primary_size,
        "code_revisions": sorted(
            {str(value["code_revision"]) for value in indexed.values()}),
        "dataset_hashes": {
            setting_key(*key): str(value["dataset_hash"])
            for key, value in sorted(indexed.items())
        },
        "failed_settings": [
            key for key in campaign_setting_keys() if key not in supplied
        ],
        # Diagnostics are regenerated here rather than copied from the record, so
        # the reported gate and shuffle numbers come out of the same saved rows
        # and prediction maps that the endpoint does.
        # Each setting replays at its own recorded resample count and confidence,
        # so the regenerated diagnostic equals the one the fit stored rather than
        # a differently resampled estimate of the same quantity.
        "diagnostics": {
            setting_key(*key): regenerate_setting_diagnostics(value)
            for key, value in sorted(indexed.items())
        },
        "contrasts": {},
        "replication": {},
        **_roster_bindings(indexed, rosters),
    }

    sizes = sorted({key[2] for key in indexed} | {primary_size})
    seeds = sorted({key[1] for key in indexed} | set(SEEDS))
    for arm_name, arm in ARMS.items():
        report["contrasts"][arm_name] = {}
        report["replication"][arm_name] = {}
        for size in sizes:
            by_seed = {
                str(seed): _seed_contrast(
                    indexed,
                    arm=arm,
                    seed=seed,
                    size=size,
                    baseline_regime=baseline_regime,
                    contrast_regime=contrast_regime,
                    root_seed=root_seed,
                    n_resamples=n_resamples,
                    confidence=confidence,
                )
                for seed in seeds
            }
            report["contrasts"][arm_name][str(size)] = by_seed
            report["replication"][arm_name][str(size)] = _replication(
                by_seed,
                required_families=required_families,
                contrast_regime=contrast_regime,
            )

    if not frozen:
        # A rehearsal runs on a different shape, and its scores do not enter the
        # paper. Making that mechanical means a rehearsal record can never reach
        # a publication path, however its numbers happen to land.
        for by_size in report["replication"].values():
            for summary in by_size.values():
                summary["promoted"] = False
                summary["publication_path"] = "rehearsal_not_publishable"
    primary = report["replication"][primary_arm][str(primary_size)]
    report["primary"] = {
        "arm": primary_arm,
        "full_method": ARMS[primary_arm]["full"],
        "control_method": ARMS[primary_arm]["control"],
        "size": primary_size,
        "hypothesis": (
            "the larger evolution step increases the incremental predictive benefit "
            "of the noisy measurement relative to the shipped step"
        ),
        "promotion_rule": (
            "strictly positive 95% contrast intervals in both families in at least "
            "two of the three seeds"
        ),
        **primary,
    }
    return report


def _roster_bindings(
    indexed: Mapping[tuple[str, int, int], Mapping[str, object]],
    rosters: Mapping[str, Mapping[str, object]] | None,
) -> dict:
    """Tie each setting's descriptive run artifact to the dataset it scored."""
    bound, missing = {}, []
    for key, record in sorted(indexed.items()):
        name = setting_key(*key)
        binding = (rosters or {}).get(name)
        if binding is None:
            missing.append(name)
            continue
        if str(binding["dataset_hash"]) != str(record["dataset_hash"]):
            raise ValueError(
                f"{name}: the roster reports dataset {binding['dataset_hash']!r} "
                f"and the record reports {record['dataset_hash']!r}"
            )
        bound[name] = {
            "artifact_id": str(binding["artifact_id"]),
            "dataset_hash": str(binding["dataset_hash"]),
            "dataset_manifest_sha256": str(binding["dataset_manifest_sha256"]),
            "methods": sorted(str(value) for value in binding["methods"]),
        }
    return {"roster_artifacts": bound, "settings_without_a_roster": missing}


def _seed_contrast(
    indexed: Mapping[tuple[str, int, int], Mapping[str, object]],
    *,
    arm: Mapping[str, str],
    seed: int,
    size: int,
    baseline_regime: str,
    contrast_regime: str,
    root_seed: int,
    n_resamples: int,
    confidence: float,
) -> dict:
    """Contrast one seed and size, or name the record that made it impossible."""
    regimes = {}
    for label in (baseline_regime, contrast_regime):
        record = indexed.get((label, seed, size))
        if record is None:
            return {
                "status": "failed_setting",
                "reason": "missing_record",
                "missing": [setting_key(label, seed, size)],
            }
        for method in (arm["full"], arm["control"]):
            if method not in record["test_predictions"]:
                return {
                    "status": "failed_setting",
                    "reason": "missing_arm_predictions",
                    "missing": [f"{setting_key(label, seed, size)}/{method}"],
                }
        regimes[label] = {
            "items": record["test_items"],
            "predictions_by_method": record["test_predictions"],
        }
    return evaluate_gain_contrast(
        regimes,
        full_method=arm["full"],
        control_method=arm["control"],
        baseline_regime=baseline_regime,
        contrast_regime=contrast_regime,
        confidence=confidence,
        n_resamples=n_resamples,
        root_seed=_setting_stream_seed(root_seed, seed, size),
    )


def _setting_stream_seed(root_seed: int, seed: int, size: int) -> int:
    """One reproducible root per seed and size, derived from the campaign root."""
    entropy = np.random.SeedSequence([root_seed, seed, size])
    return int(entropy.generate_state(1, dtype=np.uint32)[0])


def _replication(
    by_seed: Mapping[str, Mapping[str, object]],
    *,
    required_families: int,
    contrast_regime: str,
) -> dict:
    """Apply the frozen promotion rule to one arm and size.

    A successful seed needs strictly positive contrast intervals in every
    required family of that same seed. Two successes promote, and a dissenting
    third is named rather than hidden. Reversal is the mirror rule, so a
    consistently reversed contrast is reported as evidence against the
    predeclared direction rather than as an inconclusive result.
    """
    successful, reversed_seeds, evaluated, failed = [], [], [], []
    endpoint_support: dict[str, dict[str, bool]] = {}
    for label, result in sorted(by_seed.items(), key=lambda pair: int(pair[0])):
        families = {
            name: value
            for name, value in result.get("families", {}).items()
            if value.get("status") == "evaluated"
        }
        if len(families) < required_families:
            failed.append(label)
            continue
        evaluated.append(label)
        contrasts = [value["contrast"] for value in families.values()]
        if all(value["excludes_zero_above"] for value in contrasts):
            successful.append(label)
        elif all(value["excludes_zero_below"] for value in contrasts):
            reversed_seeds.append(label)
        endpoint_support[label] = {
            name: bool(
                value["regimes"][contrast_regime]["gain_interval"]
                ["excludes_zero_above"]
            )
            for name, value in sorted(families.items())
        }

    return {
        "successful_seeds": successful,
        "reversed_seeds": reversed_seeds,
        "evaluated_seeds": evaluated,
        "failed_seeds": failed,
        "successes": len(successful),
        "promoted": len(successful) >= 2,
        "publication_path": _publication_path(
            successes=len(successful), reversals=len(reversed_seeds)),
        "positive_endpoint_benefit_supported": endpoint_support,
        "endpoint_caveat": (
            "promotion licenses an increase in signed relative gain only; a positive "
            "contrast between two negative gains is not a win, and positive benefit "
            "at the contrast endpoint needs that endpoint's own interval"
        ),
    }


def _publication_path(*, successes: int, reversals: int) -> str:
    if successes >= 3:
        return "promote_three_of_three"
    if successes == 2:
        return "promote_two_of_three_with_dissent"
    if successes == 1:
        return "no_headline_one_replication"
    # Two jointly reversed seeds and no successful one, mirroring the two-of-three
    # promotion threshold. Requiring every evaluated seed to reverse as well would
    # make an unavailable third seed strengthen the claim that an inconclusive
    # third seed weakens, which reverses the direction information should move in.
    if reversals >= 2:
        return "evidence_against_predeclared_direction"
    return "no_headline_inconclusive"


def build_campaign_tables(report: Mapping[str, object]) -> dict:
    """Flatten the report into the rows the manuscript's tables print.

    Every number here comes from the report, which comes from the records, so a
    printed table cannot disagree with the file that produced it.
    """
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {REPORT_SCHEMA_VERSION!r}")
    endpoints, contrasts, replication = [], [], []
    for arm_name, by_size in sorted(report["contrasts"].items()):
        for size, by_seed in sorted(by_size.items(), key=lambda pair: int(pair[0])):
            for seed, result in sorted(by_seed.items(), key=lambda pair: int(pair[0])):
                for family, value in sorted(result.get("families", {}).items()):
                    for regime, summary in value["regimes"].items():
                        row = {
                            "arm": arm_name,
                            "size": int(size),
                            "seed": int(seed),
                            "family": family,
                            "regime": regime,
                            "available": summary is not None,
                            "reason": value["reason"],
                        }
                        # A regime that never produced this family has no endpoint
                        # to print. The row still appears, so an unavailable
                        # comparison reads as unavailable rather than as absent.
                        interval = (summary or {}).get("gain_interval") or {}
                        row.update({
                            key: None if summary is None else summary[key]
                            for key in ("full_mae", "control_mae",
                                        "absolute_reduction", "gain")
                        })
                        row["gain_lower"] = interval.get("lower")
                        row["gain_upper"] = interval.get("upper")
                        endpoints.append(row)
                    contrast = value["contrast"] or {}
                    contrasts.append({
                        "arm": arm_name,
                        "size": int(size),
                        "seed": int(seed),
                        "family": family,
                        "status": value["status"],
                        "reason": value["reason"],
                        "estimate": contrast.get("estimate"),
                        "lower": contrast.get("lower"),
                        "upper": contrast.get("upper"),
                        "excludes_zero_above": contrast.get("excludes_zero_above"),
                    })
    for arm_name, by_size in sorted(report["replication"].items()):
        for size, summary in sorted(by_size.items(), key=lambda pair: int(pair[0])):
            replication.append({
                "arm": arm_name,
                "size": int(size),
                "successes": summary["successes"],
                "successful_seeds": list(summary["successful_seeds"]),
                "reversed_seeds": list(summary["reversed_seeds"]),
                "failed_seeds": list(summary["failed_seeds"]),
                "promoted": summary["promoted"],
                "publication_path": summary["publication_path"],
            })
    gates, training_shuffle, fixed_model_shuffle = [], [], []
    for setting, diagnostics in sorted(report.get("diagnostics", {}).items()):
        for method, gate in sorted(diagnostics["gates"].items()):
            gates.append({
                "setting": setting,
                "full_method": method,
                "status": gate["status"],
                "reason": gate["reason"],
                "passing_families": list(gate["passing_families"]),
            })
        for method, by_family in sorted(
                diagnostics["training_shuffle_differences"].items()):
            for family, interval in sorted(by_family.items()):
                training_shuffle.append({
                    "setting": setting,
                    "method": method,
                    "family": family,
                    "estimate": interval["estimate"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                })
        for method, by_family in sorted(
                diagnostics["fixed_model_shuffle_mae"].items()):
            for family, summary in sorted(by_family.items()):
                fixed_model_shuffle.append({
                    "setting": setting,
                    "method": method,
                    "family": family,
                    "mean_mae": summary["mean"],
                    "min_mae": summary["min"],
                    "max_mae": summary["max"],
                    "permutations": len(summary["per_seed"]),
                })
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "failed_settings": list(report["failed_settings"]),
        "endpoints": endpoints,
        "contrasts": contrasts,
        "replication": replication,
        "gates": gates,
        "training_shuffle": training_shuffle,
        "fixed_model_shuffle": fixed_model_shuffle,
    }
