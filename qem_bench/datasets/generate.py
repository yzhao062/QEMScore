"""Deterministic dataset generation with named seed streams and a canonical hash.

The circuit stream for instance i is SeedSequence(master_seed, spawn_key=(0, i)).
The sampler and mixed-noise stream for that instance's measurement group is
SeedSequence(master_seed, spawn_key=(1, i)). Random-family builders split the
stored circuit seed into documented substreams. Every derived integer seed is
stored on its items.

All current observables are Z-type, so one computational-basis measurement per
(circuit, noise, shots) configuration estimates every sibling observable. Each
measurement group is executed once and charged once.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

import qem_bench
from qem_bench.circuits.heisenberg import (
    HeisenbergParams,
    build_heisenberg_circuit,
    sample_heisenberg_params,
)
from qem_bench.circuits.near_clifford import (
    NearCliffordParams,
    build_near_clifford_circuit,
    sample_near_clifford_params,
)
from qem_bench.circuits.qaoa import QAOAParams, build_qaoa_circuit, sample_qaoa_params
from qem_bench.circuits.random_clifford import (
    RandomCliffordParams,
    build_random_clifford_circuit,
    sample_random_clifford_params,
)
from qem_bench.circuits.tfi import TFIParams, build_tfi_circuit, sample_tfi_params
from qem_bench.datasets.schema import (
    FAMILY_LABEL_METHODS,
    FAMILY_STRATA,
    FEATURE_SPEC_VERSION,
    FEATURES,
    validate_item,
)
from qem_bench.datasets.splits import SplitSpec
from qem_bench.labels.statevector import ideal_expectation as statevector_expectation
from qem_bench.labels.stim_labels import ideal_expectation as stim_expectation
from qem_bench.noise.models import DEFAULT_NOISE_FAMILY, SEVERITY_GRIDS
from qem_bench.observables import z_expectation_from_counts, z_support_label
from qem_bench.reproducibility import environment_contract
from qem_bench.sampling import sample_counts
from qem_bench.validation import LEGACY_SCHEMA_VERSION

SEED_FORMULA = (
    "circuit stream: SeedSequence(master, spawn_key=(0, instance)); "
    "sampler and mixed-noise stream (one per measurement group): "
    "SeedSequence(master, spawn_key=(1, instance)); "
    "random-family builder substreams under circuit_seed: one-qubit=(0,), "
    "pairing=(1,), entangler=(2,), near-Clifford insertion=(3,)"
)

PRESETS: dict[str, dict] = {
    # Extended smoke tier: runs in minutes on a laptop.
    "t0-smoke": {
        "master_seed": 20260719,
        "family": "tfi",
        "stratum": "continuous_regression",
        "label_method": "statevector",
        "n_qubits": [4, 5, 6],
        "steps": [1, 2, 3],
        "dt": 0.2,
        "n_train": 48,
        "n_test": 32,
        "shots": 2048,
        "severities": ["L1", "L2", "L3"],
        "observables": ["z_mid", "zz_mid"],
    },
    # Micro tiers for unit tests and CI.
    "t0-micro": {
        "master_seed": 7,
        "family": "tfi",
        "stratum": "continuous_regression",
        "label_method": "statevector",
        "n_qubits": [3],
        "steps": [1, 2],
        "dt": 0.2,
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
    "t0-qaoa-micro": {
        "master_seed": 17,
        "family": "qaoa",
        "stratum": "continuous_regression",
        "label_method": "statevector",
        "n_qubits": [4, 6],
        "p": [1, 2],
        "graph_classes": ["path", "cycle", "erdos_renyi", "3_regular"],
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
    "t0-heisenberg-micro": {
        "master_seed": 27,
        "family": "heisenberg",
        "stratum": "continuous_regression",
        "label_method": "statevector",
        "n_qubits": [3, 4],
        "steps": [1, 2],
        "dt": 0.15,
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
    "t0-rc-micro": {
        "master_seed": 17,
        "family": "random_clifford",
        "stratum": "clifford_control",
        "label_method": "stim",
        "n_qubits": [3, 4, 5, 6],
        "depth": [2, 4],
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
    "t0-nc-micro": {
        "master_seed": 23,
        "family": "near_clifford",
        "stratum": "continuous_regression",
        "label_method": "statevector",
        "n_qubits": [3, 4, 5, 6],
        "depth": [2, 4],
        "non_clifford_count": [1, 2, 3],
        "theta": [0.4487989505128276, 0.6283185307179586],
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
}

# These named fixtures retain their original item serializer. Their golden
# hashes live only in tests and tools, never in production validation. The
# strings detach serializer selection from later mutations of the public table.
_FROZEN_LEGACY_CONFIGURATIONS = MappingProxyType(
    {
        name: json.dumps(
            {**PRESETS[name], "noise_family": DEFAULT_NOISE_FAMILY},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for name in ("t0-micro", "t0-smoke")
    }
)
LEGACY_FROZEN_PRESETS = frozenset(_FROZEN_LEGACY_CONFIGURATIONS)

PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD = (
    "physical_identity_encoding_profiles"
)
LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE = "legacy-binary64-v1"
LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE = "legacy-tfi-rounded-12-v1"
UNKNOWN_IDENTITY_ENCODING_PROFILE = "unknown"
PHYSICAL_IDENTITY_ENCODING_PROFILE_DOMAINS = MappingProxyType(
    {
        family: frozenset(
            (
                LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
                LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE,
            )
            if family == "tfi"
            else (LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,)
        )
        for family in FAMILY_STRATA
    }
)


def _uses_frozen_legacy_serializer(preset_name: object, config: dict) -> bool:
    if not isinstance(preset_name, str) or preset_name not in LEGACY_FROZEN_PRESETS:
        return False
    actual = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return actual == _FROZEN_LEGACY_CONFIGURATIONS[preset_name]


def validate_physical_identity_encoding_profiles(
    value: object,
    *,
    families: set[str] | frozenset[str],
    allow_unknown: bool = False,
) -> dict[str, str]:
    """Validate and canonically order a closed per-family profile map."""

    expected_families = set(families)
    unknown_expected = expected_families - set(
        PHYSICAL_IDENTITY_ENCODING_PROFILE_DOMAINS
    )
    if unknown_expected:
        raise ValueError(
            "physical identity encoding profiles contain unknown families: "
            f"{sorted(unknown_expected)}"
        )
    if not expected_families:
        raise ValueError(
            "physical identity encoding profiles require at least one family"
        )
    if not isinstance(value, Mapping):
        raise ValueError("physical identity encoding profiles must be an object")
    if any(not isinstance(family, str) for family in value):
        raise ValueError(
            "physical identity encoding profile family names must be strings"
        )
    actual_families = set(value)
    unknown_declared = actual_families - set(
        PHYSICAL_IDENTITY_ENCODING_PROFILE_DOMAINS
    )
    if unknown_declared:
        raise ValueError(
            "physical identity encoding profiles contain unknown families: "
            f"{sorted(unknown_declared)}"
        )
    if actual_families != expected_families:
        raise ValueError(
            "physical identity encoding profile families do not match dataset "
            f"families: declared={sorted(actual_families)}, "
            f"dataset={sorted(expected_families)}"
        )

    profiles = {}
    for family in sorted(expected_families):
        profile = value[family]
        allowed = PHYSICAL_IDENTITY_ENCODING_PROFILE_DOMAINS[family]
        if not isinstance(profile, str) or (
            profile not in allowed
            and not (allow_unknown and profile == UNKNOWN_IDENTITY_ENCODING_PROFILE)
        ):
            raise ValueError(
                "unknown physical identity encoding profile for family "
                f"{family!r}: {profile!r}; expected one of {sorted(allowed)}"
            )
        profiles[family] = profile
    return profiles


def legacy_physical_identity_encoding_profiles(
    preset_name: object, config: object
) -> dict[str, str]:
    """Return the profile selected by the legacy generator serializer."""

    if not isinstance(config, Mapping):
        raise ValueError("legacy dataset config must be an object")
    canonical_config = dict(config)
    family = canonical_config.get("family")
    if family not in PHYSICAL_IDENTITY_ENCODING_PROFILE_DOMAINS:
        raise ValueError(f"unknown circuit family {family!r}")
    profile = (
        LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
        if _uses_frozen_legacy_serializer(preset_name, canonical_config)
        else LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
    )
    return {str(family): profile}

CircuitParams = (
    TFIParams
    | QAOAParams
    | HeisenbergParams
    | RandomCliffordParams
    | NearCliffordParams
)


def _observable_support(name: str, params: CircuitParams) -> tuple[int, ...]:
    mid = params.n_qubits // 2
    if name == "z_mid":
        return (mid,)
    if name == "zz_mid":
        return (mid - 1, mid)
    if name == "zz_edge":
        if not isinstance(params, QAOAParams):
            raise ValueError("zz_edge is defined only for QAOA circuits")
        if not params.edges:
            raise ValueError("zz_edge requires a QAOA graph with at least one edge")
        return tuple(params.edges[0])
    raise ValueError(f"unknown observable {name}")


def _seed_int(master: int, spawn_key: tuple[int, ...]) -> int:
    sequence = np.random.SeedSequence(master, spawn_key=spawn_key)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sample_and_build_circuit(
    cfg: dict, rng: np.random.Generator, instance: int, circuit_seed: int
) -> tuple[CircuitParams, object]:
    family = cfg["family"]
    if family == "tfi":
        params = sample_tfi_params(
            rng,
            cfg["n_qubits"],
            cfg["steps"],
            cfg["dt"],
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_tfi_circuit(params)
    if family == "qaoa":
        params = sample_qaoa_params(
            rng,
            cfg["n_qubits"],
            cfg["p"],
            cfg["graph_classes"],
            instance=instance,
            circuit_seed=circuit_seed,
            er_edge_probability=cfg.get("er_edge_probability"),
        )
        return params, build_qaoa_circuit(params)
    if family == "heisenberg":
        params = sample_heisenberg_params(
            rng,
            cfg["n_qubits"],
            cfg["steps"],
            cfg["dt"],
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_heisenberg_circuit(params)
    if family == "random_clifford":
        params = sample_random_clifford_params(
            rng,
            cfg["n_qubits"],
            cfg["depth"],
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_random_clifford_circuit(params)
    if family == "near_clifford":
        params = sample_near_clifford_params(
            rng,
            cfg["n_qubits"],
            cfg["depth"],
            cfg["non_clifford_count"],
            cfg["theta"],
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_near_clifford_circuit(params)
    raise ValueError(f"unknown circuit family {family!r}")


def _family_parameter_fields(params: CircuitParams) -> dict:
    if isinstance(params, TFIParams):
        return {
            "steps": params.steps,
            "j": params.j,
            "h": params.h,
            "dt": params.dt,
        }
    if isinstance(params, QAOAParams):
        return {
            "p": params.p,
            "graph_class": params.graph_class,
            "edges": [list(edge) for edge in params.edges],
            "edge_probability": params.edge_probability,
            "gammas": list(params.gammas),
            "betas": list(params.betas),
        }
    if isinstance(params, HeisenbergParams):
        return {
            "steps": params.steps,
            "jx": params.jx,
            "jy": params.jy,
            "jz": params.jz,
            "dt": params.dt,
        }
    if isinstance(params, RandomCliffordParams):
        return {"depth": params.depth}
    return {
        "depth": params.depth,
        "non_clifford_count": params.non_clifford_count,
        "theta": params.theta,
    }


def _frozen_legacy_parameter_fields(params: CircuitParams) -> dict:
    fields = _family_parameter_fields(params)
    if isinstance(params, TFIParams):
        fields["j"] = round(params.j, 12)
        fields["h"] = round(params.h, 12)
    return fields


def _canonical_lines(items: list[dict]) -> list[str]:
    ordered = sorted(items, key=lambda item: item["item_id"])
    return [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in ordered]


def dataset_hash(items: list[dict]) -> str:
    payload = "\n".join(_canonical_lines(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_shots(items: list[dict], split: str | None = None) -> int:
    """Total circuit evaluations across unique measurement groups."""
    seen: dict[str, int] = {}
    for item in items:
        if split is not None and item["split"] != split:
            continue
        group = item["measurement_group"]
        if group in seen and seen[group] != item["shots"]:
            raise ValueError(f"measurement group {group} has conflicting shots values")
        seen[group] = item["shots"]
    return sum(seen.values())


def _physical_observables(items: list[dict], role: str) -> set[tuple[int, str]]:
    return {
        (item["n_qubits"], item["pauli_label"])
        for item in items
        if item["split"] == role
    }


def _require_source_observable_closure(
    items: list[dict], *, split_name: str
) -> None:
    missing = sorted(
        _physical_observables(items, "test")
        - _physical_observables(items, "train")
    )
    if missing:
        raise ValueError(
            f"split {split_name!r} is not source-observable closed; "
            f"test observables absent from training: {missing!r}"
        )


def _construct_legacy_source_observable_closure(
    items: list[dict], *, n_train: int, split_name: str
) -> None:
    if not (
        _physical_observables(items, "test")
        - _physical_observables(items, "train")
    ):
        return

    rows_by_group: dict[str, list[dict]] = {}
    for item in items:
        rows_by_group.setdefault(str(item["measurement_group"]), []).append(item)
    observables_by_group = {
        group: {
            (item["n_qubits"], item["pauli_label"])
            for item in rows
        }
        for group, rows in rows_by_group.items()
    }
    original_train = {
        group
        for group, rows in rows_by_group.items()
        if rows[0]["split"] == "train"
    }

    uncovered = set().union(*observables_by_group.values())
    selected: list[str] = []
    while uncovered and len(selected) < n_train:
        candidates = [
            group
            for group in rows_by_group
            if group not in selected and observables_by_group[group] & uncovered
        ]
        group = min(
            candidates,
            key=lambda candidate: (
                -len(observables_by_group[candidate] & uncovered),
                candidate not in original_train,
                candidate,
            ),
        )
        selected.append(group)
        uncovered -= observables_by_group[group]

    if uncovered:
        _require_source_observable_closure(items, split_name=split_name)

    for group in sorted(original_train):
        if len(selected) == n_train:
            break
        if group not in selected:
            selected.append(group)
    for group in sorted(rows_by_group):
        if len(selected) == n_train:
            break
        if group not in selected:
            selected.append(group)

    selected_groups = set(selected)
    for group, rows in rows_by_group.items():
        role = "train" if group in selected_groups else "test"
        for item in rows:
            item["split"] = role
    _require_source_observable_closure(items, split_name=split_name)


def generate(
    preset: str | dict | SplitSpec,
    out_dir: str | Path,
    master_seed: int | None = None,
    *,
    noise_family: str | None = None,
) -> dict:
    """Generate a dataset directory and return its manifest."""
    if isinstance(preset, str):
        from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split

        if preset in SPLIT_PRESETS:
            if noise_family is not None:
                raise ValueError(
                    "split-v2 noise families are declared by SplitSpec, not an override"
                )
            return generate_split(preset, out_dir, master_seed=master_seed)
    elif isinstance(preset, SplitSpec) or (
        isinstance(preset, dict) and "split_id" in preset
    ):
        from qem_bench.datasets.split_generate import generate_split

        if noise_family is not None:
            raise ValueError(
                "split-v2 noise families are declared by SplitSpec, not an override"
            )
        return generate_split(preset, out_dir, master_seed=master_seed)

    cfg = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
    preset_name = preset if isinstance(preset, str) else cfg.get("name", "custom")
    if master_seed is not None:
        if type(master_seed) is not int or master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        cfg["master_seed"] = master_seed
    if noise_family is not None:
        cfg["noise_family"] = str(noise_family)

    family = str(cfg["family"])
    if family not in FAMILY_STRATA:
        raise ValueError(f"unknown circuit family {family!r}")
    expected_stratum = FAMILY_STRATA[family]
    expected_label_method = FAMILY_LABEL_METHODS[family]
    if cfg.get("stratum", expected_stratum) != expected_stratum:
        raise ValueError(f"family {family!r} requires stratum {expected_stratum!r}")
    if cfg.get("label_method", expected_label_method) != expected_label_method:
        raise ValueError(
            f"family {family!r} requires label_method {expected_label_method!r}"
        )
    cfg["stratum"] = expected_stratum
    cfg["label_method"] = expected_label_method

    selected_noise_family = str(cfg.get("noise_family", DEFAULT_NOISE_FAMILY))
    cfg["noise_family"] = selected_noise_family
    if selected_noise_family not in SEVERITY_GRIDS:
        names = ", ".join(SEVERITY_GRIDS)
        raise ValueError(
            f"unknown noise family {selected_noise_family!r}; expected one of: {names}"
        )
    unknown_severities = [
        level
        for level in cfg["severities"]
        if level not in SEVERITY_GRIDS[selected_noise_family]
    ]
    if unknown_severities:
        raise ValueError(
            f"unknown severities for {selected_noise_family}: {unknown_severities}"
        )

    if type(cfg["master_seed"]) is not int or cfg["master_seed"] < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    for field, minimum in (("n_train", 0), ("n_test", 0), ("shots", 1)):
        value = cfg[field]
        if type(value) is not int or value < minimum:
            bound = "positive" if minimum == 1 else "nonnegative"
            raise ValueError(f"{field} must be a {bound} integer")

    artifact_environment_contract = environment_contract()
    master = cfg["master_seed"]
    physical_identity_encoding_profiles = (
        legacy_physical_identity_encoding_profiles(preset_name, cfg)
    )
    frozen_legacy_serializer = (
        physical_identity_encoding_profiles[family]
        == LEGACY_TFI_ROUNDED_IDENTITY_ENCODING_PROFILE
    )
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"dataset artifact path already exists: {out}") from exc

    n_total = cfg["n_train"] + cfg["n_test"]
    items: list[dict] = []
    label_evals = {"statevector": 0, "stim": 0}

    for instance in range(n_total):
        split = "train" if instance < cfg["n_train"] else "test"
        circuit_seed = _seed_int(master, (0, instance))
        rng = np.random.default_rng(
            np.random.SeedSequence(master, spawn_key=(0, instance))
        )
        params, circuit = _sample_and_build_circuit(
            cfg, rng, instance, circuit_seed
        )
        parameter_fields = (
            _frozen_legacy_parameter_fields(params)
            if frozen_legacy_serializer
            else _family_parameter_fields(params)
        )
        severity = cfg["severities"][instance % len(cfg["severities"])]

        group_id = f"{family}-{instance:04d}-g0"
        sampler_seed = _seed_int(master, (1, instance))
        counts, structure = sample_counts(
            circuit,
            severity=severity,
            shots=cfg["shots"],
            sampler_seed=sampler_seed,
            transpile_seed=circuit_seed,
            noise_family=selected_noise_family,
        )

        for observable in cfg["observables"]:
            support = _observable_support(observable, params)
            pauli = z_support_label(params.n_qubits, support)
            if expected_label_method == "statevector":
                ideal = statevector_expectation(circuit, pauli)
            else:
                ideal = stim_expectation(circuit, pauli)
            label_evals[expected_label_method] += 1

            noisy, noisy_stderr = z_expectation_from_counts(
                counts, support, cfg["shots"]
            )
            item = {
                "item_id": f"{family}-{instance:04d}-{observable}",
                "family": family,
                "stratum": expected_stratum,
                "split": split,
                "instance": instance,
                "n_qubits": params.n_qubits,
                "circuit_seed": circuit_seed,
                "observable": observable,
                "pauli_label": pauli,
                "obs_locality": len(support),
                "noise_family": selected_noise_family,
                "severity": severity,
                "shots": cfg["shots"],
                "measurement_group": group_id,
                "sampler_seed": sampler_seed,
                "noisy_expectation": round(noisy, 12),
                "noisy_stderr": round(noisy_stderr, 12),
                "ideal_expectation": round(ideal, 12),
                "label_method": expected_label_method,
                "two_qubit_gates": structure["two_qubit_gates"],
                "transpiled_depth": structure["transpiled_depth"],
            }
            item.update(parameter_fields)
            validate_item(item, schema_version=LEGACY_SCHEMA_VERSION)
            items.append(item)

    _construct_legacy_source_observable_closure(
        items,
        n_train=cfg["n_train"],
        split_name=str(preset_name),
    )
    _require_source_observable_closure(items, split_name=str(preset_name))

    written_items = items
    if frozen_legacy_serializer:
        compatible_items = []
        for item in items:
            compatible = dict(item)
            compatible.pop("stratum")
            compatible_items.append(compatible)
        written_items = compatible_items

    lines = _canonical_lines(written_items)
    (out / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import qiskit
    import qiskit_aer
    import sklearn
    import stim

    manifest = {
        "dataset_schema_version": LEGACY_SCHEMA_VERSION,
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD: (
            physical_identity_encoding_profiles
        ),
        "environment_contract": artifact_environment_contract,
        "preset": preset_name,
        "config": cfg,
        "master_seed": master,
        "seed_formula": SEED_FORMULA,
        "feature_spec": {"version": FEATURE_SPEC_VERSION, "features": FEATURES},
        "severity_grid": SEVERITY_GRIDS[selected_noise_family],
        "severity_grids": SEVERITY_GRIDS,
        "counts": {
            "items": len(written_items),
            "train_items": sum(
                item["split"] == "train" for item in written_items
            ),
            "test_items": sum(item["split"] == "test" for item in written_items),
            "instances": n_total,
            "measurement_groups": len(
                {item["measurement_group"] for item in written_items}
            ),
        },
        "generation_ledger": {
            "train_circuit_evals": group_shots(written_items, "train"),
            "test_circuit_evals": group_shots(written_items, "test"),
            "label_evals_statevector": label_evals["statevector"],
            "label_evals_stim": label_evals["stim"],
            "note": (
                "circuit evals are counted once per measurement group; "
                "label_evals are exact-simulation calls, logged separately, "
                "never circuit evaluations"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "qiskit": qiskit.__version__,
            "qiskit-aer": qiskit_aer.__version__,
            "stim": stim.__version__,
            "qem-bench": qem_bench.__version__,
        },
        "dataset_hash": dataset_hash(written_items),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
