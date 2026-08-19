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
from pathlib import Path

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
# hashes live only in tests and tools, never in production validation.
LEGACY_FROZEN_PRESETS = frozenset({"t0-micro", "t0-smoke"})


def _uses_frozen_legacy_serializer(preset_name: object, config: dict) -> bool:
    if not isinstance(preset_name, str) or preset_name not in LEGACY_FROZEN_PRESETS:
        return False
    expected = {**PRESETS[preset_name], "noise_family": DEFAULT_NOISE_FAMILY}
    return config == expected

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
            "j": round(params.j, 12),
            "h": round(params.h, 12),
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
        cfg["master_seed"] = int(master_seed)
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

    artifact_environment_contract = environment_contract()
    master = int(cfg["master_seed"])
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
            item.update(_family_parameter_fields(params))
            validate_item(item)
            items.append(item)

    written_items = items
    if _uses_frozen_legacy_serializer(preset_name, cfg):
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
