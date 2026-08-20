"""Deterministic split-v2 generation and its three-level hash chain."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import qem_bench
from qem_bench.datasets.schema import (
    FAMILY_LABEL_METHODS,
    FEATURE_SPEC_VERSION,
    FEATURES,
    canonical_physical_circuit_identity,
    validate_item,
)
from qem_bench.datasets.splits import (
    CircuitPool,
    SplitSpec,
    resolve_split_spec,
)
from qem_bench.labels.statevector import ideal_expectation as statevector_expectation
from qem_bench.labels.stim_labels import ideal_expectation as stim_expectation
from qem_bench.noise.models import DEFAULT_NOISE_FAMILY, SEVERITY_GRIDS
from qem_bench.observables import z_expectation_from_counts, z_support_label
from qem_bench.reproducibility import environment_contract
from qem_bench.sampling import sample_counts
from qem_bench.validation import (
    SPLIT_SEED_FORMULA,
    SPLIT_SCHEMA_VERSION,
    canonical_hash,
    canonical_item_lines,
    canonical_json,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    split_spec_hash,
)


def _s0_t0_micro() -> SplitSpec:
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": [DEFAULT_NOISE_FAMILY],
            "noise_strength": ["L1"],
            "circuit_family": ["tfi"],
            "family_native_depth": [1],
            "observable_class": ["z_mid", "zz_mid"],
            "shots": [512],
        },
        n_qubits=[3],
        role_counts={"train": 5, "validation": 2, "test": 2},
        family_parameters={"tfi": {"dt": 0.2}},
    )


SPLIT_PRESETS: dict[str, SplitSpec] = {"s0-t0-micro": _s0_t0_micro()}


def _seed_int(master: int, spawn_key: tuple[int, ...]) -> int:
    sequence = np.random.SeedSequence(master, spawn_key=spawn_key)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _transpile_seed_from_circuit_id(circuit_id: str) -> int:
    prefix = "circuit-"
    digest = circuit_id.removeprefix(prefix)
    if (
        not circuit_id.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid canonical circuit ID {circuit_id!r}")
    return int(digest, 16)


def _write_sidecar(
    root: Path,
    category: str,
    value: Mapping[str, Any],
    sidecar_hashes: dict[str, str],
) -> tuple[str, str]:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    relative = (Path("sidecars") / category / f"{digest}.json").as_posix()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"content-address collision at {relative}")
    else:
        path.write_bytes(payload)
    sidecar_hashes[relative] = digest
    return digest, relative


def _pool_config(pool: CircuitPool) -> dict[str, Any]:
    return {
        "family": pool.family,
        **pool.to_dict()["parameter_grid"],
    }


def _counts(items: list[dict]) -> dict[str, Any]:
    roles = ("train", "validation", "test")
    strata = sorted({str(item["stratum"]) for item in items})
    cells = sorted({str(item["cell_id"]) for item in items})
    return {
        "items": len(items),
        "items_by_role": {
            role: sum(item["split"] == role for item in items) for role in roles
        },
        "items_by_cell": {
            cell: sum(item["cell_id"] == cell for item in items) for cell in cells
        },
        "items_by_stratum": {
            stratum: sum(item["stratum"] == stratum for item in items)
            for stratum in strata
        },
        "circuits": len({item["circuit_id"] for item in items}),
        "measurement_groups": len(
            {item["measurement_group"] for item in items}
        ),
    }


def _group_evals(items: list[dict], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in sorted({str(item[field]) for item in items}):
        groups: dict[str, int] = {}
        for item in items:
            if str(item[field]) != value:
                continue
            group = str(item["measurement_group"])
            shots = int(item["shots"])
            if group in groups and groups[group] != shots:
                raise ValueError(
                    f"measurement group {group} has conflicting shots values"
                )
            groups[group] = shots
        result[value] = sum(groups.values())
    return result


def _generation_ledger(items: list[dict], label_evals: Mapping[str, int]) -> dict:
    by_role = _group_evals(items, "split")
    return {
        "circuit_evals_by_role": by_role,
        "circuit_evals_by_cell": _group_evals(items, "cell_id"),
        "circuit_evals_by_stratum": _group_evals(items, "stratum"),
        "total_circuit_evals": sum(by_role.values()),
        "label_evals_by_method": dict(sorted(label_evals.items())),
        "note": (
            "circuit evaluations are counted once per physical measurement group; "
            "exact-label calls are logged separately"
        ),
    }


def _versions() -> dict[str, str]:
    import qiskit
    import qiskit_aer
    import sklearn
    import stim

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "qiskit-aer": qiskit_aer.__version__,
        "stim": stim.__version__,
        "qem-bench": qem_bench.__version__,
    }


def generate_split(
    split: str | SplitSpec | Mapping[str, Any],
    out_dir: str | Path,
    master_seed: int | None = None,
) -> dict:
    """Generate one immutable split-v2 artifact."""
    if isinstance(split, str):
        try:
            spec = SPLIT_PRESETS[split]
        except KeyError as exc:
            raise ValueError(f"unknown split preset {split!r}") from exc
        dataset_id = split
        default_master_seed = 7
    else:
        spec = split if isinstance(split, SplitSpec) else SplitSpec.from_dict(split)
        spec_digest = split_spec_hash(spec)
        dataset_id = f"{spec.split_id.lower()}-{spec_digest[:12]}"
        default_master_seed = 7
    if master_seed is not None and type(master_seed) is not int:
        raise ValueError("master_seed must be a nonnegative integer")
    master = default_master_seed if master_seed is None else master_seed
    if master < 0:
        raise ValueError("master_seed must be a nonnegative integer")

    artifact_environment_contract = environment_contract()
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"dataset artifact path already exists: {out}") from exc

    resolved = resolve_split_spec(spec)
    spec_digest = split_spec_hash(spec)
    sidecar_hashes: dict[str, str] = {}

    from qem_bench.datasets.generate import (
        LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE,
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD,
        _family_parameter_fields,
        _observable_support,
        _require_source_observable_closure,
        _sample_and_build_circuit,
        validate_physical_identity_encoding_profiles,
    )

    families = {pool.family for pool in resolved.circuit_pools}
    physical_identity_encoding_profiles = (
        validate_physical_identity_encoding_profiles(
            {
                family: LEGACY_BINARY64_IDENTITY_ENCODING_PROFILE
                for family in families
            },
            families=families,
        )
    )

    circuits: dict[str, list[dict[str, Any]]] = {}
    for pool in resolved.circuit_pools:
        cfg = _pool_config(pool)
        pool_circuits: list[dict[str, Any]] = []
        for local_instance in range(pool.n_instances):
            spawn_key = tuple(pool.pool_seed_key) + (0, local_instance)
            circuit_seed = _seed_int(master, spawn_key)
            rng = np.random.default_rng(
                np.random.SeedSequence(master, spawn_key=spawn_key)
            )
            params, circuit = _sample_and_build_circuit(
                cfg, rng, local_instance, circuit_seed
            )
            parameter_fields = _family_parameter_fields(params)
            descriptor, circuit_id = canonical_physical_circuit_identity(
                {
                    "family": pool.family,
                    "n_qubits": params.n_qubits,
                    "circuit_seed": circuit_seed,
                    **parameter_fields,
                }
            )
            circuit_hash, circuit_sidecar = _write_sidecar(
                out, "circuits", descriptor, sidecar_hashes
            )
            pool_circuits.append(
                {
                    "circuit": circuit,
                    "params": params,
                    "parameter_fields": parameter_fields,
                    "circuit_seed": circuit_seed,
                    "circuit_id": circuit_id,
                    "circuit_hash": circuit_hash,
                    "circuit_sidecar": circuit_sidecar,
                    "local_instance": local_instance,
                }
            )
        circuits[pool.circuit_pool_id] = pool_circuits

    items: list[dict] = []
    ideal_cache: dict[tuple[str, str], float] = {}
    label_evals = {"statevector": 0, "stim": 0}
    pool_by_id = {
        pool.circuit_pool_id: pool for pool in resolved.circuit_pools
    }

    for cell in resolved.cells:
        pool = pool_by_id[cell.circuit_pool_id]
        noise_family = str(cell.axis_values["noise_family"])
        if noise_family not in SEVERITY_GRIDS:
            raise ValueError(f"unknown noise family {noise_family!r}")
        if cell.severity not in SEVERITY_GRIDS[noise_family]:
            raise ValueError(
                f"unknown severity {cell.severity!r} for {noise_family!r}"
            )
        noise_descriptor = {
            "noise_config_id": cell.noise_config_id,
            "noise_family": noise_family,
            "severity": cell.severity,
            "parameters": SEVERITY_GRIDS[noise_family][cell.severity],
        }
        noise_hash, noise_sidecar = _write_sidecar(
            out, "noise", noise_descriptor, sidecar_hashes
        )

        for circuit_record in circuits[cell.circuit_pool_id]:
            local_instance = int(circuit_record["local_instance"])
            sampler_spawn_key = tuple(cell.cell_seed_key) + (
                1,
                local_instance,
                cell.replicate,
                0,
            )
            sampler_seed = _seed_int(master, sampler_spawn_key)
            counts, structure = sample_counts(
                circuit_record["circuit"],
                severity=cell.severity,
                shots=cell.shots,
                sampler_seed=sampler_seed,
                transpile_seed=_transpile_seed_from_circuit_id(
                    str(circuit_record["circuit_id"])
                ),
                noise_family=noise_family,
            )
            counts_descriptor = {
                "cell_id": cell.cell_id,
                "circuit_id": circuit_record["circuit_id"],
                "shots": cell.shots,
                "sampler_seed": sampler_seed,
                "counts": {str(key): int(value) for key, value in sorted(counts.items())},
            }
            counts_hash, counts_sidecar = _write_sidecar(
                out, "counts", counts_descriptor, sidecar_hashes
            )
            group_id = (
                f"{cell.cell_id}:{circuit_record['circuit_id']}:"
                f"i{local_instance}:r{cell.replicate}:g0"
            )

            for observable in cell.observable_ids:
                support = _observable_support(observable, circuit_record["params"])
                pauli = z_support_label(
                    circuit_record["params"].n_qubits, support
                )
                observable_descriptor = {
                    "observable_id": observable,
                    "n_qubits": circuit_record["params"].n_qubits,
                    "pauli_label": pauli,
                    "support": list(support),
                }
                observable_hash, observable_sidecar = _write_sidecar(
                    out, "observables", observable_descriptor, sidecar_hashes
                )
                cache_key = (str(circuit_record["circuit_id"]), observable)
                label_method = FAMILY_LABEL_METHODS[pool.family]
                if cache_key not in ideal_cache:
                    if label_method == "statevector":
                        ideal = statevector_expectation(
                            circuit_record["circuit"], pauli
                        )
                    else:
                        ideal = stim_expectation(circuit_record["circuit"], pauli)
                    ideal_cache[cache_key] = float(ideal)
                    label_evals[label_method] += 1
                noisy, noisy_stderr = z_expectation_from_counts(
                    counts, support, cell.shots
                )
                item_descriptor = {
                    "cell_id": cell.cell_id,
                    "circuit_id": circuit_record["circuit_id"],
                    "instance": local_instance,
                    "observable_id": observable,
                    "replicate": cell.replicate,
                }
                item = {
                    "item_id": f"item-{canonical_hash(item_descriptor)}",
                    "dataset_schema_version": SPLIT_SCHEMA_VERSION,
                    "split_id": spec.split_id,
                    "split_axis": spec.split_axis,
                    "partition_id": cell.partition_id,
                    "split": cell.split,
                    "domain": cell.domain,
                    "cell_id": cell.cell_id,
                    "circuit_pool_id": cell.circuit_pool_id,
                    "circuit_id": circuit_record["circuit_id"],
                    "circuit_hash": circuit_record["circuit_hash"],
                    "circuit_sidecar": circuit_record["circuit_sidecar"],
                    "family": pool.family,
                    "stratum": pool.stratum,
                    "instance": local_instance,
                    "n_qubits": circuit_record["params"].n_qubits,
                    "circuit_seed": circuit_record["circuit_seed"],
                    "observable": observable,
                    "observable_id": observable,
                    "observable_hash": observable_hash,
                    "observable_sidecar": observable_sidecar,
                    "pauli_label": pauli,
                    "obs_locality": len(support),
                    "noise_family": noise_family,
                    "noise_config_id": cell.noise_config_id,
                    "noise_config_hash": noise_hash,
                    "noise_sidecar": noise_sidecar,
                    "severity": cell.severity,
                    "shots": cell.shots,
                    "replicate": cell.replicate,
                    "measurement_group": group_id,
                    "measurement_plan_id": cell.measurement_plan_id,
                    "sampler_seed": sampler_seed,
                    "counts_hash": counts_hash,
                    "counts_sidecar": counts_sidecar,
                    "raw_circuit_evals": cell.shots,
                    "noisy_expectation": round(noisy, 12),
                    "noisy_stderr": round(noisy_stderr, 12),
                    "ideal_expectation": round(ideal_cache[cache_key], 12),
                    "label_method": label_method,
                    "feature_spec_id": FEATURE_SPEC_VERSION,
                    "axis_values": cell.to_dict()["axis_values"],
                    "two_qubit_gates": structure["two_qubit_gates"],
                    "transpiled_depth": structure["transpiled_depth"],
                }
                item.update(circuit_record["parameter_fields"])
                validate_item(item, schema_version=SPLIT_SCHEMA_VERSION)
                items.append(item)

    _require_source_observable_closure(
        items,
        split_name=f"{spec.split_id} ({dataset_id})",
    )
    items.sort(key=lambda item: item["item_id"])
    cell_hashes = cell_item_stream_hashes(items)
    pools_payload = [pool.to_dict() for pool in resolved.circuit_pools]
    cells_payload = []
    for cell in resolved.cells:
        payload = cell.to_dict()
        payload["item_stream_hash"] = cell_hashes[cell.cell_id]
        cells_payload.append(payload)

    registry_configs = json.loads(canonical_json(SEVERITY_GRIDS))
    manifest = {
        "dataset_schema_version": SPLIT_SCHEMA_VERSION,
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD: (
            physical_identity_encoding_profiles
        ),
        "environment_contract": artifact_environment_contract,
        "dataset_id": dataset_id,
        "preset": dataset_id if isinstance(split, str) else None,
        "split_spec": spec.to_dict(),
        "split_spec_hash": spec_digest,
        "reporting": {
            "headline_strata": ["continuous_regression"],
            "control_strata": ["clifford_control"],
        },
        "master_seed": master,
        "seed_scheme": {
            "version": "split-v1",
            "master_seed": master,
            "formula": SPLIT_SEED_FORMULA,
        },
        "noise_registry": {
            "version": "severity-grid-v1",
            "hash": canonical_hash(registry_configs),
            "configs": registry_configs,
        },
        "circuit_pools": pools_payload,
        "cells": cells_payload,
        "feature_spec": {
            "version": FEATURE_SPEC_VERSION,
            "features": list(FEATURES),
        },
        "counts": _counts(items),
        "generation_ledger": _generation_ledger(items, label_evals),
        "versions": _versions(),
        "sidecar_hashes": dict(sorted(sidecar_hashes.items())),
        "items_hash": item_stream_hash(items),
    }
    manifest["dataset_hash"] = split_dataset_hash(
        spec_hash=spec_digest,
        items_hash=manifest["items_hash"],
    )

    (out / "items.jsonl").write_text(
        "\n".join(canonical_item_lines(items)) + "\n", encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["SPLIT_PRESETS", "SPLIT_SEED_FORMULA", "generate_split"]
