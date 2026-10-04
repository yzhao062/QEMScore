#!/usr/bin/env python3
"""Shot-count sweep generator and pre-fit verification for QEMScore.

Implements the frozen rule: docs/frozen-rules/2026-10-03-shot-sweep.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import qiskit
import qiskit_aer
import scipy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import qemscore
from qemscore.datasets.schema import (
    build_circuit_from_canonical_descriptor,
)
from qemscore.datasets.split_generate import (
    _counts,
    _generation_ledger,
    _transpile_seed_from_circuit_id,
    _versions,
)
from qemscore.noise.models import (
    BASIS_GATES,
    SEVERITY_GRIDS,
    build_noise_model,
)
from qemscore.observables import z_expectation_from_counts
from qemscore.sampling import _MOD, sample_counts
from qemscore.validation import (
    canonical_item_lines,
    canonical_json,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    split_spec_hash,
    validate_split_artifact,
)

PRIMARY_KEYS = ("shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640")
EXACT_SHOT_PLACEHOLDER = 1048576  # 2^20
SOURCE_SHOTS = 2048  # the shot count of the source datasets
# Fields a level may change relative to its source (rule check 3); axis_values["shots"] too.
LEVEL_FIELDS = frozenset({"shots", "raw_circuit_evals", "noisy_expectation", "noisy_stderr",
                          "counts_hash", "counts_sidecar"})
CHECK_EXACT_SHOTS = 131072  # the rule's check 2 compares the exact level with this level


def _gate_only_noise_model(
    noise_family: str, severity: str, sampler_seed: int, n_qubits: int
) -> tuple[NoiseModel, float]:
    """Construct gate-only noise model and return (noise_model, p_ro)."""
    if noise_family != "depolarizing_readout":
        raise ValueError(f"exact mode only supports depolarizing_readout, got {noise_family}")
    nm_full = build_noise_model(
        noise_family, severity, seed=sampler_seed, n_qubits=n_qubits
    )
    cfg = SEVERITY_GRIDS[noise_family][severity]
    p_ro = float(cfg["p_ro"])
    nm_gate = NoiseModel(basis_gates=BASIS_GATES)
    for gate in ("sx", "x"):
        if gate in nm_full._default_quantum_errors:
            nm_gate.add_all_qubit_quantum_error(
                nm_full._default_quantum_errors[gate], [gate]
            )
    if "cx" in nm_full._default_quantum_errors:
        nm_gate.add_all_qubit_quantum_error(
            nm_full._default_quantum_errors["cx"], ["cx"]
        )
    return nm_gate, p_ro


def _process_measurement_group_chunk(
    chunk_tasks: list[dict[str, Any]],
    shots: int | None,
    exact: bool,
    out_dir_str: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Worker function to process a chunk of measurement groups."""
    out_dir = Path(out_dir_str)
    counts_dir = out_dir / "sidecars" / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)

    item_updates: dict[str, dict[str, Any]] = {}
    sidecar_hashes: dict[str, str] = {}

    for task in chunk_tasks:
        circuit_desc = task["circuit_desc"]
        circuit_id = task["circuit_id"]
        tseed = task["tseed"]
        severity = task["severity"]
        noise_family = task["noise_family"]
        sampler_seed = task["sampler_seed"]
        cell_id = task["cell_id"]
        items_info = task["items"]

        circuit = build_circuit_from_canonical_descriptor(circuit_desc)

        if exact:
            if noise_family != "depolarizing_readout":
                raise ValueError(f"exact mode refuses noise family {noise_family}")

            # Transpile circuit exactly as sample_counts does
            measured = circuit.copy()
            measured.measure_all()
            tcirc = transpile(
                measured,
                basis_gates=BASIS_GATES,
                optimization_level=1,
                seed_transpiler=int(tseed % _MOD),
            )

            exact_structure = {
                "two_qubit_gates": int(tcirc.count_ops().get("cx", 0)),
                "transpiled_depth": int(tcirc.depth()),
            }
            for it in items_info:
                if (
                    exact_structure["two_qubit_gates"] != it["stored_two_qubit_gates"]
                    or exact_structure["transpiled_depth"] != it["stored_transpiled_depth"]
                ):
                    raise AssertionError(
                        f"structure mismatch on item {it['item_id']}: "
                        f"transpiled {exact_structure} != stored "
                        f"({it['stored_two_qubit_gates']}, {it['stored_transpiled_depth']})"
                    )

            # Strip final measurements and the barrier added by measure_all
            c_clean = QuantumCircuit(tcirc.num_qubits)
            for instr in tcirc.data:
                if instr.operation.name not in ("measure", "barrier"):
                    c_clean.append(instr.operation, instr.qubits)
            c_clean.save_probabilities()

            nm_gate, p_ro = _gate_only_noise_model(
                noise_family, severity, sampler_seed, tcirc.num_qubits
            )
            sim_dm = AerSimulator(method="density_matrix", noise_model=nm_gate)
            res = sim_dm.run(c_clean).result()
            probs = res.data(0)["probabilities"]
            prob_sha256 = hashlib.sha256(
                np.ascontiguousarray(probs, dtype=np.float64).tobytes()
            ).hexdigest()

            counts_desc = {
                "cell_id": cell_id,
                "circuit_id": circuit_id,
                "shots": EXACT_SHOT_PLACEHOLDER,
                "sampler_seed": sampler_seed,
                "exact": True,
                "p_ro": p_ro,
                "support": [it["support"] for it in items_info],
                "prob_sha256": prob_sha256,
            }
            payload = (canonical_json(counts_desc) + "\n").encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            rel_path = f"sidecars/counts/{digest}.json"
            target_file = out_dir / rel_path
            if not target_file.exists():
                target_file.write_bytes(payload)
            sidecar_hashes[rel_path] = digest

            for it in items_info:
                support = it["support"]
                mask = sum(1 << q for q in support)
                signs = np.array(
                    [1 - 2 * (bin(k & mask).count("1") % 2) for k in range(len(probs))]
                )
                r_star = float(np.dot(probs, signs) * ((1 - 2 * p_ro) ** len(support)))

                item_updates[it["item_id"]] = {
                    "shots": EXACT_SHOT_PLACEHOLDER,
                    "raw_circuit_evals": EXACT_SHOT_PLACEHOLDER,
                    "noisy_expectation": round(r_star, 12),
                    "noisy_stderr": 0.0,
                    "counts_hash": digest,
                    "counts_sidecar": rel_path,
                }
        else:
            assert shots is not None
            counts, structure = sample_counts(
                circuit,
                severity,
                shots,
                sampler_seed,
                tseed,
                noise_family=noise_family,
            )

            # Assert structure matches stored item fields
            for it in items_info:
                if (
                    structure["two_qubit_gates"] != it["stored_two_qubit_gates"]
                    or structure["transpiled_depth"] != it["stored_transpiled_depth"]
                ):
                    raise AssertionError(
                        f"structure mismatch on item {it['item_id']}: "
                        f"returned {structure} != stored ({it['stored_two_qubit_gates']}, {it['stored_transpiled_depth']})"
                    )

            counts_desc = {
                "cell_id": cell_id,
                "circuit_id": circuit_id,
                "shots": shots,
                "sampler_seed": sampler_seed,
                "counts": {str(k): int(v) for k, v in sorted(counts.items())},
            }
            payload = (canonical_json(counts_desc) + "\n").encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            rel_path = f"sidecars/counts/{digest}.json"
            target_file = out_dir / rel_path
            if not target_file.exists():
                target_file.write_bytes(payload)
            sidecar_hashes[rel_path] = digest

            for it in items_info:
                support = it["support"]
                noisy, stderr = z_expectation_from_counts(counts, support, shots)
                item_updates[it["item_id"]] = {
                    "shots": shots,
                    "raw_circuit_evals": shots,
                    "noisy_expectation": round(noisy, 12),
                    "noisy_stderr": round(stderr, 12),
                    "counts_hash": digest,
                    "counts_sidecar": rel_path,
                }

    return item_updates, sidecar_hashes


def _worker_wrapper(args: tuple) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    return _process_measurement_group_chunk(*args)


def generate(
    source_dir: Path,
    out_dir: Path,
    *,
    shots: int | None = None,
    exact: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    """Regenerate a dataset with varied shots or exact infinite-shot limits."""
    import time
    started = time.perf_counter()

    if exact and shots is not None:
        raise ValueError("specify either --shots or --exact, not both")
    if not exact and (shots is None or shots < 1):
        raise ValueError("--shots must be a positive integer when not in --exact mode")

    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = source_dir / "manifest.json"
    items_path = source_dir / "items.jsonl"
    if not manifest_path.exists() or not items_path.exists():
        raise SystemExit(f"source {source_dir} is missing manifest.json or items.jsonl")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Preload observables
    obs_sidecars: dict[str, dict] = {}
    for it in source_items:
        rel = it["observable_sidecar"]
        if rel not in obs_sidecars:
            obs_sidecars[rel] = json.loads((source_dir / rel).read_text(encoding="utf-8"))

    # Copy static sidecars: circuits, noise, observables
    import shutil
    for cat in ("circuits", "noise", "observables"):
        src_cat = source_dir / "sidecars" / cat
        dst_cat = out_dir / "sidecars" / cat
        if src_cat.exists():
            dst_cat.parent.mkdir(parents=True, exist_ok=True)
            if not dst_cat.exists():
                shutil.copytree(src_cat, dst_cat)

    # Group items by measurement_group
    by_group: dict[str, list[dict]] = defaultdict(list)
    for it in source_items:
        by_group[it["measurement_group"]].append(it)

    group_tasks = []
    for group_id, g_items in by_group.items():
        # Assert every item in a group shares circuit, sampler seed, severity, and noise family
        cids = {it["circuit_id"] for it in g_items}
        sseeds = {it["sampler_seed"] for it in g_items}
        sevs = {it["severity"] for it in g_items}
        fams = {it["noise_family"] for it in g_items}
        if len(cids) != 1 or len(sseeds) != 1 or len(sevs) != 1 or len(fams) != 1:
            raise AssertionError(f"measurement group {group_id} has inconsistent properties")

        first = g_items[0]
        circuit_id = first["circuit_id"]
        cdesc = json.loads((source_dir / first["circuit_sidecar"]).read_text(encoding="utf-8"))
        tseed = _transpile_seed_from_circuit_id(circuit_id)

        items_info = [
            {
                "item_id": it["item_id"],
                "observable_id": it["observable_id"],
                "support": obs_sidecars[it["observable_sidecar"]]["support"],
                "stored_two_qubit_gates": it["two_qubit_gates"],
                "stored_transpiled_depth": it["transpiled_depth"],
            }
            for it in g_items
        ]

        group_tasks.append({
            "group_id": group_id,
            "cell_id": first["cell_id"],
            "circuit_id": circuit_id,
            "circuit_desc": cdesc,
            "tseed": tseed,
            "severity": first["severity"],
            "noise_family": first["noise_family"],
            "sampler_seed": first["sampler_seed"],
            "items": items_info,
        })

    # Chunk tasks across workers
    chunk_size = max(1, len(group_tasks) // (workers * 4))
    chunks = [
        group_tasks[i : i + chunk_size]
        for i in range(0, len(group_tasks), chunk_size)
    ]

    all_updates: dict[str, dict[str, Any]] = {}
    new_counts_sidecars: dict[str, str] = {}

    worker_args = [
        (chunk, shots, exact, str(out_dir))
        for chunk in chunks
    ]

    if workers <= 1:
        for arg in worker_args:
            upds, sids = _worker_wrapper(arg)
            all_updates.update(upds)
            new_counts_sidecars.update(sids)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for upds, sids in executor.map(_worker_wrapper, worker_args):
                all_updates.update(upds)
                new_counts_sidecars.update(sids)

    # Build updated items
    final_shots = EXACT_SHOT_PLACEHOLDER if exact else shots
    updated_items = []
    for it in source_items:
        item_id = it["item_id"]
        new_it = dict(it)
        if item_id in all_updates:
            new_it.update(all_updates[item_id])
        new_axis = dict(new_it["axis_values"])
        new_axis["shots"] = final_shots
        new_it["axis_values"] = new_axis
        updated_items.append(new_it)

    # Write items.jsonl
    items_content = "\n".join(canonical_item_lines(updated_items)) + "\n"
    (out_dir / "items.jsonl").write_text(items_content, encoding="utf-8")

    # Update manifest
    updated_spec = dict(source_manifest["split_spec"])
    updated_fixed_axes = dict(updated_spec["fixed_axes"])
    updated_fixed_axes["shots"] = [final_shots]
    updated_spec["fixed_axes"] = updated_fixed_axes
    new_spec_hash = split_spec_hash(updated_spec)

    cell_hashes = cell_item_stream_hashes(updated_items)
    updated_cells = []
    for cell in source_manifest["cells"]:
        c = dict(cell)
        c["shots"] = final_shots
        c_axis = dict(c["axis_values"])
        c_axis["shots"] = final_shots
        c["axis_values"] = c_axis
        c["item_stream_hash"] = cell_hashes[c["cell_id"]]
        updated_cells.append(c)

    combined_sidecars = dict(source_manifest["sidecar_hashes"])
    combined_sidecars = {k: v for k, v in combined_sidecars.items() if not k.startswith("sidecars/counts/")}
    combined_sidecars.update(new_counts_sidecars)

    new_items_hash = item_stream_hash(updated_items)
    new_dataset_hash = split_dataset_hash(
        spec_hash=new_spec_hash, items_hash=new_items_hash
    )

    label_evals = {k: int(v) for k, v in source_manifest["generation_ledger"]["label_evals_by_method"].items()}
    new_manifest = {
        "dataset_schema_version": source_manifest["dataset_schema_version"],
        "physical_identity_encoding_profiles": source_manifest["physical_identity_encoding_profiles"],
        "environment_contract": source_manifest["environment_contract"],
        "dataset_id": source_manifest["dataset_id"],
        "preset": source_manifest.get("preset"),
        "split_spec": updated_spec,
        "split_spec_hash": new_spec_hash,
        "reporting": source_manifest["reporting"],
        "master_seed": source_manifest["master_seed"],
        "seed_scheme": source_manifest["seed_scheme"],
        "noise_registry": source_manifest["noise_registry"],
        "circuit_pools": source_manifest["circuit_pools"],
        "cells": updated_cells,
        "feature_spec": source_manifest["feature_spec"],
        "counts": _counts(updated_items),
        "generation_ledger": _generation_ledger(updated_items, label_evals),
        "versions": _versions(),
        "sidecar_hashes": dict(sorted(combined_sidecars.items())),
        "items_hash": new_items_hash,
        "dataset_hash": new_dataset_hash,
    }
    if exact:
        new_manifest["shot_level"] = "exact"

    (out_dir / "manifest.json").write_text(
        json.dumps(new_manifest, indent=2) + "\n", encoding="utf-8"
    )

    # The 2,048-shot regeneration is an ordinary split artifact. Every other level keeps
    # the source's cell identifiers and sampler seeds (needed for pairing), whose cell
    # keys name 2,048 shots, so it is validated as a derived sweep artifact instead.
    if not exact and shots == SOURCE_SHOTS:
        validate_split_artifact(out_dir)
    else:
        validate_sweep_artifact(source_dir, out_dir, shots=None if exact else shots, exact=exact)

    elapsed = time.perf_counter() - started

    # Write sweep_level.json
    gen_file = Path(__file__).resolve()
    sweep_level_data = {
        "shot_level": "exact" if exact else shots,
        "source_dir": str(source_dir),
        "source_dataset_hash": source_manifest["dataset_hash"],
        "source_items_jsonl_sha256": hashlib.sha256((source_dir / "items.jsonl").read_bytes()).hexdigest(),
        "generator": "tools/shot_sweep.py",
        "generator_sha256": hashlib.sha256(gen_file.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit.__version__,
            "qiskit-aer": qiskit_aer.__version__,
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "wall_time_seconds": elapsed,
    }
    (out_dir / "sweep_level.json").write_text(
        json.dumps(sweep_level_data, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "out_dir": str(out_dir),
        "shot_level": "exact" if exact else shots,
        "n_items": len(updated_items),
        "dataset_hash": new_dataset_hash,
        "items_hash": new_items_hash,
        "wall_time_seconds": elapsed,
    }


def _read_items(directory: Path) -> list[dict]:
    return [json.loads(line) for line in
            (directory / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_sweep_artifact(
    source_dir: Path, out_dir: Path, *, shots: int | None = None, exact: bool = False
) -> dict[str, Any]:
    """Validate a derived sweep level against its source; raise ValueError on any mismatch.

    The source must pass the package's ``validate_split_artifact``. The level must hold the
    same items in the same order, differ only in the fields of ``LEVEL_FIELDS`` and the shot
    entry of ``axis_values``, and carry the requested shot count. Every counts sidecar must
    hash to its recorded digest and agree with its item; at a finite level the counts must
    total the shot count and reproduce each item's estimate and standard error. The
    manifest's hashes, cells, counts, ledger, and sidecar table must match the items.
    """
    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if exact == (shots is not None):
        raise ValueError("give either shots or exact=True")
    level_shots = EXACT_SHOT_PLACEHOLDER if exact else int(shots)
    validate_split_artifact(source_dir)
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    source_items = _read_items(source_dir)
    items = _read_items(out_dir)

    if [it["item_id"] for it in items] != [it["item_id"] for it in source_items]:
        raise ValueError("level items differ from the source's items or their order")
    supports: dict[str, list[int]] = {}
    for src, out in zip(source_items, items):
        item_id = out["item_id"]
        for key in set(src) | set(out):
            if key in LEVEL_FIELDS:
                continue
            if key == "axis_values":
                a = {k: v for k, v in src[key].items() if k != "shots"}
                b = {k: v for k, v in out[key].items() if k != "shots"}
                if a != b:
                    raise ValueError(f"item {item_id}: axis_values differ beyond shots")
            elif src.get(key) != out.get(key):
                raise ValueError(f"item {item_id}: field {key} differs from the source")
        if out["shots"] != level_shots or out["raw_circuit_evals"] != level_shots \
                or out["axis_values"].get("shots") != level_shots:
            raise ValueError(f"item {item_id}: shot fields do not name level {level_shots}")
        rel = out["counts_sidecar"]
        if rel != f"sidecars/counts/{out['counts_hash']}.json":
            raise ValueError(f"item {item_id}: counts sidecar path does not match its hash")
        payload = (out_dir / rel).read_bytes()
        if hashlib.sha256(payload).hexdigest() != out["counts_hash"]:
            raise ValueError(f"item {item_id}: counts sidecar does not hash to counts_hash")
        desc = json.loads(payload)
        for key in ("cell_id", "circuit_id", "sampler_seed"):
            if desc.get(key) != out[key]:
                raise ValueError(f"item {item_id}: counts sidecar disagrees on {key}")
        if desc.get("shots") != level_shots:
            raise ValueError(f"item {item_id}: counts sidecar shots {desc.get('shots')}")
        obs_rel = out["observable_sidecar"]
        if obs_rel not in supports:
            supports[obs_rel] = json.loads(
                (out_dir / obs_rel).read_text(encoding="utf-8"))["support"]
        support = supports[obs_rel]
        if exact:
            if desc.get("exact") is not True or out["noisy_stderr"] != 0.0:
                raise ValueError(f"item {item_id}: exact item without an exact sidecar")
            if list(support) not in [list(s) for s in desc.get("support", [])]:
                raise ValueError(f"item {item_id}: support missing from its exact sidecar")
        else:
            counts = desc.get("counts")
            if not isinstance(counts, dict) or sum(int(v) for v in counts.values()) != level_shots:
                raise ValueError(f"item {item_id}: counts do not total {level_shots}")
            noisy, stderr = z_expectation_from_counts(counts, support, level_shots)
            if round(noisy, 12) != out["noisy_expectation"] or \
                    round(stderr, 12) != out["noisy_stderr"]:
                raise ValueError(f"item {item_id}: estimate does not follow from its counts")

    if manifest["items_hash"] != item_stream_hash(items):
        raise ValueError("manifest items_hash does not match the items")
    if manifest["split_spec_hash"] != split_spec_hash(manifest["split_spec"]):
        raise ValueError("manifest split_spec_hash does not match its split_spec")
    if manifest["split_spec"]["fixed_axes"].get("shots") != [level_shots]:
        raise ValueError("manifest split_spec does not name the level's shot count")
    if manifest["dataset_hash"] != split_dataset_hash(
            spec_hash=manifest["split_spec_hash"], items_hash=manifest["items_hash"]):
        raise ValueError("manifest dataset_hash does not match its hashes")
    for key in ("dataset_schema_version", "dataset_id", "master_seed", "seed_scheme",
                "noise_registry", "circuit_pools", "feature_spec"):
        if manifest.get(key) != source_manifest.get(key):
            raise ValueError(f"manifest {key} differs from the source")
    cell_hashes = cell_item_stream_hashes(items)
    if [c["cell_id"] for c in manifest["cells"]] != [c["cell_id"] for c in source_manifest["cells"]]:
        raise ValueError("manifest cells differ from the source's cells")
    for cell in manifest["cells"]:
        if cell["shots"] != level_shots or cell["item_stream_hash"] != cell_hashes[cell["cell_id"]]:
            raise ValueError(f"manifest cell {cell['cell_id']} does not match the items")
    if manifest["counts"] != _counts(items):
        raise ValueError("manifest counts do not match the items")
    label_evals = {k: int(v) for k, v in
                   source_manifest["generation_ledger"]["label_evals_by_method"].items()}
    if manifest["generation_ledger"] != _generation_ledger(items, label_evals):
        raise ValueError("manifest generation ledger does not match the items")
    table = manifest["sidecar_hashes"]
    counts_entries = {k for k in table if k.startswith("sidecars/counts/")}
    if counts_entries != {it["counts_sidecar"] for it in items}:
        raise ValueError("manifest counts sidecars differ from those the items name")
    for rel in counts_entries:
        if table[rel] != hashlib.sha256((out_dir / rel).read_bytes()).hexdigest():
            raise ValueError(f"manifest hash of {rel} does not match the file")
    for rel, digest in source_manifest["sidecar_hashes"].items():
        if rel.startswith("sidecars/counts/"):
            continue
        if table.get(rel) != digest or (out_dir / rel).read_bytes() != (source_dir / rel).read_bytes():
            raise ValueError(f"sidecar {rel} differs from the source")
    if set(table) - counts_entries != {k for k in source_manifest["sidecar_hashes"]
                                       if not k.startswith("sidecars/counts/")}:
        raise ValueError("manifest sidecar table differs from the source beyond counts")
    return {"items": len(items), "shot_level": "exact" if exact else level_shots, "passed": True}


def check_faithful(
    source_dir: Path, regen_dir: Path, out_path: Path | None = None
) -> dict[str, Any]:
    """Check faithful reproduction at 2048 shots."""
    source_dir = Path(source_dir).resolve()
    regen_dir = Path(regen_dir).resolve()

    items_source = (source_dir / "items.jsonl").read_bytes()
    items_regen = (regen_dir / "items.jsonl").read_bytes()
    items_byte_equal = items_source == items_regen

    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    regen_manifest = json.loads((regen_dir / "manifest.json").read_text(encoding="utf-8"))

    manifest_diffs: dict[str, Any] = {}
    all_keys = sorted(set(source_manifest) | set(regen_manifest))
    for k in all_keys:
        if source_manifest.get(k) != regen_manifest.get(k):
            manifest_diffs[k] = {
                "source": source_manifest.get(k),
                "regen": regen_manifest.get(k),
            }

    # Counts sidecars check
    source_rows = [json.loads(line) for line in items_source.decode("utf-8").splitlines() if line.strip()]
    regen_rows = [json.loads(line) for line in items_regen.decode("utf-8").splitlines() if line.strip()]

    source_counts_sidecars = sorted({r["counts_sidecar"] for r in source_rows})
    differing_sidecars = []
    for sc in source_counts_sidecars:
        src_path = source_dir / sc
        reg_path = regen_dir / sc
        if not reg_path.exists() or src_path.read_bytes() != reg_path.read_bytes():
            differing_sidecars.append(sc)

    # Diff in r and differing measurement groups
    differing_groups = set()
    max_r_diff = 0.0
    regen_by_id = {r["item_id"]: r for r in regen_rows}
    for s_item in source_rows:
        r_item = regen_by_id.get(s_item["item_id"])
        if r_item is None:
            differing_groups.add(s_item["measurement_group"])
            continue
        # Counts can differ while both parities, and so the estimates, agree.
        if (r_item.get("counts_hash") != s_item.get("counts_hash")
                or s_item["counts_sidecar"] in differing_sidecars):
            differing_groups.add(s_item["measurement_group"])
        diff = abs(float(s_item["noisy_expectation"]) - float(r_item["noisy_expectation"]))
        if diff > max_r_diff:
            max_r_diff = diff
        if diff > 1e-12:
            differing_groups.add(s_item["measurement_group"])

    passed = (
        items_byte_equal
        and not differing_sidecars
        and len(differing_groups) == 0
        and max_r_diff == 0.0
    )

    report = {
        "command": "check-faithful",
        "source_dir": str(source_dir),
        "regen_dir": str(regen_dir),
        "items_byte_equal": items_byte_equal,
        "manifest_equal": len(manifest_diffs) == 0,
        "manifest_diffs": manifest_diffs,
        "counts_sidecars_equal": len(differing_sidecars) == 0,
        "differing_counts_sidecars_count": len(differing_sidecars),
        "differing_measurement_groups_count": len(differing_groups),
        "max_r_diff": max_r_diff,
        "passed": passed,
    }

    if out_path is not None:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def check_exact(
    exact_dir: Path, finite_dir: Path, out_path: Path | None = None
) -> dict[str, Any]:
    """Check that finite shot estimates lie within 6 sigma of exact estimates."""
    exact_dir = Path(exact_dir).resolve()
    finite_dir = Path(finite_dir).resolve()

    level_path = finite_dir / "sweep_level.json"
    if not level_path.exists():
        raise SystemExit(f"{finite_dir} missing sweep_level.json")
    finite_level_data = json.loads(level_path.read_text(encoding="utf-8"))
    exact_level_path = exact_dir / "sweep_level.json"
    if not exact_level_path.exists():
        raise SystemExit(f"{exact_dir} missing sweep_level.json")
    exact_level_data = json.loads(exact_level_path.read_text(encoding="utf-8"))
    if exact_level_data.get("shot_level") != "exact":
        raise SystemExit(f"{exact_dir} is not the exact level: "
                         f"shot_level={exact_level_data.get('shot_level')!r}")
    if finite_level_data.get("shot_level") != CHECK_EXACT_SHOTS:
        raise SystemExit(f"check 2 compares the exact level with the {CHECK_EXACT_SHOTS}-shot "
                         f"level; {finite_dir} has shot_level={finite_level_data.get('shot_level')!r}")
    n_shots = CHECK_EXACT_SHOTS

    exact_items = [
        json.loads(line)
        for line in (exact_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finite_items = [
        json.loads(line)
        for line in (finite_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finite_by_id = {it["item_id"]: it for it in finite_items}
    exact_ids = {it["item_id"] for it in exact_items}
    if exact_ids != set(finite_by_id) or len(exact_ids) != len(exact_items):
        raise SystemExit("check 2 needs the same items at both levels: "
                         f"{len(exact_ids ^ set(finite_by_id))} item identifiers differ")
    bad_shots = [it["item_id"] for it in finite_items if it.get("shots") != CHECK_EXACT_SHOTS]
    bad_shots += [it["item_id"] for it in exact_items
                  if it.get("shots") != EXACT_SHOT_PLACEHOLDER or it.get("noisy_stderr") != 0.0]
    if bad_shots:
        raise SystemExit(f"check 2: {len(bad_shots)} items carry the wrong shot fields, "
                         f"for example {bad_shots[:3]}")

    items_checked = 0
    max_z = 0.0
    failed_items = []
    worst_items = []

    for ex in exact_items:
        item_id = ex["item_id"]
        if item_id not in finite_by_id:
            raise ValueError(f"item {item_id} in exact dir not found in finite dir")
        fin = finite_by_id[item_id]
        r_star = float(ex["noisy_expectation"])
        r_fin = float(fin["noisy_expectation"])

        stderr = math.sqrt(max(0.0, 1.0 - r_star**2) / n_shots)
        diff = abs(r_fin - r_star)
        z = diff / stderr if stderr > 1e-15 else 0.0

        if z > max_z:
            max_z = z

        record = {
            "item_id": item_id,
            "circuit_id": ex.get("circuit_id", ""),
            "r_star": r_star,
            "r_finite": r_fin,
            "diff": diff,
            "stderr": stderr,
            "z": z,
        }
        if z > 6.0:
            failed_items.append(record)

        items_checked += 1
        worst_items.append(record)

    worst_items.sort(key=lambda x: x["z"], reverse=True)
    top_worst = worst_items[:5]

    passed = len(failed_items) == 0

    report = {
        "command": "check-exact",
        "exact_dir": str(exact_dir),
        "finite_dir": str(finite_dir),
        "finite_shots": n_shots,
        "items_checked": items_checked,
        "failed_count": len(failed_items),
        "max_z": max_z,
        "worst_items": top_worst,
        "passed": passed,
    }

    if out_path is not None:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def check_caches(
    reference_dir: Path, level_dir: Path, out_path: Path | None = None
) -> dict[str, Any]:
    """Check that level ladder output caches match the 2048 reference cache."""
    reference_dir = Path(reference_dir).resolve()
    level_dir = Path(level_dir).resolve()

    ALLOWED_ITEM_DIFF_FIELDS = {
        "shots",
        "raw_circuit_evals",
        "noisy_expectation",
        "noisy_stderr",
        "counts_hash",
        "counts_sidecar",
    }

    order_matches: dict[str, bool] = {}
    pickle_top_diffs: dict[str, list[str]] = {}
    row_discrepancies: dict[str, list[dict]] = {}

    # Every primary dataset is compared; a missing file at either level fails below.
    keys_to_check = list(PRIMARY_KEYS)

    for key in keys_to_check:
        ref_order_path = reference_dir / "cache" / f"{key}.order.json"
        lvl_order_path = level_dir / "cache" / f"{key}.order.json"
        if not ref_order_path.exists() or not lvl_order_path.exists():
            raise SystemExit(f"missing {key}.order.json in {ref_order_path} or {lvl_order_path}")

        ref_order = json.loads(ref_order_path.read_text(encoding="utf-8"))
        lvl_order = json.loads(lvl_order_path.read_text(encoding="utf-8"))
        order_matches[key] = ref_order == lvl_order

        ref_pkl_path = reference_dir / "cache" / f"{key}.pkl"
        lvl_pkl_path = level_dir / "cache" / f"{key}.pkl"
        if not ref_pkl_path.exists() or not lvl_pkl_path.exists():
            raise SystemExit(f"missing {key}.pkl in {ref_pkl_path} or {lvl_pkl_path}")

        ref_data = pickle.loads(ref_pkl_path.read_bytes())
        lvl_data = pickle.loads(lvl_pkl_path.read_bytes())

        # Top-level pickle keys
        diff_keys = []
        for k in sorted(set(ref_data) | set(lvl_data)):
            if k in ("train", "validation", "test", "prediction_rows"):
                continue
            if k == "dataset_hash":
                # Differs by construction; reported, not a failure.
                continue
            if ref_data.get(k) != lvl_data.get(k):
                diff_keys.append(k)
        pickle_top_diffs[key] = diff_keys

        # Compare rows
        discrepancies = []
        dataset_hash_differs = ref_data.get("dataset_hash") != lvl_data.get("dataset_hash")
        pickle_top_diffs[key] = pickle_top_diffs[key] + (
            ["dataset_hash (allowed)"] if dataset_hash_differs else [])
        role_rows = [(role, ref_data.get(role, []), lvl_data.get(role, []))
                     for role in ("train", "validation", "test")]
        ref_pred = ref_data.get("prediction_rows", {}) or {}
        lvl_pred = lvl_data.get("prediction_rows", {}) or {}
        if sorted(ref_pred) != sorted(lvl_pred):
            discrepancies.append({"role": "prediction_rows",
                                  "error": f"roles {sorted(ref_pred)} vs {sorted(lvl_pred)}"})
        for role in sorted(set(ref_pred) & set(lvl_pred)):
            role_rows.append((f"prediction_rows/{role}", ref_pred[role], lvl_pred[role]))
        for role, ref_rows, lvl_rows in role_rows:
            if len(ref_rows) != len(lvl_rows):
                discrepancies.append({
                    "role": role,
                    "error": f"length mismatch: ref={len(ref_rows)} vs lvl={len(lvl_rows)}",
                })
                continue

            for idx, (r_item, l_item) in enumerate(zip(ref_rows, lvl_rows)):
                all_fields = sorted(set(r_item) | set(l_item))
                for fld in all_fields:
                    if fld in ALLOWED_ITEM_DIFF_FIELDS:
                        continue
                    if fld == "axis_values":
                        r_axis = {ak: av for ak, av in r_item.get("axis_values", {}).items() if ak != "shots"}
                        l_axis = {ak: av for ak, av in l_item.get("axis_values", {}).items() if ak != "shots"}
                        if r_axis != l_axis:
                            discrepancies.append({
                                "role": role,
                                "index": idx,
                                "item_id": r_item.get("item_id"),
                                "field": "axis_values (non-shot)",
                                "ref": r_axis,
                                "lvl": l_axis,
                            })
                    else:
                        if r_item.get(fld) != l_item.get(fld):
                            discrepancies.append({
                                "role": role,
                                "index": idx,
                                "item_id": r_item.get("item_id"),
                                "field": fld,
                                "ref": r_item.get(fld),
                                "lvl": l_item.get(fld),
                            })
        row_discrepancies[key] = discrepancies

    # Check descriptors/ byte identity
    ref_desc_dir = reference_dir / "descriptors"
    lvl_desc_dir = level_dir / "descriptors"
    desc_diffs = []
    if ref_desc_dir.exists() and lvl_desc_dir.exists():
        ref_files = sorted(f.name for f in ref_desc_dir.iterdir() if f.is_file())
        lvl_files = sorted(f.name for f in lvl_desc_dir.iterdir() if f.is_file())
        if ref_files != lvl_files:
            desc_diffs.append(f"file list mismatch: ref={ref_files} vs lvl={lvl_files}")
        for fn in ref_files:
            if (ref_desc_dir / fn).read_bytes() != (lvl_desc_dir / fn).read_bytes():
                desc_diffs.append(f"content mismatch in {fn}")
    else:
        desc_diffs.append("descriptors directory missing")

    # Check encoder-cache/ byte identity
    ref_enc_dir = reference_dir / "encoder-cache"
    lvl_enc_dir = level_dir / "encoder-cache"
    enc_diffs = []
    if ref_enc_dir.exists() and lvl_enc_dir.exists():
        ref_files = sorted(f.name for f in ref_enc_dir.iterdir() if f.is_file())
        lvl_files = sorted(f.name for f in lvl_enc_dir.iterdir() if f.is_file())
        if ref_files != lvl_files:
            enc_diffs.append(f"file list mismatch: ref={ref_files} vs lvl={lvl_files}")
        for fn in ref_files:
            if not (lvl_enc_dir / fn).exists():
                continue
            if fn.endswith(".npz"):
                with np.load(ref_enc_dir / fn, allow_pickle=False) as ra, \
                        np.load(lvl_enc_dir / fn, allow_pickle=False) as la:
                    if sorted(ra.files) != sorted(la.files) or any(
                            ra[name].dtype != la[name].dtype
                            or ra[name].shape != la[name].shape
                            or ra[name].tobytes() != la[name].tobytes()
                            for name in ra.files):
                        enc_diffs.append(f"array mismatch in {fn}")
            elif fn.endswith(".json"):
                volatile = {"data_dir", "seconds"}
                rj = json.loads((ref_enc_dir / fn).read_text(encoding="utf-8"))
                lj = json.loads((lvl_enc_dir / fn).read_text(encoding="utf-8"))
                if ({k: v for k, v in rj.items() if k not in volatile}
                        != {k: v for k, v in lj.items() if k not in volatile}):
                    enc_diffs.append(f"metadata mismatch in {fn} (data_dir and seconds ignored)")
            elif (ref_enc_dir / fn).read_bytes() != (lvl_enc_dir / fn).read_bytes():
                enc_diffs.append(f"content mismatch in {fn}")
    elif ref_enc_dir.exists() or lvl_enc_dir.exists():
        enc_diffs.append("encoder-cache directory present at only one level")
    # Absent at both levels: sweep preparation skips the R4 encoding.

    passed = (
        all(order_matches.values())
        and all(all(k == "dataset_hash (allowed)" for k in v) for v in pickle_top_diffs.values())
        and all(len(d) == 0 for d in row_discrepancies.values())
        and len(desc_diffs) == 0
        and len(enc_diffs) == 0
    )

    report = {
        "command": "check-caches",
        "reference_dir": str(reference_dir),
        "level_dir": str(level_dir),
        "order_matches": order_matches,
        "pickle_top_diffs": pickle_top_diffs,
        "row_discrepancies_count": {k: len(v) for k, v in row_discrepancies.items()},
        "descriptors_byte_identical": len(desc_diffs) == 0,
        "descriptors_diffs": desc_diffs,
        "encoder_cache_identical": len(enc_diffs) == 0,
        "encoder_cache_diffs": enc_diffs,
        "passed": passed,
    }

    if out_path is not None:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def validate_exact_toy_circuit(
    large_shots: int = 1_000_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate exact density matrix simulation on an asymmetric 3-qubit circuit."""
    qc = QuantumCircuit(3)
    qc.rx(0.2, 0)
    qc.rx(0.8, 1)
    qc.rx(1.5, 2)
    tseed = 12345
    m_qc = qc.copy()
    m_qc.measure_all()
    tcirc = transpile(
        m_qc, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=tseed % _MOD
    )

    nm_full = build_noise_model("depolarizing_readout", "L1", seed=seed, n_qubits=3)
    cfg = SEVERITY_GRIDS["depolarizing_readout"]["L1"]
    p_ro = float(cfg["p_ro"])

    nm_gate = NoiseModel(basis_gates=BASIS_GATES)
    for gate in ("sx", "x"):
        if gate in nm_full._default_quantum_errors:
            nm_gate.add_all_qubit_quantum_error(
                nm_full._default_quantum_errors[gate], [gate]
            )
    if "cx" in nm_full._default_quantum_errors:
        nm_gate.add_all_qubit_quantum_error(
            nm_full._default_quantum_errors["cx"], ["cx"]
        )

    c_clean = QuantumCircuit(tcirc.num_qubits)
    for instr in tcirc.data:
        if instr.operation.name not in ("measure", "barrier"):
            c_clean.append(instr.operation, instr.qubits)
    c_clean.save_probabilities()

    sim_dm = AerSimulator(method="density_matrix", noise_model=nm_gate)
    res_dm = sim_dm.run(c_clean).result()
    probs = res_dm.data(0)["probabilities"]

    sim_shots = AerSimulator(noise_model=nm_full, seed_simulator=seed % _MOD)
    counts = sim_shots.run(tcirc, shots=large_shots).result().get_counts()

    checks = []
    observables = [("Z0", [0]), ("Z1", [1]), ("Z2", [2]), ("Z0Z1", [0, 1])]
    for name, support in observables:
        mask = sum(1 << q for q in support)
        signs = np.array(
            [1 - 2 * (bin(k & mask).count("1") % 2) for k in range(len(probs))]
        )
        exact_val = float(np.dot(probs, signs) * ((1 - 2 * p_ro) ** len(support)))

        shot_val, shot_stderr = z_expectation_from_counts(counts, support, large_shots)
        diff = abs(exact_val - shot_val)
        sigma = diff / shot_stderr if shot_stderr > 0 else 0.0

        checks.append({
            "observable": name,
            "support": support,
            "exact_value": exact_val,
            "shot_value": shot_val,
            "shot_stderr": shot_stderr,
            "diff": diff,
            "sigma": sigma,
            "agrees_within_3sigma": bool(sigma < 3.0),
        })

    return {
        "large_shots": large_shots,
        "all_agree": all(c["agrees_within_3sigma"] for c in checks),
        "checks": checks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="regenerate dataset with varied shots or exact")
    gen.add_argument("--source", type=Path, required=True, help="source dataset directory")
    gen.add_argument("--shots", type=int, default=None, help="finite shot count")
    gen.add_argument("--exact", action="store_true", help="exact infinite-shot noisy expectation")
    gen.add_argument("--out", type=Path, required=True, help="output directory")
    gen.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="number of parallel worker processes",
    )

    cf = sub.add_parser("check-faithful", help="check faithful reproduction at 2048 shots")
    cf.add_argument("--source", type=Path, required=True)
    cf.add_argument("--regen", type=Path, required=True)
    cf.add_argument("--report", "--out", dest="out", type=Path, default=None)

    ce = sub.add_parser("check-exact", help="check exact vs finite shot level")
    ce.add_argument("--exact", type=Path, required=True)
    ce.add_argument("--finite", type=Path, required=True)
    ce.add_argument("--report", "--out", dest="out", type=Path, default=None)

    cc = sub.add_parser("check-caches", help="check level caches against 2048 reference")
    cc.add_argument("--reference", type=Path, required=True)
    cc.add_argument("--level", type=Path, required=True)
    cc.add_argument("--report", "--out", dest="out", type=Path, default=None)

    val = sub.add_parser("validate-exact", help="validate exact simulation against 1M shots on a toy circuit")
    val.add_argument("--shots", type=int, default=1_000_000)

    args = parser.parse_args(argv)

    if args.command == "generate":
        info = generate(
            args.source,
            args.out,
            shots=args.shots,
            exact=args.exact,
            workers=args.workers,
        )
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "check-faithful":
        rep = check_faithful(args.source, args.regen, args.out)
        print(json.dumps(rep, indent=2))
        return 0 if rep["passed"] else 1

    if args.command == "check-exact":
        rep = check_exact(args.exact, args.finite, args.out)
        print(json.dumps(rep, indent=2))
        return 0 if rep["passed"] else 1

    if args.command == "check-caches":
        rep = check_caches(args.reference, args.level, args.out)
        print(json.dumps(rep, indent=2))
        return 0 if rep["passed"] else 1

    if args.command == "validate-exact":
        res = validate_exact_toy_circuit(large_shots=args.shots)
        print(json.dumps(res, indent=2))
        return 0 if res["all_agree"] else 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
