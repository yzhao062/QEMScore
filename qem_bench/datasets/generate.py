"""Deterministic dataset generation with named seed streams and a canonical hash.

Seed formula (recorded in every manifest): the circuit stream for instance i is
SeedSequence(master_seed, spawn_key=(0, i)); the sampler stream for instance i's
measurement group is SeedSequence(master_seed, spawn_key=(1, i)). Every derived
integer seed is stored on its items, so any item regenerates without replaying the
stream order. Items are written in canonical order (sorted by item_id) and the
dataset hash is the SHA-256 of the canonical JSON lines, so parallel or reordered
generation cannot change the hash.

Measurement-group contract: the slice's observables are all Z-type, so one
computational-basis measurement per (circuit, noise, shots) configuration estimates
every observable. Each group is executed once, both observable rows derive from the
same counts (sharing shot noise, recorded via measurement_group), and the ledger
charges the group's shots once.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

import qem_bench
from qem_bench.circuits.tfi import build_tfi_circuit, sample_tfi_params
from qem_bench.datasets.schema import FEATURE_SPEC_VERSION, FEATURES, validate_item
from qem_bench.labels.statevector import ideal_expectation
from qem_bench.noise.models import SEVERITY_GRID
from qem_bench.observables import z_expectation_from_counts, z_support_label
from qem_bench.sampling import sample_counts

SEED_FORMULA = (
    "circuit stream: SeedSequence(master, spawn_key=(0, instance)); "
    "sampler stream (one per measurement group): SeedSequence(master, spawn_key=(1, instance))"
)

PRESETS: dict[str, dict] = {
    # Paper-adjacent smoke tier: runs in minutes on a laptop, exercises every path.
    "t0-smoke": {
        "master_seed": 20260719,
        "family": "tfi",
        "n_qubits": [4, 5, 6],
        "steps": [1, 2, 3],
        "dt": 0.2,
        "n_train": 48,
        "n_test": 32,
        "shots": 2048,
        "severities": ["L1", "L2", "L3"],
        "observables": ["z_mid", "zz_mid"],
    },
    # Micro tier for unit tests and CI.
    "t0-micro": {
        "master_seed": 7,
        "family": "tfi",
        "n_qubits": [3],
        "steps": [1, 2],
        "dt": 0.2,
        "n_train": 10,
        "n_test": 6,
        "shots": 512,
        "severities": ["L1", "L2"],
        "observables": ["z_mid", "zz_mid"],
    },
}


def _observable_support(name: str, n_qubits: int) -> tuple[int, ...]:
    mid = n_qubits // 2
    if name == "z_mid":
        return (mid,)
    if name == "zz_mid":
        return (mid - 1, mid)
    raise ValueError(f"unknown observable {name}")


def _seed_int(master: int, spawn_key: tuple[int, ...]) -> int:
    ss = np.random.SeedSequence(master, spawn_key=spawn_key)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _canonical_lines(items: list[dict]) -> list[str]:
    ordered = sorted(items, key=lambda it: it["item_id"])
    return [json.dumps(it, sort_keys=True, separators=(",", ":")) for it in ordered]


def dataset_hash(items: list[dict]) -> str:
    payload = "\n".join(_canonical_lines(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_shots(items: list[dict], split: str | None = None) -> int:
    """Total circuit evaluations across unique measurement groups (optionally per split).

    Raises on a group whose rows disagree about shots: such a group cannot have come
    from one shared execution, and charging it once would falsify the ledger.
    """
    seen: dict[str, int] = {}
    for it in items:
        if split is not None and it["split"] != split:
            continue
        group = it["measurement_group"]
        if group in seen and seen[group] != it["shots"]:
            raise ValueError(f"measurement group {group} has conflicting shots values")
        seen[group] = it["shots"]
    return sum(seen.values())


def generate(preset: str | dict, out_dir: str | Path, master_seed: int | None = None) -> dict:
    """Generate a dataset directory (items.jsonl + manifest.json); return the manifest."""
    cfg = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
    preset_name = preset if isinstance(preset, str) else cfg.get("name", "custom")
    if master_seed is not None:
        cfg["master_seed"] = int(master_seed)
    master = int(cfg["master_seed"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_total = cfg["n_train"] + cfg["n_test"]
    items: list[dict] = []
    label_evals = 0

    for i in range(n_total):
        split = "train" if i < cfg["n_train"] else "test"
        circuit_seed = _seed_int(master, (0, i))
        rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(0, i)))
        params = sample_tfi_params(
            rng, cfg["n_qubits"], cfg["steps"], cfg["dt"], instance=i, circuit_seed=circuit_seed
        )
        circuit = build_tfi_circuit(params)
        severity = cfg["severities"][i % len(cfg["severities"])]

        # One measurement group per instance: a single noisy execution whose counts
        # serve every Z-type observable of this circuit.
        group_id = f"{cfg['family']}-{i:04d}-g0"
        sampler_seed = _seed_int(master, (1, i))
        counts, feats = sample_counts(
            circuit,
            severity=severity,
            shots=cfg["shots"],
            sampler_seed=sampler_seed,
            transpile_seed=circuit_seed,
        )

        for obs_name in cfg["observables"]:
            support = _observable_support(obs_name, params.n_qubits)
            pauli = z_support_label(params.n_qubits, support)
            y = ideal_expectation(circuit, pauli)
            label_evals += 1
            r, r_err = z_expectation_from_counts(counts, support, cfg["shots"])
            item = {
                "item_id": f"{cfg['family']}-{i:04d}-{obs_name}",
                "family": cfg["family"],
                "split": split,
                "instance": i,
                "n_qubits": params.n_qubits,
                "steps": params.steps,
                "j": round(params.j, 12),
                "h": round(params.h, 12),
                "dt": params.dt,
                "circuit_seed": circuit_seed,
                "observable": obs_name,
                "pauli_label": pauli,
                "obs_locality": len(support),
                "noise_family": "depolarizing_readout",
                "severity": severity,
                "shots": cfg["shots"],
                "measurement_group": group_id,
                "sampler_seed": sampler_seed,
                "noisy_expectation": round(r, 12),
                "noisy_stderr": round(r_err, 12),
                "ideal_expectation": round(y, 12),
                "label_method": "statevector",
                "two_qubit_gates": feats["two_qubit_gates"],
                "transpiled_depth": feats["transpiled_depth"],
            }
            validate_item(item)
            items.append(item)

    lines = _canonical_lines(items)
    (out / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import qiskit
    import qiskit_aer
    import sklearn

    manifest = {
        "preset": preset_name,
        "config": cfg,
        "master_seed": master,
        "seed_formula": SEED_FORMULA,
        "feature_spec": {"version": FEATURE_SPEC_VERSION, "features": FEATURES},
        "severity_grid": SEVERITY_GRID,
        "counts": {
            "items": len(items),
            "train_items": sum(1 for it in items if it["split"] == "train"),
            "test_items": sum(1 for it in items if it["split"] == "test"),
            "instances": n_total,
            "measurement_groups": len({it["measurement_group"] for it in items}),
        },
        "generation_ledger": {
            "train_circuit_evals": group_shots(items, "train"),
            "test_circuit_evals": group_shots(items, "test"),
            "label_evals_statevector": label_evals,
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
            "qem-bench": qem_bench.__version__,
        },
        "dataset_hash": dataset_hash(items),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
