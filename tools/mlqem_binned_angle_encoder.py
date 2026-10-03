"""ML-QEM binned-angle circuit feature encoder lifted from released code.

Upstream provenance:
  Repository: https://github.com/qiskit-community/ml-qem
  Commit: b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3 (branch research)
  Source file: docs/tutorials/mlp.py
  SHA-256 of mlp.py: 6d621283927baa80d6ec8377b5bc8c55cb297512deb7e501bc131d5c427d04fe

This module lifts the released function `count_gates_by_rotation_angle` verbatim
by AST from the upstream repository, preserving its exact binning definition
and native-gate set as called by `encode_data_v2_ecr` in the tutorial notebooks
(e.g., h36_ising_4q_hardware_plot.ipynb).

Circuit rebuilding and transpilation:
  - Circuits are reconstructed using QEMScore's canonical descriptor builder
    `qemscore.datasets.schema.build_circuit_from_canonical_descriptor`.
  - Transpilation executes QEMScore's standard generator/sampling pipeline
    `qiskit.transpile(measured, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=tseed % _MOD)`
    matching `qemscore.sampling.sample_counts` and `tools.audit_campaign._compile`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, transpile

from qemscore.datasets.schema import (
    FAMILY_REQUIRED_FIELDS,
    build_circuit_from_canonical_descriptor,
    canonical_physical_circuit_descriptor,
)
from qemscore.datasets.split_generate import _transpile_seed_from_circuit_id
from qemscore.noise.models import BASIS_GATES
from qemscore.sampling import _MOD

UPSTREAM_COMMIT = "b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3"
UPSTREAM_MLP_SHA256 = "6d621283927baa80d6ec8377b5bc8c55cb297512deb7e501bc131d5c427d04fe"

# Released parameters from docs/tutorials/mlp.py (encode_data_v2_ecr):
# When invoked for CX-based backends (e.g., h36 tutorial / S1 reanalysis):
#   two_q_gate = 'cx'
#   gates_set = [two_q_gate] + ['sx', 'x', 'id', 'rz'] -> ['cx', 'sx', 'x', 'id', 'rz']
#   bin_size = 0.025 * np.pi
#   num_angle_bins = int(np.ceil(4 * np.pi / bin_size)) = 160
#   scaling = 0.01 (applied to gate counts and angle bin counts in encode_data_v2_ecr)
TWO_Q_GATE = "cx"
GATES_SET: list[str] = [TWO_Q_GATE, "sx", "x", "id", "rz"]
BIN_SIZE: float = 0.025 * np.pi
NUM_ANGLE_BINS: int = int(np.ceil(4 * np.pi / BIN_SIZE))
BIN_EDGES: np.ndarray = np.arange(-2 * np.pi, 2 * np.pi + BIN_SIZE, BIN_SIZE)
BIN_LABELS: list[str] = [
    f"{left:.2f} to {right:.2f}"
    for left, right in zip(BIN_EDGES[:-1], BIN_EDGES[1:])
]

FEATURE_NAMES: list[str] = list(GATES_SET) + list(BIN_LABELS)


def get_feature_names() -> list[str]:
    """Return the ordered list of 165 feature names (5 gate counts + 160 angle bins)."""
    return list(FEATURE_NAMES)


def find_upstream_mlp_path() -> Path:
    """Locate upstream docs/tutorials/mlp.py relative to repo, env, or working directory."""
    candidates = []
    if "MLQEM_PATH" in os.environ:
        candidates.append(Path(os.environ["MLQEM_PATH"]))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "_upstream" / "ml-qem" / "docs" / "tutorials" / "mlp.py")
    candidates.append(Path.cwd() / "_upstream" / "ml-qem" / "docs" / "tutorials" / "mlp.py")
    candidates.append(repo_root / "reanalysis" / "scripts" / "s1_ml_qem" / "repo" / "docs" / "tutorials" / "mlp.py")

    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not locate upstream mlp.py at b1eccf8. Checked: {[str(c) for c in candidates]}"
    )


def lift_count_gates_by_rotation_angle(mlp_path: Path | str | None = None) -> tuple[Callable, str, str]:
    """Lift count_gates_by_rotation_angle verbatim from mlp.py via AST.

    Returns:
        (function, source_code_segment, file_sha256)
    """
    path = Path(mlp_path) if mlp_path is not None else find_upstream_mlp_path()
    src = path.read_text(encoding="utf-8")
    actual_sha256 = hashlib.sha256(src.encode("utf-8")).hexdigest()
    if actual_sha256 != UPSTREAM_MLP_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {UPSTREAM_MLP_SHA256}, got {actual_sha256}"
        )

    tree = ast.parse(src)
    ns: dict[str, Any] = {"np": np}
    fn_segment = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "count_gates_by_rotation_angle":
            fn_segment = ast.get_source_segment(src, node)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            break

    if "count_gates_by_rotation_angle" not in ns or fn_segment is None:
        raise RuntimeError(f"count_gates_by_rotation_angle not found in {path}")

    return ns["count_gates_by_rotation_angle"], fn_segment, actual_sha256


# Lift function at import time
count_gates_by_rotation_angle, _LIFTED_SOURCE, LIFTED_FILE_SHA256 = (
    lift_count_gates_by_rotation_angle()
)


def rebuild_circuit_for_item(
    item: Mapping[str, Any], dataset_dir: str | Path | None = None
) -> QuantumCircuit:
    """Rebuild an untranspiled QuantumCircuit from an item or its sidecar descriptor.

    Delegates to QEMScore's canonical descriptor builder `build_circuit_from_canonical_descriptor`.
    """
    # 1. If sidecar path is present and dataset_dir is provided, read the canonical sidecar
    if "circuit_sidecar" in item and dataset_dir is not None:
        sidecar_path = Path(dataset_dir) / str(item["circuit_sidecar"])
        if sidecar_path.is_file():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                descriptor = json.load(f)
            return build_circuit_from_canonical_descriptor(descriptor)

    # 2. If item contains a nested "parameters" mapping:
    family = str(item["family"])
    n_qubits = int(item["n_qubits"])
    circuit_seed = int(item.get("circuit_seed", 0))

    if "parameters" in item and isinstance(item["parameters"], Mapping):
        parameters = dict(item["parameters"])
    else:
        # 3. Extract parameter fields directly from item
        req_params = FAMILY_REQUIRED_FIELDS.get(family, ())
        parameters = {param: item[param] for param in req_params if param in item}

    canonical = canonical_physical_circuit_descriptor({
        "family": family,
        "n_qubits": n_qubits,
        "circuit_seed": circuit_seed,
        **parameters,
    })
    return build_circuit_from_canonical_descriptor(canonical)


def transpile_circuit_for_item(
    circuit: QuantumCircuit, item: Mapping[str, Any]
) -> QuantumCircuit:
    """Transpile a QuantumCircuit using QEMScore's exact generator/sampling configuration.

    Appends measure_all(), targets BASIS_GATES at optimization_level=1, and seeds
    the transpiler using the split-v2 circuit digest seed or legacy circuit_seed.
    """
    if "circuit_id" in item and str(item["circuit_id"]).startswith("circuit-"):
        tseed = _transpile_seed_from_circuit_id(str(item["circuit_id"]))
    else:
        tseed = int(item.get("circuit_seed", 0))

    measured = circuit.copy()
    measured.measure_all()
    return transpile(
        measured,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=int(tseed % _MOD),
    )


def rebuild_and_transpile_circuit(
    item: Mapping[str, Any], dataset_dir: str | Path | None = None
) -> QuantumCircuit:
    """Rebuild and transpile the circuit for an item."""
    circ = rebuild_circuit_for_item(item, dataset_dir=dataset_dir)
    return transpile_circuit_for_item(circ, item)


def encode_item_circuit(
    item: Mapping[str, Any],
    dataset_dir: str | Path | None = None,
    *,
    scaled: bool = True,
) -> list[float]:
    """Encode an item's transpiled circuit into ML-QEM binned rotation angle features.

    Features:
      - 5 native gate counts: ['cx', 'sx', 'x', 'id', 'rz']
      - 160 rotation angle bins over [-2*pi, 2*pi] with bin_size = 0.025 * pi

    By default `scaled=True` applies the 0.01 multiplicative scaling from
    `encode_data_v2_ecr` (`torch.tensor(...) * 0.01`). If `scaled=False`, returns
    raw count values as floats.
    """
    tcirc = rebuild_and_transpile_circuit(item, dataset_dir=dataset_dir)

    # 1. Native gate counts
    gate_counts_all = tcirc.count_ops()
    gate_counts = [float(gate_counts_all.get(key, 0)) for key in GATES_SET]

    # 2. Verbatim binned rotation angles
    angle_counts = [float(c) for c in count_gates_by_rotation_angle(tcirc, BIN_SIZE)]

    factor = 0.01 if scaled else 1.0
    return [(val * factor) for val in (gate_counts + angle_counts)]


def verify_dataset(
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """Verify all unique circuits in dataset_dir, reporting mismatches and information metrics.

    Checks:
      1. Every unique circuit's rebuilt transpiled CX count and depth equals the
         item's two_qubit_gates and transpiled_depth.
      2. Computes distinct encoded feature vectors and collisions for TFI and Heisenberg
         training circuits.
      3. Verifies Heisenberg coupling permutation invariance (jx, jy, jz).
    """
    path = Path(dataset_dir)
    items_path = path / "items.jsonl"
    if not items_path.is_file():
        raise FileNotFoundError(f"items.jsonl not found at {items_path}")

    unique_circuits: dict[str, dict[str, Any]] = {}
    train_circuits: dict[str, dict[str, Any]] = {}

    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cid = str(item["circuit_id"])
            if cid not in unique_circuits:
                unique_circuits[cid] = item
            if item.get("split") == "train" and cid not in train_circuits:
                train_circuits[cid] = item

    mismatches: list[dict[str, Any]] = []
    for cid, item in unique_circuits.items():
        tcirc = rebuild_and_transpile_circuit(item, dataset_dir=path)
        actual_cx = int(tcirc.count_ops().get("cx", 0))
        actual_depth = int(tcirc.depth())
        expected_cx = int(item["two_qubit_gates"])
        expected_depth = int(item["transpiled_depth"])
        if actual_cx != expected_cx or actual_depth != expected_depth:
            mismatches.append({
                "circuit_id": cid,
                "item_id": item["item_id"],
                "expected": {"two_qubit_gates": expected_cx, "transpiled_depth": expected_depth},
                "actual": {"two_qubit_gates": actual_cx, "transpiled_depth": actual_depth},
            })

    family_reports: dict[str, dict[str, Any]] = {}
    for fam in ("tfi", "heisenberg"):
        fam_circuits = {cid: it for cid, it in train_circuits.items() if it["family"] == fam}
        vec_to_circs: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for cid, it in fam_circuits.items():
            vec = tuple(encode_item_circuit(it, dataset_dir=path, scaled=False))
            vec_to_circs.setdefault(vec, []).append(it)

        distinct_count = len(vec_to_circs)
        collision_groups = [circs for circs in vec_to_circs.values() if len(circs) > 1]
        collisions = sum(len(circs) - 1 for circs in collision_groups)
        family_reports[fam] = {
            "total_train_circuits": len(fam_circuits),
            "distinct_vectors": distinct_count,
            "collisions": collisions,
            "collision_groups": len(collision_groups),
            "max_group_size": max(len(circs) for circs in vec_to_circs.values()) if vec_to_circs else 0,
        }

    # Heisenberg permutation check
    test_h1 = {
        "family": "heisenberg",
        "n_qubits": 4,
        "circuit_seed": 42,
        "circuit_id": "circuit-0000000000000000000000000000000000000000000000000000000000000001",
        "parameters": {"steps": 1, "jx": 0.5, "jy": 0.8, "jz": 1.0, "dt": 0.15},
    }
    test_h2 = {
        "family": "heisenberg",
        "n_qubits": 4,
        "circuit_seed": 42,
        "circuit_id": "circuit-0000000000000000000000000000000000000000000000000000000000000002",
        "parameters": {"steps": 1, "jx": 0.8, "jy": 0.5, "jz": 1.0, "dt": 0.15},
    }
    vh1 = encode_item_circuit(test_h1, scaled=False)
    vh2 = encode_item_circuit(test_h2, scaled=False)
    heisenberg_can_distinguish_couplings = (vh1 != vh2)

    return {
        "total_unique_circuits": len(unique_circuits),
        "mismatches": mismatches,
        "family_reports": family_reports,
        "heisenberg_can_distinguish_couplings": heisenberg_can_distinguish_couplings,
    }


if __name__ == "__main__":
    import sys

    default_dataset = Path("data/regen-shipped-s101-n640")
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dataset
    print(f"Verifying dataset: {dataset}")
    report = verify_dataset(dataset)
    print(f"Total unique circuits: {report['total_unique_circuits']}")
    print(f"Mismatches: {len(report['mismatches'])}")
    for fam, f_rep in report["family_reports"].items():
        print(f"\n{fam.upper()} training circuits:")
        print(f"  Total circuits: {f_rep['total_train_circuits']}")
        print(f"  Distinct encoded vectors: {f_rep['distinct_vectors']}")
        print(f"  Collisions: {f_rep['collisions']}")
        print(f"  Collision groups: {f_rep['collision_groups']}")
        print(f"  Max collision group size: {f_rep['max_group_size']}")
    print(f"\nHeisenberg distinguishes jx, jy, jz: {report['heisenberg_can_distinguish_couplings']}")
