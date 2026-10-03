"""Tests for tools.mlqem_binned_angle_encoder.

Verifies:
1. Upstream AST lifting provenance and SHA-256 integrity.
2. Feature schema (5 native gates + 160 rotation angle bins).
3. Deterministic encoding across a small 3-circuit test suite.
4. Information loss: Heisenberg coupling permutation invariance (v2 == v3).
5. Generator path fidelity: rebuilt transpiled circuits match recorded item structure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

try:
    from tools.mlqem_binned_angle_encoder import (
        BIN_LABELS,
        BIN_SIZE,
        FEATURE_NAMES,
        GATES_SET,
        LIFTED_FILE_SHA256,
        NUM_ANGLE_BINS,
        UPSTREAM_COMMIT,
        UPSTREAM_MLP_SHA256,
        count_gates_by_rotation_angle,
        encode_item_circuit,
        get_feature_names,
        lift_count_gates_by_rotation_angle,
        rebuild_and_transpile_circuit,
        rebuild_circuit_for_item,
    )
except FileNotFoundError as exc:
    pytest.skip(
        f"Upstream ml-qem source not available: {exc}",
        allow_module_level=True,
    )

pytestmark = pytest.mark.filterwarnings(
    "ignore:Treating CircuitInstruction as an iterable:DeprecationWarning"
)

# A regenerated primary dataset; the dataset test is skipped when it is absent.
DATASET_DIR = Path(
    os.environ.get("QEMSCORE_REGEN_S101", "data/regen-shipped-s101-n640")
)


def test_lifted_encoder_provenance():
    """Verify count_gates_by_rotation_angle is lifted verbatim from mlp.py at commit b1eccf8."""
    assert UPSTREAM_COMMIT.startswith("b1eccf8")
    assert LIFTED_FILE_SHA256 == UPSTREAM_MLP_SHA256

    fn, segment, sha = lift_count_gates_by_rotation_angle()
    assert callable(fn)
    assert fn.__name__ == "count_gates_by_rotation_angle"
    assert "def count_gates_by_rotation_angle" in segment
    assert sha == UPSTREAM_MLP_SHA256


def test_feature_names_and_shapes():
    """Verify feature dimension is 5 native gates + 160 angle bins = 165 features."""
    names = get_feature_names()
    assert len(names) == 165
    assert len(FEATURE_NAMES) == 165
    assert names == FEATURE_NAMES
    assert names[:5] == ["cx", "sx", "x", "id", "rz"]
    assert len(BIN_LABELS) == 160
    assert NUM_ANGLE_BINS == 160
    assert names[5:] == BIN_LABELS


def test_three_circuits_deterministic_encoding():
    """Deterministic test on three distinct circuits:

    - Circuit 1: TFI (4 qubits, 1 Trotter step, dt=0.2, j=0.5, h=0.8)
    - Circuit 2: Heisenberg (4 qubits, 1 Trotter step, dt=0.15, jx=0.5, jy=0.8, jz=1.0)
    - Circuit 3: Heisenberg permuted couplings (jx=0.8, jy=0.5, jz=1.0)

    Verifies:
      * Determinism (re-encoding gives bit-identical vectors).
      * Scaling behavior (scaled=True is 0.01 * scaled=False).
      * Heisenberg permutation collision: Circuit 2 and Circuit 3 map to identical vectors.
    """
    item1 = {
        "family": "tfi",
        "n_qubits": 4,
        "circuit_seed": 100,
        "circuit_id": "circuit-0000000000000000000000000000000000000000000000000000000000000001",
        "parameters": {"steps": 1, "j": 0.5, "h": 0.8, "dt": 0.2},
    }
    item2 = {
        "family": "heisenberg",
        "n_qubits": 4,
        "circuit_seed": 200,
        "circuit_id": "circuit-0000000000000000000000000000000000000000000000000000000000000002",
        "parameters": {"steps": 1, "jx": 0.5, "jy": 0.8, "jz": 1.0, "dt": 0.15},
    }
    item3 = {
        "family": "heisenberg",
        "n_qubits": 4,
        "circuit_seed": 200,
        "circuit_id": "circuit-0000000000000000000000000000000000000000000000000000000000000003",
        "parameters": {"steps": 1, "jx": 0.8, "jy": 0.5, "jz": 1.0, "dt": 0.15},
    }

    # 1. Circuit 1 (TFI)
    v1_unscaled = encode_item_circuit(item1, scaled=False)
    v1_scaled = encode_item_circuit(item1, scaled=True)
    assert len(v1_unscaled) == 165
    assert len(v1_scaled) == 165
    assert np.allclose(np.array(v1_scaled), np.array(v1_unscaled) * 0.01)

    tc1 = rebuild_and_transpile_circuit(item1)
    cx1 = tc1.count_ops().get("cx", 0)
    sx1 = tc1.count_ops().get("sx", 0)
    rz1 = tc1.count_ops().get("rz", 0)
    assert cx1 == 6
    assert sx1 == 8
    assert rz1 == 15
    assert v1_unscaled[:5] == [6.0, 8.0, 0.0, 0.0, 15.0]

    # 2. Circuit 2 (Heisenberg)
    v2_unscaled = encode_item_circuit(item2, scaled=False)
    v2_scaled = encode_item_circuit(item2, scaled=True)
    tc2 = rebuild_and_transpile_circuit(item2)
    assert tc2.count_ops().get("cx", 0) == 18
    assert tc2.count_ops().get("sx", 0) == 18
    assert tc2.count_ops().get("rz", 0) == 27
    assert v2_unscaled[:5] == [18.0, 18.0, 0.0, 0.0, 27.0]

    # 3. Circuit 3 (Heisenberg with jx and jy permuted)
    v3_unscaled = encode_item_circuit(item3, scaled=False)
    v3_scaled = encode_item_circuit(item3, scaled=True)
    assert v2_unscaled == v3_unscaled
    assert v2_scaled == v3_scaled

    # Re-encoding produces bit-identical results (determinism)
    assert encode_item_circuit(item1, scaled=True) == v1_scaled
    assert encode_item_circuit(item2, scaled=True) == v2_scaled


@pytest.mark.skipif(not DATASET_DIR.is_dir(), reason="Regen-shipped s101 dataset not available")
def test_dataset_rebuild_structure_fidelity():
    """Verify that rebuilding and transpiling dataset items reproduces exact recorded metrics."""
    items_path = DATASET_DIR / "items.jsonl"
    tested_families = set()
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            fam = item["family"]
            if fam not in tested_families:
                tcirc = rebuild_and_transpile_circuit(item, dataset_dir=DATASET_DIR)
                assert int(tcirc.count_ops().get("cx", 0)) == int(item["two_qubit_gates"])
                assert int(tcirc.depth()) == int(item["transpiled_depth"])

                encoded = encode_item_circuit(item, dataset_dir=DATASET_DIR)
                assert len(encoded) == 165
                tested_families.add(fam)
            if tested_families == {"tfi", "heisenberg"}:
                break
