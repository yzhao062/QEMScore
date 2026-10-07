"""Exact labels and exact circuit parameter recovery for ML-QEM data.

Governed by docs/frozen-rules/2026-10-06-round9-follow-ups.md.
Extracts exact statevector labels and circuit parameters from Trotter circuits.
Validates archived noisy simulator labels against binomial shot noise.
"""

import os
import sys
import json
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from reanalysis.mlqem.encode import (
    load_split_data,
    compute_file_sha256,
    SETTINGS,
    get_setting_config,
)

SETTING_DIRS: Dict[str, str] = {
    "no_readout": "ising_init_from_qasm_no_readout",
    "readout": "ising_init_from_qasm",
    "coherent": "ising_init_from_qasm_coherent",
}


def canonical_setting(setting: str) -> str:
    """Maps raw directory name to its canonical setting name."""
    for name, dir_name in SETTING_DIRS.items():
        if setting == dir_name:
            return name
    return setting


BASIS_MAP = {"X": 0, "Y": 1, "Z": 2}
BASIS_INV_MAP = {0: "X", 1: "Y", 2: "Z"}

PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def exact_labels(circuit: Any) -> np.ndarray:
    """Computes exact statevector labels for a quantum circuit.

    The circuit can be a QuantumCircuit or a QASM string.
    Convention matches ML-QEM cal_z_exp.
    Label i corresponds to clbit n-1-i.
    Returns expectation values in range [-1.0, 1.0] with shape (len(clbits),).
    """
    if isinstance(circuit, str):
        qc = QuantumCircuit.from_qasm_str(circuit)
    elif isinstance(circuit, QuantumCircuit):
        qc = circuit
    else:
        raise TypeError(f"Expected QuantumCircuit or QASM string, got {type(circuit)}")

    meas = {}
    for inst in qc.data:
        if inst.operation.name == "measure":
            c_idx = qc.find_bit(inst.clbits[0]).index
            q_idx = qc.find_bit(inst.qubits[0]).index
            meas[c_idx] = q_idx

    if not meas:
        raise ValueError("Circuit contains no measurement operations.")

    bare = qc.remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(bare).probabilities()
    idx = np.arange(2 ** bare.num_qubits)
    out = []
    n_clbits = len(meas)
    for i in range(n_clbits):
        target_clbit = n_clbits - 1 - i
        if target_clbit not in meas:
            raise ValueError(f"Clbit index {target_clbit} missing from measurements.")
        q = meas[target_clbit]
        p1 = probs[((idx >> q) & 1) == 1].sum()
        out.append(2.0 * p1 - 1.0)
    return np.array(out, dtype=np.float64)


def _get_single_qubit_unitary(gates: List[Tuple[str, List[Any]]]) -> np.ndarray:
    """Computes 2x2 unitary matrix for single qubit gate sequence."""
    u = np.eye(2, dtype=complex)
    for name, params in gates:
        if name == "rz":
            theta = float(params[0])
            g = np.array(
                [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
                dtype=complex,
            )
        elif name == "sx":
            g = 0.5 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=complex)
        elif name == "x":
            g = PAULI_X
        elif name == "y":
            g = PAULI_Y
        elif name == "z":
            g = PAULI_Z
        elif name == "h":
            g = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
        elif name == "sdg":
            g = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex)
        elif name == "s":
            g = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
        elif name == "id":
            g = np.eye(2, dtype=complex)
        else:
            raise ValueError(f"Unsupported single qubit operation: {name}")
        u = g @ u
    return u


def _classify_pauli(u: np.ndarray) -> str:
    """Classifies single-qubit measurement basis from unitary U."""
    if abs(u[0, 1]) < 1e-6 and abs(u[1, 0]) < 1e-6:
        return "Z"
    u_dag = u.conj().T
    p = u_dag @ PAULI_Z @ u
    for sign in (1.0, -1.0):
        if np.allclose(p, sign * PAULI_X, atol=1e-5):
            return "X"
        if np.allclose(p, sign * PAULI_Y, atol=1e-5):
            return "Y"
        if np.allclose(p, sign * PAULI_Z, atol=1e-5):
            return "Z"
    raise ValueError(f"Could not classify measured Pauli from unitary matrix:\n{u}")


def extract_basis(circuit: Any) -> str:
    """Extracts measurement basis (X, Y, or Z) from circuit."""
    if isinstance(circuit, str):
        qc = QuantumCircuit.from_qasm_str(circuit)
    elif isinstance(circuit, QuantumCircuit):
        qc = circuit
    else:
        raise TypeError(f"Expected QuantumCircuit or QASM string, got {type(circuit)}")

    meas = {}
    for inst in qc.data:
        if inst.operation.name == "measure":
            c_idx = qc.find_bit(inst.clbits[0]).index
            q_idx = qc.find_bit(inst.qubits[0]).index
            meas[c_idx] = q_idx

    barriers = [i for i, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
    if len(barriers) < 2:
        raise ValueError(f"Circuit has {len(barriers)} barriers, expected at least 2.")

    b_prev = barriers[-2]
    b_last = barriers[-1]

    qubit_bases = []
    for c_idx in sorted(meas.keys()):
        q = meas[c_idx]
        q_gates = []
        for i in range(b_prev + 1, b_last):
            inst = qc.data[i]
            if len(inst.qubits) == 1 and qc.find_bit(inst.qubits[0]).index == q:
                q_gates.append((inst.operation.name, inst.operation.params))
        u = _get_single_qubit_unitary(q_gates)
        b = _classify_pauli(u)
        qubit_bases.append(b)

    if len(set(qubit_bases)) != 1:
        raise ValueError(f"Measured qubit bases disagree: {qubit_bases}.")

    return qubit_bases[0]


def count_sandwiches(circuit: Any) -> int:
    """Counts cx, rz, cx triples on one pair with the rz on the CX target."""
    data = circuit.data
    count = 0
    for i in range(len(data) - 2):
        a, b, c = data[i], data[i + 1], data[i + 2]
        if a.operation.name == "cx" and b.operation.name == "rz" and c.operation.name == "cx":
            if a.qubits == c.qubits and len(b.qubits) == 1 and b.qubits[0] == a.qubits[1]:
                count += 1
    return count


def circuit_parameters(circuit: Any, steps: int) -> Dict[str, Any]:
    """Recovers coupling J, basis, steps, and sandwich count.

    Raises if sandwich angles disagree by more than 1e-9.
    Raises if sandwich count is not 3 * steps.
    Raises if qubit measurement bases disagree.
    At step 0, J is NaN.
    """
    if isinstance(circuit, str):
        qc = QuantumCircuit.from_qasm_str(circuit)
    elif isinstance(circuit, QuantumCircuit):
        qc = circuit
    else:
        raise TypeError(f"Expected QuantumCircuit or QASM string, got {type(circuit)}")

    sandwiches = []
    data = qc.data
    for i in range(len(data) - 2):
        op1, q1 = data[i].operation, data[i].qubits
        op2, q2 = data[i + 1].operation, data[i + 1].qubits
        op3, q3 = data[i + 2].operation, data[i + 2].qubits
        if op1.name == "cx" and op2.name == "rz" and op3.name == "cx":
            if q1 == q3 and len(q2) == 1 and q2[0] == q1[1]:
                theta = float(op2.params[0])
                sandwiches.append((q1, theta))

    n_sandwiches = len(sandwiches)
    expected_sandwiches = 3 * steps
    if n_sandwiches != expected_sandwiches:
        raise ValueError(
            f"Expected {expected_sandwiches} sandwiches for step {steps}, found {n_sandwiches}."
        )

    if steps == 0:
        j_val = float("nan")
    else:
        thetas = [s[1] for s in sandwiches]
        theta_spread = np.max(thetas) - np.min(thetas)
        if theta_spread > 1e-9:
            raise ValueError(f"Sandwich angles disagree by {theta_spread:.3e} > 1e-9.")
        j_val = float(-np.mean(thetas))

    basis_str = extract_basis(qc)

    return {
        "J": j_val,
        "basis": basis_str,
        "steps": int(steps),
        "n_sandwiches": int(n_sandwiches),
    }


def compute_validation_block(
    exact: np.ndarray,
    archived: np.ndarray,
    j_arr: np.ndarray,
    basis_arr: np.ndarray,
    steps_arr: np.ndarray,
    failures: int,
) -> Dict[str, Any]:
    """Computes validation metrics for exact versus archived labels."""
    z_scores = []
    exact_ones = 0
    flat_exact = exact.flatten()
    flat_arch = archived.flatten()

    for ex, ar in zip(flat_exact, flat_arch):
        remainder = 1.0 - (ex ** 2)
        if remainder >= 1e-6:
            var = remainder / 10000.0
            z = (ar - ex) / np.sqrt(var)
            z_scores.append(float(z))
        else:
            diff = abs(ar - ex)
            if diff >= 1e-12:
                raise ValueError(
                    f"Boundary entry mismatch: exact={ex}, archived={ar}, diff={diff} >= 1e-12."
                )
            exact_ones += 1

    z_arr = np.array(z_scores) if z_scores else np.array([0.0])
    valid_j = j_arr[steps_arr > 0]
    finite_j = valid_j[np.isfinite(valid_j)]
    j_min = float(np.min(finite_j)) if len(finite_j) > 0 else float("nan")
    j_max = float(np.max(finite_j)) if len(finite_j) > 0 else float("nan")

    counts = {
        name: int(np.sum(basis_arr == code)) for name, code in BASIS_MAP.items()
    }

    return {
        "z_mean": float(np.mean(z_arr)),
        "z_sd": float(np.std(z_arr, ddof=1)) if len(z_arr) > 1 else 0.0,
        "z_max_abs": float(np.max(np.abs(z_arr))),
        "z_gt_4_count": int(np.sum(np.abs(z_arr) > 4.0)),
        "z_gt_6_count": int(np.sum(np.abs(z_arr) > 6.0)),
        "exact_ones_count": int(exact_ones),
        "exact_ones_mismatch_count": 0,
        "j_min": j_min,
        "j_max": j_max,
        "j_range": [j_min, j_max],
        "basis_counts": counts,
        "extraction_failures": int(failures),
        "total_circuits": int(len(j_arr)),
        "total_label_points": int(len(flat_exact)),
    }


REQUIRED_GATE_SPLITS: List[Tuple[str, str]] = [
    ("no_readout", "train"),
    ("no_readout", "val"),
    ("no_readout", "val_extra"),
    ("readout", "train"),
    ("readout", "val"),
    ("readout", "val_Zonly"),
    ("coherent", "train"),
    ("coherent", "val"),
]


def evaluate_gate(exact_dir: str, frozen_rule_path: str) -> Dict[str, Any]:
    """Evaluates exact label gate across all eight required splits and writes gate.json."""
    if not os.path.isfile(frozen_rule_path):
        raise FileNotFoundError(f"Frozen rule not found: {frozen_rule_path}")

    rule_sha = compute_file_sha256(frozen_rule_path)
    splits_record: Dict[str, Any] = {}
    all_passed = True

    for setting, split in REQUIRED_GATE_SPLITS:
        split_key = f"{setting}:{split}"
        manifest_candidates = [
            os.path.join(exact_dir, f"exact-{setting}-{split}.json"),
            os.path.join(exact_dir, f"manifest-{setting}-{split}.json"),
            os.path.join(exact_dir, f"exact-{SETTING_DIRS.get(setting, setting)}-{split}.json"),
            os.path.join(exact_dir, f"manifest-{SETTING_DIRS.get(setting, setting)}-{split}.json"),
        ]
        manifest_path = next((c for c in manifest_candidates if os.path.isfile(c)), None)

        npz_candidates = [
            os.path.join(exact_dir, f"exact-{setting}-{split}.npz"),
            os.path.join(exact_dir, f"exact-{SETTING_DIRS.get(setting, setting)}-{split}.npz"),
        ]
        npz_path = next((c for c in npz_candidates if os.path.isfile(c)), None)

        if not manifest_path or not npz_path:
            splits_record[split_key] = {
                "passed": False,
                "reason": "missing_manifest_or_npz",
                "manifest_found": bool(manifest_path),
                "npz_found": bool(npz_path),
            }
            all_passed = False
            continue

        try:
            with open(manifest_path, "r") as handle:
                manifest_data = json.load(handle)
        except Exception as exc:
            splits_record[split_key] = {"passed": False, "reason": f"invalid_manifest_json: {exc}"}
            all_passed = False
            continue

        actual_npz_sha = compute_file_sha256(npz_path)
        expected_npz_sha = manifest_data.get("output_npz_sha256", manifest_data.get("npz_sha256"))
        sha_match = bool(actual_npz_sha == expected_npz_sha)

        val = manifest_data.get("validation", {})
        z_mean = float(val.get("z_mean", float("nan")))
        z_sd = float(val.get("z_sd", float("nan")))
        z_max_abs = float(val.get("z_max_abs", val.get("max_abs_z", float("nan"))))
        z_gt_6_count = int(val.get("z_gt_6_count", 0))
        mismatch_count = int(val.get("exact_ones_mismatch_count", 0))

        mean_ok = bool(abs(z_mean) <= 0.1)
        sd_ok = bool(0.9 <= z_sd <= 1.1)
        max_ok = bool((np.isnan(z_max_abs) or z_max_abs <= 6.0) and z_gt_6_count == 0)
        boundary_ok = bool(mismatch_count == 0)

        split_passed = bool(sha_match and mean_ok and sd_ok and max_ok and boundary_ok)
        if not split_passed:
            all_passed = False

        splits_record[split_key] = {
            "passed": split_passed,
            "pass": split_passed,
            "split_passed": split_passed,
            "manifest_file": os.path.basename(manifest_path),
            "npz_file": os.path.basename(npz_path),
            "npz_sha256": actual_npz_sha,
            "manifest_npz_sha256": expected_npz_sha,
            "sha_match": sha_match,
            "z_mean": z_mean,
            "z_sd": z_sd,
            "z_max_abs": z_max_abs,
            "exact_ones_mismatch_count": mismatch_count,
            "thresholds": {
                "abs_z_mean_le_0_1": mean_ok,
                "z_sd_in_0_9_1_1": sd_ok,
                "no_z_gt_6": max_ok,
                "near_pm1_equal_1e12": boundary_ok,
            },
        }

    overall_passed = bool(len(splits_record) == len(REQUIRED_GATE_SPLITS) and all_passed)
    gate_record = {
        "gate_passed": overall_passed,
        "overall_pass": overall_passed,
        "frozen_rule": os.path.abspath(frozen_rule_path),
        "frozen_rule_sha256": rule_sha,
        "thresholds": {
            "max_abs_z_mean": 0.1,
            "z_sd_bounds": [0.9, 1.1],
            "max_abs_z": 6.0,
            "boundary_diff_tolerance": 1e-12,
        },
        "required_splits_count": len(REQUIRED_GATE_SPLITS),
        "evaluated_splits_count": len(splits_record),
        "splits": splits_record,
    }

    gate_path = os.path.join(exact_dir, "gate.json")
    with open(gate_path, "w") as handle:
        json.dump(gate_record, handle, indent=2)

    return gate_record


def check_gate_approval(exact_dir: str, expected_npz_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Requires that gate.json exists in exact_dir, passed, and matches expected NPZ hashes."""
    gate_path = os.path.join(exact_dir, "gate.json")
    if not os.path.isfile(gate_path):
        raise FileNotFoundError(
            f"Gate file not found: {gate_path}. Exact label operations require gate.json to exist and pass."
        )
    with open(gate_path, "r") as handle:
        gate_data = json.load(handle)
    is_passed = gate_data.get("gate_passed", gate_data.get("overall_pass", False))
    if not is_passed:
        raise ValueError(
            f"Exact label gate check failed in {gate_path}. Operations on exact targets/descriptors are blocked."
        )

    if expected_npz_paths:
        splits = gate_data.get("splits", {})
        hash_to_split = {s_info.get("npz_sha256"): k for k, s_info in splits.items() if "npz_sha256" in s_info}
        for npz_path in expected_npz_paths:
            h = compute_file_sha256(npz_path)
            fn = os.path.basename(npz_path)
            found = False
            for s_key, s_info in splits.items():
                if s_info.get("npz_file") == fn:
                    found = True
                    if s_info.get("npz_sha256") != h:
                        raise ValueError(
                            f"{npz_path} SHA-256 {h} does not match gate.json recorded SHA-256 {s_info.get('npz_sha256')}"
                        )
                    break
            if not found and h not in hash_to_split:
                raise ValueError(f"{npz_path} is not recorded in {gate_path}")

    return gate_data


def verify_split_manifest(
    dir_name: str,
    split: str,
    files_sha256: Dict[str, str],
    manifest_path: str,
) -> None:
    """Verifies that split source files match the upstream manifest."""
    with open(manifest_path, "r") as handle:
        expected = json.load(handle)
    problems = []
    for fn, sha in sorted(files_sha256.items()):
        key = f"docs/tutorials/data/{dir_name}/{split}/{fn}"
        if key not in expected:
            problems.append(f"{key}: not found in manifest")
        elif expected[key] != sha:
            problems.append(f"{key}: SHA-256 {sha} != {expected[key]}")
    if problems:
        raise SystemExit("Split manifest verification failed:\n  " + "\n  ".join(problems))


def locate_split_dir(data_root: str, dir_name: str, split: str) -> str:
    """Locates split directory under data_root."""
    candidates = [
        os.path.join(data_root, dir_name, split),
        os.path.join(data_root, "data", dir_name, split),
        os.path.join(data_root, "docs/tutorials/data", dir_name, split),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Could not locate split directory for {dir_name}/{split} under {data_root}. "
        f"Checked: {candidates}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute exact ML-QEM labels and parameters.")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing ml-qem datasets")
    parser.add_argument("--settings", nargs="+",
                        default=["no_readout", "readout", "coherent"],
                        help="Settings to process (e.g. no_readout, readout, coherent)")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to process (train, val, val_extra, val_Zonly)")
    parser.add_argument("--out-dir", type=str, default="artifacts/mlqem-exact",
                        help="Output directory for exact npz and json files")
    parser.add_argument("--frozen-rule", type=str, required=True,
                        help="Path to frozen rule file (must exist)")
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(15)),
                        help="Trotter steps to include (default: 0 to 14)")
    parser.add_argument("--no-manifest-check", action="store_true",
                        help="Skip checking source files against upstream manifest")
    return parser.parse_args()


def process_split(
    setting: str,
    split: str,
    data_root: str,
    out_dir: str,
    steps: List[int],
    frozen_rule_path: str,
    check_manifest: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    """Processes a single setting and split, writing npz and json files."""
    config = get_setting_config(setting)
    dir_name = config["dir_name"]
    split_dir = locate_split_dir(data_root, dir_name, split)

    entries, steps_list, circuit_ids, files_sha256 = load_split_data(split_dir, steps)

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_data_sha256.json")
    if check_manifest:
        verify_split_manifest(dir_name, split, files_sha256, manifest_path)

    exact_list = []
    archived_list = []
    j_list = []
    basis_list = []
    steps_out = []
    step_file_list = []
    entry_index_list = []
    failures = 0
    zero_sandwich: List[str] = []

    for idx, (entry, step_val, cid) in enumerate(zip(entries, steps_list, circuit_ids)):
        fn, entry_idx = cid
        circ = entry.get("circuit")
        if isinstance(circ, str):
            circ = QuantumCircuit.from_qasm_str(circ)
        elif circ is None and "circuit_twirled" in entry:
            circ = entry["circuit_twirled"]

        ideal = entry.get("ideal_exp_value")
        if ideal is None:
            ideal = entry.get("ideal_exp_val")
        if ideal is None:
            ideal = entry.get("ideal")
        if isinstance(ideal, np.ndarray):
            ideal = ideal.tolist()

        ex_lbl = exact_labels(circ)
        exact_list.append(ex_lbl)
        archived_list.append(ideal)
        steps_out.append(step_val)
        step_file_list.append(fn)
        entry_index_list.append(entry_idx)

        n_sandwich = count_sandwiches(circ)
        if step_val > 0 and n_sandwich == 0:
            # The transpiler removed an rz below its tolerance and cancelled the
            # CX pairs, so the executed circuit is the J = 0 circuit. The rule
            # sets J to 0 and records the circuit; every other failure stops.
            failures += 1
            zero_sandwich.append(f"{fn}:{entry_idx}")
            j_list.append(0.0)
            basis_list.append(BASIS_MAP[extract_basis(circ)])
        else:
            params = circuit_parameters(circ, step_val)
            j_list.append(params["J"])
            basis_list.append(BASIS_MAP[params["basis"]])

    exact_arr = np.array(exact_list, dtype=np.float64)
    archived_arr = np.array(archived_list, dtype=np.float64)
    j_arr = np.array(j_list, dtype=np.float64)
    basis_arr = np.array(basis_list, dtype=np.int64)
    steps_arr = np.array(steps_out, dtype=np.int64)
    step_file_arr = np.array(step_file_list, dtype=str)
    entry_index_arr = np.array(entry_index_list, dtype=np.int64)

    val_block = compute_validation_block(
        exact=exact_arr,
        archived=archived_arr,
        j_arr=j_arr,
        basis_arr=basis_arr,
        steps_arr=steps_arr,
        failures=failures,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_base = f"exact-{setting}-{split}"
    npz_path = os.path.join(out_dir, f"{out_base}.npz")
    json_path = os.path.join(out_dir, f"{out_base}.json")

    np.savez_compressed(
        npz_path,
        exact=exact_arr,
        archived=archived_arr,
        J=j_arr,
        basis=basis_arr,
        steps=steps_arr,
        step_file=step_file_arr,
        entry_index=entry_index_arr,
    )

    npz_sha = compute_file_sha256(npz_path)
    script_sha = compute_file_sha256(os.path.abspath(__file__))
    rule_sha = compute_file_sha256(os.path.abspath(frozen_rule_path))

    manifest_record = {
        "script": os.path.basename(__file__),
        "script_sha256": script_sha,
        "frozen_rule": os.path.abspath(frozen_rule_path),
        "frozen_rule_sha256": rule_sha,
        "setting": setting,
        "split": split,
        "source_files_sha256": files_sha256,
        "manifest_checked": bool(check_manifest),
        "output_npz": os.path.basename(npz_path),
        "output_npz_sha256": npz_sha,
        "validation": val_block,
        "zero_sandwich_circuits_J_set_to_0": zero_sandwich,
    }

    with open(json_path, "w") as handle:
        json.dump(manifest_record, handle, indent=2)

    return npz_path, json_path, val_block


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate_parser = argparse.ArgumentParser(
            prog="python -m reanalysis.mlqem.exact gate",
            description="Evaluate exact label gate across all eight required splits.",
        )
        gate_parser.add_argument(
            "--exact-dir",
            type=str,
            default="artifacts/mlqem-exact",
            help="Directory containing exact npz and manifest files",
        )
        gate_parser.add_argument(
            "--frozen-rule",
            type=str,
            required=True,
            help="Path to frozen rule file (must exist)",
        )
        gate_args = gate_parser.parse_args(sys.argv[2:])
        res = evaluate_gate(exact_dir=gate_args.exact_dir, frozen_rule_path=gate_args.frozen_rule)
        gate_path = os.path.join(gate_args.exact_dir, "gate.json")
        print(f"Gate evaluation written to {gate_path}")
        print(f"Overall status: {'PASSED' if res['gate_passed'] else 'FAILED'}")
        if not res["gate_passed"]:
            sys.exit(1)
        return

    args = parse_args()
    if not args.frozen_rule or not os.path.isfile(args.frozen_rule):
        raise RuntimeError(f"Frozen rule file missing or invalid: {args.frozen_rule!r}")

    settings = [canonical_setting(s) for s in args.settings]
    for setting in settings:
        for split in args.splits:
            print(f"Processing setting={setting} split={split}...")
            npz_p, json_p, val = process_split(
                setting=setting,
                split=split,
                data_root=args.data_root,
                out_dir=args.out_dir,
                steps=args.steps,
                frozen_rule_path=args.frozen_rule,
                check_manifest=not args.no_manifest_check,
            )
            print(f"Written: {npz_p}")
            print(f"Written: {json_p}")
            print(f"Validation block for {setting} {split}:")
            print(f"  z mean: {val['z_mean']:.6f}, sd: {val['z_sd']:.6f}")
            print(f"  max |z|: {val['z_max_abs']:.4f}, count |z| > 4: {val['z_gt_4_count']}")
            print(f"  exact ones count: {val['exact_ones_count']}")
            print(f"  J range: [{val['j_min']:.6f}, {val['j_max']:.6f}]")
            print(f"  basis counts: {val['basis_counts']}")
            print(f"  extraction failures: {val['extraction_failures']}")


if __name__ == "__main__":
    main()
