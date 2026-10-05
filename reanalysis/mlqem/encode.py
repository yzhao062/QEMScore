"""Tabular feature encoding for ML-QEM replication control.

Wraps and mirrors blackwater.library.learning.mlp.encode_data, returning:
- X: torch.Tensor feature matrix (N x 58)
- y: torch.Tensor ideal label matrix (N x 4)
- noisy_col_indices: tuple (start_col, end_col) = (54, 58) identifying noisy-expectation features.

Defines the canonical experimental settings and split configurations (Part 1 & 2).
"""

import os
import re
import pickle
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeLima

from blackwater.data.utils import get_backend_properties_v1
from blackwater.library.learning.mlp import encode_data


def compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# Canonical dataset split specifications
SETTINGS: Dict[str, Dict[str, Any]] = {
    "no_readout": {
        "dir_name": "ising_init_from_qasm_no_readout",
        "train_split": "train",
        "train_steps": list(range(15)),
        "val_split": "val",
        "val_steps": list(range(15)),
        "test_split": "val_extra",
        "test_steps": list(range(15)),
        # The published notebooks remove readout errors from the simulator only;
        # the properties passed to encode_data keep FakeLima's readout errors
        # (h17_compare_over_steps.ipynb cell 3).
        "zero_readout_error": False,
        "test_is_validation": False,
    },
    "readout": {
        "dir_name": "ising_init_from_qasm",
        "train_split": "train",
        "train_steps": list(range(15)),
        "val_split": "val",
        "val_steps": list(range(15)),
        "test_split": "val_Zonly",
        "test_steps": list(range(15)),
        "zero_readout_error": False,
        "test_is_validation": False,
    },
    "coherent": {
        "dir_name": "ising_init_from_qasm_coherent",
        "train_split": "train",
        "train_steps": list(range(15)),
        "val_split": "val",
        "val_steps": list(range(15)),
        "test_split": "val",
        "test_steps": list(range(15)),
        "zero_readout_error": False,
        "test_is_validation": True,
    },
}

# Aliases matching raw directory names
SETTINGS["ising_init_from_qasm_no_readout"] = SETTINGS["no_readout"]
SETTINGS["ising_init_from_qasm"] = SETTINGS["readout"]
SETTINGS["ising_init_from_qasm_coherent"] = SETTINGS["coherent"]


def get_setting_config(setting_name: str) -> Dict[str, Any]:
    """Returns canonical configuration dictionary for a given setting name or alias."""
    if setting_name not in SETTINGS:
        raise ValueError(
            f"Unknown setting: {setting_name!r}. Supported: {list(SETTINGS.keys())}"
        )
    return SETTINGS[setting_name]


def get_lima_properties(zero_readout_error: bool = False) -> Dict[str, Any]:
    """Returns backend properties for FakeLima, tailored to the experimental setting.

    If zero_readout_error is True, readout errors across all qubits are set to 0.0
    matching the setting specification.
    """
    backend = FakeLima()
    props = get_backend_properties_v1(backend)

    if zero_readout_error:
        if "qubits_props" in props:
            for q_data in props["qubits_props"].values():
                if "readout_error" in q_data:
                    q_data["readout_error"] = 0.0

    return props


def load_split_data(
    split_dir: str,
    step_numbers: Optional[List[int]] = None
) -> Tuple[List[Dict[str, Any]], List[int], List[Tuple[str, int]], Dict[str, str]]:
    """Loads entries from step_*.pk files in split_dir.

    Returns:
        entries: list of entry dicts
        steps: list of integer Trotter steps per circuit
        circuit_ids: list of (file_name, index_in_file) per circuit
        files_sha256: dict mapping filename to its SHA-256 hash
    """
    entries = []
    steps = []
    circuit_ids = []
    files_sha256 = {}

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    if step_numbers is not None:
        # Every requested step must exist: a missing file would shrink the frozen split.
        step_files = [f"step_{step}.pk" for step in sorted(set(step_numbers))]
        missing = [name for name in step_files
                   if not os.path.isfile(os.path.join(split_dir, name))]
        if missing:
            raise FileNotFoundError(f"{split_dir}: missing requested step files {missing}")
    else:
        step_files = sorted(
            (name for name in os.listdir(split_dir) if re.fullmatch(r"step_\d+\.pk", name)),
            key=lambda name: int(name[5:-3]),
        )

    for fn in step_files:
        fp = os.path.join(split_dir, fn)
        files_sha256[fn] = compute_file_sha256(fp)
        m = re.search(r"step_(\d+)\.pk", fn)
        step_val = int(m.group(1)) if m else 0
        with open(fp, "rb") as f:
            file_entries = pickle.load(f)
        for idx, entry in enumerate(file_entries):
            entries.append(entry)
            steps.append(step_val)
            circuit_ids.append((fn, idx))

    return entries, steps, circuit_ids, files_sha256


def extract_entry_data(entries: List[Dict[str, Any]]) -> Tuple[List[QuantumCircuit], List[Any], List[Any]]:
    """Extracts circuits, ideal labels, and noisy expectation values from a list of entry dicts."""
    circuits = []
    ideal_exp_vals = []
    noisy_exp_vals = []

    for entry in entries:
        circ = entry.get("circuit")
        if isinstance(circ, str):
            circ = QuantumCircuit.from_qasm_str(circ)
        elif circ is None and "circuit_twirled" in entry:
            circ = entry["circuit_twirled"]
        circuits.append(circ)

        ideal = entry.get("ideal_exp_value")
        if ideal is None:
            ideal = entry.get("ideal_exp_val")
        if ideal is None:
            ideal = entry.get("ideal")
        if isinstance(ideal, np.ndarray):
            ideal = ideal.tolist()
        ideal_exp_vals.append(ideal)

        noisy = entry.get("noisy_exp_values")
        if noisy is None:
            noisy = entry.get("noisy_exp_val")
        if noisy is None:
            noisy = entry.get("noisy")
        if isinstance(noisy, np.ndarray):
            noisy = noisy.tolist()
        noisy_exp_vals.append(noisy)

    clean_noisy = []
    for x in noisy_exp_vals:
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
            clean_noisy.append(x[0])
        else:
            clean_noisy.append(x)

    return circuits, ideal_exp_vals, clean_noisy


def encode_entries(
    entries: List[Dict[str, Any]],
    properties: Optional[Dict[str, Any]] = None,
    num_qubits: Optional[int] = None,
    meas_bases: Optional[List[Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Builds tabular feature matrix and returns noisy column range.

    Args:
        entries: list of entry dicts containing circuit, ideal, noisy values.
        properties: backend properties dict. If None, defaults to FakeLima.
        num_qubits: number of target observables/qubits (default: inferred or 4).
        meas_bases: optional encoded measurement bases.

    Returns:
        (X, y, (noisy_start_col, noisy_end_col))
    """
    if properties is None:
        properties = get_lima_properties(zero_readout_error=False)

    circuits, ideal_exp_vals, noisy_exp_vals = extract_entry_data(entries)

    if num_qubits is None:
        if isinstance(noisy_exp_vals[0], list):
            num_qubits = len(noisy_exp_vals[0])
        elif isinstance(noisy_exp_vals[0], (float, int)):
            num_qubits = 1
        elif isinstance(noisy_exp_vals[0], np.ndarray):
            num_qubits = noisy_exp_vals[0].shape[0]

    len_vec = 8
    len_gates_set = len(properties["gates_set"])
    bin_size = 0.1 * np.pi
    num_angle_bins = int(np.ceil(4 * np.pi / bin_size))  # 40

    noisy_start_col = len_vec + len_gates_set + num_angle_bins
    noisy_end_col = noisy_start_col + num_qubits

    X, y = encode_data(
        circuits=circuits,
        properties=properties,
        ideal_exp_vals=ideal_exp_vals,
        noisy_exp_vals=noisy_exp_vals,
        num_qubits=num_qubits,
        meas_bases=meas_bases
    )

    return X, y, (noisy_start_col, noisy_end_col)


MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_data_sha256.json")


def verify_manifest(dir_name: str, input_shas: Dict[str, str]) -> None:
    """Require every data file read to be listed in the manifest with its SHA-256."""
    with open(MANIFEST, "r") as handle:
        expected = json.load(handle)
    problems = []
    for rel, sha in sorted(input_shas.items()):
        key = f"docs/tutorials/data/{dir_name}/{rel}"
        if key not in expected:
            problems.append(f"{key}: not in {os.path.basename(MANIFEST)}")
        elif expected[key] != sha:
            problems.append(f"{key}: SHA-256 {sha} != {expected[key]}")
    if problems:
        raise SystemExit("data files differ from the manifest; nothing encoded:\n  "
                         + "\n  ".join(problems))


def encode_setting_dataset(
    setting: str,
    data_root: str,
    output_dir: str,
    check_manifest: bool = True,
) -> Tuple[str, str]:
    """Encodes splits for a setting, saving cached NPZ and JSON files under <output-dir>/encoded/.

    Reuses existing encoded cache when present and input file SHA-256 hashes match.

    Returns:
        (encoded_npz_path, encoded_json_path)
    """
    config = get_setting_config(setting)
    dir_name = config["dir_name"]

    # Locate base directory for this setting
    candidates = [
        os.path.join(data_root, dir_name),
        os.path.join(data_root, "data", dir_name),
        os.path.join(data_root, "docs/tutorials/data", dir_name),
    ]
    base_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            base_dir = cand
            break

    if base_dir is None:
        raise FileNotFoundError(
            f"Could not locate dataset directory for setting {setting!r} under {data_root!r}. "
            f"Checked paths: {candidates}"
        )

    encoded_dir = os.path.join(output_dir, "encoded")
    os.makedirs(encoded_dir, exist_ok=True)
    npz_path = os.path.join(encoded_dir, f"{setting}.npz")
    json_path = os.path.join(encoded_dir, f"{setting}.json")

    train_dir = os.path.join(base_dir, config["train_split"])
    val_dir = os.path.join(base_dir, config["val_split"])
    test_dir = os.path.join(base_dir, config["test_split"])

    # Load entries and file hashes
    tr_entries, tr_steps, tr_ids, tr_shas = load_split_data(train_dir, config["train_steps"])
    va_entries, va_steps, va_ids, va_shas = load_split_data(val_dir, config["val_steps"])
    te_entries, te_steps, te_ids, te_shas = load_split_data(test_dir, config["test_steps"])

    all_input_shas = {}
    for fn, h in tr_shas.items():
        all_input_shas[f"{config['train_split']}/{fn}"] = h
    for fn, h in va_shas.items():
        all_input_shas[f"{config['val_split']}/{fn}"] = h
    for fn, h in te_shas.items():
        all_input_shas[f"{config['test_split']}/{fn}"] = h

    if check_manifest:
        verify_manifest(dir_name, all_input_shas)

    # Check cache validity
    if os.path.isfile(npz_path) and os.path.isfile(json_path) and os.path.getsize(npz_path) > 0:
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            if meta.get("input_files_sha256") == all_input_shas:
                # Cache is valid and matches
                return npz_path, json_path
        except Exception:
            pass

    # Encode splits
    props = get_lima_properties(zero_readout_error=config["zero_readout_error"])
    X_tr, y_tr, noisy_range = encode_entries(tr_entries, properties=props, num_qubits=4)
    X_va, y_va, _ = encode_entries(va_entries, properties=props, num_qubits=4)
    X_te, y_te, _ = encode_entries(te_entries, properties=props, num_qubits=4)

    start_c, end_c = noisy_range
    raw_noisy = X_te[:, start_c:end_c].numpy()

    # Save NPZ
    np.savez_compressed(
        npz_path,
        X_train=X_tr.numpy(),
        y_train=y_tr.numpy(),
        train_steps=np.array(tr_steps, dtype=int),
        train_circuit_files=np.array([item[0] for item in tr_ids]),
        train_circuit_indices=np.array([item[1] for item in tr_ids], dtype=int),
        train_circuit_ids=np.array([f"{item[0]}:{item[1]}" for item in tr_ids]),
        X_val=X_va.numpy(),
        y_val=y_va.numpy(),
        val_steps=np.array(va_steps, dtype=int),
        val_circuit_files=np.array([item[0] for item in va_ids]),
        val_circuit_indices=np.array([item[1] for item in va_ids], dtype=int),
        val_circuit_ids=np.array([f"{item[0]}:{item[1]}" for item in va_ids]),
        X_test=X_te.numpy(),
        y_test=y_te.numpy(),
        test_steps=np.array(te_steps, dtype=int),
        test_circuit_files=np.array([item[0] for item in te_ids]),
        test_circuit_indices=np.array([item[1] for item in te_ids], dtype=int),
        test_circuit_ids=np.array([f"{item[0]}:{item[1]}" for item in te_ids]),
        raw_noisy=raw_noisy,
        noisy_range=np.array([start_c, end_c], dtype=int)
    )

    # Save JSON metadata
    metadata = {
        "setting": setting,
        "dir_name": dir_name,
        "zero_readout_error": config["zero_readout_error"],
        "test_is_validation": config["test_is_validation"],
        "train_samples": int(X_tr.shape[0]),
        "val_samples": int(X_va.shape[0]),
        "test_samples": int(X_te.shape[0]),
        "feature_dim": int(X_tr.shape[1]),
        "noisy_range": [int(start_c), int(end_c)],
        "input_files_sha256": all_input_shas,
        "manifest_checked": bool(check_manifest),
        "npz_file": os.path.basename(npz_path)
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, json_path
