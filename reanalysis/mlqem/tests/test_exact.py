"""Tests for exact labels and parameter recovery in ML-QEM reanalysis.

Governed by docs/frozen-rules/2026-10-06-round9-follow-ups.md.
Tests exact_labels, cal_z_exp agreement, parameter recovery, and default path invariance.
Runs under the mlqem python environment.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import pytest

# These tests need the pinned ML-QEM environment (qiskit 0.43 and blackwater).
pytest.importorskip("qiskit")
pytest.importorskip("blackwater")

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeLima
from qiskit.quantum_info import Statevector

from reanalysis.mlqem.exact import (
    exact_labels,
    circuit_parameters,
    extract_basis,
    BASIS_MAP,
    evaluate_gate,
    check_gate_approval,
    REQUIRED_GATE_SPLITS,
)
from reanalysis.mlqem.arms import prepare_arm_features, run_arm_job
from reanalysis.mlqem import analyze as an
from reanalysis.mlqem.run import job_worker


class IsingModel:
    """Ising model generator mirroring the upstream notebook implementation."""

    class Options(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self["nq"] = 4
            self["h"] = 1.0
            self["J"] = 0.15
            self["dt"] = 0.5
            self["depth"] = 15
            self.update(*args, **kwargs)

        def config_4q_paper(self):
            self["h"] = 1.0
            self["J"] = 0.15
            self["dt"] = 0.5

    @classmethod
    def apply_quantum_circuit_layer(cls, qc: QuantumCircuit, ops: "IsingModel.Options"):
        allq = list(range(ops["nq"]))
        qc.rx(2.0 * ops["h"] * ops["dt"], allq)
        qc.barrier()
        for q0 in allq[0::2]:
            q1 = q0 + 1
            qc.cx(q0, q1)
        qc.rz(-2.0 * ops["J"] * ops["dt"], allq[1::2])
        for q0 in allq[0::2]:
            q1 = q0 + 1
            qc.cx(q0, q1)
        qc.barrier()
        for q0 in allq[1:-2:2]:
            q1 = q0 + 1
            qc.cx(q0, q1)
        qc.rz(-2.0 * ops["J"] * ops["dt"], allq[2:-1:2])
        for q0 in allq[1:-2:2]:
            q1 = q0 + 1
            qc.cx(q0, q1)
        qc.barrier()

    @classmethod
    def make_quantum_circuit(cls, ops: "IsingModel.Options") -> QuantumCircuit:
        qc = QuantumCircuit(ops["nq"])
        for _ in range(ops["depth"]):
            cls.apply_quantum_circuit_layer(qc, ops)
        if ops["measure_basis"] == "Z":
            pass
        elif ops["measure_basis"] == "X":
            qc.h(list(range(ops["nq"])))
        elif ops["measure_basis"] == "Y":
            qc.sdg(list(range(ops["nq"])))
            qc.h(list(range(ops["nq"])))
        else:
            raise ValueError("Basis must be X, Y, or Z")
        qc.measure_all()
        return qc

    @classmethod
    def make_circs_sweep(cls, ops: "IsingModel.Options", num_steps: int, measure_basis: str) -> QuantumCircuit:
        ops["measure_basis"] = measure_basis
        ops["depth"] = num_steps
        return cls.make_quantum_circuit(ops)


def construct_ising_circuit_random_init(j_val: float, basis: str, num_steps: int) -> QuantumCircuit:
    """Builds Trotter circuit matching upstream generator notebook cell 6."""
    cls = IsingModel
    ops = cls.Options()
    ops.config_4q_paper()
    ops.update({"J": j_val})
    qc_init = QuantumCircuit.from_qasm_str(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\n'
        "rz(0.0007186381718527407) q[1];\nrz(2.4917901988569855) q[1];\n"
        "rz(3.3854853863523835) q[3];\nrx(1.2846113715328817) q[3];\n"
        "cx q[3],q[0];\nrx(4.212671608894216) q[2];\ncx q[2],q[3];\n"
    )
    qc_init.barrier()
    circ = qc_init.compose(cls.make_circs_sweep(ops, num_steps, basis), list(range(4)))
    return circ


def cal_z_exp(counts: dict) -> np.ndarray:
    """Computes sigma_z expectations mirroring mbd_utils.py exactly."""
    shots = sum(list(counts.values()))
    num_qubits = len(list(counts.keys())[0])
    count_pos_z = np.zeros(num_qubits)
    for key, val in counts.items():
        count_pos_z += val * np.array(list(key), dtype=int)
    count_neg_z = np.ones(num_qubits) * shots - count_pos_z
    return (count_pos_z - count_neg_z) / shots


def test_exact_labels_hand_built():
    """Verifies exact_labels on hand-built circuits with known outcomes."""
    qc_all_zero = QuantumCircuit(4, 4)
    qc_all_zero.measure(range(4), range(4))
    res_zero = exact_labels(qc_all_zero)
    np.testing.assert_allclose(res_zero, [-1.0, -1.0, -1.0, -1.0])

    qc_x0 = QuantumCircuit(4, 4)
    qc_x0.x(0)
    qc_x0.measure(range(4), range(4))
    res_x0 = exact_labels(qc_x0)
    np.testing.assert_allclose(res_x0, [-1.0, -1.0, -1.0, 1.0])

    qc_x2 = QuantumCircuit(4, 4)
    qc_x2.x(2)
    qc_x2.measure(range(4), range(4))
    res_x2 = exact_labels(qc_x2)
    np.testing.assert_allclose(res_x2, [-1.0, 1.0, -1.0, -1.0])

    qc_all_one = QuantumCircuit(4, 4)
    qc_all_one.x(range(4))
    qc_all_one.measure(range(4), range(4))
    res_one = exact_labels(qc_all_one)
    np.testing.assert_allclose(res_one, [1.0, 1.0, 1.0, 1.0])

    qc_perm = QuantumCircuit(4, 4)
    qc_perm.x(3)
    qc_perm.measure(3, 0)
    qc_perm.measure(2, 1)
    qc_perm.measure(1, 2)
    qc_perm.measure(0, 3)
    res_perm = exact_labels(qc_perm)
    np.testing.assert_allclose(res_perm, [-1.0, -1.0, -1.0, 1.0])


def test_agreement_with_cal_z_exp():
    """Verifies agreement between exact_labels and cal_z_exp at 10^6 shots."""
    qc = QuantumCircuit(4, 4)
    qc.ry(0.3, 0)
    qc.ry(1.2, 1)
    qc.ry(2.1, 2)
    qc.ry(0.8, 3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.measure(range(4), range(4))

    exact = exact_labels(qc)

    bare = qc.remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(bare).probabilities()
    n_shots = 1000000
    rng = np.random.default_rng(20261006)
    sampled_indices = rng.choice(len(probs), size=n_shots, p=probs)
    counts = {}
    for idx in sampled_indices:
        bstr = format(idx, "04b")
        counts[bstr] = counts.get(bstr, 0) + 1

    simulated = cal_z_exp(counts)
    max_diff = np.max(np.abs(simulated - exact))
    assert max_diff < 0.01, f"Difference {max_diff:.4f} exceeds 0.01 tolerance."


def test_recovery_of_j_and_basis():
    """Recovers J to 1e-9 and measurement basis on generator circuits."""
    backend = FakeLima()
    j_values = [0.05, 0.5, 0.95]
    steps_list = [0, 1, 3, 7]
    bases = ["X", "Y", "Z"]

    for steps in steps_list:
        for basis in bases:
            for j_val in j_values:
                raw_circ = construct_ising_circuit_random_init(j_val, basis, steps)
                trans_circ = transpile(raw_circ, backend=backend, optimization_level=3)
                params = circuit_parameters(trans_circ, steps)

                assert params["steps"] == steps
                assert params["basis"] == basis
                assert params["n_sandwiches"] == 3 * steps
                if steps == 0:
                    assert np.isnan(params["J"])
                else:
                    assert abs(params["J"] - j_val) < 1e-9


def test_default_path_invariance():
    """Verifies that default paths in run.py and analyze.py produce invariant outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Temporary test rule\n")

        n_train, n_val, n_test = 20, 10, 10
        rng = np.random.default_rng(12345)
        x_tr = rng.normal(0, 1, size=(n_train, 58))
        y_tr = rng.uniform(-1, 1, size=(n_train, 4))
        x_va = rng.normal(0, 1, size=(n_val, 58))
        y_va = rng.uniform(-1, 1, size=(n_val, 4))
        x_te = rng.normal(0, 1, size=(n_test, 58))
        y_te = rng.uniform(-1, 1, size=(n_test, 4))

        meta_default = run_arm_job(
            setting="no_readout",
            model="ols",
            arm="F",
            learner_seed=1,
            X_train=x_tr,
            y_train=y_tr,
            X_test=x_te,
            y_test=y_te,
            X_val=x_va,
            y_val=y_va,
            noisy_range=(54, 58),
            output_dir=tmpdir,
            frozen_rule_path=rule_path,
            descriptors="encoding",
        )

        assert "descriptors" not in meta_default
        assert "exact_npz_sha256" not in meta_default
        assert meta_default["feature_dim"] == 58
        assert meta_default["feature_column_count"] == 58


def write_synthetic_fit(
    fits_dir: str,
    setting: str,
    model: str,
    arm: str,
    seed: int,
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    steps: np.ndarray = None,
    ids: np.ndarray = None,
    circuit_files: np.ndarray = None,
    circuit_indices: np.ndarray = None,
    rule_sha: str = "a" * 64,
    test_is_validation: bool = False,
    train_targets: str = "archived",
    exact_npz_sha256: dict = None,
):
    """Writes synthetic fit files for testing analyze."""
    n = len(targets)
    if steps is None:
        steps = np.zeros(n, dtype=int)
    if ids is None:
        ids = np.array([f"step_0.pk:{i}" for i in range(n)])
    base = os.path.join(fits_dir, f"{setting}_{model}_{arm}_seed{seed}")
    savez_dict = {
        "predictions": preds,
        "targets": targets,
        "raw_noisy": targets + 0.1,
        "test_circuit_ids": ids,
        "test_steps": steps,
    }
    if circuit_files is not None:
        savez_dict["test_circuit_files"] = circuit_files
    if circuit_indices is not None:
        savez_dict["test_circuit_indices"] = circuit_indices
    np.savez_compressed(base + ".npz", **savez_dict)
    meta = {
        "setting": setting,
        "model": model,
        "arm": arm,
        "learner_seed": seed,
        "frozen_rule_sha256": rule_sha,
        "data_files_sha256": {"train/step_0.pk": "b" * 64, "val_extra/step_0.pk": "c" * 64},
        "test_is_validation": test_is_validation,
        "train_targets": train_targets,
    }
    if exact_npz_sha256 is not None:
        meta["exact_npz_sha256"] = exact_npz_sha256
    with open(base + ".json", "w") as handle:
        json.dump(meta, handle)


def write_synthetic_exact_npz(
    exact_dir: str,
    setting: str,
    split: str,
    archived: np.ndarray,
    exact: np.ndarray,
    steps: np.ndarray = None,
    step_file: np.ndarray = None,
    entry_index: np.ndarray = None,
):
    """Writes synthetic exact npz file."""
    n = len(archived)
    os.makedirs(exact_dir, exist_ok=True)
    out_path = os.path.join(exact_dir, f"exact-{setting}-{split}.npz")
    if steps is None:
        steps = np.zeros(n, dtype=int)
    if step_file is None:
        step_file = np.array(["step_0.pk"] * n)
    if entry_index is None:
        entry_index = np.arange(n)
    np.savez_compressed(
        out_path,
        archived=archived,
        exact=exact,
        J=np.full(n, 0.5),
        basis=np.zeros(n, dtype=int),
        steps=steps,
        step_file=step_file,
        entry_index=entry_index,
    )
    return out_path


def write_synthetic_gate_json(
    exact_dir: str,
    npz_paths: list,
    overall_pass: bool = True,
    rule_sha: str = "a" * 64,
):
    """Writes synthetic gate.json file."""
    splits = {}
    for p in npz_paths:
        base = os.path.basename(p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        splits[base] = {
            "npz_file": base,
            "npz_sha256": h,
            "manifest_file": f"manifest-{base}.json",
            "manifest_sha256": "0" * 64,
            "split_passed": overall_pass,
            "pass": overall_pass,
            "failures": [] if overall_pass else ["simulated failure"],
        }
    gate_path = os.path.join(exact_dir, "gate.json")
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gate_version": 1,
                "gate_passed": overall_pass,
                "overall_pass": overall_pass,
                "frozen_rule_sha256": rule_sha,
                "splits": splits,
            },
            f,
            indent=2,
        )
    return gate_path


def test_prepare_arm_features_exact_descriptors():
    """Verifies feature preparation when using exact descriptors."""
    n_train, n_val, n_test = 20, 10, 10
    rng = np.random.default_rng(99)
    x_tr = rng.normal(0, 1, size=(n_train, 63))
    x_va = rng.normal(0, 1, size=(n_val, 63))
    x_te = rng.normal(0, 1, size=(n_test, 63))
    noisy_range = (59, 63)

    x_tr_f, x_te_f, x_va_f, cols_f = prepare_arm_features(
        x_tr, x_te, x_va, noisy_range, arm="F", learner_seed=0
    )
    assert x_tr_f.shape[1] == 63
    assert len(cols_f) == 63

    x_tr_c, x_te_c, x_va_c, cols_c = prepare_arm_features(
        x_tr, x_te, x_va, noisy_range, arm="C", learner_seed=0
    )
    assert x_tr_c.shape[1] == 59
    assert len(cols_c) == 59
    assert all(c < 59 for c in cols_c)

    x_tr_p, x_te_p, x_va_p, cols_p = prepare_arm_features(
        x_tr, x_te, x_va, noisy_range, arm="P", learner_seed=1
    )
    assert x_tr_p.shape[1] == 63
    np.testing.assert_allclose(x_tr_p[:, :59], x_tr[:, :59])
    assert not np.allclose(x_tr_p[:, 59:], x_tr[:, 59:])
    np.testing.assert_allclose(x_te_p, x_te)
    np.testing.assert_allclose(x_va_p, x_va)

    for r_arm in ("R", "Rcal"):
        x_tr_r, x_te_r, x_va_r, cols_r = prepare_arm_features(
            x_tr, x_te, x_va, noisy_range, arm=r_arm, learner_seed=0
        )
        assert x_tr_r.shape[1] == 4
        assert cols_r == [59, 60, 61, 62]


def test_exact_descriptors_run_arm_job():
    """Verifies that run_arm_job records exact descriptors metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")

        n_train, n_val, n_test = 20, 10, 10
        rng = np.random.default_rng(123)
        x_tr = rng.normal(0, 1, size=(n_train, 63))
        y_tr = rng.uniform(-1, 1, size=(n_train, 4))
        x_va = rng.normal(0, 1, size=(n_val, 63))
        y_va = rng.uniform(-1, 1, size=(n_val, 4))
        x_te = rng.normal(0, 1, size=(n_test, 63))
        y_te = rng.uniform(-1, 1, size=(n_test, 4))

        meta = run_arm_job(
            setting="no_readout",
            model="ols",
            arm="F",
            learner_seed=1,
            X_train=x_tr,
            y_train=y_tr,
            X_test=x_te,
            y_test=y_te,
            X_val=x_va,
            y_val=y_va,
            noisy_range=(59, 63),
            output_dir=tmpdir,
            frozen_rule_path=rule_path,
            descriptors="exact",
            exact_npz_sha256="dummy_exact_sha",
        )

        assert meta["descriptors"] == "exact"
        assert meta["exact_npz_sha256"] == "dummy_exact_sha"
        assert meta["feature_dim"] == 63
        assert meta["feature_column_count"] == 63


def test_analyze_default_path_invariance():
    """Verifies that analyze default output is identical with or without explicit targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(42)
        targets = rng.uniform(-1.0, 1.0, size=(n, 4))

        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    targets + 0.05,
                    targets,
                    rule_sha=rule_sha,
                )

        out_default = os.path.join(tmpdir, "out_default.json")
        out_archived = os.path.join(tmpdir, "out_archived.json")
        out_report_c = os.path.join(tmpdir, "out_report_c.json")

        cmd_base = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
        ]

        subprocess.check_call(cmd_base + ["--out", out_default])
        subprocess.check_call(cmd_base + ["--targets", "archived", "--out", out_archived])
        subprocess.check_call(cmd_base + ["--report-c-minus-r", "--out", out_report_c])

        with open(out_default) as f:
            data_def = json.load(f)
        with open(out_archived) as f:
            data_arc = json.load(f)
        with open(out_report_c) as f:
            data_c = json.load(f)

        assert data_def == data_arc
        quantities = data_def["results"]["no_readout"]["ols"]["quantities"]
        assert "C_minus_R" not in quantities
        assert "reference_error" not in quantities

        assert "C_minus_R" in data_c["results"]["no_readout"]["ols"]["quantities"]


def test_analyze_exact_rescoring():
    """Verifies that analyze scores against exact targets and computes extra metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(42)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived, exact)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    archived + 0.05,
                    archived,
                    rule_sha=rule_sha,
                )

        out_path = os.path.join(tmpdir, "out_exact.json")
        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            out_path,
        ]
        subprocess.check_call(cmd)

        with open(out_path) as f:
            data = json.load(f)

        cell = data["results"]["no_readout"]["ols"]
        assert "C_minus_R" in cell["quantities"]
        assert "F_minus_R" in cell["quantities"]
        assert "reference_error" in cell
        assert abs(cell["reference_error"] - 0.01) < 1e-6
        assert "reproduction_check" not in data
        assert data.get("reproduction_check_source") == "artifacts/mlqem-own-data/analysis.json"


def test_analyze_exact_target_mismatch_raises():
    """Verifies that analyze raises ValueError when fit targets disagree with archived array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(42)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived, exact)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        mismatched_targets = archived + 0.5
        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    mismatched_targets,
                    mismatched_targets,
                    rule_sha=rule_sha,
                )

        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            os.path.join(tmpdir, "out.json"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "fit targets do not match archived array" in res.stderr


def test_analyze_coherent_test_is_validation():
    """Verifies that coherent setting cells mark test_is_validation True in both target modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(42)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        p_npz = write_synthetic_exact_npz(exact_dir, "coherent", "val", archived, exact)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        for arm in ("F", "C", "P", "R", "Rcal"):
            write_synthetic_fit(
                fits_dir,
                "coherent",
                "ols",
                arm,
                1,
                archived + 0.05,
                archived,
                rule_sha=rule_sha,
                test_is_validation=True,
            )

        cmd_base = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "coherent",
            "--models",
            "ols",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
        ]

        out_archived = os.path.join(tmpdir, "out_archived.json")
        subprocess.check_call(cmd_base + ["--targets", "archived", "--out", out_archived])
        with open(out_archived) as f:
            data_arc = json.load(f)
        assert data_arc["results"]["coherent"]["ols"]["test_is_validation"] is True

        out_exact = os.path.join(tmpdir, "out_exact.json")
        subprocess.check_call(
            cmd_base + ["--targets", "exact", "--exact-dir", exact_dir, "--out", out_exact]
        )
        with open(out_exact) as f:
            data_exc = json.load(f)
        assert data_exc["results"]["coherent"]["ols"]["test_is_validation"] is True


def test_analyze_float32_targets_binding():
    """Verifies that float32 fit targets bind against float64 archive, keeping float64 for reference_error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(1234)
        archived_f64 = rng.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float64)
        exact_f64 = archived_f64 + 0.02
        fit_targets_f32 = archived_f64.astype(np.float32)

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived_f64, exact_f64)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    fit_targets_f32 + 0.05,
                    fit_targets_f32,
                    rule_sha=rule_sha,
                )

        out_path = os.path.join(tmpdir, "out_f32.json")
        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            out_path,
        ]
        subprocess.check_call(cmd)
        with open(out_path) as f:
            data = json.load(f)
        cell = data["results"]["no_readout"]["ols"]
        assert abs(cell["reference_error"] - 0.02) < 1e-6


def test_analyze_reordered_identifiers_fails():
    """Verifies that reordered circuit steps or circuit IDs raise ValueError in analyze."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(5678)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        steps = np.arange(n, dtype=int)
        p_npz = write_synthetic_exact_npz(
            exact_dir, "no_readout", "val_extra", archived, exact, steps=steps
        )
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        # Write fits with reversed steps
        reordered_steps = steps[::-1]
        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    archived + 0.05,
                    archived,
                    steps=reordered_steps,
                    rule_sha=rule_sha,
                )

        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            os.path.join(tmpdir, "out.json"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "test steps do not match" in res.stderr


def test_analyze_two_rules_fit_frozen_rule():
    """Verifies --fit-frozen-rule binds using fit rule hash and records both hashes in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_fit_path = os.path.join(tmpdir, "rule_fit.md")
        with open(rule_fit_path, "w") as f:
            f.write("# Rule Fit Version\n")
        rule_fit_sha = hashlib.sha256(open(rule_fit_path, "rb").read()).hexdigest()

        rule_analysis_path = os.path.join(tmpdir, "rule_analysis.md")
        with open(rule_analysis_path, "w") as f:
            f.write("# Rule Analysis Version (distinct)\n")
        rule_analysis_sha = hashlib.sha256(open(rule_analysis_path, "rb").read()).hexdigest()
        assert rule_fit_sha != rule_analysis_sha

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(999)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived, exact)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_analysis_sha)

        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    archived + 0.05,
                    archived,
                    rule_sha=rule_fit_sha,
                )

        # 1. Running with only --frozen-rule rule_analysis_path must fail because fits use rule_fit_sha
        cmd_fail = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_analysis_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            os.path.join(tmpdir, "out_fail.json"),
        ]
        res_fail = subprocess.run(cmd_fail, capture_output=True, text=True)
        assert res_fail.returncode != 0
        assert "rule SHA-256 differs from the frozen rule" in res_fail.stderr

        # 2. Running with --frozen-rule rule_analysis_path --fit-frozen-rule rule_fit_path must succeed
        out_two_rules = os.path.join(tmpdir, "out_two_rules.json")
        cmd_pass = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_analysis_path,
            "--fit-frozen-rule",
            rule_fit_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            out_two_rules,
        ]
        subprocess.check_call(cmd_pass)
        with open(out_two_rules) as f:
            data = json.load(f)
        assert data["frozen_rule_sha256"] == rule_analysis_sha
        assert data["fit_frozen_rule_sha256"] == rule_fit_sha


def make_synthetic_8_splits(exact_dir: str, val_override: dict = None, drop_split: tuple = None):
    """Creates a full suite of 8 synthetic split NPZs and manifests for gate testing."""
    os.makedirs(exact_dir, exist_ok=True)
    for setting, split in REQUIRED_GATE_SPLITS:
        if drop_split and (setting, split) == drop_split:
            continue
        n = 10
        npz_name = f"exact-{setting}-{split}.npz"
        npz_path = os.path.join(exact_dir, npz_name)
        np.savez_compressed(
            npz_path,
            archived=np.zeros((n, 4)),
            exact=np.zeros((n, 4)),
            J=np.full(n, 0.5),
            basis=np.zeros(n, dtype=int),
            steps=np.zeros(n, dtype=int),
            step_file=np.array(["step_0.pk"] * n),
            entry_index=np.arange(n),
        )
        h = hashlib.sha256(open(npz_path, "rb").read()).hexdigest()
        val = {
            "z_mean": 0.01,
            "z_sd": 1.0,
            "z_max_abs": 3.0,
            "max_abs_z": 3.0,
            "z_gt_4_count": 0,
            "z_gt_6_count": 0,
            "exact_ones_count": 0,
            "exact_ones_mismatch_count": 0,
        }
        if val_override and (setting, split) in val_override:
            val.update(val_override[(setting, split)])
        manifest = {
            "setting": setting,
            "split": split,
            "output_npz": npz_name,
            "output_npz_sha256": h,
            "npz_file": npz_name,
            "npz_sha256": h,
            "circuit_count": n,
            "validation": val,
        }
        with open(os.path.join(exact_dir, f"exact-{setting}-{split}.json"), "w") as f:
            json.dump(manifest, f, indent=2)


def test_gate_evaluation_thresholds_and_missing_split():
    """Verifies that gate evaluates all 8 splits, enforces thresholds, and flags missing splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")

        # Case 1: All 8 splits pass
        dir_pass = os.path.join(tmpdir, "exact_pass")
        make_synthetic_8_splits(dir_pass)
        res_pass = evaluate_gate(dir_pass, rule_path)
        assert res_pass["overall_pass"] is True
        assert res_pass["gate_passed"] is True
        assert os.path.isfile(os.path.join(dir_pass, "gate.json"))

        # Test CLI invocation on passing directory
        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.exact",
            "gate",
            "--exact-dir",
            dir_pass,
            "--frozen-rule",
            rule_path,
        ]
        subprocess.check_call(cmd)

        # Case 2: |mean z| > 0.1 fails
        dir_fail_mean = os.path.join(tmpdir, "exact_fail_mean")
        make_synthetic_8_splits(
            dir_fail_mean, val_override={("no_readout", "train"): {"z_mean": 0.25}}
        )
        res_fail_mean = evaluate_gate(dir_fail_mean, rule_path)
        assert res_fail_mean["overall_pass"] is False
        assert "no_readout:train" in res_fail_mean["splits"]
        assert res_fail_mean["splits"]["no_readout:train"]["split_passed"] is False

        # Case 3: z_sd out of [0.9, 1.1] fails
        dir_fail_sd = os.path.join(tmpdir, "exact_fail_sd")
        make_synthetic_8_splits(
            dir_fail_sd, val_override={("readout", "val"): {"z_sd": 1.35}}
        )
        res_fail_sd = evaluate_gate(dir_fail_sd, rule_path)
        assert res_fail_sd["overall_pass"] is False

        # Case 4: |z| > 6 fails
        dir_fail_max = os.path.join(tmpdir, "exact_fail_max")
        make_synthetic_8_splits(
            dir_fail_max, val_override={("coherent", "train"): {"z_gt_6_count": 1}}
        )
        res_fail_max = evaluate_gate(dir_fail_max, rule_path)
        assert res_fail_max["overall_pass"] is False

        # Case 5: boundary mismatch fails
        dir_fail_bound = os.path.join(tmpdir, "exact_fail_bound")
        make_synthetic_8_splits(
            dir_fail_bound, val_override={("coherent", "val"): {"exact_ones_mismatch_count": 2}}
        )
        res_fail_bound = evaluate_gate(dir_fail_bound, rule_path)
        assert res_fail_bound["overall_pass"] is False

        # Case 6: missing split fails
        dir_missing = os.path.join(tmpdir, "exact_missing")
        make_synthetic_8_splits(dir_missing, drop_split=("coherent", "val"))
        res_missing = evaluate_gate(dir_missing, rule_path)
        assert res_missing["overall_pass"] is False


def test_gate_blocks_analyze_and_run():
    """Verifies that analyze --targets exact blocks when gate.json is missing, failed, or hash mismatched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)
        os.makedirs(exact_dir)

        n = 20
        rng = np.random.default_rng(77)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.01

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived, exact)

        for model in ("ols", "rf"):
            for arm in ("F", "C", "P", "R", "Rcal"):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    model,
                    arm,
                    1,
                    archived + 0.05,
                    archived,
                    rule_sha=rule_sha,
                )

        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--settings",
            "no_readout",
            "--models",
            "ols",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "exact",
            "--exact-dir",
            exact_dir,
            "--out",
            os.path.join(tmpdir, "out.json"),
        ]

        # 1. Gate missing -> FileNotFoundError
        res1 = subprocess.run(cmd, capture_output=True, text=True)
        assert res1.returncode != 0
        assert "Gate file not found" in res1.stderr

        # 2. Gate overall_pass is False -> ValueError
        write_synthetic_gate_json(exact_dir, [p_npz], overall_pass=False, rule_sha=rule_sha)
        res2 = subprocess.run(cmd, capture_output=True, text=True)
        assert res2.returncode != 0
        assert "gate check failed" in res2.stderr

        # 3. Gate NPZ hash mismatch -> ValueError
        with open(os.path.join(exact_dir, "gate.json"), "w") as f:
            json.dump({
                "gate_passed": True,
                "overall_pass": True,
                "splits": {
                    "val_extra": {
                        "npz_file": os.path.basename(p_npz),
                        "npz_sha256": "f" * 64,
                    }
                }
            }, f)
        res3 = subprocess.run(cmd, capture_output=True, text=True)
        assert res3.returncode != 0
        assert "does not match gate.json recorded SHA-256" in res3.stderr


def test_default_fit_record_schema_matches_head(tmp_path):
    """Verifies that default fit record metadata key set equals HEAD's (N3)."""
    from reanalysis.mlqem.arms import run_arm_job

    head_keys = {
        "setting",
        "model",
        "arm",
        "learner_seed",
        "num_train_samples",
        "num_val_samples",
        "num_test_samples",
        "feature_dim",
        "feature_column_count",
        "feature_columns",
        "feature_names_hash",
        "test_is_validation",
        "frozen_rule_path",
        "frozen_rule_sha256",
        "data_files_sha256",
        "package_versions",
        "wall_time_seconds",
        "npz_file",
    }

    rule_p = tmp_path / "rule.md"
    rule_p.write_text("# Rule\n")
    out_dir = tmp_path / "fits"
    out_dir.mkdir()

    n_tr, n_va, n_te = 10, 5, 5
    rng = np.random.default_rng(42)
    x_tr = rng.normal(size=(n_tr, 58))
    y_tr = rng.uniform(-1, 1, size=(n_tr, 4))
    x_va = rng.normal(size=(n_va, 58))
    y_va = rng.uniform(-1, 1, size=(n_va, 4))
    x_te = rng.normal(size=(n_te, 58))
    y_te = rng.uniform(-1, 1, size=(n_te, 4))
    test_ids = [(f"step_0.pk", i) for i in range(n_te)]

    # 1. Default run_arm_job: descriptors='encoding', train_targets='archived'
    meta_def = run_arm_job(
        setting="no_readout",
        model="ols",
        arm="F",
        learner_seed=1,
        output_dir=str(out_dir),
        X_train=x_tr,
        y_train=y_tr,
        X_val=x_va,
        y_val=y_va,
        X_test=x_te,
        y_test=y_te,
        noisy_range=(54, 58),
        test_ids=test_ids,
        test_steps=[0] * n_te,
        frozen_rule_path=str(rule_p),
    )
    assert set(meta_def.keys()) == head_keys
    with open(os.path.join(str(out_dir), "no_readout_ols_F_seed1.json")) as f:
        disk_meta = json.load(f)
    assert set(disk_meta.keys()) == head_keys

    # 2. Exact descriptors adds descriptors and train_targets
    meta_exact_desc = run_arm_job(
        setting="no_readout",
        model="ols",
        arm="F",
        learner_seed=2,
        output_dir=str(out_dir),
        X_train=x_tr,
        y_train=y_tr,
        X_val=x_va,
        y_val=y_va,
        X_test=x_te,
        y_test=y_te,
        noisy_range=(54, 58),
        test_ids=test_ids,
        test_steps=[0] * n_te,
        frozen_rule_path=str(rule_p),
        descriptors="exact",
    )
    assert set(meta_exact_desc.keys()) == head_keys | {"descriptors", "train_targets"}

    # 3. Exact train_targets adds train_targets
    meta_exact_tt = run_arm_job(
        setting="no_readout",
        model="ols",
        arm="F",
        learner_seed=3,
        output_dir=str(out_dir),
        X_train=x_tr,
        y_train=y_tr,
        X_val=x_va,
        y_val=y_va,
        X_test=x_te,
        y_test=y_te,
        noisy_range=(54, 58),
        test_ids=test_ids,
        test_steps=[0] * n_te,
        frozen_rule_path=str(rule_p),
        train_targets="exact",
    )
    assert set(meta_exact_tt.keys()) == head_keys | {"train_targets"}


def test_analyze_fc_only_scoring():
    """Verifies --analysis-arms F C scores with only F and C fits under exact and archived targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        exact_dir = os.path.join(tmpdir, "exact")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(123)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))
        exact = archived + 0.03

        p_npz = write_synthetic_exact_npz(exact_dir, "no_readout", "val_extra", archived, exact)
        write_synthetic_gate_json(exact_dir, [p_npz], rule_sha=rule_sha)

        # Write ONLY F and C fits (no R, Rcal, P)
        for arm in ("F", "C"):
            for seed in (1, 2):
                write_synthetic_fit(
                    fits_dir,
                    "no_readout",
                    "rf",
                    arm,
                    seed,
                    archived + 0.02,
                    archived,
                    rule_sha=rule_sha,
                    train_targets="exact",
                )

        cmd_base = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--analysis-arms",
            "F",
            "C",
            "--settings",
            "no_readout",
            "--models",
            "rf",
            "--seeds",
            "1",
            "2",
            "--draws",
            "100",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
        ]

        # 1. Archived targets
        out_arc = os.path.join(tmpdir, "out_arc.json")
        subprocess.check_call(cmd_base + ["--targets", "archived", "--out", out_arc])
        with open(out_arc) as f:
            data_arc = json.load(f)
        assert data_arc["train_targets"] == "exact"
        assert data_arc["analysis_arms"] == ["F", "C"]
        assert data_arc["reproduction_check_source"] == "artifacts/mlqem-own-data/analysis.json"
        cell_arc = data_arc["results"]["no_readout"]["rf"]
        assert set(cell_arc["quantities"].keys()) == {"C", "F", "D", "D_over_C"}
        assert "label" in cell_arc["classification"]
        assert "reference_error" not in cell_arc

        # 2. Exact targets
        out_exc = os.path.join(tmpdir, "out_exc.json")
        subprocess.check_call(
            cmd_base + ["--targets", "exact", "--exact-dir", exact_dir, "--out", out_exc]
        )
        with open(out_exc) as f:
            data_exc = json.load(f)
        assert data_exc["train_targets"] == "exact"
        cell_exc = data_exc["results"]["no_readout"]["rf"]
        assert set(cell_exc["quantities"].keys()) == {"C", "F", "D", "D_over_C"}
        assert abs(cell_exc["reference_error"] - 0.03) < 1e-6


def test_analyze_fc_mixed_train_targets_rejected():
    """Verifies that analyze rejects mixed train_targets metadata in the roster."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "rule.md")
        with open(rule_path, "w") as f:
            f.write("# Rule\n")
        rule_sha = hashlib.sha256(open(rule_path, "rb").read()).hexdigest()

        fits_dir = os.path.join(tmpdir, "fits")
        os.makedirs(fits_dir)

        n = 20
        rng = np.random.default_rng(456)
        archived = rng.uniform(-1.0, 1.0, size=(n, 4))

        # F has train_targets="exact", C has train_targets="archived"
        write_synthetic_fit(
            fits_dir,
            "no_readout",
            "rf",
            "F",
            1,
            archived + 0.02,
            archived,
            rule_sha=rule_sha,
            train_targets="exact",
        )
        write_synthetic_fit(
            fits_dir,
            "no_readout",
            "rf",
            "C",
            1,
            archived + 0.02,
            archived,
            rule_sha=rule_sha,
            train_targets="archived",
        )

        cmd = [
            sys.executable,
            "-m",
            "reanalysis.mlqem.analyze",
            "--fits",
            fits_dir,
            "--analysis-arms",
            "F",
            "C",
            "--settings",
            "no_readout",
            "--models",
            "rf",
            "--seeds",
            "1",
            "--draws",
            "50",
            "--frozen-rule",
            rule_path,
            "--no-manifest-check",
            "--targets",
            "archived",
            "--out",
            os.path.join(tmpdir, "out.json"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "mixed train_targets in roster" in res.stderr


