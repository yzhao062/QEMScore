"""Unit tests on throwaway circuits for ML-QEM reanalysis pipeline.

HARD CONSTRAINT: Runs strictly on throwaway synthetic circuits (at most 40, seed 8888).
Never fits on published ML-QEM data.

Runnable with mlqem environment python (qiskit, torch, sklearn).
"""

import os
import json
import tempfile
import numpy as np
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

try:
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile
    from qiskit.providers.fake_provider import FakeLima
    from blackwater.data.utils import get_backend_properties_v1
    from blackwater.library.learning.mlp import encode_data
    from reanalysis.mlqem.encode import encode_entries, get_lima_properties
    from reanalysis.mlqem.arms import (
        prepare_arm_features,
        fit_and_predict_ols,
        fit_and_predict_rf,
        fit_and_predict_mlp,
        run_arm_job,
    )
    from reanalysis.mlqem.run import job_worker, enforce_frozen_rule
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        f"mlqem environment (Qiskit 0.43/blackwater) required for throwaway circuit tests: {exc}",
        allow_module_level=True,
    )


def generate_throwaway_dataset(n_circuits: int = 40, seed: int = 8888):
    """Generates at most 40 throwaway circuits using seed 8888 (not used in paper)."""
    assert n_circuits <= 40, "Hard constraint: at most 40 throwaway circuits allowed."
    np.random.seed(seed)
    torch.manual_seed(seed)

    backend = FakeLima()
    properties = get_backend_properties_v1(backend)

    entries = []
    for i in range(n_circuits):
        qc = random_circuit(4, depth=3, max_operands=2, seed=seed + i)
        qc_trans = transpile(qc, backend=backend, optimization_level=0)

        ideal = np.random.uniform(-1.0, 1.0, size=(4,)).tolist()
        noise = np.random.normal(0, 0.05, size=(4,))
        noisy = (np.array(ideal) + noise).tolist()

        entries.append({
            "circuit": qc_trans,
            "ideal_exp_value": ideal,
            "noisy_exp_values": [noisy],
            "circuit_depth": qc_trans.depth()
        })

    return entries, properties


def test_encode_data_equivalence():
    """Verify encode_entries matches canonical encode_data bit-for-bit."""
    entries, properties = generate_throwaway_dataset(n_circuits=40, seed=8888)
    circuits = [e["circuit"] for e in entries]
    ideal_vals = [e["ideal_exp_value"] for e in entries]
    noisy_vals = [e["noisy_exp_values"][0] for e in entries]

    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_ref, y_ref = encode_data(
        circuits=circuits,
        properties=properties,
        ideal_exp_vals=ideal_vals,
        noisy_exp_vals=noisy_vals,
        num_qubits=4
    )

    assert torch.equal(X_pkg, X_ref)
    assert torch.equal(y_pkg, y_ref)
    assert noisy_range == (54, 58)
    assert X_pkg.shape == (40, 58)


def test_arm_c_ablation():
    """Verify Arm C ablates exactly the 4 noisy-expectation columns."""
    entries, properties = generate_throwaway_dataset(n_circuits=40, seed=8888)
    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_np = X_pkg.numpy()
    X_train, X_test, X_val = X_np[:20], X_np[20:30], X_np[30:]

    X_tr_C, X_te_C, X_va_C, cols_C = prepare_arm_features(
        X_train, X_test, X_val, noisy_range, arm="C", learner_seed=0
    )
    assert X_tr_C.shape[1] == 54
    assert X_te_C.shape[1] == 54
    assert X_va_C.shape[1] == 54
    assert 54 not in cols_C
    assert 57 not in cols_C


def test_arm_p_permutation():
    """Verify Arm P permutes only training rows of noisy columns, leaving test and val untouched."""
    entries, properties = generate_throwaway_dataset(n_circuits=40, seed=8888)
    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_np = X_pkg.numpy()
    X_train, X_test, X_val = X_np[:20], X_np[20:30], X_np[30:]
    start_c, end_c = noisy_range

    X_tr_P, X_te_P, X_va_P, cols_P = prepare_arm_features(
        X_train, X_test, X_val, noisy_range, arm="P", learner_seed=42
    )

    # Non-noisy columns untouched
    non_noisy_cols = [c for c in range(X_train.shape[1]) if c < start_c or c >= end_c]
    assert np.array_equal(X_tr_P[:, non_noisy_cols], X_train[:, non_noisy_cols])

    # Noisy columns multiset preserved
    assert np.allclose(np.sort(X_tr_P[:, start_c:end_c], axis=0), np.sort(X_train[:, start_c:end_c], axis=0))

    # But rows permuted
    assert not np.array_equal(X_tr_P[:, start_c:end_c], X_train[:, start_c:end_c])

    # Test and val rows preserved exactly
    assert np.array_equal(X_te_P, X_test)
    assert np.array_equal(X_va_P, X_val)


def test_seed_determinism_and_variance():
    """Verify identical seeds produce identical predictions, different seeds differ."""
    entries, properties = generate_throwaway_dataset(n_circuits=40, seed=8888)
    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_np = X_pkg.numpy()
    y_np = y_pkg.numpy()
    X_train, X_test, X_val = X_np[:20], X_np[20:30], X_np[30:]
    y_train, y_test, y_val = y_np[:20], y_np[20:30], y_np[30:]

    X_tr_F, X_te_F, X_va_F, _ = prepare_arm_features(
        X_train, X_test, X_val, noisy_range, arm="F", learner_seed=7
    )

    # OLS determinism
    p_ols_1 = fit_and_predict_ols(X_tr_F, y_train, X_te_F)
    p_ols_2 = fit_and_predict_ols(X_tr_F, y_train, X_te_F)
    assert np.array_equal(p_ols_1, p_ols_2)

    # OLS Rcal determinism
    X_tr_Rc, X_te_Rc, _, _ = prepare_arm_features(
        X_train, X_test, X_val, noisy_range, arm="Rcal", learner_seed=7
    )
    p_rcal_1 = fit_and_predict_ols(X_tr_Rc, y_train, X_te_Rc)
    p_rcal_2 = fit_and_predict_ols(X_tr_Rc, y_train, X_te_Rc)
    assert np.array_equal(p_rcal_1, p_rcal_2)

    # RF determinism with same seed
    p_rf_1 = fit_and_predict_rf(X_tr_F, y_train, X_te_F, learner_seed=7)
    p_rf_2 = fit_and_predict_rf(X_tr_F, y_train, X_te_F, learner_seed=7)
    assert np.array_equal(p_rf_1, p_rf_2)

    # MLP determinism with same seed
    p_mlp_1 = fit_and_predict_mlp(X_tr_F, y_train, X_te_F, X_va_F, y_val, learner_seed=7, epochs=5)
    p_mlp_2 = fit_and_predict_mlp(X_tr_F, y_train, X_te_F, X_va_F, y_val, learner_seed=7, epochs=5)
    assert np.array_equal(p_mlp_1, p_mlp_2)

    # Different seeds differ
    p_rf_diff = fit_and_predict_rf(X_tr_F, y_train, X_te_F, learner_seed=8)
    assert not np.array_equal(p_rf_1, p_rf_diff)

    p_mlp_diff = fit_and_predict_mlp(X_tr_F, y_train, X_te_F, X_va_F, y_val, learner_seed=8, epochs=5)
    assert not np.array_equal(p_mlp_1, p_mlp_diff)


def test_frozen_rule_enforcement():
    """Verify safety check blocks unauthorized execution without frozen rule."""
    with pytest.raises(RuntimeError, match="FROZEN RULE BARRIER"):
        enforce_frozen_rule("")

    # Also run_arm_job requires existing frozen rule
    with pytest.raises(RuntimeError, match="FROZEN RULE BARRIER"):
        run_arm_job(
            setting="test",
            model="ols",
            arm="F",
            learner_seed=1,
            X_train=np.zeros((10, 58)),
            y_train=np.zeros((10, 4)),
            X_test=np.zeros((5, 58)),
            y_test=np.zeros((5, 4)),
            X_val=np.zeros((5, 58)),
            y_val=np.zeros((5, 4)),
            noisy_range=(54, 58),
            output_dir="/tmp/test",
            frozen_rule_path=None
        )


def test_mlp_scheduler_reads_validation_loss(monkeypatch):
    """Verify MLP ReduceLROnPlateau receives validation loss values across epochs (Item 3 & 8)."""
    entries, properties = generate_throwaway_dataset(n_circuits=30, seed=8888)
    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_np = X_pkg.numpy()
    y_np = y_pkg.numpy()
    X_train, X_test, X_val = X_np[:15], X_np[15:20], X_np[20:30]
    y_train, y_test, y_val = y_np[:15], y_np[15:20], y_np[20:30]

    captured_val_losses = []
    original_step = ReduceLROnPlateau.step

    def mock_step(self, metrics, epoch=None):
        captured_val_losses.append(float(metrics))
        return original_step(self, metrics, epoch=epoch)

    monkeypatch.setattr(ReduceLROnPlateau, "step", mock_step)

    n_epochs = 4
    fit_and_predict_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        X_val=X_val,
        y_val=y_val,
        learner_seed=123,
        epochs=n_epochs,
        batch_size=8
    )

    assert len(captured_val_losses) == n_epochs
    for loss in captured_val_losses:
        assert isinstance(loss, float)
        assert loss > 0.0
        assert np.isfinite(loss)


def test_run_py_writes_npz_and_json_for_tiny_jobs():
    """Verify run.py job_worker executes and produces valid NPZ and JSON files (Item 1 & 8)."""
    entries, properties = generate_throwaway_dataset(n_circuits=30, seed=8888)
    X_pkg, y_pkg, noisy_range = encode_entries(entries, properties=properties, num_qubits=4)
    X_np = X_pkg.numpy()
    y_np = y_pkg.numpy()
    X_train, X_test, X_val = X_np[:15], X_np[15:25], X_np[25:30]
    y_train, y_test, y_val = y_np[:15], y_np[15:25], y_np[25:30]

    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = os.path.join(tmpdir, "test_rule.md")
        with open(rule_path, "w") as f:
            f.write("# Test Frozen Rule\n")

        # Create mock encoded.npz and encoded.json
        encoded_npz = os.path.join(tmpdir, "encoded_setting.npz")
        encoded_json = os.path.join(tmpdir, "encoded_setting.json")
        test_ids = [("step_0.pk", i) for i in range(len(X_test))]
        test_steps = [0] * len(X_test)

        np.savez_compressed(
            encoded_npz,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            noisy_range=np.array(noisy_range),
            test_steps=np.array(test_steps),
            test_circuit_files=np.array([t[0] for t in test_ids]),
            test_circuit_indices=np.array([t[1] for t in test_ids]),
            test_circuit_ids=np.array([f"{t[0]}:{t[1]}" for t in test_ids]),
            train_circuit_files=np.array(["train.pk"] * len(X_train)),
            train_circuit_indices=np.array(list(range(len(X_train)))),
            train_circuit_ids=np.array([f"train.pk:{i}" for i in range(len(X_train))]),
            val_circuit_files=np.array(["val.pk"] * len(X_val)),
            val_circuit_indices=np.array(list(range(len(X_val)))),
            val_circuit_ids=np.array([f"val.pk:{i}" for i in range(len(X_val))]),
        )

        with open(encoded_json, "w") as f:
            json.dump({
                "setting": "test_setting",
                "test_is_validation": False,
                "input_files_sha256": {"test_file.pk": "abc123hash"}
            }, f)

        # Run tiny job list through job_worker
        job_spec = {
            "setting": "test_setting",
            "model": "ols",
            "arm": "F",
            "seed": 1,
            "output_dir": tmpdir,
            "encoded_npz": encoded_npz,
            "encoded_json": encoded_json,
            "frozen_rule": rule_path
        }

        res = job_worker(job_spec)
        assert res["status"] == "completed"

        out_npz = os.path.join(tmpdir, "test_setting_ols_F_seed1.npz")
        out_json = os.path.join(tmpdir, "test_setting_ols_F_seed1.json")
        assert os.path.isfile(out_npz)
        assert os.path.isfile(out_json)

        # Verify NPZ contents
        with np.load(out_npz) as data:
            assert "predictions" in data
            assert "targets" in data
            assert "raw_noisy" in data
            assert "test_circuit_files" in data
            assert "test_circuit_indices" in data
            assert "test_circuit_ids" in data
            assert "test_steps" in data
            assert data["predictions"].shape == y_test.shape
            assert len(data["test_circuit_files"]) == len(X_test)

        # Verify JSON metadata
        with open(out_json) as f:
            meta = json.load(f)
        assert meta["setting"] == "test_setting"
        assert meta["model"] == "ols"
        assert meta["arm"] == "F"
        assert meta["learner_seed"] == 1
        assert "frozen_rule_sha256" in meta
        assert meta["test_is_validation"] is False

        # Verify idempotency skip on second call
        res_repeat = job_worker(job_spec)
        assert res_repeat["status"] == "skipped"
        assert res_repeat["reason"] == "output_exists"


def test_manifest_check_refuses_unlisted_and_changed_files():
    import pytest
    from reanalysis.mlqem import encode

    with open(encode.MANIFEST) as handle:
        manifest = json.load(handle)
    key = "docs/tutorials/data/ising_init_from_qasm_no_readout/train/step_0.pk"
    good = {"train/step_0.pk": manifest[key]}
    encode.verify_manifest("ising_init_from_qasm_no_readout", good)
    with pytest.raises(SystemExit, match="SHA-256"):
        encode.verify_manifest("ising_init_from_qasm_no_readout", {"train/step_0.pk": "0" * 64})
    with pytest.raises(SystemExit, match="not in"):
        encode.verify_manifest("ising_init_from_qasm_no_readout", {"train/step_99.pk": "0" * 64})


def test_published_settings_keep_fakelima_readout_properties():
    from reanalysis.mlqem import encode

    assert all(not cfg["zero_readout_error"] for cfg in encode.SETTINGS.values())


def test_load_split_data_requires_every_requested_step(tmp_path):
    import pickle
    import pytest
    from reanalysis.mlqem import encode

    with open(tmp_path / "step_0.pk", "wb") as handle:
        pickle.dump([], handle)
    _, _, _, shas = encode.load_split_data(str(tmp_path), [0])
    assert list(shas) == ["step_0.pk"]
    with pytest.raises(FileNotFoundError, match="step_1.pk"):
        encode.load_split_data(str(tmp_path), [0, 1])


def test_environment_record_checks_commit_and_blackwater_path(tmp_path):
    import pytest
    from reanalysis.mlqem.run import environment_record

    record = environment_record(str(tmp_path), strict=False)
    assert record["mlqem_commit"] is None
    assert record["blackwater_path"].endswith("__init__.py")
    assert record["versions"]["qiskit-terra"] == "0.24.1"
    with pytest.raises(SystemExit, match="not b1eccf8"):
        environment_record(str(tmp_path), strict=True)
