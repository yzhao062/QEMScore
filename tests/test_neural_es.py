"""Tests for the well-trained MLP candidate with neural early stopping (--neural-es)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from qemscore.baselines.liao import (
    DEFAULT_LR_MIN,
    DEFAULT_LR_PLATEAU_PATIENCE,
    MLP_LEARNING_RATE,
    LiaoMLPMitigator,
    LiaoMitigator,
)
from qemscore.campaign.design import campaign_split_spec
from qemscore.datasets.split_generate import generate_split
from qemscore.validation import validate_split_artifact
from tools import descriptor_common as common
from tools import descriptor_ladder as ladder
from tools import qaoa_intermediate as qaoa

REPO = Path(__file__).resolve().parents[1]
ORACLE_PATH = Path(__file__).resolve().parent / "oracles" / "liao_default_predictions.json"
PART_A_DIR = Path("/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/ladder/runs/partA")


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _make_synthetic_items(n_train: int = 128, n_val: int = 64, n_features: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)

    X_tr = rng.normal(size=(n_train, n_features))
    y_tr = np.sum(X_tr[:, :3], axis=1) * 0.5 + rng.normal(scale=0.05, size=n_train)

    X_va = rng.normal(size=(n_val, n_features))
    y_va = np.sum(X_va[:, :3], axis=1) * 0.5 + rng.normal(scale=0.05, size=n_val)

    train_items = []
    for i in range(n_train):
        c_id = f"circ_{i // 4}"
        train_items.append({
            "circuit_id": c_id,
            "measurement_group": c_id,
            "n_qubits": 4,
            "pauli_label": "IIZZ",
            "ideal_expectation": float(y_tr[i]),
            "noisy_expectation": float(y_tr[i] + rng.normal(scale=0.02)),
            "feat": X_tr[i].tolist(),
        })

    val_items = []
    for i in range(n_val):
        c_id = f"val_circ_{i // 4}"
        val_items.append({
            "circuit_id": c_id,
            "measurement_group": c_id,
            "n_qubits": 4,
            "pauli_label": "IIZZ",
            "ideal_expectation": float(y_va[i]),
            "noisy_expectation": float(y_va[i] + rng.normal(scale=0.02)),
            "feat": X_va[i].tolist(),
        })

    builder = lambda it: it["feat"]
    names = [f"f_{j}" for j in range(n_features)]
    return train_items, val_items, builder, names


# --------------------------------------------------------------------------
# 1. Defaults bit-identical to unmodified code (Oracle Regression Test)
# --------------------------------------------------------------------------

def test_defaults_bit_identical_oracle(tmp_path):
    assert ORACLE_PATH.exists(), f"Oracle file missing: {ORACLE_PATH}"
    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))

    scratch = tmp_path / "oracle_gen"
    spec = campaign_split_spec("shipped", 640, counts={"train": 6, "validation": 4, "test": 4})
    generate_split(spec, scratch, master_seed=9001)
    rows, _ = validate_split_artifact(scratch)

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]

    # Case 1: LiaoMLPMitigator default (fixed_epochs)
    mlp1 = LiaoMLPMitigator(random_state=42).fit(train)
    val_p1 = mlp1.predict(val)
    test_p1 = mlp1.predict(test)
    assert _sha256_array(val_p1) == oracle["cases"]["mlp_fixed_epochs"]["validation_predictions"]["sha256"]
    assert _sha256_array(test_p1) == oracle["cases"]["mlp_fixed_epochs"]["test_predictions"]["sha256"]
    assert mlp1.n_iter_ == oracle["cases"]["mlp_fixed_epochs"]["n_iter"]

    # Case 2: LiaoMLPMitigator validation_patience
    mlp2 = LiaoMLPMitigator(
        random_state=42, stopping_rule="validation_patience", epochs=100, patience=20, min_delta=0.0
    ).fit(train, validation_items=val)
    val_p2 = mlp2.predict(val)
    test_p2 = mlp2.predict(test)
    assert _sha256_array(val_p2) == oracle["cases"]["mlp_validation_patience"]["validation_predictions"]["sha256"]
    assert _sha256_array(test_p2) == oracle["cases"]["mlp_validation_patience"]["test_predictions"]["sha256"]
    assert mlp2.n_iter_ == oracle["cases"]["mlp_validation_patience"]["n_iter"]

    # Case 3: LiaoMitigator default (fixed_epochs)
    mit1 = LiaoMitigator(random_state=42).fit(train, val)
    val_p3 = mit1.predict(val)
    test_p3 = mit1.predict(test)
    assert _sha256_array(val_p3) == oracle["cases"]["mitigator_fixed_epochs"]["validation_predictions"]["sha256"]
    assert _sha256_array(test_p3) == oracle["cases"]["mitigator_fixed_epochs"]["test_predictions"]["sha256"]
    assert mit1.selected_model_name_ == oracle["cases"]["mitigator_fixed_epochs"]["selected_model"]

    # Case 4: LiaoMitigator validation_patience
    mit2 = LiaoMitigator(
        random_state=42, stopping_rule="validation_patience", epochs=100, patience=20, min_delta=0.0
    ).fit(train, val)
    val_p4 = mit2.predict(val)
    test_p4 = mit2.predict(test)
    assert _sha256_array(val_p4) == oracle["cases"]["mitigator_validation_patience"]["validation_predictions"]["sha256"]
    assert _sha256_array(test_p4) == oracle["cases"]["mitigator_validation_patience"]["test_predictions"]["sha256"]
    assert mit2.selected_model_name_ == oracle["cases"]["mitigator_validation_patience"]["selected_model"]

    # Case 5: LiaoMitigator drop_features (feat-only)
    mit3 = LiaoMitigator(random_state=17, drop_features=("noisy_expectation",)).fit(train, val)
    val_p5 = mit3.predict(val)
    test_p5 = mit3.predict(test)
    assert _sha256_array(val_p5) == oracle["cases"]["mitigator_drop_features"]["validation_predictions"]["sha256"]
    assert _sha256_array(test_p5) == oracle["cases"]["mitigator_drop_features"]["test_predictions"]["sha256"]


# --------------------------------------------------------------------------
# 2. Early stopping stops before cap and restores best epoch
# --------------------------------------------------------------------------

def test_early_stopping_stops_before_cap_and_restores_best_epoch():
    train_items, val_items, builder, names = _make_synthetic_items(
        n_train=256, n_val=128, n_features=10, seed=101
    )

    cap = 1000
    patience = 20
    model = LiaoMLPMitigator(
        random_state=101,
        epochs=cap,
        stopping_rule="validation_patience",
        patience=patience,
        min_delta=0.0,
        lr_plateau_factor=0.5,
        lr_plateau_patience=10,
        lr_min=1e-5,
        feature_builder=builder,
        feature_names=names,
    ).fit(train_items, validation_items=val_items)

    assert model.n_iter_ < cap, f"Model ran full cap {cap} instead of early stopping"
    assert 1 <= model.best_epoch_ <= model.n_iter_
    assert model.best_epoch_ < model.n_iter_, "Early stopping should happen after best epoch + patience"
    assert model.n_iter_ - model.best_epoch_ >= patience

    # Validation loss at restored parameters must equal the recorded best_validation_loss_
    x_val = model._standardize(model._features(val_items, "items"))
    y_val = np.asarray([it["ideal_expectation"] for it in val_items]).reshape(-1, 1)
    val_mse = model._mse(x_val, y_val)
    assert val_mse == pytest.approx(min(model.validation_loss_curve_))
    # It must be strictly better than or equal to the final epoch loss
    assert val_mse <= model.validation_loss_curve_[-1]


# --------------------------------------------------------------------------
# 3. Plateau schedule halves rate and respects floor
# --------------------------------------------------------------------------

def test_plateau_schedule_halves_rate_and_respects_floor():
    train_items, val_items, builder, names = _make_synthetic_items(
        n_train=128, n_val=64, n_features=8, seed=202
    )

    # To trigger plateaus consistently, use a high min_delta so improvements are not registered
    lr_init = MLP_LEARNING_RATE  # 0.001
    factor = 0.5
    lr_min = 1.5e-4
    plateau_patience = 3
    early_stop_patience = 25
    epochs = 20

    model = LiaoMLPMitigator(
        random_state=202,
        epochs=epochs,
        stopping_rule="validation_patience",
        patience=early_stop_patience,
        min_delta=100.0,  # impossible to beat, guarantees consecutive stale epochs
        lr_plateau_factor=factor,
        lr_plateau_patience=plateau_patience,
        lr_min=lr_min,
        feature_builder=builder,
        feature_names=names,
    ).fit(train_items, validation_items=val_items)

    assert model.n_iter_ == epochs
    lrs = model.learning_rates_
    assert len(lrs) == epochs

    # Epoch 1..4: 1e-3 (epoch 1 establishes baseline, epochs 2, 3, 4 are stale; at end of 4, rate halves)
    for ep in range(0, 4):
        assert lrs[ep] == pytest.approx(1e-3)
    # Epoch 5..7: 5e-4 (stale 3 epochs; at end of 7, rate halves)
    for ep in range(4, 7):
        assert lrs[ep] == pytest.approx(5e-4)
    # Epoch 8..10: 2.5e-4 (stale 3 epochs; at end of 10, rate halves to 1.25e-4, floored at 1.5e-4)
    for ep in range(7, 10):
        assert lrs[ep] == pytest.approx(2.5e-4)
    # Epoch 11+: floor reached at 1.5e-4
    for ep in range(10, epochs):
        assert lrs[ep] == pytest.approx(lr_min)

    assert model.final_learning_rate_ == pytest.approx(lr_min)
    assert len(model.learning_rate_changes_) == 3  # changed at end of epochs 4, 7, 10


# --------------------------------------------------------------------------
# 4. Refusal rules
# --------------------------------------------------------------------------

def test_refusal_rules(tmp_path):
    # lr_plateau_factor with fixed_epochs is refused
    with pytest.raises(ValueError, match="lr_plateau schedule requires"):
        LiaoMLPMitigator(stopping_rule="fixed_epochs", lr_plateau_factor=0.5)

    with pytest.raises(ValueError, match="lr_plateau schedule requires"):
        LiaoMitigator(stopping_rule="fixed_epochs", lr_plateau_factor=0.5)

    # Incompatible options: --neural-es and --strong-learners
    with pytest.raises(SystemExit, match="Refusal"):
        ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                     "--out", str(tmp_path), "--neural-es", "--strong-learners", "--dry-run",
                     "--rungs", "R0", "--arms", "F", "--seeds", "1"])

    with pytest.raises(SystemExit, match="Refusal"):
        qaoa.main(["run", "--data-root", str(tmp_path / "root"), "--out", str(tmp_path),
                   "--gate-file", "g.json", "--neural-es", "--strong-learners", "--dry-run"])

    # Directory refusal:
    # 1. --neural-es refuses a directory holding fits without neural_es: true
    fit_dir = tmp_path / "fits"
    fit_dir.mkdir(parents=True)
    (fit_dir / "old_fit.json").write_text(
        json.dumps({"schema": "descriptor-information-fit-v1"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Refusal"):
        ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                     "--out", str(tmp_path), "--neural-es", "--dry-run",
                     "--rungs", "R0", "--arms", "F", "--seeds", "1"])

    # Without --neural-es, the non-neural fit is accepted
    assert ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                        "--out", str(tmp_path), "--dry-run",
                        "--rungs", "R0", "--arms", "F", "--seeds", "1"]) == 0

    # 2. vice versa: run without --neural-es refuses a directory holding fits WITH neural_es: true
    (fit_dir / "old_fit.json").unlink()
    (fit_dir / "neural_fit.json").write_text(
        json.dumps({"schema": "descriptor-information-fit-v1", "neural_es": True}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Refusal"):
        ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                     "--out", str(tmp_path), "--dry-run",
                     "--rungs", "R0", "--arms", "F", "--seeds", "1"])

    # With --neural-es, the neural-es fit directory is accepted
    assert ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                        "--out", str(tmp_path), "--neural-es", "--dry-run",
                        "--rungs", "R0", "--arms", "F", "--seeds", "1"]) == 0


# --------------------------------------------------------------------------
# 5. Job plumbing (run --dry-run --neural-es) on real prepared tree
# --------------------------------------------------------------------------

def test_job_plumbing_dry_run_real_tree(tmp_path):
    if not (PART_A_DIR / "cache").exists():
        pytest.skip(f"Real partA prepared tree not available at {PART_A_DIR}")

    # Copy prepared tree without fits to tmp_path
    scratch = tmp_path / "partA_scratch"
    scratch.mkdir(parents=True)
    for name in ("cache", "descriptors", "encoder-cache"):
        shutil.copytree(PART_A_DIR / name, scratch / name)
    shutil.copy(PART_A_DIR / "datasets.json", scratch / "datasets.json")

    # Dry-run with --neural-es lists jobs with the flag
    assert ladder.main([
        "run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
        "--out", str(scratch), "--neural-es", "--dry-run",
        "--gate-file", "g.json", "--rungs", "R0", "N1", "--arms", "F", "C", "A", "--seeds", "1"
    ]) == 0

    # Direct build_jobs test
    jobs = ladder.build_jobs(
        scratch, ["shipped-s101-n640"], [], ["R0", "N1"], ["F", "C", "A"], [1],
        strength_indicator=True, neural_es=True
    )
    assert len(jobs) > 0
    for job in jobs:
        assert job.get("neural_es") is True

    # Check that running against the real PART_A_DIR (which has non-neural fits) triggers refusal
    with pytest.raises(SystemExit, match="Refusal"):
        ladder.main([
            "run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
            "--out", str(PART_A_DIR), "--neural-es", "--dry-run",
            "--gate-file", "g.json", "--rungs", "R0", "--arms", "F", "--seeds", "1"
        ])


# --------------------------------------------------------------------------
# 6. Tiny end-to-end fit on a rehearsal dataset
# --------------------------------------------------------------------------

def test_tiny_end_to_end_fit_on_rehearsal_dataset(tmp_path):
    scratch_data = tmp_path / "data"
    spec = campaign_split_spec("shipped", 640, counts={"train": 6, "validation": 4, "test": 4})
    generate_split(spec, scratch_data, master_seed=9001)
    rows, manifest = validate_split_artifact(scratch_data)

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]

    data = {
        "seed": 9001,
        "dataset_hash": manifest["dataset_hash"],
        "train": train,
        "validation": val,
        "test": test,
        "prediction_rows": {
            "validation": val,
            "test": test,
        },
    }

    builder = lambda it: [float(x) for x in range(10)]
    names = [f"f_{i}" for i in range(10)]
    fit_dir = tmp_path / "fits"

    job = {
        "part": "A",
        "key": "rehearsal-s9001",
        "rung": "R0",
        "arm": "F",
        "fit_arm": "F",
        "learner_seed": 1,
        "shuffle_seed": None,
        "purpose": "rehearsal_test",
        "stem": "rehearsal__R0__k01__F",
        "fit_dir": str(fit_dir),
        "neural_es": True,
    }

    info = common.run_liao_fit(job, data, builder, names)
    assert info["error"] is None

    json_path = fit_dir / f"{job['stem']}.json"
    npz_path = fit_dir / f"{job['stem']}.npz"
    assert json_path.exists()
    assert npz_path.exists()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["neural_es"] is True
    assert meta["schedule_parameters"]["stopping_rule"] == "validation_patience"
    assert meta["schedule_parameters"]["epochs"] == 2000
    assert meta["schedule_parameters"]["patience"] == 50
    assert meta["schedule_parameters"]["min_delta"] == 0.0
    assert meta["schedule_parameters"]["lr_plateau_factor"] == 0.5
    assert meta["schedule_parameters"]["lr_plateau_patience"] == 10
    assert meta["schedule_parameters"]["lr_min"] == 1e-5

    assert "lr_plateau_factor" not in meta  # recorded once, under schedule_parameters
    assert meta["epochs_run"] > 0
    assert meta["best_epoch"] > 0
    assert meta["final_learning_rate"] > 0

    with np.load(npz_path) as npz:
        assert "validation" in npz
        assert "test" in npz
        assert "validation__mlp" in npz
        assert "test__mlp" in npz
        assert "validation__random_forest" in npz
        assert "test__random_forest" in npz
