"""Tests for the A/A calibration tool (tools/aa_calibration.py).

Validates:
1. Two-stage A/A bootstrap interval arithmetic and properties.
2. Random partitions calibration under the null hypothesis (two identical
   pipelines exclude zero ~5% of the time).
3. Shifted pipeline detection (two-stage interval excludes zero).
4. Single-fit A/A pairwise comparisons mirroring the frozen campaign.
5. D rule consistency under A/A.
6. Input binding and refusal checks:
   - mismatched learner seed field
   - mismatched dataset hash
   - swapped/tampered prediction array
   - missing seed
   - missing rung
7. Input hashing (fits digest, cache SHA-256, script SHA-256) and atomic output.
8. CLI end-to-end execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools import aa_calibration as aac
from tools.descriptor_ladder_analysis import Row, point_macro


def _create_toy_cache(
    path: Path,
    key: str = "shipped-s101-n640",
    dataset_seed: int = 101,
    n_circuits: int = 16,
) -> tuple[dict, Path]:
    cells = [("noise_model", sev, obs) for sev in ("L1", "L3") for obs in ("z_mid", "zz_mid")]
    rows = []
    rng = np.random.default_rng(dataset_seed)
    for c_idx in range(n_circuits):
        circ_id = f"circ-{c_idx:03d}"
        for fam in ("tfi", "heisenberg"):
            for nf, sev, obs in cells:
                ideal = float(rng.uniform(-0.5, 0.5))
                noisy = ideal + float(rng.normal(0, 0.05))
                rows.append({
                    "item_id": f"item-{fam}-{circ_id}-{sev}-{obs}",
                    "circuit_id": circ_id,
                    "family": fam,
                    "stratum": "continuous_regression",
                    "noise_family": nf,
                    "severity": sev,
                    "observable": obs,
                    "ideal_expectation": ideal,
                    "noisy_expectation": noisy,
                    "split": "test",
                })
    data = {
        "key": key,
        "seed": dataset_seed,
        "dataset_hash": f"hash-{key}",
        "test": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    return data, path


def _write_toy_fit(
    fit_dir: Path,
    key: str,
    rung: str,
    arm: str,
    seed: int,
    cache_data: dict,
    predictions: np.ndarray,
) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    tag = f"k{seed:02d}"
    stem = f"{key}__{rung}__{tag}__{arm}"

    np.savez_compressed(fit_dir / f"{stem}.npz", test=predictions)

    # Calculate exact point macro MAE per family
    family_maes = {}
    for fam in ("tfi", "heisenberg"):
        row = Row(cache_data, fam, f"{key}/{fam}")
        err = np.abs(predictions[row.index] - row.ideal)
        family_maes[fam] = float(point_macro(err, row.members, row.all_cells))

    meta = {
        "schema": "descriptor-information-fit-v1",
        "key": key,
        "dataset_seed": cache_data.get("seed", 101),
        "dataset_hash": cache_data.get("dataset_hash"),
        "rung": rung,
        "arm": arm,
        "learner_seed": seed,
        "selected_model": "mlp",
        "test_family_mae": family_maes,
    }
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------


def test_parse_seeds():
    assert aac.parse_seeds("1-20") == tuple(range(1, 21))
    assert aac.parse_seeds("21-40") == tuple(range(21, 41))
    assert aac.parse_seeds("1..10") == tuple(range(1, 11))
    assert aac.parse_seeds("1, 2, 5") == (1, 2, 5)
    assert aac.parse_seeds([1, 2, 3]) == (1, 2, 3)
    assert aac.parse_seeds(["1-5", "6-10"]) == tuple(range(1, 11))


def test_two_stage_aa_draws_properties():
    sc1, sc2, cc = aac.two_stage_aa_draws(20, 20, 160, 500, 20261002)
    assert sc1.shape == (500, 20)
    assert sc2.shape == (500, 20)
    assert cc.shape == (500, 160)
    # Sum of counts in each draw must equal n_samples drawn
    assert np.all(sc1.sum(axis=1) == 20)
    assert np.all(sc2.sum(axis=1) == 20)
    assert np.all(cc.sum(axis=1) == 160)
    # Memoization returns identical objects
    sc1_b, sc2_b, cc_b = aac.two_stage_aa_draws(20, 20, 160, 500, 20261002)
    assert sc1 is sc1_b
    assert sc2 is sc2_b
    assert cc is cc_b


def test_single_fit_circuit_draws_properties():
    cc = aac.single_fit_circuit_draws(160, 500, 20261004)
    assert cc.shape == (500, 160)
    assert np.all(cc.sum(axis=1) == 160)
    cc_b = aac.single_fit_circuit_draws(160, 500, 20261004)
    assert cc is cc_b


def test_two_identical_pipelines_exclude_zero_about_five_percent(tmp_path: Path):
    """Under the null hypothesis, 95% intervals exclude zero ~5% of the time."""
    cache_file = tmp_path / "cache" / "shipped-s101-n640.pkl"
    cache_data, _ = _create_toy_cache(cache_file, n_circuits=24)
    row = Row(cache_data, "tfi", "shipped-s101-n640/tfi")

    # Generate 40 seeds from identical pipeline with realistic variance components
    rng = np.random.default_rng(123)
    errors_40 = []
    for _ in range(40):
        seed_effect = rng.normal(0, 0.02)
        circ_effect = rng.normal(0, 0.02, size=row.n_circuits)
        item_err = 0.05 + seed_effect + circ_effect[row.item_circuit] + rng.normal(0, 0.01, size=row.n_items)
        errors_40.append(np.abs(item_err))
    errors_40 = np.asarray(errors_40)

    # 200 random partitions of the 40 seeds into two sets of 20
    rp = aac.run_random_partitions(
        errors_40, list(range(1, 41)), row,
        n_partitions=200, draws=2000,
        bootstrap_seed=20261002, partition_seed=20261003,
    )
    rate = rp["fraction_excluding_zero"]
    # 5% nominal rate; verify empirical rate is within expected simulation range [0.02, 0.09]
    assert 0.02 <= rate <= 0.09, f"Expected exclusion rate ~0.05, got {rate}"


def test_shifted_pipeline_must_exclude_zero(tmp_path: Path):
    """A pipeline with a true performance gap must exclude zero."""
    cache_file = tmp_path / "cache" / "shipped-s101-n640.pkl"
    cache_data, _ = _create_toy_cache(cache_file, n_circuits=16)
    row = Row(cache_data, "tfi", "shipped-s101-n640/tfi")

    rng = np.random.default_rng(77)
    errors_1 = np.abs(0.04 + rng.normal(0, 0.005, size=(20, row.n_items)))
    # Set 2 is shifted by +0.06 higher error
    errors_2 = np.abs(0.10 + rng.normal(0, 0.005, size=(20, row.n_items)))

    res = aac.run_two_stage_aa(
        errors_1, errors_2, row,
        draws=2000, seed=20261002,
        seeds_1=list(range(1, 21)), seeds_2=list(range(21, 41)),
    )
    assert res["excludes_zero"] is True
    assert res["excludes_zero_below"] is True
    assert res["interval"]["upper"] < 0.0


def test_single_fit_aa_pairwise(tmp_path: Path):
    """Test single-fit comparisons over 190 pairs of 20 seeds."""
    cache_file = tmp_path / "cache" / "shipped-s101-n640.pkl"
    cache_data, _ = _create_toy_cache(cache_file, n_circuits=16)
    row = Row(cache_data, "tfi", "shipped-s101-n640/tfi")

    rng = np.random.default_rng(999)
    errors_20 = np.abs(0.05 + rng.normal(0, 0.015, size=(20, row.n_items)))
    res = aac.run_single_fit_aa(errors_20, list(range(1, 21)), row, draws=500, seed=20261004)

    assert res["total_pairs"] == 190
    assert len(res["pairs"]) == 190
    assert 0.0 <= res["fraction_excluding_zero"] <= 1.0
    for p in res["pairs"]:
        assert p["seed_1"] < p["seed_2"]
        assert p["interval"]["lower"] <= p["interval"]["upper"]


def test_d_rule_aa(tmp_path: Path):
    """Test D rule comparing Set 1 and Set 2."""
    cache_file = tmp_path / "cache" / "shipped-s101-n640.pkl"
    cache_data, _ = _create_toy_cache(cache_file, n_circuits=16)
    row = Row(cache_data, "tfi", "shipped-s101-n640/tfi")

    rng = np.random.default_rng(88)
    # Set 1 and Set 2 both have C and F with no true difference (not distinguished)
    err_c1 = np.abs(0.05 + rng.normal(0, 0.005, size=(10, row.n_items)))
    err_f1 = np.abs(0.05 + rng.normal(0, 0.005, size=(10, row.n_items)))
    err_c2 = np.abs(0.05 + rng.normal(0, 0.005, size=(10, row.n_items)))
    err_f2 = np.abs(0.05 + rng.normal(0, 0.005, size=(10, row.n_items)))

    res = aac.run_d_rule_aa(
        err_c1, err_f1, err_c2, err_f2,
        list(range(1, 11)), list(range(11, 21)), row,
        draws=500, seed=20261002,
    )
    assert res["status"] == "estimated"
    assert res["set_1"]["label"] in ("not_distinguished", "measurement_adds", "measurement_hurts")
    assert res["set_2"]["label"] in ("not_distinguished", "measurement_adds", "measurement_hurts")
    assert isinstance(res["labels_differ"], bool)


# --------------------------------------------------------------------------
# Refusal tests (input binding and validation)
# --------------------------------------------------------------------------


def _setup_toy_environment(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = tmp_path / "cache"
    fits_dir = tmp_path / "fits"
    cache_file = cache_dir / "shipped-s101-n640.pkl"
    cache_data, _ = _create_toy_cache(cache_file, n_circuits=16)

    rng = np.random.default_rng(42)
    for rung in ("R0", "N1"):
        for arm in ("C", "F"):
            for s in range(1, 41):
                preds = np.asarray([r["ideal_expectation"] + float(rng.normal(0, 0.02)) for r in cache_data["test"]])
                _write_toy_fit(fits_dir, "shipped-s101-n640", rung, arm, s, cache_data, preds)
    return cache_dir, fits_dir


def test_refuse_mismatched_learner_seed(tmp_path: Path):
    """Refuse when fit JSON learner_seed disagrees with file stem."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    # Corrupt learner_seed in one JSON
    bad_json = fits_dir / "shipped-s101-n640__R0__k05__C.json"
    meta = json.loads(bad_json.read_text(encoding="utf-8"))
    meta["learner_seed"] = 99  # mismatch with k05
    bad_json.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        aac.calibrate_all(
            fit_dirs=[fits_dir], cache_dirs=[cache_dir],
            rungs=["R0", "N1"], arms=["C", "F"],
            set1_seeds="1-20", set2_seeds="21-40",
            verbose=False,
        )
    assert "learner_seed" in str(exc_info.value)


def test_refuse_mismatched_dataset_hash(tmp_path: Path):
    """Refuse when fit JSON dataset_hash disagrees with the cache."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    bad_json = fits_dir / "shipped-s101-n640__R0__k05__C.json"
    meta = json.loads(bad_json.read_text(encoding="utf-8"))
    meta["dataset_hash"] = "incorrect_dataset_hash"
    bad_json.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        aac.calibrate_all(
            fit_dirs=[fits_dir], cache_dirs=[cache_dir],
            rungs=["R0", "N1"], arms=["C", "F"],
            set1_seeds="1-20", set2_seeds="21-40",
            verbose=False,
        )
    assert "dataset_hash" in str(exc_info.value)


def test_refuse_swapped_prediction_array(tmp_path: Path):
    """Refuse when NPZ predictions fail to reproduce recorded test_family_mae."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    # Tamper with NPZ predictions for one seed
    npz_path = fits_dir / "shipped-s101-n640__R0__k05__C.npz"
    bad_preds = np.zeros(128)  # zeroes will not match recorded MAE
    np.savez_compressed(npz_path, test=bad_preds)

    with pytest.raises(SystemExit) as exc_info:
        aac.calibrate_all(
            fit_dirs=[fits_dir], cache_dirs=[cache_dir],
            rungs=["R0", "N1"], arms=["C", "F"],
            set1_seeds="1-20", set2_seeds="21-40",
            verbose=False,
        )
    assert "test_family_mae" in str(exc_info.value)


def test_refuse_missing_seed(tmp_path: Path):
    """Refuse when any requested learner seed is missing."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    # Remove seed 40
    (fits_dir / "shipped-s101-n640__R0__k40__C.json").unlink()
    (fits_dir / "shipped-s101-n640__R0__k40__C.npz").unlink()

    with pytest.raises(SystemExit) as exc_info:
        aac.calibrate_all(
            fit_dirs=[fits_dir], cache_dirs=[cache_dir],
            rungs=["R0", "N1"], arms=["C", "F"],
            set1_seeds="1-20", set2_seeds="21-40",
            verbose=False,
        )
    assert "Missing" in str(exc_info.value) or "seed" in str(exc_info.value)


def test_refuse_missing_rung_for_some_row(tmp_path: Path):
    """Refuse when an explicit rung is missing for some row."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        aac.calibrate_all(
            fit_dirs=[fits_dir], cache_dirs=[cache_dir],
            rungs=["R0", "NONEXISTENT_RUNG"], arms=["C", "F"],
            set1_seeds="1-20", set2_seeds="21-40",
            verbose=False,
        )
    assert "NONEXISTENT_RUNG" in str(exc_info.value)


# --------------------------------------------------------------------------
# Integration test & CLI
# --------------------------------------------------------------------------


def test_calibrate_all_end_to_end_and_cli(tmp_path: Path):
    """End-to-end integration test with synthetic fits on disk."""
    cache_dir, fits_dir = _setup_toy_environment(tmp_path)
    out_file = tmp_path / "out" / "calibration.json"

    # Test calibrate_all directly
    result = aac.calibrate_all(
        fit_dirs=[fits_dir],
        cache_dirs=[cache_dir],
        rungs=["R0", "N1"],
        arms=["C", "F"],
        set1_seeds="1-20",
        set2_seeds="21-40",
        draws_two_stage=100,
        draws_partitions=100,
        n_partitions=10,
        draws_single_fit=100,
        verbose=False,
    )

    assert result["schema"] == "aa-calibration-v1"
    assert "fits_sha256_digest" in result["inputs"]
    assert "shipped-s101-n640" in result["inputs"]["caches_used"]
    assert "two_stage_aa" in result["summaries"]
    assert "random_partitions" in result["summaries"]
    assert "single_fit_aa" in result["summaries"]
    assert "d_rule_consistency" in result["summaries"]

    # Test CLI execution
    exit_code = aac.main([
        "--fits", str(fits_dir),
        "--cache", str(cache_dir),
        "--rungs", "R0", "N1",
        "--arms", "C", "F",
        "--set1-seeds", "1-20",
        "--set2-seeds", "21-40",
        "--draws", "50",
        "--partition-draws", "50",
        "--partition-count", "5",
        "--single-fit-draws", "50",
        "--out", str(out_file),
        "--quiet",
    ])
    assert exit_code == 0
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        disk_result = json.load(f)
    assert disk_result["schema"] == "aa-calibration-v1"
    assert disk_result["script_sha256"] == aac.sha256_file(Path(aac.__file__).resolve())
