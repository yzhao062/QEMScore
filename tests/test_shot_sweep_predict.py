"""Tests for tools/shot_sweep_predict.py."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.shot_sweep_predict import (
    CELLS_DEF,
    FAMILIES,
    RULE_THRESHOLDS,
    SEEDS,
    load_c_fit_predictions,
    run_predict,
)

_REPO = Path(__file__).resolve().parents[1]


def _make_synthetic_cache(seed: int, slope_b: float = 0.85, intercept_a: float = -0.001) -> dict:
    """Make synthetic validation and test items for a cache."""
    val_items = []
    test_items = []
    rng = np.random.default_rng(seed)

    for fam in FAMILIES:
        for sev, obs in CELLS_DEF:
            for i in range(20):
                y_val = float(rng.uniform(-0.8, 0.8))
                r_val = float(intercept_a + slope_b * y_val + rng.normal(0, 0.01))
                val_items.append({
                    "family": fam,
                    "severity": sev,
                    "observable": obs,
                    "ideal_expectation": y_val,
                    "noisy_expectation": r_val,
                })
            for i in range(10):
                y_test = float(rng.uniform(-0.8, 0.8))
                r_test = float(intercept_a + slope_b * y_test + rng.normal(0, 0.01))
                test_items.append({
                    "family": fam,
                    "severity": sev,
                    "observable": obs,
                    "ideal_expectation": y_test,
                    "noisy_expectation": r_test,
                })

    return {
        "key": f"shipped-s{seed}-n640",
        "validation": val_items,
        "test": test_items,
    }


def _make_synthetic_c_fits(fits_dir: Path, n_items: int = 40):
    """Write synthetic C fits for learner seeds 1..20."""
    fits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for seed in SEEDS:
        for rung in ("R0", "N1", "N2", "R5"):
            for k in range(1, 21):
                p_npz = fits_dir / f"shipped-s{seed}-n640__{rung}__k{k:02d}__C.npz"
                test_preds = rng.normal(0.0, 0.05, size=n_items)
                np.savez_compressed(p_npz, test=test_preds)


def test_shot_sweep_predict_b_near_zero(tmp_path: Path):
    """Test that when |b| <= 1e-12, c_m is infinite and c_comb equals c_C."""
    level_dir = tmp_path / "level_256"
    cache_dir = level_dir / "cache"
    cache_dir.mkdir(parents=True)

    # In this cache, r is constant regardless of y, so slope b is exactly 0.0
    for seed in SEEDS:
        val_items = []
        test_items = []
        rng = np.random.default_rng(seed)
        for fam in FAMILIES:
            for sev, obs in CELLS_DEF:
                for i in range(20):
                    y_val = float(rng.uniform(-0.8, 0.8))
                    val_items.append({
                        "family": fam,
                        "severity": sev,
                        "observable": obs,
                        "ideal_expectation": y_val,
                        "noisy_expectation": 0.5,
                    })
                for i in range(10):
                    y_test = float(rng.uniform(-0.8, 0.8))
                    test_items.append({
                        "family": fam,
                        "severity": sev,
                        "observable": obs,
                        "ideal_expectation": y_test,
                        "noisy_expectation": 0.5,
                    })
        cache_data = {
            "key": f"shipped-s{seed}-n640",
            "validation": val_items,
            "test": test_items,
        }
        with open(cache_dir / f"shipped-s{seed}-n640.json", "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

    fits_dir = tmp_path / "c_fits"
    _make_synthetic_c_fits(fits_dir, n_items=80)

    # Reference analysis with intervals that match RULE_THRESHOLDS
    ref_rows = {}
    for seed in SEEDS:
        for fam in FAMILIES:
            row_k = f"shipped-s{seed}-n640/{fam}"
            ref_rows[row_k] = {"rungs": {}}
            for rung in ("R0", "N1", "N2", "R5"):
                exp_t = RULE_THRESHOLDS.get(row_k, {}).get(rung, 0.100)
                ref_rows[row_k]["rungs"][rung] = {
                    "D_over_C": {"interval": {"lower": 0.0, "upper": exp_t}},
                    "classification": {"label": "not_distinguished"},
                }

    ref_analysis_path = tmp_path / "ref_analysis.json"
    ref_analysis_path.write_text(json.dumps({"parts": {"A": {"rows": ref_rows}}}))

    floor_path = tmp_path / "floor.json"
    floor_path.write_text(json.dumps({"per_cell_floor": {}}))

    out_json = tmp_path / "predictions.json"
    run_log = tmp_path / "run.log"

    res = run_predict(
        level_dirs={"256": level_dir},
        c_fits_dir=fits_dir,
        reference_c_fits_dir=fits_dir,
        reference_analysis_path=ref_analysis_path,
        measurement_floor_path=floor_path,
        out_path=out_json,
        run_log_path=run_log,
        command_str="test_b_near_zero",
        require_all_levels=False,
    )

    # Verify that slope_b is ~0, c_m is inf, and macro_c_comb == macro_c_C
    sample_rc = next(iter(res["rule_cells"].values()))
    assert abs(sample_rc["slope_b"]) <= 1e-12
    assert math.isinf(sample_rc["c_m"])

    sample_cell = next(iter(res["classification_cells"].values()))
    assert np.isclose(sample_cell["macro_c_comb"], sample_cell["macro_c_C"])
    assert np.isclose(sample_cell["D_over_C_star"], 0.0)


def _real_inputs():
    """Real 2,048-shot inputs, from QEMSCORE_LADDER_RUNS (holding partA/) and
    QEMSCORE_STRENGTH_FITS (the descriptor-information-strength-v1 fits); skip if unset."""
    runs = os.environ.get("QEMSCORE_LADDER_RUNS")
    fits = os.environ.get("QEMSCORE_STRENGTH_FITS")
    if not runs or not fits or not (Path(runs) / "partA").is_dir() or not Path(fits).is_dir():
        pytest.skip("set QEMSCORE_LADDER_RUNS and QEMSCORE_STRENGTH_FITS to run on real data")
    return Path(runs) / "partA", Path(fits)


def test_self_check_fails_on_perturbed_floor(tmp_path: Path, capsys):
    """The 2,048-shot self-check stops before writing anything when the floor differs."""
    partA_dir, strength_fits = _real_inputs()
    floor_data = json.loads((_REPO / "artifacts/descriptor-information/posthoc-measurement-floor.json")
                            .read_text(encoding="utf-8"))
    perturbed_floor = copy.deepcopy(floor_data)
    first_cell = next(iter(perturbed_floor["per_cell_floor"].keys()))
    perturbed_floor["per_cell_floor"][first_cell]["floor_mae"] += 0.01
    perturbed_floor_path = tmp_path / "perturbed_floor.json"
    perturbed_floor_path.write_text(json.dumps(perturbed_floor))
    out_file = tmp_path / "predictions_must_not_exist.json"
    log_file = tmp_path / "log_must_not_exist.log"
    with pytest.raises(SystemExit):
        run_predict(
            level_dirs={"2048": partA_dir},
            c_fits_dir=strength_fits,
            reference_c_fits_dir=strength_fits,
            reference_analysis_path=_REPO / "artifacts/descriptor-information/strength-indicator/analysis-a.json",
            measurement_floor_path=perturbed_floor_path,
            out_path=out_file,
            run_log_path=log_file,
            command_str="test",
            require_all_levels=False,
        )
    assert "SELF-CHECK FAILURE" in capsys.readouterr().err
    assert not out_file.exists()
    assert not log_file.exists()


def test_real_2048_self_check(tmp_path: Path):
    """On the real 2,048-shot caches and strength fits, the self-check passes."""
    partA_dir, strength_fits = _real_inputs()
    out_file = tmp_path / "predictions.json"
    log_file = tmp_path / "run.log"
    data = run_predict(
        level_dirs={"2048": partA_dir},
        c_fits_dir=strength_fits,
        reference_c_fits_dir=strength_fits,
        reference_analysis_path=_REPO / "artifacts/descriptor-information/strength-indicator/analysis-a.json",
        measurement_floor_path=_REPO / "artifacts/descriptor-information/posthoc-measurement-floor.json",
        out_path=out_file,
        run_log_path=log_file,
        command_str="test",
        require_all_levels=False,
    )
    assert out_file.exists() and log_file.exists()
    assert data["schema"] == "shot-sweep-predictions-v1"
    assert data["status"] == "pre-registered prediction"
    assert len(data["classification_cells"]) == 24


def test_cli_rejects_partial_level_set(tmp_path: Path):
    """The command line takes exactly the seven levels; omitting any (here all but 2,048) fails."""
    cmd = [sys.executable, str(_REPO / "tools/shot_sweep_predict.py"),
           "--level", f"2048={tmp_path}", "--c-fits", str(tmp_path), "--reference-c-fits", str(tmp_path),
           "--reference-analysis", str(tmp_path / "a.json"), "--measurement-floor", str(tmp_path / "f.json"),
           "--out", str(tmp_path / "p.json"), "--run-log", str(tmp_path / "run.log")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "seven shot levels" in proc.stderr
    assert not (tmp_path / "p.json").exists() and not (tmp_path / "run.log").exists()


def test_missing_c_seed_is_an_error(tmp_path: Path):
    """A missing learner seed (here 20), or a lone fit, stops the prediction."""
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for k in range(1, 20):
        np.savez(fits_dir / f"shipped-s101-n640__R0__k{k:02d}__C.npz", test=np.zeros(4))
    with pytest.raises(FileNotFoundError):
        load_c_fit_predictions(fits_dir, 101, "R0")
    lone = tmp_path / "lone"
    lone.mkdir()
    np.savez(lone / "shipped-s101-n640__R0__k01__C.npz", test=np.zeros(4))
    with pytest.raises(FileNotFoundError):
        load_c_fit_predictions(lone, 101, "R0")


def test_rule_thresholds_assertion():
    """Verify that all row and rung combinations in RULE_THRESHOLDS match the table."""
    ref_path = _REPO / "artifacts/descriptor-information/strength-indicator/analysis-a.json"
    ref_data = json.loads(ref_path.read_text())["parts"]["A"]["rows"]

    for row, rung_map in RULE_THRESHOLDS.items():
        assert row in ref_data
        for rung, expected_thresh in rung_map.items():
            r_data = ref_data[row]["rungs"][rung]
            ci = r_data["D_over_C"]["interval"]
            h = (ci["upper"] - ci["lower"]) / 2.0
            thresh = round(max(0.10, 2.0 * h), 3)
            assert thresh == expected_thresh
