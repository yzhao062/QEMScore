"""Tests for Round 8 tools: tools/measurement_free_stack.py and tools/equal_spend.py.

Required checks:
1. Synthetic data where g equals y exactly (inc_r|g approx 0 and inc_g approx recal).
2. Synthetic data where g is noise (inc_g approx 0).
3. Synthetic data where r carries independent information (inc_r|g > 0).
4. The inc_r reproduction check fails on a perturbed --stacking-all.
5. Identity mismatches refused.
6. Equal-spend arithmetic on hand-computed cases (signed n*_cal) and the cost direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from qemscore.validation import item_stream_hash, split_dataset_hash
from tools import equal_spend as es
from tools import measurement_free_stack as mfs


def _create_synthetic_environment(
    root: Path,
    *,
    g_mode: str = "exact",  # "exact", "noise", "independent_r"
    n_circuits: int = 15,
    fit_hashes: dict[int, str] | None = None,
    strong_fit_override: dict | None = None,
    rungs: tuple[str, ...] = ("R0",),
):
    """Build a complete synthetic shot level environment for testing.

    Modes:
      - 'exact': g = y exactly, C_hat imperfect, r noisy.
      - 'noise': g is pure random noise uncorrelated with y.
      - 'independent_r': y has a component coming from r that C_hat and g cannot predict.
    """
    data_dir = root / "data"
    cache_dir = root / "cache"
    c_fits_dir = root / "c_fits"
    g_fits_dir = root / "g_fits"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    c_fits_dir.mkdir(parents=True, exist_ok=True)
    g_fits_dir.mkdir(parents=True, exist_ok=True)

    caches_used = {}
    rows_analysis_a = {}
    dataset_hashes = {}

    rng = np.random.default_rng(20261004)

    for seed in mfs.SEEDS:
        key = f"shipped-s{seed}-n640"
        items = []

        # Generate items: 2 splits, 2 families, 2 severities, 2 observables, n_circuits
        for split in ("validation", "test"):
            for circ_idx in range(n_circuits):
                circ_id = f"c{circ_idx:03d}"
                for fam in mfs.FAMILIES:
                    for sev in mfs.SEVERITIES:
                        for obs in mfs.OBSERVABLES:
                            item_id = f"{seed}-{split}-{fam}-{sev}-{obs}-{circ_id}"

                            if g_mode == "independent_r":
                                r_val = float(rng.uniform(0.1, 0.9))
                                base_val = float(rng.uniform(0.1, 0.9))
                                y_val = 0.5 * base_val + 0.5 * r_val
                            else:
                                y_val = float(rng.uniform(0.1, 0.9))
                                r_val = y_val + float(rng.normal(0, 0.2))

                            items.append({
                                "item_id": item_id,
                                "circuit_id": circ_id,
                                "split": split,
                                "family": fam,
                                "noise_family": fam,
                                "severity": sev,
                                "observable": obs,
                                "ideal_expectation": float(y_val),
                                "noisy_expectation": float(r_val),
                            })

        i_hash = item_stream_hash(items)
        d_hash = split_dataset_hash(spec_hash="spec-test", items_hash=i_hash)
        dataset_hashes[seed] = d_hash

        item_dir = data_dir / f"regen-{key}"
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "items.jsonl").write_text(
            "".join(json.dumps(it) + "\n" for it in items), encoding="utf-8"
        )
        (item_dir / "manifest.json").write_text(
            json.dumps({"items_hash": i_hash, "split_spec_hash": "spec-test", "dataset_hash": d_hash}),
            encoding="utf-8",
        )

        rows = {
            split: [it for it in items if it["split"] == split]
            for split in ("validation", "test")
        }
        cache_path = cache_dir / f"{key}.pkl"
        with open(cache_path, "wb") as h:
            pickle.dump({"key": key, "dataset_hash": d_hash, **rows}, h)

        cache_sha = mfs.sha256_file(cache_path)
        caches_used[key] = {"dataset_hash": d_hash, "sha256": cache_sha}

        # Generate C fits and strong learner G fits
        rec_hash = (fit_hashes or {}).get(seed, d_hash)

        val_rows = rows["validation"]
        test_rows = rows["test"]
        y_val_arr = np.asarray([r["ideal_expectation"] for r in val_rows], dtype=float)
        y_test_arr = np.asarray([r["ideal_expectation"] for r in test_rows], dtype=float)

        for rung in rungs:
            for k in range(1, 21):
                # Imperfect C prediction
                if g_mode == "independent_r":
                    r_val_arr = np.asarray([r["noisy_expectation"] for r in val_rows], dtype=float)
                    r_test_arr = np.asarray([r["noisy_expectation"] for r in test_rows], dtype=float)
                    base_val_arr = (y_val_arr - 0.5 * r_val_arr) / 0.5
                    base_test_arr = (y_test_arr - 0.5 * r_test_arr) / 0.5

                    c_val = 0.6 * base_val_arr + 0.1 + rng.normal(0, 0.02, size=len(val_rows))
                    c_test = 0.6 * base_test_arr + 0.1 + rng.normal(0, 0.02, size=len(test_rows))
                    g_val = base_val_arr + rng.normal(0, 0.02, size=len(val_rows))
                    g_test = base_test_arr + rng.normal(0, 0.02, size=len(test_rows))
                else:
                    c_val = 0.5 * y_val_arr + 0.2 + rng.normal(0, 0.05, size=len(val_rows)) + 0.0001 * k
                    c_test = 0.5 * y_test_arr + 0.2 + rng.normal(0, 0.05, size=len(test_rows)) + 0.0001 * k

                    if g_mode == "exact":
                        g_val = y_val_arr.copy()
                        g_test = y_test_arr.copy()
                    elif g_mode == "noise":
                        g_val = rng.normal(0.0, 1.0, size=len(val_rows))
                        g_test = rng.normal(0.0, 1.0, size=len(test_rows))
                    else:
                        raise ValueError(f"Unknown g_mode: {g_mode}")

                # Save C fit
                stem_c = c_fits_dir / f"{key}__{rung}__k{k:02d}__C"
                np.savez(stem_c.with_suffix(".npz"), validation=c_val, test=c_test)

                # Compute family MAE
                v_fam_mae = {}
                t_fam_mae = {}
                for fam in mfs.FAMILIES:
                    v_idx = [i for i, r in enumerate(val_rows) if r["family"] == fam]
                    t_idx = [i for i, r in enumerate(test_rows) if r["family"] == fam]
                    v_fam_mae[fam] = float(np.mean(np.abs(c_val[v_idx] - y_val_arr[v_idx])))
                    t_fam_mae[fam] = float(np.mean(np.abs(c_test[t_idx] - y_test_arr[t_idx])))

                c_meta = {
                    "key": key,
                    "dataset_seed": seed,
                    "rung": rung,
                    "arm": "C",
                    "learner_seed": k,
                    "dataset_hash": rec_hash,
                    "validation_family_mae": v_fam_mae,
                    "test_family_mae": t_fam_mae,
                }
                stem_c.with_suffix(".json").write_text(json.dumps(c_meta), encoding="utf-8")

                # Save G fit
                stem_g = g_fits_dir / f"{key}__{rung}__k{k:02d}__C"
                np.savez(
                    stem_g.with_suffix(".npz"),
                    validation__poly5_ridge=g_val,
                    test__poly5_ridge=g_test,
                )
                g_hash = rec_hash
                g_seed = seed
                g_rung = rung
                g_arm = "C"
                if strong_fit_override:
                    if "dataset_hash" in strong_fit_override:
                        g_hash = strong_fit_override["dataset_hash"]
                    if "dataset_seed" in strong_fit_override:
                        g_seed = strong_fit_override["dataset_seed"]
                    if "rung" in strong_fit_override:
                        g_rung = strong_fit_override["rung"]
                    if "arm" in strong_fit_override:
                        g_arm = strong_fit_override["arm"]

                g_meta = {
                    "key": key,
                    "dataset_seed": g_seed,
                    "rung": g_rung,
                    "arm": g_arm,
                    "learner_seed": k,
                    "dataset_hash": g_hash,
                    "strong_learners": True,
                }
                stem_g.with_suffix(".json").write_text(json.dumps(g_meta), encoding="utf-8")

        for fam in mfs.FAMILIES:
            row_k = f"{key}/{fam}"
            rows_analysis_a[row_k] = {"rungs": {}}
            for rung in rungs:
                rows_analysis_a[row_k]["rungs"][rung] = {
                    "means": {
                        "C": {"point": 0.05},
                        "F": {"point": 0.04},
                        "R": {"point": 0.15},
                    }
                }

    analysis_a_path = root / "analysis-a.json"
    analysis_a_data = {
        "inputs": {"caches_used": caches_used},
        "parts": {"A": {"rows": rows_analysis_a}},
    }
    analysis_a_path.write_text(json.dumps(analysis_a_data), encoding="utf-8")

    return data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a_path, dataset_hashes


# ==============================================================================
# Requirement 1: Synthetic data where g equals y exactly
# ==============================================================================
def test_g_equals_y_recovers_recal_and_zero_inc_r_given_g(tmp_path, monkeypatch):
    """When g equals y exactly: stack_g has error 0, inc_g approx recal, inc_r|g approx 0."""
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a, _ = _create_synthetic_environment(
        tmp_path, g_mode="exact", n_circuits=15
    )

    cells, max_g_diff = mfs.compute_level_stacks(
        "exact-level",
        c_fits_dir,
        data_dir,
        cache_dir,
        g_fits_dir,
        ["R0"],
        stacking_all_set=None,
        check_reference=False,
    )

    assert max_g_diff == 0.0
    for cell_id, cell in cells.items():
        recal_mean = cell["recal"]["mean"]
        inc_g_pt = cell["inc_g"]["point"]
        inc_rg_pt = cell["inc_r_given_g"]["point"]
        stack_g_mean = cell["stack_g"]["mean"]

        # stack_g should have near-zero test error
        assert stack_g_mean < 1e-12
        # inc_g should approximately equal recal
        assert abs(inc_g_pt - recal_mean) < 1e-12
        # inc_r|g should be approximately zero
        assert abs(inc_rg_pt) < 1e-12


# ==============================================================================
# Requirement 2: Synthetic data where g is noise
# ==============================================================================
def test_g_is_noise_gives_zero_inc_g(tmp_path, monkeypatch):
    """When g is noise: adding g does not reduce error over recal, so inc_g approx 0."""
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a, _ = _create_synthetic_environment(
        tmp_path, g_mode="noise", n_circuits=25
    )

    cells, _ = mfs.compute_level_stacks(
        "noise-level",
        c_fits_dir,
        data_dir,
        cache_dir,
        g_fits_dir,
        ["R0"],
        stacking_all_set=None,
        check_reference=False,
    )

    for cell_id, cell in cells.items():
        inc_g_pt = cell["inc_g"]["point"]
        # inc_g should be near zero
        assert abs(inc_g_pt) < 0.01


# ==============================================================================
# Requirement 3: Synthetic data where r carries independent information
# ==============================================================================
def test_r_independent_information_gives_positive_inc_r_given_g(tmp_path, monkeypatch):
    """When r carries independent signal: stack_gr has lower error than stack_g, so inc_r|g > 0."""
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a, _ = _create_synthetic_environment(
        tmp_path, g_mode="independent_r", n_circuits=15
    )

    cells, _ = mfs.compute_level_stacks(
        "indep-level",
        c_fits_dir,
        data_dir,
        cache_dir,
        g_fits_dir,
        ["R0"],
        stacking_all_set=None,
        check_reference=False,
    )

    for cell_id, cell in cells.items():
        inc_rg_pt = cell["inc_r_given_g"]["point"]
        assert inc_rg_pt > 0.01


# ==============================================================================
# Requirement 4: The inc_r reproduction check fails on perturbed --stacking-all
# ==============================================================================
def test_reproduction_check_fails_on_perturbed_stacking_all(tmp_path, monkeypatch):
    """If --stacking-all records an increment_mean differing by > 1e-12, reproduction check raises SystemExit."""
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a, _ = _create_synthetic_environment(
        tmp_path, g_mode="exact", n_circuits=10
    )

    # First run without reproduction check to get ground truth numbers
    cells, _ = mfs.compute_level_stacks(
        "test-lvl",
        c_fits_dir,
        data_dir,
        cache_dir,
        g_fits_dir,
        ["R0"],
        stacking_all_set=None,
        check_reference=False,
    )

    # Build reference stacking-all dictionary
    stacking_all_set = {"cells": {}}
    for cell_id, cell in cells.items():
        row_rung = f"{cell['row']}/{cell['rung']}"
        stacking_all_set["cells"][row_rung] = {
            "increment_mean": cell["inc_r"]["point"],
            "increment_interval": cell["inc_r"]["interval"],
            "relative_increment": cell["inc_r"]["relative"],
        }

    # Verify unperturbed reproduces cleanly
    mfs.compute_level_stacks(
        "test-lvl",
        c_fits_dir,
        data_dir,
        cache_dir,
        g_fits_dir,
        ["R0"],
        stacking_all_set=stacking_all_set,
    )

    # Perturb one cell by 1e-5
    first_cell_key = next(iter(stacking_all_set["cells"]))
    stacking_all_set["cells"][first_cell_key]["increment_mean"] += 1e-5

    with pytest.raises(SystemExit, match="reproduction difference"):
        mfs.compute_level_stacks(
            "test-lvl",
            c_fits_dir,
            data_dir,
            cache_dir,
            g_fits_dir,
            ["R0"],
            stacking_all_set=stacking_all_set,
        )


FIELD_PATHS = (
    ("increment_mean",),
    ("increment_interval", "lower"),
    ("increment_interval", "upper"),
    ("relative_increment", "point"),
    ("relative_increment", "interval", "lower"),
    ("relative_increment", "interval", "upper"),
)


def _set_field(cell, path, value):
    for part in path[:-1]:
        cell = cell[part]
    cell[path[-1]] = value


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("path", FIELD_PATHS)
def test_nonfinite_reference_values_are_refused(tmp_path, monkeypatch, path, bad):
    """A NaN or infinite reference value fails preflight and the computation entry point."""
    import copy
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, _, _ = _create_synthetic_environment(
        tmp_path, g_mode="exact", n_circuits=10
    )
    cells, _ = mfs.compute_level_stacks(
        "test-lvl", c_fits_dir, data_dir, cache_dir, g_fits_dir, ["R0"],
        stacking_all_set=None, check_reference=False,
    )
    ref = {"cells": {}}
    for cell in cells.values():
        ref["cells"][f"{cell['row']}/{cell['rung']}"] = {
            "increment_mean": cell["inc_r"]["point"],
            "increment_interval": copy.deepcopy(cell["inc_r"]["interval"]),
            "relative_increment": copy.deepcopy(cell["inc_r"]["relative"]),
        }
    # Fill every row the preflight expects, then corrupt one field of one cell.
    first = next(iter(ref["cells"].values()))
    full = {"cells": {f"shipped-s{seed}-n640/{fam}/R0": copy.deepcopy(ref["cells"].get(
        f"shipped-s{seed}-n640/{fam}/R0", first)) for seed in mfs.SEEDS for fam in mfs.FAMILIES}}
    key = next(iter(ref["cells"]))
    _set_field(full["cells"][key], path, bad)
    with pytest.raises(SystemExit, match="not finite"):
        mfs.require_reference_sets({"sets": {"sweep-2048": full}}, ["2048"], ["R0"])
    _set_field(ref["cells"][key], path, bad)
    with pytest.raises(SystemExit, match="not finite"):
        mfs.compute_level_stacks(
            "test-lvl", c_fits_dir, data_dir, cache_dir, g_fits_dir, ["R0"], stacking_all_set=ref,
        )


def test_missing_reference_cell_stops_the_run(tmp_path, monkeypatch):
    """With the check on, an absent reference set or cell is refused rather than skipped."""
    monkeypatch.setattr(mfs, "BOOTSTRAP_DRAWS", 50)
    data_dir, cache_dir, c_fits_dir, g_fits_dir, _, _ = _create_synthetic_environment(
        tmp_path, g_mode="exact", n_circuits=10
    )
    for ref in (None, {"cells": {}}):
        with pytest.raises(SystemExit, match="reference cell"):
            mfs.compute_level_stacks(
                "test-lvl", c_fits_dir, data_dir, cache_dir, g_fits_dir, ["R0"], stacking_all_set=ref,
            )


# ==============================================================================
# Requirement 5: Identity mismatches refused
# ==============================================================================
def test_identity_mismatches_refused(tmp_path):
    """Check that mismatched dataset_hash, wrong arm, wrong rung, and bad shapes are rejected."""
    # 1. Strong fit wrong dataset_hash
    _, _, _, g_fits_dir, _, hashes = _create_synthetic_environment(
        tmp_path / "t1", strong_fit_override={"dataset_hash": "bad-hash"}, n_circuits=2
    )
    cache_counts = {seed: {"validation": 16, "test": 16} for seed in mfs.SEEDS}
    with pytest.raises(SystemExit, match="dataset_hash"):
        mfs.verify_strong_fits(g_fits_dir, ["R0"], {str(s): h for s, h in hashes.items()}, cache_counts)

    # 2. Strong fit wrong arm
    _, _, _, g_fits_dir2, _, hashes2 = _create_synthetic_environment(
        tmp_path / "t2", strong_fit_override={"arm": "F"}, n_circuits=2
    )
    with pytest.raises(SystemExit, match="arm"):
        mfs.verify_strong_fits(g_fits_dir2, ["R0"], {str(s): h for s, h in hashes2.items()}, cache_counts)

    # 3. Strong fit array shape mismatch
    _, _, _, g_fits_dir3, _, hashes3 = _create_synthetic_environment(tmp_path / "t3", n_circuits=2)
    bad_counts = {seed: {"validation": 999, "test": 999} for seed in mfs.SEEDS}
    with pytest.raises(SystemExit, match="validation__poly5_ridge shape"):
        mfs.verify_strong_fits(g_fits_dir3, ["R0"], {str(s): h for s, h in hashes3.items()}, bad_counts)


# ==============================================================================
# Requirement 6: Equal-spend arithmetic on hand-computed cases (including n*_cal <= 0)
# ==============================================================================
def test_equal_spend_arithmetic_hand_computed():
    """n*_R = 960 * 2048 / (ell - 2048); n*_cal = (960 * 2048 - 320 * ell) / (ell - 2048), signed."""
    assert es.compute_n_star_R(4096) == 960.0
    assert es.compute_n_star_cal(4096) == 320.0
    assert es.compute_n_star_R(6144) == 480.0
    assert es.compute_n_star_cal(6144) == 0.0
    # ell = 8192: R crosses at 320; R_cal's crossing is (1966080 - 2621440) / 6144 = -106.67.
    assert es.compute_n_star_R(8192) == 320.0
    assert math.isclose(es.compute_n_star_cal(8192), -655360.0 / 6144.0, rel_tol=1e-12)
    assert es.compute_n_star_R(32768) == 64.0
    assert es.compute_n_star_cal(32768) < 0.0
    assert math.isclose(es.compute_n_star_R(131072), 1966080.0 / 129024.0, rel_tol=1e-12)
    assert es.compute_n_star_cal(131072) < 0.0


def test_equal_spend_cost_direction():
    """The comparator is cheaper below a positive crossing and F above it."""
    below, above = es.shot_costs(8192, 100), es.shot_costs(8192, 400)
    assert below["F"] == 2170880.0 and below["R"] == 819200.0
    assert above["F"] == 2785280.0 and above["R"] == 3276800.0
    assert below["R"] < below["F"] and above["R"] > above["F"]
    # R_cal at 4096 crosses at n* = 320.
    assert es.shot_costs(4096, 319)["R_cal"] < es.shot_costs(4096, 319)["F"]
    assert es.shot_costs(4096, 321)["R_cal"] > es.shot_costs(4096, 321)["F"]
    # A negative crossing: F costs fewer shots at every nonnegative count.
    for ell in (8192, 32768, 131072):
        for n in (0, 1, 10, 10_000):
            assert es.shot_costs(ell, n)["R_cal"] > es.shot_costs(ell, n)["F"]
    assert "0 <= n < n*" in es.COST_DIRECTION


def test_equal_spend_tool_execution(tmp_path):
    """End-to-end execution of equal_spend.py on synthetic data."""
    data_dir, cache_dir, _, _, analysis_a_path, _ = _create_synthetic_environment(
        tmp_path / "env", n_circuits=4
    )

    # Build dummy predictions.json
    rule_cells = {}
    for lvl in ("256", "2048", "8192", "exact"):
        for seed in mfs.SEEDS:
            key = f"shipped-s{seed}-n640"
            for fam in mfs.FAMILIES:
                row_key = f"{key}/{fam}"
                for s in mfs.SEVERITIES:
                    for o in mfs.OBSERVABLES:
                        # calibrated error: at 8192 shots let it be 0.03 (< F_2048 which is 0.04)
                        c_r = 0.03 if lvl == "8192" else 0.08
                        rule_cells[f"{lvl}/{row_key}/{s}/{o}"] = {"c_r": c_r}

    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps({"rule_cells": rule_cells}), encoding="utf-8")

    out_json = tmp_path / "out_equal_spend.json"

    argv = [
        "--analysis-a", f"2048={analysis_a_path}", f"8192={analysis_a_path}", f"exact={analysis_a_path}",
        "--predictions", str(pred_path),
        "--data", f"2048={data_dir}", f"8192={data_dir}", f"exact={data_dir}",
        "--cache", f"2048={cache_dir}", f"8192={cache_dir}", f"exact={cache_dir}",
        "--rungs", "R0",
        "--out", str(out_json),
    ]

    ret = es.main(argv)
    assert ret == 0
    assert out_json.exists()

    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["schema"] == "equal-spend-v1"
    assert "shipped-s101-n640/tfi/R0" in result["cells"]
    cell = result["cells"]["shipped-s101-n640/tfi/R0"]
    assert cell["F_2048"] == 0.04
    # At 8192, R_cal was 0.03 <= 0.04: its crossing is negative, so F costs fewer shots at every n.
    assert cell["levels"]["8192"]["R_cal_le_F_2048"] is True
    assert math.isclose(cell["levels"]["8192"]["n_star_cal"], -655360.0 / 6144.0, rel_tol=1e-12)
    assert cell["levels"]["8192"]["F_no_costlier_at_every_n_cal"] is True
    assert result["cost_direction"] == es.COST_DIRECTION
    assert cell["smallest_finite_level_R_cal"] == 8192


def test_dry_run_modes_for_both_tools(tmp_path):
    """Test --dry-run on both tools."""
    data_dir, cache_dir, c_fits_dir, g_fits_dir, analysis_a, _ = _create_synthetic_environment(
        tmp_path / "dry_env", n_circuits=2
    )
    ref_value = {"increment_mean": 0.0, "increment_interval": {"lower": 0.0, "upper": 0.0},
                 "relative_increment": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}}}
    full_set = {"cells": {f"shipped-s{seed}-n640/{fam}/R0": ref_value
                          for seed in mfs.SEEDS for fam in mfs.FAMILIES}}
    stacking_all_path = tmp_path / "stacking_all.json"
    stacking_all_path.write_text(json.dumps({"sets": {"sweep-2048": full_set}}), encoding="utf-8")

    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps({"rule_cells": {}}), encoding="utf-8")

    out_mfs = tmp_path / "dry_mfs.json"
    out_es = tmp_path / "dry_es.json"

    # Tool 1 dry run
    ret1 = mfs.main([
        "--c-fits", f"2048={c_fits_dir}",
        "--data", f"2048={data_dir}",
        "--cache", f"2048={cache_dir}",
        "--analysis-a", f"2048={analysis_a}",
        "--g-fits", str(g_fits_dir),
        "--stacking-all", str(stacking_all_path),
        "--rungs", "R0",
        "--out", str(out_mfs),
        "--dry-run",
    ])
    assert ret1 == 0
    assert out_mfs.exists()
    d1 = json.loads(out_mfs.read_text(encoding="utf-8"))
    assert "dry-run" in d1["status"]

    # The dry run also refuses a reference file without the level's set or one of its cells.
    partial = {"cells": dict(list(full_set["cells"].items())[1:])}
    for sets in ({}, {"sweep-2048": partial}):
        bad_path = tmp_path / "stacking_bad.json"
        bad_path.write_text(json.dumps({"sets": sets}), encoding="utf-8")
        with pytest.raises(SystemExit, match="reference values"):
            mfs.main([
                "--c-fits", f"2048={c_fits_dir}", "--data", f"2048={data_dir}",
                "--cache", f"2048={cache_dir}", "--analysis-a", f"2048={analysis_a}",
                "--g-fits", str(g_fits_dir), "--stacking-all", str(bad_path),
                "--rungs", "R0", "--dry-run",
            ])

    # Tool 2 dry run
    ret2 = es.main([
        "--analysis-a", f"2048={analysis_a}",
        "--predictions", str(pred_path),
        "--data", f"2048={data_dir}",
        "--cache", f"2048={cache_dir}",
        "--rungs", "R0",
        "--out", str(out_es),
        "--dry-run",
    ])
    assert ret2 == 0
    assert out_es.exists()
    d2 = json.loads(out_es.read_text(encoding="utf-8"))
    assert "dry-run" in d2["status"]
