"""Tests for near-Clifford constant references and generalized descriptor_ladder_posthoc CLI.

Tests:
1. Toy-data verification of near-Clifford constant-zero error, training-median error,
   and bootstrap contrasts (zero - F, zero - C, (zero - F)/zero).
2. Generalized CLI verification for descriptor_ladder_posthoc: both positional and
   flag-based (--fits, --cache, --out) invocations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools import descriptor_ladder_analysis as dla
from tools import descriptor_ladder_posthoc as posthoc
from tools import near_clifford_constant_reference as nc_ref


def _make_toy_cache(
    key: str,
    dataset_seed: int,
    cells: list[tuple[str, str, str]],
    n_circuits: int = 4,
    train_per_cell: int = 10,
    train_offset: float = 0.2,
) -> dict:
    rng = np.random.default_rng(dataset_seed)
    train_rows = []
    for c_idx in range(train_per_cell):
        for nf, sev, obs in cells:
            ideal = float(rng.uniform(-0.5, 0.5)) + train_offset
            train_rows.append({
                "item_id": f"train-{dataset_seed}-{c_idx}-{sev}-{obs}",
                "circuit_id": f"train-circ-{c_idx}",
                "family": "near_clifford",
                "noise_family": nf,
                "severity": sev,
                "observable": obs,
                "ideal_expectation": ideal,
                "noisy_expectation": 0.8 * ideal,
                "split": "train",
            })

    test_rows = []
    for c_idx in range(n_circuits):
        circuit = f"circ-{dataset_seed}-{c_idx:03d}"
        for nf, sev, obs in cells:
            ideal = float(rng.uniform(-0.8, 0.8))
            test_rows.append({
                "item_id": f"test-{dataset_seed}-{c_idx}-{sev}-{obs}",
                "circuit_id": circuit,
                "family": "near_clifford",
                "noise_family": nf,
                "severity": sev,
                "observable": obs,
                "ideal_expectation": ideal,
                "noisy_expectation": 0.7 * ideal,
                "split": "test",
            })

    return {
        "key": key,
        "seed": dataset_seed,
        "dataset_hash": f"synthetic-{key}",
        "train": train_rows,
        "test": test_rows,
    }


def _write_toy_fit(
    fit_dir: Path,
    key: str,
    rung: str,
    arm: str,
    seed: int,
    rows: list[dict],
    error_const: float,
) -> None:
    tag = f"k{seed:02d}"
    stem = f"{key}__{rung}__{tag}__{arm}"
    preds = np.empty(len(rows))
    for i, r in enumerate(rows):
        sign = 1.0 if i % 2 == 0 else -1.0
        preds[i] = float(r["ideal_expectation"]) + sign * error_const
    np.savez_compressed(fit_dir / f"{stem}.npz", test=preds)

    cells = sorted({tuple(str(r[f]) for f in dla.CELL_FIELDS) for r in rows})
    maes = []
    for cell in cells:
        members = [i for i, r in enumerate(rows)
                   if tuple(str(r[f]) for f in dla.CELL_FIELDS) == cell]
        maes.append(
            math.fsum(abs(preds[i] - rows[i]["ideal_expectation"]) for i in members)
            / len(members)
        )
    family_mae = {"near_clifford": math.fsum(maes) / len(maes)}
    meta = {
        "schema": "descriptor-information-fit-v1",
        "key": key,
        "rung": rung,
        "arm": arm,
        "learner_seed": seed,
        "selected_model": "mlp",
        "test_family_mae": family_mae,
    }
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_near_clifford_constant_reference_toy_data(tmp_path: Path):
    """Test compute_constant_references on a synthetic fixture with known math."""
    cache_dir = tmp_path / "cache"
    orig_fits_dir = tmp_path / "orig_fits"
    str_fits_dir = tmp_path / "str_fits"
    cache_dir.mkdir()
    orig_fits_dir.mkdir()
    str_fits_dir.mkdir()

    cells = [("depolarizing_readout", sev, "zz_mid") for sev in ("L1", "L3")]
    seeds = (101, 211, 307)
    learner_seeds = tuple(range(1, 21))

    toy_caches = {}
    for s in seeds:
        key = f"nc-s{s}-n640"
        cache_data = _make_toy_cache(key, s, cells, n_circuits=4, train_offset=0.1)
        toy_caches[key] = cache_data
        with open(cache_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(cache_data, f)

        # Original fits: F error 0.02, C error 0.08 at NC-R0; NC-none M error 0.015, C 0.07
        for ls in learner_seeds:
            _write_toy_fit(orig_fits_dir, key, "NC-R0", "F", ls, cache_data["test"], 0.02)
            _write_toy_fit(orig_fits_dir, key, "NC-R0", "C", ls, cache_data["test"], 0.08)
            _write_toy_fit(orig_fits_dir, key, "NC-R0", "P", ls, cache_data["test"], 0.04)
            _write_toy_fit(orig_fits_dir, key, "NC-none", "M", ls, cache_data["test"], 0.015)
            _write_toy_fit(orig_fits_dir, key, "NC-none", "C", ls, cache_data["test"], 0.07)

            # Strength fits: F error 0.018, C error 0.075 at NC-R0; NC-none M error 0.012, C 0.065
            _write_toy_fit(str_fits_dir, key, "NC-R0", "F", ls, cache_data["test"], 0.018)
            _write_toy_fit(str_fits_dir, key, "NC-R0", "C", ls, cache_data["test"], 0.075)
            _write_toy_fit(str_fits_dir, key, "NC-R0", "P", ls, cache_data["test"], 0.035)
            _write_toy_fit(str_fits_dir, key, "NC-none", "M", ls, cache_data["test"], 0.012)
            _write_toy_fit(str_fits_dir, key, "NC-none", "C", ls, cache_data["test"], 0.065)

    out_json = tmp_path / "constant_ref.json"
    result = nc_ref.compute_constant_references(
        original_fit_dirs=[orig_fits_dir],
        strength_fit_dirs=[str_fits_dir],
        cache_dirs=[cache_dir],
        full_cache_dirs=[cache_dir],
        out_path=out_json,
        draws=100,  # small draws for quick test
        seed=20261002,
        verbose=False,
    )

    assert out_json.exists()
    assert result["schema"] == "near-clifford-posthoc-constant-reference-v1"
    assert len(result["datasets"]) == 3

    for s in seeds:
        key = f"nc-s{s}-n640"
        ds = result["datasets"][key]
        cache_data = toy_caches[key]
        test_rows = cache_data["test"]
        train_rows = cache_data["train"]

        # 1. Check constant zero point
        exp_zero = dla.point_macro(
            np.abs([r["ideal_expectation"] for r in test_rows]),
            [np.array([i for i, r in enumerate(test_rows) if (r["noise_family"], r["severity"], r["observable"]) == c]) for c in cells],
            tuple(range(len(cells))),
        )
        assert abs(ds["constant_zero"]["point"] - exp_zero) < 1e-12
        interval = ds["constant_zero"]["interval"]
        assert interval["lower"] <= ds["constant_zero"]["point"] <= interval["upper"]

        # 2. Check training median
        train_by_cell = {}
        for r in train_rows:
            c = (r["noise_family"], r["severity"], r["observable"])
            train_by_cell.setdefault(c, []).append(r["ideal_expectation"])
        cell_medians = {c: float(np.median(vals)) for c, vals in train_by_cell.items()}
        test_preds = np.array([cell_medians[(r["noise_family"], r["severity"], r["observable"])] for r in test_rows])
        exp_med = dla.point_macro(
            np.abs(test_preds - [r["ideal_expectation"] for r in test_rows]),
            [np.array([i for i, r in enumerate(test_rows) if (r["noise_family"], r["severity"], r["observable"]) == c]) for c in cells],
            tuple(range(len(cells))),
        )
        assert abs(ds["training_median"]["point"] - exp_med) < 1e-12

        # 3. Check fits & contrasts
        for fit_set in ("original", "strength_indicator"):
            for rung in ("NC-R0", "NC-none"):
                r_data = ds["fits"][fit_set][rung]
                zf = r_data["zero_minus_F"]
                zc = r_data["zero_minus_C"]
                ratio = r_data["ratio_zero_minus_F_over_zero"]

                assert abs(zf["point"] - (ds["constant_zero"]["point"] - r_data["mean_F"])) < 1e-12
                assert abs(zc["point"] - (ds["constant_zero"]["point"] - r_data["mean_C"])) < 1e-12
                assert abs(ratio["point"] - (zf["point"] / ds["constant_zero"]["point"])) < 1e-12
                assert zf["interval"]["lower"] <= zf["interval"]["upper"]
                assert zc["interval"]["lower"] <= zc["interval"]["upper"]
                assert ratio["interval"]["lower"] <= ratio["interval"]["upper"]


def test_generalized_posthoc_cli_both_forms(tmp_path: Path):
    """Test that descriptor_ladder_posthoc CLI accepts both positional and flag forms."""
    # Build toy QAOA dataset and fits
    qaoa_root = tmp_path / "panelB"
    fits_dir = qaoa_root / "fits"
    cache_dir = qaoa_root / "cache"
    fits_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    qaoa_cells = [("depolarizing_readout", sev, "zz_mid") for sev in ("L1", "L3")]
    seeds = (101, 211, 307)
    learner_seeds = tuple(range(1, 21))

    for s in seeds:
        key = f"qaoa-s{s}-n640"
        cache_data = _make_toy_cache(key, s, qaoa_cells, n_circuits=4)
        with open(cache_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(cache_data, f)
        for rung in ("B-partial", "B-complete", "B-none"):
            for ls in learner_seeds:
                _write_toy_fit(fits_dir, key, rung, "F", ls, cache_data["test"], 0.02)
                _write_toy_fit(fits_dir, key, rung, "C", ls, cache_data["test"], 0.05)
                _write_toy_fit(fits_dir, key, rung, "P", ls, cache_data["test"], 0.03)

    out_flat = tmp_path / "out_flat.json"
    out_flags = tmp_path / "out_flags.json"
    out_runs = tmp_path / "out_runs.json"

    # Form 1A: Positional with --flat
    ret_flat = posthoc.main(["--flat", str(qaoa_root), str(out_flat), "--quiet"])
    assert ret_flat == 0
    assert out_flat.exists()

    # Form 2: Flag-based with --fits, --cache, and --out
    ret_flags = posthoc.main([
        "--fits", str(fits_dir),
        "--cache", str(cache_dir),
        "--out", str(out_flags),
        "--quiet",
    ])
    assert ret_flags == 0
    assert out_flags.exists()

    # Form 1B: Positional without --flat (runs directory holding partB/fits and partB/cache)
    runs_dir = tmp_path / "runs"
    runs_b_fits = runs_dir / "partB" / "fits"
    runs_b_cache = runs_dir / "partB" / "cache"
    runs_b_fits.mkdir(parents=True)
    runs_b_cache.mkdir(parents=True)
    for p in fits_dir.glob("*"):
        (runs_b_fits / p.name).write_bytes(p.read_bytes())
    for p in cache_dir.glob("*"):
        (runs_b_cache / p.name).write_bytes(p.read_bytes())

    ret_runs = posthoc.main([str(runs_dir), str(out_runs), "--quiet"])
    assert ret_runs == 0
    assert out_runs.exists()

    # Verify that all outputs match
    data_flat = json.loads(out_flat.read_text(encoding="utf-8"))
    data_flags = json.loads(out_flags.read_text(encoding="utf-8"))
    data_runs = json.loads(out_runs.read_text(encoding="utf-8"))

    assert data_flat["cells"] == data_flags["cells"]
    assert data_flat["cells"] == data_runs["cells"]


def test_posthoc_part_a_rows(tmp_path: Path):
    """--part A gives one F - R and C - R entry per dataset-seed and family row and fitted rung."""
    fits_dir = tmp_path / "fits"
    cache_dir = tmp_path / "cache"
    fits_dir.mkdir()
    cache_dir.mkdir()
    cells = [("depolarizing_readout", sev, obs) for sev in ("L1", "L3") for obs in ("z_mid", "zz_mid")]
    rungs = ("R0", "N1", "N2", "R5")
    for s in (101, 211, 307):
        key = f"shipped-s{s}-n640"
        data = _make_toy_cache(key, s, cells, n_circuits=4)
        for i, row in enumerate(data["test"]):
            row["family"] = "tfi" if (i // len(cells)) % 2 == 0 else "heisenberg"
        with open(cache_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(data, f)
        for rung in rungs:
            for ls in range(1, 21):
                for arm, err in (("F", 0.02), ("C", 0.05), ("P", 0.03)):
                    _write_toy_fit(fits_dir, key, rung, arm, ls, data["test"], err)
                    meta_path = fits_dir / f"{key}__{rung}__k{ls:02d}__{arm}.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["test_family_mae"] = {"tfi": err, "heisenberg": err}
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "part-a.json"
    assert posthoc.main(["--fits", str(fits_dir), "--cache", str(cache_dir),
                         "--part", "A", "--out", str(out), "--quiet"]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["part"] == "A"
    expected = {f"shipped-s{s}-n640/{fam}/{rung}" for s in (101, 211, 307)
                for fam in ("tfi", "heisenberg") for rung in rungs}
    assert set(result["cells"]) == expected
    for entry in result["cells"].values():
        assert entry["F_minus_R"]["point"] == pytest.approx(entry["F_point"] - entry["R_point"])
        assert entry["F_point"] == pytest.approx(0.02)
