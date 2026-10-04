#!/usr/bin/env python3
"""Post hoc analysis: strongest descriptor-only reference at each descriptor rung
and stacking increment of the noisy estimate over it (Part A spin chains).

Post hoc, descriptive analysis:
Evaluates descriptor-only references that do not read the noisy estimate r across
Part A spin chain cells (dataset seeds 101, 211, 307; families TFI and Heisenberg;
rungs R0, N1..N4, R3-TFI/Heis, R4, R5). For each cell (severity x observable):
- HistGradientBoostingRegressor on the rung's descriptor columns (excluding r),
  selecting among a fixed grid (max_iter in {200, 500}, learning_rate in {0.05, 0.1},
  max_leaf_nodes in {15, 31}; random_state 0) by validation MAE, fitted on train only.
- At R0, also includes the degree-five coupling polynomial of
  tools/refit_polynomial_diagnostic.py fitted per cell on train rows of the regenerated
  caches; compares validation macro MAE and designates the one with lower validation
  macro MAE as strongest reference.
- Scores each reference on test (macro MAE over the rule's cells, equal weights).
- Computes stacking increment of r over the strongest reference: fits y ~ a + b*ref
  and y ~ a + b*ref + c*r on validation rows, applies to test, and evaluates increment
  = recalibrated macro MAE minus stacked macro MAE.
- Calculates uncertainty via a one-stage percentile bootstrap over test circuits
  (10,000 draws, numpy.random.default_rng(20261002), circuit weights as in macro_under_weights).

Outputs: artifacts/descriptor-information/posthoc-strongest-reference.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure single-thread BLAS/OpenMP per worker process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

# Ensure tools and repo root are importable
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

from tools import descriptor_ladder as dl  # noqa: E402
from tools import descriptor_ladder_analysis as dla  # noqa: E402

RULE_SEED = 20261002
BOOTSTRAP_DRAWS = 10_000
SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
RUNGS_PER_FAMILY = {
    "tfi": ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R4", "R5"),
    "heisenberg": ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R4", "R5"),
}
POLY_FEATURES = {
    "heisenberg": ("jx", "jy", "jz"),
    "tfi": ("j", "h"),
}

GRID = [
    {"max_iter": mi, "learning_rate": lr, "max_leaf_nodes": mln, "random_state": 0}
    for mi in (200, 500)
    for lr in (0.05, 0.1)
    for mln in (15, 31)
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_circuit_counts(n_circuits: int, draws: int = BOOTSTRAP_DRAWS,
                             seed: int = RULE_SEED) -> np.ndarray:
    """One-stage bootstrap resample counts over test circuits."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, n_circuits), dtype=float)
    for b in range(draws):
        counts[b] = np.bincount(
            rng.integers(0, n_circuits, n_circuits), minlength=n_circuits
        )
    counts.flags.writeable = False
    return counts


def fit_polynomial_5(x_tr: np.ndarray, y_tr: np.ndarray,
                     x_val: np.ndarray, y_val: np.ndarray,
                     x_te: np.ndarray, y_te: np.ndarray) -> dict[str, Any]:
    """Fit degree-five polynomial in couplings alone on train rows."""
    center = x_tr.mean(axis=0)
    scale = x_tr.std(axis=0)
    assert np.all(scale > 0), "Zero scale encountered in coupling features"
    poly = PolynomialFeatures(5, include_bias=True)
    design_tr = poly.fit_transform((x_tr - center) / scale)
    beta, _, rank, _ = np.linalg.lstsq(design_tr, y_tr, rcond=None)
    with np.errstate(all="ignore"):
        val_pred = poly.transform((x_val - center) / scale) @ beta
        te_pred = poly.transform((x_te - center) / scale) @ beta
    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    te_mae = float(np.mean(np.abs(te_pred - y_te)))
    return {
        "val_mae": val_mae,
        "test_mae": te_mae,
        "val_pred": val_pred,
        "test_pred": te_pred,
        "n_params": int(design_tr.shape[1]),
    }


def fit_hgbr_cell(X_tr: np.ndarray, y_tr: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_te: np.ndarray, y_te: np.ndarray) -> dict[str, Any]:
    """Fit HistGradientBoostingRegressor over fixed grid selecting by validation MAE."""
    best_val_mae = float("inf")
    best_params = None
    best_model = None

    for params in GRID:
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_tr, y_tr)
        p_val = m.predict(X_val)
        val_mae = float(np.mean(np.abs(p_val - y_val)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = dict(params)
            best_model = m

    val_pred = best_model.predict(X_val)
    te_pred = best_model.predict(X_te)
    te_mae = float(np.mean(np.abs(te_pred - y_te)))
    return {
        "best_params": best_params,
        "val_mae": best_val_mae,
        "test_mae": te_mae,
        "val_pred": val_pred,
        "test_pred": te_pred,
    }


def _worker_fit_cell(pkg: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point for fitting reference model(s) for a single cell."""
    X_tr = pkg["X_tr"]
    y_tr = pkg["y_tr"]
    X_val = pkg["X_val"]
    y_val = pkg["y_val"]
    X_te = pkg["X_te"]
    y_te = pkg["y_te"]
    is_r0 = pkg["is_r0"]

    hgbr_res = fit_hgbr_cell(X_tr, y_tr, X_val, y_val, X_te, y_te)

    poly_res = None
    if is_r0:
        poly_res = fit_polynomial_5(
            pkg["x_tr_poly"], y_tr, pkg["x_val_poly"], y_val, pkg["x_te_poly"], y_te
        )

    return {
        "cell_idx": pkg["cell_idx"],
        "severity": pkg["severity"],
        "observable": pkg["observable"],
        "hgbr": hgbr_res,
        "poly5": poly_res,
    }


def compute_stacking_and_interval(
    val_preds_by_cell: list[np.ndarray],
    y_val_by_cell: list[np.ndarray],
    r_val_by_cell: list[np.ndarray],
    test_preds_by_cell: list[np.ndarray],
    y_te_by_cell: list[np.ndarray],
    r_te_by_cell: list[np.ndarray],
    members: list[np.ndarray],
    item_circuit: np.ndarray,
    circuit_counts: np.ndarray,
) -> dict[str, Any]:
    """Fit per-cell recalibration and stacking models on validation, evaluate test increment and bootstrap CI."""
    n_test_total = sum(len(y) for y in y_te_by_cell)
    p_rec_test_full = np.empty(n_test_total, dtype=float)
    p_st_test_full = np.empty(n_test_total, dtype=float)
    y_test_full = np.empty(n_test_total, dtype=float)

    cell_stacking_details = []

    for c_i in range(len(members)):
        v_ref = val_preds_by_cell[c_i]
        v_y = y_val_by_cell[c_i]
        v_r = r_val_by_cell[c_i]

        t_ref = test_preds_by_cell[c_i]
        t_y = y_te_by_cell[c_i]
        t_r = r_te_by_cell[c_i]
        t_idx = members[c_i]

        y_test_full[t_idx] = t_y

        # Recalibration fit: y ~ a + b * ref
        X_rec_v = np.column_stack([np.ones_like(v_ref), v_ref])
        X_rec_t = np.column_stack([np.ones_like(t_ref), t_ref])
        b_rec, _, _, _ = np.linalg.lstsq(X_rec_v, v_y, rcond=None)
        p_rec = X_rec_t @ b_rec
        p_rec_test_full[t_idx] = p_rec

        # Stacking fit: y ~ a + b * ref + c * r
        X_st_v = np.column_stack([np.ones_like(v_ref), v_ref, v_r])
        X_st_t = np.column_stack([np.ones_like(t_ref), t_ref, t_r])
        b_st, _, _, _ = np.linalg.lstsq(X_st_v, v_y, rcond=None)
        p_st = X_st_t @ b_st
        p_st_test_full[t_idx] = p_st

        c_rec_mae = float(np.mean(np.abs(p_rec - t_y)))
        c_st_mae = float(np.mean(np.abs(p_st - t_y)))
        c_ref_mae = float(np.mean(np.abs(t_ref - t_y)))

        cell_stacking_details.append({
            "recal_intercept": float(b_rec[0]),
            "recal_slope": float(b_rec[1]),
            "stack_intercept": float(b_st[0]),
            "stack_ref_coeff": float(b_st[1]),
            "stack_r_coeff": float(b_st[2]),
            "ref_test_mae": c_ref_mae,
            "recal_test_mae": c_rec_mae,
            "stack_test_mae": c_st_mae,
            "cell_increment": c_rec_mae - c_st_mae,
        })

    err_rec = np.abs(p_rec_test_full - y_test_full)
    err_st = np.abs(p_st_test_full - y_test_full)

    cells = tuple(range(len(members)))
    rec_macro_mae = dla.point_macro(err_rec, members, cells)
    stack_macro_mae = dla.point_macro(err_st, members, cells)
    increment_point = rec_macro_mae - stack_macro_mae

    # One-stage bootstrap over test circuits
    rec_draws = dla.macro_under_weights(
        err_rec[None, :], item_circuit, members, cells, circuit_counts
    )[:, 0]
    stack_draws = dla.macro_under_weights(
        err_st[None, :], item_circuit, members, cells, circuit_counts
    )[:, 0]
    incr_draws = rec_draws - stack_draws
    ci_lo, ci_hi = np.percentile(incr_draws, (2.5, 97.5))

    return {
        "recal_macro_mae": rec_macro_mae,
        "stack_macro_mae": stack_macro_mae,
        "increment_point": increment_point,
        "increment_ci_95": {"lower": float(ci_lo), "upper": float(ci_hi)},
        "excludes_zero_above": bool(ci_lo > 0.0),
        "excludes_zero_below": bool(ci_hi < 0.0),
        "cell_details": cell_stacking_details,
    }


def run_strongest_reference_analysis(
    ladder_dir: Path,
    analysis_a_path: Path,
    analysis_a_strength_path: Path,
    out_json_path: Path,
    n_workers: int = 8,
) -> dict[str, Any]:
    """Execute the full strongest-reference and stacking increment analysis."""
    t0 = time.time()
    circuit_counts = generate_circuit_counts(160, BOOTSTRAP_DRAWS, RULE_SEED)

    analysis_a = json.loads(analysis_a_path.read_text(encoding="utf-8"))["parts"]["A"]
    analysis_a_str = json.loads(analysis_a_strength_path.read_text(encoding="utf-8"))["parts"]["A"]

    # Verify R0 polynomial refit baseline
    poly_summary_path = _REPO / "artifacts" / "polynomial-refit" / "polynomial_refit_summary.json"
    poly_summary = json.loads(poly_summary_path.read_text(encoding="utf-8"))["primary_evaluations"]
    poly_ref_lookup = {
        (item["seed"], item["family"]): item for item in poly_summary
    }

    results_by_cell = {}
    total_combinations = sum(len(RUNGS_PER_FAMILY[fam]) for fam in FAMILIES) * len(SEEDS)
    print(f"Starting Analysis 1 across {total_combinations} Part A cells using {n_workers} workers...")

    # Load caches once per dataset seed
    caches = {}
    for seed in SEEDS:
        cache_path = ladder_dir / "cache" / f"shipped-s{seed}-n640.pkl"
        caches[seed] = dl.load_cache(cache_path)

    for seed in SEEDS:
        data = caches[seed]
        key = f"shipped-s{seed}-n640"

        for fam in FAMILIES:
            row_key = f"{key}/{fam}"
            row_meta = analysis_a["rows"][row_key]
            row_meta_str = analysis_a_str["rows"][row_key]

            # Test circuit structure
            row_test_items = [r for r in data["test"] if r["family"] == fam]
            circuits = sorted({str(r["circuit_id"]) for r in row_test_items})
            circ_pos = {c: i for i, c in enumerate(circuits)}
            n_circ = len(circuits)
            assert n_circ == 160, f"Expected 160 test circuits, got {n_circ}"
            item_circuit = np.asarray([circ_pos[str(r["circuit_id"])] for r in row_test_items], dtype=int)

            members = [
                np.asarray(
                    [i for i, r in enumerate(row_test_items) if r["severity"] == s and r["observable"] == o],
                    dtype=int,
                )
                for s, o in CELLS_DEF
            ]

            rungs = RUNGS_PER_FAMILY[fam]
            for rung in rungs:
                cell_key = f"{row_key}/{rung}"
                builder, names = dl.builder_for(ladder_dir, key, rung, data)
                assert names[0] == "noisy_expectation", f"First feature is {names[0]}, expected noisy_expectation"
                feat_indices = [i for i, name in enumerate(names) if name != "noisy_expectation"]

                is_r0 = (rung == "R0")
                poly_features = POLY_FEATURES[fam]

                packages = []
                for c_i, (sev, obs) in enumerate(CELLS_DEF):
                    tr_items = [
                        r for r in data["train"]
                        if r["family"] == fam and r["severity"] == sev and r["observable"] == obs
                    ]
                    val_items = [
                        r for r in data["validation"]
                        if r["family"] == fam and r["severity"] == sev and r["observable"] == obs
                    ]
                    te_idx = members[c_i]
                    te_items = [row_test_items[i] for i in te_idx]

                    X_tr = np.array([[builder(r)[i] for i in feat_indices] for r in tr_items], dtype=np.float64)
                    y_tr = np.array([r["ideal_expectation"] for r in tr_items], dtype=np.float64)
                    X_val = np.array([[builder(r)[i] for i in feat_indices] for r in val_items], dtype=np.float64)
                    y_val = np.array([r["ideal_expectation"] for r in val_items], dtype=np.float64)
                    X_te = np.array([[builder(r)[i] for i in feat_indices] for r in te_items], dtype=np.float64)
                    y_te = np.array([r["ideal_expectation"] for r in te_items], dtype=np.float64)

                    pkg: dict[str, Any] = {
                        "cell_idx": c_i,
                        "severity": sev,
                        "observable": obs,
                        "X_tr": X_tr,
                        "y_tr": y_tr,
                        "X_val": X_val,
                        "y_val": y_val,
                        "X_te": X_te,
                        "y_te": y_te,
                        "is_r0": is_r0,
                    }

                    if is_r0:
                        pkg["x_tr_poly"] = np.array([[r[f] for f in poly_features] for r in tr_items], dtype=np.float64)
                        pkg["x_val_poly"] = np.array([[r[f] for f in poly_features] for r in val_items], dtype=np.float64)
                        pkg["x_te_poly"] = np.array([[r[f] for f in poly_features] for r in te_items], dtype=np.float64)

                    packages.append(pkg)

                # Fit rule cells
                cell_fits = []
                with ProcessPoolExecutor(max_workers=min(n_workers, len(packages))) as executor:
                    for res in executor.map(_worker_fit_cell, packages):
                        cell_fits.append(res)
                cell_fits.sort(key=lambda c: c["cell_idx"])

                # Determine strongest reference
                if is_r0:
                    val_macro_hgbr = float(np.mean([c["hgbr"]["val_mae"] for c in cell_fits]))
                    test_macro_hgbr = float(np.mean([c["hgbr"]["test_mae"] for c in cell_fits]))
                    val_macro_poly5 = float(np.mean([c["poly5"]["val_mae"] for c in cell_fits]))
                    test_macro_poly5 = float(np.mean([c["poly5"]["test_mae"] for c in cell_fits]))

                    strongest_ref_name = "poly5" if val_macro_poly5 < val_macro_hgbr else "hgbr"

                    # Verify polynomial refit agreement
                    expected_poly_ref = poly_ref_lookup[(seed, fam)]
                    expected_poly_test = expected_poly_ref["refit_poly5_mae"]
                    diff_poly = abs(test_macro_poly5 - expected_poly_test)
                    assert diff_poly < 1e-12, (
                        f"R0 polynomial test error mismatch for s{seed} {fam}: "
                        f"computed {test_macro_poly5}, expected {expected_poly_test}, diff {diff_poly}"
                    )
                else:
                    val_macro_hgbr = float(np.mean([c["hgbr"]["val_mae"] for c in cell_fits]))
                    test_macro_hgbr = float(np.mean([c["hgbr"]["test_mae"] for c in cell_fits]))
                    val_macro_poly5 = None
                    test_macro_poly5 = None
                    strongest_ref_name = "hgbr"

                # Extract predictions for stacking
                val_preds_list = []
                y_val_list = []
                r_val_list = []
                test_preds_list = []
                y_te_list = []
                r_te_list = []

                for c_i, (sev, obs) in enumerate(CELLS_DEF):
                    val_items = [
                        r for r in data["validation"]
                        if r["family"] == fam and r["severity"] == sev and r["observable"] == obs
                    ]
                    te_idx = members[c_i]
                    te_items = [row_test_items[i] for i in te_idx]

                    fit_entry = cell_fits[c_i][strongest_ref_name]
                    val_preds_list.append(fit_entry["val_pred"])
                    y_val_list.append(np.array([r["ideal_expectation"] for r in val_items], dtype=float))
                    r_val_list.append(np.array([r["noisy_expectation"] for r in val_items], dtype=float))

                    test_preds_list.append(fit_entry["test_pred"])
                    y_te_list.append(np.array([r["ideal_expectation"] for r in te_items], dtype=float))
                    r_te_list.append(np.array([r["noisy_expectation"] for r in te_items], dtype=float))

                stacking = compute_stacking_and_interval(
                    val_preds_list,
                    y_val_list,
                    r_val_list,
                    test_preds_list,
                    y_te_list,
                    r_te_list,
                    members,
                    item_circuit,
                    circuit_counts,
                )

                # Fetch mean C and F and check
                rung_meta = row_meta["rungs"][rung]
                rung_meta_str = row_meta_str["rungs"][rung]
                mean_c = rung_meta["means"]["C"]["point"]
                mean_f = rung_meta["means"]["F"]["point"]
                obs_d_over_c = rung_meta["D_over_C"]["point"]

                mean_c_str = rung_meta_str["means"]["C"]["point"]
                mean_f_str = rung_meta_str["means"]["F"]["point"]
                obs_d_over_c_str = rung_meta_str["D_over_C"]["point"]

                ref_val_macro = val_macro_poly5 if strongest_ref_name == "poly5" else val_macro_hgbr
                ref_test_macro = test_macro_poly5 if strongest_ref_name == "poly5" else test_macro_hgbr

                entry_cell = {
                    "row": row_key,
                    "rung": rung,
                    "dataset_seed": seed,
                    "family": fam,
                    "strongest_reference": strongest_ref_name,
                    "validation_macro_mae": ref_val_macro,
                    "test_macro_mae": ref_test_macro,
                    "mean_C": mean_c,
                    "mean_F": mean_f,
                    "observed_D_over_C": obs_d_over_c,
                    "mean_C_strength": mean_c_str,
                    "mean_F_strength": mean_f_str,
                    "observed_D_over_C_strength": obs_d_over_c_str,
                    "ratio_ref_over_mean_C": ref_test_macro / mean_c,
                    "ratio_ref_over_mean_C_strength": ref_test_macro / mean_c_str,
                    "recalibrated_macro_mae": stacking["recal_macro_mae"],
                    "stacked_macro_mae": stacking["stack_macro_mae"],
                    "stacking_increment": stacking["increment_point"],
                    "stacking_increment_interval": stacking["increment_ci_95"],
                    "excludes_zero_above": stacking["excludes_zero_above"],
                    "rule_cells": {},
                }

                if is_r0:
                    entry_cell["r0_references"] = {
                        "poly5": {
                            "validation_macro_mae": val_macro_poly5,
                            "test_macro_mae": test_macro_poly5,
                        },
                        "hgbr": {
                            "validation_macro_mae": val_macro_hgbr,
                            "test_macro_mae": test_macro_hgbr,
                        },
                    }

                for c_i, (sev, obs) in enumerate(CELLS_DEF):
                    ckey = f"{sev}/{obs}"
                    st_d = stacking["cell_details"][c_i]
                    c_fit = cell_fits[c_i]
                    strongest_sub = c_fit[strongest_ref_name]

                    rc_dict: dict[str, Any] = {
                        "severity": sev,
                        "observable": obs,
                        "ref_val_mae": strongest_sub["val_mae"],
                        "ref_test_mae": strongest_sub["test_mae"],
                        "recal_intercept": st_d["recal_intercept"],
                        "recal_slope": st_d["recal_slope"],
                        "stack_intercept": st_d["stack_intercept"],
                        "stack_ref_coeff": st_d["stack_ref_coeff"],
                        "stack_r_coeff": st_d["stack_r_coeff"],
                        "recal_test_mae": st_d["recal_test_mae"],
                        "stack_test_mae": st_d["stack_test_mae"],
                        "cell_increment": st_d["cell_increment"],
                    }
                    if is_r0:
                        rc_dict["candidates"] = {
                            "poly5": {
                                "val_mae": c_fit["poly5"]["val_mae"],
                                "test_mae": c_fit["poly5"]["test_mae"],
                            },
                            "hgbr": {
                                "val_mae": c_fit["hgbr"]["val_mae"],
                                "test_mae": c_fit["hgbr"]["test_mae"],
                                "best_params": c_fit["hgbr"]["best_params"],
                            },
                        }
                    else:
                        rc_dict["best_params"] = strongest_sub["best_params"]

                    entry_cell["rule_cells"][ckey] = rc_dict

                results_by_cell[cell_key] = entry_cell

    out_json = {
        "schema": "qem-descriptor-information-posthoc-strongest-reference-v1",
        "status": "post hoc, descriptive; paper results and analysis-a were known",
        "description": (
            "Strongest descriptor-only reference models (HGBR grid selected on validation, "
            "degree-5 coupling polynomial at R0) and stacking increment of noisy measurement r "
            "over strongest reference on Part A spin chains."
        ),
        "analysis_script": "tools/strongest_reference.py",
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap": {
            "seed": RULE_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "method": "one-stage percentile bootstrap over test circuits",
        },
        "grid": {
            "max_iter": [200, 500],
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "random_state": 0,
        },
        "cells": results_by_cell,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_json, indent=1), encoding="utf-8")
    print(f"Wrote {len(results_by_cell)} cell results to {out_json_path} in {time.time() - t0:.2f}s")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partA-ladder",
        type=Path,
        required=True,
        help="Path to Part A ladder directory (containing cache/, descriptors/, encoder-cache/)",
    )
    parser.add_argument(
        "--analysis-a",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "analysis-a.json",
        help="Path to original analysis-a.json",
    )
    parser.add_argument(
        "--analysis-a-strength",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "strength-indicator" / "analysis-a.json",
        help="Path to strength indicator analysis-a.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "posthoc-strongest-reference.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes for cell fits",
    )
    args = parser.parse_args()

    run_strongest_reference_analysis(
        ladder_dir=args.partA_ladder,
        analysis_a_path=args.analysis_a,
        analysis_a_strength_path=args.analysis_a_strength,
        out_json_path=args.out,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
