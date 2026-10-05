#!/usr/bin/env python3
"""Post hoc analysis of learner adequacy and seed ensembling (Parts A, B, NC).

Status: post hoc, descriptive; evaluating learner stability and seed-ensemble estimand.
The frozen rule (docs/frozen-rules/2026-10-02-descriptor-information.md) specifies
evaluating learners per seed and averaging the per-seed macro MAEs (estimand
D = mean_k(C_k - F_k)). This post hoc analysis evaluates learner instability
across learner seeds and evaluates the seed-ensemble estimand: averaging each
arm's predictions over the twenty seeds before scoring (E_C = macro MAE of the
ensemble prediction for C, same for F; D_ens = E_C - E_F; D_ens / E_C).

Inputs:
  --fits DIR (repeatable): directory of fit records (<stem>.json and <stem>.npz)
  --cache DIR (repeatable): directory of data caches (<key>.pkl or <key>.json.gz)
  --out PATH: output path for the JSON results
  --analysis-a PATH: optional path to reference analysis-a.json to assert 1e-12 match

Computations per cell (where C and F both have seeds 1 to 20):
1. Per-seed spread: for C, F, and D (D_k = C_k - F_k), min, max, median, SD (ddof=1),
   and mean over learner seeds 1 to 20.
2. Seed-ensemble estimand: E_C = macro MAE of the mean prediction over seeds for C;
   E_F = same for F; D_ens = E_C - E_F; D_ens / E_C.
   Interval: two-stage bootstrap using bootstrap_draws(n_seeds, n_circuits, draws, seed).
   In each draw, the ensemble prediction is the seed-count-weighted mean of the
   seeds' predictions, and macro MAE uses the draw's circuit weights.
   Labels: classified into measurement_adds, measurement_hurts, or not_distinguished.
3. Sign flip: whether the point sign of D_ens differs from that of mean D.
4. Pooled Part A R0: pooled D = mean over the six rows of D, pooled D/C = (sum of D) /
   (sum of mean C), for standard and ensemble estimands.
   Joint bootstrap: within each dataset seed (101, 211, 307), learner seeds are
   resampled once and applied to both family rows (tfi, heisenberg); circuits
   are resampled separately per family row; dataset seeds are drawn independently.

Usage:
  PYTHONPATH=. python tools/learner_adequacy.py \\
      --fits RUNS/fits --cache RUNS/cache --out artifacts/.../posthoc-learner-adequacy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import tools.descriptor_ladder_analysis as dla  # noqa: E402

RULE_SEED = dla.RULE_SEED
DEFAULT_DRAWS = dla.DEFAULT_DRAWS
PERCENTILES = dla.PERCENTILES
DATASET_SEEDS = dla.DATASET_SEEDS
POINT_CHECK_TOLERANCE = 1e-12


def _to_relative_name(path: Path) -> str:
    path = Path(path).resolve()
    # Try relative to repo
    try:
        return str(path.relative_to(_REPO.resolve()))
    except ValueError:
        pass
    # Name release-asset inputs from the asset directory onward
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("descriptor-information"):
            return str(Path(*parts[idx:]))
    return path.name


def compute_spread(values: np.ndarray) -> dict:
    """Min, max, median, SD (ddof=1), and mean over learner seeds."""
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_cell_ensemble(
    row: dla.Row,
    c_preds: np.ndarray,          # (n_seeds, n_items)
    f_preds: np.ndarray,          # (n_seeds, n_items)
    seed_counts: np.ndarray,      # (draws, n_seeds)
    circuit_counts: np.ndarray,   # (draws, n_circuits)
    seeds: list[int],
) -> dict:
    """Compute the seed-ensemble point estimand and two-stage bootstrap interval."""
    n_seeds = len(seeds)
    draws = seed_counts.shape[0]

    # Point estimand: mean prediction over seeds, then absolute error
    c_mean_pred = c_preds.mean(axis=0)
    f_mean_pred = f_preds.mean(axis=0)

    c_ens_err = np.abs(c_mean_pred - row.ideal)
    f_ens_err = np.abs(f_mean_pred - row.ideal)

    e_c_point = dla.point_macro(c_ens_err, row.members, row.all_cells)
    e_f_point = dla.point_macro(f_ens_err, row.members, row.all_cells)
    d_ens_point = e_c_point - e_f_point

    # Two-stage bootstrap:
    # In each draw, ensemble prediction is seed-count-weighted mean of predictions
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        c_draw_preds = (seed_counts @ c_preds) / float(n_seeds)
        f_draw_preds = (seed_counts @ f_preds) / float(n_seeds)

    if not (np.all(np.isfinite(c_draw_preds)) and np.all(np.isfinite(f_draw_preds))):
        raise FloatingPointError("non-finite draw predictions in ensemble bootstrap")

    c_draw_err = np.abs(c_draw_preds - row.ideal)
    f_draw_err = np.abs(f_draw_preds - row.ideal)

    # Macro MAE under draw's circuit weights
    c_macro = np.zeros(draws)
    f_macro = np.zeros(draws)
    for cell in row.all_cells:
        idx = row.members[cell]
        w = circuit_counts[:, row.item_circuit[idx]]
        w_sum = w.sum(axis=1)
        c_macro += np.sum(w * c_draw_err[:, idx], axis=1) / w_sum
        f_macro += np.sum(w * f_draw_err[:, idx], axis=1) / w_sum

    c_macro /= len(row.all_cells)
    f_macro /= len(row.all_cells)

    if not (np.all(np.isfinite(c_macro)) and np.all(np.isfinite(f_macro))):
        raise FloatingPointError("non-finite macro MAE under bootstrap weights")

    d_ens_draws = c_macro - f_macro

    d_ens_entry = dla.interval_entry(d_ens_draws, d_ens_point, n_seeds, seeds)
    classification = dla.classify(d_ens_entry, e_c_point)

    # D_ens / E_C ratio
    if e_c_point <= 0.0 or np.any(c_macro <= 0.0):
        d_ens_over_ec = {"status": "undefined", "reason": "E_C is not positive in every draw"}
    else:
        ratio_draws = d_ens_draws / c_macro
        d_ens_over_ec = dla.interval_entry(ratio_draws, d_ens_point / e_c_point, n_seeds, seeds)

    return {
        "E_C": float(e_c_point),
        "E_F": float(e_f_point),
        "D_ens": d_ens_entry,
        "D_ens_over_E_C": d_ens_over_ec,
        "classification": classification,
        "draws_D_ens": d_ens_draws,
        "draws_E_C": c_macro,
        "draws_E_F": f_macro,
    }


def generate_pooled_draws(
    dataset_seeds: tuple[int, ...],
    row_circuits: dict[str, int],
    n_seeds: int = 20,
    draws: int = DEFAULT_DRAWS,
    seed: int = RULE_SEED,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Joint bootstrap draws for pooled Part A R0.

    Seeding documentation:
    A single fresh numpy.random.default_rng(seed) is instantiated once.
    For each draw b in range(draws):
        For each dataset seed ds in sorted(dataset_seeds): (101, 211, 307)
            1. Draw learner seeds once (integers(0, n_seeds, n_seeds))
               and store count vector via np.bincount(..., minlength=n_seeds).
               The same seed count vector is shared by both family rows (tfi and heisenberg).
            2. Resample circuits separately for family 'tfi' (integers(0, n_circ_tfi, n_circ_tfi)).
            3. Resample circuits separately for family 'heisenberg' (integers(0, n_circ_heis, n_circ_heis)).

    Properties:
    - Within one dataset seed, the two family rows share one seed draw.
    - Test circuits are resampled separately per family row.
    - The three dataset seeds are drawn independently.
    """
    rng = np.random.default_rng(seed)
    seed_counts_by_ds: dict[int, np.ndarray] = {
        ds: np.zeros((draws, n_seeds)) for ds in dataset_seeds
    }
    circuit_counts_by_row: dict[str, np.ndarray] = {
        label: np.zeros((draws, n_circ)) for label, n_circ in row_circuits.items()
    }

    for b in range(draws):
        for ds in sorted(dataset_seeds):
            sc = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
            seed_counts_by_ds[ds][b] = sc
            for family in ("tfi", "heisenberg"):
                label = f"shipped-s{ds}-n640/{family}"
                n_circ = row_circuits[label]
                cc = np.bincount(rng.integers(0, n_circ, n_circ), minlength=n_circ)
                circuit_counts_by_row[label][b] = cc

    return seed_counts_by_ds, circuit_counts_by_row


def compute_pooled_r0(
    r0_rows: list[dict],
    seed_counts_by_ds: dict[int, np.ndarray],
    circuit_counts_by_row: dict[str, np.ndarray],
    n_seeds: int = 20,
) -> dict:
    """Compute pooled R0 standard and ensemble estimands over the six Part A rows."""
    draws = next(iter(seed_counts_by_ds.values())).shape[0]

    # Standard per-row draws under the joint bootstrap
    d_draws_by_row = []
    c_draws_by_row = []
    dens_draws_by_row = []
    cens_draws_by_row = []

    d_points = []
    c_points = []
    dens_points = []
    cens_points = []

    for rdata in r0_rows:
        label = rdata["label"]
        ds = rdata["dataset_seed"]
        row = rdata["row"]
        c_preds = rdata["c_preds"]
        f_preds = rdata["f_preds"]

        sc = seed_counts_by_ds[ds]
        cc = circuit_counts_by_row[label]

        # Points
        c_pts = rdata["c_seed_points"]
        f_pts = rdata["f_seed_points"]
        mean_c = float(c_pts.mean())
        mean_d = float((c_pts - f_pts).mean())
        d_points.append(mean_d)
        c_points.append(mean_c)

        # Standard draws
        c_err_seeds = np.abs(c_preds - row.ideal)
        f_err_seeds = np.abs(f_preds - row.ideal)
        macro_c = dla.macro_under_weights(
            c_err_seeds, row.item_circuit, row.members, row.all_cells, cc
        )
        macro_f = dla.macro_under_weights(
            f_err_seeds, row.item_circuit, row.members, row.all_cells, cc
        )
        mean_c_draw = np.sum(sc * macro_c, axis=1) / float(n_seeds)
        mean_f_draw = np.sum(sc * macro_f, axis=1) / float(n_seeds)
        d_draw = mean_c_draw - mean_f_draw
        d_draws_by_row.append(d_draw)
        c_draws_by_row.append(mean_c_draw)

        # Ensemble point & draws under joint bootstrap
        c_mean_pred = c_preds.mean(axis=0)
        f_mean_pred = f_preds.mean(axis=0)
        e_c = dla.point_macro(np.abs(c_mean_pred - row.ideal), row.members, row.all_cells)
        e_f = dla.point_macro(np.abs(f_mean_pred - row.ideal), row.members, row.all_cells)
        d_ens = e_c - e_f
        dens_points.append(d_ens)
        cens_points.append(e_c)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            c_draw_preds = (sc @ c_preds) / float(n_seeds)
            f_draw_preds = (sc @ f_preds) / float(n_seeds)
        c_draw_err = np.abs(c_draw_preds - row.ideal)
        f_draw_err = np.abs(f_draw_preds - row.ideal)

        c_macro_ens = np.zeros(draws)
        f_macro_ens = np.zeros(draws)
        for cell in row.all_cells:
            idx = row.members[cell]
            w = cc[:, row.item_circuit[idx]]
            w_sum = w.sum(axis=1)
            c_macro_ens += np.sum(w * c_draw_err[:, idx], axis=1) / w_sum
            f_macro_ens += np.sum(w * f_draw_err[:, idx], axis=1) / w_sum
        c_macro_ens /= len(row.all_cells)
        f_macro_ens /= len(row.all_cells)
        dens_draw = c_macro_ens - f_macro_ens

        dens_draws_by_row.append(dens_draw)
        cens_draws_by_row.append(c_macro_ens)

    # Arrays of shape (6, draws)
    d_draws_arr = np.array(d_draws_by_row)
    c_draws_arr = np.array(c_draws_by_row)
    dens_draws_arr = np.array(dens_draws_by_row)
    cens_draws_arr = np.array(cens_draws_by_row)

    # Standard pooled
    pooled_d_point = float(np.mean(d_points))
    pooled_dc_point = float(np.sum(d_points) / np.sum(c_points))
    pooled_d_draws = np.mean(d_draws_arr, axis=0)
    pooled_dc_draws = np.sum(d_draws_arr, axis=0) / np.sum(c_draws_arr, axis=0)

    # Ensemble pooled
    pooled_dens_point = float(np.mean(dens_points))
    pooled_densc_point = float(np.sum(dens_points) / np.sum(cens_points))
    pooled_dens_draws = np.mean(dens_draws_arr, axis=0)
    pooled_densc_draws = np.sum(dens_draws_arr, axis=0) / np.sum(cens_draws_arr, axis=0)

    pooled_d_entry = dla.interval_entry(pooled_d_draws, pooled_d_point, n_seeds)
    pooled_dc_entry = dla.interval_entry(pooled_dc_draws, pooled_dc_point, n_seeds)
    pooled_dens_entry = dla.interval_entry(pooled_dens_draws, pooled_dens_point, n_seeds)
    pooled_densc_entry = dla.interval_entry(pooled_densc_draws, pooled_densc_point, n_seeds)

    mean_pooled_c = float(np.mean(c_points))
    mean_pooled_ec = float(np.mean(cens_points))

    pooled_d_class = dla.classify(pooled_d_entry, mean_pooled_c)
    pooled_dens_class = dla.classify(pooled_dens_entry, mean_pooled_ec)

    sign_flip = bool(
        (pooled_dens_point > 0.0 and pooled_d_point < 0.0)
        or (pooled_dens_point < 0.0 and pooled_d_point > 0.0)
    )

    return {
        "rows": [r["label"] for r in r0_rows],
        "standard": {
            "pooled_D": pooled_d_entry,
            "pooled_D_over_C": pooled_dc_entry,
            "classification": pooled_d_class,
        },
        "ensemble": {
            "pooled_D_ens": pooled_dens_entry,
            "pooled_D_ens_over_C": pooled_densc_entry,
            "classification": pooled_dens_class,
        },
        "sign_flip": sign_flip,
        "seeding_description": (
            "Single fresh default_rng(seed) at seed 20261002. For each draw b in range(10000), "
            "iterates over dataset seeds (101, 211, 307): draws learner seeds once (integers(0, 20, 20)) "
            "shared by both family rows (tfi and heisenberg), then draws circuits separately for tfi "
            "and heisenberg. The three dataset seeds are drawn independently."
        ),
    }


def analyze_learner_adequacy(
    fit_dirs: list[Path],
    cache_dirs: list[Path],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = RULE_SEED,
    analysis_a_path: Path | None = None,
    verbose: bool = True,
    dataset_seeds: tuple[int, ...] = DATASET_SEEDS,
) -> dict:
    started = time.perf_counter()
    store = dla.FitStore(fit_dirs, False)
    caches, shadowed = dla.discover_caches(cache_dirs)

    # Reference analysis-a if provided
    ref_analysis_a: dict | None = None
    if analysis_a_path is not None and Path(analysis_a_path).exists():
        with open(analysis_a_path, encoding="utf-8") as f:
            ref_analysis_a = json.load(f)

    result: dict = {
        "schema": "descriptor-information-posthoc-learner-adequacy-v1",
        "status": "post hoc, descriptive; evaluating learner stability and seed-ensemble estimand",
        "script_sha256": dla.sha256_file(Path(__file__)),
        "bootstrap": {
            "seed": seed,
            "draws": draws,
            "percentiles": list(PERCENTILES),
            "rng_scheme": (
                "Cell bootstrap: per-cell fresh default_rng(seed) via bootstrap_draws(20, n_circuits, draws, seed). "
                "Pooled R0 bootstrap: single default_rng(seed), learner seeds resampled once per dataset seed and applied "
                "to both family rows; circuits resampled separately per row; dataset seeds drawn independently."
            ),
        },
        "inputs": {
            "fits": [_to_relative_name(Path(d)) for d in fit_dirs],
            "cache": [_to_relative_name(Path(d)) for d in cache_dirs],
            "n_fit_files": store.n_files,
            "fits_sha256_digest": store.inputs_digest,
            "caches_shadowed": shadowed,
        },
        "parts": {},
        "summary": {
            "n_cells_analyzed": 0,
            "n_sign_flips": 0,
            "sign_flip_cells": [],
        },
    }

    if ref_analysis_a is not None:
        result["inputs"]["reference_analysis_a"] = _to_relative_name(Path(analysis_a_path))

    r0_rows_for_pooling = []
    r0_circuits_for_pooling = {}
    point_checks = 0
    max_c_diff = 0.0
    max_f_diff = 0.0
    max_d_diff = 0.0

    if verbose:
        print(f"{'Part':4s} {'Row':30s} {'Rung':10s} {'mean D':10s} {'D label':18s} "
              f"{'D_ens':10s} {'D_ens [CI]':24s} {'D_ens label':18s} {'Flip?':6s} {'SD(D)':10s}")
        print("-" * 135)

    for part_name, spec in dla.PARTS.items():
        keys = sorted(k for k in store.keys() if spec["key_pattern"].match(k))
        if not keys:
            continue

        part_out: dict = {"title": spec["title"], "rows": {}}

        for key in keys:
            dataset_seed = int(spec["key_pattern"].match(key).group(1))
            if key not in caches:
                continue
            cache_data = dla.load_cache(caches[key])
            families = spec["families"] or (None,)

            for family in families:
                label = f"{key}/{family}" if family else key
                row = dla.Row(cache_data, family, label)
                row_out: dict = {"key": key, "dataset_seed": dataset_seed, "family": family, "rungs": {}}

                for rung in spec["rungs"]:
                    allowed = spec["rung_families"].get(rung)
                    if allowed is not None and family not in allowed:
                        continue

                    arms = dla.arms_at(store, spec, key, rung)
                    seeds = list(range(1, 21))
                    if set(arms["C"].keys()) != set(seeds) or set(arms["F"].keys()) != set(seeds):
                        continue

                    rec_c = dla.RowEstimator._records(arms["C"], seeds)
                    rec_f = dla.RowEstimator._records(arms["F"], seeds)

                    # Predictions: (20, n_items)
                    c_preds = np.array([store.array(r, "test")[row.index] for r in rec_c])
                    f_preds = np.array([store.array(r, "test")[row.index] for r in rec_f])

                    # 1. Per-seed spread
                    est = dla.RowEstimator(row, store, 1, seed, 20)
                    pts_c = est.point_values(rec_c, "test", row.all_cells)
                    pts_f = est.point_values(rec_f, "test", row.all_cells)
                    pts_d = pts_c - pts_f

                    spread_c = compute_spread(pts_c)
                    spread_f = compute_spread(pts_f)
                    spread_d = compute_spread(pts_d)

                    mean_c = spread_c["mean"]
                    mean_f = spread_f["mean"]
                    mean_d = spread_d["mean"]

                    # Assertions against reference analysis-a
                    if ref_analysis_a is not None:
                        ref_cell = ref_analysis_a["parts"][part_name]["rows"][label]["rungs"][rung]
                        rc = ref_cell["means"]["C"]["point"]
                        rf = ref_cell["means"]["F"]["point"]
                        rd = ref_cell["D"]["point"]
                        diff_c = abs(mean_c - rc)
                        diff_f = abs(mean_f - rf)
                        diff_d = abs(mean_d - rd)
                        max_c_diff = max(max_c_diff, diff_c)
                        max_f_diff = max(max_f_diff, diff_f)
                        max_d_diff = max(max_d_diff, diff_d)
                        point_checks += 1
                        assert diff_c <= POINT_CHECK_TOLERANCE, f"{label} {rung} C: {diff_c}"
                        assert diff_f <= POINT_CHECK_TOLERANCE, f"{label} {rung} F: {diff_f}"
                        assert diff_d <= POINT_CHECK_TOLERANCE, f"{label} {rung} D: {diff_d}"

                    # 2. Standard estimand bootstrap (Script A arithmetic)
                    seed_counts, circuit_counts = dla.bootstrap_draws(20, row.n_circuits, draws, seed)
                    macro_c_seeds = dla.macro_under_weights(
                        np.abs(c_preds - row.ideal), row.item_circuit, row.members, row.all_cells, circuit_counts
                    )
                    macro_f_seeds = dla.macro_under_weights(
                        np.abs(f_preds - row.ideal), row.item_circuit, row.members, row.all_cells, circuit_counts
                    )
                    c_draws = np.sum(seed_counts * macro_c_seeds, axis=1) / 20.0
                    f_draws = np.sum(seed_counts * macro_f_seeds, axis=1) / 20.0
                    d_draws = c_draws - f_draws

                    d_entry = dla.interval_entry(d_draws, mean_d, 20, seeds)
                    d_class = dla.classify(d_entry, mean_c)
                    d_over_c_entry = (
                        dla.interval_entry(d_draws / c_draws, mean_d / mean_c, 20, seeds)
                        if mean_c > 0.0 and np.all(c_draws > 0.0)
                        else {"status": "undefined"}
                    )

                    # 3. Seed-ensemble estimand
                    ens_res = compute_cell_ensemble(row, c_preds, f_preds, seed_counts, circuit_counts, seeds)

                    # 4. Point sign flip
                    sign_flip = bool(
                        (ens_res["D_ens"]["point"] > 0.0 and mean_d < 0.0)
                        or (ens_res["D_ens"]["point"] < 0.0 and mean_d > 0.0)
                    )

                    cell_out = {
                        "spread": {
                            "C": spread_c,
                            "F": spread_f,
                            "D": spread_d,
                        },
                        "standard": {
                            "mean_C": mean_c,
                            "mean_F": mean_f,
                            "mean_D": d_entry,
                            "D_over_C": d_over_c_entry,
                            "classification": d_class,
                        },
                        "ensemble": {
                            "E_C": ens_res["E_C"],
                            "E_F": ens_res["E_F"],
                            "D_ens": ens_res["D_ens"],
                            "D_ens_over_E_C": ens_res["D_ens_over_E_C"],
                            "classification": ens_res["classification"],
                        },
                        "sign_flip": sign_flip,
                    }

                    row_out["rungs"][rung] = cell_out
                    result["summary"]["n_cells_analyzed"] += 1
                    if sign_flip:
                        result["summary"]["n_sign_flips"] += 1
                        result["summary"]["sign_flip_cells"].append(f"{label} {rung}")

                    if verbose:
                        dens_pt = ens_res["D_ens"]["point"]
                        dens_lo = ens_res["D_ens"]["interval"]["lower"]
                        dens_hi = ens_res["D_ens"]["interval"]["upper"]
                        ci_str = f"[{dens_lo:+.6f}, {dens_hi:+.6f}]"
                        flip_str = "YES" if sign_flip else "no"
                        sd_str = f"{spread_d['sd']:.6f}" if spread_d["sd"] is not None else "-"
                        print(f"{part_name:4s} {label:30s} {rung:10s} {mean_d:+.6f} "
                              f"{d_class['label']:18s} {dens_pt:+.6f} {ci_str:24s} "
                              f"{ens_res['classification']['label']:18s} {flip_str:6s} {sd_str:10s}")

                    # Accumulate Part A R0 rows for pooled estimate
                    if part_name == "A" and rung == "R0":
                        r0_rows_for_pooling.append({
                            "label": label,
                            "dataset_seed": dataset_seed,
                            "family": family,
                            "row": row,
                            "c_preds": c_preds,
                            "f_preds": f_preds,
                            "c_seed_points": pts_c,
                            "f_seed_points": pts_f,
                        })
                        r0_circuits_for_pooling[label] = row.n_circuits

                part_out["rows"][label] = row_out

            result["parts"][part_name] = part_out

    # Pooled Part A R0
    if len(r0_rows_for_pooling) == 6:
        sc_by_ds, cc_by_row = generate_pooled_draws(
            tuple(dataset_seeds), r0_circuits_for_pooling, n_seeds=20, draws=draws, seed=seed
        )
        pooled_r0 = compute_pooled_r0(r0_rows_for_pooling, sc_by_ds, cc_by_row, n_seeds=20)
        result["pooled_R0"] = pooled_r0
        if verbose:
            std_d = pooled_r0["standard"]["pooled_D"]["point"]
            std_dc = pooled_r0["standard"]["pooled_D_over_C"]["point"]
            ens_d = pooled_r0["ensemble"]["pooled_D_ens"]["point"]
            ens_dc = pooled_r0["ensemble"]["pooled_D_ens_over_C"]["point"]
            print("-" * 135)
            print(f"Pooled Part A R0 (6 rows):")
            print(f"  Standard: pooled D = {std_d:+.6f} [{pooled_r0['standard']['pooled_D']['interval']['lower']:+.6f}, "
                  f"{pooled_r0['standard']['pooled_D']['interval']['upper']:+.6f}], "
                  f"pooled D/C = {std_dc:+.6f} [{pooled_r0['standard']['pooled_D_over_C']['interval']['lower']:+.6f}, "
                  f"{pooled_r0['standard']['pooled_D_over_C']['interval']['upper']:+.6f}] -> {pooled_r0['standard']['classification']['label']}")
            print(f"  Ensemble: pooled D_ens = {ens_d:+.6f} [{pooled_r0['ensemble']['pooled_D_ens']['interval']['lower']:+.6f}, "
                  f"{pooled_r0['ensemble']['pooled_D_ens']['interval']['upper']:+.6f}], "
                  f"pooled D_ens/C = {ens_dc:+.6f} [{pooled_r0['ensemble']['pooled_D_ens_over_C']['interval']['lower']:+.6f}, "
                  f"{pooled_r0['ensemble']['pooled_D_ens_over_C']['interval']['upper']:+.6f}] -> {pooled_r0['ensemble']['classification']['label']}")
            print(f"  Pooled point sign flip: {'YES' if pooled_r0['sign_flip'] else 'no'}")

    if ref_analysis_a is not None:
        result["verification"] = {
            "n_checked_against_analysis_a": point_checks,
            "max_abs_diff_mean_C": max_c_diff,
            "max_abs_diff_mean_F": max_f_diff,
            "max_abs_diff_mean_D": max_d_diff,
            "tolerance": POINT_CHECK_TOLERANCE,
        }

    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", action="append", type=Path, default=[],
                        help="a directory of <stem>.json + <stem>.npz fits (repeatable)")
    parser.add_argument("--cache", action="append", type=Path, default=[],
                        help="a directory of <key>.pkl or <key>.json.gz caches (repeatable)")
    parser.add_argument("--out", type=Path, default=None,
                        help="path to write output JSON")
    parser.add_argument("--analysis-a", type=Path, default=None,
                        help="optional reference analysis-a.json to assert 1e-12 point agreement")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                        help="bootstrap draws (default: 10,000)")
    parser.add_argument("--bootstrap-seed", type=int, default=RULE_SEED,
                        help="RNG seed for bootstrap draws (default: 20261002)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress verbose per-cell table")
    parser.add_argument("--dataset-seeds", type=int, nargs="+", default=list(DATASET_SEEDS),
                        help="dataset seeds of the pooled Part A R0 estimate (default: %(default)s)")
    args = parser.parse_args(argv)

    if not args.fits or not args.cache or args.out is None:
        parser.error("--fits, --cache, and --out are required")

    analysis_a_path = args.analysis_a
    if analysis_a_path is None:
        # Check adjacent analysis-a.json
        candidate = args.out.parent / "analysis-a.json"
        if candidate.exists():
            analysis_a_path = candidate

    result = analyze_learner_adequacy(
        args.fits,
        args.cache,
        draws=args.draws,
        seed=args.bootstrap_seed,
        analysis_a_path=analysis_a_path,
        verbose=not args.quiet,
        dataset_seeds=tuple(args.dataset_seeds),
    )

    dla._write_json(args.out, result)
    print(f"wrote {args.out} in {result['elapsed_seconds']:.1f}s "
          f"({result['summary']['n_cells_analyzed']} cells analyzed, {result['summary']['n_sign_flips']} sign flips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
