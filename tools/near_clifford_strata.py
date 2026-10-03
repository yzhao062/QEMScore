#!/usr/bin/env python3
"""Post hoc descriptive analysis of near-Clifford label strata and the continuous positive control.

Task items addressed:
1. Near-Clifford label distributions for dataset seeds 101, 211, 307 across splits (train, validation, test),
   identifying exact counts and fractions of labels equal to 0, +1, -1, and other values (rounded to 1e-10).
2. Stratified test errors on near-Clifford datasets (y == 0 vs y != 0) using NC-R0 fits (arms C, F, P)
   and NC-none fits (arms C, M), plus A and R:
   - Macro MAE separately on items with y = 0 and items with y != 0 (equal weight over present cells).
   - Mean over seeds of C, F, M, and D = C - F per stratum.
   - Circuit bootstrap interval for D per stratum (two-stage over seeds and circuits, 10,000 draws, default_rng(20261002)).
   - Share of C's error coming from the y != 0 items.
3. Continuous positive control:
   - From artifacts/descriptor-information/analysis-a.json, report D, D/C and their two-stage intervals for R5 and R3-TFI (Part A).
   - Report the number of distinct test labels per spin-chain row (demonstrating label continuity).

Usage:
    PYTHONPATH=. python tools/near_clifford_strata.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

# Ensure tools directory is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools"))
sys.path.insert(0, str(_REPO_ROOT))

import descriptor_ladder_analysis as dla  # noqa: E402

DATASET_SEEDS = (101, 211, 307)
LEARNER_SEEDS = tuple(range(1, 21))
DEFAULT_DRAWS = 10_000
RULE_SEED = 20261002
PERCENTILES = (2.5, 97.5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Task 2: Near-Clifford Label Distribution
# --------------------------------------------------------------------------


def analyze_label_distribution(nc_data_root: Path) -> dict[str, Any]:
    """Compute label counts and proportions for 0, +1, -1, other for each seed and split."""
    dist_out = {}
    for seed in DATASET_SEEDS:
        ds_name = f"nc-s{seed}-n640"
        items_path = nc_data_root / ds_name / "items.jsonl"
        splits: dict[str, list[float]] = {"train": [], "validation": [], "test": []}
        with open(items_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                split = record["split"]
                if split in splits:
                    splits[split].append(float(record["ideal_expectation"]))

        seed_dist = {}
        for split_name, raw_labels in splits.items():
            total = len(raw_labels)
            rounded = [round(x, 10) for x in raw_labels]
            c0 = sum(1 for x in rounded if x == 0.0)
            cp1 = sum(1 for x in rounded if x == 1.0)
            cm1 = sum(1 for x in rounded if x == -1.0)
            other_vals = [x for x in rounded if x not in (0.0, 1.0, -1.0)]
            c_other = len(other_vals)
            uniq_other = sorted(list(set(other_vals)))

            seed_dist[split_name] = {
                "total_items": total,
                "counts": {
                    "zero": c0,
                    "plus_one": cp1,
                    "minus_one": cm1,
                    "other": c_other,
                },
                "proportions": {
                    "zero": c0 / total if total > 0 else 0.0,
                    "plus_one": cp1 / total if total > 0 else 0.0,
                    "minus_one": cm1 / total if total > 0 else 0.0,
                    "other": c_other / total if total > 0 else 0.0,
                },
                "other_values": uniq_other,
            }
        dist_out[str(seed)] = seed_dist
    return dist_out


# --------------------------------------------------------------------------
# Task 3: Stratified Test Errors and Two-Stage Circuit Bootstrap
# --------------------------------------------------------------------------


def compute_stratified_errors(
    cache_dir: Path,
    fits_dir: Path,
    inputs_dir: Path,
    draws: int = DEFAULT_DRAWS,
    seed: int = RULE_SEED,
) -> dict[str, Any]:
    """Compute stratified macro MAE and two-stage circuit bootstrap intervals."""
    # Memoized bootstrap draws (20 seeds, 160 circuits, draws, seed)
    seed_counts, circuit_counts = dla.bootstrap_draws(len(LEARNER_SEEDS), 160, draws, seed)

    stratified_results = {}

    for ds_seed in DATASET_SEEDS:
        key = f"nc-s{ds_seed}-n640"
        pkl_path = cache_dir / f"{key}.pkl"
        with open(pkl_path, "rb") as f:
            cache_data = pickle.load(f)
        test_items = cache_data["test"]
        n_test = len(test_items)
        ideal = np.array([float(r["ideal_expectation"]) for r in test_items])
        noisy = np.array([float(r["noisy_expectation"]) for r in test_items])
        cells = sorted(list({(str(r["severity"]), str(r["observable"])) for r in test_items}))
        circuits = sorted({str(r["circuit_id"]) for r in test_items})
        c_pos = {c: i for i, c in enumerate(circuits)}
        item_circuit = np.array([c_pos[str(r["circuit_id"])] for r in test_items])

        # Load A predictions from Appendix N inputs
        inp_pred_path = inputs_dir / f"s{ds_seed}/test_predictions.json.gz"
        with gzip.open(inp_pred_path, "rt", encoding="utf-8") as f:
            inp_preds = json.load(f)
        pred_A = np.array([inp_preds["A"][str(r["item_id"])] for r in test_items])
        pred_R = noisy

        # Load fitted models: NC-R0 (C, F, P) and NC-none (C, M)
        c_r0 = np.array([np.load(fits_dir / f"{key}__NC-R0__k{k:02d}__C.npz")["test"] for k in LEARNER_SEEDS])
        f_r0 = np.array([np.load(fits_dir / f"{key}__NC-R0__k{k:02d}__F.npz")["test"] for k in LEARNER_SEEDS])
        p_r0 = np.array([np.load(fits_dir / f"{key}__NC-R0__k{k:02d}__P.npz")["test"] for k in LEARNER_SEEDS])
        c_none = np.array([np.load(fits_dir / f"{key}__NC-none__k{k:02d}__C.npz")["test"] for k in LEARNER_SEEDS])
        m_none = np.array([np.load(fits_dir / f"{key}__NC-none__k{k:02d}__M.npz")["test"] for k in LEARNER_SEEDS])

        err_c_r0 = np.abs(c_r0 - ideal[None, :])
        err_f_r0 = np.abs(f_r0 - ideal[None, :])
        err_p_r0 = np.abs(p_r0 - ideal[None, :])
        err_c_none = np.abs(c_none - ideal[None, :])
        err_m_none = np.abs(m_none - ideal[None, :])
        err_a = np.abs(pred_A - ideal)[None, :]
        err_r = np.abs(pred_R - ideal)[None, :]

        # Stratum masks
        is_zero = np.isclose(ideal, 0.0, atol=1e-10)
        strata_masks = {
            "all": np.ones(n_test, dtype=bool),
            "y_eq_0": is_zero,
            "y_neq_0": ~is_zero,
        }

        # Cell counts per stratum and check for empty cells
        cell_counts_info = {}
        for sname, mask in strata_masks.items():
            cell_counts = {}
            for cell in cells:
                c_idx = np.array([(str(r["severity"]), str(r["observable"])) == cell for r in test_items])
                cnt = int(np.sum(c_idx & mask))
                cell_counts[f"{cell[0]}/{cell[1]}"] = cnt
            empty_cells = [cell_str for cell_str, cnt in cell_counts.items() if cnt == 0]
            cell_counts_info[sname] = {
                "cell_item_counts": cell_counts,
                "empty_cells": empty_cells,
                "has_empty_cells": len(empty_cells) > 0,
            }

        # Compute share of C's error from y != 0 items
        mask_nz = strata_masks["y_neq_0"]
        share_c_r0 = float(np.mean(np.sum(err_c_r0[:, mask_nz], axis=1) / np.sum(err_c_r0, axis=1)))
        share_c_none = float(np.mean(np.sum(err_c_none[:, mask_nz], axis=1) / np.sum(err_c_none, axis=1)))

        strata_eval = {}
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for sname, mask in strata_masks.items():
                def evaluate_macro_and_bootstrap(err_mat: np.ndarray) -> tuple[np.ndarray, float]:
                    # err_mat is (n_models, n_test)
                    cell_maes = []
                    cell_present = []
                    for cell in cells:
                        c_idx = np.array([(str(r["severity"]), str(r["observable"])) == cell for r in test_items])
                        combo_idx = c_idx & mask
                        sub_items = np.flatnonzero(combo_idx)
                        if len(sub_items) == 0:
                            cell_maes.append(np.zeros((draws, err_mat.shape[0])))
                            cell_present.append(np.zeros(draws, dtype=bool))
                            continue
                        sub_circuits = item_circuit[sub_items]
                        w = circuit_counts[:, sub_circuits]
                        w_sum = w.sum(axis=1, keepdims=True)
                        pres = (w_sum.squeeze(-1) > 0)
                        cell_present.append(pres)
                        w_safe = np.where(w_sum > 0, w_sum, 1.0)
                        cell_mae = (w @ err_mat[:, sub_items].T) / w_safe
                        cell_maes.append(cell_mae)

                    cell_maes_arr = np.array(cell_maes)  # (n_cells, draws, n_models)
                    cell_pres_arr = np.array(cell_present, dtype=float)[:, :, None]  # (n_cells, draws, 1)
                    n_pres = cell_pres_arr.sum(axis=0)  # (draws, 1)
                    macro_draws = (cell_maes_arr * cell_pres_arr).sum(axis=0) / n_pres  # (draws, n_models)

                    # Point estimate: unit weight over cells present in stratum
                    cell_pts = []
                    for cell in cells:
                        c_idx = np.array([(str(r["severity"]), str(r["observable"])) == cell for r in test_items])
                        combo_idx = c_idx & mask
                        if np.sum(combo_idx) > 0:
                            cell_pts.append(np.mean(err_mat[:, combo_idx], axis=1))
                    pt_val = float(np.mean(np.mean(cell_pts, axis=0)))
                    return macro_draws, pt_val

                macro_c_r0, pt_c_r0 = evaluate_macro_and_bootstrap(err_c_r0)
                macro_f_r0, pt_f_r0 = evaluate_macro_and_bootstrap(err_f_r0)
                macro_p_r0, pt_p_r0 = evaluate_macro_and_bootstrap(err_p_r0)
                macro_c_none, pt_c_none = evaluate_macro_and_bootstrap(err_c_none)
                macro_m_none, pt_m_none = evaluate_macro_and_bootstrap(err_m_none)
                macro_a, pt_a = evaluate_macro_and_bootstrap(err_a)
                macro_r, pt_r = evaluate_macro_and_bootstrap(err_r)

                # D = C - F (NC-R0)
                d_draws = np.sum(seed_counts * (macro_c_r0 - macro_f_r0), axis=1) / float(len(LEARNER_SEEDS))
                pt_d = pt_c_r0 - pt_f_r0
                lo_d, hi_d = np.percentile(d_draws, PERCENTILES)

                # D_none = C_none - M (NC-none)
                dn_draws = np.sum(seed_counts * (macro_c_none - macro_m_none), axis=1) / float(len(LEARNER_SEEDS))
                pt_dn = pt_c_none - pt_m_none
                lo_dn, hi_dn = np.percentile(dn_draws, PERCENTILES)

                # E = M - F
                e_draws = np.sum(seed_counts * (macro_m_none - macro_f_r0), axis=1) / float(len(LEARNER_SEEDS))
                pt_e = pt_m_none - pt_f_r0
                lo_e, hi_e = np.percentile(e_draws, PERCENTILES)

                strata_eval[sname] = {
                    "item_count": int(np.sum(mask)),
                    "points": {
                        "NC_R0_C": pt_c_r0,
                        "NC_R0_F": pt_f_r0,
                        "NC_R0_P": pt_p_r0,
                        "NC_none_C": pt_c_none,
                        "NC_none_M": pt_m_none,
                        "A": pt_a,
                        "R": pt_r,
                    },
                    "D_NC_R0": {
                        "status": "estimated",
                        "point": pt_d,
                        "interval": {"lower": float(lo_d), "upper": float(hi_d)},
                        "excludes_zero_above": bool(lo_d > 0.0),
                        "draw_sd": float(np.std(d_draws, ddof=1)),
                    },
                    "D_NC_none": {
                        "status": "estimated",
                        "point": pt_dn,
                        "interval": {"lower": float(lo_dn), "upper": float(hi_dn)},
                        "excludes_zero_above": bool(lo_dn > 0.0),
                        "draw_sd": float(np.std(dn_draws, ddof=1)),
                    },
                    "E_contrast": {
                        "status": "estimated",
                        "point": pt_e,
                        "interval": {"lower": float(lo_e), "upper": float(hi_e)},
                        "draw_sd": float(np.std(e_draws, ddof=1)),
                    },
                }

        stratified_results[key] = {
            "dataset_seed": ds_seed,
            "cell_counts": cell_counts_info,
            "share_of_c_error_from_y_neq_0": {
                "NC_R0_C": share_c_r0,
                "NC_none_C": share_c_none,
            },
            "strata": strata_eval,
        }

    return stratified_results


# --------------------------------------------------------------------------
# Task 4: Continuous Positive Control (Part A, R5 and R3-TFI)
# --------------------------------------------------------------------------


def extract_continuous_positive_control(
    analysis_a_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Extract D, D/C and intervals for R5 and R3-TFI from analysis-a.json and label stats."""
    with open(analysis_a_path, "r", encoding="utf-8") as f:
        analysis_a = json.load(f)

    part_a_rows = analysis_a["parts"]["A"]["rows"]
    rungs_to_extract = ("R5", "R3-TFI")

    positive_control_out = {
        "analysis_a_file": (str(analysis_a_path.resolve().relative_to(_REPO_ROOT)) if analysis_a_path.resolve().is_relative_to(_REPO_ROOT) else analysis_a_path.name),
        "rows": {},
        "spin_chain_label_continuity": {},
    }

    # Extract distinct test labels from caches
    for ds_seed in DATASET_SEEDS:
        key = f"shipped-s{ds_seed}-n640"
        pkl_path = cache_dir / f"{key}.pkl"
        with open(pkl_path, "rb") as f:
            cdata = pickle.load(f)
        test_items = cdata["test"]

        for fam in ("tfi", "heisenberg"):
            row_key = f"{key}/{fam}"
            fam_items = [it for it in test_items if it.get("family") == fam]
            labels = [float(it["ideal_expectation"]) for it in fam_items]
            rounded = {round(x, 10) for x in labels}
            positive_control_out["spin_chain_label_continuity"][row_key] = {
                "dataset_seed": ds_seed,
                "family": fam,
                "total_test_items": len(fam_items),
                "distinct_test_labels": len(rounded),
                "min_label": float(min(labels)),
                "max_label": float(max(labels)),
                "is_continuous": bool(len(rounded) == 320),
            }

    # Extract R5 and R3-TFI endpoints from analysis-a
    for row_key in sorted(part_a_rows.keys()):
        row_data = part_a_rows[row_key]
        rungs_data = row_data["rungs"]
        row_extract = {}
        for rung in rungs_to_extract:
            if rung in rungs_data:
                rd = rungs_data[rung]
                row_extract[rung] = {
                    "D": rd.get("D"),
                    "D_over_C": rd.get("D_over_C"),
                    "means": {
                        arm: rd["means"][arm].get("point")
                        for arm in ("C", "F", "P", "M", "A", "R")
                        if arm in rd.get("means", {})
                    },
                }
        if row_extract:
            positive_control_out["rows"][row_key] = row_extract

    return positive_control_out


# --------------------------------------------------------------------------
# Main Execution and Printing
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post hoc descriptive analysis: near-Clifford label strata and continuous positive control."
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("descriptor-information-v1/fits"),
        help="Path to fits directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("descriptor-information-v1/cache"),
        help="Path to cache directory",
    )
    parser.add_argument(
        "--nc-data-dir",
        type=Path,
        default=Path("data"),
        help="Path to near-Clifford datasets directory",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts/near-clifford-positive-control/inputs",
        help="Path to near-Clifford inputs directory",
    )
    parser.add_argument(
        "--analysis-a",
        type=Path,
        default=_REPO_ROOT / "artifacts/descriptor-information/analysis-a.json",
        help="Path to analysis-a.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_REPO_ROOT / "artifacts/near-clifford-positive-control/posthoc-label-strata.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help="Number of bootstrap draws (default: 10,000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=RULE_SEED,
        help="RNG seed for bootstrap draws (default: 20261002)",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("Near-Clifford Strata & Continuous Positive Control Analysis (Post Hoc)")
    print(f"Bootstrap: {args.draws} draws, seed {args.bootstrap_seed}")
    print("================================================================================\n")

    # 1. Label distribution
    print("1. Computing near-Clifford label distributions...")
    label_dist = analyze_label_distribution(args.nc_data_dir)

    # 2. Stratified errors
    print("2. Computing stratified test errors on near-Clifford datasets...")
    stratified_errors = compute_stratified_errors(
        args.cache_dir, args.fits_dir, args.inputs_dir, args.draws, args.bootstrap_seed
    )

    # 3. Continuous positive control
    print("3. Extracting continuous positive control data (Part A, R5 and R3-TFI)...")
    continuous_pc = extract_continuous_positive_control(args.analysis_a, args.cache_dir)

    # Combine into master JSON
    output_payload = {
        "schema": "near-clifford-posthoc-label-strata-v1",
        "status": "post hoc, descriptive; the paper's results were known",
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "bootstrap": {
            "draws": args.draws,
            "seed": args.bootstrap_seed,
            "percentiles": list(PERCENTILES),
        },
        "label_distribution": label_dist,
        "stratified_errors": stratified_errors,
        "continuous_positive_control": continuous_pc,
    }

    write_json_atomic(args.out_json, output_payload)
    print(f"\nSuccessfully wrote JSON output to: {args.out_json}\n")

    # Print summary tables to stdout
    print("\n" + "=" * 80)
    print("TABLE 1: Near-Clifford Label Distribution by Seed and Split")
    print("=" * 80)
    print(f"{'Seed':<6}{'Split':<12}{'Total':<8}{'y = 0':<16}{'y = +1':<16}{'y = -1':<16}{'Other':<16}{'Other Values'}")
    print("-" * 105)
    for seed_str, sdata in label_dist.items():
        for split_str in ("train", "validation", "test"):
            sp = sdata[split_str]
            c = sp["counts"]
            p = sp["proportions"]
            ov = ", ".join(f"{v:+.4f}" for v in sp["other_values"]) if sp["other_values"] else "none"
            print(
                f"{seed_str:<6}{split_str:<12}{sp['total_items']:<8}"
                f"{c['zero']} ({p['zero']:.1%}){' ':<5}"
                f"{c['plus_one']} ({p['plus_one']:.1%}){' ':<5}"
                f"{c['minus_one']} ({p['minus_one']:.1%}){' ':<5}"
                f"{c['other']} ({p['other']:.2%}){' ':<5}"
                f"{ov}"
            )

    print("\n" + "=" * 80)
    print("TABLE 2: Stratified Test Errors (NC-R0, NC-none, A, R) and D = C - F Interval")
    print("=" * 80)
    print(
        f"{'Dataset':<16}{'Stratum':<10}{'Items':<8}{'C':<10}{'F':<10}{'M':<10}{'P':<10}{'A':<10}{'R':<10}{'D = C - F':<12}{'95% CI':<24}"
    )
    print("-" * 120)
    for ds_key, ds_res in stratified_errors.items():
        for sname, s_label in (("all", "All"), ("y_eq_0", "y = 0"), ("y_neq_0", "y != 0")):
            st = ds_res["strata"][sname]
            pts = st["points"]
            d_ent = st["D_NC_R0"]
            d_pt = d_ent["point"]
            lo = d_ent["interval"]["lower"]
            hi = d_ent["interval"]["upper"]
            ci_str = f"[{lo:+.5f}, {hi:+.5f}]"
            print(
                f"{ds_key:<16}{s_label:<10}{st['item_count']:<8}"
                f"{pts['NC_R0_C']:.5f}   {pts['NC_R0_F']:.5f}   {pts['NC_none_M']:.5f}   "
                f"{pts['NC_R0_P']:.5f}   {pts['A']:.5f}   {pts['R']:.5f}   "
                f"{d_pt:+.5f}     {ci_str:<24}"
            )

    print("\n" + "=" * 80)
    print("TABLE 3: Share of C's Error Coming from y != 0 Items")
    print("=" * 80)
    print(f"{'Dataset':<16}{'y != 0 Item Share':<22}{'NC-R0 C Error Share':<24}{'NC-none C Error Share':<24}")
    print("-" * 86)
    for ds_key, ds_res in stratified_errors.items():
        all_cnt = ds_res["strata"]["all"]["item_count"]
        nz_cnt = ds_res["strata"]["y_neq_0"]["item_count"]
        item_share = nz_cnt / all_cnt
        sh_r0 = ds_res["share_of_c_error_from_y_neq_0"]["NC_R0_C"]
        sh_none = ds_res["share_of_c_error_from_y_neq_0"]["NC_none_C"]
        print(f"{ds_key:<16}{nz_cnt}/{all_cnt} ({item_share:.2%}){' ':<7}{sh_r0:.2%}{' ':<17}{sh_none:.2%}")

    print("\n" + "=" * 80)
    print("TABLE 4: Continuous Positive Control (Part A, R5 and R3-TFI)")
    print("=" * 80)
    print(f"{'Row':<32}{'Rung':<10}{'Distinct y':<12}{'Label Range':<24}{'D [95% CI]':<32}{'D/C [95% CI]'}")
    print("-" * 125)
    for row_key, rungs in continuous_pc["rows"].items():
        continuity = continuous_pc["spin_chain_label_continuity"][row_key]
        n_dist = continuity["distinct_test_labels"]
        l_min, l_max = continuity["min_label"], continuity["max_label"]
        range_str = f"[{l_min:.4f}, {l_max:.4f}]"

        for rung_name in ("R5", "R3-TFI"):
            if rung_name in rungs:
                rd = rungs[rung_name]
                d = rd["D"]
                doc = rd["D_over_C"]
                d_str = f"{d['point']:+.5f} [{d['interval']['lower']:+.5f}, {d['interval']['upper']:+.5f}]"
                doc_str = f"{doc['point']:.2%} [{doc['interval']['lower']:.2%}, {doc['interval']['upper']:.2%}]"
                print(f"{row_key:<32}{rung_name:<10}{n_dist:<12}{range_str:<24}{d_str:<32}{doc_str}")


if __name__ == "__main__":
    main()
