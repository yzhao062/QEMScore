#!/usr/bin/env python3
"""Predictions file computation for the shot-count sweep of QEMScore.

Frozen rule: docs/frozen-rules/2026-10-03-shot-sweep.md
Section: Prediction Computed Before Any Fit That Reads r

Computes:
- Per level and rule cell:
  - c_m: test MAE of de-attenuated estimate (r - a)/b (or inf if |b| <= 1e-12)
  - c_r: test MAE of per-cell linear calibration of r
  - two-fold cross-fitted c_m on validation rows
- Per rung and rule cell:
  - c_C: mean over learner seeds 1..20 from sweep C fits and reference C fits
- Per classification cell:
  - D/C* predicted ceiling: 1 - macro(c_comb) / macro(c_C)
  - D/C* descriptive ceiling using c_r
  - "adds" threshold max(0.10, 2h)
  - predicted label ("measurement_adds" if D/C* >= thresh, "not_distinguished" if D/C* <= 0.05, else None)
  - reference label and baseline
  - label-changing flag
- Summary counts: H1 parts (b) and (c) qualifying cells/rows, H2 label-changing cells/rows, predicted cells.

Runs 2,048-shot self-check against posthoc-measurement-floor.json before writing.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any
from collections.abc import Sequence
import warnings
from zoneinfo import ZoneInfo

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
RUNGS = ("R0", "N1", "N2", "R5")
CANONICAL_LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")

RULE_FILE = "docs/frozen-rules/2026-10-03-shot-sweep.md"

RULE_THRESHOLDS = {
    "shipped-s101-n640/tfi": {"R0": 0.385, "N1": 0.159, "N2": 0.100},
    "shipped-s101-n640/heisenberg": {"R0": 0.446, "N1": 0.208, "N2": 0.128},
    "shipped-s211-n640/tfi": {"R0": 0.490, "N1": 0.184, "N2": 0.100},
    "shipped-s211-n640/heisenberg": {"R0": 0.360, "N1": 0.225, "N2": 0.108},
    "shipped-s307-n640/tfi": {"R0": 0.547, "N1": 0.100, "N2": 0.100},
    "shipped-s307-n640/heisenberg": {"R0": 0.247, "N1": 0.194, "N2": 0.100},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_relative(p: Path | str, base: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(base.resolve()))
    except ValueError:
        return os.path.relpath(Path(p).resolve(), base.resolve())


def find_cache_file(cache_dir: Path, seed: int) -> Path:
    cache_dir = Path(cache_dir)
    candidates = [
        cache_dir / f"shipped-s{seed}-n640.pkl",
        cache_dir / f"shipped-s{seed}-n640.json.gz",
        cache_dir / f"shipped-s{seed}-n640.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    matches = sorted(cache_dir.glob(f"*s{seed}*.pkl")) + sorted(cache_dir.glob(f"*s{seed}*.json*"))
    matches = [m for m in matches if not m.name.endswith(".order.json")]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No cache file for seed {seed} found in {cache_dir}")


def load_cache(cache_path: Path) -> dict:
    if cache_path.name.endswith(".pkl"):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    elif cache_path.name.endswith(".json.gz"):
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    elif cache_path.name.endswith(".json"):
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported cache file format: {cache_path}")


def load_c_fit_predictions(fits_dir: Path, seed: int, rung: str) -> np.ndarray:
    """Test predictions of C at learner seeds 1 to 20, shape (20, n_test_items).

    Every seed is required: an interrupted first fitting stage must stop the
    prediction instead of averaging whatever fits exist.
    """
    fits_dir = Path(fits_dir)
    preds = []
    for k in range(1, 21):
        path = fits_dir / f"shipped-s{seed}-n640__{rung}__k{k:02d}__C.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{fits_dir}: C fits for shipped-s{seed}-n640/{rung} must include every "
                f"learner seed from 1 through 20; {path.name} is missing")
        with np.load(path) as data:
            preds.append(np.asarray(data["test"], dtype=float))
    return np.asarray(preds, dtype=float)


def compute_c_C_for_fits(
    fits_dir: Path,
    cache_by_seed: dict[int, dict],
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict[str, float]:
    """Compute c_C for each row, rung, and rule cell from C fits."""
    c_C_dict = {}
    for seed in dataset_seeds:
        key = f"shipped-s{seed}-n640"
        test_items = cache_by_seed[seed]["test"]
        for rung in RUNGS:
            c_preds = load_c_fit_predictions(fits_dir, seed, rung)
            n_seeds = c_preds.shape[0]

            for fam in FAMILIES:
                row_key = f"{key}/{fam}"
                for sev, obs in CELLS_DEF:
                    cell_k = f"{row_key}/{rung}/{sev}/{obs}"
                    cell_idx = [
                        i
                        for i, r in enumerate(test_items)
                        if str(r.get("family")) == fam
                        and str(r.get("severity")) == sev
                        and str(r.get("observable")) == obs
                    ]
                    if not cell_idx:
                        raise ValueError(f"No test items for {cell_k}")
                    y_test = np.asarray(
                        [test_items[i]["ideal_expectation"] for i in cell_idx], dtype=float
                    )

                    seed_maes = []
                    for s in range(n_seeds):
                        p_s = c_preds[s, cell_idx]
                        mae_s = math.fsum(np.abs(p_s - y_test).tolist()) / len(cell_idx)
                        seed_maes.append(mae_s)
                    c_C_dict[cell_k] = float(np.mean(seed_maes))
    return c_C_dict


def append_run_log(log_path: Path, output_path: Path, command: str) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_sha = sha256_file(output_path)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    try:
        now_pac = now_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    except Exception:
        now_pac = now_utc.astimezone(datetime.timezone(datetime.timedelta(hours=-7)))
    line = f"{out_sha} {now_utc.isoformat()} {now_pac.isoformat()} {command}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def run_predict(
    level_dirs: dict[str, Path],
    c_fits_dir: Path,
    reference_c_fits_dir: Path,
    reference_analysis_path: Path,
    measurement_floor_path: Path,
    out_path: Path,
    run_log_path: Path,
    command_str: str,
    require_all_levels: bool = True,
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict:
    repo_root = _REPO
    if require_all_levels and set(level_dirs) != set(CANONICAL_LEVELS):
        raise ValueError(
            f"the frozen prediction requires exactly the seven shot levels "
            f"{list(CANONICAL_LEVELS)}; got {sorted(level_dirs)}")

    # 1. Load reference analysis and measurement floor
    if not reference_analysis_path.exists():
        raise FileNotFoundError(f"Reference analysis file not found: {reference_analysis_path}")
    ref_analysis = json.loads(reference_analysis_path.read_text(encoding="utf-8"))
    ref_rows = ref_analysis["parts"]["A"]["rows"]

    if not measurement_floor_path.exists():
        raise FileNotFoundError(f"Measurement floor file not found: {measurement_floor_path}")
    floor_payload = json.loads(measurement_floor_path.read_text(encoding="utf-8"))
    floor_dict = floor_payload.get("per_cell_floor", floor_payload)

    # 2. Load caches for each level
    level_caches: dict[str, dict[int, dict]] = {}
    for lvl_name, p in level_dirs.items():
        cache_dir = p / "cache" if (p / "cache").is_dir() else p
        level_caches[lvl_name] = {}
        for seed in dataset_seeds:
            cf = find_cache_file(cache_dir, seed)
            level_caches[lvl_name][seed] = load_cache(cf)

    # Reference 2,048 cache used for C test alignment
    if "2048" in level_caches:
        cache_2048 = level_caches["2048"]
    elif require_all_levels:
        raise ValueError("the 2,048-shot level is required")
    else:
        # Helper tests on a partial level set only; the command line never gets here.
        cache_2048 = next(iter(level_caches.values()))

    # 3. Compute c_C from sweep fits and reference fits
    c_C_sweep = compute_c_C_for_fits(c_fits_dir, cache_2048, dataset_seeds=dataset_seeds)
    c_C_ref = compute_c_C_for_fits(reference_c_fits_dir, cache_2048, dataset_seeds=dataset_seeds)

    # 4. Compute per-level and rule-cell quantities: c_m, c_r, cross-fitted c_m
    rule_cells_out: dict[str, dict[str, Any]] = {}
    level_2048_rule_cells: dict[str, dict[str, Any]] = {}

    for lvl_name, seed_cache in level_dirs.items():
        caches = level_caches[lvl_name]
        for seed in dataset_seeds:
            key = f"shipped-s{seed}-n640"
            v_items = caches[seed]["validation"]
            t_items = caches[seed]["test"]

            for fam in FAMILIES:
                row_key = f"{key}/{fam}"
                for sev, obs in CELLS_DEF:
                    cell_key = f"{row_key}/{sev}/{obs}"
                    level_cell_key = f"{lvl_name}/{cell_key}"

                    v_sub = [
                        r
                        for r in v_items
                        if str(r.get("family")) == fam
                        and str(r.get("severity")) == sev
                        and str(r.get("observable")) == obs
                    ]
                    t_sub = [
                        r
                        for r in t_items
                        if str(r.get("family")) == fam
                        and str(r.get("severity")) == sev
                        and str(r.get("observable")) == obs
                    ]

                    y_v = np.asarray([r["ideal_expectation"] for r in v_sub], dtype=float)
                    r_v = np.asarray([r["noisy_expectation"] for r in v_sub], dtype=float)
                    y_t = np.asarray([r["ideal_expectation"] for r in t_sub], dtype=float)
                    r_t = np.asarray([r["noisy_expectation"] for r in t_sub], dtype=float)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)
                        # (a) OLS regression of r on y: r = a + b * y
                        b, a = np.polyfit(y_v, r_v, 1)
                        if abs(b) <= 1e-12:
                            c_m = float("inf")
                        else:
                            r_deatt_t = (r_t - a) / b
                            c_m = float(math.fsum(np.abs(r_deatt_t - y_t).tolist()) / len(y_t))

                        # (b) Linear calibration of r: y ~ a_cal + c * r
                        c, a_cal = np.polyfit(r_v, y_v, 1)
                        y_cal_t = a_cal + c * r_t
                        c_r = float(math.fsum(np.abs(y_cal_t - y_t).tolist()) / len(y_t))

                        # (c) Two-fold cross-fitting on validation rows
                        n_v = len(v_sub)
                        half = n_v // 2
                        if half == 0:
                            c_m_cross = c_m
                        else:
                            b1, a1 = np.polyfit(y_v[:half], r_v[:half], 1)
                            if abs(b1) <= 1e-12:
                                pred2 = np.full(n_v - half, np.inf)
                            else:
                                pred2 = (r_v[half:] - a1) / b1

                            b2, a2 = np.polyfit(y_v[half:], r_v[half:], 1)
                            if abs(b2) <= 1e-12:
                                pred1 = np.full(half, np.inf)
                            else:
                                pred1 = (r_v[:half] - a2) / b2

                            pred_cv = np.concatenate([pred1, pred2])
                            if np.any(np.isinf(pred_cv)):
                                c_m_cross = float("inf")
                            else:
                                c_m_cross = float(math.fsum(np.abs(pred_cv - y_v).tolist()) / n_v)

                    rc_record = {
                        "level": lvl_name,
                        "row": row_key,
                        "severity": sev,
                        "observable": obs,
                        "slope_b": float(b),
                        "intercept_a": float(a),
                        "c_m": c_m,
                        "c_r": c_r,
                        "c_m_cross_fitted": c_m_cross,
                        "calibration_slope_c": float(c),
                        "calibration_intercept_a": float(a_cal),
                    }
                    rule_cells_out[level_cell_key] = rc_record

                    if lvl_name == "2048":
                        level_2048_rule_cells[cell_key] = rc_record

    # 5. 2,048-shot self-check against per_cell_floor
    if "2048" in level_dirs:
        diffs = []
        for cell_key, rc in sorted(level_2048_rule_cells.items()):
            if cell_key not in floor_dict:
                diffs.append(f"{cell_key}: missing from per_cell_floor")
                continue
            ref_floor = floor_dict[cell_key]
            d_cr = abs(rc["c_r"] - ref_floor["floor_mae"])
            d_a = abs(rc["intercept_a"] - ref_floor["intercept_a"])
            d_b = abs(rc["slope_b"] - ref_floor["slope_b"])
            if d_cr > 1e-12 or d_a > 1e-12 or d_b > 1e-12:
                diffs.append(
                    f"{cell_key}: "
                    f"c_r={rc['c_r']:.12e} vs floor_mae={ref_floor['floor_mae']:.12e} (diff={d_cr:.2e}); "
                    f"a={rc['intercept_a']:.12e} vs intercept_a={ref_floor['intercept_a']:.12e} (diff={d_a:.2e}); "
                    f"b={rc['slope_b']:.12e} vs slope_b={ref_floor['slope_b']:.12e} (diff={d_b:.2e})"
                )
        if diffs:
            print("SELF-CHECK FAILURE: 2,048-shot calibration floor mismatch!", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            sys.exit(1)

    # 6. Compute classification cells and thresholds
    classification_cells: dict[str, dict[str, Any]] = {}
    thresholds_computed: dict[str, dict[str, float]] = {}

    for lvl_name in level_dirs.keys():
        for seed in dataset_seeds:
            key = f"shipped-s{seed}-n640"
            for fam in FAMILIES:
                row_key = f"{key}/{fam}"
                for rung in RUNGS:
                    class_key = f"{lvl_name}/{row_key}/{rung}"

                    # Compute macro c_C, macro c_comb, macro c_comb_cr over the 4 rule cells
                    c_C_vals = []
                    c_comb_vals = []
                    c_comb_cr_vals = []

                    for sev, obs in CELLS_DEF:
                        c_k = f"{row_key}/{rung}/{sev}/{obs}"
                        rc_k = f"{lvl_name}/{row_key}/{sev}/{obs}"

                        c_C_val = c_C_sweep[c_k]
                        rc_val = rule_cells_out[rc_k]
                        c_m_val = rc_val["c_m"]
                        c_r_val = rc_val["c_r"]

                        if math.isinf(c_m_val) or abs(rc_val["slope_b"]) <= 1e-12:
                            c_comb = c_C_val
                        else:
                            c_comb = (c_C_val**-2.0 + c_m_val**-2.0)**-0.5

                        c_comb_cr = (c_C_val**-2.0 + c_r_val**-2.0)**-0.5

                        c_C_vals.append(c_C_val)
                        c_comb_vals.append(c_comb)
                        c_comb_cr_vals.append(c_comb_cr)

                    macro_c_C = float(np.mean(c_C_vals))
                    macro_c_comb = float(np.mean(c_comb_vals))
                    macro_c_comb_cr = float(np.mean(c_comb_cr_vals))

                    D_over_C_star = 1.0 - (macro_c_comb / macro_c_C) if macro_c_C > 0 else 0.0
                    D_over_C_star_cr = 1.0 - (macro_c_comb_cr / macro_c_C) if macro_c_C > 0 else 0.0

                    # Reference analysis interval & threshold
                    if row_key in ref_rows:
                        ref_rung = ref_rows[row_key]["rungs"][rung]
                        ci = ref_rung["D_over_C"]["interval"]
                        h = (ci["upper"] - ci["lower"]) / 2.0
                        raw_thresh = max(0.10, 2.0 * h)
                        adds_thresh = round(raw_thresh, 3)
                        ref_label = ref_rung["classification"]["label"]
                        ref_baseline = "not_distinguished" if ref_label == "measurement_hurts" else ref_label
                    else:
                        h = float("nan")
                        adds_thresh = 0.10
                        ref_label = None
                        ref_baseline = None

                    thresholds_computed.setdefault(row_key, {})[rung] = adds_thresh

                    # Label prediction
                    if D_over_C_star >= adds_thresh:
                        pred_label = "measurement_adds"
                    elif D_over_C_star <= 0.05:
                        pred_label = "not_distinguished"
                    else:
                        pred_label = None

                    label_changing = bool(pred_label is not None and pred_label != ref_baseline)

                    classification_cells[class_key] = {
                        "level": lvl_name,
                        "row": row_key,
                        "rung": rung,
                        "macro_c_C": macro_c_C,
                        "macro_c_comb": macro_c_comb,
                        "macro_c_comb_cr": macro_c_comb_cr,
                        "D_over_C_star": D_over_C_star,
                        "D_over_C_star_descriptive_cr": D_over_C_star_cr,
                        "h": float(h),
                        "adds_threshold": adds_thresh,
                        "predicted_label": pred_label,
                        "reference_label": ref_label,
                        "reference_label_baseline": ref_baseline,
                        "label_changing": label_changing,
                    }

    # 7. Assert computed thresholds match the rule's table
    for row, rung_map in RULE_THRESHOLDS.items():
        if row in thresholds_computed:
            for rung, exp_thresh in rung_map.items():
                calc_thresh = thresholds_computed[row][rung]
                assert calc_thresh == exp_thresh, (
                    f"Threshold mismatch for {row} {rung}: computed {calc_thresh} != table {exp_thresh}"
                )

    # 8. Compute required counts
    non_2048_cells = [c for c in classification_cells.values() if str(c["level"]) != "2048"]

    # H1 qualifying: N1 and N2, non-2048, D_over_C_star >= 0.10
    h1_qual_cells = [c for c in non_2048_cells if c["rung"] in ("N1", "N2") and c["D_over_C_star"] >= 0.10]
    h1_qual_cell_count = len(h1_qual_cells)
    h1_qual_row_count = len({c["row"] for c in h1_qual_cells})
    h1_testable = bool(h1_qual_cell_count >= 10 and h1_qual_row_count >= 2)

    # H2 label-changing: R0, N1, N2, non-2048, label_changing == True
    h2_cells = [c for c in non_2048_cells if c["rung"] in ("R0", "N1", "N2")]
    h2_lc_cells = [c for c in h2_cells if c["label_changing"]]
    h2_lc_cell_count = len(h2_lc_cells)
    h2_lc_row_count = len({c["row"] for c in h2_lc_cells})
    h2_testable = bool(h2_lc_cell_count >= 6 and h2_lc_row_count >= 2)

    h2_predicted_cells = [c for c in h2_cells if c["predicted_label"] is not None]
    h2_predicted_cell_count = len(h2_predicted_cells)

    counts = {
        "h1_qualifying_cells": h1_qual_cell_count,
        "h1_qualifying_rows": h1_qual_row_count,
        "h1_testable": h1_testable,
        "h2_label_changing_cells": h2_lc_cell_count,
        "h2_label_changing_rows": h2_lc_row_count,
        "h2_testable": h2_testable,
        "h2_predicted_cells": h2_predicted_cell_count,
        "total_classification_cells": len(classification_cells),
    }

    # Format c_C output table
    c_C_output = {}
    for cell_k, sweep_val in sorted(c_C_sweep.items()):
        c_C_output[cell_k] = {
            "c_C": sweep_val,
            "c_C_reference": c_C_ref[cell_k],
        }

    rule_path = repo_root / RULE_FILE
    rule_sha = sha256_file(rule_path) if rule_path.exists() else None
    script_sha = sha256_file(Path(__file__).resolve())

    output_payload = {
        "schema": "shot-sweep-predictions-v1",
        "status": "pre-registered prediction",
        "rule_file": RULE_FILE,
        "rule_file_sha256": rule_sha,
        "script": "tools/shot_sweep_predict.py",
        "script_sha256": script_sha,
        "inputs": {
            "levels": {lvl: make_relative(p, repo_root) for lvl, p in level_dirs.items()},
            "c_fits": make_relative(c_fits_dir, repo_root),
            "reference_c_fits": make_relative(reference_c_fits_dir, repo_root),
            "reference_analysis": make_relative(reference_analysis_path, repo_root),
            "measurement_floor": make_relative(measurement_floor_path, repo_root),
        },
        "counts": counts,
        "rule_cells": rule_cells_out,
        "c_C_by_rung_cell": c_C_output,
        "classification_cells": classification_cells,
    }

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".tmp")
    tmp_out.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    tmp_out.replace(out_path)
    print(f"Wrote predictions to {out_path}")

    append_run_log(run_log_path, out_path, command_str)
    print(f"Recorded SHA-256 and timestamp to run log {run_log_path}")
    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        action="append",
        required=True,
        help="Shot level and directory in format <shot_level>=<ladder out dir or cache dir>",
    )
    parser.add_argument(
        "--c-fits",
        type=Path,
        required=True,
        help="Directory containing this sweep's 2,048-shot C fits",
    )
    parser.add_argument(
        "--reference-c-fits",
        type=Path,
        required=True,
        help="Directory containing the 2026-10-03 reference C fits",
    )
    parser.add_argument(
        "--reference-analysis",
        type=Path,
        required=True,
        help="Path to analysis-a.json of 2026-10-03",
    )
    parser.add_argument(
        "--measurement-floor",
        type=Path,
        required=True,
        help="Path to posthoc-measurement-floor.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for predictions.json",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        required=True,
        help="Run log file to append output SHA-256 and timestamps",
    )
    parser.add_argument(
        "--dataset-seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Dataset seeds (default: %(default)s)",
    )

    args = parser.parse_args()

    level_dirs: dict[str, Path] = {}
    for entry in args.level:
        if "=" not in entry:
            raise ValueError(f"Invalid --level format '{entry}', expected <shot_level>=<path>")
        lvl, p = entry.split("=", 1)
        level_dirs[lvl.strip()] = Path(p.strip())

    run_predict(
        level_dirs=level_dirs,
        c_fits_dir=args.c_fits,
        reference_c_fits_dir=args.reference_c_fits,
        reference_analysis_path=args.reference_analysis,
        measurement_floor_path=args.measurement_floor,
        out_path=args.out,
        run_log_path=args.run_log,
        command_str=" ".join(sys.argv),
        dataset_seeds=tuple(args.dataset_seeds),
    )


if __name__ == "__main__":
    main()
