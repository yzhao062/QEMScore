#!/usr/bin/env python3
"""Post hoc constant prediction and training-median references for near-Clifford datasets.

For each near-Clifford dataset (nc-s101, nc-s211, nc-s307) and the rungs
NC-R0 and NC-none, computes the test macro MAE of:
(a) the constant prediction 0
(b) the per-cell training-median prediction (from the full caches' training rows).

With script A's two-stage bootstrap counts (bootstrap_draws, seed 20261002,
learner seeds then circuits), gives intervals for:
- constant-zero error
- zero - F (F's improvement over the constant)
- zero - C
- ratio (zero - F) / zero
for both the original fits and the strength-indicator fits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "tools"))
sys.path.insert(0, str(repo))

import descriptor_ladder_analysis as dla  # noqa: E402

NC_KEYS = (f"nc-s{s}-n640" for s in dla.DATASET_SEEDS)


def compute_constant_references(
    original_fit_dirs: list[Path],
    strength_fit_dirs: list[Path],
    cache_dirs: list[Path],
    full_cache_dirs: list[Path] | None = None,
    out_path: Path | None = None,
    *,
    analysis_a_orig_path: Path | None = None,
    analysis_a_strength_path: Path | None = None,
    draws: int = dla.DEFAULT_DRAWS,
    seed: int = dla.RULE_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    test_caches, _ = dla.discover_caches([Path(d) for d in cache_dirs])
    full_caches: dict[str, Path] = {}
    if full_cache_dirs:
        full_caches, _ = dla.discover_caches([Path(d) for d in full_cache_dirs])

    orig_store = dla.FitStore([Path(d) for d in original_fit_dirs], False)
    str_store = dla.FitStore([Path(d) for d in strength_fit_dirs], False)
    spec = dla.PARTS["NC"]

    orig_analysis_a = None
    if analysis_a_orig_path and Path(analysis_a_orig_path).exists():
        with open(analysis_a_orig_path, "r", encoding="utf-8") as f:
            orig_analysis_a = json.load(f).get("parts", {}).get("NC", {}).get("rows", {})

    str_analysis_a = None
    if analysis_a_strength_path and Path(analysis_a_strength_path).exists():
        with open(analysis_a_strength_path, "r", encoding="utf-8") as f:
            str_analysis_a = json.load(f).get("parts", {}).get("NC", {}).get("rows", {})

    out: dict[str, Any] = {
        "schema": "near-clifford-posthoc-constant-reference-v1",
        "status": "post hoc, descriptive; constant prediction and training-median references for near-Clifford positive control",
        "analysis_script_sha256": dla.sha256_file(Path(dla.__file__)),
        "bootstrap": {
            "seed": seed,
            "draws": draws,
            "percentiles": list(dla.PERCENTILES),
        },
        "coordinator_checks": {
            "expected_constant_zero_macro_mae": {
                "nc-s101-n640": 0.0934,
                "nc-s211-n640": 0.0732,
                "nc-s307-n640": 0.0918,
            },
            "tolerance_analysis_a_check": dla.POINT_CHECK_TOLERANCE,
        },
        "datasets": {},
    }

    all_constant_zero_below_c = True

    for dataset_seed in dla.DATASET_SEEDS:
        key = f"nc-s{dataset_seed}-n640"
        if key not in test_caches:
            continue
        test_cache_data = dla.load_cache(test_caches[key])
        row = dla.Row(test_cache_data, None, key)
        cells = row.all_cells

        # Constant prediction 0
        err_zero = np.abs(row.ideal)
        zero_point = dla.point_macro(err_zero, row.members, cells)

        # Bootstrap draws for constant 0
        _, circuit_counts = dla.bootstrap_draws(20, row.n_circuits, draws, seed)
        zero_macro = dla.macro_under_weights(
            err_zero[None, :], row.item_circuit, row.members, cells, circuit_counts
        )
        zero_draws = zero_macro[:, 0]
        zero_entry = dla.interval_entry(zero_draws, zero_point, None)
        zero_entry["draws_seed_count"] = 20

        # Training-median prediction
        # Find train items: in test_cache_data if present, else in full_caches
        train_items = test_cache_data.get("train")
        if not train_items and key in full_caches:
            full_data = dla.load_cache(full_caches[key])
            train_items = full_data.get("train")

        median_point = None
        cell_medians_dict = {}
        if train_items:
            train_by_cell: dict[tuple[str, str, str], list[float]] = {}
            for r in train_items:
                c = tuple(str(r[f]) for f in dla.CELL_FIELDS)
                train_by_cell.setdefault(c, []).append(float(r["ideal_expectation"]))
            cell_medians = {
                c: float(np.median(vals)) for c, vals in train_by_cell.items()
            }
            cell_medians_dict = {
                "/".join(c): med for c, med in sorted(cell_medians.items())
            }
            pred_median = np.array(
                [cell_medians.get(row.cells[c_idx], 0.0) for c_idx in row.item_cell]
            )
            err_median = np.abs(pred_median - row.ideal)
            median_point = dla.point_macro(err_median, row.members, cells)

        ds_out: dict[str, Any] = {
            "dataset_seed": dataset_seed,
            "constant_zero": {
                "point": zero_point,
                "interval": zero_entry["interval"],
                "excludes_zero_above": zero_entry["excludes_zero_above"],
                "excludes_zero_below": zero_entry["excludes_zero_below"],
                "draw_sd": zero_entry["draw_sd"],
            },
            "training_median": {
                "point": median_point,
                "cell_medians": cell_medians_dict,
            },
            "fits": {},
        }

        if verbose:
            print(f"=== {key} ===")
            z_lo = zero_entry["interval"]["lower"]
            z_hi = zero_entry["interval"]["upper"]
            print(f"Constant zero MAE: {zero_point:.6f} [{z_lo:.6f}, {z_hi:.6f}]")
            if median_point is not None:
                print(f"Training median MAE: {median_point:.6f}")

        # Fits evaluation
        for fit_name, store, analysis_a in (
            ("original", orig_store, orig_analysis_a),
            ("strength_indicator", str_store, str_analysis_a),
        ):
            est = dla.RowEstimator(row, store, draws, seed, 20)
            fit_rungs: dict[str, Any] = {}

            for rung in spec["rungs"]:
                arms = dla.arms_at(store, spec, key, rung)
                f_entry, f_draws = est.mean_arm(arms["F"])
                c_entry, c_draws = est.mean_arm(arms["C"])

                if f_draws is None or c_draws is None:
                    continue

                f_point = f_entry["point"]
                c_point = c_entry["point"]

                # Check against analysis-a.json
                if analysis_a and key in analysis_a:
                    rung_data = analysis_a[key]["rungs"].get(rung, {})
                    exp_c = (
                        rung_data.get("means", {}).get("C", {}).get("point")
                    )
                    exp_f = (
                        rung_data.get("means", {}).get("F", {}).get("point")
                    )
                    if exp_c is not None:
                        diff_c = abs(c_point - exp_c)
                        assert diff_c <= dla.POINT_CHECK_TOLERANCE, (
                            f"{fit_name} {key} {rung}: mean C {c_point} disagrees with "
                            f"analysis-a {exp_c} (diff {diff_c})"
                        )
                    if exp_f is not None:
                        diff_f = abs(f_point - exp_f)
                        assert diff_f <= dla.POINT_CHECK_TOLERANCE, (
                            f"{fit_name} {key} {rung}: mean F {f_point} disagrees with "
                            f"analysis-a {exp_f} (diff {diff_f})"
                        )

                if rung == "NC-R0" and zero_point >= c_point:
                    all_constant_zero_below_c = False

                # zero - F
                diff_f_point = zero_point - f_point
                diff_f_draws = zero_draws - f_draws
                entry_zf = dla.interval_entry(diff_f_draws, diff_f_point, 20)

                # zero - C
                diff_c_point = zero_point - c_point
                diff_c_draws = zero_draws - c_draws
                entry_zc = dla.interval_entry(diff_c_draws, diff_c_point, 20)

                # (zero - F) / zero
                ratio_point = diff_f_point / zero_point
                ratio_draws = diff_f_draws / zero_draws
                entry_ratio = dla.interval_entry(ratio_draws, ratio_point, 20)

                rung_entry = {
                    "mean_C": c_point,
                    "mean_F": f_point,
                    "constant_zero_error": zero_entry,
                    "zero_minus_F": entry_zf,
                    "zero_minus_C": entry_zc,
                    "ratio_zero_minus_F_over_zero": entry_ratio,
                }
                fit_rungs[rung] = rung_entry

                if verbose:
                    zf_lo = entry_zf["interval"]["lower"]
                    zf_hi = entry_zf["interval"]["upper"]
                    zc_lo = entry_zc["interval"]["lower"]
                    zc_hi = entry_zc["interval"]["upper"]
                    r_pt = entry_ratio["point"]
                    r_lo = entry_ratio["interval"]["lower"]
                    r_hi = entry_ratio["interval"]["upper"]
                    print(f"  {fit_name} {rung}:")
                    print(f"    mean F: {f_point:.6f}, mean C: {c_point:.6f}")
                    print(
                        f"    zero - F: {diff_f_point:+.6f} [{zf_lo:+.6f}, {zf_hi:+.6f}]"
                    )
                    print(
                        f"    zero - C: {diff_c_point:+.6f} [{zc_lo:+.6f}, {zc_hi:+.6f}]"
                    )
                    print(
                        f"    (zero - F)/zero: {r_pt:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]"
                    )

            ds_out["fits"][fit_name] = fit_rungs

        out["datasets"][key] = ds_out

    out["coordinator_checks"][
        "all_constant_zero_below_mean_c_at_NC_R0"
    ] = all_constant_zero_below_c

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
        if verbose:
            print("wrote", out_path)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--original-fits",
        action="append",
        type=Path,
        default=[],
        help="Directories of original fits (repeatable)",
    )
    parser.add_argument(
        "--strength-fits",
        action="append",
        type=Path,
        default=[],
        help="Directories of strength-indicator fits (repeatable)",
    )
    parser.add_argument(
        "--cache",
        action="append",
        type=Path,
        default=[],
        help="Cache directories for test items (repeatable)",
    )
    parser.add_argument(
        "--full-cache",
        action="append",
        type=Path,
        default=[],
        help="Cache directories with training rows for median reference (repeatable)",
    )
    parser.add_argument(
        "--analysis-a-orig",
        type=Path,
        default=repo / "artifacts" / "descriptor-information" / "analysis-a.json",
        help="Path to original analysis-a.json for assertion check",
    )
    parser.add_argument(
        "--analysis-a-strength",
        type=Path,
        default=repo
        / "artifacts"
        / "descriptor-information"
        / "strength-indicator"
        / "analysis-a.json",
        help="Path to strength-indicator analysis-a.json for assertion check",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo
        / "artifacts"
        / "near-clifford-positive-control"
        / "posthoc-constant-reference.json",
        help="Output JSON path",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=dla.RULE_SEED)
    parser.add_argument("--draws", type=int, default=dla.DEFAULT_DRAWS)
    parser.add_argument("--quiet", action="store_true", help="Suppress printed lines")
    args = parser.parse_args(argv)

    missing = [name for name, value in (
        ("--original-fits", args.original_fits), ("--strength-fits", args.strength_fits),
        ("--cache", args.cache), ("--full-cache", args.full_cache)) if not value]
    if missing:
        parser.error("required: " + ", ".join(missing))
    original_fits = args.original_fits
    strength_fits = args.strength_fits
    caches = args.cache
    full_caches = args.full_cache

    compute_constant_references(
        original_fit_dirs=original_fits,
        strength_fit_dirs=strength_fits,
        cache_dirs=caches,
        full_cache_dirs=full_caches,
        out_path=args.out,
        analysis_a_orig_path=args.analysis_a_orig,
        analysis_a_strength_path=args.analysis_a_strength,
        draws=args.draws,
        seed=args.bootstrap_seed,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
