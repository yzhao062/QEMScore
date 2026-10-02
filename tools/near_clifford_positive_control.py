#!/usr/bin/env python3
"""Positive control: Near-Clifford under-determined descriptors (setting S0).

Generates and validates datasets under data/nc-s{seed}-n640/ for seeds (101, 211, 307).
Fits arms A (feat-only) and R (raw) once per dataset.
Fits arms F (liao), C (liao-feat-only), and P (liao-training-shuffle) for k = 1..20
and anchor k = dataset_seed.
Records convergence flags, selected candidates, macro MAE endpoints,
test predictions (compressed), and summary statistics.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import qemscore
from qemscore.baselines.controls import shuffle_noisy_items
from qemscore.baselines.liao import LiaoMitigator
from qemscore.campaign.analysis import _arm_predict, _fit_arm
from qemscore.datasets.schema import FEATURES, build_features
from qemscore.datasets.split_generate import generate_split
from qemscore.datasets.splits import SplitSpec
from qemscore.runner.run import _prediction_items, _prediction_manifest
from qemscore.validation import validate_split_artifact


DATASET_SEEDS = (101, 211, 307)
LEARNER_SEEDS = tuple(range(1, 21))
N_QUBITS = 10
NATIVE_DEPTH = 4
NON_CLIFFORD_COUNTS = [1, 2, 3]
THETA_CHOICES = [0.4487989505128276, 0.6283185307179586]
ROLE_COUNTS = {"train": 640, "validation": 320, "test": 160}
NOISE_FAMILY = "depolarizing_readout"
SEVERITIES = ["L1", "L3"]
OBSERVABLES = ["z_mid", "zz_mid"]
SHOTS = 2048


def get_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_nc_spec() -> SplitSpec:
    """Build the frozen SplitSpec for near_clifford positive control."""
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": [NOISE_FAMILY],
            "noise_strength": list(SEVERITIES),
            "circuit_family": ["near_clifford"],
            "family_native_depth": [NATIVE_DEPTH],
            "observable_class": list(OBSERVABLES),
            "shots": [SHOTS],
        },
        n_qubits=[N_QUBITS],
        role_counts=dict(ROLE_COUNTS),
        family_parameters={
            "near_clifford": {
                "non_clifford_count": list(NON_CLIFFORD_COUNTS),
                "theta": list(THETA_CHOICES),
            }
        },
        budget_tier="H",
    )


def compute_macro_mae(
    test_items: list[dict[str, Any]],
    predictions: Sequence[float] | Mapping[str, float],
) -> float:
    """Compute untouched-test macro MAE equally weighted over the 4 cells."""
    cells = sorted(list({(it["severity"], it["observable"]) for it in test_items}))
    cell_maes = []
    for cell in cells:
        cell_items = [
            it for it in test_items if (it["severity"], it["observable"]) == cell
        ]
        cell_diffs = []
        for it in cell_items:
            if isinstance(predictions, Mapping):
                pred = predictions[str(it["item_id"])]
            else:
                pred = predictions[test_items.index(it)]
            cell_diffs.append(abs(float(pred) - float(it["ideal_expectation"])))
        cell_maes.append(float(np.mean(cell_diffs)))
    return float(np.mean(cell_maes))


def analyze_descriptors_and_labels(
    train_items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Analyze descriptor variation and ideal label spread in training rows."""
    print("  Analyzing descriptor variation across training items...")
    feat_matrix = np.asarray(
        [build_features(dict(it)) for it in train_items], dtype=float
    )
    targets = np.asarray([float(it["ideal_expectation"]) for it in train_items], dtype=float)

    varying_descriptors = {}
    constant_descriptors = {}

    for idx, feat_name in enumerate(FEATURES):
        if feat_name == "noisy_expectation":
            continue  # Measurement, not circuit descriptor
        col = feat_matrix[:, idx]
        unique_vals = sorted(list(set(col)))
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        if len(unique_vals) > 1:
            varying_descriptors[feat_name] = {
                "min": min_val,
                "max": max_val,
                "unique_values": [float(x) for x in unique_vals[:20]],
                "n_unique": len(unique_vals),
                "std": float(np.std(col)),
            }
        else:
            constant_descriptors[feat_name] = min_val

    label_stats = {
        "count": len(targets),
        "min": float(np.min(targets)),
        "max": float(np.max(targets)),
        "range": float(np.max(targets) - np.min(targets)),
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
        "median": float(np.median(targets)),
        "q25": float(np.percentile(targets, 25)),
        "q75": float(np.percentile(targets, 75)),
        "iqr": float(np.percentile(targets, 75) - np.percentile(targets, 25)),
    }

    return {
        "varying_descriptors": varying_descriptors,
        "constant_descriptors": constant_descriptors,
        "label_stats": label_stats,
    }


def fit_and_score_single_arm(
    arm: str,
    seed: int,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fit one arm with warning catching and convergence checks."""
    t0 = time.time()
    runtime_warnings = []
    runtime_warning_count = 0
    all_warnings_count = 0

    pred_manifest = _prediction_manifest(manifest)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        if arm == "F":
            # Liao full
            mitigator = _fit_arm("liao", train, validation, manifest, master_seed=seed)
            pred_arr = _arm_predict("liao", mitigator, test, pred_manifest)
            selected = mitigator.selected_model_name_
        elif arm == "C":
            # Liao capacity-matched control (drop noisy_expectation)
            mitigator = _fit_arm(
                "liao-feat-only", train, validation, manifest, master_seed=seed
            )
            pred_arr = _arm_predict("liao-feat-only", mitigator, test, pred_manifest)
            selected = mitigator.selected_model_name_
        elif arm == "P":
            # Liao training-shuffle refit
            shuffled_train = shuffle_noisy_items(train, seed=seed)
            mitigator = _fit_arm(
                "liao", shuffled_train, validation, manifest, master_seed=seed
            )
            pred_arr = _arm_predict("liao", mitigator, test, pred_manifest)
            selected = mitigator.selected_model_name_
        elif arm == "A":
            # Affine feature-only control (feat-only)
            pred_items = _prediction_items(test)
            model_a = _fit_arm("feat-only", train, validation, manifest, master_seed=seed)
            pred_arr = _arm_predict("feat-only", model_a, pred_items, pred_manifest)
            selected = "ridge_cv"
        else:
            raise ValueError(f"Unknown arm: {arm}")

        for w in caught_warnings:
            all_warnings_count += 1
            if issubclass(w.category, RuntimeWarning) or "RuntimeWarning" in w.category.__name__:
                runtime_warning_count += 1
                if len(runtime_warnings) < 3:
                    runtime_warnings.append(
                        f"{w.category.__name__}: {w.message} ({os.path.basename(w.filename)}:{w.lineno})"
                    )

    pred_list = [float(x) for x in pred_arr]
    all_finite = bool(np.all(np.isfinite(pred_arr)))
    converged = bool(runtime_warning_count == 0 and all_finite)
    mae = compute_macro_mae(test, pred_list)
    elapsed = time.time() - t0

    preds_by_id = {
        str(it["item_id"]): pred_list[i] for i, it in enumerate(test)
    }

    return {
        "arm": arm,
        "seed": seed,
        "selected_candidate": selected,
        "converged": converged,
        "all_finite": all_finite,
        "runtime_warning_count": runtime_warning_count,
        "all_warnings_count": all_warnings_count,
        "runtime_warning_samples": runtime_warnings,
        "mae": mae,
        "elapsed_seconds": elapsed,
        "predictions": preds_by_id,
    }


def run_dataset_pipeline(
    master_seed: int,
    data_dir_root: Path,
    output_dir_root: Path,
    learner_seeds: tuple[int, ...] = LEARNER_SEEDS,
) -> dict[str, Any]:
    """Complete generation, validation, fitting, and scoring for one dataset seed."""
    t_start = time.time()
    ds_name = f"nc-s{master_seed}-n640"
    data_dir = data_dir_root / ds_name
    out_dir = output_dir_root / f"s{master_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"[{ds_name}] Starting pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Generation
    spec = build_nc_spec()
    if (data_dir / "items.jsonl").exists() and (data_dir / "manifest.json").exists():
        print(f"[{ds_name}] Dataset already exists at {data_dir}. Validating...", flush=True)
        items, manifest = validate_split_artifact(data_dir)
        dataset_hash = manifest["dataset_hash"]
        print(f"[{ds_name}] Validated existing artifact: hash={dataset_hash}", flush=True)
    else:
        print(f"[{ds_name}] Generating dataset into {data_dir}...", flush=True)
        t_gen = time.time()
        manifest = generate_split(spec, data_dir, master_seed=master_seed)
        items, _ = validate_split_artifact(data_dir)
        dataset_hash = manifest["dataset_hash"]
        print(
            f"[{ds_name}] Generated & validated in {time.time() - t_gen:.1f}s: hash={dataset_hash}",
            flush=True,
        )

    # 2. Partition items
    train_items = [it for it in items if it["split"] == "train"]
    val_items = [it for it in items if it["split"] == "validation"]
    test_items = [it for it in items if it["split"] == "test"]

    print(
        f"[{ds_name}] Items count: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}",
        flush=True,
    )

    # 3. Analyze descriptor variation and ideal labels
    desc_analysis = analyze_descriptors_and_labels(train_items)

    # 4. Arm A (affine control, feat-only)
    print(f"[{ds_name}] Fitting Arm A (feat-only)...", flush=True)
    a_res = fit_and_score_single_arm("A", master_seed, train_items, val_items, test_items, manifest)
    a_mae = a_res["mae"]
    print(f"[{ds_name}] Arm A Macro MAE: {a_mae:.6f} (elapsed: {a_res['elapsed_seconds']:.2f}s)", flush=True)

    # 5. Arm R (raw noisy expectation)
    raw_preds = {str(it["item_id"]): float(it["noisy_expectation"]) for it in test_items}
    raw_pred_list = [raw_preds[str(it["item_id"])] for it in test_items]
    r_mae = compute_macro_mae(test_items, raw_pred_list)
    print(f"[{ds_name}] Arm R Macro MAE: {r_mae:.6f}", flush=True)

    # 6. Arms F, C, P across learner seeds k = 1..20 and anchor k = master_seed
    all_seeds_to_run = list(learner_seeds)
    if master_seed not in all_seeds_to_run:
        all_seeds_to_run.append(master_seed)

    per_seed_results = []
    test_predictions_by_seed = {}

    print(f"[{ds_name}] Fitting F, C, P across {len(all_seeds_to_run)} seeds...", flush=True)
    t_fits_start = time.time()

    for idx, k in enumerate(all_seeds_to_run):
        t_seed_start = time.time()
        f_res = fit_and_score_single_arm("F", k, train_items, val_items, test_items, manifest)
        c_res = fit_and_score_single_arm("C", k, train_items, val_items, test_items, manifest)
        p_res = fit_and_score_single_arm("P", k, train_items, val_items, test_items, manifest)

        f_mae = f_res["mae"]
        c_mae = c_res["mae"]
        p_mae = p_res["mae"]

        d_k = c_mae - f_mae
        d_over_c = d_k / c_mae if c_mae != 0 else float("nan")
        s_k = (a_mae - c_mae) / (a_mae - f_mae) if (a_mae - f_mae) != 0 else float("nan")
        pc_k = p_mae - c_mae
        pf_k = p_mae - f_mae

        converged_all = bool(f_res["converged"] and c_res["converged"] and p_res["converged"])
        converged_d = bool(f_res["converged"] and c_res["converged"])

        seed_entry = {
            "seed": k,
            "is_anchor": bool(k == master_seed and k not in learner_seeds),
            "mae": {"A": a_mae, "R": r_mae, "F": f_mae, "C": c_mae, "P": p_mae},
            "D": d_k,
            "D_over_C": d_over_c,
            "S": s_k,
            "P_minus_C": pc_k,
            "P_minus_F": pf_k,
            "selected_candidates": {
                "F": f_res["selected_candidate"],
                "C": c_res["selected_candidate"],
                "P": p_res["selected_candidate"],
            },
            "converged": {
                "F": f_res["converged"],
                "C": c_res["converged"],
                "P": p_res["converged"],
                "D": converged_d,
                "all": converged_all,
            },
            "runtime_warning_counts": {
                "F": f_res["runtime_warning_count"],
                "C": c_res["runtime_warning_count"],
                "P": p_res["runtime_warning_count"],
            },
            "all_finite": {
                "F": f_res["all_finite"],
                "C": c_res["all_finite"],
                "P": p_res["all_finite"],
            },
            "elapsed_seconds": time.time() - t_seed_start,
        }

        if k in learner_seeds:
            per_seed_results.append(seed_entry)

        test_predictions_by_seed[str(k)] = {
            "F": f_res["predictions"],
            "C": c_res["predictions"],
            "P": p_res["predictions"],
        }

        print(
            f"  [{ds_name}] k={k:02d} | F={f_mae:.4f} C={c_mae:.4f} P={p_mae:.4f} | "
            f"D={d_k:+.4f} (D/C={d_over_c:.2%}) S={s_k:+.3f} PF={pf_k:+.4f} | "
            f"cand=({f_res['selected_candidate']},{c_res['selected_candidate']},{p_res['selected_candidate']}) "
            f"conv={converged_all} ({time.time() - t_seed_start:.1f}s)",
            flush=True,
        )

    print(
        f"[{ds_name}] Finished all fits in {time.time() - t_fits_start:.1f}s",
        flush=True,
    )

    # Anchor fit result
    anchor_entry = (
        next((it for it in per_seed_results if it["seed"] == master_seed), None)
        if master_seed in learner_seeds
        else next((it for it in [seed_entry] if it["seed"] == master_seed), None)
    )

    # 7. Aggregate statistics over the 20 learner seeds
    def stat_dict(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    stats_20 = {
        "D": stat_dict([it["D"] for it in per_seed_results]),
        "D_over_C": stat_dict([it["D_over_C"] for it in per_seed_results]),
        "S": stat_dict([it["S"] for it in per_seed_results]),
        "P_minus_C": stat_dict([it["P_minus_C"] for it in per_seed_results]),
        "P_minus_F": stat_dict([it["P_minus_F"] for it in per_seed_results]),
        "F_mae": stat_dict([it["mae"]["F"] for it in per_seed_results]),
        "C_mae": stat_dict([it["mae"]["C"] for it in per_seed_results]),
        "P_mae": stat_dict([it["mae"]["P"] for it in per_seed_results]),
    }

    candidate_counts = {
        arm: dict(Counter(it["selected_candidates"][arm] for it in per_seed_results))
        for arm in ("F", "C", "P")
    }

    convergence_summary = {
        "n_seeds": len(per_seed_results),
        "converged_all": sum(it["converged"]["all"] for it in per_seed_results),
        "converged_D": sum(it["converged"]["D"] for it in per_seed_results),
        "converged_F": sum(it["converged"]["F"] for it in per_seed_results),
        "converged_C": sum(it["converged"]["C"] for it in per_seed_results),
        "converged_P": sum(it["converged"]["P"] for it in per_seed_results),
    }

    # Write compressed test predictions
    preds_file = out_dir / "test_predictions.json.gz"
    full_pred_obj = {
        "dataset_seed": master_seed,
        "dataset_hash": dataset_hash,
        "test_item_ids": [str(it["item_id"]) for it in test_items],
        "ideal_expectation": {str(it["item_id"]): float(it["ideal_expectation"]) for it in test_items},
        "A": a_res["predictions"],
        "R": raw_preds,
        "seeds": test_predictions_by_seed,
    }
    with gzip.open(preds_file, "wt", encoding="utf-8") as f:
        json.dump(full_pred_obj, f, indent=2)

    total_elapsed = time.time() - t_start

    # Summary dictionary
    summary = {
        "dataset_seed": master_seed,
        "dataset_name": ds_name,
        "dataset_dir": str(data_dir),
        "dataset_hash": dataset_hash,
        "validation_status": "passed",
        "total_elapsed_seconds": total_elapsed,
        "circuit_counts": {"train": 640, "validation": 320, "test": 160},
        "item_counts": {"train": len(train_items), "validation": len(val_items), "test": len(test_items)},
        "arm_errors": {
            "A_macro_mae": a_mae,
            "R_macro_mae": r_mae,
        },
        "anchor_fit": anchor_entry,
        "stats_20_seeds": stats_20,
        "candidate_counts": candidate_counts,
        "convergence_summary": convergence_summary,
        "descriptor_analysis": desc_analysis,
        "per_seed_results": per_seed_results,
    }

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{ds_name}] Saved summary to {summary_file}", flush=True)
    print(f"[{ds_name}] Saved test predictions to {preds_file} (size: {preds_file.stat().st_size / 1024:.1f} KB)", flush=True)
    print(f"[{ds_name}] Pipeline completed in {total_elapsed:.1f}s\n", flush=True)

    return summary


def run_single_dataset_worker(args_tuple: tuple) -> dict[str, Any]:
    """Module-level worker function for multiprocessing."""
    seed, data_root, out_root, learner_seeds = args_tuple
    return run_dataset_pipeline(
        master_seed=seed,
        data_dir_root=Path(data_root),
        output_dir_root=Path(out_root),
        learner_seeds=tuple(learner_seeds),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run near-Clifford positive control across dataset seeds."
    )
    parser.add_argument(
        "--dataset-seeds",
        nargs="+",
        type=int,
        default=list(DATASET_SEEDS),
        help="Dataset seeds to run (default: 101 211 307)",
    )
    parser.add_argument(
        "--learner-seeds",
        nargs="+",
        type=int,
        default=list(LEARNER_SEEDS),
        help="Learner seeds to fit (default: 1..20)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where datasets are generated",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/nc",
        help="Directory where results and predictions are written",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=3,
        help="Number of parallel worker processes (max 6)",
    )
    args = parser.parse_args()

    t_global_start = time.time()
    data_root = Path(args.data_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=======================================================")
    print(f"Near-Clifford Positive Control Master Runner")
    print(f"Dataset seeds: {args.dataset_seeds}")
    print(f"Learner seeds: {len(args.learner_seeds)} seeds ({min(args.learner_seeds)}..{max(args.learner_seeds)})")
    print(f"Data root: {data_root}")
    print(f"Output root: {out_root}")
    print(f"Worker processes: {args.processes}")
    print(f"=======================================================\n")

    tasks = [
        (seed, str(data_root), str(out_root), tuple(args.learner_seeds))
        for seed in args.dataset_seeds
    ]

    all_summaries = {}
    if args.processes > 1 and len(tasks) > 1:
        n_workers = min(args.processes, len(tasks), 6)
        print(f"Launching {len(tasks)} dataset jobs across {n_workers} processes...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(run_single_dataset_worker, tasks))
            for res in results:
                all_summaries[str(res["dataset_seed"])] = res
    else:
        for task in tasks:
            res = run_single_dataset_worker(task)
            all_summaries[str(res["dataset_seed"])] = res

    # Write combined summary
    combined_file = out_root / "all_summary.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    total_time = time.time() - t_global_start
    print(f"\n=======================================================")
    print(f"ALL DATASETS COMPLETED in {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"Master summary written to {combined_file}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
