#!/usr/bin/env python3
"""Two-stage bootstrap for the near-Clifford positive control.

Implements the frozen two-stage percentile bootstrap:
- 10,000 draws
- Random seed: 20261001
- Stage 1: Resamples the 20 learner seeds with replacement
- Stage 2: Resamples the 160 test physical circuits with replacement (shared across arms and seeds,
  keeping all 4 rows of each circuit grouped)
Computes the 2.5th to 97.5th percentile intervals for mean D, mean S, mean (P - F), and mean (P - C).
Applies the frozen decision rule and runs sensitivity analysis excluding flagged fits.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261001
PERCENTILES = (2.5, 97.5)


def build_circuit_matrices(
    test_items: list[dict[str, Any]],
    predictions_by_seed: dict[str, dict[str, dict[str, float]]],
    a_predictions: dict[str, float],
    raw_predictions: dict[str, float],
    learner_seeds: list[int],
) -> dict[str, Any]:
    """Compute per-seed, per-circuit macro errors across the 4 cells.

    Returns:
      circuit_ids: sorted list of 160 physical circuits
      E_A: shape (160,)
      E_R: shape (160,)
      errors: {"F": (n_seeds, 160), "C": (n_seeds, 160), "P": (n_seeds, 160)}
    """
    circuit_ids = sorted(list({str(it["circuit_id"]) for it in test_items}))
    cells = sorted(list({(it["severity"], it["observable"]) for it in test_items}))

    if len(circuit_ids) != 160:
        raise ValueError(f"Expected 160 test physical circuits, got {len(circuit_ids)}")
    if len(cells) != 4:
        raise ValueError(f"Expected 4 cells, got {len(cells)}")

    # Index items by (circuit_id, cell)
    item_map = {}
    for it in test_items:
        key = (str(it["circuit_id"]), (it["severity"], it["observable"]))
        item_map[key] = (str(it["item_id"]), float(it["ideal_expectation"]))

    n_seeds = len(learner_seeds)
    n_circuits = len(circuit_ids)

    # Errors for A and R
    E_A = np.zeros(n_circuits, dtype=float)
    E_R = np.zeros(n_circuits, dtype=float)

    for c_idx, cid in enumerate(circuit_ids):
        diffs_a = []
        diffs_r = []
        for cell in cells:
            item_id, target = item_map[(cid, cell)]
            diffs_a.append(abs(a_predictions[item_id] - target))
            diffs_r.append(abs(raw_predictions[item_id] - target))
        E_A[c_idx] = float(np.mean(diffs_a))
        E_R[c_idx] = float(np.mean(diffs_r))

    # Errors for F, C, P
    errors = {
        arm: np.zeros((n_seeds, n_circuits), dtype=float) for arm in ("F", "C", "P")
    }

    for s_idx, k in enumerate(learner_seeds):
        k_str = str(k)
        for arm in ("F", "C", "P"):
            preds = predictions_by_seed[k_str][arm]
            for c_idx, cid in enumerate(circuit_ids):
                cell_diffs = []
                for cell in cells:
                    item_id, target = item_map[(cid, cell)]
                    pred = preds[item_id]
                    cell_diffs.append(abs(pred - target))
                errors[arm][s_idx, c_idx] = float(np.mean(cell_diffs))

    return {
        "circuit_ids": circuit_ids,
        "E_A": E_A,
        "E_R": E_R,
        "errors": errors,
    }


def run_two_stage_bootstrap(
    E_A: np.ndarray,
    errors: dict[str, np.ndarray],
    valid_seed_indices: list[int] | None = None,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Execute two-stage percentile bootstrap.

    Stage 1: Resample learner seeds with replacement.
    Stage 2: Resample 160 circuits with replacement (shared across arms and seeds).
    """
    rng = np.random.default_rng(seed)

    if valid_seed_indices is not None:
        seed_pool = np.array(valid_seed_indices, dtype=int)
    else:
        seed_pool = np.arange(errors["F"].shape[0], dtype=int)

    n_seeds_to_sample = len(seed_pool)
    n_circuits = E_A.shape[0]

    d_draws = np.empty(n_draws, dtype=float)
    s_draws = np.empty(n_draws, dtype=float)
    s_ratio_draws = np.empty(n_draws, dtype=float)
    pf_draws = np.empty(n_draws, dtype=float)
    pc_draws = np.empty(n_draws, dtype=float)

    ef_mat = errors["F"]
    ec_mat = errors["C"]
    ep_mat = errors["P"]

    for b in range(n_draws):
        sampled_seeds = seed_pool[rng.integers(0, len(seed_pool), size=n_seeds_to_sample)]
        sampled_circuits = rng.integers(0, n_circuits, size=n_circuits)

        # Macro MAE of A on sampled circuits
        mae_a = float(np.mean(E_A[sampled_circuits]))

        # Per-seed macro MAEs on sampled circuits: shape (n_seeds_to_sample,)
        mae_f = np.mean(ef_mat[sampled_seeds][:, sampled_circuits], axis=1)
        mae_c = np.mean(ec_mat[sampled_seeds][:, sampled_circuits], axis=1)
        mae_p = np.mean(ep_mat[sampled_seeds][:, sampled_circuits], axis=1)

        # Endpoints per seed
        d_per_seed = mae_c - mae_f
        # S_k = (A - C_k) / (A - F_k)
        denom = mae_a - mae_f
        # Protect against rare division by zero if denom == 0
        denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        s_per_seed = (mae_a - mae_c) / denom_safe

        pf_per_seed = mae_p - mae_f
        pc_per_seed = mae_p - mae_c

        # Draw means over sampled seeds
        d_draws[b] = float(np.mean(d_per_seed))
        s_draws[b] = float(np.mean(s_per_seed))
        pf_draws[b] = float(np.mean(pf_per_seed))
        pc_draws[b] = float(np.mean(pc_per_seed))

        # Also ratio-of-means S
        mean_c = float(np.mean(mae_c))
        mean_f = float(np.mean(mae_f))
        denom_ratio = mae_a - mean_f if abs(mae_a - mean_f) > 1e-12 else 1e-12
        s_ratio_draws[b] = (mae_a - mean_c) / denom_ratio

    def get_interval(draws: np.ndarray) -> tuple[float, float]:
        return (
            float(np.percentile(draws, PERCENTILES[0])),
            float(np.percentile(draws, PERCENTILES[1])),
        )

    return {
        "D": get_interval(d_draws),
        "S": get_interval(s_draws),
        "S_ratio": get_interval(s_ratio_draws),
        "P_minus_F": get_interval(pf_draws),
        "P_minus_C": get_interval(pc_draws),
        "mean_draws": {
            "D": float(np.mean(d_draws)),
            "S": float(np.mean(s_draws)),
            "S_ratio": float(np.mean(s_ratio_draws)),
            "P_minus_F": float(np.mean(pf_draws)),
            "P_minus_C": float(np.mean(pc_draws)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run two-stage percentile bootstrap for near-Clifford positive control."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory containing datasets nc-s{seed}-n640",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/nc",
        help="Root directory containing results/nc/s{seed}/",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/nc/bootstrap_results.json",
        help="Output file for bootstrap results JSON",
    )
    parser.add_argument(
        "--dataset-seeds",
        nargs="+",
        type=int,
        default=[101, 211, 307],
        help="Dataset seeds to analyze",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    results_root = Path(args.results_dir).resolve()
    out_file = Path(args.output_file).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("Near-Clifford Positive Control Two-Stage Bootstrap")
    print(f"Dataset seeds: {args.dataset_seeds}")
    print(f"Bootstrap draws: {BOOTSTRAP_DRAWS}, Seed: {BOOTSTRAP_SEED}")
    print(f"Results dir: {results_root}")
    print("=======================================================\n")

    results_by_seed = {}
    pass_decision_all = True
    pass_decision_sensitivity = True

    for master_seed in args.dataset_seeds:
        seed_dir = results_root / f"s{master_seed}"
        ds_dir = data_root / f"nc-s{master_seed}-n640"

        summary_file = seed_dir / "summary.json"
        preds_file = seed_dir / "test_predictions.json.gz"
        items_file = ds_dir / "items.jsonl"

        if not summary_file.exists():
            raise FileNotFoundError(f"Missing summary.json at {summary_file}")
        if not preds_file.exists():
            raise FileNotFoundError(f"Missing test_predictions.json.gz at {preds_file}")
        if not items_file.exists():
            raise FileNotFoundError(f"Missing items.jsonl at {items_file}")

        with open(summary_file) as f:
            summary = json.load(f)

        with gzip.open(preds_file, "rt", encoding="utf-8") as f:
            preds_data = json.load(f)

        test_items = [
            json.loads(line)
            for line in items_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("split") == "test"
        ]

        learner_seeds = [it["seed"] for it in summary["per_seed_results"]]
        per_seed_results = summary["per_seed_results"]

        print(f"Processing dataset seed {master_seed} ({len(test_items)} test items, {len(learner_seeds)} seeds)...")

        mats = build_circuit_matrices(
            test_items,
            preds_data["seeds"],
            preds_data["A"],
            preds_data["R"],
            learner_seeds,
        )

        # 1. All fits bootstrap
        all_intervals = run_two_stage_bootstrap(
            mats["E_A"], mats["errors"]
        )

        # 2. Sensitivity analysis: exclude flagged fits
        valid_indices = [
            idx
            for idx, rec in enumerate(per_seed_results)
            if rec["converged"]["all"]
        ]
        valid_d_indices = [
            idx
            for idx, rec in enumerate(per_seed_results)
            if rec["converged"]["D"]
        ]

        if len(valid_indices) == 0:
            print(f"  All {len(learner_seeds)} seeds flagged for numerical RuntimeWarnings (during candidate MLP training).")
            print(f"  Sensitivity analysis: 0 unflagged fits remain; sensitivity evaluation not evaluable without unflagged fits.")
            sensitivity_intervals = None
            sens_supports = None
        elif len(valid_indices) < len(learner_seeds):
            print(f"  Flagged fits detected: {len(learner_seeds) - len(valid_indices)} flagged seeds, {len(valid_indices)} unflagged.")
            sensitivity_intervals = run_two_stage_bootstrap(
                mats["E_A"], mats["errors"], valid_seed_indices=valid_indices
            )
            sens_d_lower, sens_d_upper = sensitivity_intervals["D"]
            sens_supports = bool(sens_d_lower > 0.0)
            if not sens_supports:
                pass_decision_sensitivity = False
        else:
            print(f"  All {len(learner_seeds)} fits converged cleanly (no RuntimeWarnings, all finite).")
            sensitivity_intervals = all_intervals
            sens_supports = True

        # Check positive gap support: interval wholly above zero (lower bound > 0)
        d_lower, d_upper = all_intervals["D"]
        supports_positive_gap = bool(d_lower > 0.0)

        if not supports_positive_gap:
            pass_decision_all = False

        results_by_seed[str(master_seed)] = {
            "dataset_seed": master_seed,
            "dataset_hash": summary["dataset_hash"],
            "n_seeds": len(learner_seeds),
            "n_converged_all": len(valid_indices),
            "n_converged_D": len(valid_d_indices),
            "arm_errors": summary["arm_errors"],
            "all_fits": all_intervals,
            "sensitivity_fits": sensitivity_intervals,
            "supports_positive_gap": supports_positive_gap,
            "sensitivity_supports": sens_supports,
        }

        print(f"  Seed {master_seed} Results:")
        print(f"    Arm A MAE: {summary['arm_errors']['A_macro_mae']:.4f}, Arm R MAE: {summary['arm_errors']['R_macro_mae']:.4f}")
        print(f"    Mean D 95% CI: [{d_lower:+.4f}, {d_upper:+.4f}] -> Supports: {supports_positive_gap}")
        print(f"    Mean S 95% CI: [{all_intervals['S'][0]:+.4f}, {all_intervals['S'][1]:+.4f}]")
        print(f"    Mean P-F 95% CI: [{all_intervals['P_minus_F'][0]:+.4f}, {all_intervals['P_minus_F'][1]:+.4f}]")
        print(f"    Mean P-C 95% CI: [{all_intervals['P_minus_C'][0]:+.4f}, {all_intervals['P_minus_C'][1]:+.4f}]")
        print()

    # Final decision rule outcome
    overall_outcome = "PASS" if pass_decision_all else "FAILURE"
    outcome_sentence = (
        f"The positive control passes because the two-stage 95% bootstrap interval for mean D "
        f"lies wholly above zero for all three dataset seeds."
        if pass_decision_all
        else f"The positive control fails because at least one dataset seed's two-stage 95% bootstrap interval "
        f"for mean D reaches or crosses zero."
    )

    final_payload = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "results_by_seed": results_by_seed,
        "decision_rule": {
            "verdict": overall_outcome,
            "sentence": outcome_sentence,
            "pass_all_seeds": pass_decision_all,
            "pass_sensitivity": pass_decision_sensitivity,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("=======================================================")
    print(f"FROZEN DECISION RULE VERDICT: {overall_outcome}")
    print(f"Outcome statement: {outcome_sentence}")
    print(f"Results written to: {out_file}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
