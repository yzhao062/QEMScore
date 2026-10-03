#!/usr/bin/env python3
"""Exact sign-flip permutation tests and joint resampling for Q-LEAR ladder contrasts.

The paper reports Q-LEAR ladder contrasts (capacity A-C, base statistics C-O,
depth cut O-F, total C-F) as ten-fit means with percentile bootstrap intervals
over the six applications (backends fixed). To address referee concerns that a
percentile bootstrap over six application units under-covers, this script:
1. Computes the six per-application contrasts (each averaged equally over the
   eight backends, then over the ten fitted seeds, as the paper aggregates).
2. Computes the exact two-sided sign-flip permutation p-value under the null
   of zero mean over all 2^6 = 64 sign assignments (minimum possible two-sided
   p-value is 2/64 = 0.03125).
3. Computes a joint resampling interval (resampling the ten learner fits and
   the six applications jointly, 10,000 draws, numpy default_rng(20261002)).
4. Verifies that the ten-fit macro means match the printed manuscript values
   (A-C +0.000055/+0.000091, C-O 0.1969/0.4407, O-F 0.0222/0.0099,
   C-F 0.2191/0.4506) to the printed precision, failing loudly otherwise.

Usage:
    python scripts/qlear/signflip_qlear.py [OPTIONS]
    python scripts/qlear/signflip_qlear.py --input outputs/qlear/field-report.json --out outputs/qlear/
"""

import argparse
import itertools
import json
import os
import sys
import numpy as np

# Printed manuscript values and tolerance based on printed precision:
# Hardware: A-C +0.000055 (tol 5e-7), C-O 0.1969 (tol 5e-5), O-F 0.0222 (tol 5e-5), C-F 0.2191 (tol 5e-5)
# Simulator: A-C +0.000091 (tol 5e-7), C-O 0.4407 (tol 5e-5), O-F 0.0099 (tol 5e-5), C-F 0.4506 (tol 5e-5)
EXPECTED_PAPER_MEANS = {
    "hardware": {
        "A-C": (0.000055, 5e-7),
        "C-O": (0.1969, 5e-5),
        "O-F": (0.0222, 5e-5),
        "C-F": (0.2191, 5e-5),
    },
    "simulator": {
        "A-C": (0.000091, 5e-7),
        "C-O": (0.4407, 5e-5),
        "O-F": (0.0099, 5e-5),
        "C-F": (0.4506, 5e-5),
    },
}

ALL_SIGNS = list(itertools.product([-1.0, 1.0], repeat=6))
assert len(ALL_SIGNS) == 64


def exact_sign_flip_test(values):
    """Compute exact two-sided sign-flip permutation p-value over all 64 assignments.

    Under H0: mu = 0, signs of the 6 independent application errors are interchangeable.
    The two-sided p-value is the fraction of the 2^6 = 64 sign assignments s in {-1, +1}^6
    satisfying |mean(s * x)| >= |mean(x)|.
    """
    vals = np.asarray(values, dtype=float)
    t_obs = abs(float(np.mean(vals)))
    count = 0
    for s in ALL_SIGNS:
        t_perm = abs(float(np.mean(np.array(s) * vals)))
        if t_perm >= t_obs - 1e-12:
            count += 1
    p_val = count / 64.0
    return p_val, count


def joint_resampling_interval(fit_by_app_matrix, draws=10000, seed=20261002, confidence=0.95):
    """Compute joint percentile bootstrap interval resampling fits and applications.

    Each draw calls integers(0, 10, 10) over fits, then integers(0, 6, 6) over applications.
    Percentiles are taken at (1 - confidence)/2 and (1 + confidence)/2.
    """
    rng = np.random.default_rng(seed)
    n_fits, n_apps = fit_by_app_matrix.shape
    bootstrap_draws = np.empty(draws, dtype=float)
    for b in range(draws):
        s_idx = rng.integers(0, n_fits, n_fits)
        a_idx = rng.integers(0, n_apps, n_apps)
        bootstrap_draws[b] = np.mean(fit_by_app_matrix[s_idx[:, None], a_idx])
    lo_q = (1.0 - confidence) / 2.0 * 100.0
    hi_q = (1.0 + confidence) / 2.0 * 100.0
    lower = float(np.percentile(bootstrap_draws, lo_q))
    upper = float(np.percentile(bootstrap_draws, hi_q))
    return lower, upper


def evaluate_panel(panel_data, panel_name, draws=10000, seed=20261002):
    """Extract scores, compute contrasts, test-checks, sign-flip tests, and joint bootstrap."""
    apps = panel_data["applications"]
    backends = panel_data["backends"]
    n_apps = len(apps)
    assert n_apps == 6, f"Expected 6 applications, got {n_apps}"
    assert len(backends) == 8, f"Expected 8 backends, got {len(backends)}"

    # Precompute per-fit, per-app scores averaged over the 8 backends: shape (10 fits, 6 apps)
    # Arm A is affine (deterministic), fitted once and reused across all 10 network seeds (stored under seed "0").
    # Arms C, O, F have distinct fits for seeds 0..9.
    M = {arm: np.zeros((10, n_apps), dtype=float) for arm in ["A", "C", "O", "F"]}
    for a_idx, app in enumerate(apps):
        for b in backends:
            fname = f"{app}_{b}"
            file_entry = panel_data["per_file"][fname]
            a_val = file_entry["A"]["0"]
            for s in range(10):
                s_str = str(s)
                M["A"][s, a_idx] += a_val / len(backends)
                M["C"][s, a_idx] += file_entry["C"][s_str] / len(backends)
                M["O"][s, a_idx] += file_entry["O"][s_str] / len(backends)
                M["F"][s, a_idx] += file_entry["F"][s_str] / len(backends)

    contrasts_matrix = {
        "A-C": M["A"] - M["C"],
        "C-O": M["C"] - M["O"],
        "O-F": M["O"] - M["F"],
        "C-F": M["C"] - M["F"],
    }

    panel_results = {}
    for cname in ["A-C", "C-O", "O-F", "C-F"]:
        mat = contrasts_matrix[cname]
        # per-application contrast: mean over the 10 fits (and 8 backends)
        per_app = np.mean(mat, axis=0)
        ten_fit_mean = float(np.mean(per_app))
        n_pos = int(np.sum(per_app > 0))
        p_val, p_count = exact_sign_flip_test(per_app)
        lo_ci, hi_ci = joint_resampling_interval(mat, draws=draws, seed=seed)

        # Check against printed paper values
        exp_val, tol = EXPECTED_PAPER_MEANS[panel_name][cname]
        diff = abs(ten_fit_mean - exp_val)
        if diff > tol:
            raise AssertionError(
                f"Mismatch for {panel_name} contrast {cname}: "
                f"computed ten-fit mean {ten_fit_mean:.8f} disagrees with "
                f"printed value {exp_val} (diff={diff:.2e} > tol={tol:.2e})"
            )

        panel_results[cname] = {
            "per_application": {app: float(per_app[i]) for i, app in enumerate(apps)},
            "ten_fit_mean": ten_fit_mean,
            "count_positive": n_pos,
            "sign_flip_p": p_val,
            "sign_flip_count": p_count,
            "joint_percentile_interval": {
                "lower": lo_ci,
                "upper": hi_ci,
                "confidence": 0.95,
            },
            "paper_validation": {
                "expected": exp_val,
                "tolerance": tol,
                "diff": diff,
                "status": "PASS",
            },
        }

    return panel_results


def format_table(results, panel_name):
    """Format an ASCII/Markdown summary table for a panel."""
    lines = []
    lines.append(f"=== {panel_name.upper()} PANEL ===")
    lines.append(
        "| Contrast | Ground state | Pricing call | Pricing put | QAOA | Routing | TSP | Ten-Fit Mean | Sign-Flip p (count) | Joint 95% CI | Count Positive |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    apps_order = ["groundstate", "pricingcall", "pricingput", "qaoa", "routing", "tsp"]
    for cname in ["A-C", "C-O", "O-F", "C-F"]:
        d = results[cname]
        app_vals = [f"{d['per_application'][a]:+.6f}" for a in apps_order]
        app_cols = " | ".join(app_vals)
        mean_str = f"{d['ten_fit_mean']:+.6f}"
        p_str = f"{d['sign_flip_p']:.5f} ({d['sign_flip_count']}/64)"
        ci_str = f"[{d['joint_percentile_interval']['lower']:+.6f}, {d['joint_percentile_interval']['upper']:+.6f}]"
        pos_str = f"{d['count_positive']} of 6"
        lines.append(f"| **{cname}** | {app_cols} | {mean_str} | {p_str} | {ci_str} | {pos_str} |")
    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(repo_root, "outputs", "qlear", "field-report.json"),
        help="Path to field-report.json (default: outputs/qlear/field-report.json)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=os.path.join(repo_root, "outputs", "qlear", "qlear-signflip-report.json"),
        help="Path to write qlear-signflip-report.json (or output directory)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=10000,
        help="Number of joint bootstrap draws (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261002,
        help="Root seed for numpy default_rng (default: 20261002)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    out_path = os.path.abspath(args.out)
    if os.path.isdir(out_path) or (not out_path.endswith(".json")):
        out_path = os.path.join(out_path, "qlear-signflip-report.json")

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        field_report = json.load(f)

    out_report = {
        "schema": "qlear-signflip-report-v1",
        "description": "Exact two-sided sign-flip permutation tests and joint resampling intervals for Q-LEAR ladder contrasts",
        "input_report": os.path.relpath(input_path, repo_root) if input_path.startswith(repo_root) else input_path,
        "n_draws": args.draws,
        "seed": args.seed,
        "panels": {},
    }

    for panel_name in ["hardware", "simulator"]:
        if panel_name not in field_report["panels"]:
            raise KeyError(f"Panel '{panel_name}' not found in field-report.json")
        panel_results = evaluate_panel(
            field_report["panels"][panel_name],
            panel_name,
            draws=args.draws,
            seed=args.seed,
        )
        out_report["panels"][panel_name] = panel_results

        print(f"\n{format_table(panel_results, panel_name)}")
        print(f"All 4 {panel_name} ten-fit mean checks PASSED against manuscript values.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)
    print(f"\nWrote full sign-flip report to: {out_path}")


if __name__ == "__main__":
    main()
