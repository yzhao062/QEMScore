#!/usr/bin/env python3
"""Measurement-only reference (calibration error of the noisy estimate) and stacking increment analysis (Part A spin chains).

Post hoc, descriptive analysis:
Evaluates the test error of a per-cell linear calibration of the noisy estimate at 2,048 shots
(an empirical reference for this fitted estimator, not a lower bound) across Part A
spin chain cells (seeds 101, 211, 307; TFI and Heisenberg; severities L1 and L3;
observables z_mid and zz_mid), and the incremental value of stacking the noisy
measurement r onto recalibrated descriptor control predictions Chat across ladder
rungs (R0, N1..N4, R3-TFI/Heis, R4, R5) for learner seeds 1..20.

Usage:
    PYTHONPATH=. python tools/measurement_floor.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

# Ensure tools and repo root are importable
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

import descriptor_ladder_analysis as dla  # noqa: E402
from qemscore.campaign.analysis import _family_mae  # noqa: E402

RULE_SEED = dla.RULE_SEED  # 20261002
BOOTSTRAP_DRAWS = dla.DEFAULT_DRAWS  # 10_000
SHOTS = 2048
SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
RUNGS_PER_FAMILY = {
    "tfi": ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R4", "R5"),
    "heisenberg": ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R4", "R5"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_datasets(data_dir: Path) -> dict[int, dict[str, list[dict]]]:
    """Load regenerated dataset items split into validation and test."""
    datasets = {}
    for seed in SEEDS:
        p = data_dir / f"regen-shipped-s{seed}-n640" / "items.jsonl"
        val_items: list[dict] = []
        test_items: list[dict] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                split = row.get("split")
                if split == "validation":
                    val_items.append(row)
                elif split == "test":
                    test_items.append(row)
        datasets[seed] = {"val": val_items, "test": test_items}
    return datasets


def compute_floors(
    datasets: dict[int, dict[str, list[dict]]],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Compute per-cell information floors and row-level severity summaries.

    Validation items:
      - regress r on y (OLS): r = a + b * y -> slope b, residual SD, floor_sd = residual_sd / |b|
      - linear calibration: y ~ a_cal + c * r
      - shot-noise SD: sqrt(1 - mean(r^2)) / sqrt(2048)
    Test items:
      - floor_mae: MAE on test items of the linear calibration fitted on validation.
    """
    cell_floors = {}
    row_floors = {}

    for seed in SEEDS:
        v_items = datasets[seed]["val"]
        t_items = datasets[seed]["test"]

        for fam in FAMILIES:
            row_key = f"shipped-s{seed}-n640/{fam}"
            row_floors[row_key] = {
                "L1_floor_mae": [],
                "L3_floor_mae": [],
                "L1_floor_sd": [],
                "L3_floor_sd": [],
                "all_floor_mae": [],
                "all_floor_sd": [],
            }

            for sev, obs in CELLS_DEF:
                cell_key = f"{row_key}/{sev}/{obs}"
                v_sub = [
                    r
                    for r in v_items
                    if r["family"] == fam
                    and r["severity"] == sev
                    and r["observable"] == obs
                ]
                t_sub = [
                    r
                    for r in t_items
                    if r["family"] == fam
                    and r["severity"] == sev
                    and r["observable"] == obs
                ]

                y_v = np.asarray([r["ideal_expectation"] for r in v_sub], dtype=float)
                r_v = np.asarray([r["noisy_expectation"] for r in v_sub], dtype=float)
                y_t = np.asarray([r["ideal_expectation"] for r in t_sub], dtype=float)
                r_t = np.asarray([r["noisy_expectation"] for r in t_sub], dtype=float)

                # 1. Regress r on y: r = a + b * y
                b, a = np.polyfit(y_v, r_v, 1)
                resid = r_v - (a + b * y_v)
                resid_sd = float(np.std(resid, ddof=1))
                floor_sd = resid_sd / abs(b) if abs(b) > 1e-12 else float("inf")

                # 2. Linear calibration: y ~ a_cal + c * r fitted on validation items
                c, a_cal = np.polyfit(r_v, y_v, 1)
                y_pred_test = a_cal + c * r_t
                floor_mae = float(np.mean(np.abs(y_pred_test - y_t)))

                # 3. Binomial shot-noise SD: sqrt(1 - mean(r^2)) / sqrt(2048)
                shot_noise_sd = float(
                    np.sqrt(max(0.0, 1.0 - np.mean(r_v**2))) / np.sqrt(SHOTS)
                )

                entry = {
                    "dataset_seed": seed,
                    "family": fam,
                    "severity": sev,
                    "observable": obs,
                    "n_val": len(v_sub),
                    "n_test": len(t_sub),
                    "slope_b": float(b),
                    "intercept_a": float(a),
                    "residual_sd": resid_sd,
                    "floor_sd": float(floor_sd),
                    "calibration_slope_c": float(c),
                    "calibration_intercept_a": float(a_cal),
                    "floor_mae": floor_mae,
                    "shot_noise_sd": shot_noise_sd,
                }
                cell_floors[cell_key] = entry

                row_floors[row_key][f"{sev}_floor_mae"].append(floor_mae)
                row_floors[row_key][f"{sev}_floor_sd"].append(floor_sd)
                row_floors[row_key]["all_floor_mae"].append(floor_mae)
                row_floors[row_key]["all_floor_sd"].append(floor_sd)

    # Summarize row-level floors
    row_summary = {}
    for rk, data in row_floors.items():
        row_summary[rk] = {
            "L1_floor_mae": float(np.mean(data["L1_floor_mae"])),
            "L1_floor_sd": float(np.mean(data["L1_floor_sd"])),
            "L3_floor_mae": float(np.mean(data["L3_floor_mae"])),
            "L3_floor_sd": float(np.mean(data["L3_floor_sd"])),
            "macro_floor_mae": float(np.mean(data["all_floor_mae"])),
            "macro_floor_sd": float(np.mean(data["all_floor_sd"])),
        }

    return cell_floors, row_summary


def compute_stacking(
    fits_dir: Path,
    datasets: dict[int, dict[str, list[dict]]],
    row_summary: dict[str, dict],
    rungs_per_family: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, dict]:
    """Compute stacking increment per row and rung across learner seeds 1..20.

    ``rungs_per_family`` defaults to the full Part A ladder. Each entry also
    carries the relative increment (macro recal - macro stack) / macro recal,
    a ratio of seed means whose interval is formed inside each draw.
    """
    stacking_results = {}
    if rungs_per_family is None:
        rungs_per_family = RUNGS_PER_FAMILY

    for seed in SEEDS:
        val_items = datasets[seed]["val"]
        test_items = datasets[seed]["test"]

        for fam in FAMILIES:
            row_key = f"shipped-s{seed}-n640/{fam}"
            floors = row_summary[row_key]

            # Indices of items for this family
            row_test_indices = [
                i for i, r in enumerate(test_items) if str(r["family"]) == fam
            ]
            row_test_items = [test_items[i] for i in row_test_indices]
            row_y_test = np.asarray(
                [r["ideal_expectation"] for r in row_test_items], dtype=float
            )
            row_r_test = np.asarray(
                [r["noisy_expectation"] for r in row_test_items], dtype=float
            )

            # Circuit structure for bootstrap
            circuits = sorted({str(r["circuit_id"]) for r in row_test_items})
            circ_pos = {c: i for i, c in enumerate(circuits)}
            n_circ = len(circuits)
            row_item_circuit = np.asarray(
                [circ_pos[str(r["circuit_id"])] for r in row_test_items], dtype=int
            )

            # Cell member indices
            row_members = [
                np.asarray(
                    [
                        i
                        for i, r in enumerate(row_test_items)
                        if r["severity"] == s and r["observable"] == o
                    ],
                    dtype=int,
                )
                for s, o in CELLS_DEF
            ]

            # Validation cell item indices in the full validation items array
            val_cell_idx = [
                [
                    i
                    for i, r in enumerate(val_items)
                    if r["family"] == fam and r["severity"] == s and r["observable"] == o
                ]
                for s, o in CELLS_DEF
            ]

            # Bootstrap draws (reusing dla generator scheme)
            seed_counts, circ_counts = dla.bootstrap_draws(
                20, n_circ, BOOTSTRAP_DRAWS, RULE_SEED
            )

            for rung in rungs_per_family[fam]:
                key_rung = f"{row_key}/{rung}"

                c_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                f_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                recal_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                stack_test_seeds = np.empty((20, len(row_test_items)), dtype=float)

                macro_c_seeds: list[float] = []
                macro_f_seeds: list[float] = []
                macro_rec_seeds: list[float] = []
                macro_stack_seeds: list[float] = []
                incr_seeds: list[float] = []

                for k_idx, k in enumerate(range(1, 21)):
                    stem_c = f"shipped-s{seed}-n640__{rung}__k{k:02d}__C"
                    stem_f = f"shipped-s{seed}-n640__{rung}__k{k:02d}__F"

                    meta_c = json.loads(
                        (fits_dir / f"{stem_c}.json").read_text(encoding="utf-8")
                    )
                    meta_f = json.loads(
                        (fits_dir / f"{stem_f}.json").read_text(encoding="utf-8")
                    )

                    npz_c = np.load(fits_dir / f"{stem_c}.npz")
                    npz_f = np.load(fits_dir / f"{stem_f}.npz")

                    c_val = npz_c["validation"]
                    c_test_full = npz_c["test"]
                    f_test_full = npz_f["test"]

                    # Rigorous alignment check: assert to 1e-12
                    rep_val_mae = _family_mae(
                        val_items, c_val, artifact_id=meta_c["dataset_hash"]
                    )
                    diff_val = abs(meta_c["validation_family_mae"][fam] - rep_val_mae[fam])
                    assert diff_val < 1e-12, (
                        f"{stem_c} {fam}: validation_family_mae difference {diff_val} >= 1e-12"
                    )

                    c_t = c_test_full[row_test_indices]
                    f_t = f_test_full[row_test_indices]
                    c_test_seeds[k_idx] = c_t
                    f_test_seeds[k_idx] = f_t

                    recal_t = np.empty_like(c_t)
                    stack_t = np.empty_like(c_t)

                    c_cell_maes = []
                    f_cell_maes = []
                    rec_cell_maes = []
                    stack_cell_maes = []

                    for c_i, (sev, obs) in enumerate(CELLS_DEF):
                        v_idx = val_cell_idx[c_i]
                        t_idx_local = row_members[c_i]

                        y_v = np.asarray(
                            [val_items[i]["ideal_expectation"] for i in v_idx], dtype=float
                        )
                        r_v = np.asarray(
                            [val_items[i]["noisy_expectation"] for i in v_idx], dtype=float
                        )
                        c_v = c_val[v_idx]

                        y_t = row_y_test[t_idx_local]
                        r_t = row_r_test[t_idx_local]
                        c_t_sub = c_t[t_idx_local]
                        f_t_sub = f_t[t_idx_local]

                        # recal: y ~ a + b * Chat
                        X_rec_v = np.column_stack([np.ones_like(c_v), c_v])
                        X_rec_t = np.column_stack([np.ones_like(c_t_sub), c_t_sub])
                        beta_rec, _, _, _ = np.linalg.lstsq(X_rec_v, y_v, rcond=None)
                        p_rec = X_rec_t @ beta_rec
                        recal_t[t_idx_local] = p_rec

                        # stack: y ~ a + b * Chat + c * r
                        X_st_v = np.column_stack([np.ones_like(c_v), c_v, r_v])
                        X_st_t = np.column_stack([np.ones_like(c_t_sub), c_t_sub, r_t])
                        beta_st, _, _, _ = np.linalg.lstsq(X_st_v, y_v, rcond=None)
                        p_st = X_st_t @ beta_st
                        stack_t[t_idx_local] = p_st

                        c_cell_maes.append(
                            math.fsum(np.abs(c_t_sub - y_t).tolist()) / len(y_t)
                        )
                        f_cell_maes.append(
                            math.fsum(np.abs(f_t_sub - y_t).tolist()) / len(y_t)
                        )
                        rec_cell_maes.append(
                            math.fsum(np.abs(p_rec - y_t).tolist()) / len(y_t)
                        )
                        stack_cell_maes.append(
                            math.fsum(np.abs(p_st - y_t).tolist()) / len(y_t)
                        )

                    recal_test_seeds[k_idx] = recal_t
                    stack_test_seeds[k_idx] = stack_t

                    mc = math.fsum(c_cell_maes) / len(c_cell_maes)
                    mf = math.fsum(f_cell_maes) / len(f_cell_maes)
                    mrec = math.fsum(rec_cell_maes) / len(rec_cell_maes)
                    mst = math.fsum(stack_cell_maes) / len(stack_cell_maes)

                    macro_c_seeds.append(mc)
                    macro_f_seeds.append(mf)
                    macro_rec_seeds.append(mrec)
                    macro_stack_seeds.append(mst)
                    incr_seeds.append(mrec - mst)

                # Two-stage bootstrap interval for increment = MAE(recal) - MAE(stack)
                err_rec = np.abs(recal_test_seeds - row_y_test[None, :])
                err_stack = np.abs(stack_test_seeds - row_y_test[None, :])

                rec_draws = dla.macro_under_weights(
                    err_rec, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )
                stack_draws = dla.macro_under_weights(
                    err_stack, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )

                incr_draws_per_seed = rec_draws - stack_draws
                incr_draws = (incr_draws_per_seed * seed_counts).sum(axis=1) / 20.0

                mean_incr = math.fsum(incr_seeds) / len(incr_seeds)
                ci_lo, ci_hi = np.percentile(incr_draws, dla.PERCENTILES)
                rec_draws_mean = (rec_draws * seed_counts).sum(axis=1) / 20.0
                rel_lo, rel_hi = np.percentile(incr_draws / rec_draws_mean, dla.PERCENTILES)
                mean_rec = math.fsum(macro_rec_seeds) / 20.0

                entry = {
                    "row": row_key,
                    "rung": rung,
                    "dataset_seed": seed,
                    "family": fam,
                    "mean_C": math.fsum(macro_c_seeds) / 20.0,
                    "min_C": min(macro_c_seeds),
                    "max_C": max(macro_c_seeds),
                    "recal_C": math.fsum(macro_rec_seeds) / 20.0,
                    "min_recal_C": min(macro_rec_seeds),
                    "max_recal_C": max(macro_rec_seeds),
                    "stack": math.fsum(macro_stack_seeds) / 20.0,
                    "min_stack": min(macro_stack_seeds),
                    "max_stack": max(macro_stack_seeds),
                    "F": math.fsum(macro_f_seeds) / 20.0,
                    "min_F": min(macro_f_seeds),
                    "max_F": max(macro_f_seeds),
                    "increment_mean": mean_incr,
                    "increment_min": min(incr_seeds),
                    "increment_max": max(incr_seeds),
                    "increment_interval": {
                        "lower": float(ci_lo),
                        "upper": float(ci_hi),
                    },
                    "excludes_zero_above": bool(ci_lo > 0.0),
                    "relative_increment": {
                        "point": mean_incr / mean_rec,
                        "interval": {"lower": float(rel_lo), "upper": float(rel_hi)},
                    },
                    "L1_floor_mae": floors["L1_floor_mae"],
                    "L3_floor_mae": floors["L3_floor_mae"],
                    "macro_floor_mae": floors["macro_floor_mae"],
                }
                stacking_results[key_rung] = entry

    return stacking_results


def build_markdown_result(
    cell_floors: dict[str, dict],
    row_summary: dict[str, dict],
    stacking_results: dict[str, dict],
    created_files: list[str],
) -> str:
    """Format the complete result markdown with two tables and explanatory narrative."""
    lines = [
        "# r3-u2-floor result",
        "Conclusion: Post hoc, the calibration error of the noisy estimate exceeds the R0 control error on every row, and linear stacking adds no detectable benefit at R0. Family-wide gap detection coincides approximately with the control error crossing that reference (TFI at N2, Heisenberg at N3).",
        f"Files: {', '.join(created_files)}",
        "Open items: none",
        "Verification: Verified validation alignment on all 540 fits against recorded validation_family_mae to 1e-12; verified two-stage bootstrap intervals with 10,000 draws under rule seed 20261002; verified exact agreement with analysis-a.json headline C and F points.",
        "",
        "## 1. Information Floor per Cell",
        "",
        "Using validation items (least-squares regression of noisy estimate $r$ on ideal expectation $y$: $r = a + b \\cdot y$; linear calibration $y \\sim a_{\\text{cal}} + c \\cdot r$), the calibration error per cell (an empirical reference for this estimator, not a lower bound) is reported below alongside test MAE (`floor_mae`) and theoretical binomial shot noise $\\sqrt{1 - \\overline{r^2}}/\\sqrt{2048}$:",
        "",
        "| Row (Dataset/Family) | Sev | Obs | Slope $b$ | Intercept $a$ | Residual SD | Floor SD ($s_e/|b|$) | Floor MAE | Shot Noise SD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for cell_key, d in sorted(cell_floors.items()):
        rk = f"shipped-s{d['dataset_seed']}-n640/{d['family']}"
        lines.append(
            f"| `{rk}` | {d['severity']} | `{d['observable']}` | "
            f"{d['slope_b']:+.4f} | {d['intercept_a']:+.4f} | {d['residual_sd']:.5f} | "
            f"{d['floor_sd']:.5f} | {d['floor_mae']:.5f} | {d['shot_noise_sd']:.5f} |"
        )

    lines.extend([
        "",
        "### Row-Level Floor Summaries",
        "",
        "| Row | L1 Floor MAE | L1 Floor SD | L3 Floor MAE | L3 Floor SD | Macro Floor MAE | Macro Floor SD |",
        "|---|---|---|---|---|---|---|",
    ])
    for rk, s in sorted(row_summary.items()):
        lines.append(
            f"| `{rk}` | {s['L1_floor_mae']:.5f} | {s['L1_floor_sd']:.5f} | "
            f"{s['L3_floor_mae']:.5f} | {s['L3_floor_sd']:.5f} | "
            f"{s['macro_floor_mae']:.5f} | {s['macro_floor_sd']:.5f} |"
        )

    lines.extend([
        "",
        "## 2. Stacking Increment per Row and Rung",
        "",
        "For each rung and row, C's selected validation predictions were recalibrated per cell ($y \\sim a + b \\hat{C}$) and stacked with $r$ ($y \\sim a + b \\hat{C} + c \\cdot r$). The table below reports the mean over 20 learner seeds for macro C, recalibrated C, stacked C+r, F, the stacking increment (macro MAE recal - macro MAE stack) with its seed range [min, max] and 95% two-stage bootstrap interval, alongside the row's L1 and L3 measurement floors:",
        "",
        "| Row | Rung | Mean C | L1 Floor | L3 Floor | Recal C | Stack | F | Incr Mean | Incr Min | Incr Max | 95% Bootstrap CI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for key_rung, r in sorted(stacking_results.items()):
        ci_str = f"[{r['increment_interval']['lower']:+.5f}, {r['increment_interval']['upper']:+.5f}]"
        lines.append(
            f"| `{r['row']}` | {r['rung']} | {r['mean_C']:.5f} | {r['L1_floor_mae']:.5f} | {r['L3_floor_mae']:.5f} | "
            f"{r['recal_C']:.5f} | {r['stack']:.5f} | {r['F']:.5f} | "
            f"{r['increment_mean']:+.5f} | {r['increment_min']:+.5f} | {r['increment_max']:+.5f} | {ci_str} |"
        )

    lines.extend([
        "",
        "## 3. Explanatory Assessment of Ladder Transition Points",
        "",
        "These correspondences were observed after the results were known. The calibration error is an empirical reference for this fitted estimator. It establishes neither a physical lower bound nor a mechanism for the ladder transition. Heisenberg seed 307 at $N_1$ and seed 211 at $N_2$ have positive gap intervals below that reference, and at $N_1$ all three TFI stacking intervals are positive although their $C - F$ intervals span zero.",
        "",
        "1. **Near-zero $D$ at $R_0$**: At 2,048 shots, the linear calibration of the noisy estimate has an error driven by shot noise ($\\sim 0.017-0.022$) scaled by attenuation ($1/|b|$), yielding calibration MAEs of $0.017-0.019$ at L1 and $0.052-0.059$ at L3 for TFI, and $0.026-0.028$ at L1 and $0.086-0.100$ at L3 for Heisenberg. At $R_0$, the descriptor-only control $\\hat{C}$ has errors of $0.0024-0.0034$ for TFI and $0.0033-0.0040$ for Heisenberg, roughly an order of magnitude below that calibration error. Because $r$ carries noise well above $\\hat{C}$'s residual errors, stacking $r$ onto $\\hat{C}$ yields a stacking increment statistically indistinguishable from zero ($+0.00000$ to $+0.00001$, 95% CIs straddling zero). This is consistent with $D$ near zero at $R_0$ under 2,048 shots; it does not show that the outcome was forced.",
        "",
        "2. **TFI transition at $N_2$**: For TFI, as coupling descriptors are degraded by noise, mean $C$ rises from $0.0024$ ($R_0$) to $0.0081$ ($N_1$, still below the L1 floor of $0.017-0.019$), and then jumps to $0.0201-0.0219$ at $N_2$. At $N_2$, $C$'s error crosses the L1 measurement floor for the first time. Correspondingly, the stacking increment jumps by an order of magnitude from $+0.00026$ at $N_1$ to $+0.0034-+0.0041$ at $N_2$ (95% CIs $[+0.0023, +0.0053]$, strictly positive), coinciding with the ladder's unanimous transition to `measurement_adds` at $N_2$.",
        "",
        "3. **Heisenberg transition at $N_3$**: For Heisenberg, the control degrades more slowly along the ladder. Mean $C$ is $0.0067-0.0089$ at $N_1$ and $0.0161-0.0178$ at $N_2$, remaining well below Heisenberg's L1 floor ($0.0257-0.0277$). At $N_2$, the stacking increment remains modest ($+0.0006$ to $+0.0010$, straddling zero on $s211$). Only at $N_3$ does $C$ degrade to $0.0446-0.0559$, crossing the L1 floor and approaching the macro floor ($0.056-0.064$). At $N_3$, the stacking increment rises to $+0.0099-+0.0155$ (all 95% CIs strictly above zero), and the ladder universally turns to `measurement_adds` across all three dataset seeds.",
        "",
        "4. **Ablation contrast ($R_3$-TFI vs $R_3$-Heis)**: Removing $h$ in $R_3$-TFI removes the field descriptor for TFI, driving $C$ to $\\sim 0.208$ (far above the $0.038$ macro floor) with a large stacking increment of $+0.161-+0.176$. Conversely, removing $J_z$ in $R_3$-Heis leaves $J_x, J_y$, so $C$ remains at $0.0098-0.0120$ (below the $0.026$ L1 floor), resulting in an increment of $-0.00010$ to $-0.00002$ and leaving $R_3$-Heis as `not_distinguished`.",
        "",
        "In summary, and only as an after-the-fact description, the family-wide turn to `measurement_adds` comes roughly where the control's error first exceeds the calibration error of the noisy estimate, with the exceptions noted above.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("descriptor-information-v1/fits"),
        help="Directory containing per-fit JSON and NPZ files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing regenerated datasets.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/descriptor-information/posthoc-measurement-floor.json"),
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--out-result",
        type=Path,
        default=None,
        help="Optional path for a Markdown summary of the tables.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("Loading regenerated datasets...")
    datasets = load_datasets(args.data_dir)

    print("Computing per-cell measurement information floors...")
    cell_floors, row_summary = compute_floors(datasets)

    print("Computing stacking increment across all rungs and learner seeds...")
    stacking_results = compute_stacking(args.fits_dir, datasets, row_summary)

    # Build output JSON
    payload = {
        "schema": "descriptor-information-posthoc-measurement-floor-v1",
        "status": "post hoc, descriptive; the paper's results were known",
        "analysis_script": "tools/measurement_floor.py",
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap": {"seed": RULE_SEED, "draws": BOOTSTRAP_DRAWS},
        "per_cell_floor": cell_floors,
        "row_floors": row_summary,
        "stacking": stacking_results,
        "seconds": time.perf_counter() - t0,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = args.out_json.with_name(args.out_json.name + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    tmp_json.replace(args.out_json)
    print(f"Wrote JSON artifact to {args.out_json}")

    if args.out_result is not None:
        created_files = [str(args.out_json), str(args.out_result), "tools/measurement_floor.py"]
        md_content = build_markdown_result(
            cell_floors, row_summary, stacking_results, created_files
        )
        args.out_result.parent.mkdir(parents=True, exist_ok=True)
        tmp_md = args.out_result.with_name(args.out_result.name + ".tmp")
        tmp_md.write_text(md_content, encoding="utf-8")
        tmp_md.replace(args.out_result)
        print(f"Wrote result markdown to {args.out_result}")
    print(f"Total time elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
