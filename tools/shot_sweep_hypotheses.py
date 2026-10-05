#!/usr/bin/env python3
"""Evaluation of hypotheses H1 and H2 for the shot-count sweep of QEMScore.

Frozen rule: docs/frozen-rules/2026-10-03-shot-sweep.md
Section: Hypotheses and Decision Rules

Evaluates:
- H1 (the gap follows the ceiling):
  - (a) Spearman correlation >= 0.8 across seven levels for at least 5 of 6 series in each of N1 and N2.
  - (b) Median ratio observed D/C to D/C* between 0.6 and 1.1 on qualifying cells (D/C* >= 0.10, non-2048).
  - (c) Lower bound of observed D/C exceeds D/C* in at most 10% of qualifying cells.
  - Minimum count rule: >= 10 qualifying cells from >= 2 rows; otherwise not testable.
- H2 (the prediction locates the label change):
  - At least 80% of label-changing cells carry predicted label.
  - At least 80% of all predicted cells carry predicted label.
  - Measurement hurts counts as miss for adds, match for not distinguished (reporting match count).
  - Minimum count rule: >= 6 label-changing cells from >= 2 rows; otherwise not testable.
- Reading outcome numbers 1 through 6 (--prefit-failed forces 6).
- Descriptive series: R0 and R5 for H1, R5 and 2048 level for H2.

Computes everything independently from analysis A and analysis B; exits nonzero if any value
differs by > 1e-12 or any label or outcome differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[1]

SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
RUNGS = ("R0", "N1", "N2", "R5")
CANONICAL_LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")
RULE_FILE = "docs/frozen-rules/2026-10-03-shot-sweep.md"

LABELS_B = {
    "Measurement adds": "measurement_adds",
    "Measurement hurts": "measurement_hurts",
    "Not distinguished": "not_distinguished",
    "measurement_adds": "measurement_adds",
    "measurement_hurts": "measurement_hurts",
    "not_distinguished": "not_distinguished",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float | None, bool]:
    """Compute Spearman correlation with average ranks for ties.
    Returns (corr, is_constant).
    A series whose D/C* or observed D/C is constant across levels fails (a).
    """
    if len(x) != len(y) or len(x) < 2:
        return None, False
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return None, True
    with np.errstate(all="ignore"):
        res = spearmanr(x_arr, y_arr).statistic
        if np.isnan(res):
            return None, True
        return float(res), False


def parse_level_args(entries: list[str]) -> dict[str, Path]:
    out = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid format '{entry}', expected <level>=<path>")
        lvl, p = entry.split("=", 1)
        out[lvl.strip()] = Path(p.strip())
    return out


def extract_analysis_a(
    path: Path,
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract observed values from analysis A: out[row][rung] = {...}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["parts"]["A"]["rows"]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in dataset_seeds:
        for fam in FAMILIES:
            row_key = f"shipped-s{seed}-n640/{fam}"
            out[row_key] = {}
            if row_key not in rows:
                continue
            row_data = rows[row_key]
            for rung in RUNGS:
                if rung not in row_data["rungs"]:
                    continue
                r_rung = row_data["rungs"][rung]
                ci = r_rung["D_over_C"]["interval"]
                out[row_key][rung] = {
                    "observed_D_over_C": float(r_rung["D_over_C"]["point"]),
                    "observed_D_over_C_interval": [float(ci["lower"]), float(ci["upper"])],
                    "observed_label": str(r_rung["classification"]["label"]),
                }
    return out


def extract_analysis_b(
    path: Path,
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract observed values from analysis B: out[row][rung] = {...}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, Any]]] = {}

    # Prefer by_part_row_rung if available
    b_part_a = data.get("by_part_row_rung", {}).get("A", {})
    # Also prepare from cells list as fallback
    cells_map: dict[tuple[int, str, str], dict] = {}
    for cell in data.get("cells", []):
        if str(cell.get("part", "")).upper() == "A":
            cells_map[(int(cell["dataset_seed"]), str(cell["family"]), str(cell["rung"]))] = cell

    for seed in dataset_seeds:
        for fam in FAMILIES:
            row_key = f"shipped-s{seed}-n640/{fam}"
            b_row_key = f"s{seed}__{fam}"
            out[row_key] = {}
            for rung in RUNGS:
                cell_dict = None
                if b_row_key in b_part_a and rung in b_part_a[b_row_key]:
                    cell_dict = b_part_a[b_row_key][rung]
                elif (seed, fam, rung) in cells_map:
                    cell_dict = cells_map[(seed, fam, rung)]

                if cell_dict is not None and cell_dict.get("status") == "estimated":
                    ci = cell_dict["d_over_c_ci_95"]
                    raw_lab = cell_dict["classification_label"]
                    canon_lab = LABELS_B.get(raw_lab, raw_lab)
                    out[row_key][rung] = {
                        "observed_D_over_C": float(cell_dict["mean_d_over_c"]),
                        "observed_D_over_C_interval": [float(ci[0]), float(ci[1])],
                        "observed_label": str(canon_lab),
                    }
    return out


def expected_classification_cells(dataset_seeds: Sequence[int] = SEEDS) -> set[str]:
    return {
        f"{level}/shipped-s{seed}-n640/{family}/{rung}"
        for level in CANONICAL_LEVELS
        for seed in dataset_seeds
        for family in FAMILIES
        for rung in RUNGS
    }


def check_inventory(
    predictions: dict,
    obs_by_level: dict,
    dataset_seeds: Sequence[int] = SEEDS,
) -> None:
    """The frozen sweep needs every level and every classification cell.

    A missing record is an input error, distinct from the rule's not-testable
    outcomes, so the evaluation stops instead of shrinking a series or a
    denominator.
    """
    if set(obs_by_level) != set(CANONICAL_LEVELS):
        raise ValueError(
            f"the frozen sweep requires all seven analysis levels; got {sorted(obs_by_level)}")
    expected = expected_classification_cells(dataset_seeds=dataset_seeds)
    actual = set(predictions.get("classification_cells", {}))
    if actual != expected:
        raise ValueError(
            f"incomplete prediction inventory: missing={sorted(expected - actual)[:10]} "
            f"({len(expected - actual)} in all), unexpected={sorted(actual - expected)[:10]}")
    missing_obs = [
        f"{level}/shipped-s{seed}-n640/{family}/{rung}"
        for level in CANONICAL_LEVELS
        for seed in dataset_seeds
        for family in FAMILIES
        for rung in RUNGS
        if rung not in obs_by_level[level].get(f"shipped-s{seed}-n640/{family}", {})
    ]
    if missing_obs:
        raise ValueError(f"analysis cells missing: {missing_obs[:10]} ({len(missing_obs)} in all)")


def evaluate_pipeline(
    predictions: dict,
    obs_by_level: dict[str, dict[str, dict[str, dict[str, Any]]]],
    prefit_failed: bool = False,
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict:
    check_inventory(predictions, obs_by_level, dataset_seeds=dataset_seeds)
    available_levels = list(CANONICAL_LEVELS)

    # --- H1 (a): 12 series of 3 dataset seeds x 2 families x 2 rungs (N1, N2) ---
    series_results: dict[str, dict[str, Any]] = {}
    for rung in ("N1", "N2", "R0", "R5"):
        series_results[rung] = {}
        for seed in dataset_seeds:
            for fam in FAMILIES:
                row_key = f"shipped-s{seed}-n640/{fam}"
                x_vals = []
                y_vals = []
                for lvl in available_levels:
                    ck = f"{lvl}/{row_key}/{rung}"
                    if ck in predictions["classification_cells"]:
                        x_vals.append(float(predictions["classification_cells"][ck]["D_over_C_star"]))
                        y_vals.append(float(obs_by_level[lvl][row_key][rung]["observed_D_over_C"]))

                corr, is_const = compute_spearman(x_vals, y_vals)
                passes = bool(corr is not None and not is_const and corr >= 0.80)
                series_results[rung][row_key] = {
                    "spearman": corr,
                    "constant": is_const,
                    "passes": passes,
                    "n_levels": len(x_vals),
                    "dc_star_points": x_vals,
                    "observed_dc_points": y_vals,
                }

    n1_passes = sum(1 for v in series_results["N1"].values() if v["passes"])
    n2_passes = sum(1 for v in series_results["N2"].values() if v["passes"])
    h1_a_holds = bool(n1_passes >= 5 and n2_passes >= 5)

    # --- H1 (b) and (c): Qualifying cells (N1 and N2, non-2048, D/C* >= 0.10) ---
    qualifying_cells = []
    for lvl in available_levels:
        if str(lvl) == "2048":
            continue
        for seed in dataset_seeds:
            for fam in FAMILIES:
                row_key = f"shipped-s{seed}-n640/{fam}"
                for rung in ("N1", "N2"):
                    ck = f"{lvl}/{row_key}/{rung}"
                    if ck not in predictions["classification_cells"]:
                        continue
                    dc_star = float(predictions["classification_cells"][ck]["D_over_C_star"])
                    if dc_star >= 0.10:
                        obs = obs_by_level[lvl][row_key][rung]
                        obs_dc = float(obs["observed_D_over_C"])
                        obs_ci = obs["observed_D_over_C_interval"]
                        qualifying_cells.append({
                            "key": ck,
                            "level": lvl,
                            "row": row_key,
                            "rung": rung,
                            "D_over_C_star": dc_star,
                            "observed_D_over_C": obs_dc,
                            "observed_D_over_C_lower": float(obs_ci[0]),
                            "observed_D_over_C_upper": float(obs_ci[1]),
                            "ratio": float(obs_dc / dc_star),
                            "lower_exceeds_ceiling": bool(float(obs_ci[0]) > dc_star),
                        })

    qual_count = len(qualifying_cells)
    qual_rows = len({c["row"] for c in qualifying_cells})
    h1_testable = bool(qual_count >= 10 and qual_rows >= 2)

    if not h1_testable:
        h1_b_holds = None
        h1_c_holds = None
        median_ratio = None
        exceed_count = None
        fraction_exceeding = None
        h1_holds = None
    else:
        ratios = [c["ratio"] for c in qualifying_cells]
        median_ratio = float(np.median(ratios))
        h1_b_holds = bool(0.60 <= median_ratio <= 1.10)

        exceed_count = sum(1 for c in qualifying_cells if c["lower_exceeds_ceiling"])
        fraction_exceeding = float(exceed_count / qual_count)
        h1_c_holds = bool(fraction_exceeding <= 0.10)

        h1_holds = bool(h1_a_holds and h1_b_holds and h1_c_holds)

    # --- H2: Classification cells of R0, N1, N2 at six levels other than 2048 ---
    h2_cells: list[dict[str, Any]] = []
    r5_descriptive_cells: list[dict[str, Any]] = []
    level_2048_descriptive_cells: list[dict[str, Any]] = []

    for lvl in available_levels:
        is_2048 = (str(lvl) == "2048")
        for seed in dataset_seeds:
            for fam in FAMILIES:
                row_key = f"shipped-s{seed}-n640/{fam}"
                for rung in RUNGS:
                    ck = f"{lvl}/{row_key}/{rung}"
                    if ck not in predictions["classification_cells"]:
                        continue
                    cell_pred = predictions["classification_cells"][ck]
                    pred_label = cell_pred["predicted_label"]
                    obs = obs_by_level[lvl][row_key][rung]
                    obs_label = obs["observed_label"]
                    is_lc = bool(cell_pred["label_changing"])

                    # Match evaluation
                    if pred_label == "measurement_adds":
                        matched = bool(obs_label == "measurement_adds")
                        hurts_match = False
                    elif pred_label == "not_distinguished":
                        if obs_label == "not_distinguished":
                            matched = True
                            hurts_match = False
                        elif obs_label == "measurement_hurts":
                            matched = True
                            hurts_match = True
                        else:
                            matched = False
                            hurts_match = False
                    else:
                        matched = None
                        hurts_match = False

                    rec = {
                        "key": ck,
                        "level": lvl,
                        "row": row_key,
                        "rung": rung,
                        "D_over_C_star": float(cell_pred["D_over_C_star"]),
                        "adds_threshold": float(cell_pred["adds_threshold"]),
                        "predicted_label": pred_label,
                        "reference_label": str(cell_pred["reference_label"]),
                        "observed_label": obs_label,
                        "label_changing": is_lc,
                        "matched": matched,
                        "hurts_match": hurts_match,
                    }

                    if is_2048:
                        level_2048_descriptive_cells.append(rec)
                    elif rung == "R5":
                        r5_descriptive_cells.append(rec)
                    else:
                        h2_cells.append(rec)

    label_changing_cells = [c for c in h2_cells if c["label_changing"]]
    lc_count = len(label_changing_cells)
    lc_rows = len({c["row"] for c in label_changing_cells})
    h2_testable = bool(lc_count >= 6 and lc_rows >= 2)

    if not h2_testable:
        crit_lc_holds = None
        crit_all_holds = None
        rate_lc = None
        rate_all = None
        h2_holds = None
        hurts_matches = sum(1 for c in h2_cells if c["hurts_match"])
        lc_matches = None
        all_pred_matches = None
        all_pred_count = len([c for c in h2_cells if c["predicted_label"] is not None])
    else:
        lc_matches = sum(1 for c in label_changing_cells if c["matched"])
        rate_lc = float(lc_matches / lc_count)
        crit_lc_holds = bool(rate_lc >= 0.80)

        predicted_cells = [c for c in h2_cells if c["predicted_label"] is not None]
        all_pred_count = len(predicted_cells)
        all_pred_matches = sum(1 for c in predicted_cells if c["matched"])
        rate_all = float(all_pred_matches / all_pred_count) if all_pred_count > 0 else 0.0
        crit_all_holds = bool(rate_all >= 0.80)

        hurts_matches = sum(1 for c in predicted_cells if c["hurts_match"])
        h2_holds = bool(crit_lc_holds and crit_all_holds)

    # --- Reading outcome determination (1 to 6) ---
    if prefit_failed:
        reading_outcome = 6
        reading_statement = (
            "Outcome 6: A check before any fit fails: no fit runs, and the paper reports the sweep as not run, with the reason."
        )
    elif not h1_testable or not h2_testable:
        # Rule, Reading item 5: no conclusion from the untestable hypothesis; the
        # other is read alone, and if it fails, item 4 applies.
        reasons = []
        if not h1_testable:
            reasons.append(f"H1 not testable ({qual_count} qualifying cells, {qual_rows} rows)")
        if not h2_testable:
            reasons.append(f"H2 not testable ({lc_count} label-changing cells, {lc_rows} rows)")
        if h1_testable:
            read_alone = ("H1", h1_holds)
        elif h2_testable:
            read_alone = ("H2", h2_holds)
        else:
            read_alone = None
        if read_alone is not None and not read_alone[1]:
            reading_outcome = 4
            reading_statement = (
                f"Outcome 4 (through item 5): {'; '.join(reasons)}; {read_alone[0]}, read alone, "
                "fails, so the crossover is not supported by this test, and only the "
                "descriptive 2,048-shot statement remains."
            )
        elif read_alone is not None:
            permitted = (
                "the gap grew with the precision of r as predicted"
                if read_alone[0] == "H1"
                else "the ceiling located the label changes"
            )
            reading_outcome = 5
            reading_statement = (
                f"Outcome 5: {'; '.join(reasons)}; {read_alone[0]}, read alone, holds: "
                f"the paper may write that {permitted}."
            )
        else:
            reading_outcome = 5
            reading_statement = f"Outcome 5: {'; '.join(reasons)}; no conclusion is drawn."
    elif h1_holds and h2_holds:
        reading_outcome = 1
        reading_statement = (
            "Outcome 1: H1 and H2 hold: the paper may write that a ceiling computed from the two error scales before fitting predicted the size of the gap and where it became detectable."
        )
    elif h1_holds and not h2_holds:
        reading_outcome = 2
        reading_statement = (
            "Outcome 2: H1 holds and H2 fails: the gap grew with the precision of r as predicted, but the ceiling did not locate the label change."
        )
    elif not h1_holds and h2_holds:
        reading_outcome = 3
        reading_statement = (
            "Outcome 3: H2 holds and H1 fails: the ceiling located the label changes but did not track the size of the gap."
        )
    else:
        reading_outcome = 4
        reading_statement = (
            "Outcome 4: Both fail: the crossover is not supported by this test, and only the descriptive 2,048-shot statement remains."
        )

    return {
        "reading_outcome": reading_outcome,
        "reading_statement": reading_statement,
        "h1": {
            "testable": h1_testable,
            "qualifying_cell_count": qual_count,
            "qualifying_row_count": qual_rows,
            "holds": h1_holds,
            "part_a": {
                "holds": h1_a_holds,
                "n1_pass_count": n1_passes,
                "n2_pass_count": n2_passes,
                "series_n1": series_results["N1"],
                "series_n2": series_results["N2"],
                "series_r0_descriptive": series_results["R0"],
                "series_r5_descriptive": series_results["R5"],
            },
            "part_b": {
                "holds": h1_b_holds,
                "median_ratio": median_ratio,
                "qualifying_cells": qualifying_cells,
            },
            "part_c": {
                "holds": h1_c_holds,
                "exceeding_count": exceed_count,
                "fraction_exceeding": fraction_exceeding,
            },
        },
        "h2": {
            "testable": h2_testable,
            "label_changing_cell_count": lc_count,
            "label_changing_row_count": lc_rows,
            "holds": h2_holds,
            "criterion_label_changing": {
                "holds": crit_lc_holds,
                "n_cells": lc_count,
                "matches": lc_matches,
                "rate": rate_lc,
            },
            "criterion_all_predicted": {
                "holds": crit_all_holds,
                "n_cells": all_pred_count,
                "matches": all_pred_matches,
                "rate": rate_all,
            },
            "hurts_as_match_count": hurts_matches,
            "cells": {c["key"]: c for c in h2_cells},
            "descriptive_r5": {c["key"]: c for c in r5_descriptive_cells},
            "descriptive_2048": {c["key"]: c for c in level_2048_descriptive_cells},
        },
    }


def compare_ab_results(res_a: Any, res_b: Any, path: str = "") -> list[str]:
    diffs = []
    if type(res_a) != type(res_b):
        diffs.append(f"{path}: type mismatch {type(res_a)} vs {type(res_b)}")
        return diffs
    if isinstance(res_a, dict):
        for k in sorted(set(res_a.keys()) | set(res_b.keys())):
            if k not in res_a or k not in res_b:
                diffs.append(f"{path}.{k}: missing in one analysis")
            else:
                diffs.extend(compare_ab_results(res_a[k], res_b[k], f"{path}.{k}"))
    elif isinstance(res_a, (list, tuple)):
        if len(res_a) != len(res_b):
            diffs.append(f"{path}: length mismatch {len(res_a)} vs {len(res_b)}")
        else:
            for i, (va, vb) in enumerate(zip(res_a, res_b)):
                diffs.extend(compare_ab_results(va, vb, f"{path}[{i}]"))
    elif isinstance(res_a, float):
        if math.isnan(res_a) and math.isnan(res_b):
            pass
        elif abs(res_a - res_b) > 1e-12:
            diffs.append(f"{path}: float diff {abs(res_a - res_b):.2e} ({res_a} vs {res_b}) > 1e-12")
    else:
        if res_a != res_b:
            diffs.append(f"{path}: value mismatch {res_a!r} vs {res_b!r}")
    return diffs


def write_prefit_failure(reports: list[Path], out: Path) -> None:
    """Reading item 6: a check before any fit failed, so no fit ran.

    Neither hypothesis is evaluated; the output names the failed check reports.
    """
    if not reports:
        raise SystemExit("--prefit-failed needs at least one --failure-report")
    entries = []
    for report in reports:
        payload = json.loads(Path(report).read_text(encoding="utf-8"))
        entries.append({
            "path": str(report),
            "sha256": sha256_file(Path(report)),
            "command": payload.get("command"),
            "passed": payload.get("passed"),
        })
    if all(entry["passed"] is True for entry in entries):
        raise SystemExit("--prefit-failed was given, but every named report passed")
    output_payload = {
        "schema": "shot-sweep-hypotheses-v1",
        "frozen_rule": RULE_FILE,
        "script": "tools/shot_sweep_hypotheses.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation": {
            "reading_outcome": 6,
            "reading_statement": (
                "Outcome 6: A check before any fit fails: no fit runs, and the paper "
                "reports the sweep as not run, with the reason."),
            "h1": {"evaluated": False},
            "h2": {"evaluated": False},
            "failed_checks": entries,
        },
    }
    out_p = Path(out).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_p.with_suffix(".tmp")
    tmp_out.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    tmp_out.replace(out_p)
    print(f"Wrote outcome 6 to {out_p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions.json from tools/shot_sweep_predict.py",
    )
    parser.add_argument(
        "--analysis-a",
        action="append",
        default=[],
        help="Analysis A JSON in format <level>=<path>",
    )
    parser.add_argument(
        "--analysis-b",
        action="append",
        default=[],
        help="Analysis B JSON in format <level>=<path>",
    )
    parser.add_argument(
        "--prefit-failed",
        action="store_true",
        default=False,
        help="Reading outcome 6: a check before any fit failed (needs --failure-report)",
    )
    parser.add_argument(
        "--failure-report",
        type=Path,
        action="append",
        default=[],
        help="JSON report of the failed pre-fit check (repeatable); used with --prefit-failed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for hypotheses results JSON",
    )
    parser.add_argument(
        "--dataset-seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Dataset seeds (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.prefit_failed:
        write_prefit_failure(args.failure_report, args.out)
        return
    if args.predictions is None or not args.analysis_a or not args.analysis_b:
        parser.error("--predictions, --analysis-a, and --analysis-b are required "
                     "unless --prefit-failed is given")

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    levels_a = parse_level_args(args.analysis_a)
    levels_b = parse_level_args(args.analysis_b)

    missing = sorted(set(CANONICAL_LEVELS) - set(levels_a.keys()))
    extra = sorted(set(levels_a.keys()) - set(CANONICAL_LEVELS))
    if missing or extra:
        print(f"ERROR: the rule needs exactly the seven levels {list(CANONICAL_LEVELS)}; "
              f"missing {missing}, unexpected {extra}", file=sys.stderr)
        sys.exit(1)

    if set(levels_a.keys()) != set(levels_b.keys()):
        print(
            f"ERROR: Level mismatch between analysis A ({sorted(levels_a.keys())}) and analysis B ({sorted(levels_b.keys())})",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_seeds = tuple(args.dataset_seeds)
    obs_a: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for lvl, p in levels_a.items():
        obs_a[lvl] = extract_analysis_a(p, dataset_seeds=dataset_seeds)

    obs_b: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for lvl, p in levels_b.items():
        obs_b[lvl] = extract_analysis_b(p, dataset_seeds=dataset_seeds)

    results_a = evaluate_pipeline(predictions, obs_a, prefit_failed=args.prefit_failed, dataset_seeds=dataset_seeds)
    results_b = evaluate_pipeline(predictions, obs_b, prefit_failed=args.prefit_failed, dataset_seeds=dataset_seeds)

    # Compare results to 1e-12
    diffs = compare_ab_results(results_a, results_b, path="results")
    if diffs:
        print("ERROR: Disagreement between analysis A and analysis B results:", file=sys.stderr)
        for d in diffs:
            print(f"  {d}", file=sys.stderr)
        print("Neither hypothesis is read until the difference is traced.", file=sys.stderr)
        sys.exit(1)

    output_payload = {
        "schema": "shot-sweep-hypotheses-v1",
        "frozen_rule": RULE_FILE,
        "rule_file_sha256": predictions.get("rule_file_sha256"),
        "script": "tools/shot_sweep_hypotheses.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "predictions_file": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
        "inputs": {
            "analysis_a": {lvl: str(p) for lvl, p in levels_a.items()},
            "analysis_b": {lvl: str(p) for lvl, p in levels_b.items()},
            "prefit_failed": args.prefit_failed,
        },
        "agreement": {
            "status": "passed",
            "tolerance": 1e-12,
            "levels_compared": sorted(levels_a.keys()),
        },
        "evaluation": results_a,
    }

    out_p = Path(args.out).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_p.with_suffix(".tmp")
    tmp_out.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    tmp_out.replace(out_p)
    print(f"Wrote hypothesis results to {out_p}")
    print(f"Outcome: {results_a['reading_outcome']} - {results_a['reading_statement']}")


if __name__ == "__main__":
    main()
