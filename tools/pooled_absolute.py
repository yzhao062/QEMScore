#!/usr/bin/env python3
"""Pooled random effects and absolute-scale margin analysis.

This module pools results over six dataset seeds using DerSimonian-Laird
variance and Hartung-Knapp-Sidik-Jonkman confidence intervals. It also computes
calibrated raw estimate margins and shot-noise scales on the absolute scale.

Governed by docs/frozen-rules/2026-10-06-round9-follow-ups.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import scipy.stats

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

import measurement_floor as mf  # noqa: E402

FROZEN_RULE_PATH = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
FIRST_SEEDS = (101, 211, 307)
NEW_SEEDS = (401, 503, 607)
ALL_SEEDS = (101, 211, 307, 401, 503, 607)
FAMILIES = ("tfi", "heisenberg")

FIT_SETS = ("original 2,048", "strong 2,048", "strong exact")

RUNGS_PER_FAMILY = {
    "tfi": ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R4", "R5"),
    "heisenberg": ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R4", "R5"),
}

DEFAULT_ANALYSIS_PATHS = {
    # Every pooled fit reads the noise-strength indicator. The first seeds' original
    # candidates therefore come from the strength-indicator rerun, which matches
    # the new seeds' original-candidate specification.
    "original 2,048": {
        "first": Path("artifacts/descriptor-information/strength-indicator/analysis-a.json"),
        "new": Path("artifacts/descriptor-information/round8/fresh/orig/2048/analysis-a.json"),
    },
    "strong 2,048": {
        "first": Path("artifacts/descriptor-information/strong-learners/analysis-a.json"),
        "new": Path("artifacts/descriptor-information/round8/fresh/strong/2048/analysis-a.json"),
    },
    "strong exact": {
        "first": Path("artifacts/descriptor-information/round8/follow/B/exact/analysis-a.json"),
        "new": Path("artifacts/descriptor-information/round8/fresh/strong/exact/analysis-a.json"),
    },
}

DEFAULT_RCAL_DATA_FIRST = Path("/Users/yzhao062/qemscore-r8/assets/shot-sweep-v1/levels/shots-2048")
DEFAULT_RCAL_DATA_NEW = Path("/Users/yzhao062/qemscore-r8/fresh/levels/shots-2048")
DEFAULT_EXACT_DATA_FIRST = Path("/Users/yzhao062/qemscore-r8/assets/shot-sweep-v1/levels/shots-exact")
DEFAULT_EXACT_DATA_NEW = Path("/Users/yzhao062/qemscore-r8/fresh/levels/shots-exact")
DEFAULT_OUT = Path("artifacts/descriptor-information/round9/pooled-absolute.json")
DEFAULT_OUT_MD = Path("artifacts/descriptor-information/round9/pooled-absolute.md")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hksj_meta_analysis(y: Sequence[float], se: Sequence[float]) -> dict[str, float]:
    """Compute DerSimonian-Laird and Hartung-Knapp-Sidik-Jonkman pooled estimate.

    y is an array of point estimates.
    se is an array of bootstrap standard deviations.
    """
    y_arr = np.asarray(y, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    k = len(y_arr)
    if k < 2:
        raise ValueError("At least two estimates are required for meta-analysis.")

    df = k - 1
    v = se_arr ** 2
    if np.any(v <= 0.0):
        raise ValueError("Standard errors must be positive.")

    w_fe = 1.0 / v
    sum_w_fe = float(np.sum(w_fe))
    theta_fe = float(np.sum(w_fe * y_arr) / sum_w_fe)

    q_stat = float(np.sum(w_fe * (y_arr - theta_fe) ** 2))

    if q_stat > df:
        i2 = float((q_stat - df) / q_stat)
    else:
        i2 = 0.0

    c_dl = sum_w_fe - float(np.sum(w_fe ** 2)) / sum_w_fe
    if c_dl > 0.0 and q_stat > df:
        tau2 = float((q_stat - df) / c_dl)
    else:
        tau2 = 0.0

    w_re = 1.0 / (v + tau2)
    sum_w_re = float(np.sum(w_re))
    theta_re = float(np.sum(w_re * y_arr) / sum_w_re)

    var_hksj = float(np.sum(w_re * (y_arr - theta_re) ** 2) / (df * sum_w_re))
    se_hksj = float(np.sqrt(max(0.0, var_hksj)))

    t_crit = float(scipy.stats.t.ppf(0.975, df=df))
    ci_lower = theta_re - t_crit * se_hksj
    ci_upper = theta_re + t_crit * se_hksj

    return {
        "theta_fe": theta_fe,
        "theta_re": theta_re,
        "q_stat": q_stat,
        "df": df,
        "tau2": tau2,
        "i2": i2,
        "var_hksj": var_hksj,
        "se_hksj": se_hksj,
        "t_crit": t_crit,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def classify_pooled_label(ci_lower: float, ci_upper: float) -> str:
    """Classify pooled difference using the three-way rule."""
    if ci_lower > 0.0:
        return "F beats C"
    if ci_upper < 0.0:
        return "C beats F"
    return "not distinguished"


def classify_panel_unanimity(labels: Sequence[str]) -> str:
    """Classify panel unanimity across seeds."""
    if not labels:
        return "incomplete"
    unique = set(labels)
    if len(unique) == 1:
        return next(iter(unique))
    return "mixed"


def classify_practical_margin(
    ci_lower: float, ci_upper: float, delta: float
) -> str:
    """Classify pooled effect against practical margin delta."""
    if delta <= 0.0:
        raise ValueError("Margin delta must be positive.")
    if ci_lower > delta:
        return "practically relevant gain"
    if ci_upper < -delta:
        return "practically relevant loss"
    if ci_lower > -delta and ci_upper < delta:
        return "practically negligible"
    return "undetermined"


def normalize_seed_label(label: str | None, d_lower: float, d_upper: float) -> str:
    """Normalize label string to standard paper label format."""
    if label in ("measurement_adds", "F beats C"):
        return "F beats C"
    if label in ("measurement_hurts", "C beats F"):
        return "C beats F"
    if label in ("not_distinguished", "not distinguished"):
        return "not distinguished"
    if d_lower > 0.0:
        return "F beats C"
    if d_upper < 0.0:
        return "C beats F"
    return "not distinguished"


def extract_rows_from_analysis(analysis_data: dict) -> dict[tuple[int, str], dict]:
    """Extract rows indexed by seed and family from an analysis dictionary."""
    rows = {}
    part_a = analysis_data.get("parts", {}).get("A", {})
    rows_dict = part_a.get("rows", {})
    for row_key, row_data in rows_dict.items():
        seed = row_data.get("dataset_seed")
        family = row_data.get("family")
        if seed is None or family is None:
            match = re.search(r"s(\d+)", row_key)
            if match and seed is None:
                seed = int(match.group(1))
            for fam in FAMILIES:
                if fam in row_key and family is None:
                    family = fam
        if seed is not None and family is not None:
            rows[(int(seed), str(family))] = row_data
    return rows


def collect_fit_set_data(
    fit_set_name: str,
    first_path: Path,
    new_path: Path,
    expected_first_seeds: Sequence[int] = FIRST_SEEDS,
    expected_new_seeds: Sequence[int] = NEW_SEEDS,
) -> tuple[list[dict], dict[str, list[dict]], dict[str, list[str]]]:
    """Gather per-seed statistics and determine present and absent rungs.

    Fails loudly if any required seed is missing.
    """
    with open(first_path, "r", encoding="utf-8") as f:
        first_data = json.load(f)
    with open(new_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    first_rows = extract_rows_from_analysis(first_data)
    new_rows = extract_rows_from_analysis(new_data)

    all_per_seed: list[dict] = []
    rung_records_by_family: dict[str, list[dict]] = {fam: [] for fam in FAMILIES}
    absent_rungs_by_family: dict[str, list[str]] = {fam: [] for fam in FAMILIES}

    for fam in FAMILIES:
        for seed in expected_first_seeds:
            if (seed, fam) not in first_rows:
                raise ValueError(
                    f"Missing seed {seed} for {fit_set_name} {fam} in {first_path}"
                )
        for seed in expected_new_seeds:
            if (seed, fam) not in new_rows:
                raise ValueError(
                    f"Missing seed {seed} for {fit_set_name} {fam} in {new_path}"
                )

        candidate_rungs = RUNGS_PER_FAMILY[fam]
        for rung in candidate_rungs:
            present_seeds: list[int] = []
            seed_records: list[dict] = []

            for seed in expected_first_seeds:
                rungs_dict = first_rows[(seed, fam)].get("rungs", {})
                if rung in rungs_dict:
                    present_seeds.append(seed)
                    entry = rungs_dict[rung]
                    d_obj = entry["D"]
                    c_obj = entry.get("means", {}).get("C", {})
                    f_obj = entry.get("means", {}).get("F", {})
                    class_obj = entry.get("classification", {})
                    d_lo = float(d_obj["interval"]["lower"])
                    d_hi = float(d_obj["interval"]["upper"])
                    label = normalize_seed_label(
                        class_obj.get("label"), d_lo, d_hi
                    )
                    seed_records.append({
                        "fit_set": fit_set_name,
                        "family": fam,
                        "rung": rung,
                        "dataset_seed": seed,
                        "panel": "first",
                        "D_point": float(d_obj["point"]),
                        "D_draw_sd": float(d_obj["draw_sd"]),
                        "D_interval": {"lower": d_lo, "upper": d_hi},
                        "C_point": float(c_obj.get("point", 0.0)),
                        "F_point": float(f_obj.get("point", 0.0)),
                        "label": label,
                    })

            for seed in expected_new_seeds:
                rungs_dict = new_rows[(seed, fam)].get("rungs", {})
                if rung in rungs_dict:
                    present_seeds.append(seed)
                    entry = rungs_dict[rung]
                    d_obj = entry["D"]
                    c_obj = entry.get("means", {}).get("C", {})
                    f_obj = entry.get("means", {}).get("F", {})
                    class_obj = entry.get("classification", {})
                    d_lo = float(d_obj["interval"]["lower"])
                    d_hi = float(d_obj["interval"]["upper"])
                    label = normalize_seed_label(
                        class_obj.get("label"), d_lo, d_hi
                    )
                    seed_records.append({
                        "fit_set": fit_set_name,
                        "family": fam,
                        "rung": rung,
                        "dataset_seed": seed,
                        "panel": "new",
                        "D_point": float(d_obj["point"]),
                        "D_draw_sd": float(d_obj["draw_sd"]),
                        "D_interval": {"lower": d_lo, "upper": d_hi},
                        "C_point": float(c_obj.get("point", 0.0)),
                        "F_point": float(f_obj.get("point", 0.0)),
                        "label": label,
                    })

            total_expected = len(expected_first_seeds) + len(expected_new_seeds)
            if len(present_seeds) == 0:
                absent_rungs_by_family[fam].append(rung)
            elif len(present_seeds) == total_expected:
                all_per_seed.extend(seed_records)
                rung_records_by_family[fam].append({
                    "fit_set": fit_set_name,
                    "family": fam,
                    "rung": rung,
                    "seed_records": seed_records,
                })
            else:
                missing = [
                    s for s in list(expected_first_seeds) + list(expected_new_seeds)
                    if s not in present_seeds
                ]
                raise ValueError(
                    f"Rung {rung} in {fit_set_name} {fam} is missing seeds {missing}"
                )

    return all_per_seed, rung_records_by_family, absent_rungs_by_family


def compute_rcal_margins_from_datasets(
    datasets_dict: dict[int, dict[str, list[dict]]],
    dataset_seeds: Sequence[int] = ALL_SEEDS,
) -> dict[str, dict]:
    """Compute Rcal macro test errors and practical margins across seeds."""
    _, row_summary = mf.compute_floors(
        datasets_dict, dataset_seeds=dataset_seeds
    )
    per_seed_errors: dict[str, dict[int, float]] = {fam: {} for fam in FAMILIES}
    mean_errors: dict[str, float] = {}
    deltas: dict[str, float] = {}

    for fam in FAMILIES:
        for seed in dataset_seeds:
            key = f"shipped-s{seed}-n640/{fam}"
            if key not in row_summary:
                raise ValueError(f"Missing row summary key {key} in compute_floors")
            per_seed_errors[fam][seed] = float(row_summary[key]["macro_floor_mae"])
        mean_err = float(
            np.mean([per_seed_errors[fam][s] for s in dataset_seeds])
        )
        mean_errors[fam] = mean_err
        deltas[fam] = 0.1 * mean_err

    return {
        "per_seed_errors": per_seed_errors,
        "mean_errors": mean_errors,
        "deltas": deltas,
    }


def compute_shot_noise_scale_from_datasets(
    datasets_dict: dict[int, dict[str, list[dict]]],
    family: str,
    shots: int = 2048,
    split: str = "test",
) -> float:
    """Compute binomial shot noise scale for a family from dataset items."""
    values: list[float] = []
    for seed, splits in datasets_dict.items():
        items = splits.get(split, [])
        if not items:
            items = splits.get("val", []) + splits.get("test", [])
        for item in items:
            if item.get("family") == family:
                values.append(float(item["noisy_expectation"]))
    if not values:
        raise ValueError(
            f"No items found for family {family} in dataset split {split}"
        )
    arr = np.asarray(values, dtype=float)
    mean_1_minus_sq = float(np.mean(1.0 - (arr ** 2)))
    return float(np.sqrt(max(0.0, mean_1_minus_sq) / shots))


def build_pooled_rows(
    rung_records_by_fit_set: dict[str, dict[str, list[dict]]],
    deltas: dict[str, float],
    shot_noise_scales: dict[str, float],
    shot_noise_sources: dict[str, str],
) -> list[dict]:
    """Perform random-effects pooling and margin classification for each rung."""
    pooled_rows: list[dict] = []
    for fit_set in FIT_SETS:
        if fit_set not in rung_records_by_fit_set:
            continue
        for fam in FAMILIES:
            for record in rung_records_by_fit_set[fit_set][fam]:
                rung = record["rung"]
                seeds_data = record["seed_records"]
                d_points = [s["D_point"] for s in seeds_data]
                d_ses = [s["D_draw_sd"] for s in seeds_data]
                meta = hksj_meta_analysis(d_points, d_ses)

                pooled_d = meta["theta_re"]
                ci_lo = meta["ci_lower"]
                ci_hi = meta["ci_upper"]
                pooled_label = classify_pooled_label(ci_lo, ci_hi)

                p1_seeds = [s for s in seeds_data if s["panel"] == "first"]
                p2_seeds = [s for s in seeds_data if s["panel"] == "new"]
                p1_labels = [s["label"] for s in p1_seeds]
                p2_labels = [s["label"] for s in p2_seeds]
                p1_stmt = classify_panel_unanimity(p1_labels)
                p2_stmt = classify_panel_unanimity(p2_labels)
                first_panel_labels = {
                    str(s["dataset_seed"]): s["label"] for s in p1_seeds
                }
                new_panel_labels = {
                    str(s["dataset_seed"]): s["label"] for s in p2_seeds
                }

                delta = deltas[fam]
                verdict = classify_practical_margin(ci_lo, ci_hi, delta)
                mean_c = float(np.mean([s["C_point"] for s in seeds_data]))
                mean_f = float(np.mean([s["F_point"] for s in seeds_data]))

                pooled_rows.append({
                    "fit_set": fit_set,
                    "family": fam,
                    "rung": rung,
                    "mean_C": mean_c,
                    "mean_F": mean_f,
                    "pooled_D": pooled_d,
                    "pooled_D_interval": {"lower": ci_lo, "upper": ci_hi},
                    "se_hksj": meta["se_hksj"],
                    "tau2": meta["tau2"],
                    "I2": meta["i2"],
                    "fixed_effect_D": meta["theta_fe"],
                    "q_stat": meta["q_stat"],
                    "df": meta["df"],
                    "first_panel_statement": p1_stmt,
                    "first_panel_labels": first_panel_labels,
                    "new_panel_statement": p2_stmt,
                    "new_panel_labels": new_panel_labels,
                    "pooled_label": pooled_label,
                    "delta": delta,
                    "practical_verdict": verdict,
                    "shot_noise_scale": shot_noise_scales[fam],
                    "shot_noise_source": shot_noise_sources[fam],
                })
    return pooled_rows


def format_markdown_table(
    pooled_rows: list[dict],
    absent_rungs: dict[str, dict[str, list[str]]] | None = None,
) -> str:
    """Format pooled summary table and absent rungs into Markdown."""
    lines = [
        "| Fit Set | Family | Rung | Mean C | Pooled D [95% HKSJ CI] | tau^2 | I^2 | Panel 1 (101-307) | Panel 2 (401-607) | Pooled Label | delta | Practical Verdict | Shot Noise Scale |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pooled_rows:
        ci = r["pooled_D_interval"]
        d_str = f"{r['pooled_D']:+.5f} [{ci['lower']:+.5f}, {ci['upper']:+.5f}]"
        tau2_str = f"{r['tau2']:.2e}" if r["tau2"] > 0.0 else "0.00e+00"
        i2_str = f"{r['I2'] * 100:.1f}%"
        sn_str = f"{r['shot_noise_scale']:.5f} ({r['shot_noise_source']})"
        lines.append(
            f"| {r['fit_set']} | {r['family']} | {r['rung']} | "
            f"{r['mean_C']:.5f} | {d_str} | {tau2_str} | {i2_str} | "
            f"{r['first_panel_statement']} | {r['new_panel_statement']} | "
            f"{r['pooled_label']} | {r['delta']:.5f} | {r['practical_verdict']} | {sn_str} |"
        )

    if absent_rungs:
        lines.append("\n### Absent Rungs")
        has_absent = False
        for fit_set, fam_dict in absent_rungs.items():
            for fam, rungs in fam_dict.items():
                if rungs:
                    has_absent = True
                    lines.append(f"- {fit_set} / {fam}: {', '.join(rungs)}")
        if not has_absent:
            lines.append("- none")

    return "\n".join(lines)


def load_dataset_bundle(
    first_dir: Path | None,
    new_dir: Path | None,
    first_seeds: Sequence[int] = FIRST_SEEDS,
    new_seeds: Sequence[int] = NEW_SEEDS,
) -> dict[int, dict[str, list[dict]]]:
    """Load items from first and new directory trees."""
    combined: dict[int, dict[str, list[dict]]] = {}
    if first_dir is not None and first_dir.exists():
        d1 = mf.load_datasets(first_dir, dataset_seeds=first_seeds)
        combined.update(d1)
    if new_dir is not None and new_dir.exists():
        d2 = mf.load_datasets(new_dir, dataset_seeds=new_seeds)
        combined.update(d2)
    return combined


def run_pipeline(
    analysis_paths: dict[str, dict[str, Path]],
    frozen_rule_path: Path,
    rcal_data_first: Path | None = None,
    rcal_data_new: Path | None = None,
    exact_data_first: Path | None = None,
    exact_data_new: Path | None = None,
    delta_overrides: dict[str, float] | None = None,
    shot_noise_overrides: dict[str, tuple[float, str]] | None = None,
    out_path: Path | None = None,
    out_md_path: Path | None = None,
) -> dict:
    """Run the complete pooled and absolute-scale margin pipeline."""
    if not frozen_rule_path.exists():
        raise FileNotFoundError(f"Frozen rule file not found: {frozen_rule_path}")

    inputs_meta: dict[str, dict[str, str]] = {}
    for fit_set, p_dict in analysis_paths.items():
        for role, p in p_dict.items():
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {p}")
            key = f"{fit_set}_{role}".replace(" ", "_")
            inputs_meta[key] = {
                "path": str(p),
                "sha256": sha256_file(p),
            }

    all_per_seed: list[dict] = []
    rung_records_by_fit_set: dict[str, dict[str, list[dict]]] = {}
    absent_rungs: dict[str, dict[str, list[str]]] = {}

    for fit_set in FIT_SETS:
        p_first = analysis_paths[fit_set]["first"]
        p_new = analysis_paths[fit_set]["new"]
        per_seed, rung_recs, absent = collect_fit_set_data(
            fit_set, p_first, p_new
        )
        all_per_seed.extend(per_seed)
        rung_records_by_fit_set[fit_set] = rung_recs
        absent_rungs[fit_set] = absent

    deltas: dict[str, float] = {}
    per_seed_rcal_mae: dict[str, dict[str, float]] = {fam: {} for fam in FAMILIES}
    mean_rcal_mae: dict[str, float] = {}

    if delta_overrides:
        deltas = dict(delta_overrides)
        for fam in FAMILIES:
            mean_rcal_mae[fam] = deltas[fam] / 0.1
    else:
        datasets_2048 = load_dataset_bundle(rcal_data_first, rcal_data_new)
        if len(datasets_2048) != len(ALL_SEEDS):
            raise ValueError(
                f"Expected datasets for {len(ALL_SEEDS)} seeds, got {len(datasets_2048)}"
            )
        margin_info = compute_rcal_margins_from_datasets(
            datasets_2048, dataset_seeds=ALL_SEEDS
        )
        deltas = margin_info["deltas"]
        mean_rcal_mae = margin_info["mean_errors"]
        for fam in FAMILIES:
            per_seed_rcal_mae[fam] = {
                str(s): margin_info["per_seed_errors"][fam][s] for s in ALL_SEEDS
            }

    shot_noise_scales: dict[str, float] = {}
    shot_noise_sources: dict[str, str] = {}

    if shot_noise_overrides:
        for fam in FAMILIES:
            scale, src = shot_noise_overrides[fam]
            shot_noise_scales[fam] = scale
            shot_noise_sources[fam] = src
    else:
        exact_datasets = load_dataset_bundle(exact_data_first, exact_data_new)
        if len(exact_datasets) == len(ALL_SEEDS):
            for fam in FAMILIES:
                scale = compute_shot_noise_scale_from_datasets(
                    exact_datasets, fam, shots=2048, split="test"
                )
                shot_noise_scales[fam] = scale
                shot_noise_sources[fam] = "exact"
        else:
            if not deltas and not delta_overrides:
                noisy_datasets = datasets_2048
            else:
                noisy_datasets = load_dataset_bundle(rcal_data_first, rcal_data_new)
            for fam in FAMILIES:
                scale = compute_shot_noise_scale_from_datasets(
                    noisy_datasets, fam, shots=2048, split="test"
                )
                shot_noise_scales[fam] = scale
                shot_noise_sources[fam] = "noisy_r"

    pooled_rows = build_pooled_rows(
        rung_records_by_fit_set, deltas, shot_noise_scales, shot_noise_sources
    )

    md_table = format_markdown_table(pooled_rows, absent_rungs)

    out_payload = {
        "schema": "descriptor-information-pooled-absolute-v1",
        "frozen_rule": FROZEN_RULE_PATH,
        "rule_file_sha256": sha256_file(frozen_rule_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": inputs_meta,
        "margins": {
            "per_seed_rcal_mae": per_seed_rcal_mae,
            "mean_rcal_mae": mean_rcal_mae,
            "delta": deltas,
            "shot_noise_scale": shot_noise_scales,
            "shot_noise_source": shot_noise_sources,
        },
        "absent_rungs": absent_rungs,
        "per_seed_table": all_per_seed,
        "pooled_table": pooled_rows,
        "markdown_table": md_table,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=1)

    if out_md_path is not None:
        out_md_path = Path(out_md_path)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_table + "\n")

    return out_payload


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Pool estimates over six seeds and classify on absolute scale."
    )
    parser.add_argument(
        "--frozen-rule",
        type=Path,
        required=True,
        help="Path to governing frozen rule document",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Optional path for output Markdown table",
    )
    parser.add_argument(
        "--orig-2048-first",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["original 2,048"]["first"],
        help="Analysis JSON for original candidates at 2,048 shots, first seeds",
    )
    parser.add_argument(
        "--orig-2048-new",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["original 2,048"]["new"],
        help="Analysis JSON for original candidates at 2,048 shots, new seeds",
    )
    parser.add_argument(
        "--strong-2048-first",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["strong 2,048"]["first"],
        help="Analysis JSON for strong candidates at 2,048 shots, first seeds",
    )
    parser.add_argument(
        "--strong-2048-new",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["strong 2,048"]["new"],
        help="Analysis JSON for strong candidates at 2,048 shots, new seeds",
    )
    parser.add_argument(
        "--strong-exact-first",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["strong exact"]["first"],
        help="Analysis JSON for strong candidates at exact level, first seeds",
    )
    parser.add_argument(
        "--strong-exact-new",
        type=Path,
        default=DEFAULT_ANALYSIS_PATHS["strong exact"]["new"],
        help="Analysis JSON for strong candidates at exact level, new seeds",
    )
    parser.add_argument(
        "--rcal-data-first",
        type=Path,
        default=DEFAULT_RCAL_DATA_FIRST,
        help="Directory with 2,048-shot items for first seeds",
    )
    parser.add_argument(
        "--rcal-data-new",
        type=Path,
        default=DEFAULT_RCAL_DATA_NEW,
        help="Directory with 2,048-shot items for new seeds",
    )
    parser.add_argument(
        "--exact-data-first",
        type=Path,
        default=DEFAULT_EXACT_DATA_FIRST,
        help="Directory with exact items for first seeds",
    )
    parser.add_argument(
        "--exact-data-new",
        type=Path,
        default=DEFAULT_EXACT_DATA_NEW,
        help="Directory with exact items for new seeds",
    )
    parser.add_argument(
        "--delta-tfi",
        type=float,
        default=None,
        help="Override delta margin for transverse-field Ising",
    )
    parser.add_argument(
        "--delta-heisenberg",
        type=float,
        default=None,
        help="Override delta margin for Heisenberg",
    )
    parser.add_argument(
        "--shot-noise-tfi",
        type=float,
        default=None,
        help="Override shot-noise scale for transverse-field Ising",
    )
    parser.add_argument(
        "--shot-noise-heisenberg",
        type=float,
        default=None,
        help="Override shot-noise scale for Heisenberg",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed markdown table",
    )
    return parser


def check_single_strength_policy(analysis_paths: dict[str, dict[str, Path]]) -> None:
    """Require every pooled input to come from fits that read the strength indicator.

    Each input path must be a fit set of `tools/bayes_oracle_mapping.json`, and the
    committed evidence of released fit records must show the indicator for it.
    """
    repo = Path(__file__).resolve().parents[1]
    mapping = json.loads((repo / "tools" / "bayes_oracle_mapping.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (repo / "tools" / "bayes_oracle_strength_evidence.json").read_text(encoding="utf-8"))["fit_sets"]
    by_path = {entry["path"]: entry["fit_set"] for entry in mapping["fit_sets"]}
    for fit_set, panels in analysis_paths.items():
        for panel, path in panels.items():
            rel = Path(path)
            if rel.is_absolute():
                rel = rel.resolve().relative_to(repo)
            name = by_path.get(rel.as_posix())
            if name is None:
                raise ValueError(f"{fit_set} {panel} input {path} is not a mapped fit set")
            if not evidence[name]["strength_observed"]:
                raise ValueError(
                    f"{fit_set} {panel} input {path} ({name}) omits the strength indicator; "
                    "pooling would mix descriptor policies")


def main(argv: list[str] | None = None) -> int:
    """Execute main entry point for pooled absolute CLI."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if not args.frozen_rule.exists():
        parser.error(f"Frozen rule document not found: {args.frozen_rule}")

    analysis_paths = {
        "original 2,048": {
            "first": args.orig_2048_first,
            "new": args.orig_2048_new,
        },
        "strong 2,048": {
            "first": args.strong_2048_first,
            "new": args.strong_2048_new,
        },
        "strong exact": {
            "first": args.strong_exact_first,
            "new": args.strong_exact_new,
        },
    }

    check_single_strength_policy(analysis_paths)

    delta_overrides = None
    if args.delta_tfi is not None and args.delta_heisenberg is not None:
        delta_overrides = {
            "tfi": args.delta_tfi,
            "heisenberg": args.delta_heisenberg,
        }

    shot_noise_overrides = None
    if args.shot_noise_tfi is not None and args.shot_noise_heisenberg is not None:
        shot_noise_overrides = {
            "tfi": (args.shot_noise_tfi, "override"),
            "heisenberg": (args.shot_noise_heisenberg, "override"),
        }

    payload = run_pipeline(
        analysis_paths=analysis_paths,
        frozen_rule_path=args.frozen_rule,
        rcal_data_first=args.rcal_data_first,
        rcal_data_new=args.rcal_data_new,
        exact_data_first=args.exact_data_first,
        exact_data_new=args.exact_data_new,
        delta_overrides=delta_overrides,
        shot_noise_overrides=shot_noise_overrides,
        out_path=args.out,
        out_md_path=args.out_md,
    )

    if not args.quiet:
        print("\n" + payload["markdown_table"] + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
