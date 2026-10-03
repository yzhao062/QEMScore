#!/usr/bin/env python3
"""Post hoc analysis: seed-averaged headline numbers and shrinkage-referenced shares.

This script computes the 20-learner-seed averaged replacement for the paper's
single-fit headline shares (Finding 1) across the six primary rows:
dataset seeds 101, 211, 307 x families TFI, Heisenberg (rung R0).

Status: post hoc, descriptive (the paper's results were already known).

Quantities computed:
- Seed-averaged mean C, mean F, A, R, D = C - F, D/C, S = (A - C)/(A - F)
  using the exact two-stage bootstrap estimator and call sequence of script A
  (tools/descriptor_ladder_analysis.py), asserted to agree with analysis-a.json to
  1e-12 (bit-identical in the pinned environment; newer NumPy releases can move the
  last bits of a bootstrap standard deviation).
- Shrinkage-referenced share S_shr = (Sh - C)/(Sh - F), where Sh is the campaign's
  shrinkage control (predicting family mean label from training at zero circuit cost).
  Sh macro MAEs are asserted to match Table 17 of the paper (0.09674 / 0.21793 for s101,
  0.08866 / 0.22150 for s211, 0.10272 / 0.21445 for s307). Sh is seedless and resampled
  over circuits identically to arm A in each draw.
- Single-fit S per row from the frozen campaign record (campaign-archive-v1/tables.json).

Usage:
  PYTHONPATH=. python tools/headline_seed_averaged.py [options]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tools"))
sys.path.insert(0, str(_REPO))

import descriptor_ladder_analysis as dla  # noqa: E402

def _assert_close(actual, expected, path: str, tol: float = 1e-12) -> None:
    """Recursively compare nested values; floats must agree to an absolute tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), f"keys differ at {path}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}", tol)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"length differs at {path}"
        for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
            _assert_close(a_item, e_item, f"{path}[{idx}]", tol)
    elif isinstance(expected, float) and not isinstance(actual, bool) and isinstance(actual, (int, float)):
        both_nan = math.isnan(actual) and math.isnan(expected)
        assert both_nan or abs(actual - expected) <= tol, f"{path}: {actual!r} vs {expected!r}"
    else:
        assert actual == expected, f"{path}: {actual!r} vs {expected!r}"


# Default inputs, relative to the repository root: the extracted
# descriptor-information-v1 release asset and the campaign-archive-v1 release.
DEFAULT_ASSET_DIR = _REPO / "descriptor-information-v1"
DEFAULT_FITS_DIR = DEFAULT_ASSET_DIR / "fits"
DEFAULT_CACHE_DIR = DEFAULT_ASSET_DIR / "cache"
DEFAULT_ANALYSIS_A = _REPO / "artifacts/descriptor-information/analysis-a.json"
DEFAULT_ARCHIVE_DIR = _REPO / "campaign-archive-v1"
DEFAULT_OUT = _REPO / "artifacts/descriptor-information/posthoc-headline.json"

# Table 17 printed Sh macro MAEs (5 decimal places)
TABLE_17_SH_MACRO_MAE: dict[tuple[str, str], float] = {
    ("shipped-s101-n640", "heisenberg"): 0.09674,
    ("shipped-s101-n640", "tfi"): 0.21793,
    ("shipped-s211-n640", "heisenberg"): 0.08866,
    ("shipped-s211-n640", "tfi"): 0.22150,
    ("shipped-s307-n640", "heisenberg"): 0.10272,
    ("shipped-s307-n640", "tfi"): 0.21445,
}

PRIMARY_ROWS: list[tuple[str, str, str]] = [
    ("shipped-s101-n640", "tfi", "s101 TFI"),
    ("shipped-s101-n640", "heisenberg", "s101 Heisenberg"),
    ("shipped-s211-n640", "tfi", "s211 TFI"),
    ("shipped-s211-n640", "heisenberg", "s211 Heisenberg"),
    ("shipped-s307-n640", "tfi", "s307 TFI"),
    ("shipped-s307-n640", "heisenberg", "s307 Heisenberg"),
]


def load_archive_shares(archive_dir: Path) -> dict[tuple[str, str], dict]:
    tables_path = archive_dir / "tables.json"
    if not tables_path.is_file():
        raise FileNotFoundError(f"Missing campaign archive tables: {tables_path}")
    with open(tables_path, encoding="utf-8") as handle:
        tables = json.load(handle)
    shares = {}
    for entry in tables.get("shares", []):
        setting = str(entry.get("setting", ""))
        family = str(entry.get("family", ""))
        size = entry.get("size")
        if setting.startswith("shipped-s") and size == 640:
            shares[(setting, family)] = entry
    return shares


def run_headline_analysis(
    fits_dir: Path,
    cache_dir: Path,
    analysis_a_path: Path,
    archive_dir: Path,
    out_path: Path,
    draws: int = dla.DEFAULT_DRAWS,
    bootstrap_seed: int = dla.RULE_SEED,
) -> dict:
    # 1. Load inputs and reference analysis-a.json
    print(f"Loading fits from {fits_dir}...")
    store = dla.FitStore([fits_dir], False)
    caches, _ = dla.discover_caches([cache_dir])

    print(f"Loading reference analysis A from {analysis_a_path}...")
    with open(analysis_a_path, encoding="utf-8") as handle:
        analysis_a = json.load(handle)

    archive_shares = load_archive_shares(archive_dir)

    results: dict[str, dict] = {}
    table_rows: list[dict] = []

    print("\nComputing seed-averaged R0 estimands and shrinkage controls...")
    for setting, family, label in PRIMARY_ROWS:
        row_key = f"{setting}/{family}"
        cache_data = dla.load_cache(caches[setting])
        row = dla.Row(cache_data, family, row_key)
        est = dla.RowEstimator(row, store, draws, bootstrap_seed, 20)
        arms = dla.arms_at(store, dla.PARTS["A"], setting, "R0")

        # Compute D = C - F
        d_entry, d_parts = est.difference(arms["C"], arms["F"])
        assert d_parts is not None, f"D not estimable for {row_key}"
        seeds, n = d_parts["seeds"], d_parts["n"]
        assert n == 20, f"Expected 20 learner seeds, got {n} for {row_key}"

        # Compute D/C
        d_over_c = est.ratio(d_parts)

        # Compute S = (A - C)/(A - F)
        s_entry = dla._s_entry(est, store, setting, "R0", arms, d_parts)

        # Compute arm means
        mean_c, _ = est.mean_arm(arms["C"])
        mean_f, _ = est.mean_arm(arms["F"])
        mean_a = est.mean_single(store.seedless(setting, "R0", "A"), n)
        mean_r = est.mean_single(None, n, raw=True)

        # Assert agreement with analysis-a.json R0 values (to 1e-12)
        target_row = analysis_a["parts"]["A"]["rows"][row_key]["rungs"]["R0"]
        _assert_close(d_entry, target_row["D"], f"{row_key}.D")
        _assert_close(s_entry, target_row["S"], f"{row_key}.S")
        _assert_close(d_over_c, target_row["D_over_C"], f"{row_key}.D_over_C")
        _assert_close(mean_c, target_row["means"]["C"], f"{row_key}.mean_C")
        _assert_close(mean_f, target_row["means"]["F"], f"{row_key}.mean_F")
        _assert_close(mean_a, target_row["means"]["A"], f"{row_key}.mean_A")
        _assert_close(mean_r, target_row["means"]["R"], f"{row_key}.mean_R")

        # 2. Add Shrinkage Control Sh
        roster_path = archive_dir / "rosters" / setting / "results.json"
        with open(roster_path, encoding="utf-8") as handle:
            roster_data = json.load(handle)

        # Verify item alignment
        roster_items = [it["item_id"] for it in roster_data["test_items"]]
        cache_items = [it["item_id"] for it in cache_data["test"]]
        assert roster_items == cache_items, f"Item alignment mismatch in {setting}"

        sh_preds = np.asarray(
            roster_data["methods"]["shrinkage"]["predictions"], dtype=float
        )
        sh_errors = np.abs(sh_preds[row.index] - row.ideal)[None, :]
        sh_point = dla.point_macro(sh_errors[0], row.members, row.all_cells)

        # Assert Sh macro MAE matches Table 17
        expected_sh_mae = TABLE_17_SH_MACRO_MAE[(setting, family)]
        assert round(sh_point, 5) == expected_sh_mae, (
            f"Sh macro MAE mismatch for {row_key}: calculated {sh_point:.5f}, "
            f"expected {expected_sh_mae:.5f}"
        )

        # Sh bootstrap draws (seedless: circuit resamples only, same draws as A)
        _, circuit_counts = dla.bootstrap_draws(
            n, row.n_circuits, draws, bootstrap_seed
        )
        sh_macro = dla.macro_under_weights(
            sh_errors, row.item_circuit, row.members, row.all_cells, circuit_counts
        )
        sh_draws = sh_macro[:, 0]
        mean_sh = dla.interval_entry(sh_draws, sh_point, None)
        mean_sh["draws_seed_count"] = int(n)

        # c_draws and f_draws
        seed_counts = est.seed_counts(n)
        rec_c = est._records(arms["C"], seeds)
        rec_f = est._records(arms["F"], seeds)
        c_draws = (
            np.sum(
                seed_counts * est.macro(rec_c, "test", n, row.all_cells), axis=1
            )
            / n
        )
        f_draws = (
            np.sum(
                seed_counts * est.macro(rec_f, "test", n, row.all_cells), axis=1
            )
            / n
        )

        # Compute S_shr = (Sh - C)/(Sh - F)
        denom_shr = sh_draws - f_draws
        pt_denom_shr = sh_point - d_parts["second_point"]
        assert pt_denom_shr > 0.0, f"Sh - F nonpositive at point in {row_key}"
        nonpos_shr = int(np.sum(denom_shr <= 0.0))
        assert nonpos_shr == 0, f"Sh - F nonpositive in {nonpos_shr} draws in {row_key}"

        draws_shr = (sh_draws - c_draws) / denom_shr
        point_shr = (sh_point - d_parts["first_point"]) / pt_denom_shr
        s_shr = dla.interval_entry(draws_shr, point_shr, n, seeds)
        s_shr["definition"] = (
            "(Sh - mean C) / (Sh - mean F), ratio of means inside each draw"
        )

        # 3. Single-fit S from campaign archive
        arch_entry = archive_shares[(setting, family)]
        single_fit_s = {
            "point": float(arch_entry["S"]),
            "interval": {
                "lower": float(arch_entry["S_lower"]),
                "upper": float(arch_entry["S_upper"]),
            },
            "excludes_zero_above": bool(arch_entry.get("S_lower", 0.0) > 0.0),
            "excludes_zero_below": bool(arch_entry.get("S_upper", 0.0) < 0.0),
            "A": float(arch_entry["A"]),
            "C": float(arch_entry["C"]),
            "F": float(arch_entry["F"]),
            "R": float(arch_entry["R"]),
            "D": float(arch_entry["D"]),
        }

        # Store complete record for JSON artifact
        row_record = {
            "setting": setting,
            "family": family,
            "label": label,
            "row_key": row_key,
            "single_fit_S": single_fit_s,
            "seed_mean_S": s_entry,
            "S_shr": s_shr,
            "mean_C": mean_c,
            "mean_F": mean_f,
            "R": mean_r,
            "A": mean_a,
            "Sh": mean_sh,
            "D": d_entry,
            "D_over_C": d_over_c,
        }
        results[row_key] = row_record

        # Collect formatted values for the table
        table_rows.append(
            {
                "row": row_key,
                "single_fit_s_point": single_fit_s["point"],
                "single_fit_s_str": f"{single_fit_s['point']:.5f}",
                "single_fit_s_full": (
                    f"{single_fit_s['point']:.5f} "
                    f"[{single_fit_s['interval']['lower']:.5f}, {single_fit_s['interval']['upper']:.5f}]"
                ),
                "seed_mean_s_str": (
                    f"{s_entry['point']:.5f} "
                    f"[{s_entry['interval']['lower']:.5f}, {s_entry['interval']['upper']:.5f}]"
                ),
                "s_shr_str": (
                    f"{s_shr['point']:.5f} "
                    f"[{s_shr['interval']['lower']:.5f}, {s_shr['interval']['upper']:.5f}]"
                ),
                "mean_c_str": f"{mean_c['point']:.5f}",
                "mean_f_str": f"{mean_f['point']:.5f}",
                "r_str": f"{mean_r['point']:.5f}",
                "a_str": f"{mean_a['point']:.5f}",
                "sh_str": f"{sh_point:.5f}",
                "d_str": (
                    f"{d_entry['point']:+.5f} "
                    f"[{d_entry['interval']['lower']:+.5f}, {d_entry['interval']['upper']:+.5f}]"
                ),
                "d_over_c_str": (
                    f"{d_over_c['point']:+.5f} "
                    f"[{d_over_c['interval']['lower']:+.5f}, {d_over_c['interval']['upper']:+.5f}]"
                ),
            }
        )

        print(f"  [PASS] {row_key}:")
        print(f"         single-fit S = {single_fit_s['point']:.5f}")
        print(
            f"         seed-mean S  = {s_entry['point']:.5f} "
            f"[{s_entry['interval']['lower']:.5f}, {s_entry['interval']['upper']:.5f}]"
        )
        print(
            f"         S_shr        = {s_shr['point']:.5f} "
            f"[{s_shr['interval']['lower']:.5f}, {s_shr['interval']['upper']:.5f}]"
        )
        print(f"         Sh macro MAE = {sh_point:.5f} (expected {expected_sh_mae:.5f})")

    # 4. Construct output JSON artifact
    out_payload = {
        "schema": "descriptor-information-posthoc-headline-v1",
        "status": "post hoc, descriptive",
        "note": "Replaces single-fit headline Finding 1 with 20-learner-seed averaged estimands and shrinkage control",
        "analysis_script": "tools/headline_seed_averaged.py",
        "analysis_script_sha256": dla.sha256_file(Path(__file__).resolve()),
        "bootstrap": {
            "seed": bootstrap_seed,
            "draws": draws,
            "nominal_seeds": 20,
            "percentiles": [2.5, 97.5],
            "method": "two-stage percentile bootstrap (learner seeds then circuits)",
        },
        "table_columns": [
            "row",
            "single-fit S",
            "seed-mean S [CI]",
            "S_shr [CI]",
            "mean C",
            "mean F",
            "R",
            "A",
            "Sh",
            "D [CI]",
            "D/C [CI]",
        ],
        "table_rows": table_rows,
        "rows": results,
    }

    dla._write_json(out_path, out_payload)
    print(f"\nWrote post hoc headline JSON to {out_path}")

    # Print markdown table
    print("\n" + format_markdown_table(table_rows))
    return out_payload


def format_markdown_table(rows: list[dict]) -> str:
    lines = [
        "| row | single-fit S | seed-mean S [CI] | S_shr [CI] | mean C | mean F | R | A | Sh | D [CI] | D/C [CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['row']} | {r['single_fit_s_str']} | {r['seed_mean_s_str']} | "
            f"{r['s_shr_str']} | {r['mean_c_str']} | {r['mean_f_str']} | "
            f"{r['r_str']} | {r['a_str']} | {r['sh_str']} | {r['d_str']} | {r['d_over_c_str']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-averaged headline numbers and shrinkage-referenced shares."
    )
    parser.add_argument(
        "--fits",
        type=Path,
        default=DEFAULT_FITS_DIR,
        help="Directory containing release asset fit files (.json + .npz)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing release asset cache files (.pkl or .json.gz)",
    )
    parser.add_argument(
        "--analysis-a",
        type=Path,
        default=DEFAULT_ANALYSIS_A,
        help="Path to analysis-a.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Path to campaign-archive-v1 directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path for posthoc-headline.json",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=dla.DEFAULT_DRAWS,
        help="Bootstrap draws (default: 10,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=dla.RULE_SEED,
        help="Bootstrap seed (default: 20261002)",
    )
    args = parser.parse_args()

    run_headline_analysis(
        fits_dir=args.fits,
        cache_dir=args.cache,
        analysis_a_path=args.analysis_a,
        archive_dir=args.archive,
        out_path=args.out,
        draws=args.draws,
        bootstrap_seed=args.seed,
    )


if __name__ == "__main__":
    main()
