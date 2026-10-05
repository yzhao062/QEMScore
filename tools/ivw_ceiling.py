#!/usr/bin/env python3
"""Post hoc analysis: inverse-variance weighting (IVW) heuristic ceiling on remaining-error reduction D/C.

Post hoc, heuristic analysis:
Computes the heuristic inverse-variance weighting "ceiling" on the remaining-error
reduction D/C = 1 - MAE(combination) / MAE(reference) for each Part A cell (row x rung,
48 total: seeds 101, 211, 307; families TFI and Heisenberg; rungs R0, N1..N4, R3, R4, R5)
and each fit set (original 20-seed fits and strength-indicator fits).

Model and assumptions:
Under the heuristic assumption that errors of the descriptor model (Chat or strongest reference)
and the linearly calibrated noisy quantum measurement (r) are approximately independent and
unbiased, with mean absolute error (MAE) acting as the error scale, the optimal linear combination
has per-cell expected error scale:
    c_comb = (c_ref^-2 + c_r^-2)^(-1/2)
The resulting heuristic ceiling on remaining-error reduction is:
    ceiling(D/C) = 1 - macro(c_comb) / macro(c_ref)

Compares observed D/C from analysis-a.json with the IVW ceiling:
- Spearman rank correlation over the 48 Part A cells for each fit set.
- Ratios observed / ceiling where ceiling > 0.05.
- Cells where observed D/C exceeds ceiling by > 0.02.
- Evaluates reproduction of reviewer-reported numbers:
  (1) Spearman ~0.97 over 48 original cells;
  (2) For strength-indicator fits, TFI at 0.87 to 1.00 of ceiling from N2 to R5.

Outputs: artifacts/descriptor-information/posthoc-ivw-ceiling.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[1]

SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
SEVERITIES = ("L1", "L3")
OBSERVABLES = ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
RUNGS_PER_FAMILY = {
    "tfi": ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R4", "R5"),
    "heisenberg": ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R4", "R5"),
}

ASSUMPTION_STATEMENT = (
    "Heuristic inverse-variance weighting assumption: errors of the descriptor model "
    "and the calibrated quantum measurement are assumed independent and roughly unbiased, "
    "with MAE serving as the characteristic error scale sigma. The combined error scale is "
    "c_comb = (c_ref^-2 + c_r^-2)^(-1/2), giving ceiling(D/C) = 1 - macro(c_comb) / macro(c_ref)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_ivw_combination(c_ref: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """Compute heuristic IVW combination error scale: (c_ref^-2 + c_r^-2)^(-1/2)."""
    with np.errstate(divide="ignore"):
        inv_var = (c_ref ** -2.0) + (c_r ** -2.0)
        return inv_var ** -0.5


def run_ivw_ceiling_analysis(
    analysis_a_path: Path,
    analysis_a_strength_path: Path,
    floor_path: Path,
    strongest_ref_path: Path,
    out_json_path: Path,
    dataset_seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Execute the full IVW ceiling analysis for original and strength-indicator fits."""
    analysis_a = json.loads(analysis_a_path.read_text(encoding="utf-8"))["parts"]["A"]
    analysis_a_str = json.loads(analysis_a_strength_path.read_text(encoding="utf-8"))["parts"]["A"]
    floor_data = json.loads(floor_path.read_text(encoding="utf-8"))["per_cell_floor"]

    strongest_ref_data = None
    if strongest_ref_path.exists():
        strongest_ref_data = json.loads(strongest_ref_path.read_text(encoding="utf-8"))["cells"]

    fit_sets = {
        "original": analysis_a,
        "strength-indicator": analysis_a_str,
    }

    out_fit_sets: dict[str, Any] = {}

    for set_name, part_a in fit_sets.items():
        rows = part_a["rows"]
        cells_dict = {}
        obs_dcs = []
        ceil_dcs_from_c = []
        cell_meta_list = []

        for row_key, row_val in rows.items():
            dataset_seed = row_val["dataset_seed"]
            if dataset_seeds is not None and dataset_seed not in dataset_seeds:
                continue
            fam = row_val["family"]

            for rung_name, rung_val in row_val["rungs"].items():
                cell_key = f"{row_key}/{rung_name}"
                obs_dc = rung_val["D_over_C"]["point"]
                mean_c_reported = rung_val["means"]["C"]["point"]
                mean_f_reported = rung_val["means"]["F"]["point"]

                cell_descs = rung_val["D_by_cell_point_descriptive"]

                c_c_list = []
                c_r_list = []
                rule_cells_detail = {}

                for s, o in CELLS_DEF:
                    desc_key = f"depolarizing_readout/{s}/{o}"
                    c_c = float(cell_descs[desc_key]["point_C"])
                    floor_key = f"{row_key}/{s}/{o}"
                    c_r = float(floor_data[floor_key]["floor_mae"])

                    c_c_list.append(c_c)
                    c_r_list.append(c_r)

                    rule_cells_detail[f"{s}/{o}"] = {
                        "severity": s,
                        "observable": o,
                        "c_C": c_c,
                        "c_r": c_r,
                    }

                c_c_arr = np.array(c_c_list, dtype=float)
                c_r_arr = np.array(c_r_list, dtype=float)

                # Verify mean_C matches average of c_C
                macro_c = float(np.mean(c_c_arr))
                diff_c = abs(macro_c - mean_c_reported)
                assert diff_c < 1e-12, f"{cell_key}: macro c_C {macro_c} != mean_C {mean_c_reported}"

                c_comb_c = compute_ivw_combination(c_c_arr, c_r_arr)
                macro_comb_c = float(np.mean(c_comb_c))
                ceil_dc_c = 1.0 - (macro_comb_c / macro_c)

                # If strongest reference data exists, also compute ceiling from reference
                ceil_dc_ref = None
                ceil_dc_ref_norm_c = None
                macro_ref = None
                macro_comb_ref = None
                c_ref_arr = None

                if strongest_ref_data is not None and cell_key in strongest_ref_data:
                    s_ref_entry = strongest_ref_data[cell_key]
                    c_ref_list = []
                    for s, o in CELLS_DEF:
                        rc_k = f"{s}/{o}"
                        c_ref_val = float(s_ref_entry["rule_cells"][rc_k]["ref_test_mae"])
                        c_ref_list.append(c_ref_val)
                        rule_cells_detail[rc_k]["c_ref"] = c_ref_val
                    c_ref_arr = np.array(c_ref_list, dtype=float)
                    c_comb_ref = compute_ivw_combination(c_ref_arr, c_r_arr)
                    macro_ref = float(np.mean(c_ref_arr))
                    macro_comb_ref = float(np.mean(c_comb_ref))
                    ceil_dc_ref = 1.0 - (macro_comb_ref / macro_ref)
                    ceil_dc_ref_norm_c = 1.0 - (macro_comb_ref / macro_c)

                for idx, (s, o) in enumerate(CELLS_DEF):
                    rc_k = f"{s}/{o}"
                    rule_cells_detail[rc_k]["c_comb_C"] = float(c_comb_c[idx])
                    if c_ref_arr is not None:
                        rule_cells_detail[rc_k]["c_comb_ref"] = float(c_comb_ref[idx])

                diff_obs_ceil = obs_dc - ceil_dc_c
                ratio_obs_ceil = (obs_dc / ceil_dc_c) if ceil_dc_c > 0.05 else None

                cell_record = {
                    "row": row_key,
                    "rung": rung_name,
                    "dataset_seed": dataset_seed,
                    "family": fam,
                    "mean_C": mean_c_reported,
                    "mean_F": mean_f_reported,
                    "observed_D_over_C": obs_dc,
                    "macro_c_comb": macro_comb_c,
                    "ceiling_D_over_C_from_C": ceil_dc_c,
                    "ceiling_D_over_C_from_ref": ceil_dc_ref,
                    "ceiling_D_over_C_from_ref_norm_C": ceil_dc_ref_norm_c,
                    "observed_over_ceiling_C": ratio_obs_ceil,
                    "diff_observed_minus_ceiling_C": diff_obs_ceil,
                    "exceeds_ceiling_by_more_than_0_02": bool(diff_obs_ceil > 0.02),
                    "rule_cells": rule_cells_detail,
                }
                cells_dict[cell_key] = cell_record
                obs_dcs.append(obs_dc)
                ceil_dcs_from_c.append(ceil_dc_c)
                cell_meta_list.append({
                    "row": row_key,
                    "rung": rung_name,
                    "family": fam,
                    "dataset_seed": dataset_seed,
                    "observed_D_over_C": obs_dc,
                    "ceiling_D_over_C": ceil_dc_c,
                    "diff": diff_obs_ceil,
                    "ratio": ratio_obs_ceil,
                })

        assert len(obs_dcs) == 48, f"Expected 48 cells, got {len(obs_dcs)}"
        corr, pval = spearmanr(obs_dcs, ceil_dcs_from_c)

        cells_exceeding_002 = [
            c for c in cell_meta_list if c["diff"] > 0.02
        ]
        ratios_ceiling_over_005 = [
            c["ratio"] for c in cell_meta_list if c["ratio"] is not None
        ]

        # Reviewer checks
        tfi_n2_r5_ratios = [
            c["ratio"] for c in cell_meta_list
            if c["family"] == "tfi" and c["rung"] in ("N2", "N3", "N4", "R3-TFI", "R4", "R5")
            and c["ratio"] is not None
        ]

        summary = {
            "n_cells": len(obs_dcs),
            "spearman_correlation": float(corr),
            "spearman_pvalue": float(pval),
            "cells_exceeding_ceiling_by_more_than_0_02": cells_exceeding_002,
            "n_cells_ceiling_over_0_05": len(ratios_ceiling_over_005),
            "ratio_obs_over_ceiling_summary": {
                "min": float(min(ratios_ceiling_over_005)),
                "max": float(max(ratios_ceiling_over_005)),
                "mean": float(np.mean(ratios_ceiling_over_005)),
            },
            "tfi_n2_r5_ratios": {
                "min": float(min(tfi_n2_r5_ratios)) if tfi_n2_r5_ratios else None,
                "max": float(max(tfi_n2_r5_ratios)) if tfi_n2_r5_ratios else None,
                "values": tfi_n2_r5_ratios,
            },
        }

        out_fit_sets[set_name] = {
            "summary": summary,
            "cells": cells_dict,
        }

    # Reviewer verification statements
    orig_spearman = out_fit_sets["original"]["summary"]["spearman_correlation"]
    reproduced_orig_spearman = (round(orig_spearman, 2) == 0.97)

    str_tfi_min = out_fit_sets["strength-indicator"]["summary"]["tfi_n2_r5_ratios"]["min"]
    str_tfi_max = out_fit_sets["strength-indicator"]["summary"]["tfi_n2_r5_ratios"]["max"]
    reproduced_str_tfi = (
        round(str_tfi_min, 2) == 0.87 and round(str_tfi_max, 2) == 1.00
    )

    reviewer_verification = {
        "reviewer_spearman_original_reported": 0.97,
        "reviewer_spearman_original_reproduced": reproduced_orig_spearman,
        "reproduced_spearman_original_exact": orig_spearman,
        "reviewer_tfi_strength_range_reported": [0.87, 1.00],
        "reviewer_tfi_strength_range_reproduced": reproduced_str_tfi,
        "reproduced_tfi_strength_range_exact": [str_tfi_min, str_tfi_max],
    }

    out_json = {
        "schema": "qem-descriptor-information-posthoc-ivw-ceiling-v1",
        "status": "post hoc, heuristic; paper results and analysis-a were known",
        "analysis_script": "tools/ivw_ceiling.py",
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "assumption": ASSUMPTION_STATEMENT,
        "reviewer_verification": reviewer_verification,
        "fit_sets": out_fit_sets,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_json, indent=1), encoding="utf-8")
    print(f"Wrote IVW ceiling results to {out_json_path}")
    print(f"Original Spearman correlation: {orig_spearman:.4f} (reproduced ~0.97: {reproduced_orig_spearman})")
    print(f"Strength TFI N2..R5 ratio range: [{str_tfi_min:.4f}, {str_tfi_max:.4f}] (reproduced 0.87..1.00: {reproduced_str_tfi})")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-a",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "analysis-a.json",
        help="Path to original analysis-a.json",
    )
    parser.add_argument(
        "--analysis-a-strength",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "strength-indicator" / "analysis-a.json",
        help="Path to strength indicator analysis-a.json",
    )
    parser.add_argument(
        "--floor",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "posthoc-measurement-floor.json",
        help="Path to posthoc-measurement-floor.json",
    )
    parser.add_argument(
        "--strongest-reference",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "posthoc-strongest-reference.json",
        help="Path to posthoc-strongest-reference.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO / "artifacts" / "descriptor-information" / "posthoc-ivw-ceiling.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dataset-seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Dataset seeds (default: %(default)s)",
    )
    args = parser.parse_args()

    run_ivw_ceiling_analysis(
        analysis_a_path=args.analysis_a,
        analysis_a_strength_path=args.analysis_a_strength,
        floor_path=args.floor,
        strongest_ref_path=args.strongest_reference,
        out_json_path=args.out,
        dataset_seeds=tuple(args.dataset_seeds),
    )


if __name__ == "__main__":
    main()
