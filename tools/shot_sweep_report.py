"""Checks and expectations reported beside the shot-sweep results (rule 2026-10-03-shot-sweep.md).

The rule's sections "Expectations Stated in Advance" and "Checks Reported Beside Results"
list quantities to report after fitting; none enters a decision rule. This script computes
them from the fitted level trees, the predictions file, and each level's analysis A:

- C and A identity: the largest absolute difference between each level's C and A test and
  validation predictions and the 2,048-shot level's, per fit and overall;
- the platform check: the 2,048-shot labels on DeltaAI beside the 2026-10-03 labels, with the
  paired difference in D and D/C that analysis A records under ``comparison_vs_original``;
- Expectation 1: macro c_m per row and level, with the ratio to the 256-shot value;
- Expectation 2: at R0, each transverse-field Ising row's D/C* at the exact level, its "adds"
  threshold, and the observed label;
- Expectation 3: transverse-field Ising N2 labels at every level;
- Expectation 4: R5 labels at every level;
- Expectation 5: M's mean error at R5 (F at R5) beside the row's macro c_r at every level;
- the R0 and R5 series of observed D/C, reported beside H1.

It was written after the sweep's fits, to report what the rule lists; it decides nothing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")
REFERENCE_LEVEL = "2048"
RUNGS = ("R0", "N1", "N2", "R5")
STEM = re.compile(r"^(?P<key>shipped-s\d+-n\d+)__(?P<rung>[^_]+)__(?:k(?P<k>\d+)__)?(?P<arm>[ACFP])$")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def identity(level_dirs: dict[str, Path]) -> dict:
    """Largest |difference| of C and A predictions at each level against the 2,048-shot level."""
    ref_dir = level_dirs[REFERENCE_LEVEL] / "fits"
    stems = sorted(p.stem for p in ref_dir.glob("*.npz")
                   if (m := STEM.match(p.stem)) and m["arm"] in ("C", "A"))
    out: dict = {"fits_compared_per_level": len(stems), "levels": {}}
    overall = 0.0
    for level, root in level_dirs.items():
        if level == REFERENCE_LEVEL:
            continue
        per_level = 0.0
        nonzero = []
        for stem in stems:
            with np.load(ref_dir / f"{stem}.npz") as ref, np.load(root / "fits" / f"{stem}.npz") as cur:
                diff = max(float(np.max(np.abs(cur[role] - ref[role]))) for role in ("test", "validation"))
            per_level = max(per_level, diff)
            if diff != 0.0:
                nonzero.append({"fit": stem, "max_abs_diff": diff})
        out["levels"][level] = {"max_abs_diff": per_level, "nonzero_fits": nonzero}
        overall = max(overall, per_level)
    out["max_abs_diff_all_levels"] = overall
    return out


def _rows(analysis: dict) -> dict:
    return analysis["parts"]["A"]["rows"]


def _point(entry: dict | None) -> float | None:
    return None if not entry or entry.get("status") != "estimated" else float(entry["point"])


def report(root: Path, predictions: dict, reference: dict) -> dict:
    level_dirs = {level: root / f"shots-{level}" for level in LEVELS}
    analyses = {level: json.loads((d / "analysis" / "analysis-a.json").read_text(encoding="utf-8"))
                for level, d in level_dirs.items()}
    rows = list(_rows(analyses[REFERENCE_LEVEL]))
    cells = predictions["rule_cells"]
    classification = predictions["classification_cells"]

    def macro(level: str, row: str, field: str) -> float:
        values = [c[field] for key, c in cells.items()
                  if c["level"] == level and c["row"] == row]
        return float(np.mean(values))

    labels = {level: {row: {rung: _rows(analyses[level])[row]["rungs"][rung]["classification"]["label"]
                            for rung in RUNGS if rung in _rows(analyses[level])[row]["rungs"]}
                      for row in rows} for level in LEVELS}
    dc = {level: {row: {rung: _point(_rows(analyses[level])[row]["rungs"][rung].get("D_over_C"))
                        for rung in RUNGS if rung in _rows(analyses[level])[row]["rungs"]}
                  for row in rows} for level in LEVELS}

    platform = {}
    for row in rows:
        for rung, entry in _rows(analyses[REFERENCE_LEVEL])[row]["rungs"].items():
            comp = entry.get("comparison_vs_original") or {}
            platform[f"{row}/{rung}"] = {
                "label_deltaai": entry["classification"]["label"],
                "label_2026_10_03": _rows(reference)[row]["rungs"][rung]["classification"]["label"],
                "delta_D": _point(comp.get("delta_D")),
                "delta_D_over_C": _point(comp.get("delta_D_over_C")),
            }
    max_platform = max(abs(v["delta_D"]) for v in platform.values() if v["delta_D"] is not None)

    c_m = {row: {level: macro(level, row, "c_m") for level in LEVELS} for row in rows}
    exp1 = {row: {level: {"macro_c_m": c_m[row][level],
                          "ratio_to_256": c_m[row][level] / c_m[row]["256"]}
                  for level in LEVELS} for row in rows}
    exp2 = {}
    for row in rows:
        if not row.endswith("/tfi"):
            continue
        cell = classification[f"exact/{row}/R0"]
        exp2[row] = {"D_over_C_star_exact": cell["D_over_C_star"],
                     "adds_threshold": cell["adds_threshold"],
                     "threshold_reached": cell["D_over_C_star"] >= cell["adds_threshold"],
                     "observed_label_exact": labels["exact"][row]["R0"]}
    exp3 = {row: {level: labels[level][row]["N2"] for level in LEVELS}
            for row in rows if row.endswith("/tfi")}
    exp4 = {row: {level: labels[level][row]["R5"] for level in LEVELS} for row in rows}
    exp5 = {}
    for row in rows:
        exp5[row] = {}
        for level in LEVELS:
            means = _rows(analyses[level])[row]["rungs"]["R5"]["means"]
            m = _point(means.get("F"))
            c_r = macro(level, row, "c_r")
            exp5[row][level] = {"M_at_R5": m, "macro_c_r": c_r, "ratio": m / c_r}

    return {
        "schema": "shot-sweep-report-v1",
        "rule_file": "docs/frozen-rules/2026-10-03-shot-sweep.md",
        "note": ("Computed after fitting from the rule's lists of expectations and checks reported "
                 "beside results; no quantity here enters a decision rule."),
        "c_and_a_identity": identity(level_dirs),
        "platform_check_2048": {"cells": platform, "max_abs_delta_D": max_platform},
        "expectation_1_c_m": exp1,
        "expectation_2_r0_exact": exp2,
        "expectation_3_tfi_n2_labels": exp3,
        "expectation_4_r5_labels": exp4,
        "expectation_5_m_vs_c_r": exp5,
        "series_beside_h1": {rung: {row: [dc[level][row][rung] for level in LEVELS] for row in rows}
                             for rung in ("R0", "R5")},
        "labels": labels,
        "levels": list(LEVELS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True,
                        help="directory holding shots-<level>/ trees and predictions.json")
    parser.add_argument("--reference-analysis", type=Path,
                        default=_REPO / "artifacts/descriptor-information/strength-indicator/analysis-a.json",
                        help="analysis A of the 2026-10-03 strength-indicator fits")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    predictions = json.loads((args.runs / "predictions.json").read_text(encoding="utf-8"))
    reference = json.loads(args.reference_analysis.read_text(encoding="utf-8"))
    _write_json(args.out, report(args.runs, predictions, reference))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
