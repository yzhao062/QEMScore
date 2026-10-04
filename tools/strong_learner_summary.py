"""Comparisons the strong-learner rule reports beside its results (rule 2026-10-03-strong-learners.md).

From the rerun's analysis A (against the derived baseline), the derived baseline's analysis A
(against the 2026-10-03 macOS fits), the strong-learner report, and the fit trees, it writes:

- the rung statements of both records per family and rung, and those that changed;
- for every cell whose label differs from the derived baseline's: the paired D/C point
  difference and interval, whether that interval excludes zero, both D estimates, intervals,
  and widths, and whether the D interval narrowed, widened, or kept its width (these do not
  identify the cause of a label change), with mean C, F, and P of both records;
- every cell that reads "measurement hurts", with mean P beside mean F (both records);
- the six expectations stated in advance, evaluated as written;
- the platform comparison: the derived baseline's labels against the macOS labels, and the
  largest paired |delta D|;
- the identity of the derived baseline with the shot sweep's 2,048-shot DeltaAI fits of C and F
  at R0, N1, N2, and R5 (array differences and selected models).

It was written after the rerun's fits, to report what the rule lists; it decides nothing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SWEEP_RUNGS = ("R0", "N1", "N2", "R5")
TFI_EXPECTED_ADDS = ("N2", "N3", "N4", "R3-TFI", "R4", "R5")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _pt(entry: dict | None) -> float | None:
    return None if not entry or entry.get("status") != "estimated" else float(entry["point"])


def statements(strong: dict, derived: dict) -> dict:
    out: dict = {"by_family": {}, "changed": []}
    for part in ("A", "B"):
        for family, rungs in strong["parts"][part]["rung_statements"].items():
            entry = {}
            for rung, st in rungs.items():
                base = derived["parts"][part]["rung_statements"][family][rung]
                entry[rung] = {"strong": st["statement"], "derived": base["statement"],
                               "labels_strong": st["labels"], "labels_derived": base["labels"]}
                if st["statement"] != base["statement"]:
                    out["changed"].append(f"{part}/{family}/{rung}")
            out["by_family"][f"{part}/{family}"] = entry
    return out


def label_changes(strong: dict, derived: dict) -> list[dict]:
    rows = []
    for part in ("A", "B"):
        for row, rr in strong["parts"][part]["rows"].items():
            for rung, cell in rr["rungs"].items():
                base = derived["parts"][part]["rows"][row]["rungs"][rung]
                new, old = cell["classification"]["label"], base["classification"]["label"]
                if new == old:
                    continue
                paired = cell["comparison_vs_original"]["delta_D_over_C"]
                i_new, i_old = cell["D"]["interval"], base["D"]["interval"]
                w_new, w_old = i_new["upper"] - i_new["lower"], i_old["upper"] - i_old["lower"]
                rows.append({
                    "cell": f"{part}/{row}/{rung}", "label_strong": new, "label_derived": old,
                    "paired_D_over_C": {"point": paired["point"], "interval": paired["interval"]},
                    "paired_interval_excludes_zero": bool(paired["interval"]["lower"] > 0
                                                          or paired["interval"]["upper"] < 0),
                    "D_strong": {"point": cell["D"]["point"], "interval": i_new, "width": w_new},
                    "D_derived": {"point": base["D"]["point"], "interval": i_old, "width": w_old},
                    "D_interval_width_change": ("narrowed" if w_new < w_old
                                                else "widened" if w_new > w_old else "same"),
                    "means_strong": {arm: _pt(cell["means"].get(arm)) for arm in ("C", "F", "P")},
                    "means_derived": {arm: _pt(base["means"].get(arm)) for arm in ("C", "F", "P")},
                })
    return rows


def hurts_cells(strong: dict, derived: dict) -> list[dict]:
    """Every cell of the rerun that reads "measurement hurts", with mean P beside mean F (the rule's
    Reading section), and the derived baseline's label and means for the same cell."""
    rows = []
    for part in ("A", "B"):
        for row, rr in strong["parts"][part]["rows"].items():
            for rung, cell in rr["rungs"].items():
                if cell["classification"]["label"] != "measurement_hurts":
                    continue
                base = derived["parts"][part]["rows"][row]["rungs"][rung]
                rows.append({
                    "cell": f"{part}/{row}/{rung}",
                    "label_derived": base["classification"]["label"],
                    "means_strong": {arm: _pt(cell["means"].get(arm)) for arm in ("C", "F", "P")},
                    "means_derived": {arm: _pt(base["means"].get(arm)) for arm in ("C", "F", "P")},
                })
    return rows


def expectations(strong: dict, derived: dict, report: dict) -> dict:
    a_rows, d_rows = strong["parts"]["A"]["rows"], derived["parts"]["A"]["rows"]
    e1 = {row: _pt(a_rows[row]["rungs"]["R0"]["means"]["C"]) / _pt(d_rows[row]["rungs"]["R0"]["means"]["C"])
          for row in a_rows}
    e2 = {row: a_rows[row]["rungs"]["R0"]["classification"]["label"] for row in a_rows}
    e3 = {row: {rung: a_rows[row]["rungs"][rung]["classification"]["label"] for rung in TFI_EXPECTED_ADDS}
          for row in a_rows if row.endswith("/tfi")}
    e4 = {row: _pt(a_rows[row]["rungs"]["R5"]["comparison_vs_original"]["delta_D_over_C"]) for row in a_rows}
    e5 = {row: report["parts"]["A"]["rows"][row]["rungs"]["R0"]["selection_counts"]["C"]
          for row in report["parts"]["A"]["rows"]}
    b_rows, bd_rows = strong["parts"]["B"]["rows"], derived["parts"]["B"]["rows"]
    e6 = {row: {"F_strong": _pt(b_rows[row]["rungs"]["B-complete"]["means"]["F"]),
                "F_derived": _pt(bd_rows[row]["rungs"]["B-complete"]["means"]["F"])} for row in b_rows}
    by_seed: dict[str, int] = {}
    for row, counts in e5.items():
        seed = row.split("/")[0]
        # The two family rows of a dataset share one fit; take the smaller count to be safe.
        by_seed[seed] = min(by_seed.get(seed, counts.get("poly5_ridge", 0)), counts.get("poly5_ridge", 0))
    return {
        "1_r0_mean_C_ratio": {"values": e1, "holds": all(v < 0.25 for v in e1.values())},
        "2_r0_labels": {"values": e2, "holds": all(v != "measurement_adds" for v in e2.values())},
        "3_tfi_labels": {"values": e3, "holds": all(v == "measurement_adds"
                                                    for r in e3.values() for v in r.values())},
        "4_r5_paired_delta_D_over_C": {"values": e4, "holds": all(abs(v) < 0.05 for v in e4.values())},
        "5_r0_C_selects_poly5_ridge": {"values": by_seed, "holds": all(v >= 15 for v in by_seed.values())},
        "6_b_complete_mean_F": {"values": e6, "holds": all(v["F_strong"] < v["F_derived"] for v in e6.values())},
    }


def platform(derived: dict, macos: dict) -> dict:
    diffs, max_d = [], 0.0
    for part in ("A", "B"):
        for row, rr in derived["parts"][part]["rows"].items():
            for rung, cell in rr["rungs"].items():
                ref = macos["parts"][part]["rows"][row]["rungs"][rung]["classification"]["label"]
                if cell["classification"]["label"] != ref:
                    diffs.append({"cell": f"{part}/{row}/{rung}", "deltaai": cell["classification"]["label"],
                                  "macos": ref})
                d = _pt(cell["comparison_vs_original"]["delta_D"])
                if d is not None:
                    max_d = max(max_d, abs(d))
    return {"label_differences": diffs, "max_abs_paired_delta_D": max_d}


def sweep_identity(derived_fits: Path, sweep_fits: Path) -> dict:
    out = {"compared": 0, "max_abs_diff": 0.0, "nonzero": [], "selected_model_differs": []}
    for rung in SWEEP_RUNGS:
        for npz in sorted(sweep_fits.glob(f"shipped-s*-n640__{rung}__k*__[CF].npz")):
            mine = derived_fits / npz.name
            with np.load(npz) as a, np.load(mine) as b:
                diff = max(float(np.max(np.abs(a[r] - b[r]))) for r in ("test", "validation"))
            sel_a = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))["selected_model"]
            sel_b = json.loads(mine.with_suffix(".json").read_text(encoding="utf-8"))["selected_model"]
            if sel_a != sel_b:
                out["selected_model_differs"].append(npz.stem)
            if diff:
                out["nonzero"].append({"fit": npz.stem, "max_abs_diff": diff})
            out["max_abs_diff"] = max(out["max_abs_diff"], diff)
            out["compared"] += 1
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis-a", type=Path, required=True, help="the rerun's analysis A")
    parser.add_argument("--derived-analysis-a", type=Path, required=True,
                        help="the derived baseline's analysis A (against the macOS fits)")
    parser.add_argument("--report", type=Path, required=True, help="tools/strong_learner_report.py output")
    parser.add_argument("--macos-analysis-a", type=Path, required=True,
                        help="analysis A of the 2026-10-03 strength-indicator fits")
    parser.add_argument("--derived-fits", type=Path, default=None)
    parser.add_argument("--sweep-2048-fits", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    strong, derived = load(args.analysis_a), load(args.derived_analysis_a)
    out = {
        "schema": "strong-learner-summary-v1",
        "rule_file": "docs/frozen-rules/2026-10-03-strong-learners.md",
        "note": "Computed after the rerun's fits from the rule's reporting requirements; decides nothing.",
        "rung_statements": statements(strong, derived),
        "label_changes": label_changes(strong, derived),
        "measurement_hurts_cells": hurts_cells(strong, derived),
        "expectations": expectations(strong, derived, load(args.report)),
        "platform_derived_vs_macos": platform(derived, load(args.macos_analysis_a)),
    }
    if args.derived_fits and args.sweep_2048_fits:
        out["derived_vs_sweep_2048"] = sweep_identity(args.derived_fits, args.sweep_2048_fits)
    _write_json(args.out, out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
