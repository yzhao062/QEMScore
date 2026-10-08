"""Summaries of released predictions (round-10 rule, Part E).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part E.

Subcommands (all post hoc and descriptive; none changes a frozen label):

harm    Harm rate of the learned arms in S0 (the descriptor ladder's
        2,048-shot fits, original candidates), S2 (stronger noise), and S4
        (deeper circuits), dataset seeds 101, 211, 307. An item is harmed when
        the arm's prediction lies farther from the label than the raw estimate
        r: |F - y| > |r - y|. The rate is the unit-weight macro over the row's
        cells of the harmed fraction, averaged over learner seeds, with the
        two-stage interval of analysis A. Reported for F and, as a comparator,
        for C, with the largest error dilation max(|F - y| - |r - y|).
dp      D_P = P - F, the permuted-measurement refit minus F, at every rung of
        the ladder for the original, strength-indicator, and new-seed
        original-candidate fit sets, with the two-stage interval and the
        three-way label.
margins The practical-margin classification of Part D of the round-9 rule
        recomputed with the margin at 0.025, 0.05, 0.10 (the rule's), and 0.20
        times mean calibrated raw error.
mlqem   Collects F - Rcal and F - R from the released ML-QEM analyses
        (archived and exact targets); nothing is recomputed.

Usage:
  PYTHONPATH=. python tools/round10_summaries.py harm --assets A --shift S --frozen-rule RULE
  PYTHONPATH=. python tools/round10_summaries.py dp --assets A --fresh F --frozen-rule RULE
  PYTHONPATH=. python tools/round10_summaries.py margins --frozen-rule RULE
  PYTHONPATH=. python tools/round10_summaries.py mlqem --frozen-rule RULE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import descriptor_ladder_analysis as dla  # noqa: E402
from tools import pooled_absolute as pa  # noqa: E402
from tools import reading_rules as rr  # noqa: E402

RULE_FILE = rr.RULE_FILE
OUT_DIR = _REPO / "artifacts/descriptor-information/round10"
HARM_SETTINGS = {
    "S0": {"fits": "{ASSETS}/descriptor-information-v1/fits",
           "cache": "{ASSETS}/shot-sweep-v1/runs/shots-2048/cache",
           "analysis": "artifacts/descriptor-information/analysis-a.json"},
    "S2": {"fits": "{SHIFT}/S2/fits", "cache": "{SHIFT}/S2/cache",
           "analysis": "artifacts/descriptor-information/shift-transfer/S2/analysis-a.json"},
    "S4": {"fits": "{SHIFT}/S4/fits", "cache": "{SHIFT}/S4/cache",
           "analysis": "artifacts/descriptor-information/shift-transfer/S4/analysis-a.json"},
}
DP_SETS = {
    "original_2048": ("{ASSETS}/descriptor-information-v1/fits",
                      "{ASSETS}/shot-sweep-v1/runs/shots-2048/cache",
                      "artifacts/descriptor-information/analysis-a.json"),
    "strength_indicator_2048": ("{ASSETS}/descriptor-information-strength-v1/fits",
                                "{ASSETS}/shot-sweep-v1/runs/shots-2048/cache",
                                "artifacts/descriptor-information/strength-indicator/analysis-a.json"),
    "fresh_orig_2048": ("{FRESH}/fresh/orig/shots-2048/fits",
                        "{FRESH}/fresh/orig/shots-2048/cache",
                        "artifacts/descriptor-information/round8/fresh/orig/2048/analysis-a.json"),
}
MARGIN_MULTIPLIERS = (0.025, 0.05, 0.10, 0.20)
MLQEM_ANALYSES = {
    "archived": "artifacts/mlqem-own-data/analysis.json",
    "exact": "artifacts/mlqem-own-data/analysis-exact-targets.json",
}


def check_rule(path: Path) -> None:
    if not path.is_file() or path.resolve() != (_REPO / RULE_FILE).resolve():
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")


def mean_entry(row: dla.Row, values: np.ndarray) -> dict:
    n = values.shape[0]
    points = np.asarray([dla.point_macro(values[s], row.members, row.all_cells)
                         for s in range(n)])
    draws = rr.seed_mean_draws(row, rr.macro_draws(row, values), n)
    return rr.interval(draws, float(points.mean()))


def check_d_reproduces(entry: dict, released: dict, where: str) -> float:
    """D's point and both interval endpoints must equal analysis A."""
    diff = max(abs(entry["point"] - released["point"]),
               abs(entry["interval"]["lower"] - released["interval"]["lower"]),
               abs(entry["interval"]["upper"] - released["interval"]["upper"]))
    if diff > rr.TOL:
        raise SystemExit(f"{where}: D differs from analysis A by {diff}")
    return diff


def label(entry: dict) -> str:
    return ("F beats C" if entry["excludes_zero_above"] else
            "C beats F" if entry["excludes_zero_below"] else "not distinguished")


def cache_for(cache_dir: Path, key: str, analysis: dict) -> dict:
    path = cache_dir / f"{key}.pkl"
    if rr.sha256_file(path) != analysis["inputs"]["caches_used"][key]["sha256"]:
        raise SystemExit(f"{path}: SHA-256 differs from analysis A's cache")
    return dla.load_cache(path)


def harm(args) -> dict:
    roots = {"ASSETS": args.assets, "SHIFT": args.shift}
    out = {}
    for setting, spec in HARM_SETTINGS.items():
        analysis = json.loads((_REPO / spec["analysis"]).read_text(encoding="utf-8"))
        store = dla.FitStore([rr.resolve(spec["fits"], roots)], use_orig=False)
        cache_dir = rr.resolve(spec["cache"], roots)
        rows = {}
        for row_key, a_row in analysis["parts"]["A"]["rows"].items():
            key, family = row_key.split("/")
            data = cache_for(cache_dir, key, analysis)
            row = dla.Row(data, family, row_key)
            raw_err = np.abs(row.noisy - row.ideal)
            for rung, a_rung in a_row["rungs"].items():
                entry = {}
                for arm in ("F", "C"):
                    records = rr.seeded(store, key, rung, arm)
                    err = rr.item_errors(row, store, records, "test")
                    point_check = abs(float(np.mean([dla.point_macro(err[s], row.members,
                                                                     row.all_cells)
                                                     for s in range(len(records))]))
                                      - a_rung["means"][arm]["point"])
                    if point_check > rr.TOL:
                        raise SystemExit(f"{setting} {row_key} {rung} {arm}: mean error "
                                         f"differs from analysis A by {point_check}")
                    harmed = (err > raw_err[None, :]).astype(float)
                    dilation = (err - raw_err[None, :]).max(axis=1)
                    entry[arm] = {"harm_rate": mean_entry(row, harmed),
                                  "max_dilation_median_over_seeds": float(np.median(dilation)),
                                  "max_dilation_max_over_seeds": float(dilation.max())}
                rows[f"{row_key}/{rung}"] = entry
        out[setting] = rows
    return out


def dp(args) -> dict:
    roots = {"ASSETS": args.assets, "FRESH": args.fresh}
    out = {}
    for fit_set, (fits, cache, analysis_path) in DP_SETS.items():
        analysis = json.loads((_REPO / analysis_path).read_text(encoding="utf-8"))
        store = dla.FitStore([rr.resolve(fits, roots)], use_orig=False)
        cache_dir = rr.resolve(cache, roots)
        rows = {}
        for row_key, a_row in analysis["parts"]["A"]["rows"].items():
            key, family = row_key.split("/")
            data = cache_for(cache_dir, key, analysis)
            row = dla.Row(data, family, row_key)
            est = dla.RowEstimator(row, store, rr.DRAWS, rr.SEED, len(rr.LEARNER_SEEDS))
            for rung, a_rung in a_row["rungs"].items():
                arms = dla.arms_at(store, dla.PARTS["A"], key, rung)
                d_entry, _ = est.difference(arms["C"], arms["F"])
                check_d_reproduces(d_entry, a_rung["D"], f"{fit_set} {row_key} {rung}")
                dp_entry, _ = est.difference(arms["P"], arms["F"])
                rows[f"{row_key}/{rung}"] = {
                    "D_P": dp_entry, "D_P_label": dla.classify(dp_entry, None).get("label"),
                    "D": a_rung["D"], "D_label": a_rung["classification"].get("label"),
                    "mean_P": a_rung["means"]["P"]["point"], "mean_F": a_rung["means"]["F"]["point"],
                    "mean_C": a_rung["means"]["C"]["point"]}
        out[fit_set] = rows
    return out


def margins(args) -> dict:
    pooled = json.loads((_REPO / "artifacts/descriptor-information/round9/pooled-absolute.json")
                        .read_text(encoding="utf-8"))
    mean_rcal = pooled["margins"]["mean_rcal_mae"]
    out = {"mean_rcal_mae": mean_rcal, "multipliers": list(MARGIN_MULTIPLIERS), "rows": [],
           "counts": {}}
    for row in pooled["pooled_table"]:
        lo, hi = row["pooled_D_interval"]["lower"], row["pooled_D_interval"]["upper"]
        verdicts = {str(m): pa.classify_practical_margin(lo, hi, m * mean_rcal[row["family"]])
                    for m in MARGIN_MULTIPLIERS}
        if verdicts["0.1"] != row["practical_verdict"]:
            raise SystemExit(f"{row['fit_set']} {row['family']} {row['rung']}: the 0.10 "
                             "verdict differs from Part D's")
        out["rows"].append({"fit_set": row["fit_set"], "family": row["family"],
                            "rung": row["rung"], "pooled_D": row["pooled_D"],
                            "interval": row["pooled_D_interval"], "verdicts": verdicts})
    for m in MARGIN_MULTIPLIERS:
        counts: dict[str, int] = {}
        for r in out["rows"]:
            counts[r["verdicts"][str(m)]] = counts.get(r["verdicts"][str(m)], 0) + 1
        out["counts"][str(m)] = counts
    out["changed_vs_0.10"] = {
        str(m): [f"{r['fit_set']}/{r['family']}/{r['rung']}" for r in out["rows"]
                 if r["verdicts"][str(m)] != r["verdicts"]["0.1"]]
        for m in MARGIN_MULTIPLIERS if m != 0.10}
    return out


def mlqem(args) -> dict:
    out = {}
    for name, path in MLQEM_ANALYSES.items():
        doc = json.loads((_REPO / path).read_text(encoding="utf-8"))
        out[name] = {"file": path, "sha256": rr.sha256_file(_REPO / path), "cells": {}}
        for setting, models in doc["results"].items():
            for model, res in models.items():
                q = res["quantities"]
                out[name]["cells"][f"{setting}/{model}"] = {
                    k: q[k] for k in ("F", "R", "Rcal", "F_minus_R", "F_minus_Rcal") if k in q}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("harm", "dp", "margins", "mlqem"):
        s = sub.add_parser(name)
        s.add_argument("--frozen-rule", type=Path, required=True)
        s.add_argument("--out", type=Path, default=OUT_DIR / f"{name}.json")
        if name in ("harm", "dp"):
            s.add_argument("--assets", type=Path, required=True)
        if name == "harm":
            s.add_argument("--shift", type=Path, required=True)
        if name == "dp":
            s.add_argument("--fresh", type=Path, required=True)
    args = p.parse_args(argv)
    check_rule(args.frozen_rule)
    result = {"harm": harm, "dp": dp, "margins": margins, "mlqem": mlqem}[args.command](args)
    doc = {"schema": f"round10-{args.command}-v1", "frozen_rule": RULE_FILE,
           "rule_file_sha256": rr.sha256_file(_REPO / RULE_FILE),
           "script_sha256": rr.sha256_file(Path(__file__)), "result": result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
