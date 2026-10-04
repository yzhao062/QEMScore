#!/usr/bin/env python3
"""Simultaneous intervals for D across rungs and shot levels (post hoc).

Post hoc and descriptive; it decides nothing and changes no frozen label. Each
frozen analysis labels a cell from its pointwise 95 percent interval for mean
D = C - F. Statements that combine cells, such as the first rung or shot level
that reads "F beats C", are joint. This script recomputes analysis A's
two-stage bootstrap draws of D with that script's own classes (fresh
``default_rng(20261002)``, 10,000 draws, learner seeds then whole test
circuits), so every pointwise interval equals analysis A's. Within one dataset
seed, all rows, rungs, and levels share those draws: the same learner-seed and
circuit resamples enter every cell, and a level's C fits are the same fits at
every level. The script then widens each cell's interval to a pair of order
statistics until the band covers all cells of the dataset seed jointly in at
least 95 percent of the draws (a max-rank band), and relabels each cell from
the band. The bands hold per dataset seed; they make no simultaneous statement
across dataset seeds or fit sets.

It also reports, per cell, the one-sided 95 percent upper bound of D/C (the
95th percentile of its draws), for a stated equivalence margin.

Usage:
    PYTHONPATH=. python tools/joint_bootstrap.py \
        --set NAME LEVEL=FITS@CACHE@ANALYSIS_A [LEVEL=...] \
        --out artifacts/descriptor-information/posthoc-joint-bootstrap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))

import descriptor_ladder_analysis as dla  # noqa: E402

SCHEMA = "posthoc-joint-bootstrap-v1"
SEEDS = (101, 211, 307)
FAMILIES = ("tfi", "heisenberg")
LADDER = {"tfi": ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R4", "R5"),
          "heisenberg": ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R4", "R5")}
NOISE_LADDER = ("R0", "N1", "N2", "N3", "N4")
JOINT_LEVEL = 0.95
ONE_SIDED = 95.0
LABELS = {1: "measurement_adds", 0: "not_distinguished", -1: "measurement_hurts"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def label_of(lower: float, upper: float) -> str:
    return LABELS[1 if lower > 0.0 else (-1 if upper < 0.0 else 0)]


def max_rank_band(draws: np.ndarray, level: float = JOINT_LEVEL) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-column order-statistic limits that jointly cover ``level`` of the draws.

    ``draws`` is (B x m). Each column's draws are ranked; a draw's extremeness
    is the largest two-sided rank position over the columns. With ``critical``
    the ``level`` quantile of that extremeness, the band takes, in every column,
    the order statistics at ranks ``B - 1 - critical`` and ``critical``. Every
    draw whose extremeness is at most ``critical`` lies inside the band, so the
    returned coverage is at least ``level``. With one column the band is the
    pointwise interval up to one order statistic.
    """
    b = draws.shape[0]
    ranks = np.argsort(np.argsort(draws, axis=0, kind="stable"), axis=0, kind="stable")
    extremeness = np.maximum(ranks, b - 1 - ranks).max(axis=1)
    critical = int(np.quantile(extremeness, level, method="higher"))
    ordered = np.sort(draws, axis=0)
    lower = ordered[b - 1 - critical]
    upper = ordered[critical]
    covered = np.mean(np.all((draws >= lower) & (draws <= upper), axis=1))
    return lower, upper, float(covered)


def load_level(fits: Path, cache_dir: Path, analysis_path: Path) -> tuple:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    caches_used = analysis["inputs"]["caches_used"]
    store = dla.FitStore([fits], use_orig=False)
    caches = {}
    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        path = cache_dir / f"{key}.pkl"
        if sha256_file(path) != caches_used[key]["sha256"]:
            raise SystemExit(f"{path}: SHA-256 differs from the cache analysis A used")
        caches[key] = dla.load_cache(path)
    return analysis, store, caches


def run_set(name: str, levels: list[tuple[str, Path, Path, Path]],
            draws: int, seed: int) -> dict:
    loaded = {lvl: load_level(f, c, a) for lvl, f, c, a in levels}
    cells_out: dict[str, dict] = {}
    groups: dict[int, list[str]] = {s: [] for s in SEEDS}
    group_draws: dict[int, list[np.ndarray]] = {s: [] for s in SEEDS}
    max_interval_diff = 0.0
    for lvl, (analysis, store, caches) in loaded.items():
        rows = analysis["parts"]["A"]["rows"]
        for dseed in SEEDS:
            key = f"shipped-s{dseed}-n640"
            for fam in FAMILIES:
                row_key = f"{key}/{fam}"
                row = dla.Row(caches[key], fam, row_key)
                est = dla.RowEstimator(row, store, draws, seed, 20)
                for rung in LADDER[fam]:
                    if rung not in rows[row_key]["rungs"]:
                        continue
                    arms = dla.arms_at(store, dla.PARTS["A"], key, rung)
                    entry, parts = est.difference(arms["C"], arms["F"])
                    frozen = rows[row_key]["rungs"][rung]
                    for bound in ("lower", "upper"):
                        max_interval_diff = max(
                            max_interval_diff,
                            abs(entry["interval"][bound] - frozen["D"]["interval"][bound]))
                    ratio = parts["draws"] / parts["first_mean_draws"]
                    cell_key = f"{lvl}/{row_key}/{rung}"
                    cells_out[cell_key] = {
                        "level": lvl, "row": row_key, "rung": rung,
                        "D": entry["point"],
                        "D_pointwise": entry["interval"],
                        "label_pointwise": frozen["classification"]["label"],
                        "D_over_C": frozen["D_over_C"]["point"],
                        "D_over_C_one_sided_upper_95": float(np.percentile(ratio, ONE_SIDED)),
                    }
                    groups[dseed].append(cell_key)
                    group_draws[dseed].append(parts["draws"])
    if max_interval_diff > 1e-12:
        raise SystemExit(f"[{name}] pointwise D intervals differ from analysis A "
                         f"by {max_interval_diff}")
    coverage = {}
    for dseed in SEEDS:
        matrix = np.column_stack(group_draws[dseed])
        lower, upper, covered = max_rank_band(matrix)
        coverage[str(dseed)] = {"cells": len(groups[dseed]), "joint_coverage": covered}
        for j, cell_key in enumerate(groups[dseed]):
            cell = cells_out[cell_key]
            cell["D_simultaneous"] = {"lower": float(lower[j]), "upper": float(upper[j])}
            cell["label_simultaneous"] = label_of(lower[j], upper[j])
    return {"levels": {lvl: {"fits": display(f), "cache": display(c), "analysis_a": display(a)}
                       for lvl, f, c, a in levels},
            "pointwise_reproduces_analysis_a_max_abs_diff": max_interval_diff,
            "groups": coverage,
            "statements": statements(cells_out, list(loaded)),
            "cells": cells_out}


def rung_statement(labels: list[str]) -> str:
    return labels[0] if len(set(labels)) == 1 else "mixed"


def statements(cells: dict, levels: list[str]) -> dict:
    """Rung statements and the first noise rung reading "adds", under both bands."""
    out = {}
    for lvl in levels:
        for fam in FAMILIES:
            per_rung = {}
            for rung in LADDER[fam]:
                keys = [f"{lvl}/shipped-s{s}-n640/{fam}/{rung}" for s in SEEDS]
                if not all(k in cells for k in keys):
                    continue
                per_rung[rung] = {
                    band: rung_statement([cells[k][f"label_{band}"] for k in keys])
                    for band in ("pointwise", "simultaneous")}
            first = {}
            for band in ("pointwise", "simultaneous"):
                first[band] = next((r for r in NOISE_LADDER if r in per_rung
                                    and per_rung[r][band] == "measurement_adds"), None)
            out[f"{lvl}/{fam}"] = {"rung_statements": per_rung,
                                   "first_noise_rung_adds": first}
    if len(levels) > 1:
        # Per row and rung: the first level, in the order given, reading "adds".
        rows_rungs = sorted({(c["row"], c["rung"]) for c in cells.values()})
        out["first_level_adds"] = {
            f"{row}/{rung}": {
                band: next((lvl for lvl in levels
                            if cells.get(f"{lvl}/{row}/{rung}", {}).get(f"label_{band}")
                            == "measurement_adds"), None)
                for band in ("pointwise", "simultaneous")}
            for row, rung in rows_rungs}
    return out


def parse_level(spec: str) -> tuple[str, Path, Path, Path]:
    level, rest = spec.split("=", 1)
    fits, cache, analysis = rest.split("@")
    return level, Path(fits), Path(cache), Path(analysis)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs="+", action="append", required=True,
                        metavar="NAME LEVEL=FITS@CACHE@ANALYSIS_A")
    parser.add_argument("--draws", type=int, default=dla.DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=dla.RULE_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = {
        "schema": SCHEMA,
        "status": "post hoc, descriptive; decides nothing",
        "script": "tools/joint_bootstrap.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap": {"seed": args.seed, "draws": args.draws,
                      "joint_level": JOINT_LEVEL, "one_sided_percentile": ONE_SIDED},
        "sets": {},
    }
    for spec in args.set:
        name, levels = spec[0], [parse_level(s) for s in spec[1:]]
        print(f"[{name}] {len(levels)} level(s)", flush=True)
        payload["sets"][name] = run_set(name, levels, args.draws, args.seed)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
