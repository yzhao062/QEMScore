#!/usr/bin/env python3
"""Equal-spend analysis comparing learned mitigation against raw and calibrated measurement.

Follows rule docs/frozen-rules/2026-10-04-crossed-follow-ups.md (part H).
Descriptive; it decides nothing.

For each row (dataset seed x family) and rung (e.g., R0, N1, N2, R5):
- F_2048: mean F macro test error at 2,048 shots (from analysis A's means.F.point).
- For each shot level ell (the seven levels of the shot sweep: 256, 1024, 2048, 8192,
  32768, 131072, exact):
    R_ell     : macro test MAE of the raw estimate r against ideal expectation y over
                the row's four cells (as descriptor_ladder_posthoc.py defines the raw arm);
    R_cal,ell : macro of c_r from the predictions file.
- For each finite ell > 2,048 with R_ell <= F_2048:
    n*_R = 960 * 2048 / (ell - 2048)
- For each finite ell > 2,048 with R_cal,ell <= F_2048:
    n*_cal = (960 * 2048 - 320 * ell) / (ell - 2048), kept signed.
- Shot costs for n deployed circuits: F = 960 * 2048 + 2048 n, R = ell n, R_cal = 320 ell + ell n.
  Since ell > 2048, the comparator costs fewer shots when 0 <= n < n* and F costs fewer when
  n > n*. A crossing at or below zero means F costs no more shots at every n >= 0 (true of
  R_cal at every ell >= 6,144).
- Smallest such finite ell and whether exact level's R or R_cal is at most F_2048.

Usage:
    PYTHONPATH=. python tools/equal_spend.py \\
        --analysis-a 256=ARTIFACTS/levels/256/analysis-a.json ... exact=... \\
        --predictions ARTIFACTS/shot-sweep/predictions.json \\
        --data 256=LEVELS/shots-256 ... \\
        --cache 256=RUNS/shots-256/cache ... \\
        --rungs R0 N1 N2 R5 \\
        --out OUT_JSON \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

# Ensure tools and repo root are importable
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

import descriptor_ladder_analysis as dla  # noqa: E402
import measurement_floor as mf  # noqa: E402

SCHEMA = "equal-spend-v1"
RULE_STATUS = "follows rule 2026-10-04-crossed-follow-ups.md (part H)"
SEEDS = mf.SEEDS  # (101, 211, 307)
FAMILIES = mf.FAMILIES  # ("tfi", "heisenberg")
SEVERITIES = mf.SEVERITIES  # ("L1", "L3")
OBSERVABLES = mf.OBSERVABLES  # ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
DEFAULT_RUNGS = ("R0", "N1", "N2", "R5")
CANONICAL_LEVELS = ("256", "1024", "2048", "8192", "32768", "131072", "exact")



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def parse_key_value_pairs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid entry '{item}': expected format LEVEL=PATH")
        k, v = item.split("=", 1)
        out[k.strip()] = Path(v.strip())
    return out


def normalize_level_name(lvl: str) -> str:
    name = lvl
    if name.startswith("shots-"):
        name = name[len("shots-"):]
    elif name.startswith("sweep-"):
        name = name[len("sweep-"):]
    return name


def parse_finite_shot_count(lvl: str) -> int | None:
    norm = normalize_level_name(lvl)
    if norm.isdigit():
        return int(norm)
    return None




def compute_raw_r_macro(cache_dir: Path, data_dir: Path | None, seed: int, family: str) -> float:
    """Compute macro test MAE of raw estimate r over the row's four cells."""
    key = f"shipped-s{seed}-n640"
    label = f"{key}/{family}"

    cache_path = cache_dir / f"{key}.pkl"
    if not cache_path.exists():
        cache_path = cache_dir / f"{key}.json.gz"
    if not cache_path.exists():
        cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        if cache_path.name.endswith(".pkl"):
            with open(cache_path, "rb") as h:
                cache_data = pickle.load(h)
        else:
            cache_data = dla.load_cache(cache_path)
    elif data_dir is not None:
        # Load from items.jsonl
        items_path = data_dir / f"regen-{key}" / "items.jsonl"
        with open(items_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        test_items = [item for item in items if item.get("split") == "test"]
        cache_data = {"test": test_items}
    else:
        raise FileNotFoundError(f"Cache file for {key} not found in {cache_dir}")

    row = dla.Row(cache_data, family, label)
    raw_errors = np.abs(row.noisy - row.ideal)
    return dla.point_macro(raw_errors, row.members, row.all_cells)


def compute_n_star_R(ell: int) -> float:
    """n*_R = 960 * 2048 / (ell - 2048)."""
    return (960.0 * 2048.0) / (ell - 2048.0)


def compute_n_star_cal(ell: int) -> float:
    """Signed crossing n*_cal = (960 * 2048 - 320 * ell) / (ell - 2048).

    A value at or below zero means F costs no more shots at every n >= 0.
    """
    return (960.0 * 2048.0 - 320.0 * float(ell)) / (float(ell) - 2048.0)


def shot_costs(ell: int, n: float) -> dict[str, float]:
    """Total shots for n deployed circuits, per family and noise strength."""
    return {
        "F": 960.0 * 2048.0 + 2048.0 * n,
        "R": float(ell) * n,
        "R_cal": 320.0 * float(ell) + float(ell) * n,
    }


COST_DIRECTION = (
    "For a crossing n*, the comparator (R or R_cal at level ell > 2048) costs fewer shots than F "
    "when 0 <= n < n*, and F costs fewer when n > n*. A crossing at or below zero means F costs "
    "no more shots at every n >= 0."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--analysis-a",
        action="extend",
        nargs="+",
        required=True,
        metavar="LEVEL=PATH",
        help="Analysis A JSON paths per shot level",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to shot sweep predictions.json",
    )
    parser.add_argument(
        "--data",
        action="extend",
        nargs="+",
        required=True,
        metavar="LEVEL=DIR",
        help="Data directories per shot level",
    )
    parser.add_argument(
        "--cache",
        action="extend",
        nargs="+",
        required=True,
        metavar="LEVEL=DIR",
        help="Cache directories per shot level",
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        default=list(DEFAULT_RUNGS),
        help=f"Part A rungs to evaluate (default: {list(DEFAULT_RUNGS)})",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify inputs exist and match hashes without computing spend thresholds",
    )

    args = parser.parse_args(argv)

    analysis_a_map = parse_key_value_pairs(args.analysis_a)
    data_map = parse_key_value_pairs(args.data)
    cache_map = parse_key_value_pairs(args.cache)

    all_levels = sorted(analysis_a_map.keys())
    rungs = list(args.rungs)


    # Verify predictions file
    if not args.predictions.exists():
        raise FileNotFoundError(f"--predictions file {args.predictions} not found")
    predictions_data = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_sha = sha256_file(args.predictions)

    inputs_meta: dict[str, Any] = {
        "predictions": {"path": display(args.predictions), "sha256": predictions_sha},
        "analysis_a": {},
        "data": {},
        "cache": {},
    }

    analysis_a_data_by_level: dict[str, dict] = {}
    for lvl in all_levels:
        p_a = analysis_a_map[lvl]
        p_data = data_map.get(lvl)
        p_cache = cache_map.get(lvl)

        if not p_a.exists():
            raise FileNotFoundError(f"analysis-a for level {lvl} not found: {p_a}")
        if p_data is None or not p_data.is_dir():
            raise FileNotFoundError(f"data directory for level {lvl} not found: {p_data}")
        if p_cache is None or not p_cache.is_dir():
            raise FileNotFoundError(f"cache directory for level {lvl} not found: {p_cache}")

        sha_a = sha256_file(p_a)
        inputs_meta["analysis_a"][lvl] = {"path": display(p_a), "sha256": sha_a}
        inputs_meta["data"][lvl] = {"path": display(p_data)}
        inputs_meta["cache"][lvl] = {"path": display(p_cache)}

        analysis_a_data_by_level[lvl] = json.loads(p_a.read_text(encoding="utf-8"))

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    if args.dry_run:
        print("[DRY-RUN] Verified equal-spend inputs successfully:")
        print(f"  --predictions: {args.predictions} (sha256={predictions_sha[:16]}...)")
        for lvl in all_levels:
            print(f"  Level '{lvl}':")
            print(f"    analysis-a: {inputs_meta['analysis_a'][lvl]['path']}")
            print(f"    data:       {inputs_meta['data'][lvl]['path']}")
            print(f"    cache:      {inputs_meta['cache'][lvl]['path']}")

        if args.out is not None:
            dry_payload = {
                "schema": SCHEMA,
                "status": "dry-run: inputs verified without computation",
                "script": display(script_path),
                "script_sha256": script_sha,
                "inputs": inputs_meta,
                "rungs": rungs,
                "levels": all_levels,
            }
            write_json_atomic(args.out, dry_payload)
            print(f"[DRY-RUN] Wrote verification digest to {args.out}")
        return 0

    # Locate 2048 level for F_2048
    lvl_2048 = None
    for lvl in all_levels:
        if normalize_level_name(lvl) == "2048":
            lvl_2048 = lvl
            break
    if lvl_2048 is None:
        raise ValueError(f"Required 2,048-shot level not found in {all_levels}")

    analysis_2048 = analysis_a_data_by_level[lvl_2048]
    rows_2048 = analysis_2048["parts"]["A"]["rows"]

    # Precompute R_ell and R_cal_ell
    rule_cells = predictions_data.get("rule_cells", {})

    cells_output: dict[str, Any] = {}

    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        for fam in FAMILIES:
            row_key = f"{key}/{fam}"
            row_entry = rows_2048.get(row_key)
            if row_entry is None:
                continue

            for rung in rungs:
                if rung not in row_entry["rungs"]:
                    continue

                cell_key = f"{row_key}/{rung}"
                f_2048_val = float(row_entry["rungs"][rung]["means"]["F"]["point"])

                level_stats: dict[str, Any] = {}
                qualifying_finite_ells_R: list[int] = []
                qualifying_finite_ells_cal: list[int] = []

                exact_R_le_F = None
                exact_R_cal_le_F = None

                for lvl in all_levels:
                    norm_lvl = normalize_level_name(lvl)
                    cache_dir = cache_map[lvl]
                    data_dir = data_map[lvl]

                    # 1. R_ell: macro test MAE of raw estimate r
                    r_ell = compute_raw_r_macro(cache_dir, data_dir, seed, fam)

                    # 2. R_cal,ell: macro of c_r from predictions
                    c_r_vals = []
                    for s in SEVERITIES:
                        for o in OBSERVABLES:
                            pred_k = f"{norm_lvl}/{row_key}/{s}/{o}"
                            if pred_k not in rule_cells:
                                pred_k = f"{lvl}/{row_key}/{s}/{o}"
                            if pred_k not in rule_cells:
                                raise KeyError(f"rule cell {pred_k} not found in predictions.json")
                            c_r_vals.append(float(rule_cells[pred_k]["c_r"]))
                    r_cal_ell = math.fsum(c_r_vals) / len(c_r_vals)

                    lvl_record: dict[str, Any] = {
                        "R": float(r_ell),
                        "R_cal": float(r_cal_ell),
                    }

                    # Check finite ell > 2048
                    finite_count = parse_finite_shot_count(lvl)
                    if finite_count is not None and finite_count > 2048:
                        r_le = bool(r_ell <= f_2048_val)
                        r_cal_le = bool(r_cal_ell <= f_2048_val)
                        lvl_record["R_le_F_2048"] = r_le
                        lvl_record["R_cal_le_F_2048"] = r_cal_le

                        if r_le:
                            n_star_r = compute_n_star_R(finite_count)
                            lvl_record["n_star_R"] = float(n_star_r)
                            qualifying_finite_ells_R.append(finite_count)
                        else:
                            lvl_record["n_star_R"] = None

                        if r_cal_le:
                            n_star_c = compute_n_star_cal(finite_count)
                            lvl_record["n_star_cal"] = float(n_star_c)
                            lvl_record["F_no_costlier_at_every_n_cal"] = bool(n_star_c <= 0.0)
                            qualifying_finite_ells_cal.append(finite_count)
                        else:
                            lvl_record["n_star_cal"] = None

                    elif norm_lvl == "exact":
                        exact_R_le_F = bool(r_ell <= f_2048_val)
                        exact_R_cal_le_F = bool(r_cal_ell <= f_2048_val)
                        lvl_record["R_le_F_2048"] = exact_R_le_F
                        lvl_record["R_cal_le_F_2048"] = exact_R_cal_le_F

                    level_stats[norm_lvl] = lvl_record

                smallest_ell_R = min(qualifying_finite_ells_R) if qualifying_finite_ells_R else None
                smallest_ell_cal = min(qualifying_finite_ells_cal) if qualifying_finite_ells_cal else None

                cells_output[cell_key] = {
                    "row": row_key,
                    "rung": rung,
                    "dataset_seed": seed,
                    "family": fam,
                    "F_2048": f_2048_val,
                    "levels": level_stats,
                    "smallest_finite_level_R": smallest_ell_R,
                    "smallest_finite_level_R_cal": smallest_ell_cal,
                    "exact_R_le_F_2048": exact_R_le_F,
                    "exact_R_cal_le_F_2048": exact_R_cal_le_F,
                }

    payload = {
        "schema": SCHEMA,
        "status": RULE_STATUS,
        "script": display(script_path),
        "script_sha256": script_sha,
        "inputs": inputs_meta,
        "rungs": rungs,
        "levels": [normalize_level_name(l) for l in all_levels],
        "cost_direction": COST_DIRECTION,
        "cells": cells_output,
    }

    if args.out is not None:
        write_json_atomic(args.out, payload)
        print(f"Wrote equal-spend results to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
