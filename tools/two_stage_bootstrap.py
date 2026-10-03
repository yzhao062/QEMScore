"""Two-stage percentile bootstrap of the seed-averaged capacity-matched gap.

Implements the interval of the frozen rule (frozen-rule-seed-replication.md,
2026-10-01). For each row (dataset seed x family) and each of the 10,000
draws, the draw first resamples the learner seeds with replacement and then
resamples the row's test physical circuits with replacement. The circuit
resample is drawn once per draw and shared across the three arms and all
learner seeds, and it keeps every row of a circuit together (a circuit's four
strength-by-observable rows enter with the circuit's multiplicity). Inside a
draw, each cell's MAE is the multiplicity-weighted mean absolute error of the
items in that cell, and the macro MAE is the equal-weight mean over cells,
which at unit weights is the campaign's `_family_mae`. The statistic is the
mean over the resampled learner seeds of D = C - F (and of P - C, P - F). The
interval is the 2.5th to 97.5th percentile (numpy linear interpolation).

Each row uses a fresh `numpy.random.default_rng(20261001)`, so a row's interval
does not depend on which other rows were run. Per draw the calls are, in order,
`integers(0, n_seeds, n_seeds)` and `integers(0, n_circuits, n_circuits)`.

The decision rule is then applied: a row supports a positive gap only if its
interval for mean D lies wholly above zero.

Usage:
  python tools/two_stage_bootstrap.py --run-dir results/seedrep \
      [--draws 10000] [--seed 20261001] [--out results/seedrep/bootstrap.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from qemscore.campaign.analysis import _family_mae
from qemscore.campaign.design import FAMILIES

ARMS = ("F", "C", "P")
CELL_FIELDS = ("noise_family", "severity", "observable")
QUANTITIES = {
    "D": ("C", "F"),          # C - F
    "P_minus_C": ("P", "C"),
    "P_minus_F": ("P", "F"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_inputs(run_dir: Path, key: str, family: str, seeds: list[int]) -> dict:
    cache_gz = run_dir / "cache" / f"{key}.json.gz"
    cache_json = run_dir / "cache" / f"{key}.json"
    cache_pkl = run_dir / "cache" / f"{key}.pkl"
    if not cache_gz.is_file() and not cache_pkl.is_file() and (run_dir / "inputs" / "cache" / f"{key}.json.gz").is_file():
        cache_gz = run_dir / "inputs" / "cache" / f"{key}.json.gz"

    if cache_gz.is_file():
        with gzip.open(cache_gz, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    elif cache_json.is_file():
        with open(cache_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with open(cache_pkl, "rb") as handle:
            data = pickle.load(handle)
    test = data["test"]
    index = [i for i, row in enumerate(test) if str(row["family"]) == family]
    rows = [test[i] for i in index]
    ideal = np.asarray([float(row["ideal_expectation"]) for row in rows])
    circuits = sorted({str(row["circuit_id"]) for row in rows})
    circuit_index = {name: position for position, name in enumerate(circuits)}
    item_circuit = np.asarray([circuit_index[str(row["circuit_id"])] for row in rows])
    cells = sorted({tuple(str(row[field]) for field in CELL_FIELDS) for row in rows})
    item_cell = np.asarray(
        [cells.index(tuple(str(row[field]) for field in CELL_FIELDS)) for row in rows])
    # Every circuit holds exactly one row in every cell; the macro under circuit
    # weights relies on no cell ever losing all its weight.
    incidence = np.zeros((len(circuits), len(cells)), dtype=int)
    np.add.at(incidence, (item_circuit, item_cell), 1)
    if not np.all(incidence >= 1):
        raise ValueError(f"{key}/{family}: a circuit lacks a row in some cell")

    fits_dir = run_dir / "fits"
    if not fits_dir.is_dir() and (run_dir / "inputs" / "fits").is_dir():
        fits_dir = run_dir / "inputs" / "fits"

    errors = {arm: np.empty((len(seeds), len(rows))) for arm in ARMS}
    flags = {arm: np.zeros(len(seeds), dtype=bool) for arm in ARMS}
    flags_secondary = {arm: np.zeros(len(seeds), dtype=bool) for arm in ARMS}
    point_mae = {arm: np.empty(len(seeds)) for arm in ARMS}
    for s, k in enumerate(seeds):
        for arm in ARMS:
            stem = f"{key}__k{k:02d}__{arm}"
            meta = json.loads((fits_dir / f"{stem}.json").read_text(
                encoding="utf-8"))
            predictions = np.load(fits_dir / f"{stem}.npz")["test"]
            values = predictions[index]
            errors[arm][s] = np.abs(values - ideal)
            flags[arm][s] = bool(meta["flagged_non_converged"])
            flags_secondary[arm][s] = bool(meta["flagged_excluding_spurious_matmul_fpe"])
            point_mae[arm][s] = float(meta["test_family_mae"][family])
            # The vectorized macro has to equal the campaign helper at unit weights.
            if s == 0:
                check = _family_mae(rows, values, artifact_id=data["dataset_hash"])[family]
                if abs(check - point_mae[arm][s]) > 1e-12:
                    raise ValueError(f"{stem}: macro MAE disagrees with the fit record")
    return {
        "n_items": len(rows),
        "n_circuits": len(circuits),
        "cells": [list(cell) for cell in cells],
        "item_circuit": item_circuit,
        "item_cell": item_cell,
        "errors": errors,
        "flags": flags,
        "flags_secondary": flags_secondary,
        "point_mae": point_mae,
    }


def _draws(n_seeds: int, n_circuits: int, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    seed_counts = np.zeros((draws, n_seeds))
    circuit_counts = np.zeros((draws, n_circuits))
    for b in range(draws):
        seed_counts[b] = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
        circuit_counts[b] = np.bincount(
            rng.integers(0, n_circuits, n_circuits), minlength=n_circuits)
    return seed_counts, circuit_counts


def _macro_under_weights(errors: np.ndarray, item_circuit: np.ndarray,
                         item_cell: np.ndarray, n_cells: int,
                         circuit_counts: np.ndarray) -> np.ndarray:
    """(draws x seeds) macro MAE for item errors (seeds x items)."""
    total = np.zeros((circuit_counts.shape[0], errors.shape[0]))
    for cell in range(n_cells):
        members = np.flatnonzero(item_cell == cell)
        weights = circuit_counts[:, item_circuit[members]]          # draws x n_cell
        total += (weights @ errors[:, members].T) / weights.sum(axis=1, keepdims=True)
    return total / n_cells


def _interval(values: np.ndarray) -> dict:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper),
            "excludes_zero_above": bool(lower > 0.0),
            "excludes_zero_below": bool(upper < 0.0)}


def bootstrap_row(inputs: dict, seeds: list[int], *, draws: int, seed: int,
                  subset: dict[str, np.ndarray] | None = None) -> dict:
    """Two-stage intervals for each quantity; `subset` restricts seeds per quantity."""
    n_cells = len(inputs["cells"])
    results = {}
    for name, (first, second) in QUANTITIES.items():
        keep = (np.ones(len(seeds), dtype=bool) if subset is None else subset[name])
        kept = np.flatnonzero(keep)
        if kept.size == 0:
            results[name] = {"status": "not_estimable", "reason": "no unflagged seeds",
                             "n_seeds": 0}
            continue
        seed_counts, circuit_counts = _draws(kept.size, inputs["n_circuits"], draws, seed)
        macro = {
            arm: _macro_under_weights(
                inputs["errors"][arm][kept], inputs["item_circuit"], inputs["item_cell"],
                n_cells, circuit_counts)
            for arm in (first, second)}
        difference = macro[first] - macro[second]                       # draws x seeds
        two_stage = np.sum(seed_counts * difference, axis=1) / kept.size
        circuit_only = difference.mean(axis=1)
        point_per_seed = inputs["point_mae"][first][kept] - inputs["point_mae"][second][kept]
        seed_only = (seed_counts @ point_per_seed) / kept.size
        results[name] = {
            "status": "estimated",
            "n_seeds": int(kept.size),
            "learner_seeds": [seeds[i] for i in kept],
            "point_mean": float(point_per_seed.mean()),
            "two_stage_interval": _interval(two_stage),
            "two_stage_draw_sd": float(two_stage.std(ddof=1)),
            "diagnostic_circuit_only_interval": _interval(circuit_only),
            "diagnostic_seed_only_interval": _interval(seed_only),
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file() and (run_dir / "inputs" / "summary.json").is_file():
        summary_path = run_dir / "inputs" / "summary.json"
    elif not summary_path.is_file() and (run_dir.parent / "summary.json").is_file():
        summary_path = run_dir.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in summary["learner_seeds"]]
    if summary["fits_missing"]:
        raise SystemExit(f"fits missing: {summary['fits_missing'][:5]} ...")
    keys = sorted(summary["dataset_checks"])

    result = {
        "schema": "two-stage-bootstrap-v1",
        "frozen_rule": "frozen-rule-seed-replication.md (2026-10-01)",
        "draws": args.draws,
        "seed": args.seed,
        "rng_scheme": (
            "fresh numpy default_rng(seed) per row and per quantity; per draw "
            "integers(0, n_seeds, n_seeds) then integers(0, n_circuits, n_circuits)"
        ),
        "percentiles": [2.5, 97.5],
        "learner_seeds": seeds,
        "input_summary_sha256": _sha256(summary_path),
        "rows": {},
    }
    for key in keys:
        for family in FAMILIES:
            row = f"{key}/{family}"
            inputs = _row_inputs(run_dir, key, family, seeds)
            flags = inputs["flags"]
            flags2 = inputs["flags_secondary"]

            def subset(flag_map):
                return {name: ~(flag_map[a] | flag_map[b])
                        for name, (a, b) in QUANTITIES.items()}

            result["rows"][row] = {
                "setting": key,
                "family": family,
                "n_test_items": inputs["n_items"],
                "n_test_circuits": inputs["n_circuits"],
                "cells": inputs["cells"],
                "all_fits": bootstrap_row(inputs, seeds, draws=args.draws, seed=args.seed),
                "sensitivity_excluding_flagged_literal": bootstrap_row(
                    inputs, seeds, draws=args.draws, seed=args.seed,
                    subset=subset(flags)),
                "sensitivity_excluding_flagged_secondary": bootstrap_row(
                    inputs, seeds, draws=args.draws, seed=args.seed,
                    subset=subset(flags2)),
                "flag_counts_literal": {arm: int(flags[arm].sum()) for arm in ARMS},
                "flag_counts_secondary": {arm: int(flags2[arm].sum()) for arm in ARMS},
            }
            print(f"{row}: mean D {result['rows'][row]['all_fits']['D']['point_mean']:.6f} "
                  f"{result['rows'][row]['all_fits']['D']['two_stage_interval']}",
                  flush=True)

    supporting = [
        row for row, value in result["rows"].items()
        if value["all_fits"]["D"]["two_stage_interval"]["excludes_zero_above"]]
    count = len(supporting)
    if count >= 5:
        outcome = "keep_form_with_seed_averaged_numbers"
        sentence = ("At least five rows support a positive capacity-matched gap, so "
                    "Finding 2 keeps its form with the seed-averaged numbers.")
    elif count >= 1:
        outcome = "rewrite_with_seed_averaged_result_and_count"
        sentence = (f"{count} of 6 rows support a positive capacity-matched gap, so "
                    "Finding 2 is rewritten to report the seed-averaged result and the "
                    "count of supporting rows.")
    else:
        outcome = "remove_from_abstract"
        sentence = ("No row supports a positive capacity-matched gap, so Finding 2 "
                    "leaves the abstract and is replaced by the statement that the gap "
                    "is not distinguishable from learner-initialization variance.")
    result["decision"] = {
        "rule": "a row supports a positive gap only if its two-stage 95% interval "
                "for mean D lies wholly above zero; decision uses all fits",
        "supporting_rows": supporting,
        "n_supporting": count,
        "outcome": outcome,
        "sentence": sentence,
    }
    for label in ("literal", "secondary"):
        key_name = f"sensitivity_excluding_flagged_{label}"
        support = [
            row for row, value in result["rows"].items()
            if value[key_name]["D"].get("status") == "estimated"
            and value[key_name]["D"]["two_stage_interval"]["excludes_zero_above"]]
        estimable = [row for row, value in result["rows"].items()
                     if value[key_name]["D"].get("status") == "estimated"]
        result[f"sensitivity_{label}"] = {
            "n_rows_estimable": len(estimable),
            "supporting_rows": support,
            "n_supporting": len(support),
        }
    result["seconds"] = time.perf_counter() - started
    out = args.out or (run_dir / "bootstrap.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"decision: {sentence}", flush=True)
    print(f"wrote {out} in {result['seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
