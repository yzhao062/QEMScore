#!/usr/bin/env python3
"""Stacking increment beside D for every fit set and shot level (post hoc).

Post hoc and descriptive; it decides nothing. For each fit set given with
``--set`` (for example the original, strength-indicator, and strong-learner
fits, and each level of the shot sweep), each Part A row, and each rung with
fits, ``tools/measurement_floor.compute_stacking`` fits, per
strength-by-observable cell and learner seed, a linear recalibration of C's
prediction (y ~ a + b * Chat) and a stack with the noisy estimate
(y ~ a + b * Chat + c * r) on the validation rows, and scores both on the test
rows. The stacking increment is the recalibration's macro error minus the
stack's; the relative increment divides it by the recalibration's macro error.
Intervals are that function's two-stage bootstrap (seed 20261002, 10,000
draws). Beside each cell the script copies D, D/C, and the label from the fit
set's analysis A.

The stack reads r through one fitted coefficient per cell. A positive
increment where F reads "not distinguished" or "C beats F" shows that r carries
information that the fitted F did not use. It does not bound what the best use
of r attains.

Usage:
    PYTHONPATH=. python tools/stacking_increment_all.py \
        --set original FITS DATA CACHE artifacts/descriptor-information/analysis-a.json \
        --set sweep-exact FITS DATA CACHE artifacts/.../levels/exact/analysis-a.json \
        --pins artifacts/descriptor-information/posthoc-stacking-pins.json \
        --out artifacts/descriptor-information/posthoc-stacking-all.json

DATA is a directory holding ``regen-shipped-s<seed>-n640/items.jsonl`` and its
``manifest.json`` for the three dataset seeds, the items whose order the fits'
prediction arrays follow. CACHE holds the ``shipped-s<seed>-n640.pkl`` caches
analysis A read. Before computing anything, the script requires each manifest
to hash its items and its dataset hash to equal the one analysis A, its cache,
and every C and F fit record, so fits from one shot level cannot be paired
with data from another. It also requires the data's validation and test rows
to follow the cache's prediction order, which the item hash cannot check.
Each C and F fit read must name its own dataset seed, rung, learner seed, and
arm, and each of its prediction arrays must reproduce the per-family error the
fit recorded for that split. Before attaching analysis A's labels, the script
requires each cell's recomputed mean C and F test errors to equal analysis
A's. These are consistency checks against each fit's own record and analysis
A: they catch mispaired, reordered, or exchanged files. ``--pins`` names the
file ``tools/stacking_pins.py`` wrote from the published release archives.
Before any other check, every consumed C and F record and prediction array
must have the SHA-256 pinned there, and analysis A must be the pinned one, so
any edited or reordered fit file is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO) not in sys.path[:2]:
    sys.path.insert(1, str(_REPO))

import measurement_floor as mf  # noqa: E402

import numpy as np  # noqa: E402

from qemscore.campaign.analysis import _family_mae as family_mae  # noqa: E402

SCHEMA = "posthoc-stacking-all-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path) -> str:
    """A repository path relative to the repository; any other path as given."""
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def rungs_with_fits(fits_dir: Path) -> dict[str, tuple[str, ...]]:
    """The Part A rungs of each family whose first C fit exists for every dataset seed."""
    out = {}
    for fam, rungs in mf.RUNGS_PER_FAMILY.items():
        out[fam] = tuple(
            rung for rung in rungs
            if all((fits_dir / f"shipped-s{seed}-n640__{rung}__k01__C.json").exists()
                   for seed in mf.SEEDS))
    if not any(out.values()):
        raise SystemExit(f"{fits_dir}: no Part A C fits found")
    return out


def fits_digest(fits_dir: Path, rungs: dict[str, tuple[str, ...]]) -> str:
    """SHA-256 over the C and F fit files the stacking reads, in a fixed order."""
    digest = hashlib.sha256()
    names = sorted({
        f"shipped-s{seed}-n640__{rung}__k{k:02d}__{arm}.{ext}"
        for fam_rungs in rungs.values() for rung in fam_rungs
        for seed in mf.SEEDS for k in range(1, 21)
        for arm in ("C", "F") for ext in ("json", "npz")})
    for name in names:
        digest.update(name.encode())
        digest.update(sha256_file(fits_dir / name).encode())
    return digest.hexdigest()


def analysis_cell(analysis: dict, row: str, rung: str) -> dict:
    cell = analysis["parts"]["A"]["rows"][row]["rungs"][rung]
    d_over_c = cell["D_over_C"]
    return {
        "D": cell["D"]["point"],
        "D_interval": cell["D"]["interval"],
        "D_over_C": d_over_c["point"],
        "D_over_C_interval": d_over_c["interval"],
        "label": cell["classification"]["label"],
    }


def verify_inputs(fits_dir: Path, data_dir: Path, cache_dir: Path, analysis: dict,
                  rungs: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Bind the data, the cache, the fits, and analysis A per dataset seed.

    For each dataset seed, the data's manifest must hash its own items, and its
    dataset hash must equal the one analysis A records for that cache and the
    one every C and F fit read records. Validation MAE alone cannot catch a
    wrong shot level: C, the labels, and the row order stay the same while r
    changes. The item hash sorts items by identity, so it cannot catch
    reordered rows either: the cache analysis A read (checked by SHA-256)
    holds the validation and test rows in the order of the fits' prediction
    arrays, and the data's rows must follow that order item by item. Each C
    and F fit the stacking reads must then pass ``verify_fit``. Returns the
    verified dataset hash per dataset seed.
    """
    from qemscore.validation import item_stream_hash, split_dataset_hash

    caches_used = analysis["inputs"]["caches_used"]
    verified = {}
    for seed in mf.SEEDS:
        key = f"shipped-s{seed}-n640"
        item_dir = data_dir / f"regen-{key}"
        manifest = json.loads((item_dir / "manifest.json").read_text(encoding="utf-8"))
        with open(item_dir / "items.jsonl", "r", encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if manifest["items_hash"] != item_stream_hash(items):
            raise SystemExit(f"{item_dir}: manifest items_hash does not match the items")
        if manifest["dataset_hash"] != split_dataset_hash(
                spec_hash=manifest["split_spec_hash"], items_hash=manifest["items_hash"]):
            raise SystemExit(f"{item_dir}: manifest dataset_hash does not match its hashes")
        dataset_hash = str(manifest["dataset_hash"])
        if caches_used[key]["dataset_hash"] != dataset_hash:
            raise SystemExit(f"{item_dir}: dataset hash {dataset_hash} differs from the "
                             f"{caches_used[key]['dataset_hash']} analysis A used for {key}")
        cache_path = cache_dir / f"{key}.pkl"
        if sha256_file(cache_path) != caches_used[key]["sha256"]:
            raise SystemExit(f"{cache_path}: SHA-256 differs from the cache analysis A used")
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
        if cache.get("dataset_hash") != dataset_hash:
            raise SystemExit(f"{cache_path}: cache dataset hash differs from the data's")
        for split in ("validation", "test"):
            data_ids = [str(item["item_id"]) for item in items if item.get("split") == split]
            cache_ids = [str(row["item_id"]) for row in cache[split]]
            if data_ids != cache_ids:
                raise SystemExit(f"{item_dir}: {split} rows differ from the cache's prediction "
                                 f"order")
        rows = {split: [item for item in items if item.get("split") == split]
                for split in ("validation", "test")}
        for fam_rungs in rungs.values():
            for rung in fam_rungs:
                for k in range(1, 21):
                    for arm in ("C", "F"):
                        verify_fit(fits_dir / f"{key}__{rung}__k{k:02d}__{arm}.json",
                                   key, seed, rung, k, arm, dataset_hash, rows)
        verified[str(seed)] = dataset_hash
    return verified


def verify_fit(fit: Path, key: str, seed: int, rung: str, k: int, arm: str,
               dataset_hash: str, rows: dict[str, list[dict]], tol: float = 1e-12) -> None:
    """Require one fit's record and prediction arrays to belong to its slot.

    The record must name the slot's dataset, dataset seed, rung, learner seed,
    and arm, and the dataset hash of the data. Each prediction array the
    stacking reads must reproduce the per-family error that the fit recorded
    for that split when it was fitted. Prediction arrays exchanged between
    learner seeds, or reordered, fail this check even where cell means agree.
    """
    meta = json.loads(fit.read_text(encoding="utf-8"))
    expected = {"key": key, "dataset_seed": seed, "rung": rung, "arm": arm,
                "learner_seed": k, "dataset_hash": dataset_hash}
    for field, value in expected.items():
        if meta.get(field) != value:
            raise SystemExit(f"{fit}: fit {field} {meta.get(field)!r} differs from "
                             f"{value!r}")
    arrays = np.load(fit.with_suffix(".npz"))
    for split in ("validation", "test"):
        reproduced = family_mae(rows[split], arrays[split], artifact_id=dataset_hash)
        for fam, value in reproduced.items():
            diff = abs(meta[f"{split}_family_mae"][fam] - value)
            if diff > tol:
                raise SystemExit(f"{fit}: {split} predictions give {fam} error {value}, "
                                 f"not the recorded {split}_family_mae (difference {diff})")


def verify_against_analysis(stacking: dict, analysis: dict, tol: float = 1e-12) -> float:
    """Require each cell's recomputed mean C and F test errors to equal analysis A's.

    The stacking reads C's validation and test predictions and F's test
    predictions. ``compute_stacking`` already ties C's validation predictions
    to the validation MAE each fit records. This check ties the test
    predictions of both arms to the analysis whose labels are attached: test
    predictions in another order, or fits of another pipeline, change these
    means. Returns the largest difference.
    """
    rows = analysis["parts"]["A"]["rows"]
    largest = 0.0
    for key, entry in stacking.items():
        row, rung = key.rsplit("/", 1)
        means = rows[row]["rungs"][rung]["means"]
        for field, arm in (("mean_C", "C"), ("F", "F")):
            diff = abs(entry[field] - means[arm]["point"])
            if diff > tol:
                raise SystemExit(f"{key}: recomputed mean {arm} test error differs from "
                                 f"analysis A's by {diff}")
            largest = max(largest, diff)
    return largest


def verify_pins(name: str, fits_dir: Path, rungs: dict[str, tuple[str, ...]],
                analysis_path: Path, pins: dict) -> str:
    """Require every consumed fit file to equal the one pinned from the release asset.

    ``tools/stacking_pins.py`` read the pins from the published release
    archives. A fit set must select the pinned rungs, the analysis A must be
    the pinned one, and each consumed C and F record and prediction array must
    have the pinned SHA-256, so a reordered or edited file cannot reach a
    record. Returns the verified fit digest.
    """
    if name not in pins["sets"]:
        raise SystemExit(f"[{name}] no pinned fit set of that name")
    pin = pins["sets"][name]
    if {fam: list(r) for fam, r in rungs.items()} != pin["rungs"]:
        raise SystemExit(f"[{name}] {fits_dir}: rungs differ from the pinned rungs")
    if sha256_file(analysis_path) != pin["analysis_a_sha256"]:
        raise SystemExit(f"[{name}] {analysis_path}: SHA-256 differs from the pinned analysis A")
    for file_name, expected in pin["files_sha256"].items():
        if sha256_file(fits_dir / file_name) != expected:
            raise SystemExit(f"[{name}] {fits_dir / file_name}: SHA-256 differs from the "
                             f"file pinned from {pin['asset']}")
    digest = fits_digest(fits_dir, rungs)
    if digest != pin["fits_digest"]:
        raise SystemExit(f"[{name}] {fits_dir}: fit digest differs from the pinned digest")
    return digest


def run_set(name: str, fits_dir: Path, data_dir: Path, cache_dir: Path,
            analysis_path: Path, pins: dict) -> dict:
    rungs = rungs_with_fits(fits_dir)
    pinned_digest = verify_pins(name, fits_dir, rungs, analysis_path, pins)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    dataset_hashes = verify_inputs(fits_dir, data_dir, cache_dir, analysis, rungs)
    datasets = mf.load_datasets(data_dir)
    _, row_summary = mf.compute_floors(datasets)
    stacking = mf.compute_stacking(fits_dir, datasets, row_summary, rungs_per_family=rungs)
    analysis_max_abs_diff = verify_against_analysis(stacking, analysis)
    cells = {}
    for key, entry in sorted(stacking.items()):
        row, rung = key.rsplit("/", 1)
        cells[key] = {**entry, "analysis_a": analysis_cell(analysis, row, rung)}
    return {
        "fits_dir": display(fits_dir),
        "data_dir": display(data_dir),
        "cache_dir": display(cache_dir),
        "dataset_hash_verified": dataset_hashes,
        "mean_C_and_F_match_analysis_a_max_abs_diff": analysis_max_abs_diff,
        "fits_digest": pinned_digest,
        "fits_match_pins": True,
        "data_items_sha256": {
            str(seed): sha256_file(data_dir / f"regen-shipped-s{seed}-n640" / "items.jsonl")
            for seed in mf.SEEDS},
        "analysis_a": display(analysis_path),
        "analysis_a_sha256": sha256_file(analysis_path),
        "rungs": {fam: list(r) for fam, r in rungs.items()},
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", nargs=5, action="append", required=True,
                        metavar=("NAME", "FITS", "DATA", "CACHE", "ANALYSIS_A"))
    parser.add_argument("--pins", type=Path, required=True,
                        help="fit-file pins written by tools/stacking_pins.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    names = [s[0] for s in args.set]
    if len(set(names)) != len(names):
        raise SystemExit("--set names must be unique")
    payload = {
        "schema": SCHEMA,
        "status": "post hoc, descriptive",
        "script": "tools/stacking_increment_all.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "measurement_floor_sha256": sha256_file(Path(mf.__file__).resolve()),
        "bootstrap": {"seed": mf.RULE_SEED, "draws": mf.BOOTSTRAP_DRAWS},
        "pins": display(args.pins),
        "pins_sha256": sha256_file(args.pins),
        "sets": {},
    }
    for name, fits, data, cache, analysis in args.set:
        print(f"[{name}] {fits}", flush=True)
        payload["sets"][name] = run_set(name, Path(fits), Path(data), Path(cache),
                                         Path(analysis), pins)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
