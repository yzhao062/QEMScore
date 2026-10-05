#!/usr/bin/env python3
"""Measurement-free stacking of strong-learner reference predictions and noisy measurement (Part A).

Follows rule 2026-10-04-crossed-follow-ups.md (part E).
Descriptive; it decides nothing. The rule was frozen before the tool first ran on real fits.

For each shot level given with --c-fits, --data, --cache, and --analysis-a,
each Part A row (dataset seed x family), and each specified rung, fits on the
validation rows and scores on the test rows, by np.linalg.lstsq with an intercept,
four models:
    recal    : y ~ C_hat
    stack_r  : y ~ C_hat + r
    stack_g  : y ~ C_hat + g
    stack_gr : y ~ C_hat + g + r
where:
    C_hat is the level's C fit prediction;
    g is the poly5_ridge prediction of the strong C fit with the same dataset seed,
      rung, and learner seed;
    r is the level's noisy estimate.

Computes increments per learner seed (macro over the row's four cells):
    inc_r    = recal - stack_r
    inc_g    = recal - stack_g
    inc_r|g  = stack_g - stack_gr
Relative versions divide by recal, recal, and stack_g.
Point values are seed means (relative = mean increment / mean denominator).
Intervals are the two-stage bootstrap of measurement_floor.compute_stacking
(dla.bootstrap_draws(20, n_circuits, 10000, 20261002), one call per row and level),
with relative intervals as ratios inside each draw.

Checks:
    - inc_r must reproduce the matching cell of --stacking-all within 1e-12 for
      every level present in that file;
    - g must be identical across learner seeds (reports the largest difference;
      if nonzero, uses each seed's own g and records that).

Usage:
    PYTHONPATH=. python tools/measurement_free_stack.py \\
        --c-fits 256=RUNS/shots-256/fits ... \\
        --data 256=LEVELS/shots-256 ... \\
        --cache 256=RUNS/shots-256/cache ... \\
        --analysis-a 256=ARTIFACTS/levels/256/analysis-a.json ... \\
        --g-fits ASSETS/strong-learners-v1/partA-fits \\
        --stacking-all ARTIFACTS/posthoc-stacking-all.json \\
        --rungs R0 N1 N2 \\
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
from qemscore.campaign.analysis import _family_mae as family_mae  # noqa: E402
from qemscore.validation import item_stream_hash, split_dataset_hash  # noqa: E402

SCHEMA = "measurement-free-stack-v1"
RULE_STATUS = "follows rule 2026-10-04-crossed-follow-ups.md (part E)"
RULE_SEED = dla.RULE_SEED  # 20261002
BOOTSTRAP_DRAWS = dla.DEFAULT_DRAWS  # 10_000
SEEDS = mf.SEEDS  # (101, 211, 307)
FAMILIES = mf.FAMILIES  # ("tfi", "heisenberg")
SEVERITIES = mf.SEVERITIES  # ("L1", "L3")
OBSERVABLES = mf.OBSERVABLES  # ("z_mid", "zz_mid")
CELLS_DEF = tuple((s, o) for s in SEVERITIES for o in OBSERVABLES)
DEFAULT_RUNGS = ("R0", "N1", "N2")
REFERENCE_FIELDS = (
    ("increment_mean",),
    ("increment_interval", "lower"),
    ("increment_interval", "upper"),
    ("relative_increment", "point"),
    ("relative_increment", "interval", "lower"),
    ("relative_increment", "interval", "upper"),
)


def require_reference_sets(stacking_all_data: dict, levels: list[str], rungs: list[str]) -> dict[str, dict]:
    """The inc_r reference for every requested level, row, and rung, or SystemExit.

    Level ell must have set `sweep-<ell>` in posthoc-stacking-all.json, and every cell
    `shipped-s<seed>-n640/<family>/<rung>` must carry the six values inc_r is checked against.
    """
    sets = stacking_all_data.get("sets", {})
    out, problems = {}, []
    for lvl in levels:
        name = f"sweep-{normalize_level_name(lvl)}"
        stack_set = sets.get(name)
        if stack_set is None:
            problems.append(f"{name}: set missing")
            continue
        for seed in SEEDS:
            for fam in FAMILIES:
                for rung in rungs:
                    cell_key = f"shipped-s{seed}-n640/{fam}/{rung}"
                    cell = stack_set.get("cells", {}).get(cell_key)
                    if cell is None:
                        problems.append(f"{name}/{cell_key}: cell missing")
                        continue
                    for path in REFERENCE_FIELDS:
                        value = cell
                        for part in path:
                            value = value.get(part) if isinstance(value, dict) else None
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            problems.append(f"{name}/{cell_key}: {'.'.join(path)} missing")
                        elif not math.isfinite(value):
                            problems.append(f"{name}/{cell_key}: {'.'.join(path)} is not finite")
        out[lvl] = stack_set
    if problems:
        raise SystemExit("--stacking-all lacks reference values; nothing computed:\n  "
                         + "\n  ".join(problems[:20]))
    return out



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path | str) -> str:
    """A repository path relative to the repository; any other path as given."""
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


def fits_digest_c(fits_dir: Path, rungs: list[str], seeds: tuple[int, ...] = SEEDS) -> str:
    """SHA-256 over the C fit files read, in a fixed order."""
    digest = hashlib.sha256()
    names = sorted({
        f"shipped-s{seed}-n640__{rung}__k{k:02d}__C.{ext}"
        for rung in rungs
        for seed in seeds
        for k in range(1, 21)
        for ext in ("json", "npz")
    })
    for name in names:
        p = fits_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing")
        digest.update(name.encode())
        digest.update(sha256_file(p).encode())
    return digest.hexdigest()


def verify_inputs(
    fits_dir: Path,
    data_dir: Path,
    cache_dir: Path,
    analysis: dict,
    rungs: list[str],
    tol: float = 1e-12,
    dry_run: bool = False,
) -> dict[str, str]:
    """Bind data, cache, C fits, and analysis A per dataset seed."""
    caches_used = analysis["inputs"]["caches_used"]
    verified: dict[str, str] = {}
    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        item_dir = data_dir / f"regen-{key}"
        manifest_path = item_dir / "manifest.json"
        items_path = item_dir / "items.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not items_path.exists():
            raise FileNotFoundError(f"Missing items: {items_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with open(items_path, "r", encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]

        if manifest["items_hash"] != item_stream_hash(items):
            raise SystemExit(f"{item_dir}: manifest items_hash does not match items")
        if manifest["dataset_hash"] != split_dataset_hash(
            spec_hash=manifest["split_spec_hash"], items_hash=manifest["items_hash"]
        ):
            raise SystemExit(f"{item_dir}: manifest dataset_hash does not match its hashes")

        dataset_hash = str(manifest["dataset_hash"])
        if key not in caches_used:
            raise SystemExit(f"{key} not found in analysis A caches_used")
        if caches_used[key]["dataset_hash"] != dataset_hash:
            raise SystemExit(
                f"{item_dir}: dataset hash {dataset_hash} differs from "
                f"{caches_used[key]['dataset_hash']} analysis A used for {key}"
            )

        cache_path = cache_dir / f"{key}.pkl"
        if not cache_path.exists():
            cache_path = cache_dir / f"{key}.json.gz"
        if not cache_path.exists():
            cache_path = cache_dir / f"{key}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file for {key} not found in {cache_dir}")

        if sha256_file(cache_path) != caches_used[key]["sha256"]:
            raise SystemExit(f"{cache_path}: SHA-256 differs from cache analysis A used")

        if cache_path.name.endswith(".pkl"):
            with open(cache_path, "rb") as handle:
                cache = pickle.load(handle)
        else:
            cache = dla.load_cache(cache_path)

        if cache.get("dataset_hash") != dataset_hash:
            raise SystemExit(f"{cache_path}: cache dataset hash differs from data")

        for split in ("validation", "test"):
            data_ids = [str(item["item_id"]) for item in items if item.get("split") == split]
            cache_ids = [str(row["item_id"]) for row in cache[split]]
            if data_ids != cache_ids:
                raise SystemExit(
                    f"{item_dir}: {split} rows differ from cache's prediction order"
                )

        rows = {
            split: [item for item in items if item.get("split") == split]
            for split in ("validation", "test")
        }

        for rung in rungs:
            for k in range(1, 21):
                fit_stem = fits_dir / f"{key}__{rung}__k{k:02d}__C"
                json_path = fit_stem.with_suffix(".json")
                npz_path = fit_stem.with_suffix(".npz")
                if not json_path.exists():
                    raise FileNotFoundError(f"Missing C fit json: {json_path}")
                if not npz_path.exists():
                    raise FileNotFoundError(f"Missing C fit npz: {npz_path}")

                meta = json.loads(json_path.read_text(encoding="utf-8"))
                expected = {
                    "key": key,
                    "dataset_seed": seed,
                    "rung": rung,
                    "arm": "C",
                    "learner_seed": k,
                    "dataset_hash": dataset_hash,
                }
                for field, val in expected.items():
                    if meta.get(field) != val:
                        raise SystemExit(
                            f"{json_path}: fit {field} {meta.get(field)!r} differs from {val!r}"
                        )

                if not dry_run:
                    arrays = np.load(npz_path)
                    for split in ("validation", "test"):
                        reproduced = family_mae(rows[split], arrays[split], artifact_id=dataset_hash)
                        for fam, value in reproduced.items():
                            diff = abs(meta[f"{split}_family_mae"][fam] - value)
                            if diff > tol:
                                raise SystemExit(
                                    f"{json_path}: {split} predictions give {fam} error {value}, "
                                    f"not recorded {split}_family_mae (difference {diff})"
                                )
        verified[str(seed)] = dataset_hash
    return verified


def verify_strong_fits(
    g_fits_dir: Path,
    rungs: list[str],
    dataset_hashes_2048: dict[str, str],
    cache_row_counts: dict[int, dict[str, int]],
    dry_run: bool = False,
) -> None:
    """Verify strong learner rerun C fits match 2,048-shot data and row counts."""
    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        exp_hash = dataset_hashes_2048[str(seed)]
        val_count = cache_row_counts[seed]["validation"]
        test_count = cache_row_counts[seed]["test"]

        for rung in rungs:
            for k in range(1, 21):
                fit_stem = g_fits_dir / f"{key}__{rung}__k{k:02d}__C"
                json_path = fit_stem.with_suffix(".json")
                npz_path = fit_stem.with_suffix(".npz")
                if not json_path.exists():
                    raise FileNotFoundError(f"Missing g fit json: {json_path}")
                if not npz_path.exists():
                    raise FileNotFoundError(f"Missing g fit npz: {npz_path}")

                meta = json.loads(json_path.read_text(encoding="utf-8"))
                expected = {
                    "key": key,
                    "dataset_seed": seed,
                    "rung": rung,
                    "arm": "C",
                    "learner_seed": k,
                    "dataset_hash": exp_hash,
                }
                for field, val in expected.items():
                    if meta.get(field) != val:
                        raise SystemExit(
                            f"{json_path}: strong fit {field} {meta.get(field)!r} differs from {val!r}"
                        )

                # Check arrays in NPZ
                with np.load(npz_path) as npz:
                    if "validation__poly5_ridge" not in npz.files:
                        raise SystemExit(
                            f"{npz_path}: missing 'validation__poly5_ridge' array"
                        )
                    if "test__poly5_ridge" not in npz.files:
                        raise SystemExit(f"{npz_path}: missing 'test__poly5_ridge' array")

                    v_shape = npz["validation__poly5_ridge"].shape
                    t_shape = npz["test__poly5_ridge"].shape
                    if v_shape != (val_count,):
                        raise SystemExit(
                            f"{npz_path}: validation__poly5_ridge shape {v_shape} != {(val_count,)}"
                        )
                    if t_shape != (test_count,):
                        raise SystemExit(
                            f"{npz_path}: test__poly5_ridge shape {t_shape} != {(test_count,)}"
                        )


def compute_level_stacks(
    lvl_name: str,
    c_fits_dir: Path,
    data_dir: Path,
    cache_dir: Path,
    g_fits_dir: Path,
    rungs: list[str],
    stacking_all_set: dict | None = None,
    check_reference: bool = True,
) -> tuple[dict[str, Any], float]:
    """Compute recal, stack_r, stack_g, stack_gr and increments for one shot level.

    With `check_reference` (always on in the CLI), every cell's inc_r must match its
    `stacking_all_set` cell to 1e-12; a missing cell stops the run. Tests on synthetic data
    pass `check_reference=False` to obtain the reference values themselves.
    """
    datasets = mf.load_datasets(data_dir)
    cells_out: dict[str, Any] = {}
    level_max_g_diff = 0.0

    for seed in SEEDS:
        key = f"shipped-s{seed}-n640"
        val_items = datasets[seed]["val"]
        test_items = datasets[seed]["test"]

        for fam in FAMILIES:
            row_key = f"{key}/{fam}"

            row_test_indices = [
                i for i, r in enumerate(test_items) if str(r["family"]) == fam
            ]
            row_test_items = [test_items[i] for i in row_test_indices]
            row_y_test = np.asarray(
                [r["ideal_expectation"] for r in row_test_items], dtype=float
            )
            row_r_test = np.asarray(
                [r["noisy_expectation"] for r in row_test_items], dtype=float
            )

            circuits = sorted({str(r["circuit_id"]) for r in row_test_items})
            circ_pos = {c: i for i, c in enumerate(circuits)}
            n_circ = len(circuits)
            row_item_circuit = np.asarray(
                [circ_pos[str(r["circuit_id"])] for r in row_test_items], dtype=int
            )

            row_members = [
                np.asarray(
                    [
                        i
                        for i, r in enumerate(row_test_items)
                        if r["severity"] == s and r["observable"] == o
                    ],
                    dtype=int,
                )
                for s, o in CELLS_DEF
            ]

            val_cell_idx = [
                [
                    i
                    for i, r in enumerate(val_items)
                    if r["family"] == fam and r["severity"] == s and r["observable"] == o
                ]
                for s, o in CELLS_DEF
            ]

            # One call to dla.bootstrap_draws per row and level, shared by quantities of row
            seed_counts, circ_counts = dla.bootstrap_draws(
                20, n_circ, BOOTSTRAP_DRAWS, RULE_SEED
            )

            for rung in rungs:
                cell_id = f"{lvl_name}/{row_key}/{rung}"

                # First check g stability across learner seeds 1..20
                g_val_seeds = []
                g_test_seeds = []
                for k in range(1, 21):
                    npz_g = np.load(g_fits_dir / f"{key}__{rung}__k{k:02d}__C.npz")
                    g_val_seeds.append(np.asarray(npz_g["validation__poly5_ridge"], dtype=float))
                    g_test_seeds.append(np.asarray(npz_g["test__poly5_ridge"], dtype=float))

                max_g_diff = 0.0
                for k_idx in range(1, 20):
                    diff_v = float(np.max(np.abs(g_val_seeds[k_idx] - g_val_seeds[0])))
                    diff_t = float(np.max(np.abs(g_test_seeds[k_idx] - g_test_seeds[0])))
                    max_g_diff = max(max_g_diff, diff_v, diff_t)
                level_max_g_diff = max(level_max_g_diff, max_g_diff)

                recal_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                stack_r_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                stack_g_test_seeds = np.empty((20, len(row_test_items)), dtype=float)
                stack_gr_test_seeds = np.empty((20, len(row_test_items)), dtype=float)

                macro_rec_seeds: list[float] = []
                macro_stack_r_seeds: list[float] = []
                macro_stack_g_seeds: list[float] = []
                macro_stack_gr_seeds: list[float] = []

                inc_r_seeds: list[float] = []
                inc_g_seeds: list[float] = []
                inc_rg_seeds: list[float] = []

                for k_idx, k in enumerate(range(1, 21)):
                    stem_c = f"{key}__{rung}__k{k:02d}__C"
                    npz_c = np.load(c_fits_dir / f"{stem_c}.npz")
                    c_val = np.asarray(npz_c["validation"], dtype=float)
                    c_t = np.asarray(npz_c["test"], dtype=float)[row_test_indices]

                    g_val = g_val_seeds[k_idx]
                    g_t = g_test_seeds[k_idx][row_test_indices]

                    recal_t = np.empty_like(c_t)
                    stack_r_t = np.empty_like(c_t)
                    stack_g_t = np.empty_like(c_t)
                    stack_gr_t = np.empty_like(c_t)

                    rec_cell_maes = []
                    sr_cell_maes = []
                    sg_cell_maes = []
                    sgr_cell_maes = []

                    for c_i, (sev, obs) in enumerate(CELLS_DEF):
                        v_idx = val_cell_idx[c_i]
                        t_idx_local = row_members[c_i]

                        y_v = np.asarray(
                            [val_items[i]["ideal_expectation"] for i in v_idx], dtype=float
                        )
                        r_v = np.asarray(
                            [val_items[i]["noisy_expectation"] for i in v_idx], dtype=float
                        )
                        c_v = c_val[v_idx]
                        g_v = g_val[v_idx]

                        y_t = row_y_test[t_idx_local]
                        r_t = row_r_test[t_idx_local]
                        c_t_sub = c_t[t_idx_local]
                        g_t_sub = g_t[t_idx_local]

                        ones_v = np.ones_like(c_v)
                        ones_t = np.ones_like(c_t_sub)

                        # 1. recal: y ~ C_hat
                        X_rec_v = np.column_stack([ones_v, c_v])
                        X_rec_t = np.column_stack([ones_t, c_t_sub])
                        beta_rec, _, _, _ = np.linalg.lstsq(X_rec_v, y_v, rcond=None)
                        p_rec = X_rec_t @ beta_rec
                        recal_t[t_idx_local] = p_rec

                        # 2. stack_r: y ~ C_hat + r
                        X_sr_v = np.column_stack([ones_v, c_v, r_v])
                        X_sr_t = np.column_stack([ones_t, c_t_sub, r_t])
                        beta_sr, _, _, _ = np.linalg.lstsq(X_sr_v, y_v, rcond=None)
                        p_sr = X_sr_t @ beta_sr
                        stack_r_t[t_idx_local] = p_sr

                        # 3. stack_g: y ~ C_hat + g
                        X_sg_v = np.column_stack([ones_v, c_v, g_v])
                        X_sg_t = np.column_stack([ones_t, c_t_sub, g_t_sub])
                        beta_sg, _, _, _ = np.linalg.lstsq(X_sg_v, y_v, rcond=None)
                        p_sg = X_sg_t @ beta_sg
                        stack_g_t[t_idx_local] = p_sg

                        # 4. stack_gr: y ~ C_hat + g + r
                        X_sgr_v = np.column_stack([ones_v, c_v, g_v, r_v])
                        X_sgr_t = np.column_stack([ones_t, c_t_sub, g_t_sub, r_t])
                        beta_sgr, _, _, _ = np.linalg.lstsq(X_sgr_v, y_v, rcond=None)
                        p_sgr = X_sgr_t @ beta_sgr
                        stack_gr_t[t_idx_local] = p_sgr

                        rec_cell_maes.append(
                            math.fsum(np.abs(p_rec - y_t).tolist()) / len(y_t)
                        )
                        sr_cell_maes.append(
                            math.fsum(np.abs(p_sr - y_t).tolist()) / len(y_t)
                        )
                        sg_cell_maes.append(
                            math.fsum(np.abs(p_sg - y_t).tolist()) / len(y_t)
                        )
                        sgr_cell_maes.append(
                            math.fsum(np.abs(p_sgr - y_t).tolist()) / len(y_t)
                        )

                    recal_test_seeds[k_idx] = recal_t
                    stack_r_test_seeds[k_idx] = stack_r_t
                    stack_g_test_seeds[k_idx] = stack_g_t
                    stack_gr_test_seeds[k_idx] = stack_gr_t

                    mrec = math.fsum(rec_cell_maes) / len(rec_cell_maes)
                    msr = math.fsum(sr_cell_maes) / len(sr_cell_maes)
                    msg = math.fsum(sg_cell_maes) / len(sg_cell_maes)
                    msgr = math.fsum(sgr_cell_maes) / len(sgr_cell_maes)

                    macro_rec_seeds.append(mrec)
                    macro_stack_r_seeds.append(msr)
                    macro_stack_g_seeds.append(msg)
                    macro_stack_gr_seeds.append(msgr)

                    inc_r_seeds.append(mrec - msr)
                    inc_g_seeds.append(mrec - msg)
                    inc_rg_seeds.append(msg - msgr)

                # Two-stage bootstrap
                err_rec = np.abs(recal_test_seeds - row_y_test[None, :])
                err_sr = np.abs(stack_r_test_seeds - row_y_test[None, :])
                err_sg = np.abs(stack_g_test_seeds - row_y_test[None, :])
                err_sgr = np.abs(stack_gr_test_seeds - row_y_test[None, :])

                rec_draws = dla.macro_under_weights(
                    err_rec, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )
                sr_draws = dla.macro_under_weights(
                    err_sr, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )
                sg_draws = dla.macro_under_weights(
                    err_sg, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )
                sgr_draws = dla.macro_under_weights(
                    err_sgr, row_item_circuit, row_members, (0, 1, 2, 3), circ_counts
                )

                rec_draws_mean = (rec_draws * seed_counts).sum(axis=1) / 20.0
                sr_draws_mean = (sr_draws * seed_counts).sum(axis=1) / 20.0
                sg_draws_mean = (sg_draws * seed_counts).sum(axis=1) / 20.0
                sgr_draws_mean = (sgr_draws * seed_counts).sum(axis=1) / 20.0

                mean_rec = math.fsum(macro_rec_seeds) / 20.0
                mean_sr = math.fsum(macro_stack_r_seeds) / 20.0
                mean_sg = math.fsum(macro_stack_g_seeds) / 20.0
                mean_sgr = math.fsum(macro_stack_gr_seeds) / 20.0

                # 1. inc_r = recal - stack_r
                incr_r_draws = rec_draws_mean - sr_draws_mean
                mean_inc_r = math.fsum(inc_r_seeds) / len(inc_r_seeds)
                ci_lo_r, ci_hi_r = np.percentile(incr_r_draws, dla.PERCENTILES)
                rel_lo_r, rel_hi_r = np.percentile(incr_r_draws / rec_draws_mean, dla.PERCENTILES)
                rel_point_r = mean_inc_r / mean_rec

                # 2. inc_g = recal - stack_g
                incr_g_draws = rec_draws_mean - sg_draws_mean
                mean_inc_g = math.fsum(inc_g_seeds) / len(inc_g_seeds)
                ci_lo_g, ci_hi_g = np.percentile(incr_g_draws, dla.PERCENTILES)
                rel_lo_g, rel_hi_g = np.percentile(incr_g_draws / rec_draws_mean, dla.PERCENTILES)
                rel_point_g = mean_inc_g / mean_rec

                # 3. inc_r|g = stack_g - stack_gr
                incr_rg_draws = sg_draws_mean - sgr_draws_mean
                mean_inc_rg = math.fsum(inc_rg_seeds) / len(inc_rg_seeds)
                ci_lo_rg, ci_hi_rg = np.percentile(incr_rg_draws, dla.PERCENTILES)
                rel_lo_rg, rel_hi_rg = np.percentile(incr_rg_draws / sg_draws_mean, dla.PERCENTILES)
                rel_point_rg = mean_inc_rg / mean_sg

                # Reproduction check against stacking-all (required for every cell)
                stack_key = f"{row_key}/{rung}"
                reprod_diffs = None
                if check_reference and (stacking_all_set is None
                                        or stack_key not in stacking_all_set.get("cells", {})):
                    raise SystemExit(f"{cell_id}: no --stacking-all reference cell {stack_key}")
                if check_reference:
                    ref_cell = stacking_all_set["cells"][stack_key]
                    d_mean = abs(mean_inc_r - ref_cell["increment_mean"])
                    d_lo = abs(ci_lo_r - ref_cell["increment_interval"]["lower"])
                    d_hi = abs(ci_hi_r - ref_cell["increment_interval"]["upper"])
                    d_rel_pt = abs(rel_point_r - ref_cell["relative_increment"]["point"])
                    d_rel_lo = abs(rel_lo_r - ref_cell["relative_increment"]["interval"]["lower"])
                    d_rel_hi = abs(rel_hi_r - ref_cell["relative_increment"]["interval"]["upper"])
                    diffs = (d_mean, d_lo, d_hi, d_rel_pt, d_rel_lo, d_rel_hi)
                    # Each difference is checked on its own: max() can pass over a NaN.
                    if not all(math.isfinite(d) and d <= 1e-12 for d in diffs):
                        raise SystemExit(
                            f"{cell_id}: inc_r reproduction difference {diffs} exceeds 1e-12 "
                            "or is not finite against --stacking-all"
                        )
                    max_d = max(diffs)
                    reprod_diffs = {
                        "increment_mean_diff": float(d_mean),
                        "interval_lower_diff": float(d_lo),
                        "interval_upper_diff": float(d_hi),
                        "relative_point_diff": float(d_rel_pt),
                        "relative_lower_diff": float(d_rel_lo),
                        "relative_upper_diff": float(d_rel_hi),
                        "max_difference": float(max_d),
                    }

                entry = {
                    "level": lvl_name,
                    "row": row_key,
                    "rung": rung,
                    "dataset_seed": seed,
                    "family": fam,
                    "recal": {
                        "mean": float(mean_rec),
                        "min": float(min(macro_rec_seeds)),
                        "max": float(max(macro_rec_seeds)),
                    },
                    "stack_r": {
                        "mean": float(mean_sr),
                        "min": float(min(macro_stack_r_seeds)),
                        "max": float(max(macro_stack_r_seeds)),
                    },
                    "stack_g": {
                        "mean": float(mean_sg),
                        "min": float(min(macro_stack_g_seeds)),
                        "max": float(max(macro_stack_g_seeds)),
                    },
                    "stack_gr": {
                        "mean": float(mean_sgr),
                        "min": float(min(macro_stack_gr_seeds)),
                        "max": float(max(macro_stack_gr_seeds)),
                    },
                    "inc_r": {
                        "point": float(mean_inc_r),
                        "min": float(min(inc_r_seeds)),
                        "max": float(max(inc_r_seeds)),
                        "interval": {"lower": float(ci_lo_r), "upper": float(ci_hi_r)},
                        "relative": {
                            "point": float(rel_point_r),
                            "interval": {"lower": float(rel_lo_r), "upper": float(rel_hi_r)},
                        },
                    },
                    "inc_g": {
                        "point": float(mean_inc_g),
                        "min": float(min(inc_g_seeds)),
                        "max": float(max(inc_g_seeds)),
                        "interval": {"lower": float(ci_lo_g), "upper": float(ci_hi_g)},
                        "relative": {
                            "point": float(rel_point_g),
                            "interval": {"lower": float(rel_lo_g), "upper": float(rel_hi_g)},
                        },
                    },
                    "inc_r_given_g": {
                        "point": float(mean_inc_rg),
                        "min": float(min(inc_rg_seeds)),
                        "max": float(max(inc_rg_seeds)),
                        "interval": {"lower": float(ci_lo_rg), "upper": float(ci_hi_rg)},
                        "relative": {
                            "point": float(rel_point_rg),
                            "interval": {"lower": float(rel_lo_rg), "upper": float(rel_hi_rg)},
                        },
                    },
                    "reproduction_differences": reprod_diffs,
                    "g_max_seed_diff": float(max_g_diff),
                    "g_identical_across_seeds": bool(max_g_diff == 0.0),
                }
                cells_out[cell_id] = entry

    return cells_out, level_max_g_diff




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--c-fits",
        action="extend",
        nargs="+",
        required=True,
        metavar="LEVEL=DIR",
        help="C fit directories per shot level",
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
        "--analysis-a",
        action="extend",
        nargs="+",
        required=True,
        metavar="LEVEL=PATH",
        help="Analysis A JSON paths per shot level",
    )
    parser.add_argument(
        "--g-fits",
        type=Path,
        required=True,
        help="Directory of 2,048-shot strong-learner C fits",
    )
    parser.add_argument(
        "--stacking-all",
        type=Path,
        required=True,
        help="Path to posthoc-stacking-all.json",
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
        help="Verify inputs exist and match hashes without computing stacks or increments",
    )

    args = parser.parse_args(argv)

    c_fits_map = parse_key_value_pairs(args.c_fits)
    data_map = parse_key_value_pairs(args.data)
    cache_map = parse_key_value_pairs(args.cache)
    analysis_a_map = parse_key_value_pairs(args.analysis_a)

    all_levels = sorted(c_fits_map.keys())
    if sorted(data_map.keys()) != all_levels:
        raise ValueError(f"--data levels {sorted(data_map.keys())} != --c-fits levels {all_levels}")
    if sorted(cache_map.keys()) != all_levels:
        raise ValueError(f"--cache levels {sorted(cache_map.keys())} != --c-fits levels {all_levels}")
    if sorted(analysis_a_map.keys()) != all_levels:
        raise ValueError(f"--analysis-a levels {sorted(analysis_a_map.keys())} != --c-fits levels {all_levels}")

    rungs = list(args.rungs)


    # Verify stacking-all
    if not args.stacking_all.exists():
        raise FileNotFoundError(f"--stacking-all file {args.stacking_all} not found")
    stacking_all_data = json.loads(args.stacking_all.read_text(encoding="utf-8"))
    stacking_all_sha = sha256_file(args.stacking_all)
    reference_sets = require_reference_sets(stacking_all_data, all_levels, rungs)

    # Verify input bindings per level
    inputs_meta: dict[str, Any] = {
        "stacking_all": {"path": display(args.stacking_all), "sha256": stacking_all_sha},
        "g_fits": {"path": display(args.g_fits)},
        "levels": {},
    }

    cache_row_counts_2048: dict[int, dict[str, int]] = {}
    verified_hashes_2048: dict[str, str] = {}

    for lvl in all_levels:
        c_fits_dir = c_fits_map[lvl]
        data_dir = data_map[lvl]
        cache_dir = cache_map[lvl]
        analysis_path = analysis_a_map[lvl]

        if not c_fits_dir.is_dir():
            raise FileNotFoundError(f"c-fits directory {c_fits_dir} not found")
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data directory {data_dir} not found")
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"cache directory {cache_dir} not found")
        if not analysis_path.exists():
            raise FileNotFoundError(f"analysis-a file {analysis_path} not found")

        analysis_a = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis_sha = sha256_file(analysis_path)

        verified_hashes = verify_inputs(
            c_fits_dir, data_dir, cache_dir, analysis_a, rungs, dry_run=args.dry_run
        )
        c_digest = fits_digest_c(c_fits_dir, rungs)

        items_sha = {
            str(seed): sha256_file(data_dir / f"regen-shipped-s{seed}-n640" / "items.jsonl")
            for seed in SEEDS
        }
        cache_sha = {
            str(seed): sha256_file(cache_dir / f"shipped-s{seed}-n640.pkl")
            for seed in SEEDS
            if (cache_dir / f"shipped-s{seed}-n640.pkl").exists()
        }

        inputs_meta["levels"][lvl] = {
            "c_fits": {"path": display(c_fits_dir), "fits_digest": c_digest},
            "data": {"path": display(data_dir), "items_sha256": items_sha},
            "cache": {"path": display(cache_dir), "sha256": cache_sha},
            "analysis_a": {"path": display(analysis_path), "sha256": analysis_sha},
            "dataset_hashes": verified_hashes,
        }

        # Check if this level represents 2,048 shots
        norm_lvl = normalize_level_name(lvl)
        if norm_lvl == "2048":
            verified_hashes_2048 = verified_hashes
            for seed in SEEDS:
                cp = cache_dir / f"shipped-s{seed}-n640.pkl"
                with open(cp, "rb") as h:
                    c_data = pickle.load(h)
                cache_row_counts_2048[seed] = {
                    "validation": len(c_data["validation"]),
                    "test": len(c_data["test"]),
                }

    # If 2048 shot level was not explicitly passed in level map, infer row counts from the first level
    if not verified_hashes_2048:
        # Fall back to first level's dataset hashes and row counts
        first_lvl = all_levels[0]
        verified_hashes_2048 = inputs_meta["levels"][first_lvl]["dataset_hashes"]
        c_dir = cache_map[first_lvl]
        for seed in SEEDS:
            cp = c_dir / f"shipped-s{seed}-n640.pkl"
            with open(cp, "rb") as h:
                c_data = pickle.load(h)
            cache_row_counts_2048[seed] = {
                "validation": len(c_data["validation"]),
                "test": len(c_data["test"]),
            }

    # Verify strong fits against 2,048-shot data
    if not args.g_fits.is_dir():
        raise FileNotFoundError(f"--g-fits directory {args.g_fits} not found")
    verify_strong_fits(
        args.g_fits, rungs, verified_hashes_2048, cache_row_counts_2048, dry_run=args.dry_run
    )
    g_digest = fits_digest_c(args.g_fits, rungs)
    inputs_meta["g_fits"]["fits_digest"] = g_digest

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    if args.dry_run:
        print("[DRY-RUN] Verified all input paths, hashes, and identity fields successfully:")
        print(f"  --g-fits: {args.g_fits} (fits_digest={g_digest[:16]}...)")
        print(f"  --stacking-all: {args.stacking_all} (sha256={stacking_all_sha[:16]}...)")
        for lvl in all_levels:
            lvl_meta = inputs_meta["levels"][lvl]
            print(f"  Level '{lvl}':")
            print(f"    c-fits: {lvl_meta['c_fits']['path']}")
            print(f"    data:   {lvl_meta['data']['path']}")
            print(f"    cache:  {lvl_meta['cache']['path']}")
            print(f"    analysis-a: {lvl_meta['analysis_a']['path']}")

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

    # Non-dry-run computation
    all_cells: dict[str, Any] = {}
    overall_max_g_diff = 0.0

    for lvl in all_levels:
        norm_lvl = normalize_level_name(lvl)
        stack_set = reference_sets[lvl]

        lvl_cells, lvl_g_diff = compute_level_stacks(
            norm_lvl,
            c_fits_map[lvl],
            data_map[lvl],
            cache_map[lvl],
            args.g_fits,
            rungs,
            stacking_all_set=stack_set,
        )
        all_cells.update(lvl_cells)
        overall_max_g_diff = max(overall_max_g_diff, lvl_g_diff)

    payload = {
        "schema": SCHEMA,
        "status": RULE_STATUS,
        "script": display(script_path),
        "script_sha256": script_sha,
        "inputs": inputs_meta,
        "rungs": rungs,
        "levels": all_levels,
        "g_max_seed_diff": float(overall_max_g_diff),
        "g_identical_across_seeds": bool(overall_max_g_diff == 0.0),
        "cells": all_cells,
    }

    if args.out is not None:
        write_json_atomic(args.out, payload)
        print(f"Wrote measurement-free stack results to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
