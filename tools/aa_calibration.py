#!/usr/bin/env python3
"""A/A calibration tool for the interval decision rule (QEMScore).

Calibration of the paper's two-stage bootstrap decision rule under the null
hypothesis that two samples come from identical pipelines (an A/A comparison).

Background
----------
The paper labels a comparison "F beats C" ("measurement_adds") when the
two-stage 95 percent percentile interval for the mean gap D = C - F (macro test
MAE; learner seeds 1 to 20; 10,000 draws from a fresh
`numpy.random.default_rng(20261002)`; each draw resamples learner seeds, then
whole test circuits) lies wholly above zero, "C beats F" ("measurement_hurts")
when wholly below, else "not distinguished".

This tool evaluates four complementary calibration analyses:
1. Two-stage A/A (primary):
   For each row (dataset seed x family), rung, and arm (C, F):
   Set 1 = learner seeds 1 to 20, Set 2 = learner seeds 21 to 40 (or 1..10 vs
   11..20 in 10-vs-10 stand-in mode).
   The statistic is mean macro error of Set 1 minus that of Set 2.
   Each draw resamples learner seeds within each set independently (20 with
   replacement each), then whole test circuits shared by both sets.
   Fresh `default_rng(20261002)` per row, rung, and arm, 10,000 draws.
   Reports the 95% percentile interval and whether it excludes zero.
2. Random partitions (secondary):
   For each row, rung, and arm, 200 random partitions of the 40 seeds into two
   sets of 20 (partition generator `default_rng(20261003)`), each with the
   two-stage interval at 2,000 draws; reports the fraction excluding zero.
3. Single-fit A/A:
   With seeds 1 to 20 only: for each pair of distinct seeds of the same arm
   (190 pairs), the difference of the two single fits' macro errors with a
   circuit-only bootstrap interval (2,000 draws, `default_rng(20261004)`);
   reports the fraction excluding zero.
4. The D rule under A/A:
   D computed as C(set 1) - F(set 1) beside C(set 2) - F(set 2), and the
   fraction of rows where the two sets give different labels.

Usage:
  PYTHONPATH=. python tools/aa_calibration.py \\
      --fits RUNS/fits \\
      --cache RUNS/cache \\
      --rungs R0 N1 \\
      --out results/aa_calibration.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from qemscore.campaign.analysis import _family_mae
from tools.descriptor_ladder_analysis import (
    CELL_FIELDS,
    DATASET_SEEDS,
    LABEL_ADDS,
    LABEL_HURTS,
    LABEL_NONE,
    PERCENTILES,
    Row,
    bootstrap_draws,
    load_cache,
    macro_under_weights,
    point_macro,
)

_REPO = Path(__file__).resolve().parents[1]

DEFAULT_BOOTSTRAP_SEED = 20261002
DEFAULT_PARTITION_SEED = 20261003
DEFAULT_SINGLE_FIT_SEED = 20261004

DEFAULT_DRAWS_TWO_STAGE = 10_000
DEFAULT_DRAWS_PARTITIONS = 2_000
DEFAULT_PARTITION_COUNT = 200
DEFAULT_DRAWS_SINGLE_FIT = 2_000

DEFAULT_ARMS = ("C", "F")
DEFAULT_RUNGS = ("R0", "N1")

STEM = re.compile(
    r"^(?P<key>[A-Za-z0-9.\-]+)__(?P<rung>[A-Za-z0-9.+\-]+)"
    r"(?:__(?P<tag>k\d{2,}|orig))?__(?P<arm>A|F|C|P|M|GBT)$"
)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write JSON data atomically using a temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def parse_seeds(val: str | list[str] | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Parse seed specifications like '1-20', '21-40', '1,2,3', or lists of ints."""
    if isinstance(val, (list, tuple)):
        out: list[int] = []
        for x in val:
            out.extend(parse_seeds(x))
        return tuple(sorted(set(out)))
    s = str(val).strip()
    if "-" in s and not s.startswith("-"):
        parts = s.split("-", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            return tuple(range(int(parts[0]), int(parts[1]) + 1))
    if ".." in s:
        parts = s.split("..", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            return tuple(range(int(parts[0]), int(parts[1]) + 1))
    if "," in s:
        return tuple(sorted(set(int(x.strip()) for x in s.split(",") if x.strip())))
    if " " in s:
        return tuple(sorted(set(int(x.strip()) for x in s.split() if x.strip())))
    return (int(s),)


def extract_dataset_seed(key: str, cache_data: dict | None = None) -> int:
    """Extract dataset seed integer from dataset key or cache."""
    m = re.search(r"-s(\d+)-", key)
    if m:
        return int(m.group(1))
    if cache_data and "seed" in cache_data:
        return int(cache_data["seed"])
    raise SystemExit(f"Could not determine dataset seed for key {key!r}")


def verify_fit(
    record: dict,
    cache_data: dict,
    expected_key: str,
    expected_dataset_seed: int,
    expected_rung: str,
    expected_arm: str,
    expected_learner_seed: int,
    tol: float = 1e-12,
) -> None:
    """Verify one fit's metadata and prediction array against the cache.

    Refuses (SystemExit) on:
    - Mismatched key, dataset seed, rung, arm, or learner seed
    - Mismatched dataset_hash
    - Prediction array missing or failing to reproduce recorded per-family test MAE
    """
    meta = record["meta"]
    fit_path = record["json"]

    expected = {
        "key": expected_key,
        "dataset_seed": expected_dataset_seed,
        "rung": expected_rung,
        "arm": expected_arm,
        "learner_seed": expected_learner_seed,
    }
    for field, val in expected.items():
        if field not in meta:
            raise SystemExit(f"{fit_path}: missing required field {field!r} in fit JSON")
        if meta[field] != val:
            raise SystemExit(
                f"{fit_path}: fit {field} {meta[field]!r} differs from expected {val!r}"
            )

    cache_dataset_hash = cache_data.get("dataset_hash")
    if cache_dataset_hash is not None:
        if meta.get("dataset_hash") != cache_dataset_hash:
            raise SystemExit(
                f"{fit_path}: fit dataset_hash {meta.get('dataset_hash')!r} differs from "
                f"cache dataset_hash {cache_dataset_hash!r}"
            )

    npz_path = record["npz"]
    if not npz_path.exists():
        raise SystemExit(f"{fit_path}: NPZ file {npz_path} does not exist")

    with np.load(npz_path) as handle:
        if "test" not in handle.files:
            raise SystemExit(f"{npz_path}: missing 'test' array in NPZ")
        preds = np.asarray(handle["test"], dtype=float)

    if "test" not in cache_data:
        raise SystemExit(f"Cache for {expected_key} has no test rows")

    reproduced = _family_mae(
        cache_data["test"], preds, artifact_id=str(cache_dataset_hash or "")
    )
    recorded = meta.get("test_family_mae")
    if not recorded:
        raise SystemExit(f"{fit_path}: missing 'test_family_mae' in fit JSON")

    for fam, recorded_mae in recorded.items():
        if fam not in reproduced:
            raise SystemExit(f"{fit_path}: family {fam!r} not in reproduced errors")
        diff = abs(float(recorded_mae) - reproduced[fam])
        if diff > tol:
            raise SystemExit(
                f"{fit_path}: test predictions give {fam} error {reproduced[fam]}, "
                f"not the recorded test_family_mae {recorded_mae} (difference {diff})"
            )


class CalibrationFitStore:
    """Discovers and caches fit JSON records and NPZ test predictions."""

    def __init__(self, fit_dirs: list[Path]) -> None:
        self.fit_dirs = [Path(d) for d in fit_dirs]
        self.groups: dict[tuple[str, str, str], dict[int, dict]] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self.all_files: dict[str, Path] = {}
        self.unparsed: list[str] = []
        self._load()

    def _load(self) -> None:
        seen: dict[str, str] = {}
        for fit_dir in self.fit_dirs:
            if not fit_dir.is_dir():
                raise SystemExit(f"--fits {fit_dir} is not a directory")
            for path in sorted(fit_dir.glob("*.json")):
                stem = path.stem
                match = STEM.match(stem)
                if match is None:
                    self.unparsed.append(str(path))
                    continue
                if stem in seen:
                    raise SystemExit(f"fit {stem} appears in {seen[stem]} and {fit_dir}")
                seen[stem] = str(fit_dir)
                key = match.group("key")
                rung = match.group("rung")
                tag = match.group("tag")
                arm = match.group("arm")
                if tag is None or not tag.startswith("k"):
                    continue
                seed = int(tag[1:])
                npz = path.with_suffix(".npz")
                if not npz.exists():
                    continue
                meta = json.loads(path.read_text(encoding="utf-8"))
                record = {
                    "stem": stem,
                    "json": path,
                    "npz": npz,
                    "key": key,
                    "rung": rung,
                    "arm": arm,
                    "tag": tag,
                    "seed": seed,
                    "meta": meta,
                }
                self.groups.setdefault((key, rung, arm), {})[seed] = record
                self.all_files[path.name] = path
                self.all_files[npz.name] = npz

    def array(self, record: dict, name: str = "test") -> np.ndarray:
        cache_key = (record["stem"], name)
        if cache_key not in self._arrays:
            with np.load(record["npz"]) as handle:
                self._arrays[cache_key] = np.asarray(handle[name], dtype=float)
        return self._arrays[cache_key]

    def fits_digest(self) -> str:
        """SHA-256 over all fit files read in sorted filename order, as in stacking_increment_all.py."""
        digest = hashlib.sha256()
        for name in sorted(self.all_files):
            digest.update(name.encode())
            digest.update(sha256_file(self.all_files[name]).encode())
        return digest.hexdigest()

    def get_records(self, key: str, rung: str, arm: str, seeds: tuple[int, ...] | list[int]) -> list[dict]:
        group = self.groups.get((key, rung, arm), {})
        missing = [s for s in seeds if s not in group]
        if missing:
            raise SystemExit(f"Missing requested fit for key={key!r} rung={rung!r} arm={arm!r} seed={missing[0]}")
        return [group[s] for s in seeds]


@functools.lru_cache(maxsize=None)
def two_stage_aa_draws(
    n_seeds_1: int,
    n_seeds_2: int,
    n_circuits: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent resample counts for Set 1 and Set 2, and shared circuits.

    Each draw resamples learner seeds within each set independently (with
    replacement), then whole test circuits shared by both sets.
    """
    rng = np.random.default_rng(seed)
    seed_counts_1 = np.zeros((draws, n_seeds_1))
    seed_counts_2 = np.zeros((draws, n_seeds_2))
    circuit_counts = np.zeros((draws, n_circuits))
    for b in range(draws):
        seed_counts_1[b] = np.bincount(rng.integers(0, n_seeds_1, n_seeds_1), minlength=n_seeds_1)
        seed_counts_2[b] = np.bincount(rng.integers(0, n_seeds_2, n_seeds_2), minlength=n_seeds_2)
        circuit_counts[b] = np.bincount(rng.integers(0, n_circuits, n_circuits), minlength=n_circuits)
    seed_counts_1.flags.writeable = False
    seed_counts_2.flags.writeable = False
    circuit_counts.flags.writeable = False
    return seed_counts_1, seed_counts_2, circuit_counts


@functools.lru_cache(maxsize=None)
def single_fit_circuit_draws(n_circuits: int, draws: int, seed: int) -> np.ndarray:
    """Circuit-only resample counts for single-fit bootstrap comparisons."""
    rng = np.random.default_rng(seed)
    circuit_counts = np.zeros((draws, n_circuits))
    for b in range(draws):
        circuit_counts[b] = np.bincount(rng.integers(0, n_circuits, n_circuits), minlength=n_circuits)
    circuit_counts.flags.writeable = False
    return circuit_counts


def run_two_stage_aa(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    row: Row,
    *,
    draws: int = DEFAULT_DRAWS_TWO_STAGE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    seeds_1: list[int] | None = None,
    seeds_2: list[int] | None = None,
) -> dict:
    """Primary Two-stage A/A comparison between Set 1 and Set 2.

    Computes the mean macro error of Set 1 minus that of Set 2 across `draws`
    bootstrap resamples.
    """
    n1 = errors_1.shape[0]
    n2 = errors_2.shape[0]
    seed_counts_1, seed_counts_2, circuit_counts = two_stage_aa_draws(
        n1, n2, row.n_circuits, draws, seed
    )
    errors_both = np.vstack([errors_1, errors_2])
    macro_both = macro_under_weights(
        errors_both, row.item_circuit, row.members, row.all_cells, circuit_counts
    )
    macro_1 = macro_both[:, :n1]
    macro_2 = macro_both[:, n1:]
    mean_1 = np.sum(seed_counts_1 * macro_1, axis=1) / n1
    mean_2 = np.sum(seed_counts_2 * macro_2, axis=1) / n2
    diff_draws = mean_1 - mean_2

    pt_1 = np.asarray([point_macro(errors_1[i], row.members, row.all_cells) for i in range(n1)])
    pt_2 = np.asarray([point_macro(errors_2[i], row.members, row.all_cells) for i in range(n2)])
    point_diff = float(pt_1.mean() - pt_2.mean())

    lower, upper = np.percentile(diff_draws, PERCENTILES)
    ex_above = bool(lower > 0.0)
    ex_below = bool(upper < 0.0)
    ex_zero = bool(ex_above or ex_below)

    entry = {
        "status": "estimated",
        "point": point_diff,
        "interval": {"lower": float(lower), "upper": float(upper)},
        "excludes_zero": ex_zero,
        "excludes_zero_above": ex_above,
        "excludes_zero_below": ex_below,
        "mean_set1": float(pt_1.mean()),
        "mean_set2": float(pt_2.mean()),
        "draw_sd": float(diff_draws.std(ddof=1)) if diff_draws.size > 1 else None,
        "n_seeds_set1": n1,
        "n_seeds_set2": n2,
    }
    if seeds_1 is not None:
        entry["seeds_set1"] = list(seeds_1)
    if seeds_2 is not None:
        entry["seeds_set2"] = list(seeds_2)
    return entry


def run_random_partitions(
    all_errors: np.ndarray,
    all_seeds: list[int],
    row: Row,
    *,
    n_partitions: int = DEFAULT_PARTITION_COUNT,
    draws: int = DEFAULT_DRAWS_PARTITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    partition_seed: int = DEFAULT_PARTITION_SEED,
) -> dict:
    """Secondary random partitions analysis.

    Splits the combined seeds into two equal halves over `n_partitions` random
    permutations and reports the fraction of two-stage intervals excluding zero.
    """
    k = all_errors.shape[0]
    n = k // 2
    if k % 2 != 0:
        raise ValueError(f"Total seeds ({k}) must be even to split into two equal sets")

    partition_rng = np.random.default_rng(partition_seed)
    seed_counts_1, seed_counts_2, circuit_counts = two_stage_aa_draws(
        n, n, row.n_circuits, draws, bootstrap_seed
    )
    all_macro = macro_under_weights(
        all_errors, row.item_circuit, row.members, row.all_cells, circuit_counts
    )
    all_points = np.asarray(
        [point_macro(all_errors[i], row.members, row.all_cells) for i in range(k)]
    )

    partitions_out = []
    excluded_count = 0
    for p in range(n_partitions):
        perm = partition_rng.permutation(k)
        idx1, idx2 = perm[:n], perm[n:]
        m1 = all_macro[:, idx1]
        m2 = all_macro[:, idx2]
        d_draws = np.sum(seed_counts_1 * m1, axis=1) / n - np.sum(seed_counts_2 * m2, axis=1) / n
        pt_diff = float(all_points[idx1].mean() - all_points[idx2].mean())
        lower, upper = np.percentile(d_draws, PERCENTILES)
        ex = bool(lower > 0.0 or upper < 0.0)
        if ex:
            excluded_count += 1
        partitions_out.append({
            "partition_index": p,
            "point": pt_diff,
            "interval": {"lower": float(lower), "upper": float(upper)},
            "excludes_zero": ex,
            "seeds_set1": [all_seeds[i] for i in idx1],
            "seeds_set2": [all_seeds[i] for i in idx2],
        })

    rate = excluded_count / float(n_partitions) if n_partitions > 0 else 0.0
    return {
        "status": "estimated",
        "n_partitions": n_partitions,
        "draws_per_partition": draws,
        "excluded_zero_count": excluded_count,
        "fraction_excluding_zero": rate,
        "partitions": partitions_out,
    }


def run_single_fit_aa(
    errors_20: np.ndarray,
    seeds_20: list[int],
    row: Row,
    *,
    draws: int = DEFAULT_DRAWS_SINGLE_FIT,
    seed: int = DEFAULT_SINGLE_FIT_SEED,
) -> dict:
    """Single-fit A/A comparisons with circuit-only bootstrap intervals.

    Mirrors the frozen campaign's single-fit intervals over all distinct pairs
    of seeds (190 pairs for 20 seeds).
    """
    n = errors_20.shape[0]
    circuit_counts = single_fit_circuit_draws(row.n_circuits, draws, seed)
    macro_single = macro_under_weights(
        errors_20, row.item_circuit, row.members, row.all_cells, circuit_counts
    )
    points = np.asarray([point_macro(errors_20[i], row.members, row.all_cells) for i in range(n)])

    pairs_out = []
    excluded_count = 0
    total_pairs = 0
    for i, j in itertools.combinations(range(n), 2):
        s1 = seeds_20[i]
        s2 = seeds_20[j]
        diff_draws = macro_single[:, i] - macro_single[:, j]
        pt_diff = float(points[i] - points[j])
        lower, upper = np.percentile(diff_draws, PERCENTILES)
        ex_above = bool(lower > 0.0)
        ex_below = bool(upper < 0.0)
        ex = bool(ex_above or ex_below)
        if ex:
            excluded_count += 1
        total_pairs += 1
        pairs_out.append({
            "seed_1": s1,
            "seed_2": s2,
            "point": pt_diff,
            "interval": {"lower": float(lower), "upper": float(upper)},
            "excludes_zero": ex,
            "excludes_zero_above": ex_above,
            "excludes_zero_below": ex_below,
        })

    rate = excluded_count / float(total_pairs) if total_pairs > 0 else 0.0
    return {
        "status": "estimated",
        "total_pairs": total_pairs,
        "excluded_zero_count": excluded_count,
        "fraction_excluding_zero": rate,
        "pairs": pairs_out,
    }


def classify_d(lower: float, upper: float) -> tuple[str, str]:
    """Label D interval as in descriptor_ladder_analysis.py."""
    if lower > 0.0:
        return LABEL_ADDS, "F beats C"
    elif upper < 0.0:
        return LABEL_HURTS, "C beats F"
    else:
        return LABEL_NONE, "not distinguished"


def run_d_rule_aa(
    errors_c1: np.ndarray,
    errors_f1: np.ndarray,
    errors_c2: np.ndarray,
    errors_f2: np.ndarray,
    seeds_1: list[int],
    seeds_2: list[int],
    row: Row,
    *,
    draws: int = DEFAULT_DRAWS_TWO_STAGE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """The D rule under A/A: C(set 1) - F(set 1) beside C(set 2) - F(set 2)."""
    n1 = errors_c1.shape[0]
    n2 = errors_c2.shape[0]

    # Set 1 D
    seed_counts_1, circuit_counts_1 = bootstrap_draws(n1, row.n_circuits, draws, seed)
    mc1 = macro_under_weights(errors_c1, row.item_circuit, row.members, row.all_cells, circuit_counts_1)
    mf1 = macro_under_weights(errors_f1, row.item_circuit, row.members, row.all_cells, circuit_counts_1)
    d1_draws = np.sum(seed_counts_1 * (mc1 - mf1), axis=1) / n1
    pt_c1 = np.asarray([point_macro(errors_c1[i], row.members, row.all_cells) for i in range(n1)])
    pt_f1 = np.asarray([point_macro(errors_f1[i], row.members, row.all_cells) for i in range(n1)])
    pt_d1 = float(pt_c1.mean() - pt_f1.mean())
    lo1, hi1 = np.percentile(d1_draws, PERCENTILES)
    label1, alias1 = classify_d(lo1, hi1)

    # Set 2 D
    seed_counts_2, circuit_counts_2 = bootstrap_draws(n2, row.n_circuits, draws, seed)
    mc2 = macro_under_weights(errors_c2, row.item_circuit, row.members, row.all_cells, circuit_counts_2)
    mf2 = macro_under_weights(errors_f2, row.item_circuit, row.members, row.all_cells, circuit_counts_2)
    d2_draws = np.sum(seed_counts_2 * (mc2 - mf2), axis=1) / n2
    pt_c2 = np.asarray([point_macro(errors_c2[i], row.members, row.all_cells) for i in range(n2)])
    pt_f2 = np.asarray([point_macro(errors_f2[i], row.members, row.all_cells) for i in range(n2)])
    pt_d2 = float(pt_c2.mean() - pt_f2.mean())
    lo2, hi2 = np.percentile(d2_draws, PERCENTILES)
    label2, alias2 = classify_d(lo2, hi2)

    diff = bool(label1 != label2)
    return {
        "status": "estimated",
        "set_1": {
            "seeds": list(seeds_1),
            "point_D": pt_d1,
            "mean_C": float(pt_c1.mean()),
            "mean_F": float(pt_f1.mean()),
            "interval_D": {"lower": float(lo1), "upper": float(hi1)},
            "label": label1,
            "label_alias": alias1,
        },
        "set_2": {
            "seeds": list(seeds_2),
            "point_D": pt_d2,
            "mean_C": float(pt_c2.mean()),
            "mean_F": float(pt_f2.mean()),
            "interval_D": {"lower": float(lo2), "upper": float(hi2)},
            "label": label2,
            "label_alias": alias2,
        },
        "labels_differ": diff,
    }


def discover_caches(cache_dirs: list[Path]) -> dict[str, Path]:
    """Find all unique key cache files in cache directories."""
    found: dict[str, Path] = {}
    for d in cache_dirs:
        d = Path(d)
        if not d.is_dir():
            raise SystemExit(f"--cache {d} is not a directory")
        for p in sorted(d.iterdir()):
            if p.name.endswith(".pkl"):
                k = p.name[:-4]
            elif p.name.endswith(".json.gz"):
                k = p.name[:-8]
            elif p.name.endswith(".json") and not p.name.endswith(".order.json"):
                k = p.name[:-5]
            else:
                continue
            if k not in found:
                found[k] = p
    return found


def calibrate_all(
    fit_dirs: list[Path],
    cache_dirs: list[Path],
    *,
    rungs: tuple[str, ...] | list[str] | None = None,
    arms: tuple[str, ...] | list[str] = DEFAULT_ARMS,
    set1_seeds: str | list[int] | tuple[int, ...] = "1-20",
    set2_seeds: str | list[int] | tuple[int, ...] = "21-40",
    seeds_per_set: int | None = None,
    ten_vs_ten: bool = False,
    draws_two_stage: int = DEFAULT_DRAWS_TWO_STAGE,
    draws_partitions: int = DEFAULT_DRAWS_PARTITIONS,
    n_partitions: int = DEFAULT_PARTITION_COUNT,
    draws_single_fit: int = DEFAULT_DRAWS_SINGLE_FIT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    partition_seed: int = DEFAULT_PARTITION_SEED,
    single_fit_seed: int = DEFAULT_SINGLE_FIT_SEED,
    verbose: bool = True,
) -> dict:
    """Execute all A/A calibration analyses and generate structured results."""
    started = time.perf_counter()

    if ten_vs_ten:
        set1_tuple = tuple(range(1, 11))
        set2_tuple = tuple(range(11, 21))
    elif seeds_per_set is not None:
        set1_tuple = tuple(range(1, seeds_per_set + 1))
        set2_tuple = tuple(range(seeds_per_set + 1, 2 * seeds_per_set + 1))
    else:
        set1_tuple = parse_seeds(set1_seeds)
        set2_tuple = parse_seeds(set2_seeds)

    all_seeds = tuple(sorted(set(set1_tuple) | set(set2_tuple)))
    if len(all_seeds) % 2 != 0:
        raise SystemExit(f"Total seeds ({len(all_seeds)}) must be even for random partitions")
    seeds_20 = tuple(range(1, 21))

    store = CalibrationFitStore(fit_dirs)
    cache_map = discover_caches(cache_dirs)
    if not cache_map:
        raise SystemExit("No caches found under --cache directories")

    # Discover keys from fits and cache
    fit_keys = sorted({
        m.group("key") for d in fit_dirs for p in Path(d).glob("*.json")
        if (m := STEM.match(p.stem))
    })
    keys = [k for k in fit_keys if k in cache_map]
    if not keys:
        keys = sorted(cache_map.keys())
    if not keys:
        raise SystemExit("No matching keys found between fits and caches")

    # Candidate rungs across all fits
    all_candidate_rungs = sorted({
        m.group("rung") for d in fit_dirs for p in Path(d).glob("*.json")
        if (m := STEM.match(p.stem))
    })

    # Validate or determine rungs
    if rungs is not None:
        active_rungs = list(rungs)
        for key in keys:
            for r in active_rungs:
                for arm in arms:
                    for s in all_seeds:
                        if (key, r, arm) not in store.groups or s not in store.groups[(key, r, arm)]:
                            raise SystemExit(
                                f"Rung {r!r} is not present for row key={key!r}, arm={arm!r}, seed={s}"
                            )
    else:
        active_rungs = []
        for r in all_candidate_rungs:
            present_for_all = True
            for key in keys:
                for arm in arms:
                    for s in all_seeds:
                        if (key, r, arm) not in store.groups or s not in store.groups[(key, r, arm)]:
                            present_for_all = False
                            break
                    if not present_for_all:
                        break
                if not present_for_all:
                    break
            if present_for_all:
                active_rungs.append(r)
        if not active_rungs:
            raise SystemExit("No rungs are present for all rows, arms, and requested seeds")

    caches_used: dict[str, dict] = {}
    two_stage_cells: dict[str, dict] = {}
    partition_cells: dict[str, dict] = {}
    single_fit_cells: dict[str, dict] = {}
    d_rule_cells: dict[str, dict] = {}

    for key in keys:
        c_path = cache_map[key]
        c_data = load_cache(c_path)

        # Verify cache test item_id order if order file exists
        order_path = c_path.with_name(f"{key}.order.json")
        if order_path.exists():
            order_data = json.loads(order_path.read_text(encoding="utf-8"))
            order_ids = order_data.get("test_item_ids") or order_data.get("test")
            if order_ids is not None:
                cache_ids = [str(r["item_id"]) for r in c_data["test"]]
                if cache_ids != order_ids:
                    raise SystemExit(f"{c_path}: test item_id order differs from {order_path}")

        caches_used[key] = {
            "path": str(c_path.resolve()),
            "sha256": sha256_file(c_path),
            "dataset_hash": c_data.get("dataset_hash"),
        }
        dataset_seed = extract_dataset_seed(key, c_data)

        # Verify every fit read for this key
        needed_seeds = sorted(set(all_seeds) | set(seeds_20))
        for r in active_rungs:
            for arm in arms:
                for s in needed_seeds:
                    if (key, r, arm) not in store.groups or s not in store.groups[(key, r, arm)]:
                        raise SystemExit(f"Missing required fit for key={key!r} rung={r!r} arm={arm!r} seed={s}")
                    rec = store.groups[(key, r, arm)][s]
                    verify_fit(
                        rec,
                        c_data,
                        expected_key=key,
                        expected_dataset_seed=dataset_seed,
                        expected_rung=r,
                        expected_arm=arm,
                        expected_learner_seed=s,
                    )

        for fam in ("tfi", "heisenberg"):
            row_label = f"{key}/{fam}"
            row = Row(c_data, fam, row_label)
            for rung in active_rungs:
                d_errors = {}
                for arm in arms:
                    cell_id = f"{row_label}/{rung}/{arm}"
                    rec_1 = store.get_records(key, rung, arm, set1_tuple)
                    rec_2 = store.get_records(key, rung, arm, set2_tuple)
                    rec_all = store.get_records(key, rung, arm, all_seeds)
                    rec_20 = store.get_records(key, rung, arm, seeds_20)

                    err_1 = np.asarray([np.abs(store.array(r)[row.index] - row.ideal) for r in rec_1])
                    err_2 = np.asarray([np.abs(store.array(r)[row.index] - row.ideal) for r in rec_2])
                    err_all = np.asarray([np.abs(store.array(r)[row.index] - row.ideal) for r in rec_all])
                    err_20 = np.asarray([np.abs(store.array(r)[row.index] - row.ideal) for r in rec_20])

                    d_errors[arm] = (err_1, err_2)

                    # 1. Primary Two-stage A/A
                    ts_res = run_two_stage_aa(
                        err_1, err_2, row,
                        draws=draws_two_stage, seed=bootstrap_seed,
                        seeds_1=list(set1_tuple), seeds_2=list(set2_tuple),
                    )
                    ts_res.update({"row": row_label, "rung": rung, "arm": arm})
                    two_stage_cells[cell_id] = ts_res

                    # 2. Random partitions
                    rp_res = run_random_partitions(
                        err_all, list(all_seeds), row,
                        n_partitions=n_partitions, draws=draws_partitions,
                        bootstrap_seed=bootstrap_seed, partition_seed=partition_seed,
                    )
                    rp_res.update({"row": row_label, "rung": rung, "arm": arm})
                    partition_cells[cell_id] = rp_res

                    # 3. Single-fit A/A
                    sf_res = run_single_fit_aa(
                        err_20, list(seeds_20), row,
                        draws=draws_single_fit, seed=single_fit_seed,
                    )
                    sf_res.update({"row": row_label, "rung": rung, "arm": arm})
                    single_fit_cells[cell_id] = sf_res

                # 4. D rule under A/A
                if "C" in d_errors and "F" in d_errors:
                    d_cell_id = f"{row_label}/{rung}"
                    d_res = run_d_rule_aa(
                        d_errors["C"][0], d_errors["F"][0],
                        d_errors["C"][1], d_errors["F"][1],
                        list(set1_tuple), list(set2_tuple), row,
                        draws=draws_two_stage, seed=bootstrap_seed,
                    )
                    d_res.update({"row": row_label, "rung": rung})
                    d_rule_cells[d_cell_id] = d_res

    # Build summaries
    # 1. Two-stage summary
    ts_total = len(two_stage_cells)
    ts_ex = sum(1 for c in two_stage_cells.values() if c["excludes_zero"])
    ts_summary: dict[str, object] = {
        "total_comparisons": ts_total,
        "excluded_zero_count": ts_ex,
        "excluded_zero_rate": ts_ex / ts_total if ts_total else 0.0,
        "by_arm": {},
        "by_rung": {},
        "by_arm_rung": {},
    }
    for arm in arms:
        sub = [c for c in two_stage_cells.values() if c["arm"] == arm]
        cnt = sum(1 for c in sub if c["excludes_zero"])
        ts_summary["by_arm"][arm] = {
            "total": len(sub), "excluded_zero": cnt,
            "rate": cnt / len(sub) if sub else 0.0,
        }
    for rung in active_rungs:
        sub = [c for c in two_stage_cells.values() if c["rung"] == rung]
        cnt = sum(1 for c in sub if c["excludes_zero"])
        ts_summary["by_rung"][rung] = {
            "total": len(sub), "excluded_zero": cnt,
            "rate": cnt / len(sub) if sub else 0.0,
        }
    for rung in active_rungs:
        for arm in arms:
            sub = [c for c in two_stage_cells.values() if c["rung"] == rung and c["arm"] == arm]
            cnt = sum(1 for c in sub if c["excludes_zero"])
            ts_summary["by_arm_rung"][f"{rung}__{arm}"] = {
                "total": len(sub), "excluded_zero": cnt,
                "rate": cnt / len(sub) if sub else 0.0,
            }

    # 2. Random partitions summary
    rp_total_evals = sum(c["n_partitions"] for c in partition_cells.values())
    rp_total_ex = sum(c["excluded_zero_count"] for c in partition_cells.values())
    rp_summary: dict[str, object] = {
        "total_cells": len(partition_cells),
        "total_evaluations": rp_total_evals,
        "excluded_zero_count": rp_total_ex,
        "fraction_excluding_zero": rp_total_ex / rp_total_evals if rp_total_evals else 0.0,
        "by_arm": {},
        "by_rung": {},
        "by_arm_rung": {},
    }
    for arm in arms:
        sub = [c for c in partition_cells.values() if c["arm"] == arm]
        tot = sum(c["n_partitions"] for c in sub)
        ex = sum(c["excluded_zero_count"] for c in sub)
        rp_summary["by_arm"][arm] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}
    for rung in active_rungs:
        sub = [c for c in partition_cells.values() if c["rung"] == rung]
        tot = sum(c["n_partitions"] for c in sub)
        ex = sum(c["excluded_zero_count"] for c in sub)
        rp_summary["by_rung"][rung] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}
    for rung in active_rungs:
        for arm in arms:
            sub = [c for c in partition_cells.values() if c["rung"] == rung and c["arm"] == arm]
            tot = sum(c["n_partitions"] for c in sub)
            ex = sum(c["excluded_zero_count"] for c in sub)
            rp_summary["by_arm_rung"][f"{rung}__{arm}"] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}

    # 3. Single-fit summary
    sf_total_pairs = sum(c["total_pairs"] for c in single_fit_cells.values())
    sf_total_ex = sum(c["excluded_zero_count"] for c in single_fit_cells.values())
    sf_summary: dict[str, object] = {
        "total_cells": len(single_fit_cells),
        "total_pairs": sf_total_pairs,
        "excluded_zero_count": sf_total_ex,
        "fraction_excluding_zero": sf_total_ex / sf_total_pairs if sf_total_pairs else 0.0,
        "by_arm": {},
        "by_rung": {},
        "by_arm_rung": {},
    }
    for arm in arms:
        sub = [c for c in single_fit_cells.values() if c["arm"] == arm]
        tot = sum(c["total_pairs"] for c in sub)
        ex = sum(c["excluded_zero_count"] for c in sub)
        sf_summary["by_arm"][arm] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}
    for rung in active_rungs:
        sub = [c for c in single_fit_cells.values() if c["rung"] == rung]
        tot = sum(c["total_pairs"] for c in sub)
        ex = sum(c["excluded_zero_count"] for c in sub)
        sf_summary["by_rung"][rung] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}
    for rung in active_rungs:
        for arm in arms:
            sub = [c for c in single_fit_cells.values() if c["rung"] == rung and c["arm"] == arm]
            tot = sum(c["total_pairs"] for c in sub)
            ex = sum(c["excluded_zero_count"] for c in sub)
            sf_summary["by_arm_rung"][f"{rung}__{arm}"] = {"total": tot, "excluded_zero": ex, "rate": ex / tot if tot else 0.0}

    # 4. D rule summary
    d_total_rows = len(d_rule_cells)
    d_diff_cnt = sum(1 for c in d_rule_cells.values() if c["labels_differ"])
    d_summary: dict[str, object] = {
        "total_rows": d_total_rows,
        "different_label_count": d_diff_cnt,
        "different_label_rate": d_diff_cnt / d_total_rows if d_total_rows else 0.0,
        "by_rung": {},
    }
    for rung in active_rungs:
        sub = [c for c in d_rule_cells.values() if c["rung"] == rung]
        diffs = sum(1 for c in sub if c["labels_differ"])
        d_summary["by_rung"][rung] = {
            "total": len(sub), "different_labels": diffs,
            "rate": diffs / len(sub) if sub else 0.0,
        }

    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "schema": "aa-calibration-v1",
        "script": "tools/aa_calibration.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "mode": f"{len(set1_tuple)}_vs_{len(set2_tuple)}",
            "set1_seeds": list(set1_tuple),
            "set2_seeds": list(set2_tuple),
            "all_seeds": list(all_seeds),
            "draws_two_stage": draws_two_stage,
            "draws_partitions": draws_partitions,
            "n_partitions": n_partitions,
            "draws_single_fit": draws_single_fit,
            "bootstrap_seed": bootstrap_seed,
            "partition_seed": partition_seed,
            "single_fit_seed": single_fit_seed,
            "percentiles": list(PERCENTILES),
            "rungs": list(active_rungs),
            "arms": list(arms),
        },
        "inputs": {
            "fits_dirs": [str(Path(d).resolve()) for d in fit_dirs],
            "cache_dirs": [str(Path(d).resolve()) for d in cache_dirs],
            "n_fit_files": len(store.all_files),
            "fits_sha256_digest": store.fits_digest(),
            "caches_used": caches_used,
            "rungs": list(active_rungs),
            "arms": list(arms),
            "set1_seeds": list(set1_tuple),
            "set2_seeds": list(set2_tuple),
            "all_seeds": list(all_seeds),
        },
        "timing_seconds": elapsed,
        "summaries": {
            "two_stage_aa": ts_summary,
            "random_partitions": rp_summary,
            "single_fit_aa": sf_summary,
            "d_rule_consistency": d_summary,
        },
        "results": {
            "two_stage_aa": two_stage_cells,
            "random_partitions": partition_cells,
            "single_fit_aa": single_fit_cells,
            "d_rule": d_rule_cells,
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits", action="append", type=Path, required=True, help="Fits directory (repeatable)")
    parser.add_argument("--cache", action="append", type=Path, required=True, help="Cache directory (repeatable)")
    parser.add_argument("--rungs", nargs="+", default=None, help="Rungs to analyze (default: all rungs present for every row)")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), help="Arms to analyze (default: C F)")
    parser.add_argument("--ten-vs-ten", action="store_true", help="Run 10-vs-10 stand-in version (seeds 1..10 vs 11..20)")
    parser.add_argument("--set1-seeds", default=None, help="Learner seeds for Set 1 (default: 1-20, or 1-10 with --ten-vs-ten)")
    parser.add_argument("--set2-seeds", default=None, help="Learner seeds for Set 2 (default: 21-40, or 11-20 with --ten-vs-ten)")
    parser.add_argument("--seeds-per-set", type=int, default=None, help="Learner seeds per set (alternative to set1/set2 seeds)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS_TWO_STAGE, help="Two-stage bootstrap draws (default: 10,000)")
    parser.add_argument("--partition-draws", type=int, default=DEFAULT_DRAWS_PARTITIONS, help="Random partition bootstrap draws (default: 2,000)")
    parser.add_argument("--partition-count", type=int, default=DEFAULT_PARTITION_COUNT, help="Number of random partitions (default: 200)")
    parser.add_argument("--single-fit-draws", type=int, default=DEFAULT_DRAWS_SINGLE_FIT, help="Single-fit bootstrap draws (default: 2,000)")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="RNG seed for two-stage bootstrap (default: 20261002)")
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED, help="RNG seed for partition generator (default: 20261003)")
    parser.add_argument("--single-fit-seed", type=int, default=DEFAULT_SINGLE_FIT_SEED, help="RNG seed for single-fit bootstrap (default: 20261004)")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (written atomically)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose stdout output")

    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.ten_vs_ten:
        set1 = args.set1_seeds if args.set1_seeds is not None else "1-10"
        set2 = args.set2_seeds if args.set2_seeds is not None else "11-20"
    else:
        set1 = args.set1_seeds if args.set1_seeds is not None else "1-20"
        set2 = args.set2_seeds if args.set2_seeds is not None else "21-40"

    result = calibrate_all(
        fit_dirs=args.fits,
        cache_dirs=args.cache,
        rungs=args.rungs,
        arms=args.arms,
        set1_seeds=set1,
        set2_seeds=set2,
        seeds_per_set=args.seeds_per_set,
        ten_vs_ten=args.ten_vs_ten,
        draws_two_stage=args.draws,
        draws_partitions=args.partition_draws,
        n_partitions=args.partition_count,
        draws_single_fit=args.single_fit_draws,
        bootstrap_seed=args.bootstrap_seed,
        partition_seed=args.partition_seed,
        single_fit_seed=args.single_fit_seed,
        verbose=verbose,
    )

    if args.out:
        _write_json(args.out, result)
        if verbose:
            print(f"Wrote results to {args.out}")

    if verbose:
        s_ts = result["summaries"]["two_stage_aa"]
        s_rp = result["summaries"]["random_partitions"]
        s_sf = result["summaries"]["single_fit_aa"]
        s_d = result["summaries"]["d_rule_consistency"]

        print("\n=== A/A Calibration Summary ===")
        print(f"Mode: {result['parameters']['mode']}")
        print(f"Two-stage A/A rate: {s_ts['excluded_zero_count']}/{s_ts['total_comparisons']} ({s_ts['excluded_zero_rate']:.4f})")
        print(f"Random partitions rate: {s_rp['excluded_zero_count']}/{s_rp['total_evaluations']} ({s_rp['fraction_excluding_zero']:.4f})")
        print(f"Single-fit A/A rate: {s_sf['excluded_zero_count']}/{s_sf['total_pairs']} ({s_sf['fraction_excluding_zero']:.4f})")
        print(f"D-rule label difference rate: {s_d['different_label_count']}/{s_d['total_rows']} ({s_d['different_label_rate']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
