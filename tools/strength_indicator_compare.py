#!/usr/bin/env python3
"""Paired comparison of strength-indicator fits against original fits.

Frozen rule: docs/frozen-rules/2026-10-03-strength-indicator.md
(Comparison with the original experiment):
For each row and rung, report the paired difference in D and in D/C between
the strength-indicator fits and the original fits, using the same seed and
circuit indices in each bootstrap draw (descriptive).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from tools.descriptor_ladder_analysis import (
    DATASET_SEEDS,
    DEFAULT_DRAWS,
    LEARNER_SEEDS,
    PARTS,
    RULE_SEED,
    FitStore,
    Row,
    RowEstimator,
    _compare_original,
    _fmt,
    _write_json,
    arms_at,
    discover_caches,
    load_cache,
    sha256_file,
)

RULE_FILE_PATH = "docs/frozen-rules/2026-10-03-strength-indicator.md"


def compare_fits(strength_fit_dirs: list[Path], original_fit_dirs: list[Path],
                 cache_dirs: list[Path], *, draws: int = DEFAULT_DRAWS,
                 seed: int = RULE_SEED, verbose: bool = True) -> dict:
    started = time.perf_counter()
    str_store = FitStore(strength_fit_dirs, use_orig=False)
    orig_store = FitStore(original_fit_dirs, use_orig=False)
    caches, _ = discover_caches(cache_dirs)
    nominal = len(LEARNER_SEEDS)

    rule_path = Path(__file__).resolve().parents[1] / RULE_FILE_PATH
    result: dict = {
        "schema": "descriptor-information-strength-comparison-v1",
        "frozen_rule": RULE_FILE_PATH,
        "rule_file_sha256": sha256_file(rule_path) if rule_path.exists() else None,
        "bootstrap": {
            "seed": seed, "draws": draws,
            "rng_scheme": ("paired draws using fresh numpy default_rng(seed) per row "
                           "and quantity; learner seeds then whole test circuits"),
        },
        "inputs": {
            "strength_fits": [str(Path(d).resolve()) for d in strength_fit_dirs],
            "original_fits": [str(Path(d).resolve()) for d in original_fit_dirs],
            "cache": [str(Path(d).resolve()) for d in cache_dirs],
            "n_strength_fit_files": str_store.n_files,
            "n_original_fit_files": orig_store.n_files,
        },
        "parts": {},
    }

    str_keys = str_store.keys()
    for part, spec in PARTS.items():
        keys = sorted(k for k in str_keys if spec["key_pattern"].match(k))
        if not keys:
            continue
        keys_by_seed: dict[int, str] = {}
        for key in keys:
            ds = int(spec["key_pattern"].match(key).group(1))
            keys_by_seed[ds] = key

        rows_out: dict = {}
        for dataset_seed in DATASET_SEEDS:
            key = keys_by_seed.get(dataset_seed)
            if key is None or key not in caches:
                continue
            data = load_cache(caches[key])
            families = spec["families"] or (None,)
            for family in families:
                label = f"{key}/{family}" if family else key
                try:
                    row = Row(data, family, label)
                except ValueError:
                    continue
                est_s = RowEstimator(row, str_store, draws, seed, nominal)
                est_o = RowEstimator(row, orig_store, draws, seed, nominal)
                rungs_out: dict = {}
                for rung in spec["rungs"]:
                    allowed = spec["rung_families"].get(rung)
                    if allowed is not None and family not in allowed:
                        continue
                    arms_s = arms_at(str_store, spec, key, rung)
                    comp = _compare_original(
                        est_s, est_o, str_store, orig_store, spec, key, rung, arms_s)
                    rungs_out[rung] = comp
                    if verbose and comp.get("status") == "estimated":
                        print(f"{label:34s} {rung:10s} "
                              f"delta D={_fmt(comp['delta_D'])} "
                              f"delta D/C={_fmt(comp['delta_D_over_C'])}", flush=True)
                rows_out[label] = {"dataset_seed": dataset_seed, "family": family,
                                   "rungs": rungs_out}
        result["parts"][part] = {"title": spec["title"], "rows": rows_out}

    result["seconds"] = time.perf_counter() - started
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strength-fits", action="append", type=Path, required=True,
                        help="directories of strength-indicator fits (repeatable)")
    parser.add_argument("--original-fits", action="append", type=Path, required=True,
                        help="directories of original fits (repeatable)")
    parser.add_argument("--cache", action="append", type=Path, required=True,
                        help="directories of caches (repeatable)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output comparison JSON path")
    parser.add_argument("--bootstrap-seed", type=int, default=RULE_SEED)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = compare_fits(args.strength_fits, args.original_fits, args.cache,
                          draws=args.draws, seed=args.bootstrap_seed,
                          verbose=not args.quiet)
    _write_json(args.out, report)
    print(f"wrote {args.out} in {report['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
