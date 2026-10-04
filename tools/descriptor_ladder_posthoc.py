"""Post hoc (exploratory) paired contrasts F - R and C - R, Part B (default) or Part A.

The frozen rule (docs/frozen-rules/2026-10-02-descriptor-information.md) states
no reading for a comparison of a learned arm with the raw estimate R, so this
analysis is labeled post hoc. It reuses analysis script A
(tools/descriptor_ladder_analysis.py) for every step, so the draws, the
arithmetic, and the bootstrap seed are the rule's: an arm's mean-over-seeds
draws minus R's macro error under the same circuit resample. R's draws use the
rung's D seed count, as script A does for every seedless arm.

Usage, from the repository root:
  PYTHONPATH=. python tools/descriptor_ladder_posthoc.py RUNS_DIR OUT_JSON
where RUNS_DIR holds partB/fits and partB/cache (or the release asset's fits/
and cache/ directories, passed as RUNS_DIR with --flat).
Alternatively, with explicit fits and cache directories:
  PYTHONPATH=. python tools/descriptor_ladder_posthoc.py --fits DIR [--fits DIR2 ...] \
      --cache DIR [--cache DIR2 ...] [--part A|B] [--out OUT_JSON | OUT_JSON]
With --part A the rows are the Part A dataset-seed and family rows, at every rung
with fits (as the shot-sweep rule 2026-10-03-shot-sweep.md requires).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "tools"))
sys.path.insert(0, str(repo))
import descriptor_ladder_analysis as dla  # noqa: E402


def _rows(part: str, spec: dict, caches: dict, store) -> list[tuple[str, str, object, list[str]]]:
    """(key, label, Row, rungs with fits) for every row of the part, in a fixed order."""
    rows = []
    for seed in dla.DATASET_SEEDS:
        key = f"qaoa-s{seed}-n640" if part == "B" else f"shipped-s{seed}-n640"
        data = dla.load_cache(caches[key])
        present = store.rungs_for(key) if part == "A" else set(spec["rungs"])
        for family in (spec["families"] or (None,)):
            label = f"{key}/{family}" if family else key
            rungs = [rung for rung in spec["rungs"] if rung in present
                     and (spec["rung_families"].get(rung) is None
                          or family in spec["rung_families"][rung])]
            rows.append((key, label, dla.Row(data, family, label), rungs))
    return rows


def compute_posthoc(fit_dirs: list[Path], cache_dirs: list[Path],
                    out_path: Path | None = None,
                    *, verbose: bool = True, part: str = "B") -> dict:
    store = dla.FitStore([Path(d) for d in fit_dirs], False)
    caches, _ = dla.discover_caches([Path(d) for d in cache_dirs])
    if part not in ("A", "B"):
        raise ValueError(f"part must be A or B, got {part}")
    spec = dla.PARTS[part]
    out = {"schema": "descriptor-information-posthoc-arm-minus-raw-v1",
           "status": "post hoc; the frozen rule states no reading for these contrasts",
           "analysis_script_sha256": dla.sha256_file(Path(dla.__file__)),
           "bootstrap": {"seed": dla.RULE_SEED, "draws": dla.DEFAULT_DRAWS},
           "cells": {}}
    if part == "A":
        out["part"] = "A"
    for key, label, row, rungs in _rows(part, spec, caches, store):
        est = dla.RowEstimator(row, store, dla.DEFAULT_DRAWS, dla.RULE_SEED, 20)
        cells = row.all_cells
        r_point = dla.point_macro(est.raw_errors()[0], row.members, cells)
        for rung in rungs:
            arms = dla.arms_at(store, spec, key, rung)
            _, d_parts = est.difference(arms["C"], arms["F"])
            n = d_parts["n"]
            r_draws = est.macro(None, "raw", n, cells)[:, 0]
            cell_out = {"R_point": r_point}
            for arm in ("F", "C"):
                entry, draws = est.mean_arm(arms[arm])
                contrast = dla.interval_entry(draws - r_draws, entry["point"] - r_point, n)
                cell_out[f"{arm}_point"] = entry["point"]
                cell_out[f"{arm}_minus_R"] = contrast
                lo, hi = contrast["interval"]["lower"], contrast["interval"]["upper"]
                if verbose:
                    print(f"{label} {rung:10s} {arm}-R {contrast['point']:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            out["cells"][f"{label}/{rung}"] = cell_out
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
        if verbose:
            print("wrote", out_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("positional", nargs="*", type=Path,
                        help="RUNS_DIR and/or OUT_JSON")
    parser.add_argument("--fits", action="append", default=[], type=Path,
                        help="Fit directory (repeatable)")
    parser.add_argument("--cache", action="append", default=[], type=Path,
                        help="Cache directory (repeatable)")
    parser.add_argument("--flat", action="store_true",
                        help="Look for fits/ and cache/ directly in RUNS_DIR")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress printed lines")
    parser.add_argument("--part", choices=("A", "B"), default="B",
                        help="Part B (default) or Part A rows")
    args = parser.parse_args(argv)

    if args.fits:
        fit_dirs = args.fits
        if not args.cache:
            parser.error("--cache is required when --fits is specified")
        cache_dirs = args.cache
        if args.out is not None:
            out_path = args.out
        elif args.positional:
            out_path = args.positional[0]
        else:
            parser.error("Missing output JSON path (use --out or a positional path)")
    else:
        remaining = list(args.positional)
        if args.out is not None:
            if len(remaining) < 1:
                parser.error("Usage: descriptor_ladder_posthoc.py RUNS_DIR [OUT_JSON]")
            runs = remaining[0]
            out_path = args.out
        else:
            if len(remaining) < 2:
                parser.error("Usage: descriptor_ladder_posthoc.py RUNS_DIR OUT_JSON")
            runs, out_path = remaining[0], remaining[1]
        fit_dirs = [runs / ("fits" if args.flat else "partB/fits")]
        cache_dirs = [runs / ("cache" if args.flat else "partB/cache")]

    compute_posthoc(fit_dirs, cache_dirs, out_path, verbose=not args.quiet, part=args.part)
    return 0


if __name__ == "__main__":
    sys.exit(main())
