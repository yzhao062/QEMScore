"""Post hoc (exploratory) paired contrasts F - R and C - R, Part B.

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
"""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
flat = "--flat" in sys.argv
runs, out_path = (Path(a) for a in [x for x in sys.argv[1:] if x != "--flat"][:2])
sys.path.insert(0, str(repo / "tools"))
sys.path.insert(0, str(repo))
import descriptor_ladder_analysis as dla  # noqa: E402

store = dla.FitStore([runs / ("fits" if flat else "partB/fits")], False)
caches, _ = dla.discover_caches([runs / ("cache" if flat else "partB/cache")])
spec = dla.PARTS["B"]
out = {"schema": "descriptor-information-posthoc-arm-minus-raw-v1",
       "status": "post hoc; the frozen rule states no reading for these contrasts",
       "analysis_script_sha256": dla.sha256_file(Path(dla.__file__)),
       "bootstrap": {"seed": dla.RULE_SEED, "draws": dla.DEFAULT_DRAWS},
       "cells": {}}
for seed in dla.DATASET_SEEDS:
    key = f"qaoa-s{seed}-n640"
    row = dla.Row(dla.load_cache(caches[key]), None, key)
    est = dla.RowEstimator(row, store, dla.DEFAULT_DRAWS, dla.RULE_SEED, 20)
    cells = row.all_cells
    r_point = dla.point_macro(est.raw_errors()[0], row.members, cells)
    for rung in spec["rungs"]:
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
            print(f"{seed} {rung:10s} {arm}-R {contrast['point']:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        out["cells"][f"{key}/{rung}"] = cell_out
out_path.write_text(json.dumps(out, indent=1))
print("wrote", out_path)
