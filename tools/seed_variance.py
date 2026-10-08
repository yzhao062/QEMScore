"""What the two-stage intervals condition on (round-10 rule, Part C).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part C.

For the three six-seed designs that Part D of the round-9 rule pooled
(original candidates with the strength indicator at 2,048 shots, stronger
candidates at 2,048 shots, stronger candidates at the exact level; dataset
seeds 101, 211, 307 and 401, 503, 607), and for every row and rung:

1. Paired versus independent learner-seed resampling. Analysis A resamples one
   set of learner-seed counts and applies it to C and F alike (paired). The
   independent variant draws F's seed counts from a second stream
   (``default_rng(20261003)``, the rule seed plus one), keeping C's counts and
   the shared circuit counts. Interval, width, and label of each.
2. Variance components of the seed-averaged D of one dataset seed: learner
   seeds (the variance of the 20 per-seed D values over 20), test circuits
   (the variance of the circuit-only bootstrap with all learner seeds kept),
   and the two-stage draw variance analysis A reports. Across the six dataset
   seeds, the DerSimonian-Laird between-seed variance tau^2 (training sample,
   circuit pool, and everything else a dataset seed changes) from the per-seed
   D and draw SD, as Part D computed it.
3. A prediction interval for the seed-averaged D of a new dataset seed:
   theta +/- t(0.975, 4) sqrt(tau^2 + SE^2) for its true value
   (Higgins, Thompson, and Spiegelhalter 2009) and
   theta +/- t(0.975, 4) sqrt(tau^2 + mean v + SE^2) for its estimate, with
   theta the random-effects mean and SE = (sum of random-effects weights)^-1/2.
4. First versus new seeds: the mean D of each panel, their difference, and a
   Welch t interval over the three seeds of each panel.

Usage:
  PYTHONPATH=. python tools/seed_variance.py --assets A --crossed C --fresh F \\
      --frozen-rule docs/frozen-rules/2026-10-07-round10-checks.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy.stats

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import descriptor_ladder_analysis as dla  # noqa: E402
from tools import pooled_absolute as pa  # noqa: E402
from tools import reading_rules as rr  # noqa: E402

RULE_FILE = rr.RULE_FILE
POOLED = _REPO / "artifacts/descriptor-information/round9/pooled-absolute.json"
DEFAULT_OUT = _REPO / "artifacts/descriptor-information/round10/seed-variance.json"
DEFAULT_MD = _REPO / "artifacts/descriptor-information/round10/seed-variance.md"
DESIGNS = {
    "original 2,048": ("strength_indicator_2048", "fresh_orig_2048"),
    "strong 2,048": ("strong_learners_2048", "fresh_strong_2048"),
    "strong exact": ("follow_b_exact", "fresh_strong_exact"),
}
INDEPENDENT_SEED = dla.RULE_SEED + 1
TOL = 1e-12


def independent_seed_counts(n_seeds: int, n_circuits: int) -> np.ndarray:
    """Seed counts of a second stream with the rule's call order."""
    counts, _ = dla.bootstrap_draws(n_seeds, n_circuits, rr.DRAWS, INDEPENDENT_SEED)
    return counts


def cell_components(row: dla.Row, err_c: np.ndarray, err_f: np.ndarray,
                    released_d: dict) -> dict:
    n = err_c.shape[0]
    seed_counts, _ = dla.bootstrap_draws(n, row.n_circuits, rr.DRAWS, rr.SEED)
    macro_c = rr.macro_draws(row, err_c)
    macro_f = rr.macro_draws(row, err_f)
    paired = np.sum(seed_counts * (macro_c - macro_f), axis=1) / n
    ind_counts = independent_seed_counts(n, row.n_circuits)
    independent = (np.sum(seed_counts * macro_c, axis=1)
                   - np.sum(ind_counts * macro_f, axis=1)) / n
    circuit_only = np.mean(macro_c - macro_f, axis=1)
    per_seed = np.asarray([dla.point_macro(err_c[s], row.members, row.all_cells)
                           - dla.point_macro(err_f[s], row.members, row.all_cells)
                           for s in range(n)])
    point = float(per_seed.mean())
    p_entry = rr.interval(paired, point)
    diffs = [abs(point - released_d["point"]),
             abs(p_entry["lower"] - released_d["interval"]["lower"]),
             abs(p_entry["upper"] - released_d["interval"]["upper"]),
             abs(float(paired.std(ddof=1)) - released_d["draw_sd"])]
    if max(diffs) > TOL:
        raise SystemExit(f"{row.label}: paired D differs from analysis A by {max(diffs)}")
    i_entry = rr.interval(independent, point)

    def label(e: dict) -> str:
        return ("F beats C" if e["excludes_zero_above"] else
                "C beats F" if e["excludes_zero_below"] else "not distinguished")

    return {
        "D": point,
        "per_learner_seed_D": per_seed.tolist(),
        "var_learner": float(per_seed.var(ddof=1) / n),
        "var_circuit": float(circuit_only.var(ddof=1)),
        "var_two_stage": float(paired.var(ddof=1)),
        "paired": {**p_entry, "width": p_entry["upper"] - p_entry["lower"], "label": label(p_entry)},
        "independent": {**i_entry, "width": i_entry["upper"] - i_entry["lower"],
                        "label": label(i_entry), "draw_sd": float(independent.std(ddof=1))},
        "check_vs_analysis_a": max(diffs),
    }


def pool(per_seed: dict[int, dict], released: dict | None) -> dict:
    seeds = sorted(per_seed)
    y = [per_seed[s]["D"] for s in seeds]
    se = [math.sqrt(per_seed[s]["var_two_stage"]) for s in seeds]
    meta = pa.hksj_meta_analysis(y, se)
    if released is not None:
        diff = max(abs(meta["theta_re"] - released["pooled_D"]),
                   abs(meta["tau2"] - released["tau2"]))
        if diff > 1e-12:
            raise SystemExit(f"pooled D or tau2 differs from Part D by {diff}")
    tau2 = meta["tau2"]
    w = 1.0 / (np.asarray(se) ** 2 + tau2)
    se_theta = float(math.sqrt(1.0 / w.sum()))
    k = len(seeds)
    t_pred = float(scipy.stats.t.ppf(0.975, df=k - 2))
    v_l = float(np.mean([per_seed[s]["var_learner"] for s in seeds]))
    v_c = float(np.mean([per_seed[s]["var_circuit"] for s in seeds]))
    v_2 = float(np.mean([per_seed[s]["var_two_stage"] for s in seeds]))
    theta = meta["theta_re"]
    half_true = t_pred * math.sqrt(tau2 + se_theta ** 2)
    half_obs = t_pred * math.sqrt(tau2 + v_2 + se_theta ** 2)
    return {
        "dataset_seeds": seeds,
        "theta_re": theta, "tau2": tau2, "se_theta": se_theta, "t_df": k - 2,
        "mean_var_learner": v_l, "mean_var_circuit": v_c, "mean_var_two_stage": v_2,
        # Learner and circuit components are marginal diagnostics, not a
        # partition. The fraction below divides a new seed's estimate
        # variance (apart from SE^2) into dataset and within-seed parts.
        "dataset_seed_fraction": (tau2 / (tau2 + v_2)) if (tau2 + v_2) > 0 else None,
        "prediction_interval_true": {"lower": theta - half_true, "upper": theta + half_true},
        "prediction_interval_estimate": {"lower": theta - half_obs, "upper": theta + half_obs},
        "hksj_interval": {"lower": meta["ci_lower"], "upper": meta["ci_upper"]},
    }


def first_vs_new(per_seed: dict[int, dict], first: tuple, new: tuple) -> dict:
    a = np.asarray([per_seed[s]["D"] for s in first])
    b = np.asarray([per_seed[s]["D"] for s in new])
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1)) if se > 0 else None
    diff = float(b.mean() - a.mean())
    half = float(scipy.stats.t.ppf(0.975, df=df)) * se if df else 0.0
    return {"first_mean": float(a.mean()), "new_mean": float(b.mean()),
            "new_minus_first": diff, "welch_df": df,
            "interval": {"lower": diff - half, "upper": diff + half},
            "first_labels": {str(s): per_seed[s]["paired"]["label"] for s in first},
            "new_labels": {str(s): per_seed[s]["paired"]["label"] for s in new}}


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    rule_path = _REPO / RULE_FILE
    if not args.frozen_rule.is_file() or args.frozen_rule.resolve() != rule_path.resolve():
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")
    roots = {"ASSETS": args.assets, "CROSSED": args.crossed, "FRESH": args.fresh}
    registry = json.loads(rr.REGISTRY.read_text(encoding="utf-8"))
    mapping = {e["fit_set"]: e for e in json.loads(rr.MAPPING.read_text(encoding="utf-8"))["fit_sets"]}
    pooled_rows = {(r["fit_set"], r["family"], r["rung"]): r
                   for r in json.loads(POOLED.read_text(encoding="utf-8"))["pooled_table"]}
    out_designs = {}
    for design, (first_set, new_set) in DESIGNS.items():
        cells: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        panels = {}
        for fit_set in (first_set, new_set):
            entry = registry["fit_sets"][fit_set]
            analysis = json.loads((_REPO / mapping[fit_set]["path"]).read_text(encoding="utf-8"))
            a_rows = analysis["parts"]["A"]["rows"]
            store = dla.FitStore([rr.resolve(entry["fits"], roots)], use_orig=False)
            cache_dir = rr.resolve(entry["cache"], roots)
            seeds = [int(s) for s in mapping[fit_set]["dataset_seeds"]]
            panels[fit_set] = tuple(seeds)
            print(f"[{design}] {fit_set}", flush=True)
            for seed in seeds:
                key = f"shipped-s{seed}-n640"
                cache_path = cache_dir / f"{key}.pkl"
                if rr.sha256_file(cache_path) != analysis["inputs"]["caches_used"][key]["sha256"]:
                    raise SystemExit(f"{cache_path}: SHA-256 differs from analysis A's cache")
                data = dla.load_cache(cache_path)
                for family in ("tfi", "heisenberg"):
                    row = dla.Row(data, family, f"{key}/{family}")
                    for rung, a_rung in a_rows[f"{key}/{family}"]["rungs"].items():
                        c_rec = rr.seeded(store, key, rung, "C")
                        f_rec = rr.seeded(store, key, rung, "F")
                        err_c = rr.item_errors(row, store, c_rec, "test")
                        err_f = rr.item_errors(row, store, f_rec, "test")
                        cells[(family, rung)][seed] = cell_components(row, err_c, err_f,
                                                                      a_rung["D"])
        first, new = panels[first_set], panels[new_set]
        rows = {}
        for (family, rung), per_seed in sorted(cells.items()):
            if set(per_seed) != set(first) | set(new):
                continue
            released = pooled_rows.get((design, family, rung))
            rows[f"{family}/{rung}"] = {
                "per_seed": {str(s): v for s, v in sorted(per_seed.items())},
                "pooled": pool(per_seed, released),
                "first_vs_new": first_vs_new(per_seed, first, new),
                "paired_vs_independent": {
                    "label_changes": [s for s, v in per_seed.items()
                                      if v["paired"]["label"] != v["independent"]["label"]],
                    "width_ratio_independent_over_paired": {
                        str(s): v["independent"]["width"] / v["paired"]["width"]
                        for s, v in per_seed.items() if v["paired"]["width"] > 0},
                },
            }
        out_designs[design] = {"first_set": first_set, "new_set": new_set, "rows": rows}
    return {
        "schema": "seed-variance-v1",
        "frozen_rule": RULE_FILE,
        "rule_file_sha256": rr.sha256_file(rule_path),
        "script_sha256": rr.sha256_file(Path(__file__)),
        "bootstrap": {"draws": rr.DRAWS, "seed": rr.SEED, "independent_seed": INDEPENDENT_SEED},
        "designs": out_designs,
        "seconds": time.perf_counter() - started,
    }


def markdown(result: dict) -> str:
    lines = ["# Variance Components and New-Seed Prediction Intervals", "",
             f"Rule: `{RULE_FILE}`, Part C.", "",
             "| Design | Row | Pooled D | tau^2 | Learner | Circuit | Dataset fraction | "
             "PI (true) | PI (estimate) | New - first [Welch] | Paired vs independent label changes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for design, d in result["designs"].items():
        for key, r in d["rows"].items():
            p, f = r["pooled"], r["first_vs_new"]
            share = p["dataset_seed_fraction"]
            lines.append(
                f"| {design} | {key} | {p['theta_re']:+.5f} | {p['tau2']:.2e} | "
                f"{p['mean_var_learner']:.2e} | {p['mean_var_circuit']:.2e} | "
                f"{'' if share is None else f'{share:.2f}'} | "
                f"[{p['prediction_interval_true']['lower']:+.5f}, "
                f"{p['prediction_interval_true']['upper']:+.5f}] | "
                f"[{p['prediction_interval_estimate']['lower']:+.5f}, "
                f"{p['prediction_interval_estimate']['upper']:+.5f}] | "
                f"{f['new_minus_first']:+.5f} [{f['interval']['lower']:+.5f}, "
                f"{f['interval']['upper']:+.5f}] | "
                f"{len(r['paired_vs_independent']['label_changes'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--crossed", type=Path, required=True)
    p.add_argument("--fresh", type=Path, required=True)
    p.add_argument("--frozen-rule", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = p.parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    args.md.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.out} ({result['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
