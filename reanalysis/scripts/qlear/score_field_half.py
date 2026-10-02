"""Score the fitted ladder under the frozen contract, and replay the release's
own function separately as a faithful reproduction panel.

Two scores, kept apart on purpose. The frozen attribution score is standard
Hellinger on the released support with the normalization fixed in
PLAN-preprint-scope.md. The reproduction panel is the release's own RQ2 body,
which renormalizes differently; saved analyses computed under the two cannot be
mixed into one decomposition.

Aggregation and resampling are also frozen: one distribution error per
application and backend, averaged equally across eight backends and six
applications, with the six application circuits resampled jointly across their
backends and predictors, separately for each fitted seed.
"""

import glob
import json
import os
import sys

import numpy as np

FITS = sys.argv[1]
OUT = sys.argv[2]
N_RESAMPLES = 10000
ROOT_SEED = 20260906
CONFIDENCE = 0.95

LADDER = ("A", "C", "O", "F")
SECONDARY_ARM = "C4"

# The cohort and the seed list are fixed here rather than derived from whatever
# survived. Round 12 showed that deriving them from the first surviving arm let
# a deleted application, a deleted seed and a substituted seed all pass as a
# complete estimate with zero failures reported.
EXPECTED_APPLICATIONS = ("groundstate", "pricingcall", "pricingput",
                         "qaoa", "routing", "tsp")
EXPECTED_BACKENDS = ("ibm_lagos", "ibm_nairobi", "ibm_perth", "ibmq_belem",
                     "ibmq_jakarta", "ibmq_lima", "ibmq_manila", "ibmq_quito")
EXPECTED_SEEDS = tuple(range(10))
# A is affine and deterministic, so the plan's ten seeds would produce ten
# identical fits. It is fitted once and paired with every network seed. That is
# a declared property of the arm rather than a fallback for a missing fit.
AFFINE_ARMS = ("A",)


def frozen_hellinger(target, prediction):
    """Standard Hellinger on the released support.

    Percentages become probabilities, predicted negatives are clipped to zero,
    each predicted vector is normalized by its own sum, and the released target
    vector is normalized for scoring. No target value enters prediction
    postprocessing. A nonpositive total fails the evaluation explicitly.
    """
    p = np.asarray(target, dtype=float) / 100.0
    q = np.asarray(prediction, dtype=float) / 100.0
    if p.shape != q.shape:
        return None, "shape_mismatch"
    # Checked before clipping. `np.where(q > 0, q, 0)` sends NaN to zero, so a
    # nonfinite prediction would otherwise be scored as if the model had
    # predicted zero probability for that state.
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return None, "nonfinite_vector"
    q = np.where(q > 0.0, q, 0.0)
    p_total = float(p.sum())
    q_total = float(q.sum())
    if not np.isfinite(p_total) or p_total <= 0.0:
        return None, "nonpositive_target_total"
    if not np.isfinite(q_total) or q_total <= 0.0:
        return None, "nonpositive_prediction_total"
    p = p / p_total
    q = q / q_total
    value = float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))
    if not np.isfinite(value):
        return None, "nonfinite_score"
    return value, None


def release_hellinger(target, prediction):
    """The release's own RQ2 body, transcribed, for the reproduction panel."""
    p = np.asarray(target, dtype=float).copy() / 100.0
    q = np.asarray(prediction, dtype=float).copy() / 100.0
    p[p <= 0] = 0
    q[q <= 0] = 0
    if q.sum() > 1:
        nz = len(q) - len(q[q == 0])
        if nz > 0:
            q[q > 0] = q[q > 0] - ((q.sum() - p.sum()) / nz)
        q[q < 0] = 0
    total = 0.0
    for i in range(len(p)):
        total += (np.sqrt(p[i]) - np.sqrt(q[i])) ** 2
    return float((1.0 / np.sqrt(2.0)) * np.sqrt(total))


def load_fits():
    records = {}
    fit_paths = sorted(glob.glob(os.path.join(FITS, "pred-*.json")))
    if not fit_paths:
        raise SystemExit(
            f"No pred-*.json files found in '{FITS}'. Rescoring requires "
            "refitting the 41-arm ladder first using:\n"
            f"  python scripts/qlear/fit_field_half.py <unpacked-qlear-release> <fits-out-dir>\n"
            f"and then passing <fits-out-dir> to this script."
        )
    for path in fit_paths:
        with open(path) as fh:
            rec = json.load(fh)
        key = (rec["arm"], rec["seed"])
        if key in records:
            raise SystemExit(f"duplicate fit record for {key}; refusing to "
                             "silently keep one of them")
        records[key] = rec
    return records


def require_complete_roster(records):
    """Refuse to score anything unless every declared fit is present.

    The frozen text fixes the ladder, the seeds and the six-by-eight cohort. A
    missing arm, seed or file changes the estimand, so it stops the run instead
    of quietly shrinking the average.
    """
    missing = []
    for arm in list(LADDER) + [SECONDARY_ARM]:
        seeds = (0,) if arm in AFFINE_ARMS else EXPECTED_SEEDS
        for seed in seeds:
            if (arm, seed) not in records:
                missing.append(f"{arm}-seed{seed}")
    if missing:
        raise SystemExit("missing declared fits: " + ", ".join(missing))


def per_file_scores(records, panel, scorer):
    """errors[arm][seed][file] -> score, plus the failure ledger."""
    errors, failures = {}, []
    for (arm, seed), rec in records.items():
        block = rec["panels"][panel]
        for name, cell in block.items():
            value = scorer(cell["target"], cell["prediction"])
            reason = None
            if isinstance(value, tuple):
                value, reason = value
            if value is None:
                failures.append({"arm": arm, "seed": seed, "file": name,
                                 "reason": reason})
                continue
            errors.setdefault(arm, {}).setdefault(seed, {})[name] = value
    return errors, failures


class MissingEvaluation(Exception):
    """One application-and-backend cell has no score."""


def macro(scores, applications, backends):
    """Equal weight across backends inside an application, then across the
    applications supplied. ``applications`` may repeat under a resample.

    A missing cell raises rather than averaging over the survivors. The frozen
    text says a failed circuit evaluation fails explicitly and is never dropped
    silently, and quietly averaging seven backends instead of eight would change
    the aggregation weights without changing any reported number.
    """
    per_app = []
    for app in applications:
        vals = []
        for b in backends:
            key = f"{app}_{b}"
            if key not in scores:
                raise MissingEvaluation(key)
            vals.append(scores[key])
        per_app.append(sum(vals) / len(vals))
    return sum(per_app) / len(per_app)


def main():
    os.makedirs(OUT, exist_ok=True)
    records = load_fits()
    if not records:
        raise SystemExit(f"no fitted predictions under {FITS}")
    require_complete_roster(records)

    report = {"n_resamples": N_RESAMPLES, "root_seed": ROOT_SEED,
              "confidence": CONFIDENCE, "ladder": list(LADDER),
              "secondary_arm": SECONDARY_ARM,
              "primary_endpoints": ["C-F", "O-F"],
              "identity": "D = (C - O) + (O - F)",
              "score": "standard Hellinger on the released support",
              "panels": {}}

    for panel in ("hardware", "simulator"):
        if not any(panel in r["panels"] for r in records.values()):
            continue
        errors, failures = per_file_scores(records, panel, frozen_hellinger)
        repro, _ = per_file_scores(records, panel, release_hellinger)

        # Declared, not derived. The roster and the seed list come from the
        # frozen specification so that a deleted file or fit fails loudly.
        applications = list(EXPECTED_APPLICATIONS)
        backends = list(EXPECTED_BACKENDS)
        names = [f"{a}_{b}" for a in applications for b in backends]
        seeds = list(EXPECTED_SEEDS)

        absent = []
        for arm in list(LADDER) + [SECONDARY_ARM]:
            arm_seeds = (0,) if arm in AFFINE_ARMS else seeds
            for s in arm_seeds:
                have = errors.get(arm, {}).get(s, {})
                for name in names:
                    if name not in have:
                        absent.append({"arm": arm, "seed": s, "file": name,
                                       "reason": "no score produced"})
        failures = failures + absent
        block_status = "complete" if not absent else "incomplete_roster"
        block = {"applications": applications, "backends": backends,
                 "n_files": len(names), "roster_status": block_status,
                 "failures": failures,
                 "per_seed": {}, "per_file": {}}

        rng_master = np.random.default_rng(ROOT_SEED)
        for seed in seeds:
            # A is affine and fitted once; it pairs with every network seed.
            def arm_scores(arm):
                """This arm's scores for this seed.

                An affine arm is fitted once by declaration and pairs with every
                network seed. Any other arm must supply this exact seed; falling
                back to a different one would report a substituted fit as though
                it were the declared one.
                """
                by_seed = errors[arm]
                if arm in AFFINE_ARMS:
                    return by_seed[0]
                if seed not in by_seed:
                    raise MissingEvaluation(f"{arm}-seed{seed}")
                return by_seed[seed]

            try:
                point = {arm: macro(arm_scores(arm), applications, backends)
                         for arm in list(LADDER) + [SECONDARY_ARM]}
            except MissingEvaluation as missing:
                block["per_seed"][str(seed)] = {
                    "status": "not_estimable",
                    "reason": f"missing evaluation {missing}",
                }
                continue
            gaps = {
                "C-F": point["C"] - point["F"],
                "O-F": point["O"] - point["F"],
                "C-O": point["C"] - point["O"],
                "A-F": point["A"] - point["F"],
            }
            if SECONDARY_ARM in point:
                gaps["C4-F"] = point[SECONDARY_ARM] - point["F"]
            residual = gaps["C-F"] - (gaps["C-O"] + gaps["O-F"])

            # One joint resample of the six application circuits, shared by
            # every arm and every backend, drawn per fitted seed.
            rng = np.random.default_rng(
                rng_master.integers(0, 2**63 - 1, dtype=np.int64))
            draws = {k: [] for k in gaps}
            for _ in range(N_RESAMPLES):
                drawn = [applications[i] for i in
                         rng.integers(0, len(applications), len(applications))]
                m = {arm: macro(arm_scores(arm), drawn, backends)
                     for arm in point}
                draws["C-F"].append(m["C"] - m["F"])
                draws["O-F"].append(m["O"] - m["F"])
                draws["C-O"].append(m["C"] - m["O"])
                draws["A-F"].append(m["A"] - m["F"])
                if "C4-F" in draws:
                    draws["C4-F"].append(m[SECONDARY_ARM] - m["F"])

            lo_q, hi_q = (1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100
            intervals = {}
            for k, v in draws.items():
                arr = np.asarray(v, dtype=float)
                intervals[k] = {
                    "estimate": float(gaps[k]),
                    "lower": float(np.percentile(arr, lo_q)),
                    "upper": float(np.percentile(arr, hi_q)),
                }

            block["per_seed"][str(seed)] = {
                "status": "estimated",
                "rung_errors": {k: float(v) for k, v in point.items()},
                "gaps": intervals,
                "identity_residual": float(residual),
                "epochs": {arm: records[(arm, seed)]["epochs"]
                           for arm in point if (arm, seed) in records},
                "validation_mse": {arm: records[(arm, seed)]["validation_mse"]
                                   for arm in point if (arm, seed) in records},
            }

        # All 48 paired results, reported rather than summarized away.
        for name in names:
            block["per_file"][name] = {
                arm: {str(s): errors[arm][s][name] for s in errors[arm]
                      if name in errors[arm][s]}
                for arm in errors
            }
        block["reproduction_panel"] = {
            name: {str(s): repro["F"][s][name] for s in repro["F"]
                   if name in repro["F"][s]}
            for name in names
        } if "F" in repro else {}

        report["panels"][panel] = block

    if OUT.endswith(".json"):
        out_file = OUT
        out_dir = os.path.dirname(OUT)
    else:
        out_dir = OUT
        out_file = os.path.join(OUT, "field-report.json")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out_file}")

    for panel, block in report["panels"].items():
        print(f"\n=== {panel}: {block['n_files']} files, "
              f"{len(block['per_seed'])} fitted seeds, "
              f"{len(block['failures'])} failed evaluations ===")
        for seed, s in sorted(block["per_seed"].items(), key=lambda kv: int(kv[0])):
            if s.get("status") != "estimated":
                print(f" seed {seed}: {s.get('status')} ({s.get('reason')})")
                continue
            r = s["rung_errors"]
            g = s["gaps"]
            print(f" seed {seed}: " +
                  "  ".join(f"{k}={r[k]:.4f}" for k in sorted(r)) +
                  f"   C-F={g['C-F']['estimate']:+.4f}"
                  f" [{g['C-F']['lower']:+.4f},{g['C-F']['upper']:+.4f}]"
                  f"   O-F={g['O-F']['estimate']:+.4f}"
                  f" [{g['O-F']['lower']:+.4f},{g['O-F']['upper']:+.4f}]"
                  f"   resid={s['identity_residual']:.2e}")


if __name__ == "__main__":
    main()
