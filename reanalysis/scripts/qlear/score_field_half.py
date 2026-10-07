"""Score the fitted ladder and unmitigated baselines under the frozen contract.

The frozen attribution score is standard Hellinger on the released support.
The reproduction panel is the release RQ2 body, which renormalizes differently.
Saved analyses computed under the two cannot be mixed into one decomposition.

Aggregation and resampling are frozen.
One distribution error is evaluated per application and backend.
It is averaged equally across eight backends and six applications.
Application circuits are resampled jointly across backends and predictors.

The raw baseline arm R and opposite ladder O2 are governed by
docs/frozen-rules/2026-10-06-round9-follow-ups.md.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

N_RESAMPLES = 10000
ROOT_SEED = 20260906
CONFIDENCE = 0.95

LADDER = ("A", "C", "O", "F")
SECONDARY_ARM = "C4"

EXPECTED_APPLICATIONS = ("groundstate", "pricingcall", "pricingput",
                         "qaoa", "routing", "tsp")
EXPECTED_BACKENDS = ("ibm_lagos", "ibm_nairobi", "ibm_perth", "ibmq_belem",
                     "ibmq_jakarta", "ibmq_lima", "ibmq_manila", "ibmq_quito")
EXPECTED_SEEDS = tuple(range(10))
AFFINE_ARMS = ("A",)
EXPECTED_O2_COLUMNS = [
    "Num_1Q_Gates", "Num_2Q_Gates", "circuit_depth", "circuit_width",
    "state_weight", "Avg_inverted_error_25", "Avg_inverted_error_50", "Avg_inverted_error_75"
]


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


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


def load_fits(fits_dir, include_opposite=False):
    records = {}
    fit_paths = sorted(glob.glob(os.path.join(fits_dir, "pred-*.json")))
    if not fit_paths:
        raise SystemExit(
            f"No pred-*.json files found in '{fits_dir}'. Rescoring requires "
            "refitting the ladder first using:\n"
            f"  python scripts/qlear/fit_field_half.py <unpacked-qlear-release> <fits-out-dir>\n"
            f"and then passing <fits-out-dir> to this script."
        )
    for path in fit_paths:
        with open(path) as fh:
            rec = json.load(fh)
        if not include_opposite and rec.get("arm") == "O2":
            continue
        key = (rec["arm"], rec["seed"])
        if key in records:
            raise SystemExit(f"duplicate fit record for {key}; refusing to "
                             "silently keep one of them")
        records[key] = rec
    return records


def validate_target_identity(records, raw_evals=None, active_panels=None):
    """Require identical target vectors, lengths, and support per panel and file."""
    if active_panels is None:
        active_panels = ("hardware", "simulator")

    for panel in active_panels:
        all_files = set()
        for rec in records.values():
            if panel in rec.get("panels", {}):
                all_files.update(rec["panels"][panel].keys())
        if raw_evals and panel in raw_evals:
            all_files.update(raw_evals[panel].keys())

        for fname in sorted(all_files):
            ref_target = None
            ref_source = None

            for (arm, seed), rec in sorted(records.items()):
                p_block = rec.get("panels", {}).get(panel, {})
                if fname not in p_block:
                    continue
                t = p_block[fname].get("target")
                if t is None:
                    raise SystemExit(f"Target vector missing for {arm}-seed{seed} {panel}/{fname}")
                t_arr = np.asarray(t, dtype=float)
                if len(t_arr) == 0:
                    raise SystemExit(f"Empty target vector for {arm}-seed{seed} {panel}/{fname}")
                if not np.isfinite(t_arr).all():
                    raise SystemExit(f"Non-finite target values for {arm}-seed{seed} {panel}/{fname}")
                if t_arr.sum() <= 0:
                    raise SystemExit(f"Non-positive target sum for {arm}-seed{seed} {panel}/{fname}")

                if ref_target is None:
                    ref_target = t_arr
                    ref_source = f"{arm}-seed{seed}"
                else:
                    if len(t_arr) != len(ref_target):
                        raise SystemExit(
                            f"TARGET LENGTH MISMATCH in {panel}/{fname}: "
                            f"{arm}-seed{seed} has length {len(t_arr)} but {ref_source} has {len(ref_target)}"
                        )
                    ref_supp = np.where(ref_target > 0.0)[0]
                    t_supp = np.where(t_arr > 0.0)[0]
                    if not np.array_equal(ref_supp, t_supp):
                        raise SystemExit(
                            f"TARGET SUPPORT MISMATCH in {panel}/{fname}: "
                            f"{arm}-seed{seed} support differs from {ref_source}"
                        )
                    if not np.array_equal(t_arr, ref_target):
                        raise SystemExit(
                            f"TARGET VALUE MISMATCH in {panel}/{fname}: "
                            f"{arm}-seed{seed} targets differ from {ref_source}"
                        )

            if raw_evals and panel in raw_evals and fname in raw_evals[panel]:
                t_raw, _ = raw_evals[panel][fname]
                t_raw_arr = np.asarray(t_raw, dtype=float)
                if ref_target is not None:
                    if len(t_raw_arr) != len(ref_target):
                        raise SystemExit(
                            f"RAW TARGET LENGTH MISMATCH in {panel}/{fname}: "
                            f"raw CSV has length {len(t_raw_arr)} but {ref_source} has {len(ref_target)}"
                        )
                    ref_supp = np.where(ref_target > 0.0)[0]
                    raw_supp = np.where(t_raw_arr > 0.0)[0]
                    if not np.array_equal(ref_supp, raw_supp):
                        raise SystemExit(
                            f"RAW TARGET SUPPORT MISMATCH in {panel}/{fname}: "
                            f"raw CSV support differs from {ref_source}"
                        )
                    if not np.array_equal(t_raw_arr, ref_target):
                        raise SystemExit(
                            f"RAW TARGET VALUE MISMATCH in {panel}/{fname}: "
                            f"raw CSV targets differ from {ref_source}"
                        )


def validate_o2_metadata(records, expected_release=None):
    """Check O2 recorded input columns, network kind, and release identity."""
    o2_records = [rec for (arm, seed), rec in records.items() if arm == "O2"]
    if not o2_records:
        return

    ref_rel_id = None
    for rec in o2_records:
        seed = rec.get("seed")
        kind = rec.get("kind")
        if kind != "network":
            raise SystemExit(f"O2 KIND MISMATCH for seed {seed}: expected 'network', got {kind!r}")
        cols = rec.get("columns")
        if cols != EXPECTED_O2_COLUMNS:
            raise SystemExit(
                f"O2 COLUMNS MISMATCH for seed {seed}: expected {EXPECTED_O2_COLUMNS}, got {cols}"
            )
        rel_id = rec.get("release_identity")
        if rel_id is None:
            raise SystemExit(f"O2 RELEASE IDENTITY MISSING for seed {seed}")

        if ref_rel_id is None:
            ref_rel_id = rel_id
        elif rel_id != ref_rel_id:
            raise SystemExit(
                f"O2 RELEASE IDENTITY INCONSISTENCY between seeds: {rel_id} != {ref_rel_id}"
            )

        if expected_release is not None and isinstance(rel_id, dict):
            expected_base = os.path.basename(os.path.normpath(expected_release))
            rec_base = rel_id.get("release_dir")
            if rec_base and rec_base != expected_base and expected_base not in rec_base and rec_base not in expected_base:
                raise SystemExit(
                    f"O2 RELEASE IDENTITY MISMATCH: O2 was fitted on '{rec_base}' "
                    f"but scorer is using release '{expected_base}'"
                )


def find_release_folder(release_path, folder_name):
    candidate = os.path.join(release_path, folder_name)
    if os.path.isdir(candidate):
        return candidate
    matches = glob.glob(os.path.join(release_path, "*", folder_name))
    for m in matches:
        if os.path.isdir(m):
            return m
    raise FileNotFoundError(f"Folder '{folder_name}' not found under '{release_path}'")


def load_raw_arm(release_path, panels):
    """Load observed_prob_50 from release evaluation files for arm R."""
    folder_map = {
        "hardware": "real_circuits_hardware",
        "simulator": "real_circuits",
    }
    raw_scores = {}
    raw_shas = {}
    for panel in panels:
        folder = find_release_folder(release_path, folder_map[panel])
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not csv_files:
            raise SystemExit(f"No csv files found under {folder}")
        raw_scores[panel] = {}
        for path in csv_files:
            name = os.path.basename(path)[:-4]
            raw_shas[os.path.relpath(path, release_path)] = compute_file_sha256(path)
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if not rows:
                raise SystemExit(f"{path}: empty file")
            if "observed_prob_50" not in rows[0] or "target" not in rows[0]:
                raise SystemExit(f"{path}: missing observed_prob_50 or target column")
            target = [float(r["target"]) for r in rows]
            observed = [float(r["observed_prob_50"]) for r in rows]
            raw_scores[panel][name] = (target, observed)
    return raw_scores, raw_shas


def require_complete_roster(records, include_opposite=False):
    """Refuse to score anything unless every declared fit is present."""
    missing = []
    required_arms = list(LADDER) + [SECONDARY_ARM]
    if include_opposite:
        required_arms.append("O2")
    for arm in required_arms:
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
        if panel not in rec.get("panels", {}):
            continue
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
    """Equal weight across backends inside an application, then across applications."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("fits", help="Directory containing pred-*.json files")
    parser.add_argument("out", help="Output directory or field-report.json path")
    parser.add_argument(
        "--release",
        default=None,
        help="Path to unpacked Q-LEAR release for arm R evaluation files",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Include unmitigated baseline arm R (reads observed probabilities from release)",
    )
    parser.add_argument(
        "--opposite",
        "--opposite-ladder",
        action="store_true",
        dest="opposite",
        default=False,
        help="Evaluate opposite ladder C -> O2 -> F and Shapley shares",
    )
    parser.add_argument(
        "--frozen-rule",
        default=None,
        help="Path to frozen rule file (required when --raw or --opposite is requested)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fits_dir = args.fits
    out = args.out
    include_raw = args.raw or (args.release is not None)
    include_opposite = args.opposite

    if include_raw or include_opposite:
        if not args.frozen_rule or not os.path.isfile(args.frozen_rule):
            raise RuntimeError(
                "FROZEN RULE BARRIER: Governed outcomes (arm R, opposite ladder O2, "
                "or Shapley shares) require a valid, existing --frozen-rule file. "
                f"Provided path: {args.frozen_rule!r}. Aborting immediately."
            )

    if args.raw and not args.release:
        raise SystemExit("--release <path> is required when --raw is requested.")

    records = load_fits(fits_dir, include_opposite=include_opposite)
    if not records:
        raise SystemExit(f"no fitted predictions under {fits_dir}")

    if include_opposite:
        missing_o2 = [f"O2-seed{s}" for s in EXPECTED_SEEDS if ("O2", s) not in records]
        if missing_o2:
            raise SystemExit(
                f"Opposite ladder requested via --opposite, but O2 fits are absent from '{fits_dir}': "
                + ", ".join(missing_o2)
            )
        validate_o2_metadata(records, expected_release=args.release)

    require_complete_roster(records, include_opposite=include_opposite)

    report = {
        "n_resamples": N_RESAMPLES,
        "root_seed": ROOT_SEED,
        "confidence": CONFIDENCE,
        "ladder": list(LADDER),
        "secondary_arm": SECONDARY_ARM,
        "primary_endpoints": ["C-F", "O-F"],
        "identity": "D = (C - O) + (O - F)",
        "score": "standard Hellinger on the released support",
        "panels": {},
    }

    if include_opposite:
        report["opposite_ladder"] = ["C", "O2", "F"]
        report["opposite_identity"] = "D = (C - O2) + (O2 - F)"
    if include_raw:
        report["raw_contrasts"] = ["F-R", "C-R"]

    input_shas = {}
    fit_paths = sorted(glob.glob(os.path.join(fits_dir, "pred-*.json")))
    for fp in fit_paths:
        if not include_opposite and os.path.basename(fp).startswith("pred-O2-"):
            continue
        input_shas[os.path.basename(fp)] = compute_file_sha256(fp)

    active_panels = [p for p in ("hardware", "simulator") if any(p in r.get("panels", {}) for r in records.values())]

    raw_evals = {}
    if include_raw:
        raw_evals, raw_file_shas = load_raw_arm(args.release, active_panels)
        input_shas.update(raw_file_shas)

    # Validate target identity across all arms and raw rows before scoring
    validate_target_identity(records, raw_evals=raw_evals, active_panels=active_panels)

    for panel in active_panels:
        errors, failures = per_file_scores(records, panel, frozen_hellinger)
        repro, _ = per_file_scores(records, panel, release_hellinger)

        if include_raw and panel in raw_evals:
            for name, (t_vec, obs_vec) in raw_evals[panel].items():
                val, reason = frozen_hellinger(t_vec, obs_vec)
                if val is None:
                    failures.append({"arm": "R", "seed": 0, "file": name, "reason": reason})
                else:
                    errors.setdefault("R", {}).setdefault(0, {})[name] = val

        applications = list(EXPECTED_APPLICATIONS)
        backends = list(EXPECTED_BACKENDS)
        names = [f"{a}_{b}" for a in applications for b in backends]
        seeds = list(EXPECTED_SEEDS)

        check_arms = list(LADDER) + [SECONDARY_ARM]
        if include_opposite:
            check_arms.append("O2")
        if include_raw:
            check_arms.append("R")

        absent = []
        for arm in check_arms:
            arm_seeds = (0,) if (arm in AFFINE_ARMS or arm == "R") else seeds
            for s in arm_seeds:
                have = errors.get(arm, {}).get(s, {})
                for name in names:
                    if name not in have:
                        absent.append({"arm": arm, "seed": s, "file": name,
                                       "reason": "no score produced"})
        failures = failures + absent
        block_status = "complete" if not absent else "incomplete_roster"
        block = {
            "applications": applications,
            "backends": backends,
            "n_files": len(names),
            "roster_status": block_status,
            "failures": failures,
            "per_seed": {},
            "per_file": {},
        }

        rng_master = np.random.default_rng(ROOT_SEED)
        for seed in seeds:
            def arm_scores(arm):
                by_seed = errors[arm]
                if arm in AFFINE_ARMS or arm == "R":
                    return by_seed[0]
                if seed not in by_seed:
                    raise MissingEvaluation(f"{arm}-seed{seed}")
                return by_seed[seed]

            try:
                point_arms = list(LADDER) + [SECONDARY_ARM]
                if include_opposite:
                    point_arms.append("O2")
                if include_raw:
                    point_arms.append("R")
                point = {arm: macro(arm_scores(arm), applications, backends)
                         for arm in point_arms}
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
            if include_raw:
                gaps["F-R"] = point["F"] - point["R"]
                gaps["C-R"] = point["C"] - point["R"]
            if include_opposite:
                gaps["C-O2"] = point["C"] - point["O2"]
                gaps["O2-F"] = point["O2"] - point["F"]

            residual = gaps["C-F"] - (gaps["C-O"] + gaps["O-F"])
            opposite_residual = (
                gaps["C-F"] - (gaps["C-O2"] + gaps["O2-F"]) if include_opposite else None
            )

            rng = np.random.default_rng(
                rng_master.integers(0, 2**63 - 1, dtype=np.int64))
            draws = {k: [] for k in gaps}
            s_b_draws = []
            s_d_draws = []
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
                if include_raw:
                    draws["F-R"].append(m["F"] - m["R"])
                    draws["C-R"].append(m["C"] - m["R"])
                if include_opposite:
                    draws["C-O2"].append(m["C"] - m["O2"])
                    draws["O2-F"].append(m["O2"] - m["F"])
                    delta_b = m["C"] - m["F"]
                    phi_b_draw = 0.5 * ((m["C"] - m["O"]) + (m["O2"] - m["F"]))
                    phi_d_draw = 0.5 * ((m["O"] - m["F"]) + (m["C"] - m["O2"]))
                    if delta_b != 0.0:
                        s_b_draws.append(phi_b_draw / delta_b)
                        s_d_draws.append(phi_d_draw / delta_b)
                    else:
                        s_b_draws.append(np.nan)
                        s_d_draws.append(np.nan)

            lo_q, hi_q = (1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100
            intervals = {}
            for k, v in draws.items():
                arr = np.asarray(v, dtype=float)
                intervals[k] = {
                    "estimate": float(gaps[k]),
                    "lower": float(np.percentile(arr, lo_q)),
                    "upper": float(np.percentile(arr, hi_q)),
                }

            seed_record = {
                "status": "estimated",
                "rung_errors": {k: float(v) for k, v in point.items()},
                "gaps": intervals,
                "identity_residual": float(residual),
                "epochs": {arm: records[(arm, seed)]["epochs"]
                           for arm in point if (arm, seed) in records},
                "validation_mse": {arm: records[(arm, seed)]["validation_mse"]
                                   for arm in point if (arm, seed) in records},
            }

            if include_opposite:
                delta_total = gaps["C-F"]
                phi_b = 0.5 * (gaps["C-O"] + gaps["O2-F"])
                phi_d = 0.5 * (gaps["O-F"] + gaps["C-O2"])
                s_b = (phi_b / delta_total) if delta_total != 0.0 else 0.0
                s_d = (phi_d / delta_total) if delta_total != 0.0 else 0.0
                seed_record["opposite_identity_residual"] = float(opposite_residual)
                seed_record["shapley"] = {
                    "base_execution": {
                        "value": float(phi_b),
                        "share": float(s_b),
                        "share_lower": float(np.percentile(s_b_draws, lo_q)),
                        "share_upper": float(np.percentile(s_b_draws, hi_q)),
                    },
                    "dpe": {
                        "value": float(phi_d),
                        "share": float(s_d),
                        "share_lower": float(np.percentile(s_d_draws, lo_q)),
                        "share_upper": float(np.percentile(s_d_draws, hi_q)),
                    },
                }

            block["per_seed"][str(seed)] = seed_record

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

    if include_raw or include_opposite:
        report["frozen_rule"] = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
        report["frozen_rule_sha256"] = compute_file_sha256(args.frozen_rule)
        report["script_sha256"] = compute_file_sha256(__file__)
        report["input_sha256"] = input_shas

    if out.endswith(".json"):
        out_file = out
        out_dir = os.path.dirname(out)
    else:
        out_dir = out
        out_file = os.path.join(out, "field-report.json")
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
            out_str = (
                f" seed {seed}: "
                + "  ".join(f"{k}={r[k]:.4f}" for k in sorted(r))
                + f"   C-F={g['C-F']['estimate']:+.4f}"
                f" [{g['C-F']['lower']:+.4f},{g['C-F']['upper']:+.4f}]"
                f"   O-F={g['O-F']['estimate']:+.4f}"
                f" [{g['O-F']['lower']:+.4f},{g['O-F']['upper']:+.4f}]"
            )
            if include_raw and "F-R" in g:
                out_str += (
                    f"   F-R={g['F-R']['estimate']:+.4f}"
                    f" [{g['F-R']['lower']:+.4f},{g['F-R']['upper']:+.4f}]"
                )
            if include_opposite and "shapley" in s:
                sh_b = s["shapley"]["base_execution"]
                sh_d = s["shapley"]["dpe"]
                out_str += (
                    f"   Shapley(base)={sh_b['share']:.4f}"
                    f" [{sh_b['share_lower']:.4f},{sh_b['share_upper']:.4f}]"
                    f"   Shapley(dpe)={sh_d['share']:.4f}"
                    f" [{sh_d['share_lower']:.4f},{sh_d['share_upper']:.4f}]"
                )
            out_str += f"   resid={s['identity_residual']:.2e}"
            print(out_str)


if __name__ == "__main__":
    main()
