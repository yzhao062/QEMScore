"""QRAFT matched retraining with machine identity in every arm (round-10 rule, Part D).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part D.

QRAFT's released training code adds machine identity (ComputerID) only in the
seventeen-feature arm, together with the reverse statistics, so the step from
ten to seventeen features mixes the two. This script refits, under the matched
recipes of ``score_qraft_panel.py`` (row split, panel M) and of
``score_qraft_grouped.py`` (circuit-level groups: machine and the six circuit
descriptors), these arms:

  Q3    seven descriptors                                   (published)
  Q2    Q3 + forward probability percentiles                (published)
  Q1    Q2 + reverse statistics + machine identity          (published)
  Q3m   Q3 + machine identity
  Q2m   Q2 + machine identity
  Q1nm  Q2 + reverse statistics, without machine identity

and reports, as ten-seed means with the panel's joint bootstrap intervals:

  reverse increment, machine in both arms       Q2m - Q1
  reverse increment, machine in neither arm     Q2 - Q1nm
  machine increment in the forward arm          Q2 - Q2m
  machine increment in the full arm             Q1nm - Q1
  machine increment in the descriptor arm       Q3 - Q3m
  descriptor arm minus forward arm, machine in both   Q3m - Q2m (and the ratio)
  the published step                            Q2 - Q1 = (Q2 - Q2m) + (Q2m - Q1)

The published arms are fitted first and must reproduce the released reports
exactly before any new arm is fitted.

Usage:
  python reanalysis/scripts/qraft/score_qraft_machine_matched.py <QRAFT dir> <out dir> \\
      --frozen-rule docs/frozen-rules/2026-10-07-round10-checks.md
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_qraft_panel as sp  # noqa: E402

RULE_FILE = "docs/frozen-rules/2026-10-07-round10-checks.md"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PANEL_REPORT = os.path.join(REPO, "reanalysis/outputs/qraft/round9/qraft-panel-report.json")
GROUPED_REPORT = os.path.join(REPO, "reanalysis/outputs/qraft/qraft-grouped-report.json")
CIRCUIT_DESCRIPTORS_6 = sp.DESCRIPTORS[:6]
ARMS = {
    "Q3": sp.DESCRIPTORS,
    "Q2": sp.DESCRIPTORS + sp.FORWARD,
    "Q1": sp.DESCRIPTORS + sp.FORWARD + sp.REVERSE + (sp.MACHINE,),
    "Q3m": sp.DESCRIPTORS + (sp.MACHINE,),
    "Q2m": sp.DESCRIPTORS + sp.FORWARD + (sp.MACHINE,),
    "Q1nm": sp.DESCRIPTORS + sp.FORWARD + sp.REVERSE,
}
CONTRASTS = {
    "reverse_increment_machine_in_both": ("Q2m", "Q1"),
    "reverse_increment_machine_in_neither": ("Q2", "Q1nm"),
    "machine_increment_forward_arm": ("Q2", "Q2m"),
    "machine_increment_full_arm": ("Q1nm", "Q1"),
    "machine_increment_descriptor_arm": ("Q3", "Q3m"),
    "descriptor_minus_forward_machine_in_both": ("Q3m", "Q2m"),
    "published_step": ("Q2", "Q1"),
}
PUBLISHED = ("Q3", "Q2", "Q1")
TOL = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def summarize(per_seed, draws):
    """Ten-seed means, ranges, and joint-bootstrap intervals for arms and contrasts."""
    seeds = sorted(per_seed["Q1"])
    out = {"arms": {}, "contrasts": {}}
    for arm in per_seed:
        v = [per_seed[arm][s] for s in seeds]
        out["arms"][arm] = {"features": list(ARMS[arm]), "ten_seed_mean": float(np.mean(v)),
                            "ten_seed_min": float(min(v)), "ten_seed_max": float(max(v)),
                            "lower": float(np.percentile(draws[arm], 2.5)),
                            "upper": float(np.percentile(draws[arm], 97.5))}
    # The published-arm check reads the arms only. The Q2 - Q1 interval is a
    # governed Part D contrast, so it is computed with the new arms, after the check.
    if set(per_seed) == set(PUBLISHED):
        return out
    for name, (a, b) in CONTRASTS.items():
        if a not in per_seed or b not in per_seed:
            continue
        diffs = [per_seed[a][s] - per_seed[b][s] for s in seeds]
        d = draws[a] - draws[b]
        out["contrasts"][name] = {
            "arms": [a, b], "ten_seed_mean": float(np.mean(diffs)),
            "ten_seed_min": float(min(diffs)), "ten_seed_max": float(max(diffs)),
            "seeds_positive": int(sum(x > 0 for x in diffs)),
            "lower": float(np.percentile(d, 2.5)), "upper": float(np.percentile(d, 97.5))}
    if "Q3m" not in per_seed:
        return out
    ratio = draws["Q3m"] / draws["Q2m"]
    out["ratio_Q3m_over_Q2m"] = {
        "point": out["arms"]["Q3m"]["ten_seed_mean"] / out["arms"]["Q2m"]["ten_seed_mean"],
        "lower": float(np.percentile(ratio, 2.5)), "upper": float(np.percentile(ratio, 97.5))}
    return out


def row_split(in_rows, arms):
    """Panel M of score_qraft_panel.py with the given arm set."""
    saved = sp.ARMS
    sp.ARMS = arms
    try:
        result, boot = sp.panel_matched(in_rows, include_raw=True, return_draws=True)
    finally:
        sp.ARMS = saved
    per_seed = {arm: {int(s): v for s, v in result[arm]["per_seed_macro_mae"].items()}
                for arm in arms}
    out = summarize(per_seed, boot["arm_draws"])
    out["raw_R"] = {k: result["R"][k] for k in ("ten_seed_mean", "lower", "upper")}
    return out


def grouped(in_rows, group_ids, arms):
    """Circuit-level grouping with the joint group bootstrap of score_qraft_grouped.py."""
    machines = np.array([r[sp.MACHINE] for r in in_rows])
    y = sp.numeric(in_rows, (sp.TARGET,)).ravel()
    x = {arm: sp.numeric(in_rows, cols) for arm, cols in arms.items()}
    records = []
    per_seed = {arm: {} for arm in arms}
    for seed in sp.SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=sp.TEST_FRACTION, random_state=seed)
        train, test = next(gss.split(in_rows, groups=group_ids))
        assert not set(group_ids[train]) & set(group_ids[test])
        errs = {}
        for arm in arms:
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(x[arm][train], y[train])
            errs[arm] = np.abs(model.predict(x[arm][test]) - y[test])
            per_seed[arm][seed], _ = sp.macro_mae(machines[test], errs[arm])
        records.append({"machines": machines[test], "groups": group_ids[test], "errs": errs})

    u_joint = np.unique(np.concatenate([r["groups"] for r in records]))
    k_joint = len(u_joint)
    rng = np.random.default_rng(sp.BOOTSTRAP_SEED)
    picks = rng.integers(0, k_joint, size=(sp.BOOTSTRAP_DRAWS, k_joint))
    w_joint = np.zeros((sp.BOOTSTRAP_DRAWS, k_joint))
    for b in range(sp.BOOTSTRAP_DRAWS):
        w_joint[b] = np.bincount(picks[b], minlength=k_joint)
    draws = {arm: np.zeros(sp.BOOTSTRAP_DRAWS) for arm in arms}
    for rec in records:
        u_groups = np.unique(rec["groups"])
        machs = sorted(set(rec["machines"]))
        row_group = np.searchsorted(u_groups, rec["groups"])
        row_mach = np.array([machs.index(m) for m in rec["machines"]])
        counts = np.zeros((len(u_groups), len(machs)))
        np.add.at(counts, (row_group, row_mach), 1.0)
        w = w_joint[:, np.searchsorted(u_joint, u_groups)]
        for arm in arms:
            sums = np.zeros((len(u_groups), len(machs)))
            np.add.at(sums, (row_group, row_mach), rec["errs"][arm])
            # Accelerate BLAS raises spurious flags in matmul on finite inputs
            # (Appendix M.4); silence them and require a finite result instead.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                per_m = np.column_stack([(w @ sums[:, m]) / (w @ counts[:, m])
                                         for m in range(len(machs))])
            if not np.all(np.isfinite(per_m)):
                raise FloatingPointError("non-finite grouped bootstrap draw")
            draws[arm] += per_m.mean(axis=1) / len(sp.SEEDS)
    return summarize(per_seed, draws)


def check_published(row_res, grp_res):
    panel = json.load(open(PANEL_REPORT))["panel_M_matched"]
    worst = 0.0
    for arm in ("Q3", "Q2", "Q1"):
        for k in ("ten_seed_mean", "lower", "upper"):
            worst = max(worst, abs(row_res["arms"][arm][k] - panel[arm][k]))
    for k in ("ten_seed_mean", "lower", "upper"):
        worst = max(worst, abs(row_res["raw_R"][k] - panel["R"][k]))
    grp = json.load(open(GROUPED_REPORT))["grouped_retraining_6_circuit_descriptors"]
    for arm in ("Q3", "Q2", "Q1"):
        worst = max(worst, abs(grp_res["arms"][arm]["ten_seed_mean"]
                               - grp["arms"][arm]["ten_seed_mean"]))
        lo, hi = grp["bootstrap_joint"][f"{arm.lower()}_macro_mae_95_ci"]
        worst = max(worst, abs(grp_res["arms"][arm]["lower"] - lo),
                    abs(grp_res["arms"][arm]["upper"] - hi))
    if worst > TOL:
        raise SystemExit(f"published arms differ from the released reports by {worst}")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("release", help="QRAFT directory or a directory holding QRAFT/")
    p.add_argument("outdir")
    p.add_argument("--frozen-rule", required=True)
    p.add_argument("--reproduce-only", action="store_true",
                   help="fit only the three published arms and check them; no new arm")
    args = p.parse_args()
    if os.path.realpath(args.frozen_rule) != os.path.realpath(os.path.join(REPO, RULE_FILE)):
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")
    qraft = args.release if os.path.isdir(os.path.join(args.release, "data_trained")) \
        else os.path.join(args.release, "QRAFT")
    in_csv = os.path.join(qraft, "data_processed", "inputData.csv")
    in_rows = sp.load(in_csv, sp.INPUT_COLUMNS)
    if len(in_rows) != sp.EXPECTED_INPUT_ROWS:
        raise SystemExit(f"inputData.csv: expected {sp.EXPECTED_INPUT_ROWS} rows")
    for arm, cols in ARMS.items():
        if sp.TARGET in cols:
            raise SystemExit(f"{arm}: the target is a feature")
    g6 = [tuple(r[c] for c in (sp.MACHINE,) + CIRCUIT_DESCRIPTORS_6) for r in in_rows]
    _, g6_ids = np.unique(g6, axis=0, return_inverse=True)
    # The published arms are fitted and checked first; no new arm is fitted
    # unless they reproduce the released reports.
    published = {a: ARMS[a] for a in PUBLISHED}
    worst = check_published(row_split(in_rows, published),
                            grouped(in_rows, g6_ids.ravel(), published))
    print(f"published arms reproduce within {worst}", flush=True)
    if args.reproduce_only:
        return
    row_res = row_split(in_rows, ARMS)
    grp_res = grouped(in_rows, g6_ids.ravel(), ARMS)
    if check_published(row_res, grp_res) != worst:
        raise SystemExit("published arms changed between the check and the full fit")
    report = {
        "frozen_rule": RULE_FILE,
        "frozen_rule_sha256": sha256(os.path.join(REPO, RULE_FILE)),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "input_sha256": {"inputData.csv": sha256(in_csv)},
        "published_arms_max_abs_diff": worst,
        "score": "per-row mean absolute error in probability points",
        "row_split": row_res,
        "circuit_level_groups": {**grp_res, "groups": int(len(np.unique(g6_ids)))},
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "qraft-machine-matched.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}; published arms reproduce within {worst}")


if __name__ == "__main__":
    main()
