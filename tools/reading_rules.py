"""Oracle-free reading rules for D, scored against the Bayes oracle (round-10 rule, Part A).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part A.

For every fit-set cell of the Bayes oracle's confusion table (696 cells of 44
fit sets, ``artifacts/descriptor-information/round9/bayes-oracle.json``), the
script computes quantities that read only the fits' validation and test
predictions, never the oracle:

- D = C - F of the fit set (recomputed; it must equal analysis A).
- The stacking increment of r over the fit set's C: per cell, y ~ a + b C and
  y ~ a + b C + c r fitted on validation rows and applied to test rows,
  increment = macro MAE of the first minus macro MAE of the second.
- The strongest reference Ref*: among the descriptor-only candidates of the
  same data's stronger-candidate fits (random forest, MLP, gradient boosting,
  degree-5 ridge polynomial; arm C, learner seeds 1 to 20), the candidate with
  the lowest validation error for the row's family, averaged over learner
  seeds. D_ref = Ref* - F pairs learner seed k of Ref* with learner seed k of F.
- The stacking increment of r over Ref*.

Every interval is the rule's two-stage bootstrap (learner seeds, then whole
test circuits; 10,000 draws; seed 20261002; the arithmetic of Appendix M.4),
from the same draws that analysis A used. The rules:

- rule_D: information if the D interval lies above zero (the pipeline label).
- rule_S: information if the stacking increment over C lies above zero.
- rule_R (primary): information if the D_ref interval lies above zero.
- rule_RS: rule_R and the stacking increment over Ref* lies above zero.
- rule_R_delta: the D_ref interval lies above the family's margin delta of
  Part D of the round-9 rule (a tenth of mean calibrated raw error).

Each is scored against the oracle's reading of the same cell (information,
no information at R0, undetermined).

Usage:
  PYTHONPATH=. python tools/reading_rules.py --assets A --crossed C --fresh F \\
      --frozen-rule docs/frozen-rules/2026-10-07-round10-checks.md
  PYTHONPATH=. python tools/reading_rules.py ... --reproduce-only
      (plain D for all 44 sets and the stacking increment for the ten sets of
      posthoc-stacking-all.json, checked against the released values; no new
      quantity is computed)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import descriptor_ladder_analysis as dla  # noqa: E402

RULE_FILE = "docs/frozen-rules/2026-10-07-round10-checks.md"
REGISTRY = _REPO / "tools/round10_registry.json"
MAPPING = _REPO / "tools/bayes_oracle_mapping.json"
ORACLE = _REPO / "artifacts/descriptor-information/round9/bayes-oracle.json"
POOLED = _REPO / "artifacts/descriptor-information/round9/pooled-absolute.json"
STACKING_ALL = _REPO / "artifacts/descriptor-information/posthoc-stacking-all.json"
DEFAULT_OUT = _REPO / "artifacts/descriptor-information/round10/reading-rules.json"
DEFAULT_MD = _REPO / "artifacts/descriptor-information/round10/reading-rules.md"

CANDIDATES = ("random_forest", "mlp", "hgbr", "poly5_ridge")
LEARNER_SEEDS = tuple(range(1, 21))
DRAWS = dla.DEFAULT_DRAWS
SEED = dla.RULE_SEED
TOL = 1e-12
RULES = ("rule_D", "rule_S", "rule_R", "rule_RS", "rule_R_delta")
PRIMARY_RULE = "rule_R"
# Tie order among rules eligible to replace the primary rule (Part A, Reading).
ALTERNATIVES = ("rule_D", "rule_S", "rule_RS", "rule_R_delta")
SENSITIVITY_SLACK = 0.05
CELL_KEYS = ("noise_family", "severity", "observable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(template: str, roots: dict[str, Path]) -> Path:
    for name, root in roots.items():
        template = template.replace("{" + name + "}", str(root))
    if "{" in template:
        raise SystemExit(f"unresolved root in {template}")
    return Path(template)


def group_of(fit_set: str, shot_level) -> str:
    """The row group of the paper's confusion table."""
    if str(shot_level) == "exact":
        return "exact"
    if str(shot_level) == "2048":
        return "new_2048" if fit_set.startswith("fresh_") else "first_2048"
    return "other_finite"


def truth_of(row: dict) -> str:
    if row["rung"] == "R0":
        return "no information"
    return "information" if float(row["G_star_interval"]["lower"]) > 0.0 else "undetermined"


# --------------------------------------------------------------------------
# Bootstrap arithmetic on per-seed item errors
# --------------------------------------------------------------------------


def seed_mean_draws(row: dla.Row, macro: np.ndarray, n: int) -> np.ndarray:
    seed_counts, _ = dla.bootstrap_draws(n, row.n_circuits, DRAWS, SEED)
    return np.sum(seed_counts * macro, axis=1) / n


def macro_draws(row: dla.Row, errors: np.ndarray) -> np.ndarray:
    """(draws x seeds) macro MAE of item errors under the rule's circuit counts."""
    _, circuit_counts = dla.bootstrap_draws(errors.shape[0], row.n_circuits, DRAWS, SEED)
    return dla.macro_under_weights(errors, row.item_circuit, row.members, row.all_cells,
                                   circuit_counts)


def interval(draws: np.ndarray, point: float) -> dict:
    lower, upper = np.percentile(draws, dla.PERCENTILES)
    return {"point": float(point), "lower": float(lower), "upper": float(upper),
            "excludes_zero_above": bool(lower > 0.0), "excludes_zero_below": bool(upper < 0.0)}


def difference_entry(row: dla.Row, err_a: np.ndarray, err_b: np.ndarray) -> tuple[dict, np.ndarray]:
    """Mean over seeds of macro(a) - macro(b), with the two-stage interval."""
    n = err_a.shape[0]
    pa = np.asarray([dla.point_macro(err_a[s], row.members, row.all_cells) for s in range(n)])
    pb = np.asarray([dla.point_macro(err_b[s], row.members, row.all_cells) for s in range(n)])
    macro_a = macro_draws(row, err_a)
    draws = seed_mean_draws(row, macro_a - macro_draws(row, err_b), n)
    entry = interval(draws, float((pa - pb).mean()))
    entry["first_point"] = float(pa.mean())
    entry["second_point"] = float(pb.mean())
    first_mean = seed_mean_draws(row, macro_a, n)
    if entry["first_point"] > 0.0 and np.all(first_mean > 0.0):
        entry["relative"] = interval(draws / first_mean, entry["point"] / entry["first_point"])
    return entry, draws


# --------------------------------------------------------------------------
# Validation rows and the stack
# --------------------------------------------------------------------------


class Validation:
    """The validation rows of one family, split by the test row's cells."""

    def __init__(self, data: dict, family: str, row: dla.Row) -> None:
        rows = data["validation"]
        order = (data.get("prediction_rows") or {}).get("validation")
        if order is not None and [str(r["item_id"]) for r in order] != \
                [str(r["item_id"]) for r in rows]:
            raise SystemExit(f"{row.label}: validation rows differ from the prediction order")
        self.n_total = len(rows)
        self.family = family
        self.index = np.asarray([i for i, r in enumerate(rows) if str(r["family"]) == family])
        cells = {cell: c for c, cell in enumerate(row.cells)}
        self.cell_index = [[] for _ in row.cells]
        for i in self.index:
            cell = tuple(str(rows[i][f]) for f in CELL_KEYS)
            if cell not in cells:
                raise SystemExit(f"validation cell {cell} has no test cell ({row.label})")
            self.cell_index[cells[cell]].append(int(i))
        self.cell_index = [np.asarray(ix, dtype=int) for ix in self.cell_index]
        self.y = np.asarray([float(r["ideal_expectation"]) for r in rows])
        self.r = np.asarray([float(r["noisy_expectation"]) for r in rows])

    def family_mae(self, predictions: np.ndarray) -> float:
        if predictions.shape != (self.n_total,):
            raise SystemExit(f"validation predictions have shape {predictions.shape}, "
                             f"the cache has {self.n_total} validation rows")
        values = np.abs(predictions[self.index] - self.y[self.index]).tolist()
        return math.fsum(values) / len(values)


def stack_errors(row: dla.Row, val: Validation, val_pred: np.ndarray,
                 test_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Item errors of the recalibrated prediction and of the stack with r.

    Per cell: y ~ a + b p and y ~ a + b p + c r by least squares on validation
    rows, applied to the cell's test rows (the arithmetic of
    tools/measurement_floor.compute_stacking).
    """
    recal = np.empty(row.n_items)
    stack = np.empty(row.n_items)
    test_p = test_pred[row.index]
    for c, members in enumerate(row.members):
        v = val.cell_index[c]
        p_v, y_v, r_v = val_pred[v], val.y[v], val.r[v]
        p_t, r_t = test_p[members], row.noisy[members]
        beta_rec, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(p_v), p_v]), y_v, rcond=None)
        beta_st, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(p_v), p_v, r_v]), y_v,
                                      rcond=None)
        recal[members] = np.column_stack([np.ones_like(p_t), p_t]) @ beta_rec
        stack[members] = np.column_stack([np.ones_like(p_t), p_t, r_t]) @ beta_st
    return np.abs(recal - row.ideal), np.abs(stack - row.ideal)


def stack_entry(row: dla.Row, val: Validation, store: dla.FitStore, records: list[dict],
                name: str) -> dict:
    recal, stack = [], []
    for record in records:
        val_pred = store.array(record, "validation" if name == "test" else
                               "validation__" + name[len("test__"):])
        test_pred = store.array(record, name)
        if val_pred is None or test_pred is None:
            raise SystemExit(f"{record['stem']}: {name} or its validation array is missing")
        e_rec, e_st = stack_errors(row, val, val_pred, test_pred)
        recal.append(e_rec)
        stack.append(e_st)
    entry, _ = difference_entry(row, np.asarray(recal), np.asarray(stack))
    entry["recal_point"] = entry.pop("first_point")
    entry["stack_point"] = entry.pop("second_point")
    return entry


# --------------------------------------------------------------------------
# One fit-set cell
# --------------------------------------------------------------------------


def seeded(store: dla.FitStore, key: str, rung: str, arm: str) -> list[dict]:
    group = store.seeded(key, rung, arm)
    missing = [k for k in LEARNER_SEEDS if k not in group]
    if missing:
        raise SystemExit(f"{key} {rung} {arm}: learner seeds {missing} missing")
    return [group[k] for k in LEARNER_SEEDS]


def item_errors(row: dla.Row, store: dla.FitStore, records: list[dict], name: str) -> np.ndarray:
    out = np.empty((len(records), row.n_items))
    for s, record in enumerate(records):
        values = store.array(record, name)
        if values is None:
            raise SystemExit(f"{record['stem']}: no {name} array")
        if values.shape != (row.n_test_total,):
            raise SystemExit(f"{record['stem']}: {name} has shape {values.shape}")
        out[s] = np.abs(values[row.index] - row.ideal)
    return out


def check_recorded_test(row: dla.Row, records: list[dict], errors: np.ndarray, name: str,
                        family: str) -> float:
    """Each record's family test MAE equals the recomputed one (alignment check)."""
    worst = 0.0
    for s, record in enumerate(records):
        meta = record["meta"]
        if name == "test":
            recorded = (meta.get("test_family_mae") or {}).get(family)
        else:
            recorded = ((meta.get("candidate_test_family_mae") or {})
                        .get(name[len("test__"):]) or {}).get(family)
        if recorded is None:
            raise SystemExit(f"{record['stem']}: no recorded test MAE for {name}")
        diff = abs(float(recorded) - dla.point_macro(errors[s], row.members, row.all_cells))
        if diff > TOL:
            raise SystemExit(f"{record['stem']}: {name} test MAE differs by {diff}")
        worst = max(worst, diff)
    return worst


def check_recorded_validation(val: Validation, store: dla.FitStore, records: list[dict],
                              candidate: str | None) -> float:
    worst = 0.0
    for record in records:
        meta = record["meta"]
        if candidate is None:
            recorded = (meta.get("validation_family_mae") or {}).get(val.family)
            array = store.array(record, "validation")
        else:
            recorded = ((meta.get("candidate_validation_family_mae") or {})
                        .get(candidate) or {}).get(val.family)
            array = store.array(record, "validation__" + candidate)
        if recorded is None or array is None:
            raise SystemExit(f"{record['stem']}: validation record or array missing "
                             f"({candidate or 'selected'})")
        diff = abs(float(recorded) - val.family_mae(array))
        if diff > TOL:
            raise SystemExit(f"{record['stem']}: validation MAE of {candidate or 'selected'} "
                             f"differs by {diff}")
        worst = max(worst, diff)
    return worst


def select_reference(store: dla.FitStore, records: list[dict], family: str) -> dict:
    """Ref*: the candidate with the lowest seed-mean validation error for the family."""
    means, excluded = {}, {}
    for candidate in CANDIDATES:
        values = []
        for record in records:
            value = ((record["meta"].get("candidate_validation_family_mae") or {})
                     .get(candidate) or {}).get(family)
            if value is None or not math.isfinite(float(value)) or \
                    store.array(record, "test__" + candidate) is None:
                values = None
                break
            values.append(float(value))
        if values is None:
            excluded[candidate] = "missing in at least one learner seed"
            continue
        means[candidate] = math.fsum(values) / len(values)
    if not means:
        raise SystemExit("no reference candidate available")
    order = sorted(means, key=lambda c: (means[c], CANDIDATES.index(c)))
    return {"selected": order[0], "validation_means": means, "excluded": excluded}


def analyze_cell(fit_set: str, oracle_row: dict, row: dla.Row, val: Validation,
                 store: dla.FitStore, ref_store: dla.FitStore, analysis_rung: dict,
                 delta: float, reproduce_only: bool, stacking_ref: dict | None) -> dict:
    key = f"shipped-s{oracle_row['dataset_seed']}-n640"
    rung = oracle_row["rung"]
    family = oracle_row["family"]
    c_rec = seeded(store, key, rung, "C")
    f_rec = seeded(store, key, rung, "F")
    err_c = item_errors(row, store, c_rec, "test")
    err_f = item_errors(row, store, f_rec, "test")
    checks = {"test_mae_max_diff": max(check_recorded_test(row, c_rec, err_c, "test", family),
                                       check_recorded_test(row, f_rec, err_f, "test", family))}
    d_entry, _ = difference_entry(row, err_c, err_f)

    # Plain D must equal analysis A and the oracle's pipeline label.
    released = analysis_rung["D"]
    diffs = [abs(d_entry["point"] - released["point"]),
             abs(d_entry["lower"] - released["interval"]["lower"]),
             abs(d_entry["upper"] - released["interval"]["upper"])]
    if max(diffs) > TOL:
        raise SystemExit(f"{fit_set} {key} {family} {rung}: D differs from analysis A by "
                         f"{max(diffs)}")
    checks["D_vs_analysis_a"] = max(diffs)
    label = ("F beats C" if d_entry["excludes_zero_above"] else
             "C beats F" if d_entry["excludes_zero_below"] else "not distinguished")
    if label != oracle_row["pipeline_label"]:
        raise SystemExit(f"{fit_set} {key} {family} {rung}: label {label} differs from the "
                         f"oracle row's {oracle_row['pipeline_label']}")

    out = {"fit_set": fit_set, "shot_level": oracle_row["shot_level"],
           "dataset_seed": oracle_row["dataset_seed"], "family": family, "rung": rung,
           "group": group_of(fit_set, oracle_row["shot_level"]),
           "truth": truth_of(oracle_row), "G_star": oracle_row["G_star"],
           "G_star_interval": oracle_row["G_star_interval"], "D": d_entry,
           "pipeline_label": label}

    if stacking_ref is not None or not reproduce_only:
        checks["C_validation_max_diff"] = check_recorded_validation(val, store, c_rec, None)
        out["stack_C"] = stack_entry(row, val, store, c_rec, "test")
        if stacking_ref is not None:
            sdiffs = [abs(out["stack_C"]["point"] - stacking_ref["increment_mean"]),
                      abs(out["stack_C"]["lower"] - stacking_ref["increment_interval"]["lower"]),
                      abs(out["stack_C"]["upper"] - stacking_ref["increment_interval"]["upper"])]
            if max(sdiffs) > TOL:
                raise SystemExit(f"{fit_set} {key} {family} {rung}: stacking increment "
                                 f"differs from posthoc-stacking-all by {max(sdiffs)}")
            checks["stack_C_vs_stacking_all"] = max(sdiffs)
    if reproduce_only:
        out["checks"] = checks
        return out

    r_rec = seeded(ref_store, key, rung, "C")
    for record in r_rec:
        recorded = record["meta"].get("dataset_hash")
        if recorded != row_dataset_hash(row):
            raise SystemExit(f"{record['stem']}: reference fit dataset hash differs from the cache")
    choice = select_reference(ref_store, r_rec, family)
    cstar = choice["selected"]
    checks["reference_validation_max_diff"] = check_recorded_validation(val, ref_store, r_rec,
                                                                        cstar)
    err_ref = item_errors(row, ref_store, r_rec, "test__" + cstar)
    checks["reference_test_mae_max_diff"] = check_recorded_test(row, r_rec, err_ref,
                                                                "test__" + cstar, family)
    out["reference"] = choice
    out["D_ref"], _ = difference_entry(row, err_ref, err_f)
    out["stack_ref"] = stack_entry(row, val, ref_store, r_rec, "test__" + cstar)
    out["delta"] = delta
    verdicts = {
        "rule_D": d_entry["excludes_zero_above"],
        "rule_S": out["stack_C"]["excludes_zero_above"],
        "rule_R": out["D_ref"]["excludes_zero_above"],
        "rule_RS": out["D_ref"]["excludes_zero_above"] and out["stack_ref"]["excludes_zero_above"],
        "rule_R_delta": out["D_ref"]["lower"] > delta,
    }
    out["verdicts"] = {k: ("information" if v else "no detected information")
                       for k, v in verdicts.items()}
    out["reference_beats_F"] = out["D_ref"]["excludes_zero_below"]
    out["checks"] = checks
    return out


_ROW_HASH: dict[int, str] = {}


def row_dataset_hash(row: dla.Row) -> str:
    return _ROW_HASH[id(row)]


# --------------------------------------------------------------------------
# Tallies
# --------------------------------------------------------------------------


def tally(cells: list[dict], rule: str) -> dict:
    counts = {"cells": 0, "information": 0, "detected": 0, "no_information": 0,
              "false_positives": 0, "undetermined": 0, "undetermined_flagged": 0,
              "reference_beats_F_on_information": 0}
    for c in cells:
        flag = c["verdicts"][rule] == "information"
        counts["cells"] += 1
        if c["truth"] == "information":
            counts["information"] += 1
            counts["detected"] += int(flag)
            counts["reference_beats_F_on_information"] += int(c.get("reference_beats_F", False))
        elif c["truth"] == "no information":
            counts["no_information"] += 1
            counts["false_positives"] += int(flag)
        else:
            counts["undetermined"] += 1
            counts["undetermined_flagged"] += int(flag)
    counts["sensitivity"] = (counts["detected"] / counts["information"]
                             if counts["information"] else None)
    counts["specificity"] = (1.0 - counts["false_positives"] / counts["no_information"]
                             if counts["no_information"] else None)
    return counts


def changes(cells: list[dict], rule: str) -> dict:
    """Cells whose verdict under ``rule`` differs from rule_D, by oracle reading."""
    out = defaultdict(lambda: {"gained": 0, "lost": 0})
    for c in cells:
        a = c["verdicts"]["rule_D"] == "information"
        b = c["verdicts"][rule] == "information"
        if a and not b:
            out[c["truth"]]["lost"] += 1
        elif b and not a:
            out[c["truth"]]["gained"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def recommend(tallies: dict) -> dict:
    """The recommended rule of Part A's Reading from one scope's total tallies.

    A rule is eligible when its specificity strictly exceeds the primary rule's
    and its sensitivity is at least the primary's minus 0.05. Among eligible
    rules: highest specificity, then highest sensitivity, then ALTERNATIVES
    order. With none eligible, the primary rule stays.
    """
    base = tallies[PRIMARY_RULE]
    eligible = []
    for rule in ALTERNATIVES:
        t = tallies[rule]
        if None in (t["specificity"], t["sensitivity"], base["specificity"],
                    base["sensitivity"]):
            continue
        if (t["specificity"] > base["specificity"]
                and t["sensitivity"] >= base["sensitivity"] - SENSITIVITY_SLACK - 1e-12):
            eligible.append(rule)
    if not eligible:
        return {"recommended": PRIMARY_RULE, "eligible": []}
    best = min(eligible, key=lambda r: (-tallies[r]["specificity"], -tallies[r]["sensitivity"],
                                        ALTERNATIVES.index(r)))
    return {"recommended": best, "eligible": eligible}


def summarize(cells: list[dict]) -> dict:
    scopes = {
        "all": lambda c: True,
        "fit_identity": lambda c: not c["fit_set"].endswith("_derived"),
    }
    out = {}
    for scope, keep in scopes.items():
        kept = [c for c in cells if keep(c)]
        groups = {"total": kept}
        for g in ("first_2048", "new_2048", "other_finite", "exact"):
            groups[g] = [c for c in kept if c["group"] == g]
        out[scope] = {
            g: {rule: tally(members, rule) for rule in RULES}
            for g, members in groups.items()
        }
        out[scope]["verdict_changes_vs_rule_D"] = {
            rule: changes(kept, rule) for rule in RULES if rule != "rule_D"}
    return out


def markdown(result: dict) -> str:
    lines = ["# Oracle-Free Reading Rules Scored Against the Bayes Oracle", "",
             f"Rule: `{RULE_FILE}`, Part A. Primary rule: `{PRIMARY_RULE}`. "
             f"Recommended rule: `{result['recommendation']['recommended']}` "
             f"(eligible: {result['recommendation']['eligible'] or 'none'}).", ""]
    for scope in ("all", "fit_identity"):
        lines += [f"## Scope: {scope}", "",
                  "| Group | Rule | Cells | Information | Detected | Sensitivity | "
                  "No information | False positives | Specificity | Undetermined flagged |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for group, by_rule in result["summary"][scope].items():
            if group == "verdict_changes_vs_rule_D":
                continue
            for rule, t in by_rule.items():
                sens = "" if t["sensitivity"] is None else f"{t['sensitivity']:.3f}"
                spec = "" if t["specificity"] is None else f"{t['specificity']:.3f}"
                lines.append(f"| {group} | {rule} | {t['cells']} | {t['information']} | "
                             f"{t['detected']} | {sens} | {t['no_information']} | "
                             f"{t['false_positives']} | {spec} | "
                             f"{t['undetermined_flagged']} of {t['undetermined']} |")
        lines += ["", "Verdict changes against rule_D (gained, lost) by oracle reading:", ""]
        for rule, by_truth in result["summary"][scope]["verdict_changes_vs_rule_D"].items():
            parts = [f"{t}: +{v['gained']} / -{v['lost']}" for t, v in by_truth.items()]
            lines.append(f"- {rule}: " + ("; ".join(parts) if parts else "none"))
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    rule_path = _REPO / RULE_FILE
    if not args.frozen_rule.is_file() or args.frozen_rule.resolve() != rule_path.resolve():
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")
    roots = {"ASSETS": args.assets, "CROSSED": args.crossed, "FRESH": args.fresh}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mapping = {e["fit_set"]: e for e in json.loads(MAPPING.read_text(encoding="utf-8"))["fit_sets"]}
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    pooled = json.loads(POOLED.read_text(encoding="utf-8"))
    deltas = pooled["margins"]["delta"]
    stacking_all = json.loads(STACKING_ALL.read_text(encoding="utf-8"))["sets"]
    if set(registry["fit_sets"]) != set(mapping):
        raise SystemExit("the registry and the oracle mapping list different fit sets")

    rows_by_set = defaultdict(list)
    for r in oracle["equation_2"]["rows"]:
        rows_by_set[r["fit_set"]].append(r)
    if sum(len(v) for v in rows_by_set.values()) != 696:
        raise SystemExit("the oracle's Equation 2 table does not hold 696 rows")

    only = set(args.fit_sets) if args.fit_sets else None
    ref_stores: dict[str, dla.FitStore] = {}
    cells, inputs = [], {}
    for fit_set in mapping:
        if only and fit_set not in only:
            continue
        entry = registry["fit_sets"][fit_set]
        analysis_path = _REPO / mapping[fit_set]["path"]
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        a_rows = analysis["parts"]["A"]["rows"]
        fits_dir = resolve(entry["fits"], roots)
        cache_dir = resolve(entry["cache"], roots)
        print(f"[{fit_set}] {fits_dir}", flush=True)
        store = dla.FitStore([fits_dir], use_orig=False)
        cond = registry["conditions"][entry["condition"]]
        ref_dir = resolve(cond["reference_fits"], roots)
        if not args.reproduce_only and str(ref_dir) not in ref_stores:
            ref_stores[str(ref_dir)] = dla.FitStore([ref_dir], use_orig=False)
        ref_store = ref_stores.get(str(ref_dir))
        stack_name = entry.get("stacking_all")
        stack_cells = stacking_all[stack_name]["cells"] if stack_name else None
        inputs[fit_set] = {"fits": str(fits_dir), "fits_digest": store.inputs_digest,
                           "cache": str(cache_dir), "analysis": mapping[fit_set]["path"],
                           "analysis_sha256": sha256_file(analysis_path),
                           "reference_set": cond["reference_set"],
                           "reference_fits": str(ref_dir),
                           "reference_digest": ref_store.inputs_digest if ref_store else None,
                           "caches": {}}
        by_seed = defaultdict(list)
        for r in rows_by_set[fit_set]:
            by_seed[int(r["dataset_seed"])].append(r)
        for dataset_seed, orows in sorted(by_seed.items()):
            key = f"shipped-s{dataset_seed}-n640"
            cache_path = cache_dir / f"{key}.pkl"
            recorded = analysis["inputs"]["caches_used"][key]["sha256"]
            actual = sha256_file(cache_path)
            if actual != recorded:
                raise SystemExit(f"{cache_path}: SHA-256 differs from analysis A's cache")
            data = dla.load_cache(cache_path)
            inputs[fit_set]["caches"][key] = actual
            for family in sorted({r["family"] for r in orows}):
                row = dla.Row(data, family, f"{key}/{family}")
                _ROW_HASH[id(row)] = data["dataset_hash"]
                val = Validation(data, family, row)
                for orow in sorted((r for r in orows if r["family"] == family),
                                   key=lambda r: r["rung"]):
                    a_rung = a_rows[f"{key}/{family}"]["rungs"][orow["rung"]]
                    sref = None
                    if stack_cells is not None:
                        sref = stack_cells.get(f"{key}/{family}/{orow['rung']}")
                        if sref is None:
                            raise SystemExit(f"{stack_name}: no stacking cell for "
                                             f"{key}/{family}/{orow['rung']}")
                    cells.append(analyze_cell(fit_set, orow, row, val, store, ref_store,
                                              a_rung, float(deltas[family]),
                                              args.reproduce_only, sref))
    result = {
        "schema": "reading-rules-v1",
        "frozen_rule": RULE_FILE,
        "rule_file_sha256": sha256_file(rule_path),
        "script_sha256": sha256_file(Path(__file__)),
        "registry_sha256": sha256_file(REGISTRY),
        "oracle_sha256": sha256_file(ORACLE),
        "pooled_sha256": sha256_file(POOLED),
        "mode": "reproduce-only" if args.reproduce_only else "full",
        "bootstrap": {"draws": DRAWS, "seed": SEED, "percentiles": list(dla.PERCENTILES)},
        "candidates": list(CANDIDATES),
        "primary_rule": PRIMARY_RULE,
        "deltas": deltas,
        "inputs": inputs,
        "n_cells": len(cells),
        "max_check_diffs": {
            name: max((c["checks"].get(name, 0.0) for c in cells), default=0.0)
            for name in sorted({k for c in cells for k in c["checks"]})},
        "cells": cells,
    }
    if not args.reproduce_only:
        result["summary"] = summarize(cells)
        result["recommendation"] = recommend(result["summary"]["fit_identity"]["total"])
        result["recommendation"]["scope"] = "fit_identity (522 cells)"
    result["seconds"] = time.perf_counter() - started
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--crossed", type=Path, required=True)
    p.add_argument("--fresh", type=Path, required=True)
    p.add_argument("--frozen-rule", type=Path, required=True)
    p.add_argument("--fit-sets", nargs="+", default=None, help="restrict to these fit sets")
    p.add_argument("--reproduce-only", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    out = args.out or (DEFAULT_OUT if not args.reproduce_only else
                       _REPO / "build/reading-rules-reproduce.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({result['n_cells']} cells, {result['seconds']:.0f} s)")
    print("max check differences:", result["max_check_diffs"])
    if not args.reproduce_only:
        md = args.md or DEFAULT_MD
        md.write_text(markdown(result), encoding="utf-8")
        print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
