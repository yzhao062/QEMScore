"""Report strong-learner estimands (rule 2026-10-03-strong-learners.md).

Computes:
1. Fixed-learner D with two-stage interval and descriptive label for each
   candidate present in both arms C and F (reusing Script A's RowEstimator and draws).
2. Per-family selection: the one-standard-error rule applied to each family's
   validation rows alone, with resulting D per row.
3. Selection counts per arm, row, and rung.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from qemscore.baselines.liao import LiaoValidationScore, select_one_standard_error, _validation_score
from tools.descriptor_ladder_analysis import (
    FitStore,
    Row,
    RowEstimator,
    classify,
    PARTS,
    bootstrap_draws,
    point_macro,
    macro_under_weights,
    interval_entry,
    arms_at,
    load_cache,
    sha256_file,
    DEFAULT_DRAWS,
    RULE_SEED,
)

SIMPLICITY_RANKS = {
    "random_forest": 0,
    "mlp": 1,
    "hgbr": 2,
    "poly5_ridge": 3,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _record_candidates(record: dict) -> list[str]:
    cands = record["meta"].get("candidates")
    if cands:
        return list(cands)
    with np.load(record["npz"]) as npz:
        return [k[len("test__"):] for k in npz.files if k.startswith("test__")]


def per_family_select(
    record: dict,
    store: FitStore,
    data: dict,
    family: str | None,
) -> str | None:
    """Run one-standard-error rule on one family's validation rows alone."""
    validation = data.get("validation", [])
    if family is None:
        val_indices = list(range(len(validation)))
    else:
        val_indices = [i for i, it in enumerate(validation) if str(it.get("family")) == family]

    if not val_indices:
        return None

    val_items = [validation[i] for i in val_indices]
    candidates = _record_candidates(record)
    # The fit's own circuit-evaluation count, equal across its candidates.
    recorded = record["meta"].get("validation_scores") or []
    costs = {int(row["circuit_evaluations"]) for row in recorded}
    if len(costs) != 1:
        raise ValueError(
            f"{record['meta'].get('key')}: expected one recorded circuit_evaluations "
            f"across candidates, found {sorted(costs)}")
    circuit_evaluations = costs.pop()

    scores = []
    for cand in candidates:
        arr_name = f"validation__{cand}"
        preds = store.array(record, arr_name)
        if preds is None or len(preds) < len(validation):
            continue
        val_preds = preds[val_indices]
        rank = SIMPLICITY_RANKS.get(cand, 99)
        score = _validation_score(
            cand,
            val_items,
            val_preds,
            circuit_evaluations=circuit_evaluations,
            simplicity_rank=rank,
        )
        scores.append(score)

    if not scores:
        return record["meta"].get("selected_model")

    selected, _, _ = select_one_standard_error(scores)
    return selected


def analyze_strong_report(
    fit_dirs: list[Path],
    cache_dirs: list[Path],
    draws: int = DEFAULT_DRAWS,
    seed: int = RULE_SEED,
) -> dict:
    store = FitStore(fit_dirs, False)
    caches: dict[str, Path] = {}
    for d in cache_dirs:
        for p in Path(d).glob("*.pkl"):
            caches.setdefault(p.stem, p)

    report: dict = {
        "schema": "strong-learner-report-v1",
        "frozen_rule": "docs/frozen-rules/2026-10-03-strong-learners.md",
        "draws": draws,
        "seed": seed,
        "inputs": {
            "fits": [str(Path(d).resolve()) for d in fit_dirs],
            "cache": [str(Path(d).resolve()) for d in cache_dirs],
            "n_fit_files": store.n_files,
            "fits_sha256_digest": store.inputs_digest,
        },
        "parts": {},
    }

    fit_keys = store.keys()
    for part, spec in PARTS.items():
        keys = sorted(k for k in fit_keys if spec["key_pattern"].match(k))
        if not keys:
            continue

        part_out: dict = {"title": spec["title"], "rows": {}}
        keys_by_seed: dict[int, str] = {}
        for key in keys:
            m = spec["key_pattern"].match(key)
            if m:
                keys_by_seed[int(m.group(1))] = key

        for dataset_seed, key in sorted(keys_by_seed.items()):
            if key not in caches:
                continue
            data = load_cache(caches[key])
            families = spec["families"] or (None,)

            for family in families:
                row_label = f"{key}/{family}" if family else key
                try:
                    row = Row(data, family, row_label)
                except ValueError:
                    continue

                est = RowEstimator(row, store, draws, seed, 20)
                row_out: dict = {"rungs": {}}

                present_rungs = store.rungs_for(key)
                for rung in spec["rungs"]:
                    allowed = spec["rung_families"].get(rung)
                    if allowed is not None and family not in allowed:
                        continue
                    if rung not in present_rungs:
                        continue

                    arms = arms_at(store, spec, key, rung)
                    rung_out: dict = {"fits_present": {arm: sorted(arms[arm]) for arm in ("F", "C", "P", "M")}}

                    # 1. Standard selection counts
                    sel_counts = {
                        arm: dict(sorted(Counter(str(r["meta"].get("selected_model"))
                                                 for r in arms[arm].values()).items()))
                        for arm in ("F", "C", "P") if arms[arm]
                    }
                    rung_out["selection_counts"] = sel_counts

                    # 2. Fixed-learner D for each candidate present in both arms
                    c_cands = set()
                    for r in arms["C"].values():
                        c_cands.update(r["meta"].get("candidates") or [])
                    f_cands = set()
                    for r in arms["F"].values():
                        f_cands.update(r["meta"].get("candidates") or [])
                    common_cands = sorted(c_cands & f_cands)
                    if not common_cands:
                        c_names = set(k for r in arms["C"].values() for k in _record_candidates(r))
                        f_names = set(k for r in arms["F"].values() for k in _record_candidates(r))
                        common_cands = sorted(c_names & f_names)

                    fixed_d: dict = {}
                    for cand in common_cands:
                        cand_entry, cand_parts = est.difference(arms["C"], arms["F"], name=f"test__{cand}")
                        if cand_parts is not None:
                            mean_c = cand_parts["first_point"]
                            mean_f = cand_parts["second_point"]
                            cls = classify(cand_entry, mean_c)
                            fixed_d[cand] = {
                                "D": cand_entry,
                                "mean_C": mean_c,
                                "mean_F": mean_f,
                                "classification_descriptive": cls.get("label"),
                            }
                        else:
                            fixed_d[cand] = {"status": cand_entry.get("status"), "reason": cand_entry.get("reason")}
                    rung_out["fixed_learner_D"] = fixed_d

                    # 3. Per-family selection
                    common_seeds = sorted(set(arms["C"]) & set(arms["F"]))
                    per_family_out: dict = {}
                    if common_seeds:
                        c_sel = {s: per_family_select(arms["C"][s], store, data, family) for s in common_seeds}
                        f_sel = {s: per_family_select(arms["F"][s], store, data, family) for s in common_seeds}
                        p_sel = {s: per_family_select(arms["P"][s], store, data, family)
                                 for s in sorted(arms["P"]) if s in arms["P"]}

                        per_fam_counts = {
                            "C": dict(sorted(Counter(c_sel.values()).items())),
                            "F": dict(sorted(Counter(f_sel.values()).items())),
                            "P": dict(sorted(Counter(p_sel.values()).items())),
                        }
                        per_family_out["selection_counts"] = per_fam_counts
                        per_family_out["selected_by_seed"] = {
                            "C": c_sel,
                            "F": f_sel,
                        }

                        # Compute resulting D under per-family selection
                        n_s = len(common_seeds)
                        err_c_rows = []
                        err_f_rows = []
                        valid = True
                        for s in common_seeds:
                            m_c = c_sel[s]
                            m_f = f_sel[s]
                            c_preds = store.array(arms["C"][s], f"test__{m_c}") if m_c else None
                            f_preds = store.array(arms["F"][s], f"test__{m_f}") if m_f else None
                            if c_preds is None or f_preds is None:
                                valid = False
                                break
                            err_c_rows.append(np.abs(c_preds[row.index] - row.ideal))
                            err_f_rows.append(np.abs(f_preds[row.index] - row.ideal))

                        if valid and err_c_rows and err_f_rows:
                            errors_c = np.vstack(err_c_rows)
                            errors_f = np.vstack(err_f_rows)
                            _, circuit_counts = bootstrap_draws(n_s, row.n_circuits, draws, seed)
                            macro_c = macro_under_weights(
                                errors_c, row.item_circuit, row.members, row.all_cells, circuit_counts)
                            macro_f = macro_under_weights(
                                errors_f, row.item_circuit, row.members, row.all_cells, circuit_counts)
                            seed_counts = est.seed_counts(n_s)
                            draws_d = np.sum(seed_counts * (macro_c - macro_f), axis=1) / n_s
                            draws_c = np.sum(seed_counts * macro_c, axis=1) / n_s

                            point_c_arr = np.asarray(
                                [point_macro(errors_c[i], row.members, row.all_cells) for i in range(n_s)])
                            point_f_arr = np.asarray(
                                [point_macro(errors_f[i], row.members, row.all_cells) for i in range(n_s)])
                            point_d = float((point_c_arr - point_f_arr).mean())
                            point_c = float(point_c_arr.mean())
                            point_f = float(point_f_arr.mean())

                            d_entry = interval_entry(draws_d, point_d, n_s, common_seeds)
                            cls = classify(d_entry, point_c)
                            per_family_out["D"] = d_entry
                            per_family_out["mean_C"] = point_c
                            per_family_out["mean_F"] = point_f
                            per_family_out["classification_descriptive"] = cls.get("label")
                        else:
                            per_family_out["D"] = {"status": "not_available",
                                                   "reason": "per-family candidate predictions missing"}
                    else:
                        per_family_out["status"] = "not_estimable"
                        per_family_out["reason"] = "no common learner seeds for C and F"

                    rung_out["per_family_selection"] = per_family_out
                    row_out["rungs"][rung] = rung_out

                part_out["rows"][row_label] = row_out
            report["parts"][part] = part_out

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, action="append", required=True,
                        help="Fits directory (may be repeated)")
    parser.add_argument("--cache", type=Path, action="append", required=True,
                        help="Cache directory (may be repeated)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for report JSON")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                        help=f"Number of bootstrap draws (default: {DEFAULT_DRAWS})")
    parser.add_argument("--seed", type=int, default=RULE_SEED,
                        help=f"Bootstrap random seed (default: {RULE_SEED})")
    args = parser.parse_args()

    report = analyze_strong_report(args.fits, args.cache, draws=args.draws, seed=args.seed)
    _write_json(args.out, report)
    print(f"Wrote strong learner report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
