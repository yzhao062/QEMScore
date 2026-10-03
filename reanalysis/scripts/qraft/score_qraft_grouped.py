"""Grouped validation for QRAFT panel under matched retraining.

Evaluates out-of-circuit generalization by grouping rows sharing the same
machine and descriptor values so that no partial-circuit group appears in
both training and test sets.

Evaluates three group definitions:
1. Primary: 7-descriptor tuple + Machine (3,800 groups in inputData.csv).
2. Circuit-level (Appendix C): 6 circuit descriptors + Machine (1,038 groups in
   inputData.csv, corresponding to the 735 groups in outputData.csv).
3. Structure-only (post hoc): 6 circuit descriptors alone, ignoring machine
   (864 groups in inputData.csv; 106 tuples recur across 2 to 5 machines,
   preventing recurring circuit structures from appearing on both sides of a split).
"""
import collections
import csv
import json
import os
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

TARGET = "StateRealProb"
MACHINE = "ComputerID"
CIRCUIT_DESCRIPTORS_6 = (
    "CircuitWidth", "CircuitDepth", "CircuitNumU1Gates",
    "CircuitNumU2Gates", "CircuitNumU3Gates", "CircuitNumCXGates"
)
DESCRIPTORS_7 = CIRCUIT_DESCRIPTORS_6 + ("StateHammingWeight",)
FORWARD = ("StateUpProb25", "StateUpProb50", "StateUpProb75")
REVERSE = (
    "TotalUpDnErr25", "TotalUpDnErr50", "TotalUpDnErr75",
    "StateUpDnErr25", "StateUpDnErr50", "StateUpDnErr75"
)
ARMS = {
    "Q3": DESCRIPTORS_7,
    "Q2": DESCRIPTORS_7 + FORWARD,
    "Q1": DESCRIPTORS_7 + FORWARD + REVERSE + (MACHINE,),
}
INPUT_COLUMNS = (MACHINE,) + DESCRIPTORS_7 + FORWARD + REVERSE + (TARGET,)
EXPECTED_INPUT_ROWS = 10155
EXPECTED_OUTPUT_ROWS = 1524
SEEDS = tuple(range(10))
TEST_FRACTION = 0.15
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261002


def load(path, required):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing declared columns {missing}")
    return rows


def numeric(rows, columns):
    return np.array([[float(r[c]) for c in columns] for r in rows], dtype=float)


def macro_mae(machines, err):
    """Rows equal within a machine, machines equal across, as the spec fixes."""
    out = {}
    for m in sorted(set(machines)):
        sel = machines == m
        out[m] = float(np.mean(err[sel]))
    return float(np.mean([out[m] for m in sorted(out)])), out


def run_grouped_matched(rows, group_ids, rng_seed=BOOTSTRAP_SEED):
    machines_all = np.array([r[MACHINE] for r in rows])
    y_all = numeric(rows, (TARGET,)).ravel()
    unique_groups_all = np.unique(group_ids)
    n_groups = len(unique_groups_all)

    seed_records = []
    per_seed_macro = {arm: {} for arm in ARMS}

    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
        train_idx, test_idx = next(gss.split(rows, groups=group_ids))

        # Assert disjoint groups
        train_groups = set(group_ids[train_idx])
        test_groups = set(group_ids[test_idx])
        assert not train_groups.intersection(test_groups), "Data leakage: groups overlap train and test!"

        t_mach = machines_all[test_idx]
        t_grp = group_ids[test_idx]
        t_y = y_all[test_idx]

        errs = {}
        for arm, cols in ARMS.items():
            x = numeric(rows, cols)
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(x[train_idx], y_all[train_idx])
            pred = model.predict(x[test_idx])
            err = np.abs(pred - t_y)
            errs[arm] = err
            macro, _ = macro_mae(t_mach, err)
            per_seed_macro[arm][str(seed)] = macro

        seed_records.append({
            "seed": seed,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_groups": len(test_groups),
            "machines": t_mach,
            "groups": t_grp,
            "errs": errs,
        })

    # Summary point estimates
    summary = {}
    for arm in ARMS:
        vals = [per_seed_macro[arm][str(s)] for s in SEEDS]
        summary[arm] = {
            "per_seed_macro_mae": per_seed_macro[arm],
            "ten_seed_mean": float(np.mean(vals)),
            "ten_seed_min": float(np.min(vals)),
            "ten_seed_max": float(np.max(vals)),
            "features": list(ARMS[arm]),
        }

    q3_mean = summary["Q3"]["ten_seed_mean"]
    q2_mean = summary["Q2"]["ten_seed_mean"]
    q1_mean = summary["Q1"]["ten_seed_mean"]
    ratio_q3_q2 = q3_mean / q2_mean
    diff_q3_q2 = q3_mean - q2_mean
    diff_q2_q1 = q2_mean - q1_mean

    # Vectorized group-level bootstrap across 10 seeds
    # Resamples unique test groups with replacement in each seed
    rng = np.random.default_rng(rng_seed)
    q3_boots = np.zeros(BOOTSTRAP_DRAWS, dtype=float)
    q2_boots = np.zeros(BOOTSTRAP_DRAWS, dtype=float)
    q1_boots = np.zeros(BOOTSTRAP_DRAWS, dtype=float)

    for rec in seed_records:
        t_groups = rec["groups"]
        t_machs = rec["machines"]
        u_groups = np.unique(t_groups)
        K = len(u_groups)

        unique_machs = sorted(set(t_machs))
        mach_to_idx = {m: i for i, m in enumerate(unique_machs)}
        n_m = len(unique_machs)

        # Row counts and error sums per (group, machine). A structure-only group can
        # span machines, so each row stays with its own machine and the bootstrap
        # keeps the point estimator's weighting: rows equal within a machine, then
        # machines equal.
        row_group = np.searchsorted(u_groups, t_groups)
        row_mach = np.array([mach_to_idx[m] for m in t_machs], dtype=int)
        counts = np.zeros((K, n_m), dtype=float)
        np.add.at(counts, (row_group, row_mach), 1.0)
        err_sums = {}
        for arm in ("Q3", "Q2", "Q1"):
            err_sums[arm] = np.zeros((K, n_m), dtype=float)
            np.add.at(err_sums[arm], (row_group, row_mach), rec["errs"][arm])
            # Identity resample (every group once) must reproduce the point estimator.
            identity = float(np.mean(err_sums[arm].sum(axis=0) / counts.sum(axis=0)))
            point, _ = macro_mae(t_machs, rec["errs"][arm])
            assert abs(identity - point) <= 1e-12, (arm, rec["seed"], identity, point)

        draws = rng.integers(0, K, size=(BOOTSTRAP_DRAWS, K))
        W = np.zeros((BOOTSTRAP_DRAWS, K), dtype=float)
        for b in range(BOOTSTRAP_DRAWS):
            W[b] = np.bincount(draws[b], minlength=K)

        q3_m_mae = np.zeros((BOOTSTRAP_DRAWS, n_m), dtype=float)
        q2_m_mae = np.zeros((BOOTSTRAP_DRAWS, n_m), dtype=float)
        q1_m_mae = np.zeros((BOOTSTRAP_DRAWS, n_m), dtype=float)

        for m in range(n_m):
            cnt_m = np.einsum('bk,k->b', W, counts[:, m])
            q3_m_mae[:, m] = np.einsum('bk,k->b', W, err_sums["Q3"][:, m]) / cnt_m
            q2_m_mae[:, m] = np.einsum('bk,k->b', W, err_sums["Q2"][:, m]) / cnt_m
            q1_m_mae[:, m] = np.einsum('bk,k->b', W, err_sums["Q1"][:, m]) / cnt_m

        q3_boots += q3_m_mae.mean(axis=1) / len(SEEDS)
        q2_boots += q2_m_mae.mean(axis=1) / len(SEEDS)
        q1_boots += q1_m_mae.mean(axis=1) / len(SEEDS)

    diff_boots = q3_boots - q2_boots
    ratio_boots = q3_boots / q2_boots

    ci_diff = [float(x) for x in np.percentile(diff_boots, [2.5, 97.5])]
    ci_ratio = [float(x) for x in np.percentile(ratio_boots, [2.5, 97.5])]
    ci_q3 = [float(x) for x in np.percentile(q3_boots, [2.5, 97.5])]
    ci_q2 = [float(x) for x in np.percentile(q2_boots, [2.5, 97.5])]
    ci_q1 = [float(x) for x in np.percentile(q1_boots, [2.5, 97.5])]

    return {
        "total_groups": n_groups,
        "arms": summary,
        "ratio_q3_over_q2": float(ratio_q3_q2),
        "difference_q3_minus_q2": float(diff_q3_q2),
        "difference_q2_minus_q1": float(diff_q2_q1),
        "bootstrap": {
            "n_draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "resample_unit": "partial-circuit groups",
            "difference_q3_minus_q2_95_ci": ci_diff,
            "ratio_q3_over_q2_95_ci": ci_ratio,
            "q3_macro_mae_95_ci": ci_q3,
            "q2_macro_mae_95_ci": ci_q2,
            "q1_macro_mae_95_ci": ci_q1,
        }
    }


def main():
    if len(sys.argv) == 3:
        rel_dir = sys.argv[1]
        out_dir = sys.argv[2]
    elif len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reanalysis_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        rel_dir = os.path.join(reanalysis_dir, "upstream", "QLEAR")
        out_dir = os.path.join(reanalysis_dir, "outputs", "qraft")
    else:
        raise SystemExit("Usage: score_qraft_grouped.py [<unpacked-release-dir> <output-dir>]")

    qraft_dir = rel_dir if os.path.isdir(os.path.join(rel_dir, "data_trained")) else os.path.join(rel_dir, "QRAFT")
    in_csv = os.path.join(qraft_dir, "data_processed", "inputData.csv")
    out_csv = os.path.join(qraft_dir, "data_trained", "outputData.csv")

    in_rows = load(in_csv, INPUT_COLUMNS)
    out_rows = load(out_csv, (MACHINE,) + DESCRIPTORS_7)

    if len(in_rows) != EXPECTED_INPUT_ROWS:
        raise SystemExit(f"inputData.csv: expected {EXPECTED_INPUT_ROWS} rows, found {len(in_rows)}")
    if len(out_rows) != EXPECTED_OUTPUT_ROWS:
        raise SystemExit(f"outputData.csv: expected {EXPECTED_OUTPUT_ROWS} rows, found {len(out_rows)}")

    # Define groups
    # 1. 7-descriptor groups: machine + 7 descriptors
    g7_in = [tuple(r[c] for c in (MACHINE,) + DESCRIPTORS_7) for r in in_rows]
    u7_in, g7_in_ids = np.unique(g7_in, axis=0, return_inverse=True)

    g7_out = [tuple(r[c] for c in (MACHINE,) + DESCRIPTORS_7) for r in out_rows]
    u7_out = np.unique(g7_out, axis=0)

    # 2. 6-descriptor groups: machine + 6 circuit descriptors (Appendix C definition)
    g6_in = [tuple(r[c] for c in (MACHINE,) + CIRCUIT_DESCRIPTORS_6) for r in in_rows]
    u6_in, g6_in_ids = np.unique(g6_in, axis=0, return_inverse=True)

    g6_out = [tuple(r[c] for c in (MACHINE,) + CIRCUIT_DESCRIPTORS_6) for r in out_rows]
    u6_out = np.unique(g6_out, axis=0)

    # 3. Structure-only groups: 6 circuit descriptors alone (machine ignored)
    g_struct_in = [tuple(r[c] for c in CIRCUIT_DESCRIPTORS_6) for r in in_rows]
    u_struct_in, g_struct_in_ids = np.unique(np.array(g_struct_in), axis=0, return_inverse=True)
    g_struct_in_ids = g_struct_in_ids.ravel()

    g_struct_out = [tuple(r[c] for c in CIRCUIT_DESCRIPTORS_6) for r in out_rows]
    u_struct_out = np.unique(g_struct_out, axis=0)

    # Recurrence analysis across machines
    machines_per_tuple = collections.defaultdict(set)
    for r, t in zip(in_rows, g_struct_in):
        machines_per_tuple[t].add(r[MACHINE])
    multi = {t: len(m) for t, m in machines_per_tuple.items() if len(m) > 1}
    hist = collections.Counter(multi.values())

    print("================================================================================")
    print("QRAFT GROUPED RETRAINING WITH PARTIAL-CIRCUIT VALIDATION")
    print("================================================================================")
    print(f"Processed input rows: {len(in_rows)}")
    print(f"7-descriptor partial-circuit groups in inputData.csv: {len(u7_in)}")
    print(f"7-descriptor partial-circuit groups in outputData.csv: {len(u7_out)}")
    print(f"6-circuit-descriptor groups in inputData.csv: {len(u6_in)}")
    print(f"6-circuit-descriptor groups in outputData.csv: {len(u6_out)} (paper Appendix C: 735)")
    print(f"6-descriptor structure-only groups in inputData.csv: {len(u_struct_in)}")
    print(f"6-descriptor structure-only groups in outputData.csv: {len(u_struct_out)}")
    print(f"Recurring 6-descriptor tuples across 2 to 5 machines: {len(multi)} of {len(u_struct_in)} ({dict(sorted(hist.items()))})")
    print()

    # Run matched retraining for primary 7-descriptor groups
    print("--- Running matched retraining: 7-descriptor groups (3,800 groups) ---")
    res_7 = run_grouped_matched(in_rows, g7_in_ids)

    # Run matched retraining for 6-circuit-descriptor groups
    print("--- Running matched retraining: 6-circuit-descriptor groups (1,038 groups) ---")
    res_6 = run_grouped_matched(in_rows, g6_in_ids)

    # Run matched retraining for structure-only groups (6 descriptors, machine ignored)
    print("--- Running matched retraining: structure-only groups (864 groups) ---")
    res_struct = run_grouped_matched(in_rows, g_struct_in_ids)

    report = {
        "task": "QRAFT matched retraining with grouped (circuit-level) validation",
        "description": "Grouped holdout split ensuring no partial-circuit group appears in both train and test",
        "seeds": list(SEEDS),
        "test_fraction": TEST_FRACTION,
        "score": "per-row mean absolute error in probability points",
        "weighting": "rows equal within machine, machines equal across",
        "group_counts": {
            "processed_inputs_7_descriptors": len(u7_in),
            "processed_inputs_6_circuit_descriptors": len(u6_in),
            "processed_inputs_6_descriptors_structure_only": len(u_struct_in),
            "released_predictions_7_descriptors": len(u7_out),
            "released_predictions_6_circuit_descriptors": len(u6_out),
            "released_predictions_6_descriptors_structure_only": len(u_struct_out),
            "tuples_on_more_than_one_machine": len(multi),
            "machines_per_recurring_tuple_histogram": {str(k): v for k, v in sorted(hist.items())},
        },
        "grouped_retraining_7_descriptors": res_7,
        "grouped_retraining_6_circuit_descriptors": res_6,
        "grouped_retraining_6_descriptors_structure_only": res_struct,
        "grouped_retraining_structure_only": res_struct,
        "structure_group_analysis": {
            "post_hoc": True,
            "machine_plus_6_descriptor_groups": len(u6_in),
            "six_descriptor_tuples": len(u_struct_in),
            "tuples_on_more_than_one_machine": len(multi),
            "machines_per_recurring_tuple_histogram": {str(k): v for k, v in sorted(hist.items())},
        },
    }

    # Print summary tables
    for name, res in [("Primary (7 Descriptors + Machine, 3,800 Groups)", res_7),
                      ("Circuit-Level (6 Descriptors + Machine, 1,038 Groups)", res_6),
                      ("Structure-Only (6 Descriptors Alone, 864 Groups)", res_struct)]:
        print()
        print(f"Results for {name}:")
        print(f"{'Arm':<5}{'Features':>10}{'Macro MAE Mean':>16}{'Min':>12}{'Max':>12}{'95% Bootstrap CI':>24}")
        for arm in ("Q3", "Q2", "Q1"):
            a = res["arms"][arm]
            ci = res["bootstrap"][f"{arm.lower()}_macro_mae_95_ci"]
            print(f"{arm:<5}{len(a['features']):>10}{a['ten_seed_mean']:>16.4f}{a['ten_seed_min']:>12.4f}{a['ten_seed_max']:>12.4f}{str(ci):>24}")
        print(f"Ratio Q3/Q2 (descriptors-only / forward-added): {res['ratio_q3_over_q2']:.4f}  "
              f"95% CI: [{res['bootstrap']['ratio_q3_over_q2_95_ci'][0]:.4f}, {res['bootstrap']['ratio_q3_over_q2_95_ci'][1]:.4f}]")
        print(f"Difference Q3 - Q2: {res['difference_q3_minus_q2']:.4f}  "
              f"95% CI: [{res['bootstrap']['difference_q3_minus_q2_95_ci'][0]:.4f}, {res['bootstrap']['difference_q3_minus_q2_95_ci'][1]:.4f}]")
        print(f"Difference Q2 - Q1: {res['difference_q2_minus_q1']:.4f}")
        q3 = res["arms"]["Q3"]["per_seed_macro_mae"]
        q2 = res["arms"]["Q2"]["per_seed_macro_mae"]
        q1 = res["arms"]["Q1"]["per_seed_macro_mae"]
        print(f"Ordering survived: {'YES (Q3 >> Q2 > Q1)' if res['arms']['Q3']['ten_seed_min'] > res['arms']['Q2']['ten_seed_max'] else 'NO'} "
              f"(Q3 > Q2 on {sum(q3[s] > q2[s] for s in q3)}/10 seeds; Q1 < Q2 on {sum(q1[s] < q2[s] for s in q1)}/10 seeds)")

    print()
    print("Recurrence summary across machines for 6-descriptor tuples:")
    print(f"  {len(u6_in)} groups / {len(u_struct_in)} tuples / {len(multi)} tuples on 2 to 5 machines "
          f"({hist[2]} on 2, {hist[3]} on 3, {hist[4]} on 4, {hist[5]} on 5)")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "qraft-grouped-report.json")
    with open(out_file, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
