"""Score the QRAFT panel and unmitigated baseline under the frozen contract.

Panel R reads QRAFT released predictions and unmitigated forward execution.
Panel M retrains the three learned arms under matched recipes over ten seeds.

The score is per-row mean absolute error in probability points.
Rows are weighted equally within machine and machines are weighted equally across.

For Panel R under the raw option:
Resampling uses 10000 bootstrap draws with random seed 20261002.
The resampling unit is the test row stratified by machine.
Arm R, learned arm errors, and contrasts with R use identical resampled rows.

For Panel M under the raw option:
The estimand is the fixed ten-seed mean for each arm and contrast.
Intervals come from a joint row bootstrap stratified by machine.
Resampling uses 10000 bootstrap draws with random seed 20261002.
The draw applies to every seed split simultaneously.
Rows absent from a seed split are skipped for that seed.
Within each draw, per-seed macro MAEs are averaged across the ten seeds.
Each contrast draw equals the learned arm draw minus the arm R draw.

The raw baseline arm R and contrasts are governed by
docs/frozen-rules/2026-10-06-round9-follow-ups.md.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import numpy as np

TARGET = "StateRealProb"
MACHINE = "ComputerID"
DESCRIPTORS = ("CircuitWidth", "CircuitDepth", "CircuitNumU1Gates",
               "CircuitNumU2Gates", "CircuitNumU3Gates", "CircuitNumCXGates",
               "StateHammingWeight")
FORWARD = ("StateUpProb25", "StateUpProb50", "StateUpProb75")
REVERSE = ("TotalUpDnErr25", "TotalUpDnErr50", "TotalUpDnErr75",
           "StateUpDnErr25", "StateUpDnErr50", "StateUpDnErr75")
ARMS = {
    "Q3": DESCRIPTORS,                                   # train.m tran3
    "Q2": DESCRIPTORS + FORWARD,                         # train.m tran2
    "Q1": DESCRIPTORS + FORWARD + REVERSE + (MACHINE,),  # train.m tran1
}
RELEASED_PREDICTION = {"Q1": "StatePredPrT1", "Q2": "StatePredPrT2",
                       "Q3": "StatePredPrT3"}
RAW_COLUMN = "StateUpProb50"
INPUT_COLUMNS = (MACHINE,) + DESCRIPTORS + FORWARD + REVERSE + (TARGET,)
EXPECTED_INPUT_ROWS = 10155
EXPECTED_OUTPUT_ROWS = 1524
SEEDS = tuple(range(10))
TEST_FRACTION = 0.15   # train.m holds out the last 15 percent after shuffling
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261002


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load(path, required):
    """Read a released CSV, refusing a schema that is not the declared one."""
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


def panel_released(rows, include_raw=False, return_draws=False):
    machines = np.array([r[MACHINE] for r in rows])
    y = numeric(rows, (TARGET,)).ravel()
    result = {}
    errs = {}
    for arm, column in RELEASED_PREDICTION.items():
        pred = numeric(rows, (column,)).ravel()
        err = np.abs(pred - y)
        errs[arm] = err
        macro, per_machine = macro_mae(machines, err)
        result[arm] = {"macro_mae": macro, "per_machine_mae": per_machine,
                       "features": list(ARMS[arm]), "released_column": column}

    boot_draws_dict = None
    if include_raw:
        pred_r = numeric(rows, (RAW_COLUMN,)).ravel()
        err_r = np.abs(pred_r - y)
        errs["R"] = err_r
        macro_r, per_machine_r = macro_mae(machines, err_r)
        result["R"] = {
            "macro_mae": macro_r,
            "per_machine_mae": per_machine_r,
            "features": [RAW_COLUMN],
            "released_column": RAW_COLUMN,
        }

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        unique_m = sorted(set(machines))
        m_samples = {
            m: np.where(machines == m)[0][
                rng.integers(0, np.sum(machines == m), size=(BOOTSTRAP_DRAWS, np.sum(machines == m)))
            ]
            for m in unique_m
        }
        boot_macro = {}
        for arm_k, err in errs.items():
            per_m_mae = np.column_stack([np.mean(err[m_samples[m]], axis=1) for m in unique_m])
            boot_macro[arm_k] = np.mean(per_m_mae, axis=1)
            result[arm_k]["lower"] = float(np.percentile(boot_macro[arm_k], 2.5))
            result[arm_k]["upper"] = float(np.percentile(boot_macro[arm_k], 97.5))

        contrasts = {}
        contrast_draws = {}
        for arm in ("Q1", "Q2", "Q3"):
            c_draws = boot_macro[arm] - boot_macro["R"]
            contrast_draws[arm] = c_draws
            contrasts[f"{arm}-R"] = {
                "estimate": float(result[arm]["macro_mae"] - macro_r),
                "lower": float(np.percentile(c_draws, 2.5)),
                "upper": float(np.percentile(c_draws, 97.5)),
            }
        result["contrasts_with_R"] = contrasts
        if return_draws:
            boot_draws_dict = {
                "arm_draws": boot_macro,
                "contrast_draws": contrast_draws,
            }

    if return_draws:
        return result, boot_draws_dict
    return result


def panel_matched(rows, include_raw=False, return_draws=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    machines_all = np.array([r[MACHINE] for r in rows])
    y_all = numeric(rows, (TARGET,)).ravel()
    n = len(rows)
    n_test = int(round(n * TEST_FRACTION))
    result = {arm: {"per_seed_macro_mae": {}} for arm in ARMS}
    if include_raw:
        result["R"] = {"per_seed_macro_mae": {}}
        x_raw = numeric(rows, (RAW_COLUMN,)).ravel()

    test_splits = {}
    seed_errors = {arm: {} for arm in ARMS}
    if include_raw:
        seed_errors["R"] = {}

    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(n)
        test = order[:n_test]
        train = order[n_test:]
        test_splits[seed] = test
        test_machines = machines_all[test]
        test_y = y_all[test]

        for arm, columns in ARMS.items():
            x = numeric(rows, columns)
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(x[train], y_all[train])
            err = np.abs(model.predict(x[test]) - test_y)
            seed_errors[arm][seed] = err
            macro, _ = macro_mae(test_machines, err)
            result[arm]["per_seed_macro_mae"][str(seed)] = macro

        if include_raw:
            err_r = np.abs(x_raw[test] - test_y)
            seed_errors["R"][seed] = err_r
            macro_r, _ = macro_mae(test_machines, err_r)
            result["R"]["per_seed_macro_mae"][str(seed)] = macro_r

    all_arms = list(ARMS)
    if include_raw:
        all_arms.append("R")

    for arm in all_arms:
        v = [result[arm]["per_seed_macro_mae"][str(s)] for s in SEEDS]
        feats = [RAW_COLUMN] if arm == "R" else list(ARMS[arm])
        result[arm].update(ten_seed_mean=float(np.mean(v)),
                           ten_seed_min=float(min(v)), ten_seed_max=float(max(v)),
                           features=feats)

    boot_draws_dict = None
    if include_raw:
        # Joint row bootstrap across overlapping splits stratified by machine.
        # Resample test rows once per draw, stratified by machine.
        # Apply the draw to every seed split at once (rows absent from a seed split are skipped).
        # Average the per-seed macro MAEs within each draw across the ten seeds.
        test_rows_all = np.unique(np.concatenate([test_splits[s] for s in SEEDS]))
        unique_m = sorted(set(machines_all))
        T_m = {m: np.array([idx for idx in test_rows_all if machines_all[idx] == m], dtype=int)
               for m in unique_m}

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        sampled_indices = {
            m: rng.integers(0, len(T_m[m]), size=(BOOTSTRAP_DRAWS, len(T_m[m])))
            for m in unique_m
        }

        # Precompute per-seed, per-machine test membership masks and index maps
        seed_m_info = {}
        for s in SEEDS:
            test_set_s = set(test_splits[s])
            seed_m_info[s] = {}
            test_pos_map = {row_idx: pos for pos, row_idx in enumerate(test_splits[s])}
            for m in unique_m:
                in_test_s = np.array([idx in test_set_s for idx in T_m[m]], dtype=bool)
                err_positions = np.array([test_pos_map[idx] if idx in test_set_s else -1 for idx in T_m[m]], dtype=int)
                sampled_mask = in_test_s[sampled_indices[m]]
                sampled_denom = np.sum(sampled_mask, axis=1)
                sampled_err_pos = err_positions[sampled_indices[m]]
                seed_m_info[s][m] = {
                    "mask": sampled_mask,
                    "denom": sampled_denom,
                    "err_pos": sampled_err_pos,
                }

        # Compute per-seed macro MAE draws for each arm (Q1, Q2, Q3, R)
        seed_draw_arms = {arm: [] for arm in all_arms}
        for arm in all_arms:
            for s in SEEDS:
                err_arm = seed_errors[arm][s]
                m_draws = []
                for m in unique_m:
                    info = seed_m_info[s][m]
                    mask = info["mask"]
                    denom = info["denom"]
                    err_pos = info["err_pos"]
                    sampled_err = np.where(mask, err_arm[np.maximum(err_pos, 0)], 0.0)
                    numer = np.sum(sampled_err, axis=1)
                    point_err_m = np.mean(err_arm[np.array([pos for pos, idx in enumerate(test_splits[s]) if machines_all[idx] == m])])
                    m_mean = np.where(denom > 0, numer / np.maximum(denom, 1), point_err_m)
                    m_draws.append(m_mean)
                macro_arm_s = np.mean(m_draws, axis=0)
                seed_draw_arms[arm].append(macro_arm_s)

        arm_ten_seed_draws = {}
        for arm in all_arms:
            ten_seed_draws_arm = np.mean(seed_draw_arms[arm], axis=0)
            arm_ten_seed_draws[arm] = ten_seed_draws_arm
            result[arm]["lower"] = float(np.percentile(ten_seed_draws_arm, 2.5))
            result[arm]["upper"] = float(np.percentile(ten_seed_draws_arm, 97.5))

        contrasts = {}
        contrast_draws = {}
        for arm in ARMS:
            diffs = [result[arm]["per_seed_macro_mae"][str(s)] - result["R"]["per_seed_macro_mae"][str(s)] for s in SEEDS]
            c_draws = arm_ten_seed_draws[arm] - arm_ten_seed_draws["R"]
            contrast_draws[arm] = c_draws
            contrasts[f"{arm}-R"] = {
                "ten_seed_mean": float(np.mean(diffs)),
                "ten_seed_min": float(min(diffs)),
                "ten_seed_max": float(max(diffs)),
                "lower": float(np.percentile(c_draws, 2.5)),
                "upper": float(np.percentile(c_draws, 97.5)),
                "per_seed": {str(s): float(diffs[s]) for s in SEEDS},
            }
        result["contrasts_with_R"] = contrasts
        if return_draws:
            boot_draws_dict = {
                "arm_draws": arm_ten_seed_draws,
                "contrast_draws": contrast_draws,
            }

    if return_draws:
        return result, boot_draws_dict
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("release", help="Path to unpacked release or QRAFT directory")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Include unmitigated baseline arm R and contrasts of learned arms with R",
    )
    parser.add_argument(
        "--frozen-rule",
        default=None,
        help="Path to frozen rule file (required when --raw is requested)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.raw:
        if not args.frozen_rule or not os.path.isfile(args.frozen_rule):
            raise RuntimeError(
                "FROZEN RULE BARRIER: Governed outcomes (arm R or contrasts with R) "
                "require a valid, existing --frozen-rule file. "
                f"Provided path: {args.frozen_rule!r}. Aborting immediately."
            )

    rel = args.release
    outdir = args.outdir
    qraft_dir = rel if os.path.isdir(os.path.join(rel, "data_trained")) else os.path.join(rel, "QRAFT")
    out_csv = os.path.join(qraft_dir, "data_trained", "outputData.csv")
    in_csv = os.path.join(qraft_dir, "data_processed", "inputData.csv")

    out_rows = load(out_csv, INPUT_COLUMNS + tuple(RELEASED_PREDICTION.values()))
    in_rows = load(in_csv, INPUT_COLUMNS)
    for name, rows, expected in (("outputData.csv", out_rows, EXPECTED_OUTPUT_ROWS),
                                 ("inputData.csv", in_rows, EXPECTED_INPUT_ROWS)):
        if len(rows) != expected:
            raise SystemExit(f"{name}: expected {expected} rows, found {len(rows)}")
    for arm, columns in ARMS.items():
        if TARGET in columns:
            raise SystemExit(f"{arm}: the target is in the feature set")

    panel_r = panel_released(out_rows, include_raw=args.raw)
    panel_m = panel_matched(in_rows, include_raw=args.raw)

    intervals_desc = (
        "none; see PLAN-qraft-panel.md for why"
        if not args.raw
        else "percentile bootstrap within machine for arm errors and contrasts with R"
    )
    report = {"panel_R_released": panel_r,
              "panel_M_matched": panel_m,
              "score": "per-row mean absolute error in probability points",
              "weighting": "rows equal within machine, machines equal across",
              "intervals": intervals_desc,
              "seeds": list(SEEDS), "test_fraction": TEST_FRACTION}

    if args.raw:
        report["frozen_rule"] = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
        report["frozen_rule_sha256"] = compute_file_sha256(args.frozen_rule)
        report["script_sha256"] = compute_file_sha256(__file__)
        report["input_sha256"] = {
            "outputData.csv": compute_file_sha256(out_csv),
            "inputData.csv": compute_file_sha256(in_csv),
        }

    arms_to_print = ("Q3", "Q2", "Q1", "R") if args.raw else ("Q3", "Q2", "Q1")
    print("Panel R, QRAFT's released predictions, no fitting by us")
    if args.raw:
        print(f"{'arm':<5}{'features':>10}{'macro MAE':>12}{'lower':>10}{'upper':>10}")
        for arm in arms_to_print:
            r = report["panel_R_released"][arm]
            print(f"{arm:<5}{len(r['features']):>10}{r['macro_mae']:>12.4f}{r['lower']:>10.4f}{r['upper']:>10.4f}")
    else:
        print(f"{'arm':<5}{'features':>10}{'macro MAE':>12}")
        for arm in arms_to_print:
            r = report["panel_R_released"][arm]
            print(f"{arm:<5}{len(r['features']):>10}{r['macro_mae']:>12.4f}")
    if args.raw and "contrasts_with_R" in report["panel_R_released"]:
        print("Panel R contrasts with unmitigated baseline R:")
        for c_name, c_val in report["panel_R_released"]["contrasts_with_R"].items():
            print(f"  {c_name}: estimate={c_val['estimate']:+.4f} [{c_val['lower']:+.4f}, {c_val['upper']:+.4f}]")
    print()

    print("Panel M, matched retraining, ten seeds")
    if args.raw:
        print(f"{'arm':<5}{'features':>10}{'mean':>12}{'min':>12}{'max':>12}{'lower':>10}{'upper':>10}")
        for arm in arms_to_print:
            r = report["panel_M_matched"][arm]
            print(f"{arm:<5}{len(r['features']):>10}{r['ten_seed_mean']:>12.4f}"
                  f"{r['ten_seed_min']:>12.4f}{r['ten_seed_max']:>12.4f}"
                  f"{r['lower']:>10.4f}{r['upper']:>10.4f}")
    else:
        print(f"{'arm':<5}{'features':>10}{'mean':>12}{'min':>12}{'max':>12}")
        for arm in arms_to_print:
            r = report["panel_M_matched"][arm]
            print(f"{arm:<5}{len(r['features']):>10}{r['ten_seed_mean']:>12.4f}"
                  f"{r['ten_seed_min']:>12.4f}{r['ten_seed_max']:>12.4f}")
    if args.raw and "contrasts_with_R" in report["panel_M_matched"]:
        print("Panel M contrasts with unmitigated baseline R:")
        for c_name, c_val in report["panel_M_matched"]["contrasts_with_R"].items():
            print(f"  {c_name}: ten_seed_mean={c_val['ten_seed_mean']:+.4f} [{c_val['lower']:+.4f}, {c_val['upper']:+.4f}]")

    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "qraft-panel-report.json")
    with open(out_file, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
