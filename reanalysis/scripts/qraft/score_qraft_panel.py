"""The QRAFT panel, scored exactly as PLAN-qraft-panel.md fixes it.

Panel R reads QRAFT's own released predictions and fits nothing. Panel M
retrains the same three arms under one stated recipe over ten seeds.

The arms are transcribed from QRAFT/code_train/train.m, not designed here.
StateRealProb is the response in every fitcensemble call there, so it is the
target and never an input.

The score is per-row mean absolute error in probability points, weighted
equally within each machine and then equally across machines. Hellinger is not
computable because the released output carries no circuit or state identifier.

No bootstrap interval is produced. The released split is shuffled by row, so a
circuit's states straddle train and test and no resample respecting
within-circuit correlation can be built.

Usage: score_qraft_panel.py <unpacked-release-dir> <output-dir>
"""
import csv, json, os, sys
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(__doc__.strip().splitlines()[-1])
REL, OUTDIR = sys.argv[1], sys.argv[2]

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
INPUT_COLUMNS = (MACHINE,) + DESCRIPTORS + FORWARD + REVERSE + (TARGET,)
EXPECTED_INPUT_ROWS = 10155
EXPECTED_OUTPUT_ROWS = 1524
SEEDS = tuple(range(10))
TEST_FRACTION = 0.15   # train.m holds out the last 15 percent after shuffling


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


def panel_released(rows):
    machines = np.array([r[MACHINE] for r in rows])
    y = numeric(rows, (TARGET,)).ravel()
    result = {}
    for arm, column in RELEASED_PREDICTION.items():
        pred = numeric(rows, (column,)).ravel()
        macro, per_machine = macro_mae(machines, np.abs(pred - y))
        result[arm] = {"macro_mae": macro, "per_machine_mae": per_machine,
                       "features": list(ARMS[arm]), "released_column": column}
    return result


def panel_matched(rows):
    from sklearn.ensemble import HistGradientBoostingRegressor
    machines_all = np.array([r[MACHINE] for r in rows])
    y_all = numeric(rows, (TARGET,)).ravel()
    n = len(rows)
    n_test = int(round(n * TEST_FRACTION))
    result = {arm: {"per_seed_macro_mae": {}} for arm in ARMS}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(n)
        test = order[:n_test]
        train = order[n_test:]
        for arm, columns in ARMS.items():
            x = numeric(rows, columns)
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(x[train], y_all[train])
            err = np.abs(model.predict(x[test]) - y_all[test])
            macro, _ = macro_mae(machines_all[test], err)
            result[arm]["per_seed_macro_mae"][str(seed)] = macro
    for arm in ARMS:
        v = [result[arm]["per_seed_macro_mae"][str(s)] for s in SEEDS]
        result[arm].update(ten_seed_mean=float(np.mean(v)),
                           ten_seed_min=float(min(v)), ten_seed_max=float(max(v)),
                           features=list(ARMS[arm]))
    return result


def main():
    qraft_dir = REL if os.path.isdir(os.path.join(REL, "data_trained")) else os.path.join(REL, "QRAFT")
    out_rows = load(os.path.join(qraft_dir, "data_trained", "outputData.csv"),
                    INPUT_COLUMNS + tuple(RELEASED_PREDICTION.values()))
    in_rows = load(os.path.join(qraft_dir, "data_processed", "inputData.csv"),
                   INPUT_COLUMNS)
    for name, rows, expected in (("outputData.csv", out_rows, EXPECTED_OUTPUT_ROWS),
                                 ("inputData.csv", in_rows, EXPECTED_INPUT_ROWS)):
        if len(rows) != expected:
            raise SystemExit(f"{name}: expected {expected} rows, found {len(rows)}")
    for arm, columns in ARMS.items():
        if TARGET in columns:
            raise SystemExit(f"{arm}: the target is in the feature set")

    report = {"panel_R_released": panel_released(out_rows),
              "panel_M_matched": panel_matched(in_rows),
              "score": "per-row mean absolute error in probability points",
              "weighting": "rows equal within machine, machines equal across",
              "intervals": "none; see PLAN-qraft-panel.md for why",
              "seeds": list(SEEDS), "test_fraction": TEST_FRACTION}

    print("Panel R, QRAFT's released predictions, no fitting by us")
    print(f"{'arm':<5}{'features':>10}{'macro MAE':>12}")
    for arm in ("Q3", "Q2", "Q1"):
        r = report["panel_R_released"][arm]
        print(f"{arm:<5}{len(r['features']):>10}{r['macro_mae']:>12.4f}")
    print()
    print("Panel M, matched retraining, ten seeds")
    print(f"{'arm':<5}{'features':>10}{'mean':>12}{'min':>12}{'max':>12}")
    for arm in ("Q3", "Q2", "Q1"):
        r = report["panel_M_matched"][arm]
        print(f"{arm:<5}{len(r['features']):>10}{r['ten_seed_mean']:>12.4f}"
              f"{r['ten_seed_min']:>12.4f}{r['ten_seed_max']:>12.4f}")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "qraft-panel-report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote qraft-panel-report.json")


main()
