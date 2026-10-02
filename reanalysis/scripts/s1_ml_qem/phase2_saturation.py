"""PHASE 2, descriptor-saturation diagnostic for S1 (ML-QEM).

The plan: "Fit the strongest descriptor-only predictor the released data supports,
with no capacity match and no measurement input, and report the share of target
variance it explains on held-out rows. This bounds what ANY descriptor control
could reach."

So this deliberately does NOT match the released capacity. It sweeps a wide set of
descriptor-only learners on the 165 execution-independent columns, and takes the
best held-out R2 as the bound. Taking the maximum over the sweep on held-out rows
is optimistic by construction, which is what an upper bound wants; the whole table
is reported so the optimism is visible.

No arm here reads the four execution-derived columns, and none reads the target.
"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(BASE, "..", "..", "outputs", "s1_ml_qem"))
P1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
OUT = sys.argv[2] if len(sys.argv) > 2 else P1
os.makedirs(OUT, exist_ok=True)
SEED = 0


def candidates():
    """The strongest descriptor-only predictors the released data supports.
    No capacity match: these are allowed far more capacity than the released RF(100)."""
    return [
        ("LinearRegression (= arm A)", lambda: LinearRegression()),
        ("RidgeCV", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 40)))),
        ("RandomForest n=100 (= arm C)", lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)),
        ("RandomForest n=2000", lambda: RandomForestRegressor(n_estimators=2000, random_state=SEED, n_jobs=1)),
        ("RandomForest n=2000 max_features=1.0", lambda: RandomForestRegressor(
            n_estimators=2000, max_features=1.0, random_state=SEED, n_jobs=1)),
        ("ExtraTrees n=2000", lambda: ExtraTreesRegressor(n_estimators=2000, random_state=SEED, n_jobs=1)),
        ("ExtraTrees n=2000 max_features=1.0", lambda: ExtraTreesRegressor(
            n_estimators=2000, max_features=1.0, random_state=SEED, n_jobs=1)),
        ("GradientBoosting n=500", lambda: GradientBoostingRegressor(n_estimators=500, random_state=SEED)),
        ("HistGradientBoosting max_iter=1000", lambda: HistGradientBoostingRegressor(
            max_iter=1000, random_state=SEED)),
        ("kNN k=1", lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))),
        ("kNN k=3", lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3))),
        ("kNN k=5 distance-weighted", lambda: make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))),
        ("MLP (256,256,128)", lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(256, 256, 128), max_iter=4000, random_state=SEED))),
        ("MLP (512,256) tanh", lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(512, 256), activation="tanh", max_iter=4000, random_state=SEED))),
    ]


def fit_predict(make, Xtr, ytr, Xte):
    out = np.zeros((Xte.shape[0], ytr.shape[1]))
    for q in range(ytr.shape[1]):
        m = make()
        m.fit(Xtr, ytr[:, q])
        out[:, q] = m.predict(Xte)
    return out


def main():
    d = np.load(os.path.join(P1, "s1_hw4q.npz"))
    Xtr = d["X_train"].astype(np.float64)
    Xte = d["X_test"].astype(np.float64)
    ytr = d["y_train"].astype(np.float64)
    yte = d["y_test"].astype(np.float64)
    step_te = d["step_test"].astype(int)
    n_desc = int(d["n_gate"]) + int(d["n_angle"])

    Dtr = Xtr[:, :n_desc]
    Dte = Xte[:, :n_desc]
    assert Dtr.shape[1] == 165 and Dte.shape[1] == 165

    # sanity: no measurement input, no target, inside the descriptor block
    assert not np.allclose(Dte[:, -1], yte[:, 0])
    print("descriptor-only design: train %s  test %s" % (Dtr.shape, Dte.shape))

    def rowkey(a):
        return np.round(a, 9).tobytes()
    tr_set = set(rowkey(r) for r in Dtr)
    contaminated = np.array([rowkey(r) in tr_set for r in Dte])
    clean = ~contaminated
    print("leak-4 contaminated held-out rows: %d of %d\n" % (int(contaminated.sum()), len(clean)))

    rows = []
    print("%-40s %8s %8s %8s %8s" % ("descriptor-only predictor", "R2(all)", "R2(clean)",
                                     "R2(vw)", "meanL2"))
    print("-" * 78)
    for name, make in candidates():
        pred = fit_predict(make, Dtr, ytr, Dte)
        r2_all = float(r2_score(yte, pred, multioutput="uniform_average"))
        r2_vw = float(r2_score(yte, pred, multioutput="variance_weighted"))
        r2_raw = [float(v) for v in r2_score(yte, pred, multioutput="raw_values")]
        r2_clean = float(r2_score(yte[clean], pred[clean], multioutput="uniform_average"))
        meanl2 = float(np.mean(np.sqrt(np.sum((pred - yte) ** 2, axis=1))))
        rows.append({"model": name, "r2_uniform_all": r2_all, "r2_variance_weighted_all": r2_vw,
                     "r2_per_observable_all": r2_raw, "r2_uniform_leakfree": r2_clean,
                     "mean_per_row_L2_all": meanl2})
        print("%-40s %8.4f %8.4f %8.4f %8.4f" % (name, r2_all, r2_clean, r2_vw, meanl2))

    best = max(rows, key=lambda r: r["r2_uniform_all"])
    best_clean = max(rows, key=lambda r: r["r2_uniform_leakfree"])

    print("\nSATURATION (the bound on what any descriptor control could reach)")
    print("  best held-out R2 on all 2500 rows      : %.4f   (%s)"
          % (best["r2_uniform_all"], best["model"]))
    print("    per observable: %s" % ["%.4f" % v for v in best["r2_per_observable_all"]])
    print("  best held-out R2 on leak-free rows only: %.4f   (%s)   [qualification, %d rows]"
          % (best_clean["r2_uniform_leakfree"], best_clean["model"], int(clean.sum())))

    # target variance, for context on what R2 is a share of
    print("\n  target variance per observable on held-out rows: %s"
          % ["%.4f" % v for v in yte.var(axis=0)])

    out = {
        "system": "S1 ML-QEM",
        "diagnostic": "descriptor saturation: the strongest descriptor-only predictor the released data supports, no capacity match, no measurement input",
        "design": "165 execution-independent columns (5 gate counts + 160 rotation-angle bin counts); 500 training rows; held out = the released 2500-row test split",
        "metric": "R2 on held-out rows, uniform average across the 4 observables (variance-weighted and per-observable also reported)",
        "selection": "the maximum over the sweep, taken on held-out rows because the quantity wanted is an upper bound; the full table is reported so the optimism is visible",
        "saturation_all_rows": {"r2": best["r2_uniform_all"], "model": best["model"],
                                "per_observable": best["r2_per_observable_all"]},
        "saturation_leakfree_rows": {"r2": best_clean["r2_uniform_leakfree"], "model": best_clean["model"],
                                     "rows": int(clean.sum()),
                                     "status": "qualification; the declared number is the all-rows one"},
        "target_variance_per_observable_heldout": [float(v) for v in yte.var(axis=0)],
        "leak4_contaminated_heldout_rows": int(contaminated.sum()),
        "table": rows,
    }
    with open(os.path.join(OUT, "S1_saturation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.join(OUT, "S1_saturation.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
