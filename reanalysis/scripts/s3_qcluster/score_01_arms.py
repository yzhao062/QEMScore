"""Phase 2 score, step 1: build A, C and F, fit under the matched recipe, and
compute T, K, D and S with S formed inside each bootstrap draw.

Everything numeric in this file is governed by PRE-REGISTRATION.json, which was
written before this script ran. Nothing here chooses a metric, a boundary or a
resample count after seeing a number.

Arms (learner family and retraining recipe held fixed; only inputs move):
  A  affine control            LinearRegression over the ladder's descriptors
  C  capacity-matched control  the released 7-regressor bake-off, released
                               hyperparameters, selected by mean 5-fold MSE,
                               over the SAME descriptors as A
  F  full arm                  the released 8-column feature set under the same
                               bake-off and the same selection rule

Ladders: calibration available (esp_m joins A and C) and calibration withheld
(esp_m dropped from A and C entirely; it is a circuit x calibration product with
no pure-circuit part). F is identical in both.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("QCLUSTER_REPO")
if not DEFAULT_REPO:
    candidates = [
        os.path.join(ROOT, "../../upstream/s3_qcluster"),
        os.path.join(ROOT, "../../upstream/Q-Cluster"),
        os.path.join(ROOT, "Q-Cluster"),
    ]
    DEFAULT_REPO = next((c for c in candidates if os.path.exists(c)), candidates[0])

REPO = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("QCLUSTER_OUTPUTS", os.path.join(ROOT, "../../outputs/s3_qcluster"))
TRAIN = os.path.join(REPO, "data", "training_20250307_2151.pkl") if os.path.isdir(REPO) else REPO
os.makedirs(OUTDIR, exist_ok=True)

prereg_candidates = [
    os.path.join(OUTDIR, "PRE-REGISTRATION.json"),
    os.path.join(REPO, "PRE-REGISTRATION.json") if os.path.isdir(REPO) else None,
    os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "s3_qcluster", "PRE-REGISTRATION.json")),
    os.path.join(ROOT, "PRE-REGISTRATION.json"),
]
prereg_path = next((p for p in prereg_candidates if p and os.path.exists(p)), None)
if not prereg_path:
    raise FileNotFoundError("Could not find PRE-REGISTRATION.json")
with open(prereg_path) as f:
    PRE = json.load(f)
N_BOOT = int(PRE["resample_count"])
SEED = int(PRE["rng_seed"])
assert N_BOOT == 10000 and SEED == 20260904

COLUMNS = ['Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
           'entropy', 'esp_m', 'ip Entropy', 'Target']
DESCRIPTORS = ['Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr']
CALIB = ['esp_m']
EXEC_DERIVED = ['entropy']
RELEASED_X = DESCRIPTORS + EXEC_DERIVED + CALIB

# Resolve data source: pickle or npz
candidate_npz = [
    REPO if REPO.endswith(".npz") and os.path.exists(REPO) else None,
    os.path.join(REPO, "S3_target_table.npz") if os.path.isdir(REPO) else None,
    os.path.join(OUTDIR, "S3_target_table.npz"),
]
resolved_npz = next((c for c in candidate_npz if c and os.path.exists(c)), None)

if os.path.exists(TRAIN) and not TRAIN.endswith(".npz"):
    with open(TRAIN, "rb") as f:
        df = pd.DataFrame(pickle.load(f), columns=COLUMNS).dropna().reset_index(drop=True)
elif resolved_npz:
    z = np.load(resolved_npz)
    data_dict = {
        'Machine': z['machine'],
        'Algorithm': z['circuit'],
    }
    for idx, col in enumerate(COLUMNS[2:]):
        data_dict[col] = z['numeric'][:, idx]
    df = pd.DataFrame(data_dict)
else:
    raise FileNotFoundError(f"Neither training pickle nor S3_target_table.npz found. Run build_target_table.py or specify path.")
assert df.shape == (145, 12), df.shape
assert list(df.columns[2:10]) == RELEASED_X
y = df['Target'].values.astype(float)
N = len(df)


def released_bakeoff():
    """The exact candidate set and hyperparameters of released cell 6."""
    return {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                      random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "SVR": SVR(kernel='rbf', C=100, gamma=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
    }


def oof_predictions(X, model_factory, random_state):
    """Out-of-fold predictions under the released split, one value per row."""
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    pred = np.full(N, np.nan)
    per_fold_mse, per_fold_r2 = [], []
    for tr, te in kf.split(X):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        pred[te] = p
        per_fold_mse.append(mean_squared_error(y[te], p))
        per_fold_r2.append(r2_score(y[te], p))
    assert not np.isnan(pred).any()
    return pred, float(np.mean(per_fold_mse)), float(np.mean(per_fold_r2))


def run_bakeoff(X, random_state, label):
    """Run the released selection rule and return the winner's OOF predictions."""
    table = {}
    for name in released_bakeoff():
        pred, mmse, mr2 = oof_predictions(X, lambda n=name: released_bakeoff()[n], random_state)
        table[name] = {"mean_fold_MSE": mmse, "mean_fold_R2": mr2, "pred": pred}
    winner = min(table, key=lambda n: table[n]["mean_fold_MSE"])
    winner_r2 = max(table, key=lambda n: table[n]["mean_fold_R2"])
    print("  bake-off over %s (%d features), random_state=%d" % (label, X.shape[1], random_state))
    print("    %-18s %12s %12s" % ("regressor", "mean MSE", "mean R2"))
    for name in released_bakeoff():
        mark = "  <- selected" if name == winner else ""
        print("    %-18s %12.6f %12.4f%s"
              % (name, table[name]["mean_fold_MSE"], table[name]["mean_fold_R2"], mark))
    print("    selection by mean 5-fold MSE picks %s; by mean 5-fold R2 it picks %s"
          % (winner, winner_r2))
    return winner, table


print("=" * 96)
print("RELEASED-VERSUS-REFIT GAP (the plan calls this a reportable quantity, not a thing to fix)")
print("=" * 96)
X_released = df[RELEASED_X]
winner_F40, table_F40 = run_bakeoff(X_released, 40, "the released 8-column feature set")
released_cell6 = PRE and {
    "ExtraTrees": (0.0005, 0.9643), "RandomForest": (0.0006, 0.9535),
    "GradientBoosting": (0.0006, 0.9517), "DecisionTree": (0.0010, 0.9229),
    "SVR": (0.0126, -0.0199), "KNN": (0.0026, 0.8014), "XGBoost": (0.0006, 0.9452)}
print()
print("  released cell-6 stored output versus this refit, same seed, same hyperparameters:")
print("    %-18s %10s %10s %10s %10s" % ("regressor", "MSE rel", "MSE here", "R2 rel", "R2 here"))
for name, (rm, rr) in released_cell6.items():
    print("    %-18s %10.4f %10.4f %10.4f %10.4f"
          % (name, rm, table_F40[name]["mean_fold_MSE"], rr, table_F40[name]["mean_fold_R2"]))
print("    the release rounds MSE to 4 decimals, so the MSE column is a coarse comparison;")
print("    the R2 column is the informative one. Environment drift is recorded in the Phase 1")
print("    record (sklearn 1.4.2 -> 1.9.0, numpy 1.26.4 -> 2.2.6, xgboost 2.1.4 -> 3.2.0).")
print("    ExtraTrees is the released choice and this refit reproduces that choice: %s"
      % ("yes" if winner_F40 == "ExtraTrees" else "NO -- winner here is " + winner_F40))

LADDERS = {
    "with_calibration": DESCRIPTORS + CALIB,
    "without_calibration": DESCRIPTORS,
}

results = {}
for rs in (40, 42):
    print()
    print("=" * 96)
    print("FITTING THE ARMS, KFold(n_splits=5, shuffle=True, random_state=%d)" % rs)
    print("=" * 96)
    winner_F, table_F = (winner_F40, table_F40) if rs == 40 else run_bakeoff(X_released, rs,
                                                                            "the released 8-column feature set")
    pred_F = table_F[winner_F]["pred"]

    for ladder, feats in LADDERS.items():
        print()
        print("-" * 96)
        print("LADDER: %s   A and C read %s" % (ladder, feats))
        print("-" * 96)
        Xd = df[feats]
        pred_A, mseA_fold, r2A_fold = oof_predictions(Xd, LinearRegression, rs)
        winner_C, table_C = run_bakeoff(Xd, rs, "descriptors for this ladder")
        pred_C = table_C[winner_C]["pred"]

        results[(rs, ladder)] = {
            "features_AC": feats,
            "features_F": RELEASED_X,
            "winner_C": winner_C,
            "winner_F": winner_F,
            "pred_A": pred_A, "pred_C": pred_C, "pred_F": pred_F,
        }
        print("    arm A: LinearRegression, mean 5-fold MSE %.6f, mean 5-fold R2 %.4f"
              % (mseA_fold, r2A_fold))

np.save(os.path.join(OUTDIR, "oof_predictions.npy"),
        np.array([[results[(rs, l)]["pred_%s" % a] for a in ("A", "C", "F")]
                  for rs in (40, 42) for l in LADDERS], dtype=float))
with open(os.path.join(OUTDIR, "arm_meta.json"), "w") as f:
    json.dump({"order": [[rs, l] for rs in (40, 42) for l in LADDERS],
               "winner_C": {"%d|%s" % (rs, l): results[(rs, l)]["winner_C"]
                            for rs in (40, 42) for l in LADDERS},
               "winner_F": {"%d|%s" % (rs, l): results[(rs, l)]["winner_F"]
                            for rs in (40, 42) for l in LADDERS},
               "features_AC": {l: LADDERS[l] for l in LADDERS},
               "features_F": RELEASED_X}, f, indent=2)


def mse(pred, idx):
    d = y[idx] - pred[idx]
    return float(np.dot(d, d) / len(idx))


def mae(pred, idx):
    return float(np.mean(np.abs(y[idx] - pred[idx])))


print()
print("=" * 96)
print("THE ESTIMAND: T = A - F, K = A - C, D = C - F, S = K / T")
print("  point estimates on all 145 rows; intervals from %d bootstrap draws of the row" % N_BOOT)
print("  (one row = one circuit-machine pair), with T, K, D and S formed INSIDE each draw")
print("=" * 96)

rng = np.random.default_rng(SEED)
BOOT_IDX = rng.integers(0, N, size=(N_BOOT, N))   # one shared resample design for every panel

summary = {}
for rs in (40, 42):
    for ladder in LADDERS:
        r = results[(rs, ladder)]
        pA, pC, pF = r["pred_A"], r["pred_C"], r["pred_F"]
        full = np.arange(N)
        A0, C0, F0 = mse(pA, full), mse(pC, full), mse(pF, full)
        T0, K0, D0 = A0 - F0, A0 - C0, C0 - F0
        S0 = K0 / T0

        # bootstrap: recompute every quantity inside the draw
        dA = (y - pA) ** 2
        dC = (y - pC) ** 2
        dF = (y - pF) ** 2
        Ab = dA[BOOT_IDX].mean(axis=1)
        Cb = dC[BOOT_IDX].mean(axis=1)
        Fb = dF[BOOT_IDX].mean(axis=1)
        Tb, Kb, Db = Ab - Fb, Ab - Cb, Cb - Fb
        with np.errstate(divide='ignore', invalid='ignore'):
            Sb = Kb / Tb
        n_bad = int(np.sum(~np.isfinite(Sb)))
        frac_T_nonpos = float(np.mean(Tb <= 0))
        Sfin = Sb[np.isfinite(Sb)]
        lo, hi = np.percentile(Sfin, [2.5, 97.5])

        # MAE sensitivity, declared secondary
        aA = np.abs(y - pA)[BOOT_IDX].mean(axis=1)
        aC = np.abs(y - pC)[BOOT_IDX].mean(axis=1)
        aF = np.abs(y - pF)[BOOT_IDX].mean(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Sb_mae = (aA - aC) / (aA - aF)
        Sm = Sb_mae[np.isfinite(Sb_mae)]
        lo_m, hi_m = np.percentile(Sm, [2.5, 97.5])
        S0_mae = (mae(pA, full) - mae(pC, full)) / (mae(pA, full) - mae(pF, full))

        if lo > 0.5:
            reading = "PRESENT"
        elif hi < 0.5:
            reading = "ABSENT"
        else:
            reading = "INDETERMINATE"

        key = "rs%d|%s" % (rs, ladder)
        summary[key] = dict(
            random_state=rs, ladder=ladder, winner_C=r["winner_C"], winner_F=r["winner_F"],
            A_mse=A0, C_mse=C0, F_mse=F0,
            A_r2=float(r2_score(y, pA)), C_r2=float(r2_score(y, pC)), F_r2=float(r2_score(y, pF)),
            T=T0, K=K0, D=D0, S=S0,
            S_lo=float(lo), S_hi=float(hi),
            T_lo=float(np.percentile(Tb, 2.5)), T_hi=float(np.percentile(Tb, 97.5)),
            K_lo=float(np.percentile(Kb, 2.5)), K_hi=float(np.percentile(Kb, 97.5)),
            D_lo=float(np.percentile(Db, 2.5)), D_hi=float(np.percentile(Db, 97.5)),
            frac_T_nonpos=frac_T_nonpos, n_nonfinite_S=n_bad,
            n_boot=N_BOOT, reading=reading,
            S_mae=float(S0_mae), S_mae_lo=float(lo_m), S_mae_hi=float(hi_m),
        )

        print()
        print("-" * 96)
        print("PANEL  random_state=%d   ladder=%s" % (rs, ladder))
        print("-" * 96)
        print("  arm C selected: %-16s arm F selected: %s" % (r["winner_C"], r["winner_F"]))
        print("  out-of-fold MSE   A = %.6f   C = %.6f   F = %.6f" % (A0, C0, F0))
        print("  out-of-fold R2    A = %.4f     C = %.4f     F = %.4f"
              % (r2_score(y, pA), r2_score(y, pC), r2_score(y, pF)))
        print("  T = A - F = %.6f   [%.6f, %.6f]" % (T0, np.percentile(Tb, 2.5), np.percentile(Tb, 97.5)))
        print("  K = A - C = %.6f   [%.6f, %.6f]" % (K0, np.percentile(Kb, 2.5), np.percentile(Kb, 97.5)))
        print("  D = C - F = %.6f   [%.6f, %.6f]" % (D0, np.percentile(Db, 2.5), np.percentile(Db, 97.5)))
        print("  S = K / T = %.4f   95%% interval [%.4f, %.4f] from %d draws of 145 rows"
              % (S0, lo, hi, N_BOOT))
        print("     draws with T <= 0: %.4f%%   non-finite S: %d" % (100 * frac_T_nonpos, n_bad))
        print("  READING RULE -> %s" % reading)
        print("  (sensitivity, not the reading: S on MAE = %.4f [%.4f, %.4f])"
              % (S0_mae, lo_m, hi_m))

with open(os.path.join(OUTDIR, "RESULT-arms.json"), "w") as f:
    json.dump(summary, f, indent=2)
print()
print("wrote RESULT-arms.json, arm_meta.json, oof_predictions.npy")
