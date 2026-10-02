"""Phase 2 score, step 12: does shared selection optimism manufacture the reading?

The strongest technical objection to the primary panels. Arms C and F are each
chosen by the released 7-regressor bake-off scored on the same folds they are
then reported on, while arm A is a single model that pays no selection cost. If
both C and F are optimistic by roughly the same amount d, then

    S_observed = (A - C_true + d) / (A - F_true + d)

and since T > K, adding d to both moves S toward 1. Shared optimism therefore
biases the reading toward PRESENT, and the size of that push has to be measured
rather than assumed.

This runs the identical released bake-off, but selects inside each outer training
fold instead of on the folds being scored, so the selection is paid for in both
C and F. It is NOT the released recipe and NOT the score; the released recipe
selects on the full cross-validation, which is what the primary panels reproduce.
It is here to say how much of S survives when the shared optimism is removed.
"""
import os
import json
import pickle
import warnings
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

import sys
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("QCLUSTER_REPO")
if not DEFAULT_REPO:
    candidates_list = [
        os.path.join(ROOT, "../../upstream/s3_qcluster"),
        os.path.join(ROOT, "../../upstream/Q-Cluster"),
        os.path.join(ROOT, "Q-Cluster"),
    ]
    DEFAULT_REPO = next((c for c in candidates_list if os.path.exists(c)), candidates_list[0])

if len(sys.argv) == 3:
    INPUTS_DIR = sys.argv[1]
    if os.path.exists(os.path.join(sys.argv[2], "data", "training_20250307_2151.pkl")) or sys.argv[2].endswith(".pkl"):
        REPO = sys.argv[2]
        OUTDIR = INPUTS_DIR
    else:
        REPO = DEFAULT_REPO
        OUTDIR = sys.argv[2]
elif len(sys.argv) > 3:
    INPUTS_DIR = sys.argv[1]
    REPO = sys.argv[2]
    OUTDIR = sys.argv[3]
else:
    INPUTS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QCLUSTER_INPUTS", os.path.join(ROOT, "../../outputs/s3_qcluster"))
    REPO = DEFAULT_REPO
    OUTDIR = INPUTS_DIR
os.makedirs(OUTDIR, exist_ok=True)

COLUMNS = ['Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
           'entropy', 'esp_m', 'ip Entropy', 'Target']
DESCRIPTORS = ['Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr']
CALIB = ['esp_m']
RELEASED_X = DESCRIPTORS + ['entropy'] + CALIB
LADDERS = {"with_calibration": DESCRIPTORS + CALIB, "without_calibration": DESCRIPTORS}

train_pkl = os.path.join(REPO, "data", "training_20250307_2151.pkl") if os.path.isdir(REPO) else REPO
npz_candidates = [
    os.path.join(INPUTS_DIR, "S3_target_table.npz"),
    os.path.join(OUTDIR, "S3_target_table.npz"),
    os.path.join(REPO, "S3_target_table.npz") if os.path.isdir(REPO) else None,
    REPO if REPO.endswith(".npz") and os.path.exists(REPO) else None,
    INPUTS_DIR if INPUTS_DIR.endswith(".npz") and os.path.exists(INPUTS_DIR) else None,
    os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "s3_qcluster", "S3_target_table.npz")),
]
npz_path = next((p for p in npz_candidates if p and os.path.exists(p)), None)
if os.path.exists(train_pkl):
    with open(train_pkl, "rb") as f:
        df = pd.DataFrame(pickle.load(f), columns=COLUMNS).dropna().reset_index(drop=True)
elif npz_path and os.path.exists(npz_path):
    z = np.load(npz_path)
    data_dict = {
        'Machine': z['machine'],
        'Algorithm': z['circuit'],
    }
    for idx, col in enumerate(COLUMNS[2:]):
        data_dict[col] = z['numeric'][:, idx]
    df = pd.DataFrame(data_dict)
else:
    raise FileNotFoundError("Could not find training_20250307_2151.pkl or S3_target_table.npz")

y = df['Target'].values.astype(float)
N = len(df)


def bakeoff():
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


def nested_bakeoff_oof(feats, outer):
    """Released bake-off, but selected inside each outer training fold."""
    pred = np.full(N, np.nan)
    picked = []
    for tr, te in outer:
        inner = list(KFold(n_splits=5, shuffle=True, random_state=7).split(tr))
        best, best_mse = None, np.inf
        for name in bakeoff():
            ip = np.full(len(tr), np.nan)
            for itr, ite in inner:
                m = bakeoff()[name]
                m.fit(df[feats].iloc[tr[itr]], y[tr[itr]])
                ip[ite] = m.predict(df[feats].iloc[tr[ite]])
            mse = mean_squared_error(y[tr], ip)
            if mse < best_mse:
                best, best_mse = name, mse
        m = bakeoff()[best]
        m.fit(df[feats].iloc[tr], y[tr])
        pred[te] = m.predict(df[feats].iloc[te])
        picked.append(best)
    return pred, picked


def plain_oof(feats, factory, outer):
    pred = np.full(N, np.nan)
    for tr, te in outer:
        m = factory()
        m.fit(df[feats].iloc[tr], y[tr])
        pred[te] = m.predict(df[feats].iloc[te])
    return pred


rng = np.random.default_rng(20260904)
RB = rng.integers(0, N, size=(10000, N))
out = {}

for rs in (40, 42):
    outer = list(KFold(n_splits=5, shuffle=True, random_state=rs).split(df))
    pF, pickF = nested_bakeoff_oof(RELEASED_X, outer)
    print("=" * 96)
    print("SELECTION PAID FOR INSIDE EACH TRAINING FOLD, random_state=%d" % rs)
    print("=" * 96)
    print("  arm F picked, fold by fold: %s" % pickF)
    for ladder, feats in LADDERS.items():
        pA = plain_oof(feats, LinearRegression, outer)
        pC, pickC = nested_bakeoff_oof(feats, outer)
        eA, eC, eF = (y - pA) ** 2, (y - pC) ** 2, (y - pF) ** 2
        A0, C0, F0 = eA.mean(), eC.mean(), eF.mean()
        T0, K0, D0, S0 = A0 - F0, A0 - C0, C0 - F0, (A0 - C0) / (A0 - F0)
        Sb = (eA[RB].mean(1) - eC[RB].mean(1)) / (eA[RB].mean(1) - eF[RB].mean(1))
        lo, hi = np.percentile(Sb[np.isfinite(Sb)], [2.5, 97.5])
        rd = "PRESENT" if lo > 0.5 else ("ABSENT" if hi < 0.5 else "INDETERMINATE")
        print()
        print("  %s" % ladder)
        print("    arm C picked, fold by fold: %s" % pickC)
        print("    MSE  A=%.6f  C=%.6f  F=%.6f" % (A0, C0, F0))
        print("    R2   A=%.4f    C=%.4f    F=%.4f"
              % (r2_score(y, pA), r2_score(y, pC), r2_score(y, pF)))
        print("    T=%.6f  K=%.6f  D=%.6f" % (T0, K0, D0))
        print("    S=%.4f  [%.4f, %.4f]  (reading if the rule were applied: %s)"
              % (S0, lo, hi, rd))
        out["rs%d|%s" % (rs, ladder)] = dict(
            A=float(A0), C=float(C0), F=float(F0), T=float(T0), K=float(K0), D=float(D0),
            S=float(S0), S_lo=float(lo), S_hi=float(hi), reading_if_applied=rd,
            picked_C=pickC, picked_F=pickF)
    print()

print("=" * 96)
print("COMPARISON WITH THE PRIMARY PANELS (which use the released selection)")
print("=" * 96)
prim_path = os.path.join(INPUTS_DIR, "RESULT-arms.json")
if not os.path.exists(prim_path):
    prim_path = os.path.join(ROOT, "RESULT-arms.json")
prim = json.load(open(prim_path))
print("  %-30s %10s %10s %10s" % ("panel", "S released", "S nested", "shift"))
for k in out:
    if k in prim:
        print("  %-30s %10.4f %10.4f %+10.4f"
              % (k, prim[k]["S"], out[k]["S"], out[k]["S"] - prim[k]["S"]))

with open(os.path.join(OUTDIR, "RESULT-selection-optimism.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote RESULT-selection-optimism.json")
