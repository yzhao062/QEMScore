"""Phase 2 score, step 2 (rerun, leaner): the descriptor-saturation diagnostic.

Replaces score_02_saturation.py, whose nested loop over 500-tree ensembles did
not finish in a reasonable time and was stopped. Nothing about the definition
changes; only the cost. The flat search, which is the quantity the plan actually
asks for, keeps its full candidate set. The nested version, which is the extra
honesty check, uses 3 inner folds and 300-tree ensembles.

The plan: "Fit the strongest descriptor-only predictor the released data
supports, with no capacity match and no measurement input, and report the share
of target variance it explains on held-out rows. This bounds what ANY descriptor
control could reach."

  flat    maximum out-of-fold R2 over the whole search, selected on the folds it
          is reported on. Optimistic, which is the conservative direction for an
          upper bound on what a descriptor control could reach.
  nested  the same search, but the candidate is chosen inside each outer training
          fold and only then scored on the held-out rows, so the selection is
          paid for.
"""
import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("QCLUSTER_REPO")
if not DEFAULT_REPO:
    candidates_list = [
        os.path.join(ROOT, "../../upstream/s3_qcluster"),
        os.path.join(ROOT, "../../upstream/Q-Cluster"),
        os.path.join(ROOT, "Q-Cluster"),
    ]
    DEFAULT_REPO = next((c for c in candidates_list if os.path.exists(c)), candidates_list[0])

REPO = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("QCLUSTER_OUTPUTS", os.path.join(ROOT, "../../outputs/s3_qcluster"))
TRAIN = os.path.join(REPO, "data", "training_20250307_2151.pkl") if os.path.isdir(REPO) else REPO
os.makedirs(OUTDIR, exist_ok=True)

COLUMNS = ['Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
           'entropy', 'esp_m', 'ip Entropy', 'Target']
DESCRIPTORS = ['Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr']
CALIB = ['esp_m']

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
assert df.shape == (145, 12)
y = df['Target'].values.astype(float)
groups = df['Algorithm'].values
N = len(df)
NTREE = 300


def candidates():
    """A search far larger than the released 7-regressor bake-off.
    No execution-derived input reaches any of these."""
    c = {"LinearRegression": lambda: LinearRegression()}
    for a in (0.001, 0.01, 0.1, 1.0, 10.0):
        c["Ridge(a=%g)" % a] = lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))
        c["Poly2Ridge(a=%g)" % a] = lambda a=a: make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=a))
    for a in (1e-4, 1e-3, 1e-2):
        c["Lasso(a=%g)" % a] = lambda a=a: make_pipeline(StandardScaler(),
                                                         Lasso(alpha=a, max_iter=20000))
        c["ElasticNet(a=%g)" % a] = lambda a=a: make_pipeline(
            StandardScaler(), ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000))
    for k in (1, 2, 3, 5, 7, 10):
        c["KNN(k=%d)" % k] = lambda k=k: KNeighborsRegressor(n_neighbors=k)
        c["KNNscaled(k=%d)" % k] = lambda k=k: make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    for C in (1.0, 10.0, 100.0, 1000.0):
        for g in ("scale", 0.01, 0.1, 1.0):
            c["SVRscaled(C=%g,g=%s)" % (C, g)] = lambda C=C, g=g: make_pipeline(
                StandardScaler(), SVR(kernel='rbf', C=C, gamma=g, epsilon=0.005))
    for d in (3, 5, 8, None):
        c["DecisionTree(d=%s)" % d] = lambda d=d: DecisionTreeRegressor(max_depth=d,
                                                                       random_state=42)
    for mf in (1.0, 0.5):
        for leaf in (1, 2, 4):
            c["RandomForest(mf=%g,leaf=%d)" % (mf, leaf)] = lambda mf=mf, leaf=leaf: \
                RandomForestRegressor(n_estimators=NTREE, max_features=mf,
                                      min_samples_leaf=leaf, random_state=42, n_jobs=1)
            c["ExtraTrees(mf=%g,leaf=%d)" % (mf, leaf)] = lambda mf=mf, leaf=leaf: \
                ExtraTreesRegressor(n_estimators=NTREE, max_features=mf,
                                    min_samples_leaf=leaf, random_state=42, n_jobs=1)
    for lr in (0.05, 0.1):
        for d in (2, 3, 5):
            c["GradBoost(lr=%g,d=%d)" % (lr, d)] = lambda lr=lr, d=d: \
                GradientBoostingRegressor(n_estimators=NTREE, learning_rate=lr,
                                          max_depth=d, random_state=42)
            c["XGB(lr=%g,d=%d)" % (lr, d)] = lambda lr=lr, d=d: \
                XGBRegressor(n_estimators=NTREE, learning_rate=lr, max_depth=d,
                             random_state=42, verbosity=0, n_jobs=1)
    for lr in (0.05, 0.1):
        for leaves in (7, 31):
            c["HistGB(lr=%g,l=%d)" % (lr, leaves)] = lambda lr=lr, leaves=leaves: \
                HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves,
                                              max_iter=NTREE, random_state=42)
    c["GaussianProcess"] = lambda: make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3),
                                 normalize_y=True, alpha=1e-8, random_state=42))
    return c


CAND = candidates()
print("candidate set size: %d" % len(CAND), flush=True)


def oof(X, factory, splits):
    pred = np.full(N, np.nan)
    for tr, te in splits:
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def flat_search(X, splits, label):
    t0 = time.time()
    rows = []
    for name, fac in CAND.items():
        p = oof(X, fac, splits)
        rows.append((name, r2_score(y, p), mean_squared_error(y, p)))
    rows.sort(key=lambda r: -r[1])
    print("  flat search done in %.0fs; top 10 of %d candidates:" % (time.time() - t0, len(rows)),
          flush=True)
    for name, r2, m in rows[:10]:
        print("    %-28s held-out R2 = %.4f   MSE = %.6f" % (name, r2, m), flush=True)
    return rows


def nested(X, outer_splits):
    t0 = time.time()
    pred = np.full(N, np.nan)
    picked = []
    for oi, (tr, te) in enumerate(outer_splits):
        inner = list(KFold(n_splits=3, shuffle=True, random_state=7).split(tr))
        best, best_mse = None, np.inf
        for name, fac in CAND.items():
            ip = np.full(len(tr), np.nan)
            for itr, ite in inner:
                m = fac()
                m.fit(X.iloc[tr[itr]], y[tr[itr]])
                ip[ite] = m.predict(X.iloc[tr[ite]])
            mse = mean_squared_error(y[tr], ip)
            if mse < best_mse:
                best, best_mse = name, mse
        m = CAND[best]()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
        picked.append(best)
        print("    outer fold %d picked %-28s (%.0fs elapsed)" % (oi, best, time.time() - t0),
              flush=True)
    return r2_score(y, pred), mean_squared_error(y, pred), picked


FEATURE_SETS = {
    "descriptors only (6)": DESCRIPTORS,
    "descriptors + calibration (7)": DESCRIPTORS + CALIB,
}
SPLITS = {
    "released row-shuffled split (random_state=40)":
        list(KFold(n_splits=5, shuffle=True, random_state=40).split(df)),
    "DIAGNOSTIC grouped-by-circuit split (NOT the released recipe)":
        list(GroupKFold(n_splits=5).split(df, y, groups)),
}

out = {}
for split_name, splits in SPLITS.items():
    print("=" * 96, flush=True)
    print("SATURATION under %s" % split_name, flush=True)
    print("=" * 96, flush=True)
    for fs_name, feats in FEATURE_SETS.items():
        X = df[feats]
        print(); print("-" * 96); print("FEATURE SET: %s" % fs_name); print("-" * 96, flush=True)
        rows = flat_search(X, splits, fs_name)
        bname, br2, bmse = rows[0]
        nr2, nmse, picked = nested(X, splits)
        print("  FLAT   best %-26s held-out R2 = %.4f" % (bname, br2), flush=True)
        print("  NESTED (selection paid for)          held-out R2 = %.4f" % nr2, flush=True)
        out["%s | %s" % (split_name, fs_name)] = {
            "flat_best_model": bname, "flat_R2": float(br2), "flat_MSE": float(bmse),
            "nested_R2": float(nr2), "nested_MSE": float(nmse), "nested_picked": picked,
            "flat_top10": [[n, float(r), float(m)] for n, r, m in rows[:10]],
        }
    print(flush=True)

with open(os.path.join(OUTDIR, "RESULT-saturation.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote RESULT-saturation.json", flush=True)
