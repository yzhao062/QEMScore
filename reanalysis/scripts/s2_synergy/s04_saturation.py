"""
S2 phase 2, step 4: DESCRIPTOR-SATURATION DIAGNOSTIC.

PLAN: "Fit the strongest descriptor-only predictor the released data supports,
without a capacity match and without a measurement input, and report the share
of target variance it explains on held-out rows."

Two numbers per panel:
  learned   -- best held-out R2 over a sweep of strong regressors on theta alone,
               under the SAME 10-fold split used by the arms.
  analytic  -- the released generator circ_gen.py evaluated on theta alone,
               with no measurement input and no training at all.

No measurement input anywhere in this file: z_noisy, mz_zne, mz_linear and
mz_predicted are never read.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REL = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "upstream", "s2_synergy"))
DEFAULT_OUT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs", "s2_synergy"))

REL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REL
OUT = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
os.makedirs(OUT, exist_ok=True)
FOLD_SEED = 12527150
N_FOLDS = 10
PANELS = [("Fig8a", "A"), ("Fig8b", "B"), ("Fig8c", "B")]
ALPHAS = np.logspace(-8, 4, 25)


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def models():
    """Strong descriptor-only regressors. No capacity match, no measurement input."""
    inner = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    return {
        "ols": LinearRegression(),
        "ridge_cv": GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                                 {"ridge__alpha": ALPHAS}, cv=inner, n_jobs=4),
        "poly2_ridge_cv": GridSearchCV(
            make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge()),
            {"ridge__alpha": ALPHAS}, cv=inner, n_jobs=4),
        "poly3_ridge_cv": GridSearchCV(
            make_pipeline(PolynomialFeatures(3, include_bias=False), StandardScaler(), Ridge()),
            {"ridge__alpha": ALPHAS}, cv=inner, n_jobs=4),
        "kernel_ridge_rbf": GridSearchCV(
            make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
            {"kernelridge__alpha": np.logspace(-6, 2, 17),
             "kernelridge__gamma": np.logspace(-4, 1, 16)}, cv=inner, n_jobs=4),
        "svr_rbf": GridSearchCV(
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            {"svr__C": np.logspace(-1, 4, 11), "svr__gamma": np.logspace(-4, 0, 9),
             "svr__epsilon": [0.001, 0.01]}, cv=inner, n_jobs=4),
        "gpr_rbf": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=np.ones(1) * 3.0)
                + WhiteKernel(1e-3),
                normalize_y=True, n_restarts_optimizer=3, random_state=FOLD_SEED)),
        "random_forest": RandomForestRegressor(n_estimators=500, random_state=FOLD_SEED, n_jobs=4),
        "extra_trees": ExtraTreesRegressor(n_estimators=500, random_state=FOLD_SEED, n_jobs=4),
        "grad_boost": GradientBoostingRegressor(random_state=FOLD_SEED),
        "hist_grad_boost": HistGradientBoostingRegressor(random_state=FOLD_SEED),
        "knn_cv": GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                               {"kneighborsregressor__n_neighbors": [1, 2, 3, 5, 8, 12, 20],
                                "kneighborsregressor__weights": ["uniform", "distance"]},
                               cv=inner, n_jobs=4),
        "mlp_cv": GridSearchCV(
            make_pipeline(StandardScaler(),
                          MLPRegressor(max_iter=8000, random_state=FOLD_SEED, early_stopping=False)),
            {"mlpregressor__hidden_layer_sizes": [(64,), (256,), (256, 256), (512, 512)],
             "mlpregressor__alpha": [1e-5, 1e-3, 1e-1]}, cv=inner, n_jobs=4),
    }


# ---- analytic descriptor-only predictor: the released generator -------------
ASSOC = [0, 1, 2, 3, 5, 8, 9, 11, 14, 13, 12, 15, 10, 7, 6, 4]
N, P = 16, 20
EDGES = []
for i in range(N - 1):
    if ASSOC[i] in (8, 12, 7):
        EDGES.append((i, i + 1))
        if i + 2 < N:
            EDGES.append((i, i + 2))
    elif ASSOC[i] not in (9, 15, 6):
        EDGES.append((i, i + 1))
DIM = 1 << N
_bits = ((np.arange(DIM)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
SGN = 1 - 2 * _bits
_zz = np.zeros(DIM)
for a, b in EDGES:
    _zz += SGN[:, a] * SGN[:, b]
RZZ = np.exp(-1j * (-np.pi / 2) / 2.0 * _zz)


def analytic_mz(theta_row, config):
    psi = np.zeros(DIM, dtype=np.complex128)
    psi[0] = 1.0
    for j in range(P):
        ang = theta_row if config == "A" else np.full(N, theta_row[j])
        for k in range(N):
            c, s = np.cos(ang[k] / 2.0), -1j * np.sin(ang[k] / 2.0)
            v = psi.reshape(-1, 2, 1 << k)
            a0 = v[:, 0, :].copy(); a1 = v[:, 1, :].copy()
            v[:, 0, :] = c * a0 + s * a1
            v[:, 1, :] = s * a0 + c * a1
            psi = v.reshape(-1)
        psi = psi * RZZ
    prob = np.abs(psi) ** 2
    return float(np.sum(prob[:, None] * SGN, axis=0).mean())


out = {}
for panel, cfg in PANELS:
    theta = np.load(os.path.join(REL, f"{panel}_theta.npy"))
    y = np.load(os.path.join(REL, f"{panel}_mz_exact.npy"))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(kf.split(theta))
    scores = {}
    for name in models():
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            m = models()[name]      # a fresh, unfitted estimator per fold
            m.fit(theta[tr], y[tr])
            oof[te] = m.predict(theta[te]).ravel()
        scores[name] = r2(y, oof)
    ana = np.array([analytic_mz(theta[i], cfg) for i in range(len(y))])
    best = max(scores, key=scores.get)
    out[panel] = {
        "config": cfg,
        "n": int(len(y)),
        "learned_family_scores": scores,
        "learned_saturation_best": scores[best],
        "learned_saturation_best_family": best,
        "analytic_saturation": r2(y, ana),
        "analytic_max_abs_err": float(np.max(np.abs(ana - y))),
    }
    print(f"\n{panel} (config {cfg}) descriptor-only saturation, theta alone, "
          f"10-fold held-out R2:")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {v:+.4f}")
    print(f"    -> LEARNED saturation  = {scores[best]:+.4f}  ({best})")
    print(f"    -> ANALYTIC saturation = {out[panel]['analytic_saturation']:.12f} "
          f"(released generator on theta, max|err| {out[panel]['analytic_max_abs_err']:.2e})")

fp = os.path.join(OUT, "s04_saturation.json")
with open(fp, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {fp}")
