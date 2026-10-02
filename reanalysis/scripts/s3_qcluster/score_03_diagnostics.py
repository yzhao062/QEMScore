"""Phase 2 score, step 3: diagnostics beside the score. None of this is the score.

Three things, each labeled where it sits relative to the frozen plan:

1. SELECTION-RULE ROBUSTNESS. Under random_state=42 the released selection rule
   picks XGBoost rather than ExtraTrees, while released cell 9 hard-codes
   ExtraTrees under that seed. Both are run so the reading does not rest on
   which of the two the rule lands on. This is still the released recipe.

2. LEAK CHECK 4, QUANTIFIED. The released split is shuffled by row, which the
   plan records and reports because it favours the descriptor arms. This refits
   every arm under a grouped-by-circuit split to measure how much of the reading
   that favouring accounts for. It is NOT the released recipe, it is NOT the
   score, and the reading rule is NOT applied to it.

3. DOWNSTREAM CORROBORATION on the 6 rows master_dict covers. Six rows cannot
   carry a pointwise interval and none is computed. The released loop's two
   oracle devices, the fit_x comparator that receives the ideal support size and
   the ip-Entropy case filter, are both excluded here.
"""
import os
import sys
import json
import copy
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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

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

REPO = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("QCLUSTER_OUTPUTS", os.path.join(ROOT, "../../outputs/s3_qcluster"))
TRAIN = os.path.join(REPO, "data", "training_20250307_2151.pkl") if os.path.isdir(REPO) else REPO
MASTER = os.path.join(REPO, "data", "master_dict_ibm_brussels_new_20250307_2151.pkl") if os.path.isdir(REPO) else os.path.join(os.path.dirname(REPO), "master_dict_ibm_brussels_new_20250307_2151.pkl")
os.makedirs(OUTDIR, exist_ok=True)

COLUMNS = ['Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
           'entropy', 'esp_m', 'ip Entropy', 'Target']
DESCRIPTORS = ['Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr']
CALIB = ['esp_m']
RELEASED_X = DESCRIPTORS + ['entropy'] + CALIB
LADDERS = {"with_calibration": DESCRIPTORS + CALIB, "without_calibration": DESCRIPTORS}

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
N_BOOT = 10000
SEED = 20260904


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


def oof(X, factory, splits):
    pred = np.full(N, np.nan)
    for tr, te in splits:
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    assert not np.isnan(pred).any()
    return pred


def select(X, splits):
    scores = {}
    preds = {}
    for name in bakeoff():
        p = oof(X, lambda n=name: bakeoff()[n], splits)
        preds[name] = p
        scores[name] = np.mean([mean_squared_error(y[te], p[te]) for _, te in splits])
    w = min(scores, key=scores.get)
    return w, preds, scores


def estimand(pA, pC, pF, boot_idx, label, apply_reading_rule):
    eA, eC, eF = (y - pA) ** 2, (y - pC) ** 2, (y - pF) ** 2
    A0, C0, F0 = eA.mean(), eC.mean(), eF.mean()
    T0, K0, D0, S0 = A0 - F0, A0 - C0, C0 - F0, (A0 - C0) / (A0 - F0)
    Ab, Cb, Fb = eA[boot_idx].mean(1), eC[boot_idx].mean(1), eF[boot_idx].mean(1)
    Tb, Kb, Sb = Ab - Fb, Ab - Cb, (Ab - Cb) / (Ab - Fb)
    Sf = Sb[np.isfinite(Sb)]
    lo, hi = np.percentile(Sf, [2.5, 97.5])
    rd = ("PRESENT" if lo > 0.5 else "ABSENT" if hi < 0.5 else "INDETERMINATE")
    print("  %-52s A=%.6f C=%.6f F=%.6f" % (label, A0, C0, F0))
    print("  %-52s T=%.6f K=%.6f D=%.6f" % ("", T0, K0, D0))
    print("  %-52s S=%.4f  [%.4f, %.4f]  %s"
          % ("", S0, lo, hi, ("-> " + rd) if apply_reading_rule else "(reading rule NOT applied)"))
    print("  %-52s draws with T<=0: %.3f%%" % ("", 100 * np.mean(Tb <= 0)))
    return dict(A=float(A0), C=float(C0), F=float(F0), T=float(T0), K=float(K0),
                D=float(D0), S=float(S0), S_lo=float(lo), S_hi=float(hi),
                frac_T_nonpos=float(np.mean(Tb <= 0)),
                reading=rd if apply_reading_rule else "n/a (diagnostic)")


rng = np.random.default_rng(SEED)
ROW_BOOT = rng.integers(0, N, size=(N_BOOT, N))
out = {}

print("=" * 96)
print("1. SELECTION-RULE ROBUSTNESS UNDER random_state=42 (still the released recipe)")
print("=" * 96)
print("   The released rule under seed 42 selects XGBoost; released cell 9 hard-codes")
print("   ExtraTrees under that same seed. Both are reported.")
sp42 = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
pF42_et = oof(df[RELEASED_X], lambda: ExtraTreesRegressor(n_estimators=100, random_state=42), sp42)
for ladder, feats in LADDERS.items():
    pA = oof(df[feats], LinearRegression, sp42)
    pC = oof(df[feats], lambda: ExtraTreesRegressor(n_estimators=100, random_state=42), sp42)
    print()
    out["rs42_pinned_ExtraTrees|%s" % ladder] = estimand(
        pA, pC, pF42_et, ROW_BOOT, "rs=42, family pinned to ExtraTrees, %s" % ladder, True)

print()
print("=" * 96)
print("2. LEAK CHECK 4 QUANTIFIED: THE SAME ARMS UNDER A GROUPED-BY-CIRCUIT SPLIT")
print("   NOT the released recipe. NOT the score. The reading rule is not applied.")
print("=" * 96)
gsp = list(GroupKFold(n_splits=5).split(df, y, groups))
print("   GroupKFold(5) by Algorithm: fold sizes %s, algorithms per fold %s"
      % ([len(te) for _, te in gsp], [len(set(groups[te])) for _, te in gsp]))
wF_g, predsF_g, scoresF_g = select(df[RELEASED_X], gsp)
print("   arm F bake-off winner under the grouped split: %s" % wF_g)
alg_ids = pd.factorize(groups)[0]
n_alg = alg_ids.max() + 1
rng2 = np.random.default_rng(SEED + 1)
clusters = [np.where(alg_ids == a)[0] for a in range(n_alg)]
CLUST_BOOT = np.array([np.concatenate([clusters[a] for a in rng2.integers(0, n_alg, n_alg)])[:N]
                       for _ in range(2000)])
for ladder, feats in LADDERS.items():
    pA = oof(df[feats], LinearRegression, gsp)
    wC, predsC, _ = select(df[feats], gsp)
    print()
    print("   arm C bake-off winner under the grouped split, %s: %s" % (ladder, wC))
    out["grouped_diagnostic|%s" % ladder] = estimand(
        pA, predsC[wC], predsF_g[wF_g], ROW_BOOT,
        "GROUPED DIAGNOSTIC, row bootstrap, %s" % ladder, False)
    out["grouped_diagnostic_clusterboot|%s" % ladder] = estimand(
        pA, predsC[wC], predsF_g[wF_g], CLUST_BOOT,
        "GROUPED DIAGNOSTIC, circuit-cluster bootstrap, %s" % ladder, False)

print()
print("=" * 96)
print("3. DOWNSTREAM CORROBORATION ON THE 6 master_dict ROWS")
print("   Corroborating only. Six rows carry no interval and none is computed.")
print("   No fit_x comparator and no ip-Entropy filter: neither oracle is read.")
print("=" * 96)
sys.path.insert(0, os.path.join(REPO, "src"))
import qcluster  # noqa: E402


class _Stub(object):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self._state = s

    def append(self, *a, **k):
        self.__dict__.setdefault("_i", []).append(a)

    def extend(self, *a, **k):
        self.__dict__.setdefault("_i", []).extend(a)

    def add(self, *a, **k):
        self.__dict__.setdefault("_i", []).append(a)

    def __setitem__(self, k, v):
        self.__dict__.setdefault("_m", {})[repr(k)] = v

    def __getattr__(self, i):
        if i.startswith("__") and i.endswith("__"):
            raise AttributeError(i)
        return lambda *a, **k: None


_S = {}


def _mk(mod, nm):
    if (mod, nm) not in _S:
        _S[(mod, nm)] = type("Stub_" + nm, (_Stub,), {})
    return _S[(mod, nm)]


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime",
                                    "qiskit_ibm_provider", "rustworkx", "retworkx", "symengine"):
            return _mk(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _mk(module, name)


with open(MASTER, "rb") as f:
    master = _U(f).load()

# released cell 9's own configuration: seed-42 folds, ExtraTrees, arms give prob=p_hat
arm_pred = {"A_with_calib": oof(df[LADDERS["with_calibration"]], LinearRegression, sp42),
            "A_without_calib": oof(df[LADDERS["without_calibration"]], LinearRegression, sp42),
            "C_with_calib": oof(df[LADDERS["with_calibration"]],
                                lambda: ExtraTreesRegressor(n_estimators=100, random_state=42), sp42),
            "C_without_calib": oof(df[LADDERS["without_calibration"]],
                                   lambda: ExtraTreesRegressor(n_estimators=100, random_state=42), sp42),
            "F_released_features": pF42_et}
rows = df[(df['Machine'] == 'ibm_brussels') & (df['Algorithm'].isin(master.keys()))]
print("   %d rows, all on ibm_brussels: %s" % (len(rows), sorted(rows['Algorithm'])))
down = {}
print()
print("   %-15s %8s %9s %9s %9s %9s %9s %9s" % ("algorithm", "noisy", "true p",
                                                "A wcal", "A wocal", "C wcal", "C wocal", "F"))
for _, row in rows.iterrows():
    alg = row['Algorithm']
    d = master[alg]
    cn = qcluster.multiply_dict(copy.deepcopy(d['nsm']), d['shots'])
    idl = qcluster.renormalize(d['ideal'])
    noisy_fid = qcluster.hellinger_fidelity(idl, qcluster.renormalize(copy.deepcopy(d['nsm'])))
    fids = {}
    for arm, pred in arm_pred.items():
        p = float(pred[row.name])
        km = qcluster.QCluster(1024, row['Measure'], prob=p, init_method="top_k",
                               mul=2.0, tol=0.95)
        km.fit(copy.deepcopy(cn))
        fids[arm] = qcluster.hellinger_fidelity(idl, qcluster.renormalize(km.labels_))
    down[alg] = dict(noisy=float(noisy_fid), true_p=float(row['Target']),
                     p_hat={a: float(arm_pred[a][row.name]) for a in arm_pred},
                     fidelity=fids)
    print("   %-15s %8.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
          % (alg, noisy_fid, row['Target'], fids["A_with_calib"], fids["A_without_calib"],
             fids["C_with_calib"], fids["C_without_calib"], fids["F_released_features"]))
print()
print("   mean over the 6 rows:")
for arm in arm_pred:
    print("     %-22s %.4f" % (arm, np.mean([down[a]['fidelity'][arm] for a in down])))
print("     %-22s %.4f" % ("noisy baseline", np.mean([down[a]['noisy'] for a in down])))
print()
print("   released cell-9 'Q-Cluster exact' column, for comparison with F:")
rel = {"ghz_state_n11": 0.707513, "wstate_n3": 0.889651, "bv_n14": 0.080205,
       "bv_n19": 0.067193, "wstate_n27": 0.069104, "cat_state_n22": 0.126763}
for a in sorted(rel):
    print("     %-15s released %.6f   F here %.6f" % (a, rel[a], down[a]['fidelity']["F_released_features"]))
out["downstream_6_rows"] = down

with open(os.path.join(OUTDIR, "RESULT-diagnostics.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote RESULT-diagnostics.json")
