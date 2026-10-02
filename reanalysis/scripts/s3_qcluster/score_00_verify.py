"""Phase 2 score, step 0: independent verification of everything Phase 2 relies on.

Nothing is fitted here. This re-derives, from the released bytes, the facts the
Phase 1 record asserts and that the scoring script will take as given:
  - the training table's shape, schema, hashes and grid structure
  - the notebook's own recipe (learner, hyperparameters, split, X and y slices)
  - the notebook's own selection rule for arm C (the 7-regressor bake-off)
  - which column is execution-derived and which is calibration
  - leak checks 1-4, re-run rather than inherited
A failure here aborts before any arm exists.
"""
import os
import sys
import re
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

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
OUTPUTS_DIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("QCLUSTER_OUTPUTS", os.path.join(ROOT, "../../outputs/s3_qcluster"))
TRAIN = os.path.join(REPO, "data", "training_20250307_2151.pkl")
MASTER = os.path.join(REPO, "data", "master_dict_ibm_brussels_new_20250307_2151.pkl")
NB = os.path.join(REPO, "src", "demo.ipynb")

FAILURES = []


def check(label, ok, detail=""):
    ok = bool(ok)
    if not ok:
        FAILURES.append(label)
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", label, detail))
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


print("=" * 96)
print("A. RELEASED BYTES")
print("=" * 96)
check("training table sha256 matches the Phase 1 record",
      sha256(TRAIN) == "163747ce5d43f1433dcd641d771b47fe5bd94e660640e9855cd66b1b773a1eb8",
      sha256(TRAIN)[:16] + "...")
check("master_dict sha256 matches the Phase 1 record",
      sha256(MASTER) == "cb2a5ac5c0ff33d8723f17f5341e584df601aeb1ac03f45ed1cf5a068b719712",
      sha256(MASTER)[:16] + "...")

print()
print("=" * 96)
print("B. THE NOTEBOOK'S OWN RECIPE, READ OUT OF THE RELEASED .ipynb")
print("=" * 96)
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)
srcs = ["".join(c.get("source", [])) for c in nb["cells"]]
allsrc = "\n\n".join(srcs)

for i, s in enumerate(srcs):
    if ("KFold" in s) or ("ExtraTrees" in s) or ("iloc[:, 2:10]" in s) or ("iloc[:,2:10]" in s):
        print("-" * 96)
        print("CELL %d (%s):" % (i, nb["cells"][i]["cell_type"]))
        print(s.strip()[:3000])
        print()

print("-" * 96)
# column names as the notebook constructs them
m = re.search(r"columns\s*=\s*\[([^\]]*)\]", allsrc)
nb_cols = None
if m:
    nb_cols = [t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()]
print("  columns= list found in the notebook: %s" % nb_cols)

check("notebook slices X as df.iloc[:, 2:10]",
      ("iloc[:, 2:10]" in allsrc) or ("iloc[:,2:10]" in allsrc))
check("notebook slices y as df.iloc[:, 11]",
      ("iloc[:, 11]" in allsrc) or ("iloc[:,11]" in allsrc))
check("notebook learner is ExtraTreesRegressor(n_estimators=100, random_state=42)",
      re.search(r"ExtraTreesRegressor\(\s*n_estimators\s*=\s*100\s*,\s*random_state\s*=\s*42\s*\)", allsrc) is not None)
check("notebook split is KFold(n_splits=5, shuffle=True, random_state=40)",
      re.search(r"KFold\(\s*n_splits\s*=\s*5\s*,\s*shuffle\s*=\s*True\s*,\s*random_state\s*=\s*40\s*\)", allsrc) is not None)
check("notebook also uses random_state=42 for KFold somewhere (cell 9)",
      re.search(r"KFold\(\s*n_splits\s*=\s*5\s*,\s*shuffle\s*=\s*True\s*,\s*random_state\s*=\s*42\s*\)", allsrc) is not None)
check("notebook calls df.dropna()", "dropna()" in allsrc)

# the released selection rule: the 7-regressor bake-off
bakeoff = ["RandomForestRegressor", "GradientBoostingRegressor", "ExtraTreesRegressor",
           "DecisionTreeRegressor", "SVR", "KNeighborsRegressor", "XGBRegressor"]
present = [b for b in bakeoff if b in allsrc]
check("released selection rule is a 7-regressor bake-off", len(present) == 7, str(present))
check("bake-off is scored by mean 5-fold MSE and R2",
      ("mean_squared_error" in allsrc) and ("r2_score" in allsrc))

print()
print("=" * 96)
print("C. THE TABLE: SHAPE, SCHEMA, GRID (leak check 3, re-run)")
print("=" * 96)
COLUMNS = ['Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
           'entropy', 'esp_m', 'ip Entropy', 'Target']
with open(TRAIN, "rb") as f:
    raw = pickle.load(f)
check("raw pickle is a list of 145 rows", isinstance(raw, list) and len(raw) == 145, "len=%d" % len(raw))
check("every raw row has 12 fields", all(len(r) == 12 for r in raw))
df = pd.DataFrame(raw, columns=COLUMNS)
n_before = len(df)
df = df.dropna().reset_index(drop=True)
check("df.dropna() removes nothing (row order preserved)", len(df) == n_before,
      "%d -> %d" % (n_before, len(df)))
check("frame is 145 x 12", df.shape == (145, 12), str(df.shape))
check("29 algorithms x 5 machines, complete and unduplicated",
      df['Algorithm'].nunique() == 29 and df['Machine'].nunique() == 5
      and (df.groupby(['Algorithm', 'Machine']).size() == 1).all()
      and len(df) == 145)
check("no full-row duplicates", not df.duplicated().any())
if nb_cols is not None:
    check("column labels used here match the notebook's own list", nb_cols == COLUMNS, str(nb_cols))

FEATS_RELEASED = list(df.columns[2:10])
TARGET = df.columns[11]
DESCRIPTORS = ['Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr']
EXEC_DERIVED = ['entropy']
CALIB = ['esp_m']
check("released X columns are the 8 the plan names",
      FEATS_RELEASED == DESCRIPTORS + EXEC_DERIVED + CALIB, str(FEATS_RELEASED))
check("y column is 'Target'", TARGET == 'Target')

print()
print("=" * 96)
print("D. COLUMN IDENTITY AGAINST master_dict: WHICH COLUMN IS EXECUTION-DERIVED")
print("=" * 96)


class _Stub(object):
    """Placeholder for a class this environment cannot import (release pins qiskit 1.2.4)."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self._state = state

    def append(self, *a, **k):
        self.__dict__.setdefault("_items", []).append(a)

    def extend(self, *a, **k):
        self.__dict__.setdefault("_items", []).extend(a)

    def add(self, *a, **k):
        self.__dict__.setdefault("_items", []).append(a)

    def __setitem__(self, k, v):
        self.__dict__.setdefault("_map", {})[repr(k)] = v

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)

        def _noop(*a, **k):
            return None

        return _noop


_STUBS = {}


def _make_stub(module, name):
    key = (module, name)
    if key not in _STUBS:
        _STUBS[key] = type("Stub_%s" % name, (_Stub,), {"_qualified": "%s.%s" % (module, name)})
    return _STUBS[key]


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".")[0]
        if root in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_ibm_provider",
                    "rustworkx", "retworkx", "symengine"):
            return _make_stub(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _make_stub(module, name)


with open(MASTER, "rb") as f:
    master = _StubUnpickler(f).load()
check("master_dict has 6 algorithms", len(master) == 6, str(sorted(master.keys())))


def h_norm(dist):
    v = np.array([dist[k] for k in dist], dtype=float)
    p = v / v.sum()
    p = p[p > 0]
    n_bits = len(list(dist.keys())[0])
    return float(-(p * np.log2(p)).sum() / n_bits)


bru = df[df['Machine'] == 'ibm_brussels'].set_index('Algorithm')
d_ent, d_ideal, d_esp = [], [], []
for alg in sorted(master.keys()):
    d_ent.append(abs(bru.loc[alg, 'entropy'] - h_norm(master[alg]['nsm'])))
    d_ideal.append(abs(bru.loc[alg, 'ip Entropy'] - h_norm(master[alg]['ideal'])))
    d_esp.append(abs(bru.loc[alg, 'esp_m'] - float(master[alg]['esp_m'])))
check("df['entropy'] == normalized Shannon entropy of the NOISY output (execution-derived)",
      max(d_ent) < 1e-9, "max |diff| = %.3g" % max(d_ent))
check("df['ip Entropy'] == normalized Shannon entropy of the IDEAL output (oracle)",
      max(d_ideal) < 1e-9, "max |diff| = %.3g" % max(d_ideal))
check("df['esp_m'] == master_dict esp_m (calibration)", max(d_esp) < 1e-12,
      "max |diff| = %.3g" % max(d_esp))

with open(os.path.join(REPO, "src", "qcluster.py"), "r", encoding="utf-8", errors="replace") as f:
    qc = f.read()
i = qc.find("def get_esp_modified_therm")
j = qc.find("\ndef ", i + 10)
body = qc[i:j] if i >= 0 else ""
reads = {k: body.count(k) for k in ("gate_error", "gate_length", "t1", "readout_error",
                                    "circuit_to_dag")}
never = {k: body.count(k) for k in ("nsm", "counts", "ideal", "Target", "execute", "run(")}
check("get_esp_modified_therm reads backend calibration",
      all(v > 0 for v in reads.values()), str(reads))
check("get_esp_modified_therm reads NO execution result (calibration, not execution-derived)",
      all(v == 0 for v in never.values()), str(never))
print("    -> esp_m = product over DAG nodes of per-gate reliabilities built from")
print("       backend gate_error, gate_length, t1 and readout_error; a circuit x calibration")
print("       product that cannot be split into a pure-calibration and a pure-circuit part,")
print("       so the calibration-withheld ladder must drop the column entirely.")

print()
print("=" * 96)
print("E. LEAK CHECK 1: TARGET NOT IN ANY ARM'S FEATURES, UNDER ANY NAME OR TRANSFORM")
print("=" * 96)
y = df[TARGET].values.astype(float)
ARM_FEATURES = {
    "A/C, calibration available": DESCRIPTORS + CALIB,
    "A/C, calibration withheld": DESCRIPTORS,
    "F (released feature set)": FEATS_RELEASED,
}
for arm, fs in ARM_FEATURES.items():
    check("'Target' absent from %s" % arm, TARGET not in fs)
    check("'ip Entropy' absent from %s" % arm, 'ip Entropy' not in fs)
    check("'Machine'/'Algorithm' identifiers absent from %s" % arm,
          ('Machine' not in fs) and ('Algorithm' not in fs))

print()
worst_sp = 0.0
print("  %-10s %10s %10s %10s %10s" % ("feature", "pearson", "spearman", "|rank rho|", "exact dup"))
for f in FEATS_RELEASED:
    v = df[f].values.astype(float)
    pr = stats.pearsonr(v, y)[0]
    sr = stats.spearmanr(v, y)[0]
    worst_sp = max(worst_sp, abs(sr))
    print("  %-10s %10.4f %10.4f %10.4f %10s" % (f, pr, sr, abs(sr), np.allclose(v, y)))
check("no fed feature is a monotone transform of Target", worst_sp < 0.999,
      "max |spearman| = %.4f" % worst_sp)

probes = {
    "1 - esp_m": 1.0 - df['esp_m'].values,
    "1 - esp_m**(1/Measure)": 1.0 - df['esp_m'].values ** (1.0 / df['Measure'].values),
    "1 - esp_m**(1/Qubit)": 1.0 - df['esp_m'].values ** (1.0 / df['Qubit'].values),
    "-log(esp_m)/Measure": -np.log(df['esp_m'].values) / df['Measure'].values,
    "entropy - ip Entropy": df['entropy'].values - df['ip Entropy'].values,
    "(entropy - ip Entropy)/2": (df['entropy'].values - df['ip Entropy'].values) / 2.0,
}
mads = []
for name, v in probes.items():
    fin = np.isfinite(v)
    mad = float(np.max(np.abs(v[fin] - y[fin])))
    mads.append(mad)
    print("    probe %-26s max|probe - y| = %.4g" % (name, mad))
check("no closed-form probe reproduces Target", min(mads) > 1e-6, "min max-error = %.4g" % min(mads))

print()
print("=" * 96)
print("F. LEAK CHECK 2: NO ARM READS AN ORACLE")
print("=" * 96)
check("the fit_x comparator (ideal support size) is not in any arm",
      "fit_x" not in " ".join(ARM_FEATURES.keys()), "downstream comparator, excluded by construction")
check("the ip-Entropy case filter is not applied and 'ip Entropy' is fed to no arm",
      all('ip Entropy' not in fs for fs in ARM_FEATURES.values()),
      "all 145 rows scored; no ideal-entropy filtering")
print("    for the record, the released filter 'ip Entropy > 0.6' would drop %d of 145 rows;"
      % int((df['ip Entropy'] > 0.6).sum()))
print("    Phase 2 does not apply it, so no oracle touches the row set either.")

print()
print("=" * 96)
print("G. LEAK CHECK 4: ROW-SHUFFLED SPLIT (RECORD AND REPORT, NOT A REFUSAL)")
print("=" * 96)
phase1_path = os.path.join(OUTPUTS_DIR, "PHASE1-RECORD.json")
if not os.path.exists(phase1_path):
    phase1_path = os.path.join(ROOT, "PHASE1-RECORD.json")
with open(phase1_path) as f:
    rec = json.load(f)
for rs in (40, 42):
    kf = KFold(n_splits=5, shuffle=True, random_state=rs)
    fold_of = np.empty(len(df), dtype=int)
    for i, (tr, te) in enumerate(kf.split(df)):
        fold_of[te] = i
    stored = np.array(rec["split_structure"]["fold_assignment"][str(rs)])
    check("fold assignment for random_state=%d reproduces the Phase 1 record" % rs,
          np.array_equal(fold_of, stored))
    tmp = df.copy()
    tmp['fold'] = fold_of
    nstr = int((tmp.groupby('Algorithm')['fold'].nunique() > 1).sum())
    print("     random_state=%d: %d of 29 algorithms straddle folds; fold sizes %s"
          % (rs, nstr, np.bincount(fold_of).tolist()))

alg_mean = df.groupby('Algorithm')[TARGET].transform('mean')
mac_mean = df.groupby('Machine')[TARGET].transform('mean')
r2_alg = 1.0 - ((y - alg_mean) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2_mac = 1.0 - ((y - mac_mean) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("     share of Target variance explained by algorithm identity: %.4f" % r2_alg)
print("     share of Target variance explained by machine identity:   %.4f" % r2_mac)
check("leak check 4 severity reproduces the Phase 1 record",
      abs(r2_alg - rec["split_structure"]["leak_check_4_severity"]
          ["target_variance_explained_by_Algorithm_means"]) < 1e-9
      and abs(r2_mac - rec["split_structure"]["leak_check_4_severity"]
              ["target_variance_explained_by_Machine_means"]) < 1e-9)

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)
if FAILURES:
    print("  %d CHECK(S) FAILED -> ABORT BEFORE FITTING:" % len(FAILURES))
    for f in FAILURES:
        print("    - " + f)
    raise SystemExit(2)
print("  all checks pass; leak checks 1, 2 and 3 PASS; leak check 4 RECORDED AND REPORTED.")
print("  the release is verified and Phase 2 may fit.")
