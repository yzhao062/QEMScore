"""Phase 2 score, step 8: a second, wider interval on the SAME fitted arms.

The plan declares the resample unit to be the system's own evaluation unit, and
for Q-Cluster that unit is a row of the released 145-row table. That is what the
pre-registered interval resamples, and that interval stands as the score.

But the 145 rows are 29 circuits times 5 machines, and 94.2 percent of target
variance lies between circuits. Rows of one circuit are therefore not independent
draws, and a row bootstrap will read as if the sample carried more information
than it does. This script re-resamples the identical out-of-fold predictions at
the circuit level, so the reader can see how much of the pre-registered interval's
width comes from treating five rows of a circuit as five independent rows.

This is a diagnostic on the declared interval, not a replacement for it. The
reading rule is applied to the pre-registered row-level interval; the circuit
level interval is reported beside it in the same sentence.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

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
    y = df['Target'].values.astype(float)
    algorithms = df['Algorithm'].values
elif npz_path and os.path.exists(npz_path):
    z = np.load(npz_path)
    algorithms = z["circuit"]
    y = z["numeric"][:, 9].astype(float)
else:
    raise FileNotFoundError("Could not find training_20250307_2151.pkl or S3_target_table.npz")
N = len(y)

oof_candidates = [
    os.path.join(INPUTS_DIR, "oof_predictions.npy"),
    os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "s3_qcluster", "oof_predictions.npy")),
    os.path.join(ROOT, "oof_predictions.npy"),
]
oof_path = next((p for p in oof_candidates if os.path.exists(p)), None)
if not oof_path:
    raise FileNotFoundError("Could not find oof_predictions.npy")
pred = np.load(oof_path)

meta_candidates = [
    os.path.join(INPUTS_DIR, "arm_meta.json"),
    os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "s3_qcluster", "arm_meta.json")),
    os.path.join(ROOT, "arm_meta.json"),
]
meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)
if not meta_path:
    raise FileNotFoundError("Could not find arm_meta.json")
meta = json.load(open(meta_path))
order = [tuple(o) for o in meta["order"]]

alg_ids, alg_names = pd.factorize(algorithms)
n_alg = len(alg_names)
clusters = [np.where(alg_ids == a)[0] for a in range(n_alg)]
print("circuits: %d, rows per circuit: %s" % (n_alg, sorted(set(len(c) for c in clusters))))

N_BOOT = 10000
rng = np.random.default_rng(20260904 + 2)
CB = np.array([np.concatenate([clusters[a] for a in rng.integers(0, n_alg, n_alg)])
               for _ in range(N_BOOT)])
print("circuit-level bootstrap: %d draws of %d circuits, %d rows per draw"
      % (N_BOOT, n_alg, CB.shape[1]))

out = {}
print()
print("%-34s %8s %20s %20s" % ("panel", "S", "row bootstrap (score)", "circuit bootstrap"))
for i, (rs, ladder) in enumerate(order):
    pA, pC, pF = pred[i]
    eA, eC, eF = (y - pA) ** 2, (y - pC) ** 2, (y - pF) ** 2
    S0 = (eA.mean() - eC.mean()) / (eA.mean() - eF.mean())

    rng_row = np.random.default_rng(20260904)
    RB = rng_row.integers(0, N, size=(N_BOOT, N))
    Sr = (eA[RB].mean(1) - eC[RB].mean(1)) / (eA[RB].mean(1) - eF[RB].mean(1))
    rlo, rhi = np.percentile(Sr[np.isfinite(Sr)], [2.5, 97.5])

    Sc = (eA[CB].mean(1) - eC[CB].mean(1)) / (eA[CB].mean(1) - eF[CB].mean(1))
    Tc = eA[CB].mean(1) - eF[CB].mean(1)
    clo, chi = np.percentile(Sc[np.isfinite(Sc)], [2.5, 97.5])

    read_r = "PRESENT" if rlo > 0.5 else "ABSENT" if chi < 0.5 else "INDETERMINATE"
    read_r = "PRESENT" if rlo > 0.5 else ("ABSENT" if rhi < 0.5 else "INDETERMINATE")
    read_c = "PRESENT" if clo > 0.5 else ("ABSENT" if chi < 0.5 else "INDETERMINATE")
    lbl = "rs=%d, %s" % (rs, ladder)
    print("%-34s %8.4f  [%.4f, %.4f] %s  [%.4f, %.4f] %s"
          % (lbl, S0, rlo, rhi, read_r, clo, chi, read_c))
    out[lbl] = dict(S=float(S0), row_lo=float(rlo), row_hi=float(rhi), row_reading=read_r,
                    circuit_lo=float(clo), circuit_hi=float(chi), circuit_reading=read_c,
                    circuit_frac_T_nonpos=float(np.mean(Tc <= 0)),
                    width_row=float(rhi - rlo), width_circuit=float(chi - clo))

print()
print("interval width, row versus circuit resampling:")
for lbl, v in out.items():
    print("  %-34s row %.4f   circuit %.4f   ratio %.2fx"
          % (lbl, v["width_row"], v["width_circuit"], v["width_circuit"] / v["width_row"]))
print()
print("fraction of circuit-level draws with T <= 0:")
for lbl, v in out.items():
    print("  %-34s %.3f%%" % (lbl, 100 * v["circuit_frac_T_nonpos"]))

with open(os.path.join(OUTDIR, "RESULT-cluster-bootstrap.json"), "w") as f:
    json.dump(out, f, indent=2)
print()
print("wrote RESULT-cluster-bootstrap.json")
