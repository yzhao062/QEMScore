"""
S2 phase 2, step 3: FIT THE ARMS under the matched recipe.

Everything in this file was declared in PREREG-phase2.json before it ran.
Produces out-of-fold predictions only; no estimand is formed here.

usage:  python s03_fit_arms.py <Fig8a|Fig8b|Fig8c>

Arms:
  A                 OLS on [1, theta]                      (qubits dropped: zero variance)
  C                 released architecture, synergy=False    channels (q', theta)
  F_refit_matched   released architecture, synergy=True     channels (q', theta, z_noisy)
  F_released        Fig8*_mz_predicted.npy                  (not fitted here)
"""
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "10")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REL = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "upstream", "s2_synergy"))
DEFAULT_OUT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs", "s2_synergy"))

PANEL = sys.argv[1] if len(sys.argv) > 1 else "Fig8a"
REL = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REL
OUT = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REL)

assert PANEL in ("Fig8a", "Fig8b", "Fig8c")
CONFIG = "A" if PANEL == "Fig8a" else "B"

# ---- declared constants (PREREG-phase2.json) --------------------------------
FOLD_SEED = 12527150
N_FOLDS = 10
N_REPEATS = 3
MAX_EPOCHS = 400
BATCH = 8
PATIENCE = 60
VAL_FRAC = 0.20

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(2)

# the released r2() uses the Keras-2 backend API; substitute a Keras-3 equivalent.
# It is a REPORTING metric only: not the loss, not the selection rule.
import keras
import keras.backend as _K
from keras import ops as _ops
for _n in ("sum", "square", "mean"):
    if not hasattr(_K, _n):
        setattr(_K, _n, getattr(_ops, _n))
from cnn_2D import CNN as CNN2D, r2 as r2_metric          # released file, verbatim
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold


def CNN1D(synergy):
    """1D analogue of the released cnn_2D.CNN, for configuration A (paper: input (N,3))."""
    ch = 3 if synergy else 2
    m = Sequential()
    m.add(keras.Input(shape=(None, ch)))
    m.add(Conv1D(128, activation="relu", kernel_size=3, strides=1, padding="same"))
    m.add(Conv1D(128, activation="relu", kernel_size=3, strides=1, padding="same"))
    m.add(Conv1D(128, activation="relu", kernel_size=3, strides=1, padding="same"))
    m.add(GlobalAveragePooling1D())
    for _ in range(4):
        m.add(Dense(512, activation="relu"))
    m.add(Dense(1))
    m.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae", r2_metric])
    return m


def load(name):
    return np.load(os.path.join(REL, f"{PANEL}_{name}.npy"))


theta = load("theta").astype("float64")
qubits = load("qubits").astype("float64")
z_noisy = load("z_noisy").astype("float64")
y = load("mz_exact").astype("float64")
f_released = load("mz_predicted").astype("float64")
n, n_theta = theta.shape
N_QUBITS = qubits.shape[1]
P_LAYERS = n_theta if CONFIG == "B" else None
assert n == 100

qprime = qubits / 10.0        # the paper's own normalization q' = q/10


def build_channels(with_z, cal_available):
    """Stack the CNN input channels exactly as the paper describes.

    `cal_available` selects the plan's device-calibration group. For S2 that
    group is EMPTY in both settings, so the two constructions must be identical;
    the caller asserts it rather than assuming it.
    """
    cal_channels = []            # S2 ships no calibration array, in either setting
    if cal_available:
        cal_channels = []        # available -> the empty set
    if CONFIG == "A":
        # input shape (N, ch): channel 0 = q', 1 = theta(N), 2 = z_noisy
        chans = [qprime, theta]
        if with_z:
            chans.append(z_noisy)
        chans.extend(cal_channels)
        return np.stack(chans, axis=-1).astype("float32")     # (n, N, ch)
    # configuration B: input shape (N, P, ch); theta(P) repeated N times,
    # q and z_noisy repeated P times.
    q2 = np.repeat(qprime[:, :, None], P_LAYERS, axis=2)      # (n, N, P)
    t2 = np.repeat(theta[:, None, :], N_QUBITS, axis=1)       # (n, N, P)
    chans = [q2, t2]
    if with_z:
        chans.append(np.repeat(z_noisy[:, :, None], P_LAYERS, axis=2))
    chans.extend(cal_channels)
    return np.stack(chans, axis=-1).astype("float32")         # (n, N, P, ch)


def build_affine_design(cal_available):
    """Arm A design matrix. qubits has exactly zero variance -> dropped for rank."""
    cal_cols = []                # S2 ships no calibration array, in either setting
    if cal_available:
        cal_cols = []
    cols = [np.ones((n, 1)), theta] + cal_cols
    return np.hstack(cols)


# ---- assert the two calibration variants are the SAME computation ------------
cal_identity = {
    "affine_design_identical": bool(np.array_equal(build_affine_design(True),
                                                   build_affine_design(False))),
    "C_channels_identical": bool(np.array_equal(build_channels(False, True),
                                                build_channels(False, False))),
    "F_channels_identical": bool(np.array_equal(build_channels(True, True),
                                                build_channels(True, False))),
}
assert all(cal_identity.values()), cal_identity
print(f"[{PANEL}] calibration variants constructed both ways and asserted identical: {cal_identity}")

XA = build_affine_design(True)
XC = build_channels(False, True)
XF = build_channels(True, True)
print(f"[{PANEL}] config {CONFIG}: A design {XA.shape}, C input {XC.shape}, F input {XF.shape}")

# ---- 10-fold CV, out-of-fold predictions ------------------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
oof_A = np.full(n, np.nan)
oof_C = np.full(n, np.nan)
oof_F = np.full(n, np.nan)
fold_log = []
t_start = time.time()

for fold, (tr, te) in enumerate(kf.split(np.arange(n))):
    # ---- arm A: ordinary least squares on [1, theta] ----
    coef, _, rank, _ = np.linalg.lstsq(XA[tr], y[tr], rcond=None)
    oof_A[te] = XA[te] @ coef

    # ---- inner validation split, identical for C and F_refit ----
    rng = np.random.default_rng(FOLD_SEED + fold)
    perm = rng.permutation(len(tr))
    n_val = max(2, int(round(VAL_FRAC * len(tr))))
    val_idx = tr[perm[:n_val]]
    fit_idx = tr[perm[n_val:]]

    entry = {"fold": fold, "n_train": int(len(tr)), "n_fit": int(len(fit_idx)),
             "n_val": int(n_val), "n_test": int(len(te)), "affine_rank": int(rank),
             "test_rows": te.tolist()}

    for arm, X, synergy in (("C", XC, False), ("F_refit", XF, True)):
        preds, epochs_run, best_val = [], [], []
        for rep in range(N_REPEATS):
            keras.utils.set_random_seed(FOLD_SEED + 1000 * fold + rep)
            model = CNN1D(synergy) if CONFIG == "A" else CNN2D(synergy)
            es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True, verbose=0)
            h = model.fit(X[fit_idx], y[fit_idx],
                          validation_data=(X[val_idx], y[val_idx]),
                          epochs=MAX_EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
            preds.append(model.predict(X[te], verbose=0).ravel())
            epochs_run.append(len(h.history["loss"]))
            best_val.append(float(np.min(h.history["val_loss"])))
            del model
            keras.backend.clear_session()
        p = np.mean(preds, axis=0)
        (oof_C if arm == "C" else oof_F)[te] = p
        entry[arm] = {"epochs_run": epochs_run, "best_val_loss": best_val}
    fold_log.append(entry)
    print(f"[{PANEL}] fold {fold}: done, {time.time()-t_start:.0f}s elapsed, "
          f"C epochs {entry['C']['epochs_run']}, F epochs {entry['F_refit']['epochs_run']}",
          flush=True)

assert not np.isnan(oof_A).any() and not np.isnan(oof_C).any() and not np.isnan(oof_F).any()

res = {
    "panel": PANEL,
    "config": CONFIG,
    "n": int(n),
    "recipe": {"folds": N_FOLDS, "repeats": N_REPEATS, "max_epochs": MAX_EPOCHS,
               "batch": BATCH, "patience": PATIENCE, "val_frac": VAL_FRAC,
               "fold_seed": FOLD_SEED},
    "calibration_variants_identical": cal_identity,
    "y": y.tolist(),
    "pred": {
        "A": oof_A.tolist(),
        "C": oof_C.tolist(),
        "F_refit_matched": oof_F.tolist(),
        "F_released": f_released.tolist(),
    },
    "fold_log": fold_log,
    "wall_seconds": time.time() - t_start,
}
fp = os.path.join(OUT, f"s03_arms_{PANEL}.json")
with open(fp, "w") as fh:
    json.dump(res, fh, indent=1)
print(f"[{PANEL}] wrote {fp} in {res['wall_seconds']:.0f}s")
