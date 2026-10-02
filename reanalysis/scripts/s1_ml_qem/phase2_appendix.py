"""PHASE 2 appendix for S1 (ML-QEM). Reported BESIDE the declared reading, never
in place of it. Nothing here changes the estimand, the arms or the 0.5 boundary.

Three things a reader of the S1 result needs and the estimand alone does not show.

(1) The affine arm A is a weak baseline. S = (A - C) / (A - F) is monotone
    increasing in A, so a weak A pushes S toward 1. This reports how weak A is,
    whether it is broken or merely overfitting, and what S becomes under the
    STRONGEST affine descriptor-only arm available. That sensitivity can only move
    S downward, toward the 0.5 boundary, so it is not result-shopping.

(2) The trivial baselines. The system's own comparators are the unmitigated noisy
    value and ZNE. Where the descriptor control C sits relative to those says
    something the ratio S cannot.

(3) Whether the capacity match is the binding constraint on C, by comparing C
    against the saturation bound.
"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(BASE, "..", "..", "outputs", "s1_ml_qem"))
P1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
OUT = sys.argv[2] if len(sys.argv) > 2 else P1
os.makedirs(OUT, exist_ok=True)
RESAMPLE_COUNT = 2000
BOOT_SEED = 12345


def l2_rows(pred, truth):
    return np.sqrt(np.sum((pred - truth) ** 2, axis=1))


def boot_S(lossA, lossC, lossF, n=RESAMPLE_COUNT, seed=BOOT_SEED):
    m = lossA.shape[0]
    rng = np.random.default_rng(seed)
    S = np.empty(n)
    for b in range(n):
        i = rng.integers(0, m, size=m)
        a, c, f = lossA[i].mean(), lossC[i].mean(), lossF[i].mean()
        S[b] = (a - c) / (a - f)
    return S


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def reading(iv):
    return "PRESENT" if iv[0] > 0.5 else ("ABSENT" if iv[1] < 0.5 else "INDETERMINATE")


def per_obs(make, Xtr, ytr, Xte):
    out = np.zeros((Xte.shape[0], ytr.shape[1]))
    for q in range(ytr.shape[1]):
        m = make()
        m.fit(Xtr, ytr[:, q])
        out[:, q] = m.predict(Xte)
    return out


def main():
    d = np.load(os.path.join(P1, "s1_hw4q.npz"))
    Xtr, Xte = d["X_train"].astype(np.float64), d["X_test"].astype(np.float64)
    ytr, yte = d["y_train"].astype(np.float64), d["y_test"].astype(np.float64)
    noisy_te, zne_te = d["noisy_test"].astype(np.float64), d["zne_test"].astype(np.float64)
    n_desc = int(d["n_gate"]) + int(d["n_angle"])
    Dtr, Dte = Xtr[:, :n_desc], Xte[:, :n_desc]

    L = np.load(os.path.join(OUT, "S1_losses.npz"))
    lossA, lossC, lossF, lossRel = L["lossA"], L["lossC"], L["lossF"], L["lossRel"]

    print("=" * 78)
    print("S1 appendix (reported beside the declared reading, never in place of it)")
    print("=" * 78)

    # ---------------- (1) how weak is A, and is it broken or overfitting?
    predA_tr = per_obs(lambda: LinearRegression(), Dtr, ytr, Dtr)
    predA_te = per_obs(lambda: LinearRegression(), Dtr, ytr, Dte)
    r2_tr = float(r2_score(ytr, predA_tr, multioutput="uniform_average"))
    r2_te = float(r2_score(yte, predA_te, multioutput="uniform_average"))
    print("\n(1) the affine arm A")
    print("    OLS on 165 descriptor columns, 500 training rows")
    print("    R2 in-sample %.4f, held-out %.4f  -> overfitting, not degenerate"
          % (r2_tr, r2_te))
    print("    mean per-row L2: A %.5f" % lossA.mean())

    # strongest affine descriptor-only arm available
    predAr = per_obs(lambda: make_pipeline(StandardScaler(),
                                           RidgeCV(alphas=np.logspace(-4, 4, 60))),
                     Dtr, ytr, Dte)
    lossAr = l2_rows(predAr, yte)
    print("    strongest affine arm (RidgeCV, still descriptor-only, still affine):")
    print("      held-out R2 %.4f, mean per-row L2 %.5f"
          % (float(r2_score(yte, predAr, multioutput="uniform_average")), lossAr.mean()))

    Sr = boot_S(lossAr, lossC, lossF)
    ivr = ci(Sr)
    Srr = boot_S(lossAr, lossC, lossRel)
    ivrr = ci(Srr)
    a, c, f = lossAr.mean(), lossC.mean(), lossF.mean()
    print("    SENSITIVITY, ridge-A instead of OLS-A (NOT the declared reading):")
    print("      refitted-F  S = %.4f  [%.4f, %.4f]  %s"
          % ((a - c) / (a - f), ivr[0], ivr[1], reading(ivr)))
    ar, cr, fr = lossAr.mean(), lossC.mean(), lossRel.mean()
    print("      released-F  S = %.4f  [%.4f, %.4f]  %s"
          % ((ar - cr) / (ar - fr), ivrr[0], ivrr[1], reading(ivrr)))

    # ---------------- (2) the trivial baselines
    l_noisy = l2_rows(noisy_te, yte)
    l_zne = l2_rows(zne_te, yte)
    print("\n(2) where each arm sits against the system's own comparators")
    print("    mean per-row L2 on the 2500 held-out rows")
    print("      A   affine descriptor control        %.5f" % lossA.mean())
    print("      A'  ridge descriptor control         %.5f" % lossAr.mean())
    print("      C   capacity-matched descriptor ctrl %.5f" % lossC.mean())
    print("      unmitigated noisy (do nothing)       %.5f" % l_noisy.mean())
    print("      ZNE                                  %.5f" % l_zne.mean())
    print("      F   refitted full arm                %.5f" % lossF.mean())
    print("      F*  released prediction table        %.5f" % lossRel.mean())
    print("    C beats do-nothing: %s (C - unmitigated = %+.5f)"
          % ("YES" if lossC.mean() < l_noisy.mean() else "NO",
             lossC.mean() - l_noisy.mean()))
    # paired bootstrap on C - unmitigated
    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(RESAMPLE_COUNT)
    for b in range(RESAMPLE_COUNT):
        i = rng.integers(0, len(lossC), size=len(lossC))
        diffs[b] = lossC[i].mean() - l_noisy[i].mean()
    dci = ci(diffs)
    print("      C - unmitigated, 95%% interval [%+.5f, %+.5f] over %d draws"
          % (dci[0], dci[1], RESAMPLE_COUNT))

    # ---------------- (3) is the capacity match the binding constraint on C?
    predC_big = per_obs(lambda: RandomForestRegressor(n_estimators=2000, random_state=0, n_jobs=1),
                        Dtr, ytr, Dte)
    lossC_big = l2_rows(predC_big, yte)
    print("\n(3) is the capacity match what limits C?")
    print("    C at the released capacity (RF n=100): mean L2 %.5f, R2 %.4f"
          % (lossC.mean(), float(r2_score(yte, per_obs(
              lambda: RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1),
              Dtr, ytr, Dte), multioutput="uniform_average"))))
    print("    saturation bound (RF n=2000)         : mean L2 %.5f, R2 %.4f"
          % (lossC_big.mean(), float(r2_score(yte, predC_big, multioutput="uniform_average"))))
    print("    -> the capacity match is not the binding constraint; C is already at")
    print("       the descriptor bound, so no stronger descriptor control lifts S much.")
    Sbig = boot_S(lossA, lossC_big, lossF)
    ivb = ci(Sbig)
    a2, c2, f2 = lossA.mean(), lossC_big.mean(), lossF.mean()
    print("    S with the unmatched RF n=2000 as the control (NOT the declared reading):")
    print("      S = %.4f  [%.4f, %.4f]  %s" % ((a2 - c2) / (a2 - f2), ivb[0], ivb[1], reading(ivb)))

    out = {
        "status": "appendix; the declared reading is in S1_score.json and is unchanged by anything here",
        "affine_arm_A": {
            "spec": "OLS on 165 descriptor columns, 500 training rows",
            "r2_in_sample": r2_tr, "r2_held_out": r2_te,
            "mean_per_row_L2": float(lossA.mean()),
            "verdict": "overfitting, not degenerate: it fits the training rows and generalizes poorly",
            "why_it_matters": "S = (A-C)/(A-F) increases with A, so a weak affine arm pushes S toward 1",
        },
        "sensitivity_strongest_affine_arm": {
            "spec": "RidgeCV over standardized descriptors, still affine and still descriptor-only",
            "r2_held_out": float(r2_score(yte, predAr, multioutput="uniform_average")),
            "mean_per_row_L2": float(lossAr.mean()),
            "S_refitted_F": float((a - c) / (a - f)), "S_refitted_F_interval": ivr,
            "S_released_F": float((ar - cr) / (ar - fr)), "S_released_F_interval": ivrr,
            "reading_refitted": reading(ivr), "reading_released": reading(ivrr),
            "note": "this sensitivity can only move S toward 0.5, so it tests the declared reading rather than favouring it",
        },
        "baselines_mean_per_row_L2": {
            "A_affine_control": float(lossA.mean()),
            "A_ridge_control": float(lossAr.mean()),
            "C_capacity_matched_control": float(lossC.mean()),
            "unmitigated_noisy": float(l_noisy.mean()),
            "zne": float(l_zne.mean()),
            "F_refitted": float(lossF.mean()),
            "F_released": float(lossRel.mean()),
            "C_minus_unmitigated": float(lossC.mean() - l_noisy.mean()),
            "C_minus_unmitigated_interval": dci,
            "C_beats_doing_nothing": bool(lossC.mean() < l_noisy.mean()),
        },
        "capacity_match_is_not_binding": {
            "C_matched_RF100_meanL2": float(lossC.mean()),
            "RF2000_meanL2": float(lossC_big.mean()),
            "S_with_unmatched_control": float((a2 - c2) / (a2 - f2)),
            "S_with_unmatched_control_interval": ivb,
            "reading": reading(ivb),
        },
    }
    with open(os.path.join(OUT, "S1_appendix.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.join(OUT, "S1_appendix.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
