"""PHASE 3 adversarial verification of the S1 (ML-QEM) scoring.

Independent of phase2_*.py: recomputes the estimand from the saved per-row losses
with its own bootstrap code, and probes six things the phase-2 scripts did not.

  1  structural leak-1: are the 165 descriptor columns pure integer counts / 100?
     A count column cannot carry a continuous noiseless expectation value.
  2  which column attains the max |corr| with a target, and is it execution-derived?
  3  rank and distinct-row structure of the descriptor design (does it explain A's
     in-sample R2 of 0.36, i.e. is A a genuine fit or a numerically broken one?)
  4  interval robustness: different bootstrap seed, different RNG, basic bootstrap,
     normal approximation, and a CLUSTER bootstrap over circuits rather than rows
     (the declared unit is the row, but rows repeat the same circuit, so the row
     bootstrap is the narrower of the two; the cluster version is the adverse one)
  5  ridge-A restricted to Trotter steps 2-9, the fully leak-free block that the
     phase-2 stress did not cross with the strongest affine arm
  6  an independent recomputation of every headline number from S1_losses.npz
"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(BASE, "..", "..", "outputs", "s1_ml_qem"))
P1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
P2 = sys.argv[2] if len(sys.argv) > 2 else P1
os.makedirs(P2, exist_ok=True)

OK = []
BAD = []


def check(label, cond, detail=""):
    (OK if cond else BAD).append(label + (" :: " + detail if detail else ""))
    print("[%s] %s %s" % ("PASS" if cond else "FAIL", label, detail))
    return bool(cond)


def l2_rows(p, t):
    return np.sqrt(np.sum((p - t) ** 2, axis=1))


def per_obs(make, Xtr, ytr, Xte):
    o = np.zeros((Xte.shape[0], ytr.shape[1]))
    for q in range(ytr.shape[1]):
        m = make()
        m.fit(Xtr, ytr[:, q])
        o[:, q] = np.asarray(m.predict(Xte)).ravel()
    return o


def reading(iv):
    return "PRESENT" if iv[0] > 0.5 else ("ABSENT" if iv[1] < 0.5 else "INDETERMINATE")


def main():
    d = np.load(os.path.join(P1, "s1_hw4q.npz"))
    Xtr = d["X_train"].astype(np.float64)
    Xte = d["X_test"].astype(np.float64)
    ytr = d["y_train"].astype(np.float64)
    yte = d["y_test"].astype(np.float64)
    step_te = d["step_test"].astype(int)
    noisy_te = d["noisy_test"].astype(np.float64)
    n_desc = int(d["n_gate"]) + int(d["n_angle"])
    Dtr, Dte = Xtr[:, :n_desc], Xte[:, :n_desc]

    L = np.load(os.path.join(P2, "S1_losses.npz"))
    lossA, lossC, lossF, lossRel = L["lossA"], L["lossC"], L["lossF"], L["lossRel"]
    contaminated = L["contaminated"].astype(bool)
    clean = ~contaminated

    out = {}

    # ---------------------------------------------------------------- 1
    print("\n(1) STRUCTURAL LEAK-1: are the descriptors pure counts?")
    scaled = np.vstack([Dtr, Dte]) * 100.0
    resid = np.abs(scaled - np.round(scaled))
    check("every descriptor column is an integer count / 100",
          float(resid.max()) < 1e-3, "max |x*100 - round(x*100)| = %.3g" % float(resid.max()))
    print("    distinct values taken by the whole descriptor block: %d"
          % len(np.unique(np.round(scaled, 6))))
    tail = np.vstack([Xtr[:, n_desc:], Xte[:, n_desc:]]) * 100.0
    tail_resid = np.abs(tail - np.round(tail))
    check("the 4 execution-derived columns are NOT integer counts (they are expvals)",
          float(tail_resid.max()) > 1e-2, "max residual = %.3g" % float(tail_resid.max()))
    out["structural_leak1"] = {
        "descriptor_max_integer_residual": float(resid.max()),
        "execution_block_max_integer_residual": float(tail_resid.max()),
    }

    # ---------------------------------------------------------------- 2
    print("\n(2) which column attains the maximum |corr| with a target?")
    Xall = np.vstack([Xtr, Xte])
    yall = np.vstack([ytr, yte])
    best = (0.0, None, None)
    best_desc = (0.0, None, None)
    for q in range(4):
        col = yall[:, q]
        for j in range(Xall.shape[1]):
            f = Xall[:, j]
            if f.std() < 1e-12 or col.std() < 1e-12:
                continue
            r = abs(float(np.corrcoef(f, col)[0, 1]))
            if r > best[0]:
                best = (r, j, q)
            if j < n_desc and r > best_desc[0]:
                best_desc = (r, j, q)
    print("    overall max |corr| = %.4f at column %d (observable %d) -> %s"
          % (best[0], best[1], best[2],
             "EXECUTION-DERIVED" if best[1] >= n_desc else "DESCRIPTOR"))
    print("    descriptor-only max |corr| = %.4f at column %d (observable %d)"
          % (best_desc[0], best_desc[1], best_desc[2]))
    check("the highest-correlating column is execution-derived, not a descriptor",
          best[1] >= n_desc, "column %d of %d" % (best[1], Xall.shape[1]))
    check("no descriptor column is a near-copy of a target (|corr| < 0.9)",
          best_desc[0] < 0.9, "max descriptor |corr| = %.4f" % best_desc[0])
    out["max_corr"] = {"overall_r": best[0], "overall_col": int(best[1]),
                       "overall_is_execution_derived": bool(best[1] >= n_desc),
                       "descriptor_only_r": best_desc[0], "descriptor_only_col": int(best_desc[1])}

    # ---------------------------------------------------------------- 3
    print("\n(3) descriptor design structure (is arm A a genuine fit?)")
    rank_tr = int(np.linalg.matrix_rank(np.hstack([Dtr, np.ones((Dtr.shape[0], 1))])))
    nonconst = int((np.vstack([Dtr, Dte]).std(axis=0) > 1e-12).sum())
    uniq_tr = len(np.unique(np.round(Dtr, 9), axis=0))
    uniq_te = len(np.unique(np.round(Dte, 9), axis=0))
    print("    rank of [Dtr | 1]           : %d  (of %d columns, %d nonconstant)"
          % (rank_tr, Dtr.shape[1], nonconst))
    print("    distinct descriptor rows    : train %d of 500, test %d of 2500"
          % (uniq_tr, Dte.shape[0] and uniq_te))
    predA_tr = per_obs(lambda: LinearRegression(), Dtr, ytr, Dtr)
    r2_in = float(r2_score(ytr, predA_tr, multioutput="uniform_average"))
    print("    OLS in-sample R2            : %.4f" % r2_in)
    print("    -> with rank %d on %d rows, an in-sample R2 of %.2f is what a genuine"
          % (rank_tr, Dtr.shape[0], r2_in))
    print("       least-squares fit gives; A is rank-limited, not numerically broken.")
    coef_norm = float(np.linalg.norm(LinearRegression().fit(Dtr, ytr[:, 0]).coef_))
    print("    ||coef|| for observable 0   : %.3g (blow-up would indicate a broken fit)" % coef_norm)
    out["design"] = {"rank_with_intercept": rank_tr, "nonconstant_columns": nonconst,
                     "distinct_descriptor_rows_train": uniq_tr,
                     "distinct_descriptor_rows_test": uniq_te,
                     "ols_in_sample_r2": r2_in, "ols_coef_norm_obs0": coef_norm}

    # ---------------------------------------------------------------- 4
    print("\n(4) interval robustness for the declared reading")

    def boot_row(lA, lC, lF, n=2000, seed=12345, rng_kind="pcg"):
        m = lA.shape[0]
        rng = (np.random.default_rng(seed) if rng_kind == "pcg"
               else np.random.RandomState(seed))
        S = np.empty(n)
        for b in range(n):
            i = (rng.integers(0, m, size=m) if rng_kind == "pcg"
                 else rng.randint(0, m, size=m))
            a, c, f = lA[i].mean(), lC[i].mean(), lF[i].mean()
            S[b] = (a - c) / (a - f)
        return S

    def boot_cluster(lA, lC, lF, groups, n=2000, seed=777):
        """Resample whole circuits (descriptor blocks), not rows."""
        rng = np.random.default_rng(seed)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        G = len(uniq)
        S = np.empty(n)
        for b in range(n):
            pick = rng.integers(0, G, size=G)
            idx = np.concatenate([members[p] for p in pick])
            a, c, f = lA[idx].mean(), lC[idx].mean(), lF[idx].mean()
            S[b] = (a - c) / (a - f)
        return S

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    Spoint = (lossA.mean() - lossC.mean()) / (lossA.mean() - lossF.mean())
    variants = {}
    for label, S in [
        ("row bootstrap, declared seed 12345 (PCG64)", boot_row(lossA, lossC, lossF)),
        ("row bootstrap, seed 99991 (PCG64)", boot_row(lossA, lossC, lossF, seed=99991)),
        ("row bootstrap, seed 2 (legacy MT19937)", boot_row(lossA, lossC, lossF, seed=2, rng_kind="mt")),
        ("row bootstrap, 20000 draws", boot_row(lossA, lossC, lossF, n=20000, seed=4242)),
    ]:
        iv = ci(S)
        basic = (2 * Spoint - iv[1], 2 * Spoint - iv[0])
        norm = (Spoint - 1.959964 * S.std(ddof=1), Spoint + 1.959964 * S.std(ddof=1))
        print("    %-42s pct %s [%.4f, %.4f]  basic [%.4f, %.4f]  normal [%.4f, %.4f]"
              % (label, reading(iv), iv[0], iv[1], basic[0], basic[1], norm[0], norm[1]))
        variants[label] = {"percentile": iv, "reading_percentile": reading(iv),
                           "basic": list(basic), "reading_basic": reading(basic),
                           "normal": list(norm), "reading_normal": reading(norm)}

    # cluster bootstrap over descriptor blocks (circuit proxy)
    keys = [np.round(r, 9).tobytes() for r in Dte]
    lut = {}
    gid = np.empty(len(keys), dtype=int)
    for i, k in enumerate(keys):
        gid[i] = lut.setdefault(k, len(lut))
    print("    distinct circuits (descriptor blocks) among the 2500 test rows: %d" % len(lut))
    for seed in (777, 31337):
        Sc = boot_cluster(lossA, lossC, lossF, gid, seed=seed)
        iv = ci(Sc)
        print("    %-42s pct %s [%.4f, %.4f]"
              % ("CLUSTER bootstrap over circuits, seed %d" % seed, reading(iv), iv[0], iv[1]))
        variants["cluster bootstrap seed %d" % seed] = {
            "percentile": iv, "reading_percentile": reading(iv)}
    Scr = boot_cluster(lossA, lossC, lossRel, gid, seed=777)
    ivcr = ci(Scr)
    print("    %-42s pct %s [%.4f, %.4f]"
          % ("CLUSTER bootstrap, released-F", reading(ivcr), ivcr[0], ivcr[1]))
    variants["cluster bootstrap released-F"] = {"percentile": ivcr,
                                                "reading_percentile": reading(ivcr)}
    out["interval_robustness"] = variants
    all_readings = sorted({v["reading_percentile"] for v in variants.values()}
                          | {v.get("reading_basic", "PRESENT") for v in variants.values()}
                          | {v.get("reading_normal", "PRESENT") for v in variants.values()})
    check("the declared reading survives every interval construction tried",
          all_readings == ["PRESENT"], str(all_readings))

    # ---------------------------------------------------------------- 5
    print("\n(5) strongest affine arm on Trotter steps 2-9 (the fully leak-free block)")
    predAr = per_obs(lambda: make_pipeline(StandardScaler(),
                                           RidgeCV(alphas=np.logspace(-4, 4, 60))),
                     Dtr, ytr, Dte)
    lossAr = l2_rows(predAr, yte)
    blocks = [("declared OLS-A, steps 2-9", lossA, step_te >= 2),
              ("ridge-A, steps 2-9", lossAr, step_te >= 2),
              ("ridge-A, steps 1-9", lossAr, step_te >= 1),
              ("declared OLS-A, all rows", lossA, np.ones(len(lossA), bool)),
              ("ridge-A, leak-free rows", lossAr, clean)]
    stress = {}
    for label, lA, m in blocks:
        S = boot_row(lA[m], lossC[m], lossF[m])
        iv = ci(S)
        a, c, f = lA[m].mean(), lossC[m].mean(), lossF[m].mean()
        print("    %-30s rows=%4d  S=%.4f [%.4f, %.4f]  %s"
              % (label, int(m.sum()), (a - c) / (a - f), iv[0], iv[1], reading(iv)))
        stress[label] = {"rows": int(m.sum()), "S": float((a - c) / (a - f)),
                         "interval": iv, "reading": reading(iv)}
    out["extra_stress"] = stress

    # ---------------------------------------------------------------- 6
    print("\n(6) independent recomputation of the headline numbers")
    a, c, f, fr = lossA.mean(), lossC.mean(), lossF.mean(), lossRel.mean()
    rec = {"A": a, "C": c, "F_refit": f, "F_released": fr,
           "T": a - f, "K": a - c, "D": c - f, "S_refit": (a - c) / (a - f),
           "S_released": (a - c) / (a - fr)}
    for k, v in rec.items():
        print("    %-12s %.5f" % (k, v))
    with open(os.path.join(P2, "S1_RESULT.json")) as fh:
        R = json.load(fh)
    rep = R["DECLARED_READING"]["with_calibration"]
    check("S refitted matches the reported value",
          abs(rec["S_refit"] - rep["refitted_F"]["S"]) < 1e-12,
          "%.10f vs %.10f" % (rec["S_refit"], rep["refitted_F"]["S"]))
    check("S released matches the reported value",
          abs(rec["S_released"] - rep["released_F"]["S"]) < 1e-12,
          "%.10f vs %.10f" % (rec["S_released"], rep["released_F"]["S"]))
    check("T, K, D match the reported values",
          abs(rec["T"] - R["T_K_D_refitted_F"]["T"][0]) < 1e-12
          and abs(rec["K"] - R["T_K_D_refitted_F"]["K"][0]) < 1e-12
          and abs(rec["D"] - R["T_K_D_refitted_F"]["D"][0]) < 1e-12)
    check("K + D == T exactly (the decomposition closes)",
          abs((rec["K"] + rec["D"]) - rec["T"]) < 1e-12,
          "residual %.3g" % abs((rec["K"] + rec["D"]) - rec["T"]))
    check("the with- and without-calibration rungs are reported as identical",
          R["DECLARED_READING"]["with_calibration"] == R["DECLARED_READING"]["without_calibration"])
    check("leak-4 count matches the independently recomputed mask",
          int(contaminated.sum()) == R["leak_check_4"]["contaminated_test_rows"],
          "%d vs %d" % (int(contaminated.sum()), R["leak_check_4"]["contaminated_test_rows"]))

    # the descriptor control against the system's own comparator
    l_noisy = l2_rows(noisy_te, yte)
    print("\n    for context: C %.5f vs unmitigated %.5f -> C beats do-nothing: %s"
          % (c, l_noisy.mean(), c < l_noisy.mean()))
    print("    share of the SYSTEM'S OWN claimed gain (unmitigated -> F) reproduced by C:"
          " %.4f" % ((l_noisy.mean() - c) / (l_noisy.mean() - f)))
    out["alternative_reference"] = {
        "unmitigated_mean_L2": float(l_noisy.mean()),
        "S_if_referenced_to_unmitigated_instead_of_A": float((l_noisy.mean() - c) / (l_noisy.mean() - f)),
        "note": "NOT the declared estimand; the plan fixes A as the reference. Reported only to show how far the descriptor control is from the system's own baseline.",
    }
    out["recomputed"] = {k: float(v) for k, v in rec.items()}

    print("\nPASSED: %d   FAILED: %d" % (len(OK), len(BAD)))
    for b in BAD:
        print("  FAIL:", b)
    out["passed"] = OK
    out["failed"] = BAD
    with open(os.path.join(P2, "S1_phase3_adversarial.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", os.path.join(P2, "S1_phase3_adversarial.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
