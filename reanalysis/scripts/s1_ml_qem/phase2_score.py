"""PHASE 2 (score) for S1 = ML-QEM, Liao et al., Nature Machine Intelligence 6,
1478-1486 (2024). Scoped experiment: 4-qubit TFIM Trotter circuits executed on
ibm_algiers, random forest over tabular circuit features, target = the noiseless
Z expectation value on each of 4 spins.

Governed by PLAN-independent-instance-score.md, frozen before any fit, and by
phase1/RUN-RECORD-S1.json, which declared the recipe, the loss, the resample unit
and the resample count before any arm existed.

ARMS  (learner family and retraining recipe held fixed; only inputs move)
  A   affine control      LinearRegression        on descriptor columns 0..164
  C   capacity-matched    RandomForest(n=100)     on descriptor columns 0..164
  F   full arm            RandomForest(n=100)     on all columns 0..168
  F*  released F          the released per-row prediction table (no fit)

The released recipe (h36 cell 45 / demo2 cell 39) is
    for q in range(4): RandomForestRegressor(n_estimators=100).fit(X_train, y_train[:, q])
with no hyperparameter search, no scaling and no seed. The run record declared
random_state=0 for every forest before fitting, so that the ONLY difference
between C and F is the four execution-derived columns.

LOSS  the system's own metric: per-row L2 distance between the predicted 4-vector
      and the ideal 4-vector (h36 cell 51's l2_dist), averaged over evaluated rows.
      A, C, F are therefore error levels, so T = A - F is an error reduction.

ESTIMAND, formed INSIDE each bootstrap draw, never from separately averaged numbers:
      T = A - F,  K = A - C,  D = C - F,  S = K / T

CALIBRATION LADDER  scored both ways. S1's declared third group is EMPTY in this
      experiment (encode_data_v2_ecr builds vec = [] at mlp.py:157 and its
      X[:, vec_slice] assignment at mlp.py:170 is commented out), so the two rungs
      coincide by construction. Both are built and fitted separately anyway, and
      the script asserts they come out bit-identical rather than assuming it.

Nothing here retunes the 0.5 boundary, drops a system, or reconciles the
released-vs-refitted gap.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
P1 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
if os.path.isfile(P1):
    P1 = os.path.dirname(P1)
OUT = sys.argv[2] if len(sys.argv) > 2 else P1
os.makedirs(OUT, exist_ok=True)

# ---- declared before fitting, in phase1/RUN-RECORD-S1.json ----------------
RESAMPLE_UNIT = "one test row (one circuit execution on ibm_algiers, evaluated on all 4 observables jointly)"
RESAMPLE_COUNT = 2000
RF_SEED = 0
RF_TREES = 100
BOOT_SEED = 12345
ALPHA = 0.05  # 95% pointwise percentile interval
# --------------------------------------------------------------------------


def l2_rows(pred, truth):
    """The system's own per-row loss: Euclidean distance between 4-vectors."""
    return np.sqrt(np.sum((pred - truth) ** 2, axis=1))


def fit_linear(Xtr, ytr, Xte):
    out = np.zeros((Xte.shape[0], ytr.shape[1]))
    for q in range(ytr.shape[1]):
        m = LinearRegression()
        m.fit(Xtr, ytr[:, q])
        out[:, q] = m.predict(Xte)
    return out


def fit_forest(Xtr, ytr, Xte, seed=RF_SEED, trees=RF_TREES):
    """Exactly the released recipe, one forest per observable, plus the declared seed.

    n_jobs=1 (the released recipe's default) rather than -1. The trees are identical
    either way, but joblib accumulates partial predictions in a nondeterministic order,
    which perturbs the prediction at the 1e-18 level and makes a repeat fit fail an
    exact-equality check. Serial keeps the run bit-reproducible.
    """
    out = np.zeros((Xte.shape[0], ytr.shape[1]))
    for q in range(ytr.shape[1]):
        m = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1)
        m.fit(Xtr, ytr[:, q])
        out[:, q] = m.predict(Xte)
    return out


def bootstrap_S(lossA, lossC, lossF, n_draws=RESAMPLE_COUNT, seed=BOOT_SEED):
    """Resample the evaluation unit; form T, K, D, S INSIDE each draw."""
    n = lossA.shape[0]
    rng = np.random.default_rng(seed)
    T = np.empty(n_draws)
    K = np.empty(n_draws)
    D = np.empty(n_draws)
    S = np.empty(n_draws)
    for b in range(n_draws):
        idx = rng.integers(0, n, size=n)
        a = lossA[idx].mean()
        c = lossC[idx].mean()
        f = lossF[idx].mean()
        T[b] = a - f
        K[b] = a - c
        D[b] = c - f
        S[b] = (a - c) / (a - f)
    return T, K, D, S


def pct(x, lo=100 * ALPHA / 2, hi=100 * (1 - ALPHA / 2)):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def reading(interval):
    lo, hi = interval
    if lo > 0.5:
        return "PRESENT"
    if hi < 0.5:
        return "ABSENT"
    return "INDETERMINATE"


def score_variant(name, Xtr_ac, Xte_ac, Xtr_f, Xte_f, ytr, yte, released_F=None):
    """One rung of the calibration ladder. Returns a dict; fits A, C and F."""
    t0 = time.time()
    predA = fit_linear(Xtr_ac, ytr, Xte_ac)
    predC = fit_forest(Xtr_ac, ytr, Xte_ac)
    predF = fit_forest(Xtr_f, ytr, Xte_f)
    lossA = l2_rows(predA, yte)
    lossC = l2_rows(predC, yte)
    lossF = l2_rows(predF, yte)

    res = {"variant": name, "fit_seconds": round(time.time() - t0, 1),
           "A_features": int(Xtr_ac.shape[1]), "F_features": int(Xtr_f.shape[1])}

    def one(tag, lossFx, predFx):
        T, K, D, S = bootstrap_S(lossA, lossC, lossFx)
        a, c, f = lossA.mean(), lossC.mean(), lossFx.mean()
        ci = pct(S)
        d = {
            "F_arm": tag,
            "point": {
                "A_mean_L2": float(a), "C_mean_L2": float(c), "F_mean_L2": float(f),
                "T": float(a - f), "K": float(a - c), "D": float(c - f),
                "S": float((a - c) / (a - f)),
            },
            "interval_95_percentile": {
                "T": pct(T), "K": pct(K), "D": pct(D), "S": ci,
            },
            "resample_unit": RESAMPLE_UNIT,
            "resample_count": RESAMPLE_COUNT,
            "draws_with_T_le_0": int((T <= 0).sum()),
            "draws_with_S_outside_0_1": int(((S < 0) | (S > 1)).sum()),
            "reading": reading(ci),
        }
        return d

    res["refitted_F"] = one("refitted (RF n=100, all 169 columns, seed 0)", lossF, predF)
    if released_F is not None:
        lossRel = l2_rows(released_F, yte)
        res["released_F"] = one("released prediction table (hardware_over_depth.pk df['rfr_list'])",
                                lossRel, released_F)
        res["released_vs_refitted_gap"] = {
            "released_mean_L2": float(lossRel.mean()),
            "refitted_mean_L2": float(lossF.mean()),
            "gap_released_minus_refitted": float(lossRel.mean() - lossF.mean()),
            "rows_agreeing_to_1e-9": int(np.sum(np.all(np.isclose(released_F, predF, atol=1e-9), axis=1))),
            "max_abs_row_difference": float(np.max(np.abs(released_F - predF))),
            "note": "reported, not reconciled; the plan forbids adjusting the recipe after seeing this gap",
        }
    res["_losses"] = {"A": lossA, "C": lossC, "F": lossF}
    return res


def main():
    d = np.load(os.path.join(P1, "s1_hw4q.npz"))
    Xtr = d["X_train"].astype(np.float64)
    Xte = d["X_test"].astype(np.float64)
    ytr = d["y_train"].astype(np.float64)
    yte = d["y_test"].astype(np.float64)
    step_te = d["step_test"].astype(int)
    n_gate, n_angle, n_obs = int(d["n_gate"]), int(d["n_angle"]), int(d["n_obs"])
    n_desc = n_gate + n_angle

    released_F = np.load(os.path.join(P1, "released_rfr_list.npy")).astype(np.float64)

    print("=" * 78)
    print("S1 ML-QEM  |  phase 2, score")
    print("=" * 78)
    print("train %s  test %s  target %s" % (Xtr.shape, Xte.shape, yte.shape))
    print("descriptor columns 0..%d (%d), execution-derived columns %d..%d (%d)"
          % (n_desc - 1, n_desc, n_desc, n_desc + n_obs - 1, n_obs))

    # ------------------------------------------------------------------
    # The calibration ladder. S1's declared third group is empty here, so the
    # two rungs are built explicitly and their equality is ASSERTED, not assumed.
    # ------------------------------------------------------------------
    CAL_COLS = []  # verified empty: encode_data_v2_ecr builds vec = [] (mlp.py:157)
    assert Xtr.shape[1] == n_desc + n_obs, "no room for a calibration block"
    desc_cols = list(range(n_desc))
    ac_cols_with = desc_cols + CAL_COLS       # calibration available to A and C
    ac_cols_without = desc_cols                # calibration withheld from A and C
    f_cols = list(range(Xtr.shape[1]))

    print("\ncalibration ladder")
    print("  with-calibration    A/C read %d columns" % len(ac_cols_with))
    print("  without-calibration A/C read %d columns" % len(ac_cols_without))
    print("  the declared third group is EMPTY for this experiment, so the two rungs")
    print("  coincide by construction; both are fitted separately and compared.")

    print("\nfitting the WITH-calibration rung ...")
    v_with = score_variant("with_calibration", Xtr[:, ac_cols_with], Xte[:, ac_cols_with],
                           Xtr[:, f_cols], Xte[:, f_cols], ytr, yte, released_F)
    print("fitting the WITHOUT-calibration rung ...")
    v_without = score_variant("without_calibration", Xtr[:, ac_cols_without], Xte[:, ac_cols_without],
                              Xtr[:, f_cols], Xte[:, f_cols], ytr, yte, released_F)

    identical = (np.array_equal(v_with["_losses"]["A"], v_without["_losses"]["A"])
                 and np.array_equal(v_with["_losses"]["C"], v_without["_losses"]["C"])
                 and np.array_equal(v_with["_losses"]["F"], v_without["_losses"]["F"]))
    print("\n[%s] the two calibration rungs are bit-identical"
          % ("PASS" if identical else "FAIL"))

    lossA = v_without["_losses"]["A"]
    lossC = v_without["_losses"]["C"]
    lossF = v_without["_losses"]["F"]
    lossRel = l2_rows(released_F, yte)

    for v in (v_with, v_without):
        print("\n--- %s ---" % v["variant"])
        for key in ("refitted_F", "released_F"):
            if key not in v:
                continue
            r = v[key]
            p, ci = r["point"], r["interval_95_percentile"]
            print("  F = %s" % r["F_arm"])
            print("    mean per-row L2:  A %.5f   C %.5f   F %.5f"
                  % (p["A_mean_L2"], p["C_mean_L2"], p["F_mean_L2"]))
            print("    T = A-F  %.5f   [%.5f, %.5f]" % (p["T"], ci["T"][0], ci["T"][1]))
            print("    K = A-C  %.5f   [%.5f, %.5f]" % (p["K"], ci["K"][0], ci["K"][1]))
            print("    D = C-F  %.5f   [%.5f, %.5f]" % (p["D"], ci["D"][0], ci["D"][1]))
            print("    S = K/T  %.4f    [%.4f, %.4f]   (n_draws=%d)"
                  % (p["S"], ci["S"][0], ci["S"][1], r["resample_count"]))
            print("    draws with T <= 0: %d ; reading: %s"
                  % (r["draws_with_T_le_0"], r["reading"]))

    # ------------------------------------------------------------------
    # LEAK-4 qualification, quantified. The declared reading stays the one above,
    # on all 2500 test rows. This is a reported qualification, not a substitute.
    # ------------------------------------------------------------------
    def rowkey(a):
        return np.round(a, 9).tobytes()

    tr_set = set(rowkey(r) for r in Xtr[:, :n_desc])
    contaminated = np.array([rowkey(r) in tr_set for r in Xte[:, :n_desc]])
    clean = ~contaminated
    print("\nLEAK-4 quantified: %d of %d test rows share a descriptor block with training"
          % (int(contaminated.sum()), len(contaminated)))

    def subset_block(mask, label):
        T, K, D, S = bootstrap_S(lossA[mask], lossC[mask], lossF[mask])
        a, c, f = lossA[mask].mean(), lossC[mask].mean(), lossF[mask].mean()
        ci = pct(S)
        Tr, Kr, Dr, Sr = bootstrap_S(lossA[mask], lossC[mask], lossRel[mask])
        ar, cr, fr = lossA[mask].mean(), lossC[mask].mean(), lossRel[mask].mean()
        cir = pct(Sr)
        print("  %-28s rows=%4d | refitted-F S=%.4f [%.4f, %.4f] %-13s | released-F S=%.4f [%.4f, %.4f] %s"
              % (label, int(mask.sum()), (a - c) / (a - f), ci[0], ci[1], reading(ci),
                 (ar - cr) / (ar - fr), cir[0], cir[1], reading(cir)))
        return {
            "label": label, "rows": int(mask.sum()),
            "refitted_F": {"A": float(a), "C": float(c), "F": float(f),
                           "T": float(a - f), "K": float(a - c), "D": float(c - f),
                           "S": float((a - c) / (a - f)), "S_interval": ci,
                           "reading": reading(ci)},
            "released_F": {"A": float(ar), "C": float(cr), "F": float(fr),
                           "T": float(ar - fr), "K": float(ar - cr), "D": float(cr - fr),
                           "S": float((ar - cr) / (ar - fr)), "S_interval": cir,
                           "reading": reading(cir)},
        }

    print("\n  QUALIFICATION DIAGNOSTICS (reported beside the declared reading, never in place of it)")
    quals = [subset_block(clean, "leak-free rows only"),
             subset_block(contaminated, "leak-4 contaminated rows"),
             subset_block(step_te >= 2, "Trotter steps 2-9 only")]

    # per-step description
    per_step = {}
    for s in range(10):
        m = step_te == s
        a, c, f = lossA[m].mean(), lossC[m].mean(), lossF[m].mean()
        per_step[s] = {"rows": int(m.sum()), "A": float(a), "C": float(c), "F": float(f),
                       "F_released": float(lossRel[m].mean()),
                       "S_refitted": float((a - c) / (a - f)) if (a - f) != 0 else None}
    print("\n  per Trotter step (mean per-row L2)")
    print("  step  rows      A        C        F(refit)  F(released)   S(refit)")
    for s, v in per_step.items():
        print("   %d    %4d   %7.5f  %7.5f  %7.5f   %7.5f      %s"
              % (s, v["rows"], v["A"], v["C"], v["F"], v["F_released"],
                 ("%.4f" % v["S_refitted"]) if v["S_refitted"] is not None else "n/a"))

    # ------------------------------------------------------------------
    # Descriptive only: seed spread. NOT a selection, NOT part of the reading.
    # ------------------------------------------------------------------
    print("\n  seed-spread appendix (descriptive; the declared reading uses seed 0 only)")
    seed_rows = []
    for sd in [0, 1, 2, 3, 4]:
        pC = fit_forest(Xtr[:, :n_desc], ytr, Xte[:, :n_desc], seed=sd)
        pF = fit_forest(Xtr, ytr, Xte, seed=sd)
        lc, lf = l2_rows(pC, yte), l2_rows(pF, yte)
        a, c, f = lossA.mean(), lc.mean(), lf.mean()
        seed_rows.append({"seed": sd, "C": float(c), "F": float(f),
                          "S_refitted": float((a - c) / (a - f))})
        print("    seed %d:  C %.5f  F %.5f  S %.4f" % (sd, c, f, (a - c) / (a - f)))

    out = {
        "system": "S1 ML-QEM (Liao et al., Nature Machine Intelligence 6, 1478-1486, 2024)",
        "scoped_experiment": "4-qubit TFIM Trotter on ibm_algiers hardware; RF over tabular circuit features; target = noiseless Z expectation value on each of 4 spins",
        "loss": "per-row L2 distance between predicted and ideal 4-vector, averaged over rows (the system's own metric, h36 cell 51)",
        "arms": {
            "A": "LinearRegression per observable on descriptor columns 0..164",
            "C": "RandomForestRegressor(n_estimators=100, random_state=0) per observable on descriptor columns 0..164",
            "F_refitted": "RandomForestRegressor(n_estimators=100, random_state=0) per observable on all columns 0..168",
            "F_released": "docs/paper_figures/hardware_over_depth.pk df['rfr_list'], the table that produced the published figure",
        },
        "resample_unit": RESAMPLE_UNIT,
        "resample_count": RESAMPLE_COUNT,
        "bootstrap_seed": BOOT_SEED,
        "interval": "95% pointwise percentile, S formed inside each draw",
        "calibration_ladder": {
            "third_group_columns": 0,
            "why_empty": "encode_data_v2_ecr builds vec = [] (mlp.py:157) and its X[:, vec_slice] assignment (mlp.py:170) is commented out; feature width is exactly 5 + 160 + 4 = 169. The scoped experiment's released data directory holds only the circuit batches, results.pk, results_unshuffled.pk and index_order.json, with no backend-properties artifact, and h36 imports get_backend_properties_v1 without ever calling it.",
            "rungs_bit_identical": bool(identical),
            "consequence": "the with-calibration and without-calibration rungs coincide by construction and are reported as the same number, not staged as a contrast",
        },
        "with_calibration": {k: v for k, v in v_with.items() if k != "_losses"},
        "without_calibration": {k: v for k, v in v_without.items() if k != "_losses"},
        "leak_check_4": {
            "verdict": "FIRES, recorded and reported per the plan, not aborted",
            "contaminated_test_rows": int(contaminated.sum()),
            "total_test_rows": int(len(contaminated)),
            "fraction": float(contaminated.mean()),
            "direction": "favours the descriptor arms, therefore favours a PRESENT reading",
            "qualification_diagnostics": quals,
        },
        "per_trotter_step": per_step,
        "seed_spread_appendix": {
            "status": "descriptive only; the declared reading uses random_state=0 as fixed in the run record before fitting",
            "rows": seed_rows,
        },
    }
    with open(os.path.join(OUT, "S1_score.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "S1_losses.npz"),
                        lossA=lossA, lossC=lossC, lossF=lossF, lossRel=lossRel,
                        step_te=step_te, contaminated=contaminated)
    print("\nwrote", os.path.join(OUT, "S1_score.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
