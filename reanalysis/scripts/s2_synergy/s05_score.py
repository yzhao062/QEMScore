"""
S2 phase 2, step 5: FORM THE ESTIMAND AND APPLY THE READING RULE.

T = A - F,  K = A - C,  D = C - F,  S = K / T,
all four formed INSIDE each bootstrap draw from that draw's resampled rows.
Never from separately averaged numbers.

Resample unit: the circuit, that is one released row. Count: 10000, declared in
PREREG-phase2.json before any fit. Interval: percentile, 95%.

The reading rule is applied exactly as frozen:
  interval entirely above 0.5 -> PRESENT
  interval entirely below 0.5 -> ABSENT
  interval spans 0.5          -> INDETERMINATE
"""
import json
import os

import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs", "s2_synergy"))
OUT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else OUT
os.makedirs(OUTDIR, exist_ok=True)
PANELS = ["Fig8a", "Fig8b", "Fig8c"]
N_BOOT = 10000
BOOT_SEED_BASE = 12527150
QLO, QHI = 2.5, 97.5


def read(interval):
    lo, hi = interval
    if lo > 0.5:
        return "PRESENT"
    if hi < 0.5:
        return "ABSENT"
    return "INDETERMINATE"


def score_panel(panel):
    with open(os.path.join(OUT, f"s03_arms_{panel}.json")) as f:
        d = json.load(f)
    y = np.asarray(d["y"], float)
    pA = np.asarray(d["pred"]["A"], float)
    pC = np.asarray(d["pred"]["C"], float)
    pFr = np.asarray(d["pred"]["F_refit_matched"], float)
    pFl = np.asarray(d["pred"]["F_released"], float)
    n = len(y)

    # per-row squared errors, fixed before resampling (conditional on the fitted models)
    eA = (y - pA) ** 2
    eC = (y - pC) ** 2
    eFr = (y - pFr) ** 2
    eFl = (y - pFl) ** 2

    res = {"panel": panel, "config": d["config"], "n": n,
           "calibration_variants_identical": d["calibration_variants_identical"],
           "full_sample": {}, "variants": {}}

    # descriptive: full-sample arm errors, in MSE and in the paper's own 1 - R2
    var = float(np.mean((y - y.mean()) ** 2))
    for nm, e in (("A", eA), ("C", eC), ("F_refit_matched", eFr), ("F_released", eFl)):
        res["full_sample"][nm] = {"mse": float(e.mean()),
                                  "one_minus_r2": float(e.mean() / var),
                                  "r2": float(1.0 - e.mean() / var)}
    res["full_sample"]["target_variance"] = var

    rng_master = np.random.default_rng(BOOT_SEED_BASE + PANELS.index(panel))
    idx = rng_master.integers(0, n, size=(N_BOOT, n))     # the SAME draws for both F variants

    for fname, eF in (("F_released", eFl), ("F_refit_matched", eFr)):
        # ---- point estimate on the full evaluation set ----
        T0 = eA.mean() - eF.mean()
        K0 = eA.mean() - eC.mean()
        D0 = eC.mean() - eF.mean()
        S0 = K0 / T0

        # ---- the estimand, formed inside each draw ----
        mA = eA[idx].mean(axis=1)
        mC = eC[idx].mean(axis=1)
        mF = eF[idx].mean(axis=1)
        T = mA - mF
        K = mA - mC
        D = mC - mF
        S = K / T

        ci = [float(np.percentile(S, QLO)), float(np.percentile(S, QHI))]
        entry = {
            "point": {"T": float(T0), "K": float(K0), "D": float(D0), "S": float(S0)},
            "interval_S": ci,
            "reading": read(ci),
            "interval_T": [float(np.percentile(T, QLO)), float(np.percentile(T, QHI))],
            "interval_K": [float(np.percentile(K, QLO)), float(np.percentile(K, QHI))],
            "interval_D": [float(np.percentile(D, QLO)), float(np.percentile(D, QHI))],
            "median_S": float(np.median(S)),
            "frac_draws_T_le_0": float(np.mean(T <= 0)),
            "frac_draws_K_le_0": float(np.mean(K <= 0)),
            "frac_draws_D_le_0": float(np.mean(D <= 0)),
            "resample_unit": "circuit (one released row)",
            "resample_count": N_BOOT,
            "interval_type": "percentile 95%",
        }
        # both calibration variants are the SAME computation for S2 (empty calibration
        # group either way, asserted by array equality in s03_fit_arms.py)
        res["variants"][fname] = {"with_calibration": entry, "without_calibration": entry}

    # released vs refitted F gap: a reportable quantity, not reconciled
    res["released_vs_refit_gap"] = {
        "mse_F_released": float(eFl.mean()),
        "mse_F_refit_matched": float(eFr.mean()),
        "r2_F_released": float(1 - eFl.mean() / var),
        "r2_F_refit_matched": float(1 - eFr.mean() / var),
        "delta_mse_refit_minus_released": float(eFr.mean() - eFl.mean()),
    }
    return res


allres = {}
for p in PANELS:
    r = score_panel(p)
    allres[p] = r
    print("=" * 78)
    print(f"{p}  (configuration {r['config']}, n = {r['n']} circuits)")
    fs = r["full_sample"]
    print("  full-sample arm performance on the 100 held-out rows:")
    for nm in ("A", "C", "F_refit_matched", "F_released"):
        print(f"    {nm:<17} MSE = {fs[nm]['mse']:.6e}   1-R2 = {fs[nm]['one_minus_r2']:.4f}   "
              f"R2 = {fs[nm]['r2']:+.4f}")
    for fname in ("F_released", "F_refit_matched"):
        e = r["variants"][fname]["with_calibration"]
        print(f"  --- F = {fname} ---")
        print(f"    T = A-F   point {e['point']['T']:+.6e}   95% CI "
              f"[{e['interval_T'][0]:+.4e}, {e['interval_T'][1]:+.4e}]")
        print(f"    K = A-C   point {e['point']['K']:+.6e}   95% CI "
              f"[{e['interval_K'][0]:+.4e}, {e['interval_K'][1]:+.4e}]")
        print(f"    D = C-F   point {e['point']['D']:+.6e}   95% CI "
              f"[{e['interval_D'][0]:+.4e}, {e['interval_D'][1]:+.4e}]")
        print(f"    S = K/T   point {e['point']['S']:+.4f}   median {e['median_S']:+.4f}   "
              f"95% CI [{e['interval_S'][0]:+.4f}, {e['interval_S'][1]:+.4f}]   "
              f"({e['resample_count']} draws, unit = {e['resample_unit']})")
        print(f"    share of draws with T <= 0: {e['frac_draws_T_le_0']:.4f}")
        print(f"    READING: {e['reading']}")
    g = r["released_vs_refit_gap"]
    print(f"  released-vs-refitted F gap (reportable, not reconciled): "
          f"R2 released {g['r2_F_released']:+.4f} vs refitted {g['r2_F_refit_matched']:+.4f}; "
          f"delta MSE = {g['delta_mse_refit_minus_released']:+.4e}")

fp = os.path.join(OUTDIR, "s05_score.json")
with open(fp, "w") as f:
    json.dump(allres, f, indent=2)
print("\nwrote " + fp)
