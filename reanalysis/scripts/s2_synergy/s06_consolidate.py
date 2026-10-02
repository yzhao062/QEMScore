"""
S2 phase 2, step 6: CONSOLIDATE. No new arm, no new fit.

Adds three derived checks over numbers already computed:

  (a) internal consistency -- arm A's held-out R2 must equal the OLS entry of the
      saturation sweep, since both are the same model on the same folds.
  (b) the flip threshold -- how good a descriptor control would have to be to move
      each panel from ABSENT to PRESENT, compared against the BEST descriptor-only
      predictor found by the saturation sweep. This is what the PLAN's saturation
      diagnostic is for: it bounds what any descriptor control could reach.
  (c) the calibration-variant identity flags asserted at fit time.
"""
import json
import os

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "outputs", "s2_synergy"))
OUT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else OUT
os.makedirs(OUTDIR, exist_ok=True)
PANELS = ["Fig8a", "Fig8b", "Fig8c"]

score = json.load(open(os.path.join(OUT, "s05_score.json")))
sat = json.load(open(os.path.join(OUT, "s04_saturation.json")))

final = {"tag": "S2", "phase": "2 of 3, SCORE", "panels": {}}

for p in PANELS:
    r = score[p]
    fs = r["full_sample"]
    var = fs["target_variance"]
    arm = json.load(open(os.path.join(OUT, f"s03_arms_{p}.json")))

    # (a) internal consistency
    a_r2 = fs["A"]["r2"]
    ols_r2 = sat[p]["learned_family_scores"]["ols"]
    consistent = abs(a_r2 - ols_r2) < 1e-9

    entry = {
        "config": r["config"],
        "n": r["n"],
        "calibration_variants_identical": arm["calibration_variants_identical"],
        "arm_r2": {k: fs[k]["r2"] for k in ("A", "C", "F_refit_matched", "F_released")},
        "arm_mse": {k: fs[k]["mse"] for k in ("A", "C", "F_refit_matched", "F_released")},
        "saturation_learned": sat[p]["learned_saturation_best"],
        "saturation_learned_family": sat[p]["learned_saturation_best_family"],
        "saturation_analytic": sat[p]["analytic_saturation"],
        "internal_consistency_armA_equals_sweep_ols": {
            "arm_A_r2": a_r2, "sweep_ols_r2": ols_r2, "match": bool(consistent)},
        "flip_threshold": {},
        "S": {},
    }

    for fname in ("F_released", "F_refit_matched"):
        e = r["variants"][fname]["with_calibration"]
        entry["S"][fname] = {
            "point": e["point"]["S"], "interval": e["interval_S"],
            "reading": e["reading"], "resample_unit": e["resample_unit"],
            "resample_count": e["resample_count"],
            "frac_draws_T_le_0": e["frac_draws_T_le_0"],
            "T_point": e["point"]["T"], "K_point": e["point"]["K"], "D_point": e["point"]["D"],
        }
        # S > 0.5 requires err_C < err_A - 0.5 * (err_A - err_F)
        need_mse = fs["A"]["mse"] - 0.5 * e["point"]["T"]
        entry["flip_threshold"][fname] = {
            "descriptor_control_mse_needed_for_S_above_0.5": need_mse,
            "descriptor_control_r2_needed_for_S_above_0.5": 1.0 - need_mse / var,
            "best_descriptor_only_r2_found_by_sweep": sat[p]["learned_saturation_best"],
            "actual_arm_C_r2": fs["C"]["r2"],
            "margin_r2": (1.0 - need_mse / var) - sat[p]["learned_saturation_best"],
        }
    final["panels"][p] = entry

print("=" * 90)
print("S2  CONSOLIDATED  (Synergy CNN, Cantori et al., EPJ Quantum Technology 11, 45 (2024))")
print("=" * 90)
for p in PANELS:
    e = final["panels"][p]
    print(f"\n{p}  configuration {e['config']}, n = {e['n']} circuits")
    print(f"  arm R2 on the same 100 held-out rows:  "
          f"A {e['arm_r2']['A']:+.4f}   C {e['arm_r2']['C']:+.4f}   "
          f"F_refit {e['arm_r2']['F_refit_matched']:+.4f}   F_released {e['arm_r2']['F_released']:+.4f}")
    ic = e["internal_consistency_armA_equals_sweep_ols"]
    print(f"  consistency: arm A R2 {ic['arm_A_r2']:+.6f} == sweep OLS {ic['sweep_ols_r2']:+.6f}  "
          f"-> {ic['match']}")
    print(f"  saturation:  learned {e['saturation_learned']:+.4f} ({e['saturation_learned_family']})"
          f"   analytic {e['saturation_analytic']:.12f}")
    for fname in ("F_released", "F_refit_matched"):
        s = e["S"][fname]
        ft = e["flip_threshold"][fname]
        print(f"    F = {fname:<16} S = {s['point']:+.4f}  95% CI "
              f"[{s['interval'][0]:+.4f}, {s['interval'][1]:+.4f}]  -> {s['reading']}  "
              f"({s['resample_count']} draws of {e['n']} circuits; T<=0 in "
              f"{s['frac_draws_T_le_0']*100:.2f}% of draws)")
        print(f"        to read PRESENT a descriptor control would need held-out R2 > "
              f"{ft['descriptor_control_r2_needed_for_S_above_0.5']:+.4f}; the strongest "
              f"descriptor-only predictor found reaches {ft['best_descriptor_only_r2_found_by_sweep']:+.4f} "
              f"(margin {ft['margin_r2']:+.4f})")

readings = {(p, f): final["panels"][p]["S"][f]["reading"] for p in PANELS
            for f in ("F_released", "F_refit_matched")}
final["all_readings"] = {f"{p}/{f}": v for (p, f), v in readings.items()}
final["reading_summary"] = ("every panel and every F variant reads ABSENT"
                            if set(readings.values()) == {"ABSENT"}
                            else f"mixed: {sorted(set(readings.values()))}")
print("\n" + "=" * 90)
print("READINGS: " + final["reading_summary"])
for k, v in final["all_readings"].items():
    print(f"   {k:<28} {v}")

fp = os.path.join(OUTDIR, "s06_final.json")
with open(fp, "w") as f:
    json.dump(final, f, indent=2)
print("\nwrote " + fp)
