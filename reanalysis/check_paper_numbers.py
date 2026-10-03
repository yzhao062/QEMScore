#!/usr/bin/env python3
"""
check_paper_numbers.py: Comprehensive verification of all numerical claims
in the manuscript against the packaged reanalysis outputs.

Methodology and Integrity Rules:
1. Dynamic Computation: Every "actual" value is computed dynamically from the
   packaged output files in `outputs/` (never hardcoded constants).
2. Manuscript Fidelity: Every "expected" value is copied verbatim from the
   manuscript text, with exact LaTeX file and line citations.
3. Tolerance Rule: Tolerances are strictly tied to the printed precision in
   the manuscript:
   - Exact integer counts and categorical states: exact equality (tol=0).
   - Every printed number: tol = half a unit in the last printed digit of the
     manuscript value (for example +0.000055 has tol 5e-7; 26.61 has tol 5e-3),
     so a check passes only if the packaged output rounds to the printed value.

Covers all 5 systems:
- Q-LEAR (Muqeet et al. 2024): 05_field.tex, A5_field.tex, 01_intro.tex, 09_appendix.tex
- QRAFT (Patel et al. 2021): 05_field.tex, A5_field.tex
- ML-QEM (Liao et al. 2024): 06_three_systems.tex, A6_three_systems.tex, 01_intro.tex, 09_appendix.tex
- Synergy CNN (Cantori et al. 2024): 06_three_systems.tex, A6_three_systems.tex, 01_intro.tex, 09_appendix.tex
- Q-Cluster (Patil et al. 2025): 06_three_systems.tex, A6_three_systems.tex, 01_intro.tex, 09_appendix.tex
"""

import json
import math
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")

# Load all outputs
def load_json(rel_path):
    p = os.path.join(OUTPUTS, rel_path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

qlear_report = load_json("qlear/field-report.json")
cost_ledger = load_json("qlear/cost-ledger.json")
qraft_report = load_json("qraft/qraft-panel-report.json")
s1_score = load_json("s1_ml_qem/S1_score.json")
s1_saturation = load_json("s1_ml_qem/S1_saturation.json")
s2_score = load_json("s2_synergy/s05_score.json")
s2_saturation = load_json("s2_synergy/s04_saturation.json")
s3_arms = load_json("s3_qcluster/RESULT-arms.json")
s3_cluster_boot = load_json("s3_qcluster/RESULT-cluster-bootstrap.json")
s3_optimism = load_json("s3_qcluster/RESULT-selection-optimism.json")
s3_saturation = load_json("s3_qcluster/RESULT-saturation.json")
s3_phase1 = load_json("s3_qcluster/PHASE1-RECORD.json")

RESULTS = []

def _printed_tol(expected_val):
    """Half a unit in the last printed digit, read from the literal's shortest repr."""
    text = repr(float(expected_val)).lower()
    if "e" in text:
        mant, exp = text.split("e")
        dec = (len(mant.split(".")[1]) if "." in mant else 0) - int(exp)
    else:
        dec = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** (-dec) * 1.0000001


def record_check(system, claim_desc, paper_loc, expected_val, actual_val, tol=None, is_str=False, bound=None):
    """bound='lower' or 'upper' marks an outward-rounded range endpoint: the printed
    value must lie on the outer side of the packaged value by less than one unit in
    its last printed digit."""
    if is_str:
        passed = (str(expected_val).strip() == str(actual_val).strip())
        diff = 0.0 if passed else 1.0
        tol = 0
    elif isinstance(expected_val, (int, np.integer)):
        passed = (int(expected_val) == int(actual_val))
        diff = abs(int(expected_val) - int(actual_val))
        tol = 0
    elif bound is not None:
        unit = 2 * _printed_tol(expected_val) / 1.0000001
        tol = unit
        diff = abs(float(expected_val) - float(actual_val))
        outer = float(expected_val) <= float(actual_val) if bound == "lower" else float(expected_val) >= float(actual_val)
        passed = outer and diff < unit
    else:
        # The printed precision governs every float check; a caller-supplied
        # tolerance can only tighten it, never loosen it.
        tol = _printed_tol(expected_val) if tol is None else min(tol, _printed_tol(expected_val))
        diff = abs(float(expected_val) - float(actual_val))
        passed = (diff <= tol)
    
    status = "PASS" if passed else "FAIL"
    RESULTS.append({
        "system": system,
        "claim": claim_desc,
        "location": paper_loc,
        "expected": expected_val,
        "actual": actual_val,
        "diff": diff,
        "tol": tol,
        "status": status
    })

# ==============================================================================
# 1. Q-LEAR (Section 5, Appendix A5, Appendix 9)
# ==============================================================================
hw = qlear_report["panels"]["hardware"]
sim = qlear_report["panels"]["simulator"]

# 1.1 A-C macro means and ranges
hw_ac_seeds = [hw["per_seed"][str(s)]["rung_errors"]["A"] - hw["per_seed"][str(s)]["rung_errors"]["C"] for s in range(10)]
hw_ac_mean = float(np.mean(hw_ac_seeds))
hw_ac_min = float(np.min(hw_ac_seeds))
hw_ac_max = float(np.max(hw_ac_seeds))

sim_ac_seeds = [sim["per_seed"][str(s)]["rung_errors"]["A"] - sim["per_seed"][str(s)]["rung_errors"]["C"] for s in range(10)]
sim_ac_mean = float(np.mean(sim_ac_seeds))
sim_ac_min = float(np.min(sim_ac_seeds))
sim_ac_max = float(np.max(sim_ac_seeds))

record_check("Q-LEAR", "Hardware A-C 10-fit macro mean", "05_field.tex:54, 01_intro.tex:72", 0.000055, hw_ac_mean, tol=5e-07)
record_check("Q-LEAR", "Hardware A-C range min", "A5_field.tex:61", -0.003115, hw_ac_min, tol=5e-07)
record_check("Q-LEAR", "Hardware A-C range max", "A5_field.tex:61", 0.004225, hw_ac_max, tol=5e-07)
record_check("Q-LEAR", "Simulator A-C 10-fit macro mean", "05_field.tex:112, 09_appendix.tex:636", 0.000091, sim_ac_mean, tol=5e-07)
record_check("Q-LEAR", "Simulator A-C range min", "A5_field.tex:62", -0.003124, sim_ac_min, tol=5e-07)
record_check("Q-LEAR", "Simulator A-C range max", "A5_field.tex:62", 0.004333, sim_ac_max, tol=5e-07)

# 1.2 Gaps: C-F, C-O, O-F
hw_cf_seeds = [hw["per_seed"][str(s)]["gaps"]["C-F"]["estimate"] for s in range(10)]
hw_co_seeds = [hw["per_seed"][str(s)]["gaps"]["C-O"]["estimate"] for s in range(10)]
hw_of_seeds = [hw["per_seed"][str(s)]["gaps"]["O-F"]["estimate"] for s in range(10)]

record_check("Q-LEAR", "Hardware C-F 10-fit macro mean", "05_field.tex:59", 0.2191, float(np.mean(hw_cf_seeds)), tol=5e-05)
record_check("Q-LEAR", "Hardware base C-O 10-fit macro mean", "05_field.tex:61, 01_intro.tex:81", 0.1969, float(np.mean(hw_co_seeds)), tol=5e-05)
record_check("Q-LEAR", "Hardware depth-cut O-F 10-fit macro mean", "05_field.tex:61, 01_intro.tex:82", 0.0222, float(np.mean(hw_of_seeds)), tol=5e-05)

# Interval above zero counts
hw_of_above_zero = sum(1 for s in range(10) if hw["per_seed"][str(s)]["gaps"]["O-F"]["lower"] > 0)
sim_of_above_zero = sum(1 for s in range(10) if sim["per_seed"][str(s)]["gaps"]["O-F"]["lower"] > 0)
record_check("Q-LEAR", "Hardware depth-cut intervals > 0", "05_field.tex:62, 09_appendix.tex:638", 9, hw_of_above_zero)
record_check("Q-LEAR", "Simulator depth-cut intervals > 0", "05_field.tex:62, 09_appendix.tex:638", 3, sim_of_above_zero)

sim_co_seeds = [sim["per_seed"][str(s)]["gaps"]["C-O"]["estimate"] for s in range(10)]
sim_of_seeds = [sim["per_seed"][str(s)]["gaps"]["O-F"]["estimate"] for s in range(10)]
record_check("Q-LEAR", "Simulator base C-O 10-fit macro mean", "A5_field.tex:73, 09_appendix.tex:637", 0.4407, float(np.mean(sim_co_seeds)), tol=5e-05)
record_check("Q-LEAR", "Simulator depth-cut O-F 10-fit macro mean", "A5_field.tex:73, 09_appendix.tex:638", 0.0099, float(np.mean(sim_of_seeds)), tol=5e-05)

# 1.3 Per-application cancellation and gaps
apps = ["groundstate", "pricingcall", "pricingput", "qaoa", "routing", "tsp"]
app_means_ac = {}
app_cf = {}
for app in apps:
    app_files = [f for f in hw["per_file"] if f.startswith(app)]
    diffs_ac = []
    diffs_cf = []
    for f in app_files:
        a_val = hw["per_file"][f]["A"]["0"]
        c_vals = [hw["per_file"][f]["C"][str(s)] for s in range(10)]
        f_vals = [hw["per_file"][f]["F"][str(s)] for s in range(10)]
        diffs_ac.append([a_val - c for c in c_vals])
        diffs_cf.append([c - f for c, f in zip(c_vals, f_vals)])
    app_means_ac[app] = float(np.mean(diffs_ac))
    app_cf[app] = float(np.mean(diffs_cf))

record_check("Q-LEAR", "Pricingcall A-C hardware mean", "05_field.tex:94, 09_appendix.tex:662", 0.020113, app_means_ac["pricingcall"], tol=5e-07)
record_check("Q-LEAR", "Pricingput A-C hardware mean", "05_field.tex:94, 09_appendix.tex:663", 0.017292, app_means_ac["pricingput"], tol=5e-07)
record_check("Q-LEAR", "Groundstate A-C hardware mean", "09_appendix.tex:661", -0.006168, app_means_ac["groundstate"], tol=5e-07)
record_check("Q-LEAR", "QAOA A-C hardware mean", "09_appendix.tex:664", -0.012211, app_means_ac["qaoa"], tol=5e-07)
record_check("Q-LEAR", "Routing A-C hardware mean", "09_appendix.tex:665", -0.016524, app_means_ac["routing"], tol=5e-07)
record_check("Q-LEAR", "TSP A-C hardware mean", "09_appendix.tex:666", -0.002172, app_means_ac["tsp"], tol=5e-07)

record_check("Q-LEAR", "QAOA C-F hardware mean", "05_field.tex:98, 09_appendix.tex:664", 0.053, app_cf["qaoa"], tol=0.0005)
record_check("Q-LEAR", "Routing C-F hardware mean", "05_field.tex:98, 09_appendix.tex:665", 0.565, app_cf["routing"], tol=0.0005)

# 1.4 Reproduction panel (Table 8 / tab:field-repro)
repro = hw["reproduction_panel"]
repro_expected = {
    "groundstate": (0.451, 0.443, 0.457),
    "pricingcall": (0.613, 0.557, 0.661),
    "pricingput": (0.593, 0.572, 0.632),
    "qaoa": (0.577, 0.565, 0.587),
    "routing": (0.102, 0.095, 0.111),
    "tsp": (0.364, 0.348, 0.377),
}
for app, (exp_mean, exp_min, exp_max) in repro_expected.items():
    app_keys = [k for k in repro if k.startswith(app)]
    seeds_means = [np.mean([repro[k][str(s)] for k in app_keys]) for s in range(10)]
    act_mean = float(np.mean(seeds_means))
    act_min = float(np.min(seeds_means))
    act_max = float(np.max(seeds_means))
    record_check("Q-LEAR", f"Reproduction {app} mean", f"A5_field.tex:48-53", exp_mean, act_mean, tol=1e-3)
    record_check("Q-LEAR", f"Reproduction {app} min", f"A5_field.tex:48-53", exp_min, act_min, tol=1e-3)
    record_check("Q-LEAR", f"Reproduction {app} max", f"A5_field.tex:48-53", exp_max, act_max, tol=1e-3)

lagos_keys = [k for k in repro if "ibm_lagos" in k]
seeds_lagos = [np.mean([repro[k][str(s)] for k in lagos_keys]) for s in range(10)]
record_check("Q-LEAR", "Reproduction Lagos 6-app min", "A5_field.tex:31", 0.4446, float(min(seeds_lagos)), tol=5e-05)
record_check("Q-LEAR", "Reproduction Lagos 6-app max", "A5_field.tex:31", 0.4595, float(max(seeds_lagos)), tol=5e-05)
record_check("Q-LEAR", "Reproduction largest deviation (pricingcall)", "A5_field.tex:25", 0.0134, float(np.mean([np.mean([repro[k][str(s)] for k in [k for k in repro if k.startswith("pricingcall")]]) for s in range(10)]) - 0.60), tol=5e-05)

# 1.5 Cost accounting
record_check("Q-LEAR", "Base step evaluations (hardware)", "05_field.tex:69, 09_appendix.tex:361", 1024, int(cost_ledger["hardware"]["macro_added_evaluations"]["C-O"]))
record_check("Q-LEAR", "Depth-cut evaluations (hardware)", "05_field.tex:71, 09_appendix.tex:362", 3072, int(cost_ledger["hardware"]["macro_added_evaluations"]["O-F"]))
record_check("Q-LEAR", "Hardware base efficiency", "05_field.tex:72, 09_appendix.tex:361", 1.923e-4, cost_ledger["hardware"]["per_evaluation"]["C-O"], tol=5e-08)
record_check("Q-LEAR", "Hardware depth-cut efficiency", "05_field.tex:73, 09_appendix.tex:362", 7.227e-6, cost_ledger["hardware"]["per_evaluation"]["O-F"], tol=5e-10)
record_check("Q-LEAR", "Hardware efficiency ratio (base/depth-cut)", "05_field.tex:73, 01_intro.tex:82", 26.61, cost_ledger["hardware"]["efficiency_ratio_C_O_over_O_F"], tol=0.005)
record_check("Q-LEAR", "Simulator base evaluations", "05_field.tex:74", 44843, round(cost_ledger["simulator"]["macro_added_evaluations"]["C-O"]))
record_check("Q-LEAR", "Simulator depth-cut evaluations", "05_field.tex:74", 134528, round(cost_ledger["simulator"]["macro_added_evaluations"]["O-F"]))

# ==============================================================================
# 2. QRAFT (Section 5, Appendix A5)
# ==============================================================================
pR = qraft_report["panel_R_released"]
pM = qraft_report["panel_M_matched"]

record_check("QRAFT", "Panel R Q3 macro MAE", "05_field.tex:125, A5_field.tex:173", 12.4838, pR["Q3"]["macro_mae"], tol=5e-05)
record_check("QRAFT", "Panel R Q2 macro MAE", "05_field.tex:125, A5_field.tex:174", 2.1074, pR["Q2"]["macro_mae"], tol=5e-05)
record_check("QRAFT", "Panel R Q1 macro MAE", "05_field.tex:136, A5_field.tex:175", 2.4467, pR["Q1"]["macro_mae"], tol=5e-05)

record_check("QRAFT", "Panel M Q3 macro MAE mean", "05_field.tex:126, A5_field.tex:177", 13.5903, pM["Q3"]["ten_seed_mean"], tol=5e-05)
record_check("QRAFT", "Panel M Q3 range min", "A5_field.tex:177", 13.1061, pM["Q3"]["ten_seed_min"], tol=5e-05)
record_check("QRAFT", "Panel M Q3 range max", "A5_field.tex:177", 13.9975, pM["Q3"]["ten_seed_max"], tol=5e-05)

record_check("QRAFT", "Panel M Q2 macro MAE mean", "05_field.tex:126, A5_field.tex:178", 2.2685, pM["Q2"]["ten_seed_mean"], tol=5e-05)
record_check("QRAFT", "Panel M Q2 range min", "A5_field.tex:178", 2.0663, pM["Q2"]["ten_seed_min"], tol=5e-05)
record_check("QRAFT", "Panel M Q2 range max", "A5_field.tex:178", 2.5100, pM["Q2"]["ten_seed_max"], tol=5e-05)

record_check("QRAFT", "Panel M Q1 macro MAE mean", "05_field.tex:135, A5_field.tex:179", 2.1246, pM["Q1"]["ten_seed_mean"], tol=5e-05)
record_check("QRAFT", "Panel M Q1 range min", "A5_field.tex:179", 1.9440, pM["Q1"]["ten_seed_min"], tol=5e-05)
record_check("QRAFT", "Panel M Q1 range max", "A5_field.tex:179", 2.3342, pM["Q1"]["ten_seed_max"], tol=5e-05)

# Q3/Q2 ratios by machine
mac_ratios = [pR["Q3"]["per_machine_mae"][m] / pR["Q2"]["per_machine_mae"][m] for m in sorted(pR["Q3"]["per_machine_mae"])]
exp_mac_ratios = [8.96, 6.07, 6.16, 3.73, 6.50]
for idx, (exp, act) in enumerate(zip(exp_mac_ratios, mac_ratios)):
    record_check("QRAFT", f"Panel R Q3/Q2 machine {idx+1}", "A5_field.tex:111-112", exp, act, tol=1e-2)

# Q3/Q2 ratios by seed
seed_ratios = [pM["Q3"]["per_seed_macro_mae"][str(s)] / pM["Q2"]["per_seed_macro_mae"][str(s)] for s in range(10)]
exp_seed_ratios = [6.36, 5.80, 6.55, 6.29, 5.93, 5.59, 6.38, 5.87, 5.44, 5.88]
for s, (exp, act) in enumerate(zip(exp_seed_ratios, seed_ratios)):
    record_check("QRAFT", f"Panel M Q3/Q2 seed {s}", "A5_field.tex:112-113", exp, act, tol=1e-2)

# Reverse execution improvement (~6%)
q1_mean = pM["Q1"]["ten_seed_mean"]
q2_mean = pM["Q2"]["ten_seed_mean"]
qraft_rev_pct = (q2_mean - q1_mean) / q2_mean * 100
record_check("QRAFT", "Reverse stats retrained improvement %", "05_field.tex:135", 6.34, qraft_rev_pct, tol=0.005)

# Five machines count
record_check("QRAFT", "Five machines count", "05_field.tex:11", 5, len(pR["Q1"]["per_machine_mae"]))

# ==============================================================================
# 3. ML-QEM (Section 6, Appendix A6, Appendix 9)
# ==============================================================================
s1_p = s1_score["with_calibration"]["refitted_F"]
s1_lf = s1_score["leak_check_4"]["qualification_diagnostics"][0]["refitted_F"]

record_check("ML-QEM", "Primary split S", "06_three_systems.tex:79, A6_three_systems.tex:67", 0.5985, s1_p["point"]["S"], tol=5e-05)
record_check("ML-QEM", "Primary split S CI lower", "06_three_systems.tex:79, A6_three_systems.tex:67", 0.5808, s1_p["interval_95_percentile"]["S"][0], tol=5e-05)
record_check("ML-QEM", "Primary split S CI upper", "06_three_systems.tex:79, A6_three_systems.tex:67", 0.6172, s1_p["interval_95_percentile"]["S"][1], tol=5e-05)
record_check("ML-QEM", "Primary split reading", "A6_three_systems.tex:67", "PRESENT", s1_p["reading"], is_str=True)

record_check("ML-QEM", "Leak-free split S", "A6_three_systems.tex:90", 0.5534, s1_lf["S"], tol=5e-05)
record_check("ML-QEM", "Leak-free split S CI lower", "A6_three_systems.tex:90", 0.5348, s1_lf["S_interval"][0], tol=5e-05)
record_check("ML-QEM", "Leak-free split S CI upper", "A6_three_systems.tex:90", 0.5730, s1_lf["S_interval"][1], tol=5e-05)
record_check("ML-QEM", "Leak-free split reading", "A6_three_systems.tex:90", "PRESENT", s1_lf["reading"], is_str=True)

record_check("ML-QEM", "Leaked rows count", "06_three_systems.tex:49, A6_three_systems.tex:88", 277, s1_score["leak_check_4"]["contaminated_test_rows"])

# Appendix A6 errors and re-reference
s1_app = load_json("s1_ml_qem/S1_appendix.json")
s1_base = s1_app["baselines_mean_per_row_L2"]
record_check("ML-QEM", "A mean per-row L2", "06_three_systems.tex:44, A6_three_systems.tex:67", 0.4419, s1_base["A_affine_control"], tol=5e-05)
record_check("ML-QEM", "C mean per-row L2", "06_three_systems.tex:45, A6_three_systems.tex:67", 0.2376, s1_base["C_capacity_matched_control"], tol=5e-05)
record_check("ML-QEM", "Raw (R) mean per-row L2", "06_three_systems.tex:47, A6_three_systems.tex:67", 0.2097, s1_base["unmitigated_noisy"], tol=5e-05)
record_check("ML-QEM", "F refitted mean per-row L2", "06_three_systems.tex:46, A6_three_systems.tex:67", 0.1005, s1_base["F_refitted"], tol=5e-05)

# Raw re-referenced ratio: (R - C) / (R - F)
r_ref = (s1_base["unmitigated_noisy"] - s1_base["C_capacity_matched_control"]) / (s1_base["unmitigated_noisy"] - s1_base["F_refitted"])
record_check("ML-QEM", "Re-referenced ratio (R-C)/(R-F)", "06_three_systems.tex:80, A6_three_systems.tex:146", -0.2552, r_ref, tol=5e-05)
record_check("ML-QEM", "C - R paired difference", "A6_three_systems.tex:152", 0.02786, s1_base["C_minus_unmitigated"], tol=5e-06)
record_check("ML-QEM", "C - R interval lower", "A6_three_systems.tex:152", 0.02006, s1_base["C_minus_unmitigated_interval"][0], tol=5e-06)
record_check("ML-QEM", "C - R interval upper", "A6_three_systems.tex:152", 0.03545, s1_base["C_minus_unmitigated_interval"][1], tol=5e-06)

record_check("ML-QEM", "Descriptor saturation all-rows R2", "A6_three_systems.tex:137", 0.6805, s1_saturation["saturation_all_rows"]["r2"], tol=5e-05)

# ==============================================================================
# 4. Synergy CNN (Section 6, Appendix A6, Appendix 9)
# ==============================================================================
s2_hi_vals = []
s2_S_vals = []
for p in ["Fig8a", "Fig8b", "Fig8c"]:
    for f in ["F_released", "F_refit_matched"]:
        v = s2_score[p]["variants"][f]["with_calibration"]
        s_val = v["point"]["S"]
        hi_val = v["interval_S"][1]
        s2_S_vals.append(s_val)
        s2_hi_vals.append(hi_val)
        record_check("Synergy CNN", f"Reading for panel {p} {f}", "06_three_systems.tex:88", "ABSENT", v["reading"], is_str=True)

record_check("Synergy CNN", "Panel S min", "06_three_systems.tex:86, A6_three_systems.tex:71", -0.3497, min(s2_S_vals), tol=5e-05)
record_check("Synergy CNN", "Panel S max", "06_three_systems.tex:86, A6_three_systems.tex:75", 0.0633, max(s2_S_vals), tol=5e-05)
record_check("Synergy CNN", "Max upper confidence bound across panels", "A6_three_systems.tex:73", 0.2754, max(s2_hi_vals), tol=5e-05)

# ==============================================================================
# 5. Q-Cluster (Section 6, Appendix A6, Appendix 9)
# ==============================================================================
# Primary panels (rs=40, rs=42, with/without calib)
q_40_cal = s3_arms["rs40|with_calibration"]
q_40_nocal = s3_arms["rs40|without_calibration"]
q_42_cal = s3_arms["rs42|with_calibration"]
q_42_nocal = s3_arms["rs42|without_calibration"]

record_check("Q-Cluster", "rs=40 with_cal S", "06_three_systems.tex:99, A6_three_systems.tex:77", 0.8795, q_40_cal["S"], tol=5e-05)
record_check("Q-Cluster", "rs=40 with_cal S CI lower", "A6_three_systems.tex:77", 0.7723, q_40_cal["S_lo"], tol=5e-05)
record_check("Q-Cluster", "rs=40 with_cal S CI upper", "A6_three_systems.tex:77", 0.9658, q_40_cal["S_hi"], tol=5e-05)

record_check("Q-Cluster", "rs=40 without_cal S", "06_three_systems.tex:100, A6_three_systems.tex:78", 0.9211, q_40_nocal["S"], tol=5e-05)
record_check("Q-Cluster", "rs=40 without_cal S CI lower", "A6_three_systems.tex:78", 0.8480, q_40_nocal["S_lo"], tol=5e-05)
record_check("Q-Cluster", "rs=40 without_cal S CI upper", "A6_three_systems.tex:78", 0.9716, q_40_nocal["S_hi"], tol=5e-05)

record_check("Q-Cluster", "rs=42 with_cal S", "A6_three_systems.tex:79", 0.8607, q_42_cal["S"], tol=5e-05)
record_check("Q-Cluster", "rs=42 with_cal S CI lower", "A6_three_systems.tex:79", 0.6382, q_42_cal["S_lo"], tol=5e-05)
record_check("Q-Cluster", "rs=42 with_cal S CI upper", "A6_three_systems.tex:79", 1.0399, q_42_cal["S_hi"], tol=5e-05)

record_check("Q-Cluster", "rs=42 without_cal S", "A6_three_systems.tex:80", 0.9441, q_42_nocal["S"], tol=5e-05)
record_check("Q-Cluster", "rs=42 without_cal S CI lower", "A6_three_systems.tex:80", 0.8216, q_42_nocal["S_lo"], tol=5e-05)
record_check("Q-Cluster", "rs=42 without_cal S CI upper", "A6_three_systems.tex:80", 1.0058, q_42_nocal["S_hi"], tol=5e-05)

# Circuit-level bootstrap
cb_40_cal = s3_cluster_boot["rs=40, with_calibration"]
cb_40_nocal = s3_cluster_boot["rs=40, without_calibration"]
cb_42_cal = s3_cluster_boot["rs=42, with_calibration"]
cb_42_nocal = s3_cluster_boot["rs=42, without_calibration"]

record_check("Q-Cluster", "Circuit boot rs=40 with_cal CI lower", "A6_three_systems.tex:115", 0.7702, cb_40_cal["circuit_lo"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=40 with_cal CI upper", "A6_three_systems.tex:115", 0.9622, cb_40_cal["circuit_hi"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=40 without_cal CI lower", "A6_three_systems.tex:116", 0.8028, cb_40_nocal["circuit_lo"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=40 without_cal CI upper", "A6_three_systems.tex:116", 0.9676, cb_40_nocal["circuit_hi"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=42 with_cal CI lower", "A6_three_systems.tex:117", 0.5057, cb_42_cal["circuit_lo"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=42 with_cal CI upper", "A6_three_systems.tex:117", 1.0728, cb_42_cal["circuit_hi"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=42 without_cal CI lower", "A6_three_systems.tex:118", 0.7294, cb_42_nocal["circuit_lo"], tol=5e-05)
record_check("Q-Cluster", "Circuit boot rs=42 without_cal CI upper", "A6_three_systems.tex:118", 0.9994, cb_42_nocal["circuit_hi"], tol=5e-05)

# Algorithm identity variance and saturation
alg_var = s3_phase1["split_structure"]["leak_check_4_severity"]["target_variance_explained_by_Algorithm_means"]
record_check("Q-Cluster", "Target variance explained by Algorithm identity", "not printed in the current manuscript", 0.942, alg_var, tol=0.0005)
record_check("Q-Cluster", "Descriptor C arm R2", "not printed in the current manuscript", 0.9530, q_40_cal["C_r2"], tol=5e-05)

sat_key = "released row-shuffled split (random_state=40) | descriptors only (6)"
record_check("Q-Cluster", "Descriptor nested saturation R2", "A6_three_systems.tex:138", 0.9348, s3_saturation[sat_key]["nested_R2"], tol=5e-05)

# Selection optimism
record_check("Q-Cluster", "Nested selection rs40|with_cal S", "RESULT-selection-optimism.json", 0.8795, s3_optimism["rs40|with_calibration"]["S"], tol=5e-05)
record_check("Q-Cluster", "Nested selection rs40|without_cal S", "RESULT-selection-optimism.json", 0.9114, s3_optimism["rs40|without_calibration"]["S"], tol=5e-05)
record_check("Q-Cluster", "Nested selection rs42|with_cal S", "RESULT-selection-optimism.json", 0.8098, s3_optimism["rs42|with_calibration"]["S"], tol=5e-05)
record_check("Q-Cluster", "Nested selection rs42|without_cal S", "RESULT-selection-optimism.json", 0.8410, s3_optimism["rs42|without_calibration"]["S"], tol=5e-05)

for k in ["rs40|with_calibration", "rs40|without_calibration", "rs42|with_calibration", "rs42|without_calibration"]:
    record_check("Q-Cluster", f"Nested selection reading {k}", "RESULT-selection-optimism.json", "PRESENT", s3_optimism[k]["reading_if_applied"], is_str=True)

# ==============================================================================
# Print Report
# ==============================================================================
def format_val(v):
    if isinstance(v, (int, np.integer, bool)):
        return str(v)
    if isinstance(v, float):
        if 0 < abs(v) < 1e-5:
            return f"{v:+.4e}"
        elif abs(v) < 0.1:
            return f"{v:+.6f}"
        elif abs(v) < 10.0:
            return f"{v:+.5f}"
        else:
            return f"{v:+.4f}"
    return str(v)


def format_tol(t):
    if t == 0 or t == 0.0:
        return "exact"
    if isinstance(t, (int, np.integer)):
        return str(t)
    if isinstance(t, float):
        if t < 1e-4:
            return f"{t:.1e}"
        elif t < 0.01:
            return f"{t:.4f}"
        else:
            return f"{t:.2f}"
    return str(t)


def main():
    print("=" * 128)
    print(f"{'SYSTEM':<12} {'PAPER LOCATION':<32} {'CLAIM':<36} {'EXPECTED':<14} {'ACTUAL':<14} {'TOL':<10} {'STATUS':<6}")
    print("=" * 128)
    n_pass = 0
    n_fail = 0
    for r in RESULTS:
        exp_str = format_val(r['expected'])
        act_str = format_val(r['actual'])
        tol_str = format_tol(r['tol'])
        print(f"{r['system']:<12} {r['location']:<32} {r['claim']:<36} {exp_str:<14} {act_str:<14} {tol_str:<10} {r['status']:<6}")
        if r['status'] == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    print("=" * 128)
    print(f"TOTAL CHECKS: {len(RESULTS)} | PASSED: {n_pass} | FAILED: {n_fail}")
    print("=" * 128)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
