# Round-10 Checks: Results

Rule: `docs/frozen-rules/2026-10-07-round10-checks.md`, pushed at commit 700912a. Its continuous-integration run was created at 2026-10-08T00:37:06Z; the first computation began at 00:37:20Z. Code: `tools/reading_rules.py`, `tools/bayes_oracle.py --sensitivity` with `tools/oracle_sensitivity.py`, `tools/seed_variance.py`, `tools/round10_summaries.py`, `reanalysis/scripts/qraft/score_qraft_machine_matched.py`, and `tools/rule_timestamps.py`. Every reproduction guard passed. D equals analysis A exactly on all 696 cells, and the stacking increments equal `posthoc-stacking-all.json` to 2.8e-17. The default oracle rerun equals the governed risks exactly. Part C reproduces Part D's pooled D and tau², and the published QRAFT arms reproduce exactly.

## Files

- `reading-rules.json`, `reading-rules.md`: Part A, five reading rules scored against the Bayes oracle per cell, with the reference selection, D_ref, and both stacking increments.
- `oracle-runs/*.json`, `oracle-sensitivity.json`, `oracle-sensitivity.md`: Part B, the seven oracle runs and the per-cell recount.
- `seed-variance.json`, `seed-variance.md`: Part C.
- `harm.json`, `dp.json`, `margins.json`, `mlqem.json`: Part E.
- `reanalysis/outputs/qraft/round10/qraft-machine-matched.json`: Part D.
- `artifacts/rule-timestamps/rule-timestamps.json`: Part F.

## Part A

In the 522-cell scope (fit identity), against 367 informative cells (365 resolved under Part B) and 66 no-information cells:

| Rule | Detected | Sensitivity (resolved) | False positives | Specificity |
|---|---|---|---|---|
| rule_D | 296 | 0.811 | 12 | 0.818 |
| rule_S | 322 | 0.882 | 15 | 0.773 |
| rule_R | 243 | 0.666 | 12 | 0.818 |
| rule_RS | 240 | 0.658 | 3 | 0.955 |
| rule_R_delta | 143 | 0.392 | 0 | 1.000 |

The frozen recommendation step selects `rule_RS` (the only eligible rule). Ref* is the degree-5 polynomial in 470 of 522 cells, including every R0 cell. The 53 informative cells that rule_D detects and rule_R does not lie at N1 and N2. Their median G* is 0.0015, against 0.011 for detected cells. In 60 informative cells (N1, N2, R3-Heis), the D_ref interval lies below zero. The three remaining rule_RS false positives are exact-level transverse-field Ising R0 cells (seeds 211, 307, 503).

Expectations: 1 to 5 held.

## Part B

384 oracle cells per run (both strength variants); 7 unresolved, covering four seed, family, rung, and shot-level cases, all at N1 and N2. Degree-7 surrogates lower the largest test error from 4.7e-4 to 1.7e-5 (Heisenberg) and from 7.9e-5 to 1.4e-6 (transverse-field Ising) and change G* by at most 3.2e-5. At N3, N4, R3-TFI, and R5, no variant changes G* by more than 1.1 percent. In the 522-cell scope, 2 informative cells (both misses) and 6 undetermined cells are unresolved; 296 of the 365 resolved informative cells read "F beats C".

Expectations: 1 to 4 held.

## Part C

Independent learner-seed resampling changes 3 of 240 per-seed labels; with the original candidates the width ratio has median 1.01 (0.86 to 1.89). With the original candidates, learner-seed variance is 6 to 200 times the circuit variance at R0 and N1; at N3, N4, R3-TFI, R4, and R5 the circuit variance is 9 to 520 times larger. tau² is zero in 27 of 40 rows; the largest dataset-seed fraction is 0.62 (transverse-field Ising N2, original candidates). All 40 first-versus-new Welch intervals contain zero.

Expectations: 1, 3, and 4 held. 2 failed: three R0 prediction intervals exclude zero, the two stronger-candidate exact-level rows (above zero) and stronger-candidate Heisenberg at 2,048 shots (below zero by at most 5e-6).

## Part D

| Contrast | Row split | Circuit-level groups |
|---|---|---|
| Q2m − Q1 | −0.043 [−0.072, −0.012] | −0.032 [−0.062, 0.0001] |
| Q2 − Q1nm | 0.051 [0.012, 0.091] | 0.088 [0.042, 0.141] |
| Q2 − Q2m | 0.187 [0.151, 0.224] | 0.200 [0.158, 0.251] |
| Q1nm − Q1 | 0.093 [0.069, 0.117] | 0.080 [0.054, 0.111] |
| Q3 − Q3m | −0.136 [−0.205, −0.066] | −0.154 [−0.250, −0.062] |
| Q2 − Q1 (post hoc, see the rule) | 0.144 [0.100, 0.189] | 0.169 [0.121, 0.226] |
| Q3m / Q2m | 6.59 [6.22, 6.98] | 6.42 [5.88, 7.06] |

Expectations: 1, 3, and 4 held. 2 failed: machine identity raises the descriptor arm's error under both splits.

## Part E

Harm rate of F, in percent of items with |F − y| > |r − y|:

- S0: at most 2.3 at R0 to N2; Heisenberg at most 0.4 at every rung; transverse-field Ising 5.3 and 6.7 at N3 (seeds 101 and 307) and 8.6 to 16.0 at N4, R3-TFI, and R5.
- S2: at most 1.6.
- S4: 27.9 to 53.1 (C: 19.7 to 50.3).

D_P agrees with D in 136 of 144 cells. Margin multipliers 0.025, 0.05, 0.10, 0.20 give 26, 26, 23, and 19 practically relevant rows of 40. Five rows move between relevant and negligible. ML-QEM: F − Rcal lies below zero in all 18 cells.

Expectations: 2 held. 1 failed (transverse-field Ising N3 on seeds 101 and 307, and N4, R3-TFI, R5, exceed 0.05). 3 failed (Heisenberg N1, seed 307, original candidates: D above zero, D_P contains zero). 4 failed.

## Part F

For each of the nine rules from 2026-10-02 to 2026-10-06, the rule reached GitHub before its results and before every named release upload, by 1.0 to 13.6 hours. The shift-transfer rule's time comes from its continuous-integration run (an upper bound), because the events snapshot omits that push. The two 2026-10-01 rules' results reached GitHub (2026-10-02T08:51:37Z) before their rule files (2026-10-03T07:44:17Z), as the rule states.

No deviation from the rule occurred.
