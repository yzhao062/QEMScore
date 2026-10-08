# Oracle-Free Reading Rules Scored Against the Bayes Oracle

Rule: `docs/frozen-rules/2026-10-07-round10-checks.md`, Part A. Primary rule: `rule_R`. Recommended rule: `rule_RS` (eligible: ['rule_RS']).

## Scope: all

| Group | Rule | Cells | Information | Detected | Sensitivity | No information | False positives | Specificity | Undetermined flagged |
|---|---|---|---|---|---|---|---|---|---|
| total | rule_D | 696 | 485 | 391 | 0.806 | 84 | 12 | 0.857 | 7 of 127 |
| total | rule_S | 696 | 485 | 429 | 0.885 | 84 | 27 | 0.679 | 21 of 127 |
| total | rule_R | 696 | 485 | 295 | 0.608 | 84 | 12 | 0.857 | 2 of 127 |
| total | rule_RS | 696 | 485 | 292 | 0.602 | 84 | 3 | 0.964 | 2 of 127 |
| total | rule_R_delta | 696 | 485 | 177 | 0.365 | 84 | 0 | 1.000 | 0 of 127 |
| first_2048 | rule_D | 174 | 131 | 102 | 0.779 | 30 | 0 | 1.000 | 1 of 13 |
| first_2048 | rule_S | 174 | 131 | 115 | 0.878 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_R | 174 | 131 | 92 | 0.702 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_RS | 174 | 131 | 92 | 0.702 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_R_delta | 174 | 131 | 71 | 0.542 | 30 | 0 | 1.000 | 0 of 13 |
| new_2048 | rule_D | 126 | 90 | 78 | 0.867 | 18 | 0 | 1.000 | 3 of 18 |
| new_2048 | rule_S | 126 | 90 | 81 | 0.900 | 18 | 0 | 1.000 | 9 of 18 |
| new_2048 | rule_R | 126 | 90 | 74 | 0.822 | 18 | 0 | 1.000 | 1 of 18 |
| new_2048 | rule_RS | 126 | 90 | 73 | 0.811 | 18 | 0 | 1.000 | 1 of 18 |
| new_2048 | rule_R_delta | 126 | 90 | 58 | 0.644 | 18 | 0 | 1.000 | 0 of 18 |
| other_finite | rule_D | 360 | 264 | 211 | 0.799 | 0 | 0 |  | 3 of 96 |
| other_finite | rule_S | 360 | 264 | 233 | 0.883 | 0 | 0 |  | 12 of 96 |
| other_finite | rule_R | 360 | 264 | 129 | 0.489 | 0 | 0 |  | 1 of 96 |
| other_finite | rule_RS | 360 | 264 | 127 | 0.481 | 0 | 0 |  | 1 of 96 |
| other_finite | rule_R_delta | 360 | 264 | 48 | 0.182 | 0 | 0 |  | 0 of 96 |
| exact | rule_D | 36 | 0 | 0 |  | 36 | 12 | 0.667 | 0 of 0 |
| exact | rule_S | 36 | 0 | 0 |  | 36 | 27 | 0.250 | 0 of 0 |
| exact | rule_R | 36 | 0 | 0 |  | 36 | 12 | 0.667 | 0 of 0 |
| exact | rule_RS | 36 | 0 | 0 |  | 36 | 3 | 0.917 | 0 of 0 |
| exact | rule_R_delta | 36 | 0 | 0 |  | 36 | 0 | 1.000 | 0 of 0 |

Verdict changes against rule_D (gained, lost) by oracle reading:

- rule_S: information: +55 / -17; no information: +24 / -9; undetermined: +15 / -1
- rule_R: information: +0 / -96; undetermined: +0 / -5
- rule_RS: information: +0 / -99; no information: +0 / -9; undetermined: +0 / -5
- rule_R_delta: information: +0 / -214; no information: +0 / -12; undetermined: +0 / -7

## Scope: fit_identity

| Group | Rule | Cells | Information | Detected | Sensitivity | No information | False positives | Specificity | Undetermined flagged |
|---|---|---|---|---|---|---|---|---|---|
| total | rule_D | 522 | 367 | 296 | 0.807 | 66 | 12 | 0.818 | 5 of 89 |
| total | rule_S | 522 | 367 | 322 | 0.877 | 66 | 15 | 0.773 | 13 of 89 |
| total | rule_R | 522 | 367 | 243 | 0.662 | 66 | 12 | 0.818 | 2 of 89 |
| total | rule_RS | 522 | 367 | 240 | 0.654 | 66 | 3 | 0.955 | 2 of 89 |
| total | rule_R_delta | 522 | 367 | 143 | 0.390 | 66 | 0 | 1.000 | 0 of 89 |
| first_2048 | rule_D | 174 | 131 | 102 | 0.779 | 30 | 0 | 1.000 | 1 of 13 |
| first_2048 | rule_S | 174 | 131 | 115 | 0.878 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_R | 174 | 131 | 92 | 0.702 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_RS | 174 | 131 | 92 | 0.702 | 30 | 0 | 1.000 | 0 of 13 |
| first_2048 | rule_R_delta | 174 | 131 | 71 | 0.542 | 30 | 0 | 1.000 | 0 of 13 |
| new_2048 | rule_D | 84 | 60 | 53 | 0.883 | 12 | 0 | 1.000 | 2 of 12 |
| new_2048 | rule_S | 84 | 60 | 54 | 0.900 | 12 | 0 | 1.000 | 6 of 12 |
| new_2048 | rule_R | 84 | 60 | 51 | 0.850 | 12 | 0 | 1.000 | 1 of 12 |
| new_2048 | rule_RS | 84 | 60 | 50 | 0.833 | 12 | 0 | 1.000 | 1 of 12 |
| new_2048 | rule_R_delta | 84 | 60 | 39 | 0.650 | 12 | 0 | 1.000 | 0 of 12 |
| other_finite | rule_D | 240 | 176 | 141 | 0.801 | 0 | 0 |  | 2 of 64 |
| other_finite | rule_S | 240 | 176 | 153 | 0.869 | 0 | 0 |  | 7 of 64 |
| other_finite | rule_R | 240 | 176 | 100 | 0.568 | 0 | 0 |  | 1 of 64 |
| other_finite | rule_RS | 240 | 176 | 98 | 0.557 | 0 | 0 |  | 1 of 64 |
| other_finite | rule_R_delta | 240 | 176 | 33 | 0.188 | 0 | 0 |  | 0 of 64 |
| exact | rule_D | 24 | 0 | 0 |  | 24 | 12 | 0.500 | 0 of 0 |
| exact | rule_S | 24 | 0 | 0 |  | 24 | 15 | 0.375 | 0 of 0 |
| exact | rule_R | 24 | 0 | 0 |  | 24 | 12 | 0.500 | 0 of 0 |
| exact | rule_RS | 24 | 0 | 0 |  | 24 | 3 | 0.875 | 0 of 0 |
| exact | rule_R_delta | 24 | 0 | 0 |  | 24 | 0 | 1.000 | 0 of 0 |

Verdict changes against rule_D (gained, lost) by oracle reading:

- rule_S: information: +38 / -12; no information: +12 / -9; undetermined: +9 / -1
- rule_R: information: +0 / -53; undetermined: +0 / -3
- rule_RS: information: +0 / -56; no information: +0 / -9; undetermined: +0 / -3
- rule_R_delta: information: +0 / -153; no information: +0 / -12; undetermined: +0 / -5

