# Bayes Oracle and Six-Seed Pooling: Results

Rule: `docs/frozen-rules/2026-10-06-round9-follow-ups.md`, Parts C and D (pushed at commit 82aaaf6 before either ran). Code: `tools/bayes_oracle.py` (with `tools/bayes_oracle_mapping.json` and `tools/bayes_oracle_strength_evidence.json`) and `tools/pooled_absolute.py`.

## Files

- `bayes-oracle.json`, `bayes-oracle.md`: surrogates, shot-noise checks, and oracle risks per seed, family, rung, level, and strength variant. They also hold the Equation 2 rows for all 696 included tuples (504 excluded with reasons), the 16 redrawn items, and the confusion tables of 44 fit sets.
- `pooled-absolute.json`, `pooled-absolute.md`: per-seed inputs, DerSimonian–Laird pooled D with HKSJ intervals, I², margins, and practical verdicts for three fit sets.

## Part C

All 72 surrogates have test MAE below 5e-4 (none flagged). Shot-noise z SD is 0.997 to 1.006 at every finite level. Equation 2 holds to 8.3e-17 on all 696 rows.

Mean G*/C* at 2,048 shots over six seeds (observed strength):

| Family | N1 | N2 | N3 | N4 | R3 | R5 |
|---|---|---|---|---|---|---|
| Transverse-field Ising | 0.041 | 0.197 | 0.521 | 0.749 | 0.834 | 0.829 |
| Heisenberg | 0.016 | 0.087 | 0.269 | 0.374 | 0.074 | 0.379 |

Where G* is large (N3, N4, R3-TFI, and R5 at 2,048 shots), 138 rows qualify. Their median |D − G*|/G* is 0.05. The median D/G* is 0.99 to 1.05 per fit set.

Confusion over 696 cells:

- 391 read "F beats C" with information.
- 94 are misses, at N1, N2, and R3-Heis, with G* at most 0.0015.
- 12 are false positives: R0 at the exact level in `follow_b_exact` and `fresh_strong_exact`, with D of 4e-5 to 3e-4.
- None has the wrong sign; 127 are undetermined.

Expectations:

1. Every surrogate below 1e-3: held.
2. z SD in [0.9, 1.1]: held.
3. G* interval above zero at 2,048 shots at N2, N3, N4, R3-TFI, R3-Heis, and R5 on every row: failed. The exceptions are Heisenberg N2 on seed 607 and R3-Heis on seeds 101, 401, and 607.
4. G*/C* rises from N1 to N4 on every row: held.
5. Hidden-strength G* at most observed-strength G*: failed in 30 of 216 row-level comparisons, all at N1 and N2, by at most 8.3e-5.

## Part D

Margins: delta = 0.0037 (transverse-field Ising) and 0.0059 (Heisenberg). Both 2,048-shot sets read a practically relevant gain at N3, N4, R4, R5, and R3-TFI. Every R0 and every 2,048-shot N1 reads practically negligible. Transverse-field Ising N2 is undetermined at 2,048 shots.

Expectations:

1. Original candidates with the strength indicator, R0, interval contains zero for both families: held for Heisenberg. It failed for transverse-field Ising, at −0.00037 [−0.00067, −0.00006], which is practically negligible.
2. Stronger candidates, exact level, R0, practically negligible for both families: held.
3. R5 practically relevant in every fit set: held.

No deviation from the rule occurred.
