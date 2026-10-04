# Frozen Rule: Strength-Indicator Rerun of the Descriptor-Information Experiment (QEMScore, Post-Review)

Written 2026-10-03, before any fit with the strength indicator. A fourth independent review panel (2026-10-03) raised the design question this rule tests. The original descriptor-information results (rule `2026-10-02-descriptor-information.md`) were known when this rule was written. So was the post hoc per-cell calibration and stacking analysis that motivates it.

## Question

In the original experiment, every learned arm reads a feature vector without the noise strength, although training pools the two strengths (L1 and L3). One noisy value can therefore correspond to different ideal values at the two strengths, and the full arm F must infer the strength from the noisy estimate. The per-cell calibration and stacking references are fitted per cell and so know the strength. Does giving every arm the strength change the capacity-matched comparison along the descriptor ladder?

## Change

Every arm reads one extra feature: the item's noise-strength level, coded 0 for L1 and 1 for L3. This holds at every rung of Parts A and B and on the near-Clifford datasets. Nothing else changes:
- the same datasets, caches, circuits, noisy estimates, labels, and splits;
- the same rungs and rung descriptors (R0, N1 to N4, R3-TFI, R3-Heis, R4, R5; B-partial, B-complete, B-none; NC-R0 and NC-none);
- the same arms (A, C, F, P, M), candidate families, preprocessing, selection rule, and learner seeds 1 to 20;
- the permuted-column refit P permutes the noisy-estimate column only, never the strength column.

R0 is refitted with the indicator; it does not reuse the Appendix M.4 fits. No gate applies to the new fits, because their features differ from the archived ones. The code must show, in a unit test, that with the indicator off the feature builder returns exactly the original vectors, and with it on returns the original vector plus one final column.

## Estimands, Uncertainty, and Classification

These follow rule `2026-10-02-descriptor-information.md` unchanged:
- D = C − F; D/C as a ratio of means inside each draw; E = M − F; S only when A − F is positive in every draw;
- the two-stage percentile bootstrap: 10,000 draws, a fresh `numpy.random.default_rng(20261002)` per row and quantity, learner seeds and then whole test circuits;
- the three mutually exclusive cell labels;
- rung-level statements only when all three dataset seeds agree, otherwise "mixed".

The scaled endpoint upper(D)/mean(C) is reported only beside the upper limit of the D/C interval, never in its place.

**Comparison with the original experiment.** For each row and rung, report the paired difference in D and in D/C between the strength-indicator fits and the original fits. Both use the same seed and circuit indices in each draw. This comparison is descriptive.

## Expectations Stated in Advance (Not Decision Rules)

1. R0 stays "not distinguished" on all six Part A rows.
2. On the three transverse-field Ising rows, D/C at R5 and at R3-TFI is higher than in the original fits (point estimates).
3. On the three transverse-field Ising rows, M's test error at R5 falls below 0.06; it was 0.089 to 0.094 without the indicator.
4. Transverse-field Ising turns "measurement adds" no later than N2, as before; it may turn earlier.
5. On the Heisenberg rows, the indicator changes little, because the L3 noisy estimate is nearly uninformative there.
6. No expectation is stated for Part B or for the near-Clifford datasets.

## Agreement and Release

The two independent analysis scripts must agree on point estimates and interval bounds to 1e-12, with identical labels and rung statements, as under the second amendment of the original rule. The release adds the per-fit, per-candidate predictions of the new fits beside the original asset. Any analysis added after these results are seen is labeled post hoc.

## Reporting

The paper reports these fits as a follow-up experiment whose rule was frozen before fitting and whose motivation came from review. Its results sit beside the original ladder, which stays the primary record of the frozen descriptor-information rule.
