> Note: Written 2026-10-01 before the corresponding run (file modification time 21:24 PDT on 2026-10-01; the near-Clifford fits started 04:31:56 UTC on 2026-10-02 = 21:31:56 PDT on 2026-10-01). Committed later, so the repository commit date postdates the runs.

# Frozen Design and Decision Rule: Near-Clifford Positive Control

Written 2026-10-01 before any data for this design were generated or fitted. Do not change it after results are seen; report any deviation explicitly.

## Purpose

The completed campaign (spin chains, S0) is a negative control: there the couplings identify the exact label, so a measurement-free model can match the full method. This design is the matching positive control. In it the descriptors withhold the circuit's Clifford structure and the positions of its non-Clifford gates, so the label is not determined by the learner-visible features. The question is whether the capacity-matched comparison then shows a positive gap D = C − F, that is, whether the instrument detects a learner that needs the measurement.

## Data

- **Setting.** S0 (in-distribution circuit instances), circuit family `near_clifford`, 10 qubits, `family_native_depth` = 4.
- **Family parameters.** Taken from the registered `t0-nc-micro` preset, not chosen for this run: `non_clifford_count` sampled from {1, 2, 3} and `theta` sampled from {0.4487989505128276, 0.6283185307179586}, per circuit.
- **Noise, observables, and shots.** The same as the completed campaign: noise `depolarizing_readout` at strengths L1 and L3, observables `z_mid` and `zz_mid`, 2048 shots.
- **Roles.** 640 training, 320 validation, and 160 untouched-test physical circuits per dataset.
- **Dataset seeds.** 101, 211, and 307, generated with the released split-v2 generator and validated with the released artifact validator. Record each dataset hash.

## Arms

For each dataset and each learner seed k in 1..20, fit the arms through the campaign's own code paths (`qemscore/campaign/analysis.py::_fit_arm`):

- **F (full):** `LiaoMitigator(random_state=k)`.
- **C (capacity-matched control):** `LiaoMitigator(random_state=k, drop_features=("noisy_expectation",))`.
- **P (training-shuffle refit):** the F recipe fitted on `shuffle_noisy_items(train, seed=k)`.

In addition, once per dataset (they do not depend on k):

- **A (affine feature-only control):** the registered `feat-only` method, fitted and selected through `_fit_arm`.
- **R (raw):** the noisy expectation scored as a predictor.

Also report the single campaign-style fit with k equal to the dataset seed, separately from the 20-seed statistics.

## Endpoints

Untouched-test macro MAE, equally weighted over the four strength-by-observable cells. Per dataset and seed, report:

- D_k = C_k − F_k and D_k / C_k;
- S_k = (A − C_k)/(A − F_k);
- P_k − C_k and P_k − F_k.

## Intervals

The two-stage percentile bootstrap of the frozen learner-seed rule: 10,000 draws, seed 20261001. Each draw resamples the 20 learner seeds, then resamples the 160 test circuits with all four rows of a circuit kept together, using one circuit resample shared across arms and seeds. Report the 2.5th to 97.5th percentile interval of mean D, mean S, and mean P − F.

## Decision Rule (Frozen)

- **Pass.** The positive control passes only if, for every one of the three dataset seeds, the two-stage 95 percent interval for mean D lies wholly above zero. The paper may then state that the capacity-matched comparison detects a learner that needs the measurement when descriptors under-determine the label.
- **Failure.** If any seed's interval reaches or crosses zero, the positive control fails, and the paper reports that failure plainly as a limit of the learner or the instrument.
- **Scope.** Either outcome says nothing about published mitigators, hardware, or other families.
- **Non-convergence.** Fits with non-finite values or numerical RuntimeWarnings are flagged. Results with flagged fits excluded are reported as sensitivity only. The decision uses all fits.
