# Frozen Rule: Shot-Count Sweep of the Descriptor-Information Experiment (QEMScore, Post-Review)

Written 2026-10-03 (US Pacific time), before any dataset of this sweep was generated at a shot count other than 2,048 and before any fit. A fifth internal review panel (2026-10-03) asked for this test. The results of rules `2026-10-02-descriptor-information.md` and `2026-10-03-strength-indicator.md` were known when this rule was written. So were the post hoc calibration, stacking, seed-ensemble, and inverse-variance analyses of those fits.

The prediction formula, the label thresholds, the choice of rungs, and the expectations below were set with the 2,048-shot strength-indicator results in view. Two internal critics applied an earlier draft of the formula to those results. A third check extrapolated the expected number of label-changing cells from the 2,048-shot residuals; no threshold changed after that extrapolation. The 2,048-shot level is therefore in-sample. It enters only the rank criterion, part (a) of H1, as one of the seven points of each series, and no other decision rule; it is otherwise reported separately. The only data generated before this rule was frozen is a regeneration of the 2,048-shot datasets, which reproduced the earlier files byte for byte.

## Question

The earlier rules hold the noisy estimate r at 2,048 shots. Post hoc, the measurement turned "adds" once the control's macro error reached roughly a third to a half of the macro calibration error of r. At N2 these were 0.016 to 0.022 against 0.034 to 0.064. It stayed "not distinguished" at a fifth or less (N1). That observation was never tested, because the precision of r was never varied. This rule varies it through the shot count, holding circuits, labels, splits, and descriptors fixed. It tests a prediction computed from error scales before any arm that reads r is fitted.

## Terms

A **rule cell** is one strength-by-observable combination inside a row: four per row (L1 or L3, z_mid or zz_mid). Each **classification cell** is one shot level of one row and rung: 7 levels × 6 rows × 4 rungs = 168 in all. c_m and c_r are per shot level and rule cell; c_C is per rung and rule cell. D/C*, the observed D/C, and the three labels are per classification cell. "Macro" is the equal-weight mean over a row's four rule cells.

## Design

- **Datasets.** The three regenerated primary datasets, with dataset seeds 101, 211, and 307, whose items.jsonl SHA-256 values are recorded under `datasets` in `artifacts/descriptor-information/gate/gate_r0.json`. Each family (transverse-field Ising, Heisenberg) has 640 training, 320 source-validation, and 160 test circuits, with depolarizing-plus-readout noise at L1 and L3.
- **Shot levels.** 256, 1,024, 2,048, 8,192, 32,768, and 131,072 shots, plus the exact noisy expectation. Seven levels.
- **Code.** The `qemscore` package is unchanged. The sweep adds `tools/shot_sweep.py` (generation and the three checks before any fit), `tools/shot_sweep_predict.py` (the predictions file), and `tools/shot_sweep_hypotheses.py` (H1 and H2). It also adds a `--sweep` mode to `tools/descriptor_ladder.py prepare` and `run`, with its dataset check in `tools/learner_seed_replication.py`, a `--part A` option of `tools/descriptor_ladder_posthoc.py`, and one output-only change to analysis A: a rung with no fit in any row of a family gets no rung statement, as in analysis B, while a rung fitted in only some rows stays "incomplete". With every rung fitted, analysis A's output is unchanged. All of this code, with its tests, is committed with this rule.
- **Generation path.** `tools/shot_sweep.py generate` takes every measurement group of the 2,048-shot datasets. At a finite level it rebuilds the group's circuit from its sidecar and calls `qemscore.sampling.sample_counts` unchanged. The call uses the group's stored sampler seed, its noise family and severity, and the transpile seed `split_generate.py` derives from its circuit identifier. Only the shot count changes. The generator asserts that the returned two-qubit gate count and compiled depth equal the stored ones. It writes `noisy_expectation` and `noisy_stderr` from `z_expectation_from_counts`, rounded to 12 decimals, and a counts sidecar. The 2,048-shot regeneration passes `validate_split_artifact`. The other finite levels are derived sweep artifacts, because their identifiers and sampler seeds remain those of the source, whose cell keys name 2,048 shots. Before `sweep_level.json` is written, `validate_sweep_artifact` checks the source with `validate_split_artifact` and the level against it. Items must match in inventory and order and differ only in the fields listed in check 3. Each counts sidecar must match its hash, total the shot count, and reproduce its item's estimate and standard error. The manifest's hashes, cells, counts, ledger, and sidecar table must match the items.
- **Copied fields.** Every other field is copied from the 2,048-shot rows: item, cell, circuit, and measurement-group identifiers (which still name the 2,048-shot cell), split, `ideal_expectation`, two-qubit gate count, and compiled depth. Labels are never recomputed. Each level directory records, in `sweep_level.json`, its shot level, its source's dataset hash and items.jsonl SHA-256, the generator's SHA-256, the software versions, and the platform.
- **Exact level.** r* = (1 − 2p_ro)^|S| · Tr(Z_S ρ). Here ρ is the Aer density-matrix state of the same transpiled circuit, with final measurements removed, under the severity's gate channels. |S| is the observable's support size (1 for z_mid, 2 for zz_mid), and p_ro is the symmetric readout flip probability (0.01 at L1, 0.06 at L3). The factor is exact because the readout error is the same symmetric, independent bit flip on every qubit. The generator refuses any other noise family. r* is rounded to 12 decimals and `noisy_stderr` is 0.
- **Shot field at the exact level.** Exact rows carry `shots` = `raw_circuit_evals` = 1,048,576 (2^20) as a placeholder, and their counts sidecar records the exact computation in place of counts. The level's manifest records `shot_level = "exact"`. The exact level is a derived sweep artifact. It passes `validate_sweep_artifact` without the count checks, and checks 2 and 3 cover it. Every level's log2-shots column is a finite constant within its dataset.
- **Sampling.** Each finite level is a separate sample of the same distribution, not a nested subsample. Aer chooses its simulation method as it did for the earlier datasets.
- **Rungs.** R0 (exact couplings), N1 (s = 0.01), N2 (s = 0.03), and R5 (no instance descriptors), as defined in the 2026-10-02 rule. In the strength-indicator fits, these span the control's macro error from about 0.003 (R0) to about 0.09 (Heisenberg R5) and 0.21 (transverse-field Ising R5).
- **Arms and features.** A once per dataset, rung, and level; C, F, and P at learner seeds 1 to 20. M is F at R5, the same fits, so E = M − F is defined at R0, N1, and N2. Every arm reads the noise-strength indicator of the 2026-10-03 rule. Candidates, preprocessing, the selection rule, schedules, and the permuted-column construction are unchanged.
- **Running the levels.** Each level has its own output tree with the original keys `shipped-s<seed>-n640`, and its datasets sit in directories whose names carry those keys. `tools/descriptor_ladder.py prepare --sweep` builds each tree's caches from that level's datasets. Only the 2,048-shot level can match the archived dataset hash. In place of that match, sweep mode checks the items hash against the level's manifest, and checks the test and validation circuit identifiers and ideal values against the campaign archive. Every other preparation step is unchanged, and check 3 compares the resulting caches. Fitting uses `tools/descriptor_ladder.py run --sweep --strength-indicator --rungs R0 N1 N2 R5 --gate-file artifacts/descriptor-information/gate/gate_r0.json` in three steps: (1) at the 2,048-shot level with `--arms A C`; (2) the predictions file is written, hashed, and timed; (3) at every level with `--arms A C F P`, which skips fits already present. The run log keeps both invocations at the 2,048-shot level. Both analysis scripts (with `--follow-up-rule` naming this file) and the reconcile script then run on each tree, with analysis A's statement change above.
- **Platform.** All fitting runs on NCSA DeltaAI (aarch64 Linux, Grace CPUs). Comparisons across shot levels use DeltaAI fits only. Every worker sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS to 1. Package versions match the earlier fits' records: numpy 2.2.6 and scikit-learn 1.9.1, with qiskit 2.5.2 and qiskit-aer 0.17.2 as in the environment that produced the earlier datasets. The run log records versions, numpy's BLAS, and `blas_fpe_probe()` before the first fit.
- **Reproduction gate.** The R0 reproduction gate of the 2026-10-02 rule is not a condition of this sweep. It certified the feature-builder hook, which is unchanged, and the 2026-10-03 rule already exempted indicator fits from it. The run passes the earlier record through `--gate-file`. Its R0 refits are also run on DeltaAI, and their largest difference from the archive is reported without a pass criterion.

## Checks Before Any Fit

1. **Faithfulness.** Before any other level is generated, the 2,048-shot regeneration must reproduce every measurement group's counts exactly. Every noisy estimate, standard error, gate count, and compiled depth must then match too, and items.jsonl must match byte for byte. If it passes on DeltaAI, all levels are generated there. When it fails, the number of differing groups and the largest difference in r are reported. All levels are then generated on the machine that produced the earlier datasets, after the same check passes there, and DeltaAI fits on those files, identified by SHA-256. If no machine passes, no fit runs.
2. **Exact level.** Over the same items at both levels, every item's 131,072-shot estimate must lie within six standard errors, sqrt((1 − r*²)/131,072), of r*. Otherwise no fit runs.
3. **Level caches.** `tools/shot_sweep.py check-caches` compares each level's caches with the 2,048-shot level's. Training, validation, test, and prediction rows must keep the same order. They may differ only in `shots`, `raw_circuit_evals`, `noisy_expectation`, `noisy_stderr`, `counts_hash`, `counts_sidecar`, and the shot entry of `axis_values`; the cache's recorded dataset hash differs by construction. Descriptor files, including the N-rung coupling-noise draws, must be byte-identical. Sweep preparation skips the R4 encoding, which no sweep rung reads.

## Prediction Computed Before Any Fit That Reads r

C and A read no r. Their only shot-dependent input is the log2-shots column, which is constant within a level and standardizes to exactly zero at power-of-two shot counts. The selection tie-break subtracts the same total |r − y| from both candidates. So C and A are fitted first, at the 2,048-shot level. Then `tools/shot_sweep_predict.py` computes, for each shot level, row, rung, and rule cell:
- c_m, the test mean absolute error of the de-attenuated estimate (r − a)/b. The intercept a and slope b come from a least-squares fit of r on the ideal value over the cell's validation rows (`np.polyfit(y_validation, r_validation, 1)`). They are used as fitted, whatever their sign. If |b| ≤ 1e-12, c_m is infinite and c_comb equals c_C, as in `tools/measurement_floor.py`, so cells with a near-zero b add almost nothing to D/C*;
- c_r, the test mean absolute error of the per-cell linear calibration of r (`np.polyfit(r_validation, y_validation, 1)` applied to test r), reported for Expectation 5;
- c_C, C's per-cell test error from this sweep's 2,048-shot C fits (mean over learner seeds 1 to 20); the same quantity from the 2026-10-03 fits is reported beside it;
- the predicted ceiling D/C* = 1 − macro(c_comb) / macro(c_C), with c_comb = (c_C^−2 + c_m^−2)^−1/2 in each rule cell.

c_m replaces c_r in the ceiling because c_r already shrinks toward the cell mean, which C also knows. The ceiling computed with c_r is reported beside it, descriptively. Before the file is written, each 2,048-shot rule cell must reproduce `per_cell_floor[<row>/<severity>/<observable>]` of `artifacts/descriptor-information/posthoc-measurement-floor.json` to 1e-12: c_r against `floor_mae`, a against `intercept_a`, and b against `slope_b`. If any value differs, no fit of F or P runs, and the difference is reported.

The script needs all seven levels and C's fits at all twenty learner seeds in every row and rung; a missing input stops it before anything is written. The file also lists every classification cell's predicted label, marks the label-changing cells (defined under H2), and states the number of cells that qualify for parts (b) and (c) of H1. Its SHA-256 and time are recorded in the run log before any fit of F or P. D/C* is a heuristic prediction that assumes independent, approximately unbiased errors and uses MAE as the error scale. Part (c) of H1 tests how often the observed improvement exceeds it; such exceedances do not by themselves establish circuit-dependent attenuation. The prediction and the outcome use the same test circuits. Beside c_m, the file reports c_m recomputed by two-fold cross-fitting. Each rule cell's validation rows, in cache order, are split into a first and a second half; a and b fitted on one half are scored on the other.

## Estimands, Uncertainty, and Classification

These follow rule `2026-10-03-strength-indicator.md`, and through it `2026-10-02-descriptor-information.md`, unchanged at each shot level:
- D = C − F; D/C as a ratio of means inside each draw; E = M − F; S only when A − F is positive in every draw;
- the two-stage percentile bootstrap (10,000 draws, a fresh `numpy.random.default_rng(20261002)` per row and quantity, learner seeds then whole test circuits);
- the three mutually exclusive labels; rung statements only when all three dataset seeds agree, otherwise "mixed"; the scaled endpoint upper(D)/mean(C) only beside the upper limit of the D/C interval.

The two analysis scripts must agree to 1e-12 at every level, with identical labels and rung statements. F − R, with its two-stage interval, is reported beside D at every classification cell by `tools/descriptor_ladder_posthoc.py --part A`. It is descriptive and outside the agreement criterion. Each level other than 2,048 is also analyzed with `--original-fits` pointing at the 2,048-shot DeltaAI fits, giving paired differences in D and D/C (descriptive). The 2,048-shot level is analyzed with `--original-fits` pointing at the 2026-10-03 fits (the platform check). `--original-fits` leaves every other quantity unchanged; H1 and H2 read these analyses.

## Hypotheses and Decision Rules

H1 and H2 are computed by `tools/shot_sweep_hypotheses.py` from the predictions file and analysis A, and again from analysis B. A missing level, prediction record, or analysis cell is an input error: the script stops, and it is rerun with complete inputs. Both computations must give identical results to 1e-12. If they differ, neither hypothesis is read until the difference is traced to a bug. The trace and the fix are reported as a deviation, and both computations are rerun. Spearman correlations use average ranks for ties and unrounded values.

**H1 (the gap follows the ceiling).** H1 uses N1 and N2: 12 series of three dataset seeds by two families by two rungs, each across the seven levels. It holds if all three parts hold.
- (a) In each of N1 and N2, at least five of the six series have a Spearman correlation of at least 0.8 between D/C* and the observed D/C (point estimates, seven levels). A series whose D/C* or observed D/C is constant across levels fails (a).
- (b) Over the classification cells of these series at the six levels other than 2,048 with D/C* ≥ 0.10, the median ratio of observed D/C to D/C* lies between 0.6 and 1.1.
- (c) In at most 10 percent of those cells does the lower limit of the observed D/C interval exceed D/C*.

Parts (b) and (c) need at least ten qualifying cells from at least two rows. The predictions file states the count; if it is smaller, H1 is not testable.

**H2 (the prediction locates the label change).** Each classification cell gets a predicted label. It is "measurement adds" if D/C* ≥ max(0.10, 2h), and "not distinguished" if D/C* ≤ 0.05; otherwise there is no prediction. Here h = (upper − lower)/2 of `parts.A.rows[<row>].rungs[<rung>].D_over_C.interval` in `artifacts/descriptor-information/strength-indicator/analysis-a.json`. The resulting "adds" thresholds, rounded to three decimals, are the thresholds used:

| Row | R0 | N1 | N2 |
|---|---|---|---|
| s101 transverse-field Ising | 0.385 | 0.159 | 0.100 |
| s101 Heisenberg | 0.446 | 0.208 | 0.128 |
| s211 transverse-field Ising | 0.490 | 0.184 | 0.100 |
| s211 Heisenberg | 0.360 | 0.225 | 0.108 |
| s307 transverse-field Ising | 0.547 | 0.100 | 0.100 |
| s307 Heisenberg | 0.247 | 0.194 | 0.100 |

A predicted label is label-changing when it differs from that row and rung's `classification.label` in the same file, reading "measurement hurts" as "not distinguished". Those labels are "not distinguished" at R0 and N1 in every row, where s101 transverse-field Ising R0 reads "measurement hurts". At N2 they are "measurement adds" in every row except s101 Heisenberg, which reads "not distinguished".
- H2 uses the classification cells of R0, N1, and N2 at the six levels other than 2,048.
- H2 holds if at least 80 percent of the label-changing cells carry the predicted label, and at least 80 percent of all predicted cells do.
- "Measurement hurts" counts as a miss for a predicted "adds" and as a match for a predicted "not distinguished"; the number of such matches is reported.
- If fewer than six cells are label-changing, or they come from fewer than two rows, H2 is not testable.

At N2, where h is 0.041 to 0.064, a correct D/C* just below 0.05 still reads "measurement adds" about half the time. This is accepted so that the 256-shot N2 cells stay in the test.

R0 and R5 series are reported beside H1, and R5 and the 2,048-shot level beside H2. None of them enters a decision rule, except that the 2,048-shot level is one of the seven points of each series in H1 part (a).

**Reading.** "The crossover" means this: at a fixed rung, the label turns to "measurement adds" near the shot level at which D/C* reaches that row and rung's "adds" threshold. At the N1 and N2 thresholds (0.100 to 0.225), this is where c_m falls to roughly 1.2 to 2.1 times c_C in each rule cell.
1. H1 and H2 hold: the paper may write that a ceiling computed from the two error scales before fitting predicted the size of the gap and where it became detectable. The claim is limited to 256 shots to the exact limit, this noise model, these two families, and 640 training circuits.
2. H1 holds and H2 fails: the gap grew with the precision of r as predicted, but the ceiling did not locate the label change.
3. H2 holds and H1 fails: the ceiling located the label changes but did not track the size of the gap.
4. Both fail: the crossover is not supported by this test, and only the descriptive 2,048-shot statement remains.
5. H1 or H2 not testable: the paper gives the count that fell short and draws no conclusion from that hypothesis. The other hypothesis is read alone. If H1 holds, the paper may write that the gap grew with the precision of r as predicted; if H2 holds, that the ceiling located the label changes. If it fails, item 4 applies.
6. A check before any fit fails: no fit runs, and the paper reports the sweep as not run, with the reason. The hypotheses script records this outcome from the failed check's report alone.

No outcome triggers a further reframing of the paper. Any additional analysis is labeled post hoc.

## Expectations Stated in Advance (Not Decision Rules)

1. c_m falls roughly as one over the square root of the shot count, down to a floor set by circuit-dependent noise effects at the exact level.
2. At R0, a transverse-field Ising row reads "measurement adds" at the exact level when its D/C* there reaches its H2 "adds" threshold (0.385, 0.490, and 0.547 for seeds 101, 211, and 307).
3. For transverse-field Ising, N2 keeps "measurement adds" at 2,048 shots (a platform check) and above, and loses it at 256 shots on at least one row.
4. R5 reads "measurement adds" at every level.
5. M's error at R5 lies close to the row's macro c_r at every level. It did so at 2,048 shots with the strength indicator.

## Checks Reported Beside Results

Before fitting: the three checks above, and c_m and c_r per level and rule cell. After fitting:
- **C and A identity.** C's and A's test predictions are expected to be bit-identical across the seven levels. The largest absolute difference is reported for C (per row, rung, and learner seed) and for A. A nonzero difference is reported with the fits responsible, and each level's estimands still use that level's own C and A.
- **Platform check.** The 2,048-shot DeltaAI fits are analyzed with `--original-fits` pointing at the 2026-10-03 strength-indicator fits (paired differences in D and D/C, descriptive). If a 2,048-shot label differs, both labels are reported. The 2026-10-03 record remains the primary 2,048-shot result, and this sweep's statements concern differences across levels on DeltaAI.

## Deviations

Any deviation from this rule is reported with its reason. On DeltaAI, `blas_fpe_probe()` runs before the first fit. Matrix-multiplication warnings count as spurious in the sensitivity analysis only if that probe reproduces them on finite inputs. Otherwise every RuntimeWarning counts in both flags. All fits are used.

## Release

The release adds the per-fit, per-candidate predictions of every level, the slim caches the analyses read, the level datasets and caches, the predictions file, the run log, and the generation commands. In the paper, this sweep appears as a follow-up whose rule was frozen before generation and fitting, motivated by review.
