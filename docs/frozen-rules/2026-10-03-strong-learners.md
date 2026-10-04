# Frozen Rule: Strong-Learner Rerun of the Descriptor-Information Experiment (QEMScore, Post-Review)

Written 2026-10-03 (US Pacific time), before any fit of this rerun. Implementation details were added on 2026-10-04, before the rule was frozen. A fifth internal review panel (2026-10-03) asked whether the learners that define D are adequate. The results of rules `2026-10-02-descriptor-information.md` and `2026-10-03-strength-indicator.md` were known when this rule was written. The shot-sweep rule `2026-10-03-shot-sweep.md` was frozen and its shot levels were generated; none of its fits had been read when this rule was frozen. Two post hoc analyses of the earlier fits were also known. `tools/learner_adequacy.py` gives the per-seed spread and the seed-ensemble estimand. `tools/strongest_reference.py` gives the strongest descriptor-only reference and the stacking increment of r over it. At R0 a degree-five polynomial in the couplings errs 1.4 to 7.2 percent of the strength-indicator C's error, and stacking r on it changes the error by less than 3e-6. From N1 to R5 a gradient-boosted reference fitted per cell is within 0.98 to 1.70 times the strength-indicator C's error. In Part B at B-complete, F selects the random forest in all 60 fits and errs more than the raw estimate.

Two looks at the new candidates preceded this rule. A scoping study timed both on the dataset-seed-101 cache at R0 and N2. For one cell (transverse-field Ising, L1, z_mid) it computed the test error of an unregularized degree-five polynomial in the noisy couplings: 5.85e-3 at N1 and 17.16e-3 at N2. The ridge penalty was added after that look. The post hoc reference fits the same per-cell gradient-boosted grid on the same rows without r, and that model does not depend on its seed here. C's gradient-boosted candidate therefore reproduces the reference, and its test error was known at every Part A rung. Implementation tests and plumbing runs then fitted learner seed 1 on dataset seed 101, on macOS, at R0, N2, R3-TFI, R4, R5, and B-complete. Their outputs served only as implementation checks; at R0 and N2 they showed poly5_ridge selected for C and F.

## Question

The capacity-matched control C and the full arm F each choose between a random forest and an MLP with fixed settings. At R0 both are far from the error a polynomial in the couplings attains, and at B-complete F's random forest rarely splits on r. Does the ladder reading change when both arms may also choose a tuned gradient-boosted regressor and, where couplings exist, a ridge-regularized polynomial in the couplings?

## Change

Every arm that runs the candidate search gets new candidates beside the random forest and the MLP, which stay unchanged. These arms are C, F, and P, and M as F at the M rung.

**Composition and selection.** Each new candidate is one model per fit. In each of the fit's rule cells it holds a model fitted on that cell's training rows only. Part A has eight cells per fit (family, strength, observable); Part B has two (strength, with the zz_mid observable). The candidate predicts every validation, test, and prediction row with the model of that row's cell, so its prediction vector has the cached row order. A candidate enters a fit only if it has a model in every cell. For P, only the training rows are permuted (`shuffle_noisy_items`, shuffle seed equal to the learner seed, as now); grid points, alphas, and the selection use the unpermuted validation rows. The one-standard-error rule then runs once per fit, on all validation rows, as in `LiaoMitigator.fit`. Each candidate gets one score from `qemscore.baselines.liao._validation_score`, with the original candidates' shared circuit-evaluation cost, and the scores go to `select_one_standard_error`. The simplicity ranks are 0 (random forest), 1 (MLP), 2 (hgbr), and 3 (poly5_ridge). Because every validation circuit contributes the same number of rows, the rule selects the candidate with the lowest validation error; the threshold and the ranks act only on exact ties. The two family rows of a Part A dataset share one selected candidate per fit.

**hgbr.** `sklearn.ensemble.HistGradientBoostingRegressor` on the arm's own feature vector (C without r; F with r; P with its permuted r), with `random_state` equal to the learner seed, `early_stopping=False`, and other settings at their defaults. Grid points are ordered with max_iter outermost and max_leaf_nodes innermost: (200, 0.05, 15), (200, 0.05, 31), (200, 0.1, 15), (200, 0.1, 31), (500, 0.05, 15), (500, 0.05, 31), (500, 0.1, 15), (500, 0.1, 31) for (max_iter, learning_rate, max_leaf_nodes). In each cell the point with the lowest validation mean absolute error is used; ties go to the earlier point. With these settings `random_state` has no effect.

**poly5_ridge**, in Part A at R0, N1 to N4, R3-TFI, and R3-Heis; absent at R4, R5, and in Part B. In each cell it uses the couplings that the rung's feature vector carries for that cell's family. They are read from the rung builder's columns, never from the item's stored couplings, which are exact:

| Rung | Transverse-field Ising cells | Heisenberg cells |
|---|---|---|
| R0 | j, h | jx, jy, jz |
| N1 to N4 | noisy j, h | noisy jx, jy, jz |
| R3-TFI | j | jx, jy, jz |
| R3-Heis | j, h | jx, jy |

The other family's coupling columns and the strength indicator, both constant within a cell, are never in the design. Each coupling is standardized with the mean and population standard deviation (ddof 0) of the cell's training rows and expanded with `sklearn.preprocessing.PolynomialFeatures(5, include_bias=False)`, with no further scaling. For F and P the design appends r, standardized the same way, and its products with each standardized coupling. `sklearn.linear_model.Ridge(alpha, fit_intercept=True)`, other settings at their defaults, is fitted for alpha in {1e-6, 1e-4, 1e-2, 1, 1e2, 1e4}. The alpha with the lowest validation mean absolute error in the cell is used, ties to the smaller.

**Disclosed asymmetries.** The tuned candidates' validation errors are optimistic relative to the untuned random forest and MLP, which favors their selection equally in every arm; test rows enter no choice. The polynomial gives r only a linear term and its products with the couplings; whether a richer r design would raise or lower D is not established. In C and F, hgbr and poly5_ridge give the same predictions at every learner seed, because their inputs do not change with the seed. P's inputs change with its permutation. Where every seed of C and F selects them, the learner-seed stage of the bootstrap adds no variance to D, and D_ens equals D.

**Unchanged.** The three regenerated primary datasets at 2,048 shots and the Part B datasets, with their caches, circuits, noisy estimates, labels, and splits; the rung descriptors of the 2026-10-02 rule; the noise-strength indicator of the 2026-10-03 rule in every arm; the arms (A, C, F, P, M), preprocessing, and learner seeds 1 to 20. A is the affine control and does not run the candidate search. The near-Clifford datasets are not refitted.

**Code and fit records.** `tools/descriptor_ladder.py run` and `tools/qaoa_intermediate.py run` get a `--strong-learners` option. Fit files keep their names and format. NPZ keys are `validation`, `test`, and `validation__<name>` and `test__<name>` for random_forest, mlp, hgbr, and poly5_ridge (poly5_ridge only where fitted). The JSON keeps every existing field; `validation_scores` and `candidate_*_family_mae` cover every candidate. It adds `strong_learners: true`, `candidates`, `hgbr_grid_point` and `poly5_ridge_alpha` (maps from cell to the chosen setting), and `follow_up_rule` with this rule's path and SHA-256. `selected_equals_candidate_predictions` is checked for the selected candidate. The flag window of `tools/descriptor_common.py` is unchanged; hgbr and poly5_ridge are fitted after it, and their warnings are recorded per candidate. If one of them is selected, the warnings from its fitting and prediction enter both flags.

**Identity test.** A unit test covers dataset seed 101 at rungs R0, N2, R3-TFI, R4, and R5, arms C, F, and P, learner seed 1. Its reference is an oracle file of SHA-256 digests of every NPZ array and of the JSON fields. The oracle was written with the code of the shot-sweep commit, the last commit before this option. With the option off, every NPZ array must have the same name and digest, and every JSON field must be equal except `timing`, `seconds`, `pid`, and `versions.qemscore_path`. With the option on, the random-forest and MLP candidate predictions and their entries in `validation_scores` must equal those of the option-off fit exactly. The derived baseline of each option-on fit must equal the oracle under the same comparison as the option-off fit.

## Running

The fits run on NCSA DeltaAI (aarch64 Linux) with the package versions of the earlier fits (numpy 2.2.6, scikit-learn 1.9.1). Every worker sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS to 1. The run uses the prepared trees of the 2026-10-03 run, copied: the Part A caches, rung descriptors, and R4 encoder cache, and the Part B caches. Before any fit, the cache SHA-256 values must equal those recorded under `inputs.caches_used` in `artifacts/descriptor-information/strength-indicator/analysis-a.json` (Part A s101 74f90209…, s211 6e21ac24…, s307 0eccb3a9…; Part B s101 8c84d915…, s211 cca93477…, s307 5fcfdc2d…). Both run commands check this themselves and exit before the first fit on any difference. Part B reads its datasets through a copy of the QAOA data root whose `generation_report.json` names the DeltaAI paths; the dataset files are unchanged. Its run command rewrites the Part B caches from those datasets, so its check runs after that step. The fits go to a new, empty output tree; `run --strong-learners` refuses to start if its fits directory holds a fit JSON without `strong_learners: true`. Part A uses `tools/descriptor_ladder.py run --strength-indicator --strong-learners --rungs R0 N1 N2 N3 N4 R3-TFI R3-Heis R4 R5 --arms A C F P --seeds 1-20 --gate-file artifacts/descriptor-information/gate/gate_r0.json`; Part B uses `tools/qaoa_intermediate.py run --strength-indicator --strong-learners` with the same gate file. As in the shot-sweep rule, the R0 reproduction gate is not a condition. The run log records the commit, versions, numpy's BLAS, `threadpoolctl.threadpool_info()`, and `blas_fpe_probe()` before the first fit.

## Comparators

**Derived baseline (primary comparison).** Before its own selection, each strong fit records in an `option_off` object the selection, scores, flags, and warnings that the option-off code writes. From each strong fit, `tools/strong_learner_derive.py` writes a baseline fit that restores those values. Its `test` and `validation` arrays are the stored predictions of the candidate that `select_one_standard_error` picks from the random-forest and MLP scores alone. This is exactly what the option-off code gives on the same platform, at every rung. Analysis A runs with `--original-fits` pointing at the derived baseline: paired differences in D and D/C, with selection counts. Expectations 1, 4, and 5 are read against it. At R0, N1, N2, and R5 the derived baseline must equal the shot sweep's 2,048-shot DeltaAI fits of C and F; any difference is reported.

**macOS strength-indicator fits.** The derived baseline is also analyzed with `--original-fits` pointing at the 2026-10-03 fits (`descriptor-information-strength-v1`, `fits_sha256_digest` f858b871…). This is the platform effect at every rung, reported beside the primary comparison. Both comparisons are descriptive.

## Estimands, Uncertainty, and Classification

These follow rule `2026-10-02-descriptor-information.md` unchanged: D = C − F; D/C as a ratio of means inside each draw; E = M − F; S only when A − F is positive in every draw; the two-stage percentile bootstrap (10,000 draws, a fresh `numpy.random.default_rng(20261002)` per row and quantity, learner seeds then whole test circuits); the three labels; rung statements only when all three dataset seeds agree. Both analysis scripts run with `--follow-up-rule` naming this file and must agree to 1e-12, with identical labels and rung statements.

Stated in advance here, outside the agreement criterion, and computed by code committed with this rule:
- the seed-ensemble estimand D_ens and D_ens/E_C, with intervals and labels, and the pooled Part A R0 estimate (pooled D the mean of the six rows' D; pooled D/C the sum of D over the sum of mean C; one `default_rng(20261002)` stream; per draw and per dataset seed in the order 101, 211, 307, learner seeds drawn once, then transverse-field Ising circuits, then Heisenberg circuits), by `tools/learner_adequacy.py` with `--analysis-a` naming this rerun's analysis A;
- the fixed-learner D with its two-stage interval and descriptive label, for each candidate present in both arms, from the stored candidate predictions, by `tools/strong_learner_report.py`;
- per-family selection, descriptive: the one-standard-error rule applied to each family's validation rows alone, from the stored candidate predictions, with the resulting D per row, by the same script. At R3-TFI and R3-Heis the pooled selection is driven partly by the other family, whose couplings are intact;
- the selected candidate counts per arm, row, and rung.

For every cell whose label differs from the derived baseline's, the paper reports the paired D/C point difference and its 95 percent interval. It also reports both D point estimates, both D intervals, and their widths. It records separately whether the paired D/C interval excludes zero and whether the D interval narrowed, widened, or kept its width. These summaries do not identify the cause of a label change.

## Expectations Stated in Advance (Not Decision Rules)

1. At R0, mean C falls below one quarter of the derived baseline's mean C on all six Part A rows.
2. At R0, no Part A row reads "measurement adds".
3. On the three transverse-field Ising rows, N2, N3, N4, R3-TFI, R4, and R5 read "measurement adds".
4. At R5, where the polynomial is absent, D/C changes by less than 0.05 in absolute value on every Part A row (paired point difference).
5. At R0, poly5_ridge is the selected candidate for C in at least 15 of the 20 fits of every dataset seed.
6. At B-complete, mean F falls below the derived baseline's mean F on all three Part B rows.

Expectations 1, 4, and 5 follow almost surely from the post hoc reference and serve as implementation checks. Expectations 3 and 6 are the ones this rerun can fail. No label is predicted for N1 on any row, for the Heisenberg rows at N2, N3, N4, R3-Heis, R4, and R5, or for any Part B rung. Their statements are reported against the derived baseline.

## Reading

For each family, Part B, and rung, the rung statement of this rerun is set beside the derived baseline's.
1. If no rung statement changes, the paper may write that the strength-indicator ladder reading holds when both arms may also choose a tuned gradient-boosted regressor and a ridge-regularized coupling polynomial.
2. If a rung statement changes, the paper reports that rung with both statements wherever the ladder is presented. A main-text claim about that rung states that it depends on the candidate set; neither record is cited alone for it.
3. For each family, the first rung along R0, N1, N2, N3, N4 that reads "measurement adds" is reported under both records, and any shift is stated.

Where a cell reads "measurement hurts", mean P is reported beside mean F. No outcome triggers a change of candidates, grids, or rungs. A candidate set fitted after these results are seen is post hoc and does not replace either record in the main text.

## Deviations, Release, and Reporting

On DeltaAI, `blas_fpe_probe()` runs before the first fit. Matrix-multiplication warnings count as spurious in the sensitivity analysis only if that probe reproduces them on finite inputs. All fits are used in the primary estimands. Any deviation is reported with its reason.

The release adds every fit of this rerun with its per-candidate predictions, the derived baseline fits, and the run log. In the paper this rerun appears as a follow-up whose rule was frozen before fitting and whose motivation came from review. The 2026-10-03 fits remain the primary record of the strength-indicator design. Any analysis added after these results are seen is labeled post hoc.
