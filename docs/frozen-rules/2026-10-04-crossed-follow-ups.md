# Frozen Rule: Crossed Follow-Ups on the Released Datasets (QEMScore, Post-Review)

Written 2026-10-04 (US Pacific time), before any fit named here. A seventh internal review panel (2026-10-04) asked six further questions of the released spin-chain datasets. Every result of the rules `2026-10-02-descriptor-information.md`, `2026-10-03-strength-indicator.md`, `2026-10-03-shot-sweep.md`, and `2026-10-03-strong-learners.md` was known when this rule was written. So were the post hoc analyses released with them. A reviewer's A/A computations on the released learner-seed fits were also known. Single-fit intervals excluded zero for 85.8 percent of same-pipeline seed pairs, and two-stage intervals on random 10-versus-10 splits for about 9.7 percent. The companion rule `2026-10-04-fresh-confirmation.md` covers new dataset seeds. This rule reuses the released datasets of seeds 101, 211, and 307 and their shot levels, so its results are conditional on those datasets.

## Common Design

**Data and trees.** The released datasets and the prepared trees of the shot sweep at its seven levels (256, 1,024, 2,048, 8,192, 32,768, and 131,072 shots, and the exact level). Before any fit, each tree's cache SHA-256 values must equal those under `inputs.caches_used` in that level's analysis A of the shot sweep. Every run passes that file as `--cache-record` and exits before the first fit on any difference. G reads the prepared tree of the 2026-10-03 run, which holds the R4 encoder cache; its caches are checked against the strength-indicator analysis A. The stages are in `artifacts/descriptor-information/round8/scripts/round8_stage.sh`.

**Fits.** Every arm reads the noise-strength indicator of the 2026-10-03 rule. Unless stated otherwise, arms A, C, F, and P run at learner seeds 1 to 20 with the candidates, preprocessing, selection rule, schedules, and permuted column of the earlier rules. M is F at R5. Every fit runs on NCSA DeltaAI with numpy 2.2.6 and scikit-learn 1.9.1, single-threaded (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS set to 1). Each fit set writes into its own new, empty fits directory. The run log records the commit, versions, numpy's BLAS, `threadpoolctl.threadpool_info()`, and `blas_fpe_probe()` before the first fit.

**Estimands.** These follow the earlier rules unchanged. D = C − F, and D/C is a ratio of means inside each draw. Intervals come from the two-stage percentile bootstrap: 10,000 draws, a fresh `numpy.random.default_rng(20261002)` per row and quantity, learner seeds then whole test circuits. The three labels read "F beats C" (`measurement_adds`), "C beats F" (`measurement_hurts`), and "not distinguished" in the paper. Rung statements need all three dataset seeds to agree; otherwise they read "mixed". Both analysis scripts run on every fit set with `--follow-up-rule` naming this file and must agree to 1e-12 with identical labels and rung statements.

**Comparator.** Unless stated otherwise, each new fit set is compared with the shot sweep's original-candidate DeltaAI fits at the same level, through `--original-fits` (paired differences in D and D/C). These comparisons are descriptive.

## B. Strong Candidates Crossed with Shot Precision

**Fits.** `--strong-learners`, as in the 2026-10-03 strong-learner rule, at R0, N1, N2, and R5, at the six levels other than 2,048 shots. The 2,048-shot strong fits are those of the strong-learner rerun. `tools/strong_learner_derive.py` writes each new fit's derived baseline, which must equal the shot sweep's original-candidate fit at the same level; any difference is reported.

**Reported.** For each level, row, and rung: labels, D, D/C, and intervals under both candidate sets, and selection counts. For each row at R0, N1, and N2: the first of the seven levels at which it reads "F beats C" under each set. The validation-only ceiling of the shot sweep is reported beside the strong-candidate D/C, descriptively.

**Expectations (not decision rules).**
1. At R5, every row reads "F beats C" at every level.
2. At N2, every row reads "F beats C" at 8,192 shots and above.
3. At R0, no row reads "F beats C" at 256 or 1,024 shots.

No label is predicted at R0 above 1,024 shots or at N1. **Reading.** Where the strong set reads "F beats C" at R0 at some level, the paper gives that level and candidate set wherever it states the R0 negative control.

## C. A Well-Trained MLP

**Fits.** `--neural-es`: the MLP candidate stops on the source-validation rows (`stopping_rule="validation_patience"`, patience 50 epochs with `min_delta` 0, at most 2,000 epochs, best epoch restored). Its Adam step size halves after 10 epochs without improvement, down to 1e-5. The random-forest candidate, features, selection rule, and everything else are unchanged. Rungs R0, N1, N2, and R5, at 2,048 shots and the exact level. Each fit records the epochs run, the best epoch, and the final step size.

**Disclosed asymmetry.** Early stopping reads the validation rows that the selection rule also reads. Reusing these rows can make the MLP's validation error optimistic relative to the random forest's, which favors its selection. The same procedure applies in every arm, but the amount of optimism can differ across arms; selection counts per arm are reported.

**Reported.** Labels, D, D/C, and intervals; the seed-ensemble estimand D_ens and D_ens/E_C by `tools/learner_adequacy.py`; and selection counts. Also the MLP's epochs run and best epoch, and its validation and test errors beside the original MLP's.

**Expectations (not decision rules).**
1. At R0 and 2,048 shots, the MLP candidate's mean validation error in C falls below the original MLP's on every row.
2. At R0 and 2,048 shots, no row reads "F beats C".
3. At R5, every row reads "F beats C" at both levels.

**Reading.** For each family and rung, the rung statement is set beside the original candidates' at the same level. A changed statement is reported wherever that rung's result appears, with both candidate sets named.

## D. A/A Calibration of the Decision Interval

**Fits.** Learner seeds 21 to 40 of C and F (original candidates) at R0, N1, N2, and R5, at 2,048 shots, using the shot sweep's 2,048-shot tree. Seeds 1 to 20 are the shot sweep's 2,048-shot DeltaAI fits, so both sets come from one platform and one tree.

**Analysis** by `tools/aa_calibration.py`, for each row, rung, and arm (48 comparisons):
1. *Two-stage A/A (primary).* The statistic is the mean macro error of seeds 1 to 20 minus that of seeds 21 to 40. Each of 10,000 draws resamples the 20 seeds of each set independently, then whole test circuits shared by both sets, with a fresh `default_rng(20261002)` per comparison. The outcome is whether the 95 percent percentile interval excludes zero.
2. *Random partitions.* 200 random partitions of the 40 seeds into two sets of 20 (partitions from `default_rng(20261003)`), each scored as in item 1 with 2,000 draws. The outcome is the fraction of partitions excluding zero, per comparison and overall.
3. *Single-fit A/A.* For every pair of distinct seeds among 1 to 20 of one arm: the difference of the two fits' macro errors, with a circuit-only bootstrap interval. The interval uses 2,000 draws from `default_rng(20261004)`, and the outcome is the fraction excluding zero.
4. *The label rule under A/A.* The rule's label computed from seeds 1 to 20 and from seeds 21 to 40, and the fraction of row-rung cells whose two labels differ.

The JSON output records every interval, the summaries, and the SHA-256 values of its inputs and of the script.

**Expectations (not decision rules).**
1. The overall fraction in item 2 lies between 0.02 and 0.10.
2. The fraction in item 3 exceeds 0.5.

**Reading.** The paper reports the rates of items 1 to 4 in the main text whatever they are. The rates are conditional on the released training data and test circuits. The 200 partitions overlap, so they are not 200 independent experiments, and a single-fit difference can be a real difference between two fitted models. The paper does not present the rates as coverage over newly drawn training sets or as a general false-positive rate of C − F. If the overall rate of item 2 exceeds 0.10, the paper states that rate wherever it reads a label whose interval endpoint lies within one draw standard deviation of zero.

## E. A Measurement-Free Stack

**Inputs.** No new fits. Ĉ is the prediction of the shot sweep's original-candidate C fit at each level, rung, and learner seed. g is the `poly5_ridge` candidate prediction of the strong-learner rerun's C fit at 2,048 shots. It has the same dataset seed, rung, and learner seed. It is a degree-five ridge polynomial in the rung's couplings and reads no r. C reads no r, so Ĉ and g are the same at every level; only r changes.

**Analysis** by `tools/measurement_free_stack.py`, at R0, N1, and N2 at every level. Per strength-by-observable cell and learner seed, it fits four least-squares models on the validation rows. Each is scored on the test rows:
- recalibration, y ~ a + bĈ;
- the stack with r, y ~ a + bĈ + cr;
- the measurement-free stack, y ~ a + bĈ + cg;
- the stack with both, y ~ a + bĈ + cg + dr.

From their macro test errors it reports three increments with two-stage intervals. The increment inc_r is recalibration minus the stack with r, as in `tools/stacking_increment_all.py`. inc_g is recalibration minus the measurement-free stack. inc_r|g is the measurement-free stack minus the stack with both. Each relative increment divides by the error it subtracts from. The script checks that inc_r reproduces `posthoc-stacking-all.json` to 1e-12 at every cell. It also checks that g is identical across learner seeds; if g differs, it reports the largest difference and uses each seed's own g.

**Disclosed asymmetry.** g's ridge penalty was chosen on the same validation rows that fit the stack's coefficients, so the measurement-free stack is slightly optimistic.

**Expectations (not decision rules).**
1. At R0, the relative inc_g exceeds the relative inc_r on every row at every level.
2. At R0 and the exact level, the relative inc_r|g is below 0.05 on every row.
3. At N2 and the exact level, the interval of inc_r|g lies above zero on every row.

**Reading.** At R0 the label is a deterministic function of the couplings, so no input adds information beyond them; any increment there measures how far C is from that function. Both relative increments inc_r and inc_g divide by the recalibration error. Expectation 1 is therefore the direct comparison of the two stacks: it holds exactly when the measurement-free stack has the smaller test error. The paper may write that a measurement-free function of the couplings matches the R0 stacking increment of r only if expectations 1 and 2 both hold at the exact level. The claim then rests on point estimates and the bound of expectation 2; it is not an equivalence test. g's double use of the validation rows (the disclosed asymmetry) is stated beside it. Otherwise the paper reports the three increments descriptively, the remaining inc_r|g as what a linear stack of r adds to these two predictors of the same function.

## G. Training Size

**Fits.** `--train-size N` for N in 40, 80, 160, and 320 training circuits per family. Each fit uses the first N training circuits of its dataset in the generator's prefix order, with all of a circuit's rows; validation and test rows are unchanged. Rungs R0, N1, N2, R4, and R5, at 2,048 shots. N = 640 is the shot sweep's 2,048-shot fits, and at R4, which the sweep did not fit, the strong-learner rerun's derived baseline. Each fit records its training size, its number of training rows, and the SHA-256 of its sorted training circuit identifiers.

**Reported.** Labels, D, D/C, and intervals per size, and F − R by `tools/descriptor_ladder_posthoc.py --part A`. Each size is paired with N = 640 through `--original-fits` (paired differences in D and D/C). The stage links the N = 640 fits named above into one directory and stops if any C or F fit of the three datasets, five rungs, and 20 learner seeds is missing. The shot sweep's 2,048-shot caches have the same SHA-256 values as G's, so the pairs share their test circuits. For each row and rung: the largest size below 640 at which the label differs from that at 640.

**Expectations (not decision rules).**
1. At R0, D/C at N = 40 exceeds D/C at N = 640 on at least five of the six rows.
2. At R5, every row reads "F beats C" at every size.

**Reading.** The sizes are nested prefixes of one generator order, and the validation set stays fixed as training shrinks. The curve is therefore conditional on that order; it does not estimate the variation across subsets of one size. Suppose R0 reads "F beats C" at some size on some row. Then the paper states the training size of the negative control wherever it states that control, and names the sizes at which it fails.

## H. Equal Spend

**Inputs.** No new fits: the shot sweep's datasets and its original-candidate fits.

**Analysis** by `tools/equal_spend.py`, at R0, N1, N2, and R5 for each row, descriptive. F at 2,048 shots spends 2,048 shots on each of 960 training and validation circuits per family and noise strength, then 2,048 shots on each deployed circuit. For n deployed circuits it costs 960 × 2,048 + 2,048n shots. Exact labels are free in this accounting. The raw estimate R spends ℓ shots per deployed circuit at level ℓ, ℓn in all. The calibrated estimate R_cal is the per-cell linear calibration of r fitted on the validation rows (c_r of the shot sweep). It also spends ℓ shots on each of the 320 validation circuits, 320ℓ + ℓn in all. For each finite level ℓ above 2,048 shots where the macro error of R or R_cal is at most F's mean macro error, the script reports the signed crossing n*. For R it is n* = 960 × 2,048 / (ℓ − 2,048); for R_cal it is n* = (960 × 2,048 − 320ℓ) / (ℓ − 2,048). Since ℓ exceeds 2,048, the comparator costs fewer shots than F when 0 ≤ n < n*, and F costs fewer when n > n*. A crossing at or below zero means that F costs no more shots at any deployment count. This holds for R_cal at every ℓ of at least 6,144, so at all three finite levels above 2,048. The exact level enters the comparison of errors only.

**Reading.** The paper reports these counts as an accounting under stated assumptions; real labels on hardware would add to F's and R_cal's spend.

## Deviations and Release

Any deviation is reported with its reason. On DeltaAI, `blas_fpe_probe()` runs before the first fit; matrix-multiplication warnings count as spurious in the sensitivity analysis only if that probe reproduces them on finite inputs. All fits are used. The release adds every fit of B, C, D, and G with its per-candidate predictions, the outputs of E and H, and the run logs. In the paper these appear as follow-ups whose rule was frozen before fitting, motivated by review; analyses chosen after their results are seen are labeled post hoc.
