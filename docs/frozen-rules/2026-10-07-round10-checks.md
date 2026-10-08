# Frozen Rule: Round-10 Checks (Oracle-Free Reading Rules, Oracle Sensitivity, Interval Targets, QRAFT With Machine Identity, Released-Prediction Summaries, and Rule Timing)

Written 2026-10-07 (US Pacific time). The tenth internal review panel prompted the six parts below. Remaining computations in Parts A to E start only after this rule and the code it names are pushed to the public repository. The Q2 − Q1 intervals of Part D were already computed but never seen, as disclosed below; they are reported post hoc. Part F reads records that GitHub keeps; its result was seen while its tool was written (below). Every quantity here is post hoc with respect to the experiments it reads: it reuses released fits, datasets, and analyses, and it changes no frozen label.

## Known When This Rule Was Written

- **Released analyses.** Every analysis file under `artifacts/` is public. Each per-cell D, label, interval, and Equation 2 term was therefore known. So were the Bayes oracle's G* and its classes, and the confusion counts: 696 cells of 44 fit sets, 522 when each source fit counts once. Of 367 distinct informative cells, 296 read "F beats C" and 71 are misses at N1, N2, and Heisenberg R3. The twelve false positives are exact-level R0 cells of stronger candidates, with D from 4e-5 to 3e-4. The pooled six-seed table and its margin verdicts at 0.10 were known.
- **Stacking.** `posthoc-stacking-all.json` covers ten fit sets (312 cells). In 44 of them the stacking increment lies above zero while F reads "not distinguished" or "measurement hurts". Of these, 23 are R0 cells from 8,192 shots up and 12 are transverse-field Ising N1 cells. The stacking increment of the other 34 fit sets was not computed.
- **Reproduction runs during tool development.** `tools/reading_rules.py --reproduce-only` ran on all 44 fit sets. It reproduced every D of analysis A (largest difference 0) and every stacking increment of the ten sets above (largest difference 2.8e-17). It computed no reference selection, no D_ref, and no stack over a reference. `tools/bayes_oracle.py --sensitivity` with default options reproduced the governed risks exactly for dataset seed 101 at 2,048 shots, N1 and R5. During code review, the same default-option run over the full schedule reproduced all 432 governed risk records exactly. No variant was run. `score_qraft_machine_matched.py --reproduce-only` reproduced the three published QRAFT arms under both splits (largest difference 0) and fitted no new arm. Every such run, by the author and by both code reviewers, also computed the Q2 − Q1 joint-bootstrap intervals under both splits. The mode neither printed nor saved their endpoints, so no record of them exists. These computations preceded this rule's public push; the final code skips them in this mode.
- **One fit record.** While the reference selection was written, one stronger-candidate fit record was printed (dataset seed 101, N1, learner seed 1, arm C). Its lowest validation error, for both families, was the degree-5 ridge polynomial. At R5 the stronger fits hold no polynomial candidate, because no coupling enters.
- **Not computed.** Harm rates, D_P = P − F, variance components, prediction intervals, first-versus-new contrasts, and margin sensitivity were not computed. Analysis A's mean P and the paper's source-validation P − F table were known. The ML-QEM analyses already report F − Rcal with intervals; Part E only collects them.
- **QRAFT.** The released panel and grouped reports are public. Machine identity enters only the seventeen-feature arm of QRAFT's released training code, so the published step from ten to seventeen features mixes machine identity with reverse statistics.

## Part A: Oracle-Free Reading Rules Scored Against the Oracle

**Cells.** The 696 fit-set cells of the governed confusion table (`artifacts/descriptor-information/round9/bayes-oracle.json`, Equation 2 rows). `tools/round10_registry.json` lists, for each of the 44 fit sets, the fits and cache it read; each cache must have the SHA-256 that its analysis A recorded.

**Reference family.** Each data condition (dataset-seed panel and shot level) has one stronger-candidate fit set on the same data, named in the registry: `strong_learners_2048`, `follow_b_<level>`, or `fresh_strong_<level>`. Its arm C stores the validation and test predictions of four descriptor-only candidates at learner seeds 1 to 20. They are a random forest, an MLP, gradient boosting (per-cell grid), and the degree-5 ridge polynomial in the couplings. A candidate missing in any learner seed is excluded and listed (the polynomial at R5). The reference reads the rung's descriptors and the noise-strength indicator. For the fit sets whose arms do not read strength, this gives the reference one descriptor more than C; the label does not depend on noise strength.

**Selection.** For each row (dataset seed and family) and rung, Ref* is the candidate with the lowest validation error for the row's family, averaged over learner seeds 1 to 20. The error is the fit records' `candidate_validation_family_mae`. Ties go to the earlier candidate in the order above. The recorded validation error of Ref* must equal the error recomputed from its validation predictions and the cache to 1e-12. No test row enters the selection.

**Quantities.** For every cell, from the fit set's arms and the reference:
- D = C − F, recomputed; it must equal analysis A to 1e-12, and its label must equal the oracle row's pipeline label.
- The stacking increment over C. Per cell of the row, y ~ a + bC and y ~ a + bC + cr are fitted by least squares on validation rows. Both fits are applied to test rows. The increment is the first macro error minus the second (the arithmetic of `tools/measurement_floor.compute_stacking`). For the ten sets of `posthoc-stacking-all.json` it must equal the released value to 1e-12.
- D_ref = Ref* − F, pairing learner seed k of Ref* with learner seed k of F, with D_ref / Ref*.
- The stacking increment of r over Ref*, as above with Ref* in place of C.

Every interval is analysis A's two-stage bootstrap (learner seeds, then whole test circuits; 10,000 draws; seed 20261002), with the same draws.

**Rules.** Each reads validation and test predictions only, never the oracle.
- `rule_D`: information if the D interval lies above zero (the pipeline label "F beats C").
- `rule_S`: information if the stacking-increment interval over C lies above zero.
- `rule_R` (primary): information if the D_ref interval lies above zero.
- `rule_RS`: `rule_R`, and the stacking-increment interval over Ref* lies above zero.
- `rule_R_delta`: the lower D_ref limit exceeds the family's margin from Part D of the round-9 rule (0.003696 for transverse-field Ising, 0.005912 for Heisenberg).

**Scoring.** The oracle's reading of each cell is "no information" at R0, "information" where the G* interval lies above zero, and "undetermined" otherwise. A rule's sensitivity is the share of information cells it flags. Its specificity is one minus the share of no-information cells it flags. The share of undetermined cells it flags is reported too. The counts use the row groups of the paper's confusion table: first seeds at 2,048 shots, new seeds at 2,048 shots, other finite levels, and the exact level. Two scopes are counted: all 696 cells, and the 522 cells left when derived fit sets are dropped. For every rule, the cells whose reading differs from `rule_D` are counted by oracle reading. Information cells where the D_ref interval lies below zero are counted.

**Running.** `PYTHONPATH=. python tools/reading_rules.py --assets ASSETS --crossed CROSSED --fresh FRESH --frozen-rule` (this file). ASSETS holds the unpacked assets of release `descriptor-information-v1`, CROSSED and FRESH the unpacked crossed-follow-up and fresh archives of release `confirmation-v1`. Outputs: `artifacts/descriptor-information/round10/reading-rules.{json,md}`.

**Expectations stated in advance (not decision rules).** In the 522-cell scope:
1. `rule_R` flags no no-information cell that `rule_D` does not flag.
2. `rule_R` detects fewer information cells than `rule_D`; the cells it loses lie at N1 and N2.
3. `rule_R_delta` flags no no-information cell and no N1 cell.
4. `rule_S` has lower specificity than `rule_D`.
5. `rule_RS` has specificity at least that of `rule_R`.

**Reading.** The default recommendation is `rule_R`. Another rule is eligible when its specificity strictly exceeds that of `rule_R` and its sensitivity is at least that of `rule_R` minus 0.05, in the 522-cell scope. Among eligible rules, the highest specificity is chosen, then the highest sensitivity. Remaining ties go to the first in the order `rule_D`, `rule_S`, `rule_RS`, `rule_R_delta`. If none is eligible, `rule_R` stays. `tools/reading_rules.py` records the selected rule. The paper reports all five rules and the selected recommendation. The selection uses the oracle benchmark itself, so it does not estimate the recommended rule's performance elsewhere. A specificity below one identifies approximation-error gains the recommended reference does not close.

## Part B: Numerical Sensitivity of the Bayes Oracle

**Runs.** `tools/bayes_oracle.py --eval-split test --sensitivity` runs on the governed schedule for all six dataset seeds. That is N1 to N4, R3-TFI, R3-Heis, and R5 at 2,048 shots, and N1 and N2 at the five other finite levels. Seven runs:
- `default`: the governed options. It must reproduce the governed risks exactly.
- `degree7`: degree-7 ridge surrogates, same alpha grid, refitted on training and validation circuits as before.
- `stream1`, `stream2`, `stream3`: independent Monte Carlo streams, whose seed words append [20261007, k] to the governed words.
- `draws5x`: five times each coupling-sample count (100,000 per circuit and 1,000,000 shared prior draws at R5), including each ESS redraw. The circuit bootstrap keeps its 10,000 draws and seed 20261002.
- `binomial`: the likelihood of r is the binomial probability of k = (r + 1)n/2 successes in n shots, with success probability (1 + e)/2. Every finite-shot r lies on that lattice (checked to 3.3e-8 at 131,072 shots).

**Reading of cells.** In each run, a cell's class is "information" when its G* interval lies above zero and "undetermined" otherwise. A cell is resolved when all six variant runs give the governed class, and unresolved otherwise. `tools/oracle_sensitivity.py` reports, per cell, every run's G* and interval, the largest change in G*, and each surrogate set's largest test error. It recounts the governed confusion table with unresolved cells apart.

**Running.** For each run NAME with its options: `PYTHONPATH=. python tools/bayes_oracle.py --eval-split test --levels 256 1024 2048 8192 32768 131072 --sensitivity [options] --out artifacts/descriptor-information/round10/oracle-runs/NAME.json --frozen-rule` (this file). Options: `degree7` `--surrogate-degree 7`; `streamK` `--stream K`; `draws5x` `--draw-multiplier 5`; `binomial` `--likelihood binomial`. Then `PYTHONPATH=. python tools/oracle_sensitivity.py --run NAME=PATH` (one per run) `--frozen-rule` (this file).

**Expectations stated in advance (not decision rules).**
1. For every dataset seed and family, the largest surrogate test error is lower at degree 7 than at degree 5.
2. No cell at N3, N4, R3-TFI, or R5 is unresolved.
3. Unresolved cells, if any, lie at N1, N2, or R3-Heis.
4. At most a fifth of the N1 and N2 cells are unresolved.

**Reading.** The paper reports the unresolved count and the recount. Misses and detections among resolved cells replace the governed tally where the paper states sensitivity. Any statement that rests on an unresolved cell is called unresolved.

## Part C: What the Intervals Condition On

**Estimands.** The paper distinguishes three targets. The first is a fitted model. The second is the fitting pipeline on one dataset seed, where learner seeds and test circuits vary and the training sample stays fixed. The third is the pipeline over regenerated datasets. The two-stage interval targets the second, and the A/A calibration calibrates the second. The six-seed pool estimates the third from six dataset seeds.

**Designs.** The three six-seed designs of Part D of the round-9 rule, at every row and every rung present for all six seeds:
- original candidates with the strength indicator at 2,048 shots (`strength_indicator_2048` and `fresh_orig_2048`);
- stronger candidates at 2,048 shots (`strong_learners_2048` and `fresh_strong_2048`);
- stronger candidates at the exact level (`follow_b_exact` and `fresh_strong_exact`).

**Quantities.** `tools/seed_variance.py` computes, per dataset seed:
- the paired interval of analysis A, which must reproduce its point, interval, and draw SD to 1e-12;
- an independent interval, in which F's learner-seed counts come from a second stream (`default_rng(20261003)`) while C keeps analysis A's counts and both keep the shared circuit counts;
- the learner-seed component (variance of the 20 per-seed D values over 20) and the test-circuit component (variance of the circuit-only bootstrap with every learner seed kept). These are marginal diagnostics; their sum need not equal the two-stage variance.

Across the six seeds, the DerSimonian–Laird between-seed variance tau² of the pooled tool, which must reproduce Part D's pooled D and tau² to 1e-12. Two prediction intervals for a new dataset seed: theta ± t(0.975, 4)·sqrt(tau² + SE²) for its true value, and the same with the mean two-stage variance added for its estimate. Here theta is the random-effects mean and SE the square root of one over the sum of random-effects weights. The dataset-seed fraction tau² / (tau² + mean two-stage variance) is the share of a new seed's estimate variance, apart from SE², that comes from changing the dataset. Finally, each panel's mean D, their difference (new minus first), and a Welch t interval over the three seeds of each panel.

**Running.** `PYTHONPATH=. python tools/seed_variance.py --assets ASSETS --crossed CROSSED --fresh FRESH --frozen-rule` (this file). Outputs: `artifacts/descriptor-information/round10/seed-variance.{json,md}`.

**Expectations stated in advance (not decision rules).**
1. Independent resampling changes the label of fewer than 5 percent of the per-seed cells.
2. At R0, every prediction interval for the true value contains zero.
3. At transverse-field Ising N2, the prediction interval for the true value lies above zero in both 2,048-shot designs.
4. Most first-versus-new Welch intervals contain zero.

**Reading.** The paper names the second estimand as the target of its intervals and confines the calibration statement to it. It reports the components and prediction intervals at R0, N1, and N2, the paired-versus-independent comparison, and the first-versus-new table.

## Part D: QRAFT With Machine Identity in Every Arm

**Data and recipes.** The QRAFT processed inputs of the Q-LEAR release (10,155 rows), as in the 2026-10-06 rule. Two splits are used, each with ten seeds and gradient boosting at `random_state` = seed. One is the row split of `score_qraft_panel.py` (Panel M, joint row bootstrap stratified by machine). The other is the circuit-level grouping of `score_qraft_grouped.py` (machine and six circuit descriptors, 1,038 groups, joint group bootstrap). Both bootstraps use 10,000 draws and seed 20261002.

**Arms.** The published arms are Q3 (seven descriptors), Q2 (Q3 and forward probabilities), and Q1 (Q2, reverse statistics, and machine identity). The new arms are Q3m (Q3 and machine), Q2m (Q2 and machine), and Q1nm (Q2 and reverse statistics, without machine). The published arms are fitted first and must reproduce the released reports exactly before any new arm is fitted.

**Contrasts.** Each is a ten-seed mean with a joint-bootstrap interval:
- the reverse increment with machine in both arms (Q2m − Q1) and in neither (Q2 − Q1nm);
- the machine increment in the forward (Q2 − Q2m), full (Q1nm − Q1), and descriptor (Q3 − Q3m) arms;
- the descriptor arm minus the forward arm with machine in both (Q3m − Q2m, and their ratio);
- the published step Q2 − Q1 (post hoc: reproduction-mode runs computed its interval before the push, unseen).

**Running.** `python reanalysis/scripts/qraft/score_qraft_machine_matched.py <QRAFT dir> reanalysis/outputs/qraft/round10 --frozen-rule` (this file).

**Expectations stated in advance (not decision rules).**
1. The published arms reproduce exactly.
2. Machine identity lowers the ten-seed error of every arm it enters, under both splits.
3. Under both splits, Q2m − Q1 is smaller than Q2 − Q1.
4. Under both splits, the Q3m − Q2m interval lies above zero, and Q3m/Q2m exceeds 4.

**Reading.** The paper attributes to reverse execution only Q2m − Q1, and reports the machine increments separately.

## Part E: Released-Prediction Summaries

`tools/round10_summaries.py`, four subcommands, outputs under `artifacts/descriptor-information/round10/`:
- `harm`: in S0 (the 2,048-shot original-candidate ladder fits) and the shift-transfer settings S2 and S4, for every row and rung, the harm rate of F and of C. An item is harmed when the arm lies farther from the label than the raw estimate, |F − y| > |r − y|. The rate is the unit-weight macro over the row's cells of the harmed fraction, averaged over learner seeds, with analysis A's two-stage interval. Each arm's mean error must equal analysis A's to 1e-12. The largest error dilation, max(|F − y| − |r − y|), is reported per learner seed (median and maximum). The shift caches regenerate with `tools/shift_transfer.py generate` and `prepare`; each must have the SHA-256 its analysis recorded.
- `dp`: D_P = P − F with analysis A's two-stage interval and three-way label, at every rung of the original, strength-indicator, and new-seed original-candidate fit sets at 2,048 shots. Each D must reproduce analysis A.
- `margins`: Part D's practical-margin verdicts with delta at 0.025, 0.05, 0.10, and 0.20 times the mean calibrated raw error. The 0.10 verdicts must equal Part D's.
- `mlqem`: F − Rcal and F − R from the released ML-QEM analyses, archived and exact targets. Nothing is recomputed.

**Running.** `PYTHONPATH=. python tools/round10_summaries.py harm --assets ASSETS --shift SHIFT --frozen-rule` (this file), with SHIFT the `--out` directory of the shift-transfer run; `dp --assets ASSETS --fresh FRESH`; `margins`; `mlqem` (each with `--frozen-rule`).

**Expectations stated in advance (not decision rules).**
1. In S0, F's harm rate lies below 0.05 at every row and rung.
2. In S4, F's harm rate exceeds its S0 value at every row and rung.
3. D_P lies above zero at every row and rung where D does.
4. No margin multiplier moves a row between "practically relevant gain" and "practically negligible".

**Reading.** The paper reports the harm rates, D_P beside D, the margin sensitivity, and F − Rcal beside F − R in its ML-QEM table.

## Part F: GitHub Records of Rule Timing

**Records.** On 2026-10-07 the responses of `gh api repos/yzhao062/QEMScore/events --paginate`, `.../releases --paginate`, and `.../actions/runs --paginate` were saved under `artifacts/rule-timestamps/`. The events API returns a bounded recent window and can omit pushes; it lacks the push of commit 24a2cbe. The continuous-integration run that the push triggered records it instead, and a run's time is an upper bound on when its commit was on GitHub. A release's `created_at` is its tagged commit's date and is not used; publication and asset upload times are.

**Running.** `python tools/rule_timestamps.py --events ... --releases ... --runs ... --frozen-rule` (this file). Output: `artifacts/rule-timestamps/rule-timestamps.json`.

**Known result.** The tool was developed on these files, so its result was seen before this rule was frozen. Every rule from 2026-10-02 on reached GitHub before the commit that first recorded its results and before the upload of the release assets holding its fits. These records date public availability; they do not date the first computation. The results of both 2026-10-01 analyses reached GitHub at 2026-10-02T08:51:37Z (commit 645db53), before their rule files reached GitHub at 2026-10-03T07:44:17Z. Their ordering before computation rests on file modification times.

**Reading.** The paper adds the GitHub-recorded times to its claim ledger. It states that the results of the two earliest analyses reached GitHub before their rule files, whose ordering before computation rests on file modification times.

## Deviations and Release

Any deviation is reported with its reason. No outcome changes a setting, model, arm, rung, level, margin, split, rule, or reference family. Any analysis chosen after these results are seen is labeled post hoc. The outputs are committed as the results of this rule, with the commit at which they ran.
