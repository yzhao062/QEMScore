# Descriptor-Information Experiment Artifacts

This directory contains configuration, gate verification, diagnostic reports, and analysis metadata for the post-campaign descriptor-information experiment.

## Experiment Overview

The descriptor-information experiment investigates whether the capacity-matched error mitigation comparison responds to the degree to which instance descriptors identify circuit labels:
- **Part A (Missing information):** Instance descriptors are degraded on fixed spin-chain circuits (TFI and Heisenberg across dataset seeds 101, 211, 307) across rungs R0, N1-N4 (additive noise), R3-TFI/R3-Heis (withheld coupling), R4 (released ML-QEM binned-angle encoding), R5 (measurement-only/no instance descriptors), and near-Clifford controls (NC-none, NC-R0).
- **Part B (High-dimensional structure):** QAOA-MaxCut random graph circuits (Erdős–Rényi and 3-regular, 10 qubits, $p \in \{1, 2\}$) across rungs B-partial (standard QAOA parameters), B-complete (including 45 edge indicators), and B-none.

The frozen decision rule governing this experiment is documented in:
`docs/frozen-rules/2026-10-02-descriptor-information.md`
frozen at commit `4c95174`.

## Directory Contents

- `gate/gate_r0.json`: Verification report demonstrating that the hook-enabled fitting pipeline reproduces archived R0 test predictions within the amended 1.4e-11 tolerance (bound stated in Appendix M.4).
- `gate/gate_r0.before-amendment.json`: Initial gate run prior to adopting the M.4 1.4e-11 reproduction tolerance.
- `reexport_check.json`: Verification report checking that exported fit predictions match pipeline cache predictions across all evaluated seeds and rungs. Path prefixes have been scrubbed to `<ladder>/`.
- `diagnostics/`:
  - `qaoa.json`: Pre-fit diagnostic report on Part B QAOA dataset properties, label standard deviations per cell/graph class, and attenuation slopes.
  - `qaoa-generation.json`: QAOA dataset generation metadata, runtime metrics, and validation logs (path prefixes scrubbed to `<ladder>/`).
  - `r4-encoder.json`: R4 diagnostic suite report evaluating distinct vector counts, kNN coupling recovery $R^2$, and bin-lookup MAE for the released ML-QEM binned-angle encoding across seeds 101, 211, 307.
- `run_config-partA.json`: Rungs, arms (A, F, C, P, M), seeds (1–20), and job counts for Part A.
- `run_config-partB.json`: Rungs, arms (A, F, C, P, GBT), seeds (1–20), and job counts for Part B.
- Analysis outputs:
  - `analysis-a.json`: Full bootstrap analysis output from `tools/descriptor_ladder_analysis.py` (Script A).
  - `analysis-b.json`: Independent bootstrap analysis output from `tools/descriptor_ladder_analysis_b.py` (Script B).
  - `posthoc-arm-minus-raw.json`: post hoc paired contrasts F - R and C - R on Part B (`tools/descriptor_ladder_posthoc.py`). The frozen rule states no reading for a learned arm against the raw estimate, so the paper labels these post hoc.
  - `posthoc-headline.json`: post hoc twenty-learner-seed values for the six primary R0 rows (`tools/headline_seed_averaged.py`): C, F, A, D, D/C, the share S, the share against the shrinkage control, and the single-fit S from `campaign-archive-v1`. It reuses script A's estimator and draws and asserts agreement with `analysis-a.json`.
  - `posthoc-measurement-floor.json`: post hoc measurement-only floor (test error of a per-cell linear calibration of the noisy estimate at 2,048 shots) and the stacking increment of the noisy estimate over the recalibrated control at every Part A rung (`tools/measurement_floor.py`).
  - `reconcile.json`: the output of `tools/descriptor_ladder_reconcile.py` on the two analyses: 63 cells, 1,668 compared values, largest difference 2.8e-15, identical labels and rung statements (passed).

- `strength-indicator/`: the follow-up in which every arm also reads the noise strength (section below): `analysis-a.json` (with the paired comparison against the original fits in each cell), `analysis-b.json`, `reconcile.json`, `comparison.json` (the same paired comparison from `tools/strength_indicator_compare.py`), and the two run configurations. Both analyses record the estimator rule under `frozen_rule` and the follow-up rule, with its SHA-256, under `follow_up_rule`. Paths are relative to the working directory of the run.

## How to Rerun the Experiment

The commands below are the ones used for the reported run (macOS, 16 workers, about 17 minutes for Part A and 3 minutes for Part B). `data/` holds the three regenerated primary datasets (`tools/regenerate_example.py`) and the three near-Clifford datasets of Appendix N (`tools/near_clifford_positive_control.py`); `ARCHIVE` is the unpacked `campaign-archive-v1` release asset; `SEEDREP` is the Appendix M.4 fit directory (`tools/learner_seed_replication.py`, or `artifacts/learner-seed-replication/inputs/fits`). Run every command with `PYTHONPATH=.`.

```bash
PRIMARY="data/regen-shipped-s101-n640 data/regen-shipped-s211-n640 data/regen-shipped-s307-n640"
NC="data/nc-s101-n640 data/nc-s211-n640 data/nc-s307-n640"

# 1. R0 gate, under the first amendment of the frozen rule (tolerance 1.4e-11).
#    AMENDMENT is the text of that amendment, verbatim from the rule file.
python tools/descriptor_ladder.py gate --datasets $PRIMARY --archive $ARCHIVE --out runs/partA \
    --tolerance 1.4e-11 --amendment "$AMENDMENT" --seedrep-fits $SEEDREP

# 2. Part A and the near-Clifford fits (1,956 fits). The run builds the rung descriptors,
#    refuses to start without a passing gate file, and ends with the re-export check.
python tools/descriptor_ladder.py run --datasets $PRIMARY --archive $ARCHIVE --out runs/partA \
    --nc-datasets $NC --nc-results <Appendix N results dir> --seedrep-fits $SEEDREP --workers 16

# 3. Part B: generate and validate the QAOA datasets, the label-free diagnostics, then 552 fits.
python tools/qaoa_intermediate.py generate --data-root runs/qaoa-data
python tools/qaoa_intermediate.py diagnostics --data-root runs/qaoa-data
python tools/qaoa_intermediate.py run --data-root runs/qaoa-data --out runs/partB \
    --gate-file runs/partA/gate/gate_r0.json --workers 16

# 4. The two independent analyses (10,000 draws, bootstrap seed 20261002 by default).
python tools/descriptor_ladder_analysis.py --fits runs/partA/fits --fits runs/partA/gate/fits \
    --fits runs/partB/fits --cache runs/partA/cache --cache runs/partB/cache \
    --out artifacts/descriptor-information/analysis-a.json
python tools/descriptor_ladder_analysis_b.py --fits runs/partA/fits runs/partA/gate/fits runs/partB/fits \
    --cache runs/partA/cache runs/partB/cache --out artifacts/descriptor-information/analysis-b.json

# 5. The agreement check required by the second amendment.
python tools/descriptor_ladder_reconcile.py artifacts/descriptor-information/analysis-a.json \
    artifacts/descriptor-information/analysis-b.json --out artifacts/descriptor-information/reconcile.json

# 6. Post hoc (not in the rule): F - R and C - R on Part B, reusing script A's estimator and draws.
python tools/descriptor_ladder_posthoc.py runs artifacts/descriptor-information/posthoc-arm-minus-raw.json

# 7. Post hoc: seed-averaged headline, measurement floor, and near-Clifford label strata.
#    ASSET is the unpacked descriptor-information-v1 release asset (below).
python tools/headline_seed_averaged.py --fits $ASSET/fits --cache $ASSET/cache --archive $ARCHIVE \
    --out artifacts/descriptor-information/posthoc-headline.json
python tools/measurement_floor.py --fits-dir $ASSET/fits --data-dir data \
    --out-json artifacts/descriptor-information/posthoc-measurement-floor.json
python tools/near_clifford_strata.py --fits-dir $ASSET/fits --cache-dir $ASSET/cache --nc-data-dir data \
    --out-json artifacts/near-clifford-positive-control/posthoc-label-strata.json
```

Every subcommand prints its options with `--help`. Both analysis scripts also run on the release asset below, which needs no fit: point `--fits` at its `fits/` directory and `--cache` at its `cache/` directory.

## Release Asset

`descriptor-information-v1.tar.xz` (about 90 MB, SHA-256 `9cf19a499122cdff92560333cb2a91b2fcdd098ad8fe0cb95e166f7d186dfa57`) will be attached to the QEMScore release that this paper cites (not yet uploaded). It holds every fit of the experiment (JSON metadata and NPZ arrays with the selected and per-candidate validation and test predictions) and slimmed test-item caches that keep only the fields the two analysis scripts read. Both scripts give the same output on the asset as on the full caches.

`descriptor-information-strength-v1.tar.xz` (about 100 MB, SHA-256 `d9bc11b4eed70d90f7b3c605c1b1eb3c9656d15fbc83b2c20fe7511a7bdae170`) will be attached to the same release. It holds the 2,502 fits of the strength-indicator follow-up in the same format, and no caches: the follow-up reads the caches of `descriptor-information-v1`. Both analysis scripts and the comparison give the same output on the two assets as on the full caches.

## Strength-Indicator Follow-Up (Rule of 2026-10-03)

A fourth review panel noted that every learned arm read a feature vector without the noise strength, although training pools the two strengths (L1 and L3), while the per-cell calibration and stacking references are fitted per cell and so know it. The rule `docs/frozen-rules/2026-10-03-strength-indicator.md` was committed and pushed (commit `f24a482`, 2026-10-04 00:55 UTC) before any fit with the indicator; the first fit ran about 20 minutes later. The original results and the post hoc per-cell analyses were known when it was written.

Every arm at every rung of Parts A and B and on the near-Clifford datasets reads one extra final feature, `noise_strength_L3` (0 for L1, 1 for L3). The permuted refit P never permutes it. R0 and NC-R0 are refitted with the indicator (seeds 1 to 20, plus the affine control), so the follow-up has no gate and no re-export: 1,950 fits for Part A and the near-Clifford datasets and 552 for Part B, on the same machine and software as the original run. Estimands, the bootstrap, and the labels are those of the original rule. The two scripts agree on 63 cells and 1,677 compared values to 4.3e-15, with identical labels and rung statements. The original ladder remains the primary record of the original rule.

```bash
# Reuse the prepared caches, rung descriptors, and R4 encoder cache of the original run
# (copied into runs-strength/ so nothing writes into runs/).
mkdir -p runs-strength/partA runs-strength/partB
cp -R runs/partA/cache runs/partA/descriptors runs/partA/encoder-cache runs-strength/partA/
cp runs/partA/datasets.json runs/partA/dataset_checks.json runs-strength/partA/
cp -R runs/partB/cache runs-strength/partB/ && cp runs/partB/datasets.json runs-strength/partB/

python tools/descriptor_ladder.py run --datasets $PRIMARY --nc-datasets $NC --archive $ARCHIVE \
    --out runs-strength/partA --gate-file runs/partA/gate/gate_r0.json --strength-indicator --workers 16
python tools/qaoa_intermediate.py run --data-root runs/qaoa-data --out runs-strength/partB \
    --gate-file runs/partA/gate/gate_r0.json --strength-indicator --workers 16

D=artifacts/descriptor-information/strength-indicator
python tools/descriptor_ladder_analysis.py --fits runs-strength/partA/fits --fits runs-strength/partB/fits \
    --cache runs-strength/partA/cache --cache runs-strength/partB/cache \
    --original-fits runs/partA/fits --original-fits runs/partA/gate/fits --original-fits runs/partB/fits \
    --follow-up-rule docs/frozen-rules/2026-10-03-strength-indicator.md --out $D/analysis-a.json
python tools/descriptor_ladder_analysis_b.py --fits runs-strength/partA/fits runs-strength/partB/fits \
    --cache runs-strength/partA/cache runs-strength/partB/cache \
    --follow-up-rule docs/frozen-rules/2026-10-03-strength-indicator.md --out $D/analysis-b.json
python tools/descriptor_ladder_reconcile.py $D/analysis-a.json $D/analysis-b.json --out $D/reconcile.json
python tools/strength_indicator_compare.py --strength-fits runs-strength/partA/fits \
    --strength-fits runs-strength/partB/fits --original-fits runs/partA/fits \
    --original-fits runs/partA/gate/fits --original-fits runs/partB/fits \
    --cache runs-strength/partA/cache --cache runs-strength/partB/cache --out $D/comparison.json
```

On the release assets, point `--fits` (or `--strength-fits`) at `descriptor-information-strength-v1/fits`, `--original-fits` at `descriptor-information-v1/fits`, and `--cache` at `descriptor-information-v1/cache`. The `--gate-file` argument only satisfies the run command's check that the hooked fitting path was gated; the new fits are not compared with the archive.

**The six expectations stated in advance** (point estimates unless an interval is named; "D/C" is the remaining-error reduction, "TFI" the transverse-field Ising rows):

1. *R0 stays not distinguished on all six Part A rows*: did not hold. Five rows are not distinguished; TFI seed 101 reads "measurement hurts" (D/C -0.216, interval [-0.419, -0.034]), so the TFI R0 rung statement is "mixed". No row reads "measurement adds". The six R0 D/C interval upper limits are -0.034 to 0.198 (0.041 to 0.434 in the original fits).
2. *TFI D/C rises at R5 and R3-TFI*: held. At R5 it is 0.814 to 0.838 (original 0.546 to 0.584), and at R3-TFI 0.813 to 0.842 (original 0.552 to 0.585). The paired differences, 0.251 to 0.270, have intervals above zero on all six cells.
3. *TFI M error at R5 below 0.06*: held, 0.035 to 0.039 (original 0.089 to 0.094). In a post hoc comparison, it is at most 0.00021 above the mean of the row's L1 and L3 calibration errors of r in `posthoc-measurement-floor.json` (0.034 to 0.039).
4. *TFI turns "measurement adds" no later than N2*: held. N1 is not distinguished on all three seeds and N2 to R5 add on all three.
5. *Heisenberg changes little*: held for D/C from N2 to R5 outside R3-Heis (paired differences at most 0.032 in size). Three cells near the decision boundary changed label: R3-Heis seed 101 (not distinguished to adds, D/C 0.090), N1 seed 307 (adds to not distinguished), and N2 seed 307 (not distinguished to adds). The R3-Heis rung statement turns "mixed", the N1 statement "not distinguished", and N2 stays "mixed".
6. *No expectation for Part B or the near-Clifford datasets*: every label is unchanged. On Part B the paired D/C differences range from about -0.081 to 0.061; on the near-Clifford datasets their absolute values are below 0.012.

## Post Hoc Analyses of Learner Adequacy, References, and Raw Contrasts (2026-10-03)

A fifth review panel asked whether the learners that define D are adequate, how strong a descriptor-only reference can be, and how the strength-aware fits compare with the raw estimate. The four analyses below answer these questions from the released fits and caches without refitting any learned arm. All four are post hoc: the original and strength-indicator results were known.

- `posthoc-learner-adequacy.json` and `strength-indicator/posthoc-learner-adequacy.json` (`tools/learner_adequacy.py`): per-seed spread of C, F, and D in every cell, and the seed-ensemble estimand D_ens, which averages each arm's test predictions over the 20 learner seeds before scoring. Intervals reuse script A's two-stage bootstrap. A pooled Part A R0 estimate resamples learner seeds once per dataset seed for both family rows, because the two families of a dataset share one fitted network. In the original fits, ensembling lowers the R0 test errors of C 2.5 to 4.8 times and of F 2.3 to 5.4 times, and it flips the point sign of D on three of six R0 rows (seed 101 Heisenberg, seed 211 TFI, seed 307 TFI). Pooled R0 D/C is -0.052 [-0.222, 0.094]; pooled D_ens/C is 0.007 [-0.472, 0.289]. Mean C, F, and D match `analysis-a.json` to 1e-12 in every cell.
- `posthoc-strongest-reference.json` (`tools/strongest_reference.py`): at every Part A rung, the strongest descriptor-only reference that never reads r, and the stacking increment of r over it with a one-stage circuit bootstrap. Per cell, a histogram gradient-boosted regressor is selected from an eight-point grid by validation error; at R0 the degree-five coupling polynomial of `tools/refit_polynomial_diagnostic.py` is also fitted. At R0 the polynomial is strongest, with test error 1.5 to 8.2 percent of mean C, and stacking r on it changes the error by less than 3e-6. From N1 to R5 the boosted reference is close to C (0.93 to 1.88 times C's error).
- `posthoc-ivw-ceiling.json` (`tools/ivw_ceiling.py`, heuristic): per cell, c_comb = (c_C^-2 + c_r^-2)^-1/2 from C's per-cell test error c_C and the calibration error c_r of `posthoc-measurement-floor.json`, and the ceiling D/C* = 1 - macro(c_comb)/macro(c_C). The assumption is independent, roughly unbiased errors with MAE as the error scale. Over the 48 Part A cells, the Spearman correlation between the ceiling and the observed D/C is 0.974 for the original fits and 0.969 for the strength fits. For the strength fits, transverse-field Ising reaches 0.87 to 1.00 of the ceiling from N2 to R5.
- `strength-indicator/posthoc-arm-minus-raw.json` (`tools/descriptor_ladder_posthoc.py --fits ... --cache ...`): F - R and C - R on Part B for the strength-indicator fits. At B-none, F - R is -0.0024, -0.0037, and -0.0006 for dataset seeds 101, 211, and 307; only seed 211's interval lies below zero. The positional form of the script still reproduces `posthoc-arm-minus-raw.json`.

```bash
ASSET=descriptor-information-v1; STRENGTH=descriptor-information-strength-v1
python tools/learner_adequacy.py --fits $ASSET/fits --cache $ASSET/cache \
    --analysis-a artifacts/descriptor-information/analysis-a.json \
    --out artifacts/descriptor-information/posthoc-learner-adequacy.json
python tools/learner_adequacy.py --fits $STRENGTH/fits --cache $ASSET/cache \
    --analysis-a artifacts/descriptor-information/strength-indicator/analysis-a.json \
    --out artifacts/descriptor-information/strength-indicator/posthoc-learner-adequacy.json
python tools/strongest_reference.py --partA-ladder runs/partA --workers 8
python tools/ivw_ceiling.py
python tools/descriptor_ladder_posthoc.py --fits $STRENGTH/fits --cache $ASSET/cache \
    --out artifacts/descriptor-information/strength-indicator/posthoc-arm-minus-raw.json
```

`tools/strongest_reference.py` reads the full Part A caches, rung descriptors, and encoder cache of the prepared run directory (`runs/partA` above), since the reference is fitted on training rows; the other three read only the release assets.

## Shot-Count Sweep (Rule of 2026-10-03)

The fifth review panel asked whether the R0 null reflects the precision of the noisy estimate r. The rule `docs/frozen-rules/2026-10-03-shot-sweep.md` was committed and pushed with its code (commit `479a9a0`) before any shot level was generated or any fit ran. Each primary dataset was regenerated at 256, 1,024, 2,048, 8,192, 32,768, and 131,072 shots, and at the exact noisy expectation. Every measurement group reuses its stored sampler seed, so only the shot count changes. A, C, F, and P were refitted at R0, N1, N2, and R5 with the strength indicator: 732 fits per level, 5,124 in all. Before any fit that reads r, `tools/shot_sweep_predict.py` wrote a ceiling D/C* for every row, rung, and level. It combines C's error with the de-attenuated error of r, and the file was hashed.

Execution followed the rule's check-1 fallback. On DeltaAI (aarch64) the 2,048-shot regeneration of seed 101 matched the archive exactly, but seed 211 stopped in `validate_split_artifact`: one stored label differs from the aarch64 statevector by 1e-12. All levels were therefore generated on the macOS machine that produced the earlier datasets, after check 1 passed there for all three datasets (no differing measurement group, `items.jsonl` byte-identical). Check 2 passed on 8,960 items per dataset (largest |z| 4.40, below 6). DeltaAI verified every copied file against the macOS SHA-256 list before preparing, check 3 passed at every level, and all fits ran on DeltaAI. The run log records each deviation, including the R0 gate refits that DeltaAI could run only for seeds 101 and 307 (largest difference from the archive 1.1e-12).

The two analysis scripts agree to 1.4e-15 at every level, with identical labels. Reading outcome 1 of the rule applies: H1 and H2 hold.
- H1 (a): across the seven levels, D/C* and the observed D/C have a Spearman correlation of at least 0.8 in every series: 0.82 to 1.00 at N1 and 1.00 at N2.
- H1 (b): over 48 qualifying cells from six rows, the median ratio of observed D/C to D/C* is 0.876.
- H1 (c): no cell's lower interval limit exceeds D/C*.
- H2: 23 of 27 label-changing cells (85 percent) and 78 of 83 predicted cells (94 percent) carry the predicted label. Three of the matches are "measurement hurts" read as "not distinguished".

Under the rule, the paper may write that a ceiling computed from the two error scales before fitting predicted the size of the gap and where it became detectable. The claim is limited to 256 shots through the exact limit, this noise model, these two families, and 640 training circuits.

`tools/shot_sweep_report.py` computes the checks and expectations that the rule reports beside the results; it was written after the fits and decides nothing.
- C's and A's test and validation predictions are bit-identical across the seven levels (252 fits per level).
- At 2,048 shots every DeltaAI label equals the 2026-10-03 macOS label; the largest paired |ΔD| is 2.7e-6.
- Expectation 1 holds for transverse-field Ising. Macro c_m falls to 0.48 to 0.52 of its 256-shot value at 1,024 shots and to 0.17 to 0.18 at 8,192. It then levels off near 0.003 at the exact level; the Heisenberg rows are noisier.
- Expectation 2 fails: at R0, transverse-field Ising seeds 101 and 211 reach their thresholds at the exact level (D/C* 0.467 and 0.504), yet read "not distinguished". No R0 row reads "measurement adds" at any level; R0 D/C lies between -0.44 and 0.14.
- Expectation 3 holds: transverse-field Ising N2 reads "measurement adds" at 1,024 shots and above, and seeds 101 and 307 lose it at 256.
- Expectation 4 holds: R5 reads "measurement adds" at every level.
- Expectation 5 holds through 8,192 shots (M over macro c_r between 0.94 and 1.05). At 32,768 and 131,072 shots the transverse-field Ising ratio is 1.13 to 1.27. At the exact level it is 0.99 to 1.29 for transverse-field Ising and 0.77 to 0.87 for Heisenberg.

`shot-sweep/` holds the predictions file, `hypotheses.json`, and `shot-sweep-report.json`. `levels/<level>/` holds each level's two analyses, reconcile check, and F − R and C − R contrasts. `checks/` holds the check reports and SHA-256 lists; the 141,267-line tree list goes with the release asset, and its SHA-256 is in `level-tree-sha256.txt.sha256`. `logs/` and `scripts/` hold both run logs and the driver scripts. The release asset `descriptor-information-shot-sweep-v1.tar.xz` (about 410 MB, SHA-256 `d39969f21e77e4ab33a1daaa7b8daedf4b93cb9753c569d575af0ec631971b16`) will hold the seven level datasets, every fit with its per-candidate predictions, the level caches, the full check reports with the tree list, and the DeltaAI run log. Paths inside the JSON files are those of the DeltaAI run.

```bash
# On the machine that produced the datasets: generate and check every level (scripts/mac_generate.sh).
python tools/shot_sweep.py generate --source data/regen-shipped-s101-n640 --shots 2048 --out levels/shots-2048/regen-shipped-s101-n640
python tools/shot_sweep.py check-faithful --source data/regen-shipped-s101-n640 --regen levels/shots-2048/regen-shipped-s101-n640 --report check1-faithful-s101.json
# ... the other seeds and levels, then the exact level and check-exact against 131,072 shots.
# On DeltaAI: prepare --sweep and check-caches per level, fit A and C at 2,048 shots, write the
# predictions file, fit A, C, F, and P at every level, analyze, reconcile, and evaluate H1 and H2
# (scripts/sweep_stage.sh stages prepare, fitAC, predict, fitall, analyze, hypotheses).
python tools/shot_sweep_report.py --runs runs --out artifacts/descriptor-information/shot-sweep/shot-sweep-report.json
```

## Strong-Learner Rerun (Rule of 2026-10-03)

The fifth review panel asked whether the learners that define D are adequate. The rule `docs/frozen-rules/2026-10-03-strong-learners.md` was committed and pushed with its code (commit `832ec36`, 2026-10-04 16:31 UTC) before any fit of the rerun; the first fit ran about five minutes later. C, F, and P may also choose a tuned per-cell gradient-boosted regressor (hgbr) and, at Part A rungs with couplings, a per-cell ridge-regularized degree-five polynomial in the couplings that the rung's feature vector carries (poly5_ridge). Each strong fit records the option-off state, from which `tools/strong_learner_derive.py` rebuilds the random-forest and MLP baseline on the same platform. That derived baseline is the primary comparator.

All 2,199 fits ran on DeltaAI (1,647 for Part A, 552 for Part B). Part A's R4 fits peak at 7 GB or more each, and the node ran out of memory twice (140 and then 48 workers). The same run command finished the remaining fits with 16 workers, skipping the fits already written; the run log records every invocation. Slurm records the Part B job as FAILED: its driver script was replaced on disk after all 552 fits were written, and a fragment of the new file then failed. Both analysis scripts agree to 2.8e-15 on the rerun and to 6.9e-15 on the derived baseline, with identical labels and 19 identical rung statements. The derived baseline equals the shot sweep's 2,048-shot fits of C and F at R0, N1, N2, and R5 exactly (480 fits, same selected models). Its labels equal the 2026-10-03 macOS labels in every cell, with a largest paired |ΔD| of 2.4e-4.

All six expectations hold (`strong-learner-summary.json`).
1. At R0, mean C falls to 1.4 to 7.2 percent of the derived baseline's.
2. No R0 row reads "measurement adds".
3. The three transverse-field Ising rows read "measurement adds" at N2, N3, N4, R3-TFI, R4, and R5.
4. At R5, D/C changes by at most 0.0131 (paired point difference).
5. At R0, C selects poly5_ridge in all 20 fits of every dataset seed.
6. At B-complete, mean F falls to 0.52 to 0.56 of the derived baseline's.

Two rung statements change, so both records are reported for those rungs. Transverse-field Ising N1 moves from "not distinguished" to mixed (seeds 211 and 307 read "measurement adds"). Heisenberg R0 moves from "not distinguished" to "measurement hurts" on all three seeds. The first rung reading "measurement adds" is unchanged: N2 for transverse-field Ising and N3 for Heisenberg. Seven cells change label, and every D interval narrows; the paired D/C interval excludes zero in three of them (`label_changes`). These summaries do not identify the cause of a label change.

With the stronger candidates, R0 D/C lies between -0.017 and 0.000 on the six rows, with upper interval limits of at most 0.007. Pooled over the six rows, D/C is -0.012 [-0.018, -0.006]; the seed-ensemble estimate equals it, because poly5_ridge is selected at every seed. In every cell that reads "measurement hurts" (five cells; `measurement_hurts_cells`), mean P lies within 1.4 percent of mean C, while F errs 1.1 to 4.3 percent more than C. In Part B, F selects hgbr in all 60 B-complete fits and C the random forest. Post hoc, F − R is negative in all nine Part B cells, and its interval lies below zero for seed 211 at B-complete and B-none (`posthoc-arm-minus-raw-partB.json`).

`strong-learners/` holds both analyses and the reconcile check for the rerun and for the derived baseline, `learner-adequacy.json`, `strong-learner-report.json` (fixed-learner D, per-family selection, selection counts), `strong-learner-summary.json`, the post hoc F − R and C − R contrasts, both run configurations, the run log, and the driver scripts. The release asset `descriptor-information-strong-learners-v1.tar.xz` (about 237 MB, SHA-256 `80fcccc657fafcfe68d55a899cf5008977c2e6e7e0994cffadb4e71a32d04643`) will hold every fit with its per-candidate predictions, the derived baseline fits, and the run log.

```bash
# On DeltaAI (scripts/strong_stage.sh and scripts/strong_resume.sh), with copies of the prepared trees.
python tools/descriptor_ladder.py run --strength-indicator --strong-learners --rungs R0 N1 N2 N3 N4 R3-TFI R3-Heis R4 R5 \
    --arms A C F P --seeds 1-20 --datasets $PRIMARY --archive $ARCHIVE --gate-file artifacts/descriptor-information/gate/gate_r0.json \
    --out runs/partA-strong --workers 16
python tools/qaoa_intermediate.py run --strength-indicator --strong-learners --data-root runs/qaoa-data \
    --gate-file artifacts/descriptor-information/gate/gate_r0.json --out runs/partB-strong --workers 72
python tools/strong_learner_derive.py --fits runs/partA-strong/fits --fits runs/partB-strong/fits --out runs/derived-baseline/fits
D=artifacts/descriptor-information/strong-learners; RULE=docs/frozen-rules/2026-10-03-strong-learners.md
python tools/descriptor_ladder_analysis.py --fits runs/partA-strong/fits --fits runs/partB-strong/fits \
    --cache runs/partA-strong/cache --cache runs/partB-strong/cache --original-fits runs/derived-baseline/fits \
    --follow-up-rule $RULE --out $D/analysis-a.json
python tools/descriptor_ladder_analysis_b.py --fits runs/partA-strong/fits runs/partB-strong/fits \
    --cache runs/partA-strong/cache runs/partB-strong/cache --follow-up-rule $RULE --out $D/analysis-b.json
python tools/descriptor_ladder_reconcile.py $D/analysis-a.json $D/analysis-b.json --out $D/reconcile.json
# The same three commands on runs/derived-baseline/fits (with --original-fits at the 2026-10-03 fits)
# give derived-analysis-a.json, derived-analysis-b.json, and derived-reconcile.json.
python tools/learner_adequacy.py --fits runs/partA-strong/fits --fits runs/partB-strong/fits \
    --cache runs/partA-strong/cache --cache runs/partB-strong/cache --analysis-a $D/analysis-a.json --out $D/learner-adequacy.json
python tools/strong_learner_report.py --fits runs/partA-strong/fits --fits runs/partB-strong/fits \
    --cache runs/partA-strong/cache --cache runs/partB-strong/cache --out $D/strong-learner-report.json
python tools/strong_learner_summary.py --analysis-a $D/analysis-a.json --derived-analysis-a $D/derived-analysis-a.json \
    --report $D/strong-learner-report.json --macos-analysis-a artifacts/descriptor-information/strength-indicator/analysis-a.json \
    --derived-fits runs/derived-baseline/fits --sweep-2048-fits runs/shots-2048/fits --out $D/strong-learner-summary.json
# Post hoc: F - R and C - R on the rerun.
python tools/descriptor_ladder_posthoc.py --part B --fits runs/partB-strong/fits --cache runs/partB-strong/cache --out $D/posthoc-arm-minus-raw-partB.json
python tools/descriptor_ladder_posthoc.py --part A --fits runs/partA-strong/fits --cache runs/partA-strong/cache --out $D/posthoc-arm-minus-raw-partA.json
```
