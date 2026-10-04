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
