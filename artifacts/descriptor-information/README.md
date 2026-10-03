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
  - `reconcile.json`: Cross-script reconciliation report confirming point estimates, confidence intervals, labels, and rung statements agree to rule tolerance.

  - `reconcile.json`: the output of `tools/descriptor_ladder_reconcile.py` on the two analyses: 63 cells, 1,668 compared values, largest difference 2.8e-15, identical labels and rung statements (passed).

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
