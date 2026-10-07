# ML-QEM on Its Own Data: The Capacity-Matched Control

This directory applies the paper's capacity-matched control to a published learned mitigator on its own data: ML-QEM (Liao et al., *Nature Machine Intelligence* 6, 1478 to 1486, 2024; arXiv:2309.17368). The frozen rule is `docs/frozen-rules/2026-10-04-mlqem-own-data.md`; it fixes the data, encoding, models, arms, estimands, and reading before any fit on these data.

## Settings and Splits

Three simulated four-qubit Trotterized transverse-field Ising settings under FakeLima noise, Trotter steps 0 to 14 (`SETTINGS` in `encode.py`):

| Setting | Directory | Training | Validation | Test |
|---|---|---|---|---|
| `no_readout` | `ising_init_from_qasm_no_readout` | `train/` (4,500) | `val/` (1,500) | `val_extra/` (4,500) |
| `readout` | `ising_init_from_qasm` | `train/` (4,500) | `val/` (1,500) | `val_Zonly/` steps 0 to 14 (2,250) |
| `coherent` | `ising_init_from_qasm_coherent` | `train/` (4,500) | `val/` (1,500) | `val/` (1,500; no separate test split is released) |

The `no_readout` test circuits are those behind the published predictions in `docs/paper_figures/no_readout_over_depths.pk` for steps 0 to 14. The published readout and coherent predictions (300 circuits per step) match no circuit of any released split.

## Published Errors

`python -m reanalysis.mlqem.published_errors --data-root upstream/ml-qem` (mlqem environment) computes, from the three figure files and over steps 0 to 14 (4,500 circuits each), the mean over circuits of the mean absolute error over the four observables:

| Setting (figure file) | Raw noisy | Random forest | MLP | OLS, all features | OLS, noisy values | GNN | ZNE |
|---|---|---|---|---|---|---|---|
| No readout (`no_readout_over_depths.pk`) | 0.060945 | 0.015510 | 0.022274 | 0.046249 | n/a | 0.051041 | 0.041865 |
| Readout (`incoherent_over_depths.pk`) | 0.080840 | 0.016152 | 0.024036 | 0.048497 | 0.049822 | 0.052030 | 0.064073 |
| Coherent (`coherent_over_depths.pk`) | 0.107758 | 0.024191 | 0.064121 | 0.085423 | 0.087167 | 0.086222 | 0.093211 |

Only the no-readout row is on this analysis's test circuits; its random-forest and MLP values are the references of the reproduction check (`analyze.PUBLISHED_ERRORS`).

## Encoding, Models, and Arms

- Encoding: `blackwater.library.learning.mlp.encode_data` as `h17_compare_over_steps.ipynb` calls it (58 features: eight FakeLima device-noise values, six gate counts, forty angle bins, four noisy values). FakeLima's properties are used unchanged in every setting, as in the published notebooks.
- Random forest: one `RandomForestRegressor` per observable, 300 trees, every other setting at its default (no depth limit), as `h15_random_forest.ipynb` cell 11 sets it; `random_state` = learner seed + observable index. The `rfr_list_maxdepth10.pk` in the readout and coherent model directories is a variant that `h17_compare_over_steps.ipynb` does not load.
- MLP: `MLP1` (64 hidden ReLU units), Adam (1e-3), mean squared error, batch 32, 100 epochs; `ReduceLROnPlateau` (factor 0.1, patience 15, floor 1e-5) steps on the validation loss, as `h10_mlp.ipynb` cell 13 does.
- OLS: `LinearRegression` on all features.
- Arms: F (all features), C (the four noisy columns removed), P (the noisy columns permuted together across training rows, seeded by the learner seed), R (the raw noisy value), and Rcal (OLS on the noisy columns alone, the published `ols`; one fit per setting, `<setting>_ols_Rcal_seed1`, used by every model's comparison).

## Environments

- `reanalysis/environments/mlqem.yml`: Python 3.9.23 and pip 25.2 from conda-forge; every other package from PyPI at the exact versions of `mlqem-requirements.txt` (qiskit 0.43.2, qiskit-terra 0.24.1, qiskit-aer 0.12.1, pandas 1.5.3, numpy 1.23.5, scikit-learn 1.6.1, torch 2.8.0, torch-geometric 2.6.1). `mamba env create -f reanalysis/environments/mlqem.yml` builds it. Used by `encode.py`, `arms.py`, `run.py`, and `published_errors.py`. After creating it, install `blackwater` from the pinned checkout without touching the pins: `conda run -n mlqem pip install --no-deps -e upstream/ml-qem`. Unless `--no-manifest-check` is given, `run.py` stops before encoding if the checkout is not at `b1eccf8` or `blackwater` is imported from elsewhere; it appends the commit, the module path, and the package versions to `<output-dir>/run-log.jsonl`.
- The QEMScore `py312` environment runs `analyze.py`, which reads only NPZ and JSON files and reuses `tools/descriptor_ladder_analysis.bootstrap_draws`.

## Data

`reanalysis/download_upstream.sh` fetches `qiskit-community/ml-qem` at commit `b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3`, the commit the paper's other ML-QEM analyses use. Every file read here is identical at the repository's latest commit, `faf3e44`, which changes only the README. `upstream_data_sha256.json` lists the SHA-256 of all 120 data files the experiment reads and of the three published figure files. Encoding stops before any fit if a requested step file is missing, unlisted, or changed (`--no-manifest-check` exists only for rehearsals on generated circuits).

## Commands

```bash
# Fits (mlqem environment); every job requires the frozen rule and records its SHA-256.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  $MLQEM_PY -m reanalysis.mlqem.run --settings no_readout readout coherent --models ols rf mlp \
    --arms F C P R Rcal --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --data-root upstream/ml-qem --output-dir artifacts/mlqem-own-data/fits --workers 8 \
    --frozen-rule docs/frozen-rules/2026-10-04-mlqem-own-data.md
# Estimands, intervals, and labels (py312 environment).
PYTHONPATH=. python -m reanalysis.mlqem.analyze --fits artifacts/mlqem-own-data/fits \
    --out artifacts/mlqem-own-data/analysis.json --settings no_readout readout coherent \
    --models ols rf mlp --draws 10000 --bootstrap-seed 20261002 \
    --frozen-rule docs/frozen-rules/2026-10-04-mlqem-own-data.md
```

`analyze.py` reports, per setting and model, C, F, P, R, Rcal, D = C − F, D/C, F − R, F − Rcal, and P − F, with two-stage percentile intervals (learner seeds, then whole test circuits) and the label of D, plus a descriptive per-step table. Deterministic arms repeat their single prediction across the 20 learner-seed slots. Before computing, it requires every fit and its record, the rule's SHA-256, the manifest's data SHA-256 values, and identical circuit identifiers, targets, and steps across arms and seeds; any gap stops it. `reproduction_check` holds the rule's random-forest ratio and verdict and the MLP's ratio, each with the per-seed spread, and each cell carries `test_is_validation`.

## Exact Labels and Exact Descriptors

Round 9 follow-up analyses support exact statevector labels and exact circuit descriptors under `docs/frozen-rules/2026-10-06-round9-follow-ups.md`.

### 1. Generating Exact Labels and Manifests

`reanalysis.mlqem.exact` calculates exact statevector labels and circuit parameters for each circuit. It extracts the coupling constant J, the measurement basis, and Trotter step count.

```bash
PYTHONPATH=. $MLQEM_PY -m reanalysis.mlqem.exact \
    --data-root <upstream-dir> \
    --settings no_readout readout coherent \
    --splits <splits> \
    --out-dir <exact-dir> \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md
```

Each output NPZ file contains exact labels, archived labels, coupling J, basis indicators, and Trotter steps. The companion JSON manifest records source file and output SHA-256 digests with a statistical validation block.

Once all eight setting and split manifests are generated, evaluate the quality gate:

```bash
PYTHONPATH=. $MLQEM_PY -m reanalysis.mlqem.exact gate \
    --exact-dir <exact-dir> \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md
```

The gate verifies binomial shot-noise thresholds across all eight splits and writes `gate.json`.

### 2. Rescoring Against Exact Targets

`analyze.py` accepts `--targets {archived,exact}` (default `archived`), `--exact-dir <exact-dir>`, and `--fit-frozen-rule <rule>`.

```bash
PYTHONPATH=. python -m reanalysis.mlqem.analyze \
    --fits <fits-dir> \
    --out <out-path> \
    --settings no_readout readout coherent \
    --models ols rf mlp \
    --draws 10000 --bootstrap-seed 20261002 \
    --targets exact --exact-dir <exact-dir> \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md \
    --fit-frozen-rule docs/frozen-rules/2026-10-04-mlqem-own-data.md
```

Under `--targets exact`, the analyzer requires `gate.json` to exist and pass. It checks that archived labels in the exact NPZ match stored fit targets in float32 precision. It also verifies circuit identifiers and steps in order. It then rescores all arms against the exact targets. Each cell reports `C_minus_R`, `F_minus_R`, and `reference_error`. The output references the round-8 reproduction report.

### 3. Fitting with Exact Descriptors and Exact Targets

`run.py` accepts `--descriptors {encoding,exact}` (default `encoding`), `--train-targets {archived,exact}` (default `archived`), and `--exact-dir <exact-dir>`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  $MLQEM_PY -m reanalysis.mlqem.run \
    --settings no_readout readout coherent --models ols rf mlp \
    --arms F C P R Rcal --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --descriptors exact --train-targets archived --exact-dir <exact-dir> \
    --data-root <upstream-dir> --output-dir <fits-dir> --workers 8 \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md
```

When `--descriptors exact` is specified, five extra features append to the descriptor block. Learned arm F reads 63 features, arm C reads 59 features, and arm P permutes only the four noisy columns. When `--train-targets exact` is specified, training and validation use exact labels, while test targets remain archived references. Fits record `train_targets`, `descriptors`, and input SHA-256 values in their metadata records.

### 4. Secondary Refits and F/C Scoring

The primary cell (no readout error, random forest) receives two secondary fit sets at learner seeds 1 to 20. Both use exact training targets (`--train-targets exact`).

```bash
# Secondary refit with published encoding
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  $MLQEM_PY -m reanalysis.mlqem.run \
    --settings no_readout --models rf --arms F C --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --descriptors encoding --train-targets exact --exact-dir artifacts/mlqem-exact \
    --data-root <upstream-dir> \
    --output-dir artifacts/mlqem-own-data/round9/exact-training-encoding/fits \
    --workers 8 --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md

# Secondary refit with exact descriptors
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  $MLQEM_PY -m reanalysis.mlqem.run \
    --settings no_readout --models rf --arms F C --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --descriptors exact --train-targets exact --exact-dir artifacts/mlqem-exact \
    --data-root <upstream-dir> \
    --output-dir artifacts/mlqem-own-data/round9/exact-training-exact/fits \
    --workers 8 --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md
```

Each fits directory is scored against both exact and archived targets using `--analysis-arms F C`. Set `<descriptors>` to `encoding` for `exact-training-encoding` and to `exact` for `exact-training-exact`; the analyzer rejects a fit whose descriptor mode differs.

```bash
# Score against exact targets
PYTHONPATH=. python -m reanalysis.mlqem.analyze \
    --fits artifacts/mlqem-own-data/round9/<variant>/fits \
    --out artifacts/mlqem-own-data/round9/<variant>/analysis-exact.json \
    --analysis-arms F C --settings no_readout --models rf \
    --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --draws 10000 --bootstrap-seed 20261002 \
    --descriptors <descriptors> \
    --targets exact --exact-dir artifacts/mlqem-exact \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md

# Score against archived targets
PYTHONPATH=. python -m reanalysis.mlqem.analyze \
    --fits artifacts/mlqem-own-data/round9/<variant>/fits \
    --out artifacts/mlqem-own-data/round9/<variant>/analysis-archived.json \
    --analysis-arms F C --settings no_readout --models rf \
    --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --draws 10000 --bootstrap-seed 20261002 \
    --descriptors <descriptors> \
    --targets archived \
    --frozen-rule docs/frozen-rules/2026-10-06-round9-follow-ups.md
```

Under `--analysis-arms F C`, the analyzer evaluates only arms F and C. It computes C, F, D, and D/C with bootstrap intervals, as well as the D label. It requires identical `train_targets` across the roster and records it in each analysis report. Under `--targets exact`, it also reports `reference_error`.

## Tests

`reanalysis/mlqem/tests/test_throwaway.py` (mlqem environment) tests the encoding against `encode_data`, the arms, determinism, the runner, the scheduler's validation loss, the manifest and step-file checks, and the environment check on generated circuits only. `reanalysis/mlqem/tests/test_exact.py` (mlqem environment) tests exact labels on hand-built circuits, agreement with `cal_z_exp`, recovery of J and basis, and invariance of default paths. `tests/test_mlqem_reanalysis.py` (py312) tests `analyze.py` on synthetic fits: the shared Rcal file, missing fits, reordered circuits, identity mismatches, the coherent note, the manifest inventory, and the reproduction bounds.
