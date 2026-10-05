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

## Tests

`reanalysis/mlqem/tests/test_throwaway.py` (mlqem environment) tests the encoding against `encode_data`, the arms, determinism, the runner, the scheduler's validation loss, the manifest and step-file checks, and the environment check on generated circuits only. `tests/test_mlqem_reanalysis.py` (py312) tests `analyze.py` on synthetic fits: the shared Rcal file, missing fits, reordered circuits, identity mismatches, the coherent note, the manifest inventory, and the reproduction bounds.
