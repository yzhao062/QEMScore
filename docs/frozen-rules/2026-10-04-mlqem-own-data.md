# Frozen Rule: The Capacity-Matched Control on a Published Pipeline's Own Data (ML-QEM, QEMScore, Post-Review)

Written 2026-10-04 (US Pacific time), before any model was fitted on the ML-QEM data named here. Three internal review panels (rounds 5 to 7) asked for the control to be applied to a published learned mitigator on its own data. QEM-Bench could not serve: its data repository holds no data. The paper's R4 rung already applies ML-QEM's gate-count and angle-bin encoding to the paper's spin chains; this rule applies the paper's control to ML-QEM's own circuits, encoding, and models.

Known when this rule was written: the ML-QEM paper (Liao et al., Nature Machine Intelligence 6, 1478 to 1486, 2024) and its repository. A feasibility study read the repository's notebooks, loaded its data files, and computed the errors of the published predictions stored in `docs/paper_figures/`. Over Trotter steps 0 to 14, the published random-forest test errors are 0.015510, 0.016152, and 0.024191 in the three settings below. The raw noisy errors are 0.060945, 0.080840, and 0.107758. `reanalysis/mlqem/published_errors.py` computes these values from the figure files; `reanalysis/mlqem/README.md` lists them with those of the other published models. Only the no-readout predictions are on released circuits (all 4,500 are the no-readout `val_extra/` circuits of steps 0 to 14). The readout and coherent predictions (300 circuits per step) match no circuit of any released split, so their errors serve for orientation only. Code was tested only on circuits generated for the purpose with a throwaway seed. No model was fitted on the data below, and no error of a refitted model on them was computed.

## Question

In ML-QEM's own simulated Trotterized transverse-field Ising settings, with its own feature encoding and models, how far do the published learners' gains depend on the noisy expectation values?

## Data

The repository `qiskit-community/ml-qem` at commit `b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3`, which `reanalysis/download_upstream.sh` already fetches for the paper's other ML-QEM analyses. Every file read here is identical at the repository's latest commit, `faf3e44`, which changes only the README. Each data file must match the SHA-256 recorded in `reanalysis/mlqem/upstream_data_sha256.json`, which lists all 120 files of the splits below. Encoding stops before the first fit on any requested step file that is missing, unlisted, or changed. Each circuit is a four-qubit first-order Trotter circuit of 0 to 14 steps, transpiled for the FakeLima device. Its coupling J is drawn uniformly from [0, 1], and its measurement basis from X, Y, and Z. Its labels are the four ideal single-qubit expectation values, and its noisy values come from the noisy simulator.

| Setting | Directory | Training | Validation | Test |
|---|---|---|---|---|
| No readout error | `ising_init_from_qasm_no_readout` | `train/` (4,500) | `val/` (1,500) | `val_extra/` (4,500) |
| Readout error | `ising_init_from_qasm` | `train/` (4,500) | `val/` (1,500) | `val_Zonly/` steps 0 to 14 (2,250) |
| Coherent over-rotation | `ising_init_from_qasm_coherent` | `train/` (4,500) | `val/` (1,500) | `val/` (1,500) |

Counts are circuits, all over Trotter steps 0 to 14. The no-readout test set holds the circuits behind the published predictions for those steps. The coherent setting releases no separate test split, so its validation split is also its test split, and its MLP results are marked accordingly.

## Encoding, Models, and Arms

**Encoding.** `blackwater.library.learning.mlp.encode_data` is called as in `h17_compare_over_steps.ipynb`. It gives 58 features per circuit: eight device-noise values of FakeLima, six gate counts, forty rotation-angle bins of width 0.1π, and the four noisy expectation values. The device-noise values come from FakeLima's properties unchanged in every setting, as in the published notebooks; within a setting they are constant.

**Models**, as the published notebooks set them (`h15_random_forest.ipynb`, `h10_mlp.ipynb`, `h12_ols.ipynb`):
- random forest: one `RandomForestRegressor` per observable with 300 trees and every other setting at its default (no depth limit, all features per split), as `h15_random_forest.ipynb` cell 11 sets it. That cell saves `model/ising_init_from_qasm_no_readout/rfr_list.pk`, the file `h17_compare_over_steps.ipynb` loads; the file itself is not released. The readout and coherent model directories hold a depth-10 variant, `rfr_list_maxdepth10.pk`, which h17 does not load. Here each forest's `random_state` is the learner seed plus the observable index;
- MLP: `MLP1` (one hidden layer of 64 ReLU units, four outputs) trained with Adam (step size 1e-3), mean squared error, batch size 32, and 100 epochs. `ReduceLROnPlateau` (factor 0.1, patience 15, floor 1e-5) steps on the validation loss each epoch; the final model predicts. Torch, numpy, and Python are seeded with the learner seed, with deterministic algorithms on one CPU thread;
- OLS: `LinearRegression` on all features.

**Arms.** F reads all 58 features. C is the same model with the four noisy columns removed. P permutes the four noisy columns together across training rows, with a permutation seeded by the learner seed; validation and test rows stay unpermuted. R is the raw noisy value. Rcal is OLS on the four noisy columns alone, the published `ols`; it is one deterministic fit per setting, and every model's comparison uses it. The random forest and the MLP run F, C, and P at learner seeds 1 to 20. OLS runs F and C once, being deterministic, and P at seeds 1 to 20; Rcal runs once.

**Running.** `python -m reanalysis.mlqem.run --data-root <ml-qem checkout> --frozen-rule` names this file. It runs all three settings and models, arms F, C, P, R, and Rcal, and learner seeds 1 to 20. The environment is `reanalysis/environments/mlqem.yml` with its lock file `mlqem-requirements.txt` (Python 3.9, qiskit 0.43.2, qiskit-aer 0.12.1, pandas 1.5.3, numpy 1.23.5, scikit-learn 1.6.1, torch 2.8.0, all from PyPI). `blackwater` is installed from the checkout with `pip install --no-deps -e`. OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS are set to 1. Before encoding, the runner requires the checkout to be at the commit above and `blackwater` to be imported from it. It then appends the commit, the module path, and the package versions to the run log. Every fit records its setting, model, arm, learner seed, feature columns, data-file SHA-256 values, this rule's SHA-256, versions, and wall time.

## Estimands, Uncertainty, and Labels

For each test circuit, the error is the mean over its four observables of |prediction − ideal|; an arm's error is the mean over test circuits. For each setting and model: C, F, P, R, and Rcal; D = C − F; D/C as a ratio of means inside each draw; F − R; F − Rcal; and P − F. `reanalysis/mlqem/analyze.py` computes them in the QEMScore environment, with the two-stage percentile bootstrap of `tools/descriptor_ladder_analysis.bootstrap_draws`: 10,000 draws, a fresh `numpy.random.default_rng(20261002)` per setting, model, and quantity, learner seeds then whole test circuits. A deterministic arm repeats its single prediction across the 20 learner-seed slots. Before computing anything, the analysis requires every fit of the requested settings, models, arms, and seeds. Every fit must carry this rule's SHA-256, the same data-file SHA-256 values (checked against the manifest), and the same circuit identifiers, targets, and steps in the same order. A missing or mismatched fit stops the analysis. Each coherent cell carries the note that its test split is its validation split. The labels are those of the earlier rules: "F beats C" if the D interval lies above zero, "C beats F" if below, and "not distinguished" otherwise. Per-step values of D and F − R are reported as points, descriptively.

## Reproduction Check

On the no-readout test set, `analyze.py` divides the mean over the 20 learner seeds of F's random-forest error by the published 0.015510 on the same circuits. The check passes if the ratio lies in [0.9, 1.1], bounds included. The output gives the ratio, the verdict, and the smallest, largest, and standard deviation of the 20 per-seed errors. The tolerance is a practical agreement criterion fixed in advance, not an interval calibrated against the variation of one historical fit. The published forests cannot be reproduced bit for bit. Their `random_state` is unset, so they draw from numpy's global generator after `fix_random_seed(0)` (h15 cell 9). The notebooks also read the training files in `os.listdir` order. The MLP's ratio to the published 0.022274 (h10 cells 10 to 13, also after `fix_random_seed(0)`) is reported with the same spread and no criterion. Every cell is reported whatever the verdict.

## Expectations Stated in Advance (Not Decision Rules)

1. The random-forest reproduction check passes.
2. For the random forest, F − R lies below zero in all three settings.

No label of D is predicted for any setting or model.

## Reading

The primary cell is the random forest in the no-readout setting, the published best learner on the published test circuits. Every other setting and model is reported beside it.
1. If the reproduction check passes, the paper reports D, D/C, and the label of every cell as the control applied to the published pipeline on its own data. "F beats C" means that the noisy values carry part of that pipeline's gain over a descriptor-only refit; D/C gives the share. "Not distinguished" means that the pipeline's accuracy is not distinguished from such a refit at 20 learner seeds. "C beats F" means that the refit without the noisy values is more accurate.
2. If the check fails, the paper reports a refit that does not reproduce the published error, with both values. It then draws no conclusion about the published pipeline itself.

No outcome changes a setting, model, arm, or split. Any analysis chosen after the results are seen is labeled post hoc.

## Deviations and Release

Any deviation is reported with its reason. The release adds every fit's predictions and records, the encoded feature matrices with the SHA-256 of their inputs, the analysis output, and the run log. The ML-QEM data are not redistributed; the commit and the file SHA-256 values identify them.
