# Degree-Five Polynomial Diagnostic Refit

This tool independently reproduces and scripts the degree-five polynomial diagnostic reported in the manuscript (Section 4, Appendix A.4, Table `tab:poly-degree-5`).

## Purpose

The manuscript reports that an ordinary least-squares degree-five polynomial in the Hamiltonian coupling parameters alone, reading no measurement, achieves test macro MAE between 0.0000424 and 0.000292 across all six primary campaign evaluations (seeds 101, 211, 307 for Heisenberg and TFI at shipped step sizes, 640 training circuits). This error is 12 to 55 times lower than the full nonlinear measurement-using method ($F$).

Historically, the numbers were preserved as a transcribed summary (`historical-polynomial.json`). This tool provides a fully self-contained, deterministic refit directly from the package's circuit generators and the public campaign archive.

## Specification

The fit recipe follows the frozen reviewer diagnostic:
1. **Features**: Coupling parameters only:
   - Transverse-Field Ising (TFI): $J, h$ (2 continuous features).
   - Heisenberg: $J_x, J_y, J_z$ (3 continuous features).
   No quantum measurement, shot count, circuit identifier, or noise parameter is used as a predictor.
2. **Basis**: Total-degree monomials up to degree 5 with bias (`sklearn.preprocessing.PolynomialFeatures(5, include_bias=True)`):
   - TFI: $\binom{2 + 5}{5} = 21$ coefficients per observable.
   - Heisenberg: $\binom{3 + 5}{5} = 56$ coefficients per observable.
3. **Normalization**: Training inputs are standardized (mean-centered and divided by standard deviation) using training rows only.
4. **Training Rows and Collapse**:
   - 640 unique physical circuits per family.
   - Multiple severity copies in the raw dataset ($L_1, L_3$) share identical ideal labels for a given physical circuit and observable. These copies are collapsed to one training target per `(circuit_id, observable)`.
   - Separate fits are performed for each family and observable.
5. **Optimization**: Ordinary least squares via `numpy.linalg.lstsq(design, y, rcond=None)`. All design matrices have full column rank (21/21 and 56/56).
6. **Evaluation & Scoring**:
   - Evaluated on all untouched-test items (160 physical circuits $\times$ 2 severities $\times$ 2 observables = 640 test items per family per seed; 1,280 items per seed).
   - Scored with the manuscript's four-cell macro MAE metric, giving equal weight to the four cells: $(L_1, Z_{\text{mid}})$, $(L_1, ZZ_{\text{mid}})$, $(L_3, Z_{\text{mid}})$, $(L_3, ZZ_{\text{mid}})$.
7. **Paired Circuit-Blocked Bootstrap**:
   - Post-campaign paired bootstrap for $F - \text{Poly-5}$ on test rows.
   - Resamples physical circuits with replacement (10,000 draws, seed 20260904), keeping all 4 cell rows of a circuit together.

## Usage

Run with the project's Python 3.12 environment:

```bash
# from the repository root
PYTHONPATH=. python tools/refit_polynomial_diagnostic.py \
    /path/to/campaign-archive-v1 \
    artifacts/polynomial-refit
```

Optional arguments:
- `--couplings-csv <path>`: Path to a pre-extracted couplings CSV/gzip (e.g. `probe-couplings-shipped-n640.csv.gz`). If omitted, the tool regenerates circuits, couplings, and noiseless ideal labels on the fly using the package generator.
- `--n-bootstrap <int>`: Number of bootstrap draws (default: 10,000).
- `--bootstrap-seed <int>`: Base random seed for circuit resampling (default: 20260904).

## Output Files

1. `polynomial_test_predictions.csv`:
   - 3,840 rows (1,280 test items $\times$ 3 seeds).
   - Columns: `setting,seed,family,circuit_id,item_id,observable,severity,noise_family,ideal_expectation,control_C_pred,full_F_pred,poly5_pred,poly5_abs_error`.
2. `polynomial_refit_summary.json`:
   - Six-row macro MAE and per-cell MAEs.
   - Point comparison against `historical-polynomial.json` and Table `tab:poly-degree-5`.
   - 95% paired circuit-blocked bootstrap confidence intervals for $F - \text{Poly-5}$.
