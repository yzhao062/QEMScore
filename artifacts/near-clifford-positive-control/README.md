# Near-Clifford Positive Control Artifact

This directory contains the release artifact for the near-Clifford positive control experiment across dataset seeds 101, 211, and 307 (setting S0, 10 qubits, native depth 4, 1–3 non-Clifford gates; 20 learner seeds per dataset seed).

## Contents

- `bootstrap_results.json`: Two-stage percentile bootstrap results (10,000 draws, seed 20261001) for the capacity-matched gap $D = C - F$, $S$, $P - F$, and $P - C$.
- `summary-s{101,211,307}.json`: Per-dataset-seed execution summaries with candidate choices, arm errors, and runtime diagnostics.
- `all_summary.json`: Combined execution summary for all three dataset seeds.
- `posthoc-label-strata.json`: post hoc label distribution of the three near-Clifford datasets, test errors and the gap D split into items with a zero label and items with a nonzero label, and the continuous-label comparison (R5 against R3-TFI from the descriptor-information experiment). Written by `tools/near_clifford_strata.py`; the command is step 7 of `artifacts/descriptor-information/README.md`.
- `inputs/`: Self-contained input files required to recompute the bootstrap intervals from scratch:
  - `s101/`, `s211/`, `s307/`:
    - `summary.json`: Dataset-seed run summary.
    - `test_predictions.json.gz`: Compressed test predictions for arms A, R, and learner seeds 1..20 for arms F, C, and P.
    - `test_items.json.gz`: Compact compressed test items retaining only the fields read by the bootstrap: `item_id`, `circuit_id`, `family`, `severity`, `observable`, `ideal_expectation`, `noisy_expectation`, and `split`.
  - `all_summary.json`: Local-path-free combined summary.

## How to Recompute

To recompute `bootstrap_results.json` from scratch using only the released inputs, run from the repository root:

```bash
python tools/two_stage_bootstrap_nc.py \
    --inputs-dir artifacts/near-clifford-positive-control/inputs \
    --output-file artifacts/near-clifford-positive-control/bootstrap_results.json
```

Or equivalently, using `--results-dir` and `--data-dir`:

```bash
python tools/two_stage_bootstrap_nc.py \
    --results-dir artifacts/near-clifford-positive-control/inputs \
    --data-dir artifacts/near-clifford-positive-control/inputs \
    --output-file artifacts/near-clifford-positive-control/bootstrap_results.json
```

Parameters:
- Draws: 10,000
- RNG Seed: 20261001
- Percentiles: [2.5, 97.5]

Note: When no fits pass the numerical RuntimeWarning filter, the sensitivity analysis decision is recorded as `"not_estimable"`. The frozen decision rule uses all fits.

See `docs/frozen-rules/2026-10-01-near-clifford-positive-control.md` for the pre-registered decision rule and specification.
