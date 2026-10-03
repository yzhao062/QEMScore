# Learner-Seed Replication Artifact

This directory contains the replication artifact for the learner-seed experiment across the 6 primary settings (dataset seeds 101, 211, and 307; circuit families transverse-field Ising [TFI] and Heisenberg; 20 learner seeds per setting).

## Contents

- `bootstrap.json`: Two-stage percentile bootstrap results (10,000 draws, seed 20261001) for the capacity-matched gap $D = C - F$ and diagnostic quantities $P - C$ and $P - F$.
- `dataset_checks.json`: Integrity verification hashes for regenerated shipped datasets.
- `summary.json`: Summary metadata of all fitted models, candidate selections, and test errors.
- `inputs/`: Self-contained input files required to recompute the bootstrap intervals from scratch:
  - `fits/`: Per-fit metadata JSONs and test predictions (`.npz`) for all 20 learner seeds (`k01`..`k20`) across arms F, C, and P (plus anchor seeds).
  - `cache/`: Compact compressed test item caches (`shipped-s*.json.gz`) retaining only the schema fields read by the bootstrap: `item_id`, `circuit_id`, `family`, `noise_family`, `severity`, `observable`, `ideal_expectation`, `noisy_expectation`, `split`, and `stratum`.
  - `summary.json`: Local-path-free copy of the replication summary.

## How to Recompute

To recompute `bootstrap.json` from scratch using only the released inputs, run from the repository root:

```bash
python tools/two_stage_bootstrap.py \
    --run-dir artifacts/learner-seed-replication/inputs \
    --out artifacts/learner-seed-replication/bootstrap.json
```

Or equivalently, pointing directly at the artifact directory:

```bash
python tools/two_stage_bootstrap.py \
    --run-dir artifacts/learner-seed-replication \
    --out artifacts/learner-seed-replication/bootstrap.json
```

Parameters:
- Draws: 10,000
- RNG Seed: 20261001 (fresh default_rng per row and quantity)
- Percentiles: [2.5, 97.5]

See `docs/frozen-rules/2026-10-01-learner-seed-replication.md` for the pre-registered decision rule and specification.
