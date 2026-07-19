# qem-bench

A controlled benchmark for distribution-shift reliability in learned quantum error
mitigation (QEM). Learned mitigators map noisy expectation values to ideal ones;
qem-bench measures when they stay accurate, when they overcorrect and make the raw
estimate worse, and whether pre-label risk signals can route risky cases to a fixed
fallback.

Status: the shipped slice runs five circuit families end to end: transverse-field
Ising, QAOA-MaxCut, Heisenberg, random Clifford, and near-Clifford. It provides six
noise families on the L1 to L4 grid, exact statevector and Stim labels, raw and ridge
paths, and digital zero-noise extrapolation with a behavioral Mitiq cross-check. MAE
and excess absolute loss use three-bucket circuit-evaluation accounting. The frozen
surrogate controls (feature-only, noisy-value-only, shrinkage, and
shuffled-noisy-value) and the surrogate alarm remain part of every run. The S1 to S6
split grammar, CDR and vnCDR, and the remaining planned baselines arrive later.

## Quick Start

```bash
pip install -e .
qem-bench generate --preset t0-smoke --out data/t0
qem-bench run --data data/t0 --out results/t0
```

Small integration presets are also available as `t0-micro`, `t0-qaoa-micro`,
`t0-heisenberg-micro`, `t0-rc-micro`, and `t0-nc-micro`.

`generate` writes a deterministic dataset (items plus a manifest with named seed
streams and a canonical dataset hash). `run` trains the learned baseline on the train
split only, scores every method on the test split, and prints an accuracy, harm, and
budget table.

## Design Rules Carried by This Slice

- Exact ideal labels come from simulation and are logged separately; they are never
  counted as circuit evaluations.
- The budget unit is the circuit evaluation (backend calls times shots), reported in
  three buckets: training data, mitigation-specific extra circuits, and test-time
  base measurement.
- Train, validation, and test circuit instances are disjoint; test labels never touch
  model selection.
- The headline harm endpoint is per-item excess absolute loss over the raw estimate,
  reported as distributions and totals; overcorrection rate is descriptive.
- Learned baselines are metadata-free by default: model inputs carry circuit and
  observable structure plus the noisy estimate, never exact noise parameters.

The paper-grade design (split grammar S0 to S6, baseline roster, reject-option
reference policy, statistical protocol) lives in the private planning hub and lands
here incrementally.
