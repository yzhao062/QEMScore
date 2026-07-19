# qem-bench

A controlled benchmark for distribution-shift reliability in learned quantum error
mitigation (QEM). Learned mitigators map noisy expectation values to ideal ones;
qem-bench measures when they stay accurate, when they overcorrect and make the raw
estimate worse, and whether pre-label risk signals can route risky cases to a fixed
fallback.

Status: walking-skeleton stage. One vertical slice runs end to end: transverse-field
Ising (Trotter) circuits, a depolarizing-plus-readout noise model at three severity
levels, exact statevector labels, a raw baseline, and a ridge prediction path, scored
with MAE and excess absolute loss under three-bucket circuit-evaluation accounting.
The slice ships the frozen surrogate-control family (feature-only, noisy-value-only,
shrinkage, shuffled-noisy-value) and a surrogate alarm: a mitigation interpretation
of any learned score is gated on those controls, and on this slice the alarm fires,
so the ridge score is a plumbing checksum rather than evidence of mitigation.

## Quick Start

```bash
pip install -e .
qem-bench generate --preset t0-smoke --out data/t0
qem-bench run --data data/t0 --out results/t0
```

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
