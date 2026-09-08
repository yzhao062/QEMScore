# QEM-Bench campaign archive, version 1

The outputs of the frozen controlled campaign reported in the QEM-Bench paper.
This archive exists so that every number the paper states about that campaign can
be checked by someone who did not run it.

## What campaign this is

| | |
|---|---|
| Started | 2026-09-06T07:51:32Z |
| Design frozen | 2026-09-06T08:59:14Z |
| Completed | 2026-09-06T11:59:50Z |
| Code revision | `22fb3fd3c9e12fffd1c8a5dbee9cee1e4d56b4fb` |
| Root seed | 20260904 |
| Bootstrap draws, confidence | 10,000 at 0.95 |
| Frozen-design manifest SHA-256 | `060a303e646ca460332a1b2737efa943496badb9dc2f0b356a92cbce707d24ed` |
| Settings | 12: two regimes (`shipped`, `large`) by three seeds (101, 211, 307) by two training sizes (160, 640) |

`report.json` records `design_frozen`, `freeze_verified`, `fit_bindings_verified`,
`audit_complete` and `fitted_from_a_clean_tree`, all true, and the twelve dataset
hashes.

## What is in it

| Path | Contents |
|---|---|
| `report.json` | The campaign report: shares, contrasts, gates, diagnostics, replication, roster and audit status, dataset hashes |
| `tables.json` | The tabulated results |
| `audit.json` | Per-setting independent audit checks and their counts |
| `campaign-manifest.json` | The frozen design: settings, roles, circuits per family, audit rules, environment, code and dataset identities |
| `structure.json` | The run's structural record |
| `STATUS.txt`, `campaign.log` | Run status and log |
| `fit-bindings/` | One binding per setting, tying each fit to the run that produced it |
| `rosters/<setting>/results.json` | Per-method test predictions, per-cell error metrics, macro summaries and the circuit-evaluation ledger, for all eight rostered methods |
| `rosters/<setting>/binding.json` | The roster's binding record |
| `records/<setting>.json` | Selection and evaluation roles, counts, selected configurations, validation and untouched-test predictions, and item identities |
| `data-manifests/<setting>/manifest.json` | The dataset manifest for that setting, including per-item identities and hashes |

Eight methods appear in every roster: `raw` (baseline), `zne` (QEM baseline),
`liao` (competitor), `ridge` (learned), `feat-only`, `noisy-only` and `shrinkage`
(controls), and `shuf-noisy` (diagnostic). The paper's four-arm decomposition
uses `raw`, `feat-only`, the capacity-matched control derived in the contrast
stage, and `liao`.

## What is not in it, and why

The generated item streams and their sidecars, `data/<setting>/items.jsonl` and
`data/<setting>/sidecars/`, are excluded. They are 771 MB and the release's
deterministic generators reproduce them from the recorded seed and dependency
lock. Each setting's dataset manifest is included here precisely so that a
regenerated stream can be checked: `report.json` carries the twelve dataset
hashes the campaign computed, and a regeneration that does not match them is
detectable rather than silent.

This is a judgement, not a certainty. If the generators or the dependency lock
drift far enough that regeneration stops matching, this archive will document
the mismatch but will not repair it. The excluded material is 771 MB and can be
supplied as a second version if that becomes necessary.

## How to verify

`SHA256SUMS.txt` lists every file in this archive except itself and this manifest.
From the extracted directory:

```
sha256sum -c SHA256SUMS.txt
```

67 files, 277,466,583 bytes. The tree fingerprint, formed by hashing the UTF-8
concatenation of `<file SHA-256>  <relative POSIX path>\n` over those files in
sorted relative-path order, is:

```
c3b79a0766c1d48117995826a2ee48a1f0b2a1955931f314a9d298dbc9611821
```

## What this archive does not establish

It holds what the campaign recorded. It is not an independent re-derivation.

The circuit-evaluation ledger in each roster entry carries `B_train`, `B_extra`,
`B_pred`, `nominal_total`, `realized_test_total` and
`circuit_evals_per_mitigated_expectation`. Those counts are each method's own
declaration, computed by the runner from the values a method returns, and
`audit.json` checks none of them. Read them as declared cost, not as measured
cost.

The audit covers circuit identity, sidecars, observables, noisy estimates, exact
and independent labels, structure, histogram replay, parameter draws, pool
descriptors and zero-noise replay. Its per-setting counts are in `audit.json`.
Anything outside that list is unaudited.
