<a id="readme-top"></a>

<div align="center">

# qem-bench

**Benchmark primitives for distribution-shift reliability in learned quantum error mitigation.**

[Install](#install) &nbsp;|&nbsp;
[Quickstart](#quickstart) &nbsp;|&nbsp;
[Reproducibility](#reproducibility) &nbsp;|&nbsp;
[Citation](#citation)

</div>

`qem-bench` is a classically simulated benchmark for studying whether learned quantum error
mitigation (QEM) remains reliable under distribution shift with explicit circuit-evaluation budget
accounting. It generates deterministic circuit datasets, runs the methods implemented in this
release, and writes accuracy, harm, and ledger results. The split-v2 runner enforces declared tier
feasibility before method execution. It does not construct or enforce a matched-budget comparison
across methods.

> [!NOTE]
> v0.1.0 ships executable benchmark primitives and legacy-v1 micro presets. It also ships split-v2
> generation, validation, and runner paths, but no populated paper-scale split-v2 dataset.

## Install

Python 3.10 or later is required. For the frozen Windows AMD64 environment, use CPython 3.12.13
and install the exact transitive lock before installing the package:

```powershell
$env:QEM_BENCH_CI_LOCK_SHA256 = python tools/verify_ci_lock.py --digest-only
python -m pip install --only-binary=:all: -r requirements/ci-py312.lock
python -m pip install --no-deps .
python -m pip check
```

The lock is the environment for reproducing the two frozen item streams. An ordinary install from
the declared compatibility ranges is `python -m pip install .`, but it does not carry the frozen
hash guarantee.

## Quickstart

This example was run with the locked CPython 3.12 environment and the installed wheel:

```powershell
qem-bench generate --preset t0-micro --out data/t0-micro
qem-bench run --data data/t0-micro --out results/t0-micro
```

The first command writes `items.jsonl` and `manifest.json`. It generates 32 observable items and
prints the frozen `t0-micro` hash shown below. The second command writes `results.json` and prints a
table for raw, ridge, digital ZNE, the restricted-feature Liao-style ablation, and the four
surrogate controls. Treat the numerical output as a smoke-test result, not as a scientific finding.

## Included in v0.1.0

| Surface | Shipped behavior |
|---|---|
| Circuits | TFI, QAOA-MaxCut, Heisenberg, random Clifford, and near-Clifford families |
| Noise | Six named families, each on a monotone L1 to L4 placeholder severity grid |
| Labels | Exact statevector labels and Stim labels, logged separately from circuit evaluations |
| Methods | Raw estimates, ridge regression, local digital ZNE, and a registered Liao-style competitor path marked as a restricted-feature ablation, not a reproduction of the published feature encoding |
| Controls | Feature-only, noisy-value-only, shrinkage, and shuffled-noisy-value controls with an alarm |
| Splits | Closed S0 to S6 grammar with deterministic resolution, generation, validation, and split-v2 runner support |
| Accounting | Training, mitigation-extra, and test-prediction ledgers; split-v2 preflight uses per-method-cell combined caps L 2,500,000, M 25,000,000, and H 250,000,000 |
| Analysis | Per-cell metrics, statistical summaries, inference utilities, and a report layer |
| Calibration | A [severity-calibration proposal](qem_bench/noise/data/severity-calibration.proposal.json) that does not alter the shipped placeholder grids |
| Outputs | Canonical item streams, manifests, method and harm metrics, per-method ledgers, and report artifacts |

The local digital ZNE path uses deterministic global unitary folding at scale factors 1, 3, and 5.
Its behavior is cross-checked against Mitiq by the test suite.

<details>
<summary><b>Current Boundary and Non-Goals</b></summary>

v0.1.0 implements the S0 to S6 split grammar and its generation, validation, and runner paths. It
has no populated paper-scale split-v2 dataset. CDR and vnCDR runner methods remain absent; the
generated CDR viability probe is feasibility evidence, not a method implementation. A Liao-style
competitor path is present, explicitly a restricted-feature ablation rather than the headline
paper competitor. The reject-and-fallback diagnostic is not shipped: run-v2 artifacts cannot
establish that a method's predictions were produced without access to the test labels, so a
reject score fitted from one cannot be shown to be label-free. That layer returns when fitting
runs inside a trusted stage before labels are exposed, or when artifacts carry a verifiable
producer signature. The L1 to L4
values remain placeholders; the shipped severity-calibration proposal is not applied. The package
uses no quantum hardware and makes no quantum-advantage claim.

The runner checks declared tier feasibility. It does not equalize realized method costs, schedule a
matched-budget roster, or establish that two reported methods used matched budgets.

Prospective protocol decisions maintained outside this repository do not define shipped behavior.
For v0.1.0, the source code, generated manifests, `PROTOCOL-TRACE.md`, and tests govern the
executable contract that a public reader can inspect.

</details>

## Extension Interface

v0.1.0 exposes built-in methods through the CLI and Python modules. Python callers can add phased
methods with `MethodRegistration`, `register_method`, and `unregister_method`. This programmatic
interface is tested, but it is not declared stable across v0.x releases.

## Reproducibility

The frozen dataset hashes are SHA-256 digests of canonical JSON item rows sorted by `item_id`:

| Preset | Frozen Dataset Hash |
|---|---|
| `t0-micro` | `b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7` |
| `t0-smoke` | `edf5837e0da10f1dc93151d5b29d1855f66ad76341ff6c28019096308f8fcdf3` |

These hashes guarantee exact item-row identity only for the named preset, its default configuration,
CPython 3.12.13 on Windows AMD64, `PYTHONHASHSEED=0`, and
`requirements/ci-py312.lock` with canonical-LF SHA-256
`20ce2461bc2e6f70e14eeb0feaf3fe7e3cb9e9af75031da04671eab99c2bfd70`. They cover sampled noisy
values, exact labels, seeds, circuit metadata, and split assignments serialized in the rows. They do
not guarantee identical result files, method performance, wall-clock time, or cross-platform item
streams. Git pins the lock to LF on checkout, and the verifier also normalizes CRLF checkouts before
computing this digest.

Regenerate and check both streams with:

```powershell
$env:PYTHONHASHSEED = "0"
$env:QEM_BENCH_CI_LOCK_SHA256 = python tools/verify_ci_lock.py --digest-only
qem-bench generate --preset t0-micro --out data/frozen/t0-micro
qem-bench generate --preset t0-smoke --out data/frozen/t0-smoke
python tools/assert_frozen_hashes.py --root data/frozen
```

Every dataset manifest records package versions, seed derivation, the active severity grids, the
feature specification, and the dataset hash. A frozen-artifact claim must also record the expected
and observed lock digests and whether they match. Circuit cost is reported as backend calls times
shots in three buckets. Exact-simulation label calls are recorded separately and never counted as
circuit evaluations.

## Citation

Use GitHub's **Cite this repository** control or the software metadata in [`CITATION.cff`](CITATION.cff).
There is no paper citation in this release. The citation file contains a marked placeholder for one
after a paper reference becomes public and verifiable.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the locked setup, test commands, frozen-hash change
rule, and CI gate. Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

`qem-bench` is distributed under the [BSD 2-Clause License](LICENSE).
