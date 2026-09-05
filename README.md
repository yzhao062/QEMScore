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
> v0.1.0 is an alpha research artifact: eight registered methods, including four controls or
> diagnostics; in-distribution S0 and five working OOD axes (S1, S2, S3, S4, S6).
> S5 observable extrapolation cannot generate a dataset. CDR, vnCDR, budget equalization, and
> risk routing are absent. No populated paper-scale dataset or production findings ship.
> Start with the [tiny quickstart](#quickstart); use the [locked recipe](#reproducibility) to
> reproduce the two frozen item streams.

## Install

Python 3.10 or later is required. Use a virtual environment. The commands below use PowerShell
and assume that environment is active. From a downloaded release wheel, with no clone needed:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pip install ./qem_bench-0.1.0-py3-none-any.whl
python -m pip check
```

The MKL setting avoids a complex Schur decomposition crash in the Windows Miniforge build.
On a POSIX shell, set the same environment variables with `export NAME=value`.
If you have a source checkout or unpacked source distribution instead, run `python -m pip install .`
from its root. These ordinary installs resolve the declared compatibility ranges and do not carry
the frozen hash guarantee. To install the exact lock, follow the
[reproducibility recipe](#reproducibility).

The wheel includes the preset generators and noise JSON resources, so it can generate the tiny
datasets locally. It does not bundle pre-generated CI datasets, the test suite, or paper-scale
data. The source distribution includes the tests, lock, verification tools, and example report.

## Quickstart

Run these commands from a writable directory, including outside a source checkout:

```powershell
qem-bench generate --preset t0-micro --out data/t0-micro
qem-bench run --data data/t0-micro --out results/t0-micro
```

The first command writes `items.jsonl` and `manifest.json`. It generates 32 observable items and
prints their dataset hash. Compare it to the frozen value only under the locked recipe below.
The second command writes `results.json` and prints a
table for raw, ridge, digital ZNE, the restricted-feature Liao-style ablation, and the four
controls or diagnostics. This legacy-v1 preset has train and test roles and does not establish
OOD reliability. Treat its numerical output as a smoke-test result.

For the built-in split-v2 S0 preset, which also has a source-validation role:

```powershell
qem-bench generate --preset s0-t0-micro --out data/s0-t0-micro
qem-bench run --data data/s0-t0-micro --tier L --out results/s0-t0-micro
python -c "from qem_bench.validation import validate_split_artifact; items, manifest = validate_split_artifact('data/s0-t0-micro'); print(len(items), 'validated items')"
```

Use a new output directory for each split-v2 generation. The console script has exactly two
subcommands, `generate` and `run`; `qem-bench --help` lists them. Validation is a Python API,
including `validate_split_artifact` for split-v2 datasets and
`qem_bench.runner.run.validate_run_artifact` for result dictionaries. There is no `validate`
subcommand or config-file CLI. Custom OOD datasets use the Python `SplitSpec` and `generate_split`
APIs; the only built-in split-v2 CLI preset is `s0-t0-micro`.

Reports use a separate module entry point and require the `report` extra. With the same wheel
available, render the quickstart result using:

```powershell
python -m pip install "./qem_bench-0.1.0-py3-none-any.whl[report]"
python -c "import json; from pathlib import Path; Path('results/report-manifest.json').write_text(json.dumps({'schema_version': 'qem-bench-report-manifest-v1', 'runs': [{'results': 't0-micro/results.json'}]}), encoding='utf-8')"
python -m qem_bench.reports --manifest results/report-manifest.json --out results/report
```

For a source install, the extra is `python -m pip install '.[report]'`. A report manifest is
distinct from a dataset manifest; its result paths are relative to its own directory. The
default report covers `raw`, `ridge`, and `zne` and writes `table1.tex`, `figure1.pdf`, and
`report-trace.json`. Keep legacy-v1 and split-v2 runs in separate report manifests.

## Included in v0.1.0

| Surface | Shipped behavior |
|---|---|
| Circuits | TFI, QAOA-MaxCut, Heisenberg, random Clifford, and near-Clifford families |
| Noise | Six named families, each on a monotone L1 to L4 placeholder severity grid |
| Labels | Exact statevector labels and Stim labels, logged separately from circuit evaluations |
| Methods | `raw`, `ridge`, `zne` (local digital ZNE), and `liao` (restricted-feature ablation) |
| Controls and diagnostics | `feat-only`, `noisy-only`, `shrinkage`, and `shuf-noisy`, with a surrogate-control alarm |
| Splits | S0 plus S1, S2, S3, S4, S6 generation, validation, and runner support; S5 has grammar and validation definitions but refuses generation |
| Accounting | Training, mitigation-extra, and test-prediction ledgers; split-v2 preflight uses per-method-cell combined caps L 2,500,000, M 25,000,000, and H 250,000,000 |
| Analysis | Per-cell metrics, statistical summaries, inference utilities, the `evaluate_incremental_value` gate, and a report layer |
| Calibration | A [severity-calibration proposal](qem_bench/noise/data/severity-calibration.proposal.json) that does not alter the shipped placeholder grids |
| Outputs | Canonical item streams, manifests, method and harm metrics, per-method ledgers, and report artifacts |

The local digital ZNE path uses deterministic global unitary folding at scale factors 1, 3, and 5.
Its behavior is cross-checked against Mitiq by the test suite.

The `liao` arm omits the published native-gate count vector, angle bins, and sparse
Pauli-observable encoding. It is a restricted-feature ablation, not a reproduction of the
published method. Legacy-v1 uses its random-forest arm; split-v2 selects between random-forest
and MLP arms on source validation data. The MLP epoch count, stopping rule, dropout, and weight
decay are package declarations, not verified published settings.

| Split | Axis | Release status |
|---|---|---|
| S0 | Circuit instance, in-distribution | Generates and runs; built-in `s0-t0-micro` preset |
| S1 | Noise family | Generates and runs through the Python split API |
| S2 | Noise strength | Generates and runs through the Python split API |
| S3 | Circuit family | Generates and runs through the Python split API |
| S4 | Family-native depth | Generates and runs through the Python split API |
| S5 | Observable class | Refuses generation when validation or test observables are absent from training |
| S6 | Shot count | Generates and runs through the Python split API |

OOD training and validation stay in the source domain; testing uses the target domain.
Random Clifford data are labeled as the Clifford-control stratum, and `generate_report`
enforces the separation: Clifford-control runs never enter the continuous-regression
headline, and they are rendered as their own stratum section.

The Python API `qem_bench.stats.evaluate_incremental_value` compares a full model with the
`feat-only` and `noisy-only` controls on source validation, using circuit-blocked paired
intervals. It returns a dictionary whose `status` is `passed`, `failed`, or `not_evaluable`;
compare that field explicitly, because the dictionary is always truthy. The runner does not
invoke this gate. `tools/measure_incremental_value.py` drives it end to end.

## Not implemented yet

CDR and vnCDR have no runner methods; the CDR cost formula and viability probe do not implement
them. Lasso, HistGradientBoosting, and ensemble-combination runner methods are absent as well.
The random forest inside `liao` is one ablation arm. There is no reject-and-fallback or risk-routing
layer, and it is outside the scope of the first preprint.

The runner checks declared tier feasibility. It does not equalize realized method costs, spend
unused allowance on extra shots, or schedule a matched-budget roster. Same-tier results can have
different realized costs. S5 remains unavailable for dataset generation despite its grammar and
validator definitions. A four-command, config-first CLI is not implemented.

No populated paper-scale split-v2 dataset or production experiment ships. L1 to L4 severity
values remain placeholders; the calibration proposal is not applied, so equal severity labels
do not establish matched noise severity across families. The package uses no quantum hardware
and makes no quantum-advantage claim.

Prospective protocol decisions maintained outside this repository do not define shipped behavior.
For v0.1.0, the source code, generated manifests, `PROTOCOL-TRACE.md`, and tests govern the
executable contract that a public reader can inspect.

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

Use a source checkout or unpacked source distribution for this recipe: `requirements/` and
`tools/` are not wheel contents. The recorded lock targets CPython 3.12.13 on Windows AMD64;
the current CI workflow uses CPython 3.12.10. Local release checks also reproduce both streams
on Miniforge CPython 3.12.12. These are distinct environments, not a cross-version guarantee.

In a fresh environment, from the source root, install the lock, then regenerate and check both
streams. On Windows, keep the checkout and output paths short to accommodate split-v2 sidecars.

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONHASHSEED = "0"
$env:QEM_BENCH_CI_LOCK_SHA256 = python tools/verify_ci_lock.py --digest-only
if ($LASTEXITCODE -ne 0) { throw "CI lock verification failed" }
python -m pip install --only-binary=:all: -r requirements/ci-py312.lock
python -m pip install --no-deps .
python -m pip check
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
The software citation names Yue Zhao, University of Southern California, and version 0.1.0.
There is no verified paper citation yet. The citation file has a commented, single-line
`preferred-citation` template to fill with the public preprint metadata when available.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the locked setup, test commands, frozen-hash change
rule, and CI gate. Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

`qem-bench` is distributed under the [BSD 2-Clause License](LICENSE).
