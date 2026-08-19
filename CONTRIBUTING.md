# Contributing to qem-bench

Contributions should preserve the public, executable contract in the source, generated manifests,
`PROTOCOL-TRACE.md`, and tests. Open an issue before a change that alters benchmark semantics,
dataset schemas, dependency pins, cost accounting, or frozen item streams.

## Development Setup

Use CPython 3.12.13 on Windows AMD64 for the frozen-stream checks:

```powershell
$env:QEM_BENCH_CI_LOCK_SHA256 = python tools/verify_ci_lock.py --digest-only
python -m pip install --only-binary=:all: -r requirements/ci-py312.lock
python -m pip install --no-deps -e .
python -m pip check
```

Run the complete suite and the end-to-end frozen checks:

```powershell
python -m pytest -q --strict-markers tests
$env:PYTHONHASHSEED = "0"
qem-bench generate --preset t0-micro --out data/frozen/t0-micro
qem-bench generate --preset t0-smoke --out data/frozen/t0-smoke
python tools/assert_frozen_hashes.py --root data/frozen
```

## Frozen Item Streams

The two frozen hashes are:

- `t0-micro`: `b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7`
- `t0-smoke`: `edf5837e0da10f1dc93151d5b29d1855f66ad76341ff6c28019096308f8fcdf3`

A change that alters either hash requires an explicit protocol decision. Do not make a failing check
pass by replacing an expected hash in code, tests, workflows, or documentation. The proposed change
must identify the changed item fields, explain why the stream should change, and record the effect
on reproducibility before any new value is declared.

## Pull Request Gate

The required `foundation-gate` aggregates the frozen-stream, smoke, and Mitiq reference jobs. Changes
to the dependency contract, its verification tools, or the dependency-drift workflow also run the
dependency-drift workflow. A pull request is ready for review only when the local commands above and
all applicable CI jobs pass.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
