"""Re-derive a campaign dataset: every parameter draw, and a sampled replay.

Two passes, with different reach.

The exhaustive pass covers every pool instance. It rebuilds each pool descriptor
and checks that its content hash is the recorded pool identity, derives the pool
seed key from that descriptor, and replays every instance's parameter draw from
the master seed with a reimplementation of the draw order. It then rebuilds the
circuit, checks the canonical identity, compiles it, and compares the two-qubit
gate count and the compiled depth against every row that circuit produced. This
is the pass that sees a shifted random stream: a fault that consumes one extra
draw leaves every recorded seed intact and every recorded parameter consistent
with the circuit built from it, so nothing that reads the stored values can
detect it.

The sampled pass covers the frozen index rule. It replays the base measurement
with the recorded sampler seed and the digest-derived compilation seed and
compares the histogram exactly, recomputes the exact label, the noisy estimate
and the observable support, and, given a run artifact, replays each audited test
group's folded executions at the declared scales and compares the extrapolated
prediction. The scale seeds are derived from the documented rule rather than by
calling the shipped helper, because that rule is part of what is under audit.

What is coverage and what is a cross-check. `validate_split_artifact` already
recomputes each exact label from the circuit and each noisy estimate from the
stored counts, and already refuses a tampered sidecar through the hash chain. It
never replays a parameter draw, never re-executes the sampler, and never checks
the compiled structure. The report labels every check `new` or `rederived` on
that basis.

What is independent. `audit_protocol.py` writes both Trotter protocols out from
their specification and evolves the state with plain arrays, importing nothing
from `qem_bench`. Each sampled circuit's gate sequence is compared against the
one the shipped builder produced, and each sampled label against an independent
evolution. That is the fault class the rest cannot reach: a label routine or a
builder that is simply wrong agrees with itself in the generator, in the
validator, and in any re-derivation that calls it.

What none of it establishes. The noise model, the sampler, the folding and the
extrapolation are still shared, and that is a bounded scope rather than a
necessity: round 5 of the plan review built the noise models separately and
still reproduced every sampled histogram exactly, so independence and exact
replay are compatible. This audit simply does not do it. The consequence is a
real fault class it passes, and round 5 executed one: a sampler that requests
the L1 model for an L3 row keeps every parameter, identity, seed and recorded
severity, and a replay through the same wrong helper agrees with it. The
separately verified frozen noise-model tests cover model construction; they do
not cover which severity the sampler asks for. Hardening this would mean
checking the model attached to the simulator against independently specified
channels for the requested severity, which is a deterministic check that does
not replace exact replay. The histogram replay also covers the sample rather
than every execution, and the zero-noise comparison lands on the prediction
because the run artifact stores no folded histogram.

The sample is fixed before any score is inspected. Per family, regime, and seed
pool, the audited indices are training {0, 79, 159, 160, 399, 639}, validation
{0, 159, 319}, and test {0, 79, 159}. Indices that do not exist at a training
size are skipped, and the smaller size reuses the identical checks, so the two
sizes audit the same validation and test circuits. Across two families, two
regimes, and three seeds that is 144 distinct circuits and 288 base measurement
groups, beside 13,440 exhaustive parameter draws.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import transpile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qem_bench.baselines.zne import (
    SCALE_FACTORS,
    extrapolate_zero_noise,
    fold_for_execution,
)
from qem_bench.datasets.schema import (
    FAMILY_REQUIRED_FIELDS,
    build_circuit_from_canonical_descriptor,
    canonical_physical_circuit_identity,
)
from qem_bench.datasets.split_generate import _transpile_seed_from_circuit_id
from qem_bench.noise.models import BASIS_GATES
from qem_bench.labels.statevector import ideal_expectation as statevector_expectation
from qem_bench.observables import z_expectation_from_counts, z_support_label
from qem_bench.sampling import sample_counts
from qem_bench.validation import validate_split_artifact

if __package__ in (None, ""):
    from audit_protocol import (
        instruction_stream,
        protocol_gates,
        z_expectation,
    )
else:
    from tools.audit_protocol import (
        instruction_stream,
        protocol_gates,
        z_expectation,
    )

SCHEMA_VERSION = "qem-bench-campaign-audit-v1"

# Which checks are coverage nothing else performs, and which re-derive what
# `validate_split_artifact` already enforces. Recorded beside the counts so a
# reader never has to infer it.
CHECK_COVERAGE = {
    "histogram_replay": "new",
    "structure": "new",
    "circuit_identity": "rederived",
    "circuit_sidecar": "rederived",
    "counts_sidecar": "rederived",
    "observable": "rederived",
    "noisy_estimate": "rederived",
    "exact_label": "rederived",
    "zne_prediction": "new",
    "pool_descriptor": "new",
    "parameter_draw": "new",
    "redrawn_identity": "new",
    "exhaustive_structure": "new",
    "independent_protocol": "independent",
    "independent_label": "independent",
}

# Frozen before any score is inspected. Round 2 of the plan review chose these to
# cover the shared training prefix and its extension in one rule.
AUDIT_INDICES: dict[str, tuple[int, ...]] = {
    "train": (0, 79, 159, 160, 399, 639),
    "validation": (0, 159, 319),
    "test": (0, 79, 159),
}
# Histograms are compared exactly: a replay that differs by one shot is a
# different execution. Recomputed floating-point values carry a declared
# tolerance, because summation order is not part of the contract.
LABEL_TOLERANCE = 1e-12
ESTIMATE_TOLERANCE = 1e-12
# The extrapolation is a weighted sum of three estimates, so its tolerance is
# looser than the estimates that feed it.
PREDICTION_TOLERANCE = 1e-9
# Redrawn couplings are compared against the serialized record, so the
# tolerance is the round trip rather than the arithmetic.
PARAMETER_TOLERANCE = 1e-12
# Two implementations accumulate floating-point error differently, so the
# independent label carries a looser tolerance than the re-derived one.
INDEPENDENT_LABEL_TOLERANCE = 1e-11
_TRANSPILE_SEED_MOD = 2**31 - 1


def audit_dataset(
    data_dir: Path,
    *,
    indices: dict[str, tuple[int, ...]] | None = None,
    roster: Mapping[str, object] | None = None,
) -> dict:
    """Re-derive the sampled circuits of one split-v2 artifact.

    Returns a report naming every check that ran and every mismatch found. A
    mismatch is recorded rather than raised, so one bad circuit does not hide the
    rest of the sample.

    ``roster`` is a run artifact scored from this dataset. When it is supplied,
    the audited test groups have their folded executions replayed at the declared
    scales and the extrapolated prediction compared against the one the roster
    recorded; without it, no zero-noise result is checked.
    """

    indices = indices or AUDIT_INDICES
    root = Path(data_dir)
    items, manifest = validate_split_artifact(root)

    rows_by_circuit: dict[str, list[dict]] = defaultdict(list)
    selected: dict[str, dict] = {}
    for item in items:
        role = str(item["split"])
        if int(item["instance"]) not in indices.get(role, ()):
            continue
        circuit = str(item["circuit_id"])
        rows_by_circuit[circuit].append(item)
        selected.setdefault(circuit, item)

    report = {
        "schema_version": SCHEMA_VERSION,
        "data_path": str(root.resolve()),
        "dataset_hash": str(manifest["dataset_hash"]),
        "split_spec_hash": str(manifest["split_spec_hash"]),
        "master_seed": int(manifest["master_seed"]),
        "audited_indices": {role: list(values) for role, values in indices.items()},
        "scope": (
            "exhaustive parameter draws and compiled structure over every pool "
            "instance; histogram and zero-noise replay over the frozen sample; "
            "the remaining checks re-derive what validate_split_artifact "
            "enforces; the noise model and sampler stay shared, so a fault "            "persisting through both generation and replay can pass; "
            "with an independent protocol and label for the sample; the folded "
            "lands on the prediction because no folded histogram is stored"
        ),
        "check_coverage": dict(CHECK_COVERAGE),
        "label_tolerance": LABEL_TOLERANCE,
        "estimate_tolerance": ESTIMATE_TOLERANCE,
        "circuits": len(selected),
        "rows": sum(len(rows) for rows in rows_by_circuit.values()),
        "measurement_groups": 0,
        "checks": defaultdict(int),
        "mismatches": [],
    }

    # The exhaustive pass runs over every pool instance, not the sample.
    _audit_parameter_draws(items, manifest, report)

    zne = _zne_targets(roster, manifest) if roster is not None else None
    report["zne_replay"] = {
        "checked": zne is not None,
        "roster_artifact_id": None if zne is None else zne["artifact_id"],
        "scales": list(SCALE_FACTORS[1:]),
        "comparison": (
            "folded executions replayed from recorded inputs and independently "
            "derived scale seeds; the extrapolated prediction is compared, "
            "because the run artifact stores no folded histogram"
        ),
    }
    for circuit_id in sorted(selected):
        rows = rows_by_circuit[circuit_id]
        _audit_circuit(root, circuit_id, rows, report, zne=zne)

    report["checks"] = dict(sorted(report["checks"].items()))
    report["passed"] = not report["mismatches"]
    return report


# The draw order each family's sampler follows, reimplemented here rather than
# imported. A fault that consumes an extra draw leaves every recorded seed intact
# and every recorded parameter self-consistent, so comparing the stored values
# against each other cannot see it; only replaying the stream can.
_DRAW_ORDER: dict[str, tuple[str, ...]] = {
    "tfi": ("j", "h"),
    "heisenberg": ("jx", "jy", "jz"),
}
_UNIFORM_RANGE = (0.2, 1.2)


def _replay_draw(master_seed: int, seed_key: tuple[int, ...], instance: int,
                 family: str, grid: Mapping[str, object]) -> dict:
    """Redraw one pool instance from the master seed and the pool descriptor."""
    spawn_key = (*seed_key, 0, instance)
    circuit_seed = int(
        np.random.SeedSequence(master_seed, spawn_key=spawn_key)
        .generate_state(1, dtype=np.uint32)[0]
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(master_seed, spawn_key=spawn_key))
    drawn = {
        "circuit_seed": circuit_seed,
        "instance": instance,
        "n_qubits": int(rng.choice(list(grid["n_qubits"]))),
        "steps": int(rng.choice(list(grid["steps"]))),
    }
    for name in _DRAW_ORDER[family]:
        drawn[name] = float(rng.uniform(*_UNIFORM_RANGE))
    drawn["dt"] = float(grid["dt"])
    return drawn


def _audit_parameter_draws(
    items: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    report: dict,
) -> None:
    """Replay every pool instance's seeded draw, not only the sampled ones.

    This is the exhaustive half of the audit. It reaches every circuit the
    campaign built, because a generation fault that shifts a random stream leaves
    the recorded seeds and the recorded parameters consistent with each other and
    is invisible to any check that reads them.
    """
    master_seed = int(manifest["master_seed"])
    rows: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for item in items:
        rows[(str(item["circuit_pool_id"]), int(item["instance"]))].append(item)

    for pool in manifest["circuit_pools"]:
        family = str(pool["family"])
        if family not in _DRAW_ORDER:
            continue
        grid = pool["parameter_grid"]
        descriptor = {
            "partition_id": pool["partition_id"],
            "split": pool["split"],
            "domain": pool["domain"],
            "family": family,
            "n_qubits": int(pool["n_qubits"]),
            "parameter_grid": grid,
        }
        pool_id = f"pool-{_content_hash(descriptor)}"
        report["checks"]["pool_descriptor"] += 1
        if pool_id != str(pool["circuit_pool_id"]):
            report["mismatches"].append({
                "circuit_pool_id": pool["circuit_pool_id"],
                "check": "pool_descriptor",
                "detail": {"rebuilt": pool_id},
            })
            continue
        seed_key = _seed_key(descriptor)

        for instance in range(int(pool["n_instances"])):
            associated = rows.get((pool_id, instance), [])
            if not associated:
                continue
            drawn = _replay_draw(master_seed, seed_key, instance, family, grid)
            report["checks"]["parameter_draw"] += 1
            disagreed = {
                name: {"recorded": associated[0][name], "redrawn": value}
                for name, value in drawn.items()
                if not _agrees(associated[0].get(name), value)
            }
            if disagreed:
                report["mismatches"].append({
                    "circuit_pool_id": pool_id,
                    "instance": instance,
                    "check": "parameter_draw",
                    "detail": disagreed,
                })
                continue

            descriptor_source = {
                "family": family,
                "n_qubits": drawn["n_qubits"],
                "circuit_seed": drawn["circuit_seed"],
                **{name: drawn[name]
                   for name in FAMILY_REQUIRED_FIELDS[family]},
            }
            circuit_descriptor, circuit_id = canonical_physical_circuit_identity(
                descriptor_source)
            report["checks"]["redrawn_identity"] += 1
            if circuit_id != str(associated[0]["circuit_id"]):
                report["mismatches"].append({
                    "circuit_pool_id": pool_id,
                    "instance": instance,
                    "check": "redrawn_identity",
                    "detail": {"recorded": associated[0]["circuit_id"],
                               "redrawn": circuit_id},
                })
                continue
            _compare_structure(circuit_descriptor, circuit_id, associated, report)


def _compare_structure(
    circuit_descriptor: Mapping[str, object],
    circuit_id: str,
    associated: Sequence[Mapping[str, object]],
    report: dict,
) -> None:
    """Recompile once and compare against every row the circuit produced.

    Comparing against one row per group would leave the other rows of that group
    unchecked, and they carry the same two columns into the feature matrix.
    """
    compiled = _compile(
        build_circuit_from_canonical_descriptor(circuit_descriptor),
        _transpile_seed_from_circuit_id(circuit_id),
    )
    for row in associated:
        report["checks"]["exhaustive_structure"] += 1
        for field in ("two_qubit_gates", "transpiled_depth"):
            if int(compiled[field]) != int(row[field]):
                report["mismatches"].append({
                    "circuit_id": circuit_id,
                    "item_id": row["item_id"],
                    "check": "exhaustive_structure",
                    "detail": {"field": field, "recorded": row[field],
                               "recomputed": compiled[field]},
                })


def _compile(circuit, transpile_seed: int) -> dict[str, int]:
    """Transpile the way the executed protocol declares, and measure the result."""
    measured = circuit.copy()
    measured.measure_all()
    compiled = transpile(
        measured,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=int(transpile_seed % _TRANSPILE_SEED_MOD),
    )
    return {
        "two_qubit_gates": int(compiled.count_ops().get("cx", 0)),
        "transpiled_depth": int(compiled.depth()),
    }


def _production_stream(circuit) -> list[tuple]:
    """Render the shipped circuit the way the independent sequence renders."""
    stream = []
    for instruction in circuit.data:
        qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        stream.append((
            instruction.operation.name,
            qubits,
            round(float(instruction.operation.params[0]), 12),
        ))
    return stream


def _content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _seed_key(value: object) -> tuple[int, ...]:
    digest = bytes.fromhex(_content_hash(value))
    return tuple(
        int.from_bytes(digest[offset:offset + 4], "big") for offset in range(0, 16, 4))


def _agrees(recorded: object, value: object) -> bool:
    if recorded is None:
        return False
    if isinstance(value, float):
        return _close(float(recorded), value, PARAMETER_TOLERANCE)
    return int(recorded) == int(value)


def _zne_targets(
    roster: Mapping[str, object], manifest: Mapping[str, object],
) -> dict:
    """Index the roster's zero-noise predictions by test item, refusing a mismatch."""
    if str(roster["dataset_hash"]) != str(manifest["dataset_hash"]):
        raise ValueError(
            f"the roster scored dataset {roster['dataset_hash']!r} and this artifact "
            f"is {manifest['dataset_hash']!r}"
        )
    spec = roster["methods"]["zne"]
    identifiers = [str(item["item_id"]) for item in roster["test_items"]]
    predictions = [float(value) for value in spec["predictions"]]
    if len(identifiers) != len(predictions):
        raise ValueError("the roster's ZNE predictions do not match its test items")
    extrapolator = str(spec["config"]["extrapolator"])
    return {
        "artifact_id": str(roster["artifact_id"]),
        "extrapolator": extrapolator,
        "master_seed": int(manifest["master_seed"]),
        "predictions": dict(zip(identifiers, predictions, strict=True)),
    }


def _scale_seeds(master_seed: int, measurement_group: str, scale_factor: int):
    """Re-derive one folded execution's seeds from the declared rule.

    Deriving them here rather than calling the shipped helper is the point: the
    rule is part of the protocol the audit checks, so reading it out of the code
    under audit would check nothing.
    """
    root = np.random.SeedSequence(int(master_seed), spawn_key=(2,))
    digest = hashlib.sha256(measurement_group.encode("utf-8")).digest()
    group_key = tuple(
        int.from_bytes(digest[index:index + 4], "little") for index in range(0, 16, 4))
    child = np.random.SeedSequence(
        root.entropy,
        spawn_key=(*root.spawn_key, *group_key, int(scale_factor)),
        pool_size=root.pool_size,
    )
    state = child.generate_state(2, dtype=np.uint32)
    return int(state[0]), int(state[1])


def _audit_circuit(
    root: Path, circuit_id: str, rows: list[dict], report: dict,
    *, zne: Mapping[str, object] | None = None,
) -> None:
    """Rebuild one physical circuit and re-derive everything recorded about it."""

    def fail(check: str, detail: object) -> None:
        report["mismatches"].append(
            {"circuit_id": circuit_id, "check": check, "detail": detail})

    first = rows[0]
    family = str(first["family"])
    # The descriptor is rebuilt from the row rather than read from the sidecar,
    # so a sidecar that disagrees with its own row is detectable.
    source = {
        "family": family,
        "n_qubits": int(first["n_qubits"]),
        "circuit_seed": int(first["circuit_seed"]),
    }
    for field in FAMILY_REQUIRED_FIELDS[family]:
        source[field] = first[field]
    descriptor, rebuilt_id = canonical_physical_circuit_identity(source)
    report["checks"]["circuit_identity"] += 1
    if rebuilt_id != circuit_id:
        fail("circuit_identity", {"recorded": circuit_id, "rebuilt": rebuilt_id})
        return

    payload = (json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")
    stored = (root / str(first["circuit_sidecar"])).read_bytes()
    report["checks"]["circuit_sidecar"] += 1
    if hashlib.sha256(stored).hexdigest() != str(first["circuit_hash"]):
        fail("circuit_sidecar", "sidecar content does not match circuit_hash")
    elif json.loads(stored) != json.loads(payload):
        fail("circuit_sidecar", "sidecar descriptor differs from the rebuilt one")

    circuit = build_circuit_from_canonical_descriptor(descriptor)
    transpile_seed = _transpile_seed_from_circuit_id(circuit_id)

    # Written from the protocol specification rather than built by the shipped
    # builder, so the two disagree if the builder itself drifted.
    gates = protocol_gates(family, source)
    report["checks"]["independent_protocol"] += 1
    produced = _production_stream(circuit)
    if instruction_stream(gates) != produced:
        fail("independent_protocol", {"independent": instruction_stream(gates)[:4],
                                      "production": produced[:4]})
        return

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row["measurement_group"])].append(row)

    for group in sorted(by_group):
        members = by_group[group]
        head = members[0]
        counts, structure = sample_counts(
            circuit,
            severity=str(head["severity"]),
            shots=int(head["shots"]),
            sampler_seed=int(head["sampler_seed"]),
            transpile_seed=transpile_seed,
            noise_family=str(head["noise_family"]),
        )
        report["measurement_groups"] += 1

        stored_counts = (root / str(head["counts_sidecar"])).read_bytes()
        recorded = json.loads(stored_counts)
        report["checks"]["counts_sidecar"] += 1
        if hashlib.sha256(stored_counts).hexdigest() != str(head["counts_hash"]):
            fail("counts_sidecar",
                 {"group": group, "detail": "content does not match counts_hash"})
        replayed = {str(key): int(value) for key, value in sorted(counts.items())}
        report["checks"]["histogram_replay"] += 1
        if replayed != {str(key): int(value)
                        for key, value in sorted(recorded["counts"].items())}:
            fail("histogram_replay", {"group": group, "shots": head["shots"]})
            continue

        report["checks"]["structure"] += 1
        for field, key in (("two_qubit_gates", "two_qubit_gates"),
                           ("transpiled_depth", "transpiled_depth")):
            if int(structure[key]) != int(head[field]):
                fail("structure", {"group": group, "field": field,
                                   "recorded": head[field],
                                   "recomputed": structure[key]})

        for row in members:
            support = _support(str(row["observable"]), int(row["n_qubits"]))
            pauli = z_support_label(int(row["n_qubits"]), support)
            report["checks"]["observable"] += 1
            if pauli != str(row["pauli_label"]) or len(support) != int(
                    row["obs_locality"]):
                fail("observable", {"item_id": row["item_id"],
                                    "recorded": row["pauli_label"],
                                    "recomputed": pauli})
                continue

            estimate, stderr = z_expectation_from_counts(
                replayed, support, int(row["shots"]))
            report["checks"]["noisy_estimate"] += 1
            for name, value, recorded_value in (
                ("noisy_expectation", estimate, row["noisy_expectation"]),
                ("noisy_stderr", stderr, row["noisy_stderr"]),
            ):
                if not _close(value, recorded_value, ESTIMATE_TOLERANCE):
                    fail("noisy_estimate", {"item_id": row["item_id"],
                                            "field": name,
                                            "recorded": recorded_value,
                                            "recomputed": value})

            report["checks"]["exact_label"] += 1
            ideal = float(statevector_expectation(circuit, pauli))
            if not _close(ideal, row["ideal_expectation"], LABEL_TOLERANCE):
                fail("exact_label", {"item_id": row["item_id"],
                                     "recorded": row["ideal_expectation"],
                                     "recomputed": ideal})

            # The same number again, from a protocol written out separately and
            # evolved with plain arrays. A fault inside the shared builder or the
            # shared label routine is invisible to the check above and not to
            # this one.
            report["checks"]["independent_label"] += 1
            independent = z_expectation(gates, int(row["n_qubits"]), support)
            if not _close(independent, row["ideal_expectation"],
                          INDEPENDENT_LABEL_TOLERANCE):
                fail("independent_label", {"item_id": row["item_id"],
                                           "recorded": row["ideal_expectation"],
                                           "independent": independent})

        if zne is not None and str(head["split"]) == "test":
            _audit_zne_group(circuit, group, members, report, zne=zne, fail=fail)


def _audit_zne_group(
    circuit,
    group: str,
    members: list[dict],
    report: dict,
    *,
    zne: Mapping[str, object],
    fail,
) -> None:
    """Replay one test group's folded executions and compare the prediction.

    The run artifact keeps no folded histogram, so the comparison lands on the
    extrapolated value. Everything before it is re-derived: the scale seeds from
    the declared rule, the folded circuit from the rebuilt logical one, and both
    scaled expectations from the replayed counts.
    """
    head = members[0]
    shots = int(head["shots"])
    scaled = []
    for scale_factor in SCALE_FACTORS[1:]:
        sampler_seed, transpile_seed = _scale_seeds(
            int(zne["master_seed"]), group, scale_factor)
        counts, _ = sample_counts(
            fold_for_execution(circuit, scale_factor, transpile_seed),
            severity=str(head["severity"]),
            shots=shots,
            sampler_seed=sampler_seed,
            transpile_seed=transpile_seed,
            noise_family=str(head["noise_family"]),
        )
        scaled.append(counts)
        report["zne_replay_executions"] = report.get("zne_replay_executions", 0) + 1

    for row in members:
        recorded = zne["predictions"].get(str(row["item_id"]))
        if recorded is None:
            fail("zne_prediction", {"item_id": row["item_id"],
                                    "detail": "the roster scored no such test item"})
            continue
        support = _support(str(row["observable"]), int(row["n_qubits"]))
        values = [float(row["noisy_expectation"])]
        for counts in scaled:
            estimate, _ = z_expectation_from_counts(counts, support, shots)
            values.append(estimate)
        replayed = extrapolate_zero_noise(values, str(zne["extrapolator"]))
        report["checks"]["zne_prediction"] += 1
        if not _close(replayed, recorded, PREDICTION_TOLERANCE):
            fail("zne_prediction", {"item_id": row["item_id"],
                                    "recorded": recorded,
                                    "recomputed": replayed})


def _support(observable: str, n_qubits: int) -> tuple[int, ...]:
    """Derive the observable's support from its name and the register width.

    Deriving it here rather than calling the generator's helper keeps the check
    independent of the code that wrote the row.
    """
    mid = n_qubits // 2
    if observable == "z_mid":
        return (mid,)
    if observable == "zz_mid":
        return (mid - 1, mid)
    raise ValueError(f"the campaign audits z_mid and zz_mid, not {observable!r}")


def _close(value: float, recorded: object, tolerance: float) -> bool:
    number = float(recorded)
    return math.isfinite(value) and math.isfinite(number) and abs(
        value - number) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="campaign root holding data/<setting>/ artifacts")
    parser.add_argument("--out", type=Path,
                        help="where to write the audit report (default <root>/audit.json)")
    parser.add_argument(
        "--skip-zne", action="store_true",
        help="do not replay folded executions even where a roster is present")
    args = parser.parse_args()
    directory = args.root / "data"
    if not directory.exists():
        raise SystemExit(f"{directory} does not exist")
    start = time.perf_counter()
    reports = {}
    for setting in sorted(path for path in directory.iterdir() if path.is_dir()):
        roster = None
        results = args.root / "rosters" / setting.name / "results.json"
        if results.is_file() and not args.skip_zne:
            roster = json.loads(results.read_text(encoding="utf-8"))
        result = audit_dataset(setting, roster=roster)
        reports[setting.name] = result
        print(f"{setting.name}: {result['circuits']} circuits, "
              f"{result['measurement_groups']} groups, "
              f"{result['checks'].get('zne_prediction', 0)} zne predictions, "
              f"{len(result['mismatches'])} mismatches, "
              f"{time.perf_counter() - start:.1f}s", flush=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "settings": reports,
        "passed": all(value["passed"] for value in reports.values()),
        "seconds": time.perf_counter() - start,
    }
    out = args.out or args.root / "audit.json"
    out.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"settings": len(reports), "passed": summary["passed"]}),
          flush=True)


if __name__ == "__main__":
    main()
