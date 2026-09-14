#!/usr/bin/env python3
"""Measure target-derived CDR training-set viability on shipped circuit grids.

The probe compiles each target into the repository execution basis, replaces
non-Clifford RZ gates on that compiled circuit, and compiles each resulting
training circuit once more as the executor would. It reports construction-space
sizes, source and executed circuit hashes, ideal-label spread, and the rank and
condition number of the affine CDR design matrix ``[1, noisy_expectation]``.

The finite-shot design values are exact marginal binomial draws from the noisy
expectations. This has the same one-observable sampling law as computational-
basis measurement while avoiding a third transpiler pass. Exact-noise design
metrics are also reported so that shot noise cannot hide structural collapse.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from qemscore.circuits.heisenberg import (
    build_heisenberg_circuit,
    sample_heisenberg_params,
)
from qemscore.circuits.qaoa import build_qaoa_circuit, sample_qaoa_params
from qemscore.circuits.tfi import build_tfi_circuit, sample_tfi_params
from qemscore.noise.models import BASIS_GATES, SEVERITY_GRIDS, build_noise_model
from qemscore.observables import z_support_label


SCHEMA_VERSION = "cdr-viability-probe-v1"
DEFAULT_SEED = 20260819
DEFAULT_FRACTION_NON_CLIFFORD = 0.1
CLIFFORD_ANGLES = np.asarray((0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0))
CLIFFORD_TOLERANCE = 1.0e-8
QISKIT_SEED_MODULUS = 2**31 - 1
PARAMETER_COUNT = 2
REVIEWED_PAYLOAD_SHA256 = (
    "043839b9eec041364a38d4b7f78fc8d7acaa8fe07a3576c871f19cd5a512ba38"
)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    family: str
    preset: str
    n_qubits: int
    depth: int
    shots: int
    graph_class: str | None = None


def _stable_seed(master: int, *parts: object) -> int:
    payload = "\0".join((str(master), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _qiskit_seed(master: int, *parts: object) -> int:
    return _stable_seed(master, *parts) % QISKIT_SEED_MODULUS


def _target_specs(families: set[str]) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    if "tfi" in families:
        for n_qubits in (3,):
            for steps in (1, 2):
                specs.append(
                    TargetSpec(
                        f"tfi-n{n_qubits}-steps{steps}",
                        "tfi",
                        "t0-micro",
                        n_qubits,
                        steps,
                        512,
                    )
                )
        for n_qubits in (4, 5, 6):
            for steps in (1, 2, 3):
                specs.append(
                    TargetSpec(
                        f"tfi-n{n_qubits}-steps{steps}",
                        "tfi",
                        "t0-smoke",
                        n_qubits,
                        steps,
                        2048,
                    )
                )
    if "qaoa" in families:
        for n_qubits in (4, 6):
            for p in (1, 2):
                for graph_class in ("path", "cycle", "erdos_renyi", "3_regular"):
                    specs.append(
                        TargetSpec(
                            f"qaoa-n{n_qubits}-p{p}-{graph_class}",
                            "qaoa",
                            "t0-qaoa-micro",
                            n_qubits,
                            p,
                            512,
                            graph_class,
                        )
                    )
    if "heisenberg" in families:
        for n_qubits in (3, 4):
            for steps in (1, 2):
                specs.append(
                    TargetSpec(
                        f"heisenberg-n{n_qubits}-steps{steps}",
                        "heisenberg",
                        "t0-heisenberg-micro",
                        n_qubits,
                        steps,
                        512,
                    )
                )
    return specs


def _build_target(spec: TargetSpec, master_seed: int, instance: int) -> tuple[Any, QuantumCircuit]:
    circuit_seed = _qiskit_seed(master_seed, "target", spec.target_id, "circuit")
    rng = np.random.default_rng(_stable_seed(master_seed, "target", spec.target_id, "parameters"))
    if spec.family == "tfi":
        params = sample_tfi_params(
            rng,
            [spec.n_qubits],
            [spec.depth],
            0.2,
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_tfi_circuit(params)
    if spec.family == "qaoa":
        assert spec.graph_class is not None
        params = sample_qaoa_params(
            rng,
            [spec.n_qubits],
            [spec.depth],
            [spec.graph_class],
            instance=instance,
            circuit_seed=circuit_seed,
        )
        return params, build_qaoa_circuit(params)
    params = sample_heisenberg_params(
        rng,
        [spec.n_qubits],
        [spec.depth],
        0.15,
        instance=instance,
        circuit_seed=circuit_seed,
    )
    return params, build_heisenberg_circuit(params)


def _pass_manager(seed: int) -> Any:
    return generate_preset_pass_manager(
        optimization_level=1,
        basis_gates=BASIS_GATES,
        seed_transpiler=int(seed),
    )


def _run_pass_manager(manager: Any, circuits: Sequence[QuantumCircuit]) -> list[QuantumCircuit]:
    if not circuits:
        return []
    try:
        result = manager.run(list(circuits), num_processes=1)
    except TypeError:
        result = [manager.run(circuit) for circuit in circuits]
    if isinstance(result, QuantumCircuit):
        return [result]
    return list(result)


def _mod_angle(value: float) -> float:
    result = float(value) % (2.0 * np.pi)
    if math.isclose(result, 2.0 * np.pi, rel_tol=0.0, abs_tol=CLIFFORD_TOLERANCE):
        return 0.0
    if math.isclose(result, 0.0, rel_tol=0.0, abs_tol=CLIFFORD_TOLERANCE):
        return 0.0
    return result


def _angle_distance(left: float, right: float) -> float:
    return abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def _is_clifford_angle(angle: float) -> bool:
    normalized = _mod_angle(angle)
    return min(_angle_distance(normalized, candidate) for candidate in CLIFFORD_ANGLES) <= CLIFFORD_TOLERANCE


def _closest_clifford(angle: float) -> tuple[float, bool]:
    normalized = _mod_angle(angle)
    distances = np.asarray([_angle_distance(normalized, candidate) for candidate in CLIFFORD_ANGLES])
    minimum = float(np.min(distances))
    indices = np.flatnonzero(np.isclose(distances, minimum, rtol=0.0, atol=CLIFFORD_TOLERANCE))
    return float(CLIFFORD_ANGLES[int(indices[0])]), len(indices) > 1


def _float_token(value: object, *, angle: bool = False) -> str:
    number = float(value)
    if angle:
        number = _mod_angle(number)
    if number == 0.0:
        number = 0.0
    return number.hex()


def _circuit_payload(circuit: QuantumCircuit) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for instruction in circuit.data:
        operation = instruction.operation
        operations.append(
            {
                "name": operation.name,
                "qubits": [circuit.find_bit(qubit).index for qubit in instruction.qubits],
                "params": [
                    _float_token(param, angle=operation.name == "rz")
                    for param in operation.params
                ],
            }
        )
    return {
        "n_qubits": circuit.num_qubits,
        "global_phase": _float_token(circuit.global_phase, angle=True),
        "operations": operations,
    }


def _circuit_hash(circuit: QuantumCircuit) -> str:
    encoded = json.dumps(
        _circuit_payload(circuit), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_sequence(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _non_clifford_rz(circuit: QuantumCircuit) -> list[tuple[int, float]]:
    unexpected = sorted({instruction.operation.name for instruction in circuit.data} - set(BASIS_GATES))
    if unexpected:
        raise RuntimeError(f"compiled circuit has gates outside the execution basis: {unexpected}")
    result: list[tuple[int, float]] = []
    for index, instruction in enumerate(circuit.data):
        if instruction.operation.name != "rz":
            continue
        angle = float(instruction.operation.params[0])
        if not _is_clifford_angle(angle):
            result.append((index, angle))
    return result


def _training_variant(
    source: QuantumCircuit,
    non_clifford: Sequence[tuple[int, float]],
    retained_count: int,
    strategy: str,
    rng: np.random.Generator,
) -> QuantumCircuit:
    retained_ordinals = set(
        int(value)
        for value in rng.choice(len(non_clifford), size=retained_count, replace=False)
    )
    replacements: dict[int, float] = {}
    for ordinal, (instruction_index, angle) in enumerate(non_clifford):
        if ordinal in retained_ordinals:
            continue
        if strategy == "closest":
            replacement, _ = _closest_clifford(angle)
        elif strategy == "uniform":
            replacement = float(CLIFFORD_ANGLES[int(rng.integers(0, len(CLIFFORD_ANGLES)))])
        else:
            raise ValueError(f"unknown replacement strategy {strategy!r}")
        replacements[instruction_index] = replacement

    variant = QuantumCircuit(source.num_qubits)
    variant.global_phase = source.global_phase
    for index, instruction in enumerate(source.data):
        qubits = [source.find_bit(qubit).index for qubit in instruction.qubits]
        operation = instruction.operation
        if index in replacements:
            operation = RZGate(replacements[index])
        variant.append(operation, qubits)
    return variant


def _observable_specs(n_qubits: int) -> list[dict[str, Any]]:
    middle = n_qubits // 2
    definitions = (("z_mid", (middle,)), ("zz_mid", (middle - 1, middle)))
    return [
        {
            "name": name,
            "support": list(support),
            "pauli_label": z_support_label(n_qubits, support),
        }
        for name, support in definitions
    ]


def _evaluate_circuits(
    circuits: Sequence[QuantumCircuit],
    observables: Sequence[dict[str, Any]],
    backend: AerSimulator,
    readout_probability: float,
    batch_size: int,
) -> list[dict[str, dict[str, float]]]:
    evaluations: list[dict[str, dict[str, float]]] = []
    for start in range(0, len(circuits), batch_size):
        batch = circuits[start : start + batch_size]
        saved_circuits: list[QuantumCircuit] = []
        ideal_batch: list[dict[str, float]] = []
        for circuit in batch:
            state = Statevector.from_instruction(circuit)
            ideal_values: dict[str, float] = {}
            saved = circuit.copy()
            for observable in observables:
                operator = SparsePauliOp(observable["pauli_label"])
                ideal_values[observable["name"]] = float(state.expectation_value(operator).real)
                saved.save_expectation_value(
                    operator,
                    list(range(circuit.num_qubits)),
                    label=f"expectation_{observable['name']}",
                )
            ideal_batch.append(ideal_values)
            saved_circuits.append(saved)

        result = backend.run(saved_circuits).result()
        for offset, ideal_values in enumerate(ideal_batch):
            data = result.data(offset)
            noisy_values: dict[str, float] = {}
            for observable in observables:
                quantum_value = float(np.real(data[f"expectation_{observable['name']}"]))
                readout_factor = (1.0 - 2.0 * readout_probability) ** len(observable["support"])
                noisy_values[observable["name"]] = quantum_value * readout_factor
            evaluations.append({"ideal": ideal_values, "noisy_exact": noisy_values})
    return evaluations


def _design_stats(values: np.ndarray) -> dict[str, Any]:
    design = np.column_stack((np.ones(values.size, dtype=float), values))
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number: float | None
    if rank < PARAMETER_COUNT:
        condition_number = None
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "parameters": PARAMETER_COUNT,
        "rank": rank,
        "condition_number": condition_number,
        "singular_values": [float(value) for value in singular_values],
        "rank_tolerance": float(tolerance),
    }


def _label_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
        "standard_deviation": float(np.std(values)),
        "unique_at_12_decimals": len({round(float(value), 12) for value in values}),
    }


def _configuration_metrics(
    target_id: str,
    strategy: str,
    retained_count: int,
    shots: int,
    report_sizes: Sequence[int],
    source_hashes: Sequence[str],
    execution_hashes: Sequence[str],
    evaluation_by_hash: dict[str, dict[str, dict[str, float]]],
    observables: Sequence[dict[str, Any]],
    master_seed: int,
) -> dict[str, Any]:
    sampled_values: dict[str, np.ndarray] = {}
    for observable in observables:
        name = observable["name"]
        exact = np.asarray(
            [evaluation_by_hash[circuit_hash]["noisy_exact"][name] for circuit_hash in execution_hashes],
            dtype=float,
        )
        probability = np.clip((1.0 + exact) / 2.0, 0.0, 1.0)
        rng = np.random.default_rng(
            _stable_seed(master_seed, "shots", target_id, strategy, retained_count, name)
        )
        plus_counts = rng.binomial(shots, probability)
        sampled_values[name] = (2.0 * plus_counts - shots) / shots

    by_size: dict[str, Any] = {}
    for sample_size in report_sizes:
        source_prefix = list(source_hashes[:sample_size])
        execution_prefix = list(execution_hashes[:sample_size])
        source_distinct = len(set(source_prefix))
        execution_distinct = len(set(execution_prefix))
        observable_metrics: dict[str, Any] = {}
        for observable in observables:
            name = observable["name"]
            ideal = np.asarray(
                [evaluation_by_hash[circuit_hash]["ideal"][name] for circuit_hash in execution_prefix],
                dtype=float,
            )
            noisy_exact = np.asarray(
                [evaluation_by_hash[circuit_hash]["noisy_exact"][name] for circuit_hash in execution_prefix],
                dtype=float,
            )
            noisy_sampled = sampled_values[name][:sample_size]
            observable_metrics[name] = {
                "ideal_labels": _label_stats(ideal),
                "noisy_exact": _label_stats(noisy_exact),
                "noisy_sampled": _label_stats(noisy_sampled),
                "design_exact": _design_stats(noisy_exact),
                "design_sampled": _design_stats(noisy_sampled),
            }
        by_size[str(sample_size)] = {
            "sample_size": sample_size,
            "source_distinct": source_distinct,
            "source_duplicate_rate": (sample_size - source_distinct) / sample_size,
            "executed_distinct": execution_distinct,
            "executed_duplicate_rate": (sample_size - execution_distinct) / sample_size,
            "source_hash_sequence_sha256": _hash_sequence(source_prefix),
            "executed_hash_sequence_sha256": _hash_sequence(execution_prefix),
            "observables": observable_metrics,
        }
    return by_size


def _construction_space_size(non_clifford_count: int, retained_count: int, strategy: str) -> int:
    selection_count = math.comb(non_clifford_count, retained_count)
    if strategy == "uniform":
        return selection_count * 4 ** (non_clifford_count - retained_count)
    return selection_count


def _readout_probability(noise_family: str, severity: str) -> float:
    config = SEVERITY_GRIDS[noise_family][severity]
    if "p_ro" not in config:
        raise ValueError(
            f"noise family {noise_family!r} has no homogeneous p_ro; the exact marginal "
            "readout correction is undefined for this probe"
        )
    return float(config["p_ro"])


def _probe_target(
    spec: TargetSpec,
    instance: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    params, logical = _build_target(spec, args.seed, instance)
    source_seed = _qiskit_seed(args.seed, "source-compile", spec.target_id)
    execution_seed = _qiskit_seed(args.seed, "execution-compile", spec.target_id)
    source_manager = _pass_manager(source_seed)
    execution_manager = _pass_manager(execution_seed)
    source = _run_pass_manager(source_manager, [logical])[0]
    target_execution = _run_pass_manager(execution_manager, [source])[0]
    non_clifford = _non_clifford_rz(source)
    non_clifford_count = len(non_clifford)
    if non_clifford_count == 0:
        raise RuntimeError(f"{spec.target_id} compiled to a Clifford circuit")

    closest_angles: list[float] = []
    tie_count = 0
    for _, angle in non_clifford:
        replacement, tied = _closest_clifford(angle)
        closest_angles.append(replacement)
        tie_count += int(tied)

    observables = _observable_specs(spec.n_qubits)
    noise_model = build_noise_model(
        args.noise_family,
        args.severity,
        seed=_qiskit_seed(args.seed, "noise", spec.target_id),
        n_qubits=spec.n_qubits,
    )
    backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )
    readout_probability = _readout_probability(args.noise_family, args.severity)

    execution_cache: dict[str, tuple[str, QuantumCircuit]] = {}
    evaluation_by_hash: dict[str, dict[str, dict[str, float]]] = {}
    target_execution_hash = _circuit_hash(target_execution)
    target_evaluation = _evaluate_circuits(
        [target_execution], observables, backend, readout_probability, args.batch_size
    )[0]
    evaluation_by_hash[target_execution_hash] = target_evaluation

    rule_counts: list[tuple[str, int]] = [
        (
            "mitiq-default-fraction-0.1",
            int(round(DEFAULT_FRACTION_NON_CLIFFORD * non_clifford_count)),
        ),
        *((f"fixed-{count}", count) for count in args.fixed_retained_counts),
    ]
    sample_cache: dict[tuple[str, int], dict[str, Any]] = {}
    configurations: list[dict[str, Any]] = []

    for strategy in args.replacement_strategies:
        for rule, retained_count in rule_counts:
            if retained_count > non_clifford_count:
                configurations.append(
                    {
                        "rule": rule,
                        "replacement_strategy": strategy,
                        "status": "infeasible",
                        "reason": f"retained count {retained_count} exceeds G={non_clifford_count}",
                        "retained_count": retained_count,
                    }
                )
                continue

            cache_key = (strategy, retained_count)
            if cache_key not in sample_cache:
                rng = np.random.default_rng(
                    _stable_seed(args.seed, "training", spec.target_id, strategy, retained_count)
                )
                source_circuits = [
                    _training_variant(source, non_clifford, retained_count, strategy, rng)
                    for _ in range(args.samples)
                ]
                source_hashes = [_circuit_hash(circuit) for circuit in source_circuits]

                missing_by_source: dict[str, QuantumCircuit] = {}
                for source_hash, circuit in zip(source_hashes, source_circuits):
                    if source_hash not in execution_cache:
                        missing_by_source.setdefault(source_hash, circuit)
                missing_hashes = list(missing_by_source)
                compiled_missing = _run_pass_manager(
                    execution_manager,
                    [missing_by_source[source_hash] for source_hash in missing_hashes],
                )
                for source_hash, executed in zip(missing_hashes, compiled_missing):
                    execution_cache[source_hash] = (_circuit_hash(executed), executed)

                execution_hashes = [execution_cache[source_hash][0] for source_hash in source_hashes]
                missing_evaluations: dict[str, QuantumCircuit] = {}
                for source_hash in source_hashes:
                    execution_hash, circuit = execution_cache[source_hash]
                    if execution_hash not in evaluation_by_hash:
                        missing_evaluations.setdefault(execution_hash, circuit)
                evaluation_hashes = list(missing_evaluations)
                evaluated = _evaluate_circuits(
                    [missing_evaluations[circuit_hash] for circuit_hash in evaluation_hashes],
                    observables,
                    backend,
                    readout_probability,
                    args.batch_size,
                )
                evaluation_by_hash.update(zip(evaluation_hashes, evaluated))

                shots = args.shots if args.shots is not None else spec.shots
                sample_cache[cache_key] = {
                    "source_hashes": source_hashes,
                    "execution_hashes": execution_hashes,
                    "metrics_by_m": _configuration_metrics(
                        spec.target_id,
                        strategy,
                        retained_count,
                        shots,
                        args.report_sizes,
                        source_hashes,
                        execution_hashes,
                        evaluation_by_hash,
                        observables,
                        args.seed,
                    ),
                }

            sampled = sample_cache[cache_key]
            space_size = _construction_space_size(non_clifford_count, retained_count, strategy)
            configurations.append(
                {
                    "rule": rule,
                    "replacement_strategy": strategy,
                    "status": "ok",
                    "retained_count": retained_count,
                    "retained_fraction": retained_count / non_clifford_count,
                    "construction_space_size_exact": str(space_size),
                    "construction_space_log10": math.log10(space_size),
                    "sample_source_hashes": sampled["source_hashes"],
                    "sample_execution_hashes": sampled["execution_hashes"],
                    "metrics_by_m": sampled["metrics_by_m"],
                }
            )

    parameters = params.to_dict()
    nearest_histogram: dict[str, int] = {}
    for angle in closest_angles:
        key = f"{angle / np.pi:.12g}*pi"
        nearest_histogram[key] = nearest_histogram.get(key, 0) + 1
    return {
        "target_id": spec.target_id,
        "family": spec.family,
        "preset": spec.preset,
        "n_qubits": spec.n_qubits,
        "depth_parameter": "p" if spec.family == "qaoa" else "steps",
        "depth": spec.depth,
        "graph_class": spec.graph_class,
        "shots": args.shots if args.shots is not None else spec.shots,
        "parameters": parameters,
        "observables": observables,
        "source_transpiler_seed": source_seed,
        "execution_transpiler_seed": execution_seed,
        "source_compiled_hash": _circuit_hash(source),
        "execution_compiled_target_hash": target_execution_hash,
        "source_compiled_depth": int(source.depth()),
        "source_compiled_operation_counts": {
            name: int(count) for name, count in sorted(source.count_ops().items())
        },
        "rz_count_total": int(source.count_ops().get("rz", 0)),
        "non_clifford_rz_count_G": non_clifford_count,
        "non_clifford_rz_angles": [float(angle) for _, angle in non_clifford],
        "closest_replacement_angles": closest_angles,
        "closest_replacement_histogram": nearest_histogram,
        "closest_half_angle_tie_count": tie_count,
        "target_expectations": target_evaluation,
        "configurations": configurations,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _artifact(args: argparse.Namespace) -> dict[str, Any]:
    families = set(args.families)
    specs = _target_specs(families)
    if args.target_limit is not None:
        specs = specs[: args.target_limit]
    targets: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        print(f"[{index + 1}/{len(specs)}] {spec.target_id}", flush=True)
        targets.append(_probe_target(spec, index, args))

    target_hashes = [target["source_compiled_hash"] for target in targets]
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "target_panel": {
            "description": (
                "One seeded target for every shipped TFI width/step pair, QAOA "
                "width/p/graph-class tuple, and Heisenberg width/step pair selected "
                "by --families."
            ),
            "target_count": len(targets),
            "source_compiled_target_hash_sequence_sha256": _hash_sequence(target_hashes),
        },
        "construction_contract": {
            "source_compilation": {
                "basis_gates": BASIS_GATES,
                "optimization_level": 1,
            },
            "non_clifford_definition": (
                "RZ angle farther than 1e-8 radians from a multiple of pi/2 modulo 2*pi"
            ),
            "selection": "uniform retained-position subset of exact size r",
            "mitiq_default_semantics": "r = round(0.1 * G), matching Mitiq 1.0.0",
            "replacement_strategies": list(args.replacement_strategies),
            "closest_tie_rule": (
                "smallest member of [0, pi/2, pi, 3*pi/2]; observed tie counts are per target"
            ),
            "execution_compilation": {
                "basis_gates": BASIS_GATES,
                "optimization_level": 1,
            },
            "hash_contract": (
                "SHA-256 of ordered instructions, qubits, parameters, and global phase; "
                "RZ parameters and global phase are normalized modulo 2*pi"
            ),
        },
        "measurement_contract": {
            "noise_family": args.noise_family,
            "severity": args.severity,
            "shots": "shipped preset value" if args.shots is None else args.shots,
            "report_sizes_m": list(args.report_sizes),
            "samples_generated_per_configuration": args.samples,
            "design_matrix": "[ones, noisy_expectation]",
            "design_parameters": PARAMETER_COUNT,
            "exact_noise": (
                "Aer density-matrix expectation with gate noise, multiplied by the exact "
                "independent symmetric-readout factor (1 - 2*p_ro)^locality"
            ),
            "finite_shot": (
                "seeded binomial marginal draw from each exact noisy Pauli expectation"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
            "mitiq": _package_version("mitiq"),
        },
        "targets": targets,
    }


def _format_float(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _format_integer_range(values: Sequence[int]) -> str:
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low} to {high}"


def _format_space_range(values: Sequence[int]) -> str:
    low, high = min(values), max(values)
    if low == high:
        return str(low) if low < 1_000_000 else f"{low:.3e}"
    low_text = str(low) if low < 1_000_000 else f"{low:.3e}"
    high_text = str(high) if high < 1_000_000 else f"{high:.3e}"
    return f"{low_text} to {high_text}"


def _summary(artifact: dict[str, Any]) -> str:
    lines = [
        "# CDR viability probe summary",
        "",
        f"Payload SHA-256: `{artifact['payload_sha256']}`.",
        f"Seed: `{artifact['seed']}`. Targets: {artifact['target_panel']['target_count']}. "
        f"Noise: `{artifact['measurement_contract']['noise_family']}` "
        f"`{artifact['measurement_contract']['severity']}`.",
        "",
        "The exact design is the two-parameter affine matrix `[1, noisy expectation]`. "
        "The sampled design uses the shipped preset shot count. Circuit hashes are taken "
        "both before and after the execution compilation; the tables below use executed hashes.",
        "",
        "## Compiled targets",
        "",
        "| Target | Preset | n | Depth | Graph | G | Total RZ | Source hash |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for target in artifact["targets"]:
        graph = target["graph_class"] or ""
        lines.append(
            f"| `{target['target_id']}` | `{target['preset']}` | {target['n_qubits']} | "
            f"{target['depth']} | {graph} | {target['non_clifford_rz_count_G']} | "
            f"{target['rz_count_total']} | `{target['source_compiled_hash']}` |"
        )

    report_sizes = artifact["measurement_contract"]["report_sizes_m"]
    primary_m = 70 if 70 in report_sizes else max(report_sizes)
    rule_order: list[tuple[str, str]] = []
    if artifact["targets"]:
        for config in artifact["targets"][0]["configurations"]:
            rule_order.append((config["replacement_strategy"], config["rule"]))

    lines.extend(
        [
            "",
            f"## Training-set metrics at m = {primary_m}",
            "",
            "Each row is one family and construction configuration. Ranges are across that "
            "family's target configurations. Rank counts use two observables per feasible target. "
            "A condition number of `inf` means that at least one exact design is rank deficient.",
        ]
    )
    family_order = ("tfi", "qaoa", "heisenberg")
    for family in family_order:
        family_targets = [target for target in artifact["targets"] if target["family"] == family]
        if not family_targets:
            continue
        lines.extend(
            [
                "",
                f"### {family}",
                "",
                "| Replacement | Retained rule | r | Exact construction space | "
                "Executed distinct | Duplicate rate | Ideal-label range | Exact rank 2/rows | "
                "Sampled rank 2/rows | Worst exact condition |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy, rule in rule_order:
            configs = []
            for target in family_targets:
                match = next(
                    config
                    for config in target["configurations"]
                    if config["replacement_strategy"] == strategy and config["rule"] == rule
                )
                if match["status"] == "ok":
                    configs.append(match)
            if not configs:
                continue
            retained = [int(config["retained_count"]) for config in configs]
            spaces = [int(config["construction_space_size_exact"]) for config in configs]
            metrics = [config["metrics_by_m"][str(primary_m)] for config in configs]
            distinct = [int(metric["executed_distinct"]) for metric in metrics]
            duplicate_rates = [float(metric["executed_duplicate_rate"]) for metric in metrics]
            observable_rows = [
                observable
                for metric in metrics
                for observable in metric["observables"].values()
            ]
            ideal_ranges = [float(row["ideal_labels"]["range"]) for row in observable_rows]
            exact_full = sum(row["design_exact"]["rank"] == PARAMETER_COUNT for row in observable_rows)
            sampled_full = sum(
                row["design_sampled"]["rank"] == PARAMETER_COUNT for row in observable_rows
            )
            exact_conditions = [row["design_exact"]["condition_number"] for row in observable_rows]
            worst_condition = (
                "inf"
                if any(value is None for value in exact_conditions)
                else _format_float(max(float(value) for value in exact_conditions if value is not None))
            )
            distinct_text = _format_integer_range(distinct)
            duplicate_text = (
                f"{100.0 * min(duplicate_rates):.1f}%"
                if min(duplicate_rates) == max(duplicate_rates)
                else f"{100.0 * min(duplicate_rates):.1f}% to {100.0 * max(duplicate_rates):.1f}%"
            )
            ideal_text = (
                f"{_format_float(min(ideal_ranges))} to {_format_float(max(ideal_ranges))}"
            )
            lines.append(
                f"| {strategy} | `{rule}` | {_format_integer_range(retained)} | "
                f"{_format_space_range(spaces)} | {distinct_text} | {duplicate_text} | "
                f"{ideal_text} | {exact_full}/{len(observable_rows)} | "
                f"{sampled_full}/{len(observable_rows)} | {worst_condition} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- `construction_space_size_exact` counts retained-position choices and, for "
            "uniform replacement, the four Clifford-angle choices at every replaced position.",
            "",
            "- Distinct and duplicate metrics use post-substitution, post-execution-compilation "
            "hashes. The JSON also retains every ordered source and executed hash.",
            "",
            "- Exact rank is the structural check. Sampled rank is reported separately because "
            "independent shot noise can give repeated circuits different noisy estimates.",
            "",
            "- Full per-target, per-observable metrics for every reported m are in the JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def payload_sha256(artifact: dict[str, Any]) -> str:
    """Return the canonical digest of the artifact payload without its digest field."""

    payload = dict(artifact)
    payload.pop("payload_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_cdr_artifacts(
    artifact_path: Path = ROOT / "tools" / "cdr_viability_probe.json",
    summary_path: Path = ROOT / "tools" / "cdr_viability_probe.summary.md",
    *,
    expected_payload_sha256: str = REVIEWED_PAYLOAD_SHA256,
) -> str:
    """Verify the reviewed payload digest and its deterministically derived summary."""

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("CDR artifact must decode to an object")
    recorded = artifact.get("payload_sha256")
    if not isinstance(recorded, str):
        raise ValueError("CDR artifact payload_sha256 is missing or is not a string")
    regenerated = payload_sha256(artifact)
    if recorded != regenerated:
        raise ValueError(
            "CDR artifact payload SHA-256 mismatch: "
            f"recorded {recorded}, regenerated {regenerated}"
        )
    if regenerated != expected_payload_sha256:
        raise ValueError(
            "CDR artifact reviewed payload drift: "
            f"expected {expected_payload_sha256}, regenerated {regenerated}"
        )
    recorded_summary = summary_path.read_text(encoding="utf-8")
    regenerated_summary = _summary(artifact)
    if recorded_summary != regenerated_summary:
        raise ValueError(
            "CDR artifact summary drift: shipped summary does not match the payload"
        )
    return regenerated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the shipped reviewed payload and summary without regenerating targets",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--report-sizes", type=int, nargs="+", default=[10, 70, 100])
    parser.add_argument(
        "--fixed-retained-counts", type=int, nargs="+", default=[0, 1, 2, 3, 5]
    )
    parser.add_argument(
        "--replacement-strategies",
        nargs="+",
        choices=("closest", "uniform"),
        default=["closest", "uniform"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("tfi", "qaoa", "heisenberg"),
        default=["tfi", "qaoa", "heisenberg"],
    )
    parser.add_argument(
        "--noise-family",
        choices=tuple(family for family, grid in SEVERITY_GRIDS.items() if "p_ro" in grid["L1"]),
        default="depolarizing_readout",
    )
    parser.add_argument("--severity", choices=("L1", "L2", "L3", "L4"), default="L2")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "tools" / "cdr_viability_probe.json"
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=ROOT / "tools" / "cdr_viability_probe.summary.md",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if not args.report_sizes or any(size < 2 or size > args.samples for size in args.report_sizes):
        parser.error("every --report-sizes value must be between 2 and --samples")
    args.report_sizes = sorted(set(args.report_sizes))
    if any(count < 0 for count in args.fixed_retained_counts):
        parser.error("--fixed-retained-counts values must be nonnegative")
    args.fixed_retained_counts = list(dict.fromkeys(args.fixed_retained_counts))
    if args.shots is not None and args.shots < 1:
        parser.error("--shots must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.target_limit is not None and args.target_limit < 1:
        parser.error("--target-limit must be positive")
    if args.output_json.resolve() == args.output_summary.resolve():
        parser.error("JSON and summary output paths must differ")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        try:
            digest = verify_cdr_artifacts(args.output_json, args.output_summary)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"CDR artifact integrity check failed: {exc}") from exc
        print(f"expected payload SHA-256:    {REVIEWED_PAYLOAD_SHA256}")
        print(f"regenerated payload SHA-256: {digest}")
        print("summary: regenerated content matches the shipped summary")
        return 0
    artifact = _artifact(args)
    artifact["payload_sha256"] = payload_sha256(artifact)
    summary = _summary(artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    args.output_summary.write_text(summary, encoding="utf-8")
    print(f"payload_sha256={artifact['payload_sha256']}")
    for output_path in (args.output_json, args.output_summary):
        try:
            display_path = output_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            display_path = output_path.name
        print(f"wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
