"""Seeded near-Clifford circuits for dense statevector labels.

A near-Clifford circuit starts from the random Clifford circuit with the same
qubit count, depth, seed, and instance. A separate named seed stream inserts a
small configured number of T or RZ(theta) gates into that instruction sequence.
The angle is required to be non-Clifford. Circuits carry no measurements.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

from qemscore.circuits.random_clifford import (
    RandomCliffordParams,
    build_random_clifford_circuit,
)

MAX_NEAR_CLIFFORD_QUBITS = 14
NON_CLIFFORD_GATES: tuple[str, ...] = ("t", "rz")


@dataclass(frozen=True)
class NearCliffordParams:
    n_qubits: int
    depth: int
    non_clifford_count: int
    theta: float
    circuit_seed: int
    instance: int

    def __post_init__(self) -> None:
        if type(self.n_qubits) is not int:
            raise ValueError("near-Clifford n_qubits must be an integer")
        if type(self.depth) is not int:
            raise ValueError("near-Clifford depth must be an integer")
        if type(self.non_clifford_count) is not int:
            raise ValueError("near-Clifford insertion count must be an integer")
        if type(self.circuit_seed) is not int:
            raise ValueError("circuit_seed must be an integer")
        if type(self.instance) is not int:
            raise ValueError("instance must be an integer")
        if not 1 <= self.n_qubits <= MAX_NEAR_CLIFFORD_QUBITS:
            raise ValueError(
                f"near-Clifford n_qubits must be in "
                f"[1, {MAX_NEAR_CLIFFORD_QUBITS}]"
            )
        if self.depth < 1:
            raise ValueError("near-Clifford depth must be positive")
        if self.non_clifford_count < 1:
            raise ValueError("near-Clifford circuits require at least one insertion")
        if self.non_clifford_count > self.n_qubits:
            raise ValueError("near-Clifford insertion count cannot exceed n_qubits")
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        quarter_turns = self.theta / (math.pi / 2.0)
        if math.isclose(
            quarter_turns, round(quarter_turns), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("theta must not be a Clifford RZ angle")
        if self.circuit_seed < 0:
            raise ValueError("circuit_seed must be nonnegative")
        if self.instance < 0:
            raise ValueError("instance must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_near_clifford_sampling_domain(
    n_qubits_choices: list[int],
    depth_choices: list[int],
    non_clifford_count_choices: list[int],
    theta_choices: list[float],
) -> None:
    """Validate the authored choices consumed before near-Clifford draws."""
    if not n_qubits_choices:
        raise ValueError("n_qubits_choices must not be empty")
    if not depth_choices:
        raise ValueError("depth_choices must not be empty")
    if not non_clifford_count_choices:
        raise ValueError("non_clifford_count_choices must not be empty")
    if not theta_choices:
        raise ValueError("theta_choices must not be empty")
    if any(
        type(n) is not int or not 1 <= n <= MAX_NEAR_CLIFFORD_QUBITS
        for n in n_qubits_choices
    ):
        raise ValueError(
            f"all near-Clifford qubit choices must be in "
            f"[1, {MAX_NEAR_CLIFFORD_QUBITS}]"
        )
    if any(type(depth) is not int or depth < 1 for depth in depth_choices):
        raise ValueError("all near-Clifford depth choices must be positive")
    if any(
        type(count) is not int or count < 1
        for count in non_clifford_count_choices
    ):
        raise ValueError("all near-Clifford insertion counts must be positive")
    for theta in theta_choices:
        if (
            isinstance(theta, (bool, np.bool_))
            or not isinstance(theta, (int, float, np.integer, np.floating))
            or not math.isfinite(theta)
        ):
            raise ValueError("all theta choices must be finite")
        quarter_turns = theta / (math.pi / 2.0)
        if math.isclose(
            quarter_turns, round(quarter_turns), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("theta choices must not contain Clifford RZ angles")
    eligible_qubits = [
        n
        for n in n_qubits_choices
        if any(count <= n for count in non_clifford_count_choices)
    ]
    if not eligible_qubits:
        raise ValueError("no n_qubits choice can realize an insertion count")
    if len(eligible_qubits) != len(n_qubits_choices):
        raise ValueError("each n_qubits choice must realize an insertion count")
    if any(
        not any(count <= n for n in n_qubits_choices)
        for count in non_clifford_count_choices
    ):
        raise ValueError("each insertion count must be realizable by n_qubits choices")


def sample_near_clifford_params(
    rng: np.random.Generator,
    n_qubits_choices: list[int],
    depth_choices: list[int],
    non_clifford_count_choices: list[int],
    theta_choices: list[float],
    instance: int,
    circuit_seed: int,
) -> NearCliffordParams:
    """Draw one near-Clifford configuration from the preset ranges."""
    validate_near_clifford_sampling_domain(
        n_qubits_choices,
        depth_choices,
        non_clifford_count_choices,
        theta_choices,
    )
    eligible_qubits = [
        n
        for n in n_qubits_choices
        if any(count <= n for count in non_clifford_count_choices)
    ]
    if type(instance) is not int or instance < 0:
        raise ValueError("instance must be a nonnegative integer")
    if type(circuit_seed) is not int or circuit_seed < 0:
        raise ValueError("circuit_seed must be a nonnegative integer")
    n_qubits = int(rng.choice(eligible_qubits))
    eligible_counts = [count for count in non_clifford_count_choices if count <= n_qubits]
    return NearCliffordParams(
        n_qubits=n_qubits,
        depth=int(rng.choice(depth_choices)),
        non_clifford_count=int(rng.choice(eligible_counts)),
        theta=float(rng.choice(theta_choices)),
        circuit_seed=circuit_seed,
        instance=instance,
    )


def build_near_clifford_circuit(params: NearCliffordParams) -> QuantumCircuit:
    """Build the corresponding random Clifford circuit with seeded insertions."""
    base_params = RandomCliffordParams(
        n_qubits=params.n_qubits,
        depth=params.depth,
        circuit_seed=params.circuit_seed,
        instance=params.instance,
    )
    base = build_random_clifford_circuit(base_params)

    insertion_rng = np.random.default_rng(
        np.random.SeedSequence(params.circuit_seed, spawn_key=(3,))
    )
    n_slots = len(base.data) + 1
    slots = insertion_rng.integers(0, n_slots, size=params.non_clifford_count)
    qubits = insertion_rng.choice(
        params.n_qubits, size=params.non_clifford_count, replace=False
    )
    gate_indices = insertion_rng.integers(
        0, len(NON_CLIFFORD_GATES), size=params.non_clifford_count
    )

    insertions: dict[int, list[tuple[str, int]]] = {}
    for slot, qubit, gate_index in zip(slots, qubits, gate_indices):
        insertions.setdefault(int(slot), []).append(
            (NON_CLIFFORD_GATES[int(gate_index)], int(qubit))
        )

    circuit = QuantumCircuit(params.n_qubits)
    for slot in range(n_slots):
        for gate_name, qubit in insertions.get(slot, []):
            if gate_name == "t":
                circuit.t(qubit)
            else:
                circuit.rz(params.theta, qubit)
        if slot < len(base.data):
            instruction = base.data[slot]
            qargs = [
                circuit.qubits[base.find_bit(qubit).index]
                for qubit in instruction.qubits
            ]
            circuit.append(instruction.operation, qargs)
    return circuit
