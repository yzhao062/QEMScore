"""Seeded random Clifford circuits with layered one- and two-qubit gates.

Each layer applies one gate from a fixed named one-qubit Clifford gate set to every
qubit. A seeded permutation then pairs qubits, and each pair receives either CZ or
CX. The circuit seed is split into named streams for gate choices, pairings, and
entangler choices so rebuilding a parameter record is independent of caller RNG
state. Circuits carry no measurements; sampling attaches them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

MAX_RANDOM_CLIFFORD_QUBITS = 20

# These names have direct Qiskit and Stim representations. Every member is a
# one-qubit Clifford gate; identity remains explicit so the layer shape is fixed.
ONE_QUBIT_CLIFFORD_GATES: tuple[str, ...] = (
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
)
TWO_QUBIT_CLIFFORD_GATES: tuple[str, ...] = ("cz", "cx")


@dataclass(frozen=True)
class RandomCliffordParams:
    n_qubits: int
    depth: int
    circuit_seed: int
    instance: int

    def __post_init__(self) -> None:
        if type(self.n_qubits) is not int:
            raise ValueError("random Clifford n_qubits must be an integer")
        if type(self.depth) is not int:
            raise ValueError("random Clifford depth must be an integer")
        if type(self.circuit_seed) is not int:
            raise ValueError("circuit_seed must be an integer")
        if type(self.instance) is not int:
            raise ValueError("instance must be an integer")
        if not 1 <= self.n_qubits <= MAX_RANDOM_CLIFFORD_QUBITS:
            raise ValueError(
                f"random Clifford n_qubits must be in "
                f"[1, {MAX_RANDOM_CLIFFORD_QUBITS}]"
            )
        if self.depth < 1:
            raise ValueError("random Clifford depth must be positive")
        if self.circuit_seed < 0:
            raise ValueError("circuit_seed must be nonnegative")
        if self.instance < 0:
            raise ValueError("instance must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


def sample_random_clifford_params(
    rng: np.random.Generator,
    n_qubits_choices: list[int],
    depth_choices: list[int],
    instance: int,
    circuit_seed: int,
) -> RandomCliffordParams:
    """Draw one random Clifford configuration from the preset ranges."""
    if not n_qubits_choices:
        raise ValueError("n_qubits_choices must not be empty")
    if not depth_choices:
        raise ValueError("depth_choices must not be empty")
    if any(
        type(n) is not int or not 1 <= n <= MAX_RANDOM_CLIFFORD_QUBITS
        for n in n_qubits_choices
    ):
        raise ValueError(
            f"all random Clifford qubit choices must be in "
            f"[1, {MAX_RANDOM_CLIFFORD_QUBITS}]"
        )
    if any(type(depth) is not int or depth < 1 for depth in depth_choices):
        raise ValueError("all random Clifford depth choices must be positive")
    if type(instance) is not int or instance < 0:
        raise ValueError("instance must be a nonnegative integer")
    if type(circuit_seed) is not int or circuit_seed < 0:
        raise ValueError("circuit_seed must be a nonnegative integer")
    return RandomCliffordParams(
        n_qubits=int(rng.choice(n_qubits_choices)),
        depth=int(rng.choice(depth_choices)),
        circuit_seed=circuit_seed,
        instance=instance,
    )


def build_random_clifford_circuit(params: RandomCliffordParams) -> QuantumCircuit:
    """Build the measurement-free circuit specified by ``params``."""
    one_qubit_rng = np.random.default_rng(
        np.random.SeedSequence(params.circuit_seed, spawn_key=(0,))
    )
    pairing_rng = np.random.default_rng(
        np.random.SeedSequence(params.circuit_seed, spawn_key=(1,))
    )
    entangler_rng = np.random.default_rng(
        np.random.SeedSequence(params.circuit_seed, spawn_key=(2,))
    )

    circuit = QuantumCircuit(params.n_qubits)
    for _ in range(params.depth):
        gate_indices = one_qubit_rng.integers(
            0, len(ONE_QUBIT_CLIFFORD_GATES), size=params.n_qubits
        )
        for qubit, gate_index in enumerate(gate_indices):
            gate_name = ONE_QUBIT_CLIFFORD_GATES[int(gate_index)]
            getattr(circuit, gate_name)(qubit)

        permutation = pairing_rng.permutation(params.n_qubits)
        for offset in range(0, params.n_qubits - 1, 2):
            first = int(permutation[offset])
            second = int(permutation[offset + 1])
            gate_name = TWO_QUBIT_CLIFFORD_GATES[
                int(entangler_rng.integers(0, len(TWO_QUBIT_CLIFFORD_GATES)))
            ]
            getattr(circuit, gate_name)(first, second)
    return circuit
