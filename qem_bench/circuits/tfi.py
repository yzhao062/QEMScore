"""Transverse-field Ising chain, first-order Trotter circuits.

H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i on an open chain. One first-order Trotter
step of duration dt applies RZZ(-2*J*dt) on every nearest-neighbor pair, then
RX(-2*h*dt) on every qubit (Qiskit convention: RZZ(theta) = exp(-i theta ZZ/2),
RX(theta) = exp(-i theta X/2), so the negative Hamiltonian needs negative angles).
Circuits carry no measurements; sampling attaches them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class TFIParams:
    n_qubits: int
    steps: int
    j: float
    h: float
    dt: float
    circuit_seed: int
    instance: int

    def __post_init__(self) -> None:
        if type(self.n_qubits) is not int or self.n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if type(self.circuit_seed) is not int or self.circuit_seed < 0:
            raise ValueError("circuit_seed must be a nonnegative integer")
        if type(self.instance) is not int or self.instance < 0:
            raise ValueError("instance must be a nonnegative integer")

    def to_dict(self) -> dict:
        return asdict(self)


def sample_tfi_params(
    rng: np.random.Generator,
    n_qubits_choices: list[int],
    steps_choices: list[int],
    dt: float,
    instance: int,
    circuit_seed: int,
) -> TFIParams:
    """Draw one circuit configuration from the preset ranges."""
    if not n_qubits_choices or any(
        type(n_qubits) is not int or n_qubits < 1
        for n_qubits in n_qubits_choices
    ):
        raise ValueError("n_qubits_choices must contain positive integers")
    if not steps_choices or any(
        type(steps) is not int or steps < 1 for steps in steps_choices
    ):
        raise ValueError("steps_choices must contain positive integers")
    if type(instance) is not int or instance < 0:
        raise ValueError("instance must be a nonnegative integer")
    if type(circuit_seed) is not int or circuit_seed < 0:
        raise ValueError("circuit_seed must be a nonnegative integer")

    return TFIParams(
        n_qubits=int(rng.choice(n_qubits_choices)),
        steps=int(rng.choice(steps_choices)),
        j=float(rng.uniform(0.2, 1.2)),
        h=float(rng.uniform(0.2, 1.2)),
        dt=dt,
        circuit_seed=circuit_seed,
        instance=instance,
    )


def build_tfi_circuit(params: TFIParams) -> QuantumCircuit:
    circ = QuantumCircuit(params.n_qubits)
    for _ in range(params.steps):
        for q in range(params.n_qubits - 1):
            circ.rzz(-2.0 * params.j * params.dt, q, q + 1)
        for q in range(params.n_qubits):
            circ.rx(-2.0 * params.h * params.dt, q)
    return circ
