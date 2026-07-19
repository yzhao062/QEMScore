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
