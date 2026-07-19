"""Anisotropic Heisenberg-chain circuits with first-order Trotter steps.

The open-chain Hamiltonian is
``H = sum_i (Jx X_i X_(i+1) + Jy Y_i Y_(i+1) + Jz Z_i Z_(i+1))``.
This implementation samples ``Jx``, ``Jy``, and ``Jz`` independently, rather than
restricting to the XXZ case ``Jx = Jy``. Setting those two stored values equal gives
the XXZ family.

Qiskit uses ``RPP(theta) = exp(-i theta P P / 2)`` for ``P`` in ``{X, Y, Z}``.
Positive Hamiltonian couplings therefore require the positive angles
``2 * Jp * dt``. Within each first-order step, bonds are visited from left to right
and each bond receives RXX, then RYY, then RZZ. Circuits carry no measurements or
state preparation; callers select the input state and observables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

MAX_QUBITS = 12


@dataclass(frozen=True)
class HeisenbergParams:
    n_qubits: int
    steps: int
    jx: float
    jy: float
    jz: float
    dt: float
    circuit_seed: int
    instance: int

    def to_dict(self) -> dict:
        return asdict(self)


def sample_heisenberg_params(
    rng: np.random.Generator,
    n_qubits_choices: list[int],
    steps_choices: list[int],
    dt: float,
    instance: int,
    circuit_seed: int,
) -> HeisenbergParams:
    """Draw one anisotropic chain configuration from the preset ranges."""
    qubit_choices = tuple(int(n) for n in n_qubits_choices)
    if not qubit_choices:
        raise ValueError("n_qubits_choices must not be empty")
    if any(n < 2 or n > MAX_QUBITS for n in qubit_choices):
        raise ValueError(f"Heisenberg supports 2 to {MAX_QUBITS} qubits")

    trotter_choices = tuple(int(steps) for steps in steps_choices)
    if not trotter_choices or any(steps < 1 for steps in trotter_choices):
        raise ValueError("steps_choices must contain positive integers")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    return HeisenbergParams(
        n_qubits=int(rng.choice(qubit_choices)),
        steps=int(rng.choice(trotter_choices)),
        jx=float(rng.uniform(0.2, 1.2)),
        jy=float(rng.uniform(0.2, 1.2)),
        jz=float(rng.uniform(0.2, 1.2)),
        dt=float(dt),
        circuit_seed=circuit_seed,
        instance=instance,
    )


def build_heisenberg_circuit(params: HeisenbergParams) -> QuantumCircuit:
    """Build the stored open-chain first-order Trotter circuit."""
    if not 2 <= params.n_qubits <= MAX_QUBITS:
        raise ValueError(f"Heisenberg supports 2 to {MAX_QUBITS} qubits")
    if params.steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(params.dt) or params.dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not all(np.isfinite(coupling) for coupling in (params.jx, params.jy, params.jz)):
        raise ValueError("Heisenberg couplings must be finite")

    circ = QuantumCircuit(params.n_qubits)
    for _ in range(params.steps):
        for q in range(params.n_qubits - 1):
            circ.rxx(2.0 * params.jx * params.dt, q, q + 1)
            circ.ryy(2.0 * params.jy * params.dt, q, q + 1)
            circ.rzz(2.0 * params.jz * params.dt, q, q + 1)
    return circ
