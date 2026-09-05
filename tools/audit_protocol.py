"""A second implementation of the two circuit protocols and their exact labels.

Round 4 of the plan review made the point precisely: an audit that calls the same
circuit builder, the same label routine and the same noise model can show that a
recorded execution disagrees with the declared protocol and seeds, but a fault
inside those shared functions is invisible to it. This module is the independent
half. Nothing here imports from `qem_bench`.

The protocols, written from their specification rather than from the code:

    TFI, one Trotter step: rzz(-2 j dt) on every neighbouring pair of an open
    chain in ascending order, then rx(-2 h dt) on every qubit in ascending order.

    Heisenberg, one Trotter step: for every neighbouring pair of an open chain in
    ascending order, rxx(2 jx dt), then ryy(2 jy dt), then rzz(2 jz dt).

Qubit 0 is the least significant bit, which is the convention the stored counts
follow. The state is carried as an n-axis tensor with one axis per qubit, so a
gate is applied by contracting its axes rather than by building a 2^n matrix.

What this deliberately does not reimplement is the noise model. Reproducing a
stored histogram bit for bit requires the sampler to consume randomness in the
same order, and a second noise construction would not, so the histogram
comparison and an independent noise model cannot both hold. The audit keeps the
exact histogram comparison and compares the noise model structurally instead.
"""

from __future__ import annotations

from collections.abc import Sequence
import cmath
import math

import numpy as np

# One entry per gate the two protocols use: name, arity, and the matrix builder.
_SQRT_HALF = 1.0 / math.sqrt(2.0)


def tfi_gates(*, n_qubits: int, steps: int, j: float, h: float, dt: float):
    """The TFI Trotter sequence, as (name, angle, qubits) triples."""
    if n_qubits < 2 or steps < 1:
        raise ValueError("the TFI protocol needs at least two qubits and one step")
    sequence = []
    for _ in range(steps):
        for qubit in range(n_qubits - 1):
            sequence.append(("rzz", -2.0 * j * dt, (qubit, qubit + 1)))
        for qubit in range(n_qubits):
            sequence.append(("rx", -2.0 * h * dt, (qubit,)))
    return sequence


def heisenberg_gates(*, n_qubits: int, steps: int, jx: float, jy: float,
                     jz: float, dt: float):
    """The anisotropic Heisenberg Trotter sequence, as triples."""
    if n_qubits < 2 or steps < 1:
        raise ValueError("the Heisenberg protocol needs two qubits and one step")
    sequence = []
    for _ in range(steps):
        for qubit in range(n_qubits - 1):
            pair = (qubit, qubit + 1)
            sequence.append(("rxx", 2.0 * jx * dt, pair))
            sequence.append(("ryy", 2.0 * jy * dt, pair))
            sequence.append(("rzz", 2.0 * jz * dt, pair))
    return sequence


def protocol_gates(family: str, parameters):
    """Dispatch to the family's sequence, refusing anything else."""
    if family == "tfi":
        return tfi_gates(
            n_qubits=int(parameters["n_qubits"]), steps=int(parameters["steps"]),
            j=float(parameters["j"]), h=float(parameters["h"]),
            dt=float(parameters["dt"]),
        )
    if family == "heisenberg":
        return heisenberg_gates(
            n_qubits=int(parameters["n_qubits"]), steps=int(parameters["steps"]),
            jx=float(parameters["jx"]), jy=float(parameters["jy"]),
            jz=float(parameters["jz"]), dt=float(parameters["dt"]),
        )
    raise ValueError(f"the campaign protocols are tfi and heisenberg, not {family!r}")


def _rx(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array([[math.cos(half), -1j * math.sin(half)],
                     [-1j * math.sin(half), math.cos(half)]], dtype=complex)


def _two_qubit_rotation(pauli: str, angle: float) -> np.ndarray:
    """exp(-i angle/2 P (x) P) for P in {X, Y, Z}, in the |q1 q0> basis of the pair."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    operator = {"x": x, "y": y, "z": z}[pauli]
    product = np.kron(operator, operator)
    half = angle / 2.0
    return math.cos(half) * np.eye(4, dtype=complex) - 1j * math.sin(half) * product


_GATES = {
    "rx": lambda angle: _rx(angle),
    "rxx": lambda angle: _two_qubit_rotation("x", angle),
    "ryy": lambda angle: _two_qubit_rotation("y", angle),
    "rzz": lambda angle: _two_qubit_rotation("z", angle),
}


def statevector(gates: Sequence[tuple], n_qubits: int) -> np.ndarray:
    """Evolve |0...0> through the sequence, one axis per qubit."""
    state = np.zeros((2,) * n_qubits, dtype=complex)
    state[(0,) * n_qubits] = 1.0
    for name, angle, qubits in gates:
        matrix = _GATES[name](angle)
        if len(qubits) == 1:
            state = np.tensordot(matrix, state, axes=([1], [qubits[0]]))
            state = np.moveaxis(state, 0, qubits[0])
            continue
        # The pair matrix is written in the |high low> basis, so the second
        # qubit of the pair carries the more significant index.
        low, high = qubits
        operator = matrix.reshape(2, 2, 2, 2)
        state = np.tensordot(operator, state, axes=([2, 3], [high, low]))
        state = np.moveaxis(state, [0, 1], [high, low])
    return state


def z_expectation(gates: Sequence[tuple], n_qubits: int,
                  support: Sequence[int]) -> float:
    """The exact expectation of the product of Z over ``support``."""
    state = statevector(gates, n_qubits)
    probabilities = np.abs(state) ** 2
    signs = np.ones((2,) * n_qubits, dtype=float)
    for qubit in support:
        shape = [1] * n_qubits
        shape[qubit] = 2
        signs = signs * np.array([1.0, -1.0]).reshape(shape)
    value = float(np.sum(probabilities * signs))
    if not math.isfinite(value):
        raise ValueError("the independent label must be finite")
    return value


def instruction_stream(gates: Sequence[tuple]) -> list[tuple]:
    """A comparable rendering of the sequence, with angles rounded for equality.

    Angles are compared at the precision the identity descriptor serializes, so
    a difference that survives rounding is a difference in the protocol rather
    than in the arithmetic.
    """
    return [
        (name, tuple(int(qubit) for qubit in qubits), round(float(angle), 12))
        for name, angle, qubits in gates
    ]


def normalized_probability(state: np.ndarray) -> float:
    """Total probability, which a correct evolution keeps at one."""
    return float(np.sum(np.abs(state) ** 2))


__all__ = [
    "heisenberg_gates",
    "instruction_stream",
    "normalized_probability",
    "protocol_gates",
    "statevector",
    "tfi_gates",
    "z_expectation",
]
