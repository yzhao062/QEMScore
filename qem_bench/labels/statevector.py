"""Exact ideal labels by dense statevector simulation.

Exact labels are the benchmark's ground truth. They are logged separately from the
circuit-evaluation ledger because they are not noisy backend calls.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def ideal_expectation(circuit: QuantumCircuit, pauli_label: str) -> float:
    """Exact <psi(c)|P|psi(c)> for a measurement-free circuit."""
    if circuit.num_clbits:
        raise ValueError("ideal_expectation expects a measurement-free circuit")
    state = Statevector.from_instruction(circuit)
    return float(state.expectation_value(SparsePauliOp(pauli_label)).real)
