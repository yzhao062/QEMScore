"""Exact Z-type Pauli labels for Clifford-only Qiskit circuits via Stim.

The converter intentionally supports only the fixed gate set emitted by the random
Clifford family. Unsupported gates fail closed. Stim evolves the stabilizer tableau
and evaluates the observable exactly; no shots or sampling are involved.
"""

from __future__ import annotations

import stim
from qiskit import QuantumCircuit

_QISKIT_TO_STIM: dict[str, tuple[str, int]] = {
    "id": ("I", 1),
    "x": ("X", 1),
    "y": ("Y", 1),
    "z": ("Z", 1),
    "h": ("H", 1),
    "s": ("S", 1),
    "sdg": ("S_DAG", 1),
    "sx": ("SQRT_X", 1),
    "sxdg": ("SQRT_X_DAG", 1),
    "cz": ("CZ", 2),
    "cx": ("CX", 2),
}


def qiskit_to_stim_circuit(circuit: QuantumCircuit) -> stim.Circuit:
    """Convert a measurement-free Qiskit circuit from the supported gate set."""
    if circuit.num_clbits:
        raise ValueError("Stim label conversion expects a measurement-free circuit")

    converted = stim.Circuit()
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in _QISKIT_TO_STIM:
            raise ValueError(
                f"unsupported gate for Stim Clifford labels: {operation.name}"
            )
        stim_name, arity = _QISKIT_TO_STIM[operation.name]
        if operation.params:
            raise ValueError(
                f"parameterized gate is unsupported for Stim labels: {operation.name}"
            )
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(qubits) != arity:
            raise ValueError(
                f"gate {operation.name} has arity {len(qubits)}, expected {arity}"
            )
        converted.append(stim_name, qubits)

    # Stim infers register size from targets. Preserve trailing idle Qiskit qubits.
    if circuit.num_qubits and converted.num_qubits < circuit.num_qubits:
        converted.append("I", [circuit.num_qubits - 1])
    return converted


def ideal_expectation(circuit: QuantumCircuit, pauli_label: str) -> float:
    """Return the exact ideal expectation of a Z-type Qiskit Pauli label."""
    if len(pauli_label) != circuit.num_qubits:
        raise ValueError(
            f"Pauli label length {len(pauli_label)} does not match "
            f"{circuit.num_qubits} circuit qubits"
        )
    if set(pauli_label) - {"I", "Z"}:
        raise ValueError("Stim labels currently support only Z-type Pauli strings")

    converted = qiskit_to_stim_circuit(circuit)
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(circuit.num_qubits)
    simulator.do_circuit(converted)

    # Qiskit writes qubit 0 at the right of a Pauli label; Stim writes it at index 0.
    observable = stim.PauliString(pauli_label[::-1])
    value = simulator.peek_observable_expectation(observable)
    if value not in (-1, 0, 1):
        raise RuntimeError(f"unexpected non-discrete stabilizer expectation: {value}")
    return float(value)


# This explicit name is convenient when both exact-label modules are imported.
stim_ideal_expectation = ideal_expectation
