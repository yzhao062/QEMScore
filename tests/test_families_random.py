"""Oracles and deterministic contracts for random and near-Clifford families."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Operator, SparsePauliOp

from qemscore.circuits.near_clifford import (
    MAX_NEAR_CLIFFORD_QUBITS,
    NearCliffordParams,
    build_near_clifford_circuit,
    sample_near_clifford_params,
)
from qemscore.circuits.random_clifford import (
    MAX_RANDOM_CLIFFORD_QUBITS,
    RandomCliffordParams,
    build_random_clifford_circuit,
    sample_random_clifford_params,
)
from qemscore.labels.statevector import ideal_expectation as statevector_expectation
from qemscore.labels.stim_labels import ideal_expectation as stim_expectation
from qemscore.observables import z_support_label


@pytest.mark.parametrize(
    ("n_qubits", "depth", "seed"),
    [(3, 3, 11), (4, 4, 29), (5, 3, 47), (6, 5, 83)],
)
def test_stim_exactly_matches_statevector_oracle(n_qubits, depth, seed):
    """Independent exact-label paths agree at the dataset's 12-digit precision."""
    params = RandomCliffordParams(n_qubits, depth, seed, instance=seed)
    circuit = build_random_clifford_circuit(params)
    Clifford(circuit)

    for locality in range(1, n_qubits + 1):
        for support in combinations(range(n_qubits), locality):
            label = z_support_label(n_qubits, support)
            stim_value = stim_expectation(circuit, label)
            statevector_value = statevector_expectation(circuit, label)
            assert stim_value == round(statevector_value, 12)


def test_stim_gate_dagger_directions_are_discriminated():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)
    circuit.sx(0)

    assert statevector_expectation(circuit, "Z") == pytest.approx(1.0)
    assert stim_expectation(circuit, "Z") == 1.0


def test_random_clifford_ideal_values_are_discrete():
    values = []
    for seed in range(12):
        params = RandomCliffordParams(5, 4, seed, instance=seed)
        circuit = build_random_clifford_circuit(params)
        for support in ((0,), (2,), (4,), (0, 1), (1, 3, 4)):
            value = statevector_expectation(circuit, z_support_label(5, support))
            values.append(value)
            assert min(abs(value - target) for target in (-1.0, 0.0, 1.0)) < 1e-12
    assert any(abs(value) < 1e-12 for value in values)
    assert any(abs(value) > 1.0 - 1e-12 for value in values)


def test_random_clifford_sampling_and_build_are_seeded():
    kwargs = {
        "n_qubits_choices": [3, 4, 5],
        "depth_choices": [2, 4],
        "instance": 7,
        "circuit_seed": 9182,
    }
    first = sample_random_clifford_params(np.random.default_rng(123), **kwargs)
    second = sample_random_clifford_params(np.random.default_rng(123), **kwargs)
    assert first == second
    assert build_random_clifford_circuit(first) == build_random_clifford_circuit(second)
    assert build_random_clifford_circuit(first) != build_random_clifford_circuit(
        replace(first, circuit_seed=first.circuit_seed + 1)
    )


@pytest.mark.parametrize("value", (2.9, 2.0, True))
@pytest.mark.parametrize("family", ("random_clifford", "near_clifford"))
def test_random_family_samplers_reject_non_exact_integer_depths(family, value):
    common = {
        "rng": np.random.default_rng(1),
        "n_qubits_choices": [4],
        "depth_choices": [value],
        "instance": 0,
        "circuit_seed": 0,
    }
    if family == "random_clifford":
        with pytest.raises(ValueError, match="depth choices"):
            sample_random_clifford_params(**common)
    else:
        with pytest.raises(ValueError, match="depth choices"):
            sample_near_clifford_params(
                **common,
                non_clifford_count_choices=[1],
                theta_choices=[np.pi / 7],
            )


@pytest.mark.parametrize(
    ("sampler", "kwargs", "message"),
    (
        (
            sample_random_clifford_params,
            {"n_qubits_choices": [4.0], "depth_choices": [2]},
            "qubit choices",
        ),
        (
            sample_near_clifford_params,
            {
                "n_qubits_choices": [True],
                "depth_choices": [2],
                "non_clifford_count_choices": [1],
                "theta_choices": [np.pi / 7],
            },
            "qubit choices",
        ),
        (
            sample_near_clifford_params,
            {
                "n_qubits_choices": [4],
                "depth_choices": [2],
                "non_clifford_count_choices": [1.9],
                "theta_choices": [np.pi / 7],
            },
            "insertion counts",
        ),
    ),
)
def test_random_family_samplers_reject_other_non_exact_integer_choices(
    sampler, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        sampler(
            np.random.default_rng(1),
            **kwargs,
            instance=0,
            circuit_seed=0,
        )


@pytest.mark.parametrize("field", ("instance", "circuit_seed"))
@pytest.mark.parametrize("value", (2.5, 2.0, True))
def test_random_family_samplers_reject_non_exact_integer_identifiers(field, value):
    kwargs = {
        "n_qubits_choices": [4],
        "depth_choices": [2],
        "instance": 0,
        "circuit_seed": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        sample_random_clifford_params(np.random.default_rng(1), **kwargs)


def test_near_clifford_sampler_rejects_boolean_theta():
    with pytest.raises(ValueError, match="theta choices"):
        sample_near_clifford_params(
            np.random.default_rng(1),
            n_qubits_choices=[4],
            depth_choices=[2],
            non_clifford_count_choices=[1],
            theta_choices=[True],
            instance=0,
            circuit_seed=0,
        )


@pytest.mark.parametrize(("n_qubits", "depth"), [(3, 2), (4, 5), (7, 3)])
def test_random_family_instruction_counts_match_parameters(n_qubits, depth):
    base_count = depth * (n_qubits + n_qubits // 2)
    rc = build_random_clifford_circuit(
        RandomCliffordParams(n_qubits, depth, circuit_seed=31, instance=0)
    )
    assert len(rc.data) == base_count

    insertion_count = min(2, n_qubits)
    nc = build_near_clifford_circuit(
        NearCliffordParams(
            n_qubits,
            depth,
            insertion_count,
            np.pi / 5,
            circuit_seed=31,
            instance=0,
        )
    )
    assert len(nc.data) == base_count + insertion_count


def test_near_clifford_sampling_build_and_statevector_label():
    kwargs = {
        "n_qubits_choices": [4, 5],
        "depth_choices": [2, 3],
        "non_clifford_count_choices": [1, 3],
        "theta_choices": [np.pi / 7, np.pi / 5],
        "instance": 9,
        "circuit_seed": 771,
    }
    first = sample_near_clifford_params(np.random.default_rng(456), **kwargs)
    second = sample_near_clifford_params(np.random.default_rng(456), **kwargs)
    assert first == second

    circuit = build_near_clifford_circuit(first)
    assert circuit == build_near_clifford_circuit(second)
    op_counts = circuit.count_ops()
    assert op_counts.get("t", 0) + op_counts.get("rz", 0) == first.non_clifford_count
    with pytest.raises(QiskitError):
        Clifford(circuit)

    label = z_support_label(first.n_qubits, (0, first.n_qubits - 1))
    value = statevector_expectation(circuit, label)
    assert np.isfinite(value)
    assert -1.0 - 1e-12 <= value <= 1.0 + 1e-12
    with pytest.raises(ValueError, match="unsupported gate"):
        stim_expectation(circuit, label)


def _is_clifford_unitary(circuit: QuantumCircuit) -> bool:
    operator = Operator(circuit)
    for basis in ("X", "Z"):
        for qubit in range(circuit.num_qubits):
            label = "".join(
                basis if index == circuit.num_qubits - 1 - qubit else "I"
                for index in range(circuit.num_qubits)
            )
            pauli = Operator(SparsePauliOp(label))
            conjugated = operator @ pauli @ operator.adjoint()
            decomposition = SparsePauliOp.from_operator(conjugated)
            coefficients = np.abs(decomposition.coeffs)
            if not (
                np.sum(coefficients > 1e-9) == 1
                and np.isclose(np.max(coefficients), 1.0)
            ):
                return False
    return True


def test_near_clifford_reviewer_configuration_is_algebraically_non_clifford():
    algebraically_clifford = 0
    for seed in range(100):
        circuit = build_near_clifford_circuit(
            NearCliffordParams(
                n_qubits=3,
                depth=2,
                non_clifford_count=2,
                theta=np.pi / 5,
                circuit_seed=seed,
                instance=seed,
            )
        )
        algebraically_clifford += int(_is_clifford_unitary(circuit))
    assert algebraically_clifford == 0


def test_family_qubit_caps_and_non_clifford_angle_are_enforced():
    with pytest.raises(ValueError, match="n_qubits"):
        RandomCliffordParams(MAX_RANDOM_CLIFFORD_QUBITS + 1, 1, 0, 0)
    with pytest.raises(ValueError, match="n_qubits"):
        NearCliffordParams(MAX_NEAR_CLIFFORD_QUBITS + 1, 1, 1, np.pi / 7, 0, 0)
    with pytest.raises(ValueError, match="Clifford RZ angle"):
        NearCliffordParams(3, 1, 1, np.pi / 2, 0, 0)
    with pytest.raises(ValueError, match="cannot exceed"):
        NearCliffordParams(3, 1, 4, np.pi / 7, 0, 0)
