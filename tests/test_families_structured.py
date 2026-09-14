"""Physics-oracle and determinism tests for structured circuit families."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qemscore.circuits.heisenberg import (
    HeisenbergParams,
    build_heisenberg_circuit,
    sample_heisenberg_params,
)
from qemscore.circuits.qaoa import (
    GRAPH_CLASSES,
    QAOAParams,
    build_qaoa_circuit,
    sample_qaoa_params,
)
from qemscore.circuits.tfi import sample_tfi_params


I = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def _operator_on_qubits(
    n_qubits: int, single_qubit_operators: dict[int, np.ndarray]
) -> np.ndarray:
    """Build an operator in Qiskit's basis order, q_(n-1) tensor ... tensor q_0."""
    result = np.array([[1.0]], dtype=complex)
    for q in reversed(range(n_qubits)):
        result = np.kron(result, single_qubit_operators.get(q, I))
    return result


def test_qaoa_unitary_matches_cost_and_mixer_product():
    gamma, beta = 0.37, 0.23
    params = QAOAParams(
        n_qubits=2,
        graph_class="path",
        edges=((0, 1),),
        p=1,
        gammas=(gamma,),
        betas=(beta,),
        circuit_seed=0,
        instance=0,
    )
    u_circuit = Operator(build_qaoa_circuit(params)).data

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    u_initial = np.kron(hadamard, hadamard)
    zz = np.kron(Z, Z)
    u_cost = np.diag(np.exp(-1j * gamma * np.diag(zz)))
    rx = np.cos(beta) * I - 1j * np.sin(beta) * X
    u_mixer = np.kron(rx, rx)
    expected = u_mixer @ u_cost @ u_initial

    assert np.allclose(u_circuit, expected, atol=1e-10)


def test_qaoa_two_layers_match_distinct_cost_and_mixer_products():
    gammas = (0.17, 0.41)
    betas = (0.29, 0.13)
    params = QAOAParams(
        n_qubits=2,
        graph_class="path",
        edges=((0, 1),),
        p=2,
        gammas=gammas,
        betas=betas,
        circuit_seed=0,
        instance=0,
    )
    u_circuit = Operator(build_qaoa_circuit(params)).data

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    expected = np.kron(hadamard, hadamard)
    zz = np.kron(Z, Z)
    for gamma, beta in zip(gammas, betas):
        cost = np.diag(np.exp(-1j * gamma * np.diag(zz)))
        rx = np.cos(beta) * I - 1j * np.sin(beta) * X
        expected = np.kron(rx, rx) @ cost @ expected

    assert np.allclose(u_circuit, expected, atol=1e-10)


@pytest.mark.parametrize("graph_class", GRAPH_CLASSES)
def test_qaoa_sampling_and_build_are_deterministic(graph_class):
    kwargs = {
        "n_qubits_choices": [4, 6],
        "p_choices": [1, 2],
        "graph_classes": [graph_class],
        "instance": 7,
        "circuit_seed": 19,
    }
    first = sample_qaoa_params(np.random.default_rng(314), **kwargs)
    second = sample_qaoa_params(np.random.default_rng(314), **kwargs)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert build_qaoa_circuit(first) == build_qaoa_circuit(first)
    if graph_class == "3_regular":
        degrees = [0] * first.n_qubits
        for q0, q1 in first.edges:
            degrees[q0] += 1
            degrees[q1] += 1
        assert degrees == [3] * first.n_qubits


@pytest.mark.parametrize("value", (1.9, 1.0, True))
def test_qaoa_sampling_rejects_non_exact_integer_depths(value):
    with pytest.raises(ValueError, match="QAOA p choices"):
        sample_qaoa_params(
            np.random.default_rng(1),
            n_qubits_choices=[4],
            p_choices=[value],
            graph_classes=["path"],
            instance=0,
            circuit_seed=0,
        )


@pytest.mark.parametrize(
    ("sampler", "kwargs", "message"),
    (
        (
            sample_tfi_params,
            {"n_qubits_choices": [3.0], "steps_choices": [1], "dt": 0.2},
            "n_qubits_choices",
        ),
        (
            sample_tfi_params,
            {"n_qubits_choices": [3], "steps_choices": [True], "dt": 0.2},
            "steps_choices",
        ),
        (
            sample_heisenberg_params,
            {"n_qubits_choices": [3.9], "steps_choices": [1], "dt": 0.2},
            "Heisenberg supports",
        ),
        (
            sample_heisenberg_params,
            {"n_qubits_choices": [3], "steps_choices": [1.0], "dt": 0.2},
            "steps_choices",
        ),
        (
            sample_qaoa_params,
            {
                "n_qubits_choices": [True],
                "p_choices": [1],
                "graph_classes": ["path"],
            },
            "QAOA supports",
        ),
    ),
)
def test_structured_samplers_reject_other_non_exact_integer_choices(
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
def test_structured_samplers_reject_non_exact_integer_identifiers(field, value):
    kwargs = {
        "n_qubits_choices": [3],
        "steps_choices": [1],
        "dt": 0.2,
        "instance": 0,
        "circuit_seed": 0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        sample_tfi_params(np.random.default_rng(1), **kwargs)


def test_qaoa_builder_rejects_non_integer_edge_endpoints():
    params = QAOAParams(
        n_qubits=2,
        graph_class="path",
        edges=((0, 1.0),),
        p=1,
        gammas=(0.2,),
        betas=(0.3,),
        circuit_seed=0,
        instance=0,
    )
    with pytest.raises(ValueError, match="endpoints must be integers"):
        build_qaoa_circuit(params)


def test_three_regular_sampler_reaches_distinct_topologies():
    triangle_counts = set()
    for seed in range(40):
        params = sample_qaoa_params(
            np.random.default_rng(seed),
            n_qubits_choices=[6],
            p_choices=[1],
            graph_classes=["3_regular"],
            instance=seed,
            circuit_seed=seed,
        )
        edges = set(params.edges)
        triangles = sum(
            (q0, q1) in edges and (q0, q2) in edges and (q1, q2) in edges
            for q0 in range(6)
            for q1 in range(q0 + 1, 6)
            for q2 in range(q1 + 1, 6)
        )
        triangle_counts.add(triangles)
    assert len(triangle_counts) >= 2


def test_heisenberg_unitary_matches_first_order_product():
    params = HeisenbergParams(
        n_qubits=3,
        steps=1,
        jx=0.7,
        jy=0.4,
        jz=0.9,
        dt=0.2,
        circuit_seed=0,
        instance=0,
    )
    u_circuit = Operator(build_heisenberg_circuit(params)).data

    expected = np.eye(2**params.n_qubits, dtype=complex)
    for q in range(params.n_qubits - 1):
        for pauli, coupling in ((X, params.jx), (Y, params.jy), (Z, params.jz)):
            pair = _operator_on_qubits(params.n_qubits, {q: pauli, q + 1: pauli})
            angle = coupling * params.dt
            rotation = np.cos(angle) * np.eye(2**params.n_qubits) - 1j * np.sin(angle) * pair
            expected = rotation @ expected

    assert np.allclose(u_circuit, expected, atol=1e-10)


def test_heisenberg_two_steps_match_squared_one_step_product():
    params = HeisenbergParams(
        n_qubits=2,
        steps=2,
        jx=0.71,
        jy=0.43,
        jz=0.89,
        dt=0.17,
        circuit_seed=0,
        instance=0,
    )
    u_circuit = Operator(build_heisenberg_circuit(params)).data

    one_step = np.eye(4, dtype=complex)
    for pauli, coupling in ((X, params.jx), (Y, params.jy), (Z, params.jz)):
        pair = np.kron(pauli, pauli)
        angle = coupling * params.dt
        rotation = np.cos(angle) * np.eye(4) - 1j * np.sin(angle) * pair
        one_step = rotation @ one_step
    expected = one_step @ one_step

    assert np.allclose(u_circuit, expected, atol=1e-10)


def test_heisenberg_sampling_and_build_are_deterministic():
    kwargs = {
        "n_qubits_choices": [3, 4, 6],
        "steps_choices": [1, 2],
        "dt": 0.15,
        "instance": 8,
        "circuit_seed": 23,
    }
    first = sample_heisenberg_params(np.random.default_rng(2718), **kwargs)
    second = sample_heisenberg_params(np.random.default_rng(2718), **kwargs)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert build_heisenberg_circuit(first) == build_heisenberg_circuit(first)
