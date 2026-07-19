"""Six-family noise-grid contracts and channel-level physics checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from qiskit import transpile
from qiskit.circuit.library import RXGate, RZXGate, RZZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Kraus, Operator, SuperOp
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
    phase_amplitude_damping_error,
    phase_damping_error,
)

from qem_bench.circuits.tfi import TFIParams, build_tfi_circuit
from qem_bench.noise.models import (
    BASIS_GATES,
    CORRELATED_CROSSTALK_SEVERITY_GRID,
    COHERENT_OVERROTATION_SEVERITY_GRID,
    DEPHASING_READOUT_SEVERITY_GRID,
    AMP_PHASE_DAMPING_READOUT_SEVERITY_GRID,
    NOISE_FAMILIES,
    SEVERITY_GRIDS,
    average_gate_infidelities,
    build_noise_model,
)

FAMILIES = (
    "depolarizing_readout",
    "dephasing_readout",
    "amp_phase_damping_readout",
    "coherent_overrotation",
    "correlated_crosstalk",
    "mixed_heterogeneous",
)
LEVELS = ("L1", "L2", "L3", "L4")
MIXED_KWARGS = {"seed": 20260719, "n_qubits": 3}


def _build(family: str, severity: str) -> NoiseModel:
    kwargs = MIXED_KWARGS if family == "mixed_heterogeneous" else {}
    return build_noise_model(family, severity, **kwargs)


def _quantum_error(model: NoiseModel, operation: str) -> QuantumError:
    matches = [
        error
        for error in model.to_dict().get("errors", [])
        if error["type"] == "qerror" and operation in error.get("operations", [])
    ]
    assert len(matches) == 1
    return QuantumError.from_dict(matches[0])


def _without_generated_ids(model_dict: dict) -> dict:
    """Remove Qiskit's random QuantumError UUIDs, which are not channel data."""
    result = copy.deepcopy(model_dict)
    for error in result.get("errors", []):
        error.pop("id", None)
    return result


def _legacy_model(severity: str) -> NoiseModel:
    """Independent copy of the pre-grid walking-skeleton implementation."""
    frozen = {
        "L1": {"p1": 0.001, "p2": 0.010, "p_ro": 0.010},
        "L2": {"p1": 0.003, "p2": 0.030, "p_ro": 0.030},
        "L3": {"p1": 0.010, "p2": 0.060, "p_ro": 0.060},
    }
    cfg = frozen[severity]
    model = NoiseModel(basis_gates=BASIS_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(cfg["p1"], 1), ["sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(cfg["p2"], 2), ["cx"])
    p_ro = cfg["p_ro"]
    model.add_all_qubit_readout_error(
        ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    )
    return model


def test_family_registry_has_four_monotone_levels():
    assert NOISE_FAMILIES == FAMILIES
    assert tuple(SEVERITY_GRIDS) == FAMILIES
    for family, grid in SEVERITY_GRIDS.items():
        assert tuple(grid) == LEVELS, family
        keys = set(grid["L1"])
        assert all(set(grid[level]) == keys for level in LEVELS)
        for key in keys:
            values = [grid[level][key] for level in LEVELS]
            assert values == sorted(values), (family, key, values)
            assert values[-1] > values[0], (family, key, values)


@pytest.fixture(scope="module")
def transpiled_operations() -> set[str]:
    params = TFIParams(
        n_qubits=3,
        steps=2,
        j=0.5,
        h=0.8,
        dt=0.2,
        circuit_seed=1,
        instance=0,
    )
    circuit = build_tfi_circuit(params)
    circuit.measure_all()
    transpiled = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=1,
    )
    return set(transpiled.count_ops())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("severity", LEVELS)
def test_every_grid_cell_covers_transpiled_operations(
    family: str, severity: str, transpiled_operations: set[str]
):
    model_dict = _build(family, severity).to_dict()
    covered = {
        operation
        for error in model_dict.get("errors", [])
        for operation in error.get("operations", [])
    }
    missing = transpiled_operations - {"rz", "barrier"} - covered
    assert not missing, f"{family} {severity} leaves {sorted(missing)} noiseless"


@pytest.mark.parametrize("family", FAMILIES)
def test_average_gate_infidelity_is_monotone(family: str):
    kwargs = MIXED_KWARGS if family == "mixed_heterogeneous" else {}
    values = [average_gate_infidelities(family, level, **kwargs) for level in LEVELS]
    for channel in ("1q", "2q"):
        sequence = [value[channel] for value in values]
        assert all(left <= right + 1e-14 for left, right in zip(sequence, sequence[1:])), (
            family,
            channel,
            sequence,
        )
        assert sequence[-1] > sequence[0], (family, channel, sequence)


@pytest.mark.parametrize("severity", LEVELS)
def test_coherent_overrotation_channels_are_unitary_and_non_clifford(severity: str):
    model = build_noise_model("coherent_overrotation", severity)
    quantum_errors = [
        error for error in model.to_dict()["errors"] if error["type"] == "qerror"
    ]
    assert {operation for error in quantum_errors for operation in error["operations"]} == {
        "sx",
        "x",
        "cx",
    }
    for error_dict in quantum_errors:
        channel = QuantumError.from_dict(error_dict).to_quantumchannel()
        kraus = Kraus(channel).data
        assert len(kraus) == 1
        with pytest.raises(QiskitError, match="Non-Clifford"):
            Clifford.from_operator(Operator(kraus[0]))


def test_damping_and_correlated_channels_match_their_declared_oracles():
    coherent_cfg = COHERENT_OVERROTATION_SEVERITY_GRID["L2"]
    coherent_model = build_noise_model("coherent_overrotation", "L2")
    coherent_oracles = {
        "sx": RXGate(np.pi * coherent_cfg["epsilon_1q"] / 2),
        "x": RXGate(np.pi * coherent_cfg["epsilon_1q"]),
        "cx": RZXGate(np.pi * coherent_cfg["epsilon_2q"] / 2),
    }
    for operation, expected in coherent_oracles.items():
        assert np.allclose(
            SuperOp(_quantum_error(coherent_model, operation)).data,
            SuperOp(expected).data,
            atol=1e-12,
        )

    dephasing_cfg = DEPHASING_READOUT_SEVERITY_GRID["L2"]
    dephasing_model = build_noise_model("dephasing_readout", "L2")
    expected_phase_1q = phase_damping_error(dephasing_cfg["lambda_1q"])
    expected_phase_part = phase_damping_error(dephasing_cfg["lambda_2q"])
    assert np.allclose(
        SuperOp(_quantum_error(dephasing_model, "sx")).data,
        SuperOp(expected_phase_1q).data,
    )
    assert np.allclose(
        SuperOp(_quantum_error(dephasing_model, "cx")).data,
        SuperOp(expected_phase_part.tensor(expected_phase_part)).data,
    )

    damping_cfg = AMP_PHASE_DAMPING_READOUT_SEVERITY_GRID["L2"]
    damping_model = build_noise_model("amp_phase_damping_readout", "L2")
    expected_damping_1q = phase_amplitude_damping_error(
        damping_cfg["amp_1q"], damping_cfg["phase_1q"]
    )
    expected_damping_part = phase_amplitude_damping_error(
        damping_cfg["amp_2q"], damping_cfg["phase_2q"]
    )
    assert np.allclose(
        SuperOp(_quantum_error(damping_model, "sx")).data,
        SuperOp(expected_damping_1q).data,
    )
    assert np.allclose(
        SuperOp(_quantum_error(damping_model, "cx")).data,
        SuperOp(expected_damping_part.tensor(expected_damping_part)).data,
    )

    correlated_cfg = CORRELATED_CROSSTALK_SEVERITY_GRID["L2"]
    correlated_model = build_noise_model("correlated_crosstalk", "L2")
    independent = depolarizing_error(correlated_cfg["p2_ind"], 1)
    expected_correlated = (
        independent.tensor(independent)
        .compose(depolarizing_error(correlated_cfg["p2_corr"], 2))
        .compose(
            coherent_unitary_error(RZZGate(correlated_cfg["theta_zz"]).to_matrix())
        )
    )
    assert np.allclose(
        SuperOp(_quantum_error(correlated_model, "cx")).data,
        SuperOp(expected_correlated).data,
        atol=1e-12,
    )


def test_mixed_heterogeneous_is_seeded_and_deterministic():
    first = build_noise_model("mixed_heterogeneous", "L3", seed=71, n_qubits=3)
    repeat = build_noise_model("mixed_heterogeneous", "L3", seed=71, n_qubits=3)
    changed = build_noise_model("mixed_heterogeneous", "L3", seed=72, n_qubits=3)
    assert first == repeat
    assert first != changed
    with pytest.raises(ValueError, match="caller-supplied seed"):
        build_noise_model("mixed_heterogeneous", "L3", n_qubits=3)


@pytest.mark.parametrize("severity", ("L1", "L2", "L3"))
def test_depolarizing_legacy_calls_preserve_the_frozen_models(severity: str):
    expected = _without_generated_ids(_legacy_model(severity).to_dict())
    assert _without_generated_ids(build_noise_model(severity).to_dict()) == expected
    assert _without_generated_ids(build_noise_model(severity=severity).to_dict()) == expected
    assert (
        _without_generated_ids(
            build_noise_model("depolarizing_readout", severity).to_dict()
        )
        == expected
    )
