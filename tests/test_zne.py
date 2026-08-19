"""Digital ZNE protocol tests, including the required Mitiq cross-check."""

from __future__ import annotations

import math
import os

import cirq
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

import qem_bench.baselines.zne as zne_module
import qem_bench.sampling as sampling_module
from qem_bench.baselines.zne import (
    RICHARDSON_WEIGHTS,
    SCALE_FACTORS,
    ZNEMitigator,
    extrapolate_zero_noise,
    fold_for_execution,
)
from qem_bench.circuits.tfi import TFIParams, build_tfi_circuit
from qem_bench.labels.statevector import ideal_expectation
from qem_bench.noise.models import BASIS_GATES
from qem_bench.observables import z_expectation_from_counts
from qem_bench.sampling import sample_counts


def _params(steps: int = 2) -> TFIParams:
    return TFIParams(
        n_qubits=3,
        steps=steps,
        j=0.63,
        h=0.81,
        dt=0.2,
        circuit_seed=17,
        instance=4,
    )


def _item(
    raw: float,
    shots: int,
    *,
    item_id: str = "tfi-zne-z-mid",
    measurement_group: str = "tfi-zne-g0",
) -> dict:
    params = _params()
    return {
        "item_id": item_id,
        "measurement_group": measurement_group,
        "family": "tfi",
        "n_qubits": params.n_qubits,
        "steps": params.steps,
        "j": params.j,
        "h": params.h,
        "dt": params.dt,
        "circuit_seed": params.circuit_seed,
        "instance": params.instance,
        "pauli_label": "IZI",
        "shots": shots,
        "severity": "L2",
        "noise_family": "depolarizing_readout",
        "noisy_expectation": raw,
    }


def _transpiled_cx_count(circuit: QuantumCircuit, seed: int = 19) -> int:
    measured = circuit.copy()
    measured.measure_all()
    executed = transpile(
        measured,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=seed,
    )
    return int(executed.count_ops().get("cx", 0))


def test_reuses_scale_one_and_charges_two_extra_executions(monkeypatch):
    """Scale one is the stored raw draw; only scales three and five execute."""
    shots = 100
    item = _item(raw=0.6, shots=shots)
    returned_expectations = iter((0.2, -0.4))
    calls: list[tuple[int, int, int]] = []

    def fake_sample_counts(
        circuit,
        severity,
        shots,
        sampler_seed,
        transpile_seed,
        *,
        noise_family="depolarizing_readout",
    ):
        del severity, noise_family
        expectation = next(returned_expectations)
        plus = int(round(shots * (1.0 + expectation) / 2.0))
        calls.append((circuit.count_ops().get("cx", 0), sampler_seed, transpile_seed))
        return {"000": plus, "010": shots - plus}, {}

    monkeypatch.setattr(zne_module, "sample_counts", fake_sample_counts)
    model = ZNEMitigator().fit([{"ideal_expectation": 999.0}])
    predictions, extra = model.predict([item], seed_stream=np.random.SeedSequence(123))

    assert len(calls) == 2
    assert calls[0][0] < calls[1][0]
    assert predictions[0] == pytest.approx(
        extrapolate_zero_noise([item["noisy_expectation"], 0.2, -0.4])
    )
    assert extra.tolist() == [2 * shots]


def test_folded_executions_are_shared_by_measurement_group(monkeypatch):
    shots = 100
    first = _item(raw=0.6, shots=shots, item_id="tfi-zne-z-mid")
    sibling = _item(raw=-0.1, shots=shots, item_id="tfi-zne-z-edge")
    sibling["pauli_label"] = "IIZ"
    returned_expectations = iter((0.2, -0.4))
    calls = []

    def fake_sample_counts(
        circuit,
        severity,
        shots,
        sampler_seed,
        transpile_seed,
        *,
        noise_family="depolarizing_readout",
    ):
        del circuit, severity, noise_family
        expectation = next(returned_expectations)
        plus = int(round(shots * (1.0 + expectation) / 2.0))
        calls.append((sampler_seed, transpile_seed))
        return {"000": plus, "010": shots - plus}, {}

    monkeypatch.setattr(zne_module, "sample_counts", fake_sample_counts)
    predictions, extra = ZNEMitigator().fit([]).predict(
        [first, sibling], seed_stream=np.random.SeedSequence(123)
    )

    assert len(calls) == 2
    assert predictions[0] == pytest.approx(extrapolate_zero_noise([0.6, 0.2, -0.4]))
    assert predictions[1] == pytest.approx(extrapolate_zero_noise([-0.1, 1.0, 1.0]))
    assert extra.tolist() == [2 * shots]


def test_noiseless_folds_and_richardson_agree_with_the_exact_value(monkeypatch):
    """At 60,000 shots, every scale and ZNE must be within six sampling sigmas."""
    shots = 60_000
    logical = build_tfi_circuit(_params())
    support = (1,)
    exact = ideal_expectation(logical, "IZI")

    # Keep the production sampling path but remove its noise model for this test.
    monkeypatch.setattr(sampling_module, "build_noise_model", lambda *args, **kwargs: None)
    counts, _ = sample_counts(
        logical,
        severity="L2",
        shots=shots,
        sampler_seed=100,
        transpile_seed=101,
    )
    raw, _ = z_expectation_from_counts(counts, support, shots)
    scale_values = [raw]
    real_sample_counts = sampling_module.sample_counts

    def record_sample_counts(
        circuit,
        severity,
        shots,
        sampler_seed,
        transpile_seed,
        *,
        noise_family="depolarizing_readout",
    ):
        result = real_sample_counts(
            circuit,
            severity,
            shots,
            sampler_seed,
            transpile_seed,
            noise_family=noise_family,
        )
        estimate, _ = z_expectation_from_counts(result[0], support, shots)
        scale_values.append(estimate)
        return result

    monkeypatch.setattr(zne_module, "sample_counts", record_sample_counts)
    prediction, extra = ZNEMitigator().fit([]).predict(
        [_item(raw, shots)], seed_stream=np.random.SeedSequence(700, spawn_key=(2,))
    )

    assert len(scale_values) == 3
    sigma = math.sqrt(max(0.0, 1.0 - exact * exact) / shots)
    for value in scale_values:
        assert value == pytest.approx(exact, abs=6.0 * sigma + 1.0 / shots)
    for left in scale_values:
        for right in scale_values:
            assert left == pytest.approx(right, abs=6.0 * math.sqrt(2.0) * sigma + 2.0 / shots)
    richardson_sigma = sigma * math.sqrt(sum(weight * weight for weight in RICHARDSON_WEIGHTS))
    assert prediction[0] == pytest.approx(exact, abs=6.0 * richardson_sigma + 1.0 / shots)
    assert extra.tolist() == [2 * shots]


class _MitiqCircuit:
    """Small wrapper that avoids Mitiq 1.0's optional QASM parser dependency."""

    __module__ = "qem_bench_zne_crosscheck"

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit


def _qiskit_to_cirq(wrapped: _MitiqCircuit) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(wrapped.circuit.num_qubits)
    moments = []
    for instruction in wrapped.circuit.data:
        operation = instruction.operation
        if operation.name == "barrier":
            continue
        indices = [wrapped.circuit.find_bit(bit).index for bit in instruction.qubits]
        if operation.name == "rz":
            converted = cirq.rz(float(operation.params[0]))(qubits[indices[0]])
        elif operation.name == "sx":
            converted = cirq.XPowGate(exponent=0.5)(qubits[indices[0]])
        elif operation.name == "x":
            converted = cirq.X(qubits[indices[0]])
        elif operation.name == "cx":
            converted = cirq.CNOT(qubits[indices[0]], qubits[indices[1]])
        else:
            raise AssertionError(f"unexpected basis operation {operation.name}")
        # One operation per moment preserves Qiskit's sequential instruction order.
        moments.append(cirq.Moment([converted]))
    return cirq.Circuit(moments)


def _cirq_to_qiskit(circuit: cirq.Circuit) -> _MitiqCircuit:
    qiskit_circuit = QuantumCircuit(len(circuit.all_qubits()))
    for operation in circuit.all_operations():
        indices = [int(qubit.x) for qubit in operation.qubits]
        gate = operation.gate
        if isinstance(gate, cirq.CXPowGate):
            assert abs(abs(float(gate.exponent)) - 1.0) < 1e-12
            qiskit_circuit.cx(indices[0], indices[1])
        elif isinstance(gate, cirq.ZPowGate):
            qiskit_circuit.rz(math.pi * float(gate.exponent), indices[0])
        elif isinstance(gate, cirq.XPowGate):
            exponent = ((float(gate.exponent) + 1.0) % 2.0) - 1.0
            if abs(abs(exponent) - 1.0) < 1e-12:
                qiskit_circuit.x(indices[0])
            elif abs(exponent - 0.5) < 1e-12:
                qiskit_circuit.sx(indices[0])
            elif abs(exponent + 0.5) < 1e-12:
                qiskit_circuit.sxdg(indices[0])
            else:
                qiskit_circuit.rx(math.pi * exponent, indices[0])
        else:
            raise AssertionError(f"unexpected Mitiq gate {gate!r}")
        # Mitiq's fold has no boundary markers. Per-operation barriers ensure
        # Qiskit's optimization pass cannot erase the reference identity folds.
        qiskit_circuit.barrier()
    return _MitiqCircuit(qiskit_circuit)


@pytest.mark.skipif(
    os.environ.get("QEM_BENCH_SKIP_MITIQ_REFERENCE") == "1",
    reason="Mitiq reference runs in its dedicated CI job",
)
def test_mitiq_global_folding_and_richardson_crosscheck():
    """Local B1 and installed Mitiq agree at 100,000 fixed-seed shots."""
    from mitiq.interface import register_mitiq_converters
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import RichardsonFactory
    from mitiq.zne.scaling import fold_global as mitiq_fold_global

    register_mitiq_converters(
        _MitiqCircuit.__module__,
        convert_to_function=_cirq_to_qiskit,
        convert_from_function=_qiskit_to_cirq,
    )

    shots = 100_000
    logical = build_tfi_circuit(_params())
    local_circuits = [fold_for_execution(logical, scale, 313) for scale in SCALE_FACTORS]
    base_cx = int(local_circuits[0].count_ops().get("cx", 0))
    seeds = {1: 901, 3: 903, 5: 905}

    def executor(circuit: _MitiqCircuit) -> float:
        cx_count = int(circuit.circuit.count_ops().get("cx", 0))
        scale = int(round(cx_count / base_cx))
        counts, _ = sample_counts(
            circuit.circuit,
            severity="L2",
            shots=shots,
            sampler_seed=seeds[scale],
            transpile_seed=313,
        )
        return z_expectation_from_counts(counts, (1,), shots)[0]

    # Mitiq 1.0 inspects the concrete return annotation and does not resolve
    # postponed annotations from ``from __future__ import annotations``.
    executor.__annotations__["return"] = float

    local_values = [executor(_MitiqCircuit(circuit)) for circuit in local_circuits]
    local_zne = extrapolate_zero_noise(local_values)

    factory = RichardsonFactory(scale_factors=list(SCALE_FACTORS))
    mitiq_zne = execute_with_zne(
        _MitiqCircuit(local_circuits[0]),
        executor,
        factory=factory,
        scale_noise=mitiq_fold_global,
    )

    # Both paths use the same Aer executor, L2 model, scale factors, shots, and
    # factor-indexed seeds. The tolerance covers independent compiled forms and
    # Richardson's amplified sampling error.
    assert factory.get_expectation_values() == pytest.approx(local_values, abs=0.015)
    assert mitiq_zne == pytest.approx(local_zne, abs=0.04)


def test_scale_three_preserves_two_qubit_gate_scaling():
    """After the chosen order, scale three is within five percent of 3x CX count."""
    logical = build_tfi_circuit(_params(steps=3))
    base = fold_for_execution(logical, 1, transpile_seed=19)
    folded = fold_for_execution(logical, 3, transpile_seed=19)
    base_cx = _transpiled_cx_count(base)
    folded_cx = _transpiled_cx_count(folded)

    assert base_cx > 0
    assert folded_cx / base_cx == pytest.approx(3.0, rel=0.05)


def test_fixed_seed_stream_is_deterministic():
    shots = 2_048
    logical = build_tfi_circuit(_params(steps=1))
    counts, _ = sample_counts(logical, "L2", shots, sampler_seed=80, transpile_seed=81)
    raw, _ = z_expectation_from_counts(counts, (1,), shots)
    item = _item(raw, shots, item_id="fixed-seed-item")
    item["steps"] = 1
    model = ZNEMitigator().fit([])
    root = np.random.SeedSequence(20260719, spawn_key=(2,))

    first = model.predict([item], seed_stream=root)
    second = model.predict([item], seed_stream=root)
    fresh_root = np.random.SeedSequence(20260719, spawn_key=(2,))
    third = model.predict([item], seed_stream=fresh_root)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[0], third[0])
    assert np.array_equal(first[1], second[1])


def test_extrapolator_contracts():
    # Richardson exactly recovers the constant coefficient of a quadratic.
    values = [2.0 + 0.3 * scale - 0.04 * scale**2 for scale in SCALE_FACTORS]
    assert extrapolate_zero_noise(values) == pytest.approx(2.0)
    assert extrapolate_zero_noise([0.7, 0.7, 0.7], "linear") == pytest.approx(0.7)
    intercept, slope = 0.43, -0.07
    sloped = [intercept + slope * scale for scale in SCALE_FACTORS]
    assert extrapolate_zero_noise(sloped, "linear") == pytest.approx(intercept)
    with pytest.raises(ValueError, match="extrapolator"):
        ZNEMitigator(extrapolator="chosen-on-test").predict(
            [_item(0.5, 10)], seed_stream=1
        )
