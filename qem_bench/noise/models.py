"""Named noise families on the shared L1 to L4 benchmark grid.

The numeric values are monotone placeholders. They await pilot calibration against
the protocol's matched-severity definition, currently proposed as average gate
infidelity per layer at a reference depth. RZ is virtual and remains noiseless.

``SEVERITY_GRID`` remains the flat depolarizing-plus-readout grid for compatibility
with the walking skeleton. ``SEVERITY_GRIDS`` and ``NOISE_FAMILIES`` provide one
four-level grid for each named family.
"""

from __future__ import annotations

from operator import index

import numpy as np
from qiskit.circuit.library import RXGate, RZXGate, RZZGate
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
    phase_amplitude_damping_error,
    phase_damping_error,
)

BASIS_GATES = ["cx", "rz", "sx", "x"]
DEFAULT_NOISE_FAMILY = "depolarizing_readout"
DEFAULT_MIXED_QUBITS = 20

# The first three levels are frozen walking-skeleton values. Do not change them:
# existing deterministic dataset hashes depend on their physical channels.
DEPOLARIZING_READOUT_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {"p1": 0.001, "p2": 0.010, "p_ro": 0.010},
    "L2": {"p1": 0.003, "p2": 0.030, "p_ro": 0.030},
    "L3": {"p1": 0.010, "p2": 0.060, "p_ro": 0.060},
    "L4": {"p1": 0.020, "p2": 0.100, "p_ro": 0.100},
}

DEPHASING_READOUT_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {"lambda_1q": 0.002, "lambda_2q": 0.010, "p_ro": 0.010},
    "L2": {"lambda_1q": 0.006, "lambda_2q": 0.030, "p_ro": 0.030},
    "L3": {"lambda_1q": 0.020, "lambda_2q": 0.060, "p_ro": 0.060},
    "L4": {"lambda_1q": 0.040, "lambda_2q": 0.100, "p_ro": 0.100},
}

AMP_PHASE_DAMPING_READOUT_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {
        "amp_1q": 0.001,
        "phase_1q": 0.001,
        "amp_2q": 0.006,
        "phase_2q": 0.004,
        "p_ro": 0.010,
    },
    "L2": {
        "amp_1q": 0.003,
        "phase_1q": 0.004,
        "amp_2q": 0.018,
        "phase_2q": 0.012,
        "p_ro": 0.030,
    },
    "L3": {
        "amp_1q": 0.008,
        "phase_1q": 0.012,
        "amp_2q": 0.040,
        "phase_2q": 0.030,
        "p_ro": 0.060,
    },
    "L4": {
        "amp_1q": 0.015,
        "phase_1q": 0.025,
        "amp_2q": 0.070,
        "phase_2q": 0.055,
        "p_ro": 0.100,
    },
}

COHERENT_OVERROTATION_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {"epsilon_1q": 0.002, "epsilon_2q": 0.004, "p_ro": 0.005},
    "L2": {"epsilon_1q": 0.004, "epsilon_2q": 0.008, "p_ro": 0.015},
    "L3": {"epsilon_1q": 0.008, "epsilon_2q": 0.016, "p_ro": 0.030},
    "L4": {"epsilon_1q": 0.016, "epsilon_2q": 0.032, "p_ro": 0.050},
}

CORRELATED_CROSSTALK_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {
        "p1": 0.0005,
        "p2_ind": 0.002,
        "p2_corr": 0.002,
        "theta_zz": 0.002,
        "p_ro": 0.008,
    },
    "L2": {
        "p1": 0.0015,
        "p2_ind": 0.006,
        "p2_corr": 0.006,
        "theta_zz": 0.006,
        "p_ro": 0.020,
    },
    "L3": {
        "p1": 0.004,
        "p2_ind": 0.015,
        "p2_corr": 0.015,
        "theta_zz": 0.012,
        "p_ro": 0.040,
    },
    "L4": {
        "p1": 0.008,
        "p2_ind": 0.030,
        "p2_corr": 0.030,
        "theta_zz": 0.024,
        "p_ro": 0.070,
    },
}

# Mixed channels use the same-severity component grids above, multiplied by this
# level scale and by a deterministic per-location jitter described below.
MIXED_HETEROGENEOUS_SEVERITY_GRID: dict[str, dict[str, float]] = {
    "L1": {"component_scale": 0.8},
    "L2": {"component_scale": 0.9},
    "L3": {"component_scale": 1.0},
    "L4": {"component_scale": 1.1},
}

SEVERITY_GRIDS: dict[str, dict[str, dict[str, float]]] = {
    "depolarizing_readout": DEPOLARIZING_READOUT_SEVERITY_GRID,
    "dephasing_readout": DEPHASING_READOUT_SEVERITY_GRID,
    "amp_phase_damping_readout": AMP_PHASE_DAMPING_READOUT_SEVERITY_GRID,
    "coherent_overrotation": COHERENT_OVERROTATION_SEVERITY_GRID,
    "correlated_crosstalk": CORRELATED_CROSSTALK_SEVERITY_GRID,
    "mixed_heterogeneous": MIXED_HETEROGENEOUS_SEVERITY_GRID,
}

# Stable public registry of family names. Grid storage remains independently
# extensible through ``SEVERITY_GRIDS``.
NOISE_FAMILIES: tuple[str, ...] = tuple(SEVERITY_GRIDS)

# Backward-compatible public name used by datasets.generate and existing callers.
SEVERITY_GRID = DEPOLARIZING_READOUT_SEVERITY_GRID

_MIXTURE_COMPONENT_FAMILIES = (
    "depolarizing_readout",
    "dephasing_readout",
    "amp_phase_damping_readout",
    "coherent_overrotation",
    "correlated_crosstalk",
)
_MIXTURE_JITTERS = (0.8, 0.9, 1.0, 1.1, 1.2)


def _depolarizing_channels(
    cfg: dict[str, float],
) -> tuple[QuantumError, QuantumError, QuantumError]:
    one_qubit = depolarizing_error(cfg["p1"], 1)
    return one_qubit, one_qubit, depolarizing_error(cfg["p2"], 2)


def _dephasing_channels(cfg: dict[str, float]) -> tuple[QuantumError, QuantumError, QuantumError]:
    one_qubit = phase_damping_error(cfg["lambda_1q"])
    two_qubit_part = phase_damping_error(cfg["lambda_2q"])
    return one_qubit, one_qubit, two_qubit_part.tensor(two_qubit_part)


def _amp_phase_channels(cfg: dict[str, float]) -> tuple[QuantumError, QuantumError, QuantumError]:
    one_qubit = phase_amplitude_damping_error(cfg["amp_1q"], cfg["phase_1q"])
    two_qubit_part = phase_amplitude_damping_error(cfg["amp_2q"], cfg["phase_2q"])
    return one_qubit, one_qubit, two_qubit_part.tensor(two_qubit_part)


def _coherent_channels(cfg: dict[str, float]) -> tuple[QuantumError, QuantumError, QuantumError]:
    # SX and X overrotate around their native X axis. The CX error is a small RZX
    # rotation, the entangling generator commonly associated with a CX realization.
    sx_error = coherent_unitary_error(RXGate(np.pi * cfg["epsilon_1q"] / 2).to_matrix())
    x_error = coherent_unitary_error(RXGate(np.pi * cfg["epsilon_1q"]).to_matrix())
    cx_error = coherent_unitary_error(RZXGate(np.pi * cfg["epsilon_2q"] / 2).to_matrix())
    return sx_error, x_error, cx_error


def _correlated_channels(cfg: dict[str, float]) -> tuple[QuantumError, QuantumError, QuantumError]:
    one_qubit = depolarizing_error(cfg["p1"], 1)
    independent_part = depolarizing_error(cfg["p2_ind"], 1)
    independent_pair = independent_part.tensor(independent_part)
    correlated_pair = depolarizing_error(cfg["p2_corr"], 2)
    zz_coupling = coherent_unitary_error(RZZGate(cfg["theta_zz"]).to_matrix())
    combined = independent_pair.compose(correlated_pair).compose(zz_coupling)
    # Collapse the Pauli-mixture product into one Kraus instruction. This retains
    # the channel exactly and avoids hundreds of serialized branches per CX error.
    cx_error = QuantumError(combined.to_quantumchannel())
    return one_qubit, one_qubit, cx_error


def _channels_for_family(
    family: str, cfg: dict[str, float]
) -> tuple[QuantumError, QuantumError, QuantumError]:
    if family == "depolarizing_readout":
        return _depolarizing_channels(cfg)
    if family == "dephasing_readout":
        return _dephasing_channels(cfg)
    if family == "amp_phase_damping_readout":
        return _amp_phase_channels(cfg)
    if family == "coherent_overrotation":
        return _coherent_channels(cfg)
    if family == "correlated_crosstalk":
        return _correlated_channels(cfg)
    raise ValueError(f"family {family!r} does not define a homogeneous channel")


def _readout_error(p: float) -> ReadoutError:
    return ReadoutError([[1 - p, p], [p, 1 - p]])


def _homogeneous_noise_model(family: str, cfg: dict[str, float]) -> NoiseModel:
    sx_error, x_error, cx_error = _channels_for_family(family, cfg)
    model = NoiseModel(basis_gates=BASIS_GATES)
    if sx_error is x_error:
        model.add_all_qubit_quantum_error(sx_error, ["sx", "x"])
    else:
        model.add_all_qubit_quantum_error(sx_error, ["sx"])
        model.add_all_qubit_quantum_error(x_error, ["x"])
    model.add_all_qubit_quantum_error(cx_error, ["cx"])
    model.add_all_qubit_readout_error(_readout_error(cfg["p_ro"]))
    return model


def _validated_int(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _draw_mixed_profile(seed: int, spawn_key: tuple[int, ...]) -> tuple[str, float]:
    state = np.random.SeedSequence(seed, spawn_key=spawn_key).generate_state(2, dtype=np.uint32)
    family = _MIXTURE_COMPONENT_FAMILIES[int(state[0]) % len(_MIXTURE_COMPONENT_FAMILIES)]
    jitter = _MIXTURE_JITTERS[int(state[1]) % len(_MIXTURE_JITTERS)]
    return family, jitter


def _scaled_component(
    family: str,
    severity: str,
    factor: float,
    cache: dict[tuple[str, float], tuple[QuantumError, QuantumError, QuantumError, float]],
) -> tuple[QuantumError, QuantumError, QuantumError, float]:
    key = (family, factor)
    if key not in cache:
        cfg = {name: value * factor for name, value in SEVERITY_GRIDS[family][severity].items()}
        cache[key] = (*_channels_for_family(family, cfg), cfg["p_ro"])
    return cache[key]


def _mixed_components(
    severity: str, seed: int, n_qubits: int
) -> tuple[
    list[tuple[int, QuantumError, QuantumError, float]],
    list[tuple[int, int, QuantumError]],
]:
    """Return deterministic per-qubit and per-directed-pair mixed components.

    Qubit ``q`` uses ``SeedSequence(seed, spawn_key=(0, q))`` to select one of the
    five non-mixed families and one multiplier from (0.8, 0.9, 1.0, 1.1, 1.2).
    Its SX, X, and readout channels come from that profile. Directed CX pair
    ``(control, target)`` uses spawn key ``(1, control, target)`` in the same way.
    The selected family's parameters at the requested severity are multiplied by
    both that jitter and the mixed grid's ``component_scale``. Independent spawn
    keys make the construction insensitive to traversal order.
    """
    level_scale = MIXED_HETEROGENEOUS_SEVERITY_GRID[severity]["component_scale"]
    cache: dict[
        tuple[str, float], tuple[QuantumError, QuantumError, QuantumError, float]
    ] = {}
    one_qubit_components = []
    for qubit in range(n_qubits):
        family, jitter = _draw_mixed_profile(seed, (0, qubit))
        sx_error, x_error, _, p_ro = _scaled_component(
            family, severity, level_scale * jitter, cache
        )
        one_qubit_components.append((qubit, sx_error, x_error, p_ro))

    two_qubit_components = []
    for control in range(n_qubits):
        for target in range(n_qubits):
            if control == target:
                continue
            family, jitter = _draw_mixed_profile(seed, (1, control, target))
            _, _, cx_error, _ = _scaled_component(
                family, severity, level_scale * jitter, cache
            )
            two_qubit_components.append((control, target, cx_error))
    return one_qubit_components, two_qubit_components


def _mixed_noise_model(severity: str, seed: int, n_qubits: int) -> NoiseModel:
    one_qubit, two_qubit = _mixed_components(severity, seed, n_qubits)
    model = NoiseModel(basis_gates=BASIS_GATES)
    for qubit, sx_error, x_error, p_ro in one_qubit:
        model.add_quantum_error(sx_error, ["sx"], [qubit])
        model.add_quantum_error(x_error, ["x"], [qubit])
        model.add_readout_error(_readout_error(p_ro), [qubit])
    for control, target, cx_error in two_qubit:
        model.add_quantum_error(cx_error, ["cx"], [control, target])
    return model


def _resolve_family_severity(family: str, severity: str | None) -> tuple[str, str]:
    if severity is None:
        if family in SEVERITY_GRID:
            return DEFAULT_NOISE_FAMILY, family
        raise TypeError("severity is required when family is given explicitly")
    if family not in NOISE_FAMILIES:
        names = ", ".join(NOISE_FAMILIES)
        raise ValueError(f"unknown noise family {family!r}; expected one of: {names}")
    if severity not in SEVERITY_GRIDS[family]:
        levels = ", ".join(SEVERITY_GRIDS[family])
        raise ValueError(f"unknown severity {severity!r} for {family}; expected one of: {levels}")
    return family, severity


def build_noise_model(
    family: str = DEFAULT_NOISE_FAMILY,
    severity: str | None = None,
    *,
    seed: int | None = None,
    n_qubits: int = DEFAULT_MIXED_QUBITS,
) -> NoiseModel:
    """Build one named noise model.

    ``build_noise_model("L1")`` and ``build_noise_model(severity="L1")`` retain the
    original depolarizing-plus-readout behavior. The mixed family requires a caller
    supplied seed. Its ``n_qubits`` controls the qubits and directed CX pairs that
    receive local errors; the default covers circuits with up to 20 qubits.
    """
    family, severity = _resolve_family_severity(family, severity)
    if family == "mixed_heterogeneous":
        if seed is None:
            raise ValueError("mixed_heterogeneous requires a caller-supplied seed")
        checked_seed = _validated_int(seed, "seed", minimum=0)
        checked_qubits = _validated_int(n_qubits, "n_qubits", minimum=1)
        return _mixed_noise_model(severity, checked_seed, checked_qubits)
    return _homogeneous_noise_model(family, SEVERITY_GRIDS[family][severity])


def _channel_infidelity(error: QuantumError) -> float:
    fidelity = float(average_gate_fidelity(error.to_quantumchannel()))
    return max(0.0, 1.0 - fidelity)


def average_gate_infidelities(
    family: str = DEFAULT_NOISE_FAMILY,
    severity: str | None = None,
    *,
    seed: int | None = None,
    n_qubits: int = DEFAULT_MIXED_QUBITS,
) -> dict[str, float]:
    """Report mean one-qubit and two-qubit channel infidelities.

    The one-qubit value weights SX and X equally. For the mixed family it also
    averages uniformly over configured qubits, while the two-qubit value averages
    over configured directed CX pairs. Readout error is not a quantum gate channel
    and is therefore excluded. Coherent-family values are exact channel metrics.
    """
    family, severity = _resolve_family_severity(family, severity)
    if family == "mixed_heterogeneous":
        if seed is None:
            raise ValueError("mixed_heterogeneous requires a caller-supplied seed")
        checked_seed = _validated_int(seed, "seed", minimum=0)
        checked_qubits = _validated_int(n_qubits, "n_qubits", minimum=2)
        one_qubit, two_qubit = _mixed_components(severity, checked_seed, checked_qubits)
        one_values = [
            _channel_infidelity(error)
            for _, sx_error, x_error, _ in one_qubit
            for error in (sx_error, x_error)
        ]
        two_values = [_channel_infidelity(error) for _, _, error in two_qubit]
    else:
        sx_error, x_error, cx_error = _channels_for_family(
            family, SEVERITY_GRIDS[family][severity]
        )
        one_values = [_channel_infidelity(sx_error), _channel_infidelity(x_error)]
        two_values = [_channel_infidelity(cx_error)]
    return {
        "1q": float(sum(one_values) / len(one_values)),
        "2q": float(sum(two_values) / len(two_values)),
    }
