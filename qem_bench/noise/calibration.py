"""Severity-calibration machinery without changing the shipped noise registry.

The primary definition is a gate-count-weighted average gate infidelity on a
named reference shape. Reference-shape and weighting names are mandatory inputs
to every scalar calculation. Solved grids returned here are proposals; callers
must not mutate :mod:`qem_bench.noise.models` with them implicitly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from math import pi, sin, sqrt
from typing import Mapping

from qiskit_aer.noise import NoiseModel

from qem_bench.noise.models import (
    BASIS_GATES,
    SEVERITY_GRIDS,
    _MIXTURE_COMPONENT_FAMILIES,
    _MIXTURE_JITTERS,
    _channels_for_family,
    _draw_mixed_profile,
    _homogeneous_noise_model,
    _readout_error,
)

ANCHOR_FAMILY = "depolarizing_readout"
LEVELS = ("L1", "L2", "L3", "L4")
MATCHING_RELATIVE_TOLERANCE = 0.01
SOLVER_RELATIVE_TOLERANCE = 1e-10
PROPOSAL_PATH = (
    files("qem_bench.noise")
    .joinpath("data")
    .joinpath("severity-calibration.proposal.json")
)
# Stable name for diagnostics. Rendering PROPOSAL_PATH itself yields the
# absolute install path, which differs per machine and leaks a user name
# into CI logs and assertion output.
PROPOSAL_RESOURCE = "qem_bench/noise/data/severity-calibration.proposal.json"

PRIMARY_REFERENCE_SHAPE = "six_qubit_random_clifford_brickwork_v1_seed1"
PRIMARY_GATE_COUNT_WEIGHTING = "error_bearing_transpiled_occurrences_v1"
MIXED_REFERENCE_SEED = 20260719
MIXED_REFERENCE_QUBITS = 6
MIXED_REFERENCE_PROFILE = "uniform_component_jitter_expectation_v1"


@dataclass(frozen=True)
class ReferenceShape:
    """Pinned operation counts for one logical layer of a named circuit shape."""

    name: str
    gate_counts: Mapping[str, int]
    logical_layers: int
    description: str


@dataclass(frozen=True)
class GateCountWeighting:
    """Per-operation multipliers used to form a normalized layer aggregate."""

    name: str
    operation_weights: Mapping[str, float]
    description: str


@dataclass(frozen=True)
class FamilyCalibration:
    """One family's proposed grid and its mismatch from the anchor."""

    family: str
    shipped_grid: dict[str, dict[str, float]]
    solved_grid: dict[str, dict[str, float]]
    relative_residuals: dict[str, dict[str, float]]
    matched: bool
    reason: str | None = None


REFERENCE_SHAPES: dict[str, ReferenceShape] = {
    PRIMARY_REFERENCE_SHAPE: ReferenceShape(
        name=PRIMARY_REFERENCE_SHAPE,
        gate_counts={"cx": 3, "rz": 12, "sx": 6, "x": 1},
        logical_layers=1,
        description=(
            "Six-qubit random-Clifford brickwork layer, circuit seed 1, transpiled "
            "with basis [cx, rz, sx, x], optimization level 1, and transpiler seed 1."
        ),
    )
}

GATE_COUNT_WEIGHTINGS: dict[str, GateCountWeighting] = {
    PRIMARY_GATE_COUNT_WEIGHTING: GateCountWeighting(
        name=PRIMARY_GATE_COUNT_WEIGHTING,
        operation_weights={"cx": 1.0, "rz": 0.0, "sx": 1.0, "x": 1.0},
        description=(
            "Normalize physical-channel infidelity by the number of error-bearing "
            "transpiled operations; virtual RZ contributes zero weight."
        ),
    )
}

_PARAMETER_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "dephasing_readout": (("lambda_1q",), ("lambda_2q",)),
    "amp_phase_damping_readout": (
        ("amp_1q", "phase_1q"),
        ("amp_2q", "phase_2q"),
    ),
    "coherent_overrotation": (("epsilon_1q",), ("epsilon_2q",)),
    "correlated_crosstalk": (
        ("p1",),
        ("p2_ind", "p2_corr", "theta_zz"),
    ),
}


def _named_reference_shape(name: str | None) -> ReferenceShape:
    if not isinstance(name, str) or not name:
        raise ValueError("reference_shape must be a nonempty registered name")
    try:
        return REFERENCE_SHAPES[name]
    except KeyError as exc:
        names = ", ".join(REFERENCE_SHAPES)
        raise ValueError(f"unknown reference_shape {name!r}; expected one of: {names}") from exc


def _named_weighting(name: str | None) -> GateCountWeighting:
    if not isinstance(name, str) or not name:
        raise ValueError("gate_count_weighting must be a nonempty registered name")
    try:
        return GATE_COUNT_WEIGHTINGS[name]
    except KeyError as exc:
        names = ", ".join(GATE_COUNT_WEIGHTINGS)
        raise ValueError(
            f"unknown gate_count_weighting {name!r}; expected one of: {names}"
        ) from exc


def _aggregate_selected_operation_infidelities(
    infidelities: Mapping[str, float],
    operations: tuple[str, ...],
    *,
    reference_shape: str,
    gate_count_weighting: str,
) -> float:
    shape = _named_reference_shape(reference_shape)
    weighting = _named_weighting(gate_count_weighting)
    if set(shape.gate_counts) != set(BASIS_GATES):
        raise ValueError("reference shape must declare every transpiled basis operation")
    if set(weighting.operation_weights) != set(BASIS_GATES):
        raise ValueError("gate-count weighting must declare every transpiled basis operation")
    unknown = set(operations) - set(BASIS_GATES)
    if unknown:
        raise ValueError(f"unknown basis operations: {sorted(unknown)}")
    missing = set(operations) - set(infidelities)
    if missing:
        raise ValueError(f"infidelities are missing operations: {sorted(missing)}")
    denominator = sum(
        shape.gate_counts[gate] * weighting.operation_weights[gate]
        for gate in operations
    )
    if denominator <= 0:
        raise ValueError("gate-count weighting has no positive reference-shape mass")
    numerator = sum(
        shape.gate_counts[gate]
        * weighting.operation_weights[gate]
        * float(infidelities[gate])
        for gate in operations
    )
    return float(numerator / denominator / shape.logical_layers)


def aggregate_operation_infidelities(
    infidelities: Mapping[str, float],
    *,
    reference_shape: str,
    gate_count_weighting: str,
) -> float:
    """Return the named gate-count-weighted scalar for one logical layer."""
    missing = set(BASIS_GATES) - set(infidelities)
    if missing:
        raise ValueError(f"infidelities are missing operations: {sorted(missing)}")
    return _aggregate_selected_operation_infidelities(
        infidelities,
        tuple(BASIS_GATES),
        reference_shape=reference_shape,
        gate_count_weighting=gate_count_weighting,
    )


def operation_infidelities_for_config(
    family: str, config: Mapping[str, float]
) -> dict[str, float]:
    """Compute per-basis-operation infidelities for a homogeneous config."""
    if family == ANCHOR_FAMILY:
        one_qubit = config["p1"] / 2.0
        return {
            "cx": 3.0 * config["p2"] / 4.0,
            "rz": 0.0,
            "sx": one_qubit,
            "x": one_qubit,
        }
    if family == "dephasing_readout":
        one_entanglement_fidelity = (1.0 + sqrt(1.0 - config["lambda_1q"])) / 2.0
        two_part_entanglement_fidelity = (
            1.0 + sqrt(1.0 - config["lambda_2q"])
        ) / 2.0
        one_qubit = 2.0 * (1.0 - one_entanglement_fidelity) / 3.0
        return {
            "cx": 4.0 * (1.0 - two_part_entanglement_fidelity**2) / 5.0,
            "rz": 0.0,
            "sx": one_qubit,
            "x": one_qubit,
        }
    if family == "amp_phase_damping_readout":
        def entanglement_fidelity(amplitude: float, phase: float) -> float:
            return (2.0 - amplitude + 2.0 * sqrt(1.0 - amplitude - phase)) / 4.0

        one_entanglement_fidelity = entanglement_fidelity(
            config["amp_1q"], config["phase_1q"]
        )
        two_part_entanglement_fidelity = entanglement_fidelity(
            config["amp_2q"], config["phase_2q"]
        )
        one_qubit = 2.0 * (1.0 - one_entanglement_fidelity) / 3.0
        return {
            "cx": 4.0 * (1.0 - two_part_entanglement_fidelity**2) / 5.0,
            "rz": 0.0,
            "sx": one_qubit,
            "x": one_qubit,
        }
    if family == "coherent_overrotation":
        epsilon_1q = config["epsilon_1q"]
        epsilon_2q = config["epsilon_2q"]
        return {
            "cx": 4.0 * sin(pi * epsilon_2q / 4.0) ** 2 / 5.0,
            "rz": 0.0,
            "sx": 2.0 * sin(pi * epsilon_1q / 4.0) ** 2 / 3.0,
            "x": 2.0 * sin(pi * epsilon_1q / 2.0) ** 2 / 3.0,
        }
    if family == "correlated_crosstalk":
        independent = config["p2_ind"]
        correlated = config["p2_corr"]
        theta = config["theta_zz"]
        independent_identity = (1.0 - 3.0 * independent / 4.0) ** 2
        independent_zz = (independent / 4.0) ** 2
        composed_identity = (1.0 - correlated) * independent_identity + correlated / 16.0
        composed_zz = (1.0 - correlated) * independent_zz + correlated / 16.0
        entanglement_fidelity = (
            composed_identity * (1.0 - sin(theta / 2.0) ** 2)
            + composed_zz * sin(theta / 2.0) ** 2
        )
        one_qubit = config["p1"] / 2.0
        return {
            "cx": 4.0 * (1.0 - entanglement_fidelity) / 5.0,
            "rz": 0.0,
            "sx": one_qubit,
            "x": one_qubit,
        }
    raise ValueError(f"family {family!r} is not homogeneous")


def average_gate_infidelities_for_config(
    family: str, config: Mapping[str, float]
) -> dict[str, float]:
    """Compute legacy channel-class diagnostics for a homogeneous config."""
    operations = operation_infidelities_for_config(family, config)
    return {
        "1q": float((operations["sx"] + operations["x"]) / 2.0),
        "2q": float(operations["cx"]),
    }


def _scaled_config(
    config: Mapping[str, float], parameters: tuple[str, ...], scale: float
) -> dict[str, float]:
    result = dict(config)
    for parameter in parameters:
        result[parameter] = float(config[parameter] * scale)
    return result


def _maximum_scale(
    family: str, config: Mapping[str, float], parameters: tuple[str, ...]
) -> float:
    if family == "dephasing_readout":
        return 1.0 / max(config[name] for name in parameters)
    if family == "amp_phase_damping_readout":
        return 1.0 / sum(config[name] for name in parameters)
    if family == "coherent_overrotation":
        return 1.0 / max(config[name] for name in parameters)
    if family == "correlated_crosstalk":
        probabilities = [config[name] for name in parameters if name != "theta_zz"]
        return 1.0 / max(probabilities)
    raise ValueError(f"family {family!r} has no scaling bound")


def _solve_parameter_group(
    family: str,
    config: Mapping[str, float],
    parameters: tuple[str, ...],
    operations: tuple[str, ...],
    target: float,
) -> dict[str, float]:
    def objective(scale: float) -> float:
        candidate = _scaled_config(config, parameters, scale)
        return _aggregate_selected_operation_infidelities(
            operation_infidelities_for_config(family, candidate),
            operations,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )

    low = 0.0
    high = min(1.0, _maximum_scale(family, config, parameters))
    maximum = _maximum_scale(family, config, parameters)
    while objective(high) < target and high < maximum:
        high = min(2.0 * high, maximum)
    if objective(high) < target:
        raise ValueError(
            f"{family} {operations} cannot reach anchor infidelity {target:.12g} "
            f"along the shipped parameter ray"
        )
    for _ in range(100):
        midpoint = (low + high) / 2.0
        value = objective(midpoint)
        if abs(value - target) <= SOLVER_RELATIVE_TOLERANCE * target:
            return _scaled_config(config, parameters, midpoint)
        if value < target:
            low = midpoint
        else:
            high = midpoint
    return _scaled_config(config, parameters, (low + high) / 2.0)


def _solve_homogeneous_grid(family: str) -> dict[str, dict[str, float]]:
    if family == ANCHOR_FAMILY:
        return deepcopy(SEVERITY_GRIDS[family])
    try:
        one_qubit_parameters, two_qubit_parameters = _PARAMETER_GROUPS[family]
    except KeyError as exc:
        raise ValueError(f"family {family!r} has no homogeneous solver") from exc
    result: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        anchor = operation_infidelities_for_config(
            ANCHOR_FAMILY, SEVERITY_GRIDS[ANCHOR_FAMILY][level]
        )
        candidate = dict(SEVERITY_GRIDS[family][level])
        one_qubit_target = _aggregate_selected_operation_infidelities(
            anchor,
            ("sx", "x"),
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        two_qubit_target = _aggregate_selected_operation_infidelities(
            anchor,
            ("cx",),
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        candidate = _solve_parameter_group(
            family, candidate, one_qubit_parameters, ("sx", "x"), one_qubit_target
        )
        candidate = _solve_parameter_group(
            family, candidate, two_qubit_parameters, ("cx",), two_qubit_target
        )
        candidate["p_ro"] = SEVERITY_GRIDS[ANCHOR_FAMILY][level]["p_ro"]
        result[level] = candidate
    return result


def solve_proposed_grids() -> dict[str, dict[str, dict[str, float]]]:
    """Return recalibrated copies of all grids, leaving ``SEVERITY_GRIDS`` untouched."""
    result = {
        family: _solve_homogeneous_grid(family)
        for family in SEVERITY_GRIDS
        if family != "mixed_heterogeneous"
    }
    result["mixed_heterogeneous"] = {
        level: {"component_scale": 1.0} for level in LEVELS
    }
    return result


def _mixed_operation_infidelities(
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int,
    n_qubits: int,
) -> dict[str, float]:
    if n_qubits < 2:
        raise ValueError("mixed calibration requires at least two qubits")
    level_scale = grids["mixed_heterogeneous"][severity]["component_scale"]
    one_qubit_values = {"sx": [], "x": []}
    for qubit in range(n_qubits):
        family, jitter = _draw_mixed_profile(seed, (0, qubit))
        config = {
            name: value * level_scale * jitter
            for name, value in grids[family][severity].items()
        }
        infidelities = operation_infidelities_for_config(family, config)
        for operation in one_qubit_values:
            one_qubit_values[operation].append(infidelities[operation])
    two_qubit_values = []
    for control in range(n_qubits):
        for target in range(n_qubits):
            if control == target:
                continue
            family, jitter = _draw_mixed_profile(seed, (1, control, target))
            config = {
                name: value * level_scale * jitter
                for name, value in grids[family][severity].items()
            }
            infidelities = operation_infidelities_for_config(family, config)
            two_qubit_values.append(infidelities["cx"])
    return {
        "cx": float(sum(two_qubit_values) / len(two_qubit_values)),
        "rz": 0.0,
        "sx": float(sum(one_qubit_values["sx"]) / len(one_qubit_values["sx"])),
        "x": float(sum(one_qubit_values["x"]) / len(one_qubit_values["x"])),
    }


def _mixed_expected_operation_infidelities(
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, float]:
    """Average exactly over the mixed model's uniform component and jitter law."""
    level_scale = grids["mixed_heterogeneous"][severity]["component_scale"]
    values = []
    for family in _MIXTURE_COMPONENT_FAMILIES:
        for jitter in _MIXTURE_JITTERS:
            config = {
                name: value * level_scale * jitter
                for name, value in grids[family][severity].items()
            }
            values.append(operation_infidelities_for_config(family, config))
    return {
        operation: float(sum(value[operation] for value in values) / len(values))
        for operation in BASIS_GATES
    }


def operation_infidelities_for_grid(
    family: str,
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int | None = None,
    n_qubits: int = MIXED_REFERENCE_QUBITS,
) -> dict[str, float]:
    """Compute per-operation infidelities from a shipped or proposed registry."""
    if family == "mixed_heterogeneous":
        if seed is None:
            return _mixed_expected_operation_infidelities(severity, grids)
        return _mixed_operation_infidelities(
            severity, grids, seed=seed, n_qubits=n_qubits
        )
    return operation_infidelities_for_config(family, grids[family][severity])


def average_gate_infidelities_for_grid(
    family: str,
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int | None = None,
    n_qubits: int = MIXED_REFERENCE_QUBITS,
) -> dict[str, float]:
    """Compute channel-class infidelities from a shipped or proposed registry."""
    operations = operation_infidelities_for_grid(
        family, severity, grids, seed=seed, n_qubits=n_qubits
    )
    return {
        "1q": float((operations["sx"] + operations["x"]) / 2.0),
        "2q": float(operations["cx"]),
    }


def per_layer_average_gate_infidelity(
    family: str,
    severity: str,
    *,
    reference_shape: str,
    gate_count_weighting: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]] = SEVERITY_GRIDS,
    seed: int | None = None,
    n_qubits: int = MIXED_REFERENCE_QUBITS,
) -> float:
    """Return the scalar severity for a registry cell under two required names."""
    infidelities = operation_infidelities_for_grid(
        family, severity, grids, seed=seed, n_qubits=n_qubits
    )
    return aggregate_operation_infidelities(
        infidelities,
        reference_shape=reference_shape,
        gate_count_weighting=gate_count_weighting,
    )


def _relative_residual(value: float, target: float) -> float:
    return float(abs(value - target) / target)


def _family_residuals(
    family: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    residuals = {}
    for level in LEVELS:
        anchor = operation_infidelities_for_grid(ANCHOR_FAMILY, level, grids)
        candidate = operation_infidelities_for_grid(family, level, grids)
        anchor_scalar = aggregate_operation_infidelities(
            anchor,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        candidate_scalar = aggregate_operation_infidelities(
            candidate,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        residuals[level] = {
            "1q_weighted": _relative_residual(
                _aggregate_selected_operation_infidelities(
                    candidate,
                    ("sx", "x"),
                    reference_shape=PRIMARY_REFERENCE_SHAPE,
                    gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
                ),
                _aggregate_selected_operation_infidelities(
                    anchor,
                    ("sx", "x"),
                    reference_shape=PRIMARY_REFERENCE_SHAPE,
                    gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
                ),
            ),
            "2q_weighted": _relative_residual(candidate["cx"], anchor["cx"]),
            "per_layer": _relative_residual(candidate_scalar, anchor_scalar),
        }
    return residuals


def solve_family_against_anchor(family: str) -> FamilyCalibration:
    """Solve and report one family against the depolarizing anchor."""
    if family not in SEVERITY_GRIDS:
        names = ", ".join(SEVERITY_GRIDS)
        raise ValueError(f"unknown family {family!r}; expected one of: {names}")
    proposed = solve_proposed_grids()
    residuals = _family_residuals(family, proposed)
    matched = all(
        value <= MATCHING_RELATIVE_TOLERANCE
        for level in residuals.values()
        for value in level.values()
    )
    reason = None
    if not matched:
        reason = (
            "The deterministic mixed profile and nonlinear parameter jitter exceed "
            "the declared relative tolerance on the named reference shape."
            if family == "mixed_heterogeneous"
            else "The solved channel metrics exceed the declared relative tolerance."
        )
    return FamilyCalibration(
        family=family,
        shipped_grid=deepcopy(SEVERITY_GRIDS[family]),
        solved_grid=deepcopy(proposed[family]),
        relative_residuals=residuals,
        matched=matched,
        reason=reason,
    )


def average_readout_probability_for_grid(
    family: str,
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int | None = None,
    n_qubits: int = MIXED_REFERENCE_QUBITS,
) -> float:
    """Return mean symmetric readout-flip probability, separate from gate AGI."""
    if family != "mixed_heterogeneous":
        return float(grids[family][severity]["p_ro"])
    level_scale = grids[family][severity]["component_scale"]
    if seed is None:
        probabilities = [
            grids[component][severity]["p_ro"] * level_scale * jitter
            for component in _MIXTURE_COMPONENT_FAMILIES
            for jitter in _MIXTURE_JITTERS
        ]
        return float(sum(probabilities) / len(probabilities))
    probabilities = []
    for qubit in range(n_qubits):
        component, jitter = _draw_mixed_profile(seed, (0, qubit))
        probabilities.append(
            grids[component][severity]["p_ro"] * level_scale * jitter
        )
    return float(sum(probabilities) / len(probabilities))


def build_noise_model_for_grid(
    family: str,
    severity: str,
    grids: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int | None = None,
    n_qubits: int = MIXED_REFERENCE_QUBITS,
) -> NoiseModel:
    """Build a noise model from an explicit shipped or proposed registry."""
    if family != "mixed_heterogeneous":
        return _homogeneous_noise_model(family, dict(grids[family][severity]))
    if seed is None:
        raise ValueError("mixed_heterogeneous requires a caller-supplied seed")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    level_scale = grids[family][severity]["component_scale"]
    model = NoiseModel(basis_gates=BASIS_GATES)
    for qubit in range(n_qubits):
        component, jitter = _draw_mixed_profile(seed, (0, qubit))
        config = {
            name: value * level_scale * jitter
            for name, value in grids[component][severity].items()
        }
        sx_error, x_error, _ = _channels_for_family(component, config)
        model.add_quantum_error(sx_error, ["sx"], [qubit])
        model.add_quantum_error(x_error, ["x"], [qubit])
        model.add_readout_error(_readout_error(config["p_ro"]), [qubit])
    for control in range(n_qubits):
        for target in range(n_qubits):
            if control == target:
                continue
            component, jitter = _draw_mixed_profile(seed, (1, control, target))
            config = {
                name: value * level_scale * jitter
                for name, value in grids[component][severity].items()
            }
            _, _, cx_error = _channels_for_family(component, config)
            model.add_quantum_error(cx_error, ["cx"], [control, target])
    return model
