"""Calibration definitions, solvers, and proposal-registry checks."""

from __future__ import annotations

import copy
import json

import pytest
from qiskit import transpile

from qemscore.circuits.random_clifford import (
    RandomCliffordParams,
    build_random_clifford_circuit,
)
from qemscore.noise.calibration import (
    ANCHOR_FAMILY,
    GATE_COUNT_WEIGHTINGS,
    LEVELS,
    MATCHING_RELATIVE_TOLERANCE,
    PRIMARY_GATE_COUNT_WEIGHTING,
    PRIMARY_REFERENCE_SHAPE,
    PROPOSAL_PATH,
    REFERENCE_SHAPES,
    aggregate_operation_infidelities,
    average_gate_infidelities_for_config,
    average_readout_probability_for_grid,
    build_noise_model_for_grid,
    operation_infidelities_for_config,
    per_layer_average_gate_infidelity,
    solve_family_against_anchor,
    solve_proposed_grids,
)
from qemscore.noise.models import BASIS_GATES, SEVERITY_GRIDS, average_gate_infidelities

HOMOGENEOUS_FAMILIES = tuple(
    family for family in SEVERITY_GRIDS if family != "mixed_heterogeneous"
)


def test_reference_shape_reproduces_pinned_transpiled_gate_counts():
    params = RandomCliffordParams(
        n_qubits=6, depth=1, circuit_seed=1, instance=0
    )
    transpiled = transpile(
        build_random_clifford_circuit(params),
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=1,
    )
    counts = {gate: int(transpiled.count_ops().get(gate, 0)) for gate in BASIS_GATES}
    assert counts == REFERENCE_SHAPES[PRIMARY_REFERENCE_SHAPE].gate_counts
    assert set(GATE_COUNT_WEIGHTINGS[PRIMARY_GATE_COUNT_WEIGHTING].operation_weights) == set(
        BASIS_GATES
    )


def test_scalar_calibration_requires_both_registered_names():
    infidelities = {"cx": 0.01, "rz": 0.0, "sx": 0.001, "x": 0.002}
    with pytest.raises(TypeError):
        aggregate_operation_infidelities(infidelities)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reference_shape"):
        aggregate_operation_infidelities(
            infidelities,
            reference_shape=None,  # type: ignore[arg-type]
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
    with pytest.raises(ValueError, match="gate_count_weighting"):
        aggregate_operation_infidelities(
            infidelities,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=None,  # type: ignore[arg-type]
        )


def test_coherent_scalar_weights_sx_and_x_as_distinct_operations():
    config = {"epsilon_1q": 0.1, "epsilon_2q": 0.2, "p_ro": 0.0}
    infidelities = operation_infidelities_for_config(
        "coherent_overrotation", config
    )
    shape = REFERENCE_SHAPES[PRIMARY_REFERENCE_SHAPE]
    weighting = GATE_COUNT_WEIGHTINGS[PRIMARY_GATE_COUNT_WEIGHTING]
    denominator = sum(
        shape.gate_counts[gate] * weighting.operation_weights[gate]
        for gate in BASIS_GATES
    )
    expected = sum(
        shape.gate_counts[gate]
        * weighting.operation_weights[gate]
        * infidelities[gate]
        for gate in BASIS_GATES
    ) / denominator
    actual = aggregate_operation_infidelities(
        infidelities,
        reference_shape=PRIMARY_REFERENCE_SHAPE,
        gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
    )
    collapsed_one_qubit = (infidelities["sx"] + infidelities["x"]) / 2.0
    collapsed_values = {
        "cx": infidelities["cx"],
        "rz": 0.0,
        "sx": collapsed_one_qubit,
        "x": collapsed_one_qubit,
    }
    collapsed = sum(
        shape.gate_counts[gate]
        * weighting.operation_weights[gate]
        * collapsed_values[gate]
        for gate in BASIS_GATES
    ) / denominator
    assert infidelities["sx"] != pytest.approx(infidelities["x"])
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(collapsed)


@pytest.mark.parametrize("family", HOMOGENEOUS_FAMILIES)
@pytest.mark.parametrize("level", LEVELS)
def test_closed_form_config_infidelities_match_shipped_qiskit_channels(
    family: str, level: str
):
    expected = average_gate_infidelities(family, level)
    actual = average_gate_infidelities_for_config(family, SEVERITY_GRIDS[family][level])
    assert actual == pytest.approx(expected, abs=2e-13)


def test_solver_matches_anchor_without_mutating_shipped_registry():
    before = copy.deepcopy(SEVERITY_GRIDS)
    proposed = solve_proposed_grids()
    assert SEVERITY_GRIDS == before
    assert proposed is not SEVERITY_GRIDS

    for family in SEVERITY_GRIDS:
        calibration = solve_family_against_anchor(family)
        assert calibration.matched, (family, calibration.relative_residuals)
        for residuals in calibration.relative_residuals.values():
            assert max(residuals.values()) <= MATCHING_RELATIVE_TOLERANCE
        for level in LEVELS:
            assert average_readout_probability_for_grid(
                family, level, proposed
            ) == pytest.approx(
                average_readout_probability_for_grid(ANCHOR_FAMILY, level, proposed),
                rel=MATCHING_RELATIVE_TOLERANCE,
            )


def test_coherent_solution_verifies_predicted_nonlinear_endpoints():
    grid = solve_proposed_grids()["coherent_overrotation"]
    assert grid["L1"]["epsilon_1q"] == pytest.approx(0.029179201405, abs=1e-11)
    assert grid["L4"]["epsilon_1q"] == pytest.approx(0.130975115225, abs=1e-11)
    assert grid["L1"]["epsilon_2q"] == pytest.approx(0.123474332407, abs=1e-11)
    assert grid["L4"]["epsilon_2q"] == pytest.approx(0.396212085515, abs=1e-11)


def test_mixed_grid_is_pinned_to_recalibrated_component_expectation():
    proposed = solve_proposed_grids()
    assert proposed["mixed_heterogeneous"] == {
        level: {"component_scale": 1.0} for level in LEVELS
    }
    for level in LEVELS:
        mixed = per_layer_average_gate_infidelity(
            "mixed_heterogeneous",
            level,
            grids=proposed,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        anchor = per_layer_average_gate_infidelity(
            ANCHOR_FAMILY,
            level,
            grids=proposed,
            reference_shape=PRIMARY_REFERENCE_SHAPE,
            gate_count_weighting=PRIMARY_GATE_COUNT_WEIGHTING,
        )
        assert mixed == pytest.approx(anchor, rel=MATCHING_RELATIVE_TOLERANCE)


def test_explicit_proposed_registry_build_does_not_replace_shipped_registry():
    proposed = solve_proposed_grids()
    proposed_model = build_noise_model_for_grid(
        "coherent_overrotation", "L1", proposed
    )
    shipped_model = build_noise_model_for_grid(
        "coherent_overrotation", "L1", SEVERITY_GRIDS
    )
    assert proposed_model != shipped_model
    with pytest.raises(ValueError, match="caller-supplied seed"):
        build_noise_model_for_grid("mixed_heterogeneous", "L1", proposed)


def test_machine_readable_proposal_matches_solver_output():
    assert PROPOSAL_PATH.is_file()
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    assert "hub_document_target" not in proposal
    assert "depth_probe" not in proposal
    assert (
        proposal["primary_matching_status"]["coherent_overrotation"]
        == "gate_count_metric_matched"
    )
    assert proposal["matching_relative_tolerance"] == MATCHING_RELATIVE_TOLERANCE
    assert proposal["reference_shape"]["name"] == PRIMARY_REFERENCE_SHAPE
    assert proposal["gate_count_weighting"]["name"] == PRIMARY_GATE_COUNT_WEIGHTING
    assert proposal["reference_shape"]["gate_counts"] == dict(
        REFERENCE_SHAPES[proposal["reference_shape"]["name"]].gate_counts
    )
    assert proposal["gate_count_weighting"]["operation_weights"] == dict(
        GATE_COUNT_WEIGHTINGS[
            proposal["gate_count_weighting"]["name"]
        ].operation_weights
    )
    assert proposal["solved_grids"] == solve_proposed_grids()


def test_published_proposal_satisfies_its_declared_weighted_tolerance():
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    tolerance = float(proposal["matching_relative_tolerance"])
    reference_shape = proposal["reference_shape"]["name"]
    gate_count_weighting = proposal["gate_count_weighting"]["name"]
    grids = proposal["solved_grids"]

    for level in LEVELS:
        anchor = per_layer_average_gate_infidelity(
            proposal["anchor_family"],
            level,
            grids=grids,
            reference_shape=reference_shape,
            gate_count_weighting=gate_count_weighting,
        )
        for family in grids:
            candidate = per_layer_average_gate_infidelity(
                family,
                level,
                grids=grids,
                reference_shape=reference_shape,
                gate_count_weighting=gate_count_weighting,
            )
            relative_residual = abs(candidate - anchor) / anchor
            assert relative_residual <= tolerance, (
                f"{family} {level} residual {relative_residual:.6%} exceeds "
                f"declared tolerance {tolerance:.6%}"
            )
