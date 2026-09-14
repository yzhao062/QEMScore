"""Dataset-item schema and the versioned feature specification.

One row represents one (circuit, observable, noise, shots) item. The feature
specification is the single source of truth for learned-model inputs. It is
metadata-free by design: circuit and observable structure plus the noisy estimate,
never exact noise parameters. Severity and stratum are reporting tags.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from qiskit import QuantumCircuit

from qemscore.circuits.heisenberg import (
    HeisenbergParams,
    build_heisenberg_circuit,
)
from qemscore.circuits.near_clifford import (
    NearCliffordParams,
    build_near_clifford_circuit,
)
from qemscore.circuits.qaoa import QAOAParams, build_qaoa_circuit
from qemscore.circuits.random_clifford import (
    RandomCliffordParams,
    build_random_clifford_circuit,
)
from qemscore.circuits.tfi import TFIParams, build_tfi_circuit

FEATURE_SPEC_VERSION = "v1"
LEGACY_SCHEMA_VERSION = "legacy-v1"
SPLIT_SCHEMA_VERSION = "split-v2"

# Order matters: models consume features in exactly this order.
FEATURES: list[str] = [
    "noisy_expectation",
    "log2_shots",
    "n_qubits",
    "family_tfi",
    "family_qaoa",
    "family_heisenberg",
    "family_random_clifford",
    "family_near_clifford",
    "steps",
    "j",
    "h",
    "jx",
    "jy",
    "jz",
    "dt",
    "qaoa_p",
    "qaoa_edge_count",
    "qaoa_gamma_0",
    "qaoa_gamma_1",
    "qaoa_beta_0",
    "qaoa_beta_1",
    "graph_path",
    "graph_cycle",
    "graph_erdos_renyi",
    "graph_3_regular",
    "rc_depth",
    "nc_non_clifford_count",
    "two_qubit_gates",
    "transpiled_depth",
    "obs_locality",
]

STRATA = frozenset({"continuous_regression", "clifford_control"})
FAMILY_STRATA: dict[str, str] = {
    "tfi": "continuous_regression",
    "qaoa": "continuous_regression",
    "heisenberg": "continuous_regression",
    "random_clifford": "clifford_control",
    "near_clifford": "continuous_regression",
}
FAMILY_LABEL_METHODS: dict[str, str] = {
    "tfi": "statevector",
    "qaoa": "statevector",
    "heisenberg": "statevector",
    "random_clifford": "stim",
    "near_clifford": "statevector",
}

REQUIRED_FIELDS: list[str] = [
    "item_id",
    "family",
    "stratum",
    "split",
    "instance",
    "n_qubits",
    "circuit_seed",
    "observable",
    "pauli_label",
    "obs_locality",
    "noise_family",
    "severity",
    "shots",
    "measurement_group",
    "sampler_seed",
    "noisy_expectation",
    "noisy_stderr",
    "ideal_expectation",
    "label_method",
    "two_qubit_gates",
    "transpiled_depth",
]

FAMILY_REQUIRED_FIELDS: dict[str, list[str]] = {
    "tfi": ["steps", "j", "h", "dt"],
    "qaoa": [
        "p",
        "graph_class",
        "edges",
        "edge_probability",
        "gammas",
        "betas",
    ],
    "heisenberg": ["steps", "jx", "jy", "jz", "dt"],
    "random_clifford": ["depth"],
    "near_clifford": ["depth", "non_clifford_count", "theta"],
}
SPLIT_ITEM_FIELDS = frozenset(
    {
        "dataset_schema_version",
        "split_id",
        "split_axis",
        "partition_id",
        "domain",
        "cell_id",
        "circuit_pool_id",
        "circuit_id",
        "circuit_hash",
        "circuit_sidecar",
        "observable_id",
        "observable_hash",
        "observable_sidecar",
        "noise_config_id",
        "noise_config_hash",
        "noise_sidecar",
        "replicate",
        "measurement_plan_id",
        "counts_hash",
        "counts_sidecar",
        "raw_circuit_evals",
        "feature_spec_id",
        "axis_values",
    }
)

# Sibling observable rows come from one shared execution. Circuit identity, noise,
# shots, seeds, stratum, label method, and transpiled structure are invariants.
GROUP_INVARIANT_FIELDS: list[str] = [
    "family",
    "stratum",
    "split",
    "instance",
    "n_qubits",
    "circuit_seed",
    "noise_family",
    "severity",
    "shots",
    "sampler_seed",
    "label_method",
    "two_qubit_gates",
    "transpiled_depth",
]

_INTEGER_FIELD_MINIMUMS: dict[str, int] = {
    "instance": 0,
    "n_qubits": 1,
    "circuit_seed": 0,
    "obs_locality": 0,
    "shots": 1,
    "sampler_seed": 0,
    "two_qubit_gates": 0,
    "transpiled_depth": 0,
    "replicate": 0,
    "raw_circuit_evals": 1,
}
_FAMILY_INTEGER_FIELD_MINIMUMS: dict[str, dict[str, int]] = {
    "tfi": {"steps": 1},
    "qaoa": {"p": 1},
    "heisenberg": {"steps": 1},
    "random_clifford": {"depth": 1},
    "near_clifford": {"depth": 1, "non_clifford_count": 0},
}
_FAMILY_QUBIT_BOUNDS: dict[str, tuple[int, int | None]] = {
    "tfi": (1, None),
    "qaoa": (2, 12),
    "heisenberg": (2, 12),
    "random_clifford": (1, 20),
    "near_clifford": (1, 14),
}
_FINITE_NUMBER_FIELDS: tuple[str, ...] = (
    "noisy_expectation",
    "noisy_stderr",
    "ideal_expectation",
)
_FAMILY_FINITE_NUMBER_FIELDS: dict[str, tuple[str, ...]] = {
    "tfi": ("j", "h", "dt"),
    "qaoa": (),
    "heisenberg": ("jx", "jy", "jz", "dt"),
    "random_clifford": (),
    "near_clifford": ("theta",),
}
_CIRCUIT_INTEGER_FIELDS = {
    "tfi": {"steps"},
    "qaoa": {"p"},
    "heisenberg": {"steps"},
    "random_clifford": {"depth"},
    "near_clifford": {"depth", "non_clifford_count"},
}
_CIRCUIT_FLOAT_FIELDS = {
    "tfi": {"j", "h", "dt"},
    "qaoa": {"edge_probability"},
    "heisenberg": {"jx", "jy", "jz", "dt"},
    "random_clifford": set(),
    "near_clifford": {"theta"},
}


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(nested)) for key, nested in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _validate_integer_field(item: dict, field: str, minimum: int) -> None:
    value = item[field]
    item_id = item.get("item_id", "?")
    if type(value) is not int:
        raise ValueError(f"item {item_id} field {field} must be an integer")
    if value < minimum:
        bound = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"item {item_id} field {field} must be {bound}")


def _validate_finite_number(item: dict, field: str, value) -> None:
    item_id = item.get("item_id", "?")
    if type(value) not in (int, float):
        raise ValueError(f"item {item_id} field {field} must be a number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"item {item_id} field {field} must be finite")


def _validate_qaoa_parameters(item: dict) -> None:
    item_id = item.get("item_id", "?")
    n_qubits = item["n_qubits"]

    graph_class = item["graph_class"]
    if not isinstance(graph_class, str) or graph_class not in {
        "path",
        "cycle",
        "erdos_renyi",
        "3_regular",
    }:
        raise ValueError(f"item {item_id} has unknown QAOA graph_class")
    if item["p"] not in (1, 2):
        raise ValueError(f"item {item_id} field p must be 1 or 2")

    edge_probability = item["edge_probability"]
    if edge_probability is not None:
        _validate_finite_number(item, "edge_probability", edge_probability)
    if (graph_class == "erdos_renyi") != (edge_probability is not None):
        raise ValueError(
            f"item {item_id} edge_probability must be present only for erdos_renyi"
        )
    if edge_probability is not None and not 0.0 <= float(edge_probability) <= 1.0:
        raise ValueError(
            f"item {item_id} field edge_probability must be in [0, 1]"
        )

    angle_ranges = {"gammas": math.pi, "betas": math.pi / 2.0}
    for field, upper in angle_ranges.items():
        values = item[field]
        if not isinstance(values, (list, tuple)) or len(values) != item["p"]:
            raise ValueError(f"item {item_id} QAOA {field} length must equal p")
        for value in values:
            _validate_finite_number(item, field, value)
            if not 0.0 <= float(value) < upper:
                rendered_upper = "pi" if field == "gammas" else "pi/2"
                raise ValueError(
                    f"item {item_id} QAOA {field} must be in [0, {rendered_upper})"
                )

    edges = item["edges"]
    if not isinstance(edges, (list, tuple)) or any(
        not isinstance(edge, (list, tuple))
        or len(edge) != 2
        or any(type(endpoint) is not int for endpoint in edge)
        for edge in edges
    ):
        raise ValueError(
            f"item {item_id} field edges must contain integer pairs"
        )

    canonical_edges: set[tuple[int, int]] = set()
    for q0, q1 in edges:
        if q0 == q1 or not (0 <= q0 < n_qubits and 0 <= q1 < n_qubits):
            raise ValueError(f"item {item_id} contains an invalid QAOA edge")
        edge = tuple(sorted((q0, q1)))
        if edge in canonical_edges:
            raise ValueError(f"item {item_id} contains a duplicate QAOA edge")
        canonical_edges.add(edge)

    if graph_class == "path":
        expected = {(q, q + 1) for q in range(n_qubits - 1)}
        if canonical_edges != expected:
            raise ValueError(f"item {item_id} edges do not form the declared path")
    elif graph_class == "cycle":
        expected = {(q, q + 1) for q in range(n_qubits - 1)} | {
            (0, n_qubits - 1)
        }
        if n_qubits < 3 or canonical_edges != expected:
            raise ValueError(f"item {item_id} edges do not form the declared cycle")
    elif graph_class == "3_regular":
        degrees = [0] * n_qubits
        for q0, q1 in canonical_edges:
            degrees[q0] += 1
            degrees[q1] += 1
        if n_qubits < 4 or n_qubits % 2 or set(degrees) != {3}:
            raise ValueError(
                f"item {item_id} edges are not a simple 3-regular graph"
            )


def _validate_circuit_fields(item: dict, family: str) -> None:
    for field in ("n_qubits", "circuit_seed"):
        _validate_integer_field(item, field, _INTEGER_FIELD_MINIMUMS[field])
    for field, minimum in _FAMILY_INTEGER_FIELD_MINIMUMS[family].items():
        _validate_integer_field(item, field, minimum)
    for field in _FAMILY_FINITE_NUMBER_FIELDS[family]:
        _validate_finite_number(item, field, item[field])

    item_id = item.get("item_id", "?")
    lower, upper = _FAMILY_QUBIT_BOUNDS[family]
    n_qubits = item["n_qubits"]
    if n_qubits < lower or (upper is not None and n_qubits > upper):
        display_family = "QAOA" if family == "qaoa" else family
        rendered_upper = "unbounded" if upper is None else str(upper)
        raise ValueError(
            f"item {item_id} {display_family} n_qubits must be in "
            f"[{lower}, {rendered_upper}]"
        )

    if family == "heisenberg" and float(item["dt"]) <= 0.0:
        raise ValueError(f"item {item_id} Heisenberg dt must be positive")
    if family == "near_clifford":
        count = item["non_clifford_count"]
        if not 1 <= count <= item["n_qubits"]:
            raise ValueError(
                f"item {item_id} non_clifford_count must be in [1, n_qubits]"
            )
        quarter_turns = float(item["theta"]) / (math.pi / 2.0)
        if math.isclose(
            quarter_turns, round(quarter_turns), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"item {item_id} theta must be non-Clifford")
    elif family == "qaoa":
        _validate_qaoa_parameters(item)


def _validate_numeric_fields(item: dict, family: str) -> None:
    for field, minimum in _INTEGER_FIELD_MINIMUMS.items():
        if field in item and field not in {"n_qubits", "circuit_seed"}:
            _validate_integer_field(item, field, minimum)
    for field in _FINITE_NUMBER_FIELDS:
        _validate_finite_number(item, field, item[field])
    _validate_circuit_fields(item, family)

    item_id = item.get("item_id", "?")
    for field in ("noisy_expectation", "ideal_expectation"):
        if not -1.0 <= float(item[field]) <= 1.0:
            raise ValueError(f"item {item_id} field {field} must be in [-1, 1]")
    if float(item["noisy_stderr"]) < 0.0:
        raise ValueError(f"item {item_id} field noisy_stderr must be nonnegative")
    stderr_limit = 1.0 / math.sqrt(item["shots"])
    if float(item["noisy_stderr"]) > stderr_limit + 1e-12:
        raise ValueError(
            f"item {item_id} field noisy_stderr exceeds its shot bound"
        )

    pauli_label = item["pauli_label"]
    n_qubits = item["n_qubits"]
    if (
        not isinstance(pauli_label, str)
        or len(pauli_label) != n_qubits
        or set(pauli_label) - {"I", "Z"}
    ):
        raise ValueError(
            f"item {item_id} field pauli_label must be an "
            "n_qubits-long I/Z string"
        )
    obs_locality = item["obs_locality"]
    if obs_locality > n_qubits:
        raise ValueError(
            f"item {item_id} field obs_locality must not exceed n_qubits"
        )
    if pauli_label.count("Z") != obs_locality:
        raise ValueError(
            f"item {item_id} field obs_locality disagrees with pauli_label"
        )


def _canonical_circuit_float(value: object) -> float:
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def canonical_physical_circuit_descriptor(
    item: Mapping[str, object],
) -> dict[str, object]:
    """Validate and project a row into the one physical-circuit descriptor."""

    family = item.get("family")
    if not isinstance(family, str) or family not in FAMILY_REQUIRED_FIELDS:
        raise ValueError(f"unknown circuit family {family!r}")
    required_fields = (
        "n_qubits",
        "circuit_seed",
        *FAMILY_REQUIRED_FIELDS[family],
    )
    nullable_fields = {"edge_probability"} if family == "qaoa" else set()
    missing = [
        field
        for field in required_fields
        if field not in item or (item[field] is None and field not in nullable_fields)
    ]
    if missing:
        raise ValueError(f"missing circuit fields: {missing}")

    projected = {
        "item_id": item.get("item_id", "?"),
        "family": family,
        **{field: item[field] for field in required_fields},
    }
    _validate_circuit_fields(projected, family)

    parameters: dict[str, object] = {}
    for field in FAMILY_REQUIRED_FIELDS[family]:
        value = item[field]
        if field in _CIRCUIT_INTEGER_FIELDS[family]:
            pass
        elif field in _CIRCUIT_FLOAT_FIELDS[family]:
            if value is not None:
                value = _canonical_circuit_float(value)
        elif field in {"gammas", "betas"}:
            value = [_canonical_circuit_float(entry) for entry in value]
        elif field == "edges":
            value = sorted([list(sorted((q0, q1))) for q0, q1 in value])
        elif field != "graph_class":
            raise ValueError(f"unclassified circuit field {field!r}")
        parameters[field] = value

    return {
        "family": family,
        "n_qubits": item["n_qubits"],
        "parameters": parameters,
        "circuit_seed": item["circuit_seed"],
    }


def build_circuit_from_canonical_descriptor(
    descriptor: Mapping[str, object],
) -> QuantumCircuit:
    """Rebuild a canonical descriptor through the benchmark family builders."""

    expected_fields = {"family", "n_qubits", "parameters", "circuit_seed"}
    actual_fields = set(descriptor)
    if actual_fields != expected_fields:
        raise ValueError(
            "circuit descriptor fields must be exactly "
            f"{sorted(expected_fields)!r}"
        )
    family = descriptor["family"]
    if not isinstance(family, str) or family not in FAMILY_REQUIRED_FIELDS:
        raise ValueError(f"unknown circuit family {family!r}")
    raw_parameters = descriptor["parameters"]
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("circuit descriptor parameters must be a mapping")
    if set(raw_parameters) != set(FAMILY_REQUIRED_FIELDS[family]):
        raise ValueError(
            f"{family} circuit descriptor parameters must be exactly "
            f"{sorted(FAMILY_REQUIRED_FIELDS[family])!r}"
        )

    canonical = canonical_physical_circuit_descriptor(
        {
            "family": family,
            "n_qubits": descriptor["n_qubits"],
            "circuit_seed": descriptor["circuit_seed"],
            **raw_parameters,
        }
    )
    if canonical != descriptor:
        raise ValueError("circuit descriptor is not canonical")
    parameters = canonical["parameters"]
    common = {
        "n_qubits": canonical["n_qubits"],
        "circuit_seed": canonical["circuit_seed"],
        "instance": 0,
    }

    if family == "tfi":
        return build_tfi_circuit(
            TFIParams(
                **common,
                steps=parameters["steps"],
                j=parameters["j"],
                h=parameters["h"],
                dt=parameters["dt"],
            )
        )
    if family == "qaoa":
        return build_qaoa_circuit(
            QAOAParams(
                **common,
                graph_class=parameters["graph_class"],
                edges=tuple(tuple(edge) for edge in parameters["edges"]),
                p=parameters["p"],
                gammas=tuple(parameters["gammas"]),
                betas=tuple(parameters["betas"]),
                edge_probability=parameters["edge_probability"],
            )
        )
    if family == "heisenberg":
        return build_heisenberg_circuit(
            HeisenbergParams(
                **common,
                steps=parameters["steps"],
                jx=parameters["jx"],
                jy=parameters["jy"],
                jz=parameters["jz"],
                dt=parameters["dt"],
            )
        )
    if family == "random_clifford":
        return build_random_clifford_circuit(
            RandomCliffordParams(**common, depth=parameters["depth"])
        )
    return build_near_clifford_circuit(
        NearCliffordParams(
            **common,
            depth=parameters["depth"],
            non_clifford_count=parameters["non_clifford_count"],
            theta=parameters["theta"],
        )
    )


def _canonical_instruction_stream(
    circuit: QuantumCircuit,
) -> list[dict[str, object]]:
    stream: list[dict[str, object]] = []
    for instruction in circuit.data:
        if instruction.clbits:
            raise ValueError(
                "physical-circuit identity does not support classical operands"
            )
        stream.append(
            {
                "name": instruction.operation.name,
                "qubits": [
                    circuit.find_bit(qubit).index for qubit in instruction.qubits
                ],
                "params": [
                    _canonical_circuit_float(value)
                    for value in instruction.operation.params
                ],
            }
        )
    return stream


def canonical_physical_circuit_identity(
    item: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Return the provenance sidecar and executable-circuit identity."""

    descriptor = canonical_physical_circuit_descriptor(item)
    circuit = build_circuit_from_canonical_descriptor(descriptor)
    executable = {
        "n_qubits": circuit.num_qubits,
        "instructions": _canonical_instruction_stream(circuit),
    }
    payload = json.dumps(
        executable,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return descriptor, f"circuit-{hashlib.sha256(payload).hexdigest()}"


def validate_item(item: dict, *, schema_version: str) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise ValueError(f"item {item.get('item_id', '?')} missing fields: {missing}")

    family = item["family"]
    if family not in FAMILY_REQUIRED_FIELDS:
        raise ValueError(f"unknown circuit family {family!r}")
    family_missing = [
        field for field in FAMILY_REQUIRED_FIELDS[family] if field not in item
    ]
    if family_missing:
        raise ValueError(
            f"item {item.get('item_id', '?')} missing {family} fields: {family_missing}"
        )
    allowed_fields = set(REQUIRED_FIELDS) | set(FAMILY_REQUIRED_FIELDS[family])
    if schema_version == SPLIT_SCHEMA_VERSION:
        allowed_fields |= SPLIT_ITEM_FIELDS
    elif schema_version != LEGACY_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema {schema_version!r}")
    unsupported = sorted(set(item) - allowed_fields)
    if unsupported:
        raise ValueError(
            f"item {item.get('item_id', '?')} has unsupported fields: {unsupported}"
        )

    _validate_numeric_fields(item, family)

    stratum = item["stratum"]
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}")
    if stratum != FAMILY_STRATA[family]:
        raise ValueError(
            f"family {family!r} requires stratum {FAMILY_STRATA[family]!r}"
        )
    if item["label_method"] != FAMILY_LABEL_METHODS[family]:
        raise ValueError(
            f"family {family!r} requires label_method "
            f"{FAMILY_LABEL_METHODS[family]!r}"
        )


def _measurement_execution_key(item: Mapping[str, object]) -> tuple[object, ...]:
    """Return the generator coordinates of one shared counts draw."""
    if item.get("dataset_schema_version") == SPLIT_SCHEMA_VERSION:
        return (
            SPLIT_SCHEMA_VERSION,
            item["cell_id"],
            item["circuit_id"],
            item["instance"],
            item["replicate"],
        )
    return LEGACY_SCHEMA_VERSION, item["family"], item["instance"]


def validate_groups(items: list[dict]) -> None:
    """Require rows to share a group exactly when they share an execution key."""
    by_group: dict[str, list[dict]] = {}
    execution_by_group: dict[object, tuple[object, ...]] = {}
    group_by_execution: dict[tuple[object, ...], object] = {}
    for item in items:
        group = item["measurement_group"]
        by_group.setdefault(group, []).append(item)
        execution = _measurement_execution_key(item)
        previous_execution = execution_by_group.setdefault(group, execution)
        if previous_execution != execution:
            raise ValueError(
                f"measurement group {group!r} spans execution configurations "
                f"{previous_execution!r} and {execution!r}"
            )
        previous = group_by_execution.setdefault(execution, group)
        if previous != group:
            raise ValueError(
                f"execution configuration {execution!r} spans measurement groups "
                f"{previous!r} and {group!r}"
            )
    for group, rows in by_group.items():
        family_fields = FAMILY_REQUIRED_FIELDS[rows[0]["family"]]
        for field in GROUP_INVARIANT_FIELDS + family_fields:
            values = {_freeze(row[field]) for row in rows}
            if len(values) > 1:
                rendered = sorted(repr(value) for value in values)
                raise ValueError(
                    f"measurement group {group} disagrees on {field}: {rendered}"
                )


def build_features(item: dict) -> list[float]:
    """Map one item row to the model input vector (FEATURE_SPEC v1)."""
    family = item["family"]
    is_tfi = family == "tfi"
    is_qaoa = family == "qaoa"
    is_heisenberg = family == "heisenberg"
    is_random_clifford = family == "random_clifford"
    is_near_clifford = family == "near_clifford"
    graph_class = item.get("graph_class") if is_qaoa else None
    gammas = item.get("gammas", ()) if is_qaoa else ()
    betas = item.get("betas", ()) if is_qaoa else ()
    values = {
        "noisy_expectation": item["noisy_expectation"],
        "log2_shots": math.log2(item["shots"]),
        "n_qubits": item["n_qubits"],
        "family_tfi": is_tfi,
        "family_qaoa": is_qaoa,
        "family_heisenberg": is_heisenberg,
        "family_random_clifford": is_random_clifford,
        "family_near_clifford": is_near_clifford,
        "steps": item.get("steps", 0) if is_tfi or is_heisenberg else 0,
        "j": item.get("j", 0.0) if is_tfi else 0.0,
        "h": item.get("h", 0.0) if is_tfi else 0.0,
        "jx": item.get("jx", 0.0) if is_heisenberg else 0.0,
        "jy": item.get("jy", 0.0) if is_heisenberg else 0.0,
        "jz": item.get("jz", 0.0) if is_heisenberg else 0.0,
        "dt": item.get("dt", 0.0) if is_tfi or is_heisenberg else 0.0,
        "qaoa_p": item.get("p", 0) if is_qaoa else 0,
        "qaoa_edge_count": len(item.get("edges", ())) if is_qaoa else 0,
        "qaoa_gamma_0": gammas[0] if len(gammas) > 0 else 0.0,
        "qaoa_gamma_1": gammas[1] if len(gammas) > 1 else 0.0,
        "qaoa_beta_0": betas[0] if len(betas) > 0 else 0.0,
        "qaoa_beta_1": betas[1] if len(betas) > 1 else 0.0,
        "graph_path": graph_class == "path",
        "graph_cycle": graph_class == "cycle",
        "graph_erdos_renyi": graph_class == "erdos_renyi",
        "graph_3_regular": graph_class == "3_regular",
        "rc_depth": item.get("depth", 0) if is_random_clifford else 0,
        "nc_non_clifford_count": (
            item.get("non_clifford_count", 0) if is_near_clifford else 0
        ),
        "two_qubit_gates": item["two_qubit_gates"],
        "transpiled_depth": item["transpiled_depth"],
        "obs_locality": item["obs_locality"],
    }
    return [float(values[name]) for name in FEATURES]
