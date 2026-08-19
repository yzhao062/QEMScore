"""Dataset-item schema and the versioned feature specification.

One row represents one (circuit, observable, noise, shots) item. The feature
specification is the single source of truth for learned-model inputs. It is
metadata-free by design: circuit and observable structure plus the noisy estimate,
never exact noise parameters. Severity and stratum are reporting tags.
"""

from __future__ import annotations

import math

FEATURE_SPEC_VERSION = "v1"

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
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"item {item_id} field {field} must be finite")


def _validate_numeric_fields(item: dict, family: str) -> None:
    for field, minimum in _INTEGER_FIELD_MINIMUMS.items():
        if field in item:
            _validate_integer_field(item, field, minimum)
    for field, minimum in _FAMILY_INTEGER_FIELD_MINIMUMS[family].items():
        _validate_integer_field(item, field, minimum)
    for field in _FINITE_NUMBER_FIELDS + _FAMILY_FINITE_NUMBER_FIELDS[family]:
        _validate_finite_number(item, field, item[field])

    if family != "qaoa":
        return
    edge_probability = item["edge_probability"]
    if edge_probability is not None:
        _validate_finite_number(item, "edge_probability", edge_probability)
    for field in ("gammas", "betas"):
        values = item[field]
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"item {item.get('item_id', '?')} field {field} must contain numbers"
            )
        for value in values:
            _validate_finite_number(item, field, value)
    edges = item["edges"]
    if not isinstance(edges, (list, tuple)) or any(
        not isinstance(edge, (list, tuple))
        or len(edge) != 2
        or any(type(endpoint) is not int for endpoint in edge)
        for edge in edges
    ):
        raise ValueError(
            f"item {item.get('item_id', '?')} field edges must contain integer pairs"
        )


def validate_item(item: dict) -> None:
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


def validate_groups(items: list[dict]) -> None:
    """Reject rows whose measurement group is internally inconsistent."""
    by_group: dict[str, list[dict]] = {}
    for item in items:
        by_group.setdefault(item["measurement_group"], []).append(item)
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
