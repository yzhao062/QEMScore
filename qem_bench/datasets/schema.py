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


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(nested)) for key, nested in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


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
