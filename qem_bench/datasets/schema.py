"""Dataset-item schema and the versioned feature specification.

One row per (circuit, observable, noise, shots) item. The feature specification is
the single source of truth for learned-model inputs. It is metadata-free by design:
circuit and observable structure plus the noisy estimate, never exact noise
parameters. Severity appears in each item as a stratification tag; the numeric noise
parameters live only in the manifest.
"""

from __future__ import annotations

import math

FEATURE_SPEC_VERSION = "v0"

# Order matters: models consume features in exactly this order.
FEATURES: list[str] = [
    "noisy_expectation",
    "log2_shots",
    "n_qubits",
    "steps",
    "j",
    "h",
    "dt",
    "two_qubit_gates",
    "transpiled_depth",
    "obs_locality",
]

REQUIRED_FIELDS: list[str] = [
    "item_id",
    "family",
    "split",
    "instance",
    "n_qubits",
    "steps",
    "j",
    "h",
    "dt",
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


# Fields every row of one measurement group must agree on: sibling observable rows
# come from a single shared execution, so circuit identity, noise, shots, seeds, and
# transpiled structure are group invariants. Only observable-specific fields differ.
GROUP_INVARIANT_FIELDS: list[str] = [
    "family",
    "instance",
    "split",
    "n_qubits",
    "steps",
    "j",
    "h",
    "dt",
    "circuit_seed",
    "noise_family",
    "severity",
    "shots",
    "sampler_seed",
    "two_qubit_gates",
    "transpiled_depth",
]


def validate_item(item: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        raise ValueError(f"item {item.get('item_id', '?')} missing fields: {missing}")


def validate_groups(items: list[dict]) -> None:
    """Reject rows whose measurement group is internally inconsistent.

    A group that disagrees on any invariant field cannot have come from one shared
    execution, so its covariance and its single ledger charge would be fiction.
    """
    by_group: dict[str, list[dict]] = {}
    for item in items:
        by_group.setdefault(item["measurement_group"], []).append(item)
    for group, rows in by_group.items():
        for field in GROUP_INVARIANT_FIELDS:
            values = {row[field] for row in rows}
            if len(values) > 1:
                raise ValueError(
                    f"measurement group {group} disagrees on {field}: {sorted(values)}"
                )


def build_features(item: dict) -> list[float]:
    """Map one item row to the model input vector (FEATURE_SPEC v0)."""
    values = {
        "noisy_expectation": item["noisy_expectation"],
        "log2_shots": math.log2(item["shots"]),
        "n_qubits": item["n_qubits"],
        "steps": item["steps"],
        "j": item["j"],
        "h": item["h"],
        "dt": item["dt"],
        "two_qubit_gates": item["two_qubit_gates"],
        "transpiled_depth": item["transpiled_depth"],
        "obs_locality": item["obs_locality"],
    }
    return [float(values[name]) for name in FEATURES]
