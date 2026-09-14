"""Closed split-contract and split-v2 integrity validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

from qemscore.budget import TIERS
from qemscore.circuits.qaoa import QAOAParams
from qemscore.datasets.schema import (
    FAMILY_LABEL_METHODS,
    FEATURE_SPEC_VERSION,
    FEATURES,
    build_circuit_from_canonical_descriptor,
    canonical_physical_circuit_identity,
    validate_groups,
    validate_item,
)
from qemscore.datasets.splits import (
    AXES,
    DEFAULT_ROLE_DOMAINS,
    ROLES,
    SPLIT_ALLOWED_COUPLINGS,
    SPLIT_AXES,
    MeasurementCell,
    SplitSpec,
    resolve_split_spec,
)
from qemscore.labels.statevector import ideal_expectation as statevector_expectation
from qemscore.labels.stim_labels import ideal_expectation as stim_expectation
from qemscore.observables import (
    z_expectation_from_counts,
    z_support_label,
)
from qemscore.noise.models import SEVERITY_GRIDS
from qemscore.reproducibility import validate_environment_contract

SPLIT_SCHEMA_VERSION = "split-v2"
LEGACY_SCHEMA_VERSION = "legacy-v1"
SPLIT_SEED_FORMULA = (
    "circuit: SeedSequence(master_seed, spawn_key=tuple(pool_seed_key) + "
    "(0, local_instance)); sampler: SeedSequence(master_seed, "
    "spawn_key=tuple(cell_seed_key) + "
    "(1, local_instance, replicate, measurement_basis_group))"
)
_FAMILY_DEPTH_FIELDS = {
    "tfi": "steps",
    "qaoa": "p",
    "heisenberg": "steps",
    "random_clifford": "depth",
    "near_clifford": "depth",
}
_FAMILY_PARAMETER_KEY_CONTRACT = MappingProxyType(
    {
        "tfi": (frozenset({"dt"}), frozenset()),
        "heisenberg": (frozenset({"dt"}), frozenset()),
        "qaoa": (
            frozenset({"graph_classes"}),
            frozenset({"er_edge_probability"}),
        ),
        "random_clifford": (frozenset(), frozenset()),
        "near_clifford": (
            frozenset({"non_clifford_count", "theta"}),
            frozenset(),
        ),
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain(nested) for nested in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def split_spec_hash(spec: SplitSpec | Mapping[str, Any]) -> str:
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)
    return canonical_hash(spec.to_dict())


def canonical_item_lines(items: Iterable[Mapping[str, Any]]) -> list[str]:
    ordered = sorted(items, key=lambda item: str(item["item_id"]))
    return [canonical_json(item) for item in ordered]


def item_stream_hash(items: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(canonical_item_lines(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cell_item_stream_hashes(
    items: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        by_cell.setdefault(str(item["cell_id"]), []).append(item)
    return {
        cell_id: item_stream_hash(rows)
        for cell_id, rows in sorted(by_cell.items())
    }


def split_dataset_hash(
    *,
    spec_hash: str,
    items_hash: str,
) -> str:
    return canonical_hash(
        {
            "split_spec_hash": spec_hash,
            "item_stream_hash": items_hash,
        }
    )


def _split_counts(items: list[dict]) -> dict[str, Any]:
    strata = sorted({str(item["stratum"]) for item in items})
    cells = sorted({str(item["cell_id"]) for item in items})
    return {
        "items": len(items),
        "items_by_role": {
            role: sum(item["split"] == role for item in items) for role in ROLES
        },
        "items_by_cell": {
            cell: sum(item["cell_id"] == cell for item in items) for cell in cells
        },
        "items_by_stratum": {
            stratum: sum(item["stratum"] == stratum for item in items)
            for stratum in strata
        },
        "circuits": len({item["circuit_id"] for item in items}),
        "measurement_groups": len(
            {item["measurement_group"] for item in items}
        ),
    }


def _group_evals(items: list[dict], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in sorted({str(item[field]) for item in items}):
        groups: dict[str, int] = {}
        for item in items:
            if str(item[field]) != value:
                continue
            group = str(item["measurement_group"])
            shots = item["shots"]
            if type(shots) is not int or shots < 1:
                raise ValueError(f"item {item.get('item_id', '?')} has invalid shots")
            if group in groups and groups[group] != shots:
                raise ValueError(
                    f"measurement group {group} has conflicting shots values"
                )
            groups[group] = shots
        result[value] = sum(groups.values())
    return result


def _label_evals(items: list[dict]) -> dict[str, int]:
    methods = sorted(set(FAMILY_LABEL_METHODS.values()))
    pairs = {
        method: {
            (str(item["circuit_id"]), str(item["observable_id"]))
            for item in items
            if item["label_method"] == method
        }
        for method in methods
    }
    return {method: len(pairs[method]) for method in methods}


def _split_generation_ledger(
    items: list[dict], label_evals: Mapping[str, int] | None = None
) -> dict[str, Any]:
    by_role = _group_evals(items, "split")
    labels = _label_evals(items) if label_evals is None else dict(label_evals)
    return {
        "circuit_evals_by_role": by_role,
        "circuit_evals_by_cell": _group_evals(items, "cell_id"),
        "circuit_evals_by_stratum": _group_evals(items, "stratum"),
        "total_circuit_evals": sum(by_role.values()),
        "label_evals_by_method": dict(sorted(labels.items())),
        "note": (
            "circuit evaluations are counted once per physical measurement group; "
            "exact-label calls are logged separately"
        ),
    }


def _axis_values(spec: SplitSpec, axis: str, domain: str) -> tuple[Any, ...]:
    if axis == spec.split_axis:
        declaration = (
            spec.source_domain if domain == "source" else spec.target_domain
        )
        return tuple(declaration[axis])
    return tuple(spec.fixed_axes[axis])


def _number_values(values: Iterable[Any], axis: str) -> tuple[float, ...]:
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{axis} values must be numeric") from exc
    return numbers


def _split_seed(master_seed: int, spawn_key: tuple[int, ...]) -> int:
    sequence = np.random.SeedSequence(master_seed, spawn_key=spawn_key)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def validate_split_spec(spec: SplitSpec | Mapping[str, Any]) -> None:
    """Validate one declaration against the closed S0 to S6 grammar."""
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)
    if spec.split_id not in SPLIT_AXES:
        raise ValueError(
            f"unknown split_id {spec.split_id!r}; expected one of "
            f"{sorted(SPLIT_AXES)}"
        )
    moved_axis = SPLIT_AXES[spec.split_id]
    expected_domain_axes = {moved_axis}
    source_axes = set(spec.source_domain)
    target_axes = set(spec.target_domain)
    if source_axes != expected_domain_axes or target_axes != expected_domain_axes:
        raise ValueError(
            f"{spec.split_id} domain declarations must contain only moved axis "
            f"{moved_axis!r}; source={sorted(source_axes)}, "
            f"target={sorted(target_axes)}"
        )
    expected_fixed = set(AXES) - ({moved_axis} if moved_axis in AXES else set())
    fixed_axes = set(spec.fixed_axes)
    unknown_axes = (source_axes | target_axes | fixed_axes) - (
        set(AXES) | {"circuit_instance"}
    )
    if unknown_axes:
        raise ValueError(f"unknown split axes: {sorted(unknown_axes)}")
    if fixed_axes != expected_fixed:
        missing = sorted(expected_fixed - fixed_axes)
        extra = sorted(fixed_axes - expected_fixed)
        raise ValueError(
            f"{spec.split_id} fixed axes do not match the closed grammar; "
            f"missing={missing}, extra={extra}"
        )

    if dict(spec.role_domains) != DEFAULT_ROLE_DOMAINS:
        raise ValueError(
            "role_domains must be train=source, validation=source, test=target"
        )
    if set(spec.role_counts) != set(ROLES):
        raise ValueError(f"role_counts must contain exactly {list(ROLES)}")
    bad_counts = {
        role: count for role, count in spec.role_counts.items() if count < 1
    }
    if bad_counts:
        raise ValueError(f"role_counts must be positive: {bad_counts}")
    if not spec.n_qubits or any(value < 1 for value in spec.n_qubits):
        raise ValueError("n_qubits must contain positive widths")
    if not spec.replicates or any(value < 0 for value in spec.replicates):
        raise ValueError("replicates must contain nonnegative integers")
    if spec.test_group_shot_budget is not None and spec.test_group_shot_budget < 1:
        raise ValueError("test_group_shot_budget must be positive when declared")
    if (
        spec.budget_tier is not None
        and spec.budget_tier.upper() not in TIERS
    ):
        raise ValueError(f"unknown budget_tier {spec.budget_tier!r}")

    expected_couplings = SPLIT_ALLOWED_COUPLINGS[spec.split_id]
    if spec.allowed_couplings != expected_couplings:
        raise ValueError(
            f"{spec.split_id} permits exactly these couplings: "
            f"{list(expected_couplings)}"
        )

    families = {
        str(value)
        for value in (
            *_axis_values(spec, "circuit_family", "source"),
            *_axis_values(spec, "circuit_family", "target"),
        )
    }
    if set(spec.family_parameters) != families:
        raise ValueError(
            "family_parameters must contain exactly the declared circuit families; "
            f"families={sorted(families)}"
        )
    unknown_families = sorted(families - set(_FAMILY_PARAMETER_KEY_CONTRACT))
    if unknown_families:
        raise ValueError(f"unknown circuit families: {unknown_families}")
    for family in sorted(families):
        parameters = spec.family_parameters[family]
        if not isinstance(parameters, Mapping):
            raise ValueError(f"family_parameters[{family!r}] must be a mapping")
        required, optional = _FAMILY_PARAMETER_KEY_CONTRACT[family]
        actual = set(parameters)
        missing = sorted(required - actual)
        extra = sorted(actual - required - optional)
        if missing or extra:
            raise ValueError(
                f"family_parameters[{family!r}] has invalid fields; "
                f"missing={missing}, extra={extra}"
            )

    for domain in ("source", "target"):
        shots = _axis_values(spec, "shots", domain)
        if any(type(value) is not int or value < 1 for value in shots):
            raise ValueError("shots values must be positive integers")
        depths = _axis_values(spec, "family_native_depth", domain)
        if any(type(value) is not int or value < 1 for value in depths):
            raise ValueError(
                "family_native_depth values must be positive integers"
            )

    from qemscore.circuits.heisenberg import (
        validate_heisenberg_sampling_domain,
    )
    from qemscore.circuits.near_clifford import (
        validate_near_clifford_sampling_domain,
    )
    from qemscore.circuits.qaoa import validate_qaoa_sampling_domain
    from qemscore.circuits.random_clifford import (
        validate_random_clifford_sampling_domain,
    )
    from qemscore.circuits.tfi import validate_tfi_sampling_domain

    depths = list(_axis_values(spec, "family_native_depth", "source"))
    for value in _axis_values(spec, "family_native_depth", "target"):
        if value not in depths:
            depths.append(value)
    for family in sorted(families):
        parameters = spec.family_parameters[family]
        for n_qubits in spec.n_qubits:
            widths = [n_qubits]
            if family == "tfi":
                validate_tfi_sampling_domain(widths, depths, parameters["dt"])
            elif family == "heisenberg":
                validate_heisenberg_sampling_domain(
                    widths,
                    depths,
                    parameters["dt"],
                )
            elif family == "qaoa":
                validate_qaoa_sampling_domain(
                    widths,
                    depths,
                    parameters["graph_classes"],
                    parameters.get("er_edge_probability"),
                )
            elif family == "random_clifford":
                validate_random_clifford_sampling_domain(widths, depths)
            elif family == "near_clifford":
                validate_near_clifford_sampling_domain(
                    widths,
                    depths,
                    parameters["non_clifford_count"],
                    parameters["theta"],
                )
            else:
                raise ValueError(f"unknown circuit family {family!r}")

    source_moved = tuple(spec.source_domain[moved_axis])
    target_moved = tuple(spec.target_domain[moved_axis])
    if spec.split_id == "S0":
        if source_moved != target_moved:
            raise ValueError(
                "S0 source and target circuit-instance distributions must match"
            )
    else:
        overlap = {
            canonical_json(value) for value in source_moved
        } & {canonical_json(value) for value in target_moved}
        if overlap:
            raise ValueError(
                f"{spec.split_id} source and target {moved_axis} values must be "
                "disjoint"
            )

    if spec.split_id == "S4":
        source = _number_values(source_moved, moved_axis)
        target = _number_values(target_moved, moved_axis)
        if min(target) <= max(source):
            raise ValueError(
                "S4 target family_native_depth values must all exceed source values"
            )
    if spec.split_id == "S6":
        source = _number_values(source_moved, moved_axis)
        target = _number_values(target_moved, moved_axis)
        if max(target) >= min(source):
            raise ValueError("S6 target shots values must all be below source values")


def _signature(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({canonical_json(value) for value in values}))


def _cell_axis_values(cell: MeasurementCell, axis: str) -> tuple[Any, ...]:
    value = cell.axis_values[axis]
    if axis == "observable_class" and isinstance(value, tuple):
        return value
    return (value,)


def _render_signature(values: tuple[str, ...]) -> list[Any]:
    return [json.loads(value) for value in values]


def _validate_closed_axis_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    """Recompute source and target signatures and reject undeclared changes."""
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)
    validate_split_spec(spec)
    cells = tuple(resolved_cells)
    if not cells:
        raise ValueError("resolved split must contain measurement cells")

    observed_roles = {cell.split for cell in cells}
    if observed_roles != set(ROLES):
        raise ValueError(
            f"resolved cells must contain exactly roles {list(ROLES)}; "
            f"found {sorted(observed_roles)}"
        )
    for cell in cells:
        if cell.split not in spec.role_domains:
            raise ValueError(f"cell {cell.cell_id} has unknown role {cell.split!r}")
        expected_domain = spec.role_domains[cell.split]
        if cell.domain != expected_domain:
            raise ValueError(
                f"cell {cell.cell_id} role {cell.split!r} requires domain "
                f"{expected_domain!r}, not {cell.domain!r}"
            )
        if set(cell.axis_values) != set(AXES):
            raise ValueError(
                f"cell {cell.cell_id} axis_values must contain exactly {list(AXES)}"
            )

    domain_signatures: dict[str, dict[str, tuple[str, ...]]] = {}
    for domain in ("source", "target"):
        domain_cells = [cell for cell in cells if cell.domain == domain]
        if not domain_cells:
            raise ValueError(f"resolved split has no {domain} cells")
        domain_signatures[domain] = {
            axis: _signature(
                value
                for cell in domain_cells
                for value in _cell_axis_values(cell, axis)
            )
            for axis in AXES
        }

    moved_axis = spec.split_axis
    for axis in AXES:
        source_signature = domain_signatures["source"][axis]
        target_signature = domain_signatures["target"][axis]
        if axis != moved_axis and source_signature != target_signature:
            raise ValueError(
                f"{spec.split_id} changes undeclared axis {axis!r}: "
                f"source={_render_signature(source_signature)}, "
                f"target={_render_signature(target_signature)}"
            )

        for domain, signature in (
            ("source", source_signature),
            ("target", target_signature),
        ):
            declaration = (
                _axis_values(spec, axis, domain)
                if axis == moved_axis
                else tuple(spec.fixed_axes[axis])
            )
            expected = _signature(declaration)
            if signature != expected:
                kind = "moved" if axis == moved_axis else "fixed"
                raise ValueError(
                    f"{spec.split_id} {domain} cells do not match declared {kind} "
                    f"axis {axis!r}: declared={_render_signature(expected)}, "
                    f"resolved={_render_signature(signature)}"
                )

    coupling_names = {
        name for cell in cells for name in cell.coupling_values
    }
    for name in sorted(coupling_names):
        source = _signature(
            cell.coupling_values[name]
            for cell in cells
            if cell.domain == "source" and name in cell.coupling_values
        )
        target = _signature(
            cell.coupling_values[name]
            for cell in cells
            if cell.domain == "target" and name in cell.coupling_values
        )
        if source != target and name not in spec.allowed_couplings:
            raise ValueError(
                f"{spec.split_id} changes undeclared coupling {name!r}: "
                f"source={_render_signature(source)}, "
                f"target={_render_signature(target)}"
            )

    source_moved = (
        _signature(spec.source_domain[moved_axis])
        if moved_axis in AXES
        else ()
    )
    if moved_axis in AXES:
        validation_signature = _signature(
            value
            for cell in cells
            if cell.split == "validation"
            for value in _cell_axis_values(cell, moved_axis)
        )
        if validation_signature != source_moved:
            raise ValueError(
                f"{spec.split_id} validation cells must use only source "
                f"{moved_axis} values"
            )


def _validate_named_contract(
    split_id: str,
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)
    if spec.split_id != split_id:
        raise ValueError(
            f"{split_id} validator cannot validate split_id {spec.split_id!r}"
        )
    _validate_closed_axis_contract(spec, resolved_cells)


def validate_s0_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S0", spec, resolved_cells)


def validate_s1_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S1", spec, resolved_cells)


def validate_s2_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S2", spec, resolved_cells)


def validate_s3_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S3", spec, resolved_cells)


def validate_s4_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S4", spec, resolved_cells)


def validate_s5_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S5", spec, resolved_cells)


def validate_s6_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    _validate_named_contract("S6", spec, resolved_cells)


_AXIS_VALIDATORS = {
    "S0": validate_s0_contract,
    "S1": validate_s1_contract,
    "S2": validate_s2_contract,
    "S3": validate_s3_contract,
    "S4": validate_s4_contract,
    "S5": validate_s5_contract,
    "S6": validate_s6_contract,
}


def validate_axis_contract(
    spec: SplitSpec | Mapping[str, Any],
    resolved_cells: Iterable[MeasurementCell],
) -> None:
    """Dispatch to the fixed validator selected by ``split_id``."""
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)
    validate_split_spec(spec)
    _AXIS_VALIDATORS[spec.split_id](spec, resolved_cells)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_GRAPH_CLASS_ALIASES = {
    "path": "path",
    "cycle": "cycle",
    "erdos_renyi": "erdos_renyi",
    "erdos-renyi": "erdos_renyi",
    "3_regular": "3_regular",
    "3-regular": "3_regular",
}


def _declared_float(value: object, *, field: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f"split parameter {field} must be a finite number")
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def _pool_parameter_mismatches(
    item: Mapping[str, Any], pool: Any
) -> dict[str, tuple[Any, Any]]:
    grid = pool.parameter_grid
    mismatches: dict[str, tuple[Any, Any]] = {}
    required, optional = _FAMILY_PARAMETER_KEY_CONTRACT[pool.family]
    required_grid = required | {"n_qubits", _FAMILY_DEPTH_FIELDS[pool.family]}
    actual_grid = set(grid)
    missing = sorted(required_grid - actual_grid)
    extra = sorted(actual_grid - required_grid - optional)
    if missing or extra:
        mismatches["parameter_grid.fields"] = (
            sorted(actual_grid),
            {"missing": missing, "extra": extra},
        )
        return mismatches
    if pool.family in {"tfi", "heisenberg"}:
        if _declared_float(item["dt"], field="dt") != _declared_float(
            grid["dt"], field="dt"
        ):
            mismatches["parameter_grid.dt"] = (item["dt"], grid["dt"])
    elif pool.family == "qaoa":
        declared = {
            _GRAPH_CLASS_ALIASES[str(value)] for value in grid["graph_classes"]
        }
        if item["graph_class"] not in declared:
            mismatches["parameter_grid.graph_classes"] = (
                item["graph_class"],
                sorted(declared),
            )
        fixed_probability = grid.get("er_edge_probability")
        if (
            item["graph_class"] == "erdos_renyi"
            and fixed_probability is not None
            and _declared_float(item["edge_probability"], field="edge_probability")
            != _declared_float(fixed_probability, field="er_edge_probability")
        ):
            mismatches["parameter_grid.er_edge_probability"] = (
                item["edge_probability"],
                fixed_probability,
            )
    elif pool.family == "near_clifford":
        declared_counts = set(grid["non_clifford_count"])
        if item["non_clifford_count"] not in declared_counts:
            mismatches["parameter_grid.non_clifford_count"] = (
                item["non_clifford_count"],
                sorted(declared_counts),
            )
        declared_thetas = {
            _declared_float(value, field="theta") for value in grid["theta"]
        }
        if _declared_float(item["theta"], field="theta") not in declared_thetas:
            mismatches["parameter_grid.theta"] = (
                item["theta"],
                sorted(declared_thetas),
            )
    return mismatches


def _validate_item_semantics(
    items: list[dict], manifest: Mapping[str, Any], expected_resolution: Any
) -> None:
    for item in items:
        validate_item(item, schema_version=manifest["dataset_schema_version"])
        if item.get("dataset_schema_version") != SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"item {item.get('item_id', '?')} has invalid "
                "dataset_schema_version"
            )
    validate_groups(items)

    expected_feature_spec = {
        "version": FEATURE_SPEC_VERSION,
        "features": list(FEATURES),
    }
    if manifest.get("feature_spec") != expected_feature_spec:
        raise ValueError("manifest feature_spec does not match the row schema")

    master_seed = manifest.get("master_seed")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("split-v2 manifest master_seed must be a nonnegative integer")
    expected_seed_scheme = {
        "version": "split-v1",
        "master_seed": master_seed,
        "formula": SPLIT_SEED_FORMULA,
    }
    if canonical_json(manifest.get("seed_scheme")) != canonical_json(
        expected_seed_scheme
    ):
        raise ValueError("split-v2 manifest seed_scheme does not match master_seed")

    cells_by_id = {
        cell.cell_id: cell for cell in expected_resolution.cells
    }
    pools_by_id = {
        pool.circuit_pool_id: pool for pool in expected_resolution.circuit_pools
    }
    observed_members: Counter[tuple[str, int, str]] = Counter()
    circuit_records: dict[tuple[str, int], tuple[Any, ...]] = {}

    for item in items:
        item_id = str(item["item_id"])
        cell = cells_by_id.get(item.get("cell_id"))
        if cell is None:
            raise ValueError(f"item {item_id} names an unknown cell")
        pool = pools_by_id[cell.circuit_pool_id]
        instance = item.get("instance")
        valid_instance = type(instance) is int and 0 <= instance < pool.n_instances
        expected = {
            "split_id": expected_resolution.spec.split_id,
            "split_axis": expected_resolution.spec.split_axis,
            "partition_id": cell.partition_id,
            "split": cell.split,
            "domain": cell.domain,
            "circuit_pool_id": cell.circuit_pool_id,
            "noise_config_id": cell.noise_config_id,
            "severity": cell.severity,
            "shots": cell.shots,
            "replicate": cell.replicate,
            "measurement_plan_id": cell.measurement_plan_id,
            "family": pool.family,
            "stratum": pool.stratum,
            "n_qubits": pool.n_qubits,
            "noise_family": cell.axis_values["noise_family"],
            "feature_spec_id": FEATURE_SPEC_VERSION,
            "raw_circuit_evals": cell.shots,
            _FAMILY_DEPTH_FIELDS[pool.family]: cell.axis_values[
                "family_native_depth"
            ],
        }
        if valid_instance:
            expected["circuit_seed"] = _split_seed(
                master_seed, tuple(pool.pool_seed_key) + (0, instance)
            )
            expected["sampler_seed"] = _split_seed(
                master_seed,
                tuple(cell.cell_seed_key) + (1, instance, cell.replicate, 0),
            )
        mismatches = {}
        for field, value in expected.items():
            expected_value = _plain(value)
            actual_value = item.get(field)
            if canonical_json(actual_value) != canonical_json(expected_value):
                mismatches[field] = (actual_value, expected_value)
        mismatches.update(_pool_parameter_mismatches(item, pool))
        expected_axes = cell.to_dict()["axis_values"]
        if canonical_json(item.get("axis_values")) != canonical_json(expected_axes):
            mismatches["axis_values"] = (item.get("axis_values"), expected_axes)
        observable_id = item.get("observable_id")
        if observable_id not in cell.observable_ids:
            mismatches["observable_id"] = (observable_id, list(cell.observable_ids))
        if item.get("observable") != observable_id:
            mismatches["observable"] = (item.get("observable"), observable_id)

        if not valid_instance:
            mismatches["instance"] = (instance, f"0 to {pool.n_instances - 1}")
        if mismatches:
            raise ValueError(f"item {item_id} disagrees with its cell: {mismatches}")

        descriptor = {
            "cell_id": cell.cell_id,
            "circuit_id": item["circuit_id"],
            "instance": instance,
            "observable_id": observable_id,
            "replicate": cell.replicate,
        }
        expected_item_id = f"item-{canonical_hash(descriptor)}"
        if item_id != expected_item_id:
            raise ValueError(
                f"item {item_id} has invalid item_id; expected {expected_item_id}"
            )
        expected_group = (
            f"{cell.cell_id}:{item['circuit_id']}:"
            f"i{instance}:r{cell.replicate}:g0"
        )
        if item.get("measurement_group") != expected_group:
            raise ValueError(
                f"item {item_id} has invalid measurement_group; "
                f"expected {expected_group}"
            )

        observed_members[(cell.cell_id, instance, str(observable_id))] += 1
        circuit_key = (pool.circuit_pool_id, instance)
        circuit_record = (
            item.get("circuit_id"),
            item.get("circuit_hash"),
            item.get("circuit_sidecar"),
            item.get("circuit_seed"),
        )
        previous = circuit_records.setdefault(circuit_key, circuit_record)
        if previous != circuit_record:
            raise ValueError(
                f"circuit pool {pool.circuit_pool_id} instance {instance} "
                "has inconsistent circuit metadata"
            )

    expected_members = Counter(
        (cell.cell_id, instance, observable_id)
        for cell in expected_resolution.cells
        for instance in range(pools_by_id[cell.circuit_pool_id].n_instances)
        for observable_id in cell.observable_ids
    )
    if observed_members != expected_members:
        raise ValueError("split rows do not exactly populate the resolved cells")


def _generator_observable_support(
    item: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> tuple[int, ...]:
    from qemscore.datasets.generate import _observable_support

    family = descriptor.get("family")
    n_qubits = descriptor.get("n_qubits")
    if family == "qaoa":
        parameters = descriptor.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("QAOA circuit parameters must be an object")
        params: Any = QAOAParams(
            n_qubits=n_qubits,
            graph_class=parameters["graph_class"],
            edges=tuple(tuple(edge) for edge in parameters["edges"]),
            p=parameters["p"],
            gammas=tuple(parameters["gammas"]),
            betas=tuple(parameters["betas"]),
            circuit_seed=descriptor["circuit_seed"],
            instance=0,
            edge_probability=parameters["edge_probability"],
        )
    else:
        params = SimpleNamespace(n_qubits=n_qubits)
    observable_id = item.get("observable_id")
    if observable_id is None:
        observable_id = item.get("observable")
    return tuple(_observable_support(observable_id, params))


def _expected_observable_semantics(
    item: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> tuple[tuple[int, ...], str]:
    item_id = str(item.get("item_id", "?"))
    observable_field = (
        "observable_id" if "observable_id" in item else "observable"
    )
    observable_id = item.get(observable_field)
    try:
        expected_support = _generator_observable_support(item, descriptor)
        expected_pauli = z_support_label(
            int(descriptor["n_qubits"]), expected_support
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"item {item_id} {observable_field} {observable_id!r} "
            f"is not defined for family {item.get('family')!r}: {exc}"
        ) from exc
    return expected_support, expected_pauli


def _validate_row_observable_semantics(
    item: Mapping[str, Any], expected_support: tuple[int, ...], expected_pauli: str
) -> None:
    item_id = str(item.get("item_id", "?"))
    observable_field = (
        "observable_id" if "observable_id" in item else "observable"
    )
    observable_id = item.get(observable_field)
    if item.get("pauli_label") != expected_pauli:
        raise ValueError(
            f"item {item_id} pauli_label disagrees with {observable_field} "
            f"{observable_id!r}: actual={item.get('pauli_label')!r}, "
            f"expected={expected_pauli!r}"
        )
    if item.get("obs_locality") != len(expected_support):
        raise ValueError(
            f"item {item_id} obs_locality disagrees with {observable_field} "
            f"{observable_id!r}: actual={item.get('obs_locality')!r}, "
            f"expected={len(expected_support)!r}"
        )


def _validate_row_ideal_semantics(
    item: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    expected_pauli: str,
    circuit_cache: dict[str, Any],
    label_cache: dict[tuple[str, str, str], float],
    *,
    circuit_key: str,
) -> None:
    item_id = str(item.get("item_id", "?"))
    if circuit_key not in circuit_cache:
        circuit_cache[circuit_key] = build_circuit_from_canonical_descriptor(
            descriptor
        )
    label_method = str(item["label_method"])
    label_key = (circuit_key, expected_pauli, label_method)
    if label_key not in label_cache:
        if label_method == "statevector":
            ideal = statevector_expectation(
                circuit_cache[circuit_key], expected_pauli
            )
        elif label_method == "stim":
            ideal = stim_expectation(circuit_cache[circuit_key], expected_pauli)
        else:
            raise ValueError(
                f"item {item_id} has unsupported label_method {label_method!r}"
            )
        label_cache[label_key] = round(float(ideal), 12)
    expected_ideal = label_cache[label_key]
    if item.get("ideal_expectation") != expected_ideal:
        raise ValueError(
            f"item {item_id} ideal_expectation disagrees with "
            f"{label_method} generator: "
            f"actual={item.get('ideal_expectation')!r}, "
            f"expected={expected_ideal!r}"
        )


def _validate_realized_row_semantics(
    item: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    circuit_cache: dict[str, Any],
    label_cache: dict[tuple[str, str, str], float],
    *,
    validate_ideal: bool = True,
) -> None:
    expected_support, expected_pauli = _expected_observable_semantics(
        item, descriptor
    )
    _validate_row_observable_semantics(item, expected_support, expected_pauli)
    if validate_ideal:
        _validate_row_ideal_semantics(
            item,
            descriptor,
            expected_pauli,
            circuit_cache,
            label_cache,
            circuit_key=canonical_json(descriptor),
        )


def _validate_sidecar_semantics(
    items: list[dict],
    sidecars: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> None:
    counts_reference_by_group: dict[object, tuple[object, object]] = {}
    for item in items:
        group = item["measurement_group"]
        reference = (item.get("counts_hash"), item.get("counts_sidecar"))
        previous = counts_reference_by_group.setdefault(group, reference)
        if previous != reference:
            raise ValueError(
                f"measurement group {group} references multiple counts draws"
            )

    circuit_cache: dict[str, Any] = {}
    label_cache: dict[tuple[str, str, str], float] = {}
    for item in items:
        item_id = str(item["item_id"])
        circuit = sidecars[str(item["circuit_sidecar"])]
        expected_circuit, expected_circuit_id = (
            canonical_physical_circuit_identity(item)
        )
        if canonical_json(circuit) != canonical_json(expected_circuit):
            raise ValueError(
                f"item {item_id} circuit sidecar is not the exact "
                "canonical descriptor"
            )
        if item["circuit_id"] != expected_circuit_id:
            raise ValueError(f"item {item_id} circuit_id does not match its sidecar")

        noise = sidecars[str(item["noise_sidecar"])]
        for field in ("noise_config_id", "noise_family", "severity"):
            if noise.get(field) != item.get(field):
                raise ValueError(
                    f"item {item_id} noise sidecar disagrees on {field}"
                )
        try:
            registry_parameters = registry["configs"][item["noise_family"]][
                item["severity"]
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"item {item_id} noise configuration is absent from the registry"
            ) from exc
        if noise.get("parameters") != registry_parameters:
            raise ValueError(
                f"item {item_id} noise sidecar parameters disagree with the registry"
            )

        observable = sidecars[str(item["observable_sidecar"])]
        for sidecar_field, row_field in (
            ("observable_id", "observable_id"),
            ("n_qubits", "n_qubits"),
            ("pauli_label", "pauli_label"),
        ):
            if observable.get(sidecar_field) != item.get(row_field):
                raise ValueError(
                    f"item {item_id} observable sidecar disagrees on {sidecar_field}"
                )
        expected_support, expected_pauli = _expected_observable_semantics(
            item, expected_circuit
        )
        support = observable.get("support")
        if support != list(expected_support):
            raise ValueError(
                f"item {item_id} observable support disagrees with observable_id "
                f"{item['observable_id']!r}: actual={support!r}, "
                f"expected={list(expected_support)!r}"
            )
        _validate_row_observable_semantics(
            item, expected_support, expected_pauli
        )
        circuit_id = str(item["circuit_id"])
        _validate_row_ideal_semantics(
            item,
            expected_circuit,
            expected_pauli,
            circuit_cache,
            label_cache,
            circuit_key=circuit_id,
        )

        counts_sidecar = sidecars[str(item["counts_sidecar"])]
        for field in ("cell_id", "circuit_id", "shots", "sampler_seed"):
            if counts_sidecar.get(field) != item.get(field):
                raise ValueError(
                    f"item {item_id} counts sidecar disagrees on {field}"
                )
        counts = counts_sidecar.get("counts")
        if not isinstance(counts, dict) or any(
            type(count) is not int or count < 0 for count in counts.values()
        ):
            raise ValueError(f"item {item_id} counts sidecar has invalid counts")
        for outcome in counts:
            if (
                not isinstance(outcome, str)
                or len(outcome) != item["n_qubits"]
                or set(outcome) - {"0", "1"}
            ):
                raise ValueError(
                    f"item {item_id} counts sidecar has an invalid outcome "
                    f"{outcome!r}"
                )
        if sum(counts.values()) != item["shots"]:
            raise ValueError(
                f"item {item_id} counts sidecar total does not equal shots"
            )
        noisy, stderr = z_expectation_from_counts(
            counts, expected_support, item["shots"]
        )
        if item.get("noisy_expectation") != round(noisy, 12):
            raise ValueError(
                f"item {item_id} noisy_expectation disagrees with counts sidecar"
            )
        if item.get("noisy_stderr") != round(stderr, 12):
            raise ValueError(
                f"item {item_id} noisy_stderr disagrees with counts sidecar"
            )


def _validate_split_tfi_profile_rows(
    items: list[dict], manifest: Mapping[str, Any], expected_resolution: Any
) -> None:
    from qemscore.circuits.tfi import sample_tfi_params
    from qemscore.datasets.generate import (
        PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD,
        _tfi_identity_fields_for_profile,
        validate_physical_identity_encoding_profiles,
    )

    if PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD not in manifest:
        return
    families = {pool.family for pool in expected_resolution.circuit_pools}
    profiles = validate_physical_identity_encoding_profiles(
        manifest[PHYSICAL_IDENTITY_ENCODING_PROFILES_FIELD],
        families=families,
    )
    profile = profiles.get("tfi")
    if profile is None:
        return

    rows_by_draw: dict[tuple[str, int], list[dict]] = {}
    for item in items:
        if item["family"] == "tfi":
            key = (str(item["circuit_pool_id"]), int(item["instance"]))
            rows_by_draw.setdefault(key, []).append(item)

    master = int(manifest["master_seed"])
    for pool in expected_resolution.circuit_pools:
        if pool.family != "tfi":
            continue
        grid = pool.parameter_grid
        for instance in range(pool.n_instances):
            spawn_key = tuple(pool.pool_seed_key) + (0, instance)
            sequence = np.random.SeedSequence(master, spawn_key=spawn_key)
            circuit_seed = int(
                sequence.generate_state(1, dtype=np.uint32)[0]
            )
            params = sample_tfi_params(
                np.random.default_rng(
                    np.random.SeedSequence(master, spawn_key=spawn_key)
                ),
                list(grid["n_qubits"]),
                list(grid["steps"]),
                grid["dt"],
                instance,
                circuit_seed,
            )
            expected = _tfi_identity_fields_for_profile(params, profile)
            rows = rows_by_draw.get((pool.circuit_pool_id, instance), [])
            if not rows or any(
                any(row[field] != value for field, value in expected.items())
                for row in rows
            ):
                raise ValueError(
                    "split TFI identity fields do not match the declared "
                    f"serializer profile {profile!r} for pool "
                    f"{pool.circuit_pool_id!r}, instance {instance}"
                )


def validate_split_artifact(data_dir: str | Path) -> tuple[list[dict], dict]:
    """Validate the split-v2 hash chain and closed axis declaration."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("dataset_schema_version")
    if version != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"split artifact requires dataset_schema_version {SPLIT_SCHEMA_VERSION!r}"
        )
    try:
        validate_environment_contract(manifest["environment_contract"])
    except KeyError as exc:
        raise ValueError("split-v2 manifest requires environment_contract") from exc
    items = [
        json.loads(line)
        for line in (root / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("dataset contains duplicate item_id values")

    spec = SplitSpec.from_dict(manifest["split_spec"])
    actual_spec_hash = split_spec_hash(spec)
    if manifest.get("split_spec_hash") != actual_spec_hash:
        raise ValueError(
            "split_spec_hash mismatch: "
            f"manifest={manifest.get('split_spec_hash')}, actual={actual_spec_hash}"
        )
    validate_split_spec(spec)

    expected_resolution = resolve_split_spec(spec)
    expected_pools = [pool.to_dict() for pool in expected_resolution.circuit_pools]
    if manifest.get("circuit_pools") != expected_pools:
        raise ValueError("resolved circuit_pools do not match split_spec")

    cells: list[MeasurementCell] = []
    stored_cell_hashes: dict[str, str] = {}
    stored_cells_without_hashes: list[dict[str, Any]] = []
    for value in manifest.get("cells", []):
        payload = dict(value)
        stored_cell_hashes[str(payload["cell_id"])] = str(
            payload.pop("item_stream_hash")
        )
        stored_cells_without_hashes.append(payload)
        cells.append(MeasurementCell(**payload))
    expected_cells = [cell.to_dict() for cell in expected_resolution.cells]
    if stored_cells_without_hashes != expected_cells:
        raise ValueError("resolved cells do not match split_spec")
    validate_axis_contract(spec, cells)

    # Integrity before semantics. An unrehashed mutation must fail the hash
    # chain rather than a row-semantic check, so tampering is reported as
    # tampering. A fully rehashed row passes integrity here and is then caught
    # by validate_item, which is where an impossible value belongs.
    actual_cell_hashes = cell_item_stream_hashes(items)
    if actual_cell_hashes != stored_cell_hashes:
        differing = sorted(
            set(actual_cell_hashes) | set(stored_cell_hashes)
        )
        differing = [
            cell_id
            for cell_id in differing
            if actual_cell_hashes.get(cell_id) != stored_cell_hashes.get(cell_id)
        ]
        raise ValueError(f"cell item_stream_hash mismatch for cells: {differing}")

    actual_items_hash = item_stream_hash(items)
    if manifest.get("items_hash") != actual_items_hash:
        raise ValueError(
            f"items_hash mismatch: manifest={manifest.get('items_hash')}, "
            f"actual={actual_items_hash}"
        )

    _validate_item_semantics(items, manifest, expected_resolution)

    actual_counts = _split_counts(items)
    if manifest.get("counts") != actual_counts:
        raise ValueError(
            f"manifest counts do not match rows: actual={actual_counts}"
        )
    actual_ledger = _split_generation_ledger(items)
    if manifest.get("generation_ledger") != actual_ledger:
        raise ValueError(
            "manifest generation_ledger does not match row-derived accounting"
        )
    declared_test_cap = spec.test_group_shot_budget
    actual_test_cost = actual_ledger["circuit_evals_by_role"]["test"]
    if declared_test_cap is not None and actual_test_cost > declared_test_cap:
        raise ValueError(
            "test circuit-evaluation budget exceeded: "
            f"declared={declared_test_cap}, actual={actual_test_cost}"
        )

    stored_sidecars = manifest.get("sidecar_hashes")
    if not isinstance(stored_sidecars, dict):
        raise ValueError("manifest sidecar_hashes must be an object")
    reference_fields = (
        ("circuit_sidecar", "circuit_hash"),
        ("noise_sidecar", "noise_config_hash"),
        ("observable_sidecar", "observable_hash"),
        ("counts_sidecar", "counts_hash"),
    )
    referenced_sidecars: dict[str, str] = {}
    for item in items:
        for path_field, hash_field in reference_fields:
            if path_field not in item or hash_field not in item:
                raise ValueError(
                    f"item {item.get('item_id', '?')} missing sidecar reference "
                    f"{path_field!r} or {hash_field!r}"
                )
            relative = str(item[path_field])
            digest = str(item[hash_field])
            if relative in referenced_sidecars and referenced_sidecars[relative] != digest:
                raise ValueError(f"conflicting row hashes for sidecar {relative}")
            referenced_sidecars[relative] = digest
    if referenced_sidecars != stored_sidecars:
        raise ValueError("row sidecar references do not match manifest sidecar_hashes")

    registry_configs = json.loads(canonical_json(SEVERITY_GRIDS))
    expected_registry = {
        "version": "severity-grid-v1",
        "hash": canonical_hash(registry_configs),
        "configs": registry_configs,
    }
    registry = manifest.get("noise_registry")
    if registry != expected_registry:
        raise ValueError(
            "noise_registry does not match the installed severity generator"
        )
    root_resolved = root.resolve()
    actual_sidecars: dict[str, str] = {}
    sidecar_payloads: dict[str, Mapping[str, Any]] = {}
    for relative, expected in sorted(stored_sidecars.items()):
        path = (root / relative).resolve()
        if root_resolved not in path.parents:
            raise ValueError(f"sidecar path escapes dataset directory: {relative!r}")
        if not path.is_file():
            raise ValueError(f"sidecar is missing: {relative}")
        actual = _file_hash(path)
        actual_sidecars[relative] = actual
        if actual != expected:
            raise ValueError(
                f"sidecar hash mismatch for {relative}: "
                f"manifest={expected}, actual={actual}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"sidecar is not valid JSON: {relative}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"sidecar must contain an object: {relative}")
        sidecar_payloads[relative] = payload

    _validate_sidecar_semantics(items, sidecar_payloads, registry)
    _validate_split_tfi_profile_rows(items, manifest, expected_resolution)

    # Local import: generate.py imports validation.py, so a module-level import
    # here would close the cycle. Ordering matters and is deliberate. An
    # unrehashed edit fails the hash chain, a rehashed but sidecar-inconsistent
    # edit fails sidecar semantics, and only a fully consistent artifact reaches
    # the closure check, which is the case this catches.
    from qemscore.datasets.generate import _require_source_observable_closure

    _require_source_observable_closure(items, split_name=spec.split_id)

    actual_dataset_hash = split_dataset_hash(
        spec_hash=actual_spec_hash,
        items_hash=actual_items_hash,
    )
    if manifest.get("dataset_hash") != actual_dataset_hash:
        raise ValueError(
            f"dataset_hash mismatch: manifest={manifest.get('dataset_hash')}, "
            f"actual={actual_dataset_hash}"
        )
    return sorted(items, key=lambda item: item["item_id"]), manifest


__all__ = [
    "LEGACY_SCHEMA_VERSION",
    "SPLIT_SEED_FORMULA",
    "SPLIT_SCHEMA_VERSION",
    "canonical_hash",
    "canonical_item_lines",
    "canonical_json",
    "cell_item_stream_hashes",
    "item_stream_hash",
    "split_dataset_hash",
    "split_spec_hash",
    "validate_axis_contract",
    "validate_s0_contract",
    "validate_s1_contract",
    "validate_s2_contract",
    "validate_s3_contract",
    "validate_s4_contract",
    "validate_s5_contract",
    "validate_s6_contract",
    "validate_split_artifact",
    "validate_split_spec",
]
