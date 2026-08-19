"""Closed split-contract and split-v2 integrity validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from qem_bench.datasets.schema import (
    FAMILY_LABEL_METHODS,
    FEATURE_SPEC_VERSION,
    FEATURES,
    validate_groups,
    validate_item,
)
from qem_bench.datasets.splits import (
    AXES,
    DEFAULT_ROLE_DOMAINS,
    ROLES,
    SPLIT_ALLOWED_COUPLINGS,
    SPLIT_AXES,
    MeasurementCell,
    SplitSpec,
    resolve_split_spec,
)
from qem_bench.observables import z_expectation_from_counts
from qem_bench.reproducibility import validate_environment_contract

SPLIT_SCHEMA_VERSION = "split-v2"
LEGACY_SCHEMA_VERSION = "legacy-v1"


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

    expected_couplings = SPLIT_ALLOWED_COUPLINGS[spec.split_id]
    if spec.allowed_couplings != expected_couplings:
        raise ValueError(
            f"{spec.split_id} permits exactly these couplings: "
            f"{list(expected_couplings)}"
        )

    families = set(_axis_values(spec, "circuit_family", "source")) | set(
        _axis_values(spec, "circuit_family", "target")
    )
    if set(spec.family_parameters) != families:
        raise ValueError(
            "family_parameters must contain exactly the declared circuit families; "
            f"families={sorted(str(value) for value in families)}"
        )

    for domain in ("source", "target"):
        shots = _number_values(_axis_values(spec, "shots", domain), "shots")
        if any(value < 1 or not value.is_integer() for value in shots):
            raise ValueError("shots values must be positive integers")
        depths = _number_values(
            _axis_values(spec, "family_native_depth", domain),
            "family_native_depth",
        )
        if any(value < 1 for value in depths):
            raise ValueError("family_native_depth values must be positive")

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


def _validate_item_semantics(
    items: list[dict], manifest: Mapping[str, Any], expected_resolution: Any
) -> None:
    for item in items:
        validate_item(item)
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
        }
        mismatches = {}
        for field, value in expected.items():
            expected_value = _plain(value)
            actual_value = item.get(field)
            if canonical_json(actual_value) != canonical_json(expected_value):
                mismatches[field] = (actual_value, expected_value)
        expected_axes = cell.to_dict()["axis_values"]
        if canonical_json(item.get("axis_values")) != canonical_json(expected_axes):
            mismatches["axis_values"] = (item.get("axis_values"), expected_axes)
        observable_id = item.get("observable_id")
        if observable_id not in cell.observable_ids:
            mismatches["observable_id"] = (observable_id, list(cell.observable_ids))
        if item.get("observable") != observable_id:
            mismatches["observable"] = (item.get("observable"), observable_id)

        instance = item.get("instance")
        if type(instance) is not int or not 0 <= instance < pool.n_instances:
            mismatches["instance"] = (instance, f"0 to {pool.n_instances - 1}")
        if mismatches:
            raise ValueError(f"item {item_id} disagrees with its cell: {mismatches}")

        descriptor = {
            "cell_id": cell.cell_id,
            "circuit_id": item["circuit_id"],
            "observable_id": observable_id,
            "replicate": cell.replicate,
        }
        expected_item_id = f"item-{canonical_hash(descriptor)}"
        if item_id != expected_item_id:
            raise ValueError(
                f"item {item_id} has invalid item_id; expected {expected_item_id}"
            )
        expected_group = (
            f"{cell.cell_id}:{item['circuit_id']}:r{cell.replicate}:g0"
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


def _validate_sidecar_semantics(
    items: list[dict],
    sidecars: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> None:
    for item in items:
        item_id = str(item["item_id"])
        circuit = sidecars[str(item["circuit_sidecar"])]
        expected_circuit = {
            "family": item["family"],
            "n_qubits": item["n_qubits"],
            "circuit_seed": item["circuit_seed"],
        }
        for field, expected in expected_circuit.items():
            if circuit.get(field) != expected:
                raise ValueError(
                    f"item {item_id} circuit sidecar disagrees on {field}"
                )
        parameters = circuit.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"item {item_id} circuit sidecar parameters are invalid")
        for field, expected in parameters.items():
            if item.get(field) != expected:
                raise ValueError(
                    f"item {item_id} circuit sidecar disagrees on parameter {field}"
                )
        expected_circuit_id = f"circuit-{canonical_hash(circuit)}"
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
        support = observable.get("support")
        if not isinstance(support, list) or item.get("obs_locality") != len(support):
            raise ValueError(
                f"item {item_id} observable sidecar disagrees on obs_locality"
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
        if sum(counts.values()) != item["shots"]:
            raise ValueError(
                f"item {item_id} counts sidecar total does not equal shots"
            )
        noisy, stderr = z_expectation_from_counts(
            counts, tuple(support), item["shots"]
        )
        if item.get("noisy_expectation") != round(noisy, 12):
            raise ValueError(
                f"item {item_id} noisy_expectation disagrees with counts sidecar"
            )
        if item.get("noisy_stderr") != round(stderr, 12):
            raise ValueError(
                f"item {item_id} noisy_stderr disagrees with counts sidecar"
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
    validate_split_spec(spec)
    actual_spec_hash = split_spec_hash(spec)
    if manifest.get("split_spec_hash") != actual_spec_hash:
        raise ValueError(
            "split_spec_hash mismatch: "
            f"manifest={manifest.get('split_spec_hash')}, actual={actual_spec_hash}"
        )

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

    registry = manifest.get("noise_registry")
    if not isinstance(registry, dict) or canonical_hash(
        registry.get("configs")
    ) != registry.get("hash"):
        raise ValueError("noise_registry hash does not match embedded configs")
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
