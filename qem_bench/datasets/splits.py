"""Closed S0 to S6 split grammar and deterministic resolution.

``SplitSpec`` is the authored declaration. ``resolve_split_spec`` compiles it
into circuit pools and atomic measurement cells. The resolved records are
immutable and their identifiers and seed keys depend on content, not authored
list order.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping

AXES = (
    "noise_family",
    "noise_strength",
    "circuit_family",
    "family_native_depth",
    "observable_class",
    "shots",
)
ROLES = ("train", "validation", "test")
DOMAINS = ("source", "target")
SPLIT_AXES = {
    "S0": "circuit_instance",
    "S1": "noise_family",
    "S2": "noise_strength",
    "S3": "circuit_family",
    "S4": "family_native_depth",
    "S5": "observable_class",
    "S6": "shots",
}
SPLIT_ALLOWED_COUPLINGS = {
    "S0": (),
    "S1": (),
    "S2": (),
    "S3": ("family_native_parameters",),
    "S4": (),
    "S5": ("aggregate_definition", "observable_locality", "observable_norm"),
    "S6": (),
}
DEFAULT_ROLE_DOMAINS = {
    "train": "source",
    "validation": "source",
    "test": "target",
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain(nested) for nested in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("SplitSpec numeric values must be finite")
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        frozen = tuple(_freeze_value(nested) for nested in value)
        return tuple(sorted(frozen, key=_canonical_json))
    return value


def _sorted_values(values: Any) -> tuple[Any, ...]:
    if isinstance(values, (str, int, float, bool)) or values is None:
        candidates = (values,)
    else:
        candidates = tuple(values)
    if not candidates:
        raise ValueError("axis value sets must not be empty")
    unique: dict[str, Any] = {}
    for value in candidates:
        frozen = _freeze_value(value)
        unique[_canonical_json(frozen)] = frozen
    return tuple(unique[key] for key in sorted(unique))


def _integer_values(
    values: Any, *, field: str, minimum: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of integers")
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of integers") from exc
    if not candidates:
        raise ValueError(f"{field} must not be empty")
    for value in candidates:
        if type(value) is not int or value < minimum:
            bound = "positive" if minimum == 1 else "nonnegative"
            raise ValueError(f"{field} must contain {bound} integers")
    return tuple(sorted(set(candidates)))


def _integer_value(value: Any, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        bound = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{field} must be a {bound} integer")
    return value


def _axis_map(values: Mapping[str, Any]) -> Mapping[str, tuple[Any, ...]]:
    return MappingProxyType(
        {
            str(axis): _sorted_values(axis_values)
            for axis, axis_values in sorted(values.items())
        }
    )


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        frozen[str(key)] = _freeze_value(value)
    return MappingProxyType(frozen)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed_key(value: Any) -> tuple[int, ...]:
    digest = bytes.fromhex(_content_hash(value))
    return tuple(
        int.from_bytes(digest[offset : offset + 4], "big")
        for offset in range(0, 16, 4)
    )


@dataclass(frozen=True)
class SplitSpec:
    """Declarative, closed split specification.

    ``budget_tier`` and ``test_group_shot_budget`` remain authored budget
    contract parameters. The ``observable_class`` axis and ``n_qubits`` remain
    authored scope parameters. The grammar assigns no paper-scale defaults.
    """

    split_id: str
    source_domain: Mapping[str, Any]
    target_domain: Mapping[str, Any]
    fixed_axes: Mapping[str, Any]
    n_qubits: tuple[int, ...] | list[int]
    role_counts: Mapping[str, int]
    family_parameters: Mapping[str, Mapping[str, Any]]
    role_domains: Mapping[str, str] = MappingProxyType(DEFAULT_ROLE_DOMAINS)
    allowed_couplings: tuple[str, ...] | list[str] = ()
    replicates: tuple[int, ...] | list[int] = (0,)
    partition_id: str = "headline"
    budget_tier: str | None = None
    test_group_shot_budget: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_id", str(self.split_id).upper())
        object.__setattr__(self, "source_domain", _axis_map(self.source_domain))
        object.__setattr__(self, "target_domain", _axis_map(self.target_domain))
        object.__setattr__(self, "fixed_axes", _axis_map(self.fixed_axes))
        object.__setattr__(
            self,
            "n_qubits",
            _integer_values(self.n_qubits, field="n_qubits", minimum=1),
        )
        role_counts = {
            str(role): _integer_value(
                count, field=f"role_counts[{role!r}]", minimum=1
            )
            for role, count in sorted(self.role_counts.items())
        }
        object.__setattr__(
            self,
            "role_counts",
            MappingProxyType(role_counts),
        )
        object.__setattr__(
            self, "family_parameters", _frozen_mapping(self.family_parameters)
        )
        object.__setattr__(
            self,
            "role_domains",
            MappingProxyType(
                {
                    str(role): str(domain)
                    for role, domain in sorted(self.role_domains.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "allowed_couplings",
            tuple(sorted({str(value) for value in self.allowed_couplings})),
        )
        object.__setattr__(
            self,
            "replicates",
            _integer_values(self.replicates, field="replicates", minimum=0),
        )
        object.__setattr__(self, "partition_id", str(self.partition_id))
        if self.budget_tier is not None:
            if type(self.budget_tier) is not str:
                raise ValueError("budget_tier must be a string when declared")
        if self.test_group_shot_budget is not None:
            object.__setattr__(
                self,
                "test_group_shot_budget",
                _integer_value(
                    self.test_group_shot_budget,
                    field="test_group_shot_budget",
                    minimum=1,
                ),
            )

    @property
    def split_axis(self) -> str:
        try:
            return SPLIT_AXES[self.split_id]
        except KeyError as exc:
            raise ValueError(f"unknown split_id {self.split_id!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "split_axis": self.split_axis,
            "role_domains": _plain(self.role_domains),
            "source_domain": _plain(self.source_domain),
            "target_domain": _plain(self.target_domain),
            "fixed_axes": _plain(self.fixed_axes),
            "n_qubits": list(self.n_qubits),
            "role_counts": _plain(self.role_counts),
            "family_parameters": _plain(self.family_parameters),
            "allowed_couplings": list(self.allowed_couplings),
            "replicates": list(self.replicates),
            "partition_id": self.partition_id,
            "budget_tier": self.budget_tier,
            "test_group_shot_budget": self.test_group_shot_budget,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitSpec:
        payload = dict(value)
        declared_axis = payload.pop("split_axis", None)
        spec = cls(**payload)
        if declared_axis is not None and declared_axis != spec.split_axis:
            raise ValueError(
                f"{spec.split_id} requires split_axis {spec.split_axis!r}, "
                f"not {declared_axis!r}"
            )
        return spec


@dataclass(frozen=True)
class CircuitPool:
    circuit_pool_id: str
    partition_id: str
    split: str
    domain: str
    stratum: str
    family: str
    n_qubits: int
    parameter_grid: Mapping[str, Any]
    n_instances: int
    pool_seed_key: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_grid", _frozen_mapping(self.parameter_grid))

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_pool_id": self.circuit_pool_id,
            "partition_id": self.partition_id,
            "split": self.split,
            "domain": self.domain,
            "stratum": self.stratum,
            "family": self.family,
            "n_qubits": self.n_qubits,
            "parameter_grid": _plain(self.parameter_grid),
            "n_instances": self.n_instances,
            "pool_seed_key": list(self.pool_seed_key),
        }


@dataclass(frozen=True)
class MeasurementCell:
    cell_id: str
    circuit_pool_id: str
    partition_id: str
    split: str
    domain: str
    stratum: str
    axis_values: Mapping[str, Any]
    coupling_values: Mapping[str, Any]
    noise_config_id: str
    severity: str
    shots: int
    replicate: int
    observable_ids: tuple[str, ...]
    measurement_plan_id: str
    cell_seed_key: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_values", _frozen_mapping(self.axis_values))
        object.__setattr__(
            self, "coupling_values", _frozen_mapping(self.coupling_values)
        )
        object.__setattr__(self, "observable_ids", tuple(self.observable_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "circuit_pool_id": self.circuit_pool_id,
            "partition_id": self.partition_id,
            "split": self.split,
            "domain": self.domain,
            "stratum": self.stratum,
            "axis_values": _plain(self.axis_values),
            "coupling_values": _plain(self.coupling_values),
            "noise_config_id": self.noise_config_id,
            "severity": self.severity,
            "shots": self.shots,
            "replicate": self.replicate,
            "observable_ids": list(self.observable_ids),
            "measurement_plan_id": self.measurement_plan_id,
            "cell_seed_key": list(self.cell_seed_key),
        }


@dataclass(frozen=True)
class ResolvedSplit:
    spec: SplitSpec
    circuit_pools: tuple[CircuitPool, ...]
    cells: tuple[MeasurementCell, ...]


def _depth_parameter(family: str) -> str:
    if family in {"tfi", "heisenberg"}:
        return "steps"
    if family == "qaoa":
        return "p"
    if family in {"random_clifford", "near_clifford"}:
        return "depth"
    raise ValueError(f"unknown circuit family {family!r}")


def _role_axes(spec: SplitSpec, role: str) -> dict[str, tuple[Any, ...]]:
    axes = {axis: tuple(values) for axis, values in spec.fixed_axes.items()}
    moved_axis = spec.split_axis
    if moved_axis in AXES:
        domain = spec.role_domains[role]
        declaration = (
            spec.source_domain if domain == "source" else spec.target_domain
        )
        axes[moved_axis] = tuple(declaration[moved_axis])
    return axes


def resolve_split_spec(spec: SplitSpec | Mapping[str, Any]) -> ResolvedSplit:
    """Compile a declaration into canonical circuit pools and measurement cells."""
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec.from_dict(spec)

    from qem_bench.validation import validate_axis_contract, validate_split_spec

    validate_split_spec(spec)
    pools: list[CircuitPool] = []
    cells: list[MeasurementCell] = []

    for role in ROLES:
        domain = spec.role_domains[role]
        axes = _role_axes(spec, role)
        families = axes["circuit_family"]
        depths = axes["family_native_depth"]
        role_pool_descriptors: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for family, depth, n_qubits in product(families, depths, spec.n_qubits):
            family = str(family)
            family_parameters = _plain(spec.family_parameters[family])
            parameter_grid = {
                **family_parameters,
                "n_qubits": [int(n_qubits)],
                _depth_parameter(family): [depth],
            }
            pool_descriptor = {
                "partition_id": spec.partition_id,
                "split": role,
                "domain": domain,
                "family": family,
                "n_qubits": int(n_qubits),
                "parameter_grid": parameter_grid,
            }
            pool_id = f"pool-{_content_hash(pool_descriptor)}"
            role_pool_descriptors.append(
                (pool_id, pool_descriptor, family_parameters)
            )

        role_pool_descriptors.sort(key=lambda value: value[0])
        total_instances = spec.role_counts[role]
        if total_instances < len(role_pool_descriptors):
            raise ValueError(
                f"role_counts[{role!r}]={total_instances} cannot populate "
                f"{len(role_pool_descriptors)} circuit pools"
            )
        base_instances, remainder = divmod(
            total_instances, len(role_pool_descriptors)
        )

        for pool_index, (
            pool_id,
            pool_descriptor,
            family_parameters,
        ) in enumerate(role_pool_descriptors):
            family = str(pool_descriptor["family"])
            n_qubits = int(pool_descriptor["n_qubits"])
            parameter_grid = pool_descriptor["parameter_grid"]
            depth = parameter_grid[_depth_parameter(family)][0]
            stratum = (
                "clifford_control"
                if family == "random_clifford"
                else "continuous_regression"
            )
            pool = CircuitPool(
                circuit_pool_id=pool_id,
                partition_id=spec.partition_id,
                split=role,
                domain=domain,
                stratum=stratum,
                family=family,
                n_qubits=int(n_qubits),
                parameter_grid=parameter_grid,
                n_instances=base_instances + (pool_index < remainder),
                pool_seed_key=_seed_key(pool_descriptor),
            )
            pools.append(pool)

            observable_ids = tuple(str(value) for value in axes["observable_class"])
            for noise_family, severity, shots, replicate in product(
                axes["noise_family"],
                axes["noise_strength"],
                axes["shots"],
                spec.replicates,
            ):
                axis_values = {
                    "noise_family": str(noise_family),
                    "noise_strength": str(severity),
                    "circuit_family": family,
                    "family_native_depth": depth,
                    "observable_class": list(observable_ids),
                    "shots": int(shots),
                }
                coupling_values = {
                    "circuit_width": int(n_qubits),
                    "family_native_parameters": family_parameters,
                    "partition_id": spec.partition_id,
                    "replicate": int(replicate),
                    "stratum": stratum,
                }
                noise_descriptor = {
                    "noise_family": str(noise_family),
                    "severity": str(severity),
                }
                noise_config_id = f"noise-{_content_hash(noise_descriptor)}"
                cell_descriptor = {
                    "circuit_pool_id": pool_id,
                    "partition_id": spec.partition_id,
                    "split": role,
                    "domain": domain,
                    "axis_values": axis_values,
                    "coupling_values": coupling_values,
                    "noise_config_id": noise_config_id,
                    "replicate": int(replicate),
                    "measurement_plan_id": "z-computational-v1",
                }
                cell_id = f"cell-{_content_hash(cell_descriptor)}"
                cells.append(
                    MeasurementCell(
                        cell_id=cell_id,
                        circuit_pool_id=pool_id,
                        partition_id=spec.partition_id,
                        split=role,
                        domain=domain,
                        stratum=stratum,
                        axis_values=axis_values,
                        coupling_values=coupling_values,
                        noise_config_id=noise_config_id,
                        severity=str(severity),
                        shots=int(shots),
                        replicate=int(replicate),
                        observable_ids=observable_ids,
                        measurement_plan_id="z-computational-v1",
                        cell_seed_key=_seed_key(cell_descriptor),
                    )
                )

    resolved = ResolvedSplit(
        spec=spec,
        circuit_pools=tuple(sorted(pools, key=lambda value: value.circuit_pool_id)),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
    )
    validate_axis_contract(spec, resolved.cells)
    return resolved


__all__ = [
    "AXES",
    "DOMAINS",
    "ROLES",
    "SPLIT_ALLOWED_COUPLINGS",
    "SPLIT_AXES",
    "CircuitPool",
    "MeasurementCell",
    "ResolvedSplit",
    "SplitSpec",
    "resolve_split_spec",
]
