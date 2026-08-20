"""Closed dataset-row schemas by circuit family and artifact version."""

from __future__ import annotations

import importlib
import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qem_bench.circuits import (
    HeisenbergParams,
    NearCliffordParams,
    QAOAParams,
    RandomCliffordParams,
    TFIParams,
    build_heisenberg_circuit,
    build_near_clifford_circuit,
    build_qaoa_circuit,
    build_random_clifford_circuit,
    build_tfi_circuit,
)
from qem_bench.datasets.generate import (
    _family_parameter_fields,
    dataset_hash,
    generate,
)
from qem_bench.datasets.schema import (
    FAMILY_REQUIRED_FIELDS,
    LEGACY_SCHEMA_VERSION,
    SPLIT_ITEM_FIELDS,
    SPLIT_SCHEMA_VERSION,
    build_circuit_from_canonical_descriptor,
    canonical_physical_circuit_descriptor,
    canonical_physical_circuit_identity,
    validate_item,
)
from qem_bench.datasets.splits import SplitSpec
from qem_bench.noise.models import DEFAULT_NOISE_FAMILY
from qem_bench.runner.run import _load
from qem_bench.validation import (
    canonical_item_lines,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    validate_split_artifact,
)

FAMILY_PRESETS = {
    "tfi": "t0-micro",
    "qaoa": "t0-qaoa-micro",
    "heisenberg": "t0-heisenberg-micro",
    "random_clifford": "t0-rc-micro",
    "near_clifford": "t0-nc-micro",
}
ALL_FAMILY_FIELDS = frozenset(
    field for fields in FAMILY_REQUIRED_FIELDS.values() for field in fields
)
TFI_FIELDS = frozenset(FAMILY_REQUIRED_FIELDS["tfi"])
FOREIGN_FIELD_CASES = tuple(
    (
        "legacy-qaoa" if field in TFI_FIELDS else "split-tfi",
        field,
    )
    for field in sorted(ALL_FAMILY_FIELDS)
)


def _read_artifact(data: Path) -> tuple[list[dict], dict]:
    items = [
        json.loads(line)
        for line in (data / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    return items, manifest


def _write_manifest(data: Path, manifest: dict) -> None:
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _write_legacy_artifact(data: Path, items: list[dict], manifest: dict) -> None:
    (data / "items.jsonl").write_text(
        "\n".join(canonical_item_lines(items)) + "\n", encoding="utf-8"
    )
    manifest["dataset_hash"] = dataset_hash(items)
    _write_manifest(data, manifest)


def _write_split_artifact(data: Path, items: list[dict], manifest: dict) -> None:
    (data / "items.jsonl").write_text(
        "\n".join(canonical_item_lines(items)) + "\n", encoding="utf-8"
    )
    manifest["items_hash"] = item_stream_hash(items)
    cell_hashes = cell_item_stream_hashes(items)
    for cell in manifest["cells"]:
        cell["item_stream_hash"] = cell_hashes[cell["cell_id"]]
    manifest["dataset_hash"] = split_dataset_hash(
        spec_hash=manifest["split_spec_hash"],
        items_hash=manifest["items_hash"],
    )
    _write_manifest(data, manifest)


@pytest.fixture(scope="module")
def schema_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("schema-closure")
    legacy = {}
    for family, preset in FAMILY_PRESETS.items():
        data = root / preset
        generate(preset, data)
        legacy[family] = data
    split = root / "s0-t0-micro"
    generate("s0-t0-micro", split)
    split_qaoa = root / "s0-qaoa-micro"
    generate(
        SplitSpec(
            split_id="S0",
            source_domain={"circuit_instance": ["sampled"]},
            target_domain={"circuit_instance": ["sampled"]},
            fixed_axes={
                "noise_family": [DEFAULT_NOISE_FAMILY],
                "noise_strength": ["L1"],
                "circuit_family": ["qaoa"],
                "family_native_depth": [1],
                "observable_class": ["z_mid"],
                "shots": [64],
            },
            n_qubits=[4],
            role_counts={"train": 1, "validation": 1, "test": 1},
            family_parameters={
                "qaoa": {
                    "graph_classes": ["erdos_renyi"],
                    "er_edge_probability": 0.5,
                }
            },
        ),
        split_qaoa,
    )
    split_near_clifford = root / "s0-near-clifford-micro"
    generate(
        SplitSpec(
            split_id="S0",
            source_domain={"circuit_instance": ["sampled"]},
            target_domain={"circuit_instance": ["sampled"]},
            fixed_axes={
                "noise_family": [DEFAULT_NOISE_FAMILY],
                "noise_strength": ["L1"],
                "circuit_family": ["near_clifford"],
                "family_native_depth": [1],
                "observable_class": ["z_mid"],
                "shots": [64],
            },
            n_qubits=[4],
            role_counts={"train": 1, "validation": 1, "test": 1},
            family_parameters={
                "near_clifford": {
                    "non_clifford_count": [1],
                    "theta": [math.pi / 4.0],
                }
            },
        ),
        split_near_clifford,
    )
    split_heisenberg = root / "s0-heisenberg-micro"
    generate(
        SplitSpec(
            split_id="S0",
            source_domain={"circuit_instance": ["sampled"]},
            target_domain={"circuit_instance": ["sampled"]},
            fixed_axes={
                "noise_family": [DEFAULT_NOISE_FAMILY],
                "noise_strength": ["L1"],
                "circuit_family": ["heisenberg"],
                "family_native_depth": [1],
                "observable_class": ["z_mid"],
                "shots": [64],
            },
            n_qubits=[4],
            role_counts={"train": 1, "validation": 1, "test": 1},
            family_parameters={"heisenberg": {"dt": 0.15}},
        ),
        split_heisenberg,
    )
    split_random_clifford = root / "s0-random-clifford-micro"
    generate(
        SplitSpec(
            split_id="S0",
            source_domain={"circuit_instance": ["sampled"]},
            target_domain={"circuit_instance": ["sampled"]},
            fixed_axes={
                "noise_family": [DEFAULT_NOISE_FAMILY],
                "noise_strength": ["L1"],
                "circuit_family": ["random_clifford"],
                "family_native_depth": [1],
                "observable_class": ["z_mid"],
                "shots": [64],
            },
            n_qubits=[4],
            role_counts={"train": 1, "validation": 1, "test": 1},
            family_parameters={"random_clifford": {}},
        ),
        split_random_clifford,
    )
    return {
        "legacy": legacy,
        "split-tfi": split,
        "split-qaoa": split_qaoa,
        "split-heisenberg": split_heisenberg,
        "split-random_clifford": split_random_clifford,
        "split-near_clifford": split_near_clifford,
    }


def _mutate_and_rehash(
    schema_artifacts,
    tmp_path: Path,
    *,
    schema_version: str,
    family: str,
    mutation,
):
    if schema_version == LEGACY_SCHEMA_VERSION:
        source = schema_artifacts["legacy"][family]
        writer = _write_legacy_artifact
        loader = _load
    else:
        source = schema_artifacts[f"split-{family}"]
        writer = _write_split_artifact
        loader = validate_split_artifact

    data = tmp_path / f"{schema_version}-{family}"
    shutil.copytree(source, data)
    items, manifest = _read_artifact(data)
    mutation(items[0])
    writer(data, items, manifest)

    if schema_version == LEGACY_SCHEMA_VERSION:
        assert manifest["dataset_hash"] == dataset_hash(items)
    else:
        assert manifest["items_hash"] == item_stream_hash(items)
        assert {
            cell["cell_id"]: cell["item_stream_hash"] for cell in manifest["cells"]
        } == cell_item_stream_hashes(items)
        assert manifest["dataset_hash"] == split_dataset_hash(
            spec_hash=manifest["split_spec_hash"],
            items_hash=manifest["items_hash"],
        )
    return loader, data


@pytest.mark.parametrize("family", sorted(FAMILY_PRESETS))
def test_each_family_accepts_own_fields_and_rejects_foreign_fields(
    schema_artifacts, family
):
    items, manifest = _load(schema_artifacts["legacy"][family])
    item = next(row for row in items if row["family"] == family)
    validate_item(item, schema_version=manifest["dataset_schema_version"])

    own_fields = set(FAMILY_REQUIRED_FIELDS[family])
    for field in sorted(ALL_FAMILY_FIELDS - own_fields):
        with pytest.raises(ValueError, match=rf"unsupported fields: \['{field}'\]"):
            validate_item(
                {**item, field: "foreign"},
                schema_version=LEGACY_SCHEMA_VERSION,
            )


@pytest.mark.protocol("QEM-P004")
def test_rehashed_row_rejects_foreign_family_field(schema_artifacts, tmp_path):
    for artifact_name, field in FOREIGN_FIELD_CASES:
        if artifact_name == "split-tfi":
            source = schema_artifacts["split-tfi"]
            data = tmp_path / field
            shutil.copytree(source, data)
            items, manifest = _read_artifact(data)
            assert items[0]["family"] == "tfi"
            assert field not in FAMILY_REQUIRED_FIELDS["tfi"]
            items[0][field] = "foreign"
            _write_split_artifact(data, items, manifest)
            loader = validate_split_artifact
        else:
            source = schema_artifacts["legacy"]["qaoa"]
            data = tmp_path / field
            shutil.copytree(source, data)
            items, manifest = _read_artifact(data)
            assert items[0]["family"] == "qaoa"
            assert field not in FAMILY_REQUIRED_FIELDS["qaoa"]
            items[0][field] = "foreign"
            _write_legacy_artifact(data, items, manifest)
            loader = _load

        with pytest.raises(
            ValueError, match=rf"unsupported fields: \['{field}'\]"
        ):
            loader(data)


@pytest.mark.protocol("QEM-P004")
def test_rehashed_legacy_row_rejects_split_only_field(schema_artifacts, tmp_path):
    for field in sorted(SPLIT_ITEM_FIELDS):
        data = tmp_path / field
        shutil.copytree(schema_artifacts["legacy"]["tfi"], data)
        items, manifest = _read_artifact(data)
        items[0][field] = "foreign"
        _write_legacy_artifact(data, items, manifest)

        with pytest.raises(
            ValueError, match=rf"unsupported fields: \['{field}'\]"
        ):
            _load(data)


def test_validate_item_rejects_unknown_schema_version(schema_artifacts):
    items, _ = _load(schema_artifacts["legacy"]["tfi"])

    with pytest.raises(ValueError, match="unsupported dataset schema 'future-v3'"):
        validate_item(items[0], schema_version="future-v3")


def test_split_row_accepts_its_family_and_version_fields(schema_artifacts):
    items, manifest = validate_split_artifact(schema_artifacts["split-tfi"])

    validate_item(items[0], schema_version=manifest["dataset_schema_version"])
    assert manifest["dataset_schema_version"] == SPLIT_SCHEMA_VERSION


def test_neighboring_binary64_tfi_parameters_have_distinct_circuit_ids():
    first_j = 1.0
    second_j = math.nextafter(first_j, math.inf)
    common = {
        "n_qubits": 2,
        "steps": 1,
        "h": 0.7,
        "dt": 1.0,
        "circuit_seed": 11,
        "instance": 0,
    }
    first = TFIParams(j=first_j, **common)
    second = TFIParams(j=second_j, **common)

    assert round(first_j, 12) == round(second_j, 12)
    first_fields = _family_parameter_fields(first)
    second_fields = _family_parameter_fields(second)
    _, first_id = canonical_physical_circuit_identity(
        {"family": "tfi", "n_qubits": 2, "circuit_seed": 11, **first_fields}
    )
    _, second_id = canonical_physical_circuit_identity(
        {"family": "tfi", "n_qubits": 2, "circuit_seed": 11, **second_fields}
    )
    maximum_unitary_delta = float(
        np.max(
            np.abs(
                Operator(build_tfi_circuit(first)).data
                - Operator(build_tfi_circuit(second)).data
            )
        )
    )

    assert maximum_unitary_delta > 0.0
    assert first_id != second_id


def test_split_tfi_descriptor_reconstructs_the_producing_operator(
    monkeypatch, tmp_path
):
    generate_module = importlib.import_module("qem_bench.datasets.generate")
    original_builder = generate_module._sample_and_build_circuit
    produced = {}

    def capture_builder(cfg, rng, instance, circuit_seed):
        params, circuit = original_builder(cfg, rng, instance, circuit_seed)
        produced[(instance, circuit_seed)] = circuit
        return params, circuit

    monkeypatch.setattr(
        generate_module, "_sample_and_build_circuit", capture_builder
    )
    data = tmp_path / "split"
    generate("s0-t0-micro", data)
    items, _ = validate_split_artifact(data)
    row = items[0]
    descriptor = canonical_physical_circuit_descriptor(row)
    sidecar = json.loads(
        (data / row["circuit_sidecar"]).read_text(encoding="utf-8")
    )
    rebuilt = build_tfi_circuit(
        TFIParams(
            n_qubits=descriptor["n_qubits"],
            circuit_seed=descriptor["circuit_seed"],
            instance=row["instance"],
            **descriptor["parameters"],
        )
    )

    assert sidecar == descriptor
    np.testing.assert_array_equal(
        Operator(rebuilt).data,
        Operator(produced[(row["instance"], row["circuit_seed"])]).data,
    )


def _circuit_record(family: str, n_qubits: int, depth: int) -> dict:
    common = {
        "family": family,
        "n_qubits": n_qubits,
        "circuit_seed": 17,
    }
    if family == "tfi":
        return {**common, "steps": depth, "j": 0.4, "h": 0.7, "dt": 0.2}
    if family == "heisenberg":
        return {
            **common,
            "steps": depth,
            "jx": 0.4,
            "jy": 0.6,
            "jz": 0.8,
            "dt": 0.15,
        }
    if family == "qaoa":
        return {
            **common,
            "graph_class": "path",
            "edges": [[q, q + 1] for q in range(max(0, n_qubits - 1))],
            "p": depth,
            "gammas": [0.4] * max(0, depth),
            "betas": [0.3] * max(0, depth),
            "edge_probability": None,
        }
    if family == "random_clifford":
        return {**common, "depth": depth}
    if family == "near_clifford":
        return {
            **common,
            "depth": depth,
            "non_clifford_count": 1,
            "theta": math.pi / 4.0,
        }
    raise AssertionError(f"unknown family {family}")


def _build_circuit_record(record: dict):
    family = record["family"]
    common = {
        "n_qubits": record["n_qubits"],
        "circuit_seed": record["circuit_seed"],
        "instance": 0,
    }
    if family == "tfi":
        return build_tfi_circuit(
            TFIParams(
                **common,
                **{field: record[field] for field in ("steps", "j", "h", "dt")},
            )
        )
    if family == "heisenberg":
        return build_heisenberg_circuit(
            HeisenbergParams(
                **common,
                **{
                    field: record[field]
                    for field in ("steps", "jx", "jy", "jz", "dt")
                },
            )
        )
    if family == "qaoa":
        return build_qaoa_circuit(
            QAOAParams(
                **common,
                graph_class=record["graph_class"],
                edges=tuple(tuple(edge) for edge in record["edges"]),
                p=record["p"],
                gammas=tuple(record["gammas"]),
                betas=tuple(record["betas"]),
                edge_probability=record["edge_probability"],
            )
        )
    if family == "random_clifford":
        return build_random_clifford_circuit(
            RandomCliffordParams(**common, depth=record["depth"])
        )
    if family == "near_clifford":
        return build_near_clifford_circuit(
            NearCliffordParams(
                **common,
                depth=record["depth"],
                non_clifford_count=record["non_clifford_count"],
                theta=record["theta"],
            )
        )
    raise AssertionError(f"unknown family {family}")


def _instruction_stream(circuit) -> tuple:
    return tuple(
        (
            instruction.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
            tuple(
                0.0 if float(value) == 0.0 else float(value)
                for value in instruction.operation.params
            ),
        )
        for instruction in circuit.data
    )


def _record_from_descriptor(descriptor: dict) -> dict:
    return {
        "family": descriptor["family"],
        "n_qubits": descriptor["n_qubits"],
        "circuit_seed": descriptor["circuit_seed"],
        **descriptor["parameters"],
    }


def _execution_control(record: dict) -> dict:
    control = dict(record)
    family = record["family"]
    if family in {"tfi", "heisenberg"}:
        control["steps"] += 1
    elif family == "qaoa":
        control["betas"] = [record["betas"][0] + 0.01, *record["betas"][1:]]
    else:
        control["depth"] += 1
    return control


@pytest.mark.parametrize("family", sorted(FAMILY_PRESETS))
def test_realized_identity_round_trips_through_each_family_builder(family):
    record = _circuit_record(family, 4, 1)
    direct = _build_circuit_record(record)
    descriptor, first_id = canonical_physical_circuit_identity(record)
    first_rebuild = build_circuit_from_canonical_descriptor(descriptor)

    second_descriptor, second_id = canonical_physical_circuit_identity(
        _record_from_descriptor(descriptor)
    )
    second_rebuild = build_circuit_from_canonical_descriptor(second_descriptor)
    direct_stream = _instruction_stream(direct)

    assert direct_stream
    assert second_descriptor == descriptor
    assert _instruction_stream(first_rebuild) == direct_stream
    assert _instruction_stream(second_rebuild) == direct_stream
    assert second_id == first_id

    control = _execution_control(record)
    control_stream = _instruction_stream(_build_circuit_record(control))
    _, control_id = canonical_physical_circuit_identity(control)
    assert control_stream != direct_stream
    assert control_id != first_id


def _equivalent_program_cases():
    tfi = _circuit_record("tfi", 3, 2)
    heisenberg = _circuit_record("heisenberg", 3, 2)
    qaoa = _circuit_record("qaoa", 4, 1)
    random_clifford = _circuit_record("random_clifford", 1, 1)
    random_clifford["circuit_seed"] = 0
    near_clifford = _circuit_record("near_clifford", 1, 1)
    near_clifford.update(circuit_seed=0, theta=math.pi / 7.0)
    empty_qaoa = {
        "family": "qaoa",
        "n_qubits": 4,
        "circuit_seed": 7,
        "graph_class": "erdos_renyi",
        "edges": [],
        "p": 1,
        "gammas": [0.4],
        "betas": [0.3],
        "edge_probability": 0.5,
    }
    return (
        pytest.param(
            tfi,
            {**tfi, "circuit_seed": 18},
            id="tfi-unused-seed",
        ),
        pytest.param(
            tfi,
            {**tfi, "j": 0.2, "h": 0.35, "dt": 0.4},
            id="tfi-equal-builder-angles",
        ),
        pytest.param(
            heisenberg,
            {**heisenberg, "circuit_seed": 18},
            id="heisenberg-unused-seed",
        ),
        pytest.param(
            heisenberg,
            {
                **heisenberg,
                "jx": 0.2,
                "jy": 0.3,
                "jz": 0.4,
                "dt": 0.3,
            },
            id="heisenberg-equal-builder-angles",
        ),
        pytest.param(
            qaoa,
            {**qaoa, "edges": [[3, 2], [1, 0], [2, 1]]},
            id="qaoa-edge-order-and-orientation",
        ),
        pytest.param(
            qaoa,
            {
                **qaoa,
                "graph_class": "erdos_renyi",
                "edge_probability": 0.25,
                "circuit_seed": 29,
            },
            id="qaoa-provenance-only-fields",
        ),
        pytest.param(
            empty_qaoa,
            {**empty_qaoa, "gammas": [0.5]},
            id="qaoa-unused-empty-edge-gamma",
        ),
        pytest.param(
            random_clifford,
            {**random_clifford, "circuit_seed": 8},
            id="random-clifford-seed-collision-0-8",
        ),
        pytest.param(
            near_clifford,
            {**near_clifford, "circuit_seed": 8},
            id="near-clifford-seed-collision-0-8",
        ),
        pytest.param(
            near_clifford,
            {**near_clifford, "theta": math.pi / 5.0},
            id="near-clifford-unused-t-theta",
        ),
    )


@pytest.mark.parametrize(("first", "equivalent"), _equivalent_program_cases())
def test_realized_identity_convergence_sweep(first, equivalent):
    first_stream = _instruction_stream(_build_circuit_record(first))
    equivalent_stream = _instruction_stream(_build_circuit_record(equivalent))
    _, first_id = canonical_physical_circuit_identity(first)
    _, equivalent_id = canonical_physical_circuit_identity(equivalent)

    assert first != equivalent
    assert equivalent_stream == first_stream
    assert equivalent_id == first_id

    control = _execution_control(first)
    control_stream = _instruction_stream(_build_circuit_record(control))
    _, control_id = canonical_physical_circuit_identity(control)
    assert control_stream != first_stream
    assert control_id != first_id


def _separation_sweep_records(family: str) -> list[dict]:
    if family == "tfi":
        base = _circuit_record(family, 3, 1)
        return [base, {**base, "j": 0.41}, {**base, "h": 0.71}]
    if family == "heisenberg":
        base = _circuit_record(family, 3, 1)
        return [base, {**base, "jx": 0.41}, {**base, "jz": 0.81}]
    if family == "qaoa":
        empty = {
            "family": "qaoa",
            "n_qubits": 4,
            "circuit_seed": 7,
            "graph_class": "erdos_renyi",
            "edges": [],
            "p": 1,
            "gammas": [0.4],
            "betas": [0.3],
            "edge_probability": 0.5,
        }
        return [
            empty,
            {**empty, "gammas": [0.5]},
            {**empty, "betas": [0.31]},
            {**empty, "edges": [[0, 1]]},
        ]
    if family == "random_clifford":
        return [
            {
                "family": family,
                "n_qubits": 1,
                "depth": 1,
                "circuit_seed": seed,
            }
            for seed in range(17)
        ]
    return [
        {
            "family": family,
            "n_qubits": 1,
            "depth": 1,
            "non_clifford_count": 1,
            "theta": theta,
            "circuit_seed": seed,
        }
        for seed, theta in (
            (0, math.pi / 7.0),
            (0, math.pi / 5.0),
            (1, math.pi / 7.0),
            (2, math.pi / 7.0),
            (5, math.pi / 7.0),
            (5, math.pi / 5.0),
            (8, math.pi / 7.0),
        )
    ]


@pytest.mark.parametrize("family", sorted(FAMILY_PRESETS))
def test_realized_identity_separation_sweep(family):
    records = _separation_sweep_records(family)
    streams = [
        _instruction_stream(_build_circuit_record(record)) for record in records
    ]
    identities = [
        canonical_physical_circuit_identity(record)[1] for record in records
    ]
    distinct_program_pairs = 0
    equivalent_program_pairs = 0
    for first, second in combinations(range(len(records)), 2):
        if streams[first] == streams[second]:
            equivalent_program_pairs += 1
            assert identities[first] == identities[second], (family, first, second)
            continue
        distinct_program_pairs += 1
        assert identities[first] != identities[second], (family, first, second)

    assert distinct_program_pairs > 0
    if family in {"qaoa", "near_clifford"}:
        assert equivalent_program_pairs > 0
        assert streams[0] == streams[1]
        assert identities[0] == identities[1]
    elif family == "random_clifford":
        assert equivalent_program_pairs > 0


@pytest.mark.parametrize(
    ("edges", "message"),
    (
        (((0, 1), (1, 0)), "duplicate edge"),
        (((0, 0),), "invalid edge"),
        (((0, 4),), "invalid edge"),
    ),
)
def test_qaoa_canonical_execution_retains_edge_validation(edges, message):
    params = QAOAParams(
        n_qubits=4,
        graph_class="erdos_renyi",
        edges=edges,
        p=1,
        gammas=(0.4,),
        betas=(0.3,),
        circuit_seed=7,
        instance=0,
        edge_probability=0.5,
    )

    with pytest.raises(ValueError, match=message):
        build_qaoa_circuit(params)


def _accepts(call) -> bool:
    try:
        call()
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("family", sorted(FAMILY_PRESETS))
def test_builder_and_schema_domains_agree_across_widths_and_depths(family):
    for n_qubits in range(0, 23):
        for depth in range(0, 4):
            record = _circuit_record(family, n_qubits, depth)
            builder_accepts = _accepts(lambda: _build_circuit_record(record))
            schema_accepts = _accepts(
                lambda: canonical_physical_circuit_descriptor(record)
            )
            assert schema_accepts == builder_accepts, (
                family,
                n_qubits,
                depth,
            )


def _build_qaoa_from_record(record: dict):
    return build_qaoa_circuit(
        QAOAParams(
            n_qubits=record["n_qubits"],
            graph_class=record["graph_class"],
            edges=tuple(tuple(edge) for edge in record["edges"]),
            p=record["p"],
            gammas=tuple(record["gammas"]),
            betas=tuple(record["betas"]),
            circuit_seed=record["circuit_seed"],
            instance=0,
            edge_probability=record["edge_probability"],
        )
    )


def test_qaoa_identity_uses_execution_and_retains_provenance_in_sidecar():
    common = {
        "family": "qaoa",
        "n_qubits": 4,
        "edges": [[0, 1], [1, 2], [2, 3]],
        "p": 1,
        "gammas": [0.4],
        "betas": [0.3],
    }
    records = (
        {**common, "graph_class": "path", "edge_probability": None, "circuit_seed": 11},
        {
            **common,
            "graph_class": "erdos_renyi",
            "edge_probability": 0.25,
            "circuit_seed": 11,
        },
        {
            **common,
            "graph_class": "erdos_renyi",
            "edge_probability": 0.25,
            "circuit_seed": 29,
        },
        {
            **common,
            "graph_class": "erdos_renyi",
            "edge_probability": 0.75,
            "circuit_seed": 11,
        },
    )
    descriptors_and_ids = [
        canonical_physical_circuit_identity(record) for record in records
    ]
    operators = [Operator(_build_qaoa_from_record(record)).data for record in records]

    assert len({identity for _, identity in descriptors_and_ids}) == 1
    sidecars = {
        json.dumps(descriptor, sort_keys=True)
        for descriptor, _ in descriptors_and_ids
    }
    assert len(sidecars) == 4
    assert all(operator.tobytes() == operators[0].tobytes() for operator in operators)
    assert descriptors_and_ids[0][0]["parameters"]["graph_class"] == "path"
    assert descriptors_and_ids[1][0]["parameters"]["edge_probability"] == 0.25
    assert descriptors_and_ids[2][0]["circuit_seed"] == 29

    changed = {**records[0], "betas": [0.31]}
    _, changed_identity = canonical_physical_circuit_identity(changed)
    changed_operator = Operator(_build_qaoa_from_record(changed)).data
    assert changed_identity != descriptors_and_ids[0][1]
    assert not np.array_equal(changed_operator, operators[0])


def test_other_family_identities_distinguish_provenance_from_execution():
    tfi_first = _circuit_record("tfi", 3, 2)
    tfi_second = {**tfi_first, "circuit_seed": 18}
    tfi_first_descriptor, tfi_first_id = canonical_physical_circuit_identity(tfi_first)
    tfi_second_descriptor, tfi_second_id = canonical_physical_circuit_identity(
        tfi_second
    )
    assert tfi_first_descriptor != tfi_second_descriptor
    assert tfi_first_id == tfi_second_id
    np.testing.assert_array_equal(
        Operator(_build_circuit_record(tfi_first)).data,
        Operator(_build_circuit_record(tfi_second)).data,
    )

    heisenberg_first = _circuit_record("heisenberg", 3, 2)
    heisenberg_second = {**heisenberg_first, "circuit_seed": 18}
    heisenberg_first_descriptor, heisenberg_first_id = (
        canonical_physical_circuit_identity(heisenberg_first)
    )
    heisenberg_second_descriptor, heisenberg_second_id = (
        canonical_physical_circuit_identity(heisenberg_second)
    )
    assert heisenberg_first_descriptor != heisenberg_second_descriptor
    assert heisenberg_first_id == heisenberg_second_id
    np.testing.assert_array_equal(
        Operator(_build_circuit_record(heisenberg_first)).data,
        Operator(_build_circuit_record(heisenberg_second)).data,
    )

    for family in ("random_clifford", "near_clifford"):
        first = _circuit_record(family, 4, 2)
        second = {**first, "circuit_seed": 18}
        first_descriptor, first_id = canonical_physical_circuit_identity(first)
        second_descriptor, second_id = canonical_physical_circuit_identity(second)
        assert first_descriptor != second_descriptor
        assert first_id != second_id
        assert not np.array_equal(
            Operator(_build_circuit_record(first)).data,
            Operator(_build_circuit_record(second)).data,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("noisy_stderr", -0.001, "field noisy_stderr must be nonnegative"),
        ("noisy_expectation", -1.001, "field noisy_expectation must be in"),
        ("noisy_expectation", 1.001, "field noisy_expectation must be in"),
        ("ideal_expectation", -1.001, "field ideal_expectation must be in"),
        ("ideal_expectation", 1.001, "field ideal_expectation must be in"),
    ),
)
@pytest.mark.parametrize(
    "schema_version", (LEGACY_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION)
)
def test_fully_rehashed_artifact_rejects_impossible_measurement_values(
    schema_artifacts,
    tmp_path,
    schema_version,
    field,
    replacement,
    message,
):
    loader, data = _mutate_and_rehash(
        schema_artifacts,
        tmp_path,
        schema_version=schema_version,
        family="tfi",
        mutation=lambda item: item.__setitem__(field, replacement),
    )

    with pytest.raises(ValueError, match=message):
        loader(data)


@pytest.mark.parametrize(
    "schema_version", (LEGACY_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION)
)
def test_fully_rehashed_artifact_rejects_stderr_above_the_shot_bound(
    schema_artifacts, tmp_path, schema_version
):
    def mutate(item):
        item["noisy_stderr"] = 1.0 / math.sqrt(item["shots"]) + 1e-6

    loader, data = _mutate_and_rehash(
        schema_artifacts,
        tmp_path,
        schema_version=schema_version,
        family="tfi",
        mutation=mutate,
    )

    with pytest.raises(ValueError, match="noisy_stderr exceeds its shot bound"):
        loader(data)


def _mutate_qaoa_domain(item: dict, case: str) -> None:
    if case == "unknown-graph":
        item["graph_class"] = "complete"
    elif case == "unsupported-p":
        item["p"] = 3
        item["gammas"] = [0.1, 0.2, 0.3]
        item["betas"] = [0.1, 0.2, 0.3]
    elif case == "edge-probability-low":
        item["graph_class"] = "erdos_renyi"
        item["edge_probability"] = -0.25
    elif case == "edge-probability-high":
        item["graph_class"] = "erdos_renyi"
        item["edge_probability"] = 1.25
    elif case == "edge-probability-on-path":
        item["graph_class"] = "path"
        item["edge_probability"] = 0.5
    elif case == "edge-probability-missing":
        item["graph_class"] = "erdos_renyi"
        item["edge_probability"] = None
    elif case == "gammas-length":
        item["gammas"] = []
    elif case == "betas-length":
        item["betas"] = []
    elif case == "width-low":
        item["n_qubits"] = 1
    elif case == "width-high":
        item["n_qubits"] = 13
    elif case == "self-loop":
        item["edges"] = [[0, 0]]
    elif case == "negative-endpoint":
        item["edges"] = [[-1, 0]]
    elif case == "endpoint-at-width":
        item["edges"] = [[0, item["n_qubits"]]]
    elif case == "duplicate-edge":
        item["edges"] = [[0, 1], [0, 1]]
    elif case == "reverse-duplicate-edge":
        item["edges"] = [[0, 1], [1, 0]]
    elif case == "path-topology":
        item["n_qubits"] = 4
        item["graph_class"] = "path"
        item["edge_probability"] = None
        item["edges"] = [[0, 1], [1, 2]]
    elif case == "cycle-topology":
        item["n_qubits"] = 4
        item["graph_class"] = "cycle"
        item["edge_probability"] = None
        item["edges"] = [[0, 1], [1, 2], [2, 3]]
    elif case == "3-regular-topology":
        item["n_qubits"] = 4
        item["graph_class"] = "3_regular"
        item["edge_probability"] = None
        item["edges"] = [[0, 1], [1, 2], [2, 3]]
    elif case == "gamma-low":
        item["gammas"][0] = -0.1
    elif case == "gamma-high":
        item["gammas"][0] = math.pi
    elif case == "beta-low":
        item["betas"][0] = -0.1
    elif case == "beta-high":
        item["betas"][0] = math.pi / 2.0
    else:
        raise AssertionError(f"unknown QAOA mutation {case}")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("unknown-graph", "unknown QAOA graph_class"),
        ("unsupported-p", "field p must be 1 or 2"),
        ("edge-probability-low", "field edge_probability must be in"),
        ("edge-probability-high", "field edge_probability must be in"),
        (
            "edge-probability-on-path",
            "edge_probability must be present only for erdos_renyi",
        ),
        (
            "edge-probability-missing",
            "edge_probability must be present only for erdos_renyi",
        ),
        ("gammas-length", "QAOA gammas length must equal p"),
        ("betas-length", "QAOA betas length must equal p"),
        ("width-low", "QAOA n_qubits must be in"),
        ("width-high", "QAOA n_qubits must be in"),
        ("self-loop", "contains an invalid QAOA edge"),
        ("negative-endpoint", "contains an invalid QAOA edge"),
        ("endpoint-at-width", "contains an invalid QAOA edge"),
        ("duplicate-edge", "contains a duplicate QAOA edge"),
        ("reverse-duplicate-edge", "contains a duplicate QAOA edge"),
        ("path-topology", "edges do not form the declared path"),
        ("cycle-topology", "edges do not form the declared cycle"),
        ("3-regular-topology", "edges are not a simple 3-regular graph"),
        ("gamma-low", "QAOA gammas must be in"),
        ("gamma-high", "QAOA gammas must be in"),
        ("beta-low", "QAOA betas must be in"),
        ("beta-high", "QAOA betas must be in"),
    ),
)
@pytest.mark.parametrize(
    "schema_version", (LEGACY_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION)
)
def test_fully_rehashed_artifact_rejects_impossible_qaoa_values(
    schema_artifacts, tmp_path, schema_version, case, message
):
    loader, data = _mutate_and_rehash(
        schema_artifacts,
        tmp_path,
        schema_version=schema_version,
        family="qaoa",
        mutation=lambda item: _mutate_qaoa_domain(item, case),
    )

    with pytest.raises(ValueError, match=message):
        loader(data)


def _mutate_near_clifford_domain(item: dict, case: str) -> None:
    if case == "zero-count":
        item["non_clifford_count"] = 0
    elif case == "count-above-width":
        item["non_clifford_count"] = item["n_qubits"] + 1
    elif case == "clifford-theta":
        item["theta"] = math.pi / 2.0
    else:
        raise AssertionError(f"unknown near-Clifford mutation {case}")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("zero-count", "non_clifford_count must be in"),
        ("count-above-width", "non_clifford_count must be in"),
        ("clifford-theta", "theta must be non-Clifford"),
    ),
)
@pytest.mark.parametrize(
    "schema_version", (LEGACY_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION)
)
def test_fully_rehashed_artifact_rejects_impossible_near_clifford_values(
    schema_artifacts, tmp_path, schema_version, case, message
):
    loader, data = _mutate_and_rehash(
        schema_artifacts,
        tmp_path,
        schema_version=schema_version,
        family="near_clifford",
        mutation=lambda item: _mutate_near_clifford_domain(item, case),
    )

    with pytest.raises(ValueError, match=message):
        loader(data)


@pytest.mark.parametrize(
    ("family", "field", "replacement", "message"),
    (
        (
            "heisenberg",
            "n_qubits",
            1,
            r"heisenberg n_qubits must be in \[2, 12\]",
        ),
        (
            "heisenberg",
            "n_qubits",
            13,
            r"heisenberg n_qubits must be in \[2, 12\]",
        ),
        ("heisenberg", "dt", 0.0, "Heisenberg dt must be positive"),
        ("heisenberg", "dt", -0.1, "Heisenberg dt must be positive"),
        (
            "random_clifford",
            "n_qubits",
            21,
            r"random_clifford n_qubits must be in \[1, 20\]",
        ),
        (
            "near_clifford",
            "n_qubits",
            15,
            r"near_clifford n_qubits must be in \[1, 14\]",
        ),
    ),
)
@pytest.mark.parametrize(
    "schema_version", (LEGACY_SCHEMA_VERSION, SPLIT_SCHEMA_VERSION)
)
def test_fully_rehashed_artifact_rejects_every_new_builder_bound(
    schema_artifacts,
    tmp_path,
    schema_version,
    family,
    field,
    replacement,
    message,
):
    loader, data = _mutate_and_rehash(
        schema_artifacts,
        tmp_path,
        schema_version=schema_version,
        family=family,
        mutation=lambda item: item.__setitem__(field, replacement),
    )

    with pytest.raises(ValueError, match=message):
        loader(data)


def test_fully_rehashed_legacy_heisenberg_zero_dt_refuses_all_32_rows(
    schema_artifacts, tmp_path
):
    data = tmp_path / "legacy-heisenberg-zero-dt"
    shutil.copytree(schema_artifacts["legacy"]["heisenberg"], data)
    items, manifest = _read_artifact(data)
    assert len(items) == 32
    for item in items:
        item["dt"] = 0.0
    _write_legacy_artifact(data, items, manifest)
    assert manifest["dataset_hash"] == dataset_hash(items)

    with pytest.raises(ValueError, match="Heisenberg dt must be positive"):
        _load(data)
