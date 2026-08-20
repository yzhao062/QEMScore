"""Stage 1 to 3 contracts for versioned split artifacts."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from qem_bench.datasets.generate import (
    PRESETS,
    _uses_frozen_legacy_serializer,
    generate,
)
from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import (
    SPLIT_ALLOWED_COUPLINGS,
    SPLIT_AXES,
    SplitSpec,
    resolve_split_spec,
)
from qem_bench.noise.models import DEFAULT_NOISE_FAMILY
from qem_bench.reproducibility import CI_LOCK_SHA256, CI_LOCK_SHA256_ENV
from qem_bench.runner.run import _load
from qem_bench.validation import (
    canonical_item_lines,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    split_spec_hash,
    validate_axis_contract,
    validate_split_artifact,
)

SPLIT_GOLDENS = {
    "s0": {
        "split_spec_hash": (
            "dd87cb9405875257eade35fa3e201575f638c37fb05bc90fd01653721c098dc6"
        ),
        "cell_hashes": {
            "cell-305b997a6ab380ec598ad1cc379ca0ce3d5f9b1d0e37e7046914d7cf71be67cd": (
                "5ca3cc6958c220076719d4d4d356171e2b68f243cfabda7fb1d44b03d533217b"
            ),
            "cell-6cf34780024766a1b3e2fcc11792bb74af7c8c780ef736e48db17d27dd9e3548": (
                "96d11761b118cb7a0eda805623ea30b07a455f5778b5d3b63e16dd460eceddf1"
            ),
            "cell-90d7f7101da1bdb2a2f1d2d8187d97ed92af0ded4191770cac8d13db351cac64": (
                "494a1d556369bf6f9dea9fc6280bb4e0c9d4b1886ac1462e1e5c8b4f874be376"
            ),
        },
        "dataset_hash": (
            "50aeecfb2d463cecb4ad5e6ecb0b49eada57278004155e4472fdccf8f14efe41"
        ),
    },
    "s1": {
        "split_spec_hash": (
            "614597593060f4c2ee328881942d91dae0ae590f88fd1c50a7b2dbbd18b2dd04"
        ),
        "cell_hashes": {
            "cell-3990d025c00443b1560ad94adef7b044628d9b3c0a61128225f97a8545134143": (
                "4e486a3dfbacf86f43d37dab5434fc47c5ecb8f18028368dc7c14f08a231d6ec"
            ),
            "cell-55fc45cb35526860e0b8f3e874cf49aec26be662bdd401162a37e845eb473e59": (
                "f2ffd621c7beaa369114d9b54797458d3d3c1694a95922f1863d469cbece507e"
            ),
            "cell-aad90f0ed7c277258eb34edab5ac9665da9c3d5aee0271e4829bf0b7c563b6fd": (
                "db86028a7d1938796b2664aff4c69c4a7a7f1cc14bd5bde4e2536fa5b237fc9a"
            ),
        },
        "dataset_hash": (
            "ae40bf2dba3d3263662aa6c6d07f7e0726b2663b352391e007be77222a688993"
        ),
    },
}


def _spec(split_id: str) -> SplitSpec:
    fixed = {
        "noise_family": ["depolarizing_readout"],
        "noise_strength": ["L1"],
        "circuit_family": ["tfi"],
        "family_native_depth": [1],
        "observable_class": ["z_mid"],
        "shots": [512],
    }
    domains = {
        "S0": ("circuit_instance", ["sampled"], ["sampled"]),
        "S1": (
            "noise_family",
            ["depolarizing_readout"],
            ["coherent_overrotation"],
        ),
        "S2": ("noise_strength", ["L1"], ["L4"]),
        "S3": ("circuit_family", ["tfi", "heisenberg"], ["qaoa"]),
        "S4": ("family_native_depth", [1], [2]),
        "S5": ("observable_class", ["z_mid"], ["zz_mid"]),
        "S6": ("shots", [512], [64]),
    }
    axis, source, target = domains[split_id]
    if axis in fixed:
        fixed.pop(axis)
    family_parameters = {"tfi": {"dt": 0.2}}
    if split_id == "S3":
        family_parameters = {
            "tfi": {"dt": 0.2},
            "heisenberg": {"dt": 0.15},
            "qaoa": {"graph_classes": ["path"]},
        }
    return SplitSpec(
        split_id=split_id,
        source_domain={axis: source},
        target_domain={axis: target},
        fixed_axes=fixed,
        n_qubits=[4],
        role_counts={"train": 5, "validation": 2, "test": 2},
        family_parameters=family_parameters,
        allowed_couplings=SPLIT_ALLOWED_COUPLINGS[split_id],
    )


def _single_family_generation_spec(
    family: str, family_parameters: dict
) -> SplitSpec:
    return SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": ["depolarizing_readout"],
            "noise_strength": ["L1"],
            "circuit_family": [family],
            "family_native_depth": [1],
            "observable_class": ["z_mid"],
            "shots": [16],
        },
        n_qubits=[4],
        role_counts={"train": 1, "validation": 1, "test": 1},
        family_parameters={family: family_parameters},
    )


@pytest.mark.parametrize("split_id", sorted(SPLIT_AXES))
def test_closed_grammar_resolves_each_split(split_id):
    resolved = resolve_split_spec(_spec(split_id))
    assert resolved.spec.split_axis == SPLIT_AXES[split_id]
    assert {cell.split for cell in resolved.cells} == {
        "train",
        "validation",
        "test",
    }
    validate_axis_contract(resolved.spec, resolved.cells)


def test_role_counts_are_allocated_across_multi_family_pools():
    resolved = resolve_split_spec(_spec("S3"))
    for role, expected in (("train", 5), ("validation", 2), ("test", 2)):
        pools = [pool for pool in resolved.circuit_pools if pool.split == role]
        assert sum(pool.n_instances for pool in pools) == expected
    assert sorted(
        pool.n_instances
        for pool in resolved.circuit_pools
        if pool.split == "train"
    ) == [2, 3]


def test_two_axis_change_is_rejected_clearly():
    resolved = resolve_split_spec(_spec("S1"))
    cells = list(resolved.cells)
    index = next(i for i, cell in enumerate(cells) if cell.domain == "target")
    axis_values = dict(cells[index].axis_values)
    axis_values["shots"] = 64
    cells[index] = replace(cells[index], axis_values=axis_values)

    with pytest.raises(ValueError, match="S1 changes undeclared axis 'shots'"):
        validate_axis_contract(resolved.spec, cells)


@pytest.mark.protocol("QEM-P003")
def test_source_only_validation_is_enforced():
    resolved = resolve_split_spec(_spec("S2"))
    cells = list(resolved.cells)
    index = next(i for i, cell in enumerate(cells) if cell.split == "validation")
    cells[index] = replace(cells[index], domain="target")

    with pytest.raises(ValueError, match="role 'validation' requires domain 'source'"):
        validate_axis_contract(resolved.spec, cells)


def test_closed_grammar_rejects_unknown_split_and_coupling():
    unknown = replace(_spec("S1"), split_id="S7")
    with pytest.raises(ValueError, match="unknown split_id"):
        resolve_split_spec(unknown)

    undeclared = replace(_spec("S3"), allowed_couplings=())
    with pytest.raises(ValueError, match="S3 permits exactly"):
        resolve_split_spec(undeclared)


def test_closed_grammar_rejects_missing_role_and_overlapping_domains():
    missing_role = replace(
        _spec("S1"), role_counts={"train": 1, "test": 1}
    )
    with pytest.raises(ValueError, match="role_counts must contain exactly"):
        resolve_split_spec(missing_role)

    overlap = replace(
        _spec("S2"), target_domain={"noise_strength": ["L1"]}
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        resolve_split_spec(overlap)


def test_directional_splits_reject_wrong_direction():
    backward_depth = replace(
        _spec("S4"),
        source_domain={"family_native_depth": [2]},
        target_domain={"family_native_depth": [1]},
    )
    with pytest.raises(ValueError, match="must all exceed source"):
        resolve_split_spec(backward_depth)

    backward_shots = replace(
        _spec("S6"),
        source_domain={"shots": [64]},
        target_domain={"shots": [512]},
    )
    with pytest.raises(ValueError, match="must all be below source"):
        resolve_split_spec(backward_shots)


def test_authored_set_order_does_not_change_resolution():
    first = replace(
        _spec("S1"),
        source_domain={
            "noise_family": ["dephasing_readout", "depolarizing_readout"]
        },
        fixed_axes={
            **dict(_spec("S1").fixed_axes),
            "observable_class": ["zz_mid", "z_mid"],
        },
    )
    second = replace(
        first,
        source_domain={
            "noise_family": ["depolarizing_readout", "dephasing_readout"]
        },
        fixed_axes={
            **dict(first.fixed_axes),
            "observable_class": ["z_mid", "zz_mid"],
        },
    )
    resolved_first = resolve_split_spec(first)
    resolved_second = resolve_split_spec(second)

    assert split_spec_hash(first) == split_spec_hash(second)
    assert [pool.to_dict() for pool in resolved_first.circuit_pools] == [
        pool.to_dict() for pool in resolved_second.circuit_pools
    ]
    assert [cell.to_dict() for cell in resolved_first.cells] == [
        cell.to_dict() for cell in resolved_second.cells
    ]


@pytest.fixture(scope="module")
def split_dataset(tmp_path_factory):
    data = tmp_path_factory.mktemp("split-v2") / "data"
    manifest = generate("s0-t0-micro", data)
    return data, manifest


def test_split_v2_is_byte_deterministic_and_seed_sensitive(tmp_path):
    first = generate("s0-t0-micro", tmp_path / "first")
    second = generate("s0-t0-micro", tmp_path / "second")
    changed = generate("s0-t0-micro", tmp_path / "changed", master_seed=8)

    assert first["split_spec_hash"] == second["split_spec_hash"]
    assert first["dataset_hash"] == second["dataset_hash"]
    assert (tmp_path / "first" / "items.jsonl").read_bytes() == (
        tmp_path / "second" / "items.jsonl"
    ).read_bytes()
    assert changed["split_spec_hash"] == first["split_spec_hash"]
    assert changed["items_hash"] != first["items_hash"]
    assert changed["dataset_hash"] != first["dataset_hash"]


def test_split_v2_manifest_stores_declaration_resolution_and_valid_hash_chain(
    split_dataset,
):
    data, manifest = split_dataset
    items, loaded_manifest = validate_split_artifact(data)

    assert manifest["dataset_schema_version"] == "split-v2"
    assert manifest["split_spec"]["split_id"] == "S0"
    assert manifest["circuit_pools"]
    assert manifest["cells"]
    assert items
    assert loaded_manifest["dataset_hash"] == manifest["dataset_hash"]


@pytest.mark.parametrize(
    (
        "family",
        "family_parameters",
        "sampled_fields",
        "pool_parameter_fields",
    ),
    (
        pytest.param(
            "tfi",
            {"dt": 0.2},
            {"j", "h"},
            {"dt", "n_qubits", "steps"},
            id="tfi",
        ),
        pytest.param(
            "heisenberg",
            {"dt": 0.15},
            {"jx", "jy", "jz"},
            {"dt", "n_qubits", "steps"},
            id="heisenberg",
        ),
        pytest.param(
            "qaoa",
            {"graph_classes": ["erdos_renyi"]},
            {"edges", "gammas", "betas", "edge_probability"},
            {"graph_classes", "n_qubits", "p"},
            id="qaoa-sampled-er-probability",
        ),
        pytest.param(
            "qaoa",
            {"graph_classes": ["erdos_renyi"], "er_edge_probability": 0.4},
            {"edges", "gammas", "betas", "edge_probability"},
            {"er_edge_probability", "graph_classes", "n_qubits", "p"},
            id="qaoa-fixed-er-probability",
        ),
        pytest.param(
            "random_clifford",
            {},
            set(),
            {"depth", "n_qubits"},
            id="random-clifford",
        ),
        pytest.param(
            "near_clifford",
            {
                "non_clifford_count": [1, 2],
                "theta": [0.4487989505128276, 0.6283185307179586],
            },
            set(),
            {"depth", "n_qubits", "non_clifford_count", "theta"},
            id="near-clifford",
        ),
    ),
)
def test_fresh_generation_for_every_family_obeys_pool_binding(
    tmp_path,
    family,
    family_parameters,
    sampled_fields,
    pool_parameter_fields,
):
    data = tmp_path / family
    manifest = generate_split(
        _single_family_generation_spec(family, family_parameters), data
    )

    items, loaded_manifest = validate_split_artifact(data)

    assert len(items) == 3
    assert {item["family"] for item in items} == {family}
    assert loaded_manifest["dataset_hash"] == manifest["dataset_hash"]
    for pool in manifest["circuit_pools"]:
        assert sampled_fields.isdisjoint(pool["parameter_grid"])
        assert set(pool["parameter_grid"]) == pool_parameter_fields


def test_split_dataset_identity_is_environment_independent(monkeypatch, tmp_path):
    monkeypatch.delenv(CI_LOCK_SHA256_ENV, raising=False)
    absent_dir = tmp_path / "absent"
    absent = generate("s0-t0-micro", absent_dir)

    monkeypatch.setenv(CI_LOCK_SHA256_ENV, CI_LOCK_SHA256)
    verified_dir = tmp_path / "verified"
    verified = generate("s0-t0-micro", verified_dir)

    assert (absent_dir / "items.jsonl").read_bytes() == (
        verified_dir / "items.jsonl"
    ).read_bytes()
    assert absent["split_spec_hash"] == verified["split_spec_hash"]
    assert absent["items_hash"] == verified["items_hash"]
    assert absent["dataset_hash"] == verified["dataset_hash"]
    assert absent["environment_contract"]["observed_ci_lock_sha256"] is None
    assert absent["environment_contract"]["lock_verified"] is False
    assert verified["environment_contract"]["observed_ci_lock_sha256"] == (
        CI_LOCK_SHA256
    )
    assert verified["environment_contract"]["lock_verified"] is True
    assert "build_hash" not in absent
    assert "build_hash" not in verified
    validate_split_artifact(absent_dir)
    validate_split_artifact(verified_dir)


def test_tiny_s0_and_heterogeneous_s1_hashes_are_golden(split_dataset, tmp_path):
    _, s0 = split_dataset
    s1_spec = replace(
        _spec("S1"),
        fixed_axes={**dict(_spec("S1").fixed_axes), "shots": [64]},
        n_qubits=[3],
        role_counts={"train": 1, "validation": 1, "test": 1},
    )
    s1 = generate_split(s1_spec, tmp_path / "s1")

    for name, manifest in (("s0", s0), ("s1", s1)):
        actual_cells = {
            cell["cell_id"]: cell["item_stream_hash"]
            for cell in manifest["cells"]
        }
        assert manifest["split_spec_hash"] == SPLIT_GOLDENS[name]["split_spec_hash"]
        assert actual_cells == SPLIT_GOLDENS[name]["cell_hashes"]
        assert manifest["dataset_hash"] == SPLIT_GOLDENS[name]["dataset_hash"]


def test_every_split_seed_is_recoverable_and_roles_are_disjoint(split_dataset):
    data, manifest = split_dataset
    items, _ = validate_split_artifact(data)
    master = manifest["master_seed"]
    pools = {
        value["circuit_pool_id"]: value for value in manifest["circuit_pools"]
    }
    cells = {value["cell_id"]: value for value in manifest["cells"]}

    for item in items:
        pool = pools[item["circuit_pool_id"]]
        circuit_key = tuple(pool["pool_seed_key"]) + (0, item["instance"])
        expected_circuit_seed = int(
            np.random.SeedSequence(master, spawn_key=circuit_key).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        cell = cells[item["cell_id"]]
        sampler_key = tuple(cell["cell_seed_key"]) + (
            1,
            item["instance"],
            item["replicate"],
            0,
        )
        expected_sampler_seed = int(
            np.random.SeedSequence(master, spawn_key=sampler_key).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        assert item["circuit_seed"] == expected_circuit_seed
        assert item["sampler_seed"] == expected_sampler_seed

    by_role = {
        role: {item["circuit_id"] for item in items if item["split"] == role}
        for role in ("train", "validation", "test")
    }
    assert by_role["train"].isdisjoint(by_role["validation"])
    assert by_role["train"].isdisjoint(by_role["test"])
    assert by_role["validation"].isdisjoint(by_role["test"])


def test_circuit_pool_reuse_within_one_role_is_legal(tmp_path):
    spec = replace(
        _spec("S0"),
        fixed_axes={
            **dict(_spec("S0").fixed_axes),
            "noise_strength": ["L1", "L2"],
            "shots": [64],
        },
        n_qubits=[3],
        role_counts={"train": 1, "validation": 1, "test": 1},
    )
    data = tmp_path / "reuse"
    generate_split(spec, data)
    items, _ = validate_split_artifact(data)
    train = [item for item in items if item["split"] == "train"]
    assert len({item["cell_id"] for item in train}) == 2
    assert len({item["circuit_id"] for item in train}) == 1


def test_three_hash_levels_change_at_their_intended_scope(split_dataset):
    data, manifest = split_dataset
    items, _ = validate_split_artifact(data)
    original_cells = cell_item_stream_hashes(items)

    changed_spec = copy.deepcopy(manifest["split_spec"])
    changed_spec["budget_tier"] = "fixture-tier"
    changed_spec_hash = split_spec_hash(changed_spec)
    assert changed_spec_hash != manifest["split_spec_hash"]
    assert item_stream_hash(items) == manifest["items_hash"]
    assert item_stream_hash(reversed(items)) == manifest["items_hash"]
    assert cell_item_stream_hashes(items) == original_cells
    spec_global = split_dataset_hash(
        spec_hash=changed_spec_hash,
        items_hash=manifest["items_hash"],
    )
    assert spec_global != manifest["dataset_hash"]

    changed_items = copy.deepcopy(items)
    changed_cell_id = changed_items[0]["cell_id"]
    changed_items[0]["noisy_expectation"] += 0.125
    changed_cells = cell_item_stream_hashes(changed_items)
    assert changed_cells[changed_cell_id] != original_cells[changed_cell_id]
    assert all(
        changed_cells[cell_id] == digest
        for cell_id, digest in original_cells.items()
        if cell_id != changed_cell_id
    )
    assert item_stream_hash(changed_items) != manifest["items_hash"]
    assert split_spec_hash(manifest["split_spec"]) == manifest["split_spec_hash"]
    item_global = split_dataset_hash(
        spec_hash=manifest["split_spec_hash"],
        items_hash=item_stream_hash(changed_items),
    )
    assert item_global != manifest["dataset_hash"]
    assert cell_item_stream_hashes(items) == original_cells


@pytest.mark.parametrize("tamper", ("item", "spec", "cell", "sidecar"))
def test_split_hash_chain_detects_tampering(split_dataset, tmp_path, tamper):
    source, _ = split_dataset
    data = tmp_path / tamper
    shutil.copytree(source, data)

    if tamper == "item":
        items_path = data / "items.jsonl"
        items = [
            json.loads(line) for line in items_path.read_text().splitlines() if line
        ]
        items[0]["noisy_expectation"] += 0.1
        items_path.write_text("\n".join(canonical_item_lines(items)) + "\n")
        message = "cell item_stream_hash mismatch"
    elif tamper == "spec":
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split_spec"]["budget_tier"] = "tampered"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        message = "split_spec_hash mismatch"
    elif tamper == "cell":
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cells"][0]["cell_seed_key"][0] += 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        message = "resolved cells do not match split_spec"
    else:
        manifest = json.loads((data / "manifest.json").read_text())
        relative = next(iter(manifest["sidecar_hashes"]))
        with (data / relative).open("ab") as stream:
            stream.write(b"tampered")
        message = "sidecar hash mismatch"

    with pytest.raises(ValueError, match=message):
        validate_split_artifact(data)


def test_generation_is_write_once(tmp_path):
    data = tmp_path / "occupied"
    generate("t0-micro", data)
    with pytest.raises(FileExistsError, match="already exists"):
        generate("t0-micro", data)


def test_legacy_schema_is_explicit_and_has_no_runtime_hash_allow_list(tmp_path):
    data = tmp_path / "legacy"
    manifest = generate("t0-micro", data)
    generation_module = importlib.import_module("qem_bench.datasets.generate")

    assert manifest["dataset_schema_version"] == "legacy-v1"
    assert not hasattr(generation_module, "FROZEN_ITEM_STREAMS")
    assert not hasattr(generation_module, "_configuration_fingerprint")


def _configuration_leaves(value, path=()):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _configuration_leaves(nested, (*path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _configuration_leaves(nested, (*path, index))
    else:
        yield path, value


def _set_configuration_leaf(value, path, replacement):
    target = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement


def _different_configuration_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return f"{value}-mutated"
    raise AssertionError(f"missing mutation for configuration value {value!r}")


@pytest.mark.parametrize("preset", ("t0-micro", "t0-smoke"))
def test_frozen_serializer_snapshot_covers_every_configuration_field(preset):
    canonical = {
        **copy.deepcopy(PRESETS[preset]),
        "noise_family": DEFAULT_NOISE_FAMILY,
    }
    assert _uses_frozen_legacy_serializer(preset, canonical)

    for field in canonical:
        missing = copy.deepcopy(canonical)
        missing.pop(field)
        assert not _uses_frozen_legacy_serializer(preset, missing), field

    leaf_paths = list(_configuration_leaves(canonical))
    assert {path[0] for path, _ in leaf_paths} == set(canonical)
    for path, value in leaf_paths:
        mutated = copy.deepcopy(canonical)
        _set_configuration_leaf(
            mutated, path, _different_configuration_value(value)
        )
        assert not _uses_frozen_legacy_serializer(preset, mutated), path

    extra = {**canonical, "unknown_knob": "ignored"}
    assert not _uses_frozen_legacy_serializer(preset, extra)


def test_nested_public_preset_mutation_selects_exact_serializer(tmp_path):
    steps = PRESETS["t0-micro"]["steps"]
    original_first_step = steps[0]
    try:
        steps[0] += 2
        config = {
            **PRESETS["t0-micro"],
            "noise_family": DEFAULT_NOISE_FAMILY,
        }
        assert not _uses_frozen_legacy_serializer("t0-micro", config)

        data = tmp_path / "nested-preset-mutation"
        manifest = generate("t0-micro", data)
        items = [
            json.loads(line)
            for line in (data / "items.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]

        assert manifest["config"]["steps"] == steps
        assert manifest["dataset_hash"] != (
            "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"
        )
        assert all(item["stratum"] == "continuous_regression" for item in items)
        assert any(
            item["j"] != round(item["j"], 12)
            or item["h"] != round(item["h"], 12)
            for item in items
        )
    finally:
        steps[0] = original_first_step


def test_unversioned_artifact_is_rejected(tmp_path):
    data = tmp_path / "unversioned"
    generate("t0-micro", data)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("dataset_schema_version")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="dataset_schema_version is required"):
        _load(data)
