import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

import qem_bench.validation as validation_module
from qem_bench.circuits.qaoa import QAOAParams
from qem_bench.datasets.generate import generate
from qem_bench.datasets.schema import (
    LEGACY_SCHEMA_VERSION,
    SPLIT_SCHEMA_VERSION,
    canonical_physical_circuit_identity,
    validate_item,
)
from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split
from qem_bench.datasets.splits import SplitSpec
from qem_bench.observables import z_support_label
from qem_bench.validation import (
    canonical_hash,
    canonical_item_lines,
    canonical_json,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    validate_split_artifact,
    validate_split_spec,
)


def _load_artifact(data: Path) -> tuple[list[dict], dict]:
    items = [
        json.loads(line)
        for line in (data / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    return items, manifest


def _write_rehashed_artifact(data: Path, items: list[dict], manifest: dict) -> None:
    cell_hashes = cell_item_stream_hashes(items)
    for cell in manifest["cells"]:
        cell["item_stream_hash"] = cell_hashes[cell["cell_id"]]
    manifest["items_hash"] = item_stream_hash(items)
    manifest["dataset_hash"] = split_dataset_hash(
        spec_hash=manifest["split_spec_hash"],
        items_hash=manifest["items_hash"],
    )
    (data / "items.jsonl").write_text(
        "\n".join(canonical_item_lines(items)) + "\n", encoding="utf-8"
    )
    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _write_sidecar(data: Path, category: str, value: dict) -> tuple[str, str]:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    relative = (Path("sidecars") / category / f"{digest}.json").as_posix()
    path = data / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest, relative


def _refresh_sidecar_hashes(items: list[dict], manifest: dict) -> None:
    reference_fields = (
        ("circuit_sidecar", "circuit_hash"),
        ("noise_sidecar", "noise_config_hash"),
        ("observable_sidecar", "observable_hash"),
        ("counts_sidecar", "counts_hash"),
    )
    manifest["sidecar_hashes"] = dict(
        sorted(
            {
                item[path_field]: item[hash_field]
                for item in items
                for path_field, hash_field in reference_fields
            }.items()
        )
    )


def _rewrite_circuit_descriptor_and_all_hashes(
    data: Path, items: list[dict], manifest: dict, mutate
) -> dict:
    old_circuit_relative = items[0]["circuit_sidecar"]
    affected = [
        item for item in items if item["circuit_sidecar"] == old_circuit_relative
    ]
    descriptor = json.loads(
        (data / old_circuit_relative).read_text(encoding="utf-8")
    )
    mutate(descriptor)
    circuit_digest, circuit_relative = _write_sidecar(
        data, "circuits", descriptor
    )
    circuit_id = f"circuit-{canonical_hash(descriptor)}"

    counts_updates = {}
    for old_counts_relative in {
        item["counts_sidecar"] for item in affected
    }:
        counts = json.loads(
            (data / old_counts_relative).read_text(encoding="utf-8")
        )
        counts["circuit_id"] = circuit_id
        counts_updates[old_counts_relative] = _write_sidecar(data, "counts", counts)

    for item in affected:
        item["circuit_id"] = circuit_id
        item["circuit_hash"] = circuit_digest
        item["circuit_sidecar"] = circuit_relative
        counts_digest, counts_relative = counts_updates[item["counts_sidecar"]]
        item["counts_hash"] = counts_digest
        item["counts_sidecar"] = counts_relative
        item["measurement_group"] = (
            f"{item['cell_id']}:{circuit_id}:"
            f"i{item['instance']}:r{item['replicate']}:g0"
        )
        item_descriptor = {
            "cell_id": item["cell_id"],
            "circuit_id": circuit_id,
            "instance": item["instance"],
            "observable_id": item["observable_id"],
            "replicate": item["replicate"],
        }
        item["item_id"] = f"item-{canonical_hash(item_descriptor)}"

    manifest["sidecar_hashes"].pop(old_circuit_relative)
    manifest["sidecar_hashes"][circuit_relative] = circuit_digest
    for old_counts_relative, (counts_digest, counts_relative) in counts_updates.items():
        manifest["sidecar_hashes"].pop(old_counts_relative)
        manifest["sidecar_hashes"][counts_relative] = counts_digest
    manifest["sidecar_hashes"] = dict(sorted(manifest["sidecar_hashes"].items()))
    _write_rehashed_artifact(data, items, manifest)
    return descriptor


def _rewrite_circuit_rows_and_all_hashes(
    data: Path, items: list[dict], manifest: dict, mutate
) -> None:
    old_circuit_relative = items[0]["circuit_sidecar"]
    affected = [
        item for item in items if item["circuit_sidecar"] == old_circuit_relative
    ]
    for item in affected:
        mutate(item)

    descriptor, circuit_id = canonical_physical_circuit_identity(affected[0])
    for item in affected[1:]:
        sibling_descriptor, sibling_id = canonical_physical_circuit_identity(item)
        assert sibling_descriptor == descriptor
        assert sibling_id == circuit_id
    circuit_digest, circuit_relative = _write_sidecar(
        data, "circuits", descriptor
    )

    counts_updates = {}
    for old_counts_relative in {
        item["counts_sidecar"] for item in affected
    }:
        counts = json.loads(
            (data / old_counts_relative).read_text(encoding="utf-8")
        )
        counts["circuit_id"] = circuit_id
        counts_updates[old_counts_relative] = _write_sidecar(data, "counts", counts)

    observable_updates = {}
    for item in affected:
        old_observable_relative = item["observable_sidecar"]
        key = (
            old_observable_relative,
            item["n_qubits"],
            item["pauli_label"],
        )
        if key not in observable_updates:
            observable = json.loads(
                (data / old_observable_relative).read_text(encoding="utf-8")
            )
            observable["n_qubits"] = item["n_qubits"]
            observable["pauli_label"] = item["pauli_label"]
            observable_updates[key] = _write_sidecar(
                data, "observables", observable
            )

    for item in affected:
        item["circuit_id"] = circuit_id
        item["circuit_hash"] = circuit_digest
        item["circuit_sidecar"] = circuit_relative
        counts_digest, counts_relative = counts_updates[item["counts_sidecar"]]
        item["counts_hash"] = counts_digest
        item["counts_sidecar"] = counts_relative
        observable_key = (
            item["observable_sidecar"],
            item["n_qubits"],
            item["pauli_label"],
        )
        observable_digest, observable_relative = observable_updates[observable_key]
        item["observable_hash"] = observable_digest
        item["observable_sidecar"] = observable_relative
        item["measurement_group"] = (
            f"{item['cell_id']}:{circuit_id}:"
            f"i{item['instance']}:r{item['replicate']}:g0"
        )
        item_descriptor = {
            "cell_id": item["cell_id"],
            "circuit_id": circuit_id,
            "instance": item["instance"],
            "observable_id": item["observable_id"],
            "replicate": item["replicate"],
        }
        item["item_id"] = f"item-{canonical_hash(item_descriptor)}"

    reference_fields = (
        ("circuit_sidecar", "circuit_hash"),
        ("noise_sidecar", "noise_config_hash"),
        ("observable_sidecar", "observable_hash"),
        ("counts_sidecar", "counts_hash"),
    )
    manifest["sidecar_hashes"] = dict(
        sorted(
            {
                item[path_field]: item[hash_field]
                for item in items
                for path_field, hash_field in reference_fields
            }.items()
        )
    )
    _write_rehashed_artifact(data, items, manifest)


def _single_family_spec(
    family: str, family_parameters: dict, *, n_qubits: int = 4
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
        n_qubits=[n_qubits],
        role_counts={"train": 1, "validation": 1, "test": 1},
        family_parameters={family: family_parameters},
    )


_VALID_FAMILY_PARAMETERS = {
    "tfi": {"dt": 0.2},
    "heisenberg": {"dt": 0.15},
    "qaoa": {"graph_classes": ["path"]},
    "random_clifford": {},
    "near_clifford": {
        "non_clifford_count": [1],
        "theta": [0.4487989505128276],
    },
}


@pytest.mark.parametrize(
    ("family", "field", "value"),
    (
        pytest.param("tfi", "j", [999.0], id="tfi-j"),
        pytest.param("tfi", "h", [999.0], id="tfi-h"),
        pytest.param("heisenberg", "jx", [999.0], id="heisenberg-jx"),
        pytest.param("heisenberg", "jy", [999.0], id="heisenberg-jy"),
        pytest.param("heisenberg", "jz", [999.0], id="heisenberg-jz"),
        pytest.param("qaoa", "graph_class", "path", id="qaoa-graph-class"),
        pytest.param("qaoa", "edges", [[0, 1]], id="qaoa-edges"),
        pytest.param(
            "qaoa", "edge_probability", 0.4, id="qaoa-edge-probability"
        ),
        pytest.param("qaoa", "gammas", [0.1], id="qaoa-gammas"),
        pytest.param("qaoa", "betas", [0.1], id="qaoa-betas"),
    ),
)
def test_family_parameter_maps_reject_sampled_output_fields(family, field, value):
    parameters = {**copy.deepcopy(_VALID_FAMILY_PARAMETERS[family]), field: value}

    with pytest.raises(
        ValueError,
        match=rf"family_parameters\['{family}'\].*extra=\['{field}'\]",
    ):
        validate_split_spec(_single_family_spec(family, parameters))


@pytest.mark.parametrize("family", sorted(_VALID_FAMILY_PARAMETERS))
def test_every_family_parameter_map_rejects_an_unknown_field(family):
    parameters = {
        **copy.deepcopy(_VALID_FAMILY_PARAMETERS[family]),
        "unknown_knob": "ignored",
    }

    with pytest.raises(
        ValueError,
        match=rf"family_parameters\['{family}'\].*extra=\['unknown_knob'\]",
    ):
        validate_split_spec(_single_family_spec(family, parameters))


@pytest.mark.parametrize(
    ("family", "parameters", "missing"),
    (
        pytest.param("tfi", {}, "dt", id="tfi-dt"),
        pytest.param("heisenberg", {}, "dt", id="heisenberg-dt"),
        pytest.param("qaoa", {}, "graph_classes", id="qaoa-graph-classes"),
        pytest.param(
            "near_clifford",
            {"theta": [0.4487989505128276]},
            "non_clifford_count",
            id="near-clifford-count",
        ),
        pytest.param(
            "near_clifford",
            {"non_clifford_count": [1]},
            "theta",
            id="near-clifford-theta",
        ),
    ),
)
def test_family_parameter_maps_require_every_generator_input(
    family, parameters, missing
):
    with pytest.raises(
        ValueError,
        match=rf"family_parameters\['{family}'\].*missing=\['{missing}'\]",
    ):
        validate_split_spec(_single_family_spec(family, parameters))


def test_reviewer_unknown_knob_artifact_is_rejected(monkeypatch, tmp_path):
    parameters = {
        **_VALID_FAMILY_PARAMETERS["tfi"],
        "j": [999.0],
        "unknown_knob": "ignored",
    }
    spec = _single_family_spec("tfi", parameters, n_qubits=3)
    data = tmp_path / "reviewer-unknown-knob"

    with monkeypatch.context() as generation_bypass:
        generation_bypass.setattr(
            validation_module, "validate_split_spec", lambda spec: None
        )
        manifest = generate_split(spec, data)

    items, _ = _load_artifact(data)
    assert all(
        {"j", "unknown_knob"} <= set(pool["parameter_grid"])
        for pool in manifest["circuit_pools"]
    )
    assert all(item["j"] != 999.0 for item in items)

    with pytest.raises(
        ValueError,
        match=(
            r"family_parameters\['tfi'\] has invalid fields; "
            r"missing=\[\], extra=\['j', 'unknown_knob'\]"
        ),
    ):
        validate_split_artifact(data)


@pytest.fixture(scope="module")
def authored_pool_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("authored-pool-artifacts")
    cases = {
        "tfi": _single_family_spec("tfi", {"dt": 0.2}),
        "heisenberg": _single_family_spec("heisenberg", {"dt": 0.15}),
        "qaoa-path": _single_family_spec(
            "qaoa", {"graph_classes": ["path"]}
        ),
        "qaoa-fixed-er": _single_family_spec(
            "qaoa",
            {"graph_classes": ["erdos_renyi"], "er_edge_probability": 0.4},
        ),
        "random-clifford": _single_family_spec("random_clifford", {}),
        "near-clifford": _single_family_spec(
            "near_clifford",
            {
                "non_clifford_count": [1],
                "theta": [0.4487989505128276],
            },
        ),
    }
    artifacts = {}
    for name, spec in cases.items():
        data = root / name
        generate_split(spec, data)
        artifacts[name] = data
    return artifacts


def _set_unauthorized_authored_value(item: dict, field: str) -> None:
    if field == "n_qubits":
        old_n_qubits = item[field]
        item[field] = old_n_qubits + 1
        item["pauli_label"] = f"I{item['pauli_label']}"
        if item["family"] == "qaoa" and item["graph_class"] == "path":
            item["edges"] = [*item["edges"], [old_n_qubits - 1, old_n_qubits]]
    elif field in {"steps", "depth"}:
        item[field] = 2
    elif field == "p":
        item[field] = 2
        item["gammas"] = [*item["gammas"], 0.1]
        item["betas"] = [*item["betas"], 0.1]
    elif field == "dt":
        item[field] = 0.3
    elif field == "graph_class":
        item[field] = "cycle"
        item["edges"] = [*item["edges"], [0, item["n_qubits"] - 1]]
    elif field == "edge_probability":
        item[field] = 0.5
    elif field == "non_clifford_count":
        item[field] = 2
    elif field == "theta":
        item[field] = 0.6283185307179586
    else:
        raise AssertionError(f"unhandled authored field {field}")


_UNAUTHORIZED_AUTHORED_FIELDS = (
    pytest.param("tfi", "n_qubits", "n_qubits", id="tfi-width"),
    pytest.param("tfi", "steps", "steps", id="tfi-depth"),
    pytest.param("tfi", "dt", "parameter_grid.dt", id="tfi-dt"),
    pytest.param("heisenberg", "n_qubits", "n_qubits", id="heisenberg-width"),
    pytest.param("heisenberg", "steps", "steps", id="heisenberg-depth"),
    pytest.param(
        "heisenberg", "dt", "parameter_grid.dt", id="heisenberg-dt"
    ),
    pytest.param("qaoa-path", "n_qubits", "n_qubits", id="qaoa-width"),
    pytest.param("qaoa-path", "p", "p", id="qaoa-depth"),
    pytest.param(
        "qaoa-path",
        "graph_class",
        "parameter_grid.graph_classes",
        id="qaoa-graph-class",
    ),
    pytest.param(
        "qaoa-fixed-er",
        "edge_probability",
        "parameter_grid.er_edge_probability",
        id="qaoa-fixed-er-probability",
    ),
    pytest.param(
        "random-clifford", "n_qubits", "n_qubits", id="random-clifford-width"
    ),
    pytest.param(
        "random-clifford", "depth", "depth", id="random-clifford-depth"
    ),
    pytest.param(
        "near-clifford", "n_qubits", "n_qubits", id="near-clifford-width"
    ),
    pytest.param(
        "near-clifford", "depth", "depth", id="near-clifford-depth"
    ),
    pytest.param(
        "near-clifford",
        "non_clifford_count",
        "parameter_grid.non_clifford_count",
        id="near-clifford-insertion-count",
    ),
    pytest.param(
        "near-clifford",
        "theta",
        "parameter_grid.theta",
        id="near-clifford-theta",
    ),
)


@pytest.mark.parametrize(
    ("artifact_name", "field", "mismatch_field"),
    _UNAUTHORIZED_AUTHORED_FIELDS,
)
def test_rehashed_rows_reject_values_outside_the_authored_pool_grid(
    tmp_path,
    authored_pool_artifacts,
    artifact_name,
    field,
    mismatch_field,
):
    data = tmp_path / artifact_name
    shutil.copytree(authored_pool_artifacts[artifact_name], data)
    items, manifest = _load_artifact(data)
    _rewrite_circuit_rows_and_all_hashes(
        data,
        items,
        manifest,
        lambda item: _set_unauthorized_authored_value(item, field),
    )

    with pytest.raises(
        ValueError,
        match=rf"disagrees with its cell.*{mismatch_field}",
    ):
        validate_split_artifact(data)


def _old_subset_sidecar_check(item: dict, circuit: dict) -> bool:
    expected_top_level = {
        "family": item["family"],
        "n_qubits": item["n_qubits"],
        "circuit_seed": item["circuit_seed"],
    }
    parameters = circuit.get("parameters")
    return (
        all(
            circuit.get(field) == expected
            for field, expected in expected_top_level.items()
        )
        and isinstance(parameters, dict)
        and all(item.get(field) == expected for field, expected in parameters.items())
        and item["circuit_id"] == f"circuit-{canonical_hash(circuit)}"
    )


def test_validation_is_the_first_qem_bench_import_in_isolated_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dependency_paths = [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    code = (
        "import sys; "
        f"sys.path[:0] = {dependency_paths!r}; "
        f"sys.path.insert(0, {str(root)!r}); "
        "import qem_bench.validation; "
        "from qem_bench.datasets import "
        "PRESETS, SPLIT_PRESETS, generate, generate_split; "
        "assert callable(generate) and callable(generate_split); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_public_generate_export_remains_callable_after_submodule_import():
    import importlib
    import qem_bench.datasets as datasets

    generate_module = importlib.import_module("qem_bench.datasets.generate")

    assert datasets.generate is generate_module.generate
    assert "generate" in dir(datasets)


def test_duplicate_axis_values_are_canonically_deduplicated(tmp_path):
    base = SPLIT_PRESETS["s0-t0-micro"]
    spec = replace(
        base,
        fixed_axes={
            **dict(base.fixed_axes),
            "noise_strength": ["L1", "L1"],
        },
    )

    manifest = generate_split(spec, tmp_path / "duplicate-axis")
    items, _ = validate_split_artifact(tmp_path / "duplicate-axis")

    assert spec.fixed_axes["noise_strength"] == ("L1",)
    assert len(manifest["cells"]) == 3
    assert len({cell["cell_id"] for cell in manifest["cells"]}) == 3
    assert len(items) == 18


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_qubits", [3.9], "n_qubits must contain positive integers"),
        (
            "role_counts",
            {"train": 5.9, "validation": 2, "test": 2},
            r"role_counts\['train'\] must be a positive integer",
        ),
        (
            "test_group_shot_budget",
            9.9,
            "test_group_shot_budget must be a positive integer",
        ),
    ),
)
def test_fractional_integer_declarations_are_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(SPLIT_PRESETS["s0-t0-micro"], **{field: value})


@pytest.mark.parametrize("value", (1.9, 1.0, True))
def test_family_native_depth_requires_an_exact_positive_integer(tmp_path, value):
    base = SPLIT_PRESETS["s0-t0-micro"]
    spec = replace(
        base,
        fixed_axes={
            **dict(base.fixed_axes),
            "family_native_depth": [value],
        },
    )

    with pytest.raises(
        ValueError, match="family_native_depth values must be positive integers"
    ):
        generate_split(spec, tmp_path / "fractional-depth")


@pytest.mark.parametrize("value", (512.0, True))
def test_shots_axis_requires_an_exact_positive_integer(value):
    base = SPLIT_PRESETS["s0-t0-micro"]
    spec = replace(
        base,
        fixed_axes={**dict(base.fixed_axes), "shots": [value]},
    )

    with pytest.raises(ValueError, match="shots values must be positive integers"):
        validate_split_spec(spec)


@pytest.mark.parametrize("value", (7.9, 7.0, True, "7"))
def test_split_master_seed_requires_an_exact_nonnegative_integer(tmp_path, value):
    with pytest.raises(
        ValueError, match="master_seed must be a nonnegative integer"
    ):
        generate_split(
            SPLIT_PRESETS["s0-t0-micro"], tmp_path / "invalid-seed", value
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_qubits", [True], "n_qubits must contain positive integers"),
        ("replicates", [False], "replicates must contain nonnegative integers"),
        (
            "role_counts",
            {"train": True, "validation": 2, "test": 2},
            r"role_counts\['train'\] must be a positive integer",
        ),
        (
            "test_group_shot_budget",
            True,
            "test_group_shot_budget must be a positive integer",
        ),
    ),
)
def test_boolean_integer_declarations_are_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(SPLIT_PRESETS["s0-t0-micro"], **{field: value})


def test_unknown_authored_budget_tier_is_rejected():
    spec = replace(SPLIT_PRESETS["s0-t0-micro"], budget_tier="unknown")

    with pytest.raises(ValueError, match="unknown budget_tier 'unknown'"):
        validate_split_spec(spec)


def test_authored_test_group_shot_budget_binds_realized_test_cost(tmp_path):
    data = tmp_path / "test-budget"
    spec = replace(
        SPLIT_PRESETS["s0-t0-micro"], test_group_shot_budget=1
    )
    generate_split(spec, data)

    with pytest.raises(
        ValueError,
        match="test circuit-evaluation budget exceeded: declared=1, actual=1024",
    ):
        validate_split_artifact(data)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_split_values_are_rejected(value):
    base = SPLIT_PRESETS["s0-t0-micro"]
    with pytest.raises(ValueError, match="numeric values must be finite"):
        replace(
            base,
            fixed_axes={
                **dict(base.fixed_axes),
                "family_native_depth": [value],
            },
        )


def test_schema_invalid_rows_are_rejected_after_all_hashes_are_recomputed(tmp_path):
    data = tmp_path / "schema-invalid"
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    items[0]["label_method"] = "unsupported-method"
    items[0]["stratum"] = "clifford_control"
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(
        ValueError,
        match="family 'tfi' requires stratum 'continuous_regression'",
    ):
        validate_split_artifact(data)


@pytest.mark.protocol("QEM-P004")
def test_result_only_fields_are_rejected_after_all_hashes_are_recomputed(tmp_path):
    data = tmp_path / "result-fields"
    generate("s0-t0-micro", data)
    original_items, original_manifest = _load_artifact(data)
    excluded = (
        ("prediction", 0.0),
        ("residual", 0.0),
        ("accept", True),
        ("method", "raw"),
        ("mae", 0.0),
        ("wall_clock_seconds", 1.0),
    )

    for field, value in excluded:
        items = copy.deepcopy(original_items)
        manifest = copy.deepcopy(original_manifest)
        items[0][field] = value
        _write_rehashed_artifact(data, items, manifest)
        with pytest.raises(ValueError, match=rf"unsupported fields: \['{field}'\]"):
            validate_split_artifact(data)


def test_realized_family_depth_matches_the_declared_cell_depth(tmp_path):
    data = tmp_path / "depth-mismatch"
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    group = items[0]["measurement_group"]
    for item in items:
        if item["measurement_group"] == group:
            item["steps"] += 1
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(ValueError, match="disagrees with its cell.*steps"):
        validate_split_artifact(data)


@pytest.mark.protocol("QEM-P006")
def test_split_manifest_seed_contract_is_recomputed(tmp_path):
    data = tmp_path / "seed-contract"
    generate("s0-t0-micro", data)
    _, original = _load_artifact(data)
    manifest_path = data / "manifest.json"

    changed_seed = copy.deepcopy(original)
    changed_seed["master_seed"] = 8
    changed_seed["seed_scheme"]["master_seed"] = 8
    manifest_path.write_text(
        json.dumps(changed_seed, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disagrees with its cell.*circuit_seed"):
        validate_split_artifact(data)

    negative_seed = copy.deepcopy(original)
    negative_seed["master_seed"] = -1
    negative_seed["seed_scheme"]["master_seed"] = -1
    manifest_path.write_text(
        json.dumps(negative_seed, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="master_seed must be a nonnegative integer"):
        validate_split_artifact(data)

    wrong_scheme = copy.deepcopy(original)
    wrong_scheme["seed_scheme"]["formula"] = "unbound"
    manifest_path.write_text(
        json.dumps(wrong_scheme, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="seed_scheme does not match master_seed"):
        validate_split_artifact(data)


def test_valid_split_artifact_passes_numeric_schema_validation(tmp_path):
    data = tmp_path / "valid-numeric-schema"
    generate("s0-t0-micro", data)

    items, manifest = validate_split_artifact(data)

    assert len(items) == 18
    assert manifest["counts"]["items"] == 18


@pytest.mark.parametrize("field", ("n_qubits", "circuit_seed", "sampler_seed"))
def test_integer_row_types_are_rejected_after_all_hashes_are_recomputed(
    tmp_path, field
):
    data = tmp_path / field
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    items[0][field] = float(items[0][field])
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(ValueError, match=rf"field {field} must be an integer"):
        validate_split_artifact(data)


def test_validate_item_rejects_wrong_numeric_types_and_nonfinite_values(tmp_path):
    data = tmp_path / "numeric-fields"
    generate("s0-t0-micro", data)
    items, _ = _load_artifact(data)
    item = items[0]

    integer_fields = (
        "instance",
        "n_qubits",
        "circuit_seed",
        "obs_locality",
        "shots",
        "sampler_seed",
        "two_qubit_gates",
        "transpiled_depth",
        "replicate",
        "raw_circuit_evals",
        "steps",
    )
    for field in integer_fields:
        mutated = {**item, field: float(item[field])}
        with pytest.raises(ValueError, match=rf"field {field} must be an integer"):
            validate_item(mutated, schema_version=SPLIT_SCHEMA_VERSION)

    finite_fields = (
        "noisy_expectation",
        "noisy_stderr",
        "ideal_expectation",
        "j",
        "h",
        "dt",
    )
    for field in finite_fields:
        mutated = {**item, field: float("inf")}
        with pytest.raises(ValueError, match=rf"field {field} must be finite"):
            validate_item(mutated, schema_version=SPLIT_SCHEMA_VERSION)


def test_validate_item_checks_qaoa_nested_numeric_fields(tmp_path):
    data = tmp_path / "qaoa-numeric-fields"
    generate("t0-qaoa-micro", data)
    items, _ = _load_artifact(data)
    item = items[0]

    validate_item(item, schema_version=LEGACY_SCHEMA_VERSION)

    mutations = (
        ("p", float(item["p"]), "field p must be an integer"),
        ("edge_probability", float("inf"), "field edge_probability must be finite"),
        ("gammas", [float("nan")], "field gammas must be finite"),
        ("betas", [True], "field betas must be a number"),
        ("edges", [[0, 1.0]], "field edges must contain integer pairs"),
    )
    for field, value, message in mutations:
        mutated = {**item, field: value}
        with pytest.raises(ValueError, match=message):
            validate_item(mutated, schema_version=LEGACY_SCHEMA_VERSION)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("domain", "target", "disagrees with its cell.*domain"),
        ("axis_values", {}, "disagrees with its cell.*axis_values"),
    ),
)
def test_row_to_cell_mismatches_survive_hash_recomputation(
    tmp_path, field, value, message
):
    data = tmp_path / field
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    # Pick a row whose current value differs from the one being written, so the
    # mutation is always a real change. Row order follows the identity hash and
    # is not stable, so items[0] may already hold the value under test.
    target = next(
        (item for item in items if item[field] != value),
        None,
    )
    assert target is not None, f"every row already has {field}={value!r}"
    target[field] = value
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(ValueError, match=message):
        validate_split_artifact(data)


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    (
        (
            "counts",
            lambda value: {**value, "items": value["items"] + 1},
            "manifest counts do not match rows",
        ),
        (
            "generation_ledger",
            lambda value: {**value, "total_circuit_evals": 0},
            "manifest generation_ledger does not match",
        ),
    ),
)
def test_manifest_accounting_is_recomputed(tmp_path, field, mutate, message):
    data = tmp_path / field
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    manifest[field] = mutate(copy.deepcopy(manifest[field]))
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(ValueError, match=message):
        validate_split_artifact(data)


def test_sidecar_semantics_are_checked_after_all_hashes_are_recomputed(tmp_path):
    data = tmp_path / "sidecar"
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    old_relative = items[0]["circuit_sidecar"]
    affected = [item for item in items if item["circuit_sidecar"] == old_relative]
    descriptor = json.loads((data / old_relative).read_text(encoding="utf-8"))
    descriptor["n_qubits"] += 1
    payload = (canonical_json(descriptor) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    new_relative = (Path("sidecars") / "circuits" / f"{digest}.json").as_posix()
    new_path = data / new_relative
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(payload)
    for item in affected:
        item["circuit_hash"] = digest
        item["circuit_sidecar"] = new_relative
    manifest["sidecar_hashes"].pop(old_relative)
    manifest["sidecar_hashes"][new_relative] = digest
    manifest["sidecar_hashes"] = dict(sorted(manifest["sidecar_hashes"].items()))
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(
        ValueError, match="circuit sidecar is not the exact canonical descriptor"
    ):
        validate_split_artifact(data)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda descriptor: descriptor["parameters"].pop("h"),
        lambda descriptor: descriptor["parameters"].__setitem__("extra", None),
    ),
    ids=("deleted-required-parameter", "added-null-parameter"),
)
def test_circuit_sidecar_is_closed_after_every_dependent_hash_is_recomputed(
    tmp_path, mutate
):
    data = tmp_path / "closed-circuit-sidecar"
    generate("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    descriptor = _rewrite_circuit_descriptor_and_all_hashes(
        data, items, manifest, mutate
    )

    assert _old_subset_sidecar_check(items[0], descriptor)

    with pytest.raises(
        ValueError, match="circuit sidecar is not the exact canonical descriptor"
    ):
        validate_split_artifact(data)


def test_rehashed_noise_sidecar_is_bound_to_installed_generator_registry(tmp_path):
    data = tmp_path / "wrong-noise-registry"
    generate_split("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    item = items[0]
    old_relative = item["noise_sidecar"]
    affected = [row for row in items if row["noise_sidecar"] == old_relative]
    noise = json.loads((data / old_relative).read_text(encoding="utf-8"))

    family = item["noise_family"]
    severity = item["severity"]
    forged_parameters = manifest["noise_registry"]["configs"][family][severity]
    forged_parameters["p1"] = 0.5
    manifest["noise_registry"]["hash"] = canonical_hash(
        manifest["noise_registry"]["configs"]
    )
    noise["parameters"] = copy.deepcopy(forged_parameters)
    digest, relative = _write_sidecar(data, "noise", noise)
    for row in affected:
        row["noise_config_hash"] = digest
        row["noise_sidecar"] = relative
    _refresh_sidecar_hashes(items, manifest)
    _write_rehashed_artifact(data, items, manifest)

    assert noise["parameters"] == forged_parameters
    assert manifest["noise_registry"]["hash"] == canonical_hash(
        manifest["noise_registry"]["configs"]
    )
    with pytest.raises(
        ValueError,
        match="noise_registry does not match the installed severity generator",
    ):
        validate_split_artifact(data)


def test_rehashed_observable_support_is_derived_from_observable_id(
    monkeypatch, tmp_path
):
    generate_module = importlib.import_module("qem_bench.datasets.generate")
    original_support = generate_module._observable_support

    def moved_z_mid(name, params):
        if name == "z_mid" and params.n_qubits == 3:
            return (0,)
        return original_support(name, params)

    data = tmp_path / "wrong-observable-support"
    with monkeypatch.context() as generation_injection:
        generation_injection.setattr(
            generate_module, "_observable_support", moved_z_mid
        )
        generate_split("s0-t0-micro", data)

    items, _ = _load_artifact(data)
    item = next(row for row in items if row["observable_id"] == "z_mid")
    observable = json.loads(
        (data / item["observable_sidecar"]).read_text(encoding="utf-8")
    )
    assert observable["support"] == [0]
    assert item["pauli_label"] == "IIZ"

    with pytest.raises(
        ValueError,
        match=(
            r"observable support disagrees with observable_id 'z_mid': "
            r"actual=\[0\], expected=\[1\]"
        ),
    ):
        validate_split_artifact(data)


def test_rehashed_pauli_label_is_derived_from_observable_id(tmp_path):
    data = tmp_path / "wrong-pauli-label"
    generate_split("s0-t0-micro", data)
    items, manifest = _load_artifact(data)
    item = next(row for row in items if row["observable_id"] == "z_mid")
    observable = json.loads(
        (data / item["observable_sidecar"]).read_text(encoding="utf-8")
    )
    assert observable["support"] == [1]
    assert item["pauli_label"] == "IZI"

    observable["pauli_label"] = "ZII"
    digest, relative = _write_sidecar(data, "observables", observable)
    item["pauli_label"] = "ZII"
    item["observable_hash"] = digest
    item["observable_sidecar"] = relative
    _refresh_sidecar_hashes(items, manifest)
    _write_rehashed_artifact(data, items, manifest)

    with pytest.raises(
        ValueError,
        match=(
            r"pauli_label disagrees with observable_id 'z_mid': "
            r"actual='ZII', expected='IZI'"
        ),
    ):
        validate_split_artifact(data)


def test_all_generator_defined_observable_ids_validate(tmp_path):
    generate_module = importlib.import_module("qem_bench.datasets.generate")
    base = _single_family_spec("qaoa", {"graph_classes": ["path"]})
    spec = replace(
        base,
        fixed_axes={
            **dict(base.fixed_axes),
            "observable_class": ["z_mid", "zz_mid", "zz_edge"],
        },
    )
    data = tmp_path / "all-observables"
    generate_split(spec, data)

    items, _ = validate_split_artifact(data)
    assert {item["observable_id"] for item in items} == {
        "z_mid",
        "zz_mid",
        "zz_edge",
    }
    for item in items:
        params = QAOAParams(
            n_qubits=item["n_qubits"],
            graph_class=item["graph_class"],
            edges=tuple(tuple(edge) for edge in item["edges"]),
            p=item["p"],
            gammas=tuple(item["gammas"]),
            betas=tuple(item["betas"]),
            circuit_seed=item["circuit_seed"],
            instance=item["instance"],
            edge_probability=item["edge_probability"],
        )
        expected_support = generate_module._observable_support(
            item["observable_id"], params
        )
        observable = json.loads(
            (data / item["observable_sidecar"]).read_text(encoding="utf-8")
        )
        assert observable["support"] == list(expected_support)
        assert item["pauli_label"] == z_support_label(
            item["n_qubits"], expected_support
        )
        assert item["obs_locality"] == len(expected_support)


@pytest.mark.parametrize(
    ("family", "label_method"),
    (
        pytest.param("tfi", "statevector", id="tfi-statevector"),
        pytest.param("qaoa", "statevector", id="qaoa-statevector"),
        pytest.param("heisenberg", "statevector", id="heisenberg-statevector"),
        pytest.param(
            "near_clifford",
            "statevector",
            id="near-clifford-statevector",
        ),
        pytest.param("random_clifford", "stim", id="random-clifford-stim"),
    ),
)
def test_rehashed_ground_truth_is_recomputed_by_every_family_label_generator(
    family, label_method, monkeypatch, tmp_path
):
    split_generate_module = importlib.import_module(
        "qem_bench.datasets.split_generate"
    )
    calls = 0
    injected_label = 0.123456789012

    def wrong_label(circuit, pauli_label):
        nonlocal calls
        calls += 1
        return injected_label

    monkeypatch.setattr(
        split_generate_module, f"{label_method}_expectation", wrong_label
    )
    data = tmp_path / family
    generate_split(
        _single_family_spec(family, copy.deepcopy(_VALID_FAMILY_PARAMETERS[family])),
        data,
    )
    items, _ = _load_artifact(data)
    assert calls == len({item["circuit_id"] for item in items})
    assert {item["label_method"] for item in items} == {label_method}
    assert {item["ideal_expectation"] for item in items} == {injected_label}

    with pytest.raises(
        ValueError,
        match=rf"ideal_expectation disagrees with {label_method} generator",
    ):
        validate_split_artifact(data)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "tfi",
        "heisenberg",
        "qaoa-path",
        "qaoa-fixed-er",
        "random-clifford",
        "near-clifford",
    ),
)
def test_recomputed_labels_accept_every_unmodified_family_artifact(
    authored_pool_artifacts, artifact_name
):
    items, _ = validate_split_artifact(authored_pool_artifacts[artifact_name])

    assert items


def test_pauli_label_and_locality_schema_matches_z_support_contract(tmp_path):
    data = tmp_path / "pauli-schema"
    generate_split("s0-t0-micro", data)
    items, _ = _load_artifact(data)
    template = items[0]

    for n_qubits in range(1, 5):
        for length in range(n_qubits + 2):
            for characters in product("IZX", repeat=length):
                pauli_label = "".join(characters)
                candidate = {
                    **template,
                    "n_qubits": n_qubits,
                    "pauli_label": pauli_label,
                    "obs_locality": pauli_label.count("Z"),
                }
                valid = length == n_qubits and "X" not in pauli_label
                if valid:
                    validate_item(candidate, schema_version=SPLIT_SCHEMA_VERSION)
                else:
                    with pytest.raises(ValueError, match="field pauli_label"):
                        validate_item(candidate, schema_version=SPLIT_SCHEMA_VERSION)

    for invalid_label in (None, 1, True):
        with pytest.raises(ValueError, match="field pauli_label"):
            validate_item(
                {
                    **template,
                    "pauli_label": invalid_label,
                    "obs_locality": 0,
                },
                schema_version=SPLIT_SCHEMA_VERSION,
            )

    n_qubits = template["n_qubits"]
    pauli_label = z_support_label(n_qubits, (n_qubits // 2,))
    for obs_locality in (-1, 0, 1, 2, 3, 4, 1.0, True):
        candidate = {
            **template,
            "pauli_label": pauli_label,
            "obs_locality": obs_locality,
        }
        if obs_locality == 1 and type(obs_locality) is int:
            validate_item(candidate, schema_version=SPLIT_SCHEMA_VERSION)
        else:
            with pytest.raises(ValueError, match="field obs_locality"):
                validate_item(candidate, schema_version=SPLIT_SCHEMA_VERSION)
