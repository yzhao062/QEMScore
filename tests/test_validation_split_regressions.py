import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qem_bench.datasets.generate import generate
from qem_bench.datasets.schema import validate_item
from qem_bench.datasets.split_generate import SPLIT_PRESETS, generate_split
from qem_bench.validation import (
    canonical_item_lines,
    canonical_json,
    cell_item_stream_hashes,
    item_stream_hash,
    split_dataset_hash,
    validate_split_artifact,
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
            validate_item(mutated)

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
            validate_item(mutated)


def test_validate_item_checks_qaoa_nested_numeric_fields(tmp_path):
    data = tmp_path / "qaoa-numeric-fields"
    generate("t0-qaoa-micro", data)
    items, _ = _load_artifact(data)
    item = items[0]

    validate_item(item)

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
            validate_item(mutated)


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
    items[0][field] = value
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
        ValueError, match="circuit sidecar disagrees on n_qubits"
    ):
        validate_split_artifact(data)
