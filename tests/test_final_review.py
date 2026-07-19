"""Mutation-sensitive contracts added for the final integration review."""

from __future__ import annotations

import json
import math
import shutil

import pytest

import qem_bench.noise as noise_package
from qem_bench.datasets.generate import PRESETS, dataset_hash, generate
from qem_bench.datasets.schema import (
    FAMILY_LABEL_METHODS,
    FAMILY_STRATA,
    FEATURES,
    build_features,
)
from qem_bench.noise import (
    DEFAULT_NOISE_FAMILY,
    NOISE_FAMILIES,
    SEVERITY_GRIDS,
    average_gate_infidelities,
)
from qem_bench.noise.models import (
    DEFAULT_NOISE_FAMILY as MODELS_DEFAULT_NOISE_FAMILY,
)
from qem_bench.noise.models import NOISE_FAMILIES as MODELS_NOISE_FAMILIES
from qem_bench.noise.models import SEVERITY_GRIDS as MODELS_SEVERITY_GRIDS
from qem_bench.noise.models import (
    average_gate_infidelities as models_average_gate_infidelities,
)
from qem_bench.runner.run import _load, run


FROZEN_HASHES = {
    "t0-micro": "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7",
    "t0-smoke": "edf5837e0da10f1dc93151d5b29d1855f66ad76341ff6c28019096308f8fcdf3",
}

FAMILY_INDICATORS = (
    "family_tfi",
    "family_qaoa",
    "family_heisenberg",
    "family_random_clifford",
    "family_near_clifford",
)

FAMILY_VALUE_FEATURES = (
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
)

REPRESENTATIVE_ITEM = {
    "noisy_expectation": 0.25,
    "shots": 1024,
    "n_qubits": 6,
    "two_qubit_gates": 11,
    "transpiled_depth": 17,
    "obs_locality": 2,
    # Nonzero values for every family source field make zero filling observable.
    "steps": 3,
    "j": 0.7,
    "h": 0.8,
    "jx": 0.31,
    "jy": 0.41,
    "jz": 0.51,
    "dt": 0.2,
    "p": 2,
    "edges": [[0, 1], [1, 2], [2, 3]],
    "gammas": [0.11, 0.22],
    "betas": [0.33, 0.44],
    "graph_class": "cycle",
    "depth": 7,
    "non_clifford_count": 2,
    "theta": 0.63,
}

FAMILY_FEATURE_CASES = (
    (
        "tfi",
        {"steps": 3.0, "j": 0.7, "h": 0.8, "dt": 0.2},
    ),
    (
        "qaoa",
        {
            "qaoa_p": 2.0,
            "qaoa_edge_count": 3.0,
            "qaoa_gamma_0": 0.11,
            "qaoa_gamma_1": 0.22,
            "qaoa_beta_0": 0.33,
            "qaoa_beta_1": 0.44,
            "graph_cycle": 1.0,
        },
    ),
    (
        "heisenberg",
        {
            "steps": 3.0,
            "jx": 0.31,
            "jy": 0.41,
            "jz": 0.51,
            "dt": 0.2,
        },
    ),
    ("random_clifford", {"rc_depth": 7.0}),
    ("near_clifford", {"nc_non_clifford_count": 2.0}),
)

NEW_PRESET_CASES = (
    ("t0-qaoa-micro", "qaoa"),
    ("t0-heisenberg-micro", "heisenberg"),
    ("t0-rc-micro", "random_clifford"),
    ("t0-nc-micro", "near_clifford"),
)


def _read_items(data_dir) -> list[dict]:
    return [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_items_and_hash(data_dir, items: list[dict]) -> None:
    lines = [
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in sorted(items, key=lambda item: item["item_id"])
    ]
    (data_dir / "items.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_hash"] = dataset_hash(items)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _generated_feature_oracle(item: dict) -> list[float]:
    expected = {name: 0.0 for name in FEATURES}
    expected.update(
        {
            "noisy_expectation": float(item["noisy_expectation"]),
            "log2_shots": math.log2(item["shots"]),
            "n_qubits": float(item["n_qubits"]),
            f"family_{item['family']}": 1.0,
            "two_qubit_gates": float(item["two_qubit_gates"]),
            "transpiled_depth": float(item["transpiled_depth"]),
            "obs_locality": float(item["obs_locality"]),
        }
    )
    family = item["family"]
    if family == "tfi":
        expected.update(
            steps=float(item["steps"]),
            j=float(item["j"]),
            h=float(item["h"]),
            dt=float(item["dt"]),
        )
    elif family == "qaoa":
        expected.update(
            qaoa_p=float(item["p"]),
            qaoa_edge_count=float(len(item["edges"])),
            qaoa_gamma_0=float(item["gammas"][0]),
            qaoa_beta_0=float(item["betas"][0]),
        )
        if item["p"] > 1:
            expected["qaoa_gamma_1"] = float(item["gammas"][1])
            expected["qaoa_beta_1"] = float(item["betas"][1])
        expected[f"graph_{item['graph_class']}"] = 1.0
    elif family == "heisenberg":
        expected.update(
            steps=float(item["steps"]),
            jx=float(item["jx"]),
            jy=float(item["jy"]),
            jz=float(item["jz"]),
            dt=float(item["dt"]),
        )
    elif family == "random_clifford":
        expected["rc_depth"] = float(item["depth"])
    elif family == "near_clifford":
        expected["nc_non_clifford_count"] = float(item["non_clifford_count"])
    else:
        raise AssertionError(f"missing feature oracle for family {family!r}")
    return [expected[name] for name in FEATURES]


@pytest.fixture(scope="module")
def frozen_micro_dataset(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("final-review-micro") / "data"
    generate("t0-micro", data_dir)
    return data_dir


@pytest.mark.parametrize(("preset", "expected_hash"), FROZEN_HASHES.items())
def test_frozen_stream_hash_and_stratum_compatibility(
    tmp_path, preset, expected_hash
):
    data_dir = tmp_path / preset
    manifest = generate(preset, data_dir)
    raw_bytes = (data_dir / "items.jsonl").read_bytes()
    raw_items = _read_items(data_dir)

    expected_config = {**PRESETS[preset], "noise_family": DEFAULT_NOISE_FAMILY}
    assert manifest["config"] == expected_config
    assert manifest["dataset_hash"] == expected_hash
    assert b'"stratum":' not in raw_bytes
    assert all("stratum" not in item for item in raw_items)

    loaded, _ = _load(data_dir)
    assert {item["stratum"] for item in loaded} == {"continuous_regression"}


def test_master_seed_override_serializes_every_stratum(tmp_path):
    data_dir = tmp_path / "seed-override"
    manifest = generate("t0-micro", data_dir, master_seed=8)
    items = _read_items(data_dir)

    assert manifest["config"]["master_seed"] == 8
    assert manifest["dataset_hash"] != FROZEN_HASHES["t0-micro"]
    assert items
    assert all(item.get("stratum") == "continuous_regression" for item in items)
    loaded, _ = _load(data_dir)
    assert len(loaded) == len(items)


def test_noise_family_override_serializes_every_stratum(tmp_path):
    data_dir = tmp_path / "noise-override"
    manifest = generate(
        "t0-micro", data_dir, noise_family="coherent_overrotation"
    )
    items = _read_items(data_dir)

    assert manifest["config"]["noise_family"] == "coherent_overrotation"
    assert manifest["dataset_hash"] != FROZEN_HASHES["t0-micro"]
    assert items
    assert all(item.get("stratum") == "continuous_regression" for item in items)
    loaded, _ = _load(data_dir)
    assert len(loaded) == len(items)


@pytest.mark.parametrize(("family", "required"), FAMILY_FEATURE_CASES)
def test_per_family_feature_oracle(family, required):
    item = {**REPRESENTATIVE_ITEM, "family": family}
    vector = build_features(item)
    values = dict(zip(FEATURES, vector))

    assert len(vector) == len(FEATURES)
    expected_indicator = f"family_{family}"
    assert sum(values[name] for name in FAMILY_INDICATORS) == 1.0
    assert values[expected_indicator] == 1.0
    assert all(
        values[name] == 0.0
        for name in FAMILY_INDICATORS
        if name != expected_indicator
    )
    for name, expected in required.items():
        assert values[name] == pytest.approx(expected)
        assert values[name] != 0.0
    assert all(
        values[name] == 0.0
        for name in FAMILY_VALUE_FEATURES
        if name not in required
    )
    assert values["noisy_expectation"] == 0.25
    assert values["log2_shots"] == 10.0
    assert values["n_qubits"] == 6.0
    assert values["two_qubit_gates"] == 11.0
    assert values["transpiled_depth"] == 17.0
    assert values["obs_locality"] == 2.0


@pytest.mark.parametrize(("preset", "family"), NEW_PRESET_CASES)
def test_new_preset_generation_and_single_stratum_load(tmp_path, preset, family):
    data_dir = tmp_path / preset
    manifest = generate(preset, data_dir)
    raw_items = _read_items(data_dir)
    expected_stratum = FAMILY_STRATA[family]
    expected_label_method = FAMILY_LABEL_METHODS[family]

    assert raw_items
    assert all(item.get("stratum") == expected_stratum for item in raw_items)
    assert {item["label_method"] for item in raw_items} == {expected_label_method}
    for method in ("statevector", "stim"):
        expected_count = len(raw_items) if method == expected_label_method else 0
        assert manifest["generation_ledger"][f"label_evals_{method}"] == expected_count
    for item in raw_items:
        vector = build_features(item)
        assert len(vector) == len(FEATURES)
        assert vector == pytest.approx(_generated_feature_oracle(item))

    loaded, loaded_manifest = _load(data_dir)
    assert {item["stratum"] for item in loaded} == {expected_stratum}
    assert loaded_manifest["dataset_hash"] == manifest["dataset_hash"]


def test_run_rejects_manifest_noise_family_mutation(tmp_path):
    tampered = tmp_path / "manifest-noise-family"
    generate("t0-qaoa-micro", tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["noise_family"] = "coherent_overrotation"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest config\.noise_family"):
        run(tampered, tmp_path / "noise-family-output")


@pytest.mark.parametrize("field", ("severity_grid", "severity_grids"))
def test_load_rejects_manifest_noise_registry_mutation(
    frozen_micro_dataset, tmp_path, field
):
    tampered = tmp_path / field
    shutil.copytree(frozen_micro_dataset, tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "severity_grid":
        manifest[field]["L1"]["p1"] += 0.001
    else:
        manifest[field][DEFAULT_NOISE_FAMILY]["L1"]["p1"] += 0.001
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match=f"manifest {field}"):
        _load(tampered)


def test_load_rejects_row_severity_outside_family_grid(
    frozen_micro_dataset, tmp_path
):
    tampered = tmp_path / "row-severity"
    shutil.copytree(frozen_micro_dataset, tampered)
    items = _read_items(tampered)
    first_group = items[0]["measurement_group"]
    for item in items:
        item["stratum"] = FAMILY_STRATA[item["family"]]
        if item["measurement_group"] == first_group:
            item["severity"] = "L9"
    _write_items_and_hash(tampered, items)

    with pytest.raises(ValueError, match="row severities are invalid"):
        _load(tampered)


def test_load_rejects_multiple_row_noise_families(
    frozen_micro_dataset, tmp_path
):
    tampered = tmp_path / "row-noise-families"
    shutil.copytree(frozen_micro_dataset, tampered)
    items = _read_items(tampered)
    first_group = items[0]["measurement_group"]
    for item in items:
        item["stratum"] = FAMILY_STRATA[item["family"]]
        if item["measurement_group"] == first_group:
            item["noise_family"] = "coherent_overrotation"
    _write_items_and_hash(tampered, items)

    with pytest.raises(ValueError, match="exactly one noise_family"):
        _load(tampered)


def test_noise_registry_public_import_contract():
    expected_exports = {
        "DEFAULT_NOISE_FAMILY",
        "NOISE_FAMILIES",
        "SEVERITY_GRIDS",
        "average_gate_infidelities",
    }
    assert expected_exports <= set(noise_package.__all__)
    assert DEFAULT_NOISE_FAMILY == MODELS_DEFAULT_NOISE_FAMILY
    assert NOISE_FAMILIES is MODELS_NOISE_FAMILIES
    assert SEVERITY_GRIDS is MODELS_SEVERITY_GRIDS
    assert average_gate_infidelities is models_average_gate_infidelities
