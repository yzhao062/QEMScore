"""Environment-lock recording and hash-binding contracts."""

from __future__ import annotations

import copy
import json

import pytest

from qemscore.datasets.generate import generate
from qemscore.reproducibility import CI_LOCK_SHA256, CI_LOCK_SHA256_ENV
from qemscore.runner.run import run, validate_run_artifact
from qemscore.validation import validate_split_artifact

MISMATCHING_LOCK_SHA256 = "0" * 64
SECOND_MISMATCHING_LOCK_SHA256 = "1" * 64
T0_MICRO_HASH = "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"


def _contract(observed: str | None) -> dict[str, object]:
    return {
        "expected_ci_lock_sha256": CI_LOCK_SHA256,
        "observed_ci_lock_sha256": observed,
        "lock_verified": observed == CI_LOCK_SHA256,
    }


def test_absent_lock_records_unverified_dataset_and_run(monkeypatch, tmp_path):
    monkeypatch.delenv(CI_LOCK_SHA256_ENV, raising=False)
    data = tmp_path / "data"
    manifest = generate("t0-micro", data)
    results = run(data, tmp_path / "run")

    assert manifest["environment_contract"] == _contract(None)
    assert results["dataset_environment_contract"] == _contract(None)
    assert results["environment_contract"] == _contract(None)
    assert validate_run_artifact(results) is results


def test_matching_lock_records_verified_dataset_and_run(monkeypatch, tmp_path):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, CI_LOCK_SHA256)
    data = tmp_path / "data"
    manifest = generate("t0-micro", data)
    results = run(data, tmp_path / "run")

    assert manifest["environment_contract"] == _contract(CI_LOCK_SHA256)
    assert results["dataset_environment_contract"]["lock_verified"] is True
    assert results["environment_contract"]["lock_verified"] is True


def test_mismatching_lock_records_unverified_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, MISMATCHING_LOCK_SHA256)
    manifest = generate("t0-micro", tmp_path / "data")

    assert manifest["environment_contract"] == _contract(MISMATCHING_LOCK_SHA256)


def test_malformed_observed_lock_is_rejected_before_writing(monkeypatch, tmp_path):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, "not-a-sha256")
    data = tmp_path / "data"

    with pytest.raises(ValueError, match="invalid observed CI lock SHA-256"):
        generate("t0-micro", data)
    assert not data.exists()


def test_split_hash_excludes_environment_contract(monkeypatch, tmp_path):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, MISMATCHING_LOCK_SHA256)
    first = generate("s0-t0-micro", tmp_path / "first")
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, SECOND_MISMATCHING_LOCK_SHA256)
    second = generate("s0-t0-micro", tmp_path / "second")

    assert first["items_hash"] == second["items_hash"]
    assert first["split_spec_hash"] == second["split_spec_hash"]
    assert first["environment_contract"] != second["environment_contract"]
    assert first["dataset_hash"] == second["dataset_hash"]
    validate_split_artifact(tmp_path / "first")
    validate_split_artifact(tmp_path / "second")


def test_legacy_hash_excludes_contract_while_run_id_binds_both(monkeypatch, tmp_path):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, CI_LOCK_SHA256)
    matching_data = tmp_path / "matching-data"
    matching_manifest = generate("t0-micro", matching_data)
    matching_run = run(matching_data, tmp_path / "matching-run")

    monkeypatch.setenv(CI_LOCK_SHA256_ENV, MISMATCHING_LOCK_SHA256)
    mismatching_data = tmp_path / "mismatching-data"
    mismatching_manifest = generate("t0-micro", mismatching_data)
    different_run_environment = run(
        matching_data, tmp_path / "different-run-environment"
    )

    monkeypatch.setenv(CI_LOCK_SHA256_ENV, CI_LOCK_SHA256)
    different_dataset_environment = run(
        mismatching_data, tmp_path / "different-dataset-environment"
    )

    assert matching_manifest["dataset_hash"] == T0_MICRO_HASH
    assert mismatching_manifest["dataset_hash"] == T0_MICRO_HASH
    assert matching_run["artifact_id"] != different_run_environment["artifact_id"]
    assert matching_run["artifact_id"] != different_dataset_environment["artifact_id"]


def test_artifact_validators_reject_invalid_environment_contracts(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(CI_LOCK_SHA256_ENV, CI_LOCK_SHA256)
    split_data = tmp_path / "split"
    generate("s0-t0-micro", split_data)
    split_manifest_path = split_data / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_manifest["environment_contract"]["lock_verified"] = False
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid environment contract"):
        validate_split_artifact(split_data)

    legacy_data = tmp_path / "legacy"
    generate("t0-micro", legacy_data)
    results = run(legacy_data, tmp_path / "run")
    tampered = copy.deepcopy(results)
    tampered["environment_contract"]["observed_ci_lock_sha256"] = "malformed"
    with pytest.raises(ValueError, match="identity inputs are invalid"):
        validate_run_artifact(tampered)
