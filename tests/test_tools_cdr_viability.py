import json
import shutil
from pathlib import Path

import pytest

from tools.cdr_viability_probe import (
    REVIEWED_PAYLOAD_SHA256,
    payload_sha256,
    verify_cdr_artifacts,
)


TOOLS_ROOT = Path(__file__).parents[1] / "tools"
ARTIFACT_PATH = TOOLS_ROOT / "cdr_viability_probe.json"
SUMMARY_PATH = TOOLS_ROOT / "cdr_viability_probe.summary.md"


def test_shipped_cdr_artifact_matches_reviewed_payload_and_summary():
    assert (
        verify_cdr_artifacts(ARTIFACT_PATH, SUMMARY_PATH)
        == REVIEWED_PAYLOAD_SHA256
    )


def test_cdr_artifact_guard_rejects_a_mutated_payload_byte(tmp_path):
    artifact_path = tmp_path / ARTIFACT_PATH.name
    summary_path = tmp_path / SUMMARY_PATH.name
    shutil.copyfile(ARTIFACT_PATH, artifact_path)
    shutil.copyfile(SUMMARY_PATH, summary_path)
    content = artifact_path.read_bytes()
    needle = b'"schema_version": "cdr-viability-probe-v1"'
    replacement = b'"schema_version": "cdr-viability-probe-w1"'
    assert content.count(needle) == 1
    artifact_path.write_bytes(content.replace(needle, replacement, 1))

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        verify_cdr_artifacts(artifact_path, summary_path)


def test_cdr_artifact_guard_rejects_resigned_payload_drift(tmp_path):
    artifact_path = tmp_path / ARTIFACT_PATH.name
    summary_path = tmp_path / SUMMARY_PATH.name
    shutil.copyfile(ARTIFACT_PATH, artifact_path)
    shutil.copyfile(SUMMARY_PATH, summary_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema_version"] = "cdr-viability-probe-w1"
    artifact["payload_sha256"] = payload_sha256(artifact)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed payload drift"):
        verify_cdr_artifacts(artifact_path, summary_path)


def test_cdr_artifact_guard_rejects_summary_drift(tmp_path):
    artifact_path = tmp_path / ARTIFACT_PATH.name
    summary_path = tmp_path / SUMMARY_PATH.name
    shutil.copyfile(ARTIFACT_PATH, artifact_path)
    shutil.copyfile(SUMMARY_PATH, summary_path)
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + "edited\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="summary drift"):
        verify_cdr_artifacts(artifact_path, summary_path)
