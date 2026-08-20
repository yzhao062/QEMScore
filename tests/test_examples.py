import hashlib
import json
import shutil
from pathlib import Path

import pytest

from qem_bench.runner.run import validate_run_artifact


EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "report-walking-skeleton"
EXPECTED_RUN_PATHS = (
    "runs/t0-heisenberg-micro/results.json",
    "runs/t0-micro/results.json",
    "runs/t0-nc-micro/results.json",
    "runs/t0-qaoa-micro/results.json",
    "runs/t0-rc-micro/results.json",
    "runs/t0-smoke/results.json",
)
RUN_ARTIFACTS = tuple(EXAMPLE_ROOT / path for path in EXPECTED_RUN_PATHS)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_walking_skeleton_authentication(root: Path) -> None:
    actual_paths = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in (root / "runs").rglob("results.json")
        )
    )
    assert actual_paths == EXPECTED_RUN_PATHS, (
        f"walking-skeleton run inventory drift: expected {EXPECTED_RUN_PATHS}, "
        f"found {actual_paths}"
    )

    manifest_path = root / "results-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    declared_paths = tuple(entry["results"] for entry in manifest["runs"])
    assert declared_paths == EXPECTED_RUN_PATHS
    assert all(set(entry) == {"results"} for entry in manifest["runs"])

    trace = _load_json(root / "report-trace.json")
    assert trace["source_manifest"] == manifest_path.name
    assert trace["source_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    source_runs = trace.get("source_runs")
    assert isinstance(source_runs, list), (
        "report trace must authenticate source_runs with path, sha256, and artifact_id"
    )
    assert all(
        isinstance(source_run, dict)
        and set(source_run) == {"path", "sha256", "artifact_id"}
        for source_run in source_runs
    )
    assert tuple(source_run["path"] for source_run in source_runs) == EXPECTED_RUN_PATHS

    artifact_ids = []
    for source_run in source_runs:
        run_path = root / source_run["path"]
        run_bytes = run_path.read_bytes()
        assert source_run["sha256"] == (
            "sha256:" + hashlib.sha256(run_bytes).hexdigest()
        ), f"run byte SHA-256 mismatch: {source_run['path']}"
        artifact = json.loads(run_bytes)
        assert source_run["artifact_id"] == artifact["artifact_id"]
        assert validate_run_artifact(artifact) is artifact
        artifact_ids.append(artifact["artifact_id"])
    assert trace["artifacts"] == artifact_ids


def _copy_authenticated_inputs(destination: Path) -> None:
    for relative_path in (
        *EXPECTED_RUN_PATHS,
        "results-manifest.json",
        "report-trace.json",
    ):
        source = EXAMPLE_ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _record_source_runs_for_mutation_probe(root: Path) -> None:
    trace_path = root / "report-trace.json"
    trace = _load_json(trace_path)
    trace["source_runs"] = []
    for relative_path in EXPECTED_RUN_PATHS:
        run_bytes = (root / relative_path).read_bytes()
        artifact = json.loads(run_bytes)
        trace["source_runs"].append(
            {
                "path": relative_path,
                "sha256": "sha256:" + hashlib.sha256(run_bytes).hexdigest(),
                "artifact_id": artifact["artifact_id"],
            }
        )
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")


def test_walking_skeleton_inventory_and_run_bytes_are_authenticated():
    _assert_walking_skeleton_authentication(EXAMPLE_ROOT)


@pytest.mark.parametrize(
    "artifact_path", RUN_ARTIFACTS, ids=lambda path: path.parent.name
)
def test_walking_skeleton_run_artifact_validates(artifact_path):
    artifact = _load_json(artifact_path)

    assert validate_run_artifact(artifact) is artifact


def test_walking_skeleton_guard_rejects_a_removed_run(tmp_path):
    _copy_authenticated_inputs(tmp_path)
    _record_source_runs_for_mutation_probe(tmp_path)
    (tmp_path / EXPECTED_RUN_PATHS[0]).unlink()

    with pytest.raises(AssertionError, match="run inventory drift"):
        _assert_walking_skeleton_authentication(tmp_path)


def test_walking_skeleton_guard_rejects_a_mutated_run_byte(tmp_path):
    _copy_authenticated_inputs(tmp_path)
    _record_source_runs_for_mutation_probe(tmp_path)
    run_path = tmp_path / EXPECTED_RUN_PATHS[0]
    content = bytearray(run_path.read_bytes())
    newline_index = content.index(ord("\n"))
    content[newline_index] = ord(" ")
    run_path.write_bytes(content)

    with pytest.raises(AssertionError, match="run byte SHA-256 mismatch"):
        _assert_walking_skeleton_authentication(tmp_path)
