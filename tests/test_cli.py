from __future__ import annotations

import json

import pytest

from qem_bench.cli import main
from qem_bench.validation import SPLIT_SCHEMA_VERSION, validate_split_artifact


def test_cli_generates_and_validates_split_v2_then_run_refuses_clearly(tmp_path, capsys):
    data_dir = tmp_path / "split-data"

    assert main(["generate", "--preset", "s0-t0-micro", "--out", str(data_dir)]) == 0
    items, manifest = validate_split_artifact(data_dir)

    assert len(items) == 18
    assert manifest["dataset_schema_version"] == SPLIT_SCHEMA_VERSION
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        main(["run", "--data", str(data_dir), "--out", str(tmp_path / "run")])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "split-v2 artifacts can be generated and validated" in stderr
    assert "run currently supports legacy-v1 artifacts only" in stderr
    assert "Traceback" not in stderr


def test_cli_legacy_generation_remains_reachable(tmp_path):
    data_dir = tmp_path / "legacy-data"

    assert main(["generate", "--preset", "t0-micro", "--out", str(data_dir)]) == 0

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_schema_version"] == "legacy-v1"
    assert manifest["dataset_hash"] == (
        "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"
    )


def test_cli_help_states_split_v2_runner_limit(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "split-v2 generation and validation" in help_text
    assert "run supports legacy-v1 only" in help_text
