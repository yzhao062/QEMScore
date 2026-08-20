from __future__ import annotations

import json

import pytest

from qem_bench.budget import TIERS
from qem_bench.cli import main
from qem_bench.validation import SPLIT_SCHEMA_VERSION, validate_split_artifact


def test_cli_generates_validates_and_runs_split_v2_by_role(tmp_path, capsys):
    data_dir = tmp_path / "split-data"
    run_dir = tmp_path / "run"

    assert main(["generate", "--preset", "s0-t0-micro", "--out", str(data_dir)]) == 0
    items, manifest = validate_split_artifact(data_dir)

    assert len(items) == 18
    assert manifest["dataset_schema_version"] == SPLIT_SCHEMA_VERSION
    capsys.readouterr()

    assert main(
        [
            "run",
            "--data",
            str(data_dir),
            "--out",
            str(run_dir),
            "--tier",
            "H",
        ]
    ) == 0
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assignment = results["role_assignment"]
    selection = results["methods"]["ridge"]["config"]["role_protocol"]

    assert assignment["item_counts"] == {"train": 10, "validation": 4, "test": 4}
    assert assignment["test_items_used_for_selection"] == 0
    assert selection["selection_item_ids_sha256"] == assignment["item_ids_sha256"][
        "validation"
    ]
    assert selection["selection_item_ids_sha256"] != assignment["item_ids_sha256"][
        "test"
    ]
    assert len(results["budget"]) == 1
    budget_cell = next(iter(results["budget"].values()))
    assert budget_cell["tier"] == "H"
    assert budget_cell["cap"] == TIERS["H"]
    assert results["methods"]["liao"]["config"]["feature_fidelity"][
        "published_liao_encoding_reproduced"
    ] is False
    output = capsys.readouterr()
    assert "roles train=10, validation=4, test=4" in output.out
    assert "test-used-for-selection=0" in output.out
    assert output.err == ""


def test_cli_split_v2_requires_a_declared_tier(tmp_path, capsys):
    data_dir = tmp_path / "split-data"
    assert main(["generate", "--preset", "s0-t0-micro", "--out", str(data_dir)]) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        main(["run", "--data", str(data_dir), "--out", str(tmp_path / "run")])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "split-v2 run requires a budget tier" in stderr
    assert "Traceback" not in stderr


def test_cli_legacy_generation_remains_reachable(tmp_path):
    data_dir = tmp_path / "legacy-data"

    assert main(["generate", "--preset", "t0-micro", "--out", str(data_dir)]) == 0

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_schema_version"] == "legacy-v1"
    assert manifest["dataset_hash"] == (
        "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7"
    )


def test_cli_help_states_split_v2_runner_support(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "generate legacy-v1 or split-v2 benchmark data" in help_text
    assert "run baselines on legacy-v1 or split-v2 data" in help_text
