"""Tests for fresh confirmation panel infrastructure (Unit r8-u1-fresh).

Covers:
1. tools/fresh_panel.py generate:
   - refuses seeds in campaign design.SEEDS (101, 211, 307)
   - refuses existing directories
   - validates split artifact and produces fresh_dataset.json with complete metadata
2. tools/fresh_panel.py verify:
   - tests byte-identical determinism check
3. tools/descriptor_ladder.py prepare --fresh:
   - prepares primary fresh datasets without archive
   - records fresh flag in datasets.json
4. tools/shot_sweep.py generate on fresh source:
   - derives shot levels (256, exact) directly from fresh source
5. tools/descriptor_ladder.py prepare --sweep --fresh --fresh-source DIR:
   - verifies items_hash and circuit IDs / ideal values against fresh source
6. tools/descriptor_ladder.py run --fresh:
   - tests --dry-run without comparing against archive records
   - runs tiny real fits on rehearsal data end to end
   - runs descriptor_ladder_analysis.py on fresh fits
7. Dataset seeds parameterization:
   - verifies tools accept --dataset-seeds
"""

import json
from pathlib import Path
import sys
import pytest

from qemscore.campaign import design
from tools import fresh_panel
from tools import descriptor_ladder
from tools import descriptor_ladder_analysis as dla
from tools import shot_sweep


def test_fresh_panel_refuses_campaign_seeds(tmp_path: Path):
    for seed in design.SEEDS:
        with pytest.raises(ValueError, match="[Rr]efus"):
            fresh_panel.generate_fresh_datasets(
                seeds=[seed],
                out_dir=tmp_path / "out",
                size=640,
                counts=(6, 4, 4),
            )


def test_fresh_panel_refuses_existing_dir(tmp_path: Path):
    out_dir = tmp_path / "out"
    target_dir = out_dir / "regen-shipped-s9001-n640"
    target_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="[Rr]efus|[Ee]xists"):
        fresh_panel.generate_fresh_datasets(
            seeds=[9001],
            out_dir=out_dir,
            size=640,
            counts=(6, 4, 4),
        )


def test_fresh_panel_generate_and_metadata(tmp_path: Path):
    out_dir = tmp_path / "data"
    res = fresh_panel.generate_fresh_datasets(
        seeds=[9001],
        out_dir=out_dir,
        size=640,
        counts=(6, 4, 4),
    )
    assert len(res) == 1
    info = res[0]
    target_dir = Path(info["target_dir"])
    assert target_dir.is_dir()
    assert (target_dir / "items.jsonl").is_file()
    assert (target_dir / "manifest.json").is_file()

    # Check fresh_dataset.json
    fresh_json = target_dir / "fresh_dataset.json"
    assert fresh_json.is_file()
    meta = json.loads(fresh_json.read_text(encoding="utf-8"))
    assert meta["seed"] == 9001
    assert meta["size"] == 640
    assert meta["counts"] == {"train": 6, "validation": 4, "test": 4}
    assert meta["dataset_hash"] == info["dataset_hash"]
    assert "sha256" in meta
    assert "items_jsonl" in meta["sha256"]
    assert "manifest_json" in meta["sha256"]
    assert "split_generate_py" in meta["sha256"]
    assert "design_py" in meta["sha256"]
    assert "fresh_panel_py" in meta["sha256"]
    assert "versions" in meta
    for k in ("python", "numpy", "scipy", "qiskit", "qiskit_aer"):
        assert k in meta["versions"]
    assert "platform" in meta
    assert "wall_time_seconds" in meta


def test_fresh_panel_verify_determinism(tmp_path: Path):
    out_dir = tmp_path / "data"
    res = fresh_panel.generate_fresh_datasets(
        seeds=[9001],
        out_dir=out_dir,
        size=640,
        counts=(6, 4, 4),
    )
    dataset_dir = Path(res[0]["target_dir"])
    verify_res = fresh_panel.verify_determinism(
        seed=9001,
        dataset_dir=dataset_dir,
        size=640,
        counts=(6, 4, 4),
    )
    assert verify_res["verified"] is True
    assert verify_res["items_byte_identical"] is True


def test_fresh_ladder_pipeline_end_to_end(tmp_path: Path):
    try:
        # The encoder module reads ML-QEM's mlp.py when it is imported.
        from tools import mlqem_binned_angle_encoder
        mlqem_binned_angle_encoder.find_upstream_mlp_path()
    except FileNotFoundError:
        pytest.skip("the R4 encoder needs ML-QEM's mlp.py at b1eccf8 (set MLQEM_PATH)")
    # 1. Generate two rehearsal datasets
    data_dir = tmp_path / "data"
    fresh_panel.generate_fresh_datasets(
        seeds=[9001, 9002],
        out_dir=data_dir,
        size=640,
        counts=(6, 4, 4),
    )
    ds_9001 = data_dir / "regen-shipped-s9001-n640"
    ds_9002 = data_dir / "regen-shipped-s9002-n640"

    # 2. Prepare ladder tree with --fresh
    ladder_dir = tmp_path / "ladder-2048"
    prepare_args = [
        "prepare",
        "--fresh",
        "--datasets", str(ds_9001), str(ds_9002),
        "--out", str(ladder_dir),
    ]
    ret = descriptor_ladder.main(prepare_args)
    assert ret == 0
    datasets_json = ladder_dir / "datasets.json"
    assert datasets_json.is_file()
    dindex = json.loads(datasets_json.read_text(encoding="utf-8"))
    assert dindex.get("_fresh") is True
    assert dindex["shipped-s9001-n640"]["fresh"] is True
    assert dindex["shipped-s9002-n640"]["fresh"] is True
    assert (ladder_dir / "cache" / "shipped-s9001-n640.pkl").is_file()
    assert (ladder_dir / "cache" / "shipped-s9002-n640.pkl").is_file()

    # 3. Derive sweep levels from 9001 fresh source
    sweep_dir = tmp_path / "sweep"
    level_256_ds = sweep_dir / "256" / "regen-shipped-s9001-n640"
    level_exact_ds = sweep_dir / "exact" / "regen-shipped-s9001-n640"

    gen_256 = shot_sweep.generate(ds_9001, level_256_ds, shots=256)
    assert gen_256["shot_level"] == 256
    assert (level_256_ds / "items.jsonl").is_file()

    gen_exact = shot_sweep.generate(ds_9001, level_exact_ds, exact=True)
    assert gen_exact["shot_level"] == "exact"
    assert (level_exact_ds / "items.jsonl").is_file()

    # 4. Prepare sweep ladder tree with --sweep --fresh --fresh-source
    ladder_256_dir = tmp_path / "ladder-256"
    prepare_sweep_args = [
        "prepare",
        "--sweep",
        "--fresh",
        "--fresh-source", str(data_dir),
        "--datasets", str(level_256_ds),
        "--out", str(ladder_256_dir),
    ]
    ret_sw = descriptor_ladder.main(prepare_sweep_args)
    assert ret_sw == 0
    assert (ladder_256_dir / "cache" / "shipped-s9001-n640.pkl").is_file()

    # 5. Run descriptor_ladder run --dry-run
    gate_file = Path(__file__).resolve().parents[1] / "artifacts/descriptor-information/gate/gate_r0.json"
    dry_run_args = [
        "run",
        "--fresh",
        "--datasets", str(ds_9001), str(ds_9002),
        "--out", str(ladder_dir),
        "--rungs", "R0", "N1", "N2", "R5",
        "--gate-file", str(gate_file),
        "--dry-run",
    ]
    ret_dry = descriptor_ladder.main(dry_run_args)
    assert ret_dry == 0

    # 6. Run tiny real fits (limit to 2 fits on learner seed 1)
    run_args = [
        "run",
        "--fresh",
        "--datasets", str(ds_9001), str(ds_9002),
        "--out", str(ladder_dir),
        "--rungs", "R0",
        "--arms", "C", "F",
        "--seeds", "1",
        "--gate-file", str(gate_file),
        "--limit-jobs", "2",
        "--workers", "1",
    ]
    ret_run = descriptor_ladder.main(run_args)
    assert ret_run == 0
    fits_dir = ladder_dir / "fits"
    assert fits_dir.is_dir()
    fit_files = list(fits_dir.glob("*.json"))
    assert len(fit_files) >= 2

    # 7. Run descriptor_ladder_analysis on rehearsal fits
    analysis_out = tmp_path / "analysis-a.json"
    analysis_args = [
        "--fits", str(fits_dir),
        "--cache", str(ladder_dir / "cache"),
        "--dataset-seeds", "9001", "9002",
        "--draws", "50",
        "--out", str(analysis_out),
        "--quiet",
    ]
    dla_ret = dla.main(analysis_args)
    assert dla_ret == 0
    assert analysis_out.is_file()
    a_data = json.loads(analysis_out.read_text(encoding="utf-8"))
    assert "parts" in a_data
    assert "A" in a_data["parts"]
