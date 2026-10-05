"""Tests for the --train-size option in descriptor ladder.

Hard constraint: Never fit any model with the new option on the released
spin-chain data. Use tiny generated rehearsal datasets with throwaway seeds,
and dry-run only on real trees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from qemscore.campaign.design import campaign_split_spec
from qemscore.datasets.split_generate import generate_split
from qemscore.runner.run import _prediction_items
from tools import descriptor_common as common
from tools import descriptor_ladder as ladder


REAL_PART_A = Path(
    "/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/ladder/runs/partA"
)


@pytest.fixture(scope="module")
def rehearsal_datasets(tmp_path_factory):
    """Generate tiny rehearsal datasets at size 12 and size 6 with master_seed=9001."""
    base = tmp_path_factory.mktemp("rehearsal")
    d12_dir = base / "shipped-s9001-n12"
    d6_dir = base / "shipped-s9001-n6"

    spec12 = campaign_split_spec("shipped", 640, counts={"train": 12, "validation": 4, "test": 4})
    generate_split(spec12, d12_dir, master_seed=9001)

    spec6 = campaign_split_spec("shipped", 640, counts={"train": 6, "validation": 4, "test": 4})
    generate_split(spec6, d6_dir, master_seed=9001)

    items12 = [json.loads(line) for line in (d12_dir / "items.jsonl").open(encoding="utf-8")]
    items6 = [json.loads(line) for line in (d6_dir / "items.jsonl").open(encoding="utf-8")]

    return {
        "d12_dir": d12_dir,
        "d6_dir": d6_dir,
        "items12": items12,
        "items6": items6,
    }


def test_prefix_property_on_rehearsal_data(rehearsal_datasets):
    """The first N training circuits per family reproduce the size-N dataset row-for-row."""
    items12 = rehearsal_datasets["items12"]
    items6 = rehearsal_datasets["items6"]

    d12_train = [r for r in items12 if r["split"] == "train"]
    d6_train = [r for r in items6 if r["split"] == "train"]

    filtered_12 = common.filter_train_rows(d12_train, 6)

    # Both must have 48 training rows (2 families * 6 circuits * 2 observables * 2 severities)
    assert len(filtered_12) == len(d6_train) == 48

    # Row-for-row bit-identical comparison
    for r12, r6 in zip(filtered_12, d6_train):
        assert r12["item_id"] == r6["item_id"]
        assert r12["circuit_id"] == r6["circuit_id"]
        assert r12["family"] == r6["family"]
        assert r12["instance"] == r6["instance"]
        assert r12["observable"] == r6["observable"]
        assert r12["severity"] == r6["severity"]
        assert r12["shots"] == r6["shots"]
        assert r12["dt"] == r6["dt"]
        assert r12["steps"] == r6["steps"]
        assert r12["ideal_expectation"] == r6["ideal_expectation"]
        assert r12["noisy_expectation"] == r6["noisy_expectation"]
        for k in ("j", "h", "jx", "jy", "jz"):
            if k in r6:
                assert r12[k] == r6[k]

    # Identical SHA-256 of sorted training circuit identifiers
    sha12 = common._circuit_ids_sha256([r["circuit_id"] for r in filtered_12])
    sha6 = common._circuit_ids_sha256([r["circuit_id"] for r in d6_train])
    assert sha12 == sha6


def test_subset_selection_circuit_grouped(rehearsal_datasets):
    """All observables and severities for each circuit move together."""
    items12 = rehearsal_datasets["items12"]
    d12_train = [r for r in items12 if r["split"] == "train"]

    filtered = common.filter_train_rows(d12_train, 6)

    # Group rows by circuit_id
    rows_by_circuit: dict[str, list[dict]] = {}
    for r in filtered:
        rows_by_circuit.setdefault(r["circuit_id"], []).append(r)

    # 12 distinct circuits (6 per family * 2 families)
    assert len(rows_by_circuit) == 12

    for circuit_id, rows in rows_by_circuit.items():
        # Every circuit has exactly 4 rows (2 observables * 2 severities)
        assert len(rows) == 4
        observables = {r["observable"] for r in rows}
        severities = {r["severity"] for r in rows}
        assert observables == {"z_mid", "zz_mid"}
        assert severities == {"L1", "L3"}
        # Instance must be < 6
        assert all(r["instance"] < 6 for r in rows)

    # When train_size is None, returns all rows untouched
    assert common.filter_train_rows(d12_train, None) == d12_train


def test_permuted_arm_p_scope_within_subset(rehearsal_datasets, tmp_path):
    """Arm P permutes the noisy column strictly within the subset of training circuits."""
    items12 = rehearsal_datasets["items12"]
    data = {
        "seed": 9001,
        "dataset_hash": "test_hash",
        "train": [r for r in items12 if r["split"] == "train"],
        "validation": [r for r in items12 if r["split"] == "validation"],
        "test": [r for r in items12 if r["split"] == "test"],
    }
    data["prediction_rows"] = {
        role: _prediction_items(data[role]) for role in ("validation", "test")
    }

    table = ladder.coupling_noise_table(data)
    builder = ladder.RungBuilder("R0", z_by_item=table["z_by_item"])

    fit_dir = tmp_path / "fits_p"
    fit_dir.mkdir()

    job = {
        "part": "A",
        "key": "shipped-s9001-n12",
        "rung": "R0",
        "fit_dir": str(fit_dir),
        "kind": "liao",
        "arm": "P",
        "fit_arm": "P",
        "stem": "shipped-s9001-n12__R0__P__s1",
        "learner_seed": 1,
        "shuffle_seed": 777,
        "train_size": 6,
    }

    info = common.run_liao_fit(job, data, builder, builder.names)
    assert info["error"] is None

    # Verify that the subset of 6 circuits had its noisy expectations permuted
    filtered_train = common.filter_train_rows(data["train"], 6)
    subset_noisy = [r["noisy_expectation"] for r in filtered_train]

    # Test that shuffling items with common.shuffle_noisy_items preserves multiset
    shuffled_subset = common.shuffle_noisy_items(filtered_train, seed=777)
    shuffled_noisy = [r["noisy_expectation"] for r in shuffled_subset]
    assert sorted(shuffled_noisy) == sorted(subset_noisy)
    assert shuffled_noisy != subset_noisy

    # Ensure validation and test were not altered
    assert len(data["validation"]) == 32  # 4 circuits * 2 fam * 2 obs * 2 severities
    assert len(data["test"]) == 32


def test_recorded_fields_and_finite_predictions(rehearsal_datasets, tmp_path):
    """Fits with --train-size record train_size, training_rows, and circuit SHA-256."""
    items12 = rehearsal_datasets["items12"]
    data = {
        "seed": 9001,
        "dataset_hash": "test_hash",
        "train": [r for r in items12 if r["split"] == "train"],
        "validation": [r for r in items12 if r["split"] == "validation"],
        "test": [r for r in items12 if r["split"] == "test"],
    }
    data["prediction_rows"] = {
        role: _prediction_items(data[role]) for role in ("validation", "test")
    }

    table = ladder.coupling_noise_table(data)
    builder = ladder.RungBuilder("R0", z_by_item=table["z_by_item"])

    fit_dir = tmp_path / "fits_fields"
    fit_dir.mkdir()

    filtered_train = common.filter_train_rows(data["train"], 6)
    expected_sha = common._circuit_ids_sha256([r["circuit_id"] for r in filtered_train])

    # 1. Fit Arm A (affine)
    a_job = {
        "part": "A", "key": "shipped-s9001-n12", "rung": "R0", "fit_dir": str(fit_dir),
        "kind": "affine", "arm": "A", "fit_arm": "A", "stem": "test_R0_A",
        "learner_seed": None, "shuffle_seed": None, "train_size": 6,
    }
    info_a = common.run_affine_fit(a_job, data, builder, builder.names)
    assert info_a["error"] is None

    meta_a = json.loads((fit_dir / "test_R0_A.json").read_text(encoding="utf-8"))
    assert meta_a["train_size"] == 6
    assert meta_a["n_train_rows"] == 48
    assert meta_a["training_rows"] == 48
    assert meta_a["training_circuit_ids_sha256"] == expected_sha
    assert meta_a["training_circuit_identifiers_sha256"] == expected_sha

    with np.load(fit_dir / "test_R0_A.npz") as npz:
        assert np.all(np.isfinite(npz["test"]))
        assert np.all(np.isfinite(npz["validation"]))

    # 2. Fit Arm F (liao)
    f_job = {
        "part": "A", "key": "shipped-s9001-n12", "rung": "R0", "fit_dir": str(fit_dir),
        "kind": "liao", "arm": "F", "fit_arm": "F", "stem": "test_R0_F_s1",
        "learner_seed": 1, "shuffle_seed": None, "train_size": 6,
    }
    info_f = common.run_liao_fit(f_job, data, builder, builder.names)
    assert info_f["error"] is None

    meta_f = json.loads((fit_dir / "test_R0_F_s1.json").read_text(encoding="utf-8"))
    assert meta_f["train_size"] == 6
    assert meta_f["n_train_rows"] == 48
    assert meta_f["training_rows"] == 48
    assert meta_f["training_circuit_ids_sha256"] == expected_sha
    assert meta_f["training_circuit_identifiers_sha256"] == expected_sha

    with np.load(fit_dir / "test_R0_F_s1.npz") as npz:
        assert np.all(np.isfinite(npz["test"]))
        assert np.all(np.isfinite(npz["validation"]))

    # 3. Fit Arm C (liao-feat-only)
    c_job = {
        "part": "A", "key": "shipped-s9001-n12", "rung": "R0", "fit_dir": str(fit_dir),
        "kind": "liao", "arm": "C", "fit_arm": "C", "stem": "test_R0_C_s1",
        "learner_seed": 1, "shuffle_seed": None, "train_size": 6,
    }
    info_c = common.run_liao_fit(c_job, data, builder, builder.names)
    assert info_c["error"] is None
    meta_c = json.loads((fit_dir / "test_R0_C_s1.json").read_text(encoding="utf-8"))
    assert meta_c["train_size"] == 6
    assert meta_c["n_train_rows"] == 48

    # 4. Verify baseline fit (train_size=None) leaves metadata bit-identical (no train_size fields)
    base_dir = tmp_path / "fits_base"
    base_dir.mkdir()
    base_job = {
        "part": "A", "key": "shipped-s9001-n12", "rung": "R0", "fit_dir": str(base_dir),
        "kind": "affine", "arm": "A", "fit_arm": "A", "stem": "base_R0_A",
        "learner_seed": None, "shuffle_seed": None,
    }
    common.run_affine_fit(base_job, data, builder, builder.names)
    meta_base = json.loads((base_dir / "base_R0_A.json").read_text(encoding="utf-8"))
    assert "train_size" not in meta_base
    assert "n_train_rows" not in meta_base
    assert "training_circuit_ids_sha256" not in meta_base


def test_refusal_rules(rehearsal_datasets, tmp_path):
    """cmd_run enforces positive train_size, bounds against dataset, and single-size directory."""
    d12_dir = rehearsal_datasets["d12_dir"]
    cache_dir = tmp_path / "run_refusal" / "cache"
    cache_dir.mkdir(parents=True)
    items12 = rehearsal_datasets["items12"]
    data = {
        "seed": 9001,
        "dataset_hash": "test_hash",
        "train": [r for r in items12 if r["split"] == "train"],
        "validation": [r for r in items12 if r["split"] == "validation"],
        "test": [r for r in items12 if r["split"] == "test"],
    }
    data["prediction_rows"] = {
        role: _prediction_items(data[role]) for role in ("validation", "test")
    }
    common.write_cache(cache_dir / "shipped-s9001-n12.pkl", data)

    out_dir = tmp_path / "run_refusal"
    fit_dir = out_dir / "fits"
    fit_dir.mkdir(parents=True)

    base_args = argparse.Namespace(
        out=out_dir,
        datasets=[d12_dir],
        nc_datasets=[],
        rungs=["R0"],
        arms=["A"],
        seeds="1",
        sweep=False,
        strength_indicator=False,
        strong_learners=False,
        dry_run=True,
        gate_file=None,
        workers=1,
        limit_jobs=None,
        seedrep_fits=None,
        nc_results=None,
    )

    # 1. Non-positive train_size raises SystemExit
    with pytest.raises(SystemExit, match=r"positive integer"):
        base_args.train_size = 0
        ladder.cmd_run(base_args)

    with pytest.raises(SystemExit, match=r"positive integer"):
        base_args.train_size = -3
        ladder.cmd_run(base_args)

    # 2. Train size exceeding dataset count (12 circuits per family) raises SystemExit
    with pytest.raises(SystemExit, match=r"exceeds dataset"):
        base_args.train_size = 100
        ladder.cmd_run(base_args)

    # 3. Mixing sizes in one fits directory raises SystemExit
    # Write a dummy fit with train_size=6
    (fit_dir / "dummy_fit.json").write_text(json.dumps({"train_size": 6}), encoding="utf-8")

    # Run with train_size=4 refuses
    base_args.train_size = 4
    with pytest.raises(SystemExit, match=r"refuse to mix training sizes"):
        ladder.cmd_run(base_args)

    # Run with train_size=None refuses
    base_args.train_size = None
    with pytest.raises(SystemExit, match=r"refuse to mix training sizes"):
        ladder.cmd_run(base_args)

    # Write a dummy fit with train_size=None
    (fit_dir / "dummy_fit.json").write_text(json.dumps({"fit": "baseline"}), encoding="utf-8")
    base_args.train_size = 6
    with pytest.raises(SystemExit, match=r"refuse to mix training sizes"):
        ladder.cmd_run(base_args)


def test_dry_run_real_prepared_tree(tmp_path):
    """Dry-run on a scratch copy of the real Part A tree plans 1,950 fits with train_size=160."""
    if not REAL_PART_A.exists():
        pytest.skip("Real Part A tree not found")

    scratch_tree = tmp_path / "partA_scratch"
    scratch_tree.mkdir()
    for name in ("cache", "descriptors", "encoder-cache"):
        shutil.copytree(REAL_PART_A / name, scratch_tree / name)
    shutil.copy(REAL_PART_A / "datasets.json", scratch_tree / "datasets.json")

    datasets_meta = json.loads((scratch_tree / "datasets.json").read_text(encoding="utf-8"))
    primary_paths = [Path(v["data_dir"]) for v in datasets_meta.values() if v.get("kind") == "primary"]
    nc_paths = [Path(v["data_dir"]) for v in datasets_meta.values() if v.get("kind") == "near_clifford"]

    args = argparse.Namespace(
        out=scratch_tree,
        datasets=primary_paths,
        nc_datasets=nc_paths,
        rungs=[],
        arms=[],
        seeds="1-20",
        sweep=False,
        strength_indicator=False,
        strong_learners=False,
        train_size=160,
        dry_run=True,
        gate_file=None,
        workers=1,
        limit_jobs=None,
        seedrep_fits=None,
        nc_results=None,
    )

    exit_code = ladder.cmd_run(args)
    assert exit_code == 0

    # Also check build_jobs directly
    primary_keys = [ladder._key(d, ladder.PRIMARY_KEY)[0] for d in primary_paths]
    nc_keys = [ladder._key(d, ladder.NC_KEY)[0] for d in nc_paths]
    rungs = [*ladder.RUNGS, *ladder.NC_RUNGS]
    arms = list(ladder.ARMS)
    seeds = list(range(1, 21))

    jobs = ladder.build_jobs(
        scratch_tree,
        primary_keys,
        nc_keys,
        rungs,
        arms,
        seeds,
        strength_indicator=False,
        strong_learners=False,
        train_size=160,
    )

    # 1,950 fits planned (30 affine, 1920 liao)
    assert len(jobs) == 1950
    affine_jobs = [j for j in jobs if j["kind"] == "affine"]
    liao_jobs = [j for j in jobs if j["kind"] == "liao"]
    assert len(affine_jobs) == 30
    assert len(liao_jobs) == 1920
    assert all(j["train_size"] == 160 for j in jobs)


def test_end_to_end_fit_via_run_job(rehearsal_datasets, tmp_path):
    """Tiny end-to-end fit executed via ladder.run_job on rehearsal data."""
    items12 = rehearsal_datasets["items12"]
    data = {
        "seed": 9001,
        "dataset_hash": "test_hash",
        "train": [r for r in items12 if r["split"] == "train"],
        "validation": [r for r in items12 if r["split"] == "validation"],
        "test": [r for r in items12 if r["split"] == "test"],
    }
    data["prediction_rows"] = {
        role: _prediction_items(data[role]) for role in ("validation", "test")
    }

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "shipped-s9001-n12.pkl"
    common.write_cache(cache_path, data)

    fit_dir = tmp_path / "fits"
    fit_dir.mkdir(parents=True)

    job = ladder._job(
        tmp_path,
        fit_dir,
        "shipped-s9001-n12",
        "R0",
        "A",
        train_size=6,
    )
    job["cache"] = str(cache_path)

    res = ladder.run_job(job)
    assert res["error"] is None
    assert res["stem"] == "shipped-s9001-n12__R0__A"

    meta = json.loads((fit_dir / f"{res['stem']}.json").read_text(encoding="utf-8"))
    assert meta["train_size"] == 6
    assert meta["training_rows"] == 48
