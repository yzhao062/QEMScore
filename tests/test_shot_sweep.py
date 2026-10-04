"""Unit tests for shot-count sweep generation and ladder plumbing."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.descriptor_ladder import (
    ARMS,
    DEFAULT_RUNGS,
    RUNGS,
    _selected,
    build_jobs,
    builder_for,
)
from tools.shot_sweep import (
    _process_measurement_group_chunk,
    check_exact,
    generate,
    main as shot_sweep_main,
    validate_exact_toy_circuit,
)


def check_faithful_cli(source_dir: Path, regen_dir: Path, out_path: Path | None = None) -> int:
    args = ["check-faithful", "--source", str(source_dir), "--regen", str(regen_dir)]
    if out_path is not None:
        args.extend(["--report", str(out_path)])
    return shot_sweep_main(args)


def check_exact_cli(exact_dir: Path, finite_dir: Path, out_path: Path | None = None) -> int:
    args = ["check-exact", "--exact", str(exact_dir), "--finite", str(finite_dir)]
    if out_path is not None:
        args.extend(["--report", str(out_path)])
    return shot_sweep_main(args)


PRIMARY_TEST_KEYS = ("shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640")


def _replicate_primary_keys(root: Path) -> None:
    """Toy trees describe seed 101; copy its cache files to the other two primary keys."""
    cache = root / "cache"
    for suffix in (".pkl", ".order.json"):
        source = cache / f"{PRIMARY_TEST_KEYS[0]}{suffix}"
        for key in PRIMARY_TEST_KEYS[1:]:
            target = cache / f"{key}{suffix}"
            if source.exists() and not target.exists():
                target.write_bytes(source.read_bytes())


def check_caches_cli(ref_dir: Path, level_dir: Path, out_path: Path | None = None,
                     replicate: bool = True) -> int:
    if replicate:
        _replicate_primary_keys(ref_dir)
        _replicate_primary_keys(level_dir)
    args = ["check-caches", "--reference", str(ref_dir), "--level", str(level_dir)]
    if out_path is not None:
        args.extend(["--report", str(out_path)])
    return shot_sweep_main(args)


def test_exact_mode_discriminating():
    """Exact density-matrix simulation agrees with 500k shots within 3 sigma on an asymmetric circuit."""
    res = validate_exact_toy_circuit(large_shots=500_000, seed=42)
    assert res["all_agree"], f"Exact mode validation failed: {res}"
    obs_map = {c["observable"]: c for c in res["checks"]}
    # Verify asymmetric values discriminating endianness / qubit permutations
    assert math.isclose(obs_map["Z0"]["exact_value"], 0.958564, abs_tol=1e-3)
    assert math.isclose(obs_map["Z1"]["exact_value"], 0.681144, abs_tol=1e-3)
    assert math.isclose(obs_map["Z2"]["exact_value"], 0.068779, abs_tol=1e-3)
    assert math.isclose(obs_map["Z0Z1"]["exact_value"], 0.652924, abs_tol=1e-3)
    for check in res["checks"]:
        assert check["agrees_within_3sigma"]
        assert check["sigma"] < 3.0


def test_exact_placeholder_fields(tmp_path: Path):
    """Exact level rows carry placeholder shots=1048576, stderr=0.0, and exact counts sidecars."""
    # Build a tiny 1-circuit task with canonical descriptor
    circuit_desc = {
        "circuit_seed": 42,
        "family": "tfi",
        "n_qubits": 4,
        "parameters": {"dt": 0.2, "h": 1.0, "j": 0.5, "steps": 1},
    }
    tasks = [{
        "group_id": "group-0001",
        "cell_id": "cell-1",
        "circuit_id": "circuit-0001",
        "circuit_desc": circuit_desc,
        "tseed": 42,
        "severity": "L1",
        "noise_family": "depolarizing_readout",
        "sampler_seed": 100,
        "items": [{
            "item_id": "item-0001",
            "observable_id": "obs-0",
            "support": [0],
            "stored_two_qubit_gates": 6,
            "stored_transpiled_depth": 15,
        }],
    }]

    out_dir = tmp_path / "exact-test-out"
    item_updates, sidecars = _process_measurement_group_chunk(
        tasks, shots=1048576, exact=True, out_dir_str=str(out_dir)
    )

    upd = item_updates["item-0001"]
    assert upd["shots"] == 1048576
    assert upd["raw_circuit_evals"] == 1048576
    assert upd["noisy_stderr"] == 0.0
    assert math.log2(upd["shots"]) == 20.0
    assert isinstance(upd["noisy_expectation"], float)

    # Check sidecar content
    sidecar_path = out_dir / upd["counts_sidecar"]
    assert sidecar_path.exists()
    sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_data["exact"] is True
    assert "p_ro" in sidecar_data
    assert sidecar_data["support"] == [[0]]
    assert "prob_sha256" in sidecar_data


def test_sweep_mode_off_unchanged(tmp_path: Path):
    """With sweep mode off, default rungs, arms, and job lists are unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--sweep", action="store_true")

    args_off = parser.parse_args([])
    rungs, arms, seeds = _selected(args_off)
    assert rungs == list(DEFAULT_RUNGS)
    assert arms == list(ARMS)
    assert seeds == list(range(1, 21))

    primary_keys = ["shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640"]
    jobs_off = build_jobs(
        tmp_path,
        primary_keys=primary_keys,
        nc_keys=[],
        rungs=rungs,
        arms=arms,
        seeds=seeds,
        strength_indicator=False,
    )
    # Check no job has strength_indicator
    assert all("strength_indicator" not in j for j in jobs_off)


def test_sweep_job_counts(tmp_path: Path):
    """Sweep mode selects rungs R0, N1, N2, R5, arms A, C, F, P, giving 732 jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--sweep", action="store_true")

    args_on = parser.parse_args(["--sweep"])
    rungs, arms, seeds = _selected(args_on)
    assert rungs == ["R0", "N1", "N2", "R5"]
    assert arms == ["A", "C", "F", "P"]
    assert seeds == list(range(1, 21))

    primary_keys = ["shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640"]
    jobs = build_jobs(
        tmp_path,
        primary_keys=primary_keys,
        nc_keys=[],
        rungs=rungs,
        arms=arms,
        seeds=seeds,
        strength_indicator=True,
    )

    # 4 rungs x 183 = 732 fits per shot level
    assert len(jobs) == 732
    # All jobs have strength_indicator set
    assert all(j.get("strength_indicator") is True for j in jobs)

    # Per rung counts: 183 jobs each
    for rung in rungs:
        rung_jobs = [j for j in jobs if j["rung"] == rung]
        assert len(rung_jobs) == 183
        # Arm A: 3 fits (1 per dataset)
        arm_a = [j for j in rung_jobs if j["arm"] == "A"]
        assert len(arm_a) == 3
        # Liao arms (C, F, P): 20 seeds x 3 datasets x 3 arms = 180
        liao_jobs = [j for j in rung_jobs if j["arm"] in ("C", "F", "P")]
        assert len(liao_jobs) == 180


def test_sweep_mode_and_arms_override(tmp_path: Path):
    """Explicit --arms and --rungs override sweep defaults (e.g. arms A C gives 252 jobs)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--sweep", action="store_true")

    # Step 1: --arms A C on sweep rungs
    args_ac = parser.parse_args(["--sweep", "--rungs", "R0", "N1", "N2", "R5", "--arms", "A", "C"])
    rungs, arms, seeds = _selected(args_ac)
    assert rungs == ["R0", "N1", "N2", "R5"]
    assert arms == ["A", "C"]

    primary_keys = ["shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640"]
    jobs_ac = build_jobs(
        tmp_path,
        primary_keys=primary_keys,
        nc_keys=[],
        rungs=rungs,
        arms=arms,
        seeds=seeds,
        strength_indicator=True,
    )
    # 4 rungs * (3 for A + 20 seeds * 3 datasets for C) = 4 * 63 = 252 fits
    assert len(jobs_ac) == 252

    # Simulate Step 1 completion: create fake fits for all 252 jobs
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for j in jobs_ac:
        (fits_dir / f"{j['stem']}.json").write_text("{}", encoding="utf-8")

    # Step 3: --arms A C F P on the same out directory skips completed A and C fits
    args_all = parser.parse_args(["--sweep", "--rungs", "R0", "N1", "N2", "R5", "--arms", "A", "C", "F", "P"])
    rungs_all, arms_all, seeds_all = _selected(args_all)
    jobs_all = build_jobs(
        tmp_path,
        primary_keys=primary_keys,
        nc_keys=[],
        rungs=rungs_all,
        arms=arms_all,
        seeds=seeds_all,
        strength_indicator=True,
    )
    assert len(jobs_all) == 732

    # Filter out existing fits (matching descriptor_ladder.py behavior)
    pending_jobs = [j for j in jobs_all if not (fits_dir / f"{j['stem']}.json").exists()]
    assert len(pending_jobs) == 732 - 252 == 480


def test_check_faithful(tmp_path: Path):
    """check-faithful passes on identical datasets and fails on perturbed expectations or counts."""
    # Setup minimal mock dataset
    source_dir = tmp_path / "mock-source"
    regen_dir = tmp_path / "mock-regen"
    source_counts = source_dir / "sidecars" / "counts"
    regen_counts = regen_dir / "sidecars" / "counts"
    source_counts.mkdir(parents=True, exist_ok=True)
    regen_counts.mkdir(parents=True, exist_ok=True)

    sidecar_content = json.dumps({"0": 1000, "1": 1048})
    sidecar_name = "abc.json"
    (source_counts / sidecar_name).write_text(sidecar_content, encoding="utf-8")
    (regen_counts / sidecar_name).write_text(sidecar_content, encoding="utf-8")

    manifest = {"dataset_id": "test", "items_hash": "hash123"}
    (source_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (regen_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    item_row = {
        "item_id": "it-1",
        "measurement_group": "mg-1",
        "noisy_expectation": 0.046875,
        "noisy_stderr": 0.022,
        "counts_sidecar": f"sidecars/counts/{sidecar_name}",
        "two_qubit_gates": 10,
        "transpiled_depth": 25,
    }
    (source_dir / "items.jsonl").write_text(json.dumps(item_row) + "\n", encoding="utf-8")
    (regen_dir / "items.jsonl").write_text(json.dumps(item_row) + "\n", encoding="utf-8")

    # Pass case
    pass_report = tmp_path / "report-pass.json"
    assert check_faithful_cli(source_dir, regen_dir, pass_report) == 0
    report_data = json.loads(pass_report.read_text(encoding="utf-8"))
    assert report_data["passed"] is True

    # Fail case 1: Perturb noisy_expectation
    bad_item_dir = tmp_path / "mock-bad-item"
    bad_item_counts = bad_item_dir / "sidecars" / "counts"
    bad_item_counts.mkdir(parents=True, exist_ok=True)
    (bad_item_counts / sidecar_name).write_text(sidecar_content, encoding="utf-8")
    (bad_item_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bad_item_row = copy.deepcopy(item_row)
    bad_item_row["noisy_expectation"] = 0.096875
    (bad_item_dir / "items.jsonl").write_text(json.dumps(bad_item_row) + "\n", encoding="utf-8")

    fail_report1 = tmp_path / "report-fail1.json"
    assert check_faithful_cli(source_dir, bad_item_dir, fail_report1) == 1
    fail_data1 = json.loads(fail_report1.read_text(encoding="utf-8"))
    assert fail_data1["differing_measurement_groups_count"] == 1
    assert math.isclose(fail_data1["max_r_diff"], 0.05, abs_tol=1e-6)

    # Fail case 2: Perturb counts sidecar
    bad_counts_dir = tmp_path / "mock-bad-counts"
    bad_counts_counts = bad_counts_dir / "sidecars" / "counts"
    bad_counts_counts.mkdir(parents=True, exist_ok=True)
    (bad_counts_counts / sidecar_name).write_text(json.dumps({"0": 999, "1": 1049}), encoding="utf-8")
    (bad_counts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bad_counts_dir / "items.jsonl").write_text(json.dumps(item_row) + "\n", encoding="utf-8")

    fail_report2 = tmp_path / "report-fail2.json"
    assert check_faithful_cli(source_dir, bad_counts_dir, fail_report2) == 1
    fail_data2 = json.loads(fail_report2.read_text(encoding="utf-8"))
    assert fail_data2["differing_counts_sidecars_count"] == 1


def test_check_exact(tmp_path: Path):
    """check-exact verifies 131,072-shot estimates lie within 6 sigma of exact r*."""
    exact_dir = tmp_path / "exact-level"
    finite_dir = tmp_path / "finite-level"
    exact_dir.mkdir(parents=True, exist_ok=True)
    finite_dir.mkdir(parents=True, exist_ok=True)

    (finite_dir / "sweep_level.json").write_text(json.dumps({"shot_level": 131072}), encoding="utf-8")
    (exact_dir / "sweep_level.json").write_text(json.dumps({"shot_level": "exact"}), encoding="utf-8")

    r_star = 0.500000000000
    # At N=131072, sigma = sqrt((1 - 0.25)/131072) = sqrt(0.75/131072) ~ 0.00239
    # 2 sigma deviation: 0.500 + 2 * 0.00239 = 0.50478
    finite_r_pass = 0.504780000000

    exact_item = {"item_id": "it-1", "noisy_expectation": r_star, "shots": 1048576, "noisy_stderr": 0.0}
    finite_item = {"item_id": "it-1", "noisy_expectation": finite_r_pass, "shots": 131072}

    (exact_dir / "items.jsonl").write_text(json.dumps(exact_item) + "\n", encoding="utf-8")
    (finite_dir / "items.jsonl").write_text(json.dumps(finite_item) + "\n", encoding="utf-8")

    pass_report = tmp_path / "exact-pass.json"
    assert check_exact_cli(exact_dir, finite_dir, pass_report) == 0
    report_data = json.loads(pass_report.read_text(encoding="utf-8"))
    assert report_data["passed"] is True
    assert report_data["failed_count"] == 0

    # Fail case: deviation of 0.05 (~20 sigma)
    finite_item_fail = {"item_id": "it-1", "noisy_expectation": 0.550000000000, "shots": 131072}
    (finite_dir / "items.jsonl").write_text(json.dumps(finite_item_fail) + "\n", encoding="utf-8")

    fail_report = tmp_path / "exact-fail.json"
    assert check_exact_cli(exact_dir, finite_dir, fail_report) == 1
    fail_data = json.loads(fail_report.read_text(encoding="utf-8"))
    assert fail_data["passed"] is False
    assert fail_data["failed_count"] == 1
    assert len(fail_data["worst_items"]) >= 1


def test_check_caches(tmp_path: Path):
    """check-caches verifies row order, allowed differences, and identical descriptors/encoder caches."""
    ref_dir = tmp_path / "ref-ladder"
    level_dir = tmp_path / "level-ladder"
    for d in (ref_dir, level_dir):
        (d / "cache").mkdir(parents=True, exist_ok=True)
        (d / "descriptors").mkdir(parents=True, exist_ok=True)
        (d / "encoder-cache").mkdir(parents=True, exist_ok=True)

    key = "shipped-s101-n640"
    order = {"validation_item_ids": ["val-1", "val-2"], "test_item_ids": ["t-1", "t-2"]}
    (ref_dir / "cache" / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
    (level_dir / "cache" / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")

    desc_bytes = b"fake descriptor cache bytes"
    (ref_dir / "descriptors" / f"{key}.descriptors.pkl").write_bytes(desc_bytes)
    (level_dir / "descriptors" / f"{key}.descriptors.pkl").write_bytes(desc_bytes)

    enc_bytes = b"fake encoder cache bytes"
    (ref_dir / "encoder-cache" / f"{key}.pt").write_bytes(enc_bytes)
    (level_dir / "encoder-cache" / f"{key}.pt").write_bytes(enc_bytes)

    ref_payload = {
        "key": key, "seed": 101, "dataset_hash": "hash-2048",
        "train": [{"item_id": "tr-1", "circuit_id": "c1", "shots": 2048, "noisy_expectation": 0.1,
                   "axis_values": {"shots": 2048, "cell": "c"}}],
        "validation": [{"item_id": "val-1", "circuit_id": "c2", "shots": 2048, "noisy_expectation": 0.2}],
        "test": [{"item_id": "t-1", "circuit_id": "c3", "shots": 2048, "noisy_expectation": 0.3}],
        "prediction_rows": {"test": [{"item_id": "t-1", "shots": 2048, "noisy_expectation": 0.3}]},
    }
    # Level payload with allowed differences (shots and noisy_expectation changed)
    level_payload = {
        "key": key, "seed": 101, "dataset_hash": "hash-8192",
        "train": [{"item_id": "tr-1", "circuit_id": "c1", "shots": 8192, "noisy_expectation": 0.105,
                   "axis_values": {"shots": 8192, "cell": "c"}}],
        "validation": [{"item_id": "val-1", "circuit_id": "c2", "shots": 8192, "noisy_expectation": 0.205}],
        "test": [{"item_id": "t-1", "circuit_id": "c3", "shots": 8192, "noisy_expectation": 0.305}],
        "prediction_rows": {"test": [{"item_id": "t-1", "shots": 8192, "noisy_expectation": 0.305}]},
    }

    with open(ref_dir / "cache" / f"{key}.pkl", "wb") as f:
        pickle.dump(ref_payload, f)
    with open(level_dir / "cache" / f"{key}.pkl", "wb") as f:
        pickle.dump(level_payload, f)

    # Pass case
    pass_report = tmp_path / "cache-pass.json"
    assert check_caches_cli(ref_dir, level_dir, pass_report) == 0
    report_data = json.loads(pass_report.read_text(encoding="utf-8"))
    assert report_data["passed"] is True

    # Fail case 1: Order mismatch
    bad_order_dir = tmp_path / "bad-order-ladder"
    for d in ("cache", "descriptors", "encoder-cache"):
        (bad_order_dir / d).mkdir(parents=True, exist_ok=True)
    bad_order = {"validation_item_ids": ["val-2", "val-1"], "test_item_ids": ["t-1", "t-2"]}
    (bad_order_dir / "cache" / f"{key}.order.json").write_text(json.dumps(bad_order), encoding="utf-8")
    (bad_order_dir / "descriptors" / f"{key}.descriptors.pkl").write_bytes(desc_bytes)
    (bad_order_dir / "encoder-cache" / f"{key}.pt").write_bytes(enc_bytes)
    with open(bad_order_dir / "cache" / f"{key}.pkl", "wb") as f:
        pickle.dump(level_payload, f)

    fail_report1 = tmp_path / "cache-fail1.json"
    assert check_caches_cli(ref_dir, bad_order_dir, fail_report1) == 1

    # Fail case 2: Unallowed field altered in cache (circuit_id altered)
    bad_field_payload = copy.deepcopy(level_payload)
    bad_field_payload["train"][0]["circuit_id"] = "different_circuit"
    bad_field_dir = tmp_path / "bad-field-ladder"
    for d in ("cache", "descriptors", "encoder-cache"):
        (bad_field_dir / d).mkdir(parents=True, exist_ok=True)
    (bad_field_dir / "cache" / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
    (bad_field_dir / "descriptors" / f"{key}.descriptors.pkl").write_bytes(desc_bytes)
    (bad_field_dir / "encoder-cache" / f"{key}.pt").write_bytes(enc_bytes)
    with open(bad_field_dir / "cache" / f"{key}.pkl", "wb") as f:
        pickle.dump(bad_field_payload, f)

    fail_report2 = tmp_path / "cache-fail2.json"
    assert check_caches_cli(ref_dir, bad_field_dir, fail_report2) == 1

    # Fail case 3: Differing descriptor file
    bad_desc_dir = tmp_path / "bad-desc-ladder"
    for d in ("cache", "descriptors", "encoder-cache"):
        (bad_desc_dir / d).mkdir(parents=True, exist_ok=True)
    (bad_desc_dir / "cache" / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
    (bad_desc_dir / "descriptors" / f"{key}.descriptors.pkl").write_bytes(b"modified descriptor bytes")
    (bad_desc_dir / "encoder-cache" / f"{key}.pt").write_bytes(enc_bytes)
    with open(bad_desc_dir / "cache" / f"{key}.pkl", "wb") as f:
        pickle.dump(level_payload, f)

    fail_report3 = tmp_path / "cache-fail3.json"
    assert check_caches_cli(ref_dir, bad_desc_dir, fail_report3) == 1


def test_faithful_regeneration_subset(tmp_path: Path):
    """Regenerating a subset at 2048 shots reproduces original items exactly."""
    data_root = os.environ.get("QEMSCORE_REGEN_DATA")
    if not data_root or not (Path(data_root) / "regen-shipped-s101-n640").is_dir():
        pytest.skip("set QEMSCORE_REGEN_DATA to the directory holding regen-shipped-s101-n640")
    source_dir = Path(data_root) / "regen-shipped-s101-n640"

    # Pick 2 circuits
    with open(source_dir / "items.jsonl") as f:
        items = [json.loads(line) for line in f]

    target_circuits = sorted({it["circuit_id"] for it in items})[:2]
    c_items = [it for it in items if it["circuit_id"] in target_circuits]

    tasks = []
    for cid in target_circuits:
        circ_items = [it for it in c_items if it["circuit_id"] == cid]
        cdesc = json.loads((source_dir / circ_items[0]["circuit_sidecar"]).read_text(encoding="utf-8"))
        by_group = {}
        for it in circ_items:
            by_group.setdefault(it["measurement_group"], []).append(it)
        for gid, g_items in by_group.items():
            first = g_items[0]
            items_info = [
                {
                    "item_id": it["item_id"],
                    "observable_id": it["observable_id"],
                    "support": json.loads((source_dir / it["observable_sidecar"]).read_text(encoding="utf-8"))["support"],
                    "stored_two_qubit_gates": it["two_qubit_gates"],
                    "stored_transpiled_depth": it["transpiled_depth"],
                }
                for it in g_items
            ]
            tasks.append({
                "group_id": gid,
                "cell_id": first["cell_id"],
                "circuit_id": cid,
                "circuit_desc": cdesc,
                "tseed": int(cid.removeprefix("circuit-"), 16),
                "severity": first["severity"],
                "noise_family": first["noise_family"],
                "sampler_seed": first["sampler_seed"],
                "items": items_info,
            })

    out_dir = tmp_path / "test-subset-out"
    item_updates, sidecars = _process_measurement_group_chunk(
        tasks[:4], shots=2048, exact=False, out_dir_str=str(out_dir)
    )

    tested_items = [it for t in tasks[:4] for it in t["items"]]
    assert len(item_updates) == len(tested_items)
    for t_it in tested_items:
        it_id = t_it["item_id"]
        orig_it = next(it for it in c_items if it["item_id"] == it_id)
        upd = item_updates[it_id]
        assert upd["noisy_expectation"] == orig_it["noisy_expectation"]
        assert upd["noisy_stderr"] == orig_it["noisy_stderr"]
        assert upd["counts_hash"] == orig_it["counts_hash"]


def test_check_caches_prediction_rows_and_encoder_metadata(tmp_path: Path):
    """Prediction rows are compared; encoder JSON ignores only data_dir and seconds."""
    key = "shipped-s101-n640"
    order = {"validation_item_ids": ["v"], "test_item_ids": ["t"]}
    def build(root, ncx, data_dir, seconds, arr):
        for d in ("cache", "descriptors", "encoder-cache"):
            (root / d).mkdir(parents=True, exist_ok=True)
        (root / "cache" / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
        (root / "descriptors" / f"{key}__coupling_noise.json").write_text("{}", encoding="utf-8")
        np.savez(root / "encoder-cache" / f"{key}.npz", vectors=np.asarray(arr, dtype=float))
        (root / "encoder-cache" / f"{key}.json").write_text(
            json.dumps({"key": key, "data_dir": data_dir, "seconds": seconds, "n_circuits": 1}),
            encoding="utf-8")
        payload = {"key": key, "seed": 101, "dataset_hash": data_dir, "train": [], "validation": [],
                   "test": [], "prediction_rows": {"test": [{"item_id": "t", "two_qubit_gates": ncx}]}}
        with open(root / "cache" / f"{key}.pkl", "wb") as f:
            pickle.dump(payload, f)
    ref, ok, bad_pred, bad_enc = (tmp_path / n for n in ("ref", "ok", "bad-pred", "bad-enc"))
    build(ref, 6, "/a", 1.0, [1.0, 2.0])
    build(ok, 6, "/b", 2.0, [1.0, 2.0])
    build(bad_pred, 7, "/b", 2.0, [1.0, 2.0])
    build(bad_enc, 6, "/b", 2.0, [1.0, 2.5])
    assert check_caches_cli(ref, ok, tmp_path / "ok.json") == 0
    assert check_caches_cli(ref, bad_pred, tmp_path / "bp.json") == 1
    assert check_caches_cli(ref, bad_enc, tmp_path / "be.json") == 1


def test_check_exact_rejects_wrong_levels(tmp_path: Path):
    """Check 2 accepts only the exact level against the 131,072-shot level."""
    def level(root, shot_level, shots, stderr):
        root.mkdir(parents=True)
        (root / "sweep_level.json").write_text(json.dumps({"shot_level": shot_level}), encoding="utf-8")
        (root / "items.jsonl").write_text(
            json.dumps({"item_id": "i1", "noisy_expectation": 0.5, "shots": shots,
                        "noisy_stderr": stderr}) + "\n", encoding="utf-8")
        return root
    exact = level(tmp_path / "exact", "exact", 1048576, 0.0)
    ok = level(tmp_path / "f131072", 131072, 131072, 0.002)
    wrong = level(tmp_path / "f256", 256, 256, 0.05)
    assert check_exact(exact, ok)["passed"] is True
    with pytest.raises(SystemExit):
        check_exact(exact, wrong)
    with pytest.raises(SystemExit):
        check_exact(ok, ok)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_sweep_prepare_rejects_changed_validation_label(tmp_path: Path):
    """Sweep preparation compares validation and test labels with the archive."""
    from qemscore.validation import item_stream_hash
    from tools.learner_seed_replication import prepare_dataset

    data_root = os.environ.get("QEMSCORE_REGEN_DATA")
    archive = os.environ.get("QEMSCORE_ARCHIVE")
    if not data_root or not archive or not (Path(data_root) / "regen-shipped-s101-n640").is_dir():
        pytest.skip("set QEMSCORE_REGEN_DATA and QEMSCORE_ARCHIVE (campaign-archive-v1) to run")
    source = Path(data_root) / "regen-shipped-s101-n640"

    def copy_with(mutate):
        target = tmp_path / mutate.__name__ / "regen-shipped-s101-n640"
        target.mkdir(parents=True)
        rows = [json.loads(line) for line in (source / "items.jsonl").read_text().splitlines() if line]
        mutate(rows)
        (target / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["items_hash"] = item_stream_hash(rows)
        (target / "manifest.json").write_text(json.dumps(manifest))
        return target

    def unchanged(rows):
        return None

    def changed_validation_label(rows):
        row = next(r for r in rows if r["split"] == "validation")
        row["ideal_expectation"] = float(row["ideal_expectation"]) + 1e-6

    check = prepare_dataset(copy_with(unchanged), Path(archive), _mkdir(tmp_path / "cache-ok"), sweep=True)
    assert check["sweep_validation_label_max_abs_diff"] <= 1e-11
    with pytest.raises(SystemExit, match="validation ideal expectation"):
        prepare_dataset(copy_with(changed_validation_label), Path(archive), _mkdir(tmp_path / "cache-bad"),
                        sweep=True)


def test_check_caches_requires_every_primary_dataset(tmp_path: Path):
    """A reference tree missing one dataset's order file fails, even with its pickle present."""
    key = PRIMARY_TEST_KEYS[0]
    payload = {"key": key, "seed": 101, "dataset_hash": "h", "train": [], "validation": [],
               "test": [{"item_id": "t", "ideal_expectation": 0.1}], "prediction_rows": {}}
    for root in (tmp_path / "ref", tmp_path / "lvl"):
        for d in ("cache", "descriptors"):
            (root / d).mkdir(parents=True)
        (root / "descriptors" / "x.json").write_text("{}", encoding="utf-8")
        for k in PRIMARY_TEST_KEYS:
            (root / "cache" / f"{k}.order.json").write_text("{}", encoding="utf-8")
            with open(root / "cache" / f"{k}.pkl", "wb") as f:
                pickle.dump(dict(payload, key=k), f)
    assert check_caches_cli(tmp_path / "ref", tmp_path / "lvl", replicate=False) == 0
    (tmp_path / "ref" / "cache" / f"{PRIMARY_TEST_KEYS[1]}.order.json").unlink()
    with pytest.raises(SystemExit):
        check_caches_cli(tmp_path / "ref", tmp_path / "lvl", replicate=False)


def test_analysis_a_omits_rungs_without_any_fit():
    """Analysis A drops a rung absent from every row of a family (as analysis B does),
    keeps "incomplete" for a rung fitted in only some rows, and is otherwise unchanged."""
    from tools import descriptor_ladder_analysis as dla

    def row(seed, family, rungs):
        return {"family": family, "dataset_seed": seed,
                "rungs": {r: {"classification": {"label": lab}} for r, lab in rungs.items()}}

    rows_out = {}
    for seed in (101, 211, 307):
        fitted = {"R0": "not_distinguished", "N1": "not_distinguished",
                  "N2": "measurement_adds", "R5": "measurement_adds"}
        if seed == 307:
            fitted.pop("N1")  # an in-scope cell missing from one row
        rows_out[f"s{seed}/tfi"] = row(seed, "tfi", fitted)
        rows_out[f"s{seed}/heisenberg"] = row(seed, "heisenberg", dict(fitted))
    statements = dla._statements(dla.PARTS["A"], rows_out)
    for family in ("tfi", "heisenberg"):
        assert set(statements[family]) == {"R0", "N1", "N2", "R5"}
        assert statements[family]["N1"]["statement"] == "incomplete"
        assert statements[family]["N2"]["statement"] == "measurement_adds"
        assert "N3" not in statements[family] and "R4" not in statements[family]


def test_four_rung_selection_reconciles(tmp_path: Path):
    """The sweep's four rungs (R0, N1, N2, R5) pass analysis A, analysis B, and reconcile."""
    import subprocess
    import sys
    fits_src = os.environ.get("QEMSCORE_STRENGTH_FITS")
    cache = os.environ.get("QEMSCORE_DI_CACHE")
    if not fits_src or not cache or not Path(fits_src).is_dir() or not Path(cache).is_dir():
        pytest.skip("set QEMSCORE_STRENGTH_FITS and QEMSCORE_DI_CACHE (descriptor-information-v1/cache)")
    fits = tmp_path / "fits"
    fits.mkdir()
    for rung in ("R0", "N1", "N2", "R5"):
        for path in Path(fits_src).glob(f"shipped-s*-n640__{rung}__*"):
            (fits / path.name).symlink_to(path)
    repo = Path(__file__).resolve().parents[1]
    run = lambda *args: subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                                       cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)})
    a = run(repo / "tools/descriptor_ladder_analysis.py", "--fits", fits, "--cache", cache,
            "--draws", 200, "--out", tmp_path / "a.json", "--quiet")
    assert a.returncode == 0, a.stderr[-2000:]
    b = run(repo / "tools/descriptor_ladder_analysis_b.py", "--fits", fits, "--cache", cache,
            "--draws", 200, "--out", tmp_path / "b.json")
    assert b.returncode == 0, b.stderr[-2000:]
    r = run(repo / "tools/descriptor_ladder_reconcile.py", tmp_path / "a.json", tmp_path / "b.json",
            "--out", tmp_path / "r.json")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is True


def test_finite_level_generation_validates_as_derived_artifact(tmp_path: Path, monkeypatch):
    """A non-2,048 level keeps the source's cell identifiers and seeds, so it passes the
    derived-artifact validator; a changed count or an immutable field is rejected.

    The sampler is stubbed with fixed fake counts, so no real sweep data is produced."""
    import tools.shot_sweep as sweep

    data_root = os.environ.get("QEMSCORE_REGEN_DATA")
    if not data_root or not (Path(data_root) / "regen-shipped-s101-n640").is_dir():
        pytest.skip("set QEMSCORE_REGEN_DATA to the directory holding regen-shipped-s101-n640")
    source = Path(data_root) / "regen-shipped-s101-n640"
    structure = {}
    for line in (source / "items.jsonl").read_text().splitlines():
        it = json.loads(line)
        structure[it["sampler_seed"]] = {"two_qubit_gates": it["two_qubit_gates"],
                                         "transpiled_depth": it["transpiled_depth"]}

    def fake_sample_counts(circuit, severity, shots, sampler_seed, transpile_seed, *, noise_family):
        n = circuit.num_qubits
        return {"0" * n: shots - shots // 4, "1" * n: shots // 4}, structure[sampler_seed]

    monkeypatch.setattr(sweep, "sample_counts", fake_sample_counts)
    out = tmp_path / "shots-256" / "regen-shipped-s101-n640"
    sweep.generate(source, out, shots=256, workers=1)
    assert json.loads((out / "sweep_level.json").read_text())["shot_level"] == 256
    with pytest.raises(Exception):
        validate_split_artifact_unchanged = __import__(
            "qemscore.validation", fromlist=["validate_split_artifact"]).validate_split_artifact
        validate_split_artifact_unchanged(out)
    assert sweep.validate_sweep_artifact(source, out, shots=256)["passed"] is True
    with pytest.raises(ValueError):
        sweep.validate_sweep_artifact(source, out, shots=1024)

    items_path = out / "items.jsonl"
    original_items = items_path.read_text()
    first = json.loads(original_items.splitlines()[0])
    sidecar = out / first["counts_sidecar"]
    original_sidecar = sidecar.read_bytes()
    desc = json.loads(original_sidecar)
    key = next(iter(desc["counts"]))
    desc["counts"][key] += 1
    sidecar.write_text(json.dumps(desc))
    with pytest.raises(ValueError):
        sweep.validate_sweep_artifact(source, out, shots=256)
    sidecar.write_bytes(original_sidecar)

    lines = original_items.splitlines()
    mutated = json.loads(lines[0])
    mutated["ideal_expectation"] = float(mutated["ideal_expectation"]) + 1e-6
    items_path.write_text("\n".join([json.dumps(mutated)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="ideal_expectation"):
        sweep.validate_sweep_artifact(source, out, shots=256)


def test_exact_level_generation_validates_as_derived_artifact(tmp_path: Path, monkeypatch):
    """The exact level runs through generate and validate_sweep_artifact end to end.

    Circuit building, transpilation, and the density-matrix simulator are stubbed with the
    stored structure and a uniform distribution, so no real exact values are produced."""
    import tools.shot_sweep as sweep

    data_root = os.environ.get("QEMSCORE_REGEN_DATA")
    if not data_root or not (Path(data_root) / "regen-shipped-s101-n640").is_dir():
        pytest.skip("set QEMSCORE_REGEN_DATA to the directory holding regen-shipped-s101-n640")
    source = Path(data_root) / "regen-shipped-s101-n640"
    structure = {}
    for line in (source / "items.jsonl").read_text().splitlines():
        it = json.loads(line)
        desc = json.loads((source / it["circuit_sidecar"]).read_text())
        structure[json.dumps(desc, sort_keys=True)] = (
            int(it["n_qubits"]), int(it["two_qubit_gates"]), int(it["transpiled_depth"]))

    class FakeCircuit:
        def __init__(self, key):
            self.key = key
        def copy(self):
            return self
        def measure_all(self):
            return None

    class FakeTranspiled:
        def __init__(self, key):
            self.num_qubits, self._cx, self._depth = structure[key]
            self.data = []
        def count_ops(self):
            return {"cx": self._cx}
        def depth(self):
            return self._depth

    class FakeSimulator:
        def __init__(self, *args, **kwargs):
            self.n = None
        def run(self, circuit):
            self.n = circuit.num_qubits
            return self
        def result(self):
            return self
        def data(self, index):
            return {"probabilities": np.full(2 ** self.n, 1.0 / 2 ** self.n)}

    monkeypatch.setattr(sweep, "build_circuit_from_canonical_descriptor",
                        lambda desc: FakeCircuit(json.dumps(desc, sort_keys=True)))
    monkeypatch.setattr(sweep, "transpile", lambda circuit, **kwargs: FakeTranspiled(circuit.key))
    monkeypatch.setattr(sweep, "AerSimulator", FakeSimulator)
    out = tmp_path / "shots-exact" / "regen-shipped-s101-n640"
    sweep.generate(source, out, exact=True, workers=1)
    assert json.loads((out / "sweep_level.json").read_text())["shot_level"] == "exact"
    assert sweep.validate_sweep_artifact(source, out, exact=True)["passed"] is True
    lines = (out / "items.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert first["shots"] == 1048576 and first["noisy_stderr"] == 0.0
    first["noisy_stderr"] = 0.001
    (out / "items.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    with pytest.raises(ValueError):
        sweep.validate_sweep_artifact(source, out, exact=True)


def test_check_faithful_counts_groups_with_changed_counts(tmp_path: Path):
    """A group whose counts change while its estimates agree is counted as differing."""
    import hashlib as _hashlib

    def level(root, counts):
        (root / "sidecars" / "counts").mkdir(parents=True)
        payload = (json.dumps({"counts": counts}, sort_keys=True) + "\n").encode()
        digest = _hashlib.sha256(payload).hexdigest()
        rel = f"sidecars/counts/{digest}.json"
        (root / rel).write_bytes(payload)
        rows = [{"item_id": f"i{k}", "measurement_group": "g1", "counts_hash": digest,
                 "counts_sidecar": rel, "noisy_expectation": 0.0} for k in (1, 2)]
        (root / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (root / "manifest.json").write_text("{}")
        return root

    source = level(tmp_path / "source", {"000": 1024, "111": 1024})
    regen = level(tmp_path / "regen", {"001": 1024, "110": 1024})
    report = tmp_path / "faithful.json"
    assert check_faithful_cli(source, regen, report) == 1
    data = json.loads(report.read_text())
    assert data["passed"] is False
    assert data["differing_measurement_groups_count"] == 1
    assert data["max_r_diff"] == 0.0
