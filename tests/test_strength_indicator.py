"""Tests for the strength-indicator option in descriptor ladder and QAOA intermediate.

Governing rule: docs/frozen-rules/2026-10-03-strength-indicator.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pytest

from qemscore.datasets.schema import FEATURES, build_features
from qemscore.runner.run import _prediction_items
from tools import descriptor_common as common
from tools import descriptor_ladder as ladder
from tools import qaoa_intermediate as qaoa
from tools import descriptor_ladder_analysis as analysis
from tools.strength_indicator_compare import compare_fits

# The prepared run directories of the README (runs/partA, runs/partB); point
# QEMSCORE_LADDER_RUNS elsewhere to use another copy. Tests that need them skip
# when they are absent.
_RUNS = Path(os.environ.get("QEMSCORE_LADDER_RUNS",
                            Path(__file__).resolve().parents[1] / "runs"))
PART_A_CACHE_DIR = _RUNS / "partA" / "cache"
PART_B_CACHE_DIR = _RUNS / "partB" / "cache"


# --------------------------------------------------------------------------
# 1. Feature builders on real caches: option off vs option on
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_cache_samples():
    """Sample items from the real caches: Part A, NC, and Part B."""
    samples = {}
    pa_path = PART_A_CACHE_DIR / "shipped-s101-n640.pkl"
    if pa_path.exists():
        samples["partA"] = common.load_cache(pa_path)
    nc_path = PART_A_CACHE_DIR / "nc-s101-n640.pkl"
    if nc_path.exists():
        samples["nc"] = common.load_cache(nc_path)
    pb_path = PART_B_CACHE_DIR / "qaoa-s101-n640.pkl"
    if pb_path.exists():
        samples["partB"] = common.load_cache(pb_path)
    return samples


def test_builder_option_off_returns_exact_original_vectors(real_cache_samples):
    """When strength_indicator is False, builder outputs are byte-identical to original."""
    if not {"partA", "partB"} <= set(real_cache_samples):
        pytest.skip("Real caches not available at test path")

    pa_data = real_cache_samples["partA"]
    table = ladder.coupling_noise_table(pa_data)
    items = pa_data["test"][:10]
    out_dir = PART_A_CACHE_DIR.parent
    key = "shipped-s101-n640"

    # Test all Part A rungs with strength_indicator=False
    for rung in ladder.RUNGS:
        builder, names = ladder.builder_for(
            out_dir, key, rung, pa_data, strength_indicator=False
        )
        assert common.STRENGTH_FEATURE_NAME not in names
        if rung == "R0":
            assert names == tuple(FEATURES)
        for item in items:
            vec = builder(item)
            if rung == "R0":
                assert vec == build_features(item)
            assert len(vec) == len(names)

    # Test Part B rungs with strength_indicator=False
    pb_data = real_cache_samples["partB"]
    pb_items = pb_data["test"][:10]
    n_qubits = int(pb_data["n_qubits"])
    for rung in qaoa.RUNGS:
        builder, names = qaoa.builder_for(
            rung, n_qubits, strength_indicator=False
        )
        assert common.STRENGTH_FEATURE_NAME not in names
        for item in pb_items:
            vec = builder(item)
            if rung == "B-partial":
                assert vec == build_features(item)
            assert len(vec) == len(names)


def test_builder_option_on_appends_single_strength_feature(real_cache_samples):
    """When strength_indicator is True, appends 0 for L1, 1 for L3, and feature name."""
    if not {"partA", "partB"} <= set(real_cache_samples):
        pytest.skip("Real caches not available at test path")

    pa_data = real_cache_samples["partA"]
    table = ladder.coupling_noise_table(pa_data)
    items = pa_data["test"][:20]
    out_dir = PART_A_CACHE_DIR.parent
    key = "shipped-s101-n640"

    # Check both severities exist in samples
    severities = {it["severity"] for it in items}
    assert "L1" in severities and "L3" in severities

    for rung in ladder.RUNGS:
        orig_builder, orig_names = ladder.builder_for(
            out_dir, key, rung, pa_data, strength_indicator=False
        )
        str_builder, str_names = ladder.builder_for(
            out_dir, key, rung, pa_data, strength_indicator=True
        )

        assert str_names == (*orig_names, common.STRENGTH_FEATURE_NAME)
        assert len(str_names) == len(orig_names) + 1

        for item in items:
            orig_vec = orig_builder(item)
            str_vec = str_builder(item)

            assert len(str_vec) == len(orig_vec) + 1
            assert str_vec[:-1] == orig_vec
            expected_strength = 0.0 if item["severity"] == "L1" else 1.0
            assert str_vec[-1] == expected_strength

    # Part B rungs
    pb_data = real_cache_samples["partB"]
    pb_items = pb_data["test"][:20]
    n_qubits = int(pb_data["n_qubits"])
    for rung in qaoa.RUNGS:
        orig_b, orig_n = qaoa.builder_for(rung, n_qubits, strength_indicator=False)
        str_b, str_n = qaoa.builder_for(rung, n_qubits, strength_indicator=True)

        assert str_n == (*orig_n, common.STRENGTH_FEATURE_NAME)
        assert len(str_n) == len(orig_n) + 1
        for item in pb_items:
            orig_vec = orig_b(item)
            str_vec = str_b(item)
            assert str_vec[:-1] == orig_vec
            expected_strength = 0.0 if item["severity"] == "L1" else 1.0
            assert str_vec[-1] == expected_strength


def test_severity_indicator_rejects_unknown_severity():
    """severity_indicator raises ValueError on any severity other than L1 and L3."""
    assert common.severity_indicator({"severity": "L1"}) == 0.0
    assert common.severity_indicator({"severity": "L3"}) == 1.0

    with pytest.raises(ValueError, match=r"[Uu]nknown.*severity"):
        common.severity_indicator({"severity": "L2"})

    with pytest.raises(ValueError, match=r"[Uu]nknown.*severity"):
        common.severity_indicator({"severity": "invalid"})

    # Check that builder also raises when item has invalid severity
    builder = ladder.RungBuilder("R0", strength_indicator=True)
    item = {"family": "tfi", "severity": "L2", "noisy_expectation": 0.5, "shots": 2048, "transpiled_depth": 10,
            "two_qubit_gates": 10, "dt": 0.1, "steps": 2, "obs_locality": 1, "n_qubits": 10}
    with pytest.raises(ValueError, match=r"[Uu]nknown.*severity"):
        builder(item)


# --------------------------------------------------------------------------
# 2. Permutation P leaves strength column unchanged
# --------------------------------------------------------------------------


def test_permuted_arm_p_leaves_strength_column_unchanged(real_cache_samples):
    """Arm P shuffles only the noisy expectation column, never the strength column."""
    if "partA" not in real_cache_samples:
        pytest.skip("Real caches not available")

    items = real_cache_samples["partA"]["test"][:30]
    shuffled = common.shuffle_noisy_items(items, seed=42)

    assert len(shuffled) == len(items)
    # Check that severities are completely preserved per item position
    for orig, shuff in zip(items, shuffled):
        assert shuff["severity"] == orig["severity"]
        assert shuff["item_id"] == orig["item_id"]

    # Noisy expectations are shuffled (not all equal)
    orig_noisy = [it["noisy_expectation"] for it in items]
    shuff_noisy = [it["noisy_expectation"] for it in shuffled]
    assert orig_noisy != shuff_noisy
    assert sorted(orig_noisy) == sorted(shuff_noisy)

    # Builder with strength_indicator=True on shuffled items
    builder = ladder.RungBuilder("R0", strength_indicator=True)
    for orig, shuff in zip(items, shuffled):
        vec_orig = builder(orig)
        vec_shuff = builder(shuff)
        # Strength indicator is last element and must match exactly
        assert vec_orig[-1] == vec_shuff[-1]
        assert vec_shuff[-1] == (0.0 if shuff["severity"] == "L1" else 1.0)


# --------------------------------------------------------------------------
# 3. Job lists and fit counts
# --------------------------------------------------------------------------


def test_job_counts_part_a_and_nc():
    """Job counts with strength-indicator on: 1,950 total fits (includes R0 and NC-R0)."""
    primary = ["shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640"]
    nc = ["nc-s101-n640", "nc-s211-n640", "nc-s307-n640"]
    rungs = [*ladder.RUNGS, *ladder.NC_RUNGS]
    arms = list(ladder.ARMS)
    seeds = list(range(1, 21))
    out = Path("/fake/out")

    # Original jobs: 1,956 fits
    orig_jobs = ladder.build_jobs(out, primary, nc, rungs, arms, seeds, strength_indicator=False)
    assert len(orig_jobs) == 1956

    # Strength jobs: 1,950 fits
    str_jobs = ladder.build_jobs(out, primary, nc, rungs, arms, seeds, strength_indicator=True)
    assert len(str_jobs) == 1950

    # Verify that all strength jobs have strength_indicator=True
    assert all(j.get("strength_indicator") is True for j in str_jobs)

    # Breakdown checks:
    # 1. R0 jobs: 3 keys * 1 arm A + 3 keys * 3 arms (F, C, P) * 20 seeds = 183 jobs
    r0_jobs = [j for j in str_jobs if j["rung"] == "R0"]
    assert len(r0_jobs) == 183
    assert len([j for j in r0_jobs if j["arm"] == "A"]) == 3
    for arm in ("F", "C", "P"):
        arm_jobs = [j for j in r0_jobs if j["arm"] == arm]
        assert len(arm_jobs) == 3 * 20
        # Check learner seeds 1..20
        for key in primary:
            k_seeds = {j["learner_seed"] for j in arm_jobs if j["key"] == key}
            assert k_seeds == set(seeds)

    # 2. NC-R0 jobs: 3 keys * 1 arm A + 3 keys * 3 arms * 20 seeds = 183 jobs
    nc_r0_jobs = [j for j in str_jobs if j["rung"] == ladder.NC_REEXPORT_RUNG]
    assert len(nc_r0_jobs) == 183
    assert len([j for j in nc_r0_jobs if j["arm"] == "A"]) == 3
    for arm in ("F", "C", "P"):
        arm_jobs = [j for j in nc_r0_jobs if j["arm"] == arm]
        assert len(arm_jobs) == 3 * 20

    # 3. NC-none jobs: 3 keys * 2 arms (M, C) * 20 seeds = 120 jobs
    nc_none_jobs = [j for j in str_jobs if j["rung"] == ladder.NC_RUNG]
    assert len(nc_none_jobs) == 120

    # Total Part A rungs (9 rungs * 183 = 1,647) + NC-R0 (183) + NC-none (120) = 1,950
    part_a_jobs = [j for j in str_jobs if j["rung"] in ladder.RUNGS]
    assert len(part_a_jobs) == 9 * 183


def test_job_counts_part_b():
    """Job counts for Part B: exactly 552 fits in both original and strength runs."""
    keys = ["qaoa-s101-n640", "qaoa-s211-n640", "qaoa-s307-n640"]
    rungs = list(qaoa.RUNGS)
    arms = list(qaoa.ARMS)
    seeds = list(range(1, 21))
    out = Path("/fake/out")

    orig_jobs = qaoa.build_jobs(out, keys, rungs, arms, seeds, strength_indicator=False)
    assert len(orig_jobs) == 552
    assert not any("strength_indicator" in j for j in orig_jobs)

    str_jobs = qaoa.build_jobs(out, keys, rungs, arms, seeds, strength_indicator=True)
    assert len(str_jobs) == 552
    assert all(j.get("strength_indicator") is True for j in str_jobs)


# --------------------------------------------------------------------------
# 4. Synthetic toy fits write metadata with strength_indicator=True
# --------------------------------------------------------------------------


def _make_toy_dataset():
    """Small synthetic dataset with L1 and L3 items."""
    rng = np.random.default_rng(123)
    roles = ("train", "validation", "test")
    counts = {"train": 20, "validation": 10, "test": 10}
    data = {"seed": 101, "dataset_hash": "toy_hash", "n_qubits": 4}
    for role in roles:
        rows = []
        for i in range(counts[role]):
            circuit = f"circ_{role}_{i}"
            for sev in ("L1", "L3"):
                ideal = float(rng.uniform(-0.8, 0.8))
                scale = 0.8 if sev == "L1" else 0.4
                noisy = scale * ideal + float(rng.normal(0, 0.05))
                rows.append({
                    "item_id": f"{circuit}_{sev}", "circuit_id": circuit,
                    "measurement_group": circuit, "n_qubits": 4, "pauli_label": "ZIII",
                    "split": role, "family": "tfi", "noise_family": "depolarizing",
                    "stratum": "continuous_regression",
                    "severity": sev, "observable": "z_mid", "shots": 2048,
                    "ideal_expectation": ideal, "noisy_expectation": noisy,
                    "dt": 0.1, "steps": 3, "two_qubit_gates": 12,
                    "transpiled_depth": 8, "obs_locality": 1,
                    "j": 0.5, "h": 0.8,
                })
        data[role] = rows
    data["prediction_rows"] = {role: _prediction_items(data[role])
                               for role in ("validation", "test")}
    return data


def test_fits_record_strength_indicator_in_metadata(tmp_path):
    """run_liao_fit and run_affine_fit write strength_indicator=True when flagged."""
    data = _make_toy_dataset()
    builder = ladder.RungBuilder("R0", strength_indicator=True)
    names = builder.names
    assert names[-1] == common.STRENGTH_FEATURE_NAME

    base_job = {
        "part": "A", "key": "toy", "rung": "R0", "fit_dir": str(tmp_path),
        "strength_indicator": True,
    }

    # 1. Liao fit (arm C)
    c_job = {**base_job, "kind": "liao", "arm": "C", "fit_arm": "C",
             "stem": "toy__R0__k01__C", "learner_seed": 1, "shuffle_seed": None}
    info_c = common.run_liao_fit(c_job, data, builder, names)
    assert info_c["error"] is None
    meta_c = json.loads((tmp_path / "toy__R0__k01__C.json").read_text())
    assert meta_c.get("strength_indicator") is True
    assert meta_c["feature_names"][-1] == common.STRENGTH_FEATURE_NAME

    # 2. Affine fit (arm A)
    a_job = {**base_job, "kind": "affine", "arm": "A", "fit_arm": "A",
             "stem": "toy__R0__A", "learner_seed": None, "shuffle_seed": None}
    info_a = common.run_affine_fit(a_job, data, builder, names)
    assert info_a["error"] is None
    meta_a = json.loads((tmp_path / "toy__R0__A.json").read_text())
    assert meta_a.get("strength_indicator") is True
    assert meta_a["feature_names"][-1] == common.STRENGTH_FEATURE_NAME


# --------------------------------------------------------------------------
# 5. Paired comparison logic
# --------------------------------------------------------------------------


def test_paired_comparison_matches_draw_differences(tmp_path):
    """Verify paired comparison computes delta_D and delta_D_over_C with shared draws."""
    orig_dir = tmp_path / "orig_fits"
    str_dir = tmp_path / "str_fits"
    cache_dir = tmp_path / "cache"
    orig_dir.mkdir()
    str_dir.mkdir()
    cache_dir.mkdir()

    data = _make_toy_dataset()
    key = "shipped-s101-n640"
    common.write_cache(cache_dir / f"{key}.pkl", data)

    ideal = np.array([r["ideal_expectation"] for r in data["test"]])
    for seed in range(1, 3):
        # Orig fits
        np.savez_compressed(orig_dir / f"{key}__R0__k{seed:02d}__C.npz",
                            test=ideal + 0.20)
        (orig_dir / f"{key}__R0__k{seed:02d}__C.json").write_text(json.dumps({
            "key": key, "rung": "R0", "arm": "C", "learner_seed": seed, "error": None
        }))
        np.savez_compressed(orig_dir / f"{key}__R0__k{seed:02d}__F.npz",
                            test=ideal + 0.10)
        (orig_dir / f"{key}__R0__k{seed:02d}__F.json").write_text(json.dumps({
            "key": key, "rung": "R0", "arm": "F", "learner_seed": seed, "error": None
        }))

        # Strength fits
        np.savez_compressed(str_dir / f"{key}__R0__k{seed:02d}__C.npz",
                            test=ideal + 0.18)
        (str_dir / f"{key}__R0__k{seed:02d}__C.json").write_text(json.dumps({
            "key": key, "rung": "R0", "arm": "C", "learner_seed": seed, "error": None,
            "strength_indicator": True
        }))
        np.savez_compressed(str_dir / f"{key}__R0__k{seed:02d}__F.npz",
                            test=ideal + 0.05)
        (str_dir / f"{key}__R0__k{seed:02d}__F.json").write_text(json.dumps({
            "key": key, "rung": "R0", "arm": "F", "learner_seed": seed, "error": None,
            "strength_indicator": True
        }))

    # Add arm A to str_dir for R0
    np.savez_compressed(str_dir / f"{key}__R0__A.npz", test=ideal + 0.25)
    (str_dir / f"{key}__R0__A.json").write_text(json.dumps({
        "key": key, "rung": "R0", "arm": "A", "error": None, "strength_indicator": True
    }))

    # Run analysis A with --original-fits
    out_a = analysis.analyze([str_dir], [cache_dir], draws=100, seed=20261002,
                             verbose=False, original_fit_dirs=[orig_dir])
    cell_a = out_a["parts"]["A"]["rows"][f"{key}/tfi"]["rungs"]["R0"]
    assert "comparison_vs_original" in cell_a
    comp = cell_a["comparison_vs_original"]
    assert comp["status"] == "estimated"
    assert comp["n_seeds"] == 2
    # delta D = (0.18 - 0.05) - (0.20 - 0.10) = 0.13 - 0.10 = 0.03
    assert np.isclose(comp["delta_D"]["point"], 0.03, atol=1e-6)

    # Run compare_fits tool and assert exact agreement
    comp_report = compare_fits([str_dir], [orig_dir], [cache_dir], draws=100,
                               seed=20261002, verbose=False)
    cell_comp = comp_report["parts"]["A"]["rows"][f"{key}/tfi"]["rungs"]["R0"]
    assert np.isclose(cell_comp["delta_D"]["point"], comp["delta_D"]["point"], atol=1e-12)
    assert np.isclose(cell_comp["delta_D_over_C"]["point"], comp["delta_D_over_C"]["point"], atol=1e-12)


# --------------------------------------------------------------------------
# 6. Option off reproduces the pre-change builders at every rung (oracle)
# --------------------------------------------------------------------------

ORACLE = Path(__file__).resolve().parent / "oracles" / "strength_indicator_option_off_vectors.json"


def test_option_off_matches_pre_change_oracle_at_every_rung():
    """Every Part A, near-Clifford, and Part B builder, option off, equals the
    vectors the pre-change code (f24a482) produced; option on appends 0 or 1."""
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    needed = [PART_A_CACHE_DIR / f"{key}.pkl" for key, entry in oracle["datasets"].items()
              if entry["part"] in ("A", "NC")]
    needed += [PART_B_CACHE_DIR / f"{key}.pkl" for key, entry in oracle["datasets"].items()
               if entry["part"] == "B"]
    if not all(path.exists() for path in needed):
        pytest.skip("Real caches not available at test path")
    for key, entry in oracle["datasets"].items():
        cache_dir = PART_B_CACHE_DIR if entry["part"] == "B" else PART_A_CACHE_DIR
        data = common.load_cache(cache_dir / f"{key}.pkl")
        by_id = {str(r["item_id"]): r for r in data["test"]}
        items = [by_id[item_id] for item_id in entry["items"]]
        assert {r["severity"] for r in items} == {"L1", "L3"}
        for rung, expected in entry["rungs"].items():
            for strength in (False, True):
                if entry["part"] == "B":
                    builder, names = qaoa.builder_for(rung, int(data["n_qubits"]),
                                                      strength_indicator=strength)
                else:
                    builder, names = ladder.builder_for(cache_dir.parent, key, rung, data,
                                                        strength_indicator=strength)
                tail = [common.STRENGTH_FEATURE_NAME] if strength else []
                assert list(names) == expected["names"] + tail, (key, rung, strength)
                for item, vector in zip(items, expected["vectors"], strict=True):
                    extra = [common.severity_indicator(item)] if strength else []
                    assert [float(v) for v in builder(item)] == vector + extra, (
                        key, rung, strength, item["item_id"])


# --------------------------------------------------------------------------
# 7. The paired comparison against an independent paired bootstrap
# --------------------------------------------------------------------------


def _write_fit(fit_dir, key, rung, arm, seed, values, strength):
    stem = f"{key}__{rung}__k{seed:02d}__{arm}"
    np.savez_compressed(fit_dir / f"{stem}.npz", test=values)
    meta = {"key": key, "rung": rung, "arm": arm, "learner_seed": seed, "error": None}
    if strength:
        meta["strength_indicator"] = True
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta))


def _independent_macro(errors, row, circuit_counts):
    """(draws x seeds) circuit-weighted macro MAE, written without the script's helper."""
    draws, n_seeds = circuit_counts.shape[0], errors.shape[0]
    out = np.zeros((draws, n_seeds))
    for b in range(draws):
        weights = circuit_counts[b][row.item_circuit]
        for s in range(n_seeds):
            cell_means = [np.sum(weights[m] * errors[s, m]) / np.sum(weights[m])
                          for m in row.members]
            out[b, s] = np.mean(cell_means)
    return out


def test_paired_comparison_interval_matches_independent_paired_bootstrap(tmp_path):
    """Errors vary across learner seeds and circuits, so only draws that share
    the learner-seed and circuit resamples across the two fit sets reproduce
    the reported interval endpoints."""
    data = _make_toy_dataset()
    key = "shipped-s101-n640"
    cache_dir, orig_dir, str_dir = (tmp_path / "cache", tmp_path / "orig", tmp_path / "str")
    for d in (cache_dir, orig_dir, str_dir):
        d.mkdir()
    common.write_cache(cache_dir / f"{key}.pkl", data)
    ideal = np.array([r["ideal_expectation"] for r in data["test"]])
    rng = np.random.default_rng(7)
    seeds = [1, 2, 3, 4]
    preds = {}
    for label, fit_dir, strength in (("orig", orig_dir, False), ("str", str_dir, True)):
        for arm, scale in (("C", 0.20), ("F", 0.12)):
            for seed in seeds:
                values = ideal + rng.normal(0.0, scale, ideal.size) * rng.uniform(0.5, 1.5)
                preds[(label, arm, seed)] = values
                _write_fit(fit_dir, key, "R0", arm, seed, values, strength)

    draws, boot_seed = 300, 20261002
    out = analysis.analyze([str_dir], [cache_dir], draws=draws, seed=boot_seed,
                           verbose=False, original_fit_dirs=[orig_dir])
    comp = out["parts"]["A"]["rows"][f"{key}/tfi"]["rungs"]["R0"]["comparison_vs_original"]
    assert comp["status"] == "estimated" and comp["learner_seeds"] == seeds

    row = analysis.Row(data, "tfi", f"{key}/tfi")
    seed_counts, circuit_counts = analysis.bootstrap_draws(len(seeds), row.n_circuits,
                                                           draws, boot_seed)

    def ratio_and_gap(label, counts):
        err = {arm: np.vstack([np.abs(preds[(label, arm, s)][row.index] - row.ideal)
                               for s in seeds]) for arm in ("C", "F")}
        mc = _independent_macro(err["C"], row, counts)
        mf = _independent_macro(err["F"], row, counts)
        gap = np.sum(seed_counts * (mc - mf), axis=1) / len(seeds)
        mean_c = np.sum(seed_counts * mc, axis=1) / len(seeds)
        return gap / mean_c, gap

    ratio_s, gap_s = ratio_and_gap("str", circuit_counts)
    ratio_o, gap_o = ratio_and_gap("orig", circuit_counts)
    for name, vec in (("delta_D", gap_s - gap_o), ("delta_D_over_C", ratio_s - ratio_o)):
        lower, upper = np.percentile(vec, (2.5, 97.5))
        assert comp[name]["interval"]["lower"] == pytest.approx(lower, abs=1e-12)
        assert comp[name]["interval"]["upper"] == pytest.approx(upper, abs=1e-12)

    # Unpaired circuit draws for the original fits give a different interval,
    # so the endpoint check above can tell paired from unpaired resampling.
    _, other_counts = analysis.bootstrap_draws(len(seeds), row.n_circuits, draws, boot_seed + 1)
    ratio_u, _ = ratio_and_gap("orig", other_counts)
    lower_u, upper_u = np.percentile(ratio_s - ratio_u, (2.5, 97.5))
    assert abs(lower_u - comp["delta_D_over_C"]["interval"]["lower"]) > 1e-6
    assert abs(upper_u - comp["delta_D_over_C"]["interval"]["upper"]) > 1e-6

    report = compare_fits([str_dir], [orig_dir], [cache_dir], draws=draws, seed=boot_seed,
                          verbose=False)
    standalone = report["parts"]["A"]["rows"][f"{key}/tfi"]["rungs"]["R0"]
    assert standalone == comp
