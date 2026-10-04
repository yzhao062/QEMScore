"""Tests for the post hoc reporting tools of 2026-10-04.

tools/stacking_increment_all.py, tools/shot_sweep_validation_ceiling.py, and
tools/joint_bootstrap.py. Synthetic checks run everywhere; the checks on the
committed records run when the records are present. Rebuilding the records
needs the release assets, which the README section describes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import joint_bootstrap as jb
from tools import shot_sweep_validation_ceiling as svc
from tools import stacking_increment_all as sia

ART = Path(__file__).resolve().parents[1] / "artifacts" / "descriptor-information"
STACKING = ART / "posthoc-stacking-all.json"
CEILING = ART / "shot-sweep" / "posthoc-validation-ceiling.json"
JOINT = ART / "posthoc-joint-bootstrap.json"


# ---- stacking_increment_all ------------------------------------------------


def test_rungs_with_fits_keeps_rungs_present_for_every_dataset_seed(tmp_path):
    for seed in (101, 211, 307):
        for rung in ("R0", "N1"):
            (tmp_path / f"shipped-s{seed}-n640__{rung}__k01__C.json").write_text("{}")
    (tmp_path / "shipped-s101-n640__N2__k01__C.json").write_text("{}")
    rungs = sia.rungs_with_fits(tmp_path)
    assert rungs == {"tfi": ("R0", "N1"), "heisenberg": ("R0", "N1")}


def test_rungs_with_fits_refuses_an_empty_directory(tmp_path):
    with pytest.raises(SystemExit):
        sia.rungs_with_fits(tmp_path)


def _simple_family_mae(rows, values, *, artifact_id):
    """Mean absolute error per family; stands in for the campaign's cell-macro MAE."""
    out = {}
    for family in sorted({row["family"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["family"] == family]
        out[family] = float(np.mean([abs(values[i] - rows[i]["ideal_expectation"])
                                     for i in idx]))
    return out


@pytest.fixture(autouse=True)
def _simple_mae(monkeypatch):
    monkeypatch.setattr(sia, "family_mae", _simple_family_mae)


def _write_level(root, noisy, fit_hashes=None):
    """A synthetic data, cache, and fit directory for one shot level.

    The items differ between levels only in the noisy estimate, as a real shot
    sweep does. The cache holds the validation and test rows in prediction
    order. Each fit records its slot and the per-family error of its
    predictions, which differ by learner seed. ``fit_hashes`` overrides the
    dataset hash the fits record.
    """
    import pickle

    from qemscore.validation import item_stream_hash, split_dataset_hash

    data_dir, cache_dir, fits_dir = root / "data", root / "cache", root / "fits"
    fits_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    hashes, caches_used = {}, {}
    for seed in (101, 211, 307):
        key = f"shipped-s{seed}-n640"
        items = [{"item_id": f"{seed}-{split}-{i}", "split": split,
                  "family": ("tfi", "heisenberg")[i % 2], "ideal_expectation": 0.1 * i,
                  "noisy_expectation": noisy + i}
                 for split in ("validation", "test") for i in range(4)]
        items_hash = item_stream_hash(items)
        dataset_hash = split_dataset_hash(spec_hash="spec", items_hash=items_hash)
        item_dir = data_dir / f"regen-{key}"
        item_dir.mkdir(parents=True)
        (item_dir / "items.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        (item_dir / "manifest.json").write_text(json.dumps(
            {"items_hash": items_hash, "split_spec_hash": "spec",
             "dataset_hash": dataset_hash}), encoding="utf-8")
        rows = {split: [item for item in items if item["split"] == split]
                for split in ("validation", "test")}
        cache_path = cache_dir / f"{key}.pkl"
        with open(cache_path, "wb") as handle:
            pickle.dump({"key": key, "dataset_hash": dataset_hash, **rows}, handle)
        recorded = (fit_hashes or {}).get(seed, dataset_hash)
        for k in range(1, 21):
            for arm in ("C", "F"):
                shift = 0.001 * k * (2 if arm == "F" else 1)
                arrays = {split: np.asarray([row["ideal_expectation"] + shift * (1 + j)
                                             for j, row in enumerate(rows[split])])
                          for split in rows}
                stem = fits_dir / f"{key}__R0__k{k:02d}__{arm}"
                np.savez(stem.with_suffix(".npz"), **arrays)
                stem.with_suffix(".json").write_text(json.dumps({
                    "key": key, "dataset_seed": seed, "rung": "R0", "arm": arm,
                    "learner_seed": k, "dataset_hash": recorded,
                    **{f"{split}_family_mae": _simple_family_mae(
                        rows[split], arrays[split], artifact_id=recorded)
                       for split in rows}}), encoding="utf-8")
        hashes[seed] = dataset_hash
        caches_used[key] = {"dataset_hash": dataset_hash,
                            "sha256": sia.sha256_file(cache_path)}
    analysis = {"inputs": {"caches_used": caches_used}}
    return data_dir, cache_dir, fits_dir, analysis, hashes


RUNGS_R0 = {"tfi": ("R0",), "heisenberg": ("R0",)}


def _reorder_split(data_dir, seed, split):
    """Reverse one split's rows in place; the item hash, which sorts, is unchanged."""
    path = data_dir / f"regen-shipped-s{seed}-n640" / "items.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [item for item in items if item["split"] == split][::-1]
    kept = iter(rows)
    items = [next(kept) if item["split"] == split else item for item in items]
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")


def test_verify_inputs_accepts_matching_data_cache_fits_and_analysis(tmp_path):
    data_dir, cache_dir, fits_dir, analysis, hashes = _write_level(tmp_path, 0.5)
    verified = sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)
    assert verified == {str(seed): h for seed, h in hashes.items()}


def test_verify_inputs_rejects_data_from_another_shot_level(tmp_path):
    # Exact-level fits, cache, and analysis paired with data whose r comes from
    # another level: C, labels, and row order agree, so only the dataset hash differs.
    _, exact_cache, exact_fits, exact_analysis, _ = _write_level(tmp_path / "exact", 0.5)
    other_data, _, _, _, _ = _write_level(tmp_path / "shots-2048", 0.47)
    with pytest.raises(SystemExit, match="analysis A used"):
        sia.verify_inputs(exact_fits, other_data, exact_cache, exact_analysis, RUNGS_R0)


def test_verify_inputs_rejects_fits_that_read_another_dataset(tmp_path):
    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(
        tmp_path, 0.5, fit_hashes={211: "other"})
    with pytest.raises(SystemExit, match="fit dataset_hash"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


def test_verify_inputs_rejects_items_that_do_not_match_their_manifest(tmp_path):
    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    items = data_dir / "regen-shipped-s307-n640" / "items.jsonl"
    items.write_text(items.read_text(encoding="utf-8").replace("0.5", "0.6", 1),
                     encoding="utf-8")
    with pytest.raises(SystemExit, match="items_hash"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


@pytest.mark.parametrize("split", ["test", "validation"])
def test_verify_inputs_rejects_reordered_rows_that_keep_the_item_hash(tmp_path, split):
    # Reordering rows keeps the sorted item hash and the dataset hash, but the
    # prediction arrays follow the cache's order, so the stack would pair the
    # wrong r and y with each prediction.
    from qemscore.validation import item_stream_hash

    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    _reorder_split(data_dir, 211, split)
    item_dir = data_dir / "regen-shipped-s211-n640"
    manifest = json.loads((item_dir / "manifest.json").read_text(encoding="utf-8"))
    items = [json.loads(line) for line in
             (item_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert item_stream_hash(items) == manifest["items_hash"]
    with pytest.raises(SystemExit, match=f"{split} rows differ"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


def test_verify_inputs_rejects_test_predictions_exchanged_between_learner_seeds(tmp_path):
    # Exchanging two seeds' test arrays keeps every cell mean, so only each
    # fit's own recorded error can catch it.
    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    one, two = (fits_dir / f"shipped-s211-n640__R0__k{k:02d}__C.npz" for k in (1, 2))
    a, b = dict(np.load(one)), dict(np.load(two))
    a["test"], b["test"] = b["test"].copy(), a["test"].copy()
    np.savez(one, **a)
    np.savez(two, **b)
    with pytest.raises(SystemExit, match="test predictions give"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


def test_verify_inputs_rejects_a_fit_record_in_another_slot(tmp_path):
    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    fit = fits_dir / "shipped-s307-n640__R0__k05__F.json"
    meta = json.loads(fit.read_text(encoding="utf-8"))
    meta["learner_seed"] = 6
    fit.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="fit learner_seed"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


def _pins_for(fits_dir, analysis_path, name="synthetic"):
    rungs = sia.rungs_with_fits(fits_dir)
    names = sorted({f"shipped-s{seed}-n640__{rung}__k{k:02d}__{arm}.{ext}"
                    for fam_rungs in rungs.values() for rung in fam_rungs
                    for seed in (101, 211, 307) for k in range(1, 21)
                    for arm in ("C", "F") for ext in ("json", "npz")})
    return {"sets": {name: {
        "asset": "synthetic-asset", "rungs": {fam: list(r) for fam, r in rungs.items()},
        "files_sha256": {n: sia.sha256_file(fits_dir / n) for n in names},
        "fits_digest": sia.fits_digest(fits_dir, rungs),
        "analysis_a_sha256": sia.sha256_file(analysis_path)}}}


def test_verify_pins_accepts_the_pinned_files(tmp_path):
    _, _, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    analysis_path = tmp_path / "analysis-a.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    pins = _pins_for(fits_dir, analysis_path)
    rungs = sia.rungs_with_fits(fits_dir)
    assert sia.verify_pins("synthetic", fits_dir, rungs, analysis_path, pins) == \
        pins["sets"]["synthetic"]["fits_digest"]


def test_verify_pins_rejects_two_predictions_exchanged_within_a_family(tmp_path):
    # Exchanging two predictions that both lie above their targets keeps every
    # recorded error and every cell mean; only the pinned file hash catches it.
    _, _, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    analysis_path = tmp_path / "analysis-a.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    pins = _pins_for(fits_dir, analysis_path)
    npz = fits_dir / "shipped-s211-n640__R0__k01__C.npz"
    arrays = dict(np.load(npz))
    arrays["test"][[0, 2]] = arrays["test"][[2, 0]]
    np.savez(npz, **arrays)
    with pytest.raises(SystemExit, match="SHA-256 differs from the file pinned"):
        sia.verify_pins("synthetic", fits_dir, sia.rungs_with_fits(fits_dir),
                        analysis_path, pins)


def test_verify_pins_rejects_another_analysis_and_an_unpinned_set(tmp_path):
    _, _, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    analysis_path = tmp_path / "analysis-a.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    pins = _pins_for(fits_dir, analysis_path)
    rungs = sia.rungs_with_fits(fits_dir)
    with pytest.raises(SystemExit, match="no pinned fit set"):
        sia.verify_pins("other", fits_dir, rungs, analysis_path, pins)
    analysis_path.write_text(json.dumps({**analysis, "x": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="pinned analysis A"):
        sia.verify_pins("synthetic", fits_dir, rungs, analysis_path, pins)


def test_stacking_pins_read_from_an_archive_equal_the_fit_digest(tmp_path):
    import tarfile

    from tools import stacking_pins as sp

    _, _, fits_dir, _, _ = _write_level(tmp_path, 0.5)
    archive = tmp_path / "asset.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(fits_dir, arcname="asset-v1/fits")
    hashes = sp.stream_hashes(archive, {"asset-v1/fits"})["asset-v1/fits"]
    rungs = sp.rungs_present(set(hashes))
    assert rungs == sia.rungs_with_fits(fits_dir)
    assert sp.digest_of(sp.consumed_names(rungs), hashes) == sia.fits_digest(fits_dir, rungs)


@pytest.mark.skipif(not STACKING.exists(), reason="stacking record not present")
def test_stacking_record_fits_match_the_pins():
    stacking = json.loads(STACKING.read_text(encoding="utf-8"))
    pins = json.loads((ART / "posthoc-stacking-pins.json").read_text(encoding="utf-8"))
    assert stacking["pins_sha256"] == sia.sha256_file(ART / "posthoc-stacking-pins.json")
    for name, fit_set in stacking["sets"].items():
        assert fit_set["fits_match_pins"] is True, name
        assert fit_set["fits_digest"] == pins["sets"][name]["fits_digest"], name


def test_verify_inputs_rejects_a_cache_analysis_a_did_not_read(tmp_path):
    data_dir, cache_dir, fits_dir, analysis, _ = _write_level(tmp_path, 0.5)
    analysis["inputs"]["caches_used"]["shipped-s101-n640"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        sia.verify_inputs(fits_dir, data_dir, cache_dir, analysis, RUNGS_R0)


def _analysis_with_means(cells):
    return {"parts": {"A": {"rows": {
        "shipped-s101-n640/tfi": {"rungs": {
            rung: {"means": {"C": {"point": c}, "F": {"point": f}}}
            for rung, (c, f) in cells.items()}}}}}}


def test_verify_against_analysis_accepts_matching_means():
    analysis = _analysis_with_means({"R0": (0.0024, 0.0027), "N1": (0.008, 0.0079)})
    stacking = {"shipped-s101-n640/tfi/R0": {"mean_C": 0.0024, "F": 0.0027},
                "shipped-s101-n640/tfi/N1": {"mean_C": 0.008 + 1e-15, "F": 0.0079}}
    assert sia.verify_against_analysis(stacking, analysis) <= 1e-12


@pytest.mark.parametrize("field", ["mean_C", "F"])
def test_verify_against_analysis_rejects_predictions_that_do_not_reproduce_it(field):
    # Reordered test predictions, or fits of another pipeline, change the mean
    # test errors that analysis A recorded for the labels the record copies.
    analysis = _analysis_with_means({"R0": (0.0024, 0.0027)})
    stacking = {"shipped-s101-n640/tfi/R0": {"mean_C": 0.0024, "F": 0.0027}}
    stacking["shipped-s101-n640/tfi/R0"][field] += 1e-6
    with pytest.raises(SystemExit, match="differs from analysis A"):
        sia.verify_against_analysis(stacking, analysis)


@pytest.mark.skipif(not STACKING.exists(), reason="stacking record not present")
def test_stacking_record_means_match_each_analysis_a():
    stacking = json.loads(STACKING.read_text(encoding="utf-8"))
    for name, fit_set in stacking["sets"].items():
        assert fit_set["mean_C_and_F_match_analysis_a_max_abs_diff"] <= 1e-12, name


@pytest.mark.skipif(not STACKING.exists(), reason="stacking record not present")
def test_stacking_record_original_set_equals_the_measurement_floor_record():
    stacking = json.loads(STACKING.read_text(encoding="utf-8"))
    floor = json.loads((ART / "posthoc-measurement-floor.json").read_text(encoding="utf-8"))
    original = stacking["sets"]["original"]["cells"]
    assert set(original) == set(floor["stacking"])
    for key, ref in floor["stacking"].items():
        for field in ("mean_C", "F", "recal_C", "stack", "increment_mean"):
            assert original[key][field] == ref[field], (key, field)
        assert original[key]["increment_interval"] == ref["increment_interval"]


@pytest.mark.skipif(not STACKING.exists(), reason="stacking record not present")
def test_stacking_record_relative_increment_is_a_ratio_of_means():
    stacking = json.loads(STACKING.read_text(encoding="utf-8"))
    for name, fit_set in stacking["sets"].items():
        for key, cell in fit_set["cells"].items():
            rel = cell["relative_increment"]
            assert rel["point"] == pytest.approx(cell["increment_mean"] / cell["recal_C"],
                                                 rel=1e-12), (name, key)
            assert rel["interval"]["lower"] <= rel["interval"]["upper"]


# ---- shot_sweep_validation_ceiling -------------------------------------------


def test_ceiling_matches_the_rule_formula_and_ignores_an_infinite_c_m():
    c_C, c_m = [0.01, 0.02], [0.01, float("inf")]
    comb = [(0.01 ** -2 + 0.01 ** -2) ** -0.5, 0.02]
    assert svc.ceiling(c_C, c_m) == pytest.approx(1 - np.mean(comb) / np.mean(c_C))


def test_predicted_label_uses_the_rule_thresholds():
    assert svc.predicted_label(0.30, 0.25) == "measurement_adds"
    assert svc.predicted_label(0.05, 0.25) == "not_distinguished"
    assert svc.predicted_label(0.10, 0.25) is None


def test_swap_ceiling_relabels_without_touching_the_original():
    predictions = {"classification_cells": {
        "a": {"D_over_C_star": 0.0, "adds_threshold": 0.2, "predicted_label": "not_distinguished",
              "reference_label_baseline": "not_distinguished", "label_changing": False},
        "b": {"D_over_C_star": 0.9, "adds_threshold": 0.2, "predicted_label": "measurement_adds",
              "reference_label_baseline": "measurement_adds", "label_changing": False}}}
    alt = svc.swap_ceiling(predictions, {"a": 0.5, "b": 0.1})
    assert alt["classification_cells"]["a"]["predicted_label"] == "measurement_adds"
    assert alt["classification_cells"]["a"]["label_changing"] is True
    assert alt["classification_cells"]["b"]["predicted_label"] is None
    assert alt["classification_cells"]["b"]["label_changing"] is False
    assert predictions["classification_cells"]["a"]["D_over_C_star"] == 0.0


def test_persistence_counts_with_the_rule_matching():
    cells = {
        "x": {"predicted_label": "measurement_adds", "label_changing": True,
              "reference_baseline": "not_distinguished", "observed_label": "measurement_hurts"},
        "y": {"predicted_label": None, "label_changing": False,
              "reference_baseline": "measurement_adds", "observed_label": "measurement_adds"},
        "z": {"predicted_label": "not_distinguished", "label_changing": False,
              "reference_baseline": "not_distinguished", "observed_label": "measurement_adds"},
    }
    out = svc.persistence(cells)
    assert out["all_eligible"] == {"n_cells": 3, "matches": 2}
    assert out["predicted"] == {"n_cells": 2, "matches": 1}
    assert out["label_changing"] == {"n_cells": 1, "matches": 1}


@pytest.mark.skipif(not CEILING.exists(), reason="validation-ceiling record not present")
def test_validation_ceiling_record_reproduces_the_frozen_evaluation():
    record = json.loads(CEILING.read_text(encoding="utf-8"))
    hypotheses = json.loads((ART / "shot-sweep" / "hypotheses.json").read_text(encoding="utf-8"))
    frozen = record["frozen"]["summary"]
    h1, h2 = hypotheses["evaluation"]["h1"], hypotheses["evaluation"]["h2"]
    assert record["frozen_ceiling_reproduced_max_abs_diff"] <= 1e-12
    assert frozen["h1_b_median_ratio"] == h1["part_b"]["median_ratio"]
    assert frozen["h1_c_exceeding"] == h1["part_c"]["exceeding_count"]
    assert frozen["h2_all_predicted"] == [h2["criterion_all_predicted"]["matches"],
                                          h2["criterion_all_predicted"]["n_cells"]]
    assert frozen["h2_label_changing"] == [h2["criterion_label_changing"]["matches"],
                                           h2["criterion_label_changing"]["n_cells"]]


# ---- joint_bootstrap -----------------------------------------------------------


def test_max_rank_band_covers_the_draws_jointly():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(4000, 12)) + rng.normal(size=(4000, 1))
    lower, upper, covered = jb.max_rank_band(draws)
    assert 0.95 <= covered <= 0.96
    pointwise = np.percentile(draws, (2.5, 97.5), axis=0)
    assert np.all(lower <= pointwise[0]) and np.all(upper >= pointwise[1])


def test_max_rank_band_endpoints_are_the_selected_order_statistics():
    # Converting the critical rank to a percentile once moved every endpoint
    # just inside its order statistic and gave 0.9475 coverage here.
    draws = np.random.default_rng(4).normal(size=(10000, 56))
    lower, upper, covered = jb.max_rank_band(draws)
    ordered = np.sort(draws, axis=0)
    critical = int(np.flatnonzero(np.all(ordered == upper, axis=1))[0])
    assert critical == 9995
    assert np.array_equal(lower, ordered[10000 - 1 - critical])
    assert covered >= 0.95
    assert covered == pytest.approx(0.9565, abs=1e-12)


def test_max_rank_band_with_one_column_is_close_to_the_pointwise_interval():
    draws = np.random.default_rng(1).normal(size=(10000, 1))
    lower, upper, covered = jb.max_rank_band(draws)
    pointwise = np.percentile(draws[:, 0], (2.5, 97.5))
    assert lower[0] == pytest.approx(pointwise[0], abs=0.01)
    assert upper[0] == pytest.approx(pointwise[1], abs=0.01)
    assert 0.95 <= covered <= 0.952


def test_label_of_matches_the_rule_labels():
    assert jb.label_of(0.1, 0.2) == "measurement_adds"
    assert jb.label_of(-0.2, -0.1) == "measurement_hurts"
    assert jb.label_of(-0.1, 0.1) == "not_distinguished"


@pytest.mark.skipif(not JOINT.exists(), reason="joint-bootstrap record not present")
def test_joint_record_reproduces_analysis_a_and_covers_jointly():
    record = json.loads(JOINT.read_text(encoding="utf-8"))
    for name, fit_set in record["sets"].items():
        assert fit_set["pointwise_reproduces_analysis_a_max_abs_diff"] <= 1e-12, name
        for group in fit_set["groups"].values():
            assert 0.95 <= group["joint_coverage"] <= 0.96, name
        for key, cell in fit_set["cells"].items():
            band, point = cell["D_simultaneous"], cell["D_pointwise"]
            assert band["lower"] <= point["lower"] + 1e-15, (name, key)
            assert band["upper"] >= point["upper"] - 1e-15, (name, key)
            assert cell["label_simultaneous"] == jb.label_of(band["lower"], band["upper"])
