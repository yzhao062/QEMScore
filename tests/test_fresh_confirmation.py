"""Tests for tools/fresh_confirmation.py on synthetic caches, fits, and analyses."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import fresh_confirmation as fc  # noqa: E402

SEEDS = [9101, 9102]


def _simple_family_mae(items, pred, artifact_id=None):
    out = {}
    for fam in fc.FAMILIES:
        idx = [i for i, r in enumerate(items) if r["family"] == fam]
        y = np.asarray([items[i]["ideal_expectation"] for i in idx])
        out[fam] = float(np.mean(np.abs(np.asarray(pred)[idx] - y)))
    return out


@pytest.fixture(autouse=True)
def _patch_family_mae(monkeypatch):
    # The real per-family error groups rows by physical circuit, which these rows lack.
    monkeypatch.setattr(fc, "_family_mae", _simple_family_mae)
CELLS = [(s, o) for s in ("L1", "L3") for o in ("z_mid", "zz_mid")]


def _validation_rows(seed: int, level: str, n_per_cell: int = 12) -> list[dict]:
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng([seed, fc.LEVELS.index(level)])
    scale = {"256": 0.08, "1024": 0.04, "2048": 0.03, "8192": 0.015,
             "32768": 0.008, "131072": 0.004, "exact": 0.001}[level]
    rows = []
    for fam in fc.FAMILIES:
        for s, o in CELLS:
            for i in range(n_per_cell):
                y = float(rng.uniform(-1, 1))
                r = 0.8 * y + 0.01 + float(noise_rng.normal(0, scale))
                rows.append({"item_id": f"{fam}-{s}-{o}-{i}", "family": fam, "severity": s,
                             "observable": o, "ideal_expectation": y, "noisy_expectation": r})
    return rows


def _write_trees(root: Path) -> dict[str, Path]:
    levels = {}
    for level in fc.LEVELS:
        tree = root / f"shots-{level}"
        (tree / "cache").mkdir(parents=True)
        for seed in SEEDS:
            cache = {"key": f"shipped-s{seed}-n640", "dataset_hash": f"hash{seed}",
                     "validation": _validation_rows(seed, level),
                     "test": [{"poison": True}]}
            with open(tree / "cache" / f"shipped-s{seed}-n640.pkl", "wb") as handle:
                pickle.dump(cache, handle)
        levels[level] = tree
    return levels


def _write_c_fits(fits: Path, levels: dict[str, Path], c_error: float = 0.02) -> None:
    fits.mkdir(parents=True)
    for seed in SEEDS:
        with open(levels["2048"] / "cache" / f"shipped-s{seed}-n640.pkl", "rb") as handle:
            validation = pickle.load(handle)["validation"]
        y = np.asarray([r["ideal_expectation"] for r in validation])
        for rung in fc.PREDICT_RUNGS:
            for k in range(1, 21):
                rng = np.random.default_rng([seed, k, fc.PREDICT_RUNGS.index(rung)])
                pred = y + rng.choice([-1.0, 1.0], size=y.size) * c_error
                stem = f"shipped-s{seed}-n640__{rung}__k{k:02d}__C"
                np.savez(fits / f"{stem}.npz", validation=pred, test=np.zeros(3))
                meta = {"key": f"shipped-s{seed}-n640", "dataset_seed": seed, "rung": rung,
                        "arm": "C", "learner_seed": k, "dataset_hash": f"hash{seed}",
                        "strength_indicator": True,
                        "validation_family_mae": fc._family_mae(validation, pred,
                                                                artifact_id=f"hash{seed}")}
                (fits / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def predict_inputs(tmp_path):
    levels = _write_trees(tmp_path / "runs")
    _write_c_fits(tmp_path / "fits", levels)
    return levels, tmp_path / "fits"


def test_predict_reads_validation_only_and_matches_formula(predict_inputs, tmp_path):
    levels, fits = predict_inputs
    payload = fc.run_predict(SEEDS, levels, fits)
    assert set(payload["cells"]) == {f"{lv}/shipped-s{s}-n640/{fam}/{rung}" for lv in fc.LEVELS
                                     for s in SEEDS for fam in fc.FAMILIES
                                     for rung in fc.PREDICT_RUNGS}
    # Hand computation of one cell from the rule's formula.
    key = "256/shipped-s9101-n640/tfi/N1"
    c_C = [payload["c_C_validation"][f"shipped-s9101-n640/tfi/N1/{s}/{o}"] for s, o in CELLS]
    c_m = [payload["rule_cells"][f"256/shipped-s9101-n640/tfi/{s}/{o}"]["c_m_cross_fitted"]
           for s, o in CELLS]
    comb = [(a ** -2 + b ** -2) ** -0.5 for a, b in zip(c_C, c_m)]
    expected = 1 - np.mean(comb) / np.mean(c_C)
    assert math.isclose(payload["cells"][key]["D_over_C_star_val"], expected, rel_tol=0, abs_tol=1e-14)
    assert all(math.isclose(v, 0.02, abs_tol=1e-12) for v in c_C)
    # Labels follow the frozen thresholds.
    for cell in payload["cells"].values():
        value, rung = cell["D_over_C_star_val"], cell["rung"]
        if rung in ("R0", "R5"):
            assert cell["predicted_label"] is None
        elif value >= fc.ADDS_THRESHOLD[rung]:
            assert cell["predicted_label"] == fc.ADDS
        elif value <= 0.05:
            assert cell["predicted_label"] == fc.NONE
        else:
            assert cell["predicted_label"] is None
    # A precise measurement gives a larger ceiling than a noisy one.
    assert (payload["cells"]["exact/shipped-s9101-n640/tfi/N1"]["D_over_C_star_val"]
            > payload["cells"]["256/shipped-s9101-n640/tfi/N1"]["D_over_C_star_val"])


def test_predict_drops_test_rows(predict_inputs, monkeypatch):
    levels, fits = predict_inputs
    seen = []
    original = fc.load_validation

    def spy(path):
        out = original(path)
        seen.append(set(out))
        return out

    monkeypatch.setattr(fc, "load_validation", spy)
    fc.run_predict(SEEDS, levels, fits)
    assert seen and all("test" not in keys for keys in seen)


def test_predict_refusals(predict_inputs, tmp_path):
    levels, fits = predict_inputs
    with pytest.raises(SystemExit, match="seven levels"):
        fc.run_predict(SEEDS, {k: v for k, v in levels.items() if k != "exact"}, fits)
    (fits / "shipped-s9101-n640__N2__k07__C.npz").unlink()
    with pytest.raises(SystemExit, match="missing"):
        fc.run_predict(SEEDS, levels, fits)


def test_predict_refuses_wrong_identity(predict_inputs):
    levels, fits = predict_inputs
    path = fits / "shipped-s9102-n640__R0__k03__C.json"
    meta = json.loads(path.read_text())
    meta["learner_seed"] = 4
    path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="learner_seed"):
        fc.run_predict(SEEDS, levels, fits)


def test_predict_refuses_strong_fit(predict_inputs):
    levels, fits = predict_inputs
    path = fits / "shipped-s9102-n640__R0__k03__C.json"
    meta = json.loads(path.read_text())
    meta["strong_learners"] = True
    path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="original-candidate"):
        fc.run_predict(SEEDS, levels, fits)


def test_predict_cli_writes_once(predict_inputs, tmp_path):
    levels, fits = predict_inputs
    out = tmp_path / "pred.json"
    log = tmp_path / "run.log"
    argv = ["predict", "--dataset-seeds", *map(str, SEEDS), "--c-fits", str(fits),
            "--out", str(out), "--run-log", str(log)]
    for level, tree in levels.items():
        argv += ["--level", f"{level}={tree}"]
    fc.main(argv)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert digest in log.read_text()
    with pytest.raises(SystemExit, match="written once"):
        fc.main(argv)


# ---------------------------------------------------------------- score

def _entry(label: str, point: float, lower: float, upper: float) -> dict:
    return {"classification": {"status": "estimated", "label": label},
            "D_over_C": {"point": point, "interval": {"lower": lower, "upper": upper}}}


def _analysis(rungs_by_row: dict, statements: dict | None = None) -> dict:
    return {"parts": {"A": {"rows": {row: {"rungs": rungs} for row, rungs in rungs_by_row.items()},
                            "rung_statements": statements or {}}}}


def _rows():
    return [fc.row_key(s, f) for s in SEEDS for f in fc.FAMILIES]


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _score_inputs(tmp_path, *, r0_label=fc.NONE, n1_point=0.30, strong_upper=0.005,
                  first=("N2", "N3"), stack_lower=0.1):
    """Analyses in which every prediction passes unless a keyword changes it."""
    preds = {"schema": fc.SCHEMA_PREDICT, "dataset_seeds": SEEDS, "cells": {}}
    original = {}
    for level in fc.LEVELS:
        rows = {}
        for row in _rows():
            fam = row.rsplit("/", 1)[1]
            rungs = {"R0": _entry(r0_label, 0.0, -0.1, 0.1),
                     "R5": _entry(fc.ADDS, 0.8, 0.7, 0.9)}
            for rung in ("N1", "N2"):
                precise = level in ("8192", "32768", "131072", "exact")
                star = 0.35 if precise else 0.02
                lab = fc.ADDS if precise else fc.NONE
                point = n1_point if precise else 0.01
                rungs[rung] = _entry(lab, point, point - 0.1, point + 0.1)
                preds["cells"][f"{level}/{row}/{rung}"] = {
                    "D_over_C_star_val": star,
                    "predicted_label": fc.ADDS if star >= fc.ADDS_THRESHOLD[rung] else fc.NONE}
            if level == "2048":
                for rung in ("N3", "N4", "R4") + (("R3-TFI",) if fam == "tfi" else ()):
                    rungs[rung] = _entry(fc.ADDS, 0.5, 0.4, 0.6)
            rows[row] = rungs
        statements = {}
        if level == "2048":
            for fam, first_rung in zip(fc.FAMILIES, first):
                statements[fam] = {r: (fc.ADDS if fc.LADDER.index(r) >= fc.LADDER.index(first_rung)
                                       else fc.NONE) for r in fc.LADDER}
        original[level] = _write(tmp_path / f"orig-{level}.json", _analysis(rows, statements))
    strong_rows = {row: {"R0": _entry(fc.NONE, 0.0, -0.01, strong_upper)} for row in _rows()}
    strong_statements = {fam: {r: (fc.ADDS if fc.LADDER.index(r) >= fc.LADDER.index(f) else fc.NONE)
                               for r in fc.LADDER} for fam, f in zip(fc.FAMILIES, ("N2", "N3"))}
    strong = {"2048": _write(tmp_path / "strong.json", _analysis(strong_rows, strong_statements))}
    exact_sha = hashlib.sha256(original["exact"].read_bytes()).hexdigest()
    stacking = _write(tmp_path / "stack.json", {"sets": {"fresh-exact": {
        "analysis_a_sha256": exact_sha,
        "cells": {f"{row}/R0": {"relative_increment": {"point": 0.3, "interval": {
            "lower": stack_lower, "upper": 0.5}}} for row in _rows()}}}})
    pred_path = _write(tmp_path / "pred.json", preds)
    return pred_path, original, strong, stacking


def test_score_all_pass(tmp_path):
    pred, original, strong, stacking = _score_inputs(tmp_path)
    out = fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")
    assert out["summary"] == {f"P{i}": "pass" for i in range(1, 9)}
    p6 = out["predictions"]["P6"]
    # 6 levels x 4 rows x 2 rungs predicted; the precise levels change the 2,048-shot label.
    assert p6["predicted"] == [48, 48]
    assert p6["label_changing"] == [32, 32]
    assert p6["kept_label_baseline"]["label_changing"] == [0, 32]


def test_score_failures(tmp_path):
    pred, original, strong, stacking = _score_inputs(tmp_path, r0_label=fc.ADDS)
    assert fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")["summary"]["P1"] == "fail"
    pred, original, strong, stacking = _score_inputs(tmp_path, strong_upper=0.02)
    assert fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")["summary"]["P2"] == "fail"
    pred, original, strong, stacking = _score_inputs(tmp_path, first=("N1", "N3"))
    assert fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")["summary"]["P3"] == "fail"
    pred, original, strong, stacking = _score_inputs(tmp_path, n1_point=0.10)
    assert fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")["summary"]["P7"] == "fail"
    pred, original, strong, stacking = _score_inputs(tmp_path, stack_lower=-0.01)
    assert fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")["summary"]["P8"] == "fail"


def test_score_refuses_mismatched_stacking_and_rows(tmp_path):
    pred, original, strong, stacking = _score_inputs(tmp_path)
    payload = json.loads(stacking.read_text())
    payload["sets"]["fresh-exact"]["analysis_a_sha256"] = "0" * 64
    _write(stacking, payload)
    with pytest.raises(SystemExit, match="exact-level analysis A"):
        fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")
    pred, original, strong, stacking = _score_inputs(tmp_path)
    payload = json.loads(pred.read_text())
    payload["dataset_seeds"] = [9101, 9103]
    _write(pred, payload)
    with pytest.raises(SystemExit, match="rows"):
        fc.run_score([9101, 9103], pred, original, strong, stacking, "fresh-exact")


def test_score_not_testable_when_few_label_changes(tmp_path):
    pred, original, strong, stacking = _score_inputs(tmp_path)
    payload = json.loads(pred.read_text())
    for key, cell in payload["cells"].items():
        cell["D_over_C_star_val"], cell["predicted_label"] = 0.02, fc.NONE
    _write(pred, payload)
    out = fc.run_score(SEEDS, pred, original, strong, stacking, "fresh-exact")
    assert out["summary"]["P6"] == "not_testable"
    assert out["summary"]["P7"] == "not_testable"
