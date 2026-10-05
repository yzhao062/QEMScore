"""Tests for reanalysis.mlqem.analyze module.

Verifies:
1. Bootstrap draws determinism and reproducibility (imported from tools.descriptor_ladder_analysis).
2. Classification logic (measurement_adds, measurement_hurts, not_distinguished).
3. OLS seed handling (deterministic C/F/Rcal broadcast, multi-seed P).
4. Every model reads the setting's OLS Rcal fit, and a missing arm stops the analysis.
5. Binding: circuit order, targets, steps, rule and data identities, and test_is_validation.
6. The reproduction record and the CLI.

Runs in QEMScore py312 environment (pure NumPy; no torch or qiskit).
"""

import os
import sys
import json
import hashlib
import subprocess
import numpy as np
import pytest

from tools.descriptor_ladder_analysis import bootstrap_draws
from reanalysis.mlqem import analyze as an
from reanalysis.mlqem.analyze import (
    classify_d,
    load_model_errors,
    analyze_setting_model,
    reproduction_record,
    LABEL_ADDS,
    LABEL_HURTS,
    LABEL_NOT_DISTINGUISHED,
    RULE_BOOTSTRAP_SEED,
)

RULE_SHA = "a" * 64
DATA_SHAS = {"train/step_0.pk": "b" * 64, "val/step_0.pk": "c" * 64}


def write_fit(fits_dir, setting, model, arm, seed, preds, targets, *, steps=None, ids=None,
              rule_sha=RULE_SHA, data_shas=None, test_is_validation=False):
    n = len(targets)
    steps = np.zeros(n, dtype=int) if steps is None else steps
    ids = np.array([f"step_0.pk:{i}" for i in range(n)]) if ids is None else ids
    base = os.path.join(fits_dir, f"{setting}_{model}_{arm}_seed{seed}")
    np.savez_compressed(base + ".npz", predictions=preds, targets=targets, raw_noisy=targets + 0.1,
                        test_circuit_ids=ids, test_steps=steps)
    meta = {"setting": setting, "model": model, "arm": arm, "learner_seed": seed,
            "frozen_rule_sha256": rule_sha,
            "data_files_sha256": DATA_SHAS if data_shas is None else data_shas,
            "test_is_validation": test_is_validation}
    with open(base + ".json", "w") as handle:
        json.dump(meta, handle)


def write_model(fits_dir, setting, model, seeds, rng, *, f_sd=0.02, c_sd=0.08, targets=None, **kw):
    """Writes every file the runner writes for one setting and model (plus the OLS Rcal fit)."""
    n = 20
    if targets is None:
        targets = rng.uniform(-1.0, 1.0, size=(n, 4))
    write_fit(fits_dir, setting, model, "R", 1, targets + 0.1, targets, **kw)
    if not os.path.isfile(os.path.join(fits_dir, f"{setting}_ols_Rcal_seed1.npz")):
        write_fit(fits_dir, setting, "ols", "Rcal", 1, targets + rng.normal(0, 0.05, (n, 4)), targets, **kw)
    fc_seeds = [1] if model == "ols" else seeds
    for s in fc_seeds:
        write_fit(fits_dir, setting, model, "F", s, targets + rng.normal(0, f_sd, (n, 4)), targets, **kw)
        write_fit(fits_dir, setting, model, "C", s, targets + rng.normal(0, c_sd, (n, 4)), targets, **kw)
    for s in seeds:
        write_fit(fits_dir, setting, model, "P", s, targets + rng.normal(0, 0.07, (n, 4)), targets, **kw)
    return targets


def test_bootstrap_draws_reproducibility():
    sc1, cc1 = bootstrap_draws(5, 10, draws=100, seed=RULE_BOOTSTRAP_SEED)
    sc2, cc2 = bootstrap_draws(5, 10, draws=100, seed=RULE_BOOTSTRAP_SEED)
    assert np.array_equal(sc1, sc2) and np.array_equal(cc1, cc2)
    sc3, _ = bootstrap_draws(5, 10, draws=100, seed=12345)
    assert not np.array_equal(sc1, sc3)


def test_classify_labels():
    adds = classify_d({"point": 0.030, "interval": {"lower": 0.015, "upper": 0.045}}, mean_c=0.10)
    hurts = classify_d({"point": -0.030, "interval": {"lower": -0.050, "upper": -0.010}}, mean_c=0.10)
    none = classify_d({"point": 0.005, "interval": {"lower": -0.010, "upper": 0.020}}, mean_c=0.10)
    assert adds["label"] == LABEL_ADDS
    assert hurts["label"] == LABEL_HURTS
    assert none["label"] == LABEL_NOT_DISTINGUISHED
    assert abs(none["largest_reduction_not_excluded"] - 0.20) < 1e-6


def test_ols_seed_handling_and_contrasts(tmp_path):
    seeds = list(range(1, 21))
    write_model(str(tmp_path), "synth", "ols", seeds, np.random.default_rng(42),
                steps=np.repeat(np.arange(4), 5))
    arm_errors, steps, files, notes = load_model_errors(str(tmp_path), "synth", "ols", seeds,
                                                        rule_sha256=RULE_SHA, manifest_check=False)
    for arm in ("C", "F", "Rcal", "R"):
        assert arm_errors[arm].shape == (20, 20)
        assert np.all(arm_errors[arm][0] == arm_errors[arm][19])
    assert not np.all(arm_errors["P"][0] == arm_errors["P"][1])
    res = analyze_setting_model(arm_errors, steps, draws=500, seed=RULE_BOOTSTRAP_SEED, meta_notes=notes)
    for q in ["C", "F", "P", "R", "Rcal", "D", "D_over_C", "F_minus_R", "F_minus_Rcal", "P_minus_F"]:
        assert q in res["quantities"]
    q = res["quantities"]
    assert abs(q["D"]["point"] - (q["C"]["point"] - q["F"]["point"])) < 1e-12
    assert abs(q["F_minus_Rcal"]["point"] - (q["F"]["point"] - q["Rcal"]["point"])) < 1e-12
    assert res["classification"]["label"] == LABEL_ADDS
    assert len(res["per_step"]) == 4 and "F_minus_Rcal" in res["per_step"][0]
    assert len(res["F_per_seed"]) == 20


@pytest.mark.parametrize("model", ["rf", "mlp"])
def test_every_model_reads_the_ols_rcal_fit(tmp_path, model):
    seeds = [1, 2, 3]
    rng = np.random.default_rng(7)
    targets = write_model(str(tmp_path), "synth", "ols", seeds, rng)
    write_model(str(tmp_path), "synth", model, seeds, rng, targets=targets)
    arm_errors, _, files, notes = load_model_errors(str(tmp_path), "synth", model, seeds,
                                                    manifest_check=False)
    assert "Rcal" in arm_errors
    assert any(f.endswith("synth_ols_Rcal_seed1.npz") for f in files)
    assert notes["rcal_source"] == "synth_ols_Rcal_seed1"
    assert arm_errors["F"].shape == (3, 20) and arm_errors["P"].shape == (3, 20)


def test_missing_rcal_stops_the_analysis(tmp_path):
    write_model(str(tmp_path), "synth", "rf", [1, 2], np.random.default_rng(3))
    os.remove(tmp_path / "synth_ols_Rcal_seed1.npz")
    with pytest.raises(FileNotFoundError):
        load_model_errors(str(tmp_path), "synth", "rf", [1, 2], manifest_check=False)


def test_missing_seed_stops_the_analysis(tmp_path):
    write_model(str(tmp_path), "synth", "rf", [1, 2], np.random.default_rng(3))
    with pytest.raises(FileNotFoundError):
        load_model_errors(str(tmp_path), "synth", "rf", [1, 2, 3], manifest_check=False)


def test_reordered_circuits_are_refused(tmp_path):
    rng = np.random.default_rng(5)
    targets = write_model(str(tmp_path), "synth", "rf", [1, 2], rng)
    ids = np.array([f"step_0.pk:{i}" for i in range(20)])
    # C at seed 2 with predictions, targets, and identifiers reversed together.
    write_fit(str(tmp_path), "synth", "rf", "C", 2, targets[::-1] + 0.05, targets[::-1], ids=ids[::-1])
    with pytest.raises(ValueError, match="differ"):
        load_model_errors(str(tmp_path), "synth", "rf", [1, 2], manifest_check=False)


@pytest.mark.parametrize("field", ["rule", "data", "flag", "job"])
def test_identity_mismatches_are_refused(tmp_path, field):
    rng = np.random.default_rng(11)
    targets = write_model(str(tmp_path), "synth", "rf", [1, 2], rng)
    kw = {}
    if field == "rule":
        kw["rule_sha"] = "d" * 64
    elif field == "data":
        kw["data_shas"] = {**DATA_SHAS, "val/step_0.pk": "e" * 64}
    elif field == "flag":
        kw["test_is_validation"] = True
    write_fit(str(tmp_path), "synth", "rf", "P", 2, targets + 0.07, targets, **kw)
    if field == "job":
        with open(tmp_path / "synth_rf_P_seed2.json") as handle:
            meta = json.load(handle)
        meta["arm"] = "F"
        with open(tmp_path / "synth_rf_P_seed2.json", "w") as handle:
            json.dump(meta, handle)
    with pytest.raises(ValueError):
        load_model_errors(str(tmp_path), "synth", "rf", [1, 2], rule_sha256=RULE_SHA, manifest_check=False)


def test_rule_sha_must_match_the_frozen_rule(tmp_path):
    write_model(str(tmp_path), "synth", "rf", [1], np.random.default_rng(2))
    with pytest.raises(ValueError, match="frozen rule"):
        load_model_errors(str(tmp_path), "synth", "rf", [1], rule_sha256="f" * 64, manifest_check=False)


def test_test_is_validation_is_preserved(tmp_path):
    write_model(str(tmp_path), "synth", "mlp", [1, 2], np.random.default_rng(9), test_is_validation=True)
    _, _, _, notes = load_model_errors(str(tmp_path), "synth", "mlp", [1, 2], manifest_check=False)
    assert notes["test_is_validation"] is True
    assert "scheduler read these labels" in notes["test_is_validation_note"]


def test_manifest_check_uses_the_upstream_manifest():
    with open(an.MANIFEST) as handle:
        manifest = json.load(handle)
    prefix = "docs/tutorials/data/ising_init_from_qasm_no_readout/"
    good = {k[len(prefix):]: v for k, v in manifest.items() if k.startswith(prefix)}
    assert len(good) == 45
    an.check_manifest("no_readout", good)
    with pytest.raises(ValueError):
        an.check_manifest("no_readout", {**good, "val/step_3.pk": "0" * 64})
    with pytest.raises(ValueError):
        an.check_manifest("no_readout", {})


def test_manifest_lists_every_split_the_rule_reads():
    with open(an.MANIFEST) as handle:
        manifest = json.load(handle)
    for dir_name, splits in [("ising_init_from_qasm_no_readout", ["train", "val", "val_extra"]),
                             ("ising_init_from_qasm", ["train", "val", "val_Zonly"]),
                             ("ising_init_from_qasm_coherent", ["train", "val"])]:
        for split in splits:
            for step in range(15):
                assert f"docs/tutorials/data/{dir_name}/{split}/step_{step}.pk" in manifest


def test_reproduction_record_bounds_are_inclusive():
    published = an.PUBLISHED_ERRORS[("no_readout", "rf")]
    for scale, passes in [(0.9, True), (1.1, True), (0.89, False), (1.11, False)]:
        # A scale exactly at a bound must not fall outside it through rounding.
        value = published * scale
        if passes:
            value = min(max(value, published * 0.9 + 1e-15), published * 1.1 - 1e-15)
        results = {"no_readout": {"rf": {"F_per_seed": [value] * 20},
                                  "mlp": {"F_per_seed": [0.03, 0.02]}}}
        rec = reproduction_record(results)
        assert rec["no_readout_rf"]["passes"] is passes
        assert rec["no_readout_rf"]["reading"].startswith("reading 1" if passes else "reading 2")
        assert rec["no_readout_mlp"]["criterion"] is None
        assert abs(rec["no_readout_mlp"]["ratio"] - 0.025 / 0.022274) < 1e-12
    assert reproduction_record({})["no_readout_rf"] == {"status": "not_computed"}


def test_cli_execution(tmp_path):
    fits = tmp_path / "fits"
    fits.mkdir()
    rule = tmp_path / "rule.md"
    rule.write_text("frozen rule\n")
    rule_sha = hashlib.sha256(rule.read_bytes()).hexdigest()
    rng = np.random.default_rng(999)
    targets = write_model(str(fits), "cli", "ols", [1, 2], rng, rule_sha=rule_sha)
    write_model(str(fits), "cli", "rf", [1, 2], rng, targets=targets, rule_sha=rule_sha)
    out = tmp_path / "out.json"
    cmd = [sys.executable, "-m", "reanalysis.mlqem.analyze", "--fits", str(fits), "--out", str(out),
           "--settings", "cli", "--models", "ols", "rf", "--seeds", "1", "2", "--draws", "100",
           "--frozen-rule", str(rule), "--no-manifest-check"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["frozen_rule_sha256"] == rule_sha
    assert data["learner_seeds"] == [1, 2]
    assert "F_minus_Rcal" in data["results"]["cli"]["rf"]["quantities"]
    assert data["results"]["cli"]["rf"]["test_is_validation"] is False
    assert data["reproduction_check"]["no_readout_rf"] == {"status": "not_computed"}
    # A missing fit fails the run rather than producing a partial report.
    os.remove(fits / "cli_rf_P_seed2.npz")
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0


@pytest.mark.parametrize("defect", ["one_column", "short_ids", "nan", "inf"])
def test_malformed_predictions_are_refused_on_load(tmp_path, defect):
    rng = np.random.default_rng(13)
    targets = write_model(str(tmp_path), "synth", "rf", [1, 2], rng)
    preds = targets + 0.05
    ids = None
    if defect == "one_column":
        preds = preds[:, :1]
    elif defect == "short_ids":
        ids = np.array([f"step_0.pk:{i}" for i in range(19)])
    elif defect == "nan":
        preds[3, 2] = np.nan
    else:
        preds[0, 0] = np.inf
    write_fit(str(tmp_path), "synth", "rf", "F", 2, preds, targets, ids=ids)
    with pytest.raises(ValueError):
        an.load_model_fits(str(tmp_path), "synth", "rf", [1, 2], manifest_check=False)


def test_partial_or_extended_manifest_mappings_are_refused():
    with open(an.MANIFEST) as handle:
        manifest = json.load(handle)
    prefix = "docs/tutorials/data/ising_init_from_qasm_no_readout/"
    good = {k[len(prefix):]: v for k, v in manifest.items() if k.startswith(prefix)}
    an.check_manifest("ising_init_from_qasm_no_readout", good)  # raw directory names map too
    for key in sorted(good):
        partial = {k: v for k, v in good.items() if k != key}
        with pytest.raises(ValueError, match="missing"):
            an.check_manifest("no_readout", partial)
    with pytest.raises(ValueError, match="unexpected"):
        an.check_manifest("no_readout", {**good, "val/step_15.pk": "0" * 64})


def test_no_cell_is_computed_when_a_later_fit_is_invalid(tmp_path, monkeypatch):
    fits = tmp_path / "fits"
    fits.mkdir()
    rule = tmp_path / "rule.md"
    rule.write_text("frozen rule\n")
    rule_sha = hashlib.sha256(rule.read_bytes()).hexdigest()
    rng = np.random.default_rng(21)
    targets = write_model(str(fits), "cli", "ols", [1, 2], rng, rule_sha=rule_sha)
    write_model(str(fits), "cli", "rf", [1, 2], rng, targets=targets, rule_sha=rule_sha)
    # The last fit the analysis would read carries another rule's SHA-256.
    write_fit(str(fits), "cli", "rf", "P", 2, targets + 0.07, targets, rule_sha="9" * 64)

    def forbidden(*args, **kwargs):
        raise AssertionError("computation started before every fit was validated")

    monkeypatch.setattr(an, "errors_from_fits", forbidden)
    monkeypatch.setattr(an, "analyze_setting_model", forbidden)
    monkeypatch.setattr(sys, "argv", ["analyze", "--fits", str(fits), "--out", str(tmp_path / "o.json"),
                                      "--settings", "cli", "--models", "ols", "rf", "--seeds", "1", "2",
                                      "--draws", "50", "--frozen-rule", str(rule), "--no-manifest-check"])
    with pytest.raises(ValueError, match="frozen_rule_sha256"):
        an.main()
    assert not (tmp_path / "o.json").exists()
