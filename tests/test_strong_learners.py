"""Tests for the strong-learner option (rule 2026-10-03-strong-learners.md).

Tests that need the prepared Part A tree skip unless QEMSCORE_LADDER_RUNS points
at a directory holding ``partA``.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from qemscore.baselines.liao import (
    LiaoValidationScore,
    select_one_standard_error,
)
from tools import descriptor_common as common
from tools import descriptor_ladder as ladder
from tools import qaoa_intermediate as qaoa
from tools import strong_learner_derive as derive_tool
from tools import strong_learner_report as report_tool

REPO = Path(__file__).resolve().parents[1]
ORACLE_PATH = Path(__file__).resolve().parent / "oracles" / "strong_learners_option_off_digests.json"
RULE_PATH = REPO / "docs" / "frozen-rules" / "2026-10-03-strong-learners.md"
GATE_FILE = REPO / "artifacts" / "descriptor-information" / "gate" / "gate_r0.json"
EXCLUDED_JSON = ("timing", "seconds", "pid")


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _ladder_runs() -> Path:
    env = os.environ.get("QEMSCORE_LADDER_RUNS")
    if not env:
        pytest.skip("QEMSCORE_LADDER_RUNS not set")
    path = Path(env)
    if not (path / "partA" / "cache").exists():
        pytest.skip(f"partA not found under {path}")
    return path


def _copy_part_a(ladder_runs: Path, scratch: Path) -> None:
    src = ladder_runs / "partA"
    scratch.mkdir(parents=True)
    for name in ("cache", "descriptors", "encoder-cache"):
        shutil.copytree(src / name, scratch / name)
    shutil.copy(src / "datasets.json", scratch / "datasets.json")


def _oracle() -> dict:
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


def _compare_with_oracle(fit_dir: Path, oracle: dict) -> None:
    """Every NPZ array name and digest, every JSON field except the excluded ones."""
    for stem, spec in oracle["fits"].items():
        with np.load(fit_dir / f"{stem}.npz") as npz:
            assert set(npz.files) == set(spec["npz_digests"]), f"{stem}: NPZ names differ"
            for name, digest in spec["npz_digests"].items():
                assert _sha256_array(npz[name]) == digest, f"{stem}: array {name} differs"
        meta = json.loads((fit_dir / f"{stem}.json").read_text(encoding="utf-8"))
        expected = spec["json_fields"]
        got = {k: v for k, v in meta.items() if k not in EXCLUDED_JSON}
        assert set(got) == set(expected), (
            f"{stem}: JSON keys differ: extra {sorted(set(got) - set(expected))}, "
            f"missing {sorted(set(expected) - set(got))}")
        for key, value in expected.items():
            if key == "versions":
                strip = lambda v: {k: x for k, x in v.items() if k != "qemscore_path"}  # noqa: E731
                assert strip(got[key]) == strip(value), f"{stem}: versions differ"
            else:
                assert got[key] == value, f"{stem}: JSON field {key} differs"


@pytest.fixture(scope="module")
def strong_run(tmp_path_factory):
    """The 15 oracle jobs run with the option on (shared by the option-on tests)."""
    ladder_runs = _ladder_runs()
    if not ORACLE_PATH.exists():
        pytest.skip("oracle file missing")
    oracle = _oracle()
    scratch = tmp_path_factory.mktemp("strong") / "partA"
    _copy_part_a(ladder_runs, scratch)
    jobs = ladder.build_jobs(
        scratch, [oracle["key"]], [], oracle["rungs"], oracle["arms"],
        [oracle["learner_seed"]], strength_indicator=True, strong_learners=True)
    assert len(jobs) == len(oracle["fits"]) == 15
    for job in jobs:
        ladder.run_job(job)
    return scratch, jobs, oracle


# --------------------------------------------------------------------------
# 1. Identity (option off) and option-on agreement
# --------------------------------------------------------------------------

def test_identity_option_off(tmp_path):
    ladder_runs = _ladder_runs()
    oracle = _oracle()
    scratch = tmp_path / "partA"
    _copy_part_a(ladder_runs, scratch)
    jobs = ladder.build_jobs(
        scratch, [oracle["key"]], [], oracle["rungs"], oracle["arms"],
        [oracle["learner_seed"]], strength_indicator=True, strong_learners=False)
    assert len(jobs) == len(oracle["fits"])
    for job in jobs:
        ladder.run_job(job)
    _compare_with_oracle(scratch / "fits", oracle)


def test_option_on_candidate_agreement(strong_run):
    scratch, jobs, oracle = strong_run
    for job in jobs:
        stem = job["stem"]
        spec = oracle["fits"][stem]
        with np.load(scratch / "fits" / f"{stem}.npz") as npz:
            for arr in ("validation__random_forest", "test__random_forest",
                        "validation__mlp", "test__mlp"):
                assert _sha256_array(npz[arr]) == spec["npz_digests"][arr], (stem, arr)
        meta = json.loads((scratch / "fits" / f"{stem}.json").read_text(encoding="utf-8"))
        assert meta["strong_learners"] is True
        want = ["random_forest", "mlp", "hgbr"] + (
            ["poly5_ridge"] if job["rung"] in common.POLY5_RUNGS else [])
        assert meta["candidates"] == want
        assert meta["follow_up_rule"]["path"] == common.STRONG_LEARNERS_RULE
        strong = {s["name"]: s for s in meta["validation_scores"]}
        off = {s["name"]: s for s in spec["json_fields"]["validation_scores"]}
        for cand in ("random_forest", "mlp"):
            assert strong[cand] == off[cand], (stem, cand)
        # The option_off object holds exactly what the option-off code wrote.
        for off_key, meta_key in derive_tool.RESTORED.items():
            assert meta["option_off"][off_key] == spec["json_fields"][meta_key], (stem, off_key)


def test_derived_equals_option_off(strong_run, tmp_path):
    """Item 5: derive from the option-on fits; the result equals the option-off oracle."""
    scratch, jobs, oracle = strong_run
    out = tmp_path / "derived"
    assert derive_tool.derive_directory([scratch / "fits"], out) == len(jobs)
    _compare_with_oracle(out, oracle)


def test_follow_up_rule_sha256_is_the_files(strong_run):
    scratch, jobs, _ = strong_run
    file_sha = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
    for job in jobs:
        meta = json.loads((scratch / "fits" / f"{job['stem']}.json").read_text(encoding="utf-8"))
        assert meta["follow_up_rule"]["sha256"] == file_sha


def test_rule_sha256_computed_from_file():
    assert common.strong_learners_rule_sha256() == hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
    assert not hasattr(common, "STRONG_LEARNERS_RULE_SHA256")


# --------------------------------------------------------------------------
# 2. Cells, coupling table, ridge and grid settings
# --------------------------------------------------------------------------

def test_cell_key_composition():
    assert common._cell_key({"family": "tfi", "severity": "L1", "observable": "z_mid"}, "A") == "tfi/L1/z_mid"
    assert common._cell_key({"severity": "L3", "observable": "zz_mid"}, "B") == "L3/zz_mid"


def _rule_table(rung: str) -> dict[str, list[str]]:
    """The rule's coupling table, written out independently of the code."""
    suffix = {"N1": "0.01", "N2": "0.03", "N3": "0.1", "N4": "0.3"}
    if rung == "R0":
        return {"tfi": ["j", "h"], "heisenberg": ["jx", "jy", "jz"]}
    if rung in suffix:
        z = suffix[rung]
        return {"tfi": [f"j+{z}z", f"h+{z}z"],
                "heisenberg": [f"jx+{z}z", f"jy+{z}z", f"jz+{z}z"]}
    if rung == "R3-TFI":
        return {"tfi": ["j"], "heisenberg": ["jx", "jy", "jz"]}
    if rung == "R3-Heis":
        return {"tfi": ["j", "h"], "heisenberg": ["jx", "jy"]}
    raise AssertionError(rung)


POLY_RUNGS = ["R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis"]


def _assert_table(rung: str, names) -> None:
    for family, expected in _rule_table(rung).items():
        cols = common._poly5_coupling_columns(rung, family, names)
        assert [names[c] for c in cols] == expected, (rung, family)


def test_poly5_rungs_are_the_rule_list():
    assert set(common.POLY5_RUNGS) == set(POLY_RUNGS)
    assert "R4" not in common.POLY5_RUNGS and "R5" not in common.POLY5_RUNGS


@pytest.mark.parametrize("rung", POLY_RUNGS)
def test_coupling_table_against_real_builder_names(rung):
    """The real RungBuilder's names (with the strength indicator), as run_job builds them."""
    z = {} if rung in ladder.NOISE_LEVELS else None
    builder = ladder.RungBuilder(rung, z_by_item=z, strength_indicator=True)
    assert builder.names[-1] == common.STRENGTH_FEATURE_NAME
    _assert_table(rung, builder.names)


def test_coupling_table_against_builder_for_with_real_cache():
    """The same through ``builder_for`` (what ``run_job`` calls) on the real cache."""
    ladder_runs = _ladder_runs()
    data = common.load_cache(ladder_runs / "partA" / "cache" / "shipped-s101-n640.pkl")
    out = ladder_runs / "partA"
    for rung in POLY_RUNGS:
        _, names = ladder.builder_for(out, "shipped-s101-n640", rung, data, strength_indicator=True)
        _assert_table(rung, names)


def test_coupling_table_toy_names():
    names_r3_heis = ["noisy_expectation", "j", "h", "jx", "jy", "noise_strength_L3"]
    assert common._poly5_coupling_columns("R3-Heis", "tfi", names_r3_heis) == [1, 2]
    assert common._poly5_coupling_columns("R3-Heis", "heisenberg", names_r3_heis) == [3, 4]
    names_r3_tfi = ["noisy_expectation", "j", "jx", "jy", "jz"]
    assert common._poly5_coupling_columns("R3-TFI", "tfi", names_r3_tfi) == [1]
    assert common._poly5_coupling_columns("R3-TFI", "heisenberg", names_r3_tfi) == [2, 3, 4]


def test_stored_couplings_never_read_at_n_rungs():
    """The columns come from the builder's names, so a stored (exact) coupling is never used."""
    names = ["noisy_expectation", "j+0.01z", "h+0.01z", "jx+0.01z", "jy+0.01z", "jz+0.01z"]
    cols = common._poly5_coupling_columns("N1", "tfi", names)
    assert [names[c] for c in cols] == ["j+0.01z", "h+0.01z"]
    # A name list carrying the exact couplings is not accepted at an N rung.
    with pytest.raises(ValueError):
        common._poly5_coupling_columns("N1", "tfi", ["noisy_expectation", "j", "h"])


def test_ridge_alphas_and_grid_order():
    assert common.RIDGE_ALPHAS == (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
    assert common.HGBR_GRID == (
        (200, 0.05, 15), (200, 0.05, 31), (200, 0.1, 15), (200, 0.1, 31),
        (500, 0.05, 15), (500, 0.05, 31), (500, 0.1, 15), (500, 0.1, 31))


def test_arm_p_permutation_scope(tmp_path, monkeypatch):
    """P's strong candidates see the training rows with r permuted by the learner seed,
    and the validation and test rows unpermuted; C and F see the training rows as cached."""
    from qemscore.baselines.controls import shuffle_noisy_items
    train = [{"item_id": f"t{i}", "noisy_expectation": float(i), "ideal_expectation": 0.5}
             for i in range(12)]
    val = [{"item_id": f"v{i}", "noisy_expectation": 100.0 + i} for i in range(3)]
    test = [{"item_id": f"e{i}", "noisy_expectation": 200.0 + i} for i in range(3)]
    data = {"train": train, "validation": val, "test": test, "seed": 1, "dataset_hash": "h",
            "prediction_rows": {"validation": val, "test": test}}
    preds = {"random_forest": [0.0] * 3, "mlp": [0.1] * 3, "hgbr": [0.2] * 3}
    _patch_toy(monkeypatch, liao_selected="random_forest", strong_preds=preds,
               maes={"random_forest": 0.1, "mlp": 0.2, "hgbr": 0.3})
    fake = common._fit_strong_learners
    seen = {}

    def capture(job, data_, builder, names, train_, validation_, cand, timing):
        seen[job["fit_arm"]] = (train_, validation_, data_["test"])
        return fake(job, data_, builder, names, train_, validation_, cand, timing)

    monkeypatch.setattr(common, "_fit_strong_learners", capture)
    for arm, k in (("P", 3), ("F", 3), ("C", 3)):
        job = dict(_toy_job(tmp_path), arm=arm, fit_arm=arm, learner_seed=k,
                   shuffle_seed=k if arm == "P" else None, stem=f"toy-{arm}")
        common.run_liao_fit(job, data, lambda it: [0.0], ["noisy_expectation"])
    p_train, p_val, p_test = seen["P"]
    assert p_train == shuffle_noisy_items(train, seed=3)
    assert [r["noisy_expectation"] for r in p_train] != [r["noisy_expectation"] for r in train]
    assert [r["item_id"] for r in p_train] == [r["item_id"] for r in train]
    assert p_val == val and p_test == test
    for arm in ("F", "C"):
        assert seen[arm] == (train, val, test)


# --------------------------------------------------------------------------
# 3. Failure propagation, option_off record, flag recomputation (toy model)
# --------------------------------------------------------------------------

def _score(name, mae, rank, evals=7):
    return LiaoValidationScore(name=name, validation_mae=mae, standard_error=0.0,
                               total_excess_absolute_loss=mae,
                               circuit_evaluations=evals, simplicity_rank=rank)


class _FakeCand:
    n_iter_ = 1
    loss_curve_ = [0.1]

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, rows):
        return self.values.copy()


class _FakeLiao:
    selected = "random_forest"

    def __init__(self, **kwargs):
        self.candidate_models_ = {"random_forest": _FakeCand([0.0, 0.0, 0.0]),
                                  "mlp": _FakeCand([0.1, 0.1, 0.1])}
        self.selected_model_name_ = type(self).selected
        self.one_standard_error_threshold_ = 0.5
        self.eligible_models_ = (type(self).selected,)
        self.validation_scores_ = [_score("random_forest", 0.1, 0), _score("mlp", 0.2, 1)]

    def fit(self, train, validation):
        return self

    def predict(self, rows):
        return self.candidate_models_[self.selected_model_name_].predict(rows)


def _toy_job(tmp_path):
    return {"part": "A", "key": "k", "rung": "R0", "arm": "F", "fit_arm": "F",
            "learner_seed": 1, "shuffle_seed": None, "strong_learners": True,
            "fit_dir": str(tmp_path / "fits"), "stem": "toy"}


def _toy_data():
    rows = [{}, {}, {}]
    return {"train": [], "validation": rows, "test": rows, "seed": 1, "dataset_hash": "h",
            "prediction_rows": {"validation": rows, "test": rows}}


def _patch_toy(monkeypatch, *, liao_selected, strong_preds, maes, raise_in_strong=None):
    _FakeLiao.selected = liao_selected
    monkeypatch.setattr(common, "LiaoMitigator", _FakeLiao)
    monkeypatch.setattr(common, "_finite_mlp", lambda m: True)
    monkeypatch.setattr(common, "_family_mae_or_none", lambda *a, **k: None)
    monkeypatch.setattr(common, "_group_cost", lambda rows: 7)
    ranks = {"random_forest": 0, "mlp": 1, "hgbr": 2}
    monkeypatch.setattr(
        common, "_validation_score",
        lambda name, validation, preds, *, circuit_evaluations, simplicity_rank:
        _score(name, maes[name], ranks[name], circuit_evaluations))

    def fake_strong(job, data, builder, names, train, validation, cand, timing):
        if raise_in_strong is not None:
            raise raise_in_strong
        cand = dict(cand)
        for role in ("validation", "test"):
            cand[f"{role}__hgbr"] = np.asarray(strong_preds["hgbr"], dtype=float)
            cand[f"{role}__random_forest"] = np.asarray(strong_preds["random_forest"], dtype=float)
            cand[f"{role}__mlp"] = np.asarray(strong_preds["mlp"], dtype=float)
        return (cand, [], {"hgbr": [], "poly5_ridge": []}, {"c": [200, 0.05, 15]}, None,
                ("random_forest", "mlp", "hgbr"))

    monkeypatch.setattr(common, "_fit_strong_learners", fake_strong)


def test_strong_failure_propagates(tmp_path, monkeypatch):
    """Item 1: a strong-learner failure is a crash, not a recorded error and a KeyError."""
    _patch_toy(monkeypatch, liao_selected="random_forest", strong_preds={},
               maes={}, raise_in_strong=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        common.run_liao_fit(_toy_job(tmp_path), _toy_data(), lambda it: [0.0], ["x"])
    assert not (tmp_path / "fits" / "toy.json").exists()


def _boom_job(job):
    raise RuntimeError("boom")


def test_drive_records_strong_failure_as_crash(tmp_path):
    job = {"stem": "toy", "fit_dir": str(tmp_path / "fits"), "kind": "liao"}
    info = common.drive([job], _boom_job, tmp_path, 1)
    assert info["crashes"] == ["toy"]


def test_option_off_record_and_new_candidate_flags(tmp_path, monkeypatch):
    """Items 3: option_off keeps the option-off values; flags follow the selected candidate."""
    nan = float("nan")
    _patch_toy(monkeypatch, liao_selected="random_forest",
               strong_preds={"hgbr": [nan, 0.0, 0.0], "random_forest": [0.0, 0.0, 0.0],
                             "mlp": [0.1, 0.1, 0.1]},
               maes={"random_forest": 0.1, "mlp": 0.2, "hgbr": 0.01})
    common.run_liao_fit(_toy_job(tmp_path), _toy_data(), lambda it: [0.0], ["x"])
    meta = json.loads((tmp_path / "fits" / "toy.json").read_text(encoding="utf-8"))
    assert meta["selected_model"] == "hgbr"
    assert meta["flagged_non_converged"] is True
    assert meta["finite_checks"]["selected_predictions"] is False
    off = meta["option_off"]
    assert off["selected_model"] == "random_forest"
    assert off["flagged"] is False and off["flagged_secondary"] is False
    assert off["finite"]["selected_predictions"] is True
    assert [s["name"] for s in off["validation_scores"]] == ["random_forest", "mlp"]
    assert off["selected_equals_candidate_predictions"] is True
    assert meta["selected_equals_candidate_predictions"] is True
    assert [s["name"] for s in meta["validation_scores"]] == ["random_forest", "mlp", "hgbr"]
    assert set(off) == set(derive_tool.RESTORED)


def test_flags_recomputed_when_rf_mlp_selection_switches(tmp_path, monkeypatch):
    """Any change of selected candidate recomputes finite and flags, not only a new one."""
    nan = float("nan")
    _patch_toy(monkeypatch, liao_selected="random_forest",
               strong_preds={"hgbr": [0.0] * 3, "random_forest": [0.0] * 3,
                             "mlp": [nan, 0.1, 0.1]},
               maes={"random_forest": 0.3, "mlp": 0.1, "hgbr": 0.5})
    common.run_liao_fit(_toy_job(tmp_path), _toy_data(), lambda it: [0.0], ["x"])
    meta = json.loads((tmp_path / "fits" / "toy.json").read_text(encoding="utf-8"))
    assert meta["selected_model"] == "mlp"
    assert meta["finite_checks"]["selected_predictions"] is False
    assert meta["flagged_non_converged"] is True and meta["flagged_excluding_spurious_matmul_fpe"] is True
    assert meta["option_off"]["flagged"] is False


def test_flags_unchanged_when_selection_agrees(tmp_path, monkeypatch):
    _patch_toy(monkeypatch, liao_selected="random_forest",
               strong_preds={"hgbr": [0.0] * 3, "random_forest": [0.0] * 3, "mlp": [0.1] * 3},
               maes={"random_forest": 0.01, "mlp": 0.2, "hgbr": 0.5})
    common.run_liao_fit(_toy_job(tmp_path), _toy_data(), lambda it: [0.0], ["x"])
    meta = json.loads((tmp_path / "fits" / "toy.json").read_text(encoding="utf-8"))
    off = meta["option_off"]
    assert meta["selected_model"] == off["selected_model"] == "random_forest"
    assert meta["flagged_non_converged"] == off["flagged"]
    assert meta["finite_checks"] == off["finite"]


# --------------------------------------------------------------------------
# 4. Refusal and cache check through the real entry points
# --------------------------------------------------------------------------

def test_refusal_rule_part_a_and_b(tmp_path):
    (tmp_path / "fits").mkdir()
    (tmp_path / "fits" / "old.json").write_text(
        json.dumps({"schema": "descriptor-information-fit-v1"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="Refusal"):
        ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                     "--out", str(tmp_path), "--strong-learners", "--dry-run",
                     "--rungs", "R0", "--arms", "A", "--seeds", "1"])
    with pytest.raises(SystemExit, match="Refusal"):
        qaoa.main(["run", "--data-root", str(tmp_path / "root"), "--out", str(tmp_path),
                   "--gate-file", "g.json", "--strong-learners", "--dry-run"])
    # Without the option the same directory is accepted.
    assert ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                        "--out", str(tmp_path), "--dry-run", "--rungs", "R0",
                        "--arms", "A", "--seeds", "1"]) == 0


def _record(tmp_path, digests: dict) -> Path:
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"inputs": {"caches_used": {
        k: {"path": "p", "sha256": v} for k, v in digests.items()}}}), encoding="utf-8")
    return path


def test_cache_check_toy_tree(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "shipped-s101-n640.pkl").write_bytes(b"alpha")
    (cache / "shipped-s211-n640.pkl").write_bytes(b"beta")
    good = {"shipped-s101-n640": hashlib.sha256(b"alpha").hexdigest(),
            "shipped-s211-n640": hashlib.sha256(b"beta").hexdigest()}
    digests = common.check_caches_recorded(
        cache, list(good), record_path=_record(tmp_path, good))
    assert {k: v["sha256"] for k, v in digests.items()} == good
    bad = dict(good, **{"shipped-s211-n640": "0" * 64})
    with pytest.raises(SystemExit, match="shipped-s211-n640"):
        common.check_caches_recorded(cache, list(good), record_path=_record(tmp_path, bad))
    with pytest.raises(SystemExit, match="shipped-s307-n640"):
        common.check_caches_recorded(cache, ["shipped-s307-n640"],
                                     record_path=_record(tmp_path, good))


def test_real_record_lists_the_six_primary_caches():
    record = json.loads((REPO / common.STRENGTH_RECORD).read_text(encoding="utf-8"))
    used = record["inputs"]["caches_used"]
    for key in ("shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640",
                "qaoa-s101-n640", "qaoa-s211-n640", "qaoa-s307-n640"):
        assert len(used[key]["sha256"]) == 64


def test_part_a_run_exits_on_cache_mismatch(tmp_path):
    """A mismatching cache stops ``run --strong-learners`` before any fit, nonzero."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "shipped-s101-n640.pkl").write_bytes(b"not the recorded cache")
    (tmp_path / "datasets.json").write_text(json.dumps({"shipped-s101-n640": {}}), encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        ladder.main(["run", "--datasets", "x/regen-shipped-s101-n640", "--archive", "x",
                     "--out", str(tmp_path), "--strong-learners", "--strength-indicator",
                     "--rungs", "R0", "--arms", "A", "--seeds", "1",
                     "--gate-file", str(GATE_FILE)])
    assert info.value.code not in (0, None)
    assert "shipped-s101-n640" in str(info.value.code)
    assert not (tmp_path / "fits").exists()
    assert not (tmp_path / "run_config.json").exists()


# --------------------------------------------------------------------------
# 5. Part B hgbr only; derive; report
# --------------------------------------------------------------------------

def test_part_b_hgbr_only():
    for rung in ("B-partial", "B-complete", "B-none"):
        assert rung not in common.POLY5_RUNGS


def _toy_strong_fit(*, arm="C"):
    scores = [_score("random_forest", 0.05, 0), _score("mlp", 0.04, 1)]
    chosen, threshold, eligible = select_one_standard_error(scores)
    rows = lambda ss: [{"name": s.name, "validation_mae": s.validation_mae,  # noqa: E731
                        "standard_error": s.standard_error,
                        "total_excess_absolute_loss": s.total_excess_absolute_loss,
                        "circuit_evaluations": s.circuit_evaluations,
                        "simplicity_rank": s.simplicity_rank} for s in ss]
    off = {
        "selected_model": chosen, "eligible_models": list(eligible),
        "one_standard_error_threshold": float(threshold),
        "validation_scores": rows(scores), "flagged": False, "flagged_secondary": False,
        "finite": {"mlp_parameters_and_loss_curve": True, "selected_predictions": True},
        "n_runtime_warnings": 0, "n_runtime_warnings_not_matmul_fpe": 0,
        "warnings_aggregated": [{"phase": "training"}],
        "test_family_mae": {"tfi": 0.04}, "validation_family_mae": {"tfi": 0.04},
        "selected_equals_candidate_predictions": True,
    }
    meta = {
        "schema": "descriptor-information-fit-v1", "key": "k", "rung": "R0", "arm": arm,
        "fit_arm": arm, "learner_seed": 1, "strong_learners": True,
        "candidates": ["random_forest", "mlp", "hgbr"],
        "hgbr_grid_point": {"c": [200, 0.05, 15]}, "poly5_ridge_alpha": None,
        "follow_up_rule": {"path": "r", "sha256": "s"}, "option_off": off,
        "selected_model": "hgbr", "eligible_models": ["hgbr"],
        "one_standard_error_threshold": 0.0,
        "validation_scores": rows(scores + [_score("hgbr", 0.01, 2)]),
        "flagged_non_converged": True, "flagged_excluding_spurious_matmul_fpe": True,
        "finite_checks": {"mlp_parameters_and_loss_curve": True, "selected_predictions": False},
        "n_runtime_warnings": 3, "n_runtime_warnings_not_matmul_fpe": 3,
        "warnings_aggregated": [{"phase": "hgbr_fit"}],
        "test_family_mae": {"tfi": 0.01}, "validation_family_mae": {"tfi": 0.01},
        "selected_equals_candidate_predictions": True,
        "candidate_test_family_mae": {"random_forest": 1, "mlp": 2, "hgbr": 3},
        "candidate_validation_family_mae": {"random_forest": 1, "mlp": 2, "hgbr": 3},
        "candidate_prediction_warnings": [{"phase": "candidate_prediction"},
                                          {"phase": "candidate_hgbr"},
                                          {"phase": "candidate_poly5_ridge"}],
        "timing": {"fit_seconds": 1.0, "candidate_hgbr_seconds": 2.0,
                   "candidate_poly5_ridge_seconds": 3.0},
    }
    arrays = {f"{role}__{name}": np.full(2, val)
              for name, val in (("random_forest", 0.5), ("mlp", 0.4), ("hgbr", 0.1))
              for role in ("validation", "test")}
    arrays["validation"] = arrays["validation__hgbr"].copy()
    arrays["test"] = arrays["test__hgbr"].copy()
    return meta, arrays, chosen


def _write_toy(fit_dir: Path, stem: str, meta: dict, arrays: dict | None) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    if arrays:
        np.savez_compressed(fit_dir / f"{stem}.npz", **arrays)


def test_strong_learner_derive(tmp_path, monkeypatch):
    meta, arrays, chosen = _toy_strong_fit()
    src, out = tmp_path / "strong", tmp_path / "derived"
    _write_toy(src, "toy__C", meta, arrays)
    _write_toy(src, "toy__A", {"arm": "A", "strong_learners": True, "follow_up_rule": {"path": "r"},
                               "n_features": 3}, {"validation": np.zeros(2), "test": np.ones(2)})
    _write_toy(src, "toy__GBT", {"arm": "GBT", "strong_learners": True, "follow_up_rule": {"path": "r"},
                                 "seconds": 1.0}, {"test": np.ones(2)})
    # A Liao fit that failed before predicting: no arrays, JSON only, as option off writes it.
    failed, _, _ = _toy_strong_fit()
    failed["option_off"] = dict(failed["option_off"], selected_model=None, eligible_models=None,
                                one_standard_error_threshold=None, validation_scores=None)
    _write_toy(src, "toy__F", dict(failed, arm="F", fit_arm="F", error="fit: boom"), None)

    used = []
    real_write = common._write_fit
    monkeypatch.setattr(common, "_write_fit", lambda *a: (used.append(a[0]["stem"]), real_write(*a))[1])
    assert derive_tool.derive_directory([src], out) == 4
    assert sorted(used) == ["toy__A", "toy__C", "toy__F", "toy__GBT"]

    derived = json.loads((out / "toy__C.json").read_text(encoding="utf-8"))
    for key in derive_tool.STRONG_FIELDS:
        assert key not in derived
    off = meta["option_off"]
    for off_key, meta_key in derive_tool.RESTORED.items():
        assert derived[meta_key] == off[off_key], meta_key
    assert derived["selected_model"] == chosen
    assert derived["candidate_test_family_mae"] == {"random_forest": 1, "mlp": 2}
    assert derived["candidate_validation_family_mae"] == {"random_forest": 1, "mlp": 2}
    assert derived["candidate_prediction_warnings"] == [{"phase": "candidate_prediction"}]
    assert derived["timing"] == {"fit_seconds": 1.0}
    with np.load(out / "toy__C.npz") as npz:
        assert list(npz.files) == ["validation", "test", "validation__random_forest",
                                   "test__random_forest", "validation__mlp", "test__mlp"]
        np.testing.assert_array_equal(npz["validation"], arrays[f"validation__{chosen}"])
        np.testing.assert_array_equal(npz["test"], arrays[f"test__{chosen}"])

    a = json.loads((out / "toy__A.json").read_text(encoding="utf-8"))
    assert a == {"arm": "A", "n_features": 3}
    g = json.loads((out / "toy__GBT.json").read_text(encoding="utf-8"))
    assert g == {"arm": "GBT", "seconds": 1.0}
    f = json.loads((out / "toy__F.json").read_text(encoding="utf-8"))
    assert f["selected_model"] is None and f["error"] == "fit: boom"
    assert not (out / "toy__F.npz").exists()


def test_derive_refuses_bad_input(tmp_path):
    meta, arrays, _ = _toy_strong_fit()
    src = tmp_path / "s"
    _write_toy(src, "a", dict(meta, strong_learners=False), arrays)
    with pytest.raises(SystemExit, match="not a strong-learner fit"):
        derive_tool.derive_directory([src], tmp_path / "o1")
    src2 = tmp_path / "s2"
    broken = {k: v for k, v in meta.items() if k != "option_off"}
    _write_toy(src2, "a", broken, arrays)
    with pytest.raises(ValueError, match="option_off"):
        derive_tool.derive_directory([src2], tmp_path / "o2")
    src3 = tmp_path / "s3"
    wrong = json.loads(json.dumps(meta))
    wrong["option_off"]["selected_model"] = "mlp" if meta["option_off"]["selected_model"] == "random_forest" else "random_forest"
    _write_toy(src3, "a", wrong, arrays)
    with pytest.raises(ValueError, match="disagrees"):
        derive_tool.derive_directory([src3], tmp_path / "o3")


def test_strong_learner_report_empty():
    report = report_tool.analyze_strong_report([], [], draws=100)
    assert report["schema"] == "strong-learner-report-v1"
    assert report["draws"] == 100


def test_per_family_select_uses_recorded_circuit_evaluations(monkeypatch):
    seen = []

    def spy(name, rows, preds, *, circuit_evaluations, simplicity_rank):
        seen.append(circuit_evaluations)
        return _score(name, {"random_forest": 0.2, "mlp": 0.1}[name], simplicity_rank,
                      circuit_evaluations)

    monkeypatch.setattr(report_tool, "_validation_score", spy)

    class Store:
        def array(self, record, name):
            return np.zeros(4)

    record = {"meta": {"candidates": ["random_forest", "mlp"], "key": "k",
                       "validation_scores": [
                           {"name": "random_forest", "circuit_evaluations": 123},
                           {"name": "mlp", "circuit_evaluations": 123}]}}
    data = {"validation": [{"family": "tfi"}, {"family": "heisenberg"},
                           {"family": "tfi"}, {"family": "heisenberg"}]}
    assert report_tool.per_family_select(record, Store(), data, "tfi") == "mlp"
    assert seen == [123, 123]
    record["meta"]["validation_scores"][1]["circuit_evaluations"] = 5
    with pytest.raises(ValueError, match="circuit_evaluations"):
        report_tool.per_family_select(record, Store(), data, "tfi")


def test_strong_learner_report_matches_script_a(tmp_path):
    """Fixed-learner D and per-family-selection D equal script A's estimator, interval,
    and label on synthetic Part A and Part B fits whose selections differ by seed and arm."""
    import pickle

    from qemscore.baselines.liao import _validation_score as score_fn
    from tools import descriptor_ladder_analysis as analysis

    draws = 2000
    fits, caches = tmp_path / "fits", tmp_path / "cache"
    fits.mkdir()
    caches.mkdir()
    candidates = ["random_forest", "mlp", "hgbr", "poly5_ridge"]
    rng = np.random.default_rng(84)
    expected = {}
    for part, key, families, rung in (
            ("A", "shipped-s101-n640", ["tfi", "heisenberg"], "R0"),
            ("B", "qaoa-s101-n640", ["qaoa"], "B-complete")):
        cands = candidates if part == "A" else candidates[:3]
        rows = [dict(family=family, circuit_id=f"{family}-{i:02d}", ideal_expectation=0.0,
                     noisy_expectation=0.5, severity=severity, observable=observable,
                     noise_family="depolarizing")
                for family in families for i in range(12) for severity in ("L1", "L3")
                for observable in (("z_mid", "zz_mid") if part == "A" else ("zz_mid",))]
        data = {"test": rows, "validation": rows}
        with (caches / f"{key}.pkl").open("wb") as handle:
            pickle.dump(data, handle)
        for learner_seed in (1, 2, 3):
            for arm_index, arm in enumerate(("C", "F", "P")):
                arrays = {}
                for c, cand in enumerate(cands):
                    winner = (learner_seed + arm_index) % len(cands)
                    arrays[f"validation__{cand}"] = np.array([
                        0.01 if c == (winner + families.index(row["family"])) % len(cands)
                        else 0.3 + 0.01 * c for row in rows])
                    arrays[f"test__{cand}"] = rng.uniform(0.0, 0.3, len(rows)) + c * 0.01
                scores = [score_fn(cand, rows, arrays[f"validation__{cand}"],
                                   circuit_evaluations=42, simplicity_rank=c)
                          for c, cand in enumerate(cands)]
                chosen, _, _ = select_one_standard_error(scores)
                arrays["test"] = arrays[f"test__{chosen}"]
                arrays["validation"] = arrays[f"validation__{chosen}"]
                meta = dict(key=key, rung=rung, arm=arm, learner_seed=learner_seed,
                            candidates=cands, selected_model=chosen,
                            validation_scores=[dict(name=s.name, circuit_evaluations=42)
                                               for s in scores])
                stem = f"{key}__{rung}__k{learner_seed:02d}__{arm}"
                (fits / f"{stem}.json").write_text(json.dumps(meta))
                np.savez(fits / f"{stem}.npz", **arrays)
        expected[(part, key, rung)] = data

    result = report_tool.analyze_strong_report([fits], [caches], draws=draws)
    store = analysis.FitStore([fits], False)
    comparisons = 0
    for (part, key, rung), data in expected.items():
        for family in (["tfi", "heisenberg"] if part == "A" else [None]):
            label = f"{key}/{family}" if family else key
            row = analysis.Row(data, family, label)
            arms = analysis.arms_at(store, analysis.PARTS[part], key, rung)
            cell = result["parts"][part]["rows"][label]["rungs"][rung]
            est = analysis.RowEstimator(row, store, draws, analysis.RULE_SEED, 20)
            for cand, entry in cell["fixed_learner_D"].items():
                d, parts = est.difference(arms["C"], arms["F"], name=f"test__{cand}")
                assert d == entry["D"]
                assert (analysis.classify(d, parts["first_point"])["label"]
                        == entry["classification_descriptive"])
                comparisons += 1

            class SelectedStore:
                def array(self, record, name):
                    selected = report_tool.per_family_select(record, store, data, family)
                    return store.array(record, f"test__{selected}")

            sel_est = analysis.RowEstimator(row, SelectedStore(), draws, analysis.RULE_SEED, 20)
            d, parts = sel_est.difference(arms["C"], arms["F"])
            assert d == cell["per_family_selection"]["D"]
            assert (analysis.classify(d, parts["first_point"])["label"]
                    == cell["per_family_selection"]["classification_descriptive"])
            comparisons += 1
            selected = {s: report_tool.per_family_select(arms["C"][s], store, data, family)
                        for s in arms["C"]}
            assert len(set(selected.values())) > 1  # selections differ across seeds
    assert comparisons == 2 * (4 + 1) + (3 + 1)
