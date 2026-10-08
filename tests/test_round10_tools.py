"""Synthetic checks of the round-10 tools (rule 2026-10-07-round10-checks).

The tools reuse the descriptor analysis's bootstrap; these tests build small
fit directories and caches with known errors and check each new quantity
against an independent computation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import scipy.stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import bayes_oracle as bo  # noqa: E402
from tools import descriptor_ladder_analysis as dla  # noqa: E402
from tools import oracle_sensitivity as osens  # noqa: E402
from tools import pooled_absolute as pa  # noqa: E402
from tools import reading_rules as rr  # noqa: E402
from tools import round10_summaries as summ  # noqa: E402
from tools import rule_timestamps as rt  # noqa: E402
from tools import seed_variance as sv  # noqa: E402

KEY = "shipped-s101-n640"
CELLS = [("depolarizing_readout", sev, obs) for sev in ("L1", "L3") for obs in ("z_mid", "zz_mid")]
N_SEEDS = 20


def _rows(split: str, n_circuits: int, rng) -> list[dict]:
    rows = []
    for c in range(n_circuits):
        for nf, sev, obs in CELLS:
            y = float(rng.uniform(-1, 1))
            rows.append({"item_id": f"{split}-{c}-{sev}-{obs}", "circuit_id": f"{split}-c{c}",
                         "family": "tfi", "noise_family": nf, "severity": sev,
                         "observable": obs, "ideal_expectation": y,
                         "noisy_expectation": 0.8 * y + float(rng.normal(0, 0.02)),
                         "split": split})
    return rows


def _write(fit_dir: Path, rung: str, arm: str, k: int, val: np.ndarray, test: np.ndarray,
           val_rows, test_rows, extra: dict | None = None) -> None:
    stem = f"{KEY}__{rung}__k{k:02d}__{arm}"
    arrays = {"validation": val, "test": test, **(extra or {})}
    np.savez(fit_dir / f"{stem}.npz", **arrays)

    def fam_mae(pred, rows):
        cells = {}
        for p, r in zip(pred, rows):
            cells.setdefault((r["severity"], r["observable"]), []).append(abs(p - r["ideal_expectation"]))
        maes = [math.fsum(v) / len(v) for v in cells.values()]
        return math.fsum(maes) / len(maes)

    cand_val, cand_test = {}, {}
    for name, arr in (extra or {}).items():
        split, cand = name.split("__")
        target = cand_val if split == "validation" else cand_test
        target[cand] = {"tfi": fam_mae(arr, val_rows if split == "validation" else test_rows)}
    meta = {"key": KEY, "rung": rung, "arm": arm, "learner_seed": k, "dataset_hash": "h",
            "test_family_mae": {"tfi": fam_mae(test, test_rows)},
            "validation_family_mae": {"tfi": fam_mae(val, val_rows)},
            "candidate_validation_family_mae": cand_val, "candidate_test_family_mae": cand_test}
    (fit_dir / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def synthetic(tmp_path):
    rng = np.random.default_rng(3)
    val_rows, test_rows = _rows("validation", 30, rng), _rows("test", 25, rng)
    data = {"key": KEY, "dataset_hash": "h", "validation": val_rows, "test": test_rows}
    fits = tmp_path / "fits"
    fits.mkdir()
    yv = np.array([r["ideal_expectation"] for r in val_rows])
    yt = np.array([r["ideal_expectation"] for r in test_rows])
    for k in range(1, N_SEEDS + 1):
        noise = lambda n, s: rng.normal(0, s, n)  # noqa: E731
        extra = {f"{split}__{cand}": (yv if split == "validation" else yt)
                 + noise(len(yv) if split == "validation" else len(yt), scale)
                 for cand, scale in (("random_forest", 0.05), ("mlp", 0.03), ("hgbr", 0.04),
                                     ("poly5_ridge", 0.01))
                 for split in ("validation", "test")}
        _write(fits, "N1", "C", k, extra["validation__mlp"], extra["test__mlp"], val_rows,
               test_rows, extra)
        _write(fits, "N1", "F", k, yv + noise(len(yv), 0.02), yt + noise(len(yt), 0.02),
               val_rows, test_rows)
        _write(fits, "N1", "P", k, yv + noise(len(yv), 0.035), yt + noise(len(yt), 0.035),
               val_rows, test_rows)
    store = dla.FitStore([fits], use_orig=False)
    row = dla.Row(data, "tfi", f"{KEY}/tfi")
    return {"data": data, "store": store, "row": row}


def test_difference_entry_equals_analysis_arithmetic(synthetic) -> None:
    store, row = synthetic["store"], synthetic["row"]
    est = dla.RowEstimator(row, store, rr.DRAWS, rr.SEED, N_SEEDS)
    c, f = store.seeded(KEY, "N1", "C"), store.seeded(KEY, "N1", "F")
    ref, _ = est.difference(c, f)
    err_c = rr.item_errors(row, store, rr.seeded(store, KEY, "N1", "C"), "test")
    err_f = rr.item_errors(row, store, rr.seeded(store, KEY, "N1", "F"), "test")
    got, _ = rr.difference_entry(row, err_c, err_f)
    assert got["point"] == pytest.approx(ref["point"], abs=1e-15)
    assert got["lower"] == pytest.approx(ref["interval"]["lower"], abs=1e-15)
    assert got["upper"] == pytest.approx(ref["interval"]["upper"], abs=1e-15)


def test_stack_recovers_label_when_r_is_exact(synthetic) -> None:
    data, row = synthetic["data"], synthetic["row"]
    for split in ("validation", "test"):
        for r in data[split]:
            r["noisy_expectation"] = 0.5 * r["ideal_expectation"] + 0.1
    row = dla.Row(data, "tfi", row.label)
    val = rr.Validation(data, "tfi", row)
    junk_v = np.zeros(len(data["validation"])) + 0.3
    junk_t = np.zeros(len(data["test"])) + 0.3
    recal, stack = rr.stack_errors(row, val, junk_v, junk_t)
    assert np.max(stack) < 1e-10
    assert np.min(recal) >= 0.0 and np.mean(recal) > 0.1


def test_reference_selection_is_validation_argmin(synthetic) -> None:
    store = synthetic["store"]
    records = rr.seeded(store, KEY, "N1", "C")
    choice = rr.select_reference(store, records, "tfi")
    assert choice["selected"] == "poly5_ridge"
    means = choice["validation_means"]
    assert means["poly5_ridge"] < means["mlp"] < means["hgbr"] < means["random_forest"]


def test_recorded_validation_check_catches_misalignment(synthetic) -> None:
    data, store, row = synthetic["data"], synthetic["store"], synthetic["row"]
    val = rr.Validation(data, "tfi", row)
    records = rr.seeded(store, KEY, "N1", "C")
    assert rr.check_recorded_validation(val, store, records, "mlp") < 1e-12
    data["validation"] = data["validation"][::-1]
    with pytest.raises(SystemExit):
        rr.check_recorded_validation(rr.Validation(data, "tfi", row), store, records, "mlp")


def test_tally_and_changes() -> None:
    def cell(truth, d, r):
        v = {rule: "no detected information" for rule in rr.RULES}
        v["rule_D"] = "information" if d else "no detected information"
        v["rule_R"] = "information" if r else "no detected information"
        return {"truth": truth, "verdicts": v, "fit_set": "x", "group": "first_2048"}
    cells = [cell("information", 1, 1), cell("information", 1, 0), cell("information", 0, 1),
             cell("no information", 1, 0), cell("no information", 0, 0), cell("undetermined", 1, 1)]
    t = rr.tally(cells, "rule_R")
    assert (t["information"], t["detected"], t["false_positives"], t["undetermined_flagged"]) == (3, 2, 0, 1)
    assert t["specificity"] == 1.0
    ch = rr.changes(cells, "rule_R")
    assert ch["information"] == {"gained": 1, "lost": 1}
    assert ch["no information"] == {"gained": 0, "lost": 1}


def test_cell_components_and_independent_resampling(synthetic) -> None:
    store, row = synthetic["store"], synthetic["row"]
    est = dla.RowEstimator(row, store, rr.DRAWS, rr.SEED, N_SEEDS)
    d_ref, _ = est.difference(store.seeded(KEY, "N1", "C"), store.seeded(KEY, "N1", "F"))
    err_c = rr.item_errors(row, store, rr.seeded(store, KEY, "N1", "C"), "test")
    err_f = rr.item_errors(row, store, rr.seeded(store, KEY, "N1", "F"), "test")
    out = sv.cell_components(row, err_c, err_f, d_ref)
    per_seed = np.array(out["per_learner_seed_D"])
    assert out["var_learner"] == pytest.approx(per_seed.var(ddof=1) / N_SEEDS)
    assert out["paired"]["lower"] == pytest.approx(d_ref["interval"]["lower"], abs=1e-15)
    # Identical errors in every learner seed: paired and independent draws coincide.
    ec, ef = np.repeat(err_c[:1], N_SEEDS, 0), np.repeat(err_f[:1], N_SEEDS, 0)
    entry, draws = rr.difference_entry(row, ec, ef)
    released = {"point": entry["point"],
                "interval": {"lower": entry["lower"], "upper": entry["upper"]},
                "draw_sd": float(draws.std(ddof=1))}
    same = sv.cell_components(row, ec, ef, released)
    assert same["var_learner"] == pytest.approx(0.0, abs=1e-30)
    assert same["independent"]["lower"] == pytest.approx(same["paired"]["lower"], abs=1e-15)
    assert same["independent"]["upper"] == pytest.approx(same["paired"]["upper"], abs=1e-15)
    with pytest.raises(SystemExit):
        sv.cell_components(row, err_c, err_f, {**released, "point": released["point"] + 1})


def test_pool_matches_part_d_and_prediction_interval() -> None:
    per_seed = {s: {"D": d, "var_two_stage": v, "var_learner": v / 3, "var_circuit": v / 2}
                for s, d, v in zip((101, 211, 307, 401, 503, 607),
                                   (0.010, 0.012, 0.008, 0.015, 0.011, 0.009),
                                   (1e-6, 2e-6, 1.5e-6, 1e-6, 3e-6, 2e-6))}
    out = sv.pool(per_seed, None)
    meta = pa.hksj_meta_analysis([v["D"] for v in per_seed.values()],
                                 [math.sqrt(v["var_two_stage"]) for v in per_seed.values()])
    assert out["theta_re"] == pytest.approx(meta["theta_re"])
    assert out["tau2"] == pytest.approx(meta["tau2"])
    w = 1 / (np.array([v["var_two_stage"] for v in per_seed.values()]) + meta["tau2"])
    half = scipy.stats.t.ppf(0.975, 4) * math.sqrt(meta["tau2"] + 1 / w.sum())
    assert out["prediction_interval_true"]["upper"] == pytest.approx(meta["theta_re"] + half)


def test_first_vs_new_is_welch() -> None:
    per_seed = {s: {"D": d, "paired": {"label": "x"}} for s, d in
                zip((101, 211, 307, 401, 503, 607), (0.01, 0.02, 0.015, 0.03, 0.025, 0.04))}
    out = sv.first_vs_new(per_seed, (101, 211, 307), (401, 503, 607))
    a, b = np.array([0.01, 0.02, 0.015]), np.array([0.03, 0.025, 0.04])
    res = scipy.stats.ttest_ind(b, a, equal_var=False)
    ci = res.confidence_interval(0.95)
    assert out["interval"]["lower"] == pytest.approx(ci.low)
    assert out["interval"]["upper"] == pytest.approx(ci.high)


def test_binomial_likelihood_matches_scipy_up_to_constant() -> None:
    shots = 2048
    k = 1300
    r = 2 * k / shots - 1
    e = np.array([0.1, 0.2, 0.25, 0.3])
    samples = np.zeros((4, 2))
    y_s = {"model": None}

    class Lin:
        def __init__(self, values):
            self.values = values

        def predict(self, _):
            return self.values

    class Ident:
        def transform(self, x):
            return x

    surr_y = {"z_mid": {"poly": Ident(), "model": Lin(np.arange(4.0))}}
    surr_e = {("z_mid", "L1"): {"poly": Ident(), "model": Lin(e)},
              ("z_mid", "L3"): {"poly": Ident(), "model": Lin(e)}}
    items = [{"item_id": "a", "observable": "z_mid", "severity": "L1", "noisy_expectation": r}]
    out = bo.compute_bayes_predictions_for_circuit(items, samples, surr_y, surr_e, shots, "tfi",
                                                   likelihood="binomial")
    logp = scipy.stats.binom.logpmf(k, shots, (1 + e) / 2)
    w = np.exp(logp - logp.max())
    assert out[2]["a"] == pytest.approx(bo.compute_effective_sample_size(w))
    with pytest.raises(ValueError):
        bad = [{**items[0], "noisy_expectation": r + 1e-4}]
        bo.compute_bayes_predictions_for_circuit(bad, samples, surr_y, surr_e, shots, "tfi",
                                                 likelihood="binomial")
    del y_s


def test_stream_option_keeps_default_draws() -> None:
    true = {"j": 0.5, "h": 0.7}
    a = bo.draw_coupling_samples("N1", "tfi", true, (0.1, -0.2), 101, 3, sample_size=50)
    b = bo.draw_coupling_samples("N1", "tfi", true, (0.1, -0.2), 101, 3, sample_size=50,
                                 stream=None)
    c = bo.draw_coupling_samples("N1", "tfi", true, (0.1, -0.2), 101, 3, sample_size=50,
                                 stream=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_surrogate_degree_option() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (200, 2))
    y = x[:, 0] ** 7 + x[:, 1]
    s5 = bo.fit_degree5_surrogate(x[:150], y[:150], x[150:], y[150:])
    s7 = bo.fit_degree5_surrogate(x[:150], y[:150], x[150:], y[150:], degree=7)
    assert s7["degree"] == 7 and s5["degree"] == 5
    assert s7["validation_mae"] < s5["validation_mae"]


def test_sensitivity_flags_require_sensitivity_mode(tmp_path) -> None:
    rule = tmp_path / "r.md"
    rule.write_text("x")
    with pytest.raises(SystemExit):
        bo.parse_args(["--frozen-rule", str(rule), "--eval-split", "test", "--stream", "1",
                       "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        bo.parse_args(["--frozen-rule", str(rule), "--eval-split", "test", "--sensitivity"])
    args = bo.parse_args(["--frozen-rule", str(rule), "--eval-split", "test", "--sensitivity",
                          "--likelihood", "binomial", "--out", str(tmp_path / "o.json")])
    assert args.likelihood == "binomial"


def test_oracle_class() -> None:
    assert osens.klass({"G_star": {"interval": {"lower": 1e-9}}}) == "information"
    assert osens.klass({"G_star": {"interval": {"lower": 0.0}}}) == "undetermined"


def test_harm_rate_mean_entry(synthetic) -> None:
    row = synthetic["row"]
    ones = np.ones((N_SEEDS, row.n_items))
    out = summ.mean_entry(row, ones)
    assert out["point"] == 1.0 and out["lower"] == 1.0 and out["upper"] == 1.0
    half = np.zeros((N_SEEDS, row.n_items))
    half[:, ::2] = 1.0
    out = summ.mean_entry(row, half)
    expected = np.mean([dla.point_macro(half[0], row.members, row.all_cells)])
    assert out["point"] == pytest.approx(expected)


def test_rule_timestamp_helpers() -> None:
    pages = '[{"a": 1}, {"a": 2}]\n[{"a": 3}]'
    assert [e["a"] for e in rt.flatten(pages)] == [1, 2, 3]
    assert rt.earliest({"created_at": "2026-10-02T00:00:00Z"}, None,
                       {"created_at": "2026-10-01T00:00:00Z"}) == "2026-10-01T00:00:00Z"
    assert rt.earliest(None) is None


def _tallies(**spec_sens):
    out = {}
    for rule in rr.RULES:
        sp, se = spec_sens.get(rule, (0.0, 0.0))
        out[rule] = {"specificity": sp, "sensitivity": se}
    return out


def test_recommendation_keeps_primary_when_none_eligible() -> None:
    t = _tallies(rule_R=(0.9, 0.8), rule_D=(0.9, 0.9), rule_S=(0.5, 0.95),
                 rule_RS=(0.9, 0.7), rule_R_delta=(1.0, 0.6))
    assert rr.recommend(t) == {"recommended": "rule_R", "eligible": []}


def test_recommendation_ranks_eligible_rules_and_breaks_ties() -> None:
    # Several eligible: highest specificity wins.
    t = _tallies(rule_R=(0.8, 0.8), rule_D=(0.85, 0.8), rule_RS=(0.95, 0.76),
                 rule_R_delta=(0.9, 0.79))
    assert rr.recommend(t)["recommended"] == "rule_RS"
    assert set(rr.recommend(t)["eligible"]) == {"rule_D", "rule_RS", "rule_R_delta"}
    # Equal specificity: higher sensitivity wins.
    t = _tallies(rule_R=(0.8, 0.8), rule_D=(0.9, 0.78), rule_RS=(0.9, 0.79))
    assert rr.recommend(t)["recommended"] == "rule_RS"
    # Exact tie: the earlier rule in the frozen order wins.
    t = _tallies(rule_R=(0.8, 0.8), rule_S=(0.9, 0.79), rule_RS=(0.9, 0.79))
    assert rr.recommend(t)["recommended"] == "rule_S"


def test_recommendation_sensitivity_boundary() -> None:
    at = _tallies(rule_R=(0.8, 0.80), rule_R_delta=(1.0, 0.75))
    assert rr.recommend(at)["recommended"] == "rule_R_delta"
    below = _tallies(rule_R=(0.8, 0.80), rule_R_delta=(1.0, 0.7499))
    assert rr.recommend(below)["recommended"] == "rule_R"
    same_spec = _tallies(rule_R=(0.8, 0.80), rule_D=(0.8, 0.99))
    assert rr.recommend(same_spec)["recommended"] == "rule_R"


def test_redraw_stream_appends_after_governed_words() -> None:
    out = bo.draw_coupling_samples("R5", "tfi", {}, (), 101, 7, sample_size=5,
                                   extra_seed_word=10, stream=2)
    rng = np.random.default_rng(np.random.SeedSequence(
        [bo.SURROGATE_SEED, 101, 7, bo.RUNG_INDEX["R5"], 10, bo.STREAM_TAG, 2]))
    assert np.array_equal(out, rng.uniform(0.2, 1.2, size=(5, 2)))
    governed = bo.draw_coupling_samples("R5", "tfi", {}, (), 101, 7, sample_size=5,
                                        extra_seed_word=10)
    rng = np.random.default_rng(np.random.SeedSequence(
        [bo.SURROGATE_SEED, 101, 7, bo.RUNG_INDEX["R5"], 10]))
    assert np.array_equal(governed, rng.uniform(0.2, 1.2, size=(5, 2)))


def test_draw_multiplier_reaches_draws_and_redraws(monkeypatch) -> None:
    items = [{"item_id": f"i{k}", "severity": sev, "observable": obs, "ideal_expectation": 0.0,
              "j": 0.5, "h": 0.5, "noisy_expectation": 0.0}
             for k, (sev, obs) in enumerate((("L1", "z_mid"), ("L1", "zz_mid"),
                                             ("L3", "z_mid"), ("L3", "zz_mid")))]
    sizes, streams, likelihoods = [], [], []

    def fake_draw(rung, fam, true_c, z, seed, idx, sample_size=0, extra_seed_word=None,
                  stream=None):
        sizes.append((sample_size, extra_seed_word))
        streams.append(stream)
        return np.zeros((4, 2))

    def fake_pred(c_items, samples, y_s, e_s, shots, fam, likelihood="gaussian"):
        likelihoods.append(likelihood)
        ids = [it["item_id"] for it in c_items]
        low = {i: (10.0 if len(likelihoods) == 1 else 500.0) for i in ids}
        return ({i: 0.1 for i in ids}, {i: 0.2 for i in ids}, low,
                {i: 0.3 for i in ids}, dict(low))

    monkeypatch.setattr(bo, "draw_coupling_samples", fake_draw)
    monkeypatch.setattr(bo, "compute_bayes_predictions_for_circuit", fake_pred)
    task = {"seed": 101, "lvl": 2048, "shots_int": 2048, "fam": "tfi", "rung": "N2",
            "sorted_circuits": ["c0"], "by_circuit": {"c0": items},
            "circ_lookup": {"c0": {"circuit_index": 0, "z": [0.0, 0.0]}},
            "y_surrs": {}, "e_surrs": {}, "stream": 3, "draw_multiplier": 4,
            "likelihood": "binomial"}
    bo.evaluate_rung_task(task)
    assert sizes == [(bo.DEFAULT_M_SAMPLES * 4, None),
                     (bo.DEFAULT_M_SAMPLES * bo.REDRAW_FACTOR * 4, 10)]
    assert streams == [3, 3] and likelihoods == ["binomial", "binomial"]


def test_estimate_prediction_interval_adds_within_variance() -> None:
    per_seed = {s: {"D": d, "var_two_stage": v, "var_learner": v / 3, "var_circuit": v / 2}
                for s, d, v in zip((101, 211, 307, 401, 503, 607),
                                   (0.010, 0.012, 0.008, 0.015, 0.011, 0.009),
                                   (1e-6, 2e-6, 1.5e-6, 1e-6, 3e-6, 2e-6))}
    out = sv.pool(per_seed, None)
    v2 = np.mean([v["var_two_stage"] for v in per_seed.values()])
    half = scipy.stats.t.ppf(0.975, 4) * math.sqrt(out["tau2"] + v2 + out["se_theta"] ** 2)
    assert out["prediction_interval_estimate"]["lower"] == pytest.approx(out["theta_re"] - half)
    assert out["dataset_seed_fraction"] == pytest.approx(out["tau2"] / (out["tau2"] + v2))


def test_sensitivity_recount_separates_unresolved(tmp_path, monkeypatch) -> None:
    def risk(lower):
        g = {"point": lower + 1e-3, "interval": {"lower": lower, "upper": lower + 2e-3}}
        return {"C_star": g, "F_star": g, "G_star": g}
    keys = {"s101__tfi__N1__shots_2048__strength_observed": 1e-4,
            "s101__tfi__N3__shots_2048__strength_observed": 5e-3}
    governed = {"oracle_risks": {k: risk(v) for k, v in keys.items()},
                "surrogates": {}, "equation_2": {"rows": [
                    {"fit_set": "x", "dataset_seed": 101, "family": "tfi", "rung": "N1",
                     "shot_level": 2048, "variant": "strength_observed",
                     "pipeline_label": "F beats C"},
                    {"fit_set": "x", "dataset_seed": 101, "family": "tfi", "rung": "N3",
                     "shot_level": 2048, "variant": "strength_observed",
                     "pipeline_label": "not distinguished"},
                    {"fit_set": "x_derived", "dataset_seed": 101, "family": "tfi", "rung": "R0",
                     "shot_level": 2048, "variant": "strength_observed",
                     "pipeline_label": "F beats C"}]}}
    gpath = tmp_path / "governed.json"
    gpath.write_text(json.dumps(governed))
    monkeypatch.setattr(osens, "GOVERNED", gpath)
    runs = []
    for name, opts in osens.REQUIRED_RUNS.items():
        risks = {k: risk(v) for k, v in keys.items()}
        if name == "degree7":  # flips the N1 cell to undetermined
            risks["s101__tfi__N1__shots_2048__strength_observed"] = risk(-1e-4)
        doc = {"schema": "bayes_oracle_sensitivity_v1", "options": {**opts, "rungs": None},
               "oracle_risks": risks, "surrogates": {}, "seconds": 1.0}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        runs.append(f"{name}={path}")
    rule = REPO / rr.RULE_FILE
    args = type("A", (), {"frozen_rule": rule, "run": runs})()
    out = osens.run(args)
    assert out["n_unresolved"] == 1
    allc = out["confusion_recount"]["all"]
    assert allc["unresolved_information"] == 1 and allc["unresolved_information_detected"] == 1
    assert allc["resolved_information_missed"] == 1
    assert allc["false_positives"] == 1
    assert out["confusion_recount"]["fit_identity"].get("no_information", 0) == 0


def test_d_reproduction_guard_checks_upper_endpoint() -> None:
    released = {"point": 0.01, "interval": {"lower": 0.005, "upper": 0.015}}
    entry = {"point": 0.01, "interval": {"lower": 0.005, "upper": 0.015}}
    assert summ.check_d_reproduces(entry, released, "x") == 0.0
    with pytest.raises(SystemExit):
        summ.check_d_reproduces(entry, {**released, "interval": {"lower": 0.005,
                                                                  "upper": 0.016}}, "x")


def test_first_push_uses_ancestry(tmp_path, monkeypatch) -> None:
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    shas = []
    for i in range(3):
        (tmp_path / f"f{i}").write_text(str(i))
        git("add", f"f{i}")
        git("commit", "-q", "-m", f"c{i}")
        shas.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(rt, "_REPO", tmp_path)
    pushes = [{"id": "e2", "created_at": "2026-10-02T00:00:00Z",
               "payload": {"head": shas[2], "before": shas[1]}},
              {"id": "e1", "created_at": "2026-10-01T00:00:00Z",
               "payload": {"head": shas[1], "before": None}}]
    assert rt.first_push(shas[0], pushes)["event_id"] == "e1"
    assert rt.first_push(shas[2], pushes)["event_id"] == "e2"
    runs = [{"id": 9, "created_at": "2026-10-03T00:00:00Z", "event": "push",
             "head_sha": shas[2], "name": "ci"}]
    assert rt.first_run(shas[1], runs)["run_id"] == 9


def test_qraft_published_check_computes_no_contrast(monkeypatch) -> None:
    from reanalysis.scripts.qraft import score_qraft_machine_matched as qmm

    class Exploding(dict):
        def items(self):
            raise AssertionError("contrast computed")
    monkeypatch.setattr(qmm, "CONTRASTS", Exploding())
    rng = np.random.default_rng(0)
    seeds = range(1, 11)

    def inputs(arms):
        per_seed = {a: {s: float(rng.uniform(0.1, 0.2)) for s in seeds} for a in arms}
        return per_seed, {a: rng.uniform(0.1, 0.2, 50) for a in arms}
    out = qmm.summarize(*inputs(qmm.PUBLISHED))
    assert set(out["arms"]) == set(qmm.PUBLISHED)
    assert out["contrasts"] == {}
    with pytest.raises(AssertionError, match="contrast computed"):
        qmm.summarize(*inputs(qmm.ARMS))


def test_reference_selection_ignores_test_error_and_breaks_ties() -> None:
    class Store:
        def __init__(self, missing=()):
            self.missing = set(missing)

        def array(self, record, name):
            return None if (record["k"], name) in self.missing else np.zeros(3)

    def records(val, test):
        return [{"k": k, "meta": {"candidate_validation_family_mae": {c: {"tfi": v}
                                                                       for c, v in val.items()},
                                  "candidate_test_family_mae": {c: {"tfi": v}
                                                                for c, v in test.items()}}}
                for k in (1, 2)]
    # Validation ranks hgbr first; test ranks poly5_ridge first.
    recs = records({"random_forest": 0.05, "mlp": 0.04, "hgbr": 0.01, "poly5_ridge": 0.03},
                   {"random_forest": 0.05, "mlp": 0.04, "hgbr": 0.03, "poly5_ridge": 0.01})
    assert rr.select_reference(Store(), recs, "tfi")["selected"] == "hgbr"
    tied = records({"random_forest": 0.05, "mlp": 0.02, "hgbr": 0.03, "poly5_ridge": 0.02},
                   {"random_forest": 0.01, "mlp": 0.02, "hgbr": 0.03, "poly5_ridge": 0.01})
    assert rr.select_reference(Store(), tied, "tfi")["selected"] == "mlp"
    choice = rr.select_reference(Store({(2, "test__mlp")}), tied, "tfi")
    assert choice["selected"] == "poly5_ridge"
    assert set(choice["excluded"]) == {"mlp"}


def test_independent_resampling_does_not_cancel_identical_arms(synthetic) -> None:
    store, row = synthetic["store"], synthetic["row"]
    base = rr.item_errors(row, store, rr.seeded(store, KEY, "N1", "C"), "test")[:1]
    # C and F identical within each learner seed, different across seeds.
    err = np.repeat(base, N_SEEDS, 0) * np.linspace(0.5, 1.5, N_SEEDS)[:, None]
    entry, draws = rr.difference_entry(row, err, err.copy())
    released = {"point": entry["point"],
                "interval": {"lower": entry["lower"], "upper": entry["upper"]},
                "draw_sd": float(draws.std(ddof=1))}
    out = sv.cell_components(row, err, err.copy(), released)
    assert out["paired"]["lower"] == out["paired"]["upper"] == 0.0
    assert out["independent"]["draw_sd"] > 1e-3
    assert out["independent"]["lower"] < 0.0 < out["independent"]["upper"]
