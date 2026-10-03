"""Rung builders and fit planning of the descriptor-information experiment.

These tests fit nothing. They pin the column layouts of the frozen rule
(docs/frozen-rules/2026-10-02-descriptor-information.md), the common random
numbers of the N rungs, and the planned fit counts.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from qemscore.datasets.schema import FEATURES, build_features
from tools import descriptor_common as common
from tools import descriptor_ladder as ladder
from tools import qaoa_intermediate as qaoa


def _spin_item(item_id, circuit_id, family, split, **couplings):
    item = {
        "item_id": item_id, "circuit_id": circuit_id, "family": family, "split": split,
        "noisy_expectation": 0.3, "shots": 2048, "n_qubits": 10, "steps": 3,
        "dt": 0.2 if family == "tfi" else 0.15, "two_qubit_gates": 54,
        "transpiled_depth": 40, "obs_locality": 1,
    }
    item.update(couplings)
    return item


@pytest.fixture()
def spin_data():
    rows = []
    circuits = [("circuit-b", "tfi", {"j": 0.5, "h": 0.9}),
                ("circuit-a", "heisenberg", {"jx": 0.3, "jy": 0.7, "jz": 1.1}),
                ("circuit-c", "tfi", {"j": 1.0, "h": 0.25})]
    for split, (circuit, family, couplings) in zip(
            ("train", "validation", "test"), circuits):
        for observable in ("z_mid", "zz_mid"):
            rows.append(_spin_item(f"item-{circuit}-{observable}", circuit, family,
                                   split, **couplings))
    data = {"seed": 211, "train": [r for r in rows if r["split"] == "train"],
            "validation": [r for r in rows if r["split"] == "validation"],
            "test": [r for r in rows if r["split"] == "test"]}
    return data, rows


def test_coupling_noise_follows_the_rule_seed_sequence(spin_data):
    data, rows = spin_data
    table = ladder.coupling_noise_table(data)
    ordered = [entry["circuit_id"] for entry in table["circuits"]]
    assert ordered == sorted({r["circuit_id"] for r in rows})
    for index, entry in enumerate(table["circuits"]):
        rng = np.random.default_rng(np.random.SeedSequence([20261002, 211, index]))
        expected = rng.standard_normal(len(ladder.COUPLINGS[entry["family"]]))
        assert entry["z"] == [float(v) for v in expected]
    # One z per circuit: both observables of a circuit share it.
    for row in rows:
        partner = next(r for r in rows if r["circuit_id"] == row["circuit_id"]
                       and r["item_id"] != row["item_id"])
        assert table["z_by_item"][row["item_id"]] == table["z_by_item"][partner["item_id"]]


@pytest.mark.parametrize("rung", ["N1", "N2", "N3", "N4"])
def test_n_rungs_change_only_the_couplings(spin_data, rung):
    data, rows = spin_data
    table = ladder.coupling_noise_table(data)
    builder = ladder.RungBuilder(rung, z_by_item=table["z_by_item"])
    s = ladder.NOISE_LEVELS[rung]
    assert len(builder.names) == len(FEATURES)
    for row in rows:
        base = build_features(row)
        noisy = builder(dict(row))
        z = dict(zip(ladder.COUPLINGS[row["family"]], table["z_by_item"][row["item_id"]]))
        for index, name in enumerate(FEATURES):
            if name in z:
                assert noisy[index] == base[index] + s * z[name]
            else:
                assert noisy[index] == base[index]
    # The same z at every rung: N4 - R0 is 30 times N1 - R0 per coupling.
    n1 = ladder.RungBuilder("N1", z_by_item=table["z_by_item"])
    n4 = ladder.RungBuilder("N4", z_by_item=table["z_by_item"])
    row = rows[0]
    base = np.asarray(build_features(row))
    assert np.allclose(np.asarray(n4(dict(row))) - base,
                       30.0 * (np.asarray(n1(dict(row))) - base))


def test_n_rungs_work_on_label_free_prediction_rows(spin_data):
    data, rows = spin_data
    table = ladder.coupling_noise_table(data)
    builder = ladder.RungBuilder("N2", z_by_item=table["z_by_item"])
    row = {k: v for k, v in rows[-1].items() if k not in ("circuit_id",)}
    assert len(builder(row)) == len(FEATURES)


def test_r3_and_r5_layouts(spin_data):
    _, rows = spin_data
    r3t = ladder.RungBuilder("R3-TFI")
    r3h = ladder.RungBuilder("R3-Heis")
    r5 = ladder.RungBuilder("R5")
    assert "h" not in r3t.names and len(r3t.names) == len(FEATURES) - 1
    assert "jz" not in r3h.names and len(r3h.names) == len(FEATURES) - 1
    assert r5.names == common.KEPT
    for row in rows:
        full = dict(zip(FEATURES, build_features(row)))
        assert r3t(dict(row)) == [full[n] for n in r3t.names]
        assert r3h(dict(row)) == [full[n] for n in r3h.names]
        assert r5(dict(row)) == [full[n] for n in common.KEPT]


def test_r4_replaces_the_spin_block_in_place(spin_data):
    _, rows = spin_data
    names = [f"e{i}" for i in range(165)]
    encoding = {r["circuit_id"]: np.arange(165, dtype=float) * 0.01 + len(r["circuit_id"])
                for r in rows}
    circuit_by_item = {r["item_id"]: r["circuit_id"] for r in rows}
    builder = ladder.RungBuilder("R4", encoding_by_circuit=encoding,
                                 circuit_by_item=circuit_by_item, encoder_names=names)
    assert len(builder.names) == len(FEATURES) - len(ladder.SPIN_BLOCK) + 165
    assert not set(ladder.SPIN_BLOCK) & set(builder.names)
    assert set(common.KEPT) <= set(builder.names)
    start = builder.names.index("mlqem:e0")
    assert builder.names[start - 1] == "family_near_clifford"
    row = rows[0]
    values = builder(dict(row))
    assert values[start:start + 165] == list(encoding[row["circuit_id"]])


def test_qaoa_builders():
    item = {"item_id": "i", "family": "qaoa", "noisy_expectation": 0.1, "shots": 2048,
            "n_qubits": 10, "p": 2, "graph_class": "3_regular",
            "edges": [[0, 3], [5, 1], [8, 9]], "gammas": [0.3, 1.2],
            "betas": [0.2, 0.4], "two_qubit_gates": 30, "transpiled_depth": 25,
            "obs_locality": 2}
    partial, partial_names = qaoa.builder_for("B-partial", 10)
    assert partial is build_features and partial_names == tuple(FEATURES)
    complete = qaoa.QAOABuilder("B-complete", 10)
    assert len(complete.names) == len(FEATURES) + 45
    values = complete(dict(item))
    assert values[:len(FEATURES)] == build_features(item)
    edges = values[len(FEATURES):]
    on = {pair for pair, value in zip(combinations(range(10), 2), edges) if value}
    assert on == {(0, 3), (1, 5), (8, 9)}
    none = qaoa.QAOABuilder("B-none", 10)
    full = dict(zip(FEATURES, build_features(item)))
    assert none(dict(item)) == [full[n] for n in common.KEPT]


def test_planned_fit_counts(tmp_path):
    primary = ["shipped-s101-n640", "shipped-s211-n640", "shipped-s307-n640"]
    nc = ["nc-s101-n640", "nc-s211-n640", "nc-s307-n640"]
    jobs = ladder.build_jobs(tmp_path, primary, nc, list(ladder.DEFAULT_RUNGS),
                             list(ladder.ARMS), list(range(1, 21)))
    new_rungs = 8 * 3 * (1 + 20 * 3) + 3 * 20 * 2
    r0_reexport = 3 * 20 * 3 + 3          # seeds 1-20 plus M.4's original-seed P
    nc_reexport = 3 * 21 * 3              # seeds 1-20 plus Appendix N's anchor
    assert len(jobs) == new_rungs + r0_reexport + nc_reexport == 1956
    assert len({job["stem"] for job in jobs}) == len(jobs)
    nc_jobs = [job for job in jobs if job["rung"] == ladder.NC_RUNG]
    assert {job["fit_arm"] for job in nc_jobs} == {"F", "C"}
    assert all(job["rung_builder"] == "R5" for job in nc_jobs)

    r0 = [job for job in jobs if job["rung"] == "R0"]
    assert len(r0) == r0_reexport and {job["arm"] for job in r0} == {"F", "C", "P"}
    orig = [job for job in r0 if "__orig__" in job["stem"]]
    assert [(job["arm"], job["learner_seed"], job["shuffle_seed"]) for job in orig] == [
        ("P", 101, 1234), ("P", 211, 1234), ("P", 307, 1234)]
    nc_r0 = [job for job in jobs if job["rung"] == ladder.NC_REEXPORT_RUNG]
    assert len(nc_r0) == nc_reexport
    assert all(job["rung_builder"] == "R0" for job in nc_r0)
    assert {job["learner_seed"] for job in nc_r0 if job["key"] == "nc-s211-n640"} == {
        *range(1, 21), 211}
    # Every P except M.4's original-seed one shuffles with its learner seed.
    assert all(job["shuffle_seed"] == job["learner_seed"]
               for job in jobs if job["fit_arm"] == "P" and job not in orig)
    assert not [job for job in jobs if job["arm"] == "A"
                and job["rung"] in ladder.REEXPORT_RUNGS]
    # Re-exports are planned first, so a mismatch shows before the new rungs run.
    first_new = next(i for i, job in enumerate(jobs)
                     if job["rung"] not in ladder.REEXPORT_RUNGS)
    assert all(job["rung"] in ladder.REEXPORT_RUNGS for job in jobs[:first_new])
    assert first_new == r0_reexport + nc_reexport

    keys = ["qaoa-s101-n640", "qaoa-s211-n640", "qaoa-s307-n640"]
    b_jobs = qaoa.build_jobs(tmp_path, keys, list(qaoa.RUNGS), list(qaoa.ARMS),
                             list(range(1, 21)))
    assert len(b_jobs) == 3 * 3 * (1 + 20 * 3) + 3
    assert sum(job["arm"] == "GBT" for job in b_jobs) == 3


def test_new_fits_refuse_without_a_passed_gate(tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        common.require_gate(tmp_path / "gate_r0.json")
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"passed": False, "max_abs_diff_test": 2e-11,
                                  "tolerance": 1e-11}))
    with pytest.raises(SystemExit, match="did not pass"):
        common.require_gate(failed)
    passed = tmp_path / "passed.json"
    passed.write_text(json.dumps({"passed": True, "criterion_id": "archive",
                                  "tolerance": 1e-11}))
    assert common.require_gate(Path(passed))["passed"] is True
    # A pass under anything but the frozen gate needs a recorded amendment.
    for changed in ({"criterion_id": "archive", "tolerance": 1.4e-11},
                    {"criterion_id": "m4-refit-identity", "tolerance": 1e-11},
                    {}):
        loose = tmp_path / "loose.json"
        loose.write_text(json.dumps({"passed": True, **changed}))
        with pytest.raises(SystemExit, match="not the frozen gate"):
            common.require_gate(loose)
    amended = tmp_path / "amended.json"
    amended.write_text(json.dumps({"passed": True, "criterion_id": "archive",
                                   "tolerance": 1.4e-11, "amendment": "text"}))
    assert common.require_gate(amended)["amendment"] == "text"


def test_gate_amendment_must_be_written_into_the_rule(tmp_path):
    rule = tmp_path / "rule.md"
    rule.write_text("## Deviations\n\n- Amendment (2026-10-02): the R0 gate tolerance\n"
                    "  is 1.4e-11, the bound the paper states.\n")
    frozen = common.check_gate_amendment(criterion="archive", tolerance=1e-11,
                                         amendment=None, seedrep_fits=None,
                                         rule_file=rule)
    assert frozen == {"amended": False, "amendment": None}
    with pytest.raises(SystemExit, match="drop --amendment"):
        common.check_gate_amendment(criterion="archive", tolerance=1e-11,
                                    amendment="x", seedrep_fits=None, rule_file=rule)
    with pytest.raises(SystemExit, match="Only the author can amend"):
        common.check_gate_amendment(criterion="archive", tolerance=1.4e-11,
                                    amendment=None, seedrep_fits=None, rule_file=rule)
    with pytest.raises(SystemExit, match="does not appear"):
        common.check_gate_amendment(criterion="archive", tolerance=1.4e-11,
                                    amendment="tolerance is 2e-11", seedrep_fits=None,
                                    rule_file=rule)
    with pytest.raises(SystemExit, match="needs --seedrep-fits"):
        common.check_gate_amendment(criterion="m4-refit-identity", tolerance=1e-11,
                                    amendment="the R0 gate tolerance is 1.4e-11",
                                    seedrep_fits=None, rule_file=rule)
    # Whitespace and line breaks do not matter; the words do.
    amended = common.check_gate_amendment(
        criterion="archive", tolerance=1.4e-11,
        amendment="the R0 gate tolerance is 1.4e-11, the bound the paper states.",
        seedrep_fits=None, rule_file=rule)
    assert amended["amended"] is True


def test_gate_decision_reports_the_frozen_verdict_beside_an_amended_one():
    observed = {"archive_max": 1.3656e-11, "refit_identity_max": 0.0, "crashes": []}
    assert common.gate_decision(criterion="archive", tolerance=1e-11, **observed) == {
        "passed": False, "frozen_criterion_passed": False}
    assert common.gate_decision(criterion="archive", tolerance=1.4e-11, **observed) == {
        "passed": True, "frozen_criterion_passed": False}
    assert common.gate_decision(criterion="m4-refit-identity", tolerance=1e-11,
                                **observed)["passed"] is True
    assert common.gate_decision(criterion="m4-refit-identity", tolerance=1e-11,
                                archive_max=0.0, refit_identity_max=1e-16,
                                crashes=[])["passed"] is False
    assert common.gate_decision(criterion="m4-refit-identity", tolerance=1e-11,
                                archive_max=0.0, refit_identity_max=None,
                                crashes=[])["passed"] is False
    assert common.gate_decision(criterion="archive", tolerance=1e-11, archive_max=0.0,
                                refit_identity_max=None, crashes=["x"]) == {
        "passed": False, "frozen_criterion_passed": False}


def _write_fit(fit_dir, stem, *, seed, shuffle, selected, test, validation=None,
               candidates=True, error=None):
    fit_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"test": np.asarray(test, dtype=float)}
    if validation is not None:
        arrays["validation"] = np.asarray(validation, dtype=float)
    if candidates:
        for role in ("validation", "test"):
            for name in common.CANDIDATES:
                arrays[f"{role}__{name}"] = arrays.get(role, arrays["test"])
    np.savez(fit_dir / f"{stem}.npz", **arrays)
    (fit_dir / f"{stem}.json").write_text(json.dumps({
        "learner_seed": seed, "shuffle_seed": shuffle, "selected_model": selected,
        "flagged_non_converged": False, "error": error}))


def test_reexport_check_on_synthetic_files(tmp_path):
    import gzip

    key, nc_key = "shipped-s101-n640", "nc-s101-n640"
    out = tmp_path / "out"
    seedrep = tmp_path / "seedrep" / "fits"
    nc_results = tmp_path / "nc"
    order = {"validation_item_ids": ["v0", "v1"], "test_item_ids": ["t0", "t1", "t2"]}
    for cache in (out / "cache", tmp_path / "seedrep" / "cache"):
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{key}.order.json").write_text(json.dumps(order))
    nc_order = {"validation_item_ids": ["a"], "test_item_ids": ["n0", "n1"]}
    (out / "cache" / f"{nc_key}.order.json").write_text(json.dumps(nc_order))

    test, val = [0.1, 0.2, 0.3], [0.4, 0.5]
    for arm, shuffle in (("F", None), ("C", None), ("P", 1)):
        _write_fit(seedrep, f"{key}__k01__{arm}", seed=1, shuffle=shuffle,
                   selected="mlp", test=test, validation=val, candidates=False)
    for arm, shuffle in (("F", None), ("C", None), ("P", 1234)):
        _write_fit(seedrep, f"{key}__orig__{arm}", seed=101, shuffle=shuffle,
                   selected="random_forest", test=test, validation=val,
                   candidates=False)
    # Re-exports: F exact, C one ulp off, P missing; orig F, C in gate/fits, orig P
    # with the wrong shuffle seed.
    _write_fit(out / "fits", f"{key}__R0__k01__F", seed=1, shuffle=None,
               selected="mlp", test=test, validation=val)
    _write_fit(out / "fits", f"{key}__R0__k01__C", seed=1, shuffle=None,
               selected="mlp", test=[0.1, np.nextafter(0.2, 1.0), 0.3], validation=val)
    for arm in ("F", "C"):
        _write_fit(out / "gate" / "fits", f"{key}__R0__orig__{arm}", seed=101,
                   shuffle=None, selected="random_forest", test=test, validation=val)
    _write_fit(out / "fits", f"{key}__R0__orig__P", seed=101, shuffle=101,
               selected="random_forest", test=test, validation=val)

    seed_dir = nc_results / "s101"
    seed_dir.mkdir(parents=True)
    by_seed = {str(k): {arm: {"n0": 0.01 * k, "n1": -0.01 * k} for arm in "FCP"}
               for k in (1, 101)}
    with gzip.open(seed_dir / "test_predictions.json.gz", "wt") as handle:
        json.dump({"test_item_ids": ["n0", "n1"], "seeds": by_seed}, handle)
    (seed_dir / "summary.json").write_text(json.dumps({
        "per_seed_results": [{"seed": 1, "selected_candidates": dict.fromkeys("FCP", "mlp")}],
        "anchor_fit": {"seed": 101, "selected_candidates": dict.fromkeys("FCP", "mlp")}}))
    for k in (1, 101):
        for arm in "FCP":
            selected = "random_forest" if (k, arm) == (101, "C") else "mlp"
            _write_fit(out / "fits", f"{nc_key}__NC-R0__k{k:02d}__{arm}", seed=k,
                       shuffle=k if arm == "P" else None, selected=selected,
                       test=[0.01 * k, -0.01 * k])

    ladder._appendix_n.cache_clear()
    report = ladder.check_reexports(out, [key], [nc_key], [1], seedrep, nc_results)
    status = {entry["stem"]: entry["status"] for entry in report["entries"]}
    assert report["row_order"]["all_match"] is True
    assert status[f"{key}__R0__k01__F"] == "exact"
    assert status[f"{key}__R0__k01__C"] == "differs"
    assert status[f"{key}__R0__k01__P"] == "missing"
    assert status[f"{key}__R0__orig__F"] == status[f"{key}__R0__orig__C"] == "exact"
    assert status[f"{key}__R0__orig__P"] == "differs"          # shuffle seed differs
    assert status[f"{nc_key}__NC-R0__k01__P"] == "exact"
    assert status[f"{nc_key}__NC-R0__k101__F"] == "exact"      # the anchor
    assert status[f"{nc_key}__NC-R0__k101__C"] == "differs"    # selection differs
    c_entry = next(e for e in report["entries"] if e["stem"] == f"{key}__R0__k01__C")
    assert 0.0 < c_entry["roles"]["test"]["max_abs_diff"] < 1e-15
    assert report["any_differs"] is True
    assert report["r0_reuses_appendix_m4"] is False
    assert report["groups"][ladder.M4_GROUP]["expected"] == 6

    # Row order is checked before any fit.
    (out / "cache" / f"{key}.order.json").write_text(json.dumps(
        {**order, "test_item_ids": ["t1", "t0", "t2"]}))
    assert ladder.reexport_preconditions(out, [key], [], seedrep, None)[
        "all_match"] is False


def _synthetic_qaoa_data(n_qubits: int = 4) -> dict:
    """Tiny QAOA-shaped rows for exercising the writers; not an experiment."""
    from qemscore.runner.run import _prediction_items

    rng = np.random.default_rng(0)
    pairs = list(combinations(range(n_qubits), 2))
    roles = {"train": 24, "validation": 8, "test": 8}
    data: dict = {"seed": 7, "dataset_hash": "synthetic", "n_qubits": n_qubits}
    for role, count in roles.items():
        rows = []
        for instance in range(count):
            edges = [list(pair) for pair in pairs if rng.random() < 0.5] or [[0, 1]]
            ideal = float(np.tanh(0.3 * len(edges) - 1.0 + 0.1 * rng.standard_normal()))
            circuit = f"circuit-{role}-{instance}"
            for severity, scale in (("L1", 0.8), ("L3", 0.3)):
                rows.append({
                    "item_id": f"{circuit}-{severity}", "circuit_id": circuit,
                    "measurement_group": f"{circuit}-{severity}", "split": role,
                    "family": "qaoa", "stratum": "continuous_regression",
                    "instance": instance, "n_qubits": n_qubits,
                    "pauli_label": "IZZI", "obs_locality": 2,
                    "observable": "zz_mid", "noise_family": "depolarizing_readout",
                    "severity": severity, "shots": 2048,
                    "noisy_expectation": scale * ideal + 0.01 * float(rng.standard_normal()),
                    "ideal_expectation": ideal, "p": 1, "graph_class": "erdos_renyi",
                    "edges": edges, "gammas": [0.4], "betas": [0.3],
                    "two_qubit_gates": 2 * len(edges), "transpiled_depth": 5 + len(edges),
                })
        data[role] = rows
    data["prediction_rows"] = {role: _prediction_items(data[role])
                               for role in ("validation", "test")}
    return data


def test_fit_writers_on_synthetic_rows(tmp_path):
    data = _synthetic_qaoa_data()
    builder = qaoa.QAOABuilder("B-complete", 4)
    base = {"part": "B", "key": "synthetic", "rung": "B-complete",
            "fit_dir": str(tmp_path), "learner_seed": 3, "shuffle_seed": None}

    info = common.run_liao_fit({**base, "arm": "C", "fit_arm": "C", "stem": "liao"},
                               data, builder, builder.names)
    assert info["error"] is None
    meta = json.loads((tmp_path / "liao.json").read_text())
    arrays = np.load(tmp_path / "liao.npz")
    assert set(arrays.files) == {
        "validation", "test", "validation__random_forest", "test__random_forest",
        "validation__mlp", "test__mlp"}
    assert meta["selected_equals_candidate_predictions"] is True
    assert meta["dropped_features"] == ["noisy_expectation"]
    assert meta["n_features"] == len(FEATURES) + 6
    assert set(meta["candidate_test_family_mae"]) == {"random_forest", "mlp"}
    assert np.array_equal(arrays["test"], arrays[f"test__{meta['selected_model']}"])

    info = common.run_liao_fit({**base, "arm": "P", "fit_arm": "P", "stem": "shuffle",
                                "shuffle_seed": 3}, data, builder, builder.names)
    assert info["error"] is None

    info = common.run_affine_fit({**base, "arm": "A", "stem": "affine"},
                                 data, builder, builder.names)
    assert info["error"] is None
    affine = json.loads((tmp_path / "affine.json").read_text())
    assert "noisy_expectation" not in affine["feature_names"]
    assert affine["selection"]["best_alpha"] in (0.1, 1.0, 10.0)

    info = qaoa.run_gbt_fit({**base, "arm": "GBT", "stem": "gbt"}, data)
    assert info["error"] is None
    gbt = json.loads((tmp_path / "gbt.json").read_text())
    assert len(gbt["validation_scores"]) == len(qaoa.GBT_GRID)
    assert "noisy_expectation" not in gbt["feature_names"]
    assert set(np.load(tmp_path / "gbt.npz").files) == {"validation", "test"}
