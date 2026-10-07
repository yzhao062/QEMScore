"""Tests for pooled random-effects and absolute-scale margin analysis.

This module tests DerSimonian-Laird variance estimation, Hartung-Knapp-Sidik-Jonkman
confidence intervals, margin classification at boundaries, and synthetic JSON parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy.stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path[:1]:
    sys.path.insert(0, str(REPO / "tools"))
if str(REPO) not in sys.path[:2]:
    sys.path.insert(1, str(REPO))

import pooled_absolute as pa  # noqa: E402


def test_dl_and_hksj_hand_computed() -> None:
    """Verify DL and HKSJ formulas against an independently computed reference to 1e-12."""
    y = np.array([0.012, 0.015, 0.008, 0.020, 0.011, 0.014], dtype=float)
    se = np.array([0.002, 0.003, 0.0015, 0.004, 0.0025, 0.002], dtype=float)

    # Independent reference calculation inside the test
    k = len(y)
    df = k - 1
    v = se ** 2
    w_fe = 1.0 / v
    w_fe_sum = float(np.sum(w_fe))
    ref_theta_fe = float(np.sum(w_fe * y) / w_fe_sum)

    ref_q = float(np.sum(w_fe * (y - ref_theta_fe) ** 2))
    ref_i2 = float((ref_q - df) / ref_q) if ref_q > df else 0.0

    c_val = w_fe_sum - float(np.sum(w_fe ** 2)) / w_fe_sum
    ref_tau2 = float((ref_q - df) / c_val) if ref_q > df and c_val > 0.0 else 0.0

    w_re = 1.0 / (v + ref_tau2)
    w_re_sum = float(np.sum(w_re))
    ref_theta_re = float(np.sum(w_re * y) / w_re_sum)

    ref_var_hksj = float(np.sum(w_re * (y - ref_theta_re) ** 2) / (df * w_re_sum))
    ref_se_hksj = float(np.sqrt(ref_var_hksj))

    t_crit = float(scipy.stats.t.ppf(0.975, df=df))
    ref_ci_lower = ref_theta_re - t_crit * ref_se_hksj
    ref_ci_upper = ref_theta_re + t_crit * ref_se_hksj

    result = pa.hksj_meta_analysis(y, se)

    assert abs(result["tau2"] - ref_tau2) < 1e-12
    assert abs(result["theta_fe"] - ref_theta_fe) < 1e-12
    assert abs(result["theta_re"] - ref_theta_re) < 1e-12
    assert abs(result["q_stat"] - ref_q) < 1e-12
    assert abs(result["i2"] - ref_i2) < 1e-12
    assert abs(result["var_hksj"] - ref_var_hksj) < 1e-12
    assert abs(result["se_hksj"] - ref_se_hksj) < 1e-12
    assert abs(result["ci_lower"] - ref_ci_lower) < 1e-12
    assert abs(result["ci_upper"] - ref_ci_upper) < 1e-12


def test_tau2_zero_when_equal() -> None:
    """Check that tau^2 is zero and interval collapses when all estimates are equal."""
    y = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005], dtype=float)
    se = np.array([0.001, 0.002, 0.0015, 0.003, 0.0025, 0.001], dtype=float)

    result = pa.hksj_meta_analysis(y, se)

    assert result["tau2"] == 0.0
    assert result["q_stat"] == 0.0
    assert result["i2"] == 0.0
    assert abs(result["theta_re"] - 0.005) < 1e-12
    assert result["var_hksj"] == 0.0
    assert result["se_hksj"] == 0.0
    assert abs(result["ci_lower"] - 0.005) < 1e-12
    assert abs(result["ci_upper"] - 0.005) < 1e-12


def test_i2_definition() -> None:
    """Verify I^2 definition across homogeneous and heterogeneous settings."""
    # Setting with Q <= df
    y_homo = np.array([0.001, 0.0011, 0.0009, 0.00105, 0.00095, 0.001], dtype=float)
    se_homo = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float)
    res_homo = pa.hksj_meta_analysis(y_homo, se_homo)
    assert res_homo["q_stat"] <= res_homo["df"]
    assert res_homo["i2"] == 0.0
    assert res_homo["tau2"] == 0.0

    # Setting with Q > df
    y_hetero = np.array([0.0, 0.02, 0.04, -0.02, 0.03, -0.01], dtype=float)
    se_hetero = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002], dtype=float)
    res_hetero = pa.hksj_meta_analysis(y_hetero, se_hetero)
    assert res_hetero["q_stat"] > res_hetero["df"]
    expected_i2 = (res_hetero["q_stat"] - res_hetero["df"]) / res_hetero["q_stat"]
    assert abs(res_hetero["i2"] - expected_i2) < 1e-12


def test_margin_classification_boundaries() -> None:
    """Check classification against practical margin delta at all boundary conditions."""
    delta = 0.01

    # Practically relevant gain: lower limit strictly above delta
    assert pa.classify_practical_margin(0.010001, 0.02, delta) == "practically relevant gain"
    assert pa.classify_practical_margin(0.015, 0.03, delta) == "practically relevant gain"

    # Boundary at +delta: lower limit equals delta is not strictly above
    assert pa.classify_practical_margin(0.01, 0.02, delta) == "undetermined"

    # Practically relevant loss: upper limit strictly below -delta
    assert pa.classify_practical_margin(-0.02, -0.010001, delta) == "practically relevant loss"
    assert pa.classify_practical_margin(-0.03, -0.015, delta) == "practically relevant loss"

    # Boundary at -delta: upper limit equals -delta is not strictly below
    assert pa.classify_practical_margin(-0.02, -0.01, delta) == "undetermined"

    # Practically negligible: interval strictly inside (-delta, delta)
    assert pa.classify_practical_margin(-0.009, 0.009, delta) == "practically negligible"
    assert pa.classify_practical_margin(0.001, 0.008, delta) == "practically negligible"
    assert pa.classify_practical_margin(-0.008, -0.001, delta) == "practically negligible"

    # Touching boundary of (-delta, delta) is not inside
    assert pa.classify_practical_margin(-0.01, 0.005, delta) == "undetermined"
    assert pa.classify_practical_margin(-0.005, 0.01, delta) == "undetermined"
    assert pa.classify_practical_margin(-0.01, 0.01, delta) == "undetermined"

    # Overlapping or spanning delta boundaries
    assert pa.classify_practical_margin(-0.005, 0.015, delta) == "undetermined"
    assert pa.classify_practical_margin(-0.015, 0.005, delta) == "undetermined"
    assert pa.classify_practical_margin(-0.02, 0.02, delta) == "undetermined"

    # Non-positive delta must fail loudly
    with pytest.raises(ValueError, match="positive"):
        pa.classify_practical_margin(0.01, 0.02, 0.0)


def test_pooled_label_classification() -> None:
    """Verify three-way pooled label classification rule."""
    assert pa.classify_pooled_label(0.0001, 0.002) == "F beats C"
    assert pa.classify_pooled_label(-0.002, -0.0001) == "C beats F"
    assert pa.classify_pooled_label(-0.001, 0.001) == "not distinguished"
    assert pa.classify_pooled_label(0.0, 0.001) == "not distinguished"
    assert pa.classify_pooled_label(-0.001, 0.0) == "not distinguished"


def test_panel_unanimity_statements() -> None:
    """Verify panel unanimity classification across three seeds."""
    assert pa.classify_panel_unanimity(["F beats C", "F beats C", "F beats C"]) == "F beats C"
    assert pa.classify_panel_unanimity(["C beats F", "C beats F", "C beats F"]) == "C beats F"
    assert pa.classify_panel_unanimity(
        ["not distinguished", "not distinguished", "not distinguished"]
    ) == "not distinguished"
    assert pa.classify_panel_unanimity(["F beats C", "not distinguished", "F beats C"]) == "mixed"
    assert pa.classify_panel_unanimity(["F beats C", "C beats F", "F beats C"]) == "mixed"
    assert pa.classify_panel_unanimity([]) == "incomplete"


def _make_synthetic_analysis_json(
    seeds: Sequence[int],
    families: Sequence[str] = ("tfi", "heisenberg"),
    rungs: Sequence[str] = ("R0", "N1", "N2", "R5"),
) -> dict:
    """Create a minimal synthetic analysis-a JSON object for testing."""
    rows = {}
    for seed in seeds:
        for fam in families:
            row_key = f"shipped-s{seed}-n640/{fam}"
            rungs_dict = {}
            for rung in rungs:
                rungs_dict[rung] = {
                    "rung": rung,
                    "means": {
                        "C": {"point": 0.015, "draw_sd": 0.001},
                        "F": {"point": 0.012, "draw_sd": 0.001},
                    },
                    "D": {
                        "point": 0.003,
                        "draw_sd": 0.0012,
                        "interval": {"lower": 0.0008, "upper": 0.0052},
                    },
                    "classification": {
                        "label": "measurement_adds",
                    },
                }
            rows[row_key] = {
                "key": f"shipped-s{seed}-n640",
                "dataset_seed": seed,
                "family": fam,
                "rungs": rungs_dict,
            }
    return {
        "schema": "descriptor-information-analysis-a-v1",
        "parts": {"A": {"rows": rows}},
    }


def test_input_collection_synthetic_json(tmp_path: Path) -> None:
    """Verify input collection and absent rung detection on synthetic analysis files."""
    first_json = _make_synthetic_analysis_json(
        pa.FIRST_SEEDS, rungs=("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
    )
    new_json = _make_synthetic_analysis_json(
        pa.NEW_SEEDS, rungs=("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
    )

    p_first = tmp_path / "orig_first.json"
    p_new = tmp_path / "orig_new.json"
    p_first.write_text(json.dumps(first_json), encoding="utf-8")
    p_new.write_text(json.dumps(new_json), encoding="utf-8")

    per_seed, rung_recs, absent = pa.collect_fit_set_data(
        "original 2,048", p_first, p_new
    )

    # 8 rungs for tfi, 8 rungs for heisenberg, 6 seeds each = 96 records
    assert len(per_seed) == 96
    assert len(rung_recs["tfi"]) == 8
    assert len(rung_recs["heisenberg"]) == 8
    assert absent["tfi"] == []
    assert absent["heisenberg"] == []

    # Now test exact fit set with absent rungs
    exact_first_json = _make_synthetic_analysis_json(
        pa.FIRST_SEEDS, rungs=("R0", "N1", "N2", "R5")
    )
    exact_new_json = _make_synthetic_analysis_json(
        pa.NEW_SEEDS, rungs=("R0", "N1", "N2", "R5")
    )
    p_exact_first = tmp_path / "exact_first.json"
    p_exact_new = tmp_path / "exact_new.json"
    p_exact_first.write_text(json.dumps(exact_first_json), encoding="utf-8")
    p_exact_new.write_text(json.dumps(exact_new_json), encoding="utf-8")

    per_seed_ex, rung_recs_ex, absent_ex = pa.collect_fit_set_data(
        "strong exact", p_exact_first, p_exact_new
    )
    # 4 rungs * 2 families * 6 seeds = 48 records
    assert len(per_seed_ex) == 48
    assert len(rung_recs_ex["tfi"]) == 4
    assert len(rung_recs_ex["heisenberg"]) == 4
    assert set(absent_ex["tfi"]) == {"N3", "N4", "R3-TFI", "R4"}
    assert set(absent_ex["heisenberg"]) == {"N3", "N4", "R3-Heis", "R4"}


def test_missing_seed_fails_loudly(tmp_path: Path) -> None:
    """Check that missing required seeds triggers a loud ValueError."""
    # Seed 211 missing from first panel
    incomplete_first = _make_synthetic_analysis_json([101, 307])
    new_json = _make_synthetic_analysis_json(pa.NEW_SEEDS)

    p_first = tmp_path / "incomp_first.json"
    p_new = tmp_path / "new.json"
    p_first.write_text(json.dumps(incomplete_first), encoding="utf-8")
    p_new.write_text(json.dumps(new_json), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing seed 211"):
        pa.collect_fit_set_data("original 2,048", p_first, p_new)


def test_incomplete_rung_fails_loudly(tmp_path: Path) -> None:
    """Check that a rung present in some seeds but missing in others fails loudly."""
    first_json = _make_synthetic_analysis_json(pa.FIRST_SEEDS)
    new_json = _make_synthetic_analysis_json(pa.NEW_SEEDS)

    # Delete rung R5 from seed 401
    del new_json["parts"]["A"]["rows"]["shipped-s401-n640/tfi"]["rungs"]["R5"]

    p_first = tmp_path / "first.json"
    p_new = tmp_path / "incomp_rung_new.json"
    p_first.write_text(json.dumps(first_json), encoding="utf-8")
    p_new.write_text(json.dumps(new_json), encoding="utf-8")

    with pytest.raises(ValueError, match="missing seeds"):
        pa.collect_fit_set_data("original 2,048", p_first, p_new)


def test_synthetic_pipeline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Run full pipeline on synthetic inputs with CLI overrides and check outputs."""
    # Synthetic inputs are not mapped fit sets; the policy check has its own test.
    monkeypatch.setattr(pa, "check_single_strength_policy", lambda paths: None)
    rule_file = tmp_path / "rule.md"
    rule_file.write_text("# Test Frozen Rule\n\nRule placeholder.\n", encoding="utf-8")

    analysis_files = {}
    for fit_set in pa.FIT_SETS:
        rungs = (
            ("R0", "N1", "N2", "R5")
            if fit_set == "strong exact"
            else ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
        )
        f1 = _make_synthetic_analysis_json(pa.FIRST_SEEDS, rungs=rungs)
        f2 = _make_synthetic_analysis_json(pa.NEW_SEEDS, rungs=rungs)
        p1 = tmp_path / f"{fit_set.replace(' ', '_')}_first.json"
        p2 = tmp_path / f"{fit_set.replace(' ', '_')}_new.json"
        p1.write_text(json.dumps(f1), encoding="utf-8")
        p2.write_text(json.dumps(f2), encoding="utf-8")
        analysis_files[fit_set] = {"first": p1, "new": p2}

    out_json = tmp_path / "pooled.json"
    out_md = tmp_path / "pooled.md"

    argv = [
        "--frozen-rule", str(rule_file),
        "--out", str(out_json),
        "--out-md", str(out_md),
        "--orig-2048-first", str(analysis_files["original 2,048"]["first"]),
        "--orig-2048-new", str(analysis_files["original 2,048"]["new"]),
        "--strong-2048-first", str(analysis_files["strong 2,048"]["first"]),
        "--strong-2048-new", str(analysis_files["strong 2,048"]["new"]),
        "--strong-exact-first", str(analysis_files["strong exact"]["first"]),
        "--strong-exact-new", str(analysis_files["strong exact"]["new"]),
        "--delta-tfi", "0.005",
        "--delta-heisenberg", "0.006",
        "--shot-noise-tfi", "0.018",
        "--shot-noise-heisenberg", "0.019",
        "--quiet",
    ]

    exit_code = pa.main(argv)
    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema"] == "descriptor-information-pooled-absolute-v1"
    assert data["frozen_rule"] == pa.FROZEN_RULE_PATH
    assert "rule_file_sha256" in data
    assert "script_sha256" in data
    assert "inputs" in data
    assert len(data["inputs"]) == 6
    assert data["margins"]["delta"]["tfi"] == 0.005
    assert data["margins"]["delta"]["heisenberg"] == 0.006
    assert len(data["pooled_table"]) == (8 + 8) + (8 + 8) + (4 + 4)
    assert "Absent Rungs" in data["markdown_table"]
    assert "strong exact / tfi: N3, N4, R3-TFI, R4" in data["markdown_table"]

    # Verify each pooled row records per-seed labels for both panels
    for row in data["pooled_table"]:
        assert "first_panel_labels" in row
        assert "new_panel_labels" in row
        assert set(row["first_panel_labels"].keys()) == {"101", "211", "307"}
        assert set(row["new_panel_labels"].keys()) == {"401", "503", "607"}
        for lbl in row["first_panel_labels"].values():
            assert lbl in ("F beats C", "C beats F", "not distinguished")
        for lbl in row["new_panel_labels"].values():
            assert lbl in ("F beats C", "C beats F", "not distinguished")


def test_synthetic_rcal_and_shot_noise() -> None:
    """Verify Rcal margin computation and shot-noise scale on synthetic items."""
    datasets = {}
    for seed in pa.ALL_SEEDS:
        val_items = []
        test_items = []
        rng = np.random.default_rng(seed)
        for fam in pa.FAMILIES:
            for sev, obs in pa.mf.CELLS_DEF:
                for _ in range(15):
                    y_v = float(rng.uniform(-1, 1))
                    r_v = 0.8 * y_v + 0.05 + float(rng.normal(0, 0.02))
                    val_items.append({
                        "family": fam, "severity": sev, "observable": obs,
                        "ideal_expectation": y_v, "noisy_expectation": r_v,
                    })
                for _ in range(10):
                    y_t = float(rng.uniform(-1, 1))
                    r_t = 0.8 * y_t + 0.05 + float(rng.normal(0, 0.02))
                    test_items.append({
                        "family": fam, "severity": sev, "observable": obs,
                        "ideal_expectation": y_t, "noisy_expectation": r_t,
                    })
        datasets[seed] = {"val": val_items, "test": test_items}

    margin_info = pa.compute_rcal_margins_from_datasets(datasets, pa.ALL_SEEDS)
    assert "deltas" in margin_info
    assert "tfi" in margin_info["deltas"]
    assert "heisenberg" in margin_info["deltas"]
    assert margin_info["deltas"]["tfi"] > 0.0
    assert margin_info["deltas"]["heisenberg"] > 0.0
    expected_delta_tfi = 0.1 * margin_info["mean_errors"]["tfi"]
    assert abs(margin_info["deltas"]["tfi"] - expected_delta_tfi) < 1e-12

    sn_scale = pa.compute_shot_noise_scale_from_datasets(
        datasets, "tfi", shots=2048, split="test"
    )
    assert sn_scale > 0.0


def test_parser_contract() -> None:
    """Verify CLI parser defaults and required arguments match the frozen rule."""
    parser = pa.build_cli_parser()

    # Without --frozen-rule, parsing must fail
    with pytest.raises(SystemExit):
        parser.parse_args([])

    rule_str = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
    args = parser.parse_args(["--frozen-rule", rule_str])

    # Check required frozen rule path
    assert args.frozen_rule == Path(rule_str)

    # Check default output filenames (with hyphens)
    assert args.out == Path("artifacts/descriptor-information/round9/pooled-absolute.json")
    assert args.out_md == Path("artifacts/descriptor-information/round9/pooled-absolute.md")

    # Check default analysis input paths
    assert args.orig_2048_first == Path("artifacts/descriptor-information/strength-indicator/analysis-a.json")
    assert args.orig_2048_new == Path(
        "artifacts/descriptor-information/round8/fresh/orig/2048/analysis-a.json"
    )
    assert args.strong_2048_first == Path(
        "artifacts/descriptor-information/strong-learners/analysis-a.json"
    )
    assert args.strong_2048_new == Path(
        "artifacts/descriptor-information/round8/fresh/strong/2048/analysis-a.json"
    )
    assert args.strong_exact_first == Path(
        "artifacts/descriptor-information/round8/follow/B/exact/analysis-a.json"
    )
    assert args.strong_exact_new == Path(
        "artifacts/descriptor-information/round8/fresh/strong/exact/analysis-a.json"
    )

    # Check default dataset roots
    assert args.rcal_data_first == Path(
        "/Users/yzhao062/qemscore-r8/assets/shot-sweep-v1/levels/shots-2048"
    )
    assert args.rcal_data_new == Path(
        "/Users/yzhao062/qemscore-r8/fresh/levels/shots-2048"
    )
    assert args.exact_data_first == Path(
        "/Users/yzhao062/qemscore-r8/assets/shot-sweep-v1/levels/shots-exact"
    )
    assert args.exact_data_new == Path(
        "/Users/yzhao062/qemscore-r8/fresh/levels/shots-exact"
    )




def test_single_strength_policy_rejects_the_original_ladder():
    """The default inputs share one strength policy; the original ladder breaks it."""
    from tools.pooled_absolute import DEFAULT_ANALYSIS_PATHS, check_single_strength_policy

    check_single_strength_policy(DEFAULT_ANALYSIS_PATHS)
    mixed = {k: dict(v) for k, v in DEFAULT_ANALYSIS_PATHS.items()}
    mixed["original 2,048"]["first"] = Path("artifacts/descriptor-information/analysis-a.json")
    with pytest.raises(ValueError, match="mix descriptor policies"):
        check_single_strength_policy(mixed)
