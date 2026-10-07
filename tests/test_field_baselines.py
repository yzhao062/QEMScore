"""Tests for the field reanalyses (Raw Baselines, Opposite Ladder Order, and Archived Fits).

Verifies:
1. Archived Q-LEAR per-fit predictions reproduce field-report.json to 1e-12 (Task C).
2. Synthetic scoring of the raw baseline arm R in Q-LEAR (Task A).
3. Two-group Shapley share identities and opposite ladder order in Q-LEAR (Task B):
   - Two-group identity: S_base + S_dpe = 1 whenever C - F is non-zero.
   - Single-order agreement: when the two orders agree, Shapley shares equal single-order shares.
   - Missing O2 fits fail clearly when --opposite is requested.
4. Synthetic scoring of the raw baseline arm R in QRAFT (Task D).
5. Default-output invariance for both Q-LEAR and QRAFT scorers when new options are off.
"""

import csv
import json
import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unittest.mock import patch
from reanalysis.scripts.qlear import score_field_half, fit_field_half
from reanalysis.scripts.qlear.check_archived_fits import verify_checksums, compare_structures
from reanalysis.scripts.qraft import score_qraft_panel


# ==============================================================================
# Helper functions to build synthetic datasets and fits
# ==============================================================================

APPLICATIONS = score_field_half.EXPECTED_APPLICATIONS
BACKENDS = score_field_half.EXPECTED_BACKENDS
SEEDS = tuple(range(10))


def create_synthetic_qlear_release(release_dir):
    """Create minimal synthetic evaluation CSV files matching Q-LEAR release structure."""
    for panel_folder in ("real_circuits_hardware", "real_circuits"):
        folder_path = os.path.join(release_dir, panel_folder)
        os.makedirs(folder_path, exist_ok=True)
        for app in APPLICATIONS:
            for b in BACKENDS:
                file_name = f"{app}_{b}.csv"
                file_path = os.path.join(folder_path, file_name)
                # 4 states per circuit
                rows = [
                    {"target": 25.0, "observed_prob_50": 20.0},
                    {"target": 25.0, "observed_prob_50": 30.0},
                    {"target": 25.0, "observed_prob_50": 22.0},
                    {"target": 25.0, "observed_prob_50": 28.0},
                ]
                with open(file_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["target", "observed_prob_50"])
                    writer.writeheader()
                    writer.writerows(rows)


def create_synthetic_qlear_fits(fits_dir, include_o2=False, agree_orders=False):
    """Create synthetic prediction files for testing scoring logic."""
    os.makedirs(fits_dir, exist_ok=True)
    arms = ["A", "C", "C4", "O", "F"]
    if include_o2:
        arms.append("O2")

    panels = ["hardware", "simulator"]
    for arm in arms:
        seeds = (0,) if arm == "A" else SEEDS
        for seed in seeds:
            rec = {
                "arm": arm,
                "seed": seed,
                "kind": "affine" if arm == "A" else "network",
                "columns": list(score_field_half.EXPECTED_O2_COLUMNS) if arm == "O2" else ["mock_col"],
                "epochs": 10 if arm != "A" else None,
                "validation_mse": 0.01,
                "panels": {},
            }
            if arm == "O2":
                rec["release_identity"] = {"release_dir": "qlear_release"}
            for panel in panels:
                block = {}
                for app in APPLICATIONS:
                    for b in BACKENDS:
                        name = f"{app}_{b}"
                        # Fixed target probabilities
                        t_vals = [25.0, 25.0, 25.0, 25.0]
                        # Different predictions per arm to test gaps and shares:
                        # Say true distribution is [25, 25, 25, 25].
                        # C has error level 0.30
                        # F has error level 0.10
                        # O has error level 0.15
                        # If agree_orders: C - O = O2 - F and O - F = C - O2
                        # Let C err = 0.30, F err = 0.10. Total gap = 0.20.
                        # C - O = 0.15, so O err = 0.15. O - F = 0.05.
                        # To agree: O2 - F = 0.15 => O2 err = 0.25.
                        # Then C - O2 = 0.30 - 0.25 = 0.05. Matches!
                        if arm == "C" or arm == "C4" or arm == "A":
                            p_vals = [55.0, 15.0, 15.0, 15.0]  # high error
                        elif arm == "F":
                            p_vals = [25.0, 25.0, 25.0, 25.0]  # zero / low error
                        elif arm == "O":
                            p_vals = [35.0, 25.0, 20.0, 20.0]  # intermediate error
                        elif arm == "O2":
                            if agree_orders:
                                x = 43.11658421013881
                                p_vals = [x, (100.0 - x) / 3.0, (100.0 - x) / 3.0, (100.0 - x) / 3.0]
                            else:
                                p_vals = [40.0, 20.0, 20.0, 20.0]
                        else:
                            p_vals = [25.0, 25.0, 25.0, 25.0]

                        block[name] = {
                            "application": app,
                            "backend": b,
                            "target": t_vals,
                            "prediction": p_vals,
                        }
                rec["panels"][panel] = block

            target_path = os.path.join(fits_dir, f"pred-{arm}-seed{seed}.json")
            with open(target_path, "w") as fh:
                json.dump(rec, fh)


def create_synthetic_qraft_data(data_dir):
    """Create synthetic inputData.csv and outputData.csv for QRAFT tests."""
    os.makedirs(data_dir, exist_ok=True)
    fieldnames_in = (
        "ComputerID", "CircuitWidth", "CircuitDepth", "CircuitNumU1Gates",
        "CircuitNumU2Gates", "CircuitNumU3Gates", "CircuitNumCXGates",
        "TotalUpDnErr25", "TotalUpDnErr50", "TotalUpDnErr75",
        "StateHammingWeight", "StateUpProb25", "StateUpProb50", "StateUpProb75",
        "StateUpDnErr25", "StateUpDnErr50", "StateUpDnErr75", "StateRealProb",
    )
    fieldnames_out = fieldnames_in + ("StatePredPrT1", "StatePredPrT2", "StatePredPrT3")

    rows_in = []
    # 5 machines, 20 rows each = 100 rows
    for m in range(5):
        for i in range(20):
            row = {k: "1.0" for k in fieldnames_in}
            row["ComputerID"] = str(m)
            row["StateRealProb"] = "25.0"
            row["StateUpProb50"] = "20.0"
            rows_in.append(row)

    rows_out = []
    for r in rows_in:
        r_out = dict(r)
        r_out["StatePredPrT1"] = "24.0"
        r_out["StatePredPrT2"] = "23.0"
        r_out["StatePredPrT3"] = "15.0"
        rows_out.append(r_out)

    os.makedirs(os.path.join(data_dir, "data_processed"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "data_trained"), exist_ok=True)

    in_path = os.path.join(data_dir, "data_processed", "inputData.csv")
    with open(in_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_in)
        writer.writeheader()
        writer.writerows(rows_in)

    out_path = os.path.join(data_dir, "data_trained", "outputData.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(rows_out)


# ==============================================================================
# Task C: Archived Fits Reproduction Test
# ==============================================================================

def test_archived_fits_reproduce_field_report(tmp_path):
    """Task C: Verify that archived fits match SHA-256 and reproduce field-report.json."""
    fits_dir = os.path.join(REPO_ROOT, "reanalysis", "outputs", "qlear", "fits")
    checksum_file = os.path.join(fits_dir, "SHA256SUMS")
    field_report_path = os.path.join(REPO_ROOT, "reanalysis", "outputs", "qlear", "field-report.json")

    assert os.path.isdir(fits_dir), f"Archived fits dir not found at {fits_dir}"
    assert os.path.isfile(checksum_file), f"SHA256SUMS not found at {checksum_file}"
    assert os.path.isfile(field_report_path), f"field-report.json not found at {field_report_path}"

    # Step 1: Checksums
    assert verify_checksums(fits_dir, checksum_file) is True

    # Step 2: Rescore
    rescored_out = str(tmp_path / "field-report.json")
    saved_argv = sys.argv
    sys.argv = ["score_field_half.py", fits_dir, rescored_out]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    # Step 3: Compare values to 1e-12
    with open(field_report_path, "r", encoding="utf-8") as fh:
        expected = json.load(fh)
    with open(rescored_out, "r", encoding="utf-8") as fh:
        actual = json.load(fh)

    # Enforce HEAD schema requirement: roster_status is complete
    for p in ("hardware", "simulator"):
        assert actual["panels"][p]["roster_status"] == "complete"
        if "roster_status" not in expected.get("panels", {}).get(p, {}):
            actual["panels"][p].pop("roster_status", None)

    ok, msg = compare_structures(expected, actual, tol=1e-12)
    assert ok is True, f"Reproduction mismatch: {msg}"


# ==============================================================================
# Task A: Synthetic Scoring of Raw Arm R in Q-LEAR
# ==============================================================================

def test_qlear_raw_arm_scoring(tmp_path):
    """Task A: Verify raw baseline arm R scoring and contrasts F-R, C-R with intervals."""
    release_dir = str(tmp_path / "qlear_release")
    fits_dir = str(tmp_path / "qlear_fits")
    out_dir = str(tmp_path / "out")
    rule_file = str(tmp_path / "rule.md")

    create_synthetic_qlear_release(release_dir)
    create_synthetic_qlear_fits(fits_dir, include_o2=False)
    with open(rule_file, "w") as fh:
        fh.write("# Frozen Rule\n")

    saved_argv = sys.argv
    sys.argv = [
        "score_field_half.py",
        fits_dir,
        out_dir,
        "--release", release_dir,
        "--raw",
        "--frozen-rule", rule_file,
    ]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir, "field-report.json"), "r") as fh:
        report = json.load(fh)

    assert "raw_contrasts" in report
    assert report["raw_contrasts"] == ["F-R", "C-R"]
    assert "frozen_rule" in report
    assert report["frozen_rule"] == "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
    assert "frozen_rule_sha256" in report
    assert "script_sha256" in report
    assert "input_sha256" in report

    for panel in ("hardware", "simulator"):
        p_block = report["panels"][panel]
        for seed_str, s_data in p_block["per_seed"].items():
            assert "R" in s_data["rung_errors"]
            gaps = s_data["gaps"]
            assert "F-R" in gaps
            assert "C-R" in gaps

            # Check point contrast identity: F-R = F - R
            f_err = s_data["rung_errors"]["F"]
            c_err = s_data["rung_errors"]["C"]
            r_err = s_data["rung_errors"]["R"]
            assert abs(gaps["F-R"]["estimate"] - (f_err - r_err)) < 1e-12
            assert abs(gaps["C-R"]["estimate"] - (c_err - r_err)) < 1e-12

            # Intervals should be populated
            assert gaps["F-R"]["lower"] <= gaps["F-R"]["upper"]
            assert gaps["C-R"]["lower"] <= gaps["C-R"]["upper"]


# ==============================================================================
# Task B: Opposite Ladder Order and Shapley Shares
# ==============================================================================

def test_qlear_opposite_ladder_and_shapley_shares(tmp_path):
    """Task B: Verify opposite ladder C -> O2 -> F, Shapley shares identity, and single-order agreement."""
    # Subtest 1: General case (arbitrary fits) -> Two-group identity: S_base + S_dpe = 1
    fits_dir1 = str(tmp_path / "fits_general")
    out_dir1 = str(tmp_path / "out_general")
    rule_file = str(tmp_path / "rule.md")
    with open(rule_file, "w") as fh:
        fh.write("# Frozen Rule\n")

    create_synthetic_qlear_fits(fits_dir1, include_o2=True, agree_orders=False)

    saved_argv = sys.argv
    sys.argv = [
        "score_field_half.py",
        fits_dir1,
        out_dir1,
        "--opposite",
        "--frozen-rule", rule_file,
    ]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir1, "field-report.json"), "r") as fh:
        report1 = json.load(fh)

    assert report1["opposite_ladder"] == ["C", "O2", "F"]
    assert report1["opposite_identity"] == "D = (C - O2) + (O2 - F)"

    for panel in ("hardware", "simulator"):
        p_block = report1["panels"][panel]
        for seed_str, s_data in p_block["per_seed"].items():
            assert "O2" in s_data["rung_errors"]
            gaps = s_data["gaps"]
            assert "C-O2" in gaps
            assert "O2-F" in gaps
            assert abs(s_data["opposite_identity_residual"]) < 1e-12

            shapley = s_data["shapley"]
            s_base = shapley["base_execution"]["share"]
            s_dpe = shapley["dpe"]["share"]
            v_base = shapley["base_execution"]["value"]
            v_dpe = shapley["dpe"]["value"]
            c_f = gaps["C-F"]["estimate"]

            # Two-group identity: phi_base + phi_dpe = C - F
            assert abs((v_base + v_dpe) - c_f) < 1e-12
            # Two-group share identity: S_base + S_dpe = 1
            assert abs((s_base + s_dpe) - 1.0) < 1e-12

    # Subtest 2: Single-order agreement: when C - O == O2 - F and O - F == C - O2
    fits_dir2 = str(tmp_path / "fits_agree")
    out_dir2 = str(tmp_path / "out_agree")
    create_synthetic_qlear_fits(fits_dir2, include_o2=True, agree_orders=True)

    sys.argv = [
        "score_field_half.py",
        fits_dir2,
        out_dir2,
        "--opposite",
        "--frozen-rule", rule_file,
    ]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir2, "field-report.json"), "r") as fh:
        report2 = json.load(fh)

    for panel in ("hardware", "simulator"):
        p_block = report2["panels"][panel]
        for seed_str, s_data in p_block["per_seed"].items():
            gaps = s_data["gaps"]
            c_f = gaps["C-F"]["estimate"]
            c_o = gaps["C-O"]["estimate"]
            o_f = gaps["O-F"]["estimate"]

            single_order_base = c_o / c_f
            single_order_dpe = o_f / c_f

            shapley = s_data["shapley"]
            s_base = shapley["base_execution"]["share"]
            s_dpe = shapley["dpe"]["share"]

            assert abs(s_base - single_order_base) < 1e-12
            assert abs(s_dpe - single_order_dpe) < 1e-12


def test_qlear_missing_o2_fails_clearly(tmp_path):
    """Task B: Verify that scorer fails clearly if O2 fits are absent and --opposite is requested."""
    fits_dir = str(tmp_path / "fits_no_o2")
    out_dir = str(tmp_path / "out")
    rule_file = str(tmp_path / "rule.md")
    with open(rule_file, "w") as fh:
        fh.write("# Frozen Rule\n")

    create_synthetic_qlear_fits(fits_dir, include_o2=False)

    saved_argv = sys.argv
    sys.argv = [
        "score_field_half.py",
        fits_dir,
        out_dir,
        "--opposite",
        "--frozen-rule", rule_file,
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "Opposite ladder requested via --opposite, but O2 fits are absent" in str(exc_info.value)
    finally:
        sys.argv = saved_argv


# ==============================================================================
# Task D: Synthetic Scoring of Raw Arm R in QRAFT
# ==============================================================================

def test_qraft_raw_arm_scoring(tmp_path):
    """Task D: Verify raw arm R scoring, contrasts, and bootstrap in QRAFT."""
    data_dir = str(tmp_path / "qraft_data")
    out_dir = str(tmp_path / "out")
    rule_file = str(tmp_path / "rule.md")

    create_synthetic_qraft_data(data_dir)
    with open(rule_file, "w") as fh:
        fh.write("# Frozen Rule\n")

    # Monkeypatch expected rows for synthetic test
    orig_exp_in = score_qraft_panel.EXPECTED_INPUT_ROWS
    orig_exp_out = score_qraft_panel.EXPECTED_OUTPUT_ROWS
    orig_boot_draws = score_qraft_panel.BOOTSTRAP_DRAWS
    score_qraft_panel.EXPECTED_INPUT_ROWS = 100
    score_qraft_panel.EXPECTED_OUTPUT_ROWS = 100
    score_qraft_panel.BOOTSTRAP_DRAWS = 100  # Fast test

    saved_argv = sys.argv
    sys.argv = [
        "score_qraft_panel.py",
        data_dir,
        out_dir,
        "--raw",
        "--frozen-rule", rule_file,
    ]
    try:
        score_qraft_panel.main()
    finally:
        sys.argv = saved_argv
        score_qraft_panel.EXPECTED_INPUT_ROWS = orig_exp_in
        score_qraft_panel.EXPECTED_OUTPUT_ROWS = orig_exp_out
        score_qraft_panel.BOOTSTRAP_DRAWS = orig_boot_draws

    with open(os.path.join(out_dir, "qraft-panel-report.json"), "r") as fh:
        report = json.load(fh)

    assert "frozen_rule" in report
    assert report["frozen_rule"] == "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
    assert "frozen_rule_sha256" in report
    assert "script_sha256" in report
    assert "input_sha256" in report

    # Panel R checks
    pR = report["panel_R_released"]
    assert "R" in pR
    assert pR["R"]["features"] == ["StateUpProb50"]
    assert "lower" in pR["R"]
    assert "upper" in pR["R"]
    assert pR["R"]["lower"] <= pR["R"]["upper"]
    for arm in ("Q1", "Q2", "Q3"):
        assert "lower" in pR[arm]
        assert "upper" in pR[arm]
        assert pR[arm]["lower"] <= pR[arm]["upper"]
    assert "contrasts_with_R" in pR
    for arm in ("Q1", "Q2", "Q3"):
        c_key = f"{arm}-R"
        assert c_key in pR["contrasts_with_R"]
        c_block = pR["contrasts_with_R"][c_key]
        assert "estimate" in c_block
        assert "lower" in c_block
        assert "upper" in c_block
        assert c_block["lower"] <= c_block["upper"]

    # Panel M checks
    pM = report["panel_M_matched"]
    assert "R" in pM
    assert pM["R"]["features"] == ["StateUpProb50"]
    assert "ten_seed_mean" in pM["R"]
    assert "lower" in pM["R"]
    assert "upper" in pM["R"]
    assert pM["R"]["lower"] <= pM["R"]["upper"]
    for arm in ("Q1", "Q2", "Q3"):
        assert "lower" in pM[arm]
        assert "upper" in pM[arm]
        assert pM[arm]["lower"] <= pM[arm]["upper"]
    assert "contrasts_with_R" in pM
    for arm in ("Q1", "Q2", "Q3"):
        c_key = f"{arm}-R"
        assert c_key in pM["contrasts_with_R"]
        c_block = pM["contrasts_with_R"][c_key]
        assert "ten_seed_mean" in c_block
        assert "lower" in c_block
        assert "upper" in c_block
        assert c_block["lower"] <= c_block["upper"]


# ==============================================================================
# Task E: Default-Output Invariance
# ==============================================================================

def test_qlear_default_output_invariance(tmp_path):
    """Task E: Verify Q-LEAR scorer default outputs do not contain any new fields when options are off."""
    fits_dir = str(tmp_path / "fits")
    out_dir = str(tmp_path / "out")
    create_synthetic_qlear_fits(fits_dir, include_o2=False)

    saved_argv = sys.argv
    sys.argv = ["score_field_half.py", fits_dir, out_dir]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir, "field-report.json"), "r") as fh:
        report = json.load(fh)

    # Ensure no new governed fields exist
    assert "opposite_ladder" not in report
    assert "raw_contrasts" not in report
    assert "frozen_rule" not in report
    assert "frozen_rule_sha256" not in report
    assert "script_sha256" not in report
    assert "input_sha256" not in report

    for panel in ("hardware", "simulator"):
        p_block = report["panels"][panel]
        assert p_block["roster_status"] == "complete"
        assert p_block["failures"] == []
        for seed_str, s_data in p_block["per_seed"].items():
            assert "R" not in s_data["rung_errors"]
            assert "O2" not in s_data["rung_errors"]
            assert "F-R" not in s_data["gaps"]
            assert "C-R" not in s_data["gaps"]
            assert "C-O2" not in s_data["gaps"]
            assert "O2-F" not in s_data["gaps"]
            assert "shapley" not in s_data
            assert "opposite_identity_residual" not in s_data


def test_qraft_default_output_invariance(tmp_path):
    """Task E: Verify QRAFT panel report default outputs do not contain any new fields when --raw is off."""
    data_dir = str(tmp_path / "qraft_data")
    out_dir = str(tmp_path / "out")
    create_synthetic_qraft_data(data_dir)

    orig_exp_in = score_qraft_panel.EXPECTED_INPUT_ROWS
    orig_exp_out = score_qraft_panel.EXPECTED_OUTPUT_ROWS
    score_qraft_panel.EXPECTED_INPUT_ROWS = 100
    score_qraft_panel.EXPECTED_OUTPUT_ROWS = 100

    saved_argv = sys.argv
    sys.argv = ["score_qraft_panel.py", data_dir, out_dir]
    try:
        score_qraft_panel.main()
    finally:
        sys.argv = saved_argv
        score_qraft_panel.EXPECTED_INPUT_ROWS = orig_exp_in
        score_qraft_panel.EXPECTED_OUTPUT_ROWS = orig_exp_out

    with open(os.path.join(out_dir, "qraft-panel-report.json"), "r") as fh:
        report = json.load(fh)

    assert "frozen_rule" not in report
    assert "frozen_rule_sha256" not in report
    assert "script_sha256" not in report
    assert "input_sha256" not in report
    assert report["intervals"] == "none; see PLAN-qraft-panel.md for why"

    pR = report["panel_R_released"]
    assert "R" not in pR
    assert "contrasts_with_R" not in pR
    for arm in ("Q1", "Q2", "Q3"):
        assert "lower" not in pR[arm]
        assert "upper" not in pR[arm]

    pM = report["panel_M_matched"]
    assert "R" not in pM
    assert "contrasts_with_R" not in pM
    for arm in ("Q1", "Q2", "Q3"):
        assert "lower" not in pM[arm]
        assert "upper" not in pM[arm]

def test_qlear_head_schema_preservation_complete_and_incomplete(tmp_path):
    """Finding N3: Verify Q-LEAR panel blocks preserve HEAD schema for complete and incomplete rosters."""
    expected_keys = {
        "applications",
        "backends",
        "n_files",
        "roster_status",
        "failures",
        "per_seed",
        "per_file",
        "reproduction_panel",
    }

    # Case 1: Complete roster
    fits_dir_c = str(tmp_path / "fits_complete")
    out_dir_c = str(tmp_path / "out_complete")
    create_synthetic_qlear_fits(fits_dir_c, include_o2=False)

    saved_argv = sys.argv
    sys.argv = ["score_field_half.py", fits_dir_c, out_dir_c]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir_c, "field-report.json"), "r") as fh:
        report_c = json.load(fh)

    for p in ("hardware", "simulator"):
        p_block = report_c["panels"][p]
        assert set(p_block.keys()) == expected_keys
        assert p_block["roster_status"] == "complete"
        assert p_block["failures"] == []
        assert len(p_block["per_seed"]) == 10
        for s_str, s_data in p_block["per_seed"].items():
            assert s_data["status"] == "estimated"
            assert "rung_errors" in s_data
            assert "gaps" in s_data

    # Case 2: Incomplete roster (induce evaluation failure for arm F seed 3 in hardware)
    fits_dir_inc = str(tmp_path / "fits_incomplete")
    out_dir_inc = str(tmp_path / "out_incomplete")
    create_synthetic_qlear_fits(fits_dir_inc, include_o2=False)

    f_path = os.path.join(fits_dir_inc, "pred-F-seed3.json")
    with open(f_path, "r") as fh:
        rec = json.load(fh)
    # Set all nonpositive predictions so frozen_hellinger returns nonpositive_prediction_total
    rec["panels"]["hardware"]["groundstate_ibm_lagos"]["prediction"] = [0.0, 0.0, 0.0, 0.0]
    with open(f_path, "w") as fh:
        json.dump(rec, fh)

    sys.argv = ["score_field_half.py", fits_dir_inc, out_dir_inc]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir_inc, "field-report.json"), "r") as fh:
        report_inc = json.load(fh)

    p_hw = report_inc["panels"]["hardware"]
    assert set(p_hw.keys()) == expected_keys
    assert p_hw["roster_status"] == "incomplete_roster"
    assert len(p_hw["failures"]) > 0
    assert p_hw["per_seed"]["3"]["status"] == "not_estimable"
    assert "missing evaluation" in p_hw["per_seed"]["3"]["reason"]
    assert p_hw["per_seed"]["0"]["status"] == "estimated"

    # Simulator panel remains complete
    p_sim = report_inc["panels"]["simulator"]
    assert set(p_sim.keys()) == expected_keys
    assert p_sim["roster_status"] == "complete"
    assert p_sim["failures"] == []


def test_shapley_two_group_mathematical_identity():
    """Verify the two-group Shapley share identity and agreement property on random inputs.

    Property 1 (Efficiency/Sum):
        phi_B + phi_D = C - F
        S_B + S_D = 1 whenever C - F != 0.

    Property 2 (Agreement):
        When Order 1 and Order 2 agree:
            (C - O) == (O2 - F) and (O - F) == (C - O2)
        then:
            phi_B = C - O  =>  S_B = (C - O) / (C - F)
            phi_D = O - F  =>  S_D = (O - F) / (C - F)
    """
    rng = np.random.default_rng(20261006)
    for _ in range(100):
        c = float(rng.uniform(0.5, 1.0))
        f = float(rng.uniform(0.0, 0.4))
        delta = c - f
        assert delta > 0

        o = float(rng.uniform(f, c))
        o2 = float(rng.uniform(f, c))

        # Order 1 increments
        d_b1 = c - o
        d_d1 = o - f
        # Order 2 increments
        d_d2 = c - o2
        d_b2 = o2 - f

        # Shapley values
        phi_b = 0.5 * (d_b1 + d_b2)
        phi_d = 0.5 * (d_d1 + d_d2)

        # Property 1: Sum identity
        assert abs((phi_b + phi_d) - delta) < 1e-14
        s_b = phi_b / delta
        s_d = phi_d / delta
        assert abs((s_b + s_d) - 1.0) < 1e-14

        # Property 2: Agreement
        o2_agree = c - o + f
        d_b2_agree = o2_agree - f
        d_d2_agree = c - o2_agree
        assert abs(d_b1 - d_b2_agree) < 1e-14
        assert abs(d_d1 - d_d2_agree) < 1e-14

        phi_b_agree = 0.5 * (d_b1 + d_b2_agree)
        phi_d_agree = 0.5 * (d_d1 + d_d2_agree)
        assert abs(phi_b_agree - d_b1) < 1e-14
        assert abs(phi_d_agree - d_d1) < 1e-14

        s_b_agree = phi_b_agree / delta
        s_d_agree = phi_d_agree / delta
        assert abs(s_b_agree - (d_b1 / delta)) < 1e-14
        assert abs(s_d_agree - (d_d1 / delta)) < 1e-14


# ==============================================================================
# Finding 7: QRAFT Joint Row Bootstrap Estimand Tests
# ==============================================================================

def test_qraft_joint_bootstrap_constant_seed_effects():
    """Finding 7: Constant per-seed errors must give a degenerate interval at their mean."""
    rows = []
    # 5 machines, 20 rows each = 100 rows
    for m in range(5):
        for _ in range(20):
            r = {c: "1.0" for c in score_qraft_panel.INPUT_COLUMNS}
            r["ComputerID"] = str(m)
            r["StateRealProb"] = "10.0"
            r["StateUpProb50"] = "10.0"  # raw error = 0
            rows.append(r)

    class MockConstantModel:
        def __init__(self, random_state=0):
            self.seed = random_state

        def fit(self, x, y):
            pass

        def predict(self, x):
            # Error = |(10.0 + seed) - 10.0| = seed
            return np.full(len(x), 10.0 + float(self.seed))

    orig_draws = score_qraft_panel.BOOTSTRAP_DRAWS
    score_qraft_panel.BOOTSTRAP_DRAWS = 1000

    try:
        with patch("sklearn.ensemble.HistGradientBoostingRegressor", MockConstantModel):
            res = score_qraft_panel.panel_matched(rows, include_raw=True)

        expected_mean = float(np.mean([float(s) for s in range(10)]))  # 4.5
        for arm in ("Q1", "Q2", "Q3"):
            contrast = res["contrasts_with_R"][f"{arm}-R"]
            assert np.isclose(contrast["ten_seed_mean"], expected_mean, atol=1e-12)
            # The interval must be degenerate at the mean
            assert np.isclose(contrast["lower"], expected_mean, atol=1e-12)
            assert np.isclose(contrast["upper"], expected_mean, atol=1e-12)
            assert contrast["per_seed"] == {str(s): float(s) for s in range(10)}
            assert np.isclose(res[arm]["lower"], expected_mean, atol=1e-12)
            assert np.isclose(res[arm]["upper"], expected_mean, atol=1e-12)
        assert np.isclose(res["R"]["lower"], 0.0, atol=1e-12)
        assert np.isclose(res["R"]["upper"], 0.0, atol=1e-12)
    finally:
        score_qraft_panel.BOOTSTRAP_DRAWS = orig_draws


def test_qraft_joint_bootstrap_overlapping_test_rows():
    """Finding 7: Verify joint row bootstrap operates across overlapping test splits."""
    rows = []
    for m in range(5):
        for i in range(20):
            r = {c: "1.0" for c in score_qraft_panel.INPUT_COLUMNS}
            r["ComputerID"] = str(m)
            r["StateRealProb"] = str(10.0 + float(i))
            r["StateUpProb50"] = "10.0"
            rows.append(r)

    orig_draws = score_qraft_panel.BOOTSTRAP_DRAWS
    score_qraft_panel.BOOTSTRAP_DRAWS = 500

    try:
        res = score_qraft_panel.panel_matched(rows, include_raw=True)
        for arm in ("Q1", "Q2", "Q3"):
            c = res["contrasts_with_R"][f"{arm}-R"]
            assert c["lower"] <= c["upper"]
            assert c["ten_seed_min"] <= c["ten_seed_mean"] <= c["ten_seed_max"]
            assert res[arm]["lower"] <= res[arm]["upper"]
        assert res["R"]["lower"] <= res["R"]["upper"]
    finally:
        score_qraft_panel.BOOTSTRAP_DRAWS = orig_draws


def test_qraft_arm_error_intervals_and_contrast_draws_identity():
    """Finding N2: Verify arm-error intervals exist and use the same draws as contrasts."""
    rows = []
    # 5 machines, 20 rows each = 100 rows
    for m in range(5):
        for i in range(20):
            r = {c: "1.0" for c in score_qraft_panel.INPUT_COLUMNS}
            r["ComputerID"] = str(m)
            r["StateRealProb"] = str(10.0 + float(i))
            r["StateUpProb50"] = str(8.0 + float(i % 3))
            # Mock released prediction columns for panel_released
            for col in score_qraft_panel.RELEASED_PREDICTION.values():
                r[col] = str(9.0 + float(i % 4))
            rows.append(r)

    orig_draws = score_qraft_panel.BOOTSTRAP_DRAWS
    score_qraft_panel.BOOTSTRAP_DRAWS = 200

    try:
        # Part 1: Panel R
        res_r, draws_r = score_qraft_panel.panel_released(rows, include_raw=True, return_draws=True)
        for arm in ("Q1", "Q2", "Q3", "R"):
            assert "lower" in res_r[arm]
            assert "upper" in res_r[arm]
            assert res_r[arm]["lower"] <= res_r[arm]["upper"]
            expected_lo = float(np.percentile(draws_r["arm_draws"][arm], 2.5))
            expected_hi = float(np.percentile(draws_r["arm_draws"][arm], 97.5))
            assert np.isclose(res_r[arm]["lower"], expected_lo, atol=1e-12)
            assert np.isclose(res_r[arm]["upper"], expected_hi, atol=1e-12)

        for arm in ("Q1", "Q2", "Q3"):
            # Contrast draw equals arm draw minus R draw
            arm_minus_r = draws_r["arm_draws"][arm] - draws_r["arm_draws"]["R"]
            assert np.allclose(draws_r["contrast_draws"][arm], arm_minus_r, atol=1e-14)
            c_block = res_r["contrasts_with_R"][f"{arm}-R"]
            assert np.isclose(c_block["lower"], float(np.percentile(arm_minus_r, 2.5)), atol=1e-12)
            assert np.isclose(c_block["upper"], float(np.percentile(arm_minus_r, 97.5)), atol=1e-12)

        # Part 2: Panel M
        res_m, draws_m = score_qraft_panel.panel_matched(rows, include_raw=True, return_draws=True)
        for arm in ("Q1", "Q2", "Q3", "R"):
            assert "lower" in res_m[arm]
            assert "upper" in res_m[arm]
            assert res_m[arm]["lower"] <= res_m[arm]["upper"]
            expected_lo = float(np.percentile(draws_m["arm_draws"][arm], 2.5))
            expected_hi = float(np.percentile(draws_m["arm_draws"][arm], 97.5))
            assert np.isclose(res_m[arm]["lower"], expected_lo, atol=1e-12)
            assert np.isclose(res_m[arm]["upper"], expected_hi, atol=1e-12)

        for arm in ("Q1", "Q2", "Q3"):
            # Contrast draw equals arm draw minus R draw
            arm_minus_r = draws_m["arm_draws"][arm] - draws_m["arm_draws"]["R"]
            assert np.allclose(draws_m["contrast_draws"][arm], arm_minus_r, atol=1e-14)
            c_block = res_m["contrasts_with_R"][f"{arm}-R"]
            assert np.isclose(c_block["lower"], float(np.percentile(arm_minus_r, 2.5)), atol=1e-12)
            assert np.isclose(c_block["upper"], float(np.percentile(arm_minus_r, 97.5)), atol=1e-12)

    finally:
        score_qraft_panel.BOOTSTRAP_DRAWS = orig_draws


# ==============================================================================
# Finding 8: Legacy Defaults and O2 Separation Tests
# ==============================================================================

def test_legacy_fit_planning(tmp_path):
    """Finding 8: Default fit planning runs legacy arms only; O2 is explicit and isolated."""
    # Test 1: Default planning without --arm targets fit-manifest.json and legacy arms only
    assert "O2" not in fit_field_half.LEGACY_ARMS
    assert "O2" in fit_field_half.O2_ARM
    assert list(fit_field_half.LEGACY_ARMS.keys()) == ["A", "C4", "C", "O", "F"]

    # Test 2: Fitting O2 without frozen rule raises barrier error
    saved_argv = sys.argv
    sys.argv = ["fit_field_half.py", "mock_release", str(tmp_path), "--arm", "O2"]
    try:
        with pytest.raises(RuntimeError) as exc_info:
            fit_field_half.main()
        assert "FROZEN RULE BARRIER" in str(exc_info.value)
    finally:
        sys.argv = saved_argv


def test_legacy_scoring_with_o2_present_reproduces_original(tmp_path):
    """Finding 8: Legacy scoring ignores O2 files when --opposite is not given."""
    fits_dir = str(tmp_path / "fits")
    out_dir = str(tmp_path / "out")

    # Create directory with both legacy arms AND O2
    create_synthetic_qlear_fits(fits_dir, include_o2=True)

    saved_argv = sys.argv
    sys.argv = ["score_field_half.py", fits_dir, out_dir]
    try:
        score_field_half.main()
    finally:
        sys.argv = saved_argv

    with open(os.path.join(out_dir, "field-report.json"), "r") as fh:
        report = json.load(fh)

    # Ensure O2 is not present anywhere in the report
    assert "opposite_ladder" not in report
    for panel in ("hardware", "simulator"):
        p_block = report["panels"][panel]
        assert p_block["roster_status"] == "complete"
        assert p_block["failures"] == []
        for fblock in p_block["per_file"].values():
            assert "O2" not in fblock
        for s_data in p_block["per_seed"].values():
            assert "O2" not in s_data["rung_errors"]
            assert "C-O2" not in s_data["gaps"]
            assert "O2-F" not in s_data["gaps"]

    # Ensure O2 prediction files are not in input_sha256
    if "input_sha256" in report:
        for fname in report["input_sha256"]:
            assert not fname.startswith("pred-O2-")


# ==============================================================================
# Finding 10: Target Identity and O2 Metadata Tests
# ==============================================================================

def test_target_identity_validation_mismatches(tmp_path):
    """Finding 10: Validate target vectors, lengths, support, and O2 metadata."""
    fits_dir = str(tmp_path / "fits_mismatch")
    out_dir = str(tmp_path / "out_mismatch")
    release_dir = str(tmp_path / "release_mismatch")
    rule_file = str(tmp_path / "rule.md")
    with open(rule_file, "w") as fh:
        fh.write("# Frozen Rule\n")

    create_synthetic_qlear_release(release_dir)
    create_synthetic_qlear_fits(fits_dir, include_o2=True)

    # Subtest 1: Target value mismatch across arms
    # Corrupt target for arm F seed 0 in one file
    f_path = os.path.join(fits_dir, "pred-F-seed0.json")
    with open(f_path, "r") as fh:
        rec = json.load(fh)
    rec["panels"]["hardware"]["groundstate_ibm_lagos"]["target"][0] = 99.0
    with open(f_path, "w") as fh:
        json.dump(rec, fh)

    saved_argv = sys.argv
    sys.argv = ["score_field_half.py", fits_dir, out_dir]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "TARGET VALUE MISMATCH" in str(exc_info.value)
    finally:
        sys.argv = saved_argv

    # Revert corrupt file
    create_synthetic_qlear_fits(fits_dir, include_o2=True)

    # Subtest 2: Target length mismatch
    with open(f_path, "r") as fh:
        rec = json.load(fh)
    rec["panels"]["hardware"]["groundstate_ibm_lagos"]["target"].append(25.0)
    with open(f_path, "w") as fh:
        json.dump(rec, fh)

    sys.argv = ["score_field_half.py", fits_dir, out_dir]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "TARGET LENGTH MISMATCH" in str(exc_info.value)
    finally:
        sys.argv = saved_argv

    # Revert
    create_synthetic_qlear_fits(fits_dir, include_o2=True)

    # Subtest 3: Raw target mismatch against learned arms
    # Corrupt raw CSV
    raw_csv = os.path.join(release_dir, "real_circuits_hardware", "groundstate_ibm_lagos.csv")
    with open(raw_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["target", "observed_prob_50"])
        writer.writeheader()
        writer.writerows([
            {"target": 99.0, "observed_prob_50": 20.0},
            {"target": 25.0, "observed_prob_50": 30.0},
            {"target": 25.0, "observed_prob_50": 22.0},
            {"target": 25.0, "observed_prob_50": 28.0},
        ])

    sys.argv = [
        "score_field_half.py", fits_dir, out_dir,
        "--release", release_dir, "--raw", "--frozen-rule", rule_file
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "RAW TARGET VALUE MISMATCH" in str(exc_info.value)
    finally:
        sys.argv = saved_argv

    # Revert raw CSV
    create_synthetic_qlear_release(release_dir)

    # Subtest 4: O2 column mismatch
    o2_path = os.path.join(fits_dir, "pred-O2-seed0.json")
    with open(o2_path, "r") as fh:
        rec = json.load(fh)
    rec["columns"] = ["wrong_column"]
    with open(o2_path, "w") as fh:
        json.dump(rec, fh)

    sys.argv = [
        "score_field_half.py", fits_dir, out_dir,
        "--opposite", "--frozen-rule", rule_file
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "O2 COLUMNS MISMATCH" in str(exc_info.value)
    finally:
        sys.argv = saved_argv

    # Revert O2 columns and test O2 release identity mismatch
    create_synthetic_qlear_fits(fits_dir, include_o2=True)
    with open(o2_path, "r") as fh:
        rec = json.load(fh)
    rec["release_identity"] = {"release_dir": "wrong_release_dir"}
    with open(o2_path, "w") as fh:
        json.dump(rec, fh)

    sys.argv = [
        "score_field_half.py", fits_dir, out_dir,
        "--release", release_dir, "--opposite", "--frozen-rule", rule_file
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            score_field_half.main()
        assert "O2 RELEASE IDENTITY MISMATCH" in str(exc_info.value)
    finally:
        sys.argv = saved_argv

