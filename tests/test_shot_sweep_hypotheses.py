"""Tests for tools/shot_sweep_hypotheses.py."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.shot_sweep_hypotheses import (
    CANONICAL_LEVELS,
    FAMILIES,
    SEEDS,
    compute_spearman,
    evaluate_pipeline,
)

_REPO = Path(__file__).resolve().parents[1]


def _build_synthetic_dataset(
    tmp_path: Path,
    h1_mode: str = "pass",  # "pass", "fail", "not_testable"
    h2_mode: str = "pass",  # "pass", "fail", "not_testable_count", "not_testable_rows"
    ab_disagree: bool = False,
    ab_label_disagree: bool = False,
    constant_series: bool = False,
) -> tuple[Path, dict[str, Path], dict[str, Path], Path]:
    """Build synthetic predictions.json, analysis-a, and analysis-b files."""
    levels = list(CANONICAL_LEVELS)

    # 1. Predictions dictionary
    class_cells = {}
    rule_cells = {}

    for lvl in levels:
        is_2048 = (lvl == "2048")
        lvl_idx = levels.index(lvl)

        for seed in SEEDS:
            for fam in FAMILIES:
                row_k = f"shipped-s{seed}-n640/{fam}"
                for rung in ("R0", "N1", "N2", "R5"):
                    ck = f"{lvl}/{row_k}/{rung}"

                    # Base D/C* values
                    if h1_mode == "not_testable":
                        # All D/C* below 0.10 -> 0 qualifying cells
                        dc_star = 0.02 + 0.01 * lvl_idx
                    else:
                        dc_star = 0.05 + 0.04 * lvl_idx  # 0.05 up to 0.29

                    adds_thresh = 0.150 if rung != "R0" else 0.350
                    ref_label = "measurement_adds" if rung == "N2" else "not_distinguished"
                    ref_baseline = ref_label

                    # Determine predicted label and label changing
                    if h2_mode == "not_testable_count":
                        # Only 2 cells label changing total
                        if lvl == "exact" and row_k == "shipped-s101-n640/tfi" and rung in ("N1", "R0"):
                            pred_label = "measurement_adds"
                            lc = True
                        else:
                            pred_label = ref_baseline
                            lc = False
                    elif h2_mode == "not_testable_rows":
                        # Label changing only on 1 row (s101/tfi)
                        if row_k == "shipped-s101-n640/tfi" and lvl in ("32768", "131072", "exact") and rung in ("N1", "R0"):
                            pred_label = "measurement_adds"
                            lc = True
                        else:
                            pred_label = ref_baseline
                            lc = False
                    else:
                        # Standard label changing setup
                        if rung in ("N1", "R0") and lvl in ("32768", "131072", "exact"):
                            pred_label = "measurement_adds"
                            lc = True
                        elif rung == "N2" and lvl in ("256", "1024"):
                            pred_label = "not_distinguished"
                            lc = True
                        else:
                            pred_label = ref_baseline
                            lc = False

                    class_cells[ck] = {
                        "level": lvl,
                        "row": row_k,
                        "rung": rung,
                        "D_over_C_star": dc_star,
                        "adds_threshold": adds_thresh,
                        "predicted_label": pred_label,
                        "reference_label": ref_label,
                        "reference_label_baseline": ref_baseline,
                        "label_changing": lc,
                    }

    predictions_data = {
        "schema": "shot-sweep-predictions-v1",
        "status": "pre-registered prediction",
        "classification_cells": class_cells,
        "rule_cells": rule_cells,
    }

    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps(predictions_data, indent=2))

    # 2. Analyses A and B
    analysis_a_files = {}
    analysis_b_files = {}

    for lvl in levels:
        lvl_idx = levels.index(lvl)
        rows_a = {}
        rows_b = {}
        cells_b = []

        for seed in SEEDS:
            for fam in FAMILIES:
                row_k = f"shipped-s{seed}-n640/{fam}"
                b_row_k = f"s{seed}__{fam}"
                rows_a[row_k] = {"rungs": {}}
                rows_b[b_row_k] = {}

                for rung in ("R0", "N1", "N2", "R5"):
                    ck = f"{lvl}/{row_k}/{rung}"
                    c_info = class_cells[ck]
                    dc_star = c_info["D_over_C_star"]

                    # Observed D/C
                    if constant_series and rung == "N1" and row_k == "shipped-s101-n640/tfi":
                        obs_dc = 0.15  # constant across levels
                    elif h1_mode == "fail":
                        obs_dc = 0.35 - 0.04 * lvl_idx  # decreasing -> negative correlation
                    else:
                        obs_dc = 0.85 * dc_star  # ratio ~0.85 (in [0.6, 1.1])

                    lo = obs_dc - 0.02
                    hi = obs_dc + 0.02

                    # Observed label
                    if h2_mode == "fail":
                        # Invert predictions
                        if c_info["predicted_label"] == "measurement_adds":
                            obs_label_a = "not_distinguished"
                        else:
                            obs_label_a = "measurement_adds"
                    else:
                        obs_label_a = c_info["predicted_label"] or c_info["reference_label_baseline"]

                    obs_label_b = obs_label_a
                    b_doc = obs_dc
                    b_lo = lo

                    if ab_disagree and lvl == "exact" and row_k == "shipped-s101-n640/tfi" and rung == "R0":
                        b_doc += 0.05
                    if ab_label_disagree and lvl == "exact" and row_k == "shipped-s101-n640/tfi" and rung == "R0":
                        obs_label_b = "measurement_hurts" if obs_label_a != "measurement_hurts" else "measurement_adds"

                    rows_a[row_k]["rungs"][rung] = {
                        "D_over_C": {"point": obs_dc, "interval": {"lower": lo, "upper": hi}},
                        "classification": {"label": obs_label_a},
                    }

                    # Analysis B labels with Title Case
                    label_map_b = {
                        "measurement_adds": "Measurement adds",
                        "measurement_hurts": "Measurement hurts",
                        "not_distinguished": "Not distinguished",
                    }
                    cell_b_record = {
                        "part": "A",
                        "dataset_seed": seed,
                        "family": fam,
                        "rung": rung,
                        "mean_d_over_c": b_doc,
                        "d_over_c_ci_95": [b_lo, hi],
                        "classification_label": label_map_b[obs_label_b],
                        "status": "estimated",
                    }
                    rows_b[b_row_k][rung] = cell_b_record
                    cells_b.append(cell_b_record)

        pa = tmp_path / f"analysis_a_{lvl}.json"
        pa.write_text(json.dumps({"parts": {"A": {"rows": rows_a}}}))
        analysis_a_files[lvl] = pa

        pb = tmp_path / f"analysis_b_{lvl}.json"
        pb.write_text(json.dumps({"by_part_row_rung": {"A": rows_b}, "cells": cells_b}))
        analysis_b_files[lvl] = pb

    out_file = tmp_path / "hypotheses_result.json"
    return pred_file, analysis_a_files, analysis_b_files, out_file


def _run_hypotheses_cli(
    pred_file: Path,
    a_files: dict[str, Path],
    b_files: dict[str, Path],
    out_file: Path,
    prefit_failed: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_REPO / "tools/shot_sweep_hypotheses.py"),
        "--predictions",
        str(pred_file),
    ]
    for lvl, p in a_files.items():
        cmd.extend(["--analysis-a", f"{lvl}={p}"])
    for lvl, p in b_files.items():
        cmd.extend(["--analysis-b", f"{lvl}={p}"])
    if prefit_failed:
        cmd.append("--prefit-failed")
    cmd.extend(["--out", str(out_file)])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_reading_outcome_1(tmp_path: Path):
    """Reading outcome 1: H1 holds and H2 holds."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="pass")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 1
    assert ev["h1"]["holds"] is True
    assert ev["h2"]["holds"] is True


def test_reading_outcome_2(tmp_path: Path):
    """Reading outcome 2: H1 holds and H2 fails."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="fail")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 2
    assert ev["h1"]["holds"] is True
    assert ev["h2"]["holds"] is False


def test_reading_outcome_3(tmp_path: Path):
    """Reading outcome 3: H2 holds and H1 fails."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="fail", h2_mode="pass")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 3
    assert ev["h1"]["holds"] is False
    assert ev["h2"]["holds"] is True


def test_reading_outcome_4(tmp_path: Path):
    """Reading outcome 4: Both fail."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="fail", h2_mode="fail")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 4
    assert ev["h1"]["holds"] is False
    assert ev["h2"]["holds"] is False


def test_reading_outcome_5_h1_not_testable_by_count(tmp_path: Path):
    """Reading outcome 5: H1 not testable because fewer than 10 qualifying cells."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="not_testable", h2_mode="pass")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 5
    assert ev["h1"]["testable"] is False
    assert ev["h1"]["qualifying_cell_count"] < 10
    assert ev["h2"]["holds"] is True


def test_reading_outcome_5_h2_not_testable_count(tmp_path: Path):
    """Reading outcome 5: H2 not testable because fewer than 6 label-changing cells."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="not_testable_count")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 5
    assert ev["h2"]["testable"] is False
    assert ev["h2"]["label_changing_cell_count"] < 6


def test_reading_outcome_5_h2_not_testable_rows(tmp_path: Path):
    """Reading outcome 5: H2 not testable because label-changing cells come from fewer than 2 rows."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="not_testable_rows")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["reading_outcome"] == 5
    assert ev["h2"]["testable"] is False
    assert ev["h2"]["label_changing_row_count"] < 2


def test_reading_outcome_6_prefit_failed(tmp_path: Path):
    """Reading outcome 6 needs only the failed check report: no predictions, fits, or analyses."""
    report = tmp_path / "check1-faithful-s101.json"
    report.write_text(json.dumps({"command": "check-faithful", "passed": False}), encoding="utf-8")
    out_file = tmp_path / "hypotheses.json"
    cmd = [sys.executable, str(_REPO / "tools/shot_sweep_hypotheses.py"), "--prefit-failed",
           "--failure-report", str(report), "--out", str(out_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(out_file.read_text())["evaluation"]
    assert ev["reading_outcome"] == 6
    assert "Outcome 6" in ev["reading_statement"]
    assert ev["h1"] == {"evaluated": False} and ev["h2"] == {"evaluated": False}
    assert ev["failed_checks"][0]["passed"] is False


def test_prefit_failed_rejects_passing_reports(tmp_path: Path):
    """A report that passed cannot justify outcome 6."""
    report = tmp_path / "check2.json"
    report.write_text(json.dumps({"command": "check-exact", "passed": True}), encoding="utf-8")
    cmd = [sys.executable, str(_REPO / "tools/shot_sweep_hypotheses.py"), "--prefit-failed",
           "--failure-report", str(report), "--out", str(tmp_path / "h.json")]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0


def test_missing_prediction_cell_exits_nonzero(tmp_path: Path):
    """One missing prediction record is an input error, not a smaller series."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="pass")
    pred = json.loads(p_file.read_text())
    pred["classification_cells"].pop("exact/shipped-s101-n640/tfi/N1")
    p_file.write_text(json.dumps(pred))
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode != 0
    assert "incomplete prediction inventory" in proc.stderr


def test_missing_prediction_level_exits_nonzero(tmp_path: Path):
    """Removing every exact-level record must not turn the reading."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="pass")
    pred = json.loads(p_file.read_text())
    pred["classification_cells"] = {k: v for k, v in pred["classification_cells"].items()
                                    if not k.startswith("exact/")}
    p_file.write_text(json.dumps(pred))
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode != 0


def test_spearman_constant_series_fails():
    """A series whose D/C* or observed D/C is constant across levels fails (a)."""
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    y_const = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    corr, is_const = compute_spearman(x, y_const)
    assert is_const is True
    assert corr is None

    corr_x_const, is_x_const = compute_spearman(y_const, x)
    assert is_x_const is True
    assert corr_x_const is None


def test_spearman_tie_ranks():
    """Spearman correlations use average ranks for ties and unrounded values."""
    from scipy.stats import pearsonr, rankdata
    x = [1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0]
    y = [2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0]
    corr, is_const = compute_spearman(x, y)
    assert not is_const
    assert corr is not None

    # Compare against explicit Pearson on average ranks
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    expected_corr = pearsonr(rx, ry)[0]
    assert np.isclose(corr, expected_corr, atol=1e-12)


def test_hurts_as_match_rule(tmp_path: Path):
    """'Measurement hurts' counts as a miss for predicted 'adds' and match for predicted 'not distinguished'."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="pass")

    # Modify one cell's observed label to measurement_hurts
    # 1. When predicted is not_distinguished: must count as match and increment hurts_as_match_count
    for lvl in CANONICAL_LEVELS:
        a_data = json.loads(a_files[lvl].read_text())
        b_data = json.loads(b_files[lvl].read_text())
        row_k = "shipped-s101-n640/tfi"
        b_row_k = "s101__tfi"
        if lvl == "256":
            # N2 at 256 is predicted not_distinguished
            a_data["parts"]["A"]["rows"][row_k]["rungs"]["N2"]["classification"]["label"] = "measurement_hurts"
            b_data["by_part_row_rung"]["A"][b_row_k]["N2"]["classification_label"] = "Measurement hurts"
            for c in b_data["cells"]:
                if c["dataset_seed"] == 101 and c["family"] == "tfi" and c["rung"] == "N2":
                    c["classification_label"] = "Measurement hurts"
        a_files[lvl].write_text(json.dumps(a_data))
        b_files[lvl].write_text(json.dumps(b_data))

    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(out_file.read_text())
    ev = res["evaluation"]
    assert ev["h2"]["hurts_as_match_count"] >= 1
    cell_key = "256/shipped-s101-n640/tfi/N2"
    cell_res = ev["h2"]["cells"][cell_key]
    assert cell_res["observed_label"] == "measurement_hurts"
    assert cell_res["predicted_label"] == "not_distinguished"
    assert cell_res["matched"] is True
    assert cell_res["hurts_match"] is True


def test_ab_disagreement_exits_nonzero(tmp_path: Path):
    """If any value differs by > 1e-12 or label differs between A and B, exit nonzero."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(
        tmp_path, h1_mode="pass", h2_mode="pass", ab_disagree=True
    )
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode != 0
    assert "Disagreement between analysis A and analysis B" in proc.stderr
    assert not out_file.exists()


def test_ab_label_disagreement_exits_nonzero(tmp_path: Path):
    """If any label differs between A and B, exit nonzero."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(
        tmp_path, h1_mode="pass", h2_mode="pass", ab_label_disagree=True
    )
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode != 0
    assert "Disagreement between analysis A and analysis B" in proc.stderr
    assert not out_file.exists()


def test_reading_item5_other_fails_gives_outcome_4(tmp_path: Path):
    """Rule, Reading item 5: H1 not testable and H2 fails, so item 4 applies."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="not_testable", h2_mode="fail")
    proc = _run_hypotheses_cli(p_file, a_files, b_files, out_file)
    assert proc.returncode == 0, proc.stderr
    ev = json.loads(out_file.read_text())["evaluation"]
    assert ev["h1"]["testable"] is False
    assert ev["h2"]["holds"] is False
    assert ev["reading_outcome"] == 4


def test_missing_level_exits_nonzero(tmp_path: Path):
    """Every one of the seven levels is required."""
    p_file, a_files, b_files, out_file = _build_synthetic_dataset(tmp_path, h1_mode="pass", h2_mode="pass")
    a_short = {k: v for k, v in a_files.items() if k != "exact"}
    b_short = {k: v for k, v in b_files.items() if k != "exact"}
    assert len(a_short) == len(a_files) - 1
    proc = _run_hypotheses_cli(p_file, a_short, b_short, out_file)
    assert proc.returncode != 0
