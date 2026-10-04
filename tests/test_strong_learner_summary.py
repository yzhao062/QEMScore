"""Tests for tools/strong_learner_summary.py (comparisons beside the strong-learner results)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import strong_learner_summary as summary

ART = Path(__file__).resolve().parents[1] / "artifacts" / "descriptor-information"


def _est(point, lower=None, upper=None):
    lower = point - 0.01 if lower is None else lower
    upper = point + 0.01 if upper is None else upper
    return {"status": "estimated", "point": point, "interval": {"lower": lower, "upper": upper}}


def _cell(label, d, d_lo, d_hi, c, f, paired=0.0, paired_lo=-0.1, paired_hi=0.1):
    return {"classification": {"label": label}, "D": _est(d, d_lo, d_hi),
            "means": {"C": _est(c), "F": _est(f), "P": _est(c)},
            "comparison_vs_original": {"delta_D_over_C": _est(paired, paired_lo, paired_hi),
                                       "delta_D": _est(d * 0.01)}}


def test_label_changes_and_statements_on_toy_records():
    row = "shipped-s101-n640/tfi"
    strong = {"parts": {
        "A": {"rows": {row: {"rungs": {"N1": _cell("measurement_adds", 0.002, 0.001, 0.003, 0.010, 0.008,
                                                   0.05, 0.01, 0.09),
                                       "N2": _cell("measurement_adds", 0.004, 0.002, 0.006, 0.020, 0.016)}}},
              "rung_statements": {"tfi": {"N1": {"statement": "measurement_adds", "labels": {"101": "measurement_adds"}},
                                          "N2": {"statement": "measurement_adds", "labels": {"101": "measurement_adds"}}}}},
        "B": {"rows": {}, "rung_statements": {}}}}
    derived = {"parts": {
        "A": {"rows": {row: {"rungs": {"N1": _cell("not_distinguished", 0.001, -0.001, 0.003, 0.011, 0.010),
                                       "N2": _cell("measurement_adds", 0.004, 0.002, 0.006, 0.020, 0.016)}}},
              "rung_statements": {"tfi": {"N1": {"statement": "not_distinguished", "labels": {"101": "not_distinguished"}},
                                          "N2": {"statement": "measurement_adds", "labels": {"101": "measurement_adds"}}}}},
        "B": {"rows": {}, "rung_statements": {}}}}
    st = summary.statements(strong, derived)
    assert st["changed"] == ["A/tfi/N1"]
    changes = summary.label_changes(strong, derived)
    assert len(changes) == 1
    ch = changes[0]
    assert ch["cell"] == f"A/{row}/N1"
    assert ch["paired_interval_excludes_zero"] is True
    assert ch["D_interval_width_change"] == "narrowed"
    assert ch["D_strong"]["width"] == pytest.approx(0.002)
    assert ch["D_derived"]["width"] == pytest.approx(0.004)


def test_sweep_identity_reports_differences(tmp_path):
    for root, shift in ((tmp_path / "derived", 0.0), (tmp_path / "sweep", 0.0)):
        root.mkdir()
        for arm in ("C", "F"):
            stem = f"shipped-s101-n640__R0__k01__{arm}"
            np.savez(root / f"{stem}.npz", test=np.array([0.1, 0.2]) + shift, validation=np.array([0.3]))
            (root / f"{stem}.json").write_text(json.dumps({"selected_model": "mlp"}))
    out = summary.sweep_identity(tmp_path / "derived", tmp_path / "sweep")
    assert out == {"compared": 2, "max_abs_diff": 0.0, "nonzero": [], "selected_model_differs": []}
    np.savez(tmp_path / "sweep" / "shipped-s101-n640__R0__k01__F.npz",
             test=np.array([0.1, 0.2 + 1e-9]), validation=np.array([0.3]))
    (tmp_path / "sweep" / "shipped-s101-n640__R0__k01__F.json").write_text(json.dumps({"selected_model": "hgbr"}))
    out = summary.sweep_identity(tmp_path / "derived", tmp_path / "sweep")
    assert out["nonzero"][0]["fit"] == "shipped-s101-n640__R0__k01__F"
    assert out["selected_model_differs"] == ["shipped-s101-n640__R0__k01__F"]


@pytest.mark.skipif(not (ART / "strong-learners" / "strong-learner-summary.json").exists(),
                    reason="strong-learner artifacts not present")
def test_summary_reproduces_the_committed_record(tmp_path):
    sl = ART / "strong-learners"
    out = tmp_path / "summary.json"
    summary.main(["--analysis-a", str(sl / "analysis-a.json"),
                  "--derived-analysis-a", str(sl / "derived-analysis-a.json"),
                  "--report", str(sl / "strong-learner-report.json"),
                  "--macos-analysis-a", str(ART / "strength-indicator" / "analysis-a.json"),
                  "--out", str(out)])
    committed = json.loads((sl / "strong-learner-summary.json").read_text(encoding="utf-8"))
    rebuilt = json.loads(out.read_text(encoding="utf-8"))
    committed.pop("derived_vs_sweep_2048")  # needs the fit trees
    assert rebuilt == committed
    assert all(e["holds"] for e in committed["expectations"].values())


def test_hurts_cells_include_cells_whose_label_did_not_change():
    row = "shipped-s101-n640/tfi"
    hurt = _cell("measurement_hurts", -0.0001, -0.0002, -0.00005, 0.004, 0.0041)
    strong = {"parts": {"A": {"rows": {row: {"rungs": {"R0": hurt,
                                                       "N2": _cell("measurement_adds", 0.004, 0.002, 0.006, 0.02, 0.016)}}}},
                        "B": {"rows": {}}}}
    derived = {"parts": {"A": {"rows": {row: {"rungs": {"R0": hurt,
                                                        "N2": _cell("measurement_adds", 0.004, 0.002, 0.006, 0.02, 0.016)}}}},
                         "B": {"rows": {}}}}
    cells = summary.hurts_cells(strong, derived)
    assert [c["cell"] for c in cells] == [f"A/{row}/R0"]
    assert cells[0]["label_derived"] == "measurement_hurts"
    assert cells[0]["means_strong"] == {"C": 0.004, "F": 0.0041, "P": 0.004}
