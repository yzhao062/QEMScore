"""Tests for tools/shot_sweep_report.py (checks and expectations beside the shot-sweep results)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools import shot_sweep_report as report

ART = Path(__file__).resolve().parents[1] / "artifacts" / "descriptor-information" / "shot-sweep"


def _fit(path: Path, test, validation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, test=np.asarray(test, dtype=float), validation=np.asarray(validation, dtype=float))


def test_identity_reports_zero_and_nonzero_differences(tmp_path):
    dirs = {level: tmp_path / f"shots-{level}" for level in ("2048", "256", "exact")}
    for level, root in dirs.items():
        _fit(root / "fits" / "shipped-s101-n640__R0__k01__C.npz", [0.1, 0.2], [0.3])
        _fit(root / "fits" / "shipped-s101-n640__R0__A.npz", [0.5, 0.6], [0.7])
        # F is not compared: it reads r, which differs across levels.
        _fit(root / "fits" / "shipped-s101-n640__R0__k01__F.npz", [float(len(level))] * 2, [0.0])
    _fit(dirs["exact"] / "fits" / "shipped-s101-n640__R0__A.npz", [0.5, 0.6 + 1e-9], [0.7])
    out = report.identity(dirs)
    assert out["fits_compared_per_level"] == 2
    assert out["levels"]["256"] == {"max_abs_diff": 0.0, "nonzero_fits": []}
    exact = out["levels"]["exact"]
    assert exact["nonzero_fits"][0]["fit"] == "shipped-s101-n640__R0__A"
    assert exact["max_abs_diff"] == pytest.approx(1e-9, rel=1e-6)
    assert out["max_abs_diff_all_levels"] == exact["max_abs_diff"]


@pytest.mark.skipif(not (ART / "levels" / "2048" / "analysis-a.json").exists(),
                    reason="shot-sweep artifacts not present")
def test_report_reproduces_the_committed_record_except_identity(tmp_path, monkeypatch):
    """Everything but the identity check (which reads the fits) reproduces from the artifacts."""
    for level in report.LEVELS:
        dest = tmp_path / f"shots-{level}" / "analysis"
        dest.mkdir(parents=True)
        shutil.copy(ART / "levels" / level / "analysis-a.json", dest / "analysis-a.json")
    committed = json.loads((ART / "shot-sweep-report.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(report, "identity", lambda dirs: committed["c_and_a_identity"])
    predictions = json.loads((ART / "predictions.json").read_text(encoding="utf-8"))
    reference = json.loads((ART.parent / "strength-indicator" / "analysis-a.json").read_text(encoding="utf-8"))
    rebuilt = json.loads(json.dumps(report.report(tmp_path, predictions, reference), allow_nan=False))
    assert rebuilt == committed
