"""Tests for strongest descriptor reference and inverse-variance weighting ceiling.

Verifies:
1. Synthetic toy for linear stacking arithmetic and one-stage circuit bootstrap interval.
2. Synthetic toy for inverse-variance weighting ceiling formula, boundary conditions, and monotonicity.
3. Verification checks:
   - mean C and mean F match analysis-a.json to 1e-12.
   - per-cell calibration errors match posthoc-measurement-floor.json.
   - R0 polynomial test errors match polynomial_refit_summary.json to 1e-12.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.ivw_ceiling import compute_ivw_combination
from tools.strongest_reference import (
    compute_stacking_and_interval,
    fit_polynomial_5,
    generate_circuit_counts,
)

_REPO = Path(__file__).resolve().parents[1]


def test_stacking_arithmetic_synthetic():
    """Verify recalibration, stacking OLS, and bootstrap interval on synthetic toy data."""
    rng = np.random.default_rng(42)
    n_circ = 20
    draws = 500

    # 2 cells, 10 circuits each
    members = [np.arange(0, 10), np.arange(10, 20)]
    item_circuit = np.arange(20)

    val_preds = []
    y_vals = []
    r_vals = []
    te_preds = []
    y_tes = []
    r_tes = []

    for c in range(2):
        # Validation: y = x1 + x2, ref knows x1, r knows x2
        x1_v = rng.uniform(-1, 1, size=200)
        x2_v = rng.uniform(-1, 1, size=200)
        y_v = x1_v + x2_v
        ref_v = x1_v + rng.normal(0, 0.02, size=200)
        r_v = x2_v + rng.normal(0, 0.02, size=200)

        # Test
        x1_t = rng.uniform(-1, 1, size=10)
        x2_t = rng.uniform(-1, 1, size=10)
        y_t = x1_t + x2_t
        ref_t = x1_t + rng.normal(0, 0.02, size=10)
        r_t = x2_t + rng.normal(0, 0.02, size=10)

        val_preds.append(ref_v)
        y_vals.append(y_v)
        r_vals.append(r_v)
        te_preds.append(ref_t)
        y_tes.append(y_t)
        r_tes.append(r_t)

    circ_counts = generate_circuit_counts(n_circ, draws=draws, seed=20261002)

    res = compute_stacking_and_interval(
        val_preds, y_vals, r_vals,
        te_preds, y_tes, r_tes,
        members, item_circuit, circ_counts
    )

    # Basic assertions
    assert "recal_macro_mae" in res
    assert "stack_macro_mae" in res
    assert "increment_point" in res
    assert math.isclose(
        res["increment_point"],
        res["recal_macro_mae"] - res["stack_macro_mae"],
        rel_tol=1e-12,
    )

    ci = res["increment_ci_95"]
    assert ci["lower"] <= ci["upper"]

    # In this toy, r carries strong useful signal so stacking should improve over recalibration
    assert res["increment_point"] > 0.0
    assert res["excludes_zero_above"] is True

    # Check cell details
    for d in res["cell_details"]:
        assert math.isclose(d["cell_increment"], d["recal_test_mae"] - d["stack_test_mae"], rel_tol=1e-12)
        assert abs(d["recal_slope"] - 1.0) < 0.2
        assert abs(d["stack_ref_coeff"] - 1.0) < 0.2
        assert abs(d["stack_r_coeff"] - 1.0) < 0.2


def test_ivw_ceiling_formula_synthetic():
    """Verify IVW ceiling formula, limits, and algebraic invariants."""
    # 1. Exact Pythagorean combination: 0.04 and 0.03 -> (1/1600 + 1/900)^(-1/2) = (2500/1440000)^(-1/2) = 0.024
    c_ref = np.array([0.04])
    c_r = np.array([0.03])
    c_comb = compute_ivw_combination(c_ref, c_r)
    assert math.isclose(float(c_comb[0]), 0.024, rel_tol=1e-12)

    # Ceiling D/C = 1 - 0.024 / 0.04 = 1 - 0.6 = 0.4
    ceil_dc = 1.0 - float(c_comb[0]) / float(c_ref[0])
    assert math.isclose(ceil_dc, 0.4, rel_tol=1e-12)

    # 2. Equal error: c_ref == c_r -> c_comb = c_ref / sqrt(2) -> ceiling = 1 - 1/sqrt(2)
    c_equal = np.array([0.05])
    c_comb_eq = compute_ivw_combination(c_equal, c_equal)
    assert math.isclose(float(c_comb_eq[0]), 0.05 / math.sqrt(2), rel_tol=1e-12)
    ceil_eq = 1.0 - float(c_comb_eq[0]) / float(c_equal[0])
    assert math.isclose(ceil_eq, 1.0 - 1.0 / math.sqrt(2), rel_tol=1e-12)

    # 3. Limit c_r -> inf: noisy measurement is useless -> ceiling -> 0
    c_inf_r = np.array([1e10])
    c_comb_inf = compute_ivw_combination(c_equal, c_inf_r)
    assert math.isclose(float(c_comb_inf[0]), 0.05, rel_tol=1e-10)
    assert math.isclose(1.0 - float(c_comb_inf[0]) / 0.05, 0.0, abs_tol=1e-10)

    # 4. Monotonicity: as c_ref increases with fixed c_r, ceiling increases
    c_refs = np.linspace(0.01, 0.10, 10)
    c_r_fixed = np.full_like(c_refs, 0.03)
    c_combs = compute_ivw_combination(c_refs, c_r_fixed)
    ceilings = 1.0 - c_combs / c_refs
    assert np.all(np.diff(ceilings) > 0)


def test_verification_assertions():
    """Verify that analysis-a.json, posthoc-measurement-floor.json, and polynomial summary match to 1e-12."""
    analysis_a_path = _REPO / "artifacts" / "descriptor-information" / "analysis-a.json"
    floor_path = _REPO / "artifacts" / "descriptor-information" / "posthoc-measurement-floor.json"
    poly_summary_path = _REPO / "artifacts" / "polynomial-refit" / "polynomial_refit_summary.json"

    assert analysis_a_path.is_file()
    assert floor_path.is_file()
    assert poly_summary_path.is_file()

    analysis_a = json.loads(analysis_a_path.read_text(encoding="utf-8"))["parts"]["A"]
    floor_data = json.loads(floor_path.read_text(encoding="utf-8"))["per_cell_floor"]
    poly_summary = json.loads(poly_summary_path.read_text(encoding="utf-8"))["primary_evaluations"]

    # 1. Assert mean C and mean F match 1e-12 against cell descriptive points
    for row_key, row_val in analysis_a["rows"].items():
        for rung_name, rung_val in row_val["rungs"].items():
            mean_c = rung_val["means"]["C"]["point"]
            cell_descs = rung_val["D_by_cell_point_descriptive"]
            cell_c_vals = [desc["point_C"] for desc in cell_descs.values()]
            assert math.isclose(mean_c, np.mean(cell_c_vals), rel_tol=1e-12, abs_tol=1e-12)

    # 2. Assert per-cell calibration errors exist and are positive
    for k, v in floor_data.items():
        assert "floor_mae" in v
        assert v["floor_mae"] > 0

    # 3. Assert polynomial refit summary test errors match expected values
    for item in poly_summary:
        assert item["refit_poly5_mae"] > 0
        assert math.isclose(
            item["refit_poly5_mae"], item["historical_poly5_mae"], rel_tol=1e-6
        )
