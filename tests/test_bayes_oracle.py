"""Unit tests for Bayes oracle computations and diagnostics.

Governing rule: docs/frozen-rules/2026-10-06-round9-follow-ups.md.
Tests synthetic one-dimensional generator against analytic Bayes risk.
Tests weighted median against fine grid posterior calculation.
Tests truncated-normal sampler bounds and analytic expectation.
Tests Equation 2 identity on synthetic numbers.
Tests surrogate fit recovery on exact degree-3 polynomials.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import warnings

import numpy as np
import pytest
from scipy.stats import norm, truncnorm
from sklearn.preprocessing import PolynomialFeatures

from tools.bayes_oracle import (
    RIDGE_ALPHAS,
    build_confusion_tables,
    check_equation_2,
    circuit_bootstrap_risks,
    compute_bayes_predictions_for_circuit,
    compute_effective_sample_size,
    fit_degree5_surrogate,
    get_exclusion_reason,
    get_governed_oracle_schedule,
    parse_args,
    partition_mapping_tuples,
    run_oracle_pipeline,
    sample_truncated_normal,
    scale_couplings,
    unscale_couplings,
    weighted_median,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_truncated_normal_bounds_and_mean():
    """Verify that truncated normal draws remain in bounds and match analytic mean."""
    rng = np.random.default_rng(20261006)
    lower = 0.2
    upper = 1.2
    test_mus = [0.15, 0.35, 0.70, 1.05, 1.25]
    test_scales = [0.01, 0.03, 0.10, 0.30]

    for mu in test_mus:
        for scale in test_scales:
            samples = sample_truncated_normal(mu, scale, lower, upper, size=100000, rng=rng)
            assert np.all(samples >= lower)
            assert np.all(samples <= upper)

            alpha = (lower - mu) / scale
            beta = (upper - mu) / scale
            phi_diff = norm.pdf(alpha) - norm.pdf(beta)
            denom = norm.cdf(beta) - norm.cdf(alpha)
            analytic_mean = mu + scale * (phi_diff / denom)

            sample_mean = float(np.mean(samples))
            sample_se = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
            assert abs(sample_mean - analytic_mean) < 4.0 * sample_se


def test_synthetic_1d_bayes_median_matches_analytic_risk():
    """Verify that the sampler C-star risk matches analytic risk on a 1D generator."""
    rng = np.random.default_rng(20261006)
    lower = 0.2
    upper = 1.2
    s = 0.05
    n_circuits = 400
    m_samples = 20000

    j_true = rng.uniform(lower, upper, size=n_circuits)
    z = rng.standard_normal(size=n_circuits)
    x_prime = j_true + s * z

    alpha = (lower - x_prime) / s
    beta = (upper - x_prime) / s
    phi_a = norm.cdf(alpha)
    phi_b = norm.cdf(beta)
    median_analytic = x_prime + s * norm.ppf(0.5 * (phi_a + phi_b))

    risk_analytic = float(np.mean(np.abs(median_analytic - j_true)))

    sample_medians = np.zeros(n_circuits, dtype=float)
    for i in range(n_circuits):
        samples_i = truncnorm.rvs(
            alpha[i], beta[i], loc=x_prime[i], scale=s, size=m_samples, random_state=rng
        )
        sample_medians[i] = float(np.median(samples_i))

    risk_sample = float(np.mean(np.abs(sample_medians - j_true)))
    risk_diff = abs(risk_sample - risk_analytic)

    assert risk_diff < 5e-4


def test_weighted_median_matches_grid_posterior():
    """Verify that weighted median matches fine grid posterior with Gaussian likelihood."""
    rng = np.random.default_rng(20261006)
    lower = 0.2
    upper = 1.2
    s = 0.08
    x_prime = 0.65
    sigma_r = 0.035
    r_obs = 0.42

    def e_func(j_val: np.ndarray) -> np.ndarray:
        return 0.75 * j_val - 0.05

    def y_func(j_val: np.ndarray) -> np.ndarray:
        return j_val * j_val

    grid = np.linspace(lower, upper, 250000)
    grid_prior = norm.pdf(grid, loc=x_prime, scale=s)
    grid_lik = norm.pdf(r_obs, loc=e_func(grid), scale=sigma_r)
    grid_posterior = grid_prior * grid_lik
    grid_posterior /= np.sum(grid_posterior)
    grid_y = y_func(grid)
    grid_med = weighted_median(grid_y, grid_posterior)

    alpha = (lower - x_prime) / s
    beta = (upper - x_prime) / s
    sample_draws = truncnorm.rvs(
        alpha, beta, loc=x_prime, scale=s, size=150000, random_state=rng
    )
    sample_weights = norm.pdf(r_obs, loc=e_func(sample_draws), scale=sigma_r)
    sample_y = y_func(sample_draws)
    sample_med = weighted_median(sample_y, sample_weights)

    assert abs(grid_med - sample_med) < 5e-4


def test_equation_2_identity_synthetic():
    """Verify that D = G-star + excess_C - excess_F holds within 1e-12."""
    rng = np.random.default_rng(42)
    n_points = 500

    c_vals = rng.uniform(0.001, 0.050, size=n_points)
    f_vals = rng.uniform(0.001, 0.050, size=n_points)
    d_vals = c_vals - f_vals

    c_star_vals = rng.uniform(0.0005, 0.020, size=n_points)
    f_star_vals = rng.uniform(0.0005, 0.020, size=n_points)
    g_star_vals = c_star_vals - f_star_vals

    excess_c = c_vals - c_star_vals
    excess_f = f_vals - f_star_vals

    eq2_reconstructed = g_star_vals + excess_c - excess_f
    max_err = float(np.max(np.abs(d_vals - eq2_reconstructed)))

    assert max_err < 1e-15


def test_surrogate_fit_recovers_degree3_polynomial():
    """Verify that degree-5 ridge recovers a degree-3 polynomial without noise to 1e-8."""
    rng = np.random.default_rng(20261006)
    n_train = 640
    n_val = 320
    n_dim = 3

    x_train = rng.uniform(0.2, 1.2, size=(n_train, n_dim))
    x_val = rng.uniform(0.2, 1.2, size=(n_val, n_dim))

    x_train_scaled = scale_couplings(x_train)
    x_val_scaled = scale_couplings(x_val)

    # Known degree-3 polynomial: y = 0.5 + x0 - 0.7*x1^2 + 0.3*x0*x2 + 0.2*x1^3
    def true_poly(x_mat: np.ndarray) -> np.ndarray:
        return (
            0.5
            + 1.0 * x_mat[:, 0]
            - 0.7 * (x_mat[:, 1] ** 2)
            + 0.3 * (x_mat[:, 0] * x_mat[:, 2])
            + 0.2 * (x_mat[:, 1] ** 3)
        )

    y_train = true_poly(x_train_scaled)
    y_val = true_poly(x_val_scaled)

    result = fit_degree5_surrogate(
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        alphas=RIDGE_ALPHAS,
        refit_train_val=True,
    )

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        pred_val = result["model"].predict(result["poly"].transform(x_val_scaled))
    max_abs_err = float(np.max(np.abs(pred_val - y_val)))

    assert max_abs_err < 1e-8


def test_effective_sample_size_properties():
    """Verify Kish effective sample size boundary conditions and sensitivity."""
    equal_weights = np.ones(100, dtype=float)
    assert math.isclose(compute_effective_sample_size(equal_weights), 100.0, rel_tol=1e-12)

    delta_weights = np.zeros(100, dtype=float)
    delta_weights[0] = 5.0
    assert math.isclose(compute_effective_sample_size(delta_weights), 1.0, rel_tol=1e-12)

    empty_weights = np.array([], dtype=float)
    assert compute_effective_sample_size(empty_weights) == 0.0


def test_circuit_bootstrap_risks_properties():
    """Verify circuit bootstrap risk calculations and monotonicity."""
    rng = np.random.default_rng(20261002)
    n_circ = 160
    n_cells = 4

    c_errors = rng.uniform(0.010, 0.020, size=(n_circ, n_cells))
    f_errors = rng.uniform(0.005, 0.010, size=(n_circ, n_cells))

    results = circuit_bootstrap_risks(c_errors, f_errors, draws=1000, seed=20261002)

    assert results["C_star"]["point"] > results["F_star"]["point"]
    assert results["G_star"]["point"] > 0.0
    assert results["G_star"]["excludes_zero_above"] is True
    assert results["G_star"]["excludes_zero_below"] is False
    assert results["G_star"]["interval"]["lower"] <= results["G_star"]["interval"]["upper"]


def test_bayes_oracle_mapping_integrity():
    """Verify that all mapped analysis files exist and contain valid structures."""
    mapping_path = REPO_ROOT / "tools" / "bayes_oracle_mapping.json"
    assert mapping_path.is_file()

    with open(mapping_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    fit_sets = doc.get("fit_sets", [])
    assert len(fit_sets) == 44

    for entry in fit_sets:
        p = REPO_ROOT / entry["path"]
        assert p.is_file(), f"File missing: {p}"
        with open(p, "r", encoding="utf-8") as af:
            adoc = json.load(af)
        rows = adoc.get("parts", {}).get("A", {}).get("rows", {})
        assert len(rows) > 0
        for rk, rdata in rows.items():
            assert "rungs" in rdata
            assert "dataset_seed" in rdata
            assert "family" in rdata


def test_confusion_tables_synthetic():
    """Verify confusion table construction and error metric accounting."""
    rows = [
        # False positive: R0 (no information) with F beats C
        {
            "fit_set": "synth_arm",
            "rung": "R0",
            "G_star_interval": {"lower": 0.0, "upper": 0.0},
            "pipeline_label": "F beats C",
        },
        # Miss: N1 (information) with not distinguished
        {
            "fit_set": "synth_arm",
            "rung": "N1",
            "G_star_interval": {"lower": 0.0001, "upper": 0.0003},
            "pipeline_label": "not distinguished",
        },
        # Wrong sign: N2 (information) with C beats F
        {
            "fit_set": "synth_arm",
            "rung": "N2",
            "G_star_interval": {"lower": 0.0002, "upper": 0.0005},
            "pipeline_label": "C beats F",
        },
        # Match: N3 (information) with F beats C
        {
            "fit_set": "synth_arm",
            "rung": "N3",
            "G_star_interval": {"lower": 0.0005, "upper": 0.0010},
            "pipeline_label": "F beats C",
        },
    ]

    res = build_confusion_tables(rows)
    assert "synth_arm" in res
    entry = res["synth_arm"]
    assert entry["total_cells"] == 4
    assert entry["false_positives"] == 1
    assert entry["misses"] == 1
    assert entry["wrong_signs"] == 1
    assert entry["matrix"]["no information"]["F beats C"] == 1
    assert entry["matrix"]["information"]["not distinguished"] == 1
    assert entry["matrix"]["information"]["C beats F"] == 1
    assert entry["matrix"]["information"]["F beats C"] == 1


def test_parser_contracts():
    """Verify governed CLI parser contracts and defaults."""
    # 1. Missing frozen rule raises SystemExit
    with pytest.raises(SystemExit):
        parse_args([])

    with pytest.raises(SystemExit):
        parse_args(["--eval-split", "test"])

    # 2. Validation split defaults
    args_val = parse_args(["--frozen-rule", "rule.md"])
    assert args_val.eval_split == "validation"
    assert args_val.levels == ["2048"]
    assert args_val.out == Path("build/bayes_oracle_results.json")
    assert args_val.summary_out is None

    # 3. Test split governed defaults
    args_test = parse_args(["--eval-split", "test", "--frozen-rule", "rule.md"])
    assert args_test.eval_split == "test"
    assert args_test.levels == ["256", "1024", "2048", "8192", "32768", "131072"]
    assert args_test.out == Path("artifacts/descriptor-information/round9/bayes-oracle.json")
    assert args_test.summary_out == Path("artifacts/descriptor-information/round9/bayes-oracle.md")

    # 4. Explicit overrides are preserved
    args_custom = parse_args([
        "--eval-split", "test",
        "--frozen-rule", "rule.md",
        "--levels", "2048",
        "--out", "custom.json",
        "--summary-out", "custom.md",
    ])
    assert args_custom.levels == ["2048"]
    assert args_custom.out == Path("custom.json")
    assert args_custom.summary_out == Path("custom.md")


def test_equation_2_loud_failures(tmp_path: Path):
    """Verify loud failure on missing files, missing keys, and empty tables."""
    # Missing analysis file raises FileNotFoundError
    bad_map = {
        "fit_sets": [
            {
                "fit_set": "bad",
                "shot_level": 2048,
                "path": "missing.json",
                "dataset_seeds": [101],
                "expected_rungs_by_family": {"tfi": ["R0"]},
                "strength_observed": True,
            }
        ]
    }
    with pytest.raises(FileNotFoundError):
        check_equation_2(bad_map, {}, tmp_path)

    # Missing parts.A.rows raises KeyError
    af = tmp_path / "analysis.json"
    with open(af, "w", encoding="utf-8") as f:
        json.dump({"parts": {}}, f)
    map_missing_rows = {
        "fit_sets": [
            {
                "fit_set": "bad_rows",
                "shot_level": 2048,
                "path": "analysis.json",
                "dataset_seeds": [101],
                "expected_rungs_by_family": {"tfi": ["R0"]},
                "strength_observed": True,
            }
        ]
    }
    with pytest.raises(KeyError):
        check_equation_2(map_missing_rows, {}, tmp_path)

    # Missing required C/F/point keys raises KeyError
    with open(af, "w", encoding="utf-8") as f:
        json.dump({
            "parts": {
                "A": {
                    "rows": {
                        "s101__tfi": {
                            "dataset_seed": 101,
                            "family": "tfi",
                            "rungs": {
                                "R0": {"means": {}, "D": {}}
                            },
                        }
                    }
                }
            }
        }, f)
    sched = {(101, "tfi", "R0", 2048)}
    with pytest.raises(KeyError):
        check_equation_2(map_missing_rows, {}, tmp_path, oracle_schedule=sched)

    # Missing scheduled oracle risk raises KeyError
    with open(af, "w", encoding="utf-8") as f:
        json.dump({
            "parts": {
                "A": {
                    "rows": {
                        "s101__tfi": {
                            "dataset_seed": 101,
                            "family": "tfi",
                            "rungs": {
                                "R0": {
                                    "means": {"C": {"point": 0.05}, "F": {"point": 0.05}},
                                    "D": {"point": 0.0, "interval": {"lower": 0, "upper": 0}},
                                }
                            },
                        }
                    }
                }
            }
        }, f)
    with pytest.raises(KeyError, match="Missing scheduled oracle risk"):
        check_equation_2(map_missing_rows, {}, tmp_path, oracle_schedule=sched)

    # Empty table raises ValueError
    empty_map = {"fit_sets": []}
    with pytest.raises(ValueError, match="empty"):
        check_equation_2(empty_map, {}, tmp_path)


def test_mapping_partition_and_exclusion_ledger():
    """Verify partition into 696 scheduled tuples (36 of them R0 at the exact level) and 504 excluded tuples."""
    mapping_path = REPO_ROOT / "tools" / "bayes_oracle_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    included, excluded = partition_mapping_tuples(doc, REPO_ROOT)
    assert len(included) == 696
    assert sum(1 for tup in included if str(tup[1]) == "exact") == 36
    assert all(tup[4] == "R0" for tup in included if str(tup[1]) == "exact")
    assert len(excluded) == 504

    # Verify 144 excluded tuples at 2048/exact: 36 R4 and 108 exact (every exact rung except R0)
    exc_2048_exact = [e for e in excluded if str(e["shot_level"]) in ("2048", "exact")]
    assert len(exc_2048_exact) == 144
    assert sum(1 for e in exc_2048_exact if e["rung"] == "R4") == 36
    assert sum(1 for e in exc_2048_exact if str(e["shot_level"]) == "exact") == 108

    for entry in excluded:
        assert "reason" in entry
        assert len(entry["reason"]) > 0


def test_surrogate_flag_propagation_inaccurate_model(tmp_path: Path):
    """Verify surrogate failure flag triggers at test MAE 1e-3 and propagates."""
    x_tr = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_tr = x_tr[:, 0] ** 2
    x_val = np.linspace(-0.9, 0.9, 10).reshape(-1, 1)
    y_val = x_val[:, 0] ** 2
    x_te = np.linspace(-0.8, 0.8, 10).reshape(-1, 1)
    y_te_clean = x_te[:, 0] ** 2

    # Accurate model: test MAE < 1e-3
    surr_clean = fit_degree5_surrogate(x_tr, y_tr, x_val, y_val, x_test=x_te, y_test=y_te_clean)
    assert surr_clean["test_mae"] < 1e-3
    assert surr_clean["flagged"] is False

    # Inaccurate model: test MAE >= 1e-3
    y_te_bad = y_te_clean + 0.05
    surr_bad = fit_degree5_surrogate(x_tr, y_tr, x_val, y_val, x_test=x_te, y_test=y_te_bad)
    assert surr_bad["test_mae"] >= 1e-3
    assert surr_bad["flagged"] is True

    # Boundary: at least 1e-3 is flagged, below is unflagged
    surr_at_1e3 = fit_degree5_surrogate(x_tr, y_tr, x_val, y_val, x_test=x_te, y_test=y_te_clean + 1e-3)
    assert surr_at_1e3["flagged"] is True

    surr_below_1e3 = fit_degree5_surrogate(x_tr, y_tr, x_val, y_val, x_test=x_te, y_test=y_te_clean + 0.99e-3)
    assert surr_below_1e3["flagged"] is False

    # Propagation through Equation 2 and confusion tables
    af = tmp_path / "analysis.json"
    with open(af, "w", encoding="utf-8") as f:
        json.dump({
            "parts": {
                "A": {
                    "rows": {
                        "s101__tfi": {
                            "dataset_seed": 101,
                            "family": "tfi",
                            "rungs": {
                                "R0": {
                                    "means": {"C": {"point": 0.05}, "F": {"point": 0.05}},
                                    "D": {"point": 0.0, "interval": {"lower": 0, "upper": 0}},
                                }
                            },
                        }
                    }
                }
            }
        }, f)
    mapping = {
        "fit_sets": [
            {
                "fit_set": "flagged_set",
                "shot_level": 2048,
                "path": "analysis.json",
                "dataset_seeds": [101],
                "expected_rungs_by_family": {"tfi": ["R0"]},
                "strength_observed": True,
            }
        ]
    }
    oracle_risks = {
        (101, "tfi", "R0", 2048, "strength_observed"): {
            "C_star": {"point": 0.0, "interval": {"lower": 0, "upper": 0}},
            "F_star": {"point": 0.0, "interval": {"lower": 0, "upper": 0}},
            "G_star": {"point": 0.0, "interval": {"lower": 0, "upper": 0}},
            "flagged_surrogate": True,
        }
    }
    sched = {(101, "tfi", "R0", 2048)}
    eq2_res = check_equation_2(mapping, oracle_risks, tmp_path, oracle_schedule=sched)
    assert eq2_res["rows"][0]["flagged_surrogate"] is True

    ctables = build_confusion_tables(eq2_res["rows"])
    assert ctables["flagged_set"]["flagged_entries"] == 1
    assert ctables["flagged_set"]["flagged_cells"] == 1


def test_full_synthetic_test_mode_pipeline(tmp_path: Path, monkeypatch):
    """Verify test split execution end to end with synthetic items."""
    data_dir = tmp_path / "data"
    seed = 101

    rule_dest = tmp_path / "docs" / "frozen-rules" / "2026-10-06-round9-follow-ups.md"
    rule_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "docs/frozen-rules/2026-10-06-round9-follow-ups.md", rule_dest)

    def make_items(shot_level: Any) -> list[dict]:
        items = []
        for fam in ("tfi", "heisenberg"):
            for split in ("train", "validation", "test"):
                for c in range(4):
                    cid = f"{fam}_{split}_{c}"
                    j_val = 0.3 + 0.1 * c
                    h_val = 0.4 + 0.1 * c
                    for s in ("L1", "L3"):
                        for o in ("z_mid", "zz_mid"):
                            y_val = 0.2 * (j_val - 0.7) + 0.1 * (h_val - 0.7)
                            e_val = y_val * 0.9
                            noisy = e_val if shot_level == "exact" else e_val + 0.001
                            item = {
                                "item_id": f"{fam}_{split}_{c}_{s}_{o}",
                                "circuit_id": cid,
                                "family": fam,
                                "split": split,
                                "severity": s,
                                "observable": o,
                                "ideal_expectation": y_val,
                                "noisy_expectation": noisy,
                            }
                            if fam == "tfi":
                                item["j"] = j_val
                                item["h"] = h_val
                            else:
                                item["jx"] = j_val
                                item["jy"] = h_val
                                item["jz"] = 0.5
                            items.append(item)
        return items

    for lvl in ("exact", 2048):
        p = data_dir / f"shots-{lvl}" / f"regen-shipped-s{seed}-n640"
        p.mkdir(parents=True, exist_ok=True)
        items = make_items(lvl)
        with open(p / "items.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = analysis_dir / "analysis-a.json"
    analysis_doc = {
        "parts": {
            "A": {
                "rows": {
                    f"s{seed}__tfi": {
                        "dataset_seed": seed,
                        "family": "tfi",
                        "rungs": {
                            "R0": {
                                "means": {
                                    "C": {"point": 0.05},
                                    "F": {"point": 0.05},
                                },
                                "D": {
                                    "point": 0.0,
                                    "interval": {"lower": -0.001, "upper": 0.001},
                                    "excludes_zero_above": False,
                                    "excludes_zero_below": False,
                                },
                            },
                            "N1": {
                                "means": {
                                    "C": {"point": 0.06},
                                    "F": {"point": 0.04},
                                },
                                "D": {
                                    "point": 0.02,
                                    "interval": {"lower": 0.01, "upper": 0.03},
                                    "excludes_zero_above": True,
                                    "excludes_zero_below": False,
                                },
                            },
                        },
                    }
                }
            }
        }
    }
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_doc, f)

    mapping_file = tmp_path / "mapping.json"
    mapping_doc = {
        "metadata": {"strength_evidence": "required"},
        "fit_sets": [
            {
                "fit_set": "synth_set_2048",
                "shot_level": 2048,
                "path": str(analysis_file.relative_to(tmp_path)),
                "dataset_seeds": [seed],
                "expected_rungs_by_family": {"tfi": ["R0", "N1"]},
                "strength_observed": True,
            }
        ]
    }
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_doc, f)
    from tools import bayes_oracle as bo_module
    synthetic_evidence = {"synth_set_2048": {
        "strength_observed": True,
        "records": [{"arm": "C", "has_strength_feature": True},
                    {"arm": "F", "has_strength_feature": True}]}}
    monkeypatch.setattr(bo_module, "load_strength_evidence", lambda path=None: synthetic_evidence)
    evidence_file = tmp_path / "evidence.json"
    evidence_file.write_text(json.dumps({"mapping_sha256": "0" * 64, "fit_sets": synthetic_evidence}))
    monkeypatch.setattr(bo_module, "STRENGTH_EVIDENCE_PATH", evidence_file)

    out_json = tmp_path / "results.json"
    out_md = tmp_path / "results.md"

    argv = [
        "--eval-split", "test",
        "--seeds", str(seed),
        "--levels", "2048",
        "--rungs", "R0", "N1",
        "--frozen-rule", str(rule_dest),
        "--mapping", str(mapping_file),
        "--data-primary", str(data_dir),
        "--data-fresh", str(data_dir),
        "--out", str(out_json),
        "--summary-out", str(out_md),
        "--workers", "1",
    ]
    args = parse_args(argv)

    import tools.bayes_oracle as bo_mod
    old_root = bo_mod.REPO_ROOT
    bo_mod.REPO_ROOT = tmp_path
    try:
        # Evidence built from another mapping stops the governed run before any test value is read.
        with pytest.raises(ValueError, match="different mapping file"):
            run_oracle_pipeline(args)
        evidence_file.write_text(json.dumps({
            "mapping_sha256": hashlib.sha256(mapping_file.read_bytes()).hexdigest(),
            "fit_sets": synthetic_evidence}))
        res = run_oracle_pipeline(args)
    finally:
        bo_mod.REPO_ROOT = old_root

    assert res["eval_split"] == "test"
    assert res["pseudo_test"] is False
    assert out_json.is_file()
    assert out_md.is_file()

    # Verify Finding 1: x_te passed to e-surrogate test diagnostics successfully
    surr_y = res["surrogates"][f"s{seed}_tfi"]["y"]["z_mid"]
    surr_e = res["surrogates"][f"s{seed}_tfi"]["e"]["z_mid_L1"]
    assert surr_y["test_mae"] is not None
    assert surr_e["test_mae"] is not None
    assert surr_y["test_mae"] < 1e-8
    assert surr_e["test_mae"] < 1e-8

    # Verify Equation 2 rows populated and verified
    assert res["equation_2"]["verified_to_1e12"] is True
    assert len(res["equation_2"]["rows"]) == 2
    assert "synth_set_2048" in res["confusion_tables"]


def test_strength_conditioning_invariance_and_sensitivity():
    """Verify that changing severity alters observed F-star but leaves hidden F-star invariant."""
    samples = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=float)
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly.fit(samples)

    from sklearn.linear_model import LinearRegression

    model_y = LinearRegression()
    model_y.fit(poly.transform(samples), np.array([0.2, 0.4, 0.6]))
    y_surrogates = {"z_mid": {"poly": poly, "model": model_y}}

    model_e_l1 = LinearRegression()
    model_e_l1.fit(poly.transform(samples), np.array([0.1, 0.2, 0.3]))

    model_e_l3 = LinearRegression()
    model_e_l3.fit(poly.transform(samples), np.array([0.5, 0.6, 0.7]))

    e_surrogates = {
        ("z_mid", "L1"): {"poly": poly, "model": model_e_l1},
        ("z_mid", "L3"): {"poly": poly, "model": model_e_l3},
    }

    it_l1 = {
        "item_id": "it_l1",
        "observable": "z_mid",
        "severity": "L1",
        "noisy_expectation": 0.25,
        "ideal_expectation": 0.30,
    }
    it_l3 = {
        "item_id": "it_l3",
        "observable": "z_mid",
        "severity": "L3",
        "noisy_expectation": 0.25,
        "ideal_expectation": 0.30,
    }

    c_preds, f_obs, ess_obs, f_hid, ess_hid = compute_bayes_predictions_for_circuit(
        [it_l1, it_l3], samples, y_surrogates, e_surrogates, shots=2048, family="tfi"
    )

    # C-star is independent of noise strength
    assert c_preds["it_l1"] == pytest.approx(c_preds["it_l3"])

    # Observed strength F-star conditions on reporting severity so it changes
    assert abs(f_obs["it_l1"] - f_obs["it_l3"]) > 1e-4

    # Hidden strength F-star marginalizes over L1 and L3 so it is invariant
    assert f_hid["it_l1"] == pytest.approx(f_hid["it_l3"], abs=1e-12)


def test_frozen_inventory_deletions_and_duplicates(tmp_path: Path):
    """Verify that deleting an expected seed, family, or rung or duplicating a tuple raises."""
    analysis_file = tmp_path / "analysis.json"

    def write_analysis(rows_dict: dict):
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump({"parts": {"A": {"rows": rows_dict}}}, f)

    valid_rows = {
        "s101__tfi": {
            "dataset_seed": 101,
            "family": "tfi",
            "rungs": {"R0": {}, "N1": {}},
        },
        "s101__heisenberg": {
            "dataset_seed": 101,
            "family": "heisenberg",
            "rungs": {"R0": {}, "N1": {}},
        },
        "s211__tfi": {
            "dataset_seed": 211,
            "family": "tfi",
            "rungs": {"R0": {}, "N1": {}},
        },
        "s211__heisenberg": {
            "dataset_seed": 211,
            "family": "heisenberg",
            "rungs": {"R0": {}, "N1": {}},
        },
    }

    mapping = {
        "fit_sets": [
            {
                "fit_set": "test_set",
                "shot_level": 2048,
                "path": "analysis.json",
                "dataset_seeds": [101, 211],
                "expected_rungs_by_family": {
                    "tfi": ["R0", "N1"],
                    "heisenberg": ["R0", "N1"],
                },
                "strength_observed": True,
            }
        ]
    }

    # Base case passes
    write_analysis(valid_rows)
    inc, exc = partition_mapping_tuples(mapping, tmp_path)
    assert len(inc) == 8

    # 1. Delete an expected seed (seed 211 omitted)
    rows_del_seed = {k: v for k, v in valid_rows.items() if not k.startswith("s211")}
    write_analysis(rows_del_seed)
    with pytest.raises(KeyError, match="Missing expected tuple"):
        partition_mapping_tuples(mapping, tmp_path)

    # 2. Delete an expected family (heisenberg omitted)
    rows_del_fam = {k: v for k, v in valid_rows.items() if not k.endswith("heisenberg")}
    write_analysis(rows_del_fam)
    with pytest.raises(KeyError, match="Missing expected tuple"):
        partition_mapping_tuples(mapping, tmp_path)

    # 3. Delete an expected rung (N1 omitted from s101__tfi)
    import copy
    rows_del_rung = copy.deepcopy(valid_rows)
    del rows_del_rung["s101__tfi"]["rungs"]["N1"]
    write_analysis(rows_del_rung)
    with pytest.raises(KeyError, match="Missing expected tuple"):
        partition_mapping_tuples(mapping, tmp_path)

    # 4. Duplicate a tuple (two rows with the same seed and family)
    rows_dup = copy.deepcopy(valid_rows)
    rows_dup["s101__tfi__dup"] = {
        "dataset_seed": 101,
        "family": "tfi",
        "rungs": {"R0": {}, "N1": {}},
    }
    write_analysis(rows_dup)
    with pytest.raises(ValueError, match="Duplicate"):
        partition_mapping_tuples(mapping, tmp_path)

    # 5. Unexpected extra rung raises ValueError
    rows_unexpected = copy.deepcopy(valid_rows)
    rows_unexpected["s101__tfi"]["rungs"]["N2"] = {}
    write_analysis(rows_unexpected)
    with pytest.raises(ValueError, match="Unexpected tuples"):
        partition_mapping_tuples(mapping, tmp_path)


def test_strength_flag_validation_disagreement(tmp_path: Path):
    """Verify that a disagreement between the mapping strength flag and run config raises."""
    analysis_file = tmp_path / "analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump({
            "parts": {
                "A": {
                    "rows": {
                        "s101__tfi": {
                            "dataset_seed": 101,
                            "family": "tfi",
                            "rungs": {"R0": {}},
                        }
                    }
                }
            }
        }, f)

    with open(tmp_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"strength_indicator": True}, f)

    bad_map = {
        "fit_sets": [
            {
                "fit_set": "disagree_set",
                "shot_level": 2048,
                "path": "analysis.json",
                "dataset_seeds": [101],
                "expected_rungs_by_family": {"tfi": ["R0"]},
                "strength_observed": False,
            }
        ]
    }
    with pytest.raises(ValueError, match="Strength flag mismatch"):
        partition_mapping_tuples(bad_map, tmp_path)

    good_map = {
        "fit_sets": [
            {
                "fit_set": "agree_set",
                "shot_level": 2048,
                "path": "analysis.json",
                "dataset_seeds": [101],
                "expected_rungs_by_family": {"tfi": ["R0"]},
                "strength_observed": True,
            }
        ]
    }
    inc, exc = partition_mapping_tuples(good_map, tmp_path)
    assert len(inc) == 1


def test_exact_level_exclusion_reason_wording():
    """Verify exact level exclusion wording matches the required specification."""
    reason = get_exclusion_reason("exact", "N1", "tfi")
    assert reason == "exact level (outside the finite-shot likelihood implementation)"




def test_redraw_is_per_item_and_variant_and_never_changes_c_star(monkeypatch):
    """A low-ESS item in one variant changes only that item's F* in that variant."""
    from tools import bayes_oracle as bo

    items = []
    for k, (sev, obs) in enumerate((("L1", "z_mid"), ("L1", "zz_mid"), ("L3", "z_mid"), ("L3", "zz_mid"))):
        items.append({"item_id": f"i{k}", "severity": sev, "observable": obs,
                      "ideal_expectation": 0.0, "j": 0.5, "h": 0.5, "noisy_expectation": 0.0})
    ids = [it["item_id"] for it in sorted(items, key=lambda x: (x["severity"], x["observable"]))]
    calls = {"n": 0}

    def fake_predictions(c_items, samples, y_s, e_s, shots, fam):
        calls["n"] += 1
        if calls["n"] == 1:
            c = {i: 0.1 for i in ids}
            f_obs = {i: 0.2 for i in ids}
            ess_obs = {i: (10.0 if i == ids[0] else 100.0) for i in ids}
            f_hid = {i: 0.3 for i in ids}
            ess_hid = {i: 100.0 for i in ids}
        else:
            c = {i: 0.9 for i in ids}
            f_obs = {i: 0.5 for i in ids}
            ess_obs = {i: 500.0 for i in ids}
            f_hid = {i: 0.7 for i in ids}
            ess_hid = {i: 500.0 for i in ids}
        return c, f_obs, ess_obs, f_hid, ess_hid

    monkeypatch.setattr(bo, "compute_bayes_predictions_for_circuit", fake_predictions)
    monkeypatch.setattr(bo, "draw_coupling_samples", lambda *a, **k: np.zeros((4, 2)))
    task = {"seed": 101, "lvl": 2048, "shots_int": 2048, "fam": "tfi", "rung": "N2",
            "sorted_circuits": ["c0"], "by_circuit": {"c0": items},
            "circ_lookup": {"c0": {"circuit_index": 0, "z": [0.0, 0.0]}},
            "y_surrs": {}, "e_surrs": {}}
    risks, log = bo.evaluate_rung_task(task)
    obs = risks[(101, "tfi", "N2", 2048, "strength_observed")]
    hid = risks[(101, "tfi", "N2", 2048, "strength_hidden")]
    assert obs["C_star"] == hid["C_star"]
    for cell in obs["per_cell_risks"]:
        assert obs["per_cell_risks"][cell]["C_star"] == hid["per_cell_risks"][cell]["C_star"]
    assert abs(obs["C_star"]["point"] - 0.1) < 1e-12
    # Observed variant: only the low-ESS item takes the redraw (0.5); the others keep 0.2.
    assert abs(obs["F_star"]["point"] - (0.5 + 0.2 + 0.2 + 0.2) / 4) < 1e-12
    # Hidden variant: no item was low, so every item keeps its first-draw value.
    assert abs(hid["F_star"]["point"] - 0.3) < 1e-12
    assert len(log) == 1 and log[0]["item_id"] == ids[0]
    assert log[0]["variant"] == "strength_observed"
    assert calls["n"] == 2


def test_strength_flag_needs_release_evidence_and_rejects_a_flipped_flag(tmp_path):
    """Flipping one derived entry's flag, or dropping its evidence, stops the inventory."""
    import copy
    from tools import bayes_oracle as bo

    mapping = json.loads((REPO_ROOT / "tools" / "bayes_oracle_mapping.json").read_text())
    evidence = bo.load_strength_evidence()
    for entry in mapping["fit_sets"]:
        assert evidence[entry["fit_set"]]["strength_observed"] == entry["strength_observed"]

    flipped = copy.deepcopy(mapping)
    for entry in flipped["fit_sets"]:
        if entry["fit_set"] == "follow_b_256_derived":
            entry["strength_observed"] = not entry["strength_observed"]
    with pytest.raises(ValueError, match="follow_b_256_derived"):
        bo.partition_mapping_tuples(flipped, REPO_ROOT)

    missing = copy.deepcopy(mapping)
    missing["fit_sets"].append(dict(missing["fit_sets"][0], fit_set="not_in_evidence"))
    with pytest.raises(KeyError, match="not_in_evidence"):
        bo.partition_mapping_tuples(missing, REPO_ROOT)
