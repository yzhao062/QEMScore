"""Bayes oracle risks and excess-error decomposition for spin chains.

Governing rule: docs/frozen-rules/2026-10-06-round9-follow-ups.md.
Fits ridge polynomial surrogates on scaled couplings.
Verifies the shot-noise variance model on training and validation splits.
Draws posterior coupling samples given rung descriptors.
Computes Bayes predictions under absolute loss for C-star and F-star.
Evaluates macro risks with a circuit-blocked bootstrap.
Compares fitted pipeline results against oracle truths via Equation 2.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, truncnorm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(REPO_ROOT))

from tools import descriptor_ladder as dl
from tools.descriptor_common import FEATURES

RULE_SEED = 20261002
SURROGATE_SEED = 20261006
BOOTSTRAP_DRAWS = 10000
DEFAULT_M_SAMPLES = 20000
DEFAULT_R5_SAMPLES = 200000
MIN_ESS = 50.0
REDRAW_FACTOR = 10

RIDGE_ALPHAS = tuple(10.0**k for k in range(-12, -1))

COUPLINGS = {
    "tfi": ("j", "h"),
    "heisenberg": ("jx", "jy", "jz"),
}
COUPLING_BOUNDS = (0.2, 1.2)

RUNG_INDEX = {
    "R0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "R3-TFI": 5,
    "R3-Heis": 6,
    "R5": 7,
}

NOISE_LEVELS = {
    "N1": 0.01,
    "N2": 0.03,
    "N3": 0.10,
    "N4": 0.30,
}

# The ladder's own constants define the rungs the fits read; keep them identical.
if {k: tuple(v) for k, v in dl.COUPLINGS.items()} != COUPLINGS or dict(dl.NOISE_LEVELS) != NOISE_LEVELS:
    raise RuntimeError("bayes_oracle constants differ from tools/descriptor_ladder.py")

SEEDS_PRIMARY = (101, 211, 307)
SEEDS_FRESH = (401, 503, 607)
ALL_SEEDS = (*SEEDS_PRIMARY, *SEEDS_FRESH)
FAMILIES = ("tfi", "heisenberg")
CELLS = (
    ("L1", "z_mid"),
    ("L1", "zz_mid"),
    ("L3", "z_mid"),
    ("L3", "zz_mid"),
)
RUNGS_2048_TFI = ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R5")
RUNGS_2048_HEIS = ("R0", "N1", "N2", "N3", "N4", "R3-Heis", "R5")
RUNGS_SWEEP = ("N1", "N2")
LEVELS = (256, 1024, 2048, 8192, 32768, 131072)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_couplings(couplings: np.ndarray) -> np.ndarray:
    """Scale couplings linearly from [0.2, 1.2] to [-1.0, 1.0]."""
    return (couplings - 0.7) / 0.5


def unscale_couplings(scaled: np.ndarray) -> np.ndarray:
    """Scale couplings linearly from [-1.0, 1.0] back to [0.2, 1.2]."""
    return scaled * 0.5 + 0.7


def fit_degree5_surrogate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    refit_train_val: bool = True,
) -> dict[str, Any]:
    """Fit a degree-5 ridge polynomial surrogate on scaled couplings."""
    poly = PolynomialFeatures(degree=5, include_bias=False)
    d_train = poly.fit_transform(x_train)
    d_val = poly.transform(x_val)

    best_alpha = None
    best_val_mae = float("inf")
    best_model = None

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(d_train, y_train)
            pred_val = model.predict(d_val)
            val_mae = float(np.mean(np.abs(pred_val - y_val)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha
                best_model = model

        final_model = best_model
        if refit_train_val:
            x_comb = np.vstack([x_train, x_val])
            y_comb = np.concatenate([y_train, y_val])
            d_comb = poly.transform(x_comb)
            final_model = Ridge(alpha=best_alpha, fit_intercept=True)
            final_model.fit(d_comb, y_comb)

        test_mae = None
        if x_test is not None and y_test is not None:
            d_test = poly.transform(x_test)
            pred_test = final_model.predict(d_test)
            test_mae = float(np.mean(np.abs(pred_test - y_test)))

    flagged = bool(test_mae is not None and test_mae >= 1e-3)
    return {
        "best_alpha": float(best_alpha),
        "validation_mae": float(best_val_mae),
        "test_mae": test_mae,
        "flagged": flagged,
        "poly": poly,
        "model": final_model,
    }


def verify_shot_noise_model(
    finite_items: list[dict[str, Any]],
    exact_items: list[dict[str, Any]],
    shots: int,
) -> dict[str, float]:
    """Verify that finite-shot noisy estimates match the binomial variance model."""
    exact_map = {
        (it["circuit_id"], it["observable"], it["severity"]): float(it["noisy_expectation"])
        for it in exact_items
    }
    z_scores = []
    for it in finite_items:
        key = (it["circuit_id"], it["observable"], it["severity"])
        if key not in exact_map:
            continue
        e = exact_map[key]
        r = float(it["noisy_expectation"])
        var = max((1.0 - e * e) / float(shots), 1e-15)
        sd = math.sqrt(var)
        z = (r - e) / sd
        z_scores.append(z)

    z_arr = np.asarray(z_scores, dtype=float)
    return {
        "n_items": int(len(z_arr)),
        "mean_z": float(np.mean(z_arr)),
        "sd_z": float(np.std(z_arr, ddof=1)),
        "max_abs_z": float(np.max(np.abs(z_arr))),
    }


def sample_truncated_normal(
    mu: float,
    scale: float,
    lower: float,
    upper: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a normal distribution truncated to the specified bounds."""
    alpha = (lower - mu) / scale
    beta = (upper - mu) / scale
    samples = truncnorm.rvs(alpha, beta, loc=mu, scale=scale, size=size, random_state=rng)
    return np.asarray(samples, dtype=float)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of values with associated non-negative weights."""
    idx = np.argsort(values)
    v_sorted = values[idx]
    w_sorted = weights[idx]
    w_cum = np.cumsum(w_sorted)
    total_w = w_cum[-1]
    cutoff = 0.5 * total_w
    k = int(np.searchsorted(w_cum, cutoff))
    if k < len(v_sorted) and abs(w_cum[k] - cutoff) < 1e-15:
        next_k = min(k + 1, len(v_sorted) - 1)
        return float(0.5 * (v_sorted[k] + v_sorted[next_k]))
    return float(v_sorted[min(k, len(v_sorted) - 1)])


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """Compute the Kish effective sample size for unnormalized weights."""
    sum_w = float(np.sum(weights))
    if sum_w <= 0.0 or math.isnan(sum_w):
        return 0.0
    sum_w_sq = float(np.sum(weights * weights))
    if sum_w_sq <= 0.0:
        return 0.0
    return (sum_w * sum_w) / sum_w_sq


def draw_coupling_samples(
    rung: str,
    family: str,
    true_couplings: dict[str, float],
    z_draws: tuple[float, ...],
    dataset_seed: int,
    circuit_index: int,
    sample_size: int = DEFAULT_M_SAMPLES,
    extra_seed_word: int | None = None,
) -> np.ndarray:
    """Draw coupling samples from the posterior given rung descriptors."""
    rung_idx = RUNG_INDEX[rung]
    seed_words = [SURROGATE_SEED, dataset_seed, circuit_index, rung_idx]
    if extra_seed_word is not None:
        seed_words.append(extra_seed_word)
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))

    c_names = COUPLINGS[family]
    n_dim = len(c_names)

    if rung in NOISE_LEVELS:
        s = NOISE_LEVELS[rung]
        samples = np.zeros((sample_size, n_dim), dtype=float)
        for i, c_name in enumerate(c_names):
            x_prime_i = true_couplings[c_name] + s * z_draws[i]
            samples[:, i] = sample_truncated_normal(
                x_prime_i, s, COUPLING_BOUNDS[0], COUPLING_BOUNDS[1], sample_size, rng
            )
        return samples

    if rung == "R3-TFI":
        if family != "tfi":
            raise ValueError("R3-TFI is defined only for transverse-field Ising")
        samples = np.zeros((sample_size, 2), dtype=float)
        samples[:, 0] = true_couplings["j"]
        samples[:, 1] = rng.uniform(COUPLING_BOUNDS[0], COUPLING_BOUNDS[1], size=sample_size)
        return samples

    if rung == "R3-Heis":
        if family != "heisenberg":
            raise ValueError("R3-Heis is defined only for Heisenberg")
        samples = np.zeros((sample_size, 3), dtype=float)
        samples[:, 0] = true_couplings["jx"]
        samples[:, 1] = true_couplings["jy"]
        samples[:, 2] = rng.uniform(COUPLING_BOUNDS[0], COUPLING_BOUNDS[1], size=sample_size)
        return samples

    if rung == "R5":
        return rng.uniform(COUPLING_BOUNDS[0], COUPLING_BOUNDS[1], size=(sample_size, n_dim))

    raise ValueError(f"Unsupported rung for sampling: {rung}")


def circuit_bootstrap_risks(
    circuit_c_errors: np.ndarray,
    circuit_f_errors: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = RULE_SEED,
) -> dict[str, Any]:
    """Compute circuit-blocked bootstrap intervals for Bayes risks."""
    n_circuits = circuit_c_errors.shape[0]
    c_macro_per_circuit = np.mean(circuit_c_errors, axis=1)
    f_macro_per_circuit = np.mean(circuit_f_errors, axis=1)

    c_point = float(np.mean(c_macro_per_circuit))
    f_point = float(np.mean(f_macro_per_circuit))
    g_point = c_point - f_point
    g_over_c_point = (g_point / c_point) if c_point > 0.0 else 0.0

    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, n_circuits, size=(draws, n_circuits))

    c_boot = np.mean(c_macro_per_circuit[resamples], axis=1)
    f_boot = np.mean(f_macro_per_circuit[resamples], axis=1)
    g_boot = c_boot - f_boot
    with np.errstate(divide="ignore", invalid="ignore"):
        g_over_c_boot = np.where(c_boot > 0.0, g_boot / c_boot, 0.0)

    def interval_dict(arr: np.ndarray, point: float) -> dict[str, Any]:
        low, high = np.percentile(arr, [2.5, 97.5])
        return {
            "point": float(point),
            "interval": {"lower": float(low), "upper": float(high)},
            "excludes_zero_above": bool(low > 0.0),
            "excludes_zero_below": bool(high < 0.0),
            "draw_sd": float(np.std(arr, ddof=1)),
        }

    per_cell_risks = {}
    for cell_idx, (sev, obs) in enumerate(CELLS):
        c_cell = circuit_c_errors[:, cell_idx]
        f_cell = circuit_f_errors[:, cell_idx]
        c_cell_point = float(np.mean(c_cell))
        f_cell_point = float(np.mean(f_cell))
        g_cell_point = c_cell_point - f_cell_point
        c_cell_boot = np.mean(c_cell[resamples], axis=1)
        f_cell_boot = np.mean(f_cell[resamples], axis=1)
        g_cell_boot = c_cell_boot - f_cell_boot
        cell_key = f"{sev}_{obs}"
        per_cell_risks[cell_key] = {
            "C_star": interval_dict(c_cell_boot, c_cell_point),
            "F_star": interval_dict(f_cell_boot, f_cell_point),
            "G_star": interval_dict(g_cell_boot, g_cell_point),
        }

    return {
        "C_star": interval_dict(c_boot, c_point),
        "F_star": interval_dict(f_boot, f_point),
        "G_star": interval_dict(g_boot, g_point),
        "G_star_over_C_star": interval_dict(g_over_c_boot, g_over_c_point),
        "per_cell_risks": per_cell_risks,
    }


def compute_bayes_predictions_for_circuit(
    circuit_items: list[dict[str, Any]],
    samples: np.ndarray,
    y_surrogates: dict[str, Any],
    e_surrogates: dict[tuple[str, str], Any],
    shots: int,
    family: str,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Compute C-star, F-star, and ESS for items belonging to one circuit.

    Evaluates both observed and hidden noise strength variants from the same coupling draws.
    """
    c_preds = {}
    f_preds_obs = {}
    ess_obs_by_item = {}
    f_preds_hid = {}
    ess_hid_by_item = {}

    samples_scaled = scale_couplings(samples)
    y_pred_samples = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for obs in ("z_mid", "zz_mid"):
            if obs in y_surrogates:
                surr_y = y_surrogates[obs]
                d_poly = surr_y["poly"].transform(samples_scaled)
                y_pred_samples[obs] = surr_y["model"].predict(d_poly)

        e_pred_samples = {}
        for obs in ("z_mid", "zz_mid"):
            for sev_key in ("L1", "L3"):
                if (obs, sev_key) in e_surrogates:
                    surr_e = e_surrogates[(obs, sev_key)]
                    d_poly_e = surr_e["poly"].transform(samples_scaled)
                    e_pred_samples[(obs, sev_key)] = surr_e["model"].predict(d_poly_e)

        for it in circuit_items:
            iid = str(it["item_id"])
            obs = str(it["observable"])
            sev = str(it["severity"])
            r = float(it["noisy_expectation"])

            y_s = y_pred_samples[obs]
            c_preds[iid] = float(np.median(y_s))

            # Observed strength variant
            e_s_obs = e_pred_samples[(obs, sev)]
            var_obs = np.maximum(1.0 - e_s_obs * e_s_obs, 1e-6) / float(shots)
            log_w_obs = -0.5 * np.log(2.0 * np.pi * var_obs) - 0.5 * (r - e_s_obs) ** 2 / var_obs
            log_w_obs_shifted = log_w_obs - np.max(log_w_obs)
            w_obs = np.exp(log_w_obs_shifted)
            ess_obs = compute_effective_sample_size(w_obs)
            ess_obs_by_item[iid] = float(ess_obs)
            f_preds_obs[iid] = weighted_median(y_s, w_obs)

            # Hidden strength variant: equal prior 1/2 on L1 and L3
            log_terms_hid = []
            for sev_k in ("L1", "L3"):
                e_s_k = e_pred_samples[(obs, sev_k)]
                var_k = np.maximum(1.0 - e_s_k * e_s_k, 1e-6) / float(shots)
                log_terms_hid.append(
                    -0.5 * np.log(2.0 * np.pi * var_k) - 0.5 * (r - e_s_k) ** 2 / var_k
                )
            log_w_hid = logsumexp(np.stack(log_terms_hid), axis=0) - math.log(2.0)
            log_w_hid_shifted = log_w_hid - np.max(log_w_hid)
            w_hid = np.exp(log_w_hid_shifted)
            ess_hid = compute_effective_sample_size(w_hid)
            ess_hid_by_item[iid] = float(ess_hid)
            f_preds_hid[iid] = weighted_median(y_s, w_hid)

    return c_preds, f_preds_obs, ess_obs_by_item, f_preds_hid, ess_hid_by_item


def evaluate_rung_task(
    task: dict[str, Any],
) -> tuple[dict[tuple[int, str, str, Any, str], dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one rung task across circuits for observed and hidden strength variants."""
    seed = int(task["seed"])
    lvl = task["lvl"]
    shots_int = int(task["shots_int"])
    fam = str(task["fam"])
    rung = str(task["rung"])
    sorted_circuits = task["sorted_circuits"]
    by_circuit = task["by_circuit"]
    circ_lookup = task["circ_lookup"]
    y_surrs = task["y_surrs"]
    e_surrs = task["e_surrs"]
    r5_shared_samples = task.get("r5_shared_samples")
    c_names = COUPLINGS[fam]
    n_circ = len(sorted_circuits)
    redrawn_circuits_log = []

    c_err_matrix_obs = np.zeros((n_circ, 4), dtype=float)
    f_err_matrix_obs = np.zeros((n_circ, 4), dtype=float)
    c_err_matrix_hid = np.zeros((n_circ, 4), dtype=float)
    f_err_matrix_hid = np.zeros((n_circ, 4), dtype=float)

    for ci, cid in enumerate(sorted_circuits):
        c_items = sorted(
            by_circuit[cid], key=lambda x: (x["severity"], x["observable"])
        )
        c_meta = circ_lookup[cid]
        c_idx = int(c_meta["circuit_index"])
        z_vals = tuple(c_meta["z"])
        true_c = {k: float(c_items[0][k]) for k in c_names}

        if rung == "R5":
            samples = r5_shared_samples
        else:
            samples = draw_coupling_samples(
                rung, fam, true_c, z_vals, seed, c_idx, sample_size=DEFAULT_M_SAMPLES
            )

        c_preds, f_preds_obs, ess_obs, f_preds_hid, ess_hid = compute_bayes_predictions_for_circuit(
            c_items, samples, y_surrs, e_surrs, shots_int, fam
        )

        # The rule redraws per item and per variant. C* never uses a redraw:
        # it is shared by both variants and must not depend on r.
        low_obs = [iid for iid, v in ess_obs.items() if v < MIN_ESS]
        low_hid = [iid for iid, v in ess_hid.items() if v < MIN_ESS]
        f_obs_use = dict(f_preds_obs)
        f_hid_use = dict(f_preds_hid)

        if low_obs or low_hid:
            redraw_size = (
                DEFAULT_R5_SAMPLES * REDRAW_FACTOR
                if rung == "R5"
                else DEFAULT_M_SAMPLES * REDRAW_FACTOR
            )
            samples_retry = draw_coupling_samples(
                rung,
                fam,
                true_c,
                z_vals,
                seed,
                c_idx,
                sample_size=redraw_size,
                extra_seed_word=10,
            )
            _c_unused, f_obs_retry, ess_obs_retry, f_hid_retry, ess_hid_retry = (
                compute_bayes_predictions_for_circuit(
                    c_items, samples_retry, y_surrs, e_surrs, shots_int, fam
                )
            )
            for variant, low, retry, ess0, ess1, target in (
                ("strength_observed", low_obs, f_obs_retry, ess_obs, ess_obs_retry, f_obs_use),
                ("strength_hidden", low_hid, f_hid_retry, ess_hid, ess_hid_retry, f_hid_use),
            ):
                for iid in low:
                    target[iid] = retry[iid]
                    redrawn_circuits_log.append({
                        "circuit_id": cid,
                        "item_id": iid,
                        "seed": seed,
                        "family": fam,
                        "rung": rung,
                        "shots": shots_int,
                        "variant": variant,
                        "initial_ess": float(ess0[iid]),
                        "redraw_ess": float(ess1[iid]),
                    })

        for cell_idx, it in enumerate(c_items):
            iid = str(it["item_id"])
            ideal = float(it["ideal_expectation"])
            c_error = abs(c_preds[iid] - ideal)
            c_err_matrix_obs[ci, cell_idx] = c_error
            c_err_matrix_hid[ci, cell_idx] = c_error
            f_err_matrix_obs[ci, cell_idx] = abs(f_obs_use[iid] - ideal)
            f_err_matrix_hid[ci, cell_idx] = abs(f_hid_use[iid] - ideal)

    risks_obs = circuit_bootstrap_risks(c_err_matrix_obs, f_err_matrix_obs, seed=RULE_SEED)
    risks_hid = circuit_bootstrap_risks(c_err_matrix_hid, f_err_matrix_hid, seed=RULE_SEED)

    y_flagged = any(surr.get("flagged", False) for surr in y_surrs.values())
    e_flagged = any(surr.get("flagged", False) for surr in e_surrs.values())
    flagged = bool(y_flagged or e_flagged)

    risks_obs["variant"] = "strength_observed"
    risks_obs["flagged_surrogate"] = flagged
    risks_hid["variant"] = "strength_hidden"
    risks_hid["flagged_surrogate"] = flagged

    result_risks = {
        (seed, fam, rung, lvl, "strength_observed"): risks_obs,
        (seed, fam, rung, lvl, "strength_hidden"): risks_hid,
    }
    return result_risks, redrawn_circuits_log


def is_scheduled_oracle_rung(shot_level: Any, rung: str, family: str) -> bool:
    """Return True if the tuple is scheduled by the Bayes oracle."""
    lvl_str = str(shot_level)
    if lvl_str == "2048":
        if family == "tfi":
            return rung in RUNGS_2048_TFI
        if family == "heisenberg":
            return rung in RUNGS_2048_HEIS
        return False
    if lvl_str in ("256", "1024", "8192", "32768", "131072"):
        return rung in RUNGS_SWEEP
    return False


def get_exclusion_reason(shot_level: Any, rung: str, family: str) -> str:
    """Return the reason why a mapped tuple is excluded from the oracle."""
    lvl_str = str(shot_level)
    if lvl_str == "exact":
        return "exact level (outside the finite-shot likelihood implementation)"
    if rung == "R4":
        return "R4 encoding (oracle does not model full phase encodings)"
    if lvl_str in ("256", "1024", "8192", "32768", "131072") and rung not in RUNGS_SWEEP:
        return f"sweep level {shot_level} unscheduled rung {rung} (oracle only schedules N1 and N2 at sweep levels)"
    if family == "tfi" and rung == "R3-Heis":
        return "mismatched family (R3-Heis on tfi)"
    if family == "heisenberg" and rung == "R3-TFI":
        return "mismatched family (R3-TFI on heisenberg)"
    return f"unscheduled rung {rung} at level {shot_level}"


def get_governed_oracle_schedule(
    seeds: tuple[int, ...] = ALL_SEEDS,
    levels: tuple[Any, ...] = LEVELS,
) -> set[tuple[int, str, str, Any]]:
    """Return the set of scheduled oracle tuples (seed, family, rung, level)."""
    sched = set()
    for s in seeds:
        for fam in FAMILIES:
            r3 = "R3-TFI" if fam == "tfi" else "R3-Heis"
            if 2048 in levels or "2048" in levels:
                for r in ("R0", "N1", "N2", "N3", "N4", r3, "R5"):
                    sched.add((s, fam, r, 2048))
                # R0 at the exact level needs no likelihood: C* = F* = 0 by construction.
                sched.add((s, fam, "R0", "exact"))
            for lvl in (256, 1024, 8192, 32768, 131072):
                if lvl in levels or str(lvl) in levels:
                    for r in RUNGS_SWEEP:
                        sched.add((s, fam, r, lvl))
    return sched


STRENGTH_EVIDENCE_PATH = Path(__file__).resolve().parent / "bayes_oracle_strength_evidence.json"
_STRENGTH_EVIDENCE: dict[str, Any] | None = None


def load_strength_evidence(path: Path | None = None) -> dict[str, Any]:
    """Load the committed strength evidence (fit set to C and F fit records)."""
    global _STRENGTH_EVIDENCE
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["fit_sets"]
    if _STRENGTH_EVIDENCE is None:
        if not STRENGTH_EVIDENCE_PATH.is_file():
            raise FileNotFoundError(f"Missing strength evidence file {STRENGTH_EVIDENCE_PATH}")
        with open(STRENGTH_EVIDENCE_PATH, "r", encoding="utf-8") as f:
            _STRENGTH_EVIDENCE = json.load(f)["fit_sets"]
    return _STRENGTH_EVIDENCE


def verify_strength_observed_flag(
    fit_entry: dict[str, Any],
    full_path: Path,
    analysis_doc: dict[str, Any],
    require_evidence: bool = False,
) -> None:
    """Check that fit entry strength_observed agrees with recorded config and features.

    The committed evidence file `tools/bayes_oracle_strength_evidence.json` holds,
    for every fit set, one C and one F fit record from the released fits (file
    name, SHA-256, and whether `noise_strength_L3` is a feature). A fit set without
    evidence, or whose evidence disagrees with the mapping flag, stops the oracle.
    """
    fit_set_name = fit_entry["fit_set"]
    flag = bool(fit_entry["strength_observed"])
    parent_dir = full_path.parent

    evidence = load_strength_evidence() if require_evidence else {fit_set_name: None}
    if require_evidence and fit_set_name not in evidence:
        raise KeyError(f"No strength evidence for fit set {fit_set_name}")
    ev = evidence[fit_set_name]
    if ev is None:
        ev = {"records": [{"arm": "C", "has_strength_feature": flag},
                          {"arm": "F", "has_strength_feature": flag}],
              "strength_observed": flag}
    arms = {rec.get("arm") for rec in ev.get("records", [])}
    if arms != {"C", "F"}:
        raise ValueError(f"Strength evidence for {fit_set_name} must hold one C and one F record")
    ev_flags = {bool(rec["has_strength_feature"]) for rec in ev["records"]}
    if ev_flags != {flag} or bool(ev.get("strength_observed")) != flag:
        raise ValueError(
            f"Strength flag mismatch for {fit_set_name}: mapping has {flag} "
            f"but the released fit records give {sorted(ev_flags)}."
        )

    for cfg_name in ("run_config.json", "run_config-partA.json", "run_config-partB.json"):
        cfg_file = parent_dir / cfg_name
        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                if isinstance(cfg_data, dict):
                    cfg_flag = bool(cfg_data.get("strength_indicator", False))
                    if flag != cfg_flag:
                        raise ValueError(
                            f"Strength flag mismatch for {fit_set_name}: mapping has "
                            f"{flag} but {cfg_file.name} records {cfg_flag}."
                        )
            except json.JSONDecodeError:
                pass

    meta_rc = analysis_doc.get("metadata", {}).get("run_config")
    if isinstance(meta_rc, dict) and "strength_indicator" in meta_rc:
        meta_flag = bool(meta_rc["strength_indicator"])
        if flag != meta_flag:
            raise ValueError(
                f"Strength flag mismatch for {fit_set_name}: mapping has "
                f"{flag} but analysis metadata records {meta_flag}."
            )

    fits_dir = parent_dir / "fits"
    if fits_dir.is_dir():
        fit_files = list(fits_dir.glob("*.json"))
        if fit_files:
            try:
                with open(fit_files[0], "r", encoding="utf-8") as f:
                    sample_fit = json.load(f)
                if isinstance(sample_fit, dict):
                    if "features" in sample_fit or "feature_names" in sample_fit:
                        feats = sample_fit.get("features") or sample_fit.get("feature_names") or []
                        has_feat = dl.STRENGTH_FEATURE_NAME in feats
                        if flag != has_feat:
                            raise ValueError(
                                f"Strength flag mismatch for {fit_set_name}: mapping has "
                                f"{flag} but sample fit has {dl.STRENGTH_FEATURE_NAME}={has_feat}."
                            )
                    if "strength_indicator" in sample_fit:
                        fit_flag = bool(sample_fit["strength_indicator"])
                        if flag != fit_flag:
                            raise ValueError(
                                f"Strength flag mismatch for {fit_set_name}: mapping has "
                                f"{flag} but sample fit records {fit_flag}."
                            )
            except json.JSONDecodeError:
                pass


def partition_mapping_tuples(
    mapping_data: dict[str, Any],
    repo_dir: Path,
    oracle_schedule: set[tuple[int, str, str, Any]] | None = None,
) -> tuple[set[tuple[str, Any, int, str, str]], list[dict[str, Any]]]:
    """Partition frozen mapping inventory into scheduled included tuples and excluded ledger."""
    if oracle_schedule is None:
        oracle_schedule = get_governed_oracle_schedule()

    expected_included = set()
    excluded_ledger = []

    for fit_entry in mapping_data.get("fit_sets", []):
        fit_set_name = fit_entry.get("fit_set")
        if not fit_set_name:
            raise KeyError("Mapping entry missing 'fit_set'")
        if "shot_level" not in fit_entry:
            raise KeyError(f"Mapping entry {fit_set_name} missing 'shot_level'")
        if "path" not in fit_entry:
            raise KeyError(f"Mapping entry {fit_set_name} missing 'path'")
        if "dataset_seeds" not in fit_entry:
            raise KeyError(f"Mapping entry {fit_set_name} missing 'dataset_seeds'")
        if "expected_rungs_by_family" not in fit_entry:
            raise KeyError(f"Mapping entry {fit_set_name} missing 'expected_rungs_by_family'")
        if "strength_observed" not in fit_entry:
            raise KeyError(f"Mapping entry {fit_set_name} missing 'strength_observed'")

        rel_path = fit_entry["path"]
        shot_level = fit_entry["shot_level"]
        seeds = [int(s) for s in fit_entry["dataset_seeds"]]
        rungs_by_fam = fit_entry["expected_rungs_by_family"]
        if not isinstance(rungs_by_fam, dict):
            raise TypeError(f"expected_rungs_by_family must be dict in {fit_set_name}")

        expected_tuples: list[tuple[str, Any, int, str, str]] = []
        for s in seeds:
            for fam in sorted(rungs_by_fam.keys()):
                rungs = rungs_by_fam[fam]
                for rung in rungs:
                    expected_tuples.append((fit_set_name, shot_level, s, fam, str(rung)))

        full_path = repo_dir / rel_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing analysis file: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            analysis_doc = json.load(f)

        verify_strength_observed_flag(
            fit_entry, full_path, analysis_doc,
            require_evidence=mapping_data.get("metadata", {}).get("strength_evidence") == "required",
        )

        rows = analysis_doc.get("parts", {}).get("A", {}).get("rows")
        if rows is None:
            raise KeyError(f"Missing parts.A.rows in analysis file: {full_path}")

        found_tuples: list[tuple[str, Any, int, str, str]] = []
        seen_row_keys = set()
        for rk, rdata in rows.items():
            if "dataset_seed" not in rdata or "family" not in rdata or "rungs" not in rdata:
                raise KeyError(f"Missing required row fields in {full_path} row {rk}")
            s = int(rdata["dataset_seed"])
            fam = str(rdata["family"])
            row_id = (s, fam)
            if row_id in seen_row_keys:
                raise ValueError(f"Duplicate row for seed {s} and family {fam} in {full_path}")
            seen_row_keys.add(row_id)

            rungs_dict = rdata["rungs"]
            if not isinstance(rungs_dict, dict):
                raise TypeError(f"rungs must be dict in {full_path} row {rk}")
            for rung in rungs_dict.keys():
                found_tuples.append((fit_set_name, shot_level, s, fam, str(rung)))

        found_counts = Counter(found_tuples)
        for tup in expected_tuples:
            count = found_counts[tup]
            if count == 0:
                raise KeyError(f"Missing expected tuple in {full_path}: {tup}")
            if count > 1:
                raise ValueError(f"Duplicate tuple in {full_path}: {tup} (count={count})")

        unexpected = set(found_tuples) - set(expected_tuples)
        if unexpected:
            raise ValueError(f"Unexpected tuples found in {full_path}: {sorted(unexpected)}")

        for tup in expected_tuples:
            _, sl, s, fam, rung = tup
            sched_key = (s, fam, rung, sl)
            if sched_key in oracle_schedule:
                expected_included.add(tup)
            else:
                excluded_ledger.append({
                    "fit_set": fit_set_name,
                    "shot_level": sl,
                    "dataset_seed": s,
                    "family": fam,
                    "rung": rung,
                    "reason": get_exclusion_reason(sl, rung, fam),
                })

    return expected_included, excluded_ledger


def check_equation_2(
    mapping_data: dict[str, Any],
    oracle_risks: dict[tuple[int, str, str, Any, str], dict[str, Any]],
    repo_dir: Path,
    oracle_schedule: set[tuple[int, str, str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify Equation 2 identity against fitted pipeline analysis records."""
    expected_included, excluded_ledger = partition_mapping_tuples(
        mapping_data, repo_dir, oracle_schedule=oracle_schedule
    )

    table = []
    produced_included = set()
    max_identity_error = 0.0

    for fit_entry in mapping_data.get("fit_sets", []):
        fit_set_name = fit_entry["fit_set"]
        rel_path = fit_entry["path"]
        shot_level = fit_entry["shot_level"]
        strength_observed = bool(fit_entry["strength_observed"])
        variant = "strength_observed" if strength_observed else "strength_hidden"
        full_path = repo_dir / rel_path
        if not full_path.is_file():
            raise FileNotFoundError(f"Missing analysis file: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            analysis_doc = json.load(f)

        rows = analysis_doc.get("parts", {}).get("A", {}).get("rows")
        if rows is None:
            raise KeyError(f"Missing parts.A.rows in {full_path}")

        for rk, rdata in rows.items():
            if "dataset_seed" not in rdata or "family" not in rdata or "rungs" not in rdata:
                raise KeyError(f"Missing required row fields in {full_path} row {rk}")
            seed = int(rdata["dataset_seed"])
            fam = str(rdata["family"])
            for rung_name, rung_data in rdata.get("rungs", {}).items():
                tup = (fit_set_name, shot_level, seed, fam, rung_name)
                if tup not in expected_included:
                    continue

                oracle_key = (seed, fam, rung_name, shot_level, variant)
                if oracle_key not in oracle_risks:
                    raise KeyError(f"Missing scheduled oracle risk for required key: {oracle_key}")

                means = rung_data.get("means")
                d_block = rung_data.get("D")
                if not isinstance(means, dict) or not isinstance(d_block, dict):
                    raise KeyError(f"Missing means or D dict in {full_path} row {rk} rung {rung_name}")
                if "C" not in means or "point" not in means["C"]:
                    raise KeyError(f"Missing means.C.point in {full_path} row {rk} rung {rung_name}")
                if "F" not in means or "point" not in means["F"]:
                    raise KeyError(f"Missing means.F.point in {full_path} row {rk} rung {rung_name}")
                if "point" not in d_block or "interval" not in d_block:
                    raise KeyError(f"Missing D.point or D.interval in {full_path} row {rk} rung {rung_name}")

                c_fit = float(means["C"]["point"])
                f_fit = float(means["F"]["point"])
                d_fit = float(d_block["point"])
                d_interval = d_block["interval"]

                pipeline_label = "not distinguished"
                if d_block.get("excludes_zero_above", False):
                    pipeline_label = "F beats C"
                elif d_block.get("excludes_zero_below", False):
                    pipeline_label = "C beats F"

                orisk = oracle_risks[oracle_key]
                if "C_star" not in orisk or "F_star" not in orisk or "G_star" not in orisk:
                    raise KeyError(f"Missing C_star, F_star, or G_star in oracle_risks[{oracle_key}]")

                c_star = float(orisk["C_star"]["point"])
                f_star = float(orisk["F_star"]["point"])
                g_star = float(orisk["G_star"]["point"])

                excess_c = c_fit - c_star
                excess_f = f_fit - f_star
                eq2_reconstructed = g_star + excess_c - excess_f
                diff = abs(d_fit - eq2_reconstructed)
                if diff > max_identity_error:
                    max_identity_error = diff

                table.append({
                    "fit_set": fit_set_name,
                    "shot_level": shot_level,
                    "dataset_seed": seed,
                    "family": fam,
                    "rung": rung_name,
                    "strength_observed": strength_observed,
                    "variant": variant,
                    "C_fitted": c_fit,
                    "F_fitted": f_fit,
                    "D_fitted": d_fit,
                    "D_interval": d_interval,
                    "pipeline_label": pipeline_label,
                    "C_star": c_star,
                    "F_star": f_star,
                    "G_star": g_star,
                    "G_star_interval": orisk["G_star"]["interval"],
                    "excess_C": excess_c,
                    "excess_F": excess_f,
                    "identity_error": diff,
                    "flagged_surrogate": bool(orisk.get("flagged_surrogate", False)),
                })
                produced_included.add(tup)

    if len(table) == 0:
        raise ValueError("Equation 2 evaluation table is empty; verification failed.")

    if produced_included != expected_included:
        missing = expected_included - produced_included
        extra = produced_included - expected_included
        raise ValueError(
            f"Produced included tuples do not match expected included set. "
            f"Missing {len(missing)} tuples, extra {len(extra)} tuples."
        )

    return {
        "verified_to_1e12": bool(max_identity_error < 1e-12),
        "max_identity_error": float(max_identity_error),
        "n_rows_evaluated": len(table),
        "rows": table,
        "excluded": excluded_ledger,
    }


def build_confusion_tables(
    eq2_table_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-tabulate oracle ground truth against pipeline classification labels."""
    by_fit_set = defaultdict(list)
    for row in eq2_table_rows:
        by_fit_set[row["fit_set"]].append(row)

    results = {}
    for fit_set_name, rows in by_fit_set.items():
        matrix = {
            "information": {"F beats C": 0, "not distinguished": 0, "C beats F": 0},
            "no information": {"F beats C": 0, "not distinguished": 0, "C beats F": 0},
            "undetermined": {"F beats C": 0, "not distinguished": 0, "C beats F": 0},
        }
        false_positives = 0
        misses = 0
        wrong_signs = 0
        flagged_entries = 0

        for r in rows:
            rung = r["rung"]
            g_low = float(r["G_star_interval"]["lower"])

            if rung == "R0":
                truth = "no information"
            elif g_low > 0.0:
                truth = "information"
            else:
                truth = "undetermined"

            plabel = r["pipeline_label"]
            matrix[truth][plabel] += 1

            if truth == "no information" and plabel == "F beats C":
                false_positives += 1
            if truth == "information" and plabel == "not distinguished":
                misses += 1
            if truth == "information" and plabel == "C beats F":
                wrong_signs += 1

            if r.get("flagged_surrogate", False):
                flagged_entries += 1

        results[fit_set_name] = {
            "strength_observed": rows[0].get("strength_observed", True) if rows else True,
            "variant": rows[0].get("variant", "strength_observed") if rows else "strength_observed",
            "total_cells": len(rows),
            "flagged_cells": flagged_entries,
            "flagged_entries": flagged_entries,
            "false_positives": false_positives,
            "misses": misses,
            "wrong_signs": wrong_signs,
            "matrix": matrix,
        }

    return results



def resolve_dataset_path(
    base_primary: Path,
    base_fresh: Path,
    seed: int,
    level: Any,
) -> Path:
    """Resolve the items.jsonl path for a given seed and shot level."""
    base = base_primary if seed in SEEDS_PRIMARY else base_fresh
    return base / f"shots-{level}" / f"regen-shipped-s{seed}-n640" / "items.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Bayes oracle script."""
    parser = argparse.ArgumentParser(
        description="Compute Bayes oracle risks for the spin-chain descriptor ladder."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(ALL_SEEDS),
        help="Dataset seeds to include (default: all six seeds).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Shot levels to evaluate (default: sweep levels for test, 2048 for validation).",
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        default=None,
        help="Rungs to evaluate (default: standard rungs for requested levels).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the computed JSON artifact.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional output path for Markdown summary report.",
    )
    parser.add_argument(
        "--frozen-rule",
        type=Path,
        required=True,
        help="Path to an existing frozen rule governing this execution.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("tools/bayes_oracle_mapping.json"),
        help="Path to the JSON mapping from fit sets to analysis files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 12),
        help="Number of worker processes for parallel evaluation.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["validation", "test"],
        default="validation",
        help="Split to evaluate: validation (pseudo-test) or test (governed).",
    )
    parser.add_argument(
        "--data-primary",
        type=Path,
        default=Path("/Users/yzhao062/qemscore-r8/assets/shot-sweep-v1/levels"),
        help="Base path to primary shot-sweep dataset levels.",
    )
    parser.add_argument(
        "--data-fresh",
        type=Path,
        default=Path("/Users/yzhao062/qemscore-r8/fresh/levels"),
        help="Base path to fresh shot-sweep dataset levels.",
    )
    parsed = parser.parse_args(argv)

    if parsed.eval_split == "test":
        if parsed.levels is None:
            parsed.levels = ["256", "1024", "2048", "8192", "32768", "131072"]
        if parsed.out is None:
            parsed.out = Path("artifacts/descriptor-information/round9/bayes-oracle.json")
        if parsed.summary_out is None:
            parsed.summary_out = Path("artifacts/descriptor-information/round9/bayes-oracle.md")
    else:
        if parsed.levels is None:
            parsed.levels = ["2048"]
        if parsed.out is None:
            parsed.out = Path("build/bayes_oracle_results.json")

    return parsed



def run_oracle_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full Bayes oracle pipeline according to configuration."""
    start_time = time.perf_counter()

    if not args.frozen_rule.is_file():
        raise FileNotFoundError(f"Frozen rule file not found: {args.frozen_rule}")

    is_governed_test = args.eval_split == "test"
    if is_governed_test:
        round9_rule = REPO_ROOT / "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
        if not round9_rule.is_file():
            raise PermissionError(
                "Governed outcome restriction: test split cannot be run until "
                "docs/frozen-rules/2026-10-06-round9-follow-ups.md is pushed."
            )

    selected_seeds = [int(s) for s in args.seeds]
    selected_levels = [int(l) if str(l).isdigit() else str(l) for l in args.levels]

    mapping_path = args.mapping
    if not mapping_path.is_file():
        mapping_path = REPO_ROOT / mapping_path
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_doc = json.load(f)
    if args.eval_split == "test" and mapping_doc.get("metadata", {}).get("strength_evidence") != "required":
        raise ValueError("A governed run needs a mapping whose metadata requires strength evidence")
    if args.eval_split == "test":
        with open(STRENGTH_EVIDENCE_PATH, "r", encoding="utf-8") as f:
            evidence_mapping_sha = json.load(f).get("mapping_sha256")
        if evidence_mapping_sha != hashlib.sha256(mapping_path.read_bytes()).hexdigest():
            raise ValueError("The strength evidence was built from a different mapping file")

    surrogate_reports = {}
    surrogate_models = {}
    exact_datasets = {}
    input_digests = {}

    for seed in selected_seeds:
        exact_path = resolve_dataset_path(args.data_primary, args.data_fresh, seed, "exact")
        if not exact_path.is_file():
            raise FileNotFoundError(f"Missing exact dataset file: {exact_path}")
        input_digests[str(exact_path)] = sha256_file(exact_path)

        exact_items = []
        with open(exact_path, "r", encoding="utf-8") as f:
            for line in f:
                exact_items.append(json.loads(line))
        exact_datasets[seed] = exact_items

        for fam in FAMILIES:
            c_names = COUPLINGS[fam]
            splits_circuits = {}
            for sp in ("train", "validation", "test"):
                seen = {}
                for it in exact_items:
                    if it["family"] == fam and it["split"] == sp:
                        cid = str(it["circuit_id"])
                        if cid not in seen:
                            seen[cid] = it
                splits_circuits[sp] = list(seen.values())

            tr_circ = splits_circuits["train"]
            val_circ = splits_circuits["validation"]
            te_circ = splits_circuits["test"] if is_governed_test else None
            if not tr_circ or not val_circ:
                continue

            x_tr = scale_couplings(np.array([[c[n] for n in c_names] for c in tr_circ]))

            x_val = scale_couplings(np.array([[c[n] for n in c_names] for c in val_circ]))
            x_te = scale_couplings(np.array([[c[n] for n in c_names] for c in te_circ])) if te_circ else None

            fam_key = f"s{seed}_{fam}"
            surrogate_reports[fam_key] = {"y": {}, "e": {}}
            surrogate_models[(seed, fam)] = {"y": {}, "e": {}}

            refit_flag = is_governed_test

            for obs in ("z_mid", "zz_mid"):
                y_tr_map = {
                    it["circuit_id"]: float(it["ideal_expectation"])
                    for it in exact_items
                    if it["family"] == fam and it["split"] == "train" and it["observable"] == obs
                }
                y_val_map = {
                    it["circuit_id"]: float(it["ideal_expectation"])
                    for it in exact_items
                    if it["family"] == fam and it["split"] == "validation" and it["observable"] == obs
                }
                y_tr = np.array([y_tr_map[c["circuit_id"]] for c in tr_circ])
                y_val = np.array([y_val_map[c["circuit_id"]] for c in val_circ])

                y_te = None
                if te_circ:
                    y_te_map = {
                        it["circuit_id"]: float(it["ideal_expectation"])
                        for it in exact_items
                        if it["family"] == fam and it["split"] == "test" and it["observable"] == obs
                    }
                    y_te = np.array([y_te_map[c["circuit_id"]] for c in te_circ])

                surr_y = fit_degree5_surrogate(
                    x_tr, y_tr, x_val, y_val, x_test=x_te, y_test=y_te, refit_train_val=refit_flag
                )
                surrogate_models[(seed, fam)]["y"][obs] = surr_y
                surrogate_reports[fam_key]["y"][obs] = {
                    "best_alpha": surr_y["best_alpha"],
                    "validation_mae": surr_y["validation_mae"],
                    "test_mae": surr_y["test_mae"],
                    "flagged": surr_y["flagged"],
                }

            for obs in ("z_mid", "zz_mid"):
                for sev in ("L1", "L3"):
                    e_tr_map = {
                        it["circuit_id"]: float(it["noisy_expectation"])
                        for it in exact_items
                        if it["family"] == fam
                        and it["split"] == "train"
                        and it["observable"] == obs
                        and it["severity"] == sev
                    }
                    e_val_map = {
                        it["circuit_id"]: float(it["noisy_expectation"])
                        for it in exact_items
                        if it["family"] == fam
                        and it["split"] == "validation"
                        and it["observable"] == obs
                        and it["severity"] == sev
                    }
                    e_tr = np.array([e_tr_map[c["circuit_id"]] for c in tr_circ])
                    e_val = np.array([e_val_map[c["circuit_id"]] for c in val_circ])

                    e_te = None
                    if te_circ:
                        e_te_map = {
                            it["circuit_id"]: float(it["noisy_expectation"])
                            for it in exact_items
                            if it["family"] == fam
                            and it["split"] == "test"
                            and it["observable"] == obs
                            and it["severity"] == sev
                        }
                        e_te = np.array([e_te_map[c["circuit_id"]] for c in te_circ])

                    surr_e = fit_degree5_surrogate(
                        x_tr, e_tr, x_val, e_val, x_test=x_te, y_test=e_te, refit_train_val=refit_flag
                    )
                    surrogate_models[(seed, fam)]["e"][(obs, sev)] = surr_e
                    surrogate_reports[fam_key]["e"][f"{obs}_{sev}"] = {
                        "best_alpha": surr_e["best_alpha"],
                        "validation_mae": surr_e["validation_mae"],
                        "test_mae": surr_e["test_mae"],
                        "flagged": surr_e["flagged"],
                    }

    shot_noise_checks = {}
    finite_items_cache = {}
    for lvl in selected_levels:
        if lvl == "exact":
            continue
        shots_int = int(lvl)
        all_finite_items = []
        for seed in selected_seeds:
            lvl_path = resolve_dataset_path(args.data_primary, args.data_fresh, seed, lvl)
            if not lvl_path.is_file():
                continue
            input_digests[str(lvl_path)] = sha256_file(lvl_path)
            items = []
            with open(lvl_path, "r", encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line))
            finite_items_cache[(seed, lvl)] = items
            train_val_items = [it for it in items if it["split"] in ("train", "validation")]
            exact_items = exact_datasets[seed]
            all_finite_items.extend(train_val_items)

        if all_finite_items:
            combined_exact = [
                it for s in selected_seeds for it in exact_datasets[s] if it["split"] in ("train", "validation")
            ]
            check_result = verify_shot_noise_model(all_finite_items, combined_exact, shots_int)
            shot_noise_checks[str(lvl)] = check_result

    oracle_risks = {}
    redrawn_circuits_log = []
    eval_split_name = args.eval_split
    tasks = []

    noise_tables = {}
    for seed in selected_seeds:
        sample_items = exact_datasets[seed]
        dataset_doc = {
            "seed": seed,
            "train": [it for it in sample_items if it["split"] == "train"],
            "validation": [it for it in sample_items if it["split"] == "validation"],
            "test": [it for it in sample_items if it["split"] == "test"],
        }
        noise_tables[seed] = dl.coupling_noise_table(dataset_doc)

    for lvl in selected_levels:
        if lvl == "exact":
            continue
        shots_int = int(lvl)
        for seed in selected_seeds:
            items_for_level = finite_items_cache.get((seed, lvl))
            if not items_for_level:
                lvl_path = resolve_dataset_path(args.data_primary, args.data_fresh, seed, lvl)
                with open(lvl_path, "r", encoding="utf-8") as f:
                    items_for_level = [json.loads(line) for line in f]
                finite_items_cache[(seed, lvl)] = items_for_level

            split_items = [it for it in items_for_level if it["split"] == eval_split_name]
            circ_table = noise_tables[seed]
            circ_lookup = {c["circuit_id"]: c for c in circ_table["circuits"]}

            for fam in FAMILIES:
                c_names = COUPLINGS[fam]
                fam_split_items = [it for it in split_items if it["family"] == fam]
                if not fam_split_items:
                    continue

                if shots_int == 2048:
                    candidate_rungs = RUNGS_2048_TFI if fam == "tfi" else RUNGS_2048_HEIS
                else:
                    candidate_rungs = RUNGS_SWEEP

                if args.rungs is not None:
                    candidate_rungs = tuple(r for r in candidate_rungs if r in args.rungs)

                by_circuit = defaultdict(list)
                for it in fam_split_items:
                    by_circuit[it["circuit_id"]].append(it)
                sorted_circuits = sorted(by_circuit.keys())
                n_circ = len(sorted_circuits)

                r5_shared_samples = None
                if "R5" in candidate_rungs:
                    r5_shared_samples = draw_coupling_samples(
                        "R5", fam, {}, (), seed, 999999, sample_size=DEFAULT_R5_SAMPLES
                    )

                y_surrs = surrogate_models[(seed, fam)]["y"]
                e_surrs = surrogate_models[(seed, fam)]["e"]

                for rung in candidate_rungs:
                    if rung == "R0":
                        y_rep = surrogate_reports[f"s{seed}_{fam}"]["y"]
                        metric_key = "test_mae" if is_governed_test else "validation_mae"
                        y_macro_surr = 0.5 * (y_rep["z_mid"][metric_key] + y_rep["zz_mid"][metric_key])
                        r0_per_cell = {}
                        for sev, obs in CELLS:
                            cell_key = f"{sev}_{obs}"
                            r0_per_cell[cell_key] = {
                                "C_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "F_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "G_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "surrogate_y_error": float(y_rep[obs][metric_key]),
                            }
                        y_flagged = any(surr.get("flagged", False) for surr in y_surrs.values())
                        for variant in ("strength_observed", "strength_hidden"):
                            oracle_risks[(seed, fam, rung, lvl, variant)] = {
                                "variant": variant,
                                "C_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "F_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "G_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "G_star_over_C_star": {"point": 0.0, "interval": {"lower": 0.0, "upper": 0.0}},
                                "surrogate_y_error": float(y_macro_surr),
                                "surrogate_error_type": metric_key,
                                "flagged_surrogate": bool(y_flagged),
                                "per_cell_risks": r0_per_cell,
                            }
                            if str(lvl) == "2048":
                                # The same zero-by-construction record serves R0 at the exact level.
                                exact_record = dict(oracle_risks[(seed, fam, rung, lvl, variant)])
                                exact_record["level_note"] = "exact level R0: zero by construction"
                                oracle_risks[(seed, fam, "R0", "exact", variant)] = exact_record
                        continue

                    tasks.append({
                        "seed": seed,
                        "lvl": lvl,
                        "shots_int": shots_int,
                        "fam": fam,
                        "rung": rung,
                        "sorted_circuits": sorted_circuits,
                        "by_circuit": dict(by_circuit),
                        "circ_lookup": circ_lookup,
                        "y_surrs": y_surrs,
                        "e_surrs": e_surrs,
                        "r5_shared_samples": r5_shared_samples,
                    })

    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for risks_map, redrawn in executor.map(evaluate_rung_task, tasks):
                oracle_risks.update(risks_map)
                redrawn_circuits_log.extend(redrawn)
    else:
        for task in tasks:
            risks_map, redrawn = evaluate_rung_task(task)
            oracle_risks.update(risks_map)
            redrawn_circuits_log.extend(redrawn)

    active_schedule = {
        key for key in get_governed_oracle_schedule()
        if key[0] in selected_seeds and (
            key[3] in selected_levels or str(key[3]) in selected_levels
            or (key[2] == "R0" and key[3] == "exact"
                and (2048 in selected_levels or "2048" in selected_levels))
        )
    }
    eq2_result = check_equation_2(mapping_doc, oracle_risks, REPO_ROOT, oracle_schedule=active_schedule)
    confusion_tables = build_confusion_tables(eq2_result["rows"])

    total_time = time.perf_counter() - start_time

    serialized_risks = {}
    for (s, fam, r, l, variant), r_data in oracle_risks.items():
        key_str = f"s{s}__{fam}__{r}__shots_{l}__{variant}"
        serialized_risks[key_str] = r_data

    result_doc = {
        "schema": "bayes_oracle_v1",
        "frozen_rule": "docs/frozen-rules/2026-10-06-round9-follow-ups.md",
        "governing_rule_provided": str(args.frozen_rule),
        "rule_file_sha256": sha256_file(args.frozen_rule),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "mapping_file_sha256": sha256_file(mapping_path),
        "eval_split": eval_split_name,
        "pseudo_test": not is_governed_test,
        "seconds": float(total_time),
        "inputs": input_digests,
        "surrogates": surrogate_reports,
        "shot_noise_checks": shot_noise_checks,
        "oracle_risks": serialized_risks,
        "redrawn_circuits": redrawn_circuits_log,
        "equation_2": {
            "verified_to_1e12": eq2_result["verified_to_1e12"],
            "max_identity_error": eq2_result["max_identity_error"],
            "n_rows_evaluated": eq2_result["n_rows_evaluated"],
            "n_excluded_evaluated": len(eq2_result["excluded"]),
            "excluded": eq2_result["excluded"],
            "rows": eq2_result["rows"] if is_governed_test else [],
            "note": (
                "Full rows recorded only on governed test runs to prevent leaking test numbers."
                if not is_governed_test
                else "Full Equation 2 evaluation across fitted pipelines."
            ),
        },
        "confusion_tables": confusion_tables if is_governed_test else {},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_lines = [
            "# Bayes Oracle Evaluation Summary",
            "",
            f"- Evaluation split: {eval_split_name} (pseudo-test: {not is_governed_test})",
            f"- Elapsed time: {total_time:.2f} seconds",
            f"- Dataset seeds: {selected_seeds}",
            f"- Shot levels: {selected_levels}",
            "",
            "## Shot-Noise Model Checks",
            "",
        ]
        for lvl_str, chk in shot_noise_checks.items():
            summary_lines.append(
                f"- Level {lvl_str}: N={chk['n_items']}, mean(z)={chk['mean_z']:.4f}, "
                f"sd(z)={chk['sd_z']:.4f}, max(|z|)={chk['max_abs_z']:.4f}"
            )
        summary_lines.append("")

        total_surrogates = 0
        total_flagged_surrogates = 0
        for fam_data in surrogate_reports.values():
            for surr_type in ("y", "e"):
                for s_entry in fam_data.get(surr_type, {}).values():
                    total_surrogates += 1
                    if s_entry.get("flagged", False):
                        total_flagged_surrogates += 1

        total_oracle_rows = len(oracle_risks)
        flagged_oracle_rows = sum(1 for r in oracle_risks.values() if r.get("flagged_surrogate", False))

        total_eq2_rows = len(eq2_result["rows"])
        flagged_eq2_rows = sum(1 for r in eq2_result["rows"] if r.get("flagged_surrogate", False))

        total_confusion_entries = sum(res["total_cells"] for res in confusion_tables.values())
        flagged_confusion_entries = sum(res.get("flagged_entries", 0) for res in confusion_tables.values())

        summary_lines.append("## Surrogate Failure Flags (test MAE >= 1e-3)")
        summary_lines.append("")
        summary_lines.append(f"- Flagged surrogates: {total_flagged_surrogates} / {total_surrogates}")
        summary_lines.append(f"- Flagged oracle rows: {flagged_oracle_rows} / {total_oracle_rows}")
        summary_lines.append(f"- Flagged Equation 2 rows: {flagged_eq2_rows} / {total_eq2_rows}")
        summary_lines.append(f"- Flagged confusion entries: {flagged_confusion_entries} / {total_confusion_entries}")
        summary_lines.append("")
        summary_lines.append("## Excluded Mapping Tuples")
        summary_lines.append("")
        summary_lines.append(f"- Total excluded tuples: {len(eq2_result['excluded'])}")
        summary_lines.append("")
        summary_lines.append("## Equation 2 Identity Verification")
        summary_lines.append("")
        summary_lines.append(f"- Verified to 1e-12: {eq2_result['verified_to_1e12']}")
        summary_lines.append(f"- Max identity error: {eq2_result['max_identity_error']:.2e}")
        summary_lines.append(f"- Evaluated pipeline rows: {eq2_result['n_rows_evaluated']}")
        summary_lines.append("")
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")


    return result_doc


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    args = parse_args(argv)
    run_oracle_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
