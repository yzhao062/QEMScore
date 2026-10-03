#!/usr/bin/env python3
"""Independent analysis script for the descriptor-information experiment.

Frozen rule: docs/frozen-rules/2026-10-02-descriptor-information.md

Implements:
- Point estimates for macro MAE of C, F, M, A, R.
- D = C - F (mean and 95% bootstrap interval).
- D/C computed as ratio of means inside each draw (mean and 95% bootstrap interval).
- E = M - F (error removed on top of measurement).
- S = (A - C)/(A - F) where defined (only when A - F is positive in every draw).
- Paired contrasts versus R0 (Part A), B-partial (Part B), or NC-R0 (Near-Clifford):
  Delta D and Delta(D/C) using the same seed and circuit indices in each draw.
- Learner-fixed secondary D (MLP candidate in both arms) and selection counts.
- Descriptive D per severity cell.
- Classification label per cell (Measurement adds, Measurement hurts, Not distinguished),
  largest non-excluded D/C, and rung-level statement per family / part.
- Deterministic two-stage percentile bootstrap with fresh default_rng(seed)
  per row and per quantity.
- Exclusion of anchor fits outside learner seeds 1..20 and disclosure in output.
- Explicit recording of skipped files and failed fits.
- Disclosure of dropped cells per row.
- Support for multiple cache directories and configurable --bootstrap-seed.
- Self-test mode (--self-test) with synthetic test datasets and known D.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

RULE_FILE = "docs/frozen-rules/2026-10-02-descriptor-information.md"
RULE_SEED = 20261002
DEFAULT_DRAWS = 10000

# Canonical definitions per rule and descriptor_ladder.py
PART_A_RUNGS = ("R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis", "R4", "R5")
NC_RUNGS = ("NC-R0", "NC-none")
PART_B_RUNGS = ("B-partial", "B-complete", "B-none")
DATASET_SEEDS = (101, 211, 307)
VALID_LEARNER_SEEDS = tuple(range(1, 21))


def generate_bootstrap_indices(
    n_seeds: int,
    n_circuits: int,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = RULE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two-stage bootstrap draw indices.

    RNG scheme per frozen rule:
    fresh numpy.random.default_rng(seed) per row and quantity.
    Each draw calls integers(0, n_seeds, n_seeds), then integers(0, n_circuits, n_circuits).
    """
    rng = np.random.default_rng(seed)
    seed_draws = np.empty((n_draws, n_seeds), dtype=np.int32)
    circ_draws = np.empty((n_draws, n_circuits), dtype=np.int32)
    for b in range(n_draws):
        seed_draws[b] = rng.integers(0, n_seeds, n_seeds)
        circ_draws[b] = rng.integers(0, n_circuits, n_circuits)
    return seed_draws, circ_draws


def compute_resampled_means(
    circ_errors: np.ndarray,
    seed_draws: np.ndarray,
    circ_draws: np.ndarray,
) -> np.ndarray:
    """Compute resampled macro error across bootstrap draws.

    circ_errors: shape (n_seeds, n_circuits) for seed-varying models,
                 or (n_circuits,) for fixed models (e.g. affine A or raw R).
    Returns: array of shape (n_draws,) containing resampled mean error for each draw.
    """
    n_draws = len(circ_draws)
    if circ_errors.ndim == 1:
        # Fixed model: only circuits resampled
        res = np.empty(n_draws, dtype=float)
        for b in range(n_draws):
            res[b] = np.mean(circ_errors[circ_draws[b]])
        return res

    # 2D array: (n_seeds, n_circuits)
    # circ_means: shape (n_draws, n_seeds)
    # circ_means[b, k] is the mean over resampled circuits for seed k in draw b
    circ_means = np.empty((n_draws, circ_errors.shape[0]), dtype=float)
    for b in range(n_draws):
        c_b = circ_draws[b]
        circ_means[b] = np.mean(circ_errors[:, c_b], axis=1)

    # Average across resampled seeds
    res = np.take_along_axis(circ_means, seed_draws, axis=1).mean(axis=1)
    return res


def compute_bootstrap_ci(draws: np.ndarray) -> list[float]:
    """Pointwise 95% bootstrap percentile interval (2.5% to 97.5%)."""
    p2_5, p97_5 = np.percentile(draws, [2.5, 97.5])
    return [float(p2_5), float(p97_5)]


class DataLoader:
    """Discovers, caches, and organizes fit files and dataset caches."""

    def __init__(
        self,
        fits_dirs: Sequence[Path | str],
        cache_dirs: Sequence[Path | str],
    ) -> None:
        self.fits_dirs = [Path(d).resolve() for d in fits_dirs]
        self.cache_dirs = [Path(d).resolve() for d in cache_dirs]
        self._cache_store: dict[str, dict] = {}
        self.fits: list[dict] = []
        self.skipped_files: list[dict[str, str]] = []
        self.failed_fits: list[dict[str, Any]] = []
        self.excluded_anchors: list[dict[str, Any]] = []
        self._scan_fits()

    def _scan_fits(self) -> None:
        for fdir in self.fits_dirs:
            if not fdir.exists():
                continue
            for json_path in sorted(fdir.glob("*.json")):
                if json_path.name.endswith(".order.json"):
                    self.skipped_files.append({
                        "path": str(json_path),
                        "reason": "order file, not a fit result",
                    })
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                except Exception as exc:
                    self.skipped_files.append({
                        "path": str(json_path),
                        "reason": f"invalid JSON: {exc}",
                    })
                    continue

                if not isinstance(meta, dict):
                    self.skipped_files.append({
                        "path": str(json_path),
                        "reason": "JSON root is not a mapping",
                    })
                    continue

                missing_keys = [k for k in ("part", "rung", "arm") if k not in meta]
                if missing_keys:
                    self.skipped_files.append({
                        "path": str(json_path),
                        "reason": f"missing metadata fields: {missing_keys}",
                    })
                    continue

                npz_path = json_path.with_suffix(".npz")
                if not npz_path.exists():
                    self.skipped_files.append({
                        "path": str(json_path),
                        "reason": f"matching npz file {npz_path.name} not found",
                    })
                    continue

                meta["_json_path"] = str(json_path)
                meta["_npz_path"] = str(npz_path)

                # Check for fit errors
                if meta.get("error") is not None:
                    self.failed_fits.append({
                        "path": str(json_path),
                        "key": meta.get("key"),
                        "part": meta.get("part"),
                        "rung": meta.get("rung"),
                        "arm": meta.get("arm"),
                        "learner_seed": meta.get("learner_seed"),
                        "error": str(meta["error"]),
                    })
                    continue

                # Check learner seed for arms that run across learner seeds
                arm = str(meta["arm"])
                if arm in ("F", "C", "P", "M"):
                    l_seed = meta.get("learner_seed")
                    if l_seed is not None and l_seed not in VALID_LEARNER_SEEDS:
                        self.excluded_anchors.append({
                            "path": str(json_path),
                            "key": meta.get("key"),
                            "part": meta.get("part"),
                            "rung": meta.get("rung"),
                            "arm": meta.get("arm"),
                            "learner_seed": l_seed,
                            "reason": f"learner_seed {l_seed} outside 1..20 (anchor/reexport fit)",
                        })
                        continue

                self.fits.append(meta)

    def load_cache(self, key: str, dataset_seed: int) -> dict:
        if key in self._cache_store:
            return self._cache_store[key]

        target_file: Path | None = None
        # 1. Exact match across cache dirs
        for cdir in self.cache_dirs:
            if not cdir.exists():
                continue
            direct = cdir / f"{key}.pkl"
            if direct.exists():
                target_file = direct
                break

        # 2. Key glob match across cache dirs
        if target_file is None:
            for cdir in self.cache_dirs:
                if not cdir.exists():
                    continue
                candidates = list(cdir.glob(f"*{key}*.pkl"))
                if candidates:
                    target_file = candidates[0]
                    break

        # 3. Fallback to seed glob only if key not found anywhere
        if target_file is None:
            for cdir in self.cache_dirs:
                if not cdir.exists():
                    continue
                candidates = list(cdir.glob(f"*s{dataset_seed}*.pkl"))
                if candidates:
                    target_file = candidates[0]
                    break

        if target_file is None or not target_file.exists():
            raise FileNotFoundError(
                f"Cache file for key {key!r} (seed {dataset_seed}) not found in {[str(d) for d in self.cache_dirs]}"
            )

        with open(target_file, "rb") as handle:
            payload = pickle.load(handle)
        self._cache_store[key] = payload
        return payload


def evaluate_circuit_errors(
    test_items: list[dict],
    family_filter: str | None,
    predictions: np.ndarray,
    observable_filter: str | None = None,
    dropped_cells: Sequence[str] = (),
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Compute macro error per circuit and per severity cell.

    test_items: full list of test items from cache.
    family_filter: family name or None.
    predictions: shape (n_items,) or (n_seeds, n_items).
    observable_filter: e.g. 'zz_mid' for Part B.
    dropped_cells: list of dropped cell names to exclude.

    Returns:
    - sorted_circuit_ids: list of unique circuit IDs in sorted order.
    - circ_macro_errors: shape (n_circuits,) or (n_seeds, n_circuits).
    - circ_sev_errors: dict of severity -> shape (n_circuits,) or (n_seeds, n_circuits).
    """
    dropped_set = set(dropped_cells)
    valid_indices = []
    for idx, it in enumerate(test_items):
        if family_filter is not None and it.get("family") != family_filter:
            continue
        if observable_filter is not None and it.get("observable") != observable_filter:
            continue
        # Check cell drop disclosure
        cell_name = f"{it.get('noise_family')}/{it.get('severity')}/{it.get('observable')}"
        if cell_name in dropped_set or f"{it.get('severity')}/{it.get('observable')}" in dropped_set:
            continue
        valid_indices.append(idx)

    selected_items = [test_items[i] for i in valid_indices]
    y_all = np.array([float(it["ideal_expectation"]) for it in selected_items])

    if predictions.ndim == 1:
        preds = predictions[valid_indices]
        abs_err = np.abs(preds - y_all)
    else:
        preds = predictions[:, valid_indices]
        abs_err = np.abs(preds - y_all[None, :])

    sorted_circuits = sorted(list(set(str(it["circuit_id"]) for it in selected_items)))
    n_circuits = len(sorted_circuits)

    # Group item indices by circuit
    circuit_item_map: dict[str, list[int]] = defaultdict(list)
    for sub_idx, it in enumerate(selected_items):
        circuit_item_map[str(it["circuit_id"])].append(sub_idx)

    all_cells = sorted(list(set((it["severity"], it["observable"]) for it in selected_items)))
    all_severities = sorted(list(set(it["severity"] for it in selected_items)))

    is_2d = abs_err.ndim == 2
    n_seeds = abs_err.shape[0] if is_2d else 1

    circ_macro = np.zeros((n_seeds, n_circuits) if is_2d else (n_circuits,), dtype=float)
    circ_sev = {
        sev: np.zeros((n_seeds, n_circuits) if is_2d else (n_circuits,), dtype=float)
        for sev in all_severities
    }

    for c_i, cid in enumerate(sorted_circuits):
        sub_indices = circuit_item_map[cid]
        c_items = [selected_items[i] for i in sub_indices]

        # Macro across declared cells for this circuit
        cell_errs = []
        for cell in all_cells:
            cell_sub_indices = [
                sub_indices[j]
                for j, it in enumerate(c_items)
                if (it["severity"], it["observable"]) == cell
            ]
            if cell_sub_indices:
                if is_2d:
                    cell_errs.append(np.mean(abs_err[:, cell_sub_indices], axis=1))
                else:
                    cell_errs.append(np.mean(abs_err[cell_sub_indices]))

        if cell_errs:
            if is_2d:
                circ_macro[:, c_i] = np.mean(cell_errs, axis=0)
            else:
                circ_macro[c_i] = np.mean(cell_errs)

        # Per severity cell
        for sev in all_severities:
            sev_sub_indices = [
                sub_indices[j]
                for j, it in enumerate(c_items)
                if it["severity"] == sev
            ]
            if sev_sub_indices:
                if is_2d:
                    circ_sev[sev][:, c_i] = np.mean(abs_err[:, sev_sub_indices], axis=1)
                else:
                    circ_sev[sev][c_i] = np.mean(abs_err[sev_sub_indices])

    return sorted_circuits, circ_macro, circ_sev


def classify_cell(d_ci_95: list[float] | None) -> str:
    """Mutually exclusive cell classification.

    1. Measurement adds: the 95 percent interval for mean D lies wholly above zero.
    2. Measurement hurts: the interval lies wholly below zero.
    3. Not distinguished: otherwise.
    If interval is None (e.g. missing seeds), returns 'incomplete'.
    """
    if d_ci_95 is None:
        return "incomplete"
    low, high = d_ci_95
    if low > 0:
        return "Measurement adds"
    elif high < 0:
        return "Measurement hurts"
    else:
        return "Not distinguished"


def determine_rung_statement(seed_results: Mapping[int, dict[str, Any]]) -> str:
    """Determine rung-level statement per family / part.

    Rule:
    - If a dataset seed's cell is absent or not estimable, say 'incomplete'
      rather than 'mixed'.
    - A statement is made only when all three dataset seeds share one label;
      otherwise the rung reads 'mixed'.
    """
    labels = []
    for d_seed in DATASET_SEEDS:
        if d_seed not in seed_results:
            return "incomplete"
        cell = seed_results[d_seed]
        if cell.get("status") != "estimated" or cell.get("classification_label") in ("incomplete", None):
            return "incomplete"
        labels.append(cell["classification_label"])

    if len(labels) == 3:
        if len(set(labels)) == 1:
            return labels[0]
        return "mixed"
    return "incomplete"


def run_analysis(
    fits_dirs: Sequence[Path | str],
    cache_dirs: Sequence[Path | str],
    n_draws: int = DEFAULT_DRAWS,
    bootstrap_seed: int = RULE_SEED,
) -> dict[str, Any]:
    """Run full analysis pipeline over fit files and cache items."""
    loader = DataLoader(fits_dirs, cache_dirs)

    # Index fits by (key, rung, arm) -> learner_seed -> fit dict
    fits_map: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for fit in loader.fits:
        key = str(fit["key"])
        rung = str(fit["rung"])
        arm = str(fit["arm"])
        l_seed = int(fit.get("learner_seed", 0))
        fits_map[(key, rung, arm)][l_seed] = fit

    results_cells = []
    by_part_row_rung: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    dropped_cells_per_dataset: dict[str, list[str]] = {}

    def load_arm_predictions(
        key_val: str,
        rung_name: str,
        arm_name: str,
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None, list[str]]:
        seed_dict = fits_map.get((key_val, rung_name, arm_name), {})
        if not seed_dict:
            return [], None, None, []
        sorted_seeds = sorted(seed_dict.keys())
        selected_preds = []
        mlp_preds = []
        selected_models = []
        for s in sorted_seeds:
            info = seed_dict[s]
            npz = np.load(info["_npz_path"])
            selected_preds.append(npz["test"])
            if "test__mlp" in npz:
                mlp_preds.append(npz["test__mlp"])
            selected_models.append(info.get("selected_model", "unknown"))

        sel_arr = np.array(selected_preds)
        mlp_arr = np.array(mlp_preds) if mlp_preds else None
        if arm_name == "A" and len(sorted_seeds) == 1:
            sel_arr = sel_arr[0]
        return sorted_seeds, sel_arr, mlp_arr, selected_models

    # Discover unique datasets: (part, dataset_seed, key)
    raw_datasets = sorted(list(set((f["part"], int(f["dataset_seed"]), f["key"]) for f in loader.fits)))
    datasets = []
    for raw_part, d_seed, key in raw_datasets:
        is_nc = "nc-" in key.lower() or any(f.get("rung", "").startswith("NC-") for f in loader.fits if f.get("key") == key)
        is_part_b = raw_part.upper() == "B" or "qaoa" in key.lower()
        eff_part = "NC" if is_nc else ("B" if is_part_b else "A")
        if (eff_part, d_seed, key) not in datasets:
            datasets.append((eff_part, d_seed, key))

    # Pre-cache baseline circuit errors for paired contrasts: (key, fam, baseline_rung) -> dict
    baseline_store: dict[tuple[str, str, str], dict[str, Any]] = {}

    for part, d_seed, key in datasets:
        is_part_b = part.upper() == "B"
        is_nc = part.upper() == "NC"
        cache = loader.load_cache(key, d_seed)
        dropped = cache.get("dropped_cells", [])
        dropped_cells_per_dataset[key] = list(dropped)
        test_items = cache["test"]

        families = sorted(list(set(it.get("family", "qaoa" if is_part_b else "spin") for it in test_items)))

        for fam in families:
            if is_part_b:
                base_rung = "B-partial"
                obs_filter = "zz_mid"
            elif is_nc or fam == "near_clifford" or "nc-" in key.lower():
                base_rung = "NC-R0"
                obs_filter = None
            else:
                base_rung = "R0"
                obs_filter = None

            b_c_seeds, b_c_preds, _, _ = load_arm_predictions(key, base_rung, "C")
            b_f_seeds, b_f_preds, _, _ = load_arm_predictions(key, base_rung, "F")
            if b_c_preds is not None and b_f_preds is not None:
                common = sorted(list(set(b_c_seeds).intersection(b_f_seeds)))
                if common:
                    c_sub = b_c_preds[[b_c_seeds.index(s) for s in common]]
                    f_sub = b_f_preds[[b_f_seeds.index(s) for s in common]]
                    _, b_c_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, c_sub, obs_filter, dropped)
                    _, b_f_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, f_sub, obs_filter, dropped)
                    baseline_store[(key, fam, base_rung)] = {
                        "c_circ": b_c_circ,
                        "f_circ": b_f_circ,
                        "seeds": common,
                    }

    # Evaluate each dataset
    for part, d_seed, key in datasets:
        is_part_b = part.upper() == "B"
        is_nc = part.upper() == "NC"
        cache = loader.load_cache(key, d_seed)
        test_items = cache["test"]
        dropped = cache.get("dropped_cells", [])

        cache_families = sorted(list(set(it.get("family", "qaoa" if is_part_b else "spin") for it in test_items)))
        r_pred = np.array([float(it["noisy_expectation"]) for it in test_items])

        rungs_for_key = sorted(list(set(f["rung"] for f in loader.fits if f["key"] == key)))
        canonical_order = PART_B_RUNGS if is_part_b else (NC_RUNGS if is_nc else PART_A_RUNGS)
        ordered_rungs = [r for r in canonical_order if r in rungs_for_key]
        for r in rungs_for_key:
            if r not in ordered_rungs:
                ordered_rungs.append(r)

        for rung in ordered_rungs:
            if rung == "R3-TFI":
                active_families = [f for f in cache_families if f == "tfi"]
            elif rung == "R3-Heis":
                active_families = [f for f in cache_families if f == "heisenberg"]
            else:
                active_families = cache_families

            if is_part_b:
                m_rung = "B-none"
                m_arm = "F"
                base_rung = "B-partial"
                obs_filter = "zz_mid"
            elif is_nc or "near_clifford" in active_families or rung.startswith("NC-") or "nc-" in key.lower():
                m_rung = "NC-none"
                m_arm = "M"
                base_rung = "NC-R0"
                obs_filter = None
            else:
                m_rung = "R5"
                m_arm = "F"
                base_rung = "R0"
                obs_filter = None

            c_seeds, c_preds, c_mlp_preds, c_models = load_arm_predictions(key, rung, "C")

            # Arm F (mitigator for current rung):
            # Per frozen rule, at NC-none, arm M plays F's role (D = C - M)
            if rung == "NC-none":
                f_seeds, f_preds, f_mlp_preds, f_models = load_arm_predictions(key, "NC-none", "M")
                if f_preds is None:
                    f_seeds, f_preds, f_mlp_preds, f_models = load_arm_predictions(key, "NC-none", "F")
            else:
                f_seeds, f_preds, f_mlp_preds, f_models = load_arm_predictions(key, rung, "F")

            _, a_preds, _, _ = load_arm_predictions(key, rung, "A")

            # Arm M (measurement-only mitigator):
            # For Part A: arm F at R5
            # For Part B: arm F at B-none
            # For Near-Clifford: arm M at NC-none
            m_seeds, m_preds, _, _ = load_arm_predictions(key, m_rung, m_arm)
            if m_preds is None:
                alt_arm = "F" if m_arm == "M" else "M"
                m_seeds, m_preds, _, _ = load_arm_predictions(key, m_rung, alt_arm)

            common_seeds = sorted(list(set(c_seeds).intersection(f_seeds))) if c_preds is not None and f_preds is not None else []

            for fam in active_families:
                row_key = f"s{d_seed}__{fam}" if not is_part_b else f"s{d_seed}"

                if not common_seeds:
                    cell_record = {
                        "part": part,
                        "dataset_seed": d_seed,
                        "family": fam if not is_part_b else "qaoa",
                        "row_key": row_key,
                        "rung": rung,
                        "status": "incomplete",
                        "reason": "no matching valid learner seeds (1..20) found for C and F",
                        "mean_c": None,
                        "mean_f": None,
                        "mean_m": None,
                        "mean_a": None,
                        "mean_r": None,
                        "mean_d": None,
                        "d_ci_95": None,
                        "mean_d_over_c": None,
                        "d_over_c_ci_95": None,
                        "e": None,
                        "e_ci_95": None,
                        "s": None,
                        "s_ci_95": None,
                        "s_defined": False,
                        "paired_contrast": None,
                        "secondary_d": None,
                        "secondary_d_ci_95": None,
                        "selection_counts": {
                            "F": dict(Counter(f_models)),
                            "C": dict(Counter(c_models)),
                        },
                        "d_per_severity": {},
                        "dropped_cells": list(dropped),
                        "classification_label": "incomplete",
                        "largest_non_excluded_d_over_c": None,
                        "n_learner_seeds": 0,
                        "n_circuits": 0,
                    }
                    results_cells.append(cell_record)
                    by_part_row_rung[part][row_key][rung] = cell_record
                    continue

                c_idx = [c_seeds.index(s) for s in common_seeds]
                f_idx = [f_seeds.index(s) for s in common_seeds]
                c_preds_aligned = c_preds[c_idx]
                f_preds_aligned = f_preds[f_idx]
                c_mlp_aligned = c_mlp_preds[c_idx] if c_mlp_preds is not None else None
                f_mlp_aligned = f_mlp_preds[f_idx] if f_mlp_preds is not None else None

                # Compute circuit errors
                cids, c_circ, c_sev = evaluate_circuit_errors(test_items, fam if not is_part_b else None, c_preds_aligned, obs_filter, dropped)
                _, f_circ, f_sev = evaluate_circuit_errors(test_items, fam if not is_part_b else None, f_preds_aligned, obs_filter, dropped)
                _, r_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, r_pred, obs_filter, dropped)

                a_circ = None
                if a_preds is not None:
                    _, a_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, a_preds, obs_filter, dropped)

                m_circ = None
                if m_preds is not None:
                    m_common = [s for s in common_seeds if s in m_seeds]
                    if len(m_common) == len(common_seeds):
                        m_idx = [m_seeds.index(s) for s in common_seeds]
                        _, m_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, m_preds[m_idx], obs_filter, dropped)

                c_mlp_circ = None
                f_mlp_circ = None
                if c_mlp_aligned is not None and f_mlp_aligned is not None:
                    _, c_mlp_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, c_mlp_aligned, obs_filter, dropped)
                    _, f_mlp_circ, _ = evaluate_circuit_errors(test_items, fam if not is_part_b else None, f_mlp_aligned, obs_filter, dropped)

                n_circuits = len(cids)
                n_seeds = len(common_seeds)

                # Point estimates
                mean_c = float(np.mean(c_circ))
                mean_f = float(np.mean(f_circ))
                mean_r = float(np.mean(r_circ))
                mean_a = float(np.mean(a_circ)) if a_circ is not None else None
                mean_m = float(np.mean(m_circ)) if m_circ is not None else None

                mean_d = mean_c - mean_f
                mean_doc = (mean_d / mean_c) if mean_c != 0 else 0.0

                # 3. E = M - F
                mean_e = None
                e_ci = None
                if m_circ is not None:
                    if np.array_equal(m_circ, f_circ):
                        mean_e = 0.0
                        e_ci = [0.0, 0.0]
                    else:
                        mean_e = float(mean_m - mean_f)

                # D per severity cell
                d_per_sev = {}
                for sev in sorted(c_sev.keys()):
                    d_per_sev[sev] = float(np.mean(c_sev[sev]) - np.mean(f_sev[sev]))

                # Bootstrap draws: fresh default_rng(bootstrap_seed) per row and quantity
                seed_draws, circ_draws = generate_bootstrap_indices(n_seeds, n_circuits, n_draws, bootstrap_seed)
                c_draws = compute_resampled_means(c_circ, seed_draws, circ_draws)
                f_draws = compute_resampled_means(f_circ, seed_draws, circ_draws)

                # 1. D = C - F
                d_draws = c_draws - f_draws
                d_ci = compute_bootstrap_ci(d_draws)

                # 2. D/C computed as ratio of means inside each draw
                with np.errstate(divide="ignore", invalid="ignore"):
                    doc_draws = np.where(c_draws != 0, d_draws / c_draws, 0.0)
                doc_ci = compute_bootstrap_ci(doc_draws)

                # 3. E = M - F (bootstrap CI)
                if m_circ is not None and e_ci is None:
                    m_draws = compute_resampled_means(m_circ, seed_draws, circ_draws)
                    e_draws = m_draws - f_draws
                    e_ci = compute_bootstrap_ci(e_draws)

                # 4. S = (A - C)/(A - F), reported only when A - F is positive in every draw
                s_val = None
                s_ci = None
                s_defined = False
                if a_circ is not None:
                    a_draws = compute_resampled_means(a_circ, seed_draws, circ_draws)
                    af_draws = a_draws - f_draws
                    if np.all(af_draws > 0):
                        s_defined = True
                        if mean_a is not None and (mean_a - mean_f) != 0:
                            s_val = float((mean_a - mean_c) / (mean_a - mean_f))
                        with np.errstate(divide="ignore", invalid="ignore"):
                            s_draws = np.where(af_draws != 0, (a_draws - c_draws) / af_draws, 0.0)
                        s_ci = compute_bootstrap_ci(s_draws)

                # 5. Paired contrasts versus baseline rung (R0, B-partial, or NC-R0)
                contrast_info = None
                b_data = baseline_store.get((key, fam, base_rung))
                if b_data is not None and b_data["seeds"] == common_seeds:
                    b_c_circ = b_data["c_circ"]
                    b_f_circ = b_data["f_circ"]
                    b_c_draws = compute_resampled_means(b_c_circ, seed_draws, circ_draws)
                    b_f_draws = compute_resampled_means(b_f_circ, seed_draws, circ_draws)

                    b_d_draws = b_c_draws - b_f_draws
                    with np.errstate(divide="ignore", invalid="ignore"):
                        b_doc_draws = np.where(b_c_draws != 0, b_d_draws / b_c_draws, 0.0)

                    delta_d_draws = d_draws - b_d_draws
                    delta_doc_draws = doc_draws - b_doc_draws

                    b_mean_d = float(np.mean(b_c_circ) - np.mean(b_f_circ))
                    b_mean_c = float(np.mean(b_c_circ))
                    b_mean_doc = (b_mean_d / b_mean_c) if b_mean_c != 0 else 0.0

                    contrast_info = {
                        "baseline_rung": base_rung,
                        "delta_d": float(mean_d - b_mean_d),
                        "delta_d_ci_95": compute_bootstrap_ci(delta_d_draws),
                        "delta_d_over_c": float(mean_doc - b_mean_doc),
                        "delta_d_over_c_ci_95": compute_bootstrap_ci(delta_doc_draws),
                    }
                elif rung == base_rung:
                    contrast_info = {
                        "baseline_rung": base_rung,
                        "delta_d": 0.0,
                        "delta_d_ci_95": [0.0, 0.0],
                        "delta_d_over_c": 0.0,
                        "delta_d_over_c_ci_95": [0.0, 0.0],
                    }

                # 6. Learner-fixed secondary D (MLP candidate in both arms)
                sec_d = None
                sec_d_ci = None
                if c_mlp_circ is not None and f_mlp_circ is not None:
                    sec_d = float(np.mean(c_mlp_circ) - np.mean(f_mlp_circ))
                    c_mlp_draws = compute_resampled_means(c_mlp_circ, seed_draws, circ_draws)
                    f_mlp_draws = compute_resampled_means(f_mlp_circ, seed_draws, circ_draws)
                    sec_d_ci = compute_bootstrap_ci(c_mlp_draws - f_mlp_draws)

                # Selection counts per cell
                sel_counts = {
                    "F": dict(Counter(f_models)),
                    "C": dict(Counter(c_models)),
                }
                if rung == "NC-none":
                    sel_counts["M"] = dict(Counter(f_models))

                # Classification label per cell
                cell_label = classify_cell(d_ci)

                # Largest non-excluded D/C: upper limit of mean D / mean C
                largest_non_excluded = float(doc_ci[1])

                cell_record = {
                    "part": part,
                    "dataset_seed": d_seed,
                    "family": fam if not is_part_b else "qaoa",
                    "row_key": row_key,
                    "rung": rung,
                    "status": "estimated",
                    "mean_c": mean_c,
                    "mean_f": mean_f,
                    "mean_m": mean_m,
                    "mean_a": mean_a,
                    "mean_r": mean_r,
                    "mean_d": mean_d,
                    "d_ci_95": d_ci,
                    "mean_d_over_c": mean_doc,
                    "d_over_c_ci_95": doc_ci,
                    "e": mean_e,
                    "e_ci_95": e_ci,
                    "s": s_val,
                    "s_ci_95": s_ci,
                    "s_defined": s_defined,
                    "paired_contrast": contrast_info,
                    "secondary_d": sec_d,
                    "secondary_d_ci_95": sec_d_ci,
                    "selection_counts": sel_counts,
                    "d_per_severity": d_per_sev,
                    "dropped_cells": list(dropped),
                    "classification_label": cell_label,
                    "largest_non_excluded_d_over_c": largest_non_excluded,
                    "n_learner_seeds": n_seeds,
                    "n_circuits": n_circuits,
                }
                results_cells.append(cell_record)
                by_part_row_rung[part][row_key][rung] = cell_record


    # Rung-level statements per family / part
    # A statement is made only when all three dataset seeds share one label;
    # otherwise the rung reads 'mixed'. If any seed is missing or not estimable,
    # says 'incomplete'.
    rung_statements: dict[str, dict[str, Any]] = defaultdict(dict)
    largest_doc_per_rung: dict[str, dict[str, Any]] = defaultdict(dict)

    # Group cell records by (part, rung, family) -> dataset_seed -> cell
    grouped_cells: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for cell in results_cells:
        grouped_cells[(cell["part"], cell["rung"], cell["family"])][cell["dataset_seed"]] = cell

    final_rung_statements: dict[str, dict[str, Any]] = defaultdict(dict)
    final_largest_doc: dict[str, dict[str, Any]] = defaultdict(dict)

    for (part, rung, fam), seed_dict in grouped_cells.items():
        is_part_b = part.upper() == "B"
        statement = determine_rung_statement(seed_dict)

        valid_docs = [
            c["largest_non_excluded_d_over_c"]
            for c in seed_dict.values()
            if c.get("largest_non_excluded_d_over_c") is not None
        ]
        doc_max = float(max(valid_docs)) if valid_docs else None

        if is_part_b:
            final_rung_statements[part][rung] = statement
            final_largest_doc[part][rung] = doc_max
        else:
            if rung not in final_rung_statements[part]:
                final_rung_statements[part][rung] = {}
                final_largest_doc[part][rung] = {}
            final_rung_statements[part][rung][fam] = statement
            final_largest_doc[part][rung][fam] = doc_max

    # Attach statement and rung-level largest non-excluded D/C to cell records
    for cell in results_cells:
        part = cell["part"]
        rung = cell["rung"]
        fam = cell["family"]
        if part.upper() == "B":
            cell["rung_statement"] = final_rung_statements[part].get(rung, "incomplete")
            cell["rung_largest_non_excluded_d_over_c"] = final_largest_doc[part].get(rung)
        else:
            cell["rung_statement"] = final_rung_statements[part].get(rung, {}).get(fam, "incomplete")
            cell["rung_largest_non_excluded_d_over_c"] = final_largest_doc[part].get(rung, {}).get(fam)

    output_payload = {
        "schema": "descriptor-ladder-analysis-v1",
        "frozen_rule": RULE_FILE,
        "rule_seed": RULE_SEED,
        "bootstrap_seed": bootstrap_seed,
        "n_draws": n_draws,
        "rung_statements": dict(final_rung_statements),
        "largest_non_excluded_d_over_c_per_rung": dict(final_largest_doc),
        "dropped_cells_per_dataset": dropped_cells_per_dataset,
        "skipped_files": loader.skipped_files,
        "failed_fits": loader.failed_fits,
        "excluded_anchors": loader.excluded_anchors,
        "cells": results_cells,
        "by_part_row_rung": dict(by_part_row_rung),
    }
    return output_payload


def write_analysis_json(path: Path | str, payload: dict) -> None:
    """Atomically write JSON result file."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, target)


# --------------------------------------------------------------------------
# Self-Test Implementation
# --------------------------------------------------------------------------


def run_self_test() -> None:
    """Fabricate synthetic datasets/fits and verify all estimands and defect fixes."""
    print("=== Running self-test ===")
    with tempfile.TemporaryDirectory(prefix="descriptor_ladder_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cache_dir_a = tmp_path / "cache_a"
        cache_dir_b = tmp_path / "cache_b"
        cache_dir_nc = tmp_path / "cache_nc"
        fits_dir = tmp_path / "fits"
        out_file = tmp_path / "analysis_out.json"

        cache_dir_a.mkdir()
        cache_dir_b.mkdir()
        cache_dir_nc.mkdir()
        fits_dir.mkdir()

        # -------------------------------------------------------------
        # Part A synthetic data (3 seeds for R0)
        # -------------------------------------------------------------
        n_circuits = 10
        n_seeds = 5
        families = ["heisenberg", "tfi"]
        severities = ["L1", "L3"]
        observables = ["z_mid", "zz_mid"]

        for d_seed in (101, 211, 307):
            dataset_key = f"shipped-s{d_seed}-n640"
            test_items = []
            item_id_counter = 0
            for fam in families:
                for c_i in range(n_circuits):
                    cid = f"circuit-{fam}-{c_i:03d}"
                    for sev in severities:
                        for obs in observables:
                            test_items.append({
                                "item_id": f"item-{item_id_counter:04d}",
                                "circuit_id": cid,
                                "family": fam,
                                "severity": sev,
                                "observable": obs,
                                "ideal_expectation": 0.5,
                                "noisy_expectation": 0.6,
                            })
                            item_id_counter += 1
            with open(cache_dir_a / f"{dataset_key}.pkl", "wb") as handle:
                pickle.dump({"test": test_items, "dropped_cells": []}, handle)

            n_items = len(test_items)

            # Arm A fit
            a_pred = np.full(n_items, 0.60)
            np.savez_compressed(fits_dir / f"{dataset_key}__R0__A.npz", test=a_pred)
            with open(fits_dir / f"{dataset_key}__R0__A.json", "w") as h:
                json.dump({
                    "schema": "descriptor-information-fit-v1",
                    "part": "A",
                    "dataset_seed": d_seed,
                    "key": dataset_key,
                    "rung": "R0",
                    "arm": "A",
                }, h)

            # Arms C and F fits (seeds 1..5)
            for s in range(1, n_seeds + 1):
                c_pred = np.full(n_items, 0.56)
                f_pred = np.full(n_items, 0.52)
                c_mlp = np.full(n_items, 0.57)
                f_mlp = np.full(n_items, 0.52)

                np.savez_compressed(
                    fits_dir / f"{dataset_key}__R0__seed{s}__C.npz",
                    test=c_pred,
                    test__mlp=c_mlp,
                )
                with open(fits_dir / f"{dataset_key}__R0__seed{s}__C.json", "w") as h:
                    json.dump({
                        "schema": "descriptor-information-fit-v1",
                        "part": "A",
                        "dataset_seed": d_seed,
                        "key": dataset_key,
                        "rung": "R0",
                        "arm": "C",
                        "learner_seed": s,
                        "selected_model": "mlp",
                    }, h)

                np.savez_compressed(
                    fits_dir / f"{dataset_key}__R0__seed{s}__F.npz",
                    test=f_pred,
                    test__mlp=f_mlp,
                )
                with open(fits_dir / f"{dataset_key}__R0__seed{s}__F.json", "w") as h:
                    json.dump({
                        "schema": "descriptor-information-fit-v1",
                        "part": "A",
                        "dataset_seed": d_seed,
                        "key": dataset_key,
                        "rung": "R0",
                        "arm": "F",
                        "learner_seed": s,
                        "selected_model": "mlp",
                    }, h)

                # R5 fits for M
                m_pred = np.full(n_items, 0.53)
                np.savez_compressed(fits_dir / f"{dataset_key}__R5__seed{s}__F.npz", test=m_pred, test__mlp=m_pred)
                with open(fits_dir / f"{dataset_key}__R5__seed{s}__F.json", "w") as h:
                    json.dump({
                        "schema": "descriptor-information-fit-v1",
                        "part": "A",
                        "dataset_seed": d_seed,
                        "key": dataset_key,
                        "rung": "R5",
                        "arm": "F",
                        "learner_seed": s,
                        "selected_model": "mlp",
                    }, h)

            # Defect 1 test: Add anchor fit with learner_seed 101 outside 1..20
            anchor_stem = f"{dataset_key}__R0__orig__F"
            np.savez_compressed(fits_dir / f"{anchor_stem}.npz", test=np.full(n_items, 0.99))
            with open(fits_dir / f"{anchor_stem}.json", "w") as h:
                json.dump({
                    "schema": "descriptor-information-fit-v1",
                    "part": "A",
                    "dataset_seed": d_seed,
                    "key": dataset_key,
                    "rung": "R0",
                    "arm": "F",
                    "learner_seed": d_seed,
                    "selected_model": "mlp",
                }, h)

        # Defect 2 test: Add a skipped file (invalid JSON & .order.json & missing npz)
        (fits_dir / "dummy.order.json").write_text("{}", encoding="utf-8")
        (fits_dir / "corrupted.json").write_text("{broken", encoding="utf-8")
        (fits_dir / "missing_npz.json").write_text(json.dumps({"part": "A", "rung": "R0", "arm": "C"}), encoding="utf-8")

        # Defect 3 test: Add a fit with error
        (fits_dir / "failed_fit.json").write_text(json.dumps({
            "schema": "descriptor-information-fit-v1",
            "part": "A",
            "dataset_seed": 101,
            "key": "shipped-s101-n640",
            "rung": "R0",
            "arm": "F",
            "learner_seed": 19,
            "error": "ConvergenceError: test failed",
        }), encoding="utf-8")
        np.savez_compressed(fits_dir / "failed_fit.npz", test=np.zeros(10))

        # -------------------------------------------------------------
        # Part B synthetic data (with dropped cells)
        # -------------------------------------------------------------
        for s_id in (101, 211, 307):
            b_items = []
            for c_i in range(10):
                cid = f"qaoa-circ-{c_i:03d}"
                for sev in ("L1", "L3"):
                    b_items.append({
                        "item_id": f"it-{s_id}-{c_i}-{sev}",
                        "circuit_id": cid,
                        "family": "qaoa",
                        "severity": sev,
                        "observable": "zz_mid",
                        "ideal_expectation": 0.5,
                        "noisy_expectation": 0.6,
                    })
            # Defect 4 test: cache records dropped cell
            dropped_for_b = ["depolarizing_readout/L3/zz_mid"] if s_id == 307 else []
            with open(cache_dir_b / f"qaoa-s{s_id}-n640.pkl", "wb") as h:
                pickle.dump({"test": b_items, "dropped_cells": dropped_for_b}, h)

            n_b = len(b_items)
            for s in range(1, 4):
                # B-partial
                np.savez_compressed(fits_dir / f"qaoa-s{s_id}-n640__B-partial__seed{s}__C.npz", test=np.full(n_b, 0.55), test__mlp=np.full(n_b, 0.55))
                with open(fits_dir / f"qaoa-s{s_id}-n640__B-partial__seed{s}__C.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "B", "dataset_seed": s_id, "key": f"qaoa-s{s_id}-n640", "rung": "B-partial", "arm": "C", "learner_seed": s, "selected_model": "mlp"}, h)
                np.savez_compressed(fits_dir / f"qaoa-s{s_id}-n640__B-partial__seed{s}__F.npz", test=np.full(n_b, 0.53), test__mlp=np.full(n_b, 0.53))
                with open(fits_dir / f"qaoa-s{s_id}-n640__B-partial__seed{s}__F.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "B", "dataset_seed": s_id, "key": f"qaoa-s{s_id}-n640", "rung": "B-partial", "arm": "F", "learner_seed": s, "selected_model": "mlp"}, h)

                # B-complete
                np.savez_compressed(fits_dir / f"qaoa-s{s_id}-n640__B-complete__seed{s}__C.npz", test=np.full(n_b, 0.55), test__mlp=np.full(n_b, 0.55))
                with open(fits_dir / f"qaoa-s{s_id}-n640__B-complete__seed{s}__C.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "B", "dataset_seed": s_id, "key": f"qaoa-s{s_id}-n640", "rung": "B-complete", "arm": "C", "learner_seed": s, "selected_model": "mlp"}, h)
                np.savez_compressed(fits_dir / f"qaoa-s{s_id}-n640__B-complete__seed{s}__F.npz", test=np.full(n_b, 0.51), test__mlp=np.full(n_b, 0.51))
                with open(fits_dir / f"qaoa-s{s_id}-n640__B-complete__seed{s}__F.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "B", "dataset_seed": s_id, "key": f"qaoa-s{s_id}-n640", "rung": "B-complete", "arm": "F", "learner_seed": s, "selected_model": "mlp"}, h)

        # -------------------------------------------------------------
        # Defect 5 test: Near-Clifford synthetic data (NC-R0 and NC-none)
        # -------------------------------------------------------------
        for s_id in (101, 211, 307):
            nc_items = []
            for c_i in range(10):
                cid = f"nc-circ-{c_i:03d}"
                for sev in ("L1", "L3"):
                    for obs in ("z_mid", "zz_mid"):
                        nc_items.append({
                            "item_id": f"nc-{s_id}-{c_i}-{sev}-{obs}",
                            "circuit_id": cid,
                            "family": "near_clifford",
                            "severity": sev,
                            "observable": obs,
                            "ideal_expectation": 0.5,
                            "noisy_expectation": 0.6,
                        })
            with open(cache_dir_nc / f"nc-s{s_id}-n640.pkl", "wb") as h:
                pickle.dump({"test": nc_items, "dropped_cells": []}, h)

            n_nc = len(nc_items)
            for s in range(1, 4):
                # NC-R0
                np.savez_compressed(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s:02d}__C.npz", test=np.full(n_nc, 0.56), test__mlp=np.full(n_nc, 0.56))
                with open(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s:02d}__C.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "A", "dataset_seed": s_id, "key": f"nc-s{s_id}-n640", "rung": "NC-R0", "arm": "C", "learner_seed": s, "selected_model": "mlp"}, h)
                np.savez_compressed(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s:02d}__F.npz", test=np.full(n_nc, 0.52), test__mlp=np.full(n_nc, 0.52))
                with open(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s:02d}__F.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "A", "dataset_seed": s_id, "key": f"nc-s{s_id}-n640", "rung": "NC-R0", "arm": "F", "learner_seed": s, "selected_model": "mlp"}, h)

                # NC-none (measurement-only rung, arm M)
                np.savez_compressed(fits_dir / f"nc-s{s_id}-n640__NC-none__k{s:02d}__M.npz", test=np.full(n_nc, 0.53), test__mlp=np.full(n_nc, 0.53))
                with open(fits_dir / f"nc-s{s_id}-n640__NC-none__k{s:02d}__M.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "A", "dataset_seed": s_id, "key": f"nc-s{s_id}-n640", "rung": "NC-none", "arm": "M", "learner_seed": s, "selected_model": "mlp"}, h)
                np.savez_compressed(fits_dir / f"nc-s{s_id}-n640__NC-none__k{s:02d}__C.npz", test=np.full(n_nc, 0.55), test__mlp=np.full(n_nc, 0.55))
                with open(fits_dir / f"nc-s{s_id}-n640__NC-none__k{s:02d}__C.json", "w") as h:
                    json.dump({"schema": "descriptor-information-fit-v1", "part": "A", "dataset_seed": s_id, "key": f"nc-s{s_id}-n640", "rung": "NC-none", "arm": "C", "learner_seed": s, "selected_model": "mlp"}, h)

            # Add NC anchor fit (learner seed = dataset seed)
            np.savez_compressed(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s_id}__F.npz", test=np.full(n_nc, 0.99))
            with open(fits_dir / f"nc-s{s_id}-n640__NC-R0__k{s_id}__F.json", "w") as h:
                json.dump({"schema": "descriptor-information-fit-v1", "part": "A", "dataset_seed": s_id, "key": f"nc-s{s_id}-n640", "rung": "NC-R0", "arm": "F", "learner_seed": s_id, "selected_model": "mlp"}, h)

        # Run analysis across multiple cache dirs and check all defects
        res = run_analysis([fits_dir], [cache_dir_a, cache_dir_b, cache_dir_nc], n_draws=200, bootstrap_seed=20261002)
        write_analysis_json(out_file, res)

        # 1. Defect 1: Anchors excluded and listed
        assert len(res["excluded_anchors"]) == 6, f"Expected 6 excluded anchors, got {len(res['excluded_anchors'])}"
        for anchor in res["excluded_anchors"]:
            assert anchor["learner_seed"] in (101, 211, 307)

        # 2. Defect 2: Skipped files recorded and listed
        assert len(res["skipped_files"]) == 3, f"Expected 3 skipped files, got {len(res['skipped_files'])}"

        # 3. Defect 3: Failed fits excluded and listed
        assert len(res["failed_fits"]) == 1, f"Expected 1 failed fit, got {len(res['failed_fits'])}"

        # 4. Defect 4: Dropped cells disclosed
        dropped_records = [c for c in res["cells"] if c["dataset_seed"] == 307 and c["part"] == "B"]
        assert any("depolarizing_readout/L3/zz_mid" in c["dropped_cells"] for c in dropped_records)

        # 5. Defect 5 & Fix 2: Near-Clifford handled with part 'NC', D, E, paired contrast
        nc_r0_cells = [c for c in res["cells"] if c["rung"] == "NC-R0"]
        assert len(nc_r0_cells) == 3
        for nc_c in nc_r0_cells:
            assert nc_c["part"] == "NC"
            assert nc_c["family"] == "near_clifford"
            assert np.isclose(nc_c["mean_d"], 0.04)
            assert np.isclose(nc_c["e"], 0.01)
            assert nc_c["paired_contrast"]["baseline_rung"] == "NC-R0"

        nc_none_cells = [c for c in res["cells"] if c["rung"] == "NC-none"]
        assert len(nc_none_cells) == 3
        for nc_c in nc_none_cells:
            assert nc_c["part"] == "NC"
            assert nc_c["family"] == "near_clifford"
            assert np.isclose(nc_c["mean_d"], 0.02)
            assert np.isclose(nc_c["e"], 0.0)
            assert nc_c["paired_contrast"]["baseline_rung"] == "NC-R0"

        # Check Part A at R0 has E populated
        r0_cells = [c for c in res["cells"] if c["rung"] == "R0" and c["part"] == "A"]
        assert len(r0_cells) == 6
        for c in r0_cells:
            assert c["e"] is not None
            assert np.isclose(c["e"], 0.01)

        # Check by_part_row_rung contains NC
        assert "NC" in res["by_part_row_rung"]

        # 6. Defect 7: Rung statement: all 3 seeds agree -> shared label
        assert res["rung_statements"]["B"]["B-complete"] == "Measurement adds"
        assert res["rung_statements"]["A"]["R0"]["heisenberg"] == "Measurement adds"
        assert res["rung_statements"]["NC"]["NC-R0"]["near_clifford"] == "Measurement adds"
        assert res["rung_statements"]["NC"]["NC-none"]["near_clifford"] == "Measurement adds"

        print("Self-test passed successfully! All 7 defect fixes verified.")


# --------------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descriptor Ladder Analysis B: estimands, bootstrap, and classification."
    )
    parser.add_argument(
        "--fits",
        nargs="+",
        action="extend",
        help="One or more directories containing fit JSON and NPZ files.",
    )
    parser.add_argument(
        "--cache",
        nargs="+",
        action="extend",
        help="One or more directories containing dataset test cache pickle files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help=f"Number of bootstrap draws (default: {DEFAULT_DRAWS}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=RULE_SEED,
        help=f"RNG seed for two-stage bootstrap (default: {RULE_SEED}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test on fabricated synthetic data with known D.",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        if not args.fits or not args.cache or not args.out:
            return

    if not args.fits or not args.cache or not args.out:
        parser.error("--fits, --cache, and --out are required unless running only --self-test")

    print(f"Starting descriptor ladder analysis b:")
    print(f"  Fits directories:  {args.fits}")
    print(f"  Cache directories: {args.cache}")
    print(f"  Output path:       {args.out}")
    print(f"  Bootstrap draws:   {args.draws}")
    print(f"  Bootstrap seed:    {args.bootstrap_seed}")

    result = run_analysis(
        args.fits,
        args.cache,
        n_draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_analysis_json(args.out, result)

    print(f"\nProcessing summary:")
    print(f"  Estimated cells:  {len(result['cells'])}")
    print(f"  Excluded anchors: {len(result['excluded_anchors'])}")
    print(f"  Failed fits:      {len(result['failed_fits'])}")
    print(f"  Skipped files:    {len(result['skipped_files'])}")

    print("\nResults summary:")
    print(f"{'Part':<6} {'Rung':<10} {'Row':<24} {'Mean D':<10} {'95% CI':<24} {'Label':<18} {'Rung Statement'}")
    print("-" * 115)
    for cell in result["cells"]:
        ci_str = (
            f"[{cell['d_ci_95'][0]:.6f}, {cell['d_ci_95'][1]:.6f}]"
            if cell.get("d_ci_95") is not None
            else "None"
        )
        mean_d_str = f"{cell['mean_d']:<10.6f}" if cell.get("mean_d") is not None else "None      "
        print(
            f"{cell['part']:<6} {cell['rung']:<10} {cell['row_key']:<24} "
            f"{mean_d_str} {ci_str:<24} {cell['classification_label']:<18} "
            f"{cell.get('rung_statement', '-')}"
        )
    print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
