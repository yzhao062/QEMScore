"""Learner-seed replication of the capacity-matched gap (frozen rule, 2026-10-01).

For each regenerated shipped-step dataset and each learner seed k, this fits the
three arms of the frozen rule through the campaign's own code path:

- F (`liao`): ``LiaoMitigator(random_state=k).fit(train, validation)``
- C (`liao-feat-only`): the same with ``drop_features=("noisy_expectation",)``
- P (`liao-training-shuffle`): F refitted on ``shuffle_noisy_items(train, seed=k)``

exactly as ``qemscore.campaign.analysis._fit_arm`` builds them (full RF and MLP
candidate search plus the one-standard-error selection inside ``LiaoMitigator``),
and predicts on ``_prediction_items`` rows, as ``build_setting_record`` does.

It also refits the original seed (k = dataset seed) for F and C and compares
the item-level validation and test predictions with the archived record, and
refits the archived training-shuffle arm (learner seed = dataset seed, shuffle
seed = TRAINING_SHUFFLE_SEED) to compare its archived validation predictions.
The original-seed fits stay out of the 20-seed statistics.

Every fit writes its own JSON and compressed prediction file as soon as it
finishes, so a partial run is usable and a rerun resumes. The summary is
rebuilt from those files.

Usage:
  python tools/learner_seed_replication.py \
      --datasets DIR [DIR ...] --archive ARCHIVE --seeds 1-20 --out OUTDIR \
      [--workers 4] [--summarize-only]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import warnings

import numpy as np

from qemscore.baselines.controls import shuffle_noisy_items
from qemscore.baselines.liao import LiaoMLPMitigator, LiaoMitigator
from qemscore.campaign.analysis import (
    TEST_ROW_FIELDS,
    VALIDATION_ROW_FIELDS,
    _assert_setting,
    _family_mae,
)
from qemscore.campaign.design import (
    ARMS,
    DROPPED_FEATURE,
    FAMILIES,
    TRAINING_SHUFFLE_ARMS,
    TRAINING_SHUFFLE_SEED,
    setting_key,
)
from qemscore.runner.run import _prediction_items

ARM_METHODS = {
    "F": ARMS["capacity_matched"]["full"],          # liao
    "C": ARMS["capacity_matched"]["control"],       # liao-feat-only
    "P": TRAINING_SHUFFLE_ARMS["liao"],             # liao-training-shuffle
}
AFFINE_CONTROL = ARMS["primary"]["control"]         # feat-only (archived A)
ROLES = ("train", "validation", "test")

_CACHE: dict[str, dict] = {}


# --------------------------------------------------------------------------
# Data loading and verification against the archive
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        elif part:
            seeds.append(int(part))
    if len(set(seeds)) != len(seeds):
        raise SystemExit("learner seeds repeat")
    return seeds


def _describe_deviation(key, manifest, archive, record, regenerated_test,
                        regenerated_validation) -> dict:
    """Name every field and cell in which the regenerated rows leave the archive."""
    differing_fields: dict[str, int] = {}
    max_ideal = 0.0
    examples = []
    comparable = (len(regenerated_test) == len(record["test_items"])
                  and len(regenerated_validation) == len(record["validation_items"]))
    if comparable:
        for role, mine, theirs in (
                ("test", regenerated_test, record["test_items"]),
                ("validation", regenerated_validation, record["validation_items"])):
            for left, right in zip(mine, theirs, strict=True):
                for field in left:
                    if left[field] != right.get(field):
                        differing_fields[field] = differing_fields.get(field, 0) + 1
                        if field == "ideal_expectation":
                            max_ideal = max(max_ideal, abs(
                                float(left[field]) - float(right[field])))
                        if len(examples) < 10:
                            examples.append({
                                "role": role, "item_id": left["item_id"], "field": field,
                                "regenerated": left[field], "archived": right.get(field),
                                "family": left["family"], "severity": left["severity"],
                                "observable": left["observable"]})
    archived_manifest_path = archive / "data-manifests" / key / "manifest.json"
    differing_cells = []
    if archived_manifest_path.exists():
        archived_manifest = json.loads(archived_manifest_path.read_text(encoding="utf-8"))
        archived_cells = {cell["cell_id"]: cell for cell in archived_manifest["cells"]}
        for cell in manifest["cells"]:
            other = archived_cells.get(cell["cell_id"], {})
            if cell.get("item_stream_hash") != other.get("item_stream_hash"):
                differing_cells.append({
                    "cell_id": cell["cell_id"], "split": cell["split"],
                    "family": cell["axis_values"]["circuit_family"],
                    "severity": cell["severity"]})
    return {
        "comparable_row_counts": comparable,
        "differing_fields": differing_fields,
        "max_abs_ideal_difference": max_ideal,
        "examples": examples,
        "cells_with_differing_item_stream_hash": differing_cells,
        "note": ("the archive keeps no training rows, so a training-row difference "
                 "is visible only through the cell hashes above"),
    }


def prepare_dataset(data_dir: Path, archive: Path, cache_dir: Path,
                    label_tolerance: float | None = None) -> dict:
    """Load one dataset, verify it against the archived record, cache its rows."""
    from qemscore.validation import validate_split_artifact

    rows, manifest = validate_split_artifact(data_dir)
    seed = int(manifest["master_seed"])
    regime = "shipped"
    # The rehearsal-free frozen size of the primary rows.
    size = 640
    key = setting_key(regime, seed, size)
    if key not in data_dir.name:
        raise SystemExit(f"{data_dir}: directory name does not name {key}")
    structure = _assert_setting(regime, seed, size, rows)
    record_path = archive / "records" / f"{key}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    roles = {role: [row for row in rows if row["split"] == role] for role in ROLES}
    checks = {
        "setting": key,
        "data_dir": str(data_dir),
        "items_jsonl_sha256": _sha256(data_dir / "items.jsonl"),
        "manifest_dataset_hash": str(manifest["dataset_hash"]),
        "archived_dataset_hash": str(record["dataset_hash"]),
        "dataset_hash_matches_archive": str(manifest["dataset_hash"])
        == str(record["dataset_hash"]),
        "split_spec_hash_matches_archive": str(manifest["split_spec_hash"])
        == str(record["split_spec_hash"]),
        "validate_split_artifact": "passed",
        "structure_counts": structure["counts"],
    }
    regenerated_test = [
        {field: row[field] for field in TEST_ROW_FIELDS} for row in roles["test"]]
    regenerated_validation = [
        {field: row[field] for field in VALIDATION_ROW_FIELDS}
        for row in roles["validation"]]
    checks["test_items_identical_to_archive"] = regenerated_test == record["test_items"]
    checks["validation_items_identical_to_archive"] = (
        regenerated_validation == record["validation_items"])
    checks["n_items"] = {role: len(roles[role]) for role in ROLES}
    exact = all(checks[name] for name in (
        "dataset_hash_matches_archive", "split_spec_hash_matches_archive",
        "test_items_identical_to_archive",
        "validation_items_identical_to_archive"))
    checks["exact_match"] = exact
    if not exact:
        checks["deviation"] = _describe_deviation(
            key, manifest, archive, record, regenerated_test, regenerated_validation)
        deviation = checks["deviation"]
        tolerated = (
            label_tolerance is not None
            and checks["split_spec_hash_matches_archive"]
            and deviation["comparable_row_counts"]
            and set(deviation["differing_fields"]) <= {"ideal_expectation"}
            and deviation["max_abs_ideal_difference"] <= label_tolerance
        )
        checks["deviation_tolerated"] = bool(tolerated)
        checks["label_tolerance"] = label_tolerance
        if not tolerated:
            raise SystemExit(f"{key}: regenerated rows do not match the archive: {checks}")
        print(f"{key}: DEVIATION tolerated: {json.dumps(deviation)[:600]}", flush=True)

    payload = {
        "key": key,
        "seed": seed,
        "train": roles["train"],
        "validation": roles["validation"],
        "test": roles["test"],
        "prediction_rows": {
            role: _prediction_items(roles[role]) for role in ("validation", "test")},
        "dataset_hash": str(manifest["dataset_hash"]),
    }
    cache_path = cache_dir / f"{key}.pkl"
    tmp = cache_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    order = {
        "test_item_ids": [str(row["item_id"]) for row in roles["test"]],
        "validation_item_ids": [str(row["item_id"]) for row in roles["validation"]],
    }
    (cache_dir / f"{key}.order.json").write_text(json.dumps(order), encoding="utf-8")
    return checks


def _load(cache_dir: str, key: str) -> dict:
    if key not in _CACHE:
        with open(Path(cache_dir) / f"{key}.pkl", "rb") as handle:
            _CACHE[key] = pickle.load(handle)
    return _CACHE[key]


# --------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------


# numpy linked against Apple Accelerate raises floating-point status flags from
# inside finite float64 matmul kernels; `blas_fpe_probe` demonstrates it on
# standard-normal inputs. Matmul has no division, so a "divide by zero" report
# from it cannot be a real event, and a real overflow or invalid value would
# leave a non-finite array that the finite checks below catch.
SPURIOUS_MATMUL_MESSAGES = frozenset({
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
})


def _warning_rows(caught, phase: str) -> list[dict]:
    """Aggregate caught warnings by kind and location, with occurrence counts."""
    counts: dict[tuple, int] = {}
    for item in caught:
        key = (
            phase,
            item.category.__name__,
            bool(issubclass(item.category, RuntimeWarning)),
            str(item.message)[:300],
            os.path.basename(str(item.filename)),
            int(item.lineno),
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {"phase": key[0], "category": key[1], "is_runtime_warning": key[2],
         "message": key[3], "file": key[4], "lineno": key[5], "count": count}
        for key, count in sorted(counts.items())
    ]


def blas_fpe_probe() -> dict:
    """Show whether a finite matmul of MLP batch shape emits RuntimeWarnings here."""
    rng = np.random.default_rng(0)
    left = rng.normal(size=(32, 9))
    right = rng.normal(size=(9, 64))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        product = left @ right
    return {
        "operation": "standard-normal (32, 9) @ (9, 64) float64, seed 0",
        "inputs_and_output_finite": bool(
            np.all(np.isfinite(left)) and np.all(np.isfinite(right))
            and np.all(np.isfinite(product))),
        "max_abs_output": float(np.max(np.abs(product))),
        "warnings": [str(item.message) for item in caught],
        "numpy_version": np.__version__,
        "blas": _blas_name(),
    }


def _blas_name() -> str:
    try:
        config = np.show_config(mode="dicts")
        return str(config["Build Dependencies"]["blas"]["name"])
    except Exception as exc:  # informational only
        return f"unknown ({type(exc).__name__})"


def _finite_mlp(model: LiaoMLPMitigator) -> bool:
    parameters_ok = all(np.all(np.isfinite(value))
                        for value in model.parameters_.values())
    curve_ok = bool(np.all(np.isfinite(np.asarray(model.loss_curve_, dtype=float))))
    return bool(parameters_ok and curve_ok)


def run_job(job: dict) -> dict:
    """Fit one arm at one learner seed on one dataset and write its outputs."""
    started = time.perf_counter()
    data = _load(job["cache_dir"], job["key"])
    arm = job["arm"]
    k = int(job["learner_seed"])
    shuffle_seed = int(job["shuffle_seed"]) if job.get("shuffle_seed") is not None else None
    drop = (DROPPED_FEATURE,) if arm == "C" else ()
    train = data["train"]
    validation = data["validation"]
    if arm == "P":
        train = shuffle_noisy_items(train, seed=shuffle_seed)

    error = None
    model = None
    with warnings.catch_warnings(record=True) as caught_fit:
        warnings.simplefilter("always")
        try:
            model = LiaoMitigator(random_state=k, drop_features=drop).fit(
                list(train), list(validation))
        except Exception as exc:  # recorded, never swallowed silently
            error = f"fit: {type(exc).__name__}: {exc}"
    fit_warnings = _warning_rows(caught_fit, "training")

    predictions: dict[str, np.ndarray] = {}
    predict_warnings: list[dict] = []
    if model is not None:
        with warnings.catch_warnings(record=True) as caught_predict:
            warnings.simplefilter("always")
            try:
                for role in ("validation", "test"):
                    predictions[role] = np.asarray(
                        model.predict(list(data["prediction_rows"][role])), dtype=float)
            except Exception as exc:
                error = f"predict: {type(exc).__name__}: {exc}"
        predict_warnings = _warning_rows(caught_predict, "prediction")

    finite = {
        "mlp_parameters_and_loss_curve": (
            _finite_mlp(model.candidate_models_["mlp"]) if model is not None else False),
        "selected_predictions": bool(
            predictions
            and all(np.all(np.isfinite(value)) for value in predictions.values())),
    }
    runtime_warnings = [row for row in fit_warnings + predict_warnings
                        if row["is_runtime_warning"]]
    # The frozen rule, literally: any non-finite value or numerical
    # RuntimeWarning during training or prediction.
    flagged = bool(error is not None or runtime_warnings or not all(finite.values()))
    # Secondary flag, reported beside the literal one and never used by the
    # decision: the same rule with the spurious Accelerate matmul flags removed.
    other_runtime = [row for row in runtime_warnings
                     if row["message"] not in SPURIOUS_MATMUL_MESSAGES]
    flagged_secondary = bool(
        error is not None or other_runtime or not all(finite.values()))

    # Candidate-level test errors, outside the flagged window: diagnostic only.
    candidate_test_mae = {}
    if model is not None and error is None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for name, candidate in model.candidate_models_.items():
                values = np.asarray(candidate.predict(list(data["prediction_rows"]["test"])),
                                    dtype=float)
                if np.all(np.isfinite(values)):
                    candidate_test_mae[name] = _family_mae(
                        data["test"], values, artifact_id=data["dataset_hash"])

    result = {
        "key": job["key"],
        "dataset_seed": int(data["seed"]),
        "arm": arm,
        "method": ARM_METHODS[arm],
        "learner_seed": k,
        "shuffle_seed": shuffle_seed,
        "purpose": job["purpose"],
        "error": error,
        "flagged_non_converged": flagged,
        "flagged_excluding_spurious_matmul_fpe": flagged_secondary,
        "finite_checks": finite,
        "n_runtime_warnings": int(sum(row["count"] for row in runtime_warnings)),
        "n_runtime_warnings_not_matmul_fpe": int(
            sum(row["count"] for row in other_runtime)),
        "warnings_aggregated": fit_warnings + predict_warnings,
        "selected_model": None if model is None else model.selected_model_name_,
        "eligible_models": None if model is None else list(model.eligible_models_),
        "one_standard_error_threshold": (
            None if model is None else model.one_standard_error_threshold_),
        "validation_scores": None if model is None else [
            {"name": s.name, "validation_mae": s.validation_mae,
             "standard_error": s.standard_error,
             "total_excess_absolute_loss": s.total_excess_absolute_loss}
            for s in model.validation_scores_],
        "mlp_n_iter": None if model is None else int(
            model.candidate_models_["mlp"].n_iter_),
        "mlp_final_train_loss": None if model is None else float(
            model.candidate_models_["mlp"].loss_curve_[-1]),
        "test_family_mae": (
            _family_mae(data["test"], predictions["test"],
                        artifact_id=data["dataset_hash"])
            if "test" in predictions and finite["selected_predictions"] else None),
        "validation_family_mae": (
            _family_mae(data["validation"], predictions["validation"],
                        artifact_id=data["dataset_hash"])
            if "validation" in predictions and finite["selected_predictions"] else None),
        "candidate_test_family_mae": candidate_test_mae,
        "seconds": time.perf_counter() - started,
        "pid": os.getpid(),
    }

    out = Path(job["fit_dir"])
    stem = job["stem"]
    if predictions:
        tmp = out / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, **{role: predictions[role] for role in predictions})
        os.replace(tmp, out / f"{stem}.npz")
    tmp_json = out / f"{stem}.json.tmp"
    tmp_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    os.replace(tmp_json, out / f"{stem}.json")
    return {"stem": stem, "seconds": result["seconds"], "flagged": flagged,
            "flagged_secondary": flagged_secondary,
            "error": error, "selected": result["selected_model"]}


def build_jobs(keys_seeds: list[tuple[str, int]], seeds: list[int], cache_dir: Path,
               fit_dir: Path) -> list[dict]:
    jobs = []
    for key, dataset_seed in keys_seeds:
        # Original-seed reproduction first, so a mismatch shows early.
        for arm, shuffle_seed, purpose in (
                ("F", None, "original_seed_check"),
                ("C", None, "original_seed_check"),
                ("P", TRAINING_SHUFFLE_SEED, "original_seed_check_archived_shuffle")):
            jobs.append({
                "key": key, "arm": arm, "learner_seed": dataset_seed,
                "shuffle_seed": shuffle_seed, "purpose": purpose,
                "stem": f"{key}__orig__{arm}",
                "cache_dir": str(cache_dir), "fit_dir": str(fit_dir)})
    for k in seeds:
        for key, _ in keys_seeds:
            for arm in ("F", "C", "P"):
                jobs.append({
                    "key": key, "arm": arm, "learner_seed": k,
                    "shuffle_seed": k if arm == "P" else None,
                    "purpose": "replication",
                    "stem": f"{key}__k{k:02d}__{arm}",
                    "cache_dir": str(cache_dir), "fit_dir": str(fit_dir)})
    return jobs


# --------------------------------------------------------------------------
# Summary, rebuilt from the per-fit files
# --------------------------------------------------------------------------


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q))
            for q in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)}


def summarize(args, seeds: list[int], dataset_checks: dict, keys_seeds,
              fit_dir: Path, cache_dir: Path, runtime: dict) -> dict:
    archive = Path(args.archive)
    summary = {
        "schema": "learner-seed-replication-summary-v1",
        "frozen_rule": "frozen-rule-seed-replication.md (2026-10-01)",
        "learner_seeds": seeds,
        "dataset_checks": dataset_checks,
        "runtime": runtime,
        "blas_fpe_probe": blas_fpe_probe(),
        "original_seed_check": {},
        "rows": {},
        "fits_missing": [],
        "fits_with_errors": [],
    }
    for key, dataset_seed in keys_seeds:
        record = json.loads((archive / "records" / f"{key}.json").read_text(
            encoding="utf-8"))
        data = _load(str(cache_dir), key)
        test_ids = [str(row["item_id"]) for row in data["test"]]
        val_ids = [str(row["item_id"]) for row in data["validation"]]
        archived_test = {
            method: np.asarray([record["test_predictions"][method][i] for i in test_ids])
            for method in (ARM_METHODS["F"], ARM_METHODS["C"], AFFINE_CONTROL)}
        archived_mae = {
            method: _family_mae(data["test"], values, artifact_id=data["dataset_hash"])
            for method, values in archived_test.items()}

        # Original-seed reproduction.
        check = {}
        for arm in ("F", "C", "P"):
            stem = f"{key}__orig__{arm}"
            meta_path = fit_dir / f"{stem}.json"
            if not meta_path.exists():
                check[arm] = {"status": "missing"}
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = np.load(fit_dir / f"{stem}.npz")
            archived_method = ARM_METHODS[arm]
            entry = {
                "method": archived_method,
                "learner_seed": meta["learner_seed"],
                "shuffle_seed": meta["shuffle_seed"],
                "selected_model": meta["selected_model"],
                "archived_selected_model":
                    record["selected_configs"][archived_method]["selected_model"],
                "flagged_non_converged": meta["flagged_non_converged"],
                "flagged_excluding_spurious_matmul_fpe":
                    meta["flagged_excluding_spurious_matmul_fpe"],
            }
            archived_val = np.asarray(
                [record["validation_predictions"][archived_method][i] for i in val_ids])
            entry["max_abs_diff_validation"] = float(
                np.max(np.abs(arrays["validation"] - archived_val)))
            if archived_method in record["test_predictions"]:
                archived = np.asarray(
                    [record["test_predictions"][archived_method][i] for i in test_ids])
                entry["max_abs_diff_test"] = float(
                    np.max(np.abs(arrays["test"] - archived)))
                entry["refit_test_family_mae"] = meta["test_family_mae"]
                entry["archived_test_family_mae"] = archived_mae[archived_method]
            else:
                entry["max_abs_diff_test"] = None
                entry["note"] = "the archive keeps validation predictions only for this arm"
            entry["archived_validation_scores"] = [
                {"name": s["name"], "validation_mae": s["validation_mae"]}
                for s in record["selected_configs"][archived_method]["validation_scores"]]
            entry["refit_validation_scores"] = [
                {"name": s["name"], "validation_mae": s["validation_mae"]}
                for s in meta["validation_scores"]]
            check[arm] = entry
        summary["original_seed_check"][key] = check

        # 20-seed rows.
        per_seed: dict[int, dict] = {}
        for k in seeds:
            entry = {}
            for arm in ("F", "C", "P"):
                stem = f"{key}__k{k:02d}__{arm}"
                meta_path = fit_dir / f"{stem}.json"
                if not meta_path.exists():
                    summary["fits_missing"].append(stem)
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["error"] is not None:
                    summary["fits_with_errors"].append({"stem": stem, "error": meta["error"]})
                entry[arm] = meta
            per_seed[k] = entry

        for family in FAMILIES:
            row_name = f"{key}/{family}"
            A = archived_mae[AFFINE_CONTROL][family]
            archived_F = archived_mae[ARM_METHODS["F"]][family]
            archived_C = archived_mae[ARM_METHODS["C"]][family]
            table = []
            for k in seeds:
                entry = per_seed[k]
                if not all(arm in entry and entry[arm]["test_family_mae"] is not None
                           for arm in ("F", "C", "P")):
                    continue
                F = entry["F"]["test_family_mae"][family]
                C = entry["C"]["test_family_mae"][family]
                P = entry["P"]["test_family_mae"][family]
                table.append({
                    "learner_seed": k,
                    "F": F, "C": C, "P": P,
                    "D": C - F,
                    "D_over_C": (C - F) / C,
                    "S": (A - C) / (A - F) if (A - F) != 0 else None,
                    "P_minus_C": P - C,
                    "P_minus_F": P - F,
                    "selected": {arm: entry[arm]["selected_model"] for arm in ("F", "C", "P")},
                    "flagged": {arm: bool(entry[arm]["flagged_non_converged"])
                                for arm in ("F", "C", "P")},
                    "flagged_secondary": {
                        arm: bool(entry[arm]["flagged_excluding_spurious_matmul_fpe"])
                        for arm in ("F", "C", "P")},
                    "mlp_n_iter": {arm: entry[arm]["mlp_n_iter"] for arm in ("F", "C", "P")},
                })

            def block(rows_in, subset_label):
                selections = {
                    arm: {name: sum(1 for row in rows_in if row["selected"][arm] == name)
                          for name in ("random_forest", "mlp")}
                    for arm in ("F", "C", "P")}
                return {
                    "subset": subset_label,
                    "n_seeds": len(rows_in),
                    "F": _stats([row["F"] for row in rows_in]),
                    "C": _stats([row["C"] for row in rows_in]),
                    "P": _stats([row["P"] for row in rows_in]),
                    "D": _stats([row["D"] for row in rows_in]),
                    "fraction_D_positive": (
                        float(np.mean([row["D"] > 0 for row in rows_in]))
                        if rows_in else None),
                    "D_over_C": {**_stats([row["D_over_C"] for row in rows_in]),
                                 "quantiles": _quantiles([row["D_over_C"] for row in rows_in])},
                    "S_mean": (float(np.mean([row["S"] for row in rows_in]))
                               if rows_in and all(row["S"] is not None for row in rows_in)
                               else None),
                    "S": _stats([row["S"] for row in rows_in if row["S"] is not None]),
                    "P_minus_C": _stats([row["P_minus_C"] for row in rows_in]),
                    "P_minus_F": _stats([row["P_minus_F"] for row in rows_in]),
                    "selected_candidates": selections,
                }

            unflagged_D = [row for row in table
                           if not (row["flagged"]["F"] or row["flagged"]["C"])]
            unflagged_all = [row for row in table if not any(row["flagged"].values())]
            unflagged2_D = [row for row in table
                            if not (row["flagged_secondary"]["F"]
                                    or row["flagged_secondary"]["C"])]
            unflagged2_all = [row for row in table
                              if not any(row["flagged_secondary"].values())]
            summary["rows"][row_name] = {
                "setting": key,
                "family": family,
                "archived_A_feat_only": A,
                "archived_original_seed": {
                    "learner_seed": dataset_seed,
                    "F": archived_F, "C": archived_C,
                    "D": archived_C - archived_F,
                    "D_over_C": (archived_C - archived_F) / archived_C,
                    "S": (A - archived_C) / (A - archived_F),
                },
                "per_seed": table,
                "all_fits": block(table, "all fits (decision basis)"),
                "excluding_flagged_F_or_C": block(
                    unflagged_D, "seeds whose F and C fits are both unflagged"),
                "excluding_any_flagged_arm": block(
                    unflagged_all, "seeds whose F, C and P fits are all unflagged"),
                "flag_counts": {
                    arm: sum(1 for row in table if row["flagged"][arm])
                    for arm in ("F", "C", "P")},
                "excluding_flagged_F_or_C_secondary": block(
                    unflagged2_D, "seeds whose F and C fits are unflagged under the "
                    "secondary flag (spurious Accelerate matmul flags removed)"),
                "excluding_any_flagged_arm_secondary": block(
                    unflagged2_all, "seeds whose F, C and P fits are unflagged under "
                    "the secondary flag"),
                "flag_counts_secondary": {
                    arm: sum(1 for row in table if row["flagged_secondary"][arm])
                    for arm in ("F", "C", "P")},
            }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--label-tolerance", type=float, default=None,
                        help="tolerate validation ideal-label differences up to this "
                             "absolute size, recorded as a deviation (default: exact)")
    parser.add_argument("--limit-jobs", type=int, default=None,
                        help="smoke test: run only the first N jobs")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    out = args.out.resolve()
    cache_dir = out / "cache"
    fit_dir = out / "fits"
    for directory in (out, cache_dir, fit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    checks_path = out / "dataset_checks.json"
    keys_seeds: list[tuple[str, int]] = []
    if args.summarize_only and checks_path.exists():
        dataset_checks = json.loads(checks_path.read_text(encoding="utf-8"))
        for data_dir in args.datasets:
            match = [key for key in dataset_checks if key in data_dir.name]
            keys_seeds.append((match[0], int(match[0].split("-s")[1].split("-")[0])))
    else:
        dataset_checks = {}
        for data_dir in args.datasets:
            started = time.perf_counter()
            checks = prepare_dataset(data_dir.resolve(), args.archive, cache_dir,
                                     label_tolerance=args.label_tolerance)
            checks["load_and_verify_seconds"] = time.perf_counter() - started
            dataset_checks[checks["setting"]] = checks
            keys_seeds.append((checks["setting"], int(checks["setting"].split("-s")[1]
                                                     .split("-")[0])))
            print(f"verified {checks['setting']} in {checks['load_and_verify_seconds']:.1f}s",
                  flush=True)
        checks_path.write_text(json.dumps(dataset_checks, indent=1), encoding="utf-8")

    fit_seconds = None
    if not args.summarize_only:
        jobs = build_jobs(keys_seeds, seeds, cache_dir, fit_dir)
        pending = [job for job in jobs if not (fit_dir / f"{job['stem']}.json").exists()]
        if args.limit_jobs is not None:
            pending = pending[: args.limit_jobs]
        print(f"{len(jobs)} jobs, {len(pending)} pending, {args.workers} workers",
              flush=True)
        fit_start = time.perf_counter()
        log = open(out / "progress.log", "a", encoding="utf-8")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, job): job for job in pending}
            done = 0
            for future in as_completed(futures):
                job = futures[future]
                done += 1
                try:
                    info = future.result()
                    line = (f"[{done}/{len(pending)}] {info['stem']} "
                            f"{info['seconds']:.1f}s sel={info['selected']} "
                            f"flagged={info['flagged']}/{info['flagged_secondary']} err={info['error']} "
                            f"elapsed={time.perf_counter() - fit_start:.0f}s")
                except Exception:
                    line = f"[{done}/{len(pending)}] {job['stem']} CRASH\n{traceback.format_exc()}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
        log.close()
        fit_seconds = time.perf_counter() - fit_start

    fit_times = []
    for path in sorted(fit_dir.glob("*.json")):
        fit_times.append(json.loads(path.read_text(encoding="utf-8"))["seconds"])
    runtime = {
        "this_invocation_wall_seconds_before_summary": time.perf_counter() - wall_start,
        "this_invocation_fit_phase_wall_seconds": fit_seconds,
        "n_fit_files": len(fit_times),
        "sum_of_per_fit_seconds": float(sum(fit_times)),
        "workers": args.workers,
    }
    summary = summarize(args, seeds, dataset_checks, keys_seeds, fit_dir, cache_dir,
                        runtime)
    target = out / "summary.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    os.replace(tmp, target)
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
