"""Shared fitting machinery for the descriptor-information experiment.

Frozen rule: docs/frozen-rules/2026-10-02-descriptor-information.md. Part A
(tools/descriptor_ladder.py) and Part B (tools/qaoa_intermediate.py) both fit
through this module, so the two parts share one fitting path, one output
format, and one convergence flag.

What one Liao fit does, exactly as tools/learner_seed_replication.py does it:

- F: ``LiaoMitigator(random_state=k, feature_builder=B, feature_names=N)``
- C: the same with ``drop_features=("noisy_expectation",)``
- P: F refitted on ``shuffle_noisy_items(train, seed=k)``

with the full random-forest and MLP candidate search and the one-standard-error
selection inside ``LiaoMitigator``, predicting on ``_prediction_items`` rows.
The only change from the campaign path is the feature-builder hook.

What one fit writes, so a partial run resumes and nothing has to be refitted:

- ``<stem>.npz`` (compressed): the selected candidate's validation and test
  predictions (``validation``, ``test``) and each candidate's predictions
  (``validation__random_forest``, ``test__random_forest``,
  ``validation__mlp``, ``test__mlp``), aligned to the cached row order;
- ``<stem>.json``: arm, rung, seeds, feature names, the selected candidate,
  the eligible set, the one-standard-error threshold, every candidate's
  validation score, per-family macro MAE for the selected candidate and for
  each candidate, timing, the literal convergence flag of Appendix M.4 and its
  secondary version without the spurious Accelerate matmul warnings.

The JSON is written last and atomically, so its presence marks a complete fit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import qemscore  # noqa: E402

if not Path(qemscore.__file__).resolve().is_relative_to(REPO_ROOT):
    raise SystemExit(
        f"qemscore resolves to {qemscore.__file__}, outside {REPO_ROOT}; "
        f"run with PYTHONPATH={REPO_ROOT}")

from qemscore.baselines.controls import shuffle_noisy_items  # noqa: E402
from qemscore.baselines.liao import LiaoMitigator  # noqa: E402
from qemscore.baselines.ridge import RidgeMitigator  # noqa: E402
from qemscore.campaign.analysis import _family_mae  # noqa: E402
from qemscore.datasets.schema import FEATURES  # noqa: E402
from qemscore.runner.run import _RidgeRunnerMethod  # noqa: E402

from tools.learner_seed_replication import (  # noqa: E402
    SPURIOUS_MATMUL_MESSAGES,
    _finite_mlp,
    _parse_seeds,
    _warning_rows,
    blas_fpe_probe,
)

RULE_FILE = "docs/frozen-rules/2026-10-02-descriptor-information.md"
RULE_SEED = 20261002
LEARNER_SEEDS = tuple(range(1, 21))
GATE_TOLERANCE = 1e-11
# The frozen gate is criterion "archive" at GATE_TOLERANCE. Any other criterion
# or tolerance is an amendment of the frozen rule: the gate refuses it unless
# the amendment text is already written into the rule file (see
# check_gate_amendment), and it records the amendment in gate_r0.json.
FROZEN_GATE_CRITERION = "archive"
GATE_CRITERIA = {
    "archive": ("max |refit - archived| over the selected candidate's test predictions "
                "of F and C at the original learner seed, through the feature-builder "
                "hook with build_features and FEATURES, at most the tolerance"),
    "m4-refit-identity": ("the hooked F and C refits at the original learner seed equal "
                          "the pre-hook Appendix M.4 refits on this machine exactly "
                          "(test and validation, every dataset); the archive difference "
                          "is reported beside it"),
}
NOISY = "noisy_expectation"
CANDIDATES = ("random_forest", "mlp")
LIAO_ARMS = ("F", "C", "P")

# Columns kept at every rung and by every arm (the arm decides whether it reads
# the noisy estimate): noisy estimate, log2 shots, qubit count, family
# indicators, two-qubit gate count, compiled depth, observable locality.
KEPT = (
    "noisy_expectation",
    "log2_shots",
    "n_qubits",
    "family_tfi",
    "family_qaoa",
    "family_heisenberg",
    "family_random_clifford",
    "family_near_clifford",
    "two_qubit_gates",
    "transpiled_depth",
    "obs_locality",
)
assert all(name in FEATURES for name in KEPT)

STRENGTH_FEATURE_NAME = "noise_strength_L3"


def severity_indicator(item: Mapping[str, object]) -> float:
    """Return 0.0 for severity 'L1', 1.0 for 'L3'; raise on any other value."""
    sev = item.get("severity")
    if sev == "L1":
        return 0.0
    if sev == "L3":
        return 1.0
    raise ValueError(f"Unknown or missing severity: {sev!r}; expected 'L1' or 'L3'")


__all__ = [
    "CANDIDATES", "FEATURES", "FROZEN_GATE_CRITERION", "GATE_CRITERIA",
    "GATE_TOLERANCE", "KEPT", "LEARNER_SEEDS", "LIAO_ARMS", "NOISY", "REPO_ROOT",
    "RULE_FILE", "RULE_SEED", "STRENGTH_FEATURE_NAME", "_parse_seeds", "blas_fpe_probe",
    "check_gate_amendment", "drive", "gate_decision", "load_cache", "print_plan",
    "require_gate", "rule_path", "run_affine_fit", "run_liao_fit", "severity_indicator",
    "sha256_file", "write_cache", "write_json",
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


_CACHE: dict[str, dict] = {}


def load_cache(path: str | Path) -> dict:
    key = str(path)
    if key not in _CACHE:
        with open(path, "rb") as handle:
            _CACHE[key] = pickle.load(handle)
    return _CACHE[key]


def _versions() -> dict:
    import sklearn

    return {"numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "qemscore_path": str(Path(qemscore.__file__).resolve().parent)}


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def rule_path() -> Path:
    return REPO_ROOT / RULE_FILE


def _normalized(text: str) -> str:
    return " ".join(text.split())


def check_gate_amendment(*, criterion: str, tolerance: float, amendment: str | None,
                         seedrep_fits: Path | None,
                         rule_file: Path | None = None) -> dict:
    """Decide, before any work, whether the requested gate is the frozen one.

    The frozen gate is criterion ``archive`` at ``GATE_TOLERANCE``. Any other
    criterion or tolerance amends the frozen rule, which only the author can
    do, and the rule requires the amendment to be disclosed before any fit. So
    an amended gate runs only when ``amendment`` is given and its text
    (whitespace-normalized) already appears in the rule file.
    """
    if criterion not in GATE_CRITERIA:
        raise SystemExit(f"unknown gate criterion {criterion!r}; "
                         f"choose from {sorted(GATE_CRITERIA)}")
    if criterion == "m4-refit-identity" and seedrep_fits is None:
        raise SystemExit("criterion m4-refit-identity needs --seedrep-fits")
    amended = criterion != FROZEN_GATE_CRITERION or tolerance != GATE_TOLERANCE
    if not amended:
        if amendment:
            raise SystemExit("--amendment was given, but the requested gate is the "
                             "frozen one (archive at 1e-11); drop --amendment")
        return {"amended": False, "amendment": None}
    if not amendment or not amendment.strip():
        raise SystemExit(
            f"criterion {criterion!r} at tolerance {tolerance!r} amends the frozen "
            f"gate (archive at {GATE_TOLERANCE!r}). Only the author can amend it: "
            "write the amendment into the rule's Deviations section first, then pass "
            "the same text with --amendment.")
    path = Path(rule_file) if rule_file is not None else rule_path()
    if _normalized(amendment) not in _normalized(path.read_text(encoding="utf-8")):
        raise SystemExit(
            f"the --amendment text does not appear in {path}; the frozen rule "
            "requires the amendment to be written into the rule before the gate runs")
    return {"amended": True, "amendment": amendment.strip()}


def gate_decision(*, criterion: str, tolerance: float, archive_max: float,
                  refit_identity_max: float | None, crashes: Sequence[str]) -> dict:
    """The gate's pass decision; the frozen criterion is always reported beside it."""
    frozen_passed = bool(archive_max <= GATE_TOLERANCE and not crashes)
    if criterion == "archive":
        passed = bool(archive_max <= tolerance and not crashes)
    elif criterion == "m4-refit-identity":
        passed = bool(refit_identity_max is not None and refit_identity_max == 0.0
                      and not crashes)
    else:
        raise ValueError(f"unknown gate criterion {criterion!r}")
    return {"passed": passed, "frozen_criterion_passed": frozen_passed}


def require_gate(gate_file: Path) -> dict:
    """Refuse every new fit unless the R0 reproduction gate passed (frozen rule).

    A gate that passed under a criterion or tolerance other than the frozen one
    counts only if it records the amendment it ran under; that amendment is
    printed so every run discloses it.
    """
    if not gate_file.exists():
        raise SystemExit(
            f"{gate_file} is missing: run `tools/descriptor_ladder.py gate` first. "
            "The frozen rule allows no new fit before the R0 gate passes.")
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if gate.get("passed") is not True:
        raise SystemExit(
            f"{gate_file}: the R0 reproduction gate did not pass "
            f"(max test difference {gate.get('max_abs_diff_test')} against tolerance "
            f"{gate.get('tolerance')}); the frozen rule allows no new fit.")
    criterion = gate.get("criterion_id", FROZEN_GATE_CRITERION)
    frozen = criterion == FROZEN_GATE_CRITERION and gate.get("tolerance") == GATE_TOLERANCE
    if not frozen and not gate.get("amendment"):
        raise SystemExit(
            f"{gate_file}: the gate passed under criterion {criterion!r} at tolerance "
            f"{gate.get('tolerance')!r}, which is not the frozen gate, and records no "
            "amendment; the frozen rule allows no new fit.")
    if gate.get("amendment"):
        print(f"NOTE: the R0 gate passed under a disclosed amendment (criterion "
              f"{criterion!r}, tolerance {gate.get('tolerance')!r}; frozen criterion "
              f"passed: {gate.get('frozen_criterion_passed')}): {gate['amendment']}",
              flush=True)
    recorded = gate.get("rule_file_sha256")
    if recorded and rule_path().exists() and recorded != sha256_file(rule_path()):
        print(f"WARNING: {RULE_FILE} changed after the gate ran (gate recorded "
              f"{recorded[:12]}); rerun the gate, which reuses its fits.", flush=True)
    return gate


# --------------------------------------------------------------------------
# One Liao fit
# --------------------------------------------------------------------------


def _family_mae_or_none(rows, values, dataset_hash):
    array = np.asarray(values, dtype=float)
    if array.shape != (len(rows),) or not np.all(np.isfinite(array)):
        return None
    return _family_mae(rows, array, artifact_id=dataset_hash)


def run_liao_fit(job: Mapping[str, object], data: Mapping[str, object],
                 builder: Callable[[dict], Sequence[float]],
                 names: Sequence[str]) -> dict:
    """Fit one Liao arm at one learner seed and write its outputs."""
    started = time.perf_counter()
    fit_arm = str(job["fit_arm"])
    if fit_arm not in LIAO_ARMS:
        raise ValueError(f"unknown Liao arm {fit_arm!r}")
    k = int(job["learner_seed"])
    shuffle_seed = (int(job["shuffle_seed"]) if job.get("shuffle_seed") is not None
                    else None)
    drop = (NOISY,) if fit_arm == "C" else ()
    train = list(data["train"])
    validation = list(data["validation"])
    if fit_arm == "P":
        if shuffle_seed is None:
            raise ValueError("arm P needs a shuffle seed")
        train = shuffle_noisy_items(train, seed=shuffle_seed)

    error = None
    model = None
    timing: dict[str, float] = {}
    with warnings.catch_warnings(record=True) as caught_fit:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        try:
            model = LiaoMitigator(
                random_state=k, drop_features=drop,
                feature_builder=builder, feature_names=tuple(names),
            ).fit(train, validation)
        except Exception as exc:  # recorded, never swallowed silently
            error = f"fit: {type(exc).__name__}: {exc}"
        timing["fit_seconds"] = time.perf_counter() - t0
    fit_warnings = _warning_rows(caught_fit, "training")

    predictions: dict[str, np.ndarray] = {}
    predict_warnings: list[dict] = []
    if model is not None:
        with warnings.catch_warnings(record=True) as caught_predict:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            try:
                for role in ("validation", "test"):
                    predictions[role] = np.asarray(
                        model.predict(list(data["prediction_rows"][role])), dtype=float)
            except Exception as exc:
                error = f"predict: {type(exc).__name__}: {exc}"
            timing["predict_seconds"] = time.perf_counter() - t0
        predict_warnings = _warning_rows(caught_predict, "prediction")

    finite = {
        "mlp_parameters_and_loss_curve": (
            _finite_mlp(model.candidate_models_["mlp"])
            if model is not None and error is None else False),
        "selected_predictions": bool(
            len(predictions) == 2
            and all(np.all(np.isfinite(value)) for value in predictions.values())),
    }
    runtime_warnings = [row for row in fit_warnings + predict_warnings
                        if row["is_runtime_warning"]]
    # Appendix M.4's rule, literally: any non-finite value or numerical
    # RuntimeWarning during training or prediction of the selected model.
    flagged = bool(error is not None or runtime_warnings or not all(finite.values()))
    # Secondary flag, reported beside it: the spurious Accelerate matmul flags removed.
    other_runtime = [row for row in runtime_warnings
                     if row["message"] not in SPURIOUS_MATMUL_MESSAGES]
    flagged_secondary = bool(error is not None or other_runtime
                             or not all(finite.values()))

    # Each candidate's predictions, outside the flagged window (the flag keeps
    # its Appendix M.4 definition); their warnings are recorded separately.
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_warnings: list[dict] = []
    selected_matches_candidate = None
    if model is not None and error is None:
        with warnings.catch_warnings(record=True) as caught_candidates:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            for name in CANDIDATES:
                candidate = model.candidate_models_[name]
                for role in ("validation", "test"):
                    candidate_predictions[f"{role}__{name}"] = np.asarray(
                        candidate.predict(list(data["prediction_rows"][role])),
                        dtype=float)
            timing["candidate_predict_seconds"] = time.perf_counter() - t0
        candidate_warnings = _warning_rows(caught_candidates, "candidate_prediction")
        selected = model.selected_model_name_
        selected_matches_candidate = bool(all(
            np.array_equal(predictions[role], candidate_predictions[f"{role}__{selected}"])
            for role in ("validation", "test")))

    dataset_hash = str(data["dataset_hash"])
    result = {
        "schema": "descriptor-information-fit-v1",
        "frozen_rule": RULE_FILE,
        "part": job["part"],
        "key": job["key"],
        "dataset_seed": int(data["seed"]),
        "dataset_hash": dataset_hash,
        "rung": job["rung"],
        "arm": job["arm"],
        "fit_arm": fit_arm,
        "method": {"F": "liao", "C": "liao-feat-only",
                   "P": "liao-training-shuffle"}[fit_arm],
        "learner_seed": k,
        "shuffle_seed": shuffle_seed,
        "purpose": job.get("purpose", "rung_fit"),
        **({"strength_indicator": True} if job.get("strength_indicator") else {}),
        "dropped_features": list(drop),
        "feature_builder": getattr(builder, "name", None),
        "n_features": len(names),
        "feature_names": list(names),
        "feature_names_sha256": _names_sha256(names),
        "error": error,
        "flagged_non_converged": flagged,
        "flagged_excluding_spurious_matmul_fpe": flagged_secondary,
        "finite_checks": finite,
        "n_runtime_warnings": int(sum(row["count"] for row in runtime_warnings)),
        "n_runtime_warnings_not_matmul_fpe": int(
            sum(row["count"] for row in other_runtime)),
        "warnings_aggregated": fit_warnings + predict_warnings,
        "candidate_prediction_warnings": candidate_warnings,
        "selected_model": None if model is None else model.selected_model_name_,
        "selected_equals_candidate_predictions": selected_matches_candidate,
        "eligible_models": None if model is None else list(model.eligible_models_),
        "one_standard_error_threshold": (
            None if model is None else model.one_standard_error_threshold_),
        "validation_scores": None if model is None else [
            {"name": s.name, "validation_mae": s.validation_mae,
             "standard_error": s.standard_error,
             "total_excess_absolute_loss": s.total_excess_absolute_loss,
             "circuit_evaluations": s.circuit_evaluations,
             "simplicity_rank": s.simplicity_rank}
            for s in model.validation_scores_],
        "mlp_n_iter": None if model is None else int(
            model.candidate_models_["mlp"].n_iter_),
        "mlp_final_train_loss": None if model is None or error is not None else float(
            model.candidate_models_["mlp"].loss_curve_[-1]),
        "test_family_mae": (
            _family_mae_or_none(data["test"], predictions["test"], dataset_hash)
            if "test" in predictions else None),
        "validation_family_mae": (
            _family_mae_or_none(data["validation"], predictions["validation"],
                                dataset_hash)
            if "validation" in predictions else None),
        "candidate_test_family_mae": {
            name: _family_mae_or_none(data["test"],
                                      candidate_predictions[f"test__{name}"],
                                      dataset_hash)
            for name in CANDIDATES if f"test__{name}" in candidate_predictions},
        "candidate_validation_family_mae": {
            name: _family_mae_or_none(data["validation"],
                                      candidate_predictions[f"validation__{name}"],
                                      dataset_hash)
            for name in CANDIDATES if f"validation__{name}" in candidate_predictions},
        "n_rows": {"train": len(train), "validation": len(validation),
                   "test": len(data["test"])},
        "timing": timing,
        "seconds": time.perf_counter() - started,
        "pid": os.getpid(),
        "versions": _versions(),
    }
    _write_fit(job, result, {**predictions, **candidate_predictions})
    return {"stem": job["stem"], "seconds": result["seconds"], "flagged": flagged,
            "flagged_secondary": flagged_secondary, "error": error,
            "selected": result["selected_model"]}


def _write_fit(job, result: dict, arrays: Mapping[str, np.ndarray]) -> None:
    out = Path(job["fit_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = str(job["stem"])
    if arrays:
        tmp = out / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, **dict(arrays))
        os.replace(tmp, out / f"{stem}.npz")
    write_json(out / f"{stem}.json", result)


# --------------------------------------------------------------------------
# One affine control fit (arm A)
# --------------------------------------------------------------------------


class _BuilderFeatureOnlyRidge(RidgeMitigator):
    """The campaign's ``FeatureOnlyControl`` over a rung's columns.

    ``FeatureOnlyControl`` deletes the noisy-expectation column from the
    schema vector; this deletes it from the rung builder's vector. With the
    default builder the matrix is the same array the campaign built.
    """

    def __init__(self, builder, names: Sequence[str], cv: int = 5,
                 random_state: int = 0) -> None:
        super().__init__(cv=cv, random_state=random_state)
        self._builder = builder
        self._n_features = len(names)
        self.columns = tuple(i for i, name in enumerate(names) if name != NOISY)
        if len(self.columns) != len(names) - (1 if NOISY in names else 0):
            raise ValueError("builder names are inconsistent")

    def _matrix(self, items: list[dict]) -> np.ndarray:
        full = np.array([self._builder(dict(it)) for it in items])
        if full.ndim != 2 or full.shape[1] != self._n_features:
            raise ValueError("feature builder output disagrees with its names")
        return full[:, list(self.columns)]


def run_affine_fit(job: Mapping[str, object], data: Mapping[str, object],
                   builder, names: Sequence[str]) -> dict:
    """Arm A: the campaign's affine feature-only control on one rung's columns.

    Fitted once per dataset and rung exactly as `_fit_arm("feat-only")` fits it:
    ridge over standardized columns at each alpha of the frozen grid on the
    training rows, then source-validation one-standard-error selection inside
    ``_RidgeRunnerMethod``; predictions on ``_prediction_items`` rows.
    """
    started = time.perf_counter()
    error = None
    predictions: dict[str, np.ndarray] = {}
    selection = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            method = _RidgeRunnerMethod(
                lambda: _BuilderFeatureOnlyRidge(builder, names),
                source_measurements=False, test_measurements=False,
                legacy_reports_alpha=False)
            method.fit(list(data["train"]), manifest={}, split_v2=True)
            method.select(list(data["validation"]))
            for role in ("validation", "test"):
                output = method.predict(list(data["prediction_rows"][role]), manifest={})
                predictions[role] = np.asarray(output.predictions, dtype=float)
            selection = output.config
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"
    dataset_hash = str(data["dataset_hash"])
    result = {
        "schema": "descriptor-information-fit-v1",
        "frozen_rule": RULE_FILE,
        "part": job["part"],
        "key": job["key"],
        "dataset_seed": int(data["seed"]),
        "dataset_hash": dataset_hash,
        "rung": job["rung"],
        "arm": "A",
        "method": "feat-only (affine ridge over the rung's columns without the "
                  "noisy estimate)",
        "purpose": job.get("purpose", "rung_fit"),
        **({"strength_indicator": True} if job.get("strength_indicator") else {}),
        "feature_builder": getattr(builder, "name", None),
        "n_features": len(names) - (1 if NOISY in names else 0),
        "feature_names": [name for name in names if name != NOISY],
        "error": error,
        "selection": _json_safe(selection),
        "warnings_aggregated": _warning_rows(caught, "training_and_prediction"),
        "test_family_mae": (_family_mae_or_none(data["test"], predictions["test"],
                                                dataset_hash)
                            if "test" in predictions else None),
        "validation_family_mae": (
            _family_mae_or_none(data["validation"], predictions["validation"],
                                dataset_hash)
            if "validation" in predictions else None),
        "seconds": time.perf_counter() - started,
        "versions": _versions(),
    }
    _write_fit(job, result, predictions)
    return {"stem": job["stem"], "seconds": result["seconds"], "flagged": None,
            "flagged_secondary": None, "error": error, "selected": "ridge"}


def _json_safe(value):
    return json.loads(json.dumps(value, default=lambda obj: repr(obj)))


# --------------------------------------------------------------------------
# Planning and driving
# --------------------------------------------------------------------------


def print_plan(jobs: Sequence[Mapping[str, object]], fit_dir: Path | None,
               aliases: Sequence[str] = ()) -> dict:
    """List every planned fit, then the counts; touches nothing on disk."""
    done = 0
    for job in jobs:
        exists = fit_dir is not None and (fit_dir / f"{job['stem']}.json").exists()
        done += exists
        print(f"{'done' if exists else 'plan'} {job['stem']}  "
              f"[{job['part']} rung={job['rung']} arm={job['arm']} "
              f"seed={job.get('learner_seed', '-')}]")
    by_rung_arm = Counter((str(job["rung"]), str(job["arm"])) for job in jobs)
    by_kind = Counter(str(job["kind"]) for job in jobs)
    print("\ncounts by rung and arm:")
    for (rung, arm), count in sorted(by_rung_arm.items()):
        print(f"  {rung:12s} {arm:4s} {count}")
    print("counts by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    for alias in aliases:
        print(f"alias (no extra fit): {alias}")
    print(f"TOTAL planned fits: {len(jobs)}  (already complete: {done}, "
          f"pending: {len(jobs) - done})")
    return {"total": len(jobs), "complete": done, "by_kind": dict(by_kind)}


def drive(jobs: Sequence[dict], worker: Callable[[dict], dict], out: Path,
          workers: int, limit: int | None = None) -> dict:
    """Run every pending job in a process pool; a rerun resumes."""
    pending = [job for job in jobs
               if not (Path(job["fit_dir"]) / f"{job['stem']}.json").exists()]
    if limit is not None:
        pending = pending[:limit]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {workers} workers", flush=True)
    started = time.perf_counter()
    crashes = []
    with open(out / "progress.log", "a", encoding="utf-8") as log:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, job): job for job in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    info = future.result()
                    line = (f"[{done}/{len(pending)}] {info['stem']} "
                            f"{info['seconds']:.1f}s sel={info['selected']} "
                            f"flagged={info['flagged']}/{info['flagged_secondary']} "
                            f"err={info['error']} "
                            f"elapsed={time.perf_counter() - started:.0f}s")
                except Exception:
                    crashes.append(job["stem"])
                    line = (f"[{done}/{len(pending)}] {job['stem']} CRASH\n"
                            f"{traceback.format_exc()}")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
    return {"pending_at_start": len(pending), "crashes": crashes,
            "wall_seconds": time.perf_counter() - started}
