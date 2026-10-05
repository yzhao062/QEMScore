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

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

from qemscore.baselines.controls import shuffle_noisy_items  # noqa: E402
from qemscore.baselines.liao import (  # noqa: E402
    LiaoMitigator,
    LiaoValidationScore,
    _group_cost,
    _validation_score,
    select_one_standard_error,
)
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

# Strong learner definitions per frozen rule 2026-10-03-strong-learners.md
STRONG_LEARNERS_RULE = "docs/frozen-rules/2026-10-03-strong-learners.md"
# Where the caches of the 2026-10-03 run are recorded (inputs.caches_used).
STRENGTH_RECORD = "artifacts/descriptor-information/strength-indicator/analysis-a.json"
HGBR_GRID = (
    (200, 0.05, 15),
    (200, 0.05, 31),
    (200, 0.1, 15),
    (200, 0.1, 31),
    (500, 0.05, 15),
    (500, 0.05, 31),
    (500, 0.1, 15),
    (500, 0.1, 31),
)
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
POLY5_RUNGS = frozenset({"R0", "N1", "N2", "N3", "N4", "R3-TFI", "R3-Heis"})

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


def _cell_key(item: Mapping[str, object], part: str) -> str:
    """Per-cell key: 8 cells in Part A (family/severity/observable), 2 in Part B (severity/observable)."""
    if part == "A":
        return f"{item['family']}/{item['severity']}/{item['observable']}"
    return f"{item['severity']}/{item['observable']}"


def _poly5_coupling_columns(rung: str, family: str, names: Sequence[str]) -> list[int]:
    """Coupling columns carried by the rung's feature vector for this family."""
    if rung == "R0":
        cols = ["j", "h"] if family == "tfi" else ["jx", "jy", "jz"]
    elif rung in ("N1", "N2", "N3", "N4"):
        s = {"N1": 0.01, "N2": 0.03, "N3": 0.10, "N4": 0.30}[rung]
        cols = [f"j+{s:g}z", f"h+{s:g}z"] if family == "tfi" else [f"jx+{s:g}z", f"jy+{s:g}z", f"jz+{s:g}z"]
    elif rung == "R3-TFI":
        cols = ["j"] if family == "tfi" else ["jx", "jy", "jz"]
    elif rung == "R3-Heis":
        cols = ["j", "h"] if family == "tfi" else ["jx", "jy"]
    else:
        return []
    return [names.index(col) for col in cols]


__all__ = [
    "CANDIDATES", "FEATURES", "FROZEN_GATE_CRITERION", "GATE_CRITERIA",
    "GATE_TOLERANCE", "HGBR_GRID", "KEPT", "LEARNER_SEEDS", "LIAO_ARMS", "NOISY",
    "POLY5_RUNGS", "REPO_ROOT", "RIDGE_ALPHAS", "RULE_FILE", "RULE_SEED",
    "STRENGTH_FEATURE_NAME", "STRENGTH_RECORD", "STRONG_LEARNERS_RULE", "check_caches_recorded",
    "strong_learners_rule_sha256",
    "_circuit_ids_sha256", "_parse_seeds", "blas_fpe_probe", "check_gate_amendment",
    "drive", "filter_train_rows", "gate_decision",
    "load_cache", "print_plan", "require_gate", "rule_path", "run_affine_fit",
    "run_liao_fit", "severity_indicator", "sha256_file", "write_cache", "write_json",
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


_RULE_SHA256_CACHE: dict[str, str] = {}


def strong_learners_rule_sha256() -> str:
    """SHA-256 of the strong-learners rule file, read once per process.

    Computed from the file so that a wording edit before commit cannot
    desynchronize the recorded hash from the file.
    """
    path = REPO_ROOT / STRONG_LEARNERS_RULE
    key = str(path)
    if key not in _RULE_SHA256_CACHE:
        _RULE_SHA256_CACHE[key] = sha256_file(path)
    return _RULE_SHA256_CACHE[key]


def check_caches_recorded(cache_dir: Path, keys: Sequence[str], *,
                          record_path: Path | None = None) -> dict:
    """Compare each primary cache's SHA-256 with ``inputs.caches_used`` of the record.

    Exits nonzero, naming every mismatch, before any fit; returns the digests
    for ``run_config.json``.
    """
    record_path = Path(record_path) if record_path else REPO_ROOT / STRENGTH_RECORD
    recorded = json.loads(record_path.read_text(encoding="utf-8"))["inputs"]["caches_used"]
    digests: dict[str, dict] = {}
    problems: list[str] = []
    for key in keys:
        path = Path(cache_dir) / f"{key}.pkl"
        if key not in recorded:
            problems.append(f"{key}: no digest recorded in {record_path}")
            continue
        if not path.exists():
            problems.append(f"{key}: {path} is missing")
            continue
        actual = sha256_file(path)
        want = str(recorded[key]["sha256"])
        digests[key] = {"path": str(path), "sha256": actual, "recorded_sha256": want}
        if actual != want:
            problems.append(f"{key}: cache SHA-256 {actual} != recorded {want} ({path})")
    if problems:
        raise SystemExit("cache SHA-256 check failed before any fit:\n  "
                         + "\n  ".join(problems))
    return digests


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


def _circuit_ids_sha256(circuit_ids: Sequence[str]) -> str:
    """SHA-256 of the sorted unique training circuit identifiers."""
    return hashlib.sha256(json.dumps(sorted(set(circuit_ids))).encode("utf-8")).hexdigest()


def filter_train_rows(train: Sequence[dict], train_size: int | None) -> list[dict]:
    """Return the first train_size circuits per family in the generator's prefix order."""
    if train_size is None:
        return list(train)
    return [row for row in train if int(row["instance"]) < train_size]


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


def _fit_strong_learners(
    job: Mapping[str, object],
    data: Mapping[str, object],
    builder: Callable[[dict], Sequence[float]],
    names: Sequence[str],
    train: list[dict],
    validation: list[dict],
    candidate_predictions: dict[str, np.ndarray],
    timing: dict[str, float],
) -> tuple[
    dict[str, np.ndarray],
    list[dict],
    dict[str, list[dict]],
    dict[str, list[int | float]],
    dict[str, float] | None,
    tuple[str, ...],
]:
    part = str(job["part"])
    rung = str(job["rung"])
    fit_arm = str(job["fit_arm"])
    k = int(job["learner_seed"])
    test = list(data["test"])

    def cell_fn(item: Mapping[str, object]) -> str:
        return _cell_key(item, part)

    train_cells = {cell_fn(it) for it in train}
    val_cells = {cell_fn(it) for it in validation}
    test_cells = {cell_fn(it) for it in test}
    expected_count = 8 if part == "A" else 2
    all_cells = sorted(train_cells & val_cells & test_cells)
    if len(all_cells) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rule cells for Part {part}, but found {len(all_cells)}: {all_cells}"
        )

    has_poly5 = (part == "A" and rung in POLY5_RUNGS)
    candidate_names = (
        ("random_forest", "mlp", "hgbr", "poly5_ridge")
        if has_poly5
        else ("random_forest", "mlp", "hgbr")
    )

    X_train_full = np.asarray([builder(dict(it)) for it in train], dtype=float)
    X_val_full = np.asarray([builder(dict(it)) for it in validation], dtype=float)
    X_test_full = np.asarray([builder(dict(it)) for it in test], dtype=float)

    if fit_arm == "C":
        drop_col = names.index(NOISY)
        hgbr_X_train = np.delete(X_train_full, drop_col, axis=1)
        hgbr_X_val = np.delete(X_val_full, drop_col, axis=1)
        hgbr_X_test = np.delete(X_test_full, drop_col, axis=1)
    else:
        hgbr_X_train = X_train_full
        hgbr_X_val = X_val_full
        hgbr_X_test = X_test_full

    y_train = np.asarray([float(it["ideal_expectation"]) for it in train], dtype=float)
    y_val = np.asarray([float(it["ideal_expectation"]) for it in validation], dtype=float)

    cell_indices_train = {cell: [i for i, it in enumerate(train) if cell_fn(it) == cell] for cell in all_cells}
    cell_indices_val = {cell: [i for i, it in enumerate(validation) if cell_fn(it) == cell] for cell in all_cells}
    cell_indices_test = {cell: [i for i, it in enumerate(test) if cell_fn(it) == cell] for cell in all_cells}

    # 1. HGBR candidate
    val_pred_hgbr = np.empty(len(validation), dtype=float)
    test_pred_hgbr = np.empty(len(test), dtype=float)
    hgbr_grid_point: dict[str, list[int | float]] = {}
    with warnings.catch_warnings(record=True) as caught_hgbr:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        for cell in all_cells:
            tr_idx = cell_indices_train[cell]
            v_idx = cell_indices_val[cell]
            te_idx = cell_indices_test[cell]

            X_tr_c = hgbr_X_train[tr_idx]
            y_tr_c = y_train[tr_idx]
            X_v_c = hgbr_X_val[v_idx]
            y_v_c = y_val[v_idx]
            X_te_c = hgbr_X_test[te_idx]

            best_mae = None
            best_point = None
            best_model = None
            for point in HGBR_GRID:
                mi, lr, mln = point
                model = HistGradientBoostingRegressor(
                    max_iter=mi,
                    learning_rate=lr,
                    max_leaf_nodes=mln,
                    random_state=k,
                    early_stopping=False,
                )
                model.fit(X_tr_c, y_tr_c)
                pred_v = model.predict(X_v_c)
                mae = float(np.mean(np.abs(pred_v - y_v_c)))
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_point = point
                    best_model = model
            hgbr_grid_point[cell] = list(best_point)
            val_pred_hgbr[v_idx] = best_model.predict(X_v_c)
            test_pred_hgbr[te_idx] = best_model.predict(X_te_c)
        timing["candidate_hgbr_seconds"] = time.perf_counter() - t0
    hgbr_warnings = _warning_rows(caught_hgbr, "candidate_hgbr")
    candidate_predictions["validation__hgbr"] = val_pred_hgbr
    candidate_predictions["test__hgbr"] = test_pred_hgbr

    # 2. poly5_ridge candidate (if applicable)
    poly5_ridge_alpha: dict[str, float] | None = None
    poly5_warnings: list[dict] = []
    if has_poly5:
        poly5_ridge_alpha = {}
        val_pred_poly = np.empty(len(validation), dtype=float)
        test_pred_poly = np.empty(len(test), dtype=float)
        with warnings.catch_warnings(record=True) as caught_poly:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            for cell in all_cells:
                tr_idx = cell_indices_train[cell]
                v_idx = cell_indices_val[cell]
                te_idx = cell_indices_test[cell]

                family = train[tr_idx[0]]["family"]
                col_indices = _poly5_coupling_columns(rung, family, names)

                C_tr = X_train_full[tr_idx][:, col_indices]
                C_v = X_val_full[v_idx][:, col_indices]
                C_te = X_test_full[te_idx][:, col_indices]

                mean_c = np.mean(C_tr, axis=0)
                std_c = np.std(C_tr, axis=0, ddof=0)
                std_c = np.where(std_c == 0.0, 1.0, std_c)
                Z_tr = (C_tr - mean_c) / std_c
                Z_v = (C_v - mean_c) / std_c
                Z_te = (C_te - mean_c) / std_c

                poly = PolynomialFeatures(5, include_bias=False)
                X_poly_tr = poly.fit_transform(Z_tr)
                X_poly_v = poly.transform(Z_v)
                X_poly_te = poly.transform(Z_te)

                if fit_arm in ("F", "P"):
                    r_idx = names.index(NOISY)
                    r_tr = X_train_full[tr_idx, r_idx]
                    r_v = X_val_full[v_idx, r_idx]
                    r_te = X_test_full[te_idx, r_idx]

                    mean_r = float(np.mean(r_tr))
                    std_r = float(np.std(r_tr, ddof=0))
                    if std_r == 0.0:
                        std_r = 1.0
                    r_tr_std = ((r_tr - mean_r) / std_r).reshape(-1, 1)
                    r_v_std = ((r_v - mean_r) / std_r).reshape(-1, 1)
                    r_te_std = ((r_te - mean_r) / std_r).reshape(-1, 1)

                    X_design_tr = np.hstack([X_poly_tr, r_tr_std, r_tr_std * Z_tr])
                    X_design_v = np.hstack([X_poly_v, r_v_std, r_v_std * Z_v])
                    X_design_te = np.hstack([X_poly_te, r_te_std, r_te_std * Z_te])
                else:
                    X_design_tr = X_poly_tr
                    X_design_v = X_poly_v
                    X_design_te = X_poly_te

                y_tr_c = y_train[tr_idx]
                y_v_c = y_val[v_idx]

                best_mae = None
                best_alpha = None
                best_model = None
                for alpha in RIDGE_ALPHAS:
                    ridge = Ridge(alpha=alpha, fit_intercept=True)
                    ridge.fit(X_design_tr, y_tr_c)
                    pred_v = ridge.predict(X_design_v)
                    mae = float(np.mean(np.abs(pred_v - y_v_c)))
                    if best_mae is None or mae < best_mae:
                        best_mae = mae
                        best_alpha = alpha
                        best_model = ridge
                poly5_ridge_alpha[cell] = float(best_alpha)
                val_pred_poly[v_idx] = best_model.predict(X_design_v)
                test_pred_poly[te_idx] = best_model.predict(X_design_te)
            timing["candidate_poly5_ridge_seconds"] = time.perf_counter() - t0
        poly5_warnings = _warning_rows(caught_poly, "candidate_poly5_ridge")
        candidate_predictions["validation__poly5_ridge"] = val_pred_poly
        candidate_predictions["test__poly5_ridge"] = test_pred_poly

    per_candidate_warnings = {
        "hgbr": hgbr_warnings,
        "poly5_ridge": poly5_warnings,
    }
    extra_warnings = hgbr_warnings + poly5_warnings
    return (
        candidate_predictions,
        extra_warnings,
        per_candidate_warnings,
        hgbr_grid_point,
        poly5_ridge_alpha,
        candidate_names,
    )


def _score_rows(scores) -> list[dict]:
    return [
        {"name": s.name, "validation_mae": s.validation_mae,
         "standard_error": s.standard_error,
         "total_excess_absolute_loss": s.total_excess_absolute_loss,
         "circuit_evaluations": s.circuit_evaluations,
         "simplicity_rank": s.simplicity_rank}
        for s in scores]


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
    train_size = job.get("train_size")
    train = filter_train_rows(data["train"], train_size)
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
            neural_es = bool(job.get("neural_es"))
            liao_kwargs = {
                "random_state": k,
                "drop_features": drop,
                "feature_builder": builder,
                "feature_names": tuple(names),
            }
            if neural_es:
                liao_kwargs.update({
                    "stopping_rule": "validation_patience",
                    "epochs": 2000,
                    "patience": 50,
                    "min_delta": 0.0,
                    "lr_plateau_factor": 0.5,
                    "lr_plateau_patience": 10,
                    "lr_min": 1e-5,
                })
            model = LiaoMitigator(**liao_kwargs).fit(train, validation)
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
    cand_names: Sequence[str] = CANDIDATES
    hgbr_grid_point: dict[str, list[int | float]] | None = None
    poly5_ridge_alpha: dict[str, float] | None = None
    # Whenever a model was fitted its own selection is recorded, even if the
    # prediction step then failed (as the option-off code does).
    selected_model_name: str | None = (
        None if model is None else model.selected_model_name_)
    one_standard_error_threshold: float | None = (
        None if model is None else model.one_standard_error_threshold_)
    eligible_models: list[str] | None = (
        None if model is None else list(model.eligible_models_))
    val_scores = None if model is None else model.validation_scores_
    warnings_agg = fit_warnings + predict_warnings
    option_off: dict | None = None
    dataset_hash = str(data["dataset_hash"])

    def _snapshot() -> dict:
        """The values the option-off code writes, as of this moment."""
        return {
            "selected_model": selected_model_name,
            "eligible_models": None if eligible_models is None else list(eligible_models),
            "one_standard_error_threshold": one_standard_error_threshold,
            "validation_scores": None if val_scores is None else _score_rows(val_scores),
            "flagged": flagged,
            "flagged_secondary": flagged_secondary,
            "finite": dict(finite),
            "n_runtime_warnings": int(sum(row["count"] for row in runtime_warnings)),
            "n_runtime_warnings_not_matmul_fpe": int(
                sum(row["count"] for row in other_runtime)),
            "warnings_aggregated": [dict(row) for row in warnings_agg],
            "test_family_mae": (
                _family_mae_or_none(data["test"], predictions["test"], dataset_hash)
                if "test" in predictions else None),
            "validation_family_mae": (
                _family_mae_or_none(data["validation"], predictions["validation"],
                                    dataset_hash)
                if "validation" in predictions else None),
            "selected_equals_candidate_predictions": selected_matches_candidate,
        }

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

        # LiaoMitigator's own selection: what the option-off code records.
        liao_selected = selected_model_name
        selected_matches_candidate = bool(all(
            np.array_equal(predictions[role],
                           candidate_predictions[f"{role}__{liao_selected}"])
            for role in ("validation", "test")))

        if job.get("strong_learners"):
            # The option-off state, recorded before the strong selection changes
            # anything: tools/strong_learner_derive.py rebuilds the baseline from it.
            option_off = _snapshot()
            # A failure here propagates: ``drive`` records the job as a crash.
            (
                candidate_predictions,
                extra_warnings,
                per_cand_warnings,
                hgbr_grid_point,
                poly5_ridge_alpha,
                cand_names,
            ) = _fit_strong_learners(
                job, data, builder, names, train, validation,
                candidate_predictions, timing
            )
            candidate_warnings.extend(extra_warnings)
            shared_cost = _group_cost([*train, *validation])
            ranks = {"random_forest": 0, "mlp": 1, "hgbr": 2, "poly5_ridge": 3}
            val_scores = [
                _validation_score(
                    name, validation, candidate_predictions[f"validation__{name}"],
                    circuit_evaluations=shared_cost, simplicity_rank=ranks[name])
                for name in cand_names
            ]
            (
                selected_model_name,
                one_standard_error_threshold,
                eligible_tuple,
            ) = select_one_standard_error(val_scores)
            eligible_models = list(eligible_tuple)

            for role in ("validation", "test"):
                predictions[role] = candidate_predictions[f"{role}__{selected_model_name}"]

            if selected_model_name != liao_selected:
                # The selected predictions differ from LiaoMitigator's: recompute
                # the finite checks and both flags. A selected new candidate's
                # fitting and prediction warnings also enter both flags.
                sel_warnings = per_cand_warnings.get(selected_model_name, [])
                runtime_warnings = runtime_warnings + [
                    row for row in sel_warnings if row["is_runtime_warning"]]
                other_runtime = [row for row in runtime_warnings
                                 if row["message"] not in SPURIOUS_MATMUL_MESSAGES]
                finite = dict(finite)
                finite["selected_predictions"] = bool(
                    len(predictions) == 2
                    and all(np.all(np.isfinite(value)) for value in predictions.values()))
                flagged = bool(error is not None or runtime_warnings
                               or not all(finite.values()))
                flagged_secondary = bool(error is not None or other_runtime
                                         or not all(finite.values()))
                warnings_agg = fit_warnings + predict_warnings + sel_warnings

            # The selected candidate's stored predictions are the fit's predictions
            # (the same arrays; non-finite values compare equal here and are caught
            # by the finite checks and the flags).
            selected_matches_candidate = bool(all(
                np.array_equal(predictions[role],
                               candidate_predictions[f"{role}__{selected_model_name}"],
                               equal_nan=True)
                for role in ("validation", "test")))

    if job.get("strong_learners") and option_off is None:
        option_off = _snapshot()
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
        **({"train_size": int(train_size),
            "n_train_rows": len(train),
            "training_rows": len(train),
            "training_circuit_ids_sha256": _circuit_ids_sha256([str(r["circuit_id"]) for r in train]),
            "training_circuit_identifiers_sha256": _circuit_ids_sha256([str(r["circuit_id"]) for r in train])}
           if train_size is not None else {}),
        **({"strength_indicator": True} if job.get("strength_indicator") else {}),
        **({"strong_learners": True} if job.get("strong_learners") else {}),
        **({"candidates": list(cand_names)} if job.get("strong_learners") else {}),
        **({"hgbr_grid_point": hgbr_grid_point} if job.get("strong_learners") else {}),
        **({"poly5_ridge_alpha": poly5_ridge_alpha} if job.get("strong_learners") else {}),
        **({"follow_up_rule": {"path": STRONG_LEARNERS_RULE,
                               "sha256": strong_learners_rule_sha256()}}
           if job.get("strong_learners") else {}),
        **({"option_off": option_off} if option_off is not None else {}),
        **({"neural_es": True} if job.get("neural_es") else {}),
        **({
            "schedule_parameters": {
                "stopping_rule": "validation_patience",
                "epochs": 2000,
                "patience": 50,
                "min_delta": 0.0,
                "lr_plateau_factor": 0.5,
                "lr_plateau_patience": 10,
                "lr_min": 1e-5,
            },
            "epochs_run": None if model is None or error is not None else int(model.candidate_models_["mlp"].n_iter_),
            "best_epoch": None if model is None or error is not None else int(model.candidate_models_["mlp"].best_epoch_),
            "final_learning_rate": None if model is None or error is not None else float(model.candidate_models_["mlp"].final_learning_rate_),
        } if job.get("neural_es") else {}),
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
        "warnings_aggregated": warnings_agg,
        "candidate_prediction_warnings": candidate_warnings,
        "selected_model": selected_model_name,
        "selected_equals_candidate_predictions": selected_matches_candidate,
        "eligible_models": eligible_models,
        "one_standard_error_threshold": one_standard_error_threshold,
        "validation_scores": None if val_scores is None else _score_rows(val_scores),
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
            for name in cand_names if f"test__{name}" in candidate_predictions},
        "candidate_validation_family_mae": {
            name: _family_mae_or_none(data["validation"],
                                      candidate_predictions[f"validation__{name}"],
                                      dataset_hash)
            for name in cand_names if f"validation__{name}" in candidate_predictions},
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
    train_size = job.get("train_size")
    train = filter_train_rows(data["train"], train_size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            method = _RidgeRunnerMethod(
                lambda: _BuilderFeatureOnlyRidge(builder, names),
                source_measurements=False, test_measurements=False,
                legacy_reports_alpha=False)
            method.fit(train, manifest={}, split_v2=True)
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
        **({"train_size": int(train_size),
            "n_train_rows": len(train),
            "training_rows": len(train),
            "training_circuit_ids_sha256": _circuit_ids_sha256([str(r["circuit_id"]) for r in train]),
            "training_circuit_identifiers_sha256": _circuit_ids_sha256([str(r["circuit_id"]) for r in train])}
           if train_size is not None else {}),
        **({"strength_indicator": True} if job.get("strength_indicator") else {}),
        **({"strong_learners": True} if job.get("strong_learners") else {}),
        **({"neural_es": True} if job.get("neural_es") else {}),
        **({"follow_up_rule": {"path": STRONG_LEARNERS_RULE,
                               "sha256": strong_learners_rule_sha256()}}
           if job.get("strong_learners") else {}),
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
        flags = []
        if job.get("strength_indicator"):
            flags.append("strength_indicator")
        if job.get("strong_learners"):
            flags.append("strong_learners")
        if job.get("neural_es"):
            flags.append("neural_es")
        flags_str = f" flags={','.join(flags)}" if flags else ""
        print(f"{'done' if exists else 'plan'} {job['stem']}  "
              f"[{job['part']} rung={job['rung']} arm={job['arm']} "
              f"seed={job.get('learner_seed', '-')}{flags_str}]")
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
