#!/usr/bin/env python3
"""Analysis module for ML-QEM replication experiment.

Reads the fit artifacts (.npz and .json) for learner seeds across settings and models,
computes circuit-level and arm-level MAEs, computes quantities (C, F, P, R, Rcal, D = C - F,
D/C, F - R, F - Rcal, P - F), applies the two-stage percentile bootstrap imported from
tools/descriptor_ladder_analysis.py, classifies label outcomes, and outputs a
comprehensive JSON report.

Runs under QEMScore py312 environment (pure NumPy; no torch or qiskit).
"""

import os
import sys
import json
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Import canonical bootstrap_draws from descriptor_ladder_analysis
from tools.descriptor_ladder_analysis import bootstrap_draws

RULE_BOOTSTRAP_SEED = 20261002
PERCENTILES = (2.5, 97.5)

LABEL_ADDS = "measurement_adds"
LABEL_HURTS = "measurement_hurts"
LABEL_NOT_DISTINGUISHED = "not_distinguished"


def compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_draws(
    errors: np.ndarray,
    seed_counts: np.ndarray,
    circuit_counts: np.ndarray
) -> np.ndarray:
    """Computes draw values across draws from errors (n_seeds x n_circuits)."""
    n_seeds = errors.shape[0]
    n_circuits = errors.shape[1]
    # Platform's Accelerate BLAS raises spurious floating-point flags in matmul (Appendix M.4)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        draw_seed_means = (circuit_counts @ errors.T) / n_circuits  # shape (draws, n_seeds)
        draw_values = np.sum(seed_counts * draw_seed_means, axis=1) / n_seeds  # shape (draws,)
    return draw_values


def format_interval(
    draw_values: np.ndarray,
    point: float,
    percentiles: Tuple[float, float] = PERCENTILES
) -> Dict[str, Any]:
    """Formats point estimate and bootstrap percentile interval."""
    lower, upper = np.percentile(draw_values, percentiles)
    return {
        "point": float(point),
        "interval": {
            "lower": float(lower),
            "upper": float(upper)
        },
        "excludes_zero_above": bool(lower > 0.0),
        "excludes_zero_below": bool(upper < 0.0),
        "draw_sd": float(np.std(draw_values, ddof=1)) if len(draw_values) > 1 else 0.0
    }


def classify_d(interval_dict: Dict[str, Any], mean_c: float) -> Dict[str, Any]:
    """Classifies D interval into measurement_adds, measurement_hurts, or not_distinguished."""
    lower = interval_dict["interval"]["lower"]
    upper = interval_dict["interval"]["upper"]
    out = {
        "status": "estimated",
        "interval_D": {"lower": lower, "upper": upper},
        "mean_D": interval_dict["point"],
        "mean_C": float(mean_c)
    }
    if lower > 0.0:
        out["label"] = LABEL_ADDS
    elif upper < 0.0:
        out["label"] = LABEL_HURTS
    else:
        out["label"] = LABEL_NOT_DISTINGUISHED
        out["largest_reduction_not_excluded"] = float(upper / mean_c) if mean_c != 0 else None
        out["largest_reduction_definition"] = "upper(D) / mean(C)"
    return out


def find_fit_file(fits_dir: str, setting: str, model: str, arm: str, seed: int) -> str:
    """Finds fit NPZ file checking setting aliases if needed."""
    aliases = [setting]
    if setting == "no_readout":
        aliases.append("ising_init_from_qasm_no_readout")
    elif setting == "ising_init_from_qasm_no_readout":
        aliases.append("no_readout")
    elif setting == "readout":
        aliases.append("ising_init_from_qasm")
    elif setting == "ising_init_from_qasm":
        aliases.append("readout")
    elif setting == "coherent":
        aliases.append("ising_init_from_qasm_coherent")
    elif setting == "ising_init_from_qasm_coherent":
        aliases.append("coherent")

    for s in aliases:
        cand = os.path.join(fits_dir, f"{s}_{model}_{arm}_seed{seed}.npz")
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"Could not find fit file for setting={setting}, model={model}, arm={arm}, seed={seed} in {fits_dir}"
    )


# Published errors over Trotter steps 0 to 14 on the no-readout test circuits (val_extra),
# computed from docs/paper_figures/no_readout_over_depths.pk; frozen in the ML-QEM rule.
PUBLISHED_ERRORS = {("no_readout", "rf"): 0.015510, ("no_readout", "mlp"): 0.022274}
REPRODUCTION_BOUNDS = (0.9, 1.1)
SETTING_DIRS = {
    "no_readout": "ising_init_from_qasm_no_readout",
    "readout": "ising_init_from_qasm",
    "coherent": "ising_init_from_qasm_coherent",
}
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_data_sha256.json")


TEST_SPLITS = {
    "no_readout": "val_extra",
    "readout": "val_Zonly",
    "coherent": "val",
}


def canonical_setting(setting: str) -> str:
    """Maps a raw ML-QEM directory name to its setting name; other names pass unchanged."""
    for name, dir_name in SETTING_DIRS.items():
        if setting == dir_name:
            return name
    return setting


def find_exact_file(exact_dir: str, setting: str) -> str:
    """Locates the exact npz file for a setting and its canonical test split."""
    can_setting = canonical_setting(setting)
    split = TEST_SPLITS.get(can_setting)
    if not split:
        raise ValueError(f"Unknown test split for setting {setting!r}")
    candidates = [
        os.path.join(exact_dir, f"exact-{can_setting}-{split}.npz"),
        os.path.join(exact_dir, f"exact-{SETTING_DIRS.get(can_setting, can_setting)}-{split}.npz"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find exact npz file for setting={setting} split={split} in {exact_dir}. "
        f"Checked: {candidates}"
    )


def check_gate_approval(exact_dir: str, expected_npz_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Requires that gate.json exists in exact_dir, passed, and matches expected NPZ hashes."""
    gate_path = os.path.join(exact_dir, "gate.json")
    if not os.path.isfile(gate_path):
        raise FileNotFoundError(
            f"Gate file not found: {gate_path}. Exact label operations require gate.json to exist and pass."
        )
    with open(gate_path, "r") as handle:
        gate_data = json.load(handle)
    is_passed = gate_data.get("gate_passed", gate_data.get("overall_pass", False))
    if not is_passed:
        raise ValueError(
            f"Exact label gate check failed in {gate_path}. Operations on exact targets/descriptors are blocked."
        )

    if expected_npz_paths:
        splits = gate_data.get("splits", {})
        hash_to_split = {s_info.get("npz_sha256"): k for k, s_info in splits.items() if "npz_sha256" in s_info}
        for npz_path in expected_npz_paths:
            h = compute_file_sha256(npz_path)
            fn = os.path.basename(npz_path)
            found = False
            for s_key, s_info in splits.items():
                if s_info.get("npz_file") == fn:
                    found = True
                    if s_info.get("npz_sha256") != h:
                        raise ValueError(
                            f"{npz_path} SHA-256 {h} does not match gate.json recorded SHA-256 {s_info.get('npz_sha256')}"
                        )
                    break
            if not found and h not in hash_to_split:
                raise ValueError(f"{npz_path} is not recorded in {gate_path}")

    return gate_data


def load_fit(fits_dir: str, setting: str, model: str, arm: str, seed: int,
             expected_descriptors: str = "encoding") -> Dict[str, Any]:
    """Loads one fit's NPZ arrays and its JSON record and checks them.

    The record must name the same job. Targets must be a nonempty (n, 4) array, predictions must
    have the same shape, identifiers and steps must have n entries, and every number must be
    finite. A failed check raises before any error is computed.
    """
    npz_path = find_fit_file(fits_dir, setting, model, arm, seed)
    json_path = npz_path[:-len(".npz")] + ".json"
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"{json_path}: the fit record is missing")
    with open(json_path, "r") as handle:
        meta = json.load(handle)
    want = {"model": model, "arm": arm, "learner_seed": seed}
    for key, value in want.items():
        if meta.get(key) != value:
            raise ValueError(f"{json_path}: {key} is {meta.get(key)!r}, expected {value!r}")
    if meta.get("setting") not in (setting, SETTING_DIRS.get(setting)):
        raise ValueError(f"{json_path}: setting is {meta.get('setting')!r}, expected {setting!r}")
    if expected_descriptors == "exact":
        if meta.get("descriptors") != "exact":
            raise ValueError(f"{json_path}: descriptors is {meta.get('descriptors')!r}, expected 'exact'")
    else:
        if meta.get("descriptors") == "exact":
            raise ValueError(f"{json_path}: fit has descriptors='exact', expected 'encoding'")
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in ("predictions", "targets", "test_circuit_ids", "test_steps")}
        for opt in ("test_circuit_files", "test_circuit_indices"):
            if opt in data:
                arrays[opt] = data[opt]
    targets = arrays["targets"]
    if targets.ndim != 2 or targets.shape[1] != 4 or len(targets) == 0:
        raise ValueError(f"{npz_path}: expected nonempty targets with four observables")
    if arrays["predictions"].shape != targets.shape:
        raise ValueError(f"{npz_path}: prediction shape {arrays['predictions'].shape} differs from "
                         f"targets {targets.shape}")
    for key in ("test_circuit_ids", "test_steps"):
        if arrays[key].shape != (len(targets),):
            raise ValueError(f"{npz_path}: {key} shape differs from target rows")
    for key in ("predictions", "targets", "test_steps"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"{npz_path}: {key} contains nonfinite values")
    return {"npz": npz_path, "json": json_path, "meta": meta, **arrays}


def check_manifest(setting: str, data_shas: Dict[str, str]) -> None:
    """Requires the fits' data-file mapping to equal the manifest's entries for the setting."""
    with open(MANIFEST, "r") as handle:
        manifest = json.load(handle)
    prefix = f"docs/tutorials/data/{SETTING_DIRS[canonical_setting(setting)]}/"
    expected = {key[len(prefix):]: value for key, value in manifest.items() if key.startswith(prefix)}
    data_shas = data_shas or {}
    missing = sorted(expected.keys() - data_shas.keys())
    unexpected = sorted(data_shas.keys() - expected.keys())
    changed = sorted(k for k in expected.keys() & data_shas.keys() if data_shas[k] != expected[k])
    if not expected or missing or unexpected or changed:
        raise ValueError(f"{setting}: manifest mismatch: missing={missing}, "
                         f"unexpected={unexpected}, changed={changed}")


def fit_plan(model: str, seeds: List[int], analysis_arms: Optional[List[str]] = None) -> List[Tuple[str, str, List[int], bool]]:
    """(arm, fitted model, seeds read, repeated across seed slots) for every arm of a cell."""
    is_ols = (model.lower() == "ols")
    full_plan = [("R", model, [1], True), ("Rcal", "ols", [1], True)]
    for arm in ("F", "C"):
        plan_seeds = [1] if is_ols else list(seeds)
        full_plan.append((arm, model, plan_seeds, is_ols))
    full_plan.append(("P", model, list(seeds), False))
    if analysis_arms is None:
        return full_plan
    valid_arms = {"R", "Rcal", "F", "C", "P"}
    unknown = set(analysis_arms) - valid_arms
    if unknown:
        raise ValueError(f"Unknown arms in --analysis-arms: {sorted(unknown)}")
    arms_set = set(analysis_arms)
    return [entry for entry in full_plan if entry[0] in arms_set]


def load_model_fits(
    fits_dir: str,
    setting: str,
    model: str,
    seeds: List[int],
    rule_sha256: Optional[str] = None,
    manifest_check: bool = True,
    targets_mode: str = "archived",
    exact_dir: Optional[str] = None,
    descriptors: str = "encoding",
    analysis_arms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Loads and binds every fit of one setting and model; computes no error.

    Every arm is required. R is the model's own seed-1 record of the raw noisy values. Rcal is
    the setting's deterministic OLS fit on the noisy columns (`<setting>_ols_Rcal_seed1`), shared
    by every model. OLS arms C and F are deterministic: seed 1 is repeated across the learner-seed
    slots. Every fit must carry the same circuit identifiers, targets, steps, data-file SHA-256
    values (equal to the manifest's when `manifest_check`), rule SHA-256 (equal to `rule_sha256`
    when given), and `test_is_validation` flag.
    """
    exact_targets = None
    archived_targets = None
    ref_error = None
    input_files: List[str] = []

    if targets_mode == "exact":
        if not exact_dir or not os.path.isdir(exact_dir):
            raise ValueError(f"With --targets exact, --exact-dir must be a valid directory, got {exact_dir!r}")
        exact_path = find_exact_file(exact_dir, setting)
        input_files.append(exact_path)
        with np.load(exact_path) as ex_data:
            exact_targets = ex_data["exact"]
            archived_targets = ex_data["archived"]
            exact_steps = ex_data["steps"]
            exact_step_file = ex_data["step_file"]
            exact_entry_index = ex_data["entry_index"]
        ref_error = float(np.mean(np.mean(np.abs(archived_targets - exact_targets), axis=1)))

    reference = None
    roster_train_targets = None
    arms: Dict[str, Tuple[List[np.ndarray], bool]] = {}
    for arm, fit_model, fit_seeds, repeat in fit_plan(model, seeds, analysis_arms=analysis_arms):
        rows = []
        for s in fit_seeds:
            fit = load_fit(fits_dir, setting, fit_model, arm, s, expected_descriptors=descriptors)
            input_files.extend([fit["npz"], fit["json"]])
            meta = fit["meta"]
            fit_tt = meta.get("train_targets", "archived")
            if roster_train_targets is None:
                roster_train_targets = fit_tt
            elif fit_tt != roster_train_targets:
                raise ValueError(
                    f"{fit['json']}: mixed train_targets in roster: {fit_tt!r} != {roster_train_targets!r}"
                )
            binding = {
                "frozen_rule_sha256": meta.get("frozen_rule_sha256"),
                "data_files_sha256": meta.get("data_files_sha256"),
                "test_is_validation": meta.get("test_is_validation"),
            }
            if reference is None:
                if rule_sha256 is not None and binding["frozen_rule_sha256"] != rule_sha256:
                    raise ValueError(f"{fit['json']}: rule SHA-256 differs from the frozen rule")
                if not isinstance(binding["test_is_validation"], bool):
                    raise ValueError(f"{fit['json']}: test_is_validation is missing")
                if manifest_check:
                    check_manifest(setting, binding["data_files_sha256"])
                reference = {**binding, "test_circuit_ids": fit["test_circuit_ids"],
                             "targets": fit["targets"], "test_steps": fit["test_steps"],
                             "first": fit["json"]}
            for key in ("frozen_rule_sha256", "data_files_sha256", "test_is_validation"):
                if binding[key] != reference[key]:
                    raise ValueError(f"{fit['json']}: {key} differs from {reference['first']}")
            for key in ("test_circuit_ids", "targets", "test_steps"):
                if not np.array_equal(fit[key], reference[key]):
                    raise ValueError(f"{fit['npz']}: {key} differ from {reference['first']}")
            if targets_mode == "exact":
                fit_targets = fit["targets"]
                if not np.array_equal(fit_targets, archived_targets.astype(fit_targets.dtype)):
                    raise ValueError(
                        f"{fit['npz']}: fit targets do not match archived array of "
                        f"{os.path.basename(exact_path)} exactly and in order"
                    )
                if not np.array_equal(fit["test_steps"], exact_steps):
                    raise ValueError(
                        f"{fit['npz']}: test steps do not match {os.path.basename(exact_path)} in order"
                    )
                if "test_circuit_files" in fit and not np.array_equal(fit["test_circuit_files"], exact_step_file):
                    raise ValueError(
                        f"{fit['npz']}: test circuit files do not match {os.path.basename(exact_path)} in order"
                    )
                if "test_circuit_indices" in fit and not np.array_equal(fit["test_circuit_indices"], exact_entry_index):
                    raise ValueError(
                        f"{fit['npz']}: test circuit indices do not match {os.path.basename(exact_path)} in order"
                    )
                exact_cids = np.array([f"{f}:{idx}" for f, idx in zip(exact_step_file, exact_entry_index)])
                if not np.array_equal(fit["test_circuit_ids"], exact_cids):
                    raise ValueError(
                        f"{fit['npz']}: test circuit identifiers do not match {os.path.basename(exact_path)} in order"
                    )
            rows.append(fit["predictions"])
        arms[arm] = (rows, repeat)

    scoring_targets = exact_targets if targets_mode == "exact" else reference["targets"]

    ids_digest = hashlib.sha256("\n".join(str(x) for x in reference["test_circuit_ids"]).encode()).hexdigest()
    notes = {
        "binding": {
            "frozen_rule_sha256": reference["frozen_rule_sha256"],
            "data_files_sha256_digest": hashlib.sha256(
                json.dumps(reference["data_files_sha256"], sort_keys=True).encode()).hexdigest(),
            "test_circuit_ids_sha256": ids_digest,
            "n_test_circuits": int(len(reference["test_circuit_ids"])),
            "manifest_checked": bool(manifest_check),
        },
        "test_is_validation": bool(reference["test_is_validation"]),
    }
    if "Rcal" in arms:
        notes["rcal_source"] = f"{setting}_ols_Rcal_seed1"
    if analysis_arms is not None or targets_mode == "exact" or descriptors == "exact" or roster_train_targets == "exact":
        notes["train_targets"] = roster_train_targets
    if targets_mode == "exact":
        notes["targets"] = "exact"
        notes["exact_npz"] = os.path.basename(exact_path)
        notes["exact_npz_sha256"] = compute_file_sha256(exact_path)
        notes["reference_error"] = ref_error
    if descriptors == "exact":
        notes["descriptors"] = "exact"
    if reference["test_is_validation"]:
        notes["test_is_validation_note"] = (
            "The test split is the validation split"
            + ("; the MLP's scheduler read these labels." if model.lower() == "mlp" else ".")
        )
    if model.lower() == "ols" and ("C" in arms or "F" in arms or "Rcal" in arms):
        notes["ols_deterministic_seeds_repeated"] = True
        notes["note"] = (
            "For OLS, arms C, F, and Rcal are deterministic; seed 1 predictions repeated "
            "across learner seeds so the two-stage bootstrap applies identically."
        )
    return {"arms": arms, "targets": reference["targets"], "scoring_targets": scoring_targets,
            "test_steps": reference["test_steps"], "input_files": input_files, "notes": notes,
            "n_seeds": len(seeds), "reference_error": ref_error, "train_targets": roster_train_targets}


def errors_from_fits(loaded: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-circuit errors (mean over the four observables) as (n_seeds x n_circuits) matrices."""
    out = {}
    scoring_targets = loaded.get("scoring_targets", loaded["targets"])
    for arm, (rows, repeat) in loaded["arms"].items():
        errs = [np.mean(np.abs(pred - scoring_targets), axis=1) for pred in rows]
        out[arm] = np.tile(errs[0], (loaded["n_seeds"], 1)) if repeat else np.array(errs)
    return out


def load_model_errors(
    fits_dir: str,
    setting: str,
    model: str,
    seeds: List[int],
    rule_sha256: Optional[str] = None,
    manifest_check: bool = True,
    targets_mode: str = "archived",
    exact_dir: Optional[str] = None,
    descriptors: str = "encoding",
    analysis_arms: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], Dict[str, Any]]:
    """`load_model_fits` then `errors_from_fits`, for one setting and model."""
    loaded = load_model_fits(
        fits_dir=fits_dir,
        setting=setting,
        model=model,
        seeds=seeds,
        rule_sha256=rule_sha256,
        manifest_check=manifest_check,
        targets_mode=targets_mode,
        exact_dir=exact_dir,
        descriptors=descriptors,
        analysis_arms=analysis_arms,
    )
    return errors_from_fits(loaded), loaded["test_steps"], loaded["input_files"], loaded["notes"]



def reproduction_record(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The rule's reproduction check: mean over learner seeds of F's error over the published one.

    The random forest in the no-readout setting passes if the ratio lies in [0.9, 1.1]
    (inclusive). The MLP's ratio is reported without a criterion.
    """
    out: Dict[str, Any] = {}
    for (setting, model), published in sorted(PUBLISHED_ERRORS.items()):
        cell = results.get(setting, {}).get(model)
        key = f"{setting}_{model}"
        if cell is None:
            out[key] = {"status": "not_computed"}
            continue
        per_seed = cell["F_per_seed"]
        refit = float(np.mean(per_seed))
        ratio = refit / published
        record = {
            "published_error": published,
            "refit_mean_error": refit,
            "ratio": ratio,
            "per_seed": {"min": float(np.min(per_seed)), "max": float(np.max(per_seed)),
                         "sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0},
        }
        if model == "rf":
            low, high = REPRODUCTION_BOUNDS
            record["bounds"] = [low, high]
            record["passes"] = bool(low <= ratio <= high)
            record["reading"] = ("reading 1: the cells are reported as the control applied to the "
                                 "published pipeline" if record["passes"] else
                                 "reading 2: the refit does not reproduce the published error; "
                                 "no conclusion about the published pipeline itself")
        else:
            record["criterion"] = None
        out[key] = record
    return out


def analyze_setting_model(
    arm_errors: Dict[str, np.ndarray],
    test_steps: np.ndarray,
    draws: int = 10000,
    seed: int = RULE_BOOTSTRAP_SEED,
    meta_notes: Optional[Dict[str, Any]] = None,
    report_c_minus_r: bool = False,
    reference_error: Optional[float] = None,
) -> Dict[str, Any]:
    """Performs point estimation, two-stage percentile bootstrap, and classification."""
    has_F = "F" in arm_errors
    has_C = "C" in arm_errors
    has_P = "P" in arm_errors
    has_R = "R" in arm_errors
    has_rcal = "Rcal" in arm_errors

    first_arm = next(iter(arm_errors.values()))
    n_seeds = first_arm.shape[0]
    n_circuits = first_arm.shape[1]

    err_F = arm_errors["F"] if has_F else None
    err_C = arm_errors["C"] if has_C else None
    err_P = arm_errors["P"] if has_P else None
    err_R = arm_errors["R"] if has_R else None
    err_Rcal = arm_errors.get("Rcal") if has_rcal else None

    # Point estimates
    pt_C = float(np.mean(err_C)) if has_C else None
    pt_F = float(np.mean(err_F)) if has_F else None
    pt_P = float(np.mean(err_P)) if has_P else None
    pt_R = float(np.mean(err_R)) if has_R else None
    pt_D = (pt_C - pt_F) if (has_C and has_F) else None
    pt_D_over_C = (pt_D / pt_C if pt_C != 0 else 0.0) if (has_C and has_F) else None
    pt_F_minus_R = (pt_F - pt_R) if (has_F and has_R) else None
    pt_P_minus_F = (pt_P - pt_F) if (has_P and has_F) else None

    quantities = {}

    # 1. C
    if has_C:
        sc_C, cc_C = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_C = evaluate_draws(err_C, sc_C, cc_C)
        quantities["C"] = format_interval(d_C, pt_C)

    # 2. F
    if has_F:
        sc_F, cc_F = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_F = evaluate_draws(err_F, sc_F, cc_F)
        quantities["F"] = format_interval(d_F, pt_F)

    # 3. P
    if has_P:
        sc_P, cc_P = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_P = evaluate_draws(err_P, sc_P, cc_P)
        quantities["P"] = format_interval(d_P, pt_P)

    # 4. R
    if has_R:
        sc_R, cc_R = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_R = evaluate_draws(err_R, sc_R, cc_R)
        quantities["R"] = format_interval(d_R, pt_R)

    # 5. Rcal (if present)
    if has_rcal:
        pt_Rcal = float(np.mean(err_Rcal))
        sc_Rc, cc_Rc = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_Rcal = evaluate_draws(err_Rcal, sc_Rc, cc_Rc)
        quantities["Rcal"] = format_interval(d_Rcal, pt_Rcal)

    # 6. D = C - F (paired inside draw)
    if has_C and has_F:
        sc_D, cc_D = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_D = evaluate_draws(err_C, sc_D, cc_D) - evaluate_draws(err_F, sc_D, cc_D)
        quantities["D"] = format_interval(d_D, pt_D)

    # 7. D / C (ratio inside draw)
    if has_C and has_F:
        sc_DC, cc_DC = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_C_dc = evaluate_draws(err_C, sc_DC, cc_DC)
        d_F_dc = evaluate_draws(err_F, sc_DC, cc_DC)
        d_DC = np.where(d_C_dc != 0, (d_C_dc - d_F_dc) / d_C_dc, 0.0)
        quantities["D_over_C"] = format_interval(d_DC, pt_D_over_C)

    # 8. F - R (paired inside draw)
    if has_F and has_R:
        sc_FR, cc_FR = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_FR = evaluate_draws(err_F, sc_FR, cc_FR) - evaluate_draws(err_R, sc_FR, cc_FR)
        quantities["F_minus_R"] = format_interval(d_FR, pt_F_minus_R)

    # 9. F - Rcal (paired inside draw, if Rcal present)
    if has_F and has_rcal:
        pt_F_minus_Rcal = pt_F - pt_Rcal
        sc_FRc, cc_FRc = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_FRcal = evaluate_draws(err_F, sc_FRc, cc_FRc) - evaluate_draws(err_Rcal, sc_FRc, cc_FRc)
        quantities["F_minus_Rcal"] = format_interval(d_FRcal, pt_F_minus_Rcal)

    # 10. P - F (paired inside draw)
    if has_P and has_F:
        sc_PF, cc_PF = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_PF = evaluate_draws(err_P, sc_PF, cc_PF) - evaluate_draws(err_F, sc_PF, cc_PF)
        quantities["P_minus_F"] = format_interval(d_PF, pt_P_minus_F)

    # 11. C - R (paired inside draw, when requested)
    if report_c_minus_r and has_C and has_R:
        pt_C_minus_R = pt_C - pt_R
        sc_CR, cc_CR = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_CR = evaluate_draws(err_C, sc_CR, cc_CR) - evaluate_draws(err_R, sc_CR, cc_CR)
        quantities["C_minus_R"] = format_interval(d_CR, pt_C_minus_R)

    # Classification from interval of D
    classification = None
    if "D" in quantities and has_C:
        classification = classify_d(quantities["D"], mean_c=pt_C)

    # Descriptive per-step table (points only)
    unique_steps = np.unique(test_steps)
    per_step = []
    mean_err_C = np.mean(err_C, axis=0) if has_C else None
    mean_err_F = np.mean(err_F, axis=0) if has_F else None
    mean_err_R = np.mean(err_R, axis=0) if has_R else None
    mean_err_Rcal = np.mean(err_Rcal, axis=0) if has_rcal else None

    for step_val in unique_steps:
        mask = (test_steps == step_val)
        step_dict = {
            "step": int(step_val),
            "circuit_count": int(np.sum(mask)),
        }
        if mean_err_C is not None and mean_err_F is not None:
            step_dict["D"] = float(np.mean(mean_err_C[mask] - mean_err_F[mask]))
        if mean_err_F is not None and mean_err_R is not None:
            step_dict["F_minus_R"] = float(np.mean(mean_err_F[mask] - mean_err_R[mask]))
        if mean_err_F is not None and mean_err_Rcal is not None:
            step_dict["F_minus_Rcal"] = float(np.mean(mean_err_F[mask] - mean_err_Rcal[mask]))
        per_step.append(step_dict)

    res = {}
    if classification is not None:
        res["classification"] = classification
    res["quantities"] = quantities
    res["per_step"] = per_step
    if has_F:
        res["F_per_seed"] = [float(x) for x in np.mean(err_F, axis=1)]
    if reference_error is not None:
        res["reference_error"] = reference_error
    if meta_notes:
        res["meta"] = meta_notes

    return res


def parse_args():
    parser = argparse.ArgumentParser(description="ML-QEM Reanalysis Evaluation & Bootstrap")
    parser.add_argument("--fits", type=str, required=True,
                        help="Directory containing per-fit .npz and .json files")
    parser.add_argument("--out", type=str, default="mlqem_reanalysis_report.json",
                        help="Path to write the output JSON report")
    parser.add_argument("--settings", nargs="+",
                        default=["no_readout", "readout", "coherent"],
                        help="Settings to evaluate")
    parser.add_argument("--models", nargs="+", default=["ols", "rf", "mlp"],
                        help="Models to evaluate: ols, rf, mlp")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)),
                        help="Learner seeds to include in bootstrap (default: 1 to 20)")
    parser.add_argument("--draws", type=int, default=10000,
                        help="Number of bootstrap draws (default: 10000)")
    parser.add_argument("--bootstrap-seed", type=int, default=RULE_BOOTSTRAP_SEED,
                        help="Random seed for bootstrap generator (default: 20261002)")
    parser.add_argument("--frozen-rule", type=str, required=True,
                        help="the ML-QEM rule; every fit must record its SHA-256")
    parser.add_argument("--no-manifest-check", action="store_true",
                        help="synthetic fits only: skip the upstream data manifest")
    parser.add_argument("--targets", choices=["archived", "exact"], default="archived",
                        help="Evaluation targets: archived (default) or exact statevector")
    parser.add_argument("--exact-dir", type=str, default=None,
                        help="Directory containing exact-<setting>-<split>.npz files")
    parser.add_argument("--report-c-minus-r", action="store_true", default=False,
                        help="Include C - R in archived targets output")
    parser.add_argument("--descriptors", choices=["encoding", "exact"], default="encoding",
                        help="Fit descriptor type: encoding (default) or exact")
    parser.add_argument("--fit-frozen-rule", type=str, default=None,
                        help="Path to rule that generated fits; defaults to --frozen-rule")
    parser.add_argument("--analysis-arms", nargs="+", default=None,
                        help="Arms to analyze (default: None for all arms [R, Rcal, F, C, P])")
    return parser.parse_args()


def main():
    args = parse_args()
    script_sha = compute_file_sha256(os.path.abspath(__file__))

    if args.targets == "exact":
        if not args.exact_dir or not os.path.isdir(args.exact_dir):
            raise ValueError(
                f"With --targets exact, --exact-dir must be an existing directory, got {args.exact_dir!r}"
            )
        expected_paths = [find_exact_file(args.exact_dir, s) for s in args.settings]
        check_gate_approval(args.exact_dir, expected_npz_paths=expected_paths)

    all_results = {}
    all_input_files = []

    rule_sha = compute_file_sha256(args.frozen_rule)
    fit_rule_path = args.fit_frozen_rule or args.frozen_rule
    fit_rule_sha = compute_file_sha256(fit_rule_path)
    settings = [canonical_setting(s) for s in args.settings]
    # Phase 1: load and bind every requested fit. A missing or mismatched fit stops the analysis
    # before any error or interval is computed.
    loaded = {}
    for setting in settings:
        for model in args.models:
            loaded[(setting, model)] = load_model_fits(
                fits_dir=args.fits,
                setting=setting,
                model=model,
                seeds=args.seeds,
                rule_sha256=fit_rule_sha,
                manifest_check=not args.no_manifest_check,
                targets_mode=args.targets,
                exact_dir=args.exact_dir,
                descriptors=args.descriptors,
                analysis_arms=args.analysis_arms,
            )

    # Check that train_targets is consistent across the entire roster
    roster_train_targets_set = {
        cell["train_targets"] for cell in loaded.values() if cell.get("train_targets") is not None
    }
    if len(roster_train_targets_set) > 1:
        raise ValueError(f"Mixed train_targets across roster: {sorted(roster_train_targets_set)}")
    common_train_targets = next(iter(roster_train_targets_set)) if roster_train_targets_set else "archived"

    # Phase 2: errors, intervals, and labels.
    for setting in settings:
        all_results[setting] = {}
        for model in args.models:
            cell = loaded[(setting, model)]
            all_input_files.extend(cell["input_files"])
            should_report_cr = (args.targets == "exact" or args.report_c_minus_r)
            ref_err = cell["reference_error"] if args.targets == "exact" else None
            res = analyze_setting_model(
                arm_errors=errors_from_fits(cell),
                test_steps=cell["test_steps"],
                draws=args.draws,
                seed=args.bootstrap_seed,
                meta_notes=cell["notes"],
                report_c_minus_r=should_report_cr,
                reference_error=ref_err,
            )
            res["test_is_validation"] = cell["notes"]["test_is_validation"]
            if args.analysis_arms is not None or args.targets == "exact" or args.descriptors == "exact" or common_train_targets == "exact":
                res["train_targets"] = cell["train_targets"]
            all_results[setting][model] = res

    # Collect input SHA-256 values
    input_shas = {}
    for f in sorted(set(all_input_files)):
        if os.path.isfile(f):
            input_shas[os.path.basename(f)] = compute_file_sha256(f)

    report = {
        "script": os.path.basename(__file__),
        "script_sha256": script_sha,
        "bootstrap_draws": args.draws,
        "bootstrap_seed": args.bootstrap_seed,
        "learner_seeds": args.seeds,
        "frozen_rule_sha256": rule_sha,
        "input_files_sha256": input_shas,
    }
    if args.targets == "exact" or args.descriptors == "exact" or args.analysis_arms is not None:
        report["fit_frozen_rule_sha256"] = fit_rule_sha

    if args.analysis_arms is not None or args.targets == "exact" or args.descriptors == "exact" or common_train_targets == "exact":
        report["train_targets"] = common_train_targets
    if args.analysis_arms is not None:
        report["analysis_arms"] = args.analysis_arms

    if args.targets == "archived" and args.analysis_arms is None:
        report["reproduction_check"] = reproduction_record(all_results)
    else:
        report["reproduction_check_source"] = "artifacts/mlqem-own-data/analysis.json"
    report["results"] = all_results

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2, allow_nan=False)

    print(f"Report written successfully to {args.out}")


if __name__ == "__main__":
    main()

