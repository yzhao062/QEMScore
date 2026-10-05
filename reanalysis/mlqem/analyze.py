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


def canonical_setting(setting: str) -> str:
    """Maps a raw ML-QEM directory name to its setting name; other names pass unchanged."""
    for name, dir_name in SETTING_DIRS.items():
        if setting == dir_name:
            return name
    return setting


def load_fit(fits_dir: str, setting: str, model: str, arm: str, seed: int) -> Dict[str, Any]:
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
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in ("predictions", "targets", "test_circuit_ids", "test_steps")}
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


def fit_plan(model: str, seeds: List[int]) -> List[Tuple[str, str, List[int], bool]]:
    """(arm, fitted model, seeds read, repeated across seed slots) for every arm of a cell."""
    is_ols = (model.lower() == "ols")
    plan = [("R", model, [1], True), ("Rcal", "ols", [1], True)]
    for arm in ("F", "C"):
        plan.append((arm, model, [1] if is_ols else list(seeds), is_ols))
    plan.append(("P", model, list(seeds), False))
    return plan


def load_model_fits(
    fits_dir: str,
    setting: str,
    model: str,
    seeds: List[int],
    rule_sha256: Optional[str] = None,
    manifest_check: bool = True,
) -> Dict[str, Any]:
    """Loads and binds every fit of one setting and model; computes no error.

    Every arm is required. R is the model's own seed-1 record of the raw noisy values. Rcal is
    the setting's deterministic OLS fit on the noisy columns (`<setting>_ols_Rcal_seed1`), shared
    by every model. OLS arms C and F are deterministic: seed 1 is repeated across the learner-seed
    slots. Every fit must carry the same circuit identifiers, targets, steps, data-file SHA-256
    values (equal to the manifest's when `manifest_check`), rule SHA-256 (equal to `rule_sha256`
    when given), and `test_is_validation` flag.
    """
    reference = None
    input_files: List[str] = []
    arms: Dict[str, Tuple[List[np.ndarray], bool]] = {}
    for arm, fit_model, fit_seeds, repeat in fit_plan(model, seeds):
        rows = []
        for s in fit_seeds:
            fit = load_fit(fits_dir, setting, fit_model, arm, s)
            input_files.extend([fit["npz"], fit["json"]])
            meta = fit["meta"]
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
            rows.append(fit["predictions"])
        arms[arm] = (rows, repeat)

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
        "rcal_source": f"{setting}_ols_Rcal_seed1",
    }
    if reference["test_is_validation"]:
        notes["test_is_validation_note"] = (
            "The test split is the validation split"
            + ("; the MLP's scheduler read these labels." if model.lower() == "mlp" else ".")
        )
    if model.lower() == "ols":
        notes["ols_deterministic_seeds_repeated"] = True
        notes["note"] = (
            "For OLS, arms C, F, and Rcal are deterministic; seed 1 predictions repeated "
            "across learner seeds so the two-stage bootstrap applies identically."
        )
    return {"arms": arms, "targets": reference["targets"], "test_steps": reference["test_steps"],
            "input_files": input_files, "notes": notes, "n_seeds": len(seeds)}


def errors_from_fits(loaded: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-circuit errors (mean over the four observables) as (n_seeds x n_circuits) matrices."""
    out = {}
    for arm, (rows, repeat) in loaded["arms"].items():
        errs = [np.mean(np.abs(pred - loaded["targets"]), axis=1) for pred in rows]
        out[arm] = np.tile(errs[0], (loaded["n_seeds"], 1)) if repeat else np.array(errs)
    return out


def load_model_errors(
    fits_dir: str,
    setting: str,
    model: str,
    seeds: List[int],
    rule_sha256: Optional[str] = None,
    manifest_check: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], Dict[str, Any]]:
    """`load_model_fits` then `errors_from_fits`, for one setting and model."""
    loaded = load_model_fits(fits_dir, setting, model, seeds, rule_sha256, manifest_check)
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
    meta_notes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Performs point estimation, two-stage percentile bootstrap, and classification."""
    n_seeds = arm_errors["F"].shape[0]
    n_circuits = arm_errors["F"].shape[1]

    err_C = arm_errors["C"]
    err_F = arm_errors["F"]
    err_P = arm_errors["P"]
    err_R = arm_errors["R"]
    has_rcal = "Rcal" in arm_errors
    err_Rcal = arm_errors.get("Rcal")

    # Point estimates
    pt_C = float(np.mean(err_C))
    pt_F = float(np.mean(err_F))
    pt_P = float(np.mean(err_P))
    pt_R = float(np.mean(err_R))
    pt_D = pt_C - pt_F
    pt_D_over_C = pt_D / pt_C if pt_C != 0 else 0.0
    pt_F_minus_R = pt_F - pt_R
    pt_P_minus_F = pt_P - pt_F

    quantities = {}

    # 1. C
    sc_C, cc_C = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_C = evaluate_draws(err_C, sc_C, cc_C)
    quantities["C"] = format_interval(d_C, pt_C)

    # 2. F
    sc_F, cc_F = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_F = evaluate_draws(err_F, sc_F, cc_F)
    quantities["F"] = format_interval(d_F, pt_F)

    # 3. P
    sc_P, cc_P = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_P = evaluate_draws(err_P, sc_P, cc_P)
    quantities["P"] = format_interval(d_P, pt_P)

    # 4. R
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
    sc_D, cc_D = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_D = evaluate_draws(err_C, sc_D, cc_D) - evaluate_draws(err_F, sc_D, cc_D)
    quantities["D"] = format_interval(d_D, pt_D)

    # 7. D / C (ratio inside draw)
    sc_DC, cc_DC = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_C_dc = evaluate_draws(err_C, sc_DC, cc_DC)
    d_F_dc = evaluate_draws(err_F, sc_DC, cc_DC)
    d_DC = np.where(d_C_dc != 0, (d_C_dc - d_F_dc) / d_C_dc, 0.0)
    quantities["D_over_C"] = format_interval(d_DC, pt_D_over_C)

    # 8. F - R (paired inside draw)
    sc_FR, cc_FR = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_FR = evaluate_draws(err_F, sc_FR, cc_FR) - evaluate_draws(err_R, sc_FR, cc_FR)
    quantities["F_minus_R"] = format_interval(d_FR, pt_F_minus_R)

    # 9. F - Rcal (paired inside draw, if Rcal present)
    if has_rcal:
        pt_F_minus_Rcal = pt_F - pt_Rcal
        sc_FRc, cc_FRc = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
        d_FRcal = evaluate_draws(err_F, sc_FRc, cc_FRc) - evaluate_draws(err_Rcal, sc_FRc, cc_FRc)
        quantities["F_minus_Rcal"] = format_interval(d_FRcal, pt_F_minus_Rcal)

    # 10. P - F (paired inside draw)
    sc_PF, cc_PF = bootstrap_draws(n_seeds, n_circuits, draws=draws, seed=seed)
    d_PF = evaluate_draws(err_P, sc_PF, cc_PF) - evaluate_draws(err_F, sc_PF, cc_PF)
    quantities["P_minus_F"] = format_interval(d_PF, pt_P_minus_F)

    # Classification from interval of D
    classification = classify_d(quantities["D"], mean_c=pt_C)

    # Descriptive per-step table (points only)
    unique_steps = np.unique(test_steps)
    per_step = []
    mean_err_C = np.mean(err_C, axis=0)
    mean_err_F = np.mean(err_F, axis=0)
    mean_err_R = np.mean(err_R, axis=0)
    mean_err_Rcal = np.mean(err_Rcal, axis=0) if has_rcal else None

    for step_val in unique_steps:
        mask = (test_steps == step_val)
        c_step_D = float(np.mean(mean_err_C[mask] - mean_err_F[mask]))
        c_step_FR = float(np.mean(mean_err_F[mask] - mean_err_R[mask]))
        step_dict = {
            "step": int(step_val),
            "circuit_count": int(np.sum(mask)),
            "D": c_step_D,
            "F_minus_R": c_step_FR
        }
        if has_rcal:
            step_dict["F_minus_Rcal"] = float(np.mean(mean_err_F[mask] - mean_err_Rcal[mask]))
        per_step.append(step_dict)

    res = {
        "classification": classification,
        "quantities": quantities,
        "per_step": per_step,
        "F_per_seed": [float(x) for x in np.mean(err_F, axis=1)],
    }
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
    return parser.parse_args()


def main():
    args = parse_args()
    script_sha = compute_file_sha256(os.path.abspath(__file__))

    all_results = {}
    all_input_files = []

    rule_sha = compute_file_sha256(args.frozen_rule)
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
                rule_sha256=rule_sha,
                manifest_check=not args.no_manifest_check,
            )
    # Phase 2: errors, intervals, and labels.
    for setting in settings:
        all_results[setting] = {}
        for model in args.models:
            cell = loaded[(setting, model)]
            all_input_files.extend(cell["input_files"])
            res = analyze_setting_model(
                arm_errors=errors_from_fits(cell),
                test_steps=cell["test_steps"],
                draws=args.draws,
                seed=args.bootstrap_seed,
                meta_notes=cell["notes"]
            )
            res["test_is_validation"] = cell["notes"]["test_is_validation"]
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
        "reproduction_check": reproduction_record(all_results),
        "results": all_results
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2, allow_nan=False)

    print(f"Report written successfully to {args.out}")


if __name__ == "__main__":
    main()
