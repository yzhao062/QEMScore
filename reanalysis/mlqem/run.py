"""CLI job runner for ML-QEM replication control.

Runs a list of (setting, model, arm, learner_seed) jobs across parallel processes,
encodes each setting dataset once with caching in <output-dir>/encoded/<setting>.npz,
skips existing results, and strictly enforces the frozen-rule constraint.
"""

import os
import sys
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

import numpy as np

from reanalysis.mlqem.encode import encode_setting_dataset, SETTINGS
from reanalysis.mlqem.arms import run_arm_job, compute_file_sha256
from reanalysis.mlqem.exact import check_gate_approval


def enforce_frozen_rule(frozen_rule: str):
    """Enforces safety check: refuses execution without an existing frozen rule file."""
    if not frozen_rule or not os.path.isfile(frozen_rule):
        raise RuntimeError(
            f"FROZEN RULE BARRIER: Refusing execution without a valid, existing "
            f"--frozen-rule file. Provided path: {frozen_rule!r}. Aborting immediately."
        )


MLQEM_COMMIT = "b1eccf8cf5ef4e9e498f3fe66e03951bc6b4a4d3"
DISTRIBUTIONS = ("qiskit", "qiskit-terra", "qiskit-aer", "numpy", "pandas", "scipy",
                 "scikit-learn", "torch", "torch-geometric", "blackwater")


def environment_record(data_root: str, strict: bool) -> Dict[str, Any]:
    """Commit, versions, and the loaded blackwater path, checked before the first fit.

    With `strict`, the data root must be an ml-qem checkout at MLQEM_COMMIT and blackwater must
    be imported from that checkout (`pip install --no-deps -e <data-root>`).
    """
    import platform
    import subprocess
    from importlib import metadata

    import blackwater

    root = os.path.realpath(data_root)
    blackwater_path = os.path.realpath(blackwater.__file__)
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "mlqem_commit": commit,
        "blackwater_path": blackwater_path,
        "python": platform.python_version(),
        "versions": versions,
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "MKL_NUM_THREADS")},
    }
    if strict:
        problems = []
        if commit != MLQEM_COMMIT:
            problems.append(f"{root} is at {commit}, not {MLQEM_COMMIT}")
        if not blackwater_path.startswith(root + os.sep):
            problems.append(f"blackwater is imported from {blackwater_path}, outside {root}")
        if problems:
            raise SystemExit("environment check failed; nothing fitted:\n  " + "\n  ".join(problems))
    return record


def _find_exact_file_for_split(exact_dir: str, setting: str, split: str) -> str:
    """Finds exact npz file for a setting and split."""
    config = SETTINGS.get(setting)
    dir_name = config["dir_name"] if config else setting
    candidates = [
        os.path.join(exact_dir, f"exact-{setting}-{split}.npz"),
        os.path.join(exact_dir, f"exact-{dir_name}-{split}.npz"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find exact npz file for setting={setting} split={split} in {exact_dir}. "
        f"Checked: {candidates}"
    )


def _load_exact_features(npz_path: str) -> Tuple[np.ndarray, str]:
    """Loads 5-column exact parameter features and file SHA-256."""
    sha = compute_file_sha256(npz_path)
    with np.load(npz_path) as ex_data:
        j_col = np.nan_to_num(ex_data["J"], nan=0.0)[:, None]
        basis_raw = ex_data["basis"]
        b0 = (basis_raw == 0).astype(float)[:, None]
        b1 = (basis_raw == 1).astype(float)[:, None]
        b2 = (basis_raw == 2).astype(float)[:, None]
        steps_col = ex_data["steps"].astype(float)[:, None]
        feats = np.hstack([j_col, b0, b1, b2, steps_col])
    return feats, sha


def parse_args():
    parser = argparse.ArgumentParser(description="ML-QEM Replication Control Runner")
    parser.add_argument("--settings", nargs="+",
                        default=["no_readout", "readout", "coherent"],
                        help="List of settings to run (e.g. no_readout, readout, coherent)")
    parser.add_argument("--models", nargs="+", default=["ols", "rf", "mlp"],
                        help="Models to fit: ols, rf, mlp")
    parser.add_argument("--arms", nargs="+", default=["F", "C", "P", "R", "Rcal"],
                        help="Arms to run: F, C, P, R, Rcal")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)),
                        help="Learner seeds (1 to 20)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing ml-qem datasets")
    parser.add_argument("--output-dir", type=str, default="results_mlqem",
                        help="Directory to save NPZ and JSON outputs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--frozen-rule", type=str, required=True,
                        help="Path to frozen rule file (strictly required)")
    parser.add_argument("--no-manifest-check", action="store_true",
                        help="rehearsal on generated data only: skip the upstream SHA-256 manifest")
    parser.add_argument("--descriptors", choices=["encoding", "exact"], default="encoding",
                        help="Descriptor set: encoding (58 cols) or exact (63 cols with exact parameters)")
    parser.add_argument("--train-targets", choices=["archived", "exact"], default="archived",
                        help="Targets used for training and validation: archived (default) or exact")
    parser.add_argument("--exact-dir", type=str, default=None,
                        help="Directory containing exact-<setting>-<split>.npz files")
    return parser.parse_args()


def job_worker(job_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Worker task executed in parallel process."""
    setting = job_dict["setting"]
    model = job_dict["model"]
    arm = job_dict["arm"]
    seed = job_dict["seed"]
    output_dir = job_dict["output_dir"]
    encoded_npz = job_dict["encoded_npz"]
    encoded_json = job_dict["encoded_json"]
    frozen_rule = job_dict["frozen_rule"]
    descriptors = job_dict.get("descriptors", "encoding")
    train_targets = job_dict.get("train_targets", "archived")

    base_name = f"{setting}_{model}_{arm}_seed{seed}"
    npz_path = os.path.join(output_dir, f"{base_name}.npz")
    json_path = os.path.join(output_dir, f"{base_name}.json")

    # Check idempotency
    if os.path.exists(npz_path) and os.path.exists(json_path) and os.path.getsize(npz_path) > 0:
        return {"status": "skipped", "job": base_name, "reason": "output_exists"}

    # Load encoded data from cache
    with np.load(encoded_npz) as raw_data:
        data = {k: raw_data[k] for k in raw_data.files}
    X_tr = data["X_train"]
    y_tr = data["y_train"]
    X_va = data["X_val"]
    y_va = data["y_val"]
    X_te = data["X_test"]
    y_te = data["y_test"]
    noisy_range = tuple(data["noisy_range"])
    test_steps = data["test_steps"].tolist()
    test_files = data["test_circuit_files"].tolist()
    test_indices = data["test_circuit_indices"].tolist()
    test_ids = list(zip(test_files, test_indices))

    with open(encoded_json, "r") as f:
        meta = json.load(f)
    test_is_val = meta.get("test_is_validation", False)
    input_shas = meta.get("input_files_sha256", {})

    exact_shas = {}
    if descriptors == "exact" or train_targets == "exact":
        exact_paths = job_dict["exact_npz_paths"]
        for split_key, y_split in (("train", y_tr), ("val", y_va), ("test", y_te)):
            ep = exact_paths[split_key]
            exact_shas[os.path.basename(ep)] = compute_file_sha256(ep)
            with np.load(ep) as ex_check:
                if f"{split_key}_circuit_files" in data:
                    if not np.array_equal(data[f"{split_key}_circuit_files"], ex_check["step_file"]):
                        raise ValueError(f"Encoded {split_key} circuit files do not match exact npz in order")
                if f"{split_key}_circuit_indices" in data:
                    if not np.array_equal(data[f"{split_key}_circuit_indices"], ex_check["entry_index"]):
                        raise ValueError(f"Encoded {split_key} circuit indices do not match exact npz in order")
                if f"{split_key}_steps" in data:
                    if not np.array_equal(data[f"{split_key}_steps"], ex_check["steps"]):
                        raise ValueError(f"Encoded {split_key} steps do not match exact npz in order")
                archived = np.asarray(ex_check["archived"], dtype=np.float64)
                if archived.shape != np.asarray(y_split).shape or not np.allclose(
                        archived, np.asarray(y_split, dtype=np.float64), rtol=0.0, atol=1e-6):
                    raise ValueError(f"Exact npz rows do not match the encoded {split_key} labels")

        if train_targets == "exact":
            with np.load(exact_paths["train"]) as ex_tr_npz:
                y_tr = ex_tr_npz["exact"]
            with np.load(exact_paths["val"]) as ex_va_npz:
                y_va = ex_va_npz["exact"]

        if descriptors == "exact":
            if tuple(int(v) for v in noisy_range) != (54, 58):
                raise ValueError(f"Expected noisy columns (54, 58), found {noisy_range}")
            exact_tr, h_tr = _load_exact_features(exact_paths["train"])
            exact_va, h_va = _load_exact_features(exact_paths["val"])
            exact_te, h_te = _load_exact_features(exact_paths["test"])
            X_tr = np.hstack([X_tr[:, :54], exact_tr, X_tr[:, 54:58]])
            X_va = np.hstack([X_va[:, :54], exact_va, X_va[:, 54:58]])
            X_te = np.hstack([X_te[:, :54], exact_te, X_te[:, 54:58]])
            noisy_range = (59, 63)

    # Execute fit and prediction
    run_arm_job(
        setting=setting,
        model=model,
        arm=arm,
        learner_seed=seed,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        X_val=X_va,
        y_val=y_va,
        noisy_range=noisy_range,
        output_dir=output_dir,
        test_ids=test_ids,
        test_steps=test_steps,
        test_is_validation=test_is_val,
        frozen_rule_path=frozen_rule,
        data_files_sha256=input_shas,
        descriptors=descriptors,
        exact_npz_sha256=exact_shas if (descriptors == "exact" or train_targets == "exact") else None,
        train_targets=train_targets,
    )

    return {"status": "completed", "job": base_name}



def main():
    args = parse_args()

    # Safety constraint: enforce frozen rule requirement unconditionally
    enforce_frozen_rule(args.frozen_rule)

    os.makedirs(args.output_dir, exist_ok=True)

    # Record (and, outside rehearsals, check) the environment before any encoding or fit.
    env = environment_record(args.data_root, strict=not args.no_manifest_check)
    env.update({"argv": sys.argv, "frozen_rule_sha256": compute_file_sha256(args.frozen_rule)})
    with open(os.path.join(args.output_dir, "run-log.jsonl"), "a") as handle:
        handle.write(json.dumps(env, sort_keys=True) + "\n")
    print(f"Environment: ml-qem {env['mlqem_commit']}, blackwater {env['blackwater_path']}")

    # Encode each setting split once with caching
    encoded_map = {}
    for setting in args.settings:
        print(f"Checking/encoding setting dataset: {setting}...")
        npz_p, json_p = encode_setting_dataset(
            setting=setting,
            data_root=args.data_root,
            output_dir=args.output_dir,
            check_manifest=not args.no_manifest_check,
        )
        encoded_map[setting] = (npz_p, json_p)

    need_exact = (args.descriptors == "exact" or args.train_targets == "exact")
    exact_paths_by_setting = {}
    if need_exact:
        if not args.exact_dir or not os.path.isdir(args.exact_dir):
            raise ValueError(
                f"With exact descriptors or train-targets, --exact-dir must be an existing directory, got {args.exact_dir!r}"
            )
        all_exact_paths = []
        for setting in args.settings:
            config = SETTINGS[setting]
            tr_p = _find_exact_file_for_split(args.exact_dir, setting, config["train_split"])
            va_p = _find_exact_file_for_split(args.exact_dir, setting, config["val_split"])
            te_p = _find_exact_file_for_split(args.exact_dir, setting, config["test_split"])
            exact_paths_by_setting[setting] = {
                "train": tr_p,
                "val": va_p,
                "test": te_p,
            }
            all_exact_paths.extend([tr_p, va_p, te_p])
        check_gate_approval(args.exact_dir, expected_npz_paths=sorted(set(all_exact_paths)))

    # Schedule jobs
    jobs = []
    for setting in args.settings:
        npz_p, json_p = encoded_map[setting]
        for model in args.models:
            for arm in args.arms:
                # Arm Rcal is only applicable to OLS
                if arm == "Rcal" and model.lower() != "ols":
                    continue

                # Determine seed list:
                # - R is raw unmitigated baseline: run once (seed 1)
                # - OLS arms C, F, Rcal are deterministic: run once (seed 1)
                # - P is permuted control (all models): run across all seeds
                # - RF and MLP arms F, C, P: run across all seeds
                if arm == "R":
                    seeds_to_run = [1]
                elif model.lower() == "ols" and arm in ["C", "F", "Rcal"]:
                    seeds_to_run = [1]
                else:
                    seeds_to_run = args.seeds

                for seed in seeds_to_run:
                    j_info = {
                        "setting": setting,
                        "model": model,
                        "arm": arm,
                        "seed": seed,
                        "output_dir": args.output_dir,
                        "encoded_npz": npz_p,
                        "encoded_json": json_p,
                        "frozen_rule": args.frozen_rule,
                        "descriptors": args.descriptors,
                        "train_targets": args.train_targets,
                    }
                    if need_exact:
                        j_info["exact_npz_paths"] = exact_paths_by_setting[setting]
                    jobs.append(j_info)


    print(f"Total jobs scheduled: {len(jobs)} across {args.workers} workers.")

    skipped = 0
    executed = 0

    if args.workers <= 1:
        for j in jobs:
            res = job_worker(j)
            if res.get("status") == "skipped":
                skipped += 1
            else:
                executed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(job_worker, j): j for j in jobs}
            for future in as_completed(futures):
                res = future.result()
                if res.get("status") == "skipped":
                    skipped += 1
                else:
                    executed += 1

    print(f"Execution summary: {executed} executed, {skipped} skipped.")


if __name__ == "__main__":
    main()
