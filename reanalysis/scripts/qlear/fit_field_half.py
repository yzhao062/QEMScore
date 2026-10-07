"""Fit the frozen ladder arms on Q-LEAR archived data.

The ladder inputs, retraining recipe, seeds, and score are fixed by the frozen plan.
Arm O2 adds the opposite ladder rung governed by docs/frozen-rules/2026-10-06-round9-follow-ups.md.
By default the script runs the legacy five arms and writes fit-manifest.json.
Arm O2 runs only when requested via --arm O2 and writes fit-manifest-O2.json.

ktrain is imported before scikit-learn, lightgbm, and xgboost on purpose.
The reverse order loads two OpenMP runtimes and the process dies with an access violation.
"""

import argparse
import hashlib
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    import ktrain  # noqa: E402  (must precede sklearn/lightgbm/xgboost)
    from ktrain import tabular  # noqa: E402
except ImportError:
    ktrain = None
    tabular = None

import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
try:
    import tensorflow as tf  # noqa: E402
except ImportError:
    tf = None
from sklearn.linear_model import LinearRegression  # noqa: E402

SEEDS = tuple(range(10))

# The release's own loader drops these three before any model sees them, in
# Train ML models.ipynb and in both RQ2 notebooks. Every published model
# therefore reads ten features rather than the thirteen in the CSV files.
DROPPED = ["Avg_inverted_error", "observed_prob_25", "observed_prob_75"]

DESC4 = ["Num_1Q_Gates", "Num_2Q_Gates", "circuit_depth", "circuit_width"]
STATE = ["state_weight"]
BASE_EXEC = ["observed_prob_50", "Avg_odds_ratio"]
DPE = ["Avg_inverted_error_25", "Avg_inverted_error_50", "Avg_inverted_error_75"]

# The frozen ladder. A shares C's inputs and is affine; C4 is the declared
# secondary; F is the full ten-feature Q-LEAR network; O2 is the opposite ladder rung.
LEGACY_ARMS = {
    "A": (DESC4 + STATE, "affine"),
    "C4": (DESC4, "network"),
    "C": (DESC4 + STATE, "network"),
    "O": (DESC4 + STATE + BASE_EXEC, "network"),
    "F": (DESC4 + STATE + BASE_EXEC + DPE, "network"),
}

O2_ARM = {
    "O2": (DESC4 + STATE + DPE, "network"),
}

ARMS = {**LEGACY_ARMS, **O2_ARM}


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load(folder, release):
    files = sorted(glob.glob(os.path.join(release, folder, "*.csv")))
    if not files:
        raise SystemExit(f"no csv files under {folder}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df.loc[:, [c for c in df.columns if c not in DROPPED]].astype("float")


def evaluation_files(folder, release):
    """One entry per application and backend, keyed by the file name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(release, folder, "*.csv"))):
        name = os.path.basename(path)[:-4]
        application, backend = name.split("_", 1)
        df = pd.read_csv(path)
        df = df.loc[:, [c for c in df.columns if c not in DROPPED]].astype("float")
        out[name] = (application, backend, df)
    return out


def fit_network(train_df, val_df, columns, seed):
    """The published recipe, on one arm's columns, under one declared seed."""
    if ktrain is None or tf is None:
        raise ImportError(
            "ktrain and tensorflow are required to fit network models. "
            "Use Python 3.10 with environments/requirements-qlear.txt."
        )
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    trn_cols = list(columns) + ["target"]
    trn, val, preproc = tabular.tabular_from_df(
        train_df.loc[:, trn_cols],
        is_regression=True,
        label_columns="target",
        val_df=val_df.loc[:, trn_cols],
        verbose=0,
    )
    model = tabular.tabular_regression_model(
        "mlp", trn,
        hidden_layers=[128, 1000, 128],
        hidden_dropouts=[0, 0.5, 0],
        metrics=["mse"],
        bn=False,
        verbose=0,
    )
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=128)
    hist = learner.autofit(lr=1e-4, early_stopping=20, reduce_on_plateau=3, verbose=0)
    predictor = ktrain.get_predictor(learner.model, preproc)
    epochs = len(hist.history.get("loss", [])) if hist is not None else None
    return predictor, epochs


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("release", help="Path to unpacked Q-LEAR release directory")
    parser.add_argument("out", help="Output directory for fit predictions")
    parser.add_argument("only_arm", nargs="?", default=None, help="Optional arm to fit")
    parser.add_argument("only_seed", nargs="?", type=int, default=None, help="Optional seed to fit")
    parser.add_argument("--arm", dest="flag_arm", default=None, help="Optional arm to fit")
    parser.add_argument("--seed", dest="flag_seed", type=int, default=None, help="Optional seed to fit")
    parser.add_argument("--frozen-rule", default=None, help="Path to frozen rule file")
    return parser.parse_args()


def main():
    args = parse_args()
    release = args.release
    out = args.out
    only_arm = args.flag_arm or args.only_arm
    only_seed = args.flag_seed if args.flag_seed is not None else args.only_seed

    if only_arm == "O2":
        if not args.frozen_rule or not os.path.isfile(args.frozen_rule):
            raise RuntimeError(
                "FROZEN RULE BARRIER: Fitting arm O2 requires a valid, existing "
                f"--frozen-rule file. Provided path: {args.frozen_rule!r}. Aborting immediately."
            )
        target_arms = O2_ARM
        manifest_filename = "fit-manifest-O2.json"
    elif only_arm is not None:
        if only_arm not in LEGACY_ARMS:
            raise SystemExit(f"Unknown arm '{only_arm}'; known arms are {list(ARMS.keys())}")
        target_arms = {only_arm: LEGACY_ARMS[only_arm]}
        manifest_filename = "fit-manifest.json"
    else:
        target_arms = LEGACY_ARMS
        manifest_filename = "fit-manifest.json"

    os.makedirs(out, exist_ok=True)
    train_df = load("training_data", release)
    val_df = load("testing_data", release)
    print(f"train rows {len(train_df)}  validation rows {len(val_df)}", flush=True)

    panels = {
        "hardware": evaluation_files("real_circuits_hardware", release),
        "simulator": evaluation_files("real_circuits", release),
    }
    for panel, files in panels.items():
        print(f"{panel}: {len(files)} application-by-backend files", flush=True)

    manifest = {
        "arms": {a: {"columns": c, "kind": k} for a, (c, k) in target_arms.items()},
        "seeds": list(SEEDS),
        "dropped_columns": DROPPED,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "panels": {p: sorted(f) for p, f in panels.items()},
        "recipe": {
            "hidden_layers": [128, 1000, 128],
            "hidden_dropouts": [0, 0.5, 0],
            "bn": False,
            "batch_size": 128,
            "lr": 1e-4,
            "early_stopping": 20,
            "reduce_on_plateau": 3,
        },
    }
    if args.frozen_rule and os.path.isfile(args.frozen_rule):
        manifest["frozen_rule"] = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
        manifest["frozen_rule_sha256"] = compute_file_sha256(args.frozen_rule)
        manifest["script_sha256"] = compute_file_sha256(__file__)

    manifest_target = os.path.join(out, manifest_filename)
    if not (manifest_filename == "fit-manifest.json" and os.path.exists(manifest_target) and only_arm is not None):
        with open(manifest_target, "w") as fh:
            json.dump(manifest, fh, indent=1)

    for arm, (columns, kind) in target_arms.items():
        seeds = (0,) if kind == "affine" else SEEDS
        if only_seed is not None:
            if only_seed not in seeds:
                continue
            seeds = (only_seed,)
        for seed in seeds:
            tag = f"{arm}-seed{seed}"
            target = os.path.join(out, f"pred-{tag}.json")
            if os.path.exists(target):
                print(f"{tag}: already present, skipping", flush=True)
                continue
            began = time.time()
            if kind == "affine":
                model = LinearRegression()
                model.fit(train_df.loc[:, columns].values, train_df["target"].values)
                predict = lambda df: model.predict(df.loc[:, columns].values)
                epochs = None
            else:
                predictor, epochs = fit_network(train_df, val_df, columns, seed)
                predict = lambda df: np.asarray(
                    predictor.predict(df.loc[:, list(columns) + ["target"]])
                ).reshape(-1)

            record = {"arm": arm, "seed": seed, "kind": kind,
                      "columns": list(columns), "epochs": epochs,
                      "validation_mse": None, "panels": {}}
            if arm == "O2":
                record["release_identity"] = {
                    "release_dir": os.path.basename(os.path.normpath(release)),
                }
            if arm == "O2" or (args.frozen_rule and os.path.isfile(args.frozen_rule)):
                record["frozen_rule"] = "docs/frozen-rules/2026-10-06-round9-follow-ups.md"
                record["frozen_rule_sha256"] = compute_file_sha256(args.frozen_rule)
                record["script_sha256"] = compute_file_sha256(__file__)

            vp = np.asarray(predict(val_df)).reshape(-1)
            record["validation_mse"] = float(
                np.mean(((val_df["target"].values / 100.0) - (vp / 100.0)) ** 2))

            for panel, files in panels.items():
                block = {}
                for name, (application, backend, df) in files.items():
                    preds = np.asarray(predict(df)).reshape(-1)
                    block[name] = {
                        "application": application,
                        "backend": backend,
                        "target": [float(x) for x in df["target"].values],
                        "prediction": [float(x) for x in preds],
                    }
                record["panels"][panel] = block

            with open(target, "w") as fh:
                json.dump(record, fh)
            print(f"{tag}: epochs={epochs} val_mse={record['validation_mse']:.6f} "
                  f"in {time.time() - began:.1f}s", flush=True)

    print("all fits complete", flush=True)


if __name__ == "__main__":
    main()
