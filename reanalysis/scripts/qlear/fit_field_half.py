"""Fit the frozen A/C/O/F/C4 ladder on Q-LEAR's archived data.

The ladder, the inputs, the retraining recipe, the seeds and the score are all
fixed in PLAN-preprint-scope.md item 4, which was frozen before the controlled
campaign ran. Nothing here chooses any of them.

ktrain is imported before scikit-learn, lightgbm and xgboost on purpose: the
reverse order loads two OpenMP runtimes and the process dies with an access
violation before reaching any model.
"""

import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")

import ktrain  # noqa: E402  (must precede sklearn/lightgbm/xgboost)
from ktrain import tabular  # noqa: E402

import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

RELEASE = sys.argv[1]
OUT = sys.argv[2]
SEEDS = tuple(range(10))

# Optional third and fourth arguments fit exactly one arm and seed, then exit.
# TensorFlow does not release a fitted graph's memory back inside one process,
# so a 41-fit loop grows until the machine kills it. One process per fit keeps
# the footprint flat; the driver loop supplies the pairs.
ONLY_ARM = sys.argv[3] if len(sys.argv) > 3 else None
ONLY_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else None

# The release's own loader drops these three before any model sees them, in
# Train ML models.ipynb and in both RQ2 notebooks. Every published model
# therefore reads ten features rather than the thirteen in the CSV files.
DROPPED = ["Avg_inverted_error", "observed_prob_25", "observed_prob_75"]

DESC4 = ["Num_1Q_Gates", "Num_2Q_Gates", "circuit_depth", "circuit_width"]
STATE = ["state_weight"]
BASE_EXEC = ["observed_prob_50", "Avg_odds_ratio"]
DPE = ["Avg_inverted_error_25", "Avg_inverted_error_50", "Avg_inverted_error_75"]

# The frozen ladder. A shares C's inputs and is affine; C4 is the declared
# secondary; F is the full ten-feature Q-LEAR network.
ARMS = {
    "A": (DESC4 + STATE, "affine"),
    "C4": (DESC4, "network"),
    "C": (DESC4 + STATE, "network"),
    "O": (DESC4 + STATE + BASE_EXEC, "network"),
    "F": (DESC4 + STATE + BASE_EXEC + DPE, "network"),
}


def load(folder):
    files = sorted(glob.glob(os.path.join(RELEASE, folder, "*.csv")))
    if not files:
        raise SystemExit(f"no csv files under {folder}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df.loc[:, [c for c in df.columns if c not in DROPPED]].astype("float")


def evaluation_files(folder):
    """One entry per application and backend, keyed by the file name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RELEASE, folder, "*.csv"))):
        name = os.path.basename(path)[:-4]
        application, backend = name.split("_", 1)
        df = pd.read_csv(path)
        df = df.loc[:, [c for c in df.columns if c not in DROPPED]].astype("float")
        out[name] = (application, backend, df)
    return out


def fit_network(train_df, val_df, columns, seed):
    """The published recipe, on one arm's columns, under one declared seed."""
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


def main():
    os.makedirs(OUT, exist_ok=True)
    train_df = load("training_data")
    val_df = load("testing_data")
    print(f"train rows {len(train_df)}  validation rows {len(val_df)}", flush=True)

    panels = {
        "hardware": evaluation_files("real_circuits_hardware"),
        "simulator": evaluation_files("real_circuits"),
    }
    for panel, files in panels.items():
        print(f"{panel}: {len(files)} application-by-backend files", flush=True)

    manifest = {
        "arms": {a: {"columns": c, "kind": k} for a, (c, k) in ARMS.items()},
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
    with open(os.path.join(OUT, "fit-manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    for arm, (columns, kind) in ARMS.items():
        if ONLY_ARM is not None and arm != ONLY_ARM:
            continue
        seeds = (0,) if kind == "affine" else SEEDS
        if ONLY_SEED is not None:
            if ONLY_SEED not in seeds:
                continue
            seeds = (ONLY_SEED,)
        for seed in seeds:
            tag = f"{arm}-seed{seed}"
            target = os.path.join(OUT, f"pred-{tag}.json")
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
