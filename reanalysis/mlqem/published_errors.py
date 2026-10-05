"""Errors of ML-QEM's published predictions over Trotter steps 0 to 14 (mlqem environment).

Reads the three figure files in docs/paper_figures/ of the ml-qem checkout (no model is fitted)
and prints, per setting and column, the mean over circuits of the mean over the four observables
of |prediction - ideal|. The no-readout values for `rfr_list` and `mlp` are the reference values
of the frozen rule's reproduction check (`analyze.PUBLISHED_ERRORS`).

Usage: python -m reanalysis.mlqem.published_errors --data-root upstream/ml-qem
"""

import argparse
import json
import os
import pickle

import numpy as np

FIGURE_FILES = {
    "no_readout": "no_readout_over_depths.pk",
    "readout": "incoherent_over_depths.pk",
    "coherent": "coherent_over_depths.pk",
}
COLUMNS = ("noisy", "rfr_list", "mlp", "ols_full", "ols", "gnn", "zne")


def published_errors(data_root: str, steps=range(15)) -> dict:
    out = {}
    for setting, name in FIGURE_FILES.items():
        with open(os.path.join(data_root, "docs", "paper_figures", name), "rb") as handle:
            df = pickle.load(handle)["df"]
        df = df[df["step"].isin(list(steps))]
        ideal = np.stack([np.ravel(v) for v in df["ideal"]])
        row = {"circuits": int(len(df))}
        for col in COLUMNS:
            if col in df.columns:
                pred = np.stack([np.ravel(v) for v in df[col]])
                row[col] = float(np.mean(np.mean(np.abs(pred - ideal), axis=1)))
        out[setting] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="ml-qem checkout")
    args = parser.parse_args()
    print(json.dumps(published_errors(args.data_root), indent=2))


if __name__ == "__main__":
    main()
