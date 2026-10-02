#!/usr/bin/env python3
"""
build_target_table.py: Build S3_target_table.npz from upstream Q-Cluster data.

Because the upstream Q-Cluster repository (https://github.com/hrushikesh890/Q-Cluster)
declares no open-source license, upstream data files and derived tables cannot be
redistributed directly in this repository.

This script reads the upstream training pickle fetched by download_upstream.sh:
  data/training_20250307_2151.pkl (SHA256: 163747ce5d43f1433dcd641d771b47fe5bd94e660640e9855cd66b1b773a1eb8)
and converts it to a structured .npz file (S3_target_table.npz) in a user-chosen
output directory for scoring.

Environment:
  Standard Python 3.8+ with numpy and pandas.
  No Qiskit dependency is required for training_20250307_2151.pkl (it contains
  only basic Python objects: lists of strings and numbers).
"""

import argparse
import hashlib
import os
import pickle
import sys
import numpy as np
import pandas as pd

EXPECTED_SHA256 = "163747ce5d43f1433dcd641d771b47fe5bd94e660640e9855cd66b1b773a1eb8"
COLUMNS = [
    'Machine', 'Algorithm', 'Qubit', 'Measure', 'rz', 'sx', 'x', 'ecr',
    'entropy', 'esp_m', 'ip Entropy', 'Target'
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Convert upstream Q-Cluster pickle to S3_target_table.npz"
    )
    parser.add_argument(
        "--upstream-path",
        "-u",
        default=None,
        help="Path to Q-Cluster repository root or directly to training_20250307_2151.pkl"
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=None,
        help="Directory where S3_target_table.npz will be written"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    reanalysis_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Resolve input path
    pickle_path = None
    if args.upstream_path:
        if os.path.isfile(args.upstream_path):
            pickle_path = args.upstream_path
        elif os.path.isdir(args.upstream_path):
            candidate = os.path.join(args.upstream_path, "data", "training_20250307_2151.pkl")
            if os.path.exists(candidate):
                pickle_path = candidate
            elif os.path.exists(os.path.join(args.upstream_path, "training_20250307_2151.pkl")):
                pickle_path = os.path.join(args.upstream_path, "training_20250307_2151.pkl")

    if not pickle_path or not os.path.exists(pickle_path):
        # Check standard default locations
        defaults = [
            os.path.join(reanalysis_dir, "upstream", "s3_qcluster", "data", "training_20250307_2151.pkl"),
            os.path.join(reanalysis_dir, "upstream", "Q-Cluster", "data", "training_20250307_2151.pkl"),
            os.path.join(script_dir, "Q-Cluster", "data", "training_20250307_2151.pkl"),
            os.path.join(os.getcwd(), "upstream", "s3_qcluster", "data", "training_20250307_2151.pkl"),
        ]
        for d in defaults:
            if os.path.exists(d):
                pickle_path = d
                break

    if not pickle_path or not os.path.exists(pickle_path):
        print(f"ERROR: Could not locate training_20250307_2151.pkl.", file=sys.stderr)
        print("Please run `bash download_upstream.sh` first or specify --upstream-path.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading upstream pickle: {pickle_path}")
    h = sha256_file(pickle_path)
    print(f"SHA-256: {h}")
    if h != EXPECTED_SHA256:
        print(f"WARNING: Checksum mismatch! Expected {EXPECTED_SHA256}, got {h}", file=sys.stderr)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data, columns=COLUMNS).dropna().reset_index(drop=True)
    if df.shape != (145, 12):
        print(f"ERROR: Expected DataFrame shape (145, 12), got {df.shape}", file=sys.stderr)
        sys.exit(1)

    machine = np.array(df['Machine'].values, dtype='<U32')
    circuit = np.array(df['Algorithm'].values, dtype='<U32')
    numeric = df.iloc[:, 2:].values.astype(float)

    # Resolve output directory
    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join(reanalysis_dir, "outputs", "s3_qcluster")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, "S3_target_table.npz")
    np.savez(out_file, machine=machine, circuit=circuit, numeric=numeric)

    out_h = sha256_file(out_file)
    print(f"Successfully wrote target table: {out_file}")
    print(f"Shape: machine={machine.shape}, circuit={circuit.shape}, numeric={numeric.shape}")
    print(f"Output SHA-256: {out_h}")


if __name__ == "__main__":
    main()
