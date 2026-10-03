"""R4 diagnostic suite: evaluates coupling information and bin-lookup predictability.

Evaluates the released ML-QEM binned-angle encoding (tools.mlqem_binned_angle_encoder, scaled=True)
across three regenerated datasets (s101, s211, s307) for TFI and Heisenberg spin chains:
  1. Distinct encoded vectors among training circuits and all circuits.
  2. kNN regression (k=5, standardized inputs) coupling recovery R^2 on test circuits
     (including sorted coupling triple recovery for Heisenberg).
  3. Bin-lookup macro MAE on ideal_expectation across 4 (observable, severity) cells per family.
  4. Sanity reference using exact couplings rounded to 2 decimals.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from tools.mlqem_binned_angle_encoder import encode_item_circuit

DATASETS = [
    "regen-shipped-s101-n640",
    "regen-shipped-s211-n640",
    "regen-shipped-s307-n640",
]

# The regenerated primary datasets (tools/regenerate_example.py); override with DATA_ROOT.
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))

OUT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "descriptor-information"
    / "diagnostics"
    / "r4-encoder.json"
)


def run_diagnostics_for_dataset(ds_path: Path) -> dict[str, Any]:
    """Run diagnostics on a single dataset directory."""
    items_file = ds_path / "items.jsonl"
    with open(items_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    # Map unique circuits
    unique_circuits: dict[str, dict[str, Any]] = {}
    for it in items:
        cid = it["circuit_id"]
        if cid not in unique_circuits:
            unique_circuits[cid] = it

    # Precompute encodings and 2-decimal rounded couplings
    t0 = time.time()
    encoded_cache: dict[str, tuple[float, ...]] = {}
    rounded_couplings_cache: dict[str, tuple[float, ...]] = {}

    for cid, it in unique_circuits.items():
        vec = tuple(encode_item_circuit(it, ds_path, scaled=True))
        encoded_cache[cid] = vec
        fam = it["family"]
        if fam == "tfi":
            rounded_couplings_cache[cid] = (round(float(it["j"]), 2), round(float(it["h"]), 2))
        else:
            rounded_couplings_cache[cid] = (
                round(float(it["jx"]), 2),
                round(float(it["jy"]), 2),
                round(float(it["jz"]), 2),
            )
    enc_time = time.time() - t0

    ds_report: dict[str, Any] = {"dataset_name": ds_path.name, "encoding_time_sec": enc_time}

    for fam in ("tfi", "heisenberg"):
        fam_circuits = [it for it in unique_circuits.values() if it["family"] == fam]
        train_c = [it for it in fam_circuits if it["split"] == "train"]
        val_c = [it for it in fam_circuits if it["split"] == "validation"]
        test_c = [it for it in fam_circuits if it["split"] == "test"]

        # 1. Distinct vectors
        train_vecs = set(encoded_cache[it["circuit_id"]] for it in train_c)
        all_vecs = set(encoded_cache[it["circuit_id"]] for it in fam_circuits)

        # 2. kNN Coupling Recovery
        X_train_enc = np.array([encoded_cache[it["circuit_id"]] for it in train_c], dtype=float)
        X_test_enc = np.array([encoded_cache[it["circuit_id"]] for it in test_c], dtype=float)

        scaler_enc = StandardScaler()
        X_tr_enc_sc = scaler_enc.fit_transform(X_train_enc)
        X_te_enc_sc = scaler_enc.transform(X_test_enc)

        X_train_rnd = np.array([rounded_couplings_cache[it["circuit_id"]] for it in train_c], dtype=float)
        X_test_rnd = np.array([rounded_couplings_cache[it["circuit_id"]] for it in test_c], dtype=float)

        scaler_rnd = StandardScaler()
        X_tr_rnd_sc = scaler_rnd.fit_transform(X_train_rnd)
        X_te_rnd_sc = scaler_rnd.transform(X_test_rnd)

        if fam == "tfi":
            y_train = np.array([[float(it["j"]), float(it["h"])] for it in train_c], dtype=float)
            y_test = np.array([[float(it["j"]), float(it["h"])] for it in test_c], dtype=float)

            # ML-QEM kNN
            knn_enc = KNeighborsRegressor(n_neighbors=5).fit(X_tr_enc_sc, y_train)
            pred_enc = knn_enc.predict(X_te_enc_sc)
            r2_enc = r2_score(y_test, pred_enc, multioutput="raw_values")
            knn_recovery_mlqem = {
                "r2_j": float(r2_enc[0]),
                "r2_h": float(r2_enc[1]),
                "mean_r2": float(np.mean(r2_enc)),
            }

            # Sanity kNN (rounded 2-decimal couplings)
            knn_rnd = KNeighborsRegressor(n_neighbors=5).fit(X_tr_rnd_sc, y_train)
            pred_rnd = knn_rnd.predict(X_te_rnd_sc)
            r2_rnd = r2_score(y_test, pred_rnd, multioutput="raw_values")
            knn_recovery_sanity = {
                "r2_j": float(r2_rnd[0]),
                "r2_h": float(r2_rnd[1]),
                "mean_r2": float(np.mean(r2_rnd)),
            }
        else:
            y_train = np.array(
                [[float(it["jx"]), float(it["jy"]), float(it["jz"])] for it in train_c], dtype=float
            )
            y_test = np.array(
                [[float(it["jx"]), float(it["jy"]), float(it["jz"])] for it in test_c], dtype=float
            )

            y_train_sorted = np.sort(y_train, axis=1)
            y_test_sorted = np.sort(y_test, axis=1)

            # ML-QEM kNN (raw)
            knn_enc = KNeighborsRegressor(n_neighbors=5).fit(X_tr_enc_sc, y_train)
            pred_enc = knn_enc.predict(X_te_enc_sc)
            r2_enc = r2_score(y_test, pred_enc, multioutput="raw_values")

            # ML-QEM kNN (sorted)
            knn_enc_s = KNeighborsRegressor(n_neighbors=5).fit(X_tr_enc_sc, y_train_sorted)
            pred_enc_s = knn_enc_s.predict(X_te_enc_sc)
            r2_enc_s = r2_score(y_test_sorted, pred_enc_s, multioutput="raw_values")

            knn_recovery_mlqem = {
                "r2_jx": float(r2_enc[0]),
                "r2_jy": float(r2_enc[1]),
                "r2_jz": float(r2_enc[2]),
                "mean_r2_raw": float(np.mean(r2_enc)),
                "r2_sorted_0": float(r2_enc_s[0]),
                "r2_sorted_1": float(r2_enc_s[1]),
                "r2_sorted_2": float(r2_enc_s[2]),
                "mean_r2_sorted": float(np.mean(r2_enc_s)),
            }

            # Sanity kNN (raw)
            knn_rnd = KNeighborsRegressor(n_neighbors=5).fit(X_tr_rnd_sc, y_train)
            pred_rnd = knn_rnd.predict(X_te_rnd_sc)
            r2_rnd = r2_score(y_test, pred_rnd, multioutput="raw_values")

            # Sanity kNN (sorted)
            knn_rnd_s = KNeighborsRegressor(n_neighbors=5).fit(X_tr_rnd_sc, y_train_sorted)
            pred_rnd_s = knn_rnd_s.predict(X_te_rnd_sc)
            r2_rnd_s = r2_score(y_test_sorted, pred_rnd_s, multioutput="raw_values")

            knn_recovery_sanity = {
                "r2_jx": float(r2_rnd[0]),
                "r2_jy": float(r2_rnd[1]),
                "r2_jz": float(r2_rnd[2]),
                "mean_r2_raw": float(np.mean(r2_rnd)),
                "r2_sorted_0": float(r2_rnd_s[0]),
                "r2_sorted_1": float(r2_rnd_s[1]),
                "r2_sorted_2": float(r2_rnd_s[2]),
                "mean_r2_sorted": float(np.mean(r2_rnd_s)),
            }

        # 3. Bin-lookup Predictor
        fam_items = [it for it in items if it["family"] == fam]
        cells = sorted(set((it["observable"], it["severity"]) for it in fam_items))
        cell_maes_enc: dict[str, float] = {}
        cell_maes_rnd: dict[str, float] = {}

        for obs, sev in cells:
            cell_key = f"{obs}_{sev}"
            tr_cell = [
                it
                for it in fam_items
                if it["observable"] == obs and it["severity"] == sev and it["split"] == "train"
            ]
            te_cell = [
                it
                for it in fam_items
                if it["observable"] == obs and it["severity"] == sev and it["split"] == "test"
            ]
            fallback = float(np.mean([it["ideal_expectation"] for it in tr_cell]))

            # ML-QEM bin lookup
            lookup_enc = defaultdict(list)
            for it in tr_cell:
                lookup_enc[encoded_cache[it["circuit_id"]]].append(float(it["ideal_expectation"]))
            lookup_enc_mean = {k: float(np.mean(v)) for k, v in lookup_enc.items()}

            errs_enc = []
            for it in te_cell:
                pred = lookup_enc_mean.get(encoded_cache[it["circuit_id"]], fallback)
                errs_enc.append(abs(pred - float(it["ideal_expectation"])))
            cell_maes_enc[cell_key] = float(np.mean(errs_enc))

            # Sanity bin lookup (rounded couplings)
            lookup_rnd = defaultdict(list)
            for it in tr_cell:
                lookup_rnd[rounded_couplings_cache[it["circuit_id"]]].append(
                    float(it["ideal_expectation"])
                )
            lookup_rnd_mean = {k: float(np.mean(v)) for k, v in lookup_rnd.items()}

            errs_rnd = []
            for it in te_cell:
                pred = lookup_rnd_mean.get(rounded_couplings_cache[it["circuit_id"]], fallback)
                errs_rnd.append(abs(pred - float(it["ideal_expectation"])))
            cell_maes_rnd[cell_key] = float(np.mean(errs_rnd))

        macro_mae_enc = float(np.mean(list(cell_maes_enc.values())))
        macro_mae_rnd = float(np.mean(list(cell_maes_rnd.values())))

        ds_report[fam] = {
            "distinct_vectors": {
                "train": len(train_vecs),
                "total_train_circuits": len(train_c),
                "all": len(all_vecs),
                "total_all_circuits": len(fam_circuits),
            },
            "coupling_recovery_knn": {
                "mlqem": knn_recovery_mlqem,
                "sanity_rounded": knn_recovery_sanity,
            },
            "bin_lookup_macro_mae": {
                "mlqem": {
                    "macro_mae": macro_mae_enc,
                    "cell_maes": cell_maes_enc,
                },
                "sanity_rounded": {
                    "macro_mae": macro_mae_rnd,
                    "cell_maes": cell_maes_rnd,
                },
            },
        }

    return ds_report


def print_markdown_tables(results: dict[str, Any]) -> None:
    """Print markdown formatted summary tables for all datasets."""
    for ds_name, ds_data in results.items():
        print(f"\n### Dataset: `{ds_name}`\n")
        print("| Family | Metric | ML-QEM Binned-Angle | Sanity Ref (Couplings 2-dec) |")
        print("| :--- | :--- | :---: | :---: |")

        # TFI
        tfi = ds_data["tfi"]
        dv = tfi["distinct_vectors"]
        knn = tfi["coupling_recovery_knn"]
        bl = tfi["bin_lookup_macro_mae"]
        d_tr, d_all = dv["train"], dv["all"]
        print(f"| **TFI** | Distinct encoded vectors (train / all) | {d_tr} / {d_all} (out of 640 / 1120) | 640 / 1120 (continuous) |")
        print(f"| | kNN R^2 (j) | {knn['mlqem']['r2_j']:.4f} | {knn['sanity_rounded']['r2_j']:.4f} |")
        print(f"| | kNN R^2 (h) | {knn['mlqem']['r2_h']:.4f} | {knn['sanity_rounded']['r2_h']:.4f} |")
        print(f"| | kNN R^2 (mean) | {knn['mlqem']['mean_r2']:.4f} | {knn['sanity_rounded']['mean_r2']:.4f} |")
        print(f"| | Bin-lookup Macro MAE | {bl['mlqem']['macro_mae']:.6f} | {bl['sanity_rounded']['macro_mae']:.6f} |")

        # Heisenberg
        hsb = ds_data["heisenberg"]
        dv_h = hsb["distinct_vectors"]
        knn_h = hsb["coupling_recovery_knn"]
        bl_h = hsb["bin_lookup_macro_mae"]
        dh_tr, dh_all = dv_h["train"], dv_h["all"]
        print(f"| **Heisenberg** | Distinct encoded vectors (train / all) | {dh_tr} / {dh_all} (out of 640 / 1120) | 640 / 1120 (continuous) |")
        print(f"| | kNN R^2 raw (jx, jy, jz) | {knn_h['mlqem']['r2_jx']:.4f}, {knn_h['mlqem']['r2_jy']:.4f}, {knn_h['mlqem']['r2_jz']:.4f} | {knn_h['sanity_rounded']['r2_jx']:.4f}, {knn_h['sanity_rounded']['r2_jy']:.4f}, {knn_h['sanity_rounded']['r2_jz']:.4f} |")
        print(f"| | kNN R^2 raw (mean) | {knn_h['mlqem']['mean_r2_raw']:.4f} | {knn_h['sanity_rounded']['mean_r2_raw']:.4f} |")
        print(f"| | kNN R^2 sorted triple (s0, s1, s2) | {knn_h['mlqem']['r2_sorted_0']:.4f}, {knn_h['mlqem']['r2_sorted_1']:.4f}, {knn_h['mlqem']['r2_sorted_2']:.4f} | {knn_h['sanity_rounded']['r2_sorted_0']:.4f}, {knn_h['sanity_rounded']['r2_sorted_1']:.4f}, {knn_h['sanity_rounded']['r2_sorted_2']:.4f} |")
        print(f"| | kNN R^2 sorted (mean) | {knn_h['mlqem']['mean_r2_sorted']:.4f} | {knn_h['sanity_rounded']['mean_r2_sorted']:.4f} |")
        print(f"| | Bin-lookup Macro MAE | {bl_h['mlqem']['macro_mae']:.6f} | {bl_h['sanity_rounded']['macro_mae']:.6f} |")


def main() -> None:
    data_root = DEFAULT_DATA_ROOT
    results: dict[str, Any] = {}

    for ds_name in DATASETS:
        ds_path = data_root / ds_name
        print(f"Processing {ds_name}...")
        results[ds_name] = run_diagnostics_for_dataset(ds_path)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full diagnostics to {OUT_PATH}")

    print_markdown_tables(results)


if __name__ == "__main__":
    main()
