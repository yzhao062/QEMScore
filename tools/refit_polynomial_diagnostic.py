"""Independent reproduction and refit of the degree-five polynomial diagnostic.

Fits ordinary least-squares degree-five polynomials in coupling parameters alone
for the six primary evaluations (seeds 101, 211, 307; families Heisenberg and TFI;
640 training circuits per family).

Evaluates predictions on all untouched-test items, recomputes control C and full
method F from archived test predictions, scores macro MAE with four-cell equal weighting,
compares against historical-polynomial.json and the manuscript's Table tab:poly-degree-5,
and calculates paired circuit-blocked bootstrap confidence intervals for F - Poly-5.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures

# Add repository root to sys.path so qemscore can be imported locally
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qemscore.datasets.generate import (
    _family_parameter_fields,
    _observable_support,
    _sample_and_build_circuit,
)
from qemscore.datasets.schema import canonical_physical_circuit_identity
from qemscore.datasets.split_generate import _pool_config, _seed_int
from qemscore.datasets.splits import SplitSpec, resolve_split_spec
from qemscore.labels.statevector import ideal_expectation as statevector_expectation
from qemscore.observables import z_support_label
from qemscore.validation import canonical_hash

SEEDS = (101, 211, 307)
FAMILIES = ("heisenberg", "tfi")
FEATURES = {
    "heisenberg": ["jx", "jy", "jz"],
    "tfi": ["j", "h"],
}
N_PARAMS = {
    "heisenberg": 56,  # comb(3 + 5, 5)
    "tfi": 21,         # comb(2 + 5, 5)
}
CELLS = [
    ("L1", "z_mid"),
    ("L1", "zz_mid"),
    ("L3", "z_mid"),
    ("L3", "zz_mid"),
]
PAPER_TABLE_VALUES = {
    (101, "heisenberg"): {"C": 0.00306, "F": 0.00345, "poly5": 0.000256, "poly_f_pct": 7.4},
    (101, "tfi"):        {"C": 0.00480, "F": 0.00234, "poly5": 0.000042, "poly_f_pct": 1.8},
    (211, "heisenberg"): {"C": 0.00447, "F": 0.00347, "poly5": 0.000292, "poly_f_pct": 8.4},
    (211, "tfi"):        {"C": 0.00300, "F": 0.00184, "poly5": 0.000051, "poly_f_pct": 2.8},
    (307, "heisenberg"): {"C": 0.00409, "F": 0.00329, "poly5": 0.000266, "poly_f_pct": 8.1},
    (307, "tfi"):        {"C": 0.00745, "F": 0.00191, "poly5": 0.000051, "poly_f_pct": 2.7},
}
HISTORICAL_VALUES = {
    (101, "heisenberg"): 0.000255615868,
    (101, "tfi"):        4.23741389e-05,
    (211, "heisenberg"): 0.000292322659,
    (211, "tfi"):        5.14294694e-05,
    (307, "heisenberg"): 0.000265768816,
    (307, "tfi"):        5.1229289e-05,
}


def regenerate_items_for_setting(
    archive_dir: Path, setting: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Regenerate circuits, couplings, and noiseless ideal labels using the clone's generators."""
    manifest_path = archive_dir / "data-manifests" / setting / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing data manifest at {manifest_path}")
    record_path = archive_dir / "records" / f"{setting}.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Missing record file at {record_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))

    spec = SplitSpec.from_dict(manifest["split_spec"])
    resolved = resolve_split_spec(spec)

    circuits: dict[str, list[dict[str, Any]]] = {}
    for pool in resolved.circuit_pools:
        cfg = _pool_config(pool)
        pool_circuits: list[dict[str, Any]] = []
        for local_instance in range(pool.n_instances):
            spawn_key = tuple(pool.pool_seed_key) + (0, local_instance)
            circuit_seed = _seed_int(seed, spawn_key)
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=spawn_key))
            params, circuit = _sample_and_build_circuit(cfg, rng, local_instance, circuit_seed)
            parameter_fields = _family_parameter_fields(params)
            descriptor, circuit_id = canonical_physical_circuit_identity(
                {
                    "family": pool.family,
                    "n_qubits": params.n_qubits,
                    "circuit_seed": circuit_seed,
                    **parameter_fields,
                }
            )
            pool_circuits.append(
                {
                    "circuit": circuit,
                    "params": params,
                    "parameter_fields": parameter_fields,
                    "circuit_id": circuit_id,
                    "local_instance": local_instance,
                }
            )
        circuits[pool.circuit_pool_id] = pool_circuits

    pool_by_id = {p.circuit_pool_id: p for p in resolved.circuit_pools}
    ideal_cache: dict[tuple[str, str], float] = {}
    items: list[dict[str, Any]] = []

    for cell in resolved.cells:
        pool = pool_by_id[cell.circuit_pool_id]
        for circ in circuits[cell.circuit_pool_id]:
            loc_inst = circ["local_instance"]
            for obs in cell.observable_ids:
                support = _observable_support(obs, circ["params"])
                pauli = z_support_label(circ["params"].n_qubits, support)
                cache_key = (str(circ["circuit_id"]), obs)
                if cache_key not in ideal_cache:
                    ideal = statevector_expectation(circ["circuit"], pauli)
                    ideal_cache[cache_key] = float(ideal)
                item_desc = {
                    "cell_id": cell.cell_id,
                    "circuit_id": circ["circuit_id"],
                    "instance": loc_inst,
                    "observable_id": obs,
                    "replicate": cell.replicate,
                }
                item_id = f"item-{canonical_hash(item_desc)}"
                items.append(
                    {
                        "item_id": item_id,
                        "setting": setting,
                        "split": cell.split,
                        "family": pool.family,
                        "circuit_id": circ["circuit_id"],
                        "instance": loc_inst,
                        "observable": obs,
                        "severity": cell.severity,
                        "noise_family": cell.axis_values["noise_family"],
                        "ideal_expectation": round(ideal_cache[cache_key], 12),
                        **circ["parameter_fields"],
                    }
                )

    # Verification against archived records
    rec_test = {it["item_id"]: it for it in record["test_items"]}
    rec_val = {it["item_id"]: it for it in record["validation_items"]}
    gen_by_id = {it["item_id"]: it for it in items}

    # Verify item identities
    assert set(rec_test.keys()).issubset(gen_by_id.keys()), f"Missing test items in {setting}"
    assert set(rec_val.keys()).issubset(gen_by_id.keys()), f"Missing validation items in {setting}"

    # Verify ideal expectation values
    max_test_diff = max(
        abs(gen_by_id[k]["ideal_expectation"] - rec_test[k]["ideal_expectation"])
        for k in rec_test
    )
    max_val_diff = max(
        abs(gen_by_id[k]["ideal_expectation"] - rec_val[k]["ideal_expectation"])
        for k in rec_val
    )
    if max_test_diff > 1e-11:
        raise ValueError(
            f"Test ideal_expectation mismatch in {setting}: max diff {max_test_diff:.2e}"
        )
    if max_val_diff > 1e-11:
        raise ValueError(
            f"Validation ideal_expectation mismatch in {setting}: max diff {max_val_diff:.2e}"
        )

    return items, record


def load_preextracted_couplings(
    couplings_csv_path: Path, archive_dir: Path
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Load items from pre-extracted probe-couplings CSV/gzip file."""
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    opener = gzip.open if str(couplings_csv_path).endswith(".gz") else open
    with opener(couplings_csv_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = dict(row)
            record["ideal_expectation"] = float(row["ideal_expectation"])
            for feature in ("jx", "jy", "jz", "j", "h"):
                record[feature] = float(row[feature]) if row[feature] != "" else None
            by_setting[row["setting"]].append(record)

    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for seed in SEEDS:
        setting = f"shipped-s{seed}-n640"
        record_path = archive_dir / "records" / f"{setting}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        items = by_setting[setting]
        results[setting] = (items, record)
    return results


def fit_and_evaluate(
    items: list[dict[str, Any]],
    record: dict[str, Any],
    seed: int,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 20260904,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit degree-5 polynomial per family and observable, score test items, and compute bootstrap CIs."""
    setting = f"shipped-s{seed}-n640"
    c_preds_record = record["test_predictions"]["liao-feat-only"]
    f_preds_record = record["test_predictions"]["liao"]

    setting_predictions: list[dict[str, Any]] = []
    setting_results: dict[str, Any] = {}

    for family in FAMILIES:
        feat_names = FEATURES[family]
        tr_items = [it for it in items if it["family"] == family and it["split"] == "train"]
        te_items = [it for it in items if it["family"] == family and it["split"] == "test"]

        # Sort test items deterministically
        te_items.sort(key=lambda r: (r["circuit_id"], r["observable"], r["severity"]))

        poly_preds: dict[str, float] = {}
        for obs in sorted({it["observable"] for it in tr_items}):
            # Collapse repeated severity copies to one target per physical circuit
            seen: dict[str, dict[str, Any]] = {}
            for it in tr_items:
                if it["observable"] == obs:
                    cid = it["circuit_id"]
                    if cid in seen:
                        assert seen[cid]["ideal_expectation"] == it["ideal_expectation"], (
                            f"Ideal expectation mismatch across severity copies for circuit {cid}"
                        )
                    seen[cid] = it

            tr_u = list(seen.values())
            assert len(tr_u) == 640, f"Expected 640 unique training circuits, got {len(tr_u)}"

            x_tr = np.array([[it[f] for f in feat_names] for it in tr_u], dtype=np.float64)
            y_tr = np.array([it["ideal_expectation"] for it in tr_u], dtype=np.float64)

            center = x_tr.mean(axis=0)
            scale = x_tr.std(axis=0)
            assert np.all(scale > 0), f"Zero scale encountered in {family} {obs}"

            poly = PolynomialFeatures(5, include_bias=True)
            design_tr = poly.fit_transform((x_tr - center) / scale)
            assert design_tr.shape[1] == N_PARAMS[family], (
                f"Design matrix columns {design_tr.shape[1]} != expected {N_PARAMS[family]}"
            )

            beta, _, rank, _ = np.linalg.lstsq(design_tr, y_tr, rcond=None)
            assert rank == design_tr.shape[1], (
                f"Rank deficient design: rank {rank} < {design_tr.shape[1]}"
            )

            te_obs = [it for it in te_items if it["observable"] == obs]
            x_te = np.array([[it[f] for f in feat_names] for it in te_obs], dtype=np.float64)
            design_te = poly.transform((x_te - center) / scale)

            # Suppress Apple Accelerate BLAS FP flags on macOS in numpy 2.2.6
            with np.errstate(all="ignore"):
                y_te_pred = design_te @ beta

            for it, pred in zip(te_obs, y_te_pred, strict=True):
                poly_preds[it["item_id"]] = float(pred)

        # Collect per-item predictions
        for it in te_items:
            iid = it["item_id"]
            p_val = poly_preds[iid]
            c_val = float(c_preds_record[iid])
            f_val = float(f_preds_record[iid])
            ideal = float(it["ideal_expectation"])
            setting_predictions.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "family": family,
                    "circuit_id": it["circuit_id"],
                    "item_id": iid,
                    "observable": it["observable"],
                    "severity": it["severity"],
                    "noise_family": it.get("noise_family", "depolarizing_readout"),
                    "ideal_expectation": ideal,
                    "control_C_pred": c_val,
                    "full_F_pred": f_val,
                    "poly5_pred": p_val,
                    "poly5_abs_error": abs(p_val - ideal),
                }
            )

        # Compute cell MAEs and macro MAE
        cell_maes_poly: dict[str, float] = {}
        cell_maes_c: dict[str, float] = {}
        cell_maes_f: dict[str, float] = {}

        for sev, obs in CELLS:
            cell_key = f"{sev}_{obs}"
            cell_items = [it for it in te_items if it["severity"] == sev and it["observable"] == obs]
            assert len(cell_items) == 160, f"Expected 160 cell items, got {len(cell_items)}"
            cell_maes_poly[cell_key] = float(
                np.mean([abs(poly_preds[it["item_id"]] - it["ideal_expectation"]) for it in cell_items])
            )
            cell_maes_c[cell_key] = float(
                np.mean([abs(c_preds_record[it["item_id"]] - it["ideal_expectation"]) for it in cell_items])
            )
            cell_maes_f[cell_key] = float(
                np.mean([abs(f_preds_record[it["item_id"]] - it["ideal_expectation"]) for it in cell_items])
            )

        poly5_macro_mae = float(np.mean(list(cell_maes_poly.values())))
        c_macro_mae = float(np.mean(list(cell_maes_c.values())))
        f_macro_mae = float(np.mean(list(cell_maes_f.values())))

        # Paired circuit-blocked bootstrap for F minus Poly-5
        circ_list = sorted({it["circuit_id"] for it in te_items})
        assert len(circ_list) == 160, f"Expected 160 test circuits, got {len(circ_list)}"
        circ_idx = {c: i for i, c in enumerate(circ_list)}

        f_matrix = np.zeros((160, 4), dtype=np.float64)
        p_matrix = np.zeros((160, 4), dtype=np.float64)
        for it in te_items:
            ci = circ_idx[it["circuit_id"]]
            cj = CELLS.index((it["severity"], it["observable"]))
            f_matrix[ci, cj] = abs(f_preds_record[it["item_id"]] - it["ideal_expectation"])
            p_matrix[ci, cj] = abs(poly_preds[it["item_id"]] - it["ideal_expectation"])

        point_diff = f_macro_mae - poly5_macro_mae

        rng = np.random.default_rng(bootstrap_seed + seed)
        draws = rng.integers(0, 160, size=(n_bootstrap, 160))
        # Vectorized macro delta across 10,000 draws
        # Shape: (n_bootstrap, 160, 4) -> mean(axis=1) is (n_bootstrap, 4) -> mean(axis=1) is (n_bootstrap,)
        boot_diffs = (f_matrix[draws] - p_matrix[draws]).mean(axis=1).mean(axis=1)
        ci_95_low, ci_95_high = np.percentile(boot_diffs, [2.5, 97.5])

        hist_poly5 = HISTORICAL_VALUES[(seed, family)]
        abs_diff_hist = abs(poly5_macro_mae - hist_poly5)
        rel_diff_hist = abs_diff_hist / hist_poly5

        paper_tab = PAPER_TABLE_VALUES[(seed, family)]
        poly_over_f_pct = (poly5_macro_mae / f_macro_mae) * 100.0
        reduction_beyond_f_pct = (1.0 - poly5_macro_mae / f_macro_mae) * 100.0

        setting_results[family] = {
            "setting": setting,
            "seed": seed,
            "family": family,
            "n_params": N_PARAMS[family],
            "control_C": c_macro_mae,
            "full_F": f_macro_mae,
            "refit_poly5_mae": poly5_macro_mae,
            "historical_poly5_mae": hist_poly5,
            "abs_diff_vs_historical": abs_diff_hist,
            "rel_diff_vs_historical": rel_diff_hist,
            "refit_poly_over_F_pct": poly_over_f_pct,
            "reduction_beyond_F_pct": reduction_beyond_f_pct,
            "paper_table": paper_tab,
            "cell_maes_poly5": cell_maes_poly,
            "cell_maes_control_C": cell_maes_c,
            "cell_maes_full_F": cell_maes_f,
            "paired_circuit_bootstrap": {
                "diff_F_minus_poly5": point_diff,
                "ci_95_low": float(ci_95_low),
                "ci_95_high": float(ci_95_high),
                "n_resamples": n_bootstrap,
                "bootstrap_seed": bootstrap_seed + seed,
            },
        }

    return setting_results, setting_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refit degree-five polynomial diagnostic on campaign primary evaluations."
    )
    parser.add_argument(
        "archive_dir",
        type=Path,
        help="Path to the extracted campaign-archive-v1 directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("build/polynomial-refit"),
        help="Path to output directory.",
    )
    parser.add_argument(
        "--couplings-csv",
        type=Path,
        default=None,
        help="Optional path to pre-extracted couplings CSV/gz. If omitted, regenerates data using clone code.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of paired circuit-bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260904,
        help="Base random seed for bootstrap (default: 20260904).",
    )
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Degree-Five Polynomial Diagnostic Independent Refit ===")
    print(f"Archive:    {archive_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Python:     {sys.version.split()[0]} ({sys.executable})")
    print(f"NumPy:      {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Timestamp:  {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    all_predictions: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    if args.couplings_csv:
        print(f"Loading pre-extracted couplings from {args.couplings_csv}...")
        preextracted = load_preextracted_couplings(args.couplings_csv, archive_dir)
        for seed in SEEDS:
            setting = f"shipped-s{seed}-n640"
            items, record = preextracted[setting]
            setting_res, preds = fit_and_evaluate(
                items, record, seed, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed
            )
            all_predictions.extend(preds)
            for family in FAMILIES:
                all_rows.append(setting_res[family])
    else:
        print("Regenerating datasets and exact ideal labels using package generator...")
        for seed in SEEDS:
            setting = f"shipped-s{seed}-n640"
            t0 = datetime.datetime.now()
            items, record = regenerate_items_for_setting(archive_dir, setting, seed)
            t1 = datetime.datetime.now()
            print(f"  [{setting}] regenerated {len(items)} items in {(t1 - t0).total_seconds():.2f}s")
            setting_res, preds = fit_and_evaluate(
                items, record, seed, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed
            )
            all_predictions.extend(preds)
            for family in FAMILIES:
                all_rows.append(setting_res[family])

    # Sort rows to standard order: (seed, family)
    all_rows.sort(key=lambda r: (r["seed"], r["family"]))

    # Output 1: CSV predictions
    csv_path = output_dir / "polynomial_test_predictions.csv"
    csv_fields = [
        "setting",
        "seed",
        "family",
        "circuit_id",
        "item_id",
        "observable",
        "severity",
        "noise_family",
        "ideal_expectation",
        "control_C_pred",
        "full_F_pred",
        "poly5_pred",
        "poly5_abs_error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_predictions)
    print(f"\nWrote per-item test predictions: {csv_path} ({len(all_predictions)} rows)")

    # Output 2: Summary JSON
    summary_data = {
        "schema_version": "qem-bench-polynomial-refit-v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "archive": {
            "name": archive_dir.name,
            # Tree fingerprint as defined in the archive's MANIFEST.md: the SHA-256 of
            # SHA256SUMS.txt, which lists every other file's hash.
            "sha256sums_digest": hashlib.sha256(
                (archive_dir / "SHA256SUMS.txt").read_bytes()
            ).hexdigest(),
        },
        "fit_recipe": (
            "Separate polynomial ordinary-least-squares for each family and observable, "
            "fixed degree 5, inputs centered and scaled on training rows only, training target "
            "collapsed to one per physical circuit and observable (640 training circuits per family), "
            "evaluated on untouched test rows without quantum measurement."
        ),
        "primary_evaluations": all_rows,
    }
    json_path = output_dir / "polynomial_refit_summary.json"
    json_path.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary JSON:              {json_path}")

    # Build and print comparison table
    print("\n" + "=" * 110)
    print(f"{'Seed':<5} {'Family':<11} {'Recomp C':<10} {'Recomp F':<10} {'Refit Poly-5':<14} {'Hist Poly-5':<14} {'Rel Diff':<10} {'Poly/F':<8} {'Paper P/F':<10} {'F - Poly [95% CI]':<22}")
    print("-" * 110)
    for r in all_rows:
        boot = r["paired_circuit_bootstrap"]
        ci_str = f"[{boot['ci_95_low']:.6f}, {boot['ci_95_high']:.6f}]"
        print(
            f"{r['seed']:<5} {r['family']:<11} "
            f"{r['control_C']:<10.5f} {r['full_F']:<10.5f} "
            f"{r['refit_poly5_mae']:<14.6e} {r['historical_poly5_mae']:<14.6e} "
            f"{r['rel_diff_vs_historical']:<10.2e} "
            f"{r['refit_poly_over_F_pct']:>6.1f}%   "
            f"{r['paper_table']['poly_f_pct']:>6.1f}%     "
            f"{ci_str:<22}"
        )
    print("=" * 110 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
