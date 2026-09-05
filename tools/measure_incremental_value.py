"""Reproduce an exploratory two-family S0 incremental-value measurement.

Artifacts go under --root, which should be a short, unused absolute path.
The gate uses source validation only and runs before the test roster. Validation
selection shares those rows, so its intervals are conditional diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qem_bench.baselines.controls import shuffle_noisy_items
from qem_bench.datasets.split_generate import generate_split
from qem_bench.datasets.splits import SplitSpec
from qem_bench.runner.metrics import build_cell_records, headline_metrics
from qem_bench.runner.run import (
    _prediction_items, _prediction_manifest, registered_methods, run,
    validate_run_artifact,
)
from qem_bench.stats.incremental_value import evaluate_incremental_value
from qem_bench.stats.bootstrap import circuit_blocked_bootstrap
from qem_bench.validation import validate_split_artifact


def _family_mae(rows, predictions):
    values = {}
    for family in sorted({row["family"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["family"] == family]
        subset = [rows[i] for i in indices]
        records = build_cell_records(
            "diagnostic", subset, np.asarray(predictions)[indices],
            [row["noisy_expectation"] for row in subset], artifact_id="pilot",
        )
        values[family] = headline_metrics(records)["mae"]
    return values


def measure_shuffle_intervals(summary_path: Path) -> dict:
    """Add paired source-validation intervals for training-shuffle diagnostics.

    Refit Liao with the same declared training permutation; use the saved ridge
    shuffle predictions. Only source rows enter the comparison; artifact loading
    still validates the complete dataset.
    """
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows, manifest = validate_split_artifact(summary["data_path"])
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    registration = next(r for r in registered_methods() if r.name == "liao")
    model = registration.factory().fit(
        shuffle_noisy_items(train, seed=1234), manifest=manifest, split_v2=True)
    model.select(validation)
    values = model.predict(_prediction_items(validation), manifest=_prediction_manifest(manifest)).predictions
    predictions = summary["validation_predictions"]
    predictions["liao-training-shuffle"] = dict(zip(
        [row["item_id"] for row in validation], map(float, values), strict=True))
    differences = {}
    for method, shuffled_method in (("ridge", "shuf-noisy"), ("liao", "liao-training-shuffle")):
        differences[method] = {}
        for family in ("tfi", "heisenberg"):
            subset = [row for row in validation if row["family"] == family]
            records = {name: build_cell_records(
                "diagnostic", subset, [predictions[name][row["item_id"]] for row in subset],
                [row["noisy_expectation"] for row in subset], artifact_id=summary["dataset_hash"],
            ) for name in (method, shuffled_method)}
            differences[method][family] = asdict(circuit_blocked_bootstrap(
                records[shuffled_method], reference_records=records[method],
                confidence=0.95, n_resamples=summary["gates"][method]["n_resamples"], seed=20260904,
            ))
    summary["training_shuffle_differences"] = differences
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="reuse a validated dataset matching this specification")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--validation", type=int, default=40)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--qubits", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--tfi-dt", type=float, default=0.2)
    parser.add_argument("--heisenberg-dt", type=float, default=0.15)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("--root must be absolute")
    args.root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    spec = SplitSpec(
        split_id="S0",
        source_domain={"circuit_instance": ["sampled"]},
        target_domain={"circuit_instance": ["sampled"]},
        fixed_axes={
            "noise_family": ["depolarizing_readout"],
            "noise_strength": ["L1", "L3"],
            "circuit_family": ["tfi", "heisenberg"],
            "family_native_depth": [args.depth],
            "observable_class": ["z_mid", "zz_mid"],
            "shots": [2048],
        },
        n_qubits=[args.qubits],
        role_counts={"train": args.train, "validation": args.validation, "test": args.test},
        family_parameters={"tfi": {"dt": args.tfi_dt},
                           "heisenberg": {"dt": args.heisenberg_dt}},
        budget_tier="H",
    )
    data = args.data or args.root / "data"
    if args.data is None:
        generate_split(spec, data, master_seed=args.seed)
    rows, manifest = validate_split_artifact(data)
    if manifest["split_spec"] != spec.to_dict() or manifest["master_seed"] != args.seed:
        raise ValueError("reused dataset must match the requested specification and seed")
    roles = {role: [row for row in rows if row["split"] == role]
             for role in ("train", "validation", "test")}
    validation = roles["validation"]
    prediction_rows = _prediction_items(validation)
    prediction_manifest = _prediction_manifest(manifest)
    print(f"generated {manifest['dataset_hash']} in {time.perf_counter()-start:.1f}s", flush=True)
    models, predictions, validation_mae = {}, {}, {}
    for registration in registered_methods():
        if registration.name == "zne":
            continue
        model = registration.factory().fit(roles["train"], manifest=manifest, split_v2=True)
        model.select(validation)
        values = model.predict(prediction_rows, manifest=prediction_manifest).predictions
        models[registration.name] = model
        predictions[registration.name] = dict(zip(
            [row["item_id"] for row in validation], map(float, values), strict=True))
        validation_mae[registration.name] = _family_mae(validation, values)
        print(f"source {registration.name}: {time.perf_counter()-start:.1f}s", flush=True)
    gates = {method: evaluate_incremental_value(
        validation, predictions, full_method=method, ratio_margin=1.05,
        confidence=0.95, n_resamples=args.resamples, seed=20260904,
    ) for method in ("ridge", "liao")}
    print("gates " + json.dumps({k: v["status"] for k, v in gates.items()}), flush=True)
    (args.root / "gate-before-test.json").write_text(
        json.dumps(gates, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # A training permutation tests learnability after refitting; fixed-model
    # validation permutations test the selected predictor's dependence on input.
    liao_registration = next(r for r in registered_methods() if r.name == "liao")
    shuffled_liao = liao_registration.factory().fit(
        shuffle_noisy_items(roles["train"], seed=1234), manifest=manifest, split_v2=True)
    shuffled_liao.select(validation)
    validation_mae["liao-training-shuffle"] = _family_mae(
        validation, shuffled_liao.predict(prediction_rows, manifest=prediction_manifest).predictions)
    shuffle_maes = {method: defaultdict(list) for method in ("ridge", "liao")}
    for shuffle_seed in range(20):
        groups = defaultdict(list)
        for index, row in enumerate(prediction_rows):
            groups[tuple(row[key] for key in ("family", "noise_family", "severity", "observable"))].append(index)
        shuffled = list(prediction_rows)
        for indices in groups.values():
            changed = shuffle_noisy_items([prediction_rows[i] for i in indices], seed=shuffle_seed)
            for index, row in zip(indices, changed, strict=True):
                shuffled[index] = row
        for method in shuffle_maes:
            values = models[method].predict(shuffled, manifest=prediction_manifest).predictions
            for family, mae in _family_mae(validation, values).items():
                shuffle_maes[method][family].append(mae)

    roster = run(data, args.root / "run")
    validate_run_artifact(roster)
    test_by_id = {row["item_id"]: row for row in roles["test"]}
    test_rows = [test_by_id[row["item_id"]] for row in roster["test_items"]]
    test_mae = {method: _family_mae(test_rows, value["predictions"])
                for method, value in roster["methods"].items()}
    summary = {
        "spec": spec.to_dict(), "master_seed": args.seed,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "data_path": str(data.resolve()),
        "dataset_hash": manifest["dataset_hash"], "run_artifact_id": roster["artifact_id"],
        "counts": {role: {"items": len(items), "circuits": len({row["circuit_id"] for row in items})}
                   for role, items in roles.items()},
        "validation_mae": validation_mae, "test_mae": test_mae, "gates": gates,
        "validation_predictions": predictions,
        "fixed_model_validation_shuffle": {
            method: {family: {"mean_mae": float(np.mean(values)),
                              "min_mae": min(values), "max_mae": max(values),
                              "permutation_maes": values}
                     for family, values in by_family.items()}
            for method, by_family in shuffle_maes.items()},
        "ledger_totals": {method: value["ledger"]["total"] for method, value in roster["methods"].items()},
        "seconds": time.perf_counter() - start,
        "scope": "exploratory S0; unequal actual budgets; conditional pointwise intervals; "
                 "validation selection and gate share rows; shuffle diagnostics are outside roster ledgers",
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("counts", "validation_mae", "test_mae", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
