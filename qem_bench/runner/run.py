"""Benchmark runner: validate the dataset, train on the train split, score every
method and control on the test split, and report accuracy, harm, and the
three-bucket circuit-evaluation ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qem_bench.baselines.controls import (
    FeatureOnlyControl,
    NoisyOnlyControl,
    ShrinkageControl,
    ShuffledNoisyControl,
)
from qem_bench.baselines.ridge import RidgeMitigator
from qem_bench.datasets.generate import dataset_hash, group_shots
from qem_bench.datasets.schema import (
    FEATURE_SPEC_VERSION,
    FEATURES,
    validate_groups,
    validate_item,
)
from qem_bench.runner.metrics import method_metrics


def _load(data_dir: Path) -> tuple[list[dict], dict]:
    """Load and validate a dataset; refuse rows that do not match their manifest."""
    items = [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    for item in items:
        validate_item(item)
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("dataset contains duplicate item_id values")
    validate_groups(items)

    items.sort(key=lambda item: item["item_id"])
    actual_hash = dataset_hash(items)
    expected_hash = manifest.get("dataset_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"dataset hash mismatch: manifest={expected_hash}, items={actual_hash}"
        )

    current_spec = {"version": FEATURE_SPEC_VERSION, "features": list(FEATURES)}
    if manifest.get("feature_spec") != current_spec:
        raise ValueError("dataset feature_spec does not match the installed code")

    ledger = manifest.get("generation_ledger", {})
    if ledger.get("train_circuit_evals") != group_shots(items, "train") or ledger.get(
        "test_circuit_evals"
    ) != group_shots(items, "test"):
        raise ValueError("manifest generation ledger does not match item rows")
    # Each row corresponds to exactly one exact-label call, and the v0 slice admits
    # only statevector labels; the count is derived from the rows' declared method.
    bad_methods = {it["label_method"] for it in items} - {"statevector"}
    if bad_methods:
        raise ValueError(f"unsupported label_method values in items: {sorted(bad_methods)}")
    statevector_rows = sum(1 for it in items if it["label_method"] == "statevector")
    if ledger.get("label_evals_statevector") != statevector_rows:
        raise ValueError("manifest label_evals_statevector does not match item rows")

    counts = manifest.get("counts", {})
    expected_counts = {
        "items": len(items),
        "train_items": sum(1 for it in items if it["split"] == "train"),
        "test_items": sum(1 for it in items if it["split"] == "test"),
        "instances": len({it["instance"] for it in items}),
        "measurement_groups": len({it["measurement_group"] for it in items}),
    }
    if counts != expected_counts:
        raise ValueError(
            f"manifest counts do not match item rows: manifest={counts}, rows={expected_counts}"
        )
    return items, manifest


def _ledger(train_evals: int, extra_evals: int, pred_evals: int, n_test: int) -> dict:
    total = train_evals + extra_evals + pred_evals
    return {
        "B_train": train_evals,
        "B_extra": extra_evals,
        "B_pred": pred_evals,
        "total": total,
        "amortized_per_test_item": total / n_test,
    }


def run(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items, manifest = _load(data_dir)
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    if not train or not test:
        raise ValueError("dataset must contain both train and test items")

    r = np.array([it["noisy_expectation"] for it in test])
    y = np.array([it["ideal_expectation"] for it in test])
    n_test = len(test)
    train_group_evals = group_shots(items, "train")
    test_group_evals = group_shots(items, "test")

    ridge = RidgeMitigator().fit(train)

    # Required surrogate and leakage controls (frozen design). Cost semantics:
    # feature-only and shrinkage consume no noisy measurements at all (their only
    # inputs are structure features and exact train labels, which are logged
    # separately); noisy-only and shuffled-noisy consume the same measurement groups
    # as the primary ridge. shuffled-noisy is a diagnostic, never a deployable method.
    controls = {
        "feat-only": (FeatureOnlyControl().fit(train), _ledger(0, 0, 0, n_test), "control"),
        "noisy-only": (
            NoisyOnlyControl().fit(train),
            _ledger(train_group_evals, 0, test_group_evals, n_test),
            "control",
        ),
        "shrinkage": (ShrinkageControl().fit(train), _ledger(0, 0, 0, n_test), "control"),
        "shuf-noisy": (
            ShuffledNoisyControl().fit(train),
            _ledger(train_group_evals, 0, test_group_evals, n_test),
            "diagnostic",
        ),
    }

    methods: dict[str, dict] = {
        "raw": {
            "predictions": r,
            "ledger": _ledger(0, 0, test_group_evals, n_test),
            "role": "baseline",
            "config": {},
        },
        "ridge": {
            "predictions": ridge.predict(test),
            "ledger": _ledger(train_group_evals, 0, test_group_evals, n_test),
            "role": "learned",
            "config": {"best_alpha": ridge.best_alpha_},
        },
    }
    for name, (model, ledger, role) in controls.items():
        methods[name] = {
            "predictions": model.predict(test),
            "ledger": ledger,
            "role": role,
            "config": {},
        }

    results: dict = {
        "dataset_hash": manifest["dataset_hash"],
        "preset": manifest["preset"],
        "feature_spec": manifest["feature_spec"],
        "n_train_items": len(train),
        "n_test_items": n_test,
        "label_evals_statevector": manifest["generation_ledger"]["label_evals_statevector"],
        "methods": {},
    }
    for name, spec in methods.items():
        results["methods"][name] = {
            "metrics": method_metrics(np.asarray(spec["predictions"]), r, y),
            "ledger": spec["ledger"],
            "role": spec["role"],
            "config": spec["config"],
        }

    # Surrogate alarm (frozen design): if removing the noisy measurement leaves
    # accuracy intact, the model class is emulating the simulator on this slice.
    ridge_mae = results["methods"]["ridge"]["metrics"]["mae"]
    feat_mae = results["methods"]["feat-only"]["metrics"]["mae"]
    results["surrogate_alarm"] = {
        "ridge_mae": ridge_mae,
        "feature_only_mae": feat_mae,
        "triggered": bool(feat_mae <= ridge_mae * 1.05),
        "note": (
            "triggered means the learned model shows no measured incremental value "
            "from the noisy measurement on this slice; treat its score as a "
            "plumbing checksum, not mitigation"
        ),
    }

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    cols = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("signed_bias", "Bias"),
        ("excess_loss_total", "ExcessLoss(sum)"),
        ("overcorrection_rate", "OCR"),
        ("physicality_violation_rate", "PhysViol"),
    ]
    print(f"\ndataset {results['dataset_hash'][:12]}  preset {results['preset']}  "
          f"test items {results['n_test_items']}")
    header = f"{'method':<11}{'role':<11}" + "".join(f"{label:>16}" for _, label in cols)
    print(header)
    for name, spec in results["methods"].items():
        met = spec["metrics"]
        row = f"{name:<11}{spec['role']:<11}" + "".join(f"{met[key]:>16.6f}" for key, _ in cols)
        print(row)
    print(f"\n{'method':<11}{'B_train':>12}{'B_extra':>12}{'B_pred':>12}{'total':>14}"
          f"{'per-test':>12}")
    for name, spec in results["methods"].items():
        led = spec["ledger"]
        print(f"{name:<11}{led['B_train']:>12}{led['B_extra']:>12}{led['B_pred']:>12}"
              f"{led['total']:>14}{led['amortized_per_test_item']:>12.1f}")
    print(f"\nexact labels (statevector calls, logged separately): "
          f"{results['label_evals_statevector']}")
    alarm = results["surrogate_alarm"]
    state = "TRIGGERED" if alarm["triggered"] else "clear"
    print(f"surrogate alarm: {state} (ridge MAE {alarm['ridge_mae']:.6f} vs "
          f"feature-only {alarm['feature_only_mae']:.6f})")
