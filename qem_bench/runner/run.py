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
from qem_bench.baselines.zne import SCALE_FACTORS, ZNEMitigator
from qem_bench.datasets.generate import (
    dataset_hash,
    frozen_item_stream_hash,
    group_shots,
)
from qem_bench.datasets.schema import (
    FAMILY_STRATA,
    FEATURE_SPEC_VERSION,
    FEATURES,
    validate_groups,
    validate_item,
)
from qem_bench.noise import SEVERITY_GRIDS
from qem_bench.runner.metrics import method_metrics

SUPPORTED_LABEL_METHODS = frozenset({"statevector", "stim"})


def _load(data_dir: Path) -> tuple[list[dict], dict]:
    """Load and validate a dataset; refuse rows that do not match their manifest."""
    items = [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    current_spec = {"version": FEATURE_SPEC_VERSION, "features": list(FEATURES)}
    if manifest.get("feature_spec") != current_spec:
        raise ValueError("dataset feature_spec does not match the installed code")

    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("dataset contains duplicate item_id values")

    items.sort(key=lambda item: item["item_id"])
    actual_hash = dataset_hash(items)
    expected_hash = manifest.get("dataset_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"dataset hash mismatch: manifest={expected_hash}, items={actual_hash}"
        )

    missing_stratum = [item for item in items if "stratum" not in item]
    if missing_stratum:
        frozen_hash = frozen_item_stream_hash(
            manifest.get("preset"), manifest.get("config")
        )
        if frozen_hash is None or actual_hash != frozen_hash:
            raise ValueError(
                "item rows missing stratum outside the frozen "
                "preset/config/dataset_hash allow-list"
            )
        for item in missing_stratum:
            family = item.get("family")
            if family not in FAMILY_STRATA:
                raise ValueError(f"unknown circuit family {family!r}")
            item["stratum"] = FAMILY_STRATA[family]

    for item in items:
        validate_item(item)
    validate_groups(items)

    row_noise_families = {item["noise_family"] for item in items}
    if len(row_noise_families) != 1:
        raise ValueError(
            "item rows must contain exactly one noise_family; "
            f"found {sorted(row_noise_families)!r}"
        )
    row_noise_family = next(iter(row_noise_families))
    if row_noise_family not in SEVERITY_GRIDS:
        raise ValueError(f"unknown row noise_family {row_noise_family!r}")

    config = manifest.get("config")
    manifest_noise_family = (
        config.get("noise_family") if isinstance(config, dict) else None
    )
    if manifest_noise_family != row_noise_family:
        raise ValueError(
            "manifest config.noise_family does not match item rows: "
            f"manifest={manifest_noise_family!r}, rows={row_noise_family!r}"
        )

    valid_severities = SEVERITY_GRIDS[row_noise_family]
    invalid_severities = sorted(
        {
            item["severity"]
            for item in items
            if item["severity"] not in valid_severities
        }
    )
    if invalid_severities:
        raise ValueError(
            f"item row severities are invalid for {row_noise_family}: "
            f"{invalid_severities!r}"
        )
    if manifest.get("severity_grid") != valid_severities:
        raise ValueError(
            "manifest severity_grid does not match the installed registry for "
            f"{row_noise_family}"
        )
    if manifest.get("severity_grids") != SEVERITY_GRIDS:
        raise ValueError("manifest severity_grids does not match the installed registry")

    ledger = manifest.get("generation_ledger", {})
    if ledger.get("train_circuit_evals") != group_shots(items, "train") or ledger.get(
        "test_circuit_evals"
    ) != group_shots(items, "test"):
        raise ValueError("manifest generation ledger does not match item rows")
    # Each row corresponds to one exact-label call. Derive and validate the count
    # independently for every supported method.
    row_methods = {item["label_method"] for item in items}
    bad_methods = row_methods - SUPPORTED_LABEL_METHODS
    if bad_methods:
        raise ValueError(f"unsupported label_method values in items: {sorted(bad_methods)}")
    label_counts = {
        method: sum(item["label_method"] == method for item in items)
        for method in sorted(SUPPORTED_LABEL_METHODS)
    }
    for method, expected in label_counts.items():
        key = f"label_evals_{method}"
        if ledger.get(key, 0) != expected:
            raise ValueError(f"manifest {key} does not match item rows")

    strata = {item["stratum"] for item in items}
    if len(strata) != 1:
        raise ValueError(
            "the Clifford control stratum is excluded from the "
            "continuous-regression headline; runner requires one stratum per dataset"
        )

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
    zne = ZNEMitigator().fit(train)
    zne_seed_stream = np.random.SeedSequence(
        int(manifest["master_seed"]), spawn_key=(2,)
    )
    zne_predictions, zne_extra_per_group = zne.predict(
        test, seed_stream=zne_seed_stream
    )
    zne_extra_evals = int(np.sum(zne_extra_per_group, dtype=np.int64))

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
        "zne": {
            "predictions": zne_predictions,
            "ledger": _ledger(0, zne_extra_evals, test_group_evals, n_test),
            "role": "qem-baseline",
            "config": {
                "scale_factors": list(SCALE_FACTORS),
                "extrapolator": zne.extrapolator,
                "scale_one": "reused stored noisy_expectation",
                "fold_order": "optimization-level-1 transpile, then global fold",
                "sharing": "one folded execution per measurement group and scale",
                "seed_stream": "SeedSequence(master_seed, spawn_key=(2,))",
            },
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
        "stratum": items[0]["stratum"],
        "n_train_items": len(train),
        "n_test_items": n_test,
        "label_evals": {
            method: int(
                manifest["generation_ledger"].get(f"label_evals_{method}", 0)
            )
            for method in sorted(SUPPORTED_LABEL_METHODS)
        },
        "methods": {},
    }
    results["label_evals_statevector"] = results["label_evals"]["statevector"]
    results["label_evals_stim"] = results["label_evals"]["stim"]
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
    header = f"{'method':<11}{'role':<13}" + "".join(f"{label:>16}" for _, label in cols)
    print(header)
    for name, spec in results["methods"].items():
        met = spec["metrics"]
        row = f"{name:<11}{spec['role']:<13}" + "".join(
            f"{met[key]:>16.6f}" for key, _ in cols
        )
        print(row)
    print(f"\n{'method':<11}{'B_train':>12}{'B_extra':>12}{'B_pred':>12}{'total':>14}"
          f"{'per-test':>12}")
    for name, spec in results["methods"].items():
        led = spec["ledger"]
        print(f"{name:<11}{led['B_train']:>12}{led['B_extra']:>12}{led['B_pred']:>12}"
              f"{led['total']:>14}{led['amortized_per_test_item']:>12.1f}")
    label_summary = ", ".join(
        f"{method}={count}" for method, count in results["label_evals"].items()
    )
    print(f"\nexact labels ({label_summary}; logged separately)")
    alarm = results["surrogate_alarm"]
    state = "TRIGGERED" if alarm["triggered"] else "clear"
    print(f"surrogate alarm: {state} (ridge MAE {alarm['ridge_mae']:.6f} vs "
          f"feature-only {alarm['feature_only_mae']:.6f})")
