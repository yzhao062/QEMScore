"""T0 smoke tests: determinism, end-to-end run, ledger conservation."""

import json

from qem_bench.datasets.generate import generate
from qem_bench.runner.run import run


def test_generate_is_deterministic(tmp_path):
    m1 = generate("t0-micro", tmp_path / "a")
    m2 = generate("t0-micro", tmp_path / "b")
    assert m1["dataset_hash"] == m2["dataset_hash"]
    lines_a = (tmp_path / "a" / "items.jsonl").read_text().splitlines()
    lines_b = (tmp_path / "b" / "items.jsonl").read_text().splitlines()
    assert lines_a == lines_b


def test_master_seed_changes_hash(tmp_path):
    m1 = generate("t0-micro", tmp_path / "a")
    m2 = generate("t0-micro", tmp_path / "b", master_seed=8)
    assert m1["dataset_hash"] != m2["dataset_hash"]


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "results"
    manifest = generate("t0-micro", data)
    results = run(data, out)

    assert (out / "results.json").exists()
    assert results["dataset_hash"] == manifest["dataset_hash"]

    raw = results["methods"]["raw"]["metrics"]
    ridge = results["methods"]["ridge"]["metrics"]
    for metrics in (raw, ridge):
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"] - 1e-12

    # Raw's excess loss over raw is identically zero by definition.
    assert raw["excess_loss_total"] == 0.0
    assert raw["overcorrection_rate"] == 0.0

    # All five methods and controls are present with a role tag.
    for name in ("raw", "ridge", "feat-only", "noisy-only", "shrinkage", "shuf-noisy"):
        assert name in results["methods"]
        assert results["methods"][name]["role"] in ("baseline", "learned", "control", "diagnostic")

    # Ledger conservation (measurement-group contract): circuit evaluations are
    # charged once per unique measurement group, never once per observable row.
    items = [
        json.loads(line) for line in (data / "items.jsonl").read_text().splitlines() if line
    ]
    train_groups = {it["measurement_group"]: it["shots"] for it in items if it["split"] == "train"}
    test_groups = {it["measurement_group"]: it["shots"] for it in items if it["split"] == "test"}
    train_evals = sum(train_groups.values())
    test_evals = sum(test_groups.values())

    ridge_ledger = results["methods"]["ridge"]["ledger"]
    assert ridge_ledger["B_train"] == train_evals
    assert ridge_ledger["B_extra"] == 0
    assert ridge_ledger["B_pred"] == test_evals
    assert ridge_ledger["total"] == train_evals + test_evals
    assert ridge_ledger["amortized_per_test_item"] == ridge_ledger["total"] / results["n_test_items"]

    raw_ledger = results["methods"]["raw"]["ledger"]
    assert raw_ledger["B_train"] == 0
    assert raw_ledger["B_pred"] == test_evals

    # Zero-measurement controls cost exactly zero circuit evaluations.
    assert results["methods"]["feat-only"]["ledger"]["total"] == 0
    assert results["methods"]["shrinkage"]["ledger"]["total"] == 0

    # Manifest generation ledger agrees with the rows, and exact labels stay separate.
    assert manifest["generation_ledger"]["train_circuit_evals"] == train_evals
    assert manifest["generation_ledger"]["test_circuit_evals"] == test_evals
    assert manifest["generation_ledger"]["label_evals_statevector"] == len(items)
    assert results["label_evals_statevector"] == len(items)

    # The surrogate alarm is always reported.
    assert "surrogate_alarm" in results
    assert isinstance(results["surrogate_alarm"]["triggered"], bool)
