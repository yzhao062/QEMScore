"""tools/neural_es_summary.py on synthetic fit records."""

import json

import pytest

from tools import neural_es_summary as nes


def _write(d, s, rung, k, *, es, val, test, epochs=100, best=60, arm="C"):
    d.mkdir(parents=True, exist_ok=True)
    meta = {"rung": rung, "arm": arm, "learner_seed": k, "neural_es": es,
            "candidate_validation_family_mae": {"mlp": {"tfi": val, "heisenberg": 2 * val}},
            "candidate_test_family_mae": {"mlp": {"tfi": test, "heisenberg": 2 * test}},
            "selected_model": "mlp", "epochs_run": epochs if es else None, "best_epoch": best if es else None}
    (d / f"shipped-s{s}-n640__{rung}__k{k:02d}__{arm}.json").write_text(json.dumps(meta))


def test_summary_ratios_and_epochs(tmp_path):
    new, orig = tmp_path / "new", tmp_path / "orig"
    for k in (1, 2):
        _write(new, 101, "R0", k, es=True, val=0.01 * k, test=0.02, epochs=100 + k, best=50 + k)
        _write(orig, 101, "R0", k, es=False, val=0.02 * k, test=0.04)
    res = nes.summarize(new, orig, ["R0"], [101], [1, 2])
    cell = res["cells"]["shipped-s101-n640/tfi/R0"]
    assert cell["validation_mlp_mean"]["ratio"] == pytest.approx(0.5)
    assert cell["test_mlp_mean"]["ratio"] == pytest.approx(0.5)
    assert cell["epochs_run"] == {"min": 101, "max": 102, "mean": 101.5}
    assert cell["selected_mlp_in_C"] == 2
    assert len(res["inputs_sha256"]) == 4


def test_missing_or_mislabelled_records_are_refused(tmp_path):
    new, orig = tmp_path / "new", tmp_path / "orig"
    _write(new, 101, "R0", 1, es=True, val=0.01, test=0.02)
    with pytest.raises(SystemExit, match="missing"):
        nes.summarize(new, orig, ["R0"], [101], [1])
    _write(orig, 101, "R0", 1, es=True, val=0.02, test=0.04)
    with pytest.raises(SystemExit, match="must not be"):
        nes.summarize(new, orig, ["R0"], [101], [1])
