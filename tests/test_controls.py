"""Regression tests for the surrogate-control family and the surrogate alarm.

These guards are mutation-sensitive on purpose: a regression that turns a
column-subset control into a full-feature ridge copy, mutates caller data, breaks a
control ledger, or silences the alarm must fail here even when everything else
stays green."""

import copy
import json

import numpy as np
import pytest

from qem_bench.baselines.controls import (
    FeatureOnlyControl,
    NoisyOnlyControl,
    ShuffledNoisyControl,
)
from qem_bench.datasets.generate import generate
from qem_bench.runner.run import run


@pytest.fixture(scope="module")
def micro_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("controls")
    data = root / "data"
    generate("t0-micro", data)
    items = [
        json.loads(line) for line in (data / "items.jsonl").read_text().splitlines() if line
    ]
    results = run(data, root / "out")
    return items, results


def _perturbed(items, field, delta):
    out = [dict(it) for it in items]
    for it in out:
        it[field] = it[field] + delta
    return out


def test_control_columns_are_exactly_the_stated_contract():
    """Freeze the control family: noisy-only is exactly (noisy_expectation,
    log2_shots); feature-only is exactly every versioned feature except
    noisy_expectation. A partial-column mutation must fail here."""
    from qem_bench.datasets.schema import FEATURES

    noisy_names = tuple(FEATURES[i] for i in NoisyOnlyControl.columns)
    assert noisy_names == ("noisy_expectation", "log2_shots")

    feature_names = tuple(FEATURES[i] for i in FeatureOnlyControl.columns)
    assert feature_names == tuple(f for f in FEATURES if f != "noisy_expectation")


def test_feature_only_ignores_the_noisy_value(micro_run):
    items, _ = micro_run
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    model = FeatureOnlyControl().fit(train)
    base = model.predict(test)
    moved = model.predict(_perturbed(test, "noisy_expectation", 0.5))
    assert np.allclose(base, moved)


def test_noisy_only_ignores_structure(micro_run):
    items, _ = micro_run
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    model = NoisyOnlyControl().fit(train)
    base = model.predict(test)
    moved = model.predict(_perturbed(test, "j", 10.0))
    assert np.allclose(base, moved)
    shifted = model.predict(_perturbed(test, "noisy_expectation", 0.5))
    assert not np.allclose(base, shifted)


def test_shuffled_noisy_does_not_mutate_inputs(micro_run):
    items, _ = micro_run
    train = [it for it in items if it["split"] == "train"]
    snapshot = copy.deepcopy(train)
    ShuffledNoisyControl().fit(train)
    assert train == snapshot


def test_control_ledgers(micro_run):
    _, results = micro_run
    ridge_ledger = results["methods"]["ridge"]["ledger"]
    for name in ("feat-only", "shrinkage"):
        ledger = results["methods"][name]["ledger"]
        assert ledger["B_train"] == 0
        assert ledger["B_extra"] == 0
        assert ledger["B_pred"] == 0
        assert ledger["total"] == 0
    for name in ("noisy-only", "shuf-noisy"):
        assert results["methods"][name]["ledger"] == ridge_ledger
    assert results["methods"]["shuf-noisy"]["role"] == "diagnostic"


def test_surrogate_alarm_rule_and_state(micro_run):
    _, results = micro_run
    alarm = results["surrogate_alarm"]
    ridge_mae = results["methods"]["ridge"]["metrics"]["mae"]
    feat_mae = results["methods"]["feat-only"]["metrics"]["mae"]
    assert alarm["ridge_mae"] == ridge_mae
    assert alarm["feature_only_mae"] == feat_mae
    assert alarm["triggered"] == (feat_mae <= ridge_mae * 1.05)
    # On the deterministic micro fixture the alarm fires: the feature-only control
    # matches the full ridge, so the ridge score is a plumbing checksum here.
    assert alarm["triggered"] is True
