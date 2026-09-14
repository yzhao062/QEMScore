"""Regression tests for the surrogate-control family and the surrogate alarm.

These guards are mutation-sensitive on purpose: a regression that turns a
column-subset control into a full-feature ridge copy, mutates caller data, breaks a
control ledger, or silences the alarm must fail here even when everything else
stays green."""

import copy
import json

import numpy as np
import pytest

from qemscore.baselines.controls import (
    FeatureOnlyControl,
    NoisyOnlyControl,
    ShuffledNoisyControl,
    ShrinkageControl,
    shuffle_noisy_items,
)
from qemscore.baselines.ridge import RidgeMitigator
from qemscore.datasets.generate import generate
from qemscore.runner.run import run


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
    from qemscore.datasets.schema import FEATURES

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


def _expected_shuffled_column(items, seed):
    """Independently derived permutation, computed without the production helper.

    Calling shuffle_noisy_items to build the oracle would compare the helper with
    itself, so this reimplements the documented contract directly: the seed selects
    a NumPy default_rng permutation of the whole noisy column.
    """
    column = np.asarray([item["noisy_expectation"] for item in items], dtype=float)
    np.random.default_rng(seed).shuffle(column)
    return [float(value) for value in column]


def test_shuffle_seed_selects_the_permutation_over_the_whole_column(micro_run):
    """The seed must choose the permutation, and it must permute every position.

    A driver requests twenty seeds for this diagnostic, so a hardcoded seed would
    report twenty identical repeats as twenty samples. Requiring only that two of
    several seeds differ would still admit a seed folded onto two values, and
    comparing against the helper itself would admit a shuffle applied to half the
    column, so every seed is checked against an independent full permutation.
    """
    items, _ = micro_run
    train = [item for item in items if item["split"] == "train"]
    assert len({item["noisy_expectation"] for item in train}) > 3

    def column(seed):
        return [item["noisy_expectation"] for item in shuffle_noisy_items(train, seed=seed)]

    assert column(1234) == column(1234)
    seeds = list(range(8))
    for seed in seeds:
        assert column(seed) == _expected_shuffled_column(train, seed)
    # Distinct seeds must give distinct permutations, not two alternating ones.
    assert len({tuple(column(seed)) for seed in seeds}) >= len(seeds) - 1


@pytest.mark.parametrize("shuffle_seed", [7, 1234, 20260904])
def test_the_shuffled_diagnostic_fits_on_its_own_seeds_permuted_measurements(
    micro_run, shuffle_seed
):
    """Preserving the caller's rows is not evidence that the fit saw shuffled ones.

    Dropping the shuffle keeps every other assertion in this file true, and fixing
    the seed inside fit would pass any single-seed check, so the diagnostic is
    compared against a plain ridge fitted on an independently permuted training
    set at the same seed, and against one fitted on the original measurements.
    """
    items, _ = micro_run
    train = [item for item in items if item["split"] == "train"]
    test = [item for item in items if item["split"] == "test"]

    reference_rows = [dict(item) for item in train]
    for item, value in zip(
        reference_rows, _expected_shuffled_column(train, shuffle_seed), strict=True
    ):
        item["noisy_expectation"] = value

    diagnostic = ShuffledNoisyControl(shuffle_seed=shuffle_seed).fit(train).predict(test)
    on_shuffled = RidgeMitigator().fit(reference_rows).predict(test)
    on_original = RidgeMitigator().fit(train).predict(test)

    assert np.allclose(diagnostic, on_shuffled)
    assert not np.allclose(diagnostic, on_original)


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


def test_shrinkage_uses_source_family_means_and_source_only_unseen_family_floor():
    train = [{"family": "tfi", "ideal_expectation": 0.2},
             {"family": "tfi", "ideal_expectation": 0.6},
             {"family": "heisenberg", "ideal_expectation": -0.8}]
    model = ShrinkageControl().fit(train)
    for role in ("train", "validation", "test"):
        rows = [{"family": family, "split": role, "ideal_expectation": 999}
                for family in ("tfi", "heisenberg", "qaoa")]
        assert model.predict(rows) == pytest.approx([0.4, -0.8, 0.0])
    model.fit([{"family": "qaoa", "ideal_expectation": 0.7}])
    assert model.predict([{"family": "tfi"}]) == pytest.approx([0.7])


def test_shrinkage_requires_nonempty_fit():
    with pytest.raises(RuntimeError, match="fit first"):
        ShrinkageControl().predict([{"family": "tfi"}])
    with pytest.raises(ValueError, match="must not be empty"):
        ShrinkageControl().fit([])


def test_generic_shuffle_changes_only_noisy_values_and_supports_liao(micro_run):
    from qemscore.baselines.liao import LiaoRandomForestMitigator

    items, _ = micro_run
    train = [item for item in items if item["split"] == "train"]
    test = [item for item in items if item["split"] == "test"]
    snapshot = copy.deepcopy(test)
    shuffled = shuffle_noisy_items(test, seed=1234)
    assert test == snapshot
    assert sorted(item["noisy_expectation"] for item in shuffled) == sorted(
        item["noisy_expectation"] for item in test)
    assert [item["noisy_expectation"] for item in shuffled] != [
        item["noisy_expectation"] for item in test]
    for before, after in zip(test, shuffled):
        assert {k: v for k, v in before.items() if k != "noisy_expectation"} == {
            k: v for k, v in after.items() if k != "noisy_expectation"}
    model = LiaoRandomForestMitigator().fit(train)
    assert not np.allclose(model.predict(test), model.predict(shuffled))
