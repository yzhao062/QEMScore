"""Contracts for the optional feature-builder hook of the Liao-style arms.

The hook lets a caller hand the random forest, the MLP, and the composite
``LiaoMitigator`` a different descriptor vector without forking the fitting
path. Two guarantees matter: leaving it unset keeps the campaign's behaviour
byte for byte, and setting it changes what every candidate reads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qemscore.baselines.liao import (
    LiaoMLPMitigator,
    LiaoMitigator,
    LiaoRandomForestMitigator,
)
from qemscore.datasets.generate import generate
from qemscore.datasets.schema import FEATURES, build_features
from qemscore.validation import LEGACY_SCHEMA_VERSION

NOISY = "noisy_expectation"


@pytest.fixture(scope="module")
def legacy_v1_items(tmp_path_factory) -> list[dict]:
    data_dir = tmp_path_factory.mktemp("liao-builder-legacy-v1") / "data"
    manifest = generate("t0-micro", data_dir)
    assert manifest["dataset_schema_version"] == LEGACY_SCHEMA_VERSION
    return [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _roles(items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train = [item for item in items if item["split"] == "train"]
    test = [item for item in items if item["split"] == "test"]
    groups = sorted({item["measurement_group"] for item in train})
    validation_groups = set(groups[-3:])
    return (
        [item for item in train if item["measurement_group"] not in validation_groups],
        [item for item in train if item["measurement_group"] in validation_groups],
        test,
    )


def _without_labels(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "ideal_expectation"} for item in items]


SUBSET_NAMES = ("noisy_expectation", "log2_shots", "n_qubits", "obs_locality", "probe")


class _SubsetBuilder:
    """A small custom builder that also counts its calls."""

    name = "subset-probe"

    def __init__(self, probe: float = 0.0) -> None:
        self.calls = 0
        self.probe = probe

    def __call__(self, item: dict) -> list[float]:
        self.calls += 1
        full = dict(zip(FEATURES, build_features(item)))
        return [full["noisy_expectation"], full["log2_shots"], full["n_qubits"],
                full["obs_locality"], self.probe * float(item["instance"])]


@pytest.mark.parametrize("drop", [(), (NOISY,)])
def test_default_builder_through_the_hook_is_bit_identical(legacy_v1_items, drop):
    """Passing build_features explicitly reproduces the default path exactly."""
    train, validation, test = _roles(legacy_v1_items)
    default = LiaoMitigator(random_state=23, drop_features=drop).fit(train, validation)
    hooked = LiaoMitigator(
        random_state=23, drop_features=drop,
        feature_builder=build_features, feature_names=FEATURES,
    ).fit(train, validation)

    assert hooked.selected_model_name_ == default.selected_model_name_
    assert hooked.validation_scores_ == default.validation_scores_
    for name in ("random_forest", "mlp"):
        left = default.candidate_models_[name].predict(test)
        right = hooked.candidate_models_[name].predict(test)
        assert np.array_equal(left, right)
    assert np.array_equal(default.predict(test), hooked.predict(test))


def test_unset_hook_keeps_the_configuration_unchanged(legacy_v1_items):
    train, validation, _ = _roles(legacy_v1_items)
    default = LiaoMitigator(random_state=5).fit(train, validation)
    assert default.feature_builder is None
    assert default.feature_names == tuple(FEATURES)
    config = default.config_
    assert "feature_builder" not in config and "feature_names" not in config
    for name in ("random_forest", "mlp"):
        candidate = config["candidate_configs"][name]
        assert "feature_builder" not in candidate and "feature_names" not in candidate


def test_custom_builder_is_used_by_every_candidate(legacy_v1_items):
    train, validation, test = _roles(legacy_v1_items)
    builder = _SubsetBuilder(probe=0.5)
    model = LiaoMitigator(
        random_state=7, feature_builder=builder, feature_names=SUBSET_NAMES,
    ).fit(train, validation)

    forest = model.candidate_models_["random_forest"]
    for estimator in forest.estimators_.values():
        assert estimator.n_features_in_ == len(SUBSET_NAMES)
    mlp = model.candidate_models_["mlp"]
    assert mlp.parameters_["w1"].shape == (len(SUBSET_NAMES), 64)
    assert builder.calls > 0

    calls_before = builder.calls
    predictions = model.predict(_without_labels(test))
    assert builder.calls == calls_before + len(test)
    assert predictions.shape == (len(test),)
    assert np.all(np.isfinite(predictions))

    config = model.config_
    assert config["feature_builder"] == "subset-probe"
    assert config["feature_names"] == list(SUBSET_NAMES)
    for name in ("random_forest", "mlp"):
        assert config["candidate_configs"][name]["feature_names"] == list(SUBSET_NAMES)
    json.dumps(config, allow_nan=False)


def test_custom_builder_changes_what_the_model_reads(legacy_v1_items):
    train, validation, test = _roles(legacy_v1_items)
    plain = LiaoMLPMitigator(
        random_state=3, feature_builder=_SubsetBuilder(probe=0.0),
        feature_names=SUBSET_NAMES).fit(train)
    probed = LiaoMLPMitigator(
        random_state=3, feature_builder=_SubsetBuilder(probe=1.0),
        feature_names=SUBSET_NAMES).fit(train)
    assert not np.allclose(plain.predict(test), probed.predict(test))
    default = LiaoMLPMitigator(random_state=3).fit(train)
    assert not np.allclose(plain.predict(test), default.predict(test))


def test_drop_features_refers_to_the_builder_names(legacy_v1_items):
    train, validation, test = _roles(legacy_v1_items)
    control = LiaoMitigator(
        random_state=11, drop_features=("probe", NOISY),
        feature_builder=_SubsetBuilder(probe=1.0), feature_names=SUBSET_NAMES,
    ).fit(train, validation)
    for estimator in control.candidate_models_["random_forest"].estimators_.values():
        assert estimator.n_features_in_ == len(SUBSET_NAMES) - 2
    moved = [dict(item, noisy_expectation=item["noisy_expectation"] + 0.25)
             for item in test]
    assert np.array_equal(control.predict(test), control.predict(moved))

    # A schema name that the builder does not emit is unknown here.
    for factory in (LiaoMitigator, LiaoRandomForestMitigator, LiaoMLPMitigator):
        with pytest.raises(ValueError, match="unknown feature names"):
            factory(random_state=0, drop_features=("j",),
                    feature_builder=_SubsetBuilder(), feature_names=SUBSET_NAMES)
        with pytest.raises(ValueError, match="at least one column"):
            factory(random_state=0, drop_features=SUBSET_NAMES,
                    feature_builder=_SubsetBuilder(), feature_names=SUBSET_NAMES)


@pytest.mark.parametrize("builder,names,error,message", [
    (None, SUBSET_NAMES, ValueError, "requires a feature_builder"),
    (_SubsetBuilder(), None, ValueError, "requires a sequence of feature_names"),
    (_SubsetBuilder(), "probe", ValueError, "requires a sequence of feature_names"),
    (_SubsetBuilder(), (), ValueError, "must not be empty"),
    (_SubsetBuilder(), ("probe", "probe"), ValueError, "must not repeat"),
    ("not-callable", SUBSET_NAMES, TypeError, "callable"),
])
def test_an_inconsistent_hook_is_refused_at_construction(builder, names, error, message):
    for factory in (LiaoMitigator, LiaoRandomForestMitigator, LiaoMLPMitigator):
        with pytest.raises(error, match=message):
            factory(random_state=0, feature_builder=builder, feature_names=names)


def test_a_builder_that_disagrees_with_its_names_fails_closed(legacy_v1_items):
    train, validation, _ = _roles(legacy_v1_items)
    too_few = SUBSET_NAMES + ("extra",)
    with pytest.raises(ValueError, match="feature_names declare"):
        LiaoRandomForestMitigator(
            random_state=0, feature_builder=_SubsetBuilder(), feature_names=too_few,
        ).fit(train)
    with pytest.raises(ValueError, match="feature_names declare"):
        LiaoMitigator(
            random_state=0, feature_builder=_SubsetBuilder(), feature_names=too_few,
        ).fit(train, validation)


def test_a_lookup_builder_works_on_label_free_prediction_rows(legacy_v1_items):
    """A rung builder keyed by item_id must not need the label or the circuit id."""
    train, validation, test = _roles(legacy_v1_items)
    table = {str(item["item_id"]): float(item["instance"]) * 0.1
             for item in train + validation + test}

    def lookup(item: dict) -> list[float]:
        return [*build_features(item), table[str(item["item_id"])]]

    names = (*FEATURES, "lookup")
    model = LiaoMitigator(random_state=2, feature_builder=lookup,
                          feature_names=names).fit(train, validation)
    rows = [{k: v for k, v in item.items() if k not in ("ideal_expectation",)}
            for item in test]
    assert np.all(np.isfinite(model.predict(rows)))
