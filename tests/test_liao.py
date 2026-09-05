"""Contracts for the restricted-feature Liao-style ablation."""

from __future__ import annotations

import json

import numpy as np
import pytest

import qem_bench.baselines.liao as liao_module
from qem_bench.baselines.liao import (
    LiaoMLPMitigator,
    LiaoMitigator,
    LiaoRandomForestMitigator,
    LiaoValidationScore,
    MLP_BATCH_SIZE,
    MLP_HIDDEN_WIDTHS,
    MLP_LEARNING_RATE,
    OUR_DECLARATION,
    RANDOM_FOREST_TREES,
    UNVERIFIED_MLP_HYPERPARAMETERS,
    select_one_standard_error,
)
from qem_bench.budget import BudgetInputs, Method, method_budget
from qem_bench.datasets.generate import generate, group_shots
from qem_bench.datasets.schema import FEATURES
from qem_bench.validation import LEGACY_SCHEMA_VERSION


@pytest.fixture(scope="module")
def legacy_v1_items(tmp_path_factory) -> list[dict]:
    data_dir = tmp_path_factory.mktemp("liao-legacy-v1") / "data"
    manifest = generate("t0-micro", data_dir)
    assert manifest["dataset_schema_version"] == LEGACY_SCHEMA_VERSION
    return [
        json.loads(line)
        for line in (data_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    return (
        [item for item in items if item["split"] == "train"],
        [item for item in items if item["split"] == "test"],
    )


def _source_train_validation(items: list[dict]) -> tuple[list[dict], list[dict]]:
    train, _ = _split(items)
    groups = sorted({item["measurement_group"] for item in train})
    validation_groups = set(groups[-3:])
    return (
        [item for item in train if item["measurement_group"] not in validation_groups],
        [item for item in train if item["measurement_group"] in validation_groups],
    )


def test_reported_architectures_and_declared_unverified_defaults(legacy_v1_items):
    train, _ = _split(legacy_v1_items)
    forest = LiaoRandomForestMitigator(random_state=17).fit(train)
    estimators = forest.estimators_
    observable_keys = {
        (item["n_qubits"], item["pauli_label"]) for item in train
    }
    assert len(observable_keys) == 2
    assert set(estimators) == observable_keys
    trees = [
        tree for estimator in estimators.values() for tree in estimator.estimators_
    ]
    assert len(trees) == RANDOM_FOREST_TREES * len(observable_keys) == 200
    assert len({id(tree) for tree in trees}) == len(trees)
    for estimator in estimators.values():
        assert estimator.n_estimators == RANDOM_FOREST_TREES == 100
        assert estimator.criterion == "squared_error"
        assert estimator.min_samples_split == 2
        assert estimator.max_features == 1

    mlp = LiaoMLPMitigator(random_state=17).fit(train)
    config = mlp.config_
    assert config["hidden_layer_sizes"] == list(MLP_HIDDEN_WIDTHS) == [64, 64]
    assert config["activation"] == "relu"
    assert config["loss"] == "mean_squared_error"
    assert config["optimizer"] == "adam"
    assert config["learning_rate_init"] == MLP_LEARNING_RATE == 0.001
    assert config["batch_size"] == MLP_BATCH_SIZE == 32
    assert set(config["unverified_hyperparameters"]) == set(
        UNVERIFIED_MLP_HYPERPARAMETERS
    )
    assert {
        declaration["provenance"]
        for declaration in config["unverified_hyperparameters"].values()
    } == {OUR_DECLARATION}
    assert {
        name: declaration["value"]
        for name, declaration in config["unverified_hyperparameters"].items()
    } == {
        "epoch_count": 200,
        "stopping_rule": "fixed_epochs",
        "dropout": 0.0,
        "weight_decay": 0.0,
    }
    assert mlp.n_iter_ == 200
    assert mlp.parameters_["w1"].shape == (len(FEATURES), 64)
    assert mlp.parameters_["w2"].shape == (64, 64)
    assert mlp.parameters_["w3"].shape == (64, 1)


def test_both_arms_train_and_predict_on_generated_legacy_v1(legacy_v1_items):
    train, test = _split(legacy_v1_items)
    target = np.asarray([item["ideal_expectation"] for item in test])
    raw = np.asarray([item["noisy_expectation"] for item in test])
    raw_mae = float(np.mean(np.abs(raw - target)))

    predictions = {
        "random_forest": LiaoRandomForestMitigator(random_state=11)
        .fit(train)
        .predict(test),
        "mlp": LiaoMLPMitigator(random_state=11).fit(train).predict(test),
    }
    assert predictions.keys() == {"random_forest", "mlp"}
    for values in predictions.values():
        assert values.shape == target.shape
        assert np.all(np.isfinite(values))
        assert float(np.mean(np.abs(values - target))) >= 0.0
    assert raw_mae >= 0.0


def test_random_forest_refuses_unseen_physical_observable(legacy_v1_items):
    train, test = _split(legacy_v1_items)
    model = LiaoRandomForestMitigator(random_state=11).fit(train)
    unseen = dict(test[0])
    unseen["pauli_label"] = "I" * unseen["n_qubits"]

    combined = model.predict(test)
    separate = np.asarray([model.predict([item])[0] for item in test])
    assert np.array_equal(combined, separate)

    with pytest.raises(ValueError, match="unseen observable"):
        model.predict([unseen])


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"n_qubits": 3},
        {"pauli_label": "IZI"},
        {"n_qubits": True, "pauli_label": "Z"},
        {"n_qubits": 0, "pauli_label": ""},
        {"n_qubits": 3.0, "pauli_label": "IZI"},
        {"n_qubits": 3, "pauli_label": 1},
        {"n_qubits": 3, "pauli_label": "IZ"},
        {"n_qubits": 3, "pauli_label": "IXI"},
    ],
)
def test_physical_observable_key_fails_closed(item):
    first_alias = {"n_qubits": 3, "pauli_label": "IZI", "observable": "z_mid"}
    second_alias = {
        "n_qubits": 3,
        "pauli_label": "IZI",
        "observable": "alias",
        "observable_id": "other-alias",
    }
    assert liao_module._observable_key(first_alias) == liao_module._observable_key(
        second_alias
    ) == (3, "IZI")

    with pytest.raises(ValueError):
        liao_module._observable_key(item)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LiaoRandomForestMitigator(random_state=23),
        lambda: LiaoMLPMitigator(random_state=23),
    ],
    ids=["random-forest", "mlp"],
)
def test_same_seed_repeats_predictions_exactly(legacy_v1_items, factory):
    train, test = _split(legacy_v1_items)
    first = factory().fit(train).predict(test)
    second = factory().fit(train).predict(test)
    assert np.array_equal(first, second)


def test_unverified_mlp_choices_are_configurable(legacy_v1_items):
    train, _ = _source_train_validation(legacy_v1_items)
    model = LiaoMLPMitigator(
        random_state=5,
        epochs=7,
        stopping_rule="fixed_epochs",
        dropout=0.2,
        weight_decay=0.01,
    ).fit(train)
    declarations = model.config_["unverified_hyperparameters"]
    assert {name: spec["value"] for name, spec in declarations.items()} == {
        "epoch_count": 7,
        "stopping_rule": "fixed_epochs",
        "dropout": 0.2,
        "weight_decay": 0.01,
    }
    assert model.n_iter_ == 7

    fit_items, validation_items = _source_train_validation(legacy_v1_items)
    early = LiaoMLPMitigator(
        random_state=5,
        epochs=12,
        stopping_rule="validation_patience",
        patience=2,
    ).fit(fit_items, validation_items=validation_items)
    assert 1 <= early.n_iter_ <= 12
    assert len(early.validation_loss_curve_) == early.n_iter_


def test_one_standard_error_rule_uses_declared_tie_break_order():
    scores = [
        LiaoValidationScore("best", 0.10, 0.03, 2.0, 100, 0),
        LiaoValidationScore("lower-excess", 0.12, 0.01, 1.0, 200, 2),
        LiaoValidationScore("outside", 0.14, 0.01, -1.0, 1, 0),
    ]
    selected, threshold, eligible = select_one_standard_error(scores)
    assert threshold == pytest.approx(0.13)
    assert eligible == ("best", "lower-excess")
    assert selected == "lower-excess"

    equal_excess = [
        LiaoValidationScore("expensive", 0.10, 0.02, 1.0, 200, 0),
        LiaoValidationScore("cheap", 0.11, 0.01, 1.0, 100, 2),
    ]
    assert select_one_standard_error(equal_excess)[0] == "cheap"

    equal_cost = [
        LiaoValidationScore("complex", 0.10, 0.02, 1.0, 100, 2),
        LiaoValidationScore("simple", 0.11, 0.01, 1.0, 100, 0),
    ]
    assert select_one_standard_error(equal_cost)[0] == "simple"


def test_composite_selects_on_disjoint_source_validation(legacy_v1_items):
    train, validation = _source_train_validation(legacy_v1_items)
    model = LiaoMitigator(random_state=31, epochs=20).fit(train, validation)
    config = model.config_
    assert set(model.candidate_models_) == {"random_forest", "mlp"}
    assert config["selected_model"] in config["eligible_models"]
    assert config["selection_rule"] == "source-validation one-standard-error"
    assert config["tie_break_order"] == [
        "total_excess_absolute_loss",
        "circuit_evaluations",
        "simplicity_rank",
    ]
    assert model.predict(validation).shape == (len(validation),)


def test_realized_ledger_matches_method_budget(legacy_v1_items):
    train, test = _split(legacy_v1_items)
    shots = {item["shots"] for item in legacy_v1_items}
    assert len(shots) == 1
    shot_count = shots.pop()
    n_train_groups = len({item["measurement_group"] for item in train})
    n_test_groups = len({item["measurement_group"] for item in test})

    expected = method_budget(
        Method.LIAO,
        BudgetInputs(
            n_train_circuits=n_train_groups,
            n_test_circuits=n_test_groups,
            shots=shot_count,
            n_scales=1,
            m=0,
        ),
    )
    realized = (
        group_shots(legacy_v1_items, "train"),
        0,
        group_shots(legacy_v1_items, "test"),
    )
    assert realized == (expected.B_train, expected.B_extra, expected.B_pred)
    assert expected.total == sum(realized)


def test_random_forest_hyperparameters_match_the_paper():
    """Pin every RF hyperparameter Liao et al. state, to the value they state.

    Section IV.1.2 reads: "For RF, we used 100 tree estimators for each
    observable ... For each tree, at least 2 samples are required to split an
    internal node, and 1 feature is considered when looking for the best split."

    ``max_features`` is therefore the integer 1, one feature per split, and not
    the scikit-learn default of 1.0 meaning all features. The two differ
    sharply: on a 12-feature regression the integer form scored 1.65x worse.
    That gap is why this must be pinned. This test covers the reported model
    architecture within the restricted-feature ablation; it does not claim
    fidelity to the paper's feature encoding.

    The authors' public research code uses scikit-learn defaults with 200 or 300
    trees, which disagrees with their own paper. This restricted-feature
    ablation follows the paper only for these reported RF hyperparameters; it
    does not reproduce the published feature encoding. Any departure belongs in
    LiaoMLPDeclarations-style declared fields rather than in a silent constant.
    """
    from qem_bench.baselines.liao import LiaoRandomForestMitigator

    config = LiaoRandomForestMitigator(random_state=0).config_
    assert config["scope"] == "one-independent-forest-per-observable"
    assert config["observable_key_fields"] == ["n_qubits", "pauli_label"]
    assert config["n_estimators_per_observable"] == 100
    assert config["min_samples_split"] == 2
    assert config["criterion"] == "squared_error"
    assert type(config["max_features"]) is int, (
        f"max_features must be the integer 1 per the paper, got "
        f"{type(config['max_features']).__name__} {config['max_features']!r}"
    )
    assert config["max_features"] == 1


def test_validation_standard_error_blocks_physical_circuits_across_severity():
    items = []
    predictions = []
    for circuit_id, loss in (("circuit-a", 0.0), ("circuit-b", 2.0)):
        for severity in ("L1", "L2"):
            items.append(
                {
                    "circuit_id": circuit_id,
                    "measurement_group": f"{circuit_id}:{severity}",
                    "ideal_expectation": 0.0,
                    "noisy_expectation": 0.0,
                }
            )
            predictions.append(loss)

    score = liao_module._validation_score(
        "probe",
        items,
        np.asarray(predictions),
        circuit_evaluations=0,
        simplicity_rank=0,
    )

    assert score.validation_mae == pytest.approx(1.0)
    assert score.standard_error == pytest.approx(1.0)


# --- Capacity-matched feature-only control -----------------------------------
# The campaign compares the ablation against a counterpart that is the same
# candidate family under the same search and seeds, with the noisy-measurement
# column physically removed. These pin what "same" and "removed" have to mean.

NOISY = "noisy_expectation"


def _perturb_noisy(items: list[dict], delta: float = 0.25) -> list[dict]:
    return [dict(item, noisy_expectation=item["noisy_expectation"] + delta)
            for item in items]


def test_the_control_cannot_read_the_noisy_value_and_the_full_arm_can(legacy_v1_items):
    """The guarantee the control exists to provide.

    Moving only the noisy column at prediction time must leave the control's
    output identical and must move the full arm's. A zeroed column would satisfy
    neither side of this, and a control fitted on a different candidate family
    would answer a different question.
    """
    train, validation = _source_train_validation(legacy_v1_items)
    _, test = _split(legacy_v1_items)
    full = LiaoMitigator(random_state=17).fit(train, validation)
    control = LiaoMitigator(random_state=17, drop_features=(NOISY,)).fit(train, validation)

    moved = _perturb_noisy(test)
    control_before, control_after = control.predict(test), control.predict(moved)
    full_before, full_after = full.predict(test), full.predict(moved)

    assert np.array_equal(control_before, control_after)
    assert not np.allclose(full_before, full_after)


def test_the_control_deletes_the_column_rather_than_zeroing_it(legacy_v1_items):
    train, validation = _source_train_validation(legacy_v1_items)
    control = LiaoMitigator(random_state=17, drop_features=(NOISY,)).fit(train, validation)
    forest = control.candidate_models_["random_forest"]
    for estimator in forest.estimators_.values():
        assert estimator.n_features_in_ == len(FEATURES) - 1

    zeroed_train = [dict(item, noisy_expectation=0.0) for item in train]
    zeroed_validation = [dict(item, noisy_expectation=0.0) for item in validation]
    zeroed = LiaoMitigator(random_state=17).fit(zeroed_train, zeroed_validation)
    _, test = _split(legacy_v1_items)
    zeroed_test = [dict(item, noisy_expectation=0.0) for item in test]
    # Both withhold the measurement, but the zeroed model still spends a split
    # candidate and a weight on a constant column, so it is a different estimator.
    assert not np.allclose(control.predict(test), zeroed.predict(zeroed_test))


def test_the_control_keeps_the_same_candidates_search_and_seed(legacy_v1_items):
    train, validation = _source_train_validation(legacy_v1_items)
    full = LiaoMitigator(random_state=17).fit(train, validation)
    control = LiaoMitigator(random_state=17, drop_features=(NOISY,)).fit(train, validation)

    assert set(control.candidate_models_) == set(full.candidate_models_)
    assert len(control.validation_scores_) == len(full.validation_scores_)
    assert control.selected_model_name_ in set(full.candidate_models_)
    for name in control.candidate_models_:
        assert control.candidate_models_[name].random_state == 17
    forest_config = control.candidate_models_["random_forest"].config_
    assert forest_config["n_estimators_per_observable"] == RANDOM_FOREST_TREES
    assert forest_config["max_features"] == 1
    mlp_config = control.candidate_models_["mlp"].config_
    assert mlp_config["hidden_layer_sizes"] == list(MLP_HIDDEN_WIDTHS)
    assert mlp_config["batch_size"] == MLP_BATCH_SIZE
    assert mlp_config["learning_rate_init"] == MLP_LEARNING_RATE
    # Selection is the same rule, run independently on each pipeline's scores.
    assert control.config_["selection_rule"] == full.config_["selection_rule"]
    assert control.config_["tie_break_order"] == full.config_["tie_break_order"]


def test_every_config_declares_what_was_dropped(legacy_v1_items):
    train, validation = _source_train_validation(legacy_v1_items)
    control = LiaoMitigator(random_state=17, drop_features=(NOISY,)).fit(train, validation)
    full = LiaoMitigator(random_state=17).fit(train, validation)

    assert control.config_["dropped_features"] == [NOISY]
    assert full.config_["dropped_features"] == []
    for name in ("random_forest", "mlp"):
        assert control.config_["candidate_configs"][name]["dropped_features"] == [NOISY]
        assert full.config_["candidate_configs"][name]["dropped_features"] == []
    json.dumps(control.config_, allow_nan=False, default=str)


@pytest.mark.parametrize("names,message", [
    (("no_such_feature",), "unknown feature names"),
    ((NOISY, NOISY), "must not repeat"),
    (tuple(FEATURES), "at least one column"),
])
def test_an_unresolvable_mask_is_refused_at_construction(names, message):
    # A silently ignored name would leave a control still reading the input it
    # was built to withhold, and no later check could tell.
    with pytest.raises(ValueError, match=message):
        LiaoMitigator(random_state=0, drop_features=names)
    with pytest.raises(ValueError, match=message):
        LiaoRandomForestMitigator(random_state=0, drop_features=names)
    with pytest.raises(ValueError, match=message):
        LiaoMLPMitigator(random_state=0, drop_features=names)
