"""Restricted-feature Liao-style ablation; not a reproduction of the
published Liao et al. feature encoding and not the headline competitor.

The architecture reported by Liao et al. fixes the random forest at 100 CART
trees and the companion network at two width-64 ReLU hidden layers trained with
MSE and Adam at learning rate 0.001 with batches of 32. The paper does not state
the network epoch count, stopping rule, dropout, or weight decay. Their defaults
below are qem-bench declarations, not reproduced settings from Liao et al. They
remain constructor arguments so a run artifact can record any predeclared change.

Both arms consume the versioned feature vector and already measured noisy
expectations. They never execute a circuit. The runner must charge the unique
source and test measurement groups once, even when it trains both arms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from qem_bench.datasets.schema import build_features

RANDOM_FOREST_NAME = "random_forest"
MLP_NAME = "mlp"

RANDOM_FOREST_TREES = 100
MLP_HIDDEN_WIDTHS = (64, 64)
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 32

DEFAULT_MLP_EPOCHS = 200
DEFAULT_MLP_STOPPING_RULE = "fixed_epochs"
DEFAULT_MLP_DROPOUT = 0.0
DEFAULT_MLP_WEIGHT_DECAY = 0.0

OUR_DECLARATION = "qem-bench declaration; not reported by Liao et al."
UNVERIFIED_MLP_HYPERPARAMETERS = (
    "epoch_count",
    "stopping_rule",
    "dropout",
    "weight_decay",
)

_STOPPING_RULES = frozenset({"fixed_epochs", "validation_patience"})
_ADAM_BETA_1 = 0.9
_ADAM_BETA_2 = 0.999
_ADAM_EPSILON = 1e-8

Item = Mapping[str, object]


def _items(items: Sequence[Item], label: str) -> list[Item]:
    rows = list(items)
    if not rows:
        raise ValueError(f"{label} must not be empty")
    return rows


def _matrix(items: Sequence[Item], label: str) -> np.ndarray:
    rows = _items(items, label)
    matrix = np.asarray([build_features(dict(item)) for item in rows], dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} features must be a finite matrix")
    return matrix


def _targets(items: Sequence[Item], label: str) -> np.ndarray:
    rows = _items(items, label)
    try:
        targets = np.asarray([item["ideal_expectation"] for item in rows], dtype=float)
    except KeyError as exc:
        raise ValueError(f"{label} items require ideal_expectation") from exc
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError(f"{label} targets must be finite")
    return targets


def _group_cost(items: Sequence[Item]) -> int:
    shots_by_group: dict[str, int] = {}
    for item in items:
        try:
            group = str(item["measurement_group"])
            shots = int(item["shots"])
        except KeyError as exc:
            raise ValueError("items require measurement_group and shots") from exc
        if not group:
            raise ValueError("measurement_group must not be empty")
        if shots <= 0 or isinstance(item["shots"], bool) or shots != item["shots"]:
            raise ValueError("shots must be a positive integer")
        previous = shots_by_group.setdefault(group, shots)
        if previous != shots:
            raise ValueError(f"measurement group {group} has conflicting shots values")
    return sum(shots_by_group.values())


def _require_disjoint_groups(
    train_items: Sequence[Item], validation_items: Sequence[Item]
) -> None:
    train_groups = {str(item["measurement_group"]) for item in train_items}
    validation_groups = {str(item["measurement_group"]) for item in validation_items}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError("train and validation measurement groups must be disjoint")


class LiaoRandomForestMitigator:
    """Reported B7 RF architecture applied to the restricted feature vector.

    Liao et al. additionally state squared-error reduction, at least two samples
    to split an internal node, and one candidate feature per split. These are
    represented directly by the scikit-learn estimator configuration.
    """

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self._model: RandomForestRegressor | None = None

    def fit(self, train_items: Sequence[Item]) -> "LiaoRandomForestMitigator":
        x = _matrix(train_items, "train_items")
        y = _targets(train_items, "train_items")
        self._model = RandomForestRegressor(
            n_estimators=RANDOM_FOREST_TREES,
            criterion="squared_error",
            min_samples_split=2,
            # Integer 1, not float 1.0. Liao et al. Section IV.1.2 states
            # "1 feature is considered when looking for the best split",
            # so one feature per split matches this reported RF hyperparameter,
            # unlike the scikit-learn default.
            max_features=1,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(x, y)
        return self

    @property
    def estimator_(self) -> RandomForestRegressor:
        if self._model is None:
            raise RuntimeError("fit first")
        return self._model

    @property
    def config_(self) -> dict[str, object]:
        return {
            "model": RANDOM_FOREST_NAME,
            "n_estimators": RANDOM_FOREST_TREES,
            "tree_algorithm": "CART",
            "criterion": "squared_error",
            "min_samples_split": 2,
            "max_features": 1,
            "random_state": self.random_state,
        }

    def predict(self, items: Sequence[Item]) -> np.ndarray:
        return np.asarray(self.estimator_.predict(_matrix(items, "items")), dtype=float)


@dataclass(frozen=True)
class LiaoMLPDeclarations:
    """Configurable qem-bench declarations for details absent from the paper."""

    epoch_count: int = DEFAULT_MLP_EPOCHS
    stopping_rule: str = DEFAULT_MLP_STOPPING_RULE
    dropout: float = DEFAULT_MLP_DROPOUT
    weight_decay: float = DEFAULT_MLP_WEIGHT_DECAY

    def with_provenance(self) -> dict[str, dict[str, object]]:
        return {
            name: {"value": value, "provenance": OUR_DECLARATION}
            for name, value in asdict(self).items()
        }


class LiaoMLPMitigator:
    """Two-hidden-layer dense ReLU network trained by deterministic Adam.

    A small NumPy implementation keeps dropout configurable without adding a
    framework dependency. Features are standardized using training rows only.
    ``fixed_epochs`` always runs ``epochs`` passes. ``validation_patience``
    requires validation rows and restores the parameters with the lowest
    validation MSE.
    """

    def __init__(
        self,
        random_state: int = 0,
        *,
        epochs: int = DEFAULT_MLP_EPOCHS,
        stopping_rule: str = DEFAULT_MLP_STOPPING_RULE,
        dropout: float = DEFAULT_MLP_DROPOUT,
        weight_decay: float = DEFAULT_MLP_WEIGHT_DECAY,
        patience: int = 20,
        min_delta: float = 0.0,
    ) -> None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if stopping_rule not in _STOPPING_RULES:
            raise ValueError(f"stopping_rule must be one of {sorted(_STOPPING_RULES)}")
        if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not np.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        if not np.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("min_delta must be finite and nonnegative")

        self.random_state = random_state
        self.epochs = epochs
        self.stopping_rule = stopping_rule
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.patience = patience
        self.min_delta = float(min_delta)

        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []
        self.n_iter_: int = 0
        self._parameters: dict[str, np.ndarray] | None = None

    @property
    def declarations(self) -> LiaoMLPDeclarations:
        return LiaoMLPDeclarations(
            epoch_count=self.epochs,
            stopping_rule=self.stopping_rule,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
        )

    @property
    def config_(self) -> dict[str, object]:
        return {
            "model": MLP_NAME,
            "hidden_layer_sizes": list(MLP_HIDDEN_WIDTHS),
            "activation": "relu",
            "loss": "mean_squared_error",
            "optimizer": "adam",
            "learning_rate_init": MLP_LEARNING_RATE,
            "batch_size": MLP_BATCH_SIZE,
            "random_state": self.random_state,
            "preprocessing": "training-only feature standardization",
            "unverified_hyperparameters": self.declarations.with_provenance(),
            "patience": self.patience,
            "min_delta": self.min_delta,
            "n_iter": self.n_iter_,
        }

    def _initialize(self, n_features: int, rng: np.random.Generator) -> None:
        widths = (n_features, *MLP_HIDDEN_WIDTHS, 1)
        self._parameters = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / widths[0]), (widths[0], widths[1])),
            "b1": np.zeros((1, widths[1]), dtype=float),
            "w2": rng.normal(0.0, np.sqrt(2.0 / widths[1]), (widths[1], widths[2])),
            "b2": np.zeros((1, widths[2]), dtype=float),
            "w3": rng.normal(0.0, np.sqrt(1.0 / widths[2]), (widths[2], widths[3])),
            "b3": np.zeros((1, widths[3]), dtype=float),
        }

    @property
    def parameters_(self) -> dict[str, np.ndarray]:
        if self._parameters is None:
            raise RuntimeError("fit first")
        return self._parameters

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("fit first")
        return (x - self.feature_mean_) / self.feature_scale_

    def _forward(
        self, x: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        parameters = self.parameters_
        z1 = x @ parameters["w1"] + parameters["b1"]
        a1 = np.maximum(z1, 0.0)
        mask1 = np.ones_like(a1)
        if rng is not None and self.dropout:
            mask1 = (rng.random(a1.shape) >= self.dropout) / (1.0 - self.dropout)
        dropped1 = a1 * mask1

        z2 = dropped1 @ parameters["w2"] + parameters["b2"]
        a2 = np.maximum(z2, 0.0)
        mask2 = np.ones_like(a2)
        if rng is not None and self.dropout:
            mask2 = (rng.random(a2.shape) >= self.dropout) / (1.0 - self.dropout)
        dropped2 = a2 * mask2
        output = dropped2 @ parameters["w3"] + parameters["b3"]
        return output, (z1, mask1, dropped1, z2, mask2, dropped2)

    def _gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        output: np.ndarray,
        cache: tuple[np.ndarray, ...],
    ) -> dict[str, np.ndarray]:
        z1, mask1, dropped1, z2, mask2, dropped2 = cache
        parameters = self.parameters_
        output_gradient = 2.0 * (output - y) / len(x)
        gradients: dict[str, np.ndarray] = {
            "w3": dropped2.T @ output_gradient + self.weight_decay * parameters["w3"],
            "b3": np.sum(output_gradient, axis=0, keepdims=True),
        }
        hidden2 = (output_gradient @ parameters["w3"].T) * mask2 * (z2 > 0.0)
        gradients["w2"] = dropped1.T @ hidden2 + self.weight_decay * parameters["w2"]
        gradients["b2"] = np.sum(hidden2, axis=0, keepdims=True)
        hidden1 = (hidden2 @ parameters["w2"].T) * mask1 * (z1 > 0.0)
        gradients["w1"] = x.T @ hidden1 + self.weight_decay * parameters["w1"]
        gradients["b1"] = np.sum(hidden1, axis=0, keepdims=True)
        return gradients

    def _mse(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions, _ = self._forward(x)
        return float(np.mean(np.square(predictions - y)))

    def fit(
        self,
        train_items: Sequence[Item],
        *,
        validation_items: Sequence[Item] | None = None,
    ) -> "LiaoMLPMitigator":
        train_rows = _items(train_items, "train_items")
        x = _matrix(train_rows, "train_items")
        y = _targets(train_rows, "train_items").reshape(-1, 1)

        validation_x: np.ndarray | None = None
        validation_y: np.ndarray | None = None
        if self.stopping_rule == "validation_patience":
            if validation_items is None:
                raise ValueError("validation_patience requires validation_items")
            validation_rows = _items(validation_items, "validation_items")
            _require_disjoint_groups(train_rows, validation_rows)
            validation_x = _matrix(validation_rows, "validation_items")
            validation_y = _targets(validation_rows, "validation_items").reshape(-1, 1)

        self.feature_mean_ = np.mean(x, axis=0)
        self.feature_scale_ = np.std(x, axis=0)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0
        x = self._standardize(x)
        if validation_x is not None:
            validation_x = self._standardize(validation_x)

        rng = np.random.default_rng(self.random_state)
        self._initialize(x.shape[1], rng)
        first_moment = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        self.n_iter_ = 0
        adam_step = 0
        best_validation_loss = np.inf
        best_parameters: dict[str, np.ndarray] | None = None
        stale_epochs = 0

        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), MLP_BATCH_SIZE):
                indices = order[start : start + MLP_BATCH_SIZE]
                batch_x = x[indices]
                batch_y = y[indices]
                output, cache = self._forward(batch_x, rng)
                gradients = self._gradients(batch_x, batch_y, output, cache)
                adam_step += 1
                for name, gradient in gradients.items():
                    first_moment[name] = (
                        _ADAM_BETA_1 * first_moment[name]
                        + (1.0 - _ADAM_BETA_1) * gradient
                    )
                    second_moment[name] = (
                        _ADAM_BETA_2 * second_moment[name]
                        + (1.0 - _ADAM_BETA_2) * np.square(gradient)
                    )
                    corrected_first = first_moment[name] / (1.0 - _ADAM_BETA_1**adam_step)
                    corrected_second = second_moment[name] / (
                        1.0 - _ADAM_BETA_2**adam_step
                    )
                    self.parameters_[name] -= MLP_LEARNING_RATE * corrected_first / (
                        np.sqrt(corrected_second) + _ADAM_EPSILON
                    )

            train_loss = self._mse(x, y)
            if not np.isfinite(train_loss):
                raise RuntimeError("MLP training produced a nonfinite loss")
            self.loss_curve_.append(train_loss)
            self.n_iter_ = epoch + 1

            if validation_x is not None and validation_y is not None:
                validation_loss = self._mse(validation_x, validation_y)
                self.validation_loss_curve_.append(validation_loss)
                if validation_loss < best_validation_loss - self.min_delta:
                    best_validation_loss = validation_loss
                    best_parameters = {
                        name: value.copy() for name, value in self.parameters_.items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break

        if best_parameters is not None:
            self._parameters = best_parameters
        return self

    def predict(self, items: Sequence[Item]) -> np.ndarray:
        x = self._standardize(_matrix(items, "items"))
        predictions, _ = self._forward(x)
        return np.asarray(predictions[:, 0], dtype=float)


@dataclass(frozen=True)
class LiaoValidationScore:
    """Source-validation quantities used by the frozen B7 selection rule."""

    name: str
    validation_mae: float
    standard_error: float
    total_excess_absolute_loss: float
    circuit_evaluations: int
    simplicity_rank: int


def select_one_standard_error(
    scores: Sequence[LiaoValidationScore],
) -> tuple[str, float, tuple[str, ...]]:
    """Apply the one-standard-error rule and the declared tie-break order."""

    candidates = list(scores)
    if not candidates:
        raise ValueError("scores must not be empty")
    if len({score.name for score in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    for score in candidates:
        numeric = (
            score.validation_mae,
            score.standard_error,
            score.total_excess_absolute_loss,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("validation scores must be finite")
        if score.validation_mae < 0.0 or score.standard_error < 0.0:
            raise ValueError("validation error and standard error must be nonnegative")
        if score.circuit_evaluations < 0 or score.simplicity_rank < 0:
            raise ValueError("cost and simplicity rank must be nonnegative")

    best = min(candidates, key=lambda score: (score.validation_mae, score.name))
    threshold = best.validation_mae + best.standard_error
    eligible = [score for score in candidates if score.validation_mae <= threshold]
    selected = min(
        eligible,
        key=lambda score: (
            score.total_excess_absolute_loss,
            score.circuit_evaluations,
            score.simplicity_rank,
            score.name,
        ),
    )
    return selected.name, float(threshold), tuple(sorted(score.name for score in eligible))


def _validation_score(
    name: str,
    validation_items: Sequence[Item],
    predictions: np.ndarray,
    *,
    circuit_evaluations: int,
    simplicity_rank: int,
) -> LiaoValidationScore:
    rows = _items(validation_items, "validation_items")
    if predictions.shape != (len(rows),) or not np.all(np.isfinite(predictions)):
        raise ValueError("validation predictions must be a finite vector aligned to rows")
    targets = _targets(rows, "validation_items")
    raw = np.asarray([item["noisy_expectation"] for item in rows], dtype=float)
    losses = np.abs(predictions - targets)
    raw_losses = np.abs(raw - targets)

    losses_by_group: dict[str, list[float]] = {}
    for item, loss in zip(rows, losses):
        circuit_id = item.get("circuit_id", item.get("measurement_group"))
        if not isinstance(circuit_id, str) or not circuit_id:
            raise ValueError(
                "validation items require circuit_id or measurement_group"
            )
        losses_by_group.setdefault(circuit_id, []).append(float(loss))
    block_losses = np.asarray(
        [np.mean(losses_by_group[group]) for group in sorted(losses_by_group)], dtype=float
    )
    standard_error = (
        float(np.std(block_losses, ddof=1) / np.sqrt(len(block_losses)))
        if len(block_losses) > 1
        else 0.0
    )
    return LiaoValidationScore(
        name=name,
        validation_mae=float(np.mean(block_losses)),
        standard_error=standard_error,
        total_excess_absolute_loss=float(np.sum(losses - raw_losses)),
        circuit_evaluations=circuit_evaluations,
        simplicity_rank=simplicity_rank,
    )


class LiaoMitigator:
    """Validation-selected restricted-feature RF and MLP arms."""

    def __init__(
        self,
        random_state: int = 0,
        *,
        epochs: int = DEFAULT_MLP_EPOCHS,
        stopping_rule: str = DEFAULT_MLP_STOPPING_RULE,
        dropout: float = DEFAULT_MLP_DROPOUT,
        weight_decay: float = DEFAULT_MLP_WEIGHT_DECAY,
        patience: int = 20,
        min_delta: float = 0.0,
    ) -> None:
        self.random_state = random_state
        self._mlp_arguments = {
            "epochs": epochs,
            "stopping_rule": stopping_rule,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "patience": patience,
            "min_delta": min_delta,
        }
        self.candidate_models_: dict[str, object] = {}
        self.validation_scores_: tuple[LiaoValidationScore, ...] = ()
        self.one_standard_error_threshold_: float | None = None
        self.eligible_models_: tuple[str, ...] = ()
        self.selected_model_name_: str | None = None

    def fit(
        self,
        train_items: Sequence[Item],
        validation_items: Sequence[Item],
    ) -> "LiaoMitigator":
        train_rows = _items(train_items, "train_items")
        validation_rows = _items(validation_items, "validation_items")
        _require_disjoint_groups(train_rows, validation_rows)

        random_forest = LiaoRandomForestMitigator(self.random_state).fit(train_rows)
        mlp = LiaoMLPMitigator(self.random_state, **self._mlp_arguments)
        if mlp.stopping_rule == "validation_patience":
            mlp.fit(train_rows, validation_items=validation_rows)
        else:
            mlp.fit(train_rows)
        self.candidate_models_ = {
            RANDOM_FOREST_NAME: random_forest,
            MLP_NAME: mlp,
        }

        shared_cost = _group_cost([*train_rows, *validation_rows])
        self.validation_scores_ = (
            _validation_score(
                RANDOM_FOREST_NAME,
                validation_rows,
                random_forest.predict(validation_rows),
                circuit_evaluations=shared_cost,
                simplicity_rank=0,
            ),
            _validation_score(
                MLP_NAME,
                validation_rows,
                mlp.predict(validation_rows),
                circuit_evaluations=shared_cost,
                simplicity_rank=1,
            ),
        )
        (
            self.selected_model_name_,
            self.one_standard_error_threshold_,
            self.eligible_models_,
        ) = select_one_standard_error(self.validation_scores_)
        return self

    @property
    def selected_model_(self) -> object:
        if self.selected_model_name_ is None:
            raise RuntimeError("fit first")
        return self.candidate_models_[self.selected_model_name_]

    @property
    def config_(self) -> dict[str, object]:
        if self.selected_model_name_ is None:
            raise RuntimeError("fit first")
        return {
            "selected_model": self.selected_model_name_,
            "selection_rule": "source-validation one-standard-error",
            "one_standard_error_threshold": self.one_standard_error_threshold_,
            "eligible_models": list(self.eligible_models_),
            "tie_break_order": [
                "total_excess_absolute_loss",
                "circuit_evaluations",
                "simplicity_rank",
            ],
            "validation_scores": [asdict(score) for score in self.validation_scores_],
            "candidate_configs": {
                RANDOM_FOREST_NAME: self.candidate_models_[RANDOM_FOREST_NAME].config_,
                MLP_NAME: self.candidate_models_[MLP_NAME].config_,
            },
        }

    def predict(self, items: Sequence[Item]) -> np.ndarray:
        return self.selected_model_.predict(items)


__all__ = [
    "DEFAULT_MLP_DROPOUT",
    "DEFAULT_MLP_EPOCHS",
    "DEFAULT_MLP_STOPPING_RULE",
    "DEFAULT_MLP_WEIGHT_DECAY",
    "LiaoMLPDeclarations",
    "LiaoMLPMitigator",
    "LiaoMitigator",
    "LiaoRandomForestMitigator",
    "LiaoValidationScore",
    "MLP_BATCH_SIZE",
    "MLP_HIDDEN_WIDTHS",
    "MLP_LEARNING_RATE",
    "OUR_DECLARATION",
    "RANDOM_FOREST_TREES",
    "UNVERIFIED_MLP_HYPERPARAMETERS",
    "select_one_standard_error",
]
