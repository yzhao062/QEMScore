"""Surrogate and leakage controls (frozen design, required family).

These controls decide whether a learned mitigator uses the noisy measurement or
merely emulates the simulator from circuit structure:

- feature-only: full features minus the noisy expectation. Costs zero circuit
  evaluations at train and test time; strong accuracy here means the model class is
  a classical surrogate of the data-generating process, not a mitigator.
- noisy-value-only: the noisy expectation (plus shots) with no circuit features.
- shrinkage: predict the train-split mean label; a zero-cost floor.
- shuffled-noisy (diagnostic): full features with the training noisy-expectation
  column permuted. Accuracy that survives the shuffle exposes surrogate behavior.

Each control shares the primary ridge's circuit-grouped selection machinery.
"""

from __future__ import annotations

import numpy as np

from qem_bench.baselines.ridge import RidgeMitigator
from qem_bench.datasets.schema import FEATURES, build_features

_NOISY_IDX = FEATURES.index("noisy_expectation")
_SHOTS_IDX = FEATURES.index("log2_shots")


class _ColumnSubsetRidge(RidgeMitigator):
    """Ridge over a fixed subset of the versioned feature vector."""

    columns: tuple[int, ...] = ()

    def _matrix(self, items: list[dict]) -> np.ndarray:
        full = np.array([build_features(it) for it in items])
        return full[:, list(self.columns)]


class FeatureOnlyControl(_ColumnSubsetRidge):
    """All features except the noisy expectation. Zero circuit-evaluation cost."""

    columns = tuple(i for i in range(len(FEATURES)) if i != _NOISY_IDX)


class NoisyOnlyControl(_ColumnSubsetRidge):
    """Only the noisy expectation and the shot count."""

    columns = (_NOISY_IDX, _SHOTS_IDX)


class ShrinkageControl:
    """Predict the train-split mean label. Zero circuit-evaluation cost."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, train_items: list[dict]) -> "ShrinkageControl":
        self._mean = float(np.mean([it["ideal_expectation"] for it in train_items]))
        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit first")
        return np.full(len(items), self._mean)


class ShuffledNoisyControl(RidgeMitigator):
    """Full features with the training noisy-expectation column permuted.

    Diagnostic only: it consumes the same measurements as the primary ridge, and its
    score is interpreted, never deployed.
    """

    def __init__(self, cv: int = 5, random_state: int = 0, shuffle_seed: int = 1234) -> None:
        super().__init__(cv=cv, random_state=random_state)
        self.shuffle_seed = shuffle_seed

    def fit(self, train_items: list[dict]) -> "RidgeMitigator":
        rng = np.random.default_rng(self.shuffle_seed)
        shuffled = [dict(it) for it in train_items]
        column = np.array([it["noisy_expectation"] for it in shuffled])
        rng.shuffle(column)
        for it, value in zip(shuffled, column):
            it["noisy_expectation"] = float(value)
        return super().fit(shuffled)
