"""Surrogate and leakage controls (frozen design, required family).

These controls decide whether a learned mitigator uses the noisy measurement or
merely emulates the simulator from circuit structure:

- feature-only: full features minus the noisy expectation. Costs zero circuit
  evaluations at train and test time; strong accuracy here means the model class is
  a classical surrogate of the data-generating process, not a mitigator.
- noisy-value-only: the noisy expectation (plus shots) with no circuit features.
- shrinkage: predict the training mean label for the family; a zero-cost floor.
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
    """Predict the training mean for each family at zero measurement cost.

    For a family absent from training (S3), use the pooled source training mean.
    Its family mean cannot be learned without target labels.
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._family_means: dict[str, float] = {}

    def fit(self, train_items: list[dict]) -> "ShrinkageControl":
        if not train_items:
            raise ValueError("train_items must not be empty")
        self._mean = float(np.mean([it["ideal_expectation"] for it in train_items]))
        self._family_means = {
            family: float(np.mean([
                it["ideal_expectation"] for it in train_items if it["family"] == family
            ]))
            for family in {it["family"] for it in train_items}
        }
        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit first")
        return np.asarray([
            self._family_means.get(it["family"], self._mean) for it in items
        ], dtype=float)


def shuffle_noisy_items(items: list[dict], *, seed: int) -> list[dict]:
    """Copy rows and permute only the noisy value, for any learned method.

    The caller chooses the intervention: training shuffle followed by refitting,
    or evaluation shuffle with the fitted model held fixed. These are different
    diagnostics and must be labeled separately. Item order and labels are kept.
    """
    shuffled = [dict(item) for item in items]
    column = np.asarray([item["noisy_expectation"] for item in items], dtype=float)
    np.random.default_rng(seed).shuffle(column)
    for item, value in zip(shuffled, column, strict=True):
        item["noisy_expectation"] = float(value)
    return shuffled


class ShuffledNoisyControl(RidgeMitigator):
    """Full features with the training noisy-expectation column permuted.

    Diagnostic only: it consumes the same measurements as the primary ridge, and its
    score is interpreted, never deployed.
    """

    def __init__(self, cv: int = 5, random_state: int = 0, shuffle_seed: int = 1234) -> None:
        super().__init__(cv=cv, random_state=random_state)
        self.shuffle_seed = shuffle_seed

    def fit(self, train_items: list[dict]) -> "RidgeMitigator":
        return super().fit(shuffle_noisy_items(train_items, seed=self.shuffle_seed))
