"""Ridge mitigator: the first learned baseline of the walking skeleton.

Selection touches training data only: the alpha grid is frozen here, and
cross-validated selection runs inside fit() with circuit-grouped folds, so
observable rows from one circuit never straddle a train/validation boundary
(frozen protocol rule). Test labels never reach this class.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qemscore.datasets.schema import build_features

ALPHA_GRID = [0.1, 1.0, 10.0]


class RidgeMitigator:
    def __init__(self, cv: int = 5, random_state: int = 0) -> None:
        self.cv = cv
        self.random_state = random_state
        self._search: GridSearchCV | None = None

    def _matrix(self, items: list[dict]) -> np.ndarray:
        """Model-input matrix; subclasses may restrict to a feature subset."""
        return np.array([build_features(it) for it in items])

    def fit(self, train_items: list[dict]) -> "RidgeMitigator":
        x = self._matrix(train_items)
        y = np.array([it["ideal_expectation"] for it in train_items])
        groups = np.array([f"{it['family']}:{it['instance']}" for it in train_items])
        if np.unique(groups).size < self.cv:
            raise ValueError(f"cv={self.cv} requires at least {self.cv} distinct circuits")
        pipeline = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(random_state=self.random_state))]
        )
        self._search = GridSearchCV(
            pipeline,
            param_grid={"ridge__alpha": ALPHA_GRID},
            cv=GroupKFold(n_splits=self.cv),
            scoring="neg_mean_absolute_error",
        )
        self._search.fit(x, y, groups=groups)
        return self

    @property
    def best_alpha_(self) -> float:
        if self._search is None:
            raise RuntimeError("fit first")
        return float(self._search.best_params_["ridge__alpha"])

    def predict(self, items: list[dict]) -> np.ndarray:
        if self._search is None:
            raise RuntimeError("fit first")
        return self._search.predict(self._matrix(items))
