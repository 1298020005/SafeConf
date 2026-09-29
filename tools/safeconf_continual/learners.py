"""Low-capacity learners for the two-memory SafeConf lifecycle.

The public learner accepts only biological transfer targets. The shared core
accepts errors from multiple upstreams after fitting a training-only error CDF.
The residual adapter is explicitly tied to one immutable upstream version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .contracts import FrozenErrorCDF


LearnerKind = Literal["ridge", "hgb"]


class NumericPreprocessor:
    """Training-only imputation/standardisation with explicit missing flags."""

    def fit(self, values: np.ndarray) -> "NumericPreprocessor":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("numeric preprocessor needs a non-trivial matrix")
        self.medians_ = np.asarray([
            np.nanmedian(values[:, j]) if np.isfinite(values[:, j]).any() else 0.0
            for j in range(values.shape[1])
        ])
        filled = np.where(np.isfinite(values), values, self.medians_)
        self.center_ = filled.mean(axis=0)
        spread = filled.std(axis=0)
        self.scale_ = np.where(spread > 1e-8, spread, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("preprocessor is not fitted")
        values = np.asarray(values, dtype=float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.medians_, values)
        return np.c_[(filled - self.center_) / self.scale_, missing]


def _regressor(kind: LearnerKind, seed: int):
    if kind == "ridge":
        return Ridge(alpha=10.0)
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=10.0,
            random_state=seed,
        )
    raise ValueError(kind)


@dataclass
class PublicBiologyLearner:
    """Predicts held-out biological transfer error, never an upstream error."""

    kind: LearnerKind = "hgb"
    seed: int = 20260929

    def fit(self, pair_features: np.ndarray, transfer_rmse: np.ndarray) -> "PublicBiologyLearner":
        labels = np.asarray(transfer_rmse, dtype=float)
        if len(labels) != len(pair_features) or not np.isfinite(labels).all():
            raise ValueError("public learner requires finite biological transfer labels")
        self.preprocessor_ = NumericPreprocessor().fit(pair_features)
        self.model_ = _regressor(self.kind, self.seed).fit(
            self.preprocessor_.transform(pair_features), labels
        )
        return self

    def predict_transfer_error(self, pair_features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(self.preprocessor_.transform(pair_features)), dtype=float)

    def retrieval_weights(
        self,
        pair_features: np.ndarray,
        support_weights: np.ndarray | None = None,
        regularization: float = 0.0,
    ) -> np.ndarray:
        score = self.predict_transfer_error(pair_features)
        spread = max(float(np.std(score)), 1e-8)
        learned = np.exp(np.clip(-(score - score.min()) / spread, -20, 20))
        learned /= learned.sum()
        if support_weights is None or regularization == 0:
            return learned
        if not 0 <= regularization <= 1:
            raise ValueError("regularization must be in [0, 1]")
        support = np.asarray(support_weights, dtype=float)
        if len(support) != len(learned) or support.sum() <= 0:
            raise ValueError("support weights do not align")
        support /= support.sum()
        weights = (1 - regularization) * learned + regularization * support
        return weights / weights.sum()


@dataclass
class SharedRiskCore:
    """Cross-upstream risk model on a training-frozen error-rank scale."""

    kind: LearnerKind = "hgb"
    seed: int = 20260929

    def fit(self, features: np.ndarray, realised_errors: np.ndarray) -> "SharedRiskCore":
        self.error_cdf_ = FrozenErrorCDF.fit(realised_errors)
        labels = self.error_cdf_.transform(realised_errors)
        self.preprocessor_ = NumericPreprocessor().fit(features)
        self.model_ = _regressor(self.kind, self.seed).fit(
            self.preprocessor_.transform(features), labels
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(self.model_.predict(self.preprocessor_.transform(features)), dtype=float),
            0,
            1,
        )


@dataclass
class ErrorResidualAdapter:
    """Personalised correction for exactly one upstream checkpoint/version."""

    upstream_model_id: str
    model_version: str
    kind: LearnerKind = "hgb"
    kappa: float = 50.0
    seed: int = 20260929

    def fit(
        self,
        features: np.ndarray,
        shared_risk: np.ndarray,
        error_rank: np.ndarray,
        n_feedback_clusters: int,
    ) -> "ErrorResidualAdapter":
        if n_feedback_clusters < 1:
            raise ValueError("an error adapter needs feedback clusters")
        target = np.asarray(error_rank, dtype=float) - np.asarray(shared_risk, dtype=float)
        self.preprocessor_ = NumericPreprocessor().fit(features)
        self.model_ = _regressor(self.kind, self.seed).fit(
            self.preprocessor_.transform(features), target
        )
        self.n_feedback_clusters_ = int(n_feedback_clusters)
        return self

    @property
    def shrinkage(self) -> float:
        n = self.n_feedback_clusters_
        return float(n / (n + self.kappa))

    def predict(self, features: np.ndarray, shared_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correction = np.asarray(
            self.model_.predict(self.preprocessor_.transform(features)), dtype=float
        ) * self.shrinkage
        return correction, np.clip(np.asarray(shared_risk, dtype=float) + correction, 0, 1)
