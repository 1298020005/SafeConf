"""Leakage-aware score channels and supervised fusion for SafeConf.

This module deliberately keeps rule-based public scores separate from
supervised fusion.  A channel may be converted to a comparable percentile
using training-side values without reading an error label.  The Ridge fusion
class, in contrast, requires an explicitly registered error-label source and
records that source in its fitted metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class TrainOnlyQuantileScale:
    """Empirical percentile transform fitted only on a training partition."""

    values: np.ndarray
    name: str

    @classmethod
    def fit(cls, values: np.ndarray, name: str) -> "TrainOnlyQuantileScale":
        x = np.asarray(values, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            raise ValueError(f"cannot fit score scale for {name}: no finite values")
        return cls(np.sort(x), name)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float).reshape(-1)
        out = np.full(len(x), np.nan, dtype=float)
        finite = np.isfinite(x)
        if len(self.values) == 1:
            out[finite] = 0.5
            return out
        # Mid-rank-like empirical CDF.  Values outside the fit range are
        # clipped to the endpoints rather than fitted using query labels.
        ranks = np.searchsorted(self.values, x[finite], side="right")
        out[finite] = np.clip(ranks / len(self.values), 0.0, 1.0)
        return out

    def audit(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "fit_rows": int(len(self.values)),
            "fit_min": float(self.values[0]),
            "fit_max": float(self.values[-1]),
            "fit_hash": __import__("hashlib").sha256(
                self.values.tobytes()
            ).hexdigest(),
            "scale_kind": "training_only_empirical_cdf",
        }


@dataclass
class TrainOnlyStandardizer:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def fit(cls, x: np.ndarray, names: list[str]) -> "TrainOnlyStandardizer":
        a = np.asarray(x, dtype=float)
        if a.ndim != 2 or len(a) < 2:
            raise ValueError("fusion standardizer needs a two-dimensional training matrix")
        med = np.array([
            np.nanmedian(a[:, j]) if np.isfinite(a[:, j]).any() else 0.0
            for j in range(a.shape[1])
        ])
        filled = np.where(np.isfinite(a), a, med)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        return cls(med, mean, np.where(std > 1e-12, std, 1.0), tuple(names))

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=float)
        missing = ~np.isfinite(a)
        filled = np.where(missing, self.medians, a)
        z = (filled - self.means) / self.scales
        return np.concatenate([z, missing.astype(float)], axis=1)


@dataclass
class OOFReadyRidgeFusion:
    """Ridge stacker whose input channels must already be fold-out-of-fold.

    ``label_source`` is mandatory so a score cannot silently become a
    zero-target-error method.  This class does not generate OOF predictions;
    the caller must generate them with the upstream risk model and pass them
    as ``base_scores_are_oof=True``.
    """

    alpha: float = 10.0
    label_source: str = "UNREGISTERED"
    target_error_budget: str = "none"

    def fit(
        self,
        channels: np.ndarray,
        labels: np.ndarray,
        channel_names: list[str],
        *,
        base_scores_are_oof: bool,
        sample_weight: np.ndarray | None = None,
    ) -> "OOFReadyRidgeFusion":
        if not base_scores_are_oof:
            raise ValueError("fusion inputs must be fold-out-of-fold predictions")
        x = np.asarray(channels, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("fusion channels and labels have incompatible sizes")
        if not self.label_source or self.label_source == "UNREGISTERED":
            raise ValueError("register the error-label source before fitting fusion")
        self.channel_names_ = tuple(channel_names)
        self.preprocessor_ = TrainOnlyStandardizer.fit(x, channel_names)
        self.model_ = Ridge(alpha=float(self.alpha), fit_intercept=True)
        self.model_.fit(
            self.preprocessor_.transform(x), y,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=float),
        )
        self.n_fit_ = len(y)
        self.label_source_ = self.label_source
        self.target_error_budget_ = self.target_error_budget
        return self

    def predict(self, channels: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fusion is not fitted")
        x = np.asarray(channels, dtype=float)
        return np.asarray(self.model_.predict(self.preprocessor_.transform(x)), dtype=float)

    def audit(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fusion is not fitted")
        return {
            "kind": "Ridge",
            "alpha": float(self.alpha),
            "channels": list(self.channel_names_),
            "n_fit": int(self.n_fit_),
            "label_source": self.label_source_,
            "target_error_budget": self.target_error_budget_,
            "base_scores_are_oof": True,
            "standardizer_has_missing_indicators": True,
            "coefficients": self.model_.coef_.tolist(),
            "intercept": float(self.model_.intercept_),
        }


def rule_score(amplitude: np.ndarray, public: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the no-error-label unified public rule and availability flags."""
    a = np.asarray(amplitude, dtype=float)
    p = np.asarray(public, dtype=float)
    has_public = np.isfinite(p)
    score = np.where(has_public, p, a)
    if not np.isfinite(score).all():
        raise ValueError("rule score has no usable amplitude/public evidence")
    return score, has_public.astype(int)


def build_fusion_channels(
    amplitude_q: np.ndarray,
    public_q: np.ndarray,
    shared_q: np.ndarray | None,
    target_q: np.ndarray | None,
    *,
    has_public: np.ndarray | None = None,
    has_shared: np.ndarray | None = None,
    has_target: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Construct one common feature layout without turning missing into zero."""
    n = len(amplitude_q)
    def channel(value, name):
        if value is None:
            return np.full(n, np.nan), np.zeros(n, dtype=int), name
        v = np.asarray(value, dtype=float).reshape(-1)
        if len(v) != n:
            raise ValueError(f"channel {name} has incompatible length")
        return v, np.isfinite(v).astype(int), name

    a = np.asarray(amplitude_q, dtype=float).reshape(-1)
    if len(a) != n:
        raise ValueError("amplitude channel has incompatible length")
    p, p_default, _ = channel(public_q, "public_q")
    s, s_default, _ = channel(shared_q, "shared_q")
    t, t_default, _ = channel(target_q, "target_q")
    hp = p_default if has_public is None else np.asarray(has_public, dtype=int)
    hs = s_default if has_shared is None else np.asarray(has_shared, dtype=int)
    ht = t_default if has_target is None else np.asarray(has_target, dtype=int)
    values = np.column_stack([a, p, s, t, hp, hs, ht])
    names = ["amplitude_q", "public_q", "shared_q", "target_q",
             "has_public", "has_shared", "has_target"]
    return values, names


def score_task(task: Mapping[str, Any], frozen_config: Mapping[str, Any]) -> dict[str, Any]:
    """The single dispatch point used by rule and supervised candidates.

    ``frozen_config`` is an already selected development-side object.  This
    function never fits a model, reads truth, or changes channel weights.
    """
    mode = str(frozen_config.get("mode", ""))
    if mode == "rule":
        score, has_public = rule_score(
            np.array([task["amplitude"]], dtype=float),
            np.array([task.get("public", np.nan)], dtype=float),
        )
        return {"risk_score": float(score[0]), "has_public": int(has_public[0]),
                "evidence_status": "public" if has_public[0] else "amplitude_only",
                "version": frozen_config.get("version", "rule_unversioned")}
    if mode == "ridge":
        model = frozen_config.get("model")
        if model is None:
            raise ValueError("ridge score config has no fitted OOF-registered model")
        channels = np.asarray(task["channels"], dtype=float).reshape(1, -1)
        score = float(model.predict(channels)[0])
        return {"risk_score": score, "has_public": int(task.get("has_public", 0)),
                "evidence_status": str(task.get("evidence_status", "supervised")),
                "version": frozen_config.get("version", "ridge_unversioned")}
    raise ValueError(f"unknown frozen score mode: {mode}")
