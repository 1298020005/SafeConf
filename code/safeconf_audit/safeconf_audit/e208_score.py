"""Frozen truth-free SafeConf-M score primitives for E208.

This module implements the score identity registered before Jiang24 test
outcomes are opened.  It deliberately accepts predictions, controls and
training-source effects only; there is no target-outcome argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


COMPONENT_NAMES = (
    "family_disagreement",
    "model_source_gap",
    "source_delta_dispersion",
    "negative_log_source_cells",
    "support_context_deficit",
)


class E208ScoreError(ValueError):
    """Raised when a registered E208 score contract is not satisfied."""


@dataclass(frozen=True)
class TaskFeatures:
    family_disagreement: float
    model_source_gap: float
    source_delta_dispersion: float
    negative_log_source_cells: float
    support_context_deficit: float
    predicted_magnitude: float

    def components(self) -> np.ndarray:
        return np.asarray(
            [
                self.family_disagreement,
                self.model_source_gap,
                self.source_delta_dispersion,
                self.negative_log_source_cells,
                self.support_context_deficit,
            ],
            dtype=np.float64,
        )


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise E208ScoreError(f"{name} must be a finite one-dimensional gene vector")
    return result


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    left = _finite_vector(left, "left")
    right = _finite_vector(right, "right")
    if left.shape != right.shape:
        raise E208ScoreError("RMSE vectors must share the registered gene axis")
    return float(np.sqrt(np.mean(np.square(left - right))))


def task_features(
    seed_predictions: np.ndarray,
    control_centroid: np.ndarray,
    source_deltas: np.ndarray,
    n_source_cells: int,
    *,
    total_other_contexts: int = 29,
) -> TaskFeatures:
    """Compute the six truth-free task features fixed for E208.

    ``seed_predictions`` is four LatentAdditive task-centroid predictions.
    ``source_deltas`` contains one train-only perturbation effect per source
    cell-state and is averaged across states, not across cells.
    """

    predictions = np.asarray(seed_predictions, dtype=np.float64)
    deltas = np.asarray(source_deltas, dtype=np.float64)
    control = _finite_vector(control_centroid, "control_centroid")
    if (
        predictions.ndim != 2
        or predictions.shape[0] != 4
        or predictions.shape[1:] != control.shape
        or not np.isfinite(predictions).all()
    ):
        raise E208ScoreError(
            "seed_predictions must contain four finite predictions on the control gene axis"
        )
    if (
        deltas.ndim != 2
        or deltas.shape[0] < 2
        or deltas.shape[1:] != control.shape
        or not np.isfinite(deltas).all()
    ):
        raise E208ScoreError(
            "source_deltas must contain at least two finite train-only context effects"
        )
    if isinstance(n_source_cells, bool) or int(n_source_cells) != n_source_cells:
        raise E208ScoreError("n_source_cells must be an integer")
    n_source_cells = int(n_source_cells)
    if n_source_cells < deltas.shape[0]:
        raise E208ScoreError("source-cell support cannot be smaller than source contexts")
    if total_other_contexts != 29 or deltas.shape[0] > total_other_contexts:
        raise E208ScoreError("E208 fixes 29 possible non-target source contexts")

    family = predictions.mean(axis=0)
    source_mean = deltas.mean(axis=0)
    disagreement = float(
        np.sqrt(np.mean(np.square(predictions - family[None, :])))
    )
    dispersion = float(
        np.sqrt(np.mean(np.square(deltas - source_mean[None, :])))
    )
    return TaskFeatures(
        family_disagreement=disagreement,
        model_source_gap=rmse(family, control + source_mean),
        source_delta_dispersion=dispersion,
        negative_log_source_cells=-math.log1p(n_source_cells),
        support_context_deficit=float(total_other_contexts - deltas.shape[0]),
        predicted_magnitude=rmse(family, control),
    )


def score_state_batch(
    components: np.ndarray,
    magnitude: np.ndarray,
) -> dict[str, np.ndarray]:
    """Standardize one registered state batch and compute S and 4:1 ranks."""

    values = np.asarray(components, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(COMPONENT_NAMES)
        or values.shape[0] < 10
        or magnitude.shape != (values.shape[0],)
        or not np.isfinite(values).all()
        or not np.isfinite(magnitude).all()
    ):
        raise E208ScoreError(
            "a state batch requires at least ten aligned finite tasks and five components"
        )
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise E208ScoreError("constant or invalid component requires state-level ABSTAIN")

    z_components = (values - center[None, :]) / scale[None, :]
    safeconf = z_components.mean(axis=1)
    magnitude_rank = rankdata(magnitude, method="average")
    safeconf_rank = rankdata(safeconf, method="average")
    n_tasks = float(values.shape[0])
    combined = (4.0 * magnitude_rank + safeconf_rank) / (5.0 * n_tasks)
    return {
        "component_center": center,
        "component_scale": scale,
        "z_components": z_components,
        "safeconf": safeconf,
        "magnitude_percentile_rank": magnitude_rank / n_tasks,
        "safeconf_percentile_rank": safeconf_rank / n_tasks,
        "safeconf_m_4to1": combined,
    }

