"""Data contracts shared by the continual SafeConf components.

The contracts deliberately separate model-independent biological memory from
model-specific realised errors.  They contain no training implementation so
that ingestion, learning, and evaluation can be audited independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Iterable

import numpy as np


def _token(*values: object) -> str:
    return sha256("\0".join(map(str, values)).encode("utf-8")).hexdigest()


def biological_task_key(
    study_id: str,
    context: str,
    perturbation_type: str,
    perturbation_target: str,
    condition: str,
    effect_contract_id: str,
) -> str:
    """Identity of one biological truth shared by every upstream prediction."""
    return _token(
        "biological-task-v1",
        study_id,
        context,
        perturbation_type,
        perturbation_target,
        condition,
        effect_contract_id,
    )


def perturbation_cluster_key(
    study_id: str, perturbation_type: str, perturbation_target: str
) -> str:
    """Strict outer-fold/bootstrap unit; contexts stay in the same cluster."""
    return _token(
        "perturbation-cluster-v1",
        study_id,
        perturbation_type,
        perturbation_target,
    )


@dataclass(frozen=True)
class PublicMemoryItem:
    experiment_id: str
    study_id: str
    context: str
    perturbation_type: str
    perturbation_target: str
    condition: str
    effect_vector_row: int
    effect_contract_id: str
    control_source: str
    gene_space_id: str
    n_cells: int
    n_guides: int | None
    n_plates: int | None
    n_batches: int | None
    guide_reproducibility: float | None
    plate_reproducibility: float | None
    split_half_stability: float | None
    batch_agreement: float | None
    provenance: str
    eligibility: bool
    timestamp: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorMemoryItem:
    upstream_model_id: str
    model_version: str
    prediction_id: str
    biological_task_key: str
    perturbation_cluster_key: str
    shared_risk: float
    realised_error: float
    error_rank: float
    shared_risk_residual: float
    feedback_timestamp: str
    provenance: str
    is_oof_or_heldout: bool

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskOutput:
    shared_risk: float
    error_correction: float
    final_risk: float
    public_memory_version: str
    shared_core_version: str
    error_adapter_version: str | None
    history_support: int
    feedback_count: int
    status: str

    def __post_init__(self) -> None:
        for value in (self.shared_risk, self.final_risk):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("risk scores must be finite values in [0, 1]")


@dataclass(frozen=True)
class FrozenErrorCDF:
    """Empirical error CDF fitted exclusively from a training partition."""

    sorted_training_errors: tuple[float, ...]

    @classmethod
    def fit(cls, errors: Iterable[float]) -> "FrozenErrorCDF":
        values = np.asarray(list(errors), dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("error CDF needs at least two finite training errors")
        return cls(tuple(map(float, np.sort(values))))

    def transform(self, errors: Iterable[float]) -> np.ndarray:
        reference = np.asarray(self.sorted_training_errors, dtype=float)
        values = np.asarray(list(errors), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("cannot rank non-finite errors")
        # Mid-rank empirical CDF, frozen after fit.
        left = np.searchsorted(reference, values, side="left")
        right = np.searchsorted(reference, values, side="right")
        return np.clip((left + right) / (2.0 * len(reference)), 0.0, 1.0)
