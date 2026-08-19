from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionBatch:
    """Scalar record table plus high-dimensional prediction arrays."""

    records: pd.DataFrame
    predicted_effects: dict[str, np.ndarray]
    true_effects: dict[str, np.ndarray]


class PredictorAdapter(Protocol):
    """A black-box perturbation predictor used by confidence scoring."""

    name: str

    def fit(self, train_tasks: list[dict]) -> None:
        ...

    def predict(self, target_tasks: list[dict]) -> PredictionBatch:
        ...


REQUIRED_PREDICTION_COLUMNS = {
    "record_id",
    "dataset_name",
    "fold_id",
    "split",
    "context",
    "perturbation",
    "predictor_name",
    "predicted_effect_key",
    "true_effect_key",
    "true_error_rmse",
}


def validate_prediction_records(records: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(records.columns))
    issues: list[str] = []
    if missing:
        issues.append("missing_columns=" + ",".join(missing))
    if "record_id" in records.columns and records["record_id"].duplicated().any():
        issues.append("duplicate_record_id")
    if "true_error_rmse" in records.columns:
        err = pd.to_numeric(records["true_error_rmse"], errors="coerce")
        if err.isna().any():
            issues.append("true_error_rmse_has_nan")
    return issues

