from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safetrans_confidence.data.records import validate_prediction_record_contract
from safetrans_confidence.predictors.base import PredictionBatch, validate_prediction_records


class PrecomputedGEARSAdapter:
    """Read per-prediction GEARS outputs for confidence scoring.

    This adapter deliberately refuses perturbation-level metric summaries.
    Confidence scoring needs one row per prediction with predicted and true
    effect vectors.  Using already evaluated MSE/metric tables as a confidence
    score would leak labels and is therefore rejected.
    """

    name = "GEARS"

    def __init__(self, records_csv: Path, predicted_npz: Path, true_npz: Path):
        self.records_csv = Path(records_csv)
        self.predicted_npz = Path(predicted_npz)
        self.true_npz = Path(true_npz)

    def fit(self, train_tasks: list[dict]) -> None:
        return None

    def predict(self, target_tasks: list[dict] | None = None) -> PredictionBatch:
        if not self.records_csv.exists():
            raise FileNotFoundError(f"GEARS prediction records not found: {self.records_csv}")
        if not self.predicted_npz.exists() or not self.true_npz.exists():
            raise FileNotFoundError("GEARS predicted/true effect NPZ files are required")
        records = pd.read_csv(self.records_csv)
        issues = validate_prediction_records(records)
        if issues:
            raise ValueError("Invalid GEARS prediction records: " + ";".join(issues))
        predicted = dict(np.load(self.predicted_npz))
        true = dict(np.load(self.true_npz))
        contract_issues = validate_prediction_record_contract(
            records,
            strict=False,
            predicted_effects=predicted,
            true_effects=true,
        )
        if contract_issues:
            raise ValueError("Invalid GEARS PredictionRecord contract: " + ";".join(contract_issues))
        return PredictionBatch(records=records, predicted_effects=predicted, true_effects=true)


def audit_gears_files(path: Path) -> pd.DataFrame:
    """Classify GEARS files as usable or metric-only for confidence scoring."""
    path = Path(path)
    rows: list[dict] = []
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        name = file.name.lower()
        usable_kind = "unknown"
        if name.endswith(".npz") and ("pred" in name or "true" in name or "effect" in name):
            usable_kind = "candidate_array"
        elif name.endswith(".csv"):
            try:
                cols = list(pd.read_csv(file, nrows=1).columns)
            except Exception:
                cols = []
            if {"predicted_effect_key", "true_effect_key"}.issubset(cols):
                usable_kind = "candidate_prediction_records"
            elif any(c.lower().endswith("mse") or c.lower() == "mse" for c in cols):
                usable_kind = "metric_only_not_usable"
            else:
                usable_kind = "csv_unknown"
            rows.append(
                {
                    "file_path": str(file),
                    "file_name": file.name,
                    "usable_kind": usable_kind,
                    "columns": "|".join(cols[:40]),
                }
            )
            continue
        rows.append(
            {
                "file_path": str(file),
                "file_name": file.name,
                "usable_kind": usable_kind,
                "columns": "",
            }
        )
    return pd.DataFrame(rows)
