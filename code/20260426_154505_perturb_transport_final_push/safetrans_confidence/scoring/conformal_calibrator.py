from __future__ import annotations

import numpy as np
import pandas as pd


def build_unsafe_flags(
    scores: pd.DataFrame,
    target_coverage: float = 0.8,
    score_name: str | None = None,
) -> pd.DataFrame:
    """Create fold-local unsafe flags from validation score quantiles.

    For confidence scores, lower values are unsafe. For risk scores, higher
    values are unsafe. Thresholds are learned from val rows when available,
    otherwise train rows. Test labels are never used to set the threshold.
    """
    if score_name is not None:
        scores = scores[scores["score_name"] == score_name].copy()
    rows: list[dict] = []
    for (dataset, fold, predictor, sname), sub in scores.groupby(
        ["dataset_name", "fold_id", "predictor_name", "score_name"], dropna=False
    ):
        ref = sub[sub["split"] == "val"]
        if ref.empty:
            ref = sub[sub["split"] == "train"]
        test = sub[sub["split"] == "test"]
        if ref.empty or test.empty:
            continue
        stype = str(sub["score_type"].iloc[0])
        ref_values = pd.to_numeric(ref["score_value"], errors="coerce").dropna()
        if ref_values.empty:
            continue
        if stype == "risk":
            threshold = float(ref_values.quantile(target_coverage))
            unsafe = pd.to_numeric(test["score_value"], errors="coerce") > threshold
        else:
            threshold = float(ref_values.quantile(1.0 - target_coverage))
            unsafe = pd.to_numeric(test["score_value"], errors="coerce") < threshold
        for (_, row), flag in zip(test.iterrows(), unsafe):
            rows.append(
                {
                    "record_id": row["record_id"],
                    "dataset_name": dataset,
                    "fold_id": int(fold),
                    "predictor_name": predictor,
                    "score_name": sname,
                    "score_type": stype,
                    "target_coverage": float(target_coverage),
                    "threshold": threshold,
                    "unsafe_flag": bool(flag) if not pd.isna(flag) else np.nan,
                    "score_value": row["score_value"],
                    "true_error_rmse": row["true_error_rmse"],
                }
            )
    return pd.DataFrame(rows)

