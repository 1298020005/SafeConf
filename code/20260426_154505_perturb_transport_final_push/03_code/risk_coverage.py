from __future__ import annotations

import numpy as np
import pandas as pd

from evaluators import deg_precision, rmse, topk_overlap


def task_errors(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(y_true)):
        rows.append(
            {
                "task_id": i,
                "rmse": rmse(y_true[i], y_pred[i]),
                "top20_overlap": topk_overlap(y_true[i], y_pred[i], 20),
                "deg_precision_top50": deg_precision(y_true[i], y_pred[i], 50),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    coverages: list[float] | None = None,
) -> pd.DataFrame:
    if coverages is None:
        coverages = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    errors = task_errors(y_true, y_pred)
    conf = np.asarray(confidence, dtype=np.float64)
    order = np.argsort(-conf)
    rows = []
    n = len(order)
    for cov in coverages:
        k = max(1, int(np.ceil(n * cov)))
        keep = order[:k]
        sub = errors.iloc[keep]
        rows.append(
            {
                "coverage": float(k / n),
                "abstention_rate": float(1.0 - k / n),
                "mean_confidence": float(conf[keep].mean()),
                "rmse": float(sub["rmse"].mean()),
                "top20_overlap": float(sub["top20_overlap"].mean()),
                "deg_precision_top50": float(sub["deg_precision_top50"].mean()),
                "n_kept": int(k),
                "n_total": int(n),
            }
        )
    return pd.DataFrame(rows)
