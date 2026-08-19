from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # Average ties without pulling scipy into the hot path.
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, ranks)
        ranks = sums[inv] / counts[inv]
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rankdata(np.asarray(a)), _rankdata(np.asarray(b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def topk_overlap(true: np.ndarray, pred: np.ndarray, k: int = 20) -> float:
    k = min(k, len(true))
    if k <= 0:
        return float("nan")
    t = set(np.argsort(-np.abs(true))[:k])
    p = set(np.argsort(-np.abs(pred))[:k])
    return float(len(t & p) / k)


def deg_precision(true: np.ndarray, pred: np.ndarray, k: int = 50) -> float:
    k = min(k, len(true))
    if k <= 0:
        return float("nan")
    true_deg = set(np.argsort(-np.abs(true))[:k])
    pred_deg = set(np.argsort(-np.abs(pred))[:k])
    return float(len(true_deg & pred_deg) / k)


def program_shift_consistency(true_program: np.ndarray, pred_program: np.ndarray) -> float:
    return pearson(np.asarray(true_program), np.asarray(pred_program))


def effect_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_program: np.ndarray | None = None,
    pred_program: np.ndarray | None = None,
) -> Dict[str, float]:
    out = {
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "top20_overlap": topk_overlap(y_true, y_pred, 20),
        "deg_precision_top50": deg_precision(y_true, y_pred, 50),
    }
    if true_program is not None and pred_program is not None:
        out["program_shift_consistency"] = program_shift_consistency(true_program, pred_program)
    else:
        out["program_shift_consistency"] = float("nan")
    return out


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [
        "pearson",
        "spearman",
        "rmse",
        "top20_overlap",
        "deg_precision_top50",
        "program_shift_consistency",
    ]
    rows = []
    for keys, sub in df.groupby(["phase", "dataset", "split_type", "model"], dropna=False):
        row = dict(zip(["phase", "dataset", "split_type", "model"], keys))
        row["n_runs"] = len(sub)
        row["n_tasks_mean"] = float(sub["n_tasks"].mean()) if "n_tasks" in sub else math.nan
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_v1_vs_v0(summary: pd.DataFrame) -> pd.DataFrame:
    return compare_model_vs_v0(summary, "V1")


def compare_model_vs_v0(summary: pd.DataFrame, model_name: str = "V2") -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    idx_cols = ["phase", "dataset", "split_type"]
    for keys, sub in summary.groupby(idx_cols, dropna=False):
        base = sub[sub["model"] == "V0"]
        v1 = sub[sub["model"] == model_name]
        if base.empty or v1.empty:
            continue
        b = base.iloc[0]
        v = v1.iloc[0]
        row = dict(zip(idx_cols, keys))
        row["model"] = model_name
        row.update(
            {
                "pearson_delta": v["pearson_mean"] - b["pearson_mean"],
                "spearman_delta": v["spearman_mean"] - b["spearman_mean"],
                "rmse_delta": v["rmse_mean"] - b["rmse_mean"],
                "top20_delta": v["top20_overlap_mean"] - b["top20_overlap_mean"],
                "deg_precision_delta": v["deg_precision_top50_mean"] - b["deg_precision_top50_mean"],
                "program_consistency_delta": v["program_shift_consistency_mean"]
                - b["program_shift_consistency_mean"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
