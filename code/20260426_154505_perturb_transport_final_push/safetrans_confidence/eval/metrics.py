from __future__ import annotations

import math

import numpy as np
import pandas as pd


def raw_spearman(score: pd.Series, error: pd.Series) -> float:
    s = pd.to_numeric(score, errors="coerce")
    e = pd.to_numeric(error, errors="coerce")
    mask = s.notna() & e.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(s[mask].corr(e[mask], method="spearman"))


def aligned_spearman(score: pd.Series, error: pd.Series, score_type: str) -> float:
    rho = raw_spearman(score, error)
    if score_type == "confidence" and np.isfinite(rho):
        return -rho
    return rho


def compute_aurc(errors: np.ndarray, scores: np.ndarray, score_type: str) -> float:
    """Area Under the Risk-Coverage curve using all thresholds.

    Lower AURC means the scorer retains low-error predictions more effectively.
    Follows Geifman & El-Yaniv (2017) definition.
    """
    errors = np.asarray(errors, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(errors) & np.isfinite(scores)
    errors, scores = errors[mask], scores[mask]
    n = len(errors)
    if n < 2:
        return float("nan")
    ascending = (score_type == "risk")
    order = np.argsort(scores) if ascending else np.argsort(-scores)
    sorted_errors = errors[order]
    cumsum = np.cumsum(sorted_errors)
    risks = cumsum / np.arange(1, n + 1)
    return float(risks.mean())


def compute_oracle_aurc(errors: np.ndarray) -> float:
    """AURC of the perfect oracle that sorts by true error ascending."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n < 2:
        return float("nan")
    sorted_errors = np.sort(errors)
    cumsum = np.cumsum(sorted_errors)
    risks = cumsum / np.arange(1, n + 1)
    return float(risks.mean())


def compute_random_aurc(errors: np.ndarray) -> float:
    """Expected AURC of a random scorer (equals mean error)."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) < 2:
        return float("nan")
    return float(errors.mean())


def compute_excess_aurc(errors: np.ndarray, scores: np.ndarray, score_type: str) -> float:
    """Excess AURC = AURC(scorer) - AURC(oracle).

    Measures how much worse the scorer is compared to a perfect oracle.
    Lower is better; zero means oracle-level performance.
    """
    aurc = compute_aurc(errors, scores, score_type)
    oracle = compute_oracle_aurc(errors)
    if not np.isfinite(aurc) or not np.isfinite(oracle):
        return float("nan")
    return aurc - oracle


def evaluate_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build eval summary and risk-coverage tables from long-form scores."""
    test = scores[scores["split"] == "test"].dropna(subset=["score_value", "true_error_rmse"]).copy()
    rows = []
    for level, group_cols in [
        ("dataset", ["dataset_family", "dataset_name", "score_name"]),
        ("family", ["dataset_family", "score_name"]),
        ("overall", ["score_name"]),
    ]:
        for key, g in test.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            meta = dict(zip(group_cols, key))
            score_type = str(g["score_type"].iloc[0])
            err = g["true_error_rmse"].to_numpy()
            scr = g["score_value"].to_numpy()
            rows.append(
                {
                    "level": level,
                    "dataset_family": meta.get("dataset_family", "ALL"),
                    "dataset_name": meta.get("dataset_name", "ALL"),
                    "score_name": meta.get("score_name"),
                    "score_type": score_type,
                    "n": int(len(g)),
                    "spearman_score_vs_rmse": raw_spearman(g["score_value"], g["true_error_rmse"]),
                    "direction_aligned_spearman": aligned_spearman(
                        g["score_value"], g["true_error_rmse"], score_type
                    ),
                    "mean_rmse": float(g["true_error_rmse"].mean()),
                    "aurc": compute_aurc(err, scr, score_type),
                    "oracle_aurc": compute_oracle_aurc(err),
                    "random_aurc": compute_random_aurc(err),
                    "excess_aurc": compute_excess_aurc(err, scr, score_type),
                }
            )
    eval_df = pd.DataFrame(rows)

    cov_rows = []
    for (_family, dataset, score), g in test.groupby(["dataset_family", "dataset_name", "score_name"]):
        score_type = str(g["score_type"].iloc[0])
        g = g.sort_values("score_value", ascending=(score_type == "risk"))
        full = float(g["true_error_rmse"].mean())
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            keep = max(1, int(math.ceil(cov * len(g))))
            kept = g.head(keep)
            mean = float(kept["true_error_rmse"].mean())
            cov_rows.append(
                {
                    "dataset_family": _family,
                    "dataset_name": dataset,
                    "score_name": score,
                    "score_type": score_type,
                    "coverage": cov,
                    "mean_rmse": mean,
                    "full_mean_rmse": full,
                    "risk_cov_improve_pct": 100.0 * (full - mean) / full if full else np.nan,
                }
            )
    cov_df = pd.DataFrame(cov_rows)
    cov80 = cov_df[np.isclose(cov_df["coverage"], 0.8)].groupby(
        ["dataset_name", "score_name"], as_index=False
    )["risk_cov_improve_pct"].mean()
    eval_df = eval_df.merge(cov80, on=["dataset_name", "score_name"], how="left")
    return eval_df, cov_df
