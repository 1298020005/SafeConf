from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _test_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "score_value", "true_error_rmse", "score_type"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores table missing columns: {missing}")
    return scores[scores["split"] == "test"].dropna(
        subset=["score_value", "true_error_rmse"]
    ).copy()


def _group_iter(test: pd.DataFrame):
    group_cols = ["dataset_family", "dataset_name", "predictor_name", "score_name"]
    for key, group in test.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        yield meta, group


def compute_high_low_rmse(scores: pd.DataFrame, fraction: float = 0.2) -> pd.DataFrame:
    """Compare retained high-confidence/low-risk predictions against the opposite tail."""
    rows: list[dict] = []
    test = _test_scores(scores)
    for meta, group in _group_iter(test):
        if len(group) < 5:
            continue
        score_type = str(group["score_type"].iloc[0])
        ascending = score_type == "risk"
        ordered = group.sort_values("score_value", ascending=ascending)
        n_tail = max(1, int(math.ceil(fraction * len(ordered))))
        good = ordered.head(n_tail)
        bad = ordered.tail(n_tail)
        rows.append(
            {
                **meta,
                "score_type": score_type,
                "tail_fraction": float(fraction),
                "n_total": int(len(ordered)),
                "n_good_tail": int(len(good)),
                "n_bad_tail": int(len(bad)),
                "good_tail_label": "high_confidence_or_low_risk",
                "bad_tail_label": "low_confidence_or_high_risk",
                "good_tail_mean_rmse": float(good["true_error_rmse"].mean()),
                "bad_tail_mean_rmse": float(bad["true_error_rmse"].mean()),
                "good_tail_median_rmse": float(good["true_error_rmse"].median()),
                "bad_tail_median_rmse": float(bad["true_error_rmse"].median()),
                "mean_rmse_improve_pct": (
                    100.0
                    * (float(bad["true_error_rmse"].mean()) - float(good["true_error_rmse"].mean()))
                    / float(bad["true_error_rmse"].mean())
                    if float(bad["true_error_rmse"].mean()) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_calibration_buckets(scores: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Bucket scores and summarize the empirical error in each bucket."""
    rows: list[dict] = []
    test = _test_scores(scores)
    for meta, group in _group_iter(test):
        if len(group) < n_bins:
            continue
        score_type = str(group["score_type"].iloc[0])
        group = group.copy()
        # Convert every score to an aligned reliability axis: larger means more reliable.
        if score_type == "risk":
            group["reliability_axis"] = -pd.to_numeric(group["score_value"], errors="coerce")
        else:
            group["reliability_axis"] = pd.to_numeric(group["score_value"], errors="coerce")
        valid = group.dropna(subset=["reliability_axis", "true_error_rmse"])
        if valid["reliability_axis"].nunique() < 2:
            continue
        try:
            valid["bucket"] = pd.qcut(
                valid["reliability_axis"], q=min(n_bins, valid["reliability_axis"].nunique()), duplicates="drop"
            )
        except ValueError:
            continue
        bucketed = valid.groupby("bucket", observed=True)
        for bucket_id, (_bucket, b) in enumerate(bucketed, start=1):
            rows.append(
                {
                    **meta,
                    "score_type": score_type,
                    "bucket_id_low_to_high_reliability": int(bucket_id),
                    "n": int(len(b)),
                    "mean_score_value": float(b["score_value"].mean()),
                    "mean_reliability_axis": float(b["reliability_axis"].mean()),
                    "mean_rmse": float(b["true_error_rmse"].mean()),
                    "median_rmse": float(b["true_error_rmse"].median()),
                }
            )
    return pd.DataFrame(rows)


def compute_failure_detection(scores: pd.DataFrame, failure_quantile: float = 0.8) -> pd.DataFrame:
    """Treat the top-error tail as failures and evaluate whether scores detect it."""
    rows: list[dict] = []
    test = _test_scores(scores)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        roc_auc_score = None
        average_precision_score = None

    for meta, group in _group_iter(test):
        if len(group) < 8:
            continue
        score_type = str(group["score_type"].iloc[0])
        threshold = float(group["true_error_rmse"].quantile(failure_quantile))
        y = (group["true_error_rmse"] >= threshold).astype(int)
        if y.nunique() < 2:
            continue
        # Higher detector_score should mean more likely to fail.
        raw_score = pd.to_numeric(group["score_value"], errors="coerce")
        detector_score = raw_score if score_type == "risk" else -raw_score
        mask = detector_score.notna()
        if int(mask.sum()) < 8 or y[mask].nunique() < 2:
            continue
        auroc = np.nan
        auprc = np.nan
        if roc_auc_score is not None:
            try:
                auroc = float(roc_auc_score(y[mask], detector_score[mask]))
                auprc = float(average_precision_score(y[mask], detector_score[mask]))
            except Exception:
                pass
        rows.append(
            {
                **meta,
                "score_type": score_type,
                "failure_quantile": float(failure_quantile),
                "n": int(mask.sum()),
                "n_failure": int(y[mask].sum()),
                "failure_threshold_rmse": threshold,
                "failure_detection_auroc": auroc,
                "failure_detection_auprc": auprc,
            }
        )
    return pd.DataFrame(rows)


def build_reliability_tables(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the extra reliability tables a confidence-scoring paper needs."""
    return {
        "HIGH_LOW_RMSE": compute_high_low_rmse(scores),
        "CALIBRATION_BUCKETS": compute_calibration_buckets(scores),
        "FAILURE_DETECTION": compute_failure_detection(scores),
    }
