"""Within-group feature normalization for cross-dataset reliability modeling.

The frozen v0.2 score normalizes each feature with a robust z-score against
fold-local train rows. That is enough for a per-dataset rule, but a confidence
model that must transfer ACROSS datasets needs features on a common, scale-free
axis: raw ``perturbation_support_count`` medians range from 1 to 66 and
``model_disagreement_rmse`` medians span a 30x range across the seven datasets,
so a model trained on raw values would learn dataset-scale artifacts instead of
reliability signal.

This module maps every leakage-safe feature to its empirical quantile in
[0, 1], using ONLY fold-local train/val reference rows to build the empirical
CDF. Test rows are mapped through the train CDF, so no held-out label or
held-out feature distribution leaks into the normalization statistics. The
result is a dataset-scale-invariant feature matrix suitable for
leave-one-dataset-out training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from safetrans_confidence.features.schema import (
    LABEL_OR_FORBIDDEN_COLUMNS,
    assert_no_label_leakage,
)

QNORM_SUFFIX = "_qnorm"

DEFAULT_REFERENCE_SPLITS = ("train", "val")
DEFAULT_GROUP_COLUMNS = ("dataset_name", "fold_id", "predictor_name")

# Leakage-safe reliability features that may be used by a transferable model.
# These never use held-out true effects; label-derived entries only use
# fold-local train labels, which is allowed for a confidence model.
TRANSFERABLE_FEATURE_CANDIDATES = (
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "historical_residual_risk",
    "model_disagreement_rmse",
    "model_disagreement_cosine",
    "ood_nearest_distance",
    "ood_mean_k_distance",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "prediction_magnitude_deviation",
    "prediction_norm_ratio",
)


def available_transferable_features(table: pd.DataFrame) -> list[str]:
    """Return transferable feature columns present in the table.

    ``perturbation_support_count`` is log1p-friendly but we quantile-normalize
    anyway, so raw counts are fine here.
    """
    cols = [c for c in TRANSFERABLE_FEATURE_CANDIDATES if c in table.columns]
    assert_no_label_leakage(cols)
    return cols


def _empirical_quantile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to their empirical quantile in [0, 1] using a reference sample.

    Uses midpoint ranks so the median of the reference maps near 0.5. Values
    outside the reference range saturate to (0, 1). NaNs map to 0.5.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    out = np.full(len(values), np.nan, dtype=float)
    vals = np.asarray(values, dtype=float)
    if ref.size == 0:
        out[:] = 0.5
        return out
    ref_sorted = np.sort(ref)
    n = ref_sorted.size
    finite = np.isfinite(vals)
    # left/right insertion points give the count of ref strictly < and <= value.
    left = np.searchsorted(ref_sorted, vals[finite], side="left")
    right = np.searchsorted(ref_sorted, vals[finite], side="right")
    midpoint_rank = (left + right) / 2.0
    out[finite] = midpoint_rank / n
    out[~finite] = 0.5
    return np.clip(out, 0.0, 1.0)


def normalize_features_within_group(
    base: pd.DataFrame,
    feature_cols: list[str] | None = None,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLUMNS,
    reference_splits: tuple[str, ...] = DEFAULT_REFERENCE_SPLITS,
) -> tuple[pd.DataFrame, list[str]]:
    """Add quantile-normalized feature columns computed within each group.

    For each ``group_cols`` group, the empirical CDF of each feature is built
    from rows whose ``split`` is in ``reference_splits`` and applied to all rows
    in the group. New columns are named ``<feature><QNORM_SUFFIX>``.

    Returns the augmented frame and the list of new normalized column names.
    """
    cols = list(feature_cols) if feature_cols is not None else available_transferable_features(base)
    bad = sorted(set(cols) & set(LABEL_OR_FORBIDDEN_COLUMNS))
    if bad:
        raise ValueError(f"refusing to normalize label/forbidden columns: {bad}")
    out = base.copy()
    norm_cols = [f"{c}{QNORM_SUFFIX}" for c in cols]
    for c in norm_cols:
        out[c] = np.nan
    if "split" not in out.columns:
        raise ValueError("normalize_features_within_group requires a 'split' column")

    present_group_cols = [c for c in group_cols if c in out.columns]
    if not present_group_cols:
        raise ValueError(f"none of the group columns are present: {group_cols}")

    for _, idx_obj in out.groupby(list(present_group_cols), dropna=False).groups.items():
        idx = list(idx_obj)
        sub = out.loc[idx]
        ref_mask = sub["split"].isin(reference_splits)
        ref_rows = sub[ref_mask]
        if ref_rows.empty:
            ref_rows = sub  # degenerate fallback: use all rows in the group
        for feat, norm_col in zip(cols, norm_cols):
            ref_values = pd.to_numeric(ref_rows[feat], errors="coerce").to_numpy()
            all_values = pd.to_numeric(sub[feat], errors="coerce").to_numpy()
            out.loc[idx, norm_col] = _empirical_quantile(all_values, ref_values)
    return out, norm_cols
