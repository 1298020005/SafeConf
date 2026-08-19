#!/usr/bin/env python3
"""Shared deterministic math for the frozen E174 conformal experiment."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


STATES = ("Rest", "Stim8hr", "Stim48hr")
MODEL_SPECS: Mapping[str, tuple[str, ...]] = {
    "state_stratum_constant": (),
    "magnitude": ("predicted_magnitude",),
    "magnitude_plus_pair_lower": ("predicted_magnitude", "pair_lower_bound_rmse"),
}
RIDGE_LAMBDA = 10.0


def target_key(frame: pd.DataFrame) -> pd.Series:
    return frame.panel_id.astype(str) + "::" + frame.perturbed_gene_id.astype(str)


def add_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "pair_lower_bound_rmse" not in result:
        result["pair_lower_bound_rmse"] = result.model_disagreement_rmse.astype(float) / 2.0
    return result


def _categorical_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    state = frame.culture_condition.astype(str)
    stratum = frame.target_stratum.astype(str)
    if not set(state).issubset(STATES):
        raise ValueError(f"unexpected states: {sorted(set(state) - set(STATES))}")
    if not set(stratum).issubset({"DONOR_UNSEEN_ONLY", "COLUMN_UNSEEN"}):
        raise ValueError("unexpected target stratum")
    matrix = np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            state.eq("Stim8hr").to_numpy(float),
            state.eq("Stim48hr").to_numpy(float),
            stratum.eq("COLUMN_UNSEEN").to_numpy(float),
        ]
    )
    return matrix, ["intercept", "state_Stim8hr", "state_Stim48hr", "column_unseen"]


def fit_ridge(frame: pd.DataFrame, outcome: str, spec: str) -> dict[str, Any]:
    if spec not in MODEL_SPECS:
        raise ValueError(spec)
    frame = add_pair_columns(frame)
    categorical, names = _categorical_design(frame)
    numeric = list(MODEL_SPECS[spec])
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    columns = [categorical]
    for column in numeric:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite feature: {column}")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not scale > 1e-12:
            raise ValueError(f"degenerate numeric feature: {column}")
        means[column], scales[column] = mean, scale
        columns.append(((values - mean) / scale)[:, None])
        names.append(f"z_{column}")
    design = np.column_stack(columns)
    y = frame[outcome].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError(f"non-finite outcome: {outcome}")
    penalty = np.eye(design.shape[1], dtype=float) * RIDGE_LAMBDA
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "schema": "safeconf_e174_ridge_base_v1",
        "spec": spec,
        "outcome": outcome,
        "ridge_lambda": RIDGE_LAMBDA,
        "feature_names": names,
        "numeric_means": means,
        "numeric_scales": scales,
        "coefficient": [float(value) for value in coefficient],
        "n_training_tasks": int(len(frame)),
        "n_training_targets": int(target_key(frame).nunique()),
    }


def predict_ridge(frame: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
    frame = add_pair_columns(frame)
    spec = str(model["spec"])
    categorical, names = _categorical_design(frame)
    columns = [categorical]
    for column in MODEL_SPECS[spec]:
        mean = float(model["numeric_means"][column])
        scale = float(model["numeric_scales"][column])
        columns.append(((frame[column].to_numpy(float) - mean) / scale)[:, None])
        names.append(f"z_{column}")
    if names != list(model["feature_names"]):
        raise ValueError("model feature order changed")
    result = np.column_stack(columns) @ np.asarray(model["coefficient"], dtype=float)
    if str(model["outcome"]) == "pair_mean_rmse":
        result = np.maximum(result, frame.pair_lower_bound_rmse.to_numpy(float))
    else:
        result = np.maximum(result, 0.0)
    if not np.isfinite(result).all():
        raise ValueError("non-finite base prediction")
    return result


def conformal_rank(n_clusters: int, coverage: float) -> int:
    if not 0 < coverage < 1 or n_clusters < 1:
        raise ValueError("invalid conformal rank input")
    return min(n_clusters, math.ceil((n_clusters + 1) * coverage))


def calibrate_cluster_upper(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    outcome: str,
    coverage: float = 0.90,
) -> dict[str, Any]:
    if len(frame) != len(base_prediction):
        raise ValueError("calibration prediction length changed")
    work = frame[["panel_id", "perturbed_gene_id", "culture_condition", outcome]].copy()
    work["base_prediction"] = np.asarray(base_prediction, dtype=float)
    work["residual"] = work[outcome].to_numpy(float) - work.base_prediction.to_numpy(float)
    work["target_key"] = target_key(frame).to_numpy()
    counts = work.groupby("target_key").culture_condition.nunique()
    if not counts.eq(3).all():
        raise ValueError("each calibration target must contain all three states")
    residual = work.groupby("target_key", sort=True).residual.max().sort_values().to_numpy(float)
    rank = conformal_rank(len(residual), coverage)
    quantile = float(residual[rank - 1])
    return {
        "schema": "safeconf_e174_cluster_conformal_upper_v1",
        "outcome": outcome,
        "coverage": coverage,
        "n_calibration_targets": int(len(residual)),
        "n_calibration_tasks": int(len(work)),
        "finite_sample_order_rank_one_based": int(rank),
        "cluster_score": "max_state(observed_error_minus_base_prediction)",
        "quantile": quantile,
        "calibration_residual_sha256_float64": hashlib.sha256(
            np.ascontiguousarray(residual, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def apply_cluster_upper(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    upper = np.asarray(base_prediction, dtype=float) + float(calibration["quantile"])
    if str(calibration["outcome"]) == "pair_mean_rmse":
        upper = np.maximum(upper, add_pair_columns(frame).pair_lower_bound_rmse.to_numpy(float))
    else:
        upper = np.maximum(upper, 0.0)
    return upper


def identity_split(frame: pd.DataFrame, salt: str) -> dict[str, str]:
    targets = frame.assign(target_key=target_key(frame))[
        ["target_key", "panel_id", "target_stratum"]
    ].drop_duplicates()
    targets["identity_sha256"] = targets.target_key.map(
        lambda value: hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()
    )
    blocks: list[pd.DataFrame] = []
    for _, block in targets.groupby(["panel_id", "target_stratum"], sort=True):
        block = block.sort_values(["identity_sha256", "target_key"], kind="stable").copy()
        n = len(block)
        n_train, n_calibration = int(round(0.60 * n)), int(round(0.20 * n))
        block["development_split"] = (
            ["train"] * n_train
            + ["calibration"] * n_calibration
            + ["evaluation"] * (n - n_train - n_calibration)
        )
        blocks.append(block)
    return pd.concat(blocks).set_index("target_key").development_split.to_dict()


def rmse_rows(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("RMSE matrix shapes changed")
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def load_npz_vectors(path: Any, expected: int, n_genes: int = 512) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = np.asarray(archive[key], dtype=np.float64)
            if value.shape != (n_genes,) or not np.isfinite(value).all():
                raise ValueError(f"invalid vector {path}/{key}/{value.shape}")
            result[str(key)] = value
    if len(result) != expected:
        raise ValueError(f"unexpected vector count {path}: {len(result)} != {expected}")
    return result
