#!/usr/bin/env python3
"""Robust leave-one-predictor-out validation for SafeConf task-risk scores.

The learned reliability model is fitted on errors from V0StrongBaseline and
ContextSimBaseline, then evaluated on an unseen retrieval-based predictor.
Two validation modes are supported:

``lopo``
    Hold out the predictor only. Training may use V0/ContextSim train+val rows
    from every dataset.

``lodo_lopo``
    Hold out both the predictor and the complete target dataset. Training uses
    V0/ContextSim train+val rows from the other datasets only.

This is evidence for task-risk transfer within the tested retrieval-predictor
family. It is not evidence that the score works for every deep predictor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from safetrans_confidence.eval.selective_prediction import (
    selective_prediction_summary,
    within_magnitude_stratum_rho,
)
from safetrans_confidence.features.normalize import (
    QNORM_SUFFIX,
    normalize_features_within_group,
)
from safetrans_confidence.scoring.error_ranker import _make_model
from safetrans_confidence.scoring.protocol_v0_2 import (
    PRIMARY_SCORE_NAME,
    assign_dataset_family,
    build_protocol_v0_2_scores,
)

TRAIN_PREDICTORS = ("V0StrongBaseline", "ContextSimBaseline")
MAIN_THIRD_PREDICTORS = ("PertMeanPredictor", "Control1NNPredictor")
TOP_FRACTIONS = (0.05, 0.10, 0.20)

# Task-level signals are identical for every predictor evaluated on a task.
TASK_LEVEL_FEATURES = [
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
]

# These features require the target predictor's output vector.
PREDICTOR_OUTPUT_FEATURES = [
    "prediction_l2_norm",
    "prediction_abs_mean",
    "prediction_magnitude_deviation",
    "prediction_norm_ratio",
]

ALL_FEATURES = TASK_LEVEL_FEATURES + PREDICTOR_OUTPUT_FEATURES
DISAGREEMENT_FEATURES = ["model_disagreement_rmse", "model_disagreement_cosine"]

FEATURE_SETS = {
    "full": ALL_FEATURES,
    "no_disagreement": [c for c in ALL_FEATURES if c not in DISAGREEMENT_FEATURES],
    "pre_model_task_only": [
        c
        for c in TASK_LEVEL_FEATURES
        if c not in DISAGREEMENT_FEATURES
    ],
}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _load_run(run_dir: Path) -> dict:
    rec = pd.read_csv(run_dir / "tables" / "PREDICTION_RECORDS.csv")
    feats = pd.read_csv(run_dir / "tables" / "CONFIDENCE_FEATURES.csv")
    with np.load(run_dir / "input" / "true_effects.npz") as true_npz:
        true_arrays = {
            k: np.asarray(true_npz[k], dtype=np.float32) for k in true_npz.files
        }
    with np.load(run_dir / "input" / "target_control_means.npz") as ctrl_npz:
        ctrl_arrays = {
            k: np.asarray(ctrl_npz[k], dtype=np.float32) for k in ctrl_npz.files
        }
    with np.load(run_dir / "input" / "predicted_effects.npz") as pred_npz:
        pred_arrays = {
            k: np.asarray(pred_npz[k], dtype=np.float32) for k in pred_npz.files
        }
    return {
        "rec": rec,
        "feats": feats,
        "true": true_arrays,
        "ctrl": ctrl_arrays,
        "pred": pred_arrays,
        "dataset": str(rec["dataset_name"].iloc[0]),
    }


def _task_table(rec: pd.DataFrame, true: dict, ctrl: dict) -> pd.DataFrame:
    """Return one task row per fold without predictor duplication."""
    cols = [
        "task_key",
        "fold_id",
        "split",
        "context",
        "perturbation",
        "true_effect_key",
        "target_control_key",
    ]
    tasks = rec[cols].drop_duplicates(["fold_id", "task_key"]).reset_index(drop=True)
    tasks["true_effect"] = tasks["true_effect_key"].map(
        lambda key: true.get(str(key))
    )
    tasks["control_mean"] = tasks["target_control_key"].map(
        lambda key: ctrl.get(str(key))
    )
    return tasks


def _valid_array(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0


def _source_rows(train: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Fold-train sources excluding the target task when scoring a train row."""
    source = train
    if str(row["split"]) == "train":
        source = source[source["task_key"].astype(str) != str(row["task_key"])]
    return source


def _fallback_mean(source: pd.DataFrame) -> np.ndarray | None:
    arrays = [a for a in source["true_effect"] if _valid_array(a)]
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def predict_pert_mean(tasks: pd.DataFrame) -> pd.DataFrame:
    """Same-perturbation mean over fold-train tasks, with self-exclusion."""
    rows: list[dict] = []
    for fold, group in tasks.groupby("fold_id", dropna=False):
        train = group[group["split"] == "train"]
        for _, row in group.iterrows():
            if not _valid_array(row["true_effect"]):
                continue
            source = _source_rows(train, row)
            same = source[
                source["perturbation"].astype(str) == str(row["perturbation"])
            ]
            pred = _fallback_mean(same)
            if pred is None:
                pred = _fallback_mean(source)
            if pred is None:
                continue
            rows.append(_prediction_row(row, int(fold), "PertMeanPredictor", pred))
    return pd.DataFrame(rows)


def _predict_control_knn(
    tasks: pd.DataFrame,
    k: int,
    predictor_name: str,
) -> pd.DataFrame:
    """Hard k-NN over same-perturbation train contexts in control space."""
    rows: list[dict] = []
    for fold, group in tasks.groupby("fold_id", dropna=False):
        train = group[group["split"] == "train"]
        for _, row in group.iterrows():
            if not _valid_array(row["true_effect"]) or not _valid_array(
                row["control_mean"]
            ):
                continue
            source = _source_rows(train, row)
            same_pert = source[
                source["perturbation"].astype(str) == str(row["perturbation"])
            ]
            other_context = same_pert[
                same_pert["context"].astype(str) != str(row["context"])
            ]
            candidates = other_context if not other_context.empty else same_pert
            valid = candidates[
                candidates["control_mean"].map(_valid_array)
                & candidates["true_effect"].map(_valid_array)
            ]
            if valid.empty:
                pred = _fallback_mean(source)
            else:
                controls = np.stack(
                    [np.asarray(v, dtype=np.float64) for v in valid["control_mean"]],
                    axis=0,
                )
                effects = np.stack(
                    [np.asarray(v, dtype=np.float32) for v in valid["true_effect"]],
                    axis=0,
                )
                query = np.asarray(row["control_mean"], dtype=np.float64)
                denom = (np.linalg.norm(controls, axis=1) + 1e-8) * (
                    np.linalg.norm(query) + 1e-8
                )
                similarities = (controls @ query) / denom
                top = np.argsort(-similarities)[: min(k, len(similarities))]
                pred = effects[top].mean(axis=0).astype(np.float32)
            if pred is None:
                continue
            rows.append(_prediction_row(row, int(fold), predictor_name, pred))
    return pd.DataFrame(rows)


def predict_control_1nn(tasks: pd.DataFrame) -> pd.DataFrame:
    """Nearest same-perturbation train context; ContextSim-like sensitivity."""
    return _predict_control_knn(tasks, k=1, predictor_name="Control1NNPredictor")


def predict_control_knn(tasks: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Legacy hard-kNN sensitivity retained for reproducibility."""
    return _predict_control_knn(tasks, k=k, predictor_name="ControlKNNPredictor")


def _prediction_row(
    row: pd.Series,
    fold: int,
    predictor_name: str,
    prediction: np.ndarray,
) -> dict:
    true_effect = np.asarray(row["true_effect"], dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32)
    return {
        "task_key": row["task_key"],
        "fold_id": fold,
        "split": row["split"],
        "context": row["context"],
        "perturbation": row["perturbation"],
        "predictor_name": predictor_name,
        "predicted_effect": pred,
        "true_effect": true_effect,
        "true_error_rmse": _rmse(true_effect, pred),
    }


THIRD_PREDICTORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "PertMeanPredictor": predict_pert_mean,
    "Control1NNPredictor": predict_control_1nn,
    "ControlKNNPredictor": predict_control_knn,
}


def _fold_train_median_norm(tasks: pd.DataFrame) -> dict[int, float]:
    result: dict[int, float] = {}
    for fold, group in tasks.groupby("fold_id", dropna=False):
        train = group[group["split"] == "train"]
        norms = [
            float(np.linalg.norm(value))
            for value in train["true_effect"]
            if _valid_array(value)
        ]
        result[int(fold)] = float(np.median(norms)) if norms else np.nan
    return result


def _magnitude_features(
    predicted_effect: np.ndarray,
    fold_median_norm: float,
) -> dict:
    l2 = float(np.linalg.norm(predicted_effect))
    abs_mean = float(np.mean(np.abs(predicted_effect)))
    reference = (
        fold_median_norm
        if np.isfinite(fold_median_norm) and fold_median_norm > 1e-9
        else 1.0
    )
    ratio = l2 / reference
    return {
        "prediction_l2_norm": l2,
        "prediction_abs_mean": abs_mean,
        "prediction_norm_ratio": ratio,
        "prediction_magnitude_deviation": abs(ratio - 1.0),
    }


def _record_id(dataset: str, predictor: str, fold: int, task_key: object) -> str:
    return f"lopo::{dataset}::{predictor}::fold{fold}::{task_key}"


def build_feature_matrix(
    run_dir: Path,
    third_predictor: str = "PertMeanPredictor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned existing/third-predictor features and diversity rows."""
    run = _load_run(run_dir)
    rec, feats = run["rec"], run["feats"]
    tasks = _task_table(rec, run["true"], run["ctrl"])
    fold_median = _fold_train_median_norm(tasks)

    existing_rows: list[dict] = []
    for _, row in rec.iterrows():
        pred = run["pred"].get(str(row["predicted_effect_key"]))
        if not _valid_array(pred):
            continue
        fold = int(row["fold_id"])
        true_effect = run["true"].get(str(row["true_effect_key"]))
        existing_rows.append(
            {
                "record_id": _record_id(
                    str(row["dataset_name"]),
                    str(row["predictor_name"]),
                    fold,
                    row["task_key"],
                ),
                "dataset_name": row["dataset_name"],
                "dataset_family": assign_dataset_family(str(row["dataset_name"])),
                "task_key": row["task_key"],
                "fold_id": fold,
                "split": row["split"],
                "context": row["context"],
                "perturbation": row["perturbation"],
                "predictor_name": row["predictor_name"],
                "true_error_rmse": float(row["true_error_rmse"]),
                "true_effect_l2_norm": (
                    float(np.linalg.norm(true_effect))
                    if _valid_array(true_effect)
                    else np.nan
                ),
                **_magnitude_features(pred, fold_median.get(fold, np.nan)),
            }
        )
    existing = pd.DataFrame(existing_rows)

    third_predictions = THIRD_PREDICTORS[third_predictor](tasks)
    third_rows: list[dict] = []
    for _, row in third_predictions.iterrows():
        fold = int(row["fold_id"])
        pred = np.asarray(row["predicted_effect"], dtype=np.float32)
        third_rows.append(
            {
                "record_id": _record_id(
                    run["dataset"], third_predictor, fold, row["task_key"]
                ),
                "dataset_name": run["dataset"],
                "dataset_family": assign_dataset_family(run["dataset"]),
                "task_key": row["task_key"],
                "fold_id": fold,
                "split": row["split"],
                "context": row["context"],
                "perturbation": row["perturbation"],
                "predictor_name": third_predictor,
                "true_error_rmse": float(row["true_error_rmse"]),
                "true_effect_l2_norm": float(
                    np.linalg.norm(np.asarray(row["true_effect"]))
                ),
                **_magnitude_features(pred, fold_median.get(fold, np.nan)),
            }
        )
    third = pd.DataFrame(third_rows)

    task_feature_columns = [
        "task_key",
        "fold_id",
        "split",
        *[c for c in TASK_LEVEL_FEATURES if c in feats.columns],
    ]
    task_features = feats[task_feature_columns].drop_duplicates(
        ["task_key", "fold_id", "split"]
    )
    merged = pd.concat([existing, third], ignore_index=True).merge(
        task_features,
        on=["task_key", "fold_id", "split"],
        how="left",
        validate="many_to_one",
    )
    merged["run_dir"] = str(run_dir)

    diversity = _predictor_diversity_rows(
        run,
        third_predictions,
        third_predictor,
    )
    return merged, diversity


def _predictor_diversity_rows(
    run: dict,
    third_predictions: pd.DataFrame,
    third_predictor: str,
) -> pd.DataFrame:
    """Compare the third predictor with the existing ContextSim prediction."""
    contextsim = run["rec"][
        run["rec"]["predictor_name"].astype(str).eq("ContextSimBaseline")
    ]
    lookup: dict[tuple[int, str, str], tuple[np.ndarray, float]] = {}
    for _, row in contextsim.iterrows():
        pred = run["pred"].get(str(row["predicted_effect_key"]))
        if _valid_array(pred):
            lookup[
                (int(row["fold_id"]), str(row["split"]), str(row["task_key"]))
            ] = (pred, float(row["true_error_rmse"]))

    rows = []
    for _, row in third_predictions.iterrows():
        key = (int(row["fold_id"]), str(row["split"]), str(row["task_key"]))
        existing = lookup.get(key)
        if existing is None:
            continue
        contextsim_pred, contextsim_error = existing
        third_pred = np.asarray(row["predicted_effect"], dtype=np.float32)
        vector_rmse = _rmse(third_pred, contextsim_pred)
        rows.append(
            {
                "dataset_name": run["dataset"],
                "third_predictor": third_predictor,
                "fold_id": int(row["fold_id"]),
                "split": row["split"],
                "task_key": row["task_key"],
                "vector_rmse_vs_contextsim": vector_rmse,
                "cosine_similarity_vs_contextsim": _cosine_similarity(
                    third_pred, contextsim_pred
                ),
                "exact_vector_match": bool(
                    np.array_equal(third_pred, contextsim_pred)
                ),
                "near_identical_vector": bool(vector_rmse <= 1e-8),
                "third_predictor_error": float(row["true_error_rmse"]),
                "contextsim_error": contextsim_error,
            }
        )
    return pd.DataFrame(rows)


def _normalization_audit(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in base.groupby(
        ["dataset_name", "fold_id", "predictor_name"],
        dropna=False,
    ):
        dataset, fold, predictor = keys
        n_train = int(group["split"].astype(str).eq("train").sum())
        n_val = int(group["split"].astype(str).eq("val").sum())
        n_test = int(group["split"].astype(str).eq("test").sum())
        n_reference = n_train + n_val
        rows.append(
            {
                "dataset_name": dataset,
                "fold_id": int(fold),
                "predictor_name": predictor,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "n_reference_train_val": n_reference,
                "reference_splits": "train,val",
                "test_used_as_reference": False,
                "status": "ok" if n_reference > 0 else "fail_no_reference_rows",
            }
        )
    audit = pd.DataFrame(rows)
    failed = audit[audit["status"] != "ok"]
    if not failed.empty:
        sample = failed.head(5).to_dict("records")
        raise ValueError(f"normalization groups lack train/val reference rows: {sample}")
    return audit


def _feature_columns(
    normalized_columns: list[str],
    feature_set: str,
) -> list[str]:
    allowed = set(FEATURE_SETS[feature_set])
    selected = [
        column
        for column in normalized_columns
        if column.removesuffix(QNORM_SUFFIX) in allowed
    ]
    if not selected:
        raise ValueError(f"feature set {feature_set!r} has no available columns")
    if feature_set == "pre_model_task_only":
        forbidden = set(DISAGREEMENT_FEATURES + PREDICTOR_OUTPUT_FEATURES)
        leaked = [
            column
            for column in selected
            if column.removesuffix(QNORM_SUFFIX) in forbidden
        ]
        if leaked:
            raise AssertionError(
                f"pre_model_task_only contains predictor-output features: {leaked}"
            )
    return selected


def _train_target_rank(train: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=train.index, dtype=float)
    groups = train.groupby(
        ["dataset_name", "fold_id", "predictor_name"],
        dropna=False,
    ).groups
    for idx_obj in groups.values():
        idx = list(idx_obj)
        errors = pd.to_numeric(train.loc[idx, "true_error_rmse"], errors="coerce")
        out.loc[idx] = errors.rank(method="average", pct=True)
    return out


def _score_metadata(row: pd.Series) -> dict:
    return {
        "record_id": row["record_id"],
        "dataset_name": row["dataset_name"],
        "dataset_family": row["dataset_family"],
        "fold_id": int(row["fold_id"]),
        "split": row["split"],
        "context": row["context"],
        "perturbation": row["perturbation"],
        "predictor_name": row["predictor_name"],
        "task_key": row["task_key"],
        "true_error_rmse": float(row["true_error_rmse"]),
        "true_effect_l2_norm": float(row["true_effect_l2_norm"]),
    }


def _fit_learned_scores(
    base: pd.DataFrame,
    third_predictor: str,
    feature_set: str,
    validation_mode: str,
    normalized_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LOPO or LODOxLOPO models and return test scores plus provenance."""
    feature_columns = _feature_columns(normalized_columns, feature_set)
    target_datasets = sorted(base["dataset_name"].dropna().astype(str).unique())
    score_rows: list[dict] = []
    provenance_rows: list[dict] = []

    heldouts: list[str | None]
    if validation_mode == "lopo":
        heldouts = [None]
    elif validation_mode == "lodo_lopo":
        heldouts = target_datasets
    else:
        raise ValueError(validation_mode)

    folds = sorted(
        pd.to_numeric(base["fold_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for heldout in heldouts:
        for fold_id in folds:
            source_mask = (
                base["predictor_name"].isin(TRAIN_PREDICTORS)
                & base["split"].isin(["train", "val"])
                & pd.to_numeric(base["fold_id"], errors="coerce").eq(fold_id)
            )
            if heldout is not None:
                source_mask &= base["dataset_name"].astype(str).ne(heldout)
            source = base[source_mask].copy()
            source["_target_rank"] = _train_target_rank(source)
            source = source.dropna(subset=["_target_rank"])
            target_mask = (
                base["predictor_name"].astype(str).eq(third_predictor)
                & base["split"].astype(str).eq("test")
                & pd.to_numeric(base["fold_id"], errors="coerce").eq(fold_id)
            )
            if heldout is None:
                target_scope = "ALL"
            else:
                target_mask &= base["dataset_name"].astype(str).eq(heldout)
                target_scope = heldout
            target = base[target_mask].copy()
            if source.empty or target.empty:
                continue

            training_datasets = sorted(
                source["dataset_name"].dropna().astype(str).unique().tolist()
            )
            if heldout is not None and heldout in training_datasets:
                raise AssertionError(
                    f"held-out dataset leaked into training: {heldout}"
                )
            if not set(source["predictor_name"].astype(str).unique()).issubset(
                set(TRAIN_PREDICTORS)
            ):
                raise AssertionError("third predictor leaked into training rows")
            source_tasks = set(
                zip(
                    source["dataset_name"].astype(str),
                    source["task_key"].astype(str),
                )
            )
            target_tasks = set(
                zip(
                    target["dataset_name"].astype(str),
                    target["task_key"].astype(str),
                )
            )
            overlap = source_tasks & target_tasks
            if overlap:
                raise AssertionError(
                    "outer-fold task leakage between learned-model source and "
                    f"target: {sorted(overlap)[:3]}"
                )

            model = _make_model(
                seed + 1009 * fold_id,
                model_type="histgbt",
                n_train_rows=len(source),
            )
            model.fit(
                source[feature_columns].fillna(0.5).to_numpy(),
                source["_target_rank"].to_numpy(),
            )
            predictions = model.predict(
                target[feature_columns].fillna(0.5).to_numpy()
            )
            score_name = f"{validation_mode}_{feature_set}"
            for (_, row), value in zip(target.iterrows(), predictions):
                score_rows.append(
                    {
                        **_score_metadata(row),
                        "validation_mode": validation_mode,
                        "feature_set": feature_set,
                        "score_name": score_name,
                        "score_type": "risk",
                        "score_value": float(value),
                        "heldout_dataset": heldout or "",
                    }
                )
            provenance_rows.append(
                {
                    "validation_mode": validation_mode,
                    "feature_set": feature_set,
                    "fold_id": fold_id,
                    "heldout_dataset": heldout or "",
                    "target_scope": target_scope,
                    "third_predictor": third_predictor,
                    "n_train_rows": int(len(source)),
                    "n_test_rows": int(len(target)),
                    "training_datasets": ";".join(training_datasets),
                    "training_predictors": ";".join(
                        sorted(source["predictor_name"].astype(str).unique())
                    ),
                    "feature_columns": ";".join(feature_columns),
                    "third_predictor_error_used_for_fit": False,
                    "heldout_dataset_error_used_for_fit": False,
                    "source_target_task_overlap": 0,
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(provenance_rows)


def _baseline_scores(
    base: pd.DataFrame,
    third_predictor: str,
    seed: int,
) -> pd.DataFrame:
    target = base[
        base["predictor_name"].astype(str).eq(third_predictor)
        & base["split"].astype(str).eq("test")
    ].copy()
    target = target.sort_values(
        ["dataset_name", "fold_id", "task_key"],
        kind="stable",
    )
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for random_value, (_, row) in zip(rng.random(len(target)), target.iterrows()):
        metadata = _score_metadata(row)
        rows.append(
            {
                **metadata,
                "validation_mode": "baseline",
                "feature_set": "none",
                "score_name": "random",
                "score_type": "risk",
                "score_value": float(random_value),
                "heldout_dataset": "",
            }
        )
        rows.append(
            {
                **metadata,
                "validation_mode": "baseline",
                "feature_set": "none",
                "score_name": "predicted_magnitude",
                "score_type": "risk",
                "score_value": float(row["prediction_l2_norm"]),
                "heldout_dataset": "",
            }
        )

    protocol_scores, _ = build_protocol_v0_2_scores(
        base[base["predictor_name"].astype(str).eq(third_predictor)].copy(),
        include_evaluation_labels=True,
    )
    primary = protocol_scores[
        protocol_scores["score_name"].astype(str).eq(PRIMARY_SCORE_NAME)
        & protocol_scores["split"].astype(str).eq("test")
    ].copy()
    metadata_lookup = (
        target[
            [
                "record_id",
                "task_key",
                "true_effect_l2_norm",
                "dataset_family",
            ]
        ]
        .drop_duplicates("record_id")
        .set_index("record_id")
    )
    for _, row in primary.iterrows():
        record_id = str(row["record_id"])
        if record_id not in metadata_lookup.index:
            continue
        metadata = metadata_lookup.loc[record_id]
        rows.append(
            {
                "record_id": record_id,
                "dataset_name": row["dataset_name"],
                "dataset_family": metadata["dataset_family"],
                "fold_id": int(row["fold_id"]),
                "split": row["split"],
                "context": row["context"],
                "perturbation": row["perturbation"],
                "predictor_name": row["predictor_name"],
                "task_key": metadata["task_key"],
                "true_error_rmse": float(row["true_error_rmse"]),
                "true_effect_l2_norm": float(metadata["true_effect_l2_norm"]),
                "validation_mode": "baseline",
                "feature_set": "none",
                "score_name": "frozen_v0_2",
                "score_type": "confidence",
                "score_value": float(row["score_value"]),
                "heldout_dataset": "",
            }
        )
    return pd.DataFrame(rows)


def _risk_axis(frame: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(frame["score_value"], errors="coerce")
    return score.where(frame["score_type"].astype(str).eq("risk"), -score)


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return rankdata(values, method="average")
    except Exception:
        return pd.Series(values).rank(method="average").to_numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _partial_spearman(
    risk: np.ndarray,
    error: np.ndarray,
    control: np.ndarray,
) -> float:
    mask = np.isfinite(risk) & np.isfinite(error) & np.isfinite(control)
    if int(mask.sum()) < 5:
        return float("nan")
    rr = _rankdata(risk[mask])
    re = _rankdata(error[mask])
    rc = _rankdata(control[mask])
    r_xy = _pearson(rr, re)
    r_xc = _pearson(rr, rc)
    r_yc = _pearson(re, rc)
    if not all(np.isfinite(v) for v in (r_xy, r_xc, r_yc)):
        return float("nan")
    denom = math.sqrt(max(0.0, (1.0 - r_xc**2) * (1.0 - r_yc**2)))
    return float((r_xy - r_xc * r_yc) / denom) if denom > 1e-12 else np.nan


def _top_fraction_enrichments(
    errors: np.ndarray,
    risk: np.ndarray,
) -> dict[str, float]:
    n = len(errors)
    if n == 0:
        return {f"top{int(frac * 100)}_enrichment": np.nan for frac in TOP_FRACTIONS}
    risk_order = np.argsort(-risk)
    error_order = np.argsort(-errors)
    result = {}
    for fraction in TOP_FRACTIONS:
        n_top = max(1, int(math.ceil(n * fraction)))
        top_risk = set(risk_order[:n_top].tolist())
        worst_error = set(error_order[:n_top].tolist())
        precision = len(top_risk & worst_error) / n_top
        expected = n_top / n
        result[f"top{int(fraction * 100)}_enrichment"] = (
            precision / expected if expected > 0 else np.nan
        )
    return result


def _metric_bundle(
    errors: np.ndarray,
    risk: np.ndarray,
    magnitude: np.ndarray,
) -> dict[str, float]:
    mask = np.isfinite(errors) & np.isfinite(risk)
    errors = errors[mask]
    risk = risk[mask]
    magnitude = magnitude[mask]
    selective = selective_prediction_summary(errors, risk)
    return {
        "aligned_rho": _spearman(risk, errors),
        "partial_rho_control_magnitude": _partial_spearman(
            risk, errors, magnitude
        ),
        "aurc": selective["aurc"],
        "excess_aurc": selective["excess_aurc"],
        "aurc_reduction_vs_random_pct": selective[
            "aurc_reduction_vs_random_pct"
        ],
        **_top_fraction_enrichments(errors, risk),
    }


BOOTSTRAP_METRICS = (
    "aligned_rho",
    "partial_rho_control_magnitude",
    "excess_aurc",
    "aurc_reduction_vs_random_pct",
    "top5_enrichment",
    "top10_enrichment",
    "top20_enrichment",
)


def _clustered_bootstrap(
    group: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    work = group.dropna(
        subset=["true_error_rmse", "risk_axis", "true_effect_l2_norm"]
    ).copy()
    point = _metric_bundle(
        work["true_error_rmse"].to_numpy(dtype=float),
        work["risk_axis"].to_numpy(dtype=float),
        work["true_effect_l2_norm"].to_numpy(dtype=float),
    )
    clusters = []
    for _, cluster in work.groupby("task_key", dropna=False, sort=False):
        clusters.append(
            (
                cluster["true_error_rmse"].to_numpy(dtype=float),
                cluster["risk_axis"].to_numpy(dtype=float),
                cluster["true_effect_l2_norm"].to_numpy(dtype=float),
            )
        )
    result = {
        **{f"{metric}_point": point[metric] for metric in BOOTSTRAP_METRICS},
        "n": int(len(work)),
        "n_task_clusters": int(len(clusters)),
        "n_bootstrap": int(n_bootstrap),
    }
    if len(clusters) < 4 or n_bootstrap <= 0:
        for metric in BOOTSTRAP_METRICS:
            result[f"{metric}_ci_low"] = np.nan
            result[f"{metric}_ci_high"] = np.nan
        return result

    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(n_bootstrap):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        errors = np.concatenate([clusters[i][0] for i in picked])
        risk = np.concatenate([clusters[i][1] for i in picked])
        magnitude = np.concatenate([clusters[i][2] for i in picked])
        metrics = _metric_bundle(errors, risk, magnitude)
        for metric in BOOTSTRAP_METRICS:
            value = metrics[metric]
            if np.isfinite(value):
                samples[metric].append(float(value))
    for metric in BOOTSTRAP_METRICS:
        values = samples[metric]
        if values:
            low, high = np.quantile(values, [0.025, 0.975])
            result[f"{metric}_ci_low"] = float(low)
            result[f"{metric}_ci_high"] = float(high)
        else:
            result[f"{metric}_ci_low"] = np.nan
            result[f"{metric}_ci_high"] = np.nan
    return result


def _evaluate_scores(
    scores: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluated = scores.copy()
    evaluated["risk_axis"] = _risk_axis(evaluated)
    point_rows = []
    bootstrap_rows = []
    retrieval_rows = []
    group_columns = [
        "dataset_name",
        "predictor_name",
        "validation_mode",
        "feature_set",
        "score_name",
    ]
    for group_index, (keys, group) in enumerate(
        evaluated.groupby(group_columns, dropna=False, sort=True)
    ):
        metadata = dict(zip(group_columns, keys))
        boot = _clustered_bootstrap(
            group,
            n_bootstrap=n_bootstrap,
            seed=seed + group_index * 1009,
        )
        point_rows.append(
            {
                **metadata,
                "n": boot["n"],
                "n_task_clusters": boot["n_task_clusters"],
                **{
                    metric: boot[f"{metric}_point"]
                    for metric in BOOTSTRAP_METRICS
                },
            }
        )
        bootstrap_rows.append({**metadata, **boot})
        for fraction in TOP_FRACTIONS:
            prefix = f"top{int(fraction * 100)}_enrichment"
            retrieval_rows.append(
                {
                    **metadata,
                    "top_fraction": fraction,
                    "enrichment": boot[f"{prefix}_point"],
                    "enrichment_ci_low": boot[f"{prefix}_ci_low"],
                    "enrichment_ci_high": boot[f"{prefix}_ci_high"],
                    "n": boot["n"],
                    "n_task_clusters": boot["n_task_clusters"],
                }
            )
    return (
        pd.DataFrame(point_rows),
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(retrieval_rows),
    )


def _within_magnitude_table(scores: pd.DataFrame) -> pd.DataFrame:
    work = scores.copy()
    work["risk_axis"] = _risk_axis(work)
    rows = []
    group_columns = [
        "dataset_name",
        "predictor_name",
        "validation_mode",
        "feature_set",
        "score_name",
    ]
    for keys, group in work.groupby(group_columns, dropna=False, sort=True):
        result = within_magnitude_stratum_rho(
            group,
            "risk_axis",
            "true_error_rmse",
            "true_effect_l2_norm",
            n_bins=4,
        )
        if result.empty:
            continue
        metadata = dict(zip(group_columns, keys))
        for key, value in reversed(list(metadata.items())):
            result.insert(0, key, value)
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _diversity_summary(diversity_rows: pd.DataFrame) -> pd.DataFrame:
    test = diversity_rows[
        diversity_rows["split"].astype(str).eq("test")
    ].copy()
    rows = []
    for (dataset, predictor), group in test.groupby(
        ["dataset_name", "third_predictor"],
        dropna=False,
    ):
        rows.append(
            {
                "dataset_name": dataset,
                "third_predictor": predictor,
                "n": int(len(group)),
                "exact_vector_match_pct": float(
                    100.0 * group["exact_vector_match"].mean()
                ),
                "near_identical_vector_pct": float(
                    100.0 * group["near_identical_vector"].mean()
                ),
                "mean_vector_rmse_vs_contextsim": float(
                    group["vector_rmse_vs_contextsim"].mean()
                ),
                "median_vector_rmse_vs_contextsim": float(
                    group["vector_rmse_vs_contextsim"].median()
                ),
                "mean_cosine_similarity_vs_contextsim": float(
                    group["cosine_similarity_vs_contextsim"].mean()
                ),
                "error_spearman_vs_contextsim": _spearman(
                    group["third_predictor_error"].to_numpy(dtype=float),
                    group["contextsim_error"].to_numpy(dtype=float),
                ),
                "independent_under_50pct_near_identical": bool(
                    group["near_identical_vector"].mean() < 0.5
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    out_path: Path,
    ladder: pd.DataFrame,
    diversity: pd.DataFrame,
) -> dict:
    pert = ladder[
        ladder["predictor_name"].astype(str).eq("PertMeanPredictor")
    ]
    task_only = pert[
        pert["validation_mode"].astype(str).eq("lopo")
        & pert["feature_set"].astype(str).eq("pre_model_task_only")
    ]
    lodo_lopo = pert[
        pert["validation_mode"].astype(str).eq("lodo_lopo")
        & pert["feature_set"].astype(str).eq("full")
    ]
    n_task_only_positive = int(
        (task_only["partial_rho_control_magnitude"] > 0).sum()
    )
    n_lodo_lopo_positive = int(
        (lodo_lopo["partial_rho_control_magnitude"] > 0).sum()
    )
    n_task_only_datasets = int(task_only["dataset_name"].nunique())
    n_lodo_lopo_datasets = int(lodo_lopo["dataset_name"].nunique())
    formal_gate_evaluated = (
        n_task_only_datasets == 7 and n_lodo_lopo_datasets == 7
    )
    if not formal_gate_evaluated:
        task_only_gate = "smoke_not_evaluated"
    elif n_task_only_positive >= 5:
        task_only_gate = "strong_pre_model_task_risk_evidence"
    elif n_task_only_positive == 4:
        task_only_gate = "exploratory_pre_model_task_risk_evidence"
    else:
        task_only_gate = "do_not_use_as_core_evidence"

    control = diversity[
        diversity["third_predictor"].astype(str).eq("Control1NNPredictor")
    ]
    n_control_datasets = int(control["dataset_name"].nunique())
    independent_datasets = int(
        control["independent_under_50pct_near_identical"].sum()
    )
    if n_control_datasets != 7:
        control_role = "smoke_not_evaluated"
    else:
        control_role = (
            "sensitivity_only"
            if independent_datasets < 4
            else "mechanistically_close_secondary_predictor"
        )
    if not formal_gate_evaluated:
        lodo_gate = "smoke_not_evaluated"
    else:
        lodo_gate = (
            "cross_dataset_cross_predictor_signal"
            if n_lodo_lopo_positive >= 4
            else "ordinary_lopo_only"
        )
    decision = {
        "pertmean_pre_model_task_only_positive_datasets": n_task_only_positive,
        "pertmean_pre_model_task_only_total_datasets": n_task_only_datasets,
        "pre_model_task_only_gate": task_only_gate,
        "pertmean_lodo_lopo_full_positive_datasets": n_lodo_lopo_positive,
        "pertmean_lodo_lopo_full_total_datasets": n_lodo_lopo_datasets,
        "lodo_lopo_gate": lodo_gate,
        "control1nn_independent_datasets": independent_datasets,
        "control1nn_total_datasets": n_control_datasets,
        "control1nn_role": control_role,
        "formal_seven_dataset_gate_evaluated": formal_gate_evaluated,
    }
    lines = [
        "# LOPO robustness audit",
        "",
        "## Decision",
        "",
        "- PertMean pre_model_task_only positive partial rho: "
        f"{n_task_only_positive}/{n_task_only_datasets}",
        f"- pre_model_task_only gate: `{task_only_gate}`",
        "- PertMean LODOxLOPO full positive partial rho: "
        f"{n_lodo_lopo_positive}/{n_lodo_lopo_datasets}",
        f"- LODOxLOPO gate: `{lodo_gate}`",
        "- Control1NN datasets below 50% near-identical to ContextSim: "
        f"{independent_datasets}/{n_control_datasets}",
        f"- Control1NN role: `{control_role}`",
        "",
        "## Interpretation boundary",
        "",
        "These experiments test task-risk transfer among retrieval-based predictors. "
        "They do not establish universal transfer to deep predictors such as GEARS or CPA.",
        "",
        "`pre_model_task_only` does not use the target predictor output, but it still "
        "requires fold-local historical perturbation effects and the target context control profile.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return decision


def _write_small_outputs(
    directory: Path,
    ladder: pd.DataFrame,
    bootstrap: pd.DataFrame,
    retrieval: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    diversity: pd.DataFrame,
    within_magnitude: pd.DataFrame,
    normalization_audit: pd.DataFrame,
    provenance: pd.DataFrame,
    status: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ladder.to_csv(directory / "LOPO_BASELINE_LADDER.csv", index=False)
    bootstrap.to_csv(directory / "LOPO_BOOTSTRAP_CI.csv", index=False)
    retrieval.to_csv(directory / "LOPO_BAD_RETRIEVAL.csv", index=False)
    feature_ablation.to_csv(
        directory / "LOPO_FEATURE_ABLATION.csv", index=False
    )
    diversity.to_csv(directory / "LOPO_PREDICTOR_DIVERSITY.csv", index=False)
    within_magnitude.to_csv(
        directory / "LOPO_WITHIN_MAGNITUDE_STRATUM.csv", index=False
    )
    normalization_audit.to_csv(
        directory / "LOPO_NORMALIZATION_AUDIT.csv", index=False
    )
    provenance.to_csv(directory / "LOPO_TRAINING_PROVENANCE.csv", index=False)
    (directory / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    run_dirs: list[Path],
    out_dir: Path,
    third_predictor: str = "PertMeanPredictor",
    validation_mode: str = "both",
    n_bootstrap: int = 1000,
    evidence_dir: Path | None = None,
    seed: int = 5201,
) -> dict:
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    predictors = (
        list(MAIN_THIRD_PREDICTORS)
        if third_predictor == "all"
        else [third_predictor]
    )
    validation_modes = (
        ["lopo", "lodo_lopo"]
        if validation_mode == "both"
        else [validation_mode.replace("-", "_")]
    )

    all_scores = []
    all_diversity = []
    all_normalization_audits = []
    all_provenance = []
    feature_matrices = {}

    for predictor in predictors:
        frames = []
        diversity_frames = []
        for run_dir in run_dirs:
            frame, diversity = build_feature_matrix(run_dir, predictor)
            frames.append(frame)
            diversity_frames.append(diversity)
        base = pd.concat(frames, ignore_index=True)
        normalization_audit = _normalization_audit(base)
        base, normalized_columns = normalize_features_within_group(
            base,
            [column for column in ALL_FEATURES if column in base.columns],
            group_cols=("dataset_name", "fold_id", "predictor_name"),
            reference_splits=("train", "val"),
        )
        feature_path = tables / f"LOPO_FEATURE_MATRIX_{predictor}.csv"
        base.to_csv(feature_path, index=False)
        feature_matrices[predictor] = str(feature_path)

        scores = [_baseline_scores(base, predictor, seed)]
        provenance_frames = []
        for mode in validation_modes:
            for feature_set in FEATURE_SETS:
                learned, provenance = _fit_learned_scores(
                    base,
                    predictor,
                    feature_set,
                    mode,
                    normalized_columns,
                    seed,
                )
                scores.append(learned)
                provenance_frames.append(provenance)
        all_scores.append(pd.concat(scores, ignore_index=True))
        all_diversity.append(pd.concat(diversity_frames, ignore_index=True))
        all_normalization_audits.append(normalization_audit)
        all_provenance.append(
            pd.concat(provenance_frames, ignore_index=True)
        )

    score_table = pd.concat(all_scores, ignore_index=True)
    diversity_rows = pd.concat(all_diversity, ignore_index=True)
    normalization_audit = pd.concat(
        all_normalization_audits, ignore_index=True
    ).drop_duplicates()
    provenance = pd.concat(all_provenance, ignore_index=True)
    score_table.to_csv(tables / "LOPO_ALL_SCORES.csv", index=False)
    diversity_rows.to_csv(tables / "LOPO_DIVERSITY_ROWS.csv", index=False)

    ladder, bootstrap, retrieval = _evaluate_scores(
        score_table,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    feature_ablation = ladder[
        ladder["validation_mode"].isin(["lopo", "lodo_lopo"])
    ].copy()
    diversity = _diversity_summary(diversity_rows)
    within_magnitude = _within_magnitude_table(score_table)

    ladder.to_csv(tables / "LOPO_BASELINE_LADDER.csv", index=False)
    bootstrap.to_csv(tables / "LOPO_BOOTSTRAP_CI.csv", index=False)
    retrieval.to_csv(tables / "LOPO_BAD_RETRIEVAL.csv", index=False)
    feature_ablation.to_csv(
        tables / "LOPO_FEATURE_ABLATION.csv", index=False
    )
    diversity.to_csv(tables / "LOPO_PREDICTOR_DIVERSITY.csv", index=False)
    within_magnitude.to_csv(
        tables / "LOPO_WITHIN_MAGNITUDE_STRATUM.csv", index=False
    )
    normalization_audit.to_csv(
        tables / "LOPO_NORMALIZATION_AUDIT.csv", index=False
    )
    provenance.to_csv(tables / "LOPO_TRAINING_PROVENANCE.csv", index=False)

    decision = _write_report(
        reports / "LOPO_ROBUSTNESS_REPORT.md",
        ladder,
        diversity,
    )
    status = {
        "out_dir": str(out_dir),
        "evidence_dir": str(evidence_dir) if evidence_dir else "",
        "third_predictors": predictors,
        "validation_modes": validation_modes,
        "feature_sets": list(FEATURE_SETS),
        "n_datasets": int(score_table["dataset_name"].nunique()),
        "n_score_rows": int(len(score_table)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "feature_matrices": feature_matrices,
        "frozen_protocol_modified": False,
        "status": "ok",
        "decision": decision,
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if evidence_dir is not None:
        _write_small_outputs(
            evidence_dir,
            ladder,
            bootstrap,
            retrieval,
            feature_ablation,
            diversity,
            within_magnitude,
            normalization_audit,
            provenance,
            status,
        )
        _write_report(
            evidence_dir / "LOPO_ROBUSTNESS_REPORT.md",
            ladder,
            diversity,
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust LOPO/LODOxLOPO validation on unseen predictors."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--third-predictor",
        default="PertMeanPredictor",
        choices=[*THIRD_PREDICTORS, "all"],
    )
    parser.add_argument(
        "--validation-mode",
        default="both",
        choices=["lopo", "lodo-lopo", "both"],
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5201)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.run_dirs,
                args.out_dir,
                third_predictor=args.third_predictor,
                validation_mode=args.validation_mode,
                n_bootstrap=args.bootstrap,
                evidence_dir=args.evidence_dir,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
