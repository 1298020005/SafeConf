#!/usr/bin/env python3
"""Run the preregistered E1-E4 SafeConf risk audits.

The audits consume the frozen PertMean LOPO feature matrix produced by
``run_lopo_third_predictor``. They do not rebuild tasks or modify protocol
v0.2.

E1
    Six-group learned-risk ablation with paired task-cluster bootstrap.
E2
    Dataset-local LOPO with nested, group-cross-fitted magnitude residuals.
E3
    Target/feature permutation nulls and missingness-only diagnostics.
E4
    HistGBT seed and model-configuration stability.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safetrans_confidence.cli.run_lopo_third_predictor import (
    ALL_FEATURES,
    TRAIN_PREDICTORS,
    _metric_bundle,
    _partial_spearman,
    _spearman,
    _top_fraction_enrichments,
    _train_target_rank,
)
from safetrans_confidence.eval.selective_prediction import (
    selective_prediction_summary,
)
from safetrans_confidence.features.normalize import QNORM_SUFFIX

THIRD_PREDICTOR = "PertMeanPredictor"
DEFAULT_SEED = 5201
SEEDS = (42, 123, 256, 314, 500, 789, 1024, 2048, 3141, 5201)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "context": (
        "context_similarity_max",
        "context_similarity_mean",
    ),
    "support": ("perturbation_support_count",),
    "historical": (
        "perturbation_effect_stability",
        "perturbation_effect_variance",
        "historical_residual_risk",
    ),
    "disagreement": (
        "model_disagreement_rmse",
        "model_disagreement_cosine",
    ),
    "ood": (
        "ood_nearest_distance",
        "ood_mean_k_distance",
    ),
    "prediction_output": (
        "prediction_magnitude_deviation",
        "prediction_l2_norm",
        "prediction_abs_mean",
        "prediction_norm_ratio",
    ),
}

CONFIGS: dict[str, dict] = {
    "config_A_shallow_fast": {
        "max_iter": 80,
        "max_leaf_nodes": 8,
        "learning_rate": 0.08,
    },
    "config_B_default": {
        "max_iter": 160,
        "max_leaf_nodes": 12,
        "learning_rate": 0.04,
    },
    "config_C_deeper_slow": {
        "max_iter": 320,
        "max_leaf_nodes": 16,
        "learning_rate": 0.02,
    },
    "config_D_shallower": {
        "max_iter": 160,
        "max_leaf_nodes": 6,
        "learning_rate": 0.04,
    },
    "config_E_deeper": {
        "max_iter": 160,
        "max_leaf_nodes": 24,
        "learning_rate": 0.04,
    },
    "config_F_elasticnet": {"model_type": "elasticnet"},
}


def _feature_group_contract() -> None:
    flat = [feature for group in FEATURE_GROUPS.values() for feature in group]
    if len(ALL_FEATURES) != 14:
        raise AssertionError(f"expected 14 LOPO features, found {len(ALL_FEATURES)}")
    if len(flat) != 14 or len(set(flat)) != 14:
        raise AssertionError("feature groups must contain 14 unique features")
    if set(flat) != set(ALL_FEATURES):
        missing = sorted(set(ALL_FEATURES) - set(flat))
        extra = sorted(set(flat) - set(ALL_FEATURES))
        raise AssertionError(f"feature-group mismatch: missing={missing}, extra={extra}")


def _load_matrix(path: Path) -> pd.DataFrame:
    _feature_group_contract()
    frame = pd.read_csv(path)
    required = {
        "record_id",
        "dataset_name",
        "task_key",
        "fold_id",
        "split",
        "predictor_name",
        "true_error_rmse",
        "true_effect_l2_norm",
        *ALL_FEATURES,
        *[f"{feature}{QNORM_SUFFIX}" for feature in ALL_FEATURES],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature matrix is missing columns: {missing}")
    if frame["dataset_name"].nunique() != 7:
        raise ValueError("formal preregistration requires exactly seven datasets")
    return frame


def _qnorm_features(raw_features: Iterable[str]) -> list[str]:
    return [f"{feature}{QNORM_SUFFIX}" for feature in raw_features]


def _make_histgbt(seed: int, params: dict | None = None):
    values = {
        "max_iter": 160,
        "max_leaf_nodes": 12,
        "learning_rate": 0.04,
        "l2_regularization": 0.05,
        "random_state": seed,
    }
    values.update(params or {})
    return HistGradientBoostingRegressor(**values)


def _make_config_model(config_name: str, seed: int):
    config = CONFIGS[config_name]
    if config.get("model_type") == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=20000,
                random_state=seed,
            ),
        )
    return _make_histgbt(seed, config)


def _training_source(
    base: pd.DataFrame,
    heldout_dataset: str | None = None,
    dataset_local: str | None = None,
    fold_id: int | None = None,
) -> pd.DataFrame:
    mask = (
        base["predictor_name"].isin(TRAIN_PREDICTORS)
        & base["split"].isin(["train", "val"])
    )
    if heldout_dataset is not None:
        mask &= base["dataset_name"].astype(str).ne(heldout_dataset)
    if dataset_local is not None:
        mask &= base["dataset_name"].astype(str).eq(dataset_local)
    if fold_id is not None:
        mask &= pd.to_numeric(base["fold_id"], errors="coerce").eq(fold_id)
    source = base[mask].copy()
    if source.empty:
        raise ValueError("empty training source")
    return source


def _target_rows(
    base: pd.DataFrame,
    dataset: str | None = None,
    fold_id: int | None = None,
) -> pd.DataFrame:
    mask = (
        base["predictor_name"].astype(str).eq(THIRD_PREDICTOR)
        & base["split"].astype(str).eq("test")
    )
    if dataset is not None:
        mask &= base["dataset_name"].astype(str).eq(dataset)
    if fold_id is not None:
        mask &= pd.to_numeric(base["fold_id"], errors="coerce").eq(fold_id)
    return base[mask].copy()


def _fit_rank_model(
    source: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    model_factory: Callable[[int], object] | None = None,
    target_values: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    train = source.copy()
    if target_values is None:
        train["_audit_target"] = _train_target_rank(train)
    else:
        train["_audit_target"] = np.asarray(target_values, dtype=float)
    train = train.dropna(subset=["_audit_target"])
    factory = model_factory or (lambda value: _make_histgbt(value))
    model = factory(seed)
    model.fit(
        train[feature_columns].fillna(0.5).to_numpy(),
        train["_audit_target"].to_numpy(dtype=float),
    )
    return np.asarray(
        model.predict(target[feature_columns].fillna(0.5).to_numpy()),
        dtype=float,
    )


def _score_frame(target: pd.DataFrame, score: np.ndarray, name: str) -> pd.DataFrame:
    columns = [
        "record_id",
        "dataset_name",
        "task_key",
        "fold_id",
        "split",
        "predictor_name",
        "true_error_rmse",
        "true_effect_l2_norm",
    ]
    out = target[columns].copy()
    out["score_name"] = name
    out["risk_axis"] = np.asarray(score, dtype=float)
    return out


def _score_variant(
    base: pd.DataFrame,
    raw_features: list[str],
    validation_mode: str,
    seed: int,
    score_name: str,
    model_factory: Callable[[int], object] | None = None,
) -> pd.DataFrame:
    feature_columns = _qnorm_features(raw_features)
    rows = []
    if validation_mode == "lopo":
        folds = sorted(
            pd.to_numeric(base["fold_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        for fold_id in folds:
            source = _training_source(base, fold_id=fold_id)
            target = _target_rows(base, fold_id=fold_id)
            _assert_no_task_overlap(source, target)
            score = _fit_rank_model(
                source,
                target,
                feature_columns,
                seed + 1009 * fold_id,
                model_factory=model_factory,
            )
            rows.append(_score_frame(target, score, score_name))
    elif validation_mode == "lodo_lopo":
        for dataset in sorted(base["dataset_name"].dropna().astype(str).unique()):
            folds = sorted(
                pd.to_numeric(base["fold_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            for fold_id in folds:
                source = _training_source(
                    base,
                    heldout_dataset=dataset,
                    fold_id=fold_id,
                )
                target = _target_rows(
                    base,
                    dataset=dataset,
                    fold_id=fold_id,
                )
                _assert_no_task_overlap(source, target)
                score = _fit_rank_model(
                    source,
                    target,
                    feature_columns,
                    seed + 1009 * fold_id,
                    model_factory=model_factory,
                )
                rows.append(_score_frame(target, score, score_name))
    else:
        raise ValueError(validation_mode)
    return pd.concat(rows, ignore_index=True)


def _assert_no_task_overlap(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
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
            "outer-fold task leakage between source and target: "
            f"{sorted(overlap)[:3]}"
        )


def _point_metrics(group: pd.DataFrame, risk_col: str = "risk_axis") -> dict:
    return _metric_bundle(
        pd.to_numeric(group["true_error_rmse"], errors="coerce").to_numpy(),
        pd.to_numeric(group[risk_col], errors="coerce").to_numpy(),
        pd.to_numeric(group["true_effect_l2_norm"], errors="coerce").to_numpy(),
    )


def _paired_task_bootstrap_delta(
    full: pd.DataFrame,
    alternative: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    keys = [
        "record_id",
        "dataset_name",
        "task_key",
        "true_error_rmse",
        "true_effect_l2_norm",
    ]
    merged = full[keys + ["risk_axis"]].merge(
        alternative[["record_id", "risk_axis"]],
        on="record_id",
        how="inner",
        suffixes=("_full", "_alternative"),
        validate="one_to_one",
    )
    if len(merged) != len(full) or len(merged) != len(alternative):
        raise ValueError("paired score tables are not perfectly aligned")
    cluster_positions = [
        np.asarray(positions, dtype=int)
        for positions in merged.groupby("task_key", sort=False).indices.values()
    ]
    errors = pd.to_numeric(
        merged["true_error_rmse"], errors="coerce"
    ).to_numpy(dtype=float)
    magnitude = pd.to_numeric(
        merged["true_effect_l2_norm"], errors="coerce"
    ).to_numpy(dtype=float)
    full_risk = pd.to_numeric(
        merged["risk_axis_full"], errors="coerce"
    ).to_numpy(dtype=float)
    alternative_risk = pd.to_numeric(
        merged["risk_axis_alternative"], errors="coerce"
    ).to_numpy(dtype=float)
    point_full = _metric_bundle(errors, full_risk, magnitude)
    point_alt = _metric_bundle(errors, alternative_risk, magnitude)
    metrics = {
        "partial_rho_control_magnitude": (
            "partial_rho_control_magnitude",
            1.0,
        ),
        "aurc_reduction_vs_random_pct": (
            "aurc_reduction_vs_random_pct",
            1.0,
        ),
        "top10_enrichment": ("top10_enrichment", 1.0),
    }
    result = {
        "n": int(len(merged)),
        "n_task_clusters": int(len(cluster_positions)),
        "n_bootstrap": int(n_bootstrap),
    }
    for output_name, (metric_name, direction) in metrics.items():
        result[f"delta_{output_name}_point"] = direction * (
            point_full[metric_name] - point_alt[metric_name]
        )
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in metrics}
    singleton_clusters = all(len(positions) == 1 for positions in cluster_positions)
    singleton_positions = (
        np.asarray([positions[0] for positions in cluster_positions], dtype=int)
        if singleton_clusters
        else None
    )
    for _ in range(n_bootstrap):
        picks = rng.integers(
            0,
            len(cluster_positions),
            size=len(cluster_positions),
        )
        sample_positions = (
            singleton_positions[picks]
            if singleton_positions is not None
            else np.concatenate([cluster_positions[index] for index in picks])
        )
        sample_full = _metric_bundle(
            errors[sample_positions],
            full_risk[sample_positions],
            magnitude[sample_positions],
        )
        sample_alt = _metric_bundle(
            errors[sample_positions],
            alternative_risk[sample_positions],
            magnitude[sample_positions],
        )
        for output_name, (metric_name, direction) in metrics.items():
            value = direction * (
                sample_full[metric_name] - sample_alt[metric_name]
            )
            if np.isfinite(value):
                samples[output_name].append(float(value))
    for name, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        result[f"delta_{name}_ci_low"] = float(low)
        result[f"delta_{name}_ci_high"] = float(high)
    return result


def run_e1(
    base: pd.DataFrame,
    out_dir: Path,
    n_bootstrap: int,
    seed: int,
    n_jobs: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_frames = []
    delta_rows = []
    full_features = list(ALL_FEATURES)
    for mode in ("lopo", "lodo_lopo"):
        full = _score_variant(
            base,
            full_features,
            mode,
            seed,
            score_name=f"{mode}_full",
        )
        full["validation_mode"] = mode
        full["dropped_group"] = "none"
        score_frames.append(full)
        for group_name, group_features in FEATURE_GROUPS.items():
            kept = [
                feature
                for feature in full_features
                if feature not in set(group_features)
            ]
            dropped = _score_variant(
                base,
                kept,
                mode,
                seed,
                score_name=f"{mode}_drop_{group_name}",
            )
            dropped["validation_mode"] = mode
            dropped["dropped_group"] = group_name
            score_frames.append(dropped)
            datasets = sorted(base["dataset_name"].unique())
            bootstrap_results = Parallel(n_jobs=n_jobs)(
                delayed(_paired_task_bootstrap_delta)(
                    full[full["dataset_name"].astype(str).eq(str(dataset))],
                    dropped[
                        dropped["dataset_name"].astype(str).eq(str(dataset))
                    ],
                    n_bootstrap=n_bootstrap,
                    seed=seed
                    + 1009 * list(FEATURE_GROUPS).index(group_name)
                    + 7919 * (0 if mode == "lopo" else 1)
                    + 17 * datasets.index(dataset),
            )
            for dataset in datasets
            )
            for dataset, result in zip(datasets, bootstrap_results):
                delta_rows.append(
                    {
                        "dataset_name": dataset,
                        "validation_mode": mode,
                        "dropped_group": group_name,
                        "dropped_features": ";".join(group_features),
                        "n_features_kept": len(kept),
                        **result,
                    }
                )
    scores = pd.concat(score_frames, ignore_index=True)
    deltas = pd.DataFrame(delta_rows)
    group_gate = (
        deltas[deltas["validation_mode"].eq("lopo")]
        .assign(
            ci_positive=lambda frame: frame[
                "delta_partial_rho_control_magnitude_ci_low"
            ]
            > 0
        )
        .groupby("dropped_group")["ci_positive"]
        .sum()
        .sort_values(ascending=False)
    )
    passing_groups = group_gate[group_gate >= 3].index.tolist()
    decision = {
        "passing_groups": passing_groups,
        "n_passing_groups": len(passing_groups),
        "gate": "pass" if len(passing_groups) >= 2 else "fail",
        "gate_note": (
            "Unadjusted paired bootstrap CIs are an across-dataset consistency "
            "gate, not multiplicity-adjusted inference."
        ),
    }
    scores.to_csv(out_dir / "E1_ALL_SCORES.csv", index=False)
    deltas.to_csv(out_dir / "E1_GROUP_ABLATION_DELTAS.csv", index=False)
    pd.DataFrame(
        [
            {
                "dropped_group": group,
                "n_datasets_ci_positive_lopo": int(count),
                "passes_three_of_seven": bool(count >= 3),
            }
            for group, count in group_gate.items()
        ]
    ).to_csv(out_dir / "E1_GROUP_GATE.csv", index=False)
    return decision


@dataclass
class _NaturalSplineCalibrator:
    design_info: object | None = None
    coefficients: np.ndarray | None = None
    x_min: float | None = None
    x_max: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NaturalSplineCalibrator":
        from patsy import dmatrix

        x_values = np.asarray(x, dtype=float)
        self.x_min = float(np.nanmin(x_values))
        self.x_max = float(np.nanmax(x_values))
        design = dmatrix(
            "cr(x, df=4)",
            {"x": x_values},
            return_type="dataframe",
        )
        self.design_info = design.design_info
        self.coefficients = np.linalg.lstsq(
            np.asarray(design, dtype=float),
            np.asarray(y, dtype=float),
            rcond=None,
        )[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        from patsy import build_design_matrices

        if (
            self.design_info is None
            or self.coefficients is None
            or self.x_min is None
            or self.x_max is None
        ):
            raise RuntimeError("spline calibrator is not fitted")
        # Patsy's natural spline basis rejects values beyond the fitted knots.
        # Clipping also makes the sensitivity model's extrapolation policy match
        # the primary isotonic model's out_of_bounds="clip" behavior.
        x_values = np.clip(
            np.asarray(x, dtype=float),
            self.x_min,
            self.x_max,
        )
        design = build_design_matrices(
            [self.design_info],
            {"x": x_values},
        )[0]
        return np.asarray(design, dtype=float) @ self.coefficients


def _make_calibrator(method: str):
    if method == "isotonic":
        return IsotonicRegression(increasing=True, out_of_bounds="clip")
    if method == "natural_spline_df4":
        return _NaturalSplineCalibrator()
    raise ValueError(method)


def _robust_z(values: pd.Series, reference: pd.Series) -> np.ndarray:
    values_num = pd.to_numeric(values, errors="coerce")
    ref = pd.to_numeric(reference, errors="coerce")
    median = float(ref.median())
    scale = float(ref.quantile(0.75) - ref.quantile(0.25))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(ref.std())
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return ((values_num.fillna(median) - median) / scale).to_numpy(dtype=float)


def _cross_fitted_expected_error(
    source: pd.DataFrame,
    magnitude_z: np.ndarray,
    method: str,
    n_splits: int = 3,
) -> tuple[np.ndarray, pd.DataFrame]:
    groups = source["task_key"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(source), np.nan, dtype=float)
    audit_rows = []
    for inner_fold, (train_idx, holdout_idx) in enumerate(
        splitter.split(np.zeros(len(source)), groups=groups)
    ):
        train_tasks = set(groups[train_idx])
        holdout_tasks = set(groups[holdout_idx])
        overlap = train_tasks & holdout_tasks
        if overlap:
            raise AssertionError(f"inner task leakage: {sorted(overlap)[:3]}")
        calibrator = _make_calibrator(method)
        calibrator.fit(
            magnitude_z[train_idx],
            source.iloc[train_idx]["true_error_rmse"].to_numpy(dtype=float),
        )
        oof[holdout_idx] = calibrator.predict(magnitude_z[holdout_idx])
        audit_rows.append(
            {
                "inner_fold": inner_fold,
                "n_train_rows": len(train_idx),
                "n_holdout_rows": len(holdout_idx),
                "n_train_tasks": len(train_tasks),
                "n_holdout_tasks": len(holdout_tasks),
                "task_overlap": len(overlap),
            }
        )
    if not np.isfinite(oof).all():
        raise AssertionError("OOF expected errors contain non-finite values")
    return oof, pd.DataFrame(audit_rows)


def _e2_outer_fold(
    base: pd.DataFrame,
    dataset: str,
    fold_id: int,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = _training_source(
        base,
        dataset_local=dataset,
        fold_id=fold_id,
    ).reset_index(drop=True)
    target = _target_rows(base, dataset=dataset, fold_id=fold_id).reset_index(
        drop=True
    )
    if target.empty:
        raise ValueError(f"empty E2 target: {dataset} fold {fold_id}")
    source_mag_z = _robust_z(
        source["prediction_l2_norm"],
        source["prediction_l2_norm"],
    )
    target_mag_z = _robust_z(
        target["prediction_l2_norm"],
        source["prediction_l2_norm"],
    )
    oof_expected, inner_audit = _cross_fitted_expected_error(
        source,
        source_mag_z,
        method,
    )
    residual_train = (
        source["true_error_rmse"].to_numpy(dtype=float) - oof_expected
    )
    feature_columns = _qnorm_features(ALL_FEATURES)
    model = _make_histgbt(seed)
    model.fit(
        source[feature_columns].fillna(0.5).to_numpy(),
        residual_train,
    )
    predicted_residual = model.predict(
        target[feature_columns].fillna(0.5).to_numpy()
    )
    calibrator = _make_calibrator(method)
    calibrator.fit(
        source_mag_z,
        source["true_error_rmse"].to_numpy(dtype=float),
    )
    expected_test = calibrator.predict(target_mag_z)
    result = target[
        [
            "record_id",
            "dataset_name",
            "task_key",
            "fold_id",
            "true_error_rmse",
            "true_effect_l2_norm",
            "prediction_l2_norm",
        ]
    ].copy()
    result["calibration_method"] = method
    result["magnitude_z"] = target_mag_z
    result["expected_error_magnitude"] = expected_test
    result["true_residual_error"] = (
        result["true_error_rmse"].to_numpy(dtype=float) - expected_test
    )
    result["predicted_residual_error"] = predicted_residual
    result["combined_predicted_error"] = expected_test + predicted_residual
    inner_audit.insert(0, "calibration_method", method)
    inner_audit.insert(0, "fold_id", fold_id)
    inner_audit.insert(0, "dataset_name", dataset)
    return result, inner_audit


def _paired_bootstrap_e2(
    group: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    cluster_positions = [
        np.asarray(positions, dtype=int)
        for positions in group.reset_index(drop=True)
        .groupby("task_key", sort=False)
        .indices.values()
    ]
    group = group.reset_index(drop=True)

    def metrics(frame: pd.DataFrame) -> tuple[float, float]:
        residual_partial = _partial_spearman(
            frame["predicted_residual_error"].to_numpy(dtype=float),
            frame["true_residual_error"].to_numpy(dtype=float),
            frame["true_effect_l2_norm"].to_numpy(dtype=float),
        )
        errors = frame["true_error_rmse"].to_numpy(dtype=float)
        magnitude_summary = selective_prediction_summary(
            errors,
            frame["expected_error_magnitude"].to_numpy(dtype=float),
        )
        combined_summary = selective_prediction_summary(
            errors,
            frame["combined_predicted_error"].to_numpy(dtype=float),
        )
        aurc_improvement = (
            magnitude_summary["aurc"] - combined_summary["aurc"]
        )
        return residual_partial, aurc_improvement

    point_partial, point_aurc = metrics(group)
    rng = np.random.default_rng(seed)
    partial_samples = []
    aurc_samples = []
    singleton_clusters = all(len(positions) == 1 for positions in cluster_positions)
    singleton_positions = (
        np.asarray([positions[0] for positions in cluster_positions], dtype=int)
        if singleton_clusters
        else None
    )
    for _ in range(n_bootstrap):
        picks = rng.integers(
            0,
            len(cluster_positions),
            size=len(cluster_positions),
        )
        sample_positions = (
            singleton_positions[picks]
            if singleton_positions is not None
            else np.concatenate([cluster_positions[index] for index in picks])
        )
        sample = group.iloc[sample_positions]
        partial, aurc = metrics(sample)
        if np.isfinite(partial):
            partial_samples.append(partial)
        if np.isfinite(aurc):
            aurc_samples.append(aurc)
    partial_low, partial_high = (
        np.quantile(partial_samples, [0.025, 0.975])
        if partial_samples
        else (np.nan, np.nan)
    )
    aurc_low, aurc_high = (
        np.quantile(aurc_samples, [0.025, 0.975])
        if aurc_samples
        else (np.nan, np.nan)
    )
    return {
        "n": int(len(group)),
        "n_task_clusters": len(cluster_positions),
        "n_bootstrap": n_bootstrap,
        "residual_partial_rho_point": point_partial,
        "residual_partial_rho_ci_low": partial_low,
        "residual_partial_rho_ci_high": partial_high,
        "aurc_improvement_magnitude_minus_combined_point": point_aurc,
        "aurc_improvement_magnitude_minus_combined_ci_low": aurc_low,
        "aurc_improvement_magnitude_minus_combined_ci_high": aurc_high,
    }


def run_e2(
    base: pd.DataFrame,
    out_dir: Path,
    n_bootstrap: int,
    seed: int,
    n_jobs: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    audit_rows = []
    for dataset in sorted(base["dataset_name"].unique()):
        folds = sorted(
            pd.to_numeric(
                base[base["dataset_name"].astype(str).eq(str(dataset))][
                    "fold_id"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )
        for fold_id in folds:
            for method in ("isotonic", "natural_spline_df4"):
                scores, audit = _e2_outer_fold(
                    base,
                    str(dataset),
                    int(fold_id),
                    method,
                    seed,
                )
                score_rows.append(scores)
                audit_rows.append(audit)
    scores = pd.concat(score_rows, ignore_index=True)
    audit = pd.concat(audit_rows, ignore_index=True)
    grouped_scores = list(
        scores.groupby(["dataset_name", "calibration_method"], sort=True)
    )
    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(_paired_bootstrap_e2)(
            group,
            n_bootstrap=n_bootstrap,
            seed=seed + index * 1009,
        )
        for index, (_, group) in enumerate(grouped_scores)
    )
    summary_rows = [
        {
            "dataset_name": dataset,
            "calibration_method": method,
            **result,
        }
        for ((dataset, method), _), result in zip(
            grouped_scores,
            bootstrap_results,
        )
    ]
    summary = pd.DataFrame(summary_rows)
    primary = summary[summary["calibration_method"].eq("isotonic")]
    n_partial = int((primary["residual_partial_rho_ci_low"] > 0).sum())
    n_aurc = int(
        (
            primary[
                "aurc_improvement_magnitude_minus_combined_ci_low"
            ]
            > 0
        ).sum()
    )
    decision = {
        "isotonic_partial_ci_positive_datasets": n_partial,
        "isotonic_aurc_improvement_ci_positive_datasets": n_aurc,
        "partial_gate": "pass" if n_partial >= 4 else "fail",
        "aurc_gate": "pass" if n_aurc >= 4 else "fail",
        "overall_gate": "pass" if n_partial >= 4 and n_aurc >= 4 else "fail",
    }
    scores.to_csv(out_dir / "E2_TEST_SCORES.csv", index=False)
    audit.to_csv(out_dir / "E2_INNER_SPLIT_AUDIT.csv", index=False)
    summary.to_csv(out_dir / "E2_MAGNITUDE_RESIDUAL_SUMMARY.csv", index=False)
    return decision


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    observed = values[finite]
    if not len(observed):
        return pd.Series(result, index=p_values.index)
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(observed), dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[finite] = restored
    return pd.Series(result, index=p_values.index)


def _empirical_upper_tail_p(point: float, null_values: np.ndarray) -> float:
    finite_null = np.asarray(null_values, dtype=float)
    finite_null = finite_null[np.isfinite(finite_null)]
    if not np.isfinite(point) or not len(finite_null):
        return float("nan")
    return float(
        (1 + int(np.sum(finite_null >= point))) / (1 + len(finite_null))
    )


def _shuffle_task_paired_targets(
    source: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    result = pd.Series(np.nan, index=source.index, dtype=float)
    for _, group in source.groupby(["dataset_name", "fold_id"], sort=False):
        task_predictor = group.pivot_table(
            index="task_key",
            columns="predictor_name",
            values="true_error_rmse",
            aggfunc="first",
        )
        tasks = task_predictor.index.to_numpy()
        shuffled = rng.permutation(tasks)
        mapping = dict(zip(tasks, shuffled))
        lookup = {
            (str(task), str(predictor)): float(value)
            for task, row in task_predictor.iterrows()
            for predictor, value in row.items()
            if pd.notna(value)
        }
        for idx, row in group.iterrows():
            source_task = mapping[row["task_key"]]
            result.loc[idx] = lookup[
                (str(source_task), str(row["predictor_name"]))
            ]
    if result.isna().any():
        raise AssertionError("task-paired target permutation produced missing values")
    return result


def _rank_target_values(
    source: pd.DataFrame,
    values: pd.Series | np.ndarray,
) -> pd.Series:
    ranked = pd.Series(np.nan, index=source.index, dtype=float)
    raw = pd.Series(np.asarray(values, dtype=float), index=source.index)
    for _, indices_obj in source.groupby(
        ["dataset_name", "fold_id", "predictor_name"],
        sort=False,
    ).groups.items():
        indices = list(indices_obj)
        ranked.loc[indices] = raw.loc[indices].rank(
            method="average",
            pct=True,
        )
    return ranked


def _jointly_shuffle_features(
    source: pd.DataFrame,
    feature_columns: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    shuffled = source.copy()
    for _, indices_obj in source.groupby(
        ["dataset_name", "fold_id", "predictor_name"],
        sort=False,
    ).groups.items():
        indices = np.asarray(list(indices_obj))
        permuted = rng.permutation(indices)
        shuffled.loc[indices, feature_columns] = source.loc[
            permuted, feature_columns
        ].to_numpy()
    return shuffled


def _score_from_source(
    source: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    target_values: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    score = _fit_rank_model(
        source,
        target,
        feature_columns,
        seed,
        target_values=target_values,
    )
    return _score_frame(target, score, "audit")


def _score_foldwise_from_source(
    source: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    target_values: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    target_series = (
        pd.Series(np.asarray(target_values, dtype=float), index=source.index)
        if target_values is not None
        else None
    )
    rows = []
    folds = sorted(
        pd.to_numeric(target["fold_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for fold_id in folds:
        source_fold = source[
            pd.to_numeric(source["fold_id"], errors="coerce").eq(fold_id)
        ]
        target_fold = target[
            pd.to_numeric(target["fold_id"], errors="coerce").eq(fold_id)
        ]
        _assert_no_task_overlap(source_fold, target_fold)
        fold_values = (
            target_series.loc[source_fold.index] if target_series is not None else None
        )
        rows.append(
            _score_from_source(
                source_fold,
                target_fold,
                feature_columns,
                seed + 1009 * fold_id,
                target_values=fold_values,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _partial_by_dataset(scores: pd.DataFrame) -> dict[str, float]:
    return {
        str(dataset): _point_metrics(group)[
            "partial_rho_control_magnitude"
        ]
        for dataset, group in scores.groupby("dataset_name", sort=True)
    }


def _missingness_columns(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    columns = []
    for feature in ALL_FEATURES:
        column = f"{feature}__missing"
        out[column] = pd.to_numeric(out[feature], errors="coerce").isna().astype(float)
        columns.append(column)
    return out, columns


def _paired_bootstrap_score_delta(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    result = _paired_task_bootstrap_delta(
        primary,
        comparator,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        key: value
        for key, value in result.items()
        if "partial_rho" in key or key in {"n", "n_task_clusters", "n_bootstrap"}
    }


def run_e3(
    base: pd.DataFrame,
    out_dir: Path,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    qnorm = _qnorm_features(ALL_FEATURES)
    source = _training_source(base)
    target = _target_rows(base)
    observed = _score_foldwise_from_source(source, target, qnorm, seed)
    observed_partial = _partial_by_dataset(observed)

    base_missing, missing_columns = _missingness_columns(base)
    source_missing = _training_source(base_missing)
    target_missing = _target_rows(base_missing)
    missingness_only = _score_foldwise_from_source(
        source_missing,
        target_missing,
        missing_columns,
        seed,
    )
    plus_columns = qnorm + missing_columns
    full_plus_missingness = _score_foldwise_from_source(
        source_missing,
        target_missing,
        plus_columns,
        seed,
    )
    missing_observed_partial = _partial_by_dataset(missingness_only)

    null_rows = []
    for permutation in range(n_permutations):
        if permutation % 10 == 0:
            print(
                f"E3 permutation {permutation}/{n_permutations}",
                flush=True,
            )
        rng = np.random.default_rng(seed + permutation * 1009)
        shuffled_targets = _rank_target_values(
            source,
            _shuffle_task_paired_targets(source, rng),
        )
        target_null = _score_foldwise_from_source(
            source,
            target,
            qnorm,
            seed + permutation,
            target_values=shuffled_targets,
        )
        missing_null = _score_foldwise_from_source(
            source_missing,
            target_missing,
            missing_columns,
            seed + permutation,
            target_values=shuffled_targets,
        )
        shuffled_source = _jointly_shuffle_features(source, qnorm, rng)
        feature_null = _score_foldwise_from_source(
            shuffled_source,
            target,
            qnorm,
            seed + permutation,
        )
        target_partial = _partial_by_dataset(target_null)
        feature_partial = _partial_by_dataset(feature_null)
        missing_partial = _partial_by_dataset(missing_null)
        for dataset in sorted(observed_partial):
            null_rows.extend(
                [
                    {
                        "dataset_name": dataset,
                        "permutation": permutation,
                        "null_type": "shuffled_train_targets",
                        "partial_rho": target_partial[dataset],
                    },
                    {
                        "dataset_name": dataset,
                        "permutation": permutation,
                        "null_type": "jointly_shuffled_train_features",
                        "partial_rho": feature_partial[dataset],
                    },
                    {
                        "dataset_name": dataset,
                        "permutation": permutation,
                        "null_type": "missingness_shuffled_train_targets",
                        "partial_rho": missing_partial[dataset],
                    },
                ]
            )
    nulls = pd.DataFrame(null_rows)
    p_rows = []
    for dataset in sorted(observed_partial):
        comparisons = {
            "shuffled_train_targets": observed_partial[dataset],
            "jointly_shuffled_train_features": observed_partial[dataset],
            "missingness_shuffled_train_targets": missing_observed_partial[dataset],
        }
        for null_type, point in comparisons.items():
            values = nulls[
                nulls["dataset_name"].astype(str).eq(dataset)
                & nulls["null_type"].eq(null_type)
            ]["partial_rho"].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            empirical_p = _empirical_upper_tail_p(point, values)
            p_rows.append(
                {
                    "dataset_name": dataset,
                    "comparison": null_type,
                    "observed_partial_rho": point,
                    "n_finite_null_permutations": int(len(finite_values)),
                    "null_mean_partial_rho": (
                        float(np.mean(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "null_sd_partial_rho": (
                        float(np.std(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "empirical_p_one_sided": empirical_p,
                }
            )
    p_values = pd.DataFrame(p_rows)
    p_values["bh_fdr"] = (
        p_values.groupby("comparison")["empirical_p_one_sided"]
        .transform(_bh_adjust)
        .astype(float)
    )

    delta_rows = []
    for index, dataset in enumerate(sorted(observed_partial)):
        primary_ds = full_plus_missingness[
            full_plus_missingness["dataset_name"].astype(str).eq(dataset)
        ]
        comparator_ds = observed[
            observed["dataset_name"].astype(str).eq(dataset)
        ]
        delta_rows.append(
            {
                "dataset_name": dataset,
                **_paired_bootstrap_score_delta(
                    primary_ds,
                    comparator_ds,
                    n_bootstrap=n_bootstrap,
                    seed=seed + index * 1009,
                ),
            }
        )
    missing_delta = pd.DataFrame(delta_rows)
    target_q = p_values[
        p_values["comparison"].eq("shuffled_train_targets")
    ]["bh_fdr"]
    feature_q = p_values[
        p_values["comparison"].eq("jointly_shuffled_train_features")
    ]["bh_fdr"]
    missing_q = p_values[
        p_values["comparison"].eq("missingness_shuffled_train_targets")
    ]["bh_fdr"]
    n_missing_delta = int(
        (
            missing_delta[
                "delta_partial_rho_control_magnitude_ci_low"
            ]
            > 0
        ).sum()
    )
    decision = {
        "observed_vs_target_null_fdr_significant_datasets": int((target_q < 0.05).sum()),
        "observed_vs_feature_null_fdr_significant_datasets": int((feature_q < 0.05).sum()),
        "missingness_only_fdr_significant_datasets": int((missing_q < 0.05).sum()),
        "full_plus_missingness_ci_positive_datasets": n_missing_delta,
        "target_null_gate": "pass" if int((target_q < 0.05).sum()) >= 5 else "fail",
        "feature_null_gate": "pass" if int((feature_q < 0.05).sum()) >= 5 else "fail",
        "missingness_gate": (
            "pass"
            if int((missing_q < 0.05).sum()) == 0 and n_missing_delta < 3
            else "diagnose"
        ),
    }
    observed.assign(model="current_full_values").to_csv(
        out_dir / "E3_OBSERVED_FULL_SCORES.csv", index=False
    )
    missingness_only.assign(model="missingness_only").to_csv(
        out_dir / "E3_MISSINGNESS_ONLY_SCORES.csv", index=False
    )
    full_plus_missingness.assign(model="full_values_plus_missingness").to_csv(
        out_dir / "E3_FULL_PLUS_MISSINGNESS_SCORES.csv", index=False
    )
    nulls.to_csv(out_dir / "E3_PERMUTATION_NULLS.csv", index=False)
    p_values.to_csv(out_dir / "E3_EMPIRICAL_PVALUES.csv", index=False)
    missing_delta.to_csv(
        out_dir / "E3_MISSINGNESS_PAIRED_DELTA.csv", index=False
    )
    return decision


def run_e4(base: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for model_seed in SEEDS:
        scores = _score_variant(
            base,
            list(ALL_FEATURES),
            "lopo",
            model_seed,
            score_name=f"seed_{model_seed}",
        )
        for dataset, group in scores.groupby("dataset_name", sort=True):
            seed_rows.append(
                {
                    "dataset_name": dataset,
                    "seed": model_seed,
                    **_point_metrics(group),
                }
            )
    seed_results = pd.DataFrame(seed_rows)
    seed_summary = (
        seed_results.groupby("dataset_name")[
            "partial_rho_control_magnitude"
        ]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
            n_positive=lambda values: int((values > 0).sum()),
        )
        .reset_index()
    )

    config_rows = []
    for config_name in CONFIGS:
        scores = _score_variant(
            base,
            list(ALL_FEATURES),
            "lopo",
            seed,
            score_name=config_name,
            model_factory=lambda model_seed, name=config_name: _make_config_model(
                name, model_seed
            ),
        )
        for dataset, group in scores.groupby("dataset_name", sort=True):
            config_rows.append(
                {
                    "dataset_name": dataset,
                    "config_name": config_name,
                    **_point_metrics(group),
                }
            )
    config_results = pd.DataFrame(config_rows)
    config_summary = (
        config_results.assign(
            positive=lambda frame: frame[
                "partial_rho_control_magnitude"
            ]
            > 0
        )
        .groupby("dataset_name")["positive"]
        .sum()
        .rename("n_positive_configs")
        .reset_index()
    )
    n_seed_stable = int((seed_summary["n_positive"] >= 9).sum())
    n_config_stable = int((config_summary["n_positive_configs"] >= 3).sum())
    decision = {
        "datasets_with_at_least_9_of_10_positive_seeds": n_seed_stable,
        "datasets_with_at_least_3_of_6_positive_configs": n_config_stable,
        "seed_gate": "pass" if n_seed_stable >= 5 else "fail",
        "config_gate": "pass" if n_config_stable >= 5 else "fail",
        "note": (
            "HistGBT is close to deterministic; configuration sensitivity is "
            "more informative than seed sensitivity."
        ),
    }
    seed_results.to_csv(out_dir / "E4_SEED_RESULTS.csv", index=False)
    seed_summary.to_csv(out_dir / "E4_SEED_SUMMARY.csv", index=False)
    config_results.to_csv(out_dir / "E4_CONFIG_RESULTS.csv", index=False)
    config_summary.to_csv(out_dir / "E4_CONFIG_SUMMARY.csv", index=False)
    return decision


def _write_status(
    out_dir: Path,
    matrix_path: Path,
    decisions: dict,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "unknown"
    status = {
        "status": "ok",
        "code_commit": code_commit,
        "feature_matrix": str(matrix_path),
        "feature_matrix_rows": int(
            sum(1 for _ in matrix_path.open("r", encoding="utf-8")) - 1
        ),
        "experiments": list(decisions),
        "decisions": decisions,
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "n_jobs": args.jobs,
        "seed": args.seed,
        "elapsed_seconds": elapsed_seconds,
        "frozen_protocol_modified": False,
        "learned_model_outer_fold_policy": (
            "one model per outer fold; source=train+val and target=test from "
            "the same fold only"
        ),
        "source_target_task_overlap_required": 0,
        "preregistration_correction": (
            "Permutation empirical p-values are expected to be small when the "
            "observed model exceeds its null; the supplied p>0.10 direction "
            "was corrected before execution."
        ),
    }
    (out_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_report(out_dir: Path, decisions: dict) -> Path:
    lines = [
        "# E1-E4 preregistered audit report",
        "",
        "This report records gate outcomes without changing frozen protocol v0.2.",
        "",
        "## Decisions",
        "",
    ]
    for experiment, decision in decisions.items():
        lines.extend(
            [
                f"### {experiment.upper()}",
                "",
                "```json",
                json.dumps(decision, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## E3 gate correction",
            "",
            "A useful observed model should exceed the permutation null, so its "
            "one-sided empirical p-value should be small. The supplied "
            "`p > 0.10` direction was reversed before execution. Missingness-only "
            "remains a negative control and should not be significant.",
            "",
        ]
    )
    path = out_dir / "E1_E4_GATE_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _copy_compact_evidence(runtime_dir: Path, evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {
        "E1_GROUP_ABLATION_DELTAS.csv",
        "E1_GROUP_GATE.csv",
        "E2_INNER_SPLIT_AUDIT.csv",
        "E2_MAGNITUDE_RESIDUAL_SUMMARY.csv",
        "E3_EMPIRICAL_PVALUES.csv",
        "E3_MISSINGNESS_PAIRED_DELTA.csv",
        "E4_SEED_RESULTS.csv",
        "E4_SEED_SUMMARY.csv",
        "E4_CONFIG_RESULTS.csv",
        "E4_CONFIG_SUMMARY.csv",
        "RUN_STATUS.json",
        "E1_E4_GATE_REPORT.md",
    }
    for path in runtime_dir.rglob("*"):
        if path.is_file() and path.name in keep_names:
            destination = evidence_dir / path.relative_to(runtime_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["e1", "e2", "e3", "e4"],
        default=["e1", "e2", "e3", "e4"],
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=7)
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = _load_matrix(args.feature_matrix)
    decisions = {}
    runners = {
        "e1": lambda: run_e1(
            base,
            args.out_dir / "E1_group_ablation",
            args.bootstrap,
            args.seed,
            args.jobs,
        ),
        "e2": lambda: run_e2(
            base,
            args.out_dir / "E2_magnitude_residual",
            args.bootstrap,
            args.seed,
            args.jobs,
        ),
        "e3": lambda: run_e3(
            base,
            args.out_dir / "E3_negative_controls",
            args.permutations,
            args.bootstrap,
            args.seed,
        ),
        "e4": lambda: run_e4(
            base, args.out_dir / "E4_model_stability", args.seed
        ),
    }
    for experiment in args.experiments:
        print(f"Starting {experiment.upper()}", flush=True)
        decisions[experiment] = runners[experiment]()
    _write_status(
        args.out_dir,
        args.feature_matrix,
        decisions,
        args,
        time.time() - started,
    )
    _write_report(args.out_dir, decisions)
    if args.evidence_dir is not None:
        _copy_compact_evidence(args.out_dir, args.evidence_dir)
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
