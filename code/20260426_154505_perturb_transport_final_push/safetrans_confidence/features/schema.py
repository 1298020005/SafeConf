from __future__ import annotations

import pandas as pd


FEATURE_GROUPS: dict[str, list[str]] = {
    "evidence": [
        "context_similarity_max",
        "context_similarity_mean",
        "perturbation_support_count",
        "perturbation_effect_stability",
        "perturbation_effect_variance",
        "historical_residual_risk",
    ],
    "disagreement": [
        "model_disagreement_rmse",
        "model_disagreement_cosine",
        "gears_uncertainty",
    ],
    "ood": [
        "ood_nearest_distance",
        "prediction_magnitude_deviation",
        "prediction_l2_norm",
        "prediction_abs_mean",
        "fold_train_median_effect_norm",
        "prediction_norm_ratio",
    ],
}

FROZEN_PRIMARY_FEATURES = {
    "context_similarity_max",
    "perturbation_support_count",
    "model_disagreement_rmse",
}
BLIND_PRIMARY_FEATURES = FROZEN_PRIMARY_FEATURES
EVALUATION_DIAGNOSTIC_FEATURES = {
    "context_similarity_mean",
    "perturbation_effect_stability",
    "perturbation_effect_variance",
    "historical_residual_risk",
    "model_disagreement_cosine",
    "gears_uncertainty",
    "ood_nearest_distance",
    "prediction_magnitude_deviation",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "fold_train_median_effect_norm",
    "prediction_norm_ratio",
}
LABEL_OR_FORBIDDEN_COLUMNS = {
    "true_error_rmse",
    "true_error_cosine",
    "true_effect_key",
    "true_effect",
    "true_effect_l2_norm",
    "true_effect_abs_mean",
    "normalized_rmse",
    "failure_label",
    "score_value",
}


def available_feature_columns(table: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for group_cols in FEATURE_GROUPS.values():
        for col in group_cols:
            if col in table.columns and col not in cols:
                cols.append(col)
    return cols


def assert_no_label_leakage(feature_cols: list[str]) -> None:
    bad = sorted(set(feature_cols) & LABEL_OR_FORBIDDEN_COLUMNS)
    if bad:
        raise ValueError(f"label/leakage columns are not allowed as confidence features: {bad}")


def build_feature_schema_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            rows.append(
                {
                    "feature_group": group_name,
                    "feature_name": col,
                    "present": bool(col in table.columns),
                    "allowed_for_scorer": bool(col not in LABEL_OR_FORBIDDEN_COLUMNS),
                    "human_meaning": _human_meaning(col),
                }
            )
    return pd.DataFrame(rows)


def build_feature_provenance_table(table: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    present_cols = set(table.columns) if table is not None else set()
    for group_name, cols in FEATURE_GROUPS.items():
        for col in cols:
            meta = _feature_provenance(col)
            missing = _feature_missingness_summary(table, col)
            rows.append(
                {
                    "feature_group": group_name,
                    "feature_name": col,
                    "present": bool(col in present_cols) if table is not None else None,
                    "allowed_source_object": meta["allowed_source_object"],
                    "fold_scope": meta["fold_scope"],
                    "normalization_reference": meta["normalization_reference"],
                    "missingness_rule": meta["missingness_rule"],
                    "train_empty_fallback": meta["train_empty_fallback"],
                    "label_derived": meta["label_derived"],
                    "uses_heldout_true_effects": meta["uses_heldout_true_effects"],
                    "allowed_for_blind_primary_features": col in BLIND_PRIMARY_FEATURES,
                    "allowed_for_frozen_primary_score": col in FROZEN_PRIMARY_FEATURES,
                    "allowed_for_evaluation_diagnostics": col in EVALUATION_DIAGNOSTIC_FEATURES,
                    "forbidden_input_columns": meta["forbidden_input_columns"],
                    "leakage_status": _leakage_status(col, meta),
                    **missing,
                    "human_meaning": _human_meaning(col),
                }
            )
    return pd.DataFrame(rows)


def build_feature_missingness(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in available_feature_columns(table):
        series = pd.to_numeric(table[col], errors="coerce")
        rows.append(
            {
                "feature_name": col,
                "n_rows": int(len(series)),
                "n_missing": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()) if len(series) else 1.0,
                "mean": float(series.mean()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def _feature_missingness_summary(table: pd.DataFrame | None, feature_name: str) -> dict:
    empty = {
        "n_rows": None,
        "n_missing": None,
        "missing_rate": None,
        "n_datasets_present": None,
        "n_datasets_with_missing": None,
    }
    if table is None or feature_name not in table.columns:
        return empty
    series = pd.to_numeric(table[feature_name], errors="coerce")
    out = {
        "n_rows": int(len(series)),
        "n_missing": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else 1.0,
        "n_datasets_present": None,
        "n_datasets_with_missing": None,
    }
    if "dataset_name" in table.columns:
        dataset_rows = []
        for dataset, group in table.groupby("dataset_name", dropna=False):
            values = pd.to_numeric(group[feature_name], errors="coerce")
            dataset_rows.append(
                {
                    "dataset_name": dataset,
                    "has_any_value": bool(values.notna().any()),
                    "has_missing": bool(values.isna().any()),
                }
            )
        out["n_datasets_present"] = int(sum(row["has_any_value"] for row in dataset_rows))
        out["n_datasets_with_missing"] = int(sum(row["has_missing"] for row in dataset_rows))
    return out


def _feature_provenance(feature_name: str) -> dict:
    default = {
        "allowed_source_object": "unspecified",
        "fold_scope": "unspecified",
        "normalization_reference": "none",
        "missingness_rule": "not specified",
        "train_empty_fallback": "not specified",
        "label_derived": False,
        "uses_heldout_true_effects": False,
        "forbidden_input_columns": "true_error_rmse,true_error_cosine,true_effect_l2_norm,true_effect_abs_mean,normalized_rmse,failure_label,score_value",
    }
    meta = {
        "context_similarity_max": {
            "allowed_source_object": "target_control_means.npz and fold-train target_control_key rows",
            "fold_scope": "fold-local train reference controls; target task control vector only",
            "normalization_reference": "protocol scorer robust z-score against train rows within dataset/fold/predictor",
            "missingness_rule": "NaN only if no finite train-control similarity can be computed",
            "train_empty_fallback": "hard failure; no train tasks for fold",
        },
        "context_similarity_mean": {
            "allowed_source_object": "target_control_means.npz and fold-train target_control_key rows",
            "fold_scope": "fold-local train reference controls; target task control vector only",
            "normalization_reference": "none in frozen primary score; diagnostic raw feature",
            "missingness_rule": "NaN only if no finite train-control similarity can be computed",
            "train_empty_fallback": "hard failure; no train tasks for fold",
        },
        "perturbation_support_count": {
            "allowed_source_object": "PredictionRecord task identity columns and fold-train perturbation/context rows",
            "fold_scope": "fold-local train task labels excluding same context where possible",
            "normalization_reference": "protocol scorer robust z-score of log1p(count) against train rows",
            "missingness_rule": "integer count; zero support is valid",
            "train_empty_fallback": "hard failure; no train tasks for fold",
        },
        "perturbation_effect_stability": {
            "allowed_source_object": "fold-train true_effects.npz for same-perturbation source contexts",
            "fold_scope": "fold-local train labels only; no target held-out true effect",
            "normalization_reference": "none in frozen primary score; diagnostic raw feature",
            "missingness_rule": "NaN if fewer than two source train effects are available",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
        "perturbation_effect_variance": {
            "allowed_source_object": "fold-train true_effects.npz for same-perturbation source contexts",
            "fold_scope": "fold-local train labels only; no target held-out true effect",
            "normalization_reference": "none in frozen primary score; diagnostic raw feature",
            "missingness_rule": "NaN if fewer than two source train effects are available",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
        "historical_residual_risk": {
            "allowed_source_object": "fold-train true_effects.npz leave-one-context-out residuals",
            "fold_scope": "fold-local train labels only; no target held-out true effect",
            "normalization_reference": "none in frozen primary score; diagnostic raw feature",
            "missingness_rule": "NaN if no train perturbation has enough alternate-context source effects",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
        "model_disagreement_rmse": {
            "allowed_source_object": "predicted_effects.npz for expected aligned predictor pair",
            "fold_scope": "same task and split; predictions only",
            "normalization_reference": "protocol scorer robust z-score against train rows within dataset/fold/predictor",
            "missingness_rule": "strict mode fails if expected predictor set is missing; non-strict mode emits NaN",
            "train_empty_fallback": "hard failure if protocol scorer has no train/val reference rows",
        },
        "model_disagreement_cosine": {
            "allowed_source_object": "predicted_effects.npz for expected aligned predictor pair",
            "fold_scope": "same task and split; predictions only",
            "normalization_reference": "none in frozen primary score; diagnostic raw feature",
            "missingness_rule": "NaN if expected predictor set is missing",
            "train_empty_fallback": "not applicable",
        },
        "gears_uncertainty": {
            "allowed_source_object": "native predictor uncertainty output when supplied by adapter",
            "fold_scope": "predictor output; no SafeConf label lookup",
            "normalization_reference": "none in frozen primary score; adapter-specific diagnostic",
            "missingness_rule": "missing unless predictor exports uncertainty",
            "train_empty_fallback": "not applicable",
        },
        "ood_nearest_distance": {
            "allowed_source_object": "fold-train feature/reference space if implemented",
            "fold_scope": "fold-local train reference only",
            "normalization_reference": "diagnostic; not used by frozen primary score",
            "missingness_rule": "fast recovery path intentionally writes NaN",
            "train_empty_fallback": "not applicable in current fast path",
        },
        "prediction_magnitude_deviation": {
            "allowed_source_object": "predicted_effects.npz and fold-train true_effects.npz median effect norm",
            "fold_scope": "target prediction plus fold-train label scale; no target held-out true effect",
            "normalization_reference": "fold_train_median_effect_norm",
            "missingness_rule": "NaN if fold-train effect norm reference is unavailable",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
        "prediction_l2_norm": {
            "allowed_source_object": "predicted_effects.npz",
            "fold_scope": "target prediction only",
            "normalization_reference": "none",
            "missingness_rule": "NaN only if predicted effect vector is unavailable or invalid",
            "train_empty_fallback": "not applicable",
        },
        "prediction_abs_mean": {
            "allowed_source_object": "predicted_effects.npz",
            "fold_scope": "target prediction only",
            "normalization_reference": "none",
            "missingness_rule": "NaN only if predicted effect vector is unavailable or invalid",
            "train_empty_fallback": "not applicable",
        },
        "fold_train_median_effect_norm": {
            "allowed_source_object": "fold-train true_effects.npz",
            "fold_scope": "fold-local train labels only",
            "normalization_reference": "none; used as diagnostic scale reference",
            "missingness_rule": "NaN if no finite train effect norm exists",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
        "prediction_norm_ratio": {
            "allowed_source_object": "predicted_effects.npz and fold_train_median_effect_norm",
            "fold_scope": "target prediction plus fold-local train label scale",
            "normalization_reference": "fold_train_median_effect_norm",
            "missingness_rule": "NaN if train effect norm reference is unavailable",
            "train_empty_fallback": "hard failure; no train tasks for fold",
            "label_derived": True,
        },
    }.get(feature_name, {})
    return {**default, **meta}


def _leakage_status(feature_name: str, meta: dict) -> str:
    if feature_name in FROZEN_PRIMARY_FEATURES and meta.get("label_derived"):
        return "fail_primary_label_derived"
    if meta.get("uses_heldout_true_effects"):
        return "fail_uses_heldout_true_effects"
    if feature_name in FROZEN_PRIMARY_FEATURES:
        return "pass_frozen_primary"
    if meta.get("label_derived"):
        return "evaluation_only_label_derived"
    return "diagnostic_or_secondary"


def _human_meaning(feature_name: str) -> str:
    meanings = {
        "context_similarity_max": "target context 与训练 context 最像的程度，越高通常越可信",
        "context_similarity_mean": "target context 与训练 context 的平均相似度",
        "perturbation_support_count": "同一 perturbation 在训练里有多少 source contexts 支持",
        "perturbation_effect_stability": "同一 perturbation 在不同训练 contexts 下 effect 是否稳定",
        "perturbation_effect_variance": "同一 perturbation 的 effect 波动大小",
        "historical_residual_risk": "训练内部 leave-one-context-out 的历史误差，越高越危险",
        "model_disagreement_rmse": "不同 predictor 对同一 task 的预测差异，越高越危险",
        "model_disagreement_cosine": "不同 predictor effect 方向差异",
        "gears_uncertainty": "GEARS 自带 uncertainty，如果 predictor 输出了才有",
        "ood_nearest_distance": "target task 到训练任务最近邻距离，越远越危险",
        "prediction_magnitude_deviation": "预测 effect 幅度是否异常",
        "prediction_l2_norm": "预测 effect 向量长度",
        "prediction_abs_mean": "预测 effect 平均绝对值",
        "fold_train_median_effect_norm": "fold 内 train true effects 的中位尺度，用于诊断归一化",
        "prediction_norm_ratio": "预测 effect 长度相对 train effect 尺度的比例",
    }
    return meanings.get(feature_name, "")
