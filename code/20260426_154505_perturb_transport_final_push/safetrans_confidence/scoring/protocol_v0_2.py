from __future__ import annotations

import numpy as np
import pandas as pd

from safetrans_confidence.data.dataset_ontology import (
    assign_as_run_scoring_family,
    dataset_names_for_as_run_family,
)

PRIMARY_SCORE_NAME = "protocol_v0_2_family_confidence"
STABILITY_SCORE_NAME = "protocol_v0_2_with_stability_confidence"

SCORE_ID_COLUMNS = [
    "record_id",
    "dataset_name",
    "dataset_family",
    "fold_id",
    "split",
    "context",
    "perturbation",
    "predictor_name",
]
PRIMARY_FEATURE_COLUMNS = [
    "context_similarity_max",
    "perturbation_support_count",
    "model_disagreement_rmse",
]
SECONDARY_FEATURE_COLUMNS = [
    "historical_residual_risk",
    "ood_nearest_distance",
    "prediction_magnitude_deviation",
    "perturbation_effect_stability",
]

CHEM_DATASETS = frozenset(dataset_names_for_as_run_family("chem_robust"))


def zscore_by_ref(values: pd.Series, ref: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    ref = pd.to_numeric(ref, errors="coerce")
    med = ref.median()
    if pd.isna(med):
        med = 0.0
    scale = ref.quantile(0.75) - ref.quantile(0.25)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = ref.std()
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (values.fillna(med) - med) / scale


def assign_dataset_family(dataset_name: str) -> str:
    return assign_as_run_scoring_family(dataset_name, default="gene_main")


def build_protocol_v0_2_primary_scores(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen primary v0.2 score without evaluation labels.

    This path is intentionally label-blind: it does not require or emit
    true-effect keys, true-effect arrays, or true-error columns. Those fields
    belong to the downstream evaluation stage.
    """
    scoring_base = _prepare_scoring_base(base)
    _require_columns(scoring_base, SCORE_ID_COLUMNS + PRIMARY_FEATURE_COLUMNS, "primary protocol scoring")
    protocol, formulas = _compute_primary_protocol(scoring_base)
    rows: list[dict] = []
    _add_score_rows(rows, scoring_base, PRIMARY_SCORE_NAME, "confidence", protocol, include_evaluation_labels=False)
    return pd.DataFrame(rows), formulas


def build_protocol_v0_2_scores(
    base: pd.DataFrame,
    include_evaluation_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescore merged records+features with frozen protocol v0.2 formulas.

    This legacy/evaluation table includes diagnostic scores and, by default,
    carries `true_error_rmse` forward for downstream metrics. Use
    `build_protocol_v0_2_primary_scores` for deployable blind scoring.
    """
    scoring_base = _prepare_scoring_base(base)
    required = SCORE_ID_COLUMNS + PRIMARY_FEATURE_COLUMNS + SECONDARY_FEATURE_COLUMNS
    if include_evaluation_labels:
        required.append("true_error_rmse")
    _require_columns(scoring_base, required, "protocol evaluation scoring")

    rows: list[dict] = []
    rng = np.random.default_rng(5201)

    def add_score(score_name: str, score_type: str, values: pd.Series) -> None:
        _add_score_rows(rows, scoring_base, score_name, score_type, values, include_evaluation_labels)

    add_score("random_score", "confidence", pd.Series(rng.random(len(scoring_base)), index=scoring_base.index))
    add_score("context_similarity_score", "confidence", scoring_base["context_similarity_max"])
    add_score(
        "support_count_score",
        "confidence",
        np.log1p(scoring_base["perturbation_support_count"].astype(float)),
    )
    add_score("model_disagreement_risk", "risk", scoring_base["model_disagreement_rmse"])
    add_score("historical_residual_risk", "risk", scoring_base["historical_residual_risk"])
    add_score("ood_distance_risk", "risk", scoring_base["ood_nearest_distance"])
    add_score("prediction_magnitude_risk", "risk", scoring_base["prediction_magnitude_deviation"])

    protocol, formula_frame = _compute_primary_protocol(scoring_base)
    protocol_with_stability = pd.Series(np.nan, index=scoring_base.index, dtype=float)

    for (dataset, fold, predictor), idx_obj in scoring_base.groupby(
        ["dataset_name", "fold_id", "predictor_name"]
    ).groups.items():
        idx = list(idx_obj)
        sub = scoring_base.loc[idx]
        train = sub[sub["split"] == "train"]
        if train.empty:
            train = sub[sub["split"].isin(["train", "val"])]
        family = str(sub["dataset_family"].iloc[0])
        z_ctx = zscore_by_ref(sub["context_similarity_max"], train["context_similarity_max"])
        z_support = zscore_by_ref(
            np.log1p(sub["perturbation_support_count"].astype(float)),
            np.log1p(train["perturbation_support_count"].astype(float)),
        )
        z_dis = zscore_by_ref(sub["model_disagreement_rmse"], train["model_disagreement_rmse"])
        z_ood = zscore_by_ref(sub["ood_nearest_distance"], train["ood_nearest_distance"])
        z_stab = zscore_by_ref(
            sub["perturbation_effect_stability"], train["perturbation_effect_stability"]
        )
        if family == "chem_robust":
            protocol_with_stability.loc[idx] = z_support - z_dis
        else:
            protocol_with_stability.loc[idx] = z_ctx + z_support + z_stab - z_dis - z_ood

    add_score(PRIMARY_SCORE_NAME, "confidence", protocol)
    add_score(STABILITY_SCORE_NAME, "confidence", protocol_with_stability)
    return pd.DataFrame(rows), formula_frame


def _prepare_scoring_base(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    if "dataset_family" not in out.columns and "dataset_name" in out.columns:
        out["dataset_family"] = out["dataset_name"].map(assign_dataset_family)
    return out


def _require_columns(frame: pd.DataFrame, columns: list[str], purpose: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {purpose}: {', '.join(missing)}")


def _score_row_metadata(row: pd.Series) -> dict:
    return {
        "record_id": row["record_id"],
        "dataset_name": row["dataset_name"],
        "dataset_family": row["dataset_family"],
        "fold_id": int(row["fold_id"]),
        "split": row["split"],
        "context": row["context"],
        "perturbation": row["perturbation"],
        "predictor_name": row["predictor_name"],
    }


def _add_score_rows(
    rows: list[dict],
    base: pd.DataFrame,
    score_name: str,
    score_type: str,
    values: pd.Series,
    include_evaluation_labels: bool,
) -> None:
    for idx, value in values.items():
        row = base.loc[idx]
        out = {
            **_score_row_metadata(row),
            "score_name": score_name,
            "score_type": score_type,
            "score_value": float(value) if pd.notna(value) else np.nan,
        }
        if include_evaluation_labels:
            out["true_error_rmse"] = float(row["true_error_rmse"])
        rows.append(out)


def _compute_primary_protocol(base: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    protocol = pd.Series(np.nan, index=base.index, dtype=float)
    formula_rows = []

    for (dataset, fold, predictor), idx_obj in base.groupby(
        ["dataset_name", "fold_id", "predictor_name"]
    ).groups.items():
        idx = list(idx_obj)
        sub = base.loc[idx]
        train = sub[sub["split"] == "train"]
        if train.empty:
            train = sub[sub["split"].isin(["train", "val"])]
        family = str(sub["dataset_family"].iloc[0])
        z_ctx = zscore_by_ref(sub["context_similarity_max"], train["context_similarity_max"])
        z_support = zscore_by_ref(
            np.log1p(sub["perturbation_support_count"].astype(float)),
            np.log1p(train["perturbation_support_count"].astype(float)),
        )
        z_dis = zscore_by_ref(sub["model_disagreement_rmse"], train["model_disagreement_rmse"])
        if family == "chem_robust":
            protocol.loc[idx] = z_support - z_dis
            formula = "log_support - model_disagreement; stability_weight=0"
        else:
            protocol.loc[idx] = z_ctx + z_support - z_dis
            formula = "context_similarity + log_support - model_disagreement"
        formula_rows.append(
            {
                "dataset_name": dataset,
                "dataset_family": family,
                "fold_id": int(fold),
                "predictor_name": predictor,
                "protocol_formula": formula,
            }
        )
    return protocol, pd.DataFrame(formula_rows)
