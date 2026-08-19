from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from safetrans_confidence.features.schema import assert_no_label_leakage

DEFAULT_ERROR_RANKER_FEATURES = [
    "context_similarity_max",
    "context_similarity_mean",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "prediction_magnitude_deviation",
    "model_disagreement_rmse",
    "ood_nearest_distance",
    "historical_residual_risk",
]

LINEAR_ERROR_RANKER_FEATURES = [
    "context_similarity_max",
    "perturbation_support_count",
    "perturbation_effect_stability",
    "prediction_magnitude_deviation",
    "model_disagreement_rmse",
    "ood_nearest_distance",
]


@dataclass(frozen=True)
class ErrorRankerConfig:
    score_name: str = "safeconf_error_ranker_risk"
    min_train_rows: int = 8
    random_state: int = 5201


def _make_model(random_state: int, model_type: str = "histgbt", n_train_rows: int | None = None):
    if model_type == "elasticnet":
        try:
            from sklearn.linear_model import ElasticNetCV
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            cv = max(2, min(5, int(n_train_rows or 5)))
            return make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=[0.2, 0.5, 0.8, 1.0],
                    alphas=np.logspace(-4, 1, 40),
                    cv=cv,
                    max_iter=20000,
                    random_state=random_state,
                ),
            )
        except Exception:
            from sklearn.linear_model import ElasticNet
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=random_state),
            )

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=160,
            max_leaf_nodes=12,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=random_state,
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        )


def _prepare_xy(train: pd.DataFrame, target: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    target_x = target[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_x.fillna(med), target_x.fillna(med)


def build_error_ranker_scores(
    base: pd.DataFrame,
    feature_cols: list[str] | None = None,
    config: ErrorRankerConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train fold-local shallow risk models and score test predictions.

    This is intentionally not a perturbation predictor. It only estimates the
    expected error of an already produced prediction.  For each
    (dataset, fold, predictor), it trains on train+val rows and predicts test
    rows only; no test labels are used during fitting.
    """
    cfg = config or ErrorRankerConfig()
    cols = [c for c in (feature_cols or DEFAULT_ERROR_RANKER_FEATURES) if c in base.columns]
    assert_no_label_leakage(cols)
    rows: list[dict] = []
    status_rows: list[dict] = []

    group_cols = ["dataset_name", "fold_id", "predictor_name"]
    for (dataset, fold, predictor), sub in base.groupby(group_cols, dropna=False):
        train = sub[sub["split"].isin(["train", "val"])].copy()
        test = sub[sub["split"] == "test"].copy()
        status = {
            "dataset_name": dataset,
            "fold_id": int(fold),
            "predictor_name": predictor,
            "n_train_pool": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(cols)),
            "score_name": cfg.score_name,
        }
        if len(cols) == 0 or len(test) == 0 or len(train) < cfg.min_train_rows:
            status["status"] = "skipped_too_few_rows_or_features"
            status_rows.append(status)
            continue
        y = pd.to_numeric(train["true_error_rmse"], errors="coerce")
        valid = y.notna()
        train = train.loc[valid].copy()
        y = y.loc[valid]
        if len(train) < cfg.min_train_rows or float(y.std(ddof=0)) <= 1e-12:
            status["status"] = "skipped_degenerate_target"
            status_rows.append(status)
            continue

        train_x, test_x = _prepare_xy(train, test, cols)
        model = _make_model(cfg.random_state)
        model.fit(train_x, y)
        pred_risk = model.predict(test_x)
        for (_, row), value in zip(test.iterrows(), pred_risk):
            rows.append(
                {
                    "record_id": row["record_id"],
                    "dataset_name": row["dataset_name"],
                    "dataset_family": row.get("dataset_family", ""),
                    "fold_id": int(row["fold_id"]),
                    "split": row["split"],
                    "context": row["context"],
                    "perturbation": row["perturbation"],
                    "predictor_name": row["predictor_name"],
                    "score_name": cfg.score_name,
                    "score_type": "risk",
                    "score_value": float(value),
                    "true_error_rmse": float(row["true_error_rmse"]),
                }
            )
        status["status"] = "ok"
        status_rows.append(status)

    return pd.DataFrame(rows), pd.DataFrame(status_rows)
