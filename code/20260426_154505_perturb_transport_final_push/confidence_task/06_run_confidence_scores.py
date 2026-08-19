#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


BASE_FEATURES = [
    "context_similarity_max",
    "perturbation_effect_stability",
    "perturbation_support_count",
    "prediction_l2_norm",
    "prediction_abs_mean",
    "model_disagreement_rmse",
]


def minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    lo = x.min(skipna=True)
    hi = x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - lo) / (hi - lo)


def median_impute_fit_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    med = train[cols].median(numeric_only=True)
    med = med.fillna(0.0)
    x_train = train[cols].fillna(med).to_numpy(dtype=np.float64)
    x_test = test[cols].fillna(med).to_numpy(dtype=np.float64)
    return x_train, x_test


def shallow_regressor(seed: int):
    try:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=seed)
    except Exception:
        return None


def add_score_rows(rows: list[dict], df: pd.DataFrame, score_name: str, score_type: str, values: pd.Series) -> None:
    for idx, value in values.items():
        if pd.isna(value):
            continue
        rec = df.loc[idx]
        rows.append(
            {
                "record_id": rec["record_id"],
                "dataset_name": rec["dataset_name"],
                "fold_id": int(rec["fold_id"]),
                "split": rec["split"],
                "context": rec["context"],
                "perturbation": rec["perturbation"],
                "predictor_name": rec["predictor_name"],
                "score_name": score_name,
                "score_type": score_type,
                "score_value": float(value),
                "true_error_rmse": float(rec["true_error_rmse"]),
            }
        )


def build_learned_risk(df: pd.DataFrame, rows: list[dict], seed: int) -> dict:
    model = shallow_regressor(seed)
    if model is None:
        return {"status": "skipped", "reason": "sklearn is unavailable"}

    feature_cols = [c for c in BASE_FEATURES if c in df.columns]
    status_rows = []
    for (dataset_name, predictor_name, fold_id), sub in df.groupby(["dataset_name", "predictor_name", "fold_id"], dropna=False):
        val = sub[sub["split"] == "val"].copy()
        test = sub[sub["split"] == "test"].copy()
        if len(val) < 2 or test.empty:
            status_rows.append(
                {
                    "dataset_name": dataset_name,
                    "predictor_name": predictor_name,
                    "fold_id": int(fold_id),
                    "status": "skipped",
                    "n_val_train": int(len(val)),
                    "n_test": int(len(test)),
                    "reason": "too few val rows or no test rows",
                }
            )
            continue
        x_train, x_test = median_impute_fit_transform(val, test, feature_cols)
        y_train = val["true_error_rmse"].to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model
            fitted.fit(x_train, y_train)
        pred = fitted.predict(x_test)
        for (_, rec), value in zip(test.iterrows(), pred):
            rows.append(
                {
                    "record_id": rec["record_id"],
                    "dataset_name": rec["dataset_name"],
                    "fold_id": int(rec["fold_id"]),
                    "split": rec["split"],
                    "context": rec["context"],
                    "perturbation": rec["perturbation"],
                    "predictor_name": rec["predictor_name"],
                    "score_name": "learned_risk_score",
                    "score_type": "risk",
                    "score_value": float(value),
                    "true_error_rmse": float(rec["true_error_rmse"]),
                }
            )
        status_rows.append(
            {
                "dataset_name": dataset_name,
                "predictor_name": predictor_name,
                "fold_id": int(fold_id),
                "status": "ok_exploratory",
                "n_val_train": int(len(val)),
                "n_test": int(len(test)),
                "feature_cols": ",".join(feature_cols),
            }
        )
    return {"status": "ok", "fold_status": status_rows, "note": "learned_risk_score trains only on val rows and scores test rows; sample size is very small."}


def write_report(path: Path, score_df: pd.DataFrame, diagnostics: dict, feature_missing: dict) -> None:
    lines = [
        "# Confidence Scores Report",
        "",
        f"- Score rows: {len(score_df)}",
        f"- Score names: {', '.join(sorted(score_df['score_name'].unique())) if not score_df.empty else 'none'}",
        f"- learned_risk_score status: {diagnostics.get('learned_risk', {}).get('status')}",
        "",
        "## Feature Missingness Used For Scores",
        "",
        "| feature | missing_rate |",
        "| --- | ---: |",
    ]
    for feature, rate in feature_missing.items():
        lines.append(f"| `{feature}` | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `confidence` scores are higher-is-better.",
            "- `risk` scores are higher-error-prone.",
            "- `learned_risk_score` is shallow and exploratory here because each fold has only a small validation set.",
            "- No test row is used to train `learned_risk_score`; each fold trains on its val rows and scores its test rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confidence/risk score baselines in long format.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--features-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "tables" / "CONFIDENCE_FEATURES.csv"))
    parser.add_argument("--out-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "tables" / "CONFIDENCE_SCORES.csv"))
    parser.add_argument("--report-md", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "reports" / "confidence_scores_report.md"))
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    records = pd.read_csv(args.records_csv)
    features = pd.read_csv(args.features_csv)
    df = records.merge(features, on=["record_id", "task_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"], how="left")
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    add_score_rows(rows, df, "random_score", "confidence", pd.Series(rng.random(len(df)), index=df.index))
    add_score_rows(rows, df, "context_similarity_score", "confidence", df["context_similarity_max"])
    add_score_rows(rows, df, "perturbation_stability_score", "confidence", df["perturbation_effect_stability"])
    add_score_rows(rows, df, "support_count_score", "confidence", np.log1p(pd.to_numeric(df["perturbation_support_count"], errors="coerce")))
    add_score_rows(rows, df, "prediction_magnitude_risk", "risk", df["prediction_l2_norm"])
    add_score_rows(rows, df, "model_disagreement_risk", "risk", df["model_disagreement_rmse"])

    combined = (
        minmax(df["context_similarity_max"]).fillna(0.0)
        + minmax(df["perturbation_effect_stability"]).fillna(0.0)
        + minmax(np.log1p(pd.to_numeric(df["perturbation_support_count"], errors="coerce"))).fillna(0.0)
    )
    disagreement = minmax(df["model_disagreement_rmse"])
    if disagreement.notna().mean() >= 0.5:
        combined = combined - disagreement.fillna(disagreement.median())
        divisor = 4.0
    else:
        divisor = 3.0
    combined = combined / divisor
    add_score_rows(rows, df, "simple_combined_confidence", "confidence", combined)

    learned_status = build_learned_risk(df, rows, args.seed)
    score_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    report_md = Path(args.report_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out_csv, index=False)
    feature_missing = {feature: float(df[feature].isna().mean()) for feature in BASE_FEATURES if feature in df.columns}
    diagnostics = {"learned_risk": learned_status, "score_rows": int(len(score_df))}
    write_report(report_md, score_df, diagnostics, feature_missing)
    print(json.dumps({"out_csv": str(out_csv), "report_md": str(report_md), **diagnostics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
