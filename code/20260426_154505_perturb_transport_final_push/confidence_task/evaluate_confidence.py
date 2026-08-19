#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.notna().sum() < 3 or y.notna().sum() < 3:
        return float("nan")
    return float(x.corr(y, method=method))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence/risk scores against true prediction error.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--scores-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "CONFIDENCE_SCORES.csv"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records_csv)
    scores = pd.read_csv(args.scores_csv)
    df = records.merge(scores, on="record_id", how="inner")
    summary_rows = []
    rc_rows = []
    hl_rows = []
    bucket_rows = []
    groups = ["dataset_name", "predictor_name", "fold_id", "score_name"]
    for keys, sub in df.groupby(groups, dropna=False):
        dataset_name, predictor_name, fold_id, score_name = keys
        score_type = str(sub["score_type"].iloc[0])
        error = pd.to_numeric(sub["true_error_rmse"], errors="coerce")
        score = pd.to_numeric(sub["score_value"], errors="coerce")
        signed_score = -score if score_type == "confidence" else score
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "predictor_name": predictor_name,
                "fold_id": fold_id,
                "score_name": score_name,
                "score_type": score_type,
                "n": int(len(sub)),
                "spearman_score_vs_error": corr(signed_score, error, "spearman"),
                "pearson_score_vs_error": corr(signed_score, error, "pearson"),
            }
        )
        order = np.argsort(score.to_numpy() if score_type == "risk" else -score.to_numpy())
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            k = max(1, int(np.ceil(len(order) * cov)))
            kept = sub.iloc[order[:k]]
            rc_rows.append(
                {
                    "dataset_name": dataset_name,
                    "predictor_name": predictor_name,
                    "fold_id": fold_id,
                    "score_name": score_name,
                    "coverage": float(k / len(order)),
                    "n_kept": int(k),
                    "mean_true_error_rmse": float(kept["true_error_rmse"].mean()),
                    "median_true_error_rmse": float(kept["true_error_rmse"].median()),
                }
            )
        q20 = score.quantile(0.2)
        q80 = score.quantile(0.8)
        low_good = sub[score <= q20] if score_type == "risk" else sub[score >= q80]
        high_bad = sub[score >= q80] if score_type == "risk" else sub[score <= q20]
        hl_rows.append(
            {
                "dataset_name": dataset_name,
                "predictor_name": predictor_name,
                "fold_id": fold_id,
                "score_name": score_name,
                "low_risk_or_high_conf_n": int(len(low_good)),
                "low_risk_or_high_conf_rmse": float(low_good["true_error_rmse"].mean()) if len(low_good) else float("nan"),
                "high_risk_or_low_conf_n": int(len(high_bad)),
                "high_risk_or_low_conf_rmse": float(high_bad["true_error_rmse"].mean()) if len(high_bad) else float("nan"),
            }
        )
        try:
            bins = pd.qcut(score.rank(method="first"), q=min(5, len(score)), labels=False)
            for b, bsub in sub.groupby(bins):
                bucket_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "predictor_name": predictor_name,
                        "fold_id": fold_id,
                        "score_name": score_name,
                        "bucket": int(b),
                        "n": int(len(bsub)),
                        "mean_score": float(bsub["score_value"].mean()),
                        "mean_true_error_rmse": float(bsub["true_error_rmse"].mean()),
                    }
                )
        except Exception:
            pass

    summary = pd.DataFrame(summary_rows)
    risk_cov = pd.DataFrame(rc_rows)
    high_low = pd.DataFrame(hl_rows)
    buckets = pd.DataFrame(bucket_rows)
    summary.to_csv(out_dir / "CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    risk_cov.to_csv(out_dir / "RISK_COVERAGE.csv", index=False)
    high_low.to_csv(out_dir / "HIGH_LOW_CONFIDENCE_RMSE.csv", index=False)
    buckets.to_csv(out_dir / "CALIBRATION_BUCKETS.csv", index=False)
    (out_dir / "FAILURE_DETECTION.csv").write_text("status,message\nskipped,AUROC/AUPRC will be added after MVP Step 4 validation\n", encoding="utf-8")
    payload = {"n_summary_rows": int(len(summary)), "best_by_spearman": summary.sort_values("spearman_score_vs_error", ascending=False).head(10).to_dict("records") if len(summary) else []}
    (out_dir / "CONFIDENCE_EVAL_SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "confidence_eval_report.md").write_text(f"# Confidence evaluation report\n\n- summary_rows: {len(summary)}\n- risk_coverage_rows: {len(risk_cov)}\n\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
