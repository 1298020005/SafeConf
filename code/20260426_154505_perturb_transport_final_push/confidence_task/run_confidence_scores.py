#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def minmax(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi) or abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - lo) / (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long-format confidence/risk score baselines.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--features-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp" / "CONFIDENCE_FEATURES.csv"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records_csv)
    features = pd.read_csv(args.features_csv)
    df = records.merge(features, on="record_id", how="left")
    rng = np.random.default_rng(args.seed)
    rows = []
    random_values = pd.Series(rng.random(len(df)), index=df.index)
    score_defs = [
        ("random_score", "confidence", random_values),
        ("context_similarity_score", "confidence", minmax(df["context_similarity_max"])),
        ("perturbation_stability_score", "confidence", minmax(df["perturbation_effect_stability"])),
        ("support_count_score", "confidence", minmax(np.log1p(df["perturbation_support_count"]))),
        ("prediction_magnitude_risk", "risk", minmax(df["prediction_l2_norm"])),
        ("model_disagreement_risk", "risk", minmax(df["model_disagreement_rmse"])),
    ]
    skipped = []
    for score_name, score_type, values in score_defs:
        if pd.Series(values).notna().sum() == 0:
            skipped.append(score_name)
            continue
        for rec_id, value in zip(df["record_id"], values):
            if pd.notna(value):
                rows.append({"record_id": rec_id, "score_name": score_name, "score_type": score_type, "score_value": float(value)})
    out = pd.DataFrame(rows)
    out_path = out_dir / "CONFIDENCE_SCORES.csv"
    out.to_csv(out_path, index=False)
    report = report_dir / "confidence_scores_report.md"
    report.write_text(
        "\n".join(
            [
                "# Confidence scores report",
                "",
                f"- records: {len(records)}",
                f"- score_rows: {len(out)}",
                f"- skipped_scores: {', '.join(skipped) if skipped else 'none'}",
                "- learned_risk_score is intentionally left for the next pass after Step 4 validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"confidence_scores_csv": str(out_path), "n_rows": int(len(out)), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
