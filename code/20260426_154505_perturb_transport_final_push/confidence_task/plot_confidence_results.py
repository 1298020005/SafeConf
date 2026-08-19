#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, facecolor="white")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simple white-background figures for the confidence MVP.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    records = pd.read_csv(out_dir / "PREDICTION_RECORDS.csv")
    scores = pd.read_csv(out_dir / "CONFIDENCE_SCORES.csv")
    merged = records.merge(scores, on="record_id", how="inner")

    plt.figure(figsize=(7, 5))
    for name, sub in merged.groupby("score_name"):
        plt.scatter(sub["score_value"], sub["true_error_rmse"], s=18, alpha=0.55, label=name)
    plt.xlabel("score value")
    plt.ylabel("true error RMSE")
    plt.legend(fontsize=7)
    savefig(fig_dir / "confidence_vs_error_scatter.png")

    rc = pd.read_csv(out_dir / "RISK_COVERAGE.csv")
    plt.figure(figsize=(7, 5))
    for name, sub in rc.groupby("score_name"):
        curve = sub.groupby("coverage", as_index=False)["mean_true_error_rmse"].mean()
        plt.plot(curve["coverage"], curve["mean_true_error_rmse"], marker="o", label=name)
    plt.xlabel("coverage")
    plt.ylabel("mean true error RMSE")
    plt.legend(fontsize=7)
    savefig(fig_dir / "risk_coverage.png")

    hl = pd.read_csv(out_dir / "HIGH_LOW_CONFIDENCE_RMSE.csv")
    plot_df = hl.groupby("score_name", as_index=False)[["low_risk_or_high_conf_rmse", "high_risk_or_low_conf_rmse"]].mean()
    plot_df.plot(x="score_name", kind="bar", figsize=(8, 5))
    plt.ylabel("mean RMSE")
    savefig(fig_dir / "high_low_confidence_rmse.png")

    ev = pd.read_csv(out_dir / "CONFIDENCE_EVAL_SUMMARY.csv")
    comp = ev.groupby("score_name", as_index=False)["spearman_score_vs_error"].mean().sort_values("spearman_score_vs_error")
    comp.plot(x="score_name", y="spearman_score_vs_error", kind="barh", figsize=(7, 5), legend=False)
    plt.xlabel("Spearman(score-as-risk, error)")
    savefig(fig_dir / "baseline_spearman_comparison.png")

    buckets = pd.read_csv(out_dir / "CALIBRATION_BUCKETS.csv")
    plt.figure(figsize=(7, 5))
    for name, sub in buckets.groupby("score_name"):
        curve = sub.groupby("bucket", as_index=False)["mean_true_error_rmse"].mean()
        plt.plot(curve["bucket"], curve["mean_true_error_rmse"], marker="o", label=name)
    plt.xlabel("score bucket")
    plt.ylabel("mean true error RMSE")
    plt.legend(fontsize=7)
    savefig(fig_dir / "calibration_buckets.png")
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
