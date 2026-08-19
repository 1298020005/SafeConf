#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def choose_scores(summary: pd.DataFrame, max_scores: int = 4) -> list[str]:
    overall = summary[summary["scope"] == "overall"].copy()
    chosen: list[str] = []
    if "random_score" in set(overall["score_name"]):
        chosen.append("random_score")
    ranked = overall.sort_values("direction_aligned_spearman", ascending=False)["score_name"].tolist()
    for score in ranked:
        if score not in chosen:
            chosen.append(score)
        if len(chosen) >= max_scores:
            break
    return chosen


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, facecolor="white")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple white-background figures for the final confidence MVP.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(table_dir / "CONFIDENCE_SCORES.csv")
    summary = pd.read_csv(table_dir / "CONFIDENCE_EVAL_SUMMARY.csv")
    coverage = pd.read_csv(table_dir / "RISK_COVERAGE.csv")
    high_low = pd.read_csv(table_dir / "HIGH_LOW_CONFIDENCE_RMSE.csv")
    buckets = pd.read_csv(table_dir / "CALIBRATION_BUCKETS.csv")
    chosen = choose_scores(summary)

    plot_scores = scores[(scores["split"] == "test") & (scores["score_name"].isin(chosen))].copy()
    plt.figure(figsize=(8, 5))
    for score_name, sub in plot_scores.groupby("score_name"):
        plt.scatter(sub["score_value"], sub["true_error_rmse"], s=28, alpha=0.65, label=score_name)
    plt.xlabel("score value")
    plt.ylabel("true error RMSE")
    plt.title("Confidence / risk score vs prediction error")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    savefig(fig_dir / "confidence_vs_error_scatter.png")

    cov = coverage[(coverage["scope"] == "overall") & (coverage["score_name"].isin(chosen))].copy()
    plt.figure(figsize=(8, 5))
    for score_name, sub in cov.groupby("score_name"):
        sub = sub.sort_values("coverage")
        plt.plot(sub["coverage"], sub["mean_true_error_rmse"], marker="o", linewidth=2, label=score_name)
    plt.xlabel("coverage")
    plt.ylabel("mean true error RMSE")
    plt.title("Risk-coverage curve")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    savefig(fig_dir / "risk_coverage.png")

    hl = high_low[(high_low["scope"] == "overall") & (high_low["score_name"].isin(chosen))].copy()
    x = range(len(hl))
    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.18 for i in x], hl["good_mean_rmse"], width=0.36, label="high-conf / low-risk")
    plt.bar([i + 0.18 for i in x], hl["bad_mean_rmse"], width=0.36, label="low-conf / high-risk")
    plt.xticks(list(x), hl["score_name"], rotation=30, ha="right")
    plt.ylabel("mean RMSE")
    plt.title("High-confidence vs low-confidence RMSE")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    savefig(fig_dir / "high_low_confidence_rmse.png")

    comp = summary[summary["scope"] == "overall"].copy().sort_values("direction_aligned_spearman")
    plt.figure(figsize=(8, 5))
    plt.barh(comp["score_name"], comp["direction_aligned_spearman"])
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("direction-aligned Spearman with error")
    plt.title("Score ranking by Spearman")
    plt.grid(axis="x", alpha=0.2)
    savefig(fig_dir / "baseline_spearman_comparison.png")

    buck = buckets[(buckets["scope"] == "overall") & (buckets["score_name"].isin(chosen))].copy()
    plt.figure(figsize=(8, 5))
    for score_name, sub in buck.groupby("score_name"):
        sub = sub.sort_values("bucket")
        plt.plot(sub["bucket"], sub["mean_true_error_rmse"], marker="o", linewidth=2, label=score_name)
    plt.xlabel("score bucket, low to high")
    plt.ylabel("mean true error RMSE")
    plt.title("Calibration buckets")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    savefig(fig_dir / "calibration_buckets.png")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
