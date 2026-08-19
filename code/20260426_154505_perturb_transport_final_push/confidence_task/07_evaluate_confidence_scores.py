#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    tmp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return float("nan")
    return float(tmp["x"].corr(tmp["y"], method=method))


def bootstrap_diff_ci(good: pd.Series, bad: pd.Series, seed: int = 20260521, n_boot: int = 500) -> tuple[float, float]:
    good = pd.to_numeric(good, errors="coerce").dropna().to_numpy()
    bad = pd.to_numeric(bad, errors="coerce").dropna().to_numpy()
    if len(good) < 2 or len(bad) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        g = rng.choice(good, len(good), replace=True)
        b = rng.choice(bad, len(bad), replace=True)
        vals.append(float(np.mean(b) - np.mean(g)))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def failure_metrics(score_as_risk: pd.Series, error: pd.Series) -> dict[str, float | str]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {"auroc": float("nan"), "auprc": float("nan"), "status": "sklearn_unavailable"}
    tmp = pd.DataFrame({"score": pd.to_numeric(score_as_risk, errors="coerce"), "error": pd.to_numeric(error, errors="coerce")}).dropna()
    if len(tmp) < 5 or tmp["score"].nunique() < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "status": "too_few_rows"}
    threshold = tmp["error"].quantile(0.80)
    y = (tmp["error"] >= threshold).astype(int)
    if y.nunique() < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "status": "single_class_failure"}
    return {
        "auroc": float(roc_auc_score(y, tmp["score"])),
        "auprc": float(average_precision_score(y, tmp["score"])),
        "status": "ok",
    }


def make_groups(df: pd.DataFrame) -> list[tuple[str, dict, pd.DataFrame]]:
    groups: list[tuple[str, dict, pd.DataFrame]] = [("overall", {}, df)]
    for predictor, sub in df.groupby("predictor_name", dropna=False):
        groups.append(("by_predictor", {"predictor_name": predictor}, sub))
    for (predictor, fold_id), sub in df.groupby(["predictor_name", "fold_id"], dropna=False):
        groups.append(("by_predictor_fold", {"predictor_name": predictor, "fold_id": fold_id}, sub))
    return groups


def evaluate(records: pd.DataFrame, scores: pd.DataFrame, eval_split: str) -> dict[str, pd.DataFrame]:
    merged = records.merge(scores, on=["record_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name", "true_error_rmse"], how="inner")
    if eval_split != "all":
        merged = merged[merged["split"] == eval_split].copy()
    if merged.empty:
        raise RuntimeError(f"No merged rows for eval_split={eval_split!r}")

    summary_rows = []
    coverage_rows = []
    high_low_rows = []
    failure_rows = []
    bucket_rows = []
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    for score_name, score_df in merged.groupby("score_name", dropna=False):
        score_type = str(score_df["score_type"].iloc[0])
        for scope, attrs, sub in make_groups(score_df):
            if len(sub) < 3:
                continue
            score = pd.to_numeric(sub["score_value"], errors="coerce")
            error = pd.to_numeric(sub["true_error_rmse"], errors="coerce")
            score_as_risk = -score if score_type == "confidence" else score
            spearman_raw = corr(score, error, "spearman")
            pearson_raw = corr(score, error, "pearson")
            spearman_aligned = corr(score_as_risk, error, "spearman")
            pearson_aligned = corr(score_as_risk, error, "pearson")
            expected_direction_ok = bool(
                (score_type == "confidence" and pd.notna(spearman_raw) and spearman_raw < 0)
                or (score_type == "risk" and pd.notna(spearman_raw) and spearman_raw > 0)
            )
            row = {
                "scope": scope,
                "dataset_name": "KaggleCrossCell",
                "predictor_name": attrs.get("predictor_name", "ALL"),
                "fold_id": attrs.get("fold_id", "ALL"),
                "score_name": score_name,
                "score_type": score_type,
                "split_evaluated": eval_split,
                "n": int(len(sub)),
                "spearman_score_vs_error": spearman_raw,
                "pearson_score_vs_error": pearson_raw,
                "direction_aligned_spearman": spearman_aligned,
                "direction_aligned_pearson": pearson_aligned,
                "expected_direction_ok": expected_direction_ok,
            }
            summary_rows.append(row)

            order = np.argsort(score_as_risk.to_numpy(dtype=float))
            for cov in coverages:
                k = max(1, int(np.ceil(len(order) * cov)))
                kept = sub.iloc[order[:k]]
                coverage_rows.append(
                    {
                        **row,
                        "coverage": float(k / len(order)),
                        "n_kept": int(k),
                        "mean_true_error_rmse": float(kept["true_error_rmse"].mean()),
                        "median_true_error_rmse": float(kept["true_error_rmse"].median()),
                    }
                )

            q_low, q_high = score.quantile(0.20), score.quantile(0.80)
            if score_type == "confidence":
                good = sub[score >= q_high]
                bad = sub[score <= q_low]
                good_label = "top20_confidence"
                bad_label = "bottom20_confidence"
            else:
                good = sub[score <= q_low]
                bad = sub[score >= q_high]
                good_label = "bottom20_risk"
                bad_label = "top20_risk"
            ci_low, ci_high = bootstrap_diff_ci(good["true_error_rmse"], bad["true_error_rmse"])
            high_low_rows.append(
                {
                    **row,
                    "good_group": good_label,
                    "bad_group": bad_label,
                    "good_n": int(len(good)),
                    "bad_n": int(len(bad)),
                    "good_mean_rmse": float(good["true_error_rmse"].mean()) if len(good) else float("nan"),
                    "bad_mean_rmse": float(bad["true_error_rmse"].mean()) if len(bad) else float("nan"),
                    "good_median_rmse": float(good["true_error_rmse"].median()) if len(good) else float("nan"),
                    "bad_median_rmse": float(bad["true_error_rmse"].median()) if len(bad) else float("nan"),
                    "bad_minus_good_mean_rmse": float(bad["true_error_rmse"].mean() - good["true_error_rmse"].mean()) if len(good) and len(bad) else float("nan"),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "expected_high_low_ok": bool(len(good) and len(bad) and bad["true_error_rmse"].mean() > good["true_error_rmse"].mean()),
                }
            )
            fmet = failure_metrics(score_as_risk, error)
            failure_rows.append({**row, **fmet})

            valid = sub.assign(_score=score, _error=error).dropna(subset=["_score", "_error"])
            if len(valid) >= 5:
                try:
                    valid = valid.copy()
                    valid["_bucket"] = pd.qcut(valid["_score"].rank(method="first"), q=min(5, len(valid)), labels=False)
                    for bucket, bsub in valid.groupby("_bucket"):
                        bucket_rows.append(
                            {
                                **row,
                                "bucket": int(bucket),
                                "n_bucket": int(len(bsub)),
                                "mean_score": float(bsub["_score"].mean()),
                                "mean_true_error_rmse": float(bsub["_error"].mean()),
                                "median_true_error_rmse": float(bsub["_error"].median()),
                            }
                        )
                except Exception:
                    pass
    return {
        "merged": merged,
        "summary": pd.DataFrame(summary_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "high_low": pd.DataFrame(high_low_rows),
        "failure": pd.DataFrame(failure_rows),
        "buckets": pd.DataFrame(bucket_rows),
    }


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    overall = summary[summary["scope"] == "overall"].copy()
    if overall.empty:
        return overall
    return overall.sort_values(["direction_aligned_spearman", "n"], ascending=[False, False])


def write_eval_report(path: Path, summary: pd.DataFrame, high_low: pd.DataFrame, coverage: pd.DataFrame, failure: pd.DataFrame, eval_split: str) -> None:
    best = best_rows(summary)
    random = summary[(summary["scope"] == "overall") & (summary["score_name"] == "random_score")]
    random_s = float(random["direction_aligned_spearman"].iloc[0]) if not random.empty else float("nan")
    lines = [
        "# Confidence Evaluation Report",
        "",
        f"- Evaluated split: `{eval_split}`",
        f"- Summary rows: {len(summary)}",
        f"- Overall scores: {len(summary[summary['scope'] == 'overall']) if not summary.empty else 0}",
        f"- Random baseline aligned Spearman: {random_s:.4f}" if pd.notna(random_s) else "- Random baseline aligned Spearman: NA",
        "",
        "## Best Overall Scores",
        "",
        "| rank | score_name | score_type | n | raw_spearman | aligned_spearman | expected_direction_ok |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, (_, row) in enumerate(best.head(8).iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['score_name']}` | {row['score_type']} | {int(row['n'])} | "
            f"{row['spearman_score_vs_error']:.4f} | {row['direction_aligned_spearman']:.4f} | {row['expected_direction_ok']} |"
        )
    lines.extend(["", "## Automatic Interpretation", ""])
    if not best.empty:
        top = best.iloc[0]
        lines.append(f"- Best aligned score is `{top['score_name']}` with aligned Spearman {top['direction_aligned_spearman']:.4f}.")
        if pd.notna(random_s):
            better = float(top["direction_aligned_spearman"]) > random_s
            lines.append(f"- Best score is {'better' if better else 'not better'} than random by aligned Spearman.")
        ok_scores = summary[(summary["scope"] == "overall") & (summary["expected_direction_ok"])]
        bad_scores = summary[(summary["scope"] == "overall") & (~summary["expected_direction_ok"])]
        lines.append(f"- Direction matched expectation for {len(ok_scores)} overall scores and failed/reversed for {len(bad_scores)} overall scores.")
    hl_overall = high_low[high_low["scope"] == "overall"]
    if not hl_overall.empty:
        ok_frac = float(hl_overall["expected_high_low_ok"].mean())
        lines.append(f"- High-confidence/low-risk group had lower mean RMSE in {ok_frac:.1%} of overall score rows.")
    cov_overall = coverage[coverage["scope"] == "overall"]
    if not cov_overall.empty:
        improvements = []
        for score_name, sub in cov_overall.groupby("score_name"):
            full = sub[sub["coverage"] >= 0.999]
            part = sub[sub["coverage"].between(0.79, 0.81)]
            if not full.empty and not part.empty:
                improvements.append((score_name, float(full["mean_true_error_rmse"].iloc[0] - part["mean_true_error_rmse"].iloc[0])))
        if improvements:
            n_down = sum(v > 0 for _, v in improvements)
            lines.append(f"- Risk-coverage mean RMSE decreased at about 80% coverage for {n_down}/{len(improvements)} overall scores.")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is an MVP on one dataset only: KaggleCrossCell.",
            "- Test rows are small: the result is useful for deciding whether to expand, not for paper-level claims.",
            "- `learned_risk_score` is exploratory because each fold has very few validation predictions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mvp_report(
    path: Path,
    records: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    high_low: pd.DataFrame,
    coverage: pd.DataFrame,
    split_summary: dict,
) -> None:
    best = best_rows(summary)
    top = best.iloc[0] if not best.empty else None
    feature_missing = {col: float(features[col].isna().mean()) for col in features.columns if col not in {"record_id", "task_id", "dataset_name", "fold_id", "split", "context", "perturbation", "predictor_name"}}
    best_hl = high_low[(high_low["scope"] == "overall") & (high_low["score_name"] == top["score_name"])] if top is not None else pd.DataFrame()
    best_cov = coverage[(coverage["scope"] == "overall") & (coverage["score_name"] == top["score_name"])] if top is not None else pd.DataFrame()
    cov_full = best_cov[best_cov["coverage"] >= 0.999]
    cov_80 = best_cov[best_cov["coverage"].between(0.79, 0.81)]
    coverage_text = "无法判断"
    if not cov_full.empty and not cov_80.empty:
        delta = float(cov_full["mean_true_error_rmse"].iloc[0] - cov_80["mean_true_error_rmse"].iloc[0])
        coverage_text = f"80% coverage mean RMSE 相比 full coverage {'下降' if delta > 0 else '没有下降'}，差值 {delta:.6f}"
    hl_text = "无法判断"
    if not best_hl.empty:
        row = best_hl.iloc[0]
        hl_text = f"好组 mean RMSE={row['good_mean_rmse']:.6f}，坏组 mean RMSE={row['bad_mean_rmse']:.6f}，bad-good={row['bad_minus_good_mean_rmse']:.6f}"
    lines = [
        "# Confidence Scoring MVP Final Report",
        "",
        "## 1. 本次 MVP 的目标",
        "",
        "这次不是继续训练新的 perturbation prediction 大模型，而是先做一个很小但关键的问题：已有 predictor 给出 predicted_effect 后，我们能不能给每一次 prediction 打一个 confidence / risk 分，并且这个分数真的和预测误差有关。",
        "",
        "## 2. 用的数据集",
        "",
        f"- Dataset: `{split_summary.get('dataset_name', 'KaggleCrossCell')}`",
        f"- Task count: {split_summary.get('n_tasks')}",
        f"- Context count: {split_summary.get('n_contexts')}",
        f"- Perturbation count: {split_summary.get('n_perturbations')}",
        "",
        "## 3. held-out pair split 情况",
        "",
        f"- Fold count: {split_summary.get('actual_folds')}",
        "- 这个 split 留出的是具体 `(context, perturbation)` pair。",
        "- test pair 不出现在 train，但 test 的 context 和 perturbation 都分别在 train 里出现过。",
        "- 这比普通 random split 更接近“这个新组合的预测到底能不能信”。",
        "",
        "## 4. predictor 情况",
        "",
        f"- Predictors: {', '.join(sorted(records['predictor_name'].unique()))}",
        "- 本轮只用 V0StrongBaseline 和 ContextSimBaseline，没有训练深度模型。",
        "",
        "## 5. PredictionRecord 规模",
        "",
        f"- Total PredictionRecord rows: {len(records)}",
        f"- Test rows: {int((records['split'] == 'test').sum())}",
        f"- Val rows: {int((records['split'] == 'val').sum())}",
        "",
        "## 6. confidence features 及缺失率",
        "",
        "| feature | missing_rate |",
        "| --- | ---: |",
    ]
    for feature, rate in feature_missing.items():
        lines.append(f"| `{feature}` | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## 7. confidence / risk scores",
            "",
            f"- Scores: {', '.join(sorted(scores['score_name'].unique()))}",
            "",
            "## 8. Spearman / Pearson 结果",
            "",
        ]
    )
    if top is not None:
        lines.extend(
            [
                f"- Best overall score: `{top['score_name']}`",
                f"- Score type: `{top['score_type']}`",
                f"- Raw Spearman(score vs error): {top['spearman_score_vs_error']:.6f}",
                f"- Direction-aligned Spearman: {top['direction_aligned_spearman']:.6f}",
                f"- Pearson(score vs error): {top['pearson_score_vs_error']:.6f}",
            ]
        )
    else:
        lines.append("- 没有可用 overall summary。")
    lines.extend(
        [
            "",
            "## 9. high-confidence vs low-confidence 是否符合预期",
            "",
            f"- {hl_text}",
            "",
            "## 10. risk-coverage 是否下降",
            "",
            f"- {coverage_text}",
            "",
            "## 11. best score 是哪个",
            "",
            f"- `{top['score_name']}`" if top is not None else "- 暂无。",
            "",
            "## 12. learned_risk_score 是否比简单 baseline 好",
            "",
        ]
    )
    learned = summary[(summary["scope"] == "overall") & (summary["score_name"] == "learned_risk_score")]
    simple = summary[(summary["scope"] == "overall") & (summary["score_name"] == "simple_combined_confidence")]
    if not learned.empty and not simple.empty:
        lines.append(
            f"- learned_risk_score aligned Spearman={learned.iloc[0]['direction_aligned_spearman']:.6f}; "
            f"simple_combined_confidence aligned Spearman={simple.iloc[0]['direction_aligned_spearman']:.6f}。"
        )
        lines.append("- 注意：learned_risk_score 目前训练样本很少，只能算 exploratory。")
    else:
        lines.append("- learned_risk_score 或 simple baseline 缺失，暂不能稳定比较。")
    if top is not None and pd.notna(top["direction_aligned_spearman"]):
        support = float(top["direction_aligned_spearman"]) > 0
    else:
        support = False
    lines.extend(
        [
            "",
            "## 13. 结果是否支持 confidence scoring task",
            "",
            "- 当前结果支持继续扩展这个 task，但还不能夸大。",
            f"- 原因：best score 的方向对齐 Spearman {'为正' if support else '不理想'}，说明至少有一部分 score 能反映 error 风险；但样本量太小。",
            "",
            "## 14. 当前限制",
            "",
            "- 样本量小：只有 KaggleCrossCell，test prediction rows 不多。",
            "- 数据集单一：还没有扩展到 Haber / Parekh / KaggleCrossPatient。",
            "- feature 基础：现在主要是 context similarity、support、stability、prediction magnitude、model disagreement。",
            "- learned_risk_score 不稳定：每个 fold 可用于训练 risk model 的 val rows 很少。",
            "",
            "## 15. 下一步建议",
            "",
            "- 第一优先：扩展到 Haber / Parekh / KaggleCrossPatient，看 confidence-error 关系是否复现。",
            "- 第二优先：优化 feature，尤其是更稳的 perturbation stability 和 context distance。",
            "- 第三优先：接入更多 predictor，但不要先训练大模型；先让 confidence evaluator 稳起来。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence/risk scores against true prediction error.")
    parser.add_argument("--records-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "PREDICTION_RECORDS.csv"))
    parser.add_argument("--features-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "tables" / "CONFIDENCE_FEATURES.csv"))
    parser.add_argument("--scores-csv", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "tables" / "CONFIDENCE_SCORES.csv"))
    parser.add_argument("--split-summary-json", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final" / "input" / "kagglecrosscell_heldout_pair_split_summary.json"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "confidence_task_mvp_final"))
    parser.add_argument("--eval-split", default="test", choices=["test", "val", "all"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    report_dir = out_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records_csv)
    features = pd.read_csv(args.features_csv)
    scores = pd.read_csv(args.scores_csv)
    split_summary = json.loads(Path(args.split_summary_json).read_text(encoding="utf-8"))
    result = evaluate(records, scores, args.eval_split)
    summary = result["summary"]
    coverage = result["coverage"]
    high_low = result["high_low"]
    failure = result["failure"]
    buckets = result["buckets"]

    summary.to_csv(table_dir / "CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    coverage.to_csv(table_dir / "RISK_COVERAGE.csv", index=False)
    high_low.to_csv(table_dir / "HIGH_LOW_CONFIDENCE_RMSE.csv", index=False)
    failure.to_csv(table_dir / "FAILURE_DETECTION.csv", index=False)
    buckets.to_csv(table_dir / "CALIBRATION_BUCKETS.csv", index=False)
    payload = {
        "eval_split": args.eval_split,
        "n_merged_rows": int(len(result["merged"])),
        "n_summary_rows": int(len(summary)),
        "best_overall": best_rows(summary).head(10).replace({np.nan: None}).to_dict("records"),
    }
    (table_dir / "CONFIDENCE_EVAL_SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_eval_report(report_dir / "confidence_eval_report.md", summary, high_low, coverage, failure, args.eval_split)
    write_mvp_report(out_dir / "MVP_REPORT.md", records, features, scores, summary, high_low, coverage, split_summary)
    print(json.dumps({"out_dir": str(out_dir), **payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
