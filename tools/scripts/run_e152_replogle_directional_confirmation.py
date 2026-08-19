#!/usr/bin/env python3
"""E152: frozen Directional-SafeConf confirmation on Replogle K562/RPE1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
E151 = ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714"
E151_ROOT = E151 / "Replogle_two_cellline"
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
E149 = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714"
MANIFEST = E149 / "manifests/E149_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E152_replogle_directional_confirmation_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
SCORES_BEFORE_TRUTH = TABLES / "E152_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv"
SEED = 202607152
N_BOOTSTRAP = 3000
ENDPOINTS = [
    "error_centered_pearson_mean",
    "error_centered_cosine_mean",
    "direction_error_rank_target",
]
SCORES = [
    "directional_risk_frozen",
    "baseline_predicted_magnitude",
    "safeconf_calibrated_pair_risk",
    "risk_model_disagreement",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


E134 = load_module(
    "e134_for_e152", ROOT / "tools/scripts/run_e134_systema_exact_expression_space_audit.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def freeze_scores() -> dict[str, object]:
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    model = json.loads(MODEL.read_text())
    contract = json.loads((E149 / "RUN_STATUS.json").read_text())
    e151_status = json.loads((E151 / "RUN_STATUS.json").read_text())
    if sha256(MODEL) != contract["frozen_direction_model_sha256"]:
        raise RuntimeError("E135 model differs from the hash frozen in E149")
    if e151_status["strict_issue_count"] != 0:
        raise RuntimeError("E151 prediction records did not pass strict validation")
    if e151_status["n_primary_unique_heldout_context_tasks"] != 256:
        raise RuntimeError("E151 primary task count differs from E149")
    deployable = [
        "fold_id",
        "task_id",
        "setting",
        "context",
        "perturbation",
        *model["features_in_order"],
    ]
    tasks = pd.read_csv(E151_ROOT / "PRIMARY_TASK_RISK_TABLE.csv", usecols=deployable)
    if len(tasks) != 256 or tasks.duplicated(["context", "perturbation"]).any():
        raise RuntimeError("direction score input is not 256 unique held-out-context tasks")
    matrix = tasks[model["features_in_order"]].to_numpy(float)
    tasks["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(
        model["coefficients_in_order"], float
    )
    tasks["target_truth_used_for_score_or_transform"] = False
    tasks["frozen_model_sha256"] = sha256(MODEL)
    tasks.to_csv(SCORES_BEFORE_TRUTH, index=False)
    status = {
        "phase": "scores_frozen_before_replogle_directional_truth_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_primary_unique_tasks": len(tasks),
        "score_file": str(SCORES_BEFORE_TRUTH.relative_to(ROOT)),
        "score_file_sha256": sha256(SCORES_BEFORE_TRUTH),
        "model_file_sha256": sha256(MODEL),
        "manifest_sha256": sha256(MANIFEST),
        "e151_status_sha256": sha256(E151 / "RUN_STATUS.json"),
        "target_truth_columns_read": [],
        "absolute_error_columns_read": [],
        "model_refit": False,
    }
    (OUT / "SCORE_FREEZE_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    (OUT / "ANALYSIS_CONTRACT.md").write_text(
        "# E152 分析合同｜Replogle 方向风险确认\n\n"
        "E149 已固定 128 个扰动、两个细胞系留出折、256 个唯一主任务、E135 模型哈希和 gate。"
        "E152 第一阶段只读取四个部署特征并冻结方向风险；第二阶段才读取预测与真实向量。\n\n"
        "主终点为两模型平均的 Systema-centered Pearson error 与 cosine error。每折转换为 percentile rank 后"
        "取平均形成复合方向误差。按 128 个 perturbation 整簇 bootstrap 3,000 次；同一基因的 K562/RPE1"
        "任务同步重采样。gate 要求两个终点宏平均均大于0，且复合终点95%区间下界大于0。\n\n"
        "该数据没有参与 E135 方向模型开发，但两个细胞系属于同一研究，目标 control 可见；结果只支持"
        " control-observed 跨细胞系复制，不能写成跨研究或完全 zero-shot。\n"
    )
    (OUT / "README_先看这个.md").write_text(
        "# E152 先看这个\n\n先读 `ANALYSIS_CONTRACT.md`；完成后读 `reports/E152_REPORT.md`。\n"
    )
    return status


def exact_primary_audit(score_status: dict[str, object]):
    if sha256(SCORES_BEFORE_TRUTH) != score_status["score_file_sha256"]:
        raise RuntimeError("E152 frozen score file changed")
    spec = {
        "root": E151_ROOT,
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": MANIFEST,
        "cache": Path(
            "/home/yyf/data/safeconf_e112_external/"
            "Replogle_two_cellline_CONTROL_ONLY_512.npz"
        ),
    }
    all_tasks, source_audit = E134.audit_dataset("Replogle_two_cellline", spec)
    scores = pd.read_csv(SCORES_BEFORE_TRUTH)
    keys = ["fold_id", "task_id", "setting", "context", "perturbation"]
    tasks = all_tasks.merge(
        scores[
            keys
            + [
                "directional_risk_frozen",
                "frozen_model_sha256",
                "target_truth_used_for_score_or_transform",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(tasks) != 256 or tasks.duplicated(["context", "perturbation"]).any():
        raise RuntimeError("exact audit did not retain the 256 frozen primary tasks")
    rank_parts = []
    for endpoint in ENDPOINTS[:2]:
        rank_parts.append(
            tasks.groupby("fold_id")[endpoint].transform(
                lambda values: rankdata(values) / len(values)
            )
        )
    tasks["direction_error_rank_target"] = np.mean(np.stack(rank_parts), axis=0)
    return tasks, source_audit, len(all_tasks) - len(tasks)


def fold_metrics(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold_id, group in tasks.groupby("fold_id", sort=True):
        for score in SCORES:
            for endpoint in ENDPOINTS:
                rows.append(
                    {
                        "fold_id": fold_id,
                        "score": score,
                        "endpoint": endpoint,
                        "n_tasks": len(group),
                        "spearman": rho(group[score], group[endpoint]),
                    }
                )
    folds = pd.DataFrame(rows)
    macro = folds.groupby(["score", "endpoint"], as_index=False).agg(
        n_folds=("fold_id", "nunique"),
        fold_macro_spearman=("spearman", "mean"),
        min_fold_spearman=("spearman", "min"),
    )
    return folds, macro


def cluster_bootstrap(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    perturbations = sorted(tasks.perturbation.astype(str).unique())
    perturbation_index = {value: index for index, value in enumerate(perturbations)}
    folds = []
    for _, fold in tasks.groupby("fold_id", sort=False):
        folds.append(
            {
                "cluster": np.asarray(
                    [perturbation_index[str(value)] for value in fold.perturbation], int
                ),
                "scores": {score: fold[score].to_numpy(float) for score in SCORES},
                "endpoints": {
                    endpoint: fold[endpoint].to_numpy(float) for endpoint in ENDPOINTS
                },
            }
        )
    rows = []
    for draw in range(N_BOOTSTRAP):
        counts = rng.multinomial(
            len(perturbations), np.full(len(perturbations), 1 / len(perturbations))
        )
        row = {"draw": draw}
        for score in SCORES:
            for endpoint in ENDPOINTS:
                estimates = []
                for fold in folds:
                    indices = np.repeat(
                        np.arange(len(fold["cluster"])), counts[fold["cluster"]]
                    )
                    estimates.append(
                        rho(fold["scores"][score][indices], fold["endpoints"][endpoint][indices])
                    )
                row[f"{score}__{endpoint}"] = float(np.nanmean(estimates))
        for endpoint in ENDPOINTS:
            row[f"delta_directional_minus_magnitude__{endpoint}"] = (
                row[f"directional_risk_frozen__{endpoint}"]
                - row[f"baseline_predicted_magnitude__{endpoint}"]
            )
        rows.append(row)
    draws = pd.DataFrame(rows)
    summary = []
    for column in draws.columns[1:]:
        values = draws[column].to_numpy(float)
        summary.append(
            {
                "metric": column,
                "bootstrap_draws": N_BOOTSTRAP,
                "ci_low_2.5pct": np.nanquantile(values, 0.025),
                "median": np.nanmedian(values),
                "ci_high_97.5pct": np.nanquantile(values, 0.975),
                "fraction_above_zero": np.nanmean(values > 0),
            }
        )
    return draws, pd.DataFrame(summary)


def write_report(
    tasks: pd.DataFrame,
    macro: pd.DataFrame,
    bootstrap: pd.DataFrame,
    baseline: pd.DataFrame,
    passed: bool,
    excluded_diagnostics: int,
) -> None:
    direction = macro[macro.score.eq("directional_risk_frozen")].set_index("endpoint")
    boot = bootstrap.set_index("metric")
    combined = boot.loc["directional_risk_frozen__direction_error_rank_target"]
    delta = boot.loc["delta_directional_minus_magnitude__direction_error_rank_target"]
    pivot = macro.pivot(index="score", columns="endpoint", values="fold_macro_spearman")
    lines = [
        "# E152｜Replogle K562/RPE1 方向风险确认",
        "",
        f"## 预注册 gate：{'通过' if passed else '未通过'}",
        "",
        f"冻结方向风险在 centered Pearson error 上的两折宏平均 Spearman 为 "
        f"**{direction.loc['error_centered_pearson_mean', 'fold_macro_spearman']:.3f}**，"
        f"centered cosine error 为 **{direction.loc['error_centered_cosine_mean', 'fold_macro_spearman']:.3f}**。"
        f"复合方向 rank 的 perturbation-cluster bootstrap 95% CI 为 "
        f"**[{combined['ci_low_2.5pct']:.3f}, {combined['ci_high_97.5pct']:.3f}]**。",
        "",
        f"相对 predicted magnitude 的复合方向 Δρ bootstrap 95% CI 为 "
        f"**[{delta['ci_low_2.5pct']:.3f}, {delta['ci_high_97.5pct']:.3f}]**。",
        "",
        "| score | Pearson ρ | cosine ρ | combined rank ρ |",
        "|---|---:|---:|---:|",
    ]
    for score, row in pivot.iterrows():
        lines.append(
            f"| {score} | {row['error_centered_pearson_mean']:.3f} | "
            f"{row['error_centered_cosine_mean']:.3f} | "
            f"{row['direction_error_rank_target']:.3f} |"
        )
    lines += [
        "",
        "## 简单预测器",
        "",
        "| tasks | ensemble RMSE | training perturbed-centroid RMSE | ensemble − simple | ensemble胜出比例 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(
            f"| {row.n_tasks} | {row.rmse_ensemble_mean:.4f} | "
            f"{row.rmse_training_perturbed_mean:.4f} | "
            f"{row.ensemble_minus_simple_baseline:+.4f} | "
            f"{row.fraction_tasks_ensemble_beats_simple_baseline:.1%} |"
        )
    lines += [
        "",
        "## 覆盖与边界",
        "",
        f"- 主分析包含 {len(tasks)} 个唯一 held-out cell-line × perturbation 任务；"
        f"{excluded_diagnostics} 个 source-context 诊断任务不进入 gate。",
        "- E135 模型没有在 Replogle 上重拟合；风险分数先冻结，随后才读取方向真值。",
        "- 两个细胞系来自同一研究，目标细胞系 control 可见。通过 gate 也只表示同研究内跨细胞系复制，"
        "不等于跨研究泛化或新湿实验确认。",
    ]
    (REPORTS / "E152_REPORT.md").write_text("\n".join(lines) + "\n")


def analyze() -> dict[str, object]:
    score_status = json.loads((OUT / "SCORE_FREEZE_STATUS.json").read_text())
    tasks, source_audit, excluded = exact_primary_audit(score_status)
    folds, macro = fold_metrics(tasks)
    draws, bootstrap = cluster_bootstrap(tasks)
    baseline = E134.E133.baseline_summary(tasks)
    direction = macro[macro.score.eq("directional_risk_frozen")].set_index("endpoint")
    combined = bootstrap.set_index("metric").loc[
        "directional_risk_frozen__direction_error_rank_target"
    ]
    passed = bool(
        direction.loc["error_centered_pearson_mean", "fold_macro_spearman"] > 0
        and direction.loc["error_centered_cosine_mean", "fold_macro_spearman"] > 0
        and combined["ci_low_2.5pct"] > 0
    )
    tasks.to_csv(TABLES / "E152_TASK_AUDIT.csv", index=False)
    folds.to_csv(TABLES / "E152_FOLD_METRICS.csv", index=False)
    macro.to_csv(TABLES / "E152_MACRO_SUMMARY.csv", index=False)
    draws.to_csv(TABLES / "E152_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    bootstrap.to_csv(TABLES / "E152_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    baseline.to_csv(TABLES / "E152_SIMPLE_BASELINE_SUMMARY.csv", index=False)
    pd.DataFrame([source_audit]).to_csv(TABLES / "E152_SOURCE_TRUTH_AUDIT.csv", index=False)
    write_report(tasks, macro, bootstrap, baseline, passed, excluded)
    status = {
        "experiment": "E152_replogle_directional_confirmation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "n_folds": int(tasks.fold_id.nunique()),
        "n_primary_unique_tasks": len(tasks),
        "n_source_context_diagnostic_tasks_excluded": excluded,
        "bootstrap_draws": N_BOOTSTRAP,
        "score_file_sha256": score_status["score_file_sha256"],
        "model_file_sha256": score_status["model_file_sha256"],
        "all_source_truth_checks_pass": bool(
            source_audit["truth_reconstruction_pass_atol_1e-5"]
        ),
        "preregistered_directional_gate_passed": passed,
        "model_refit_on_replogle": False,
        "replogle_truth_used_for_score_or_transform": False,
        "claim_scope": "same-study control-observed cross-cell-line replication",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(macro.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(baseline.to_string(index=False))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only and args.analyze_only:
        raise ValueError("choose one phase")
    if args.freeze_only:
        print(json.dumps(freeze_scores(), ensure_ascii=False, indent=2))
    elif args.analyze_only:
        analyze()
    else:
        freeze_scores()
        analyze()


if __name__ == "__main__":
    main()
