#!/usr/bin/env python3
"""E139: confirm the frozen E135 directional risk head on Nadig."""

from __future__ import annotations

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
E138_ROOT = ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline"
MODEL_PATH = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
CONTRACT_STATUS = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714/RUN_STATUS.json"
OUT = ROOT / "docs/实验结果/E139_nadig_directional_confirmation_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
SCORES_BEFORE_TRUTH = TABLES / "E139_DIRECTIONAL_SCORES_BEFORE_TRUTH.csv"
SEED = 202607139
N_BOOTSTRAP = 3000
ENDPOINTS = ["error_centered_pearson_mean", "error_centered_cosine_mean", "direction_error_rank_target"]
SCORES = ["directional_risk_frozen", "baseline_predicted_magnitude", "safeconf_calibrated_pair_risk", "risk_model_disagreement"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E134 = load_module("e134_for_e139", ROOT / "tools/scripts/run_e134_systema_exact_expression_space_audit.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def score_without_truth():
    model = json.loads(MODEL_PATH.read_text())
    contract = json.loads(CONTRACT_STATUS.read_text())
    if sha256(MODEL_PATH) != contract["frozen_direction_model_sha256"]:
        raise RuntimeError("frozen model hash differs from E136 contract")
    deployable = ["fold_id", "task_id", "setting", "context", "perturbation", *model["features_in_order"]]
    tasks = pd.read_csv(E138_ROOT / "TASK_RISK_TABLE.csv", usecols=deployable)
    matrix = tasks[model["features_in_order"]].to_numpy(float)
    tasks["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(model["coefficients_in_order"], float)
    tasks["target_truth_used_for_score_or_transform"] = False
    tasks["frozen_model_sha256"] = contract["frozen_direction_model_sha256"]
    tasks.to_csv(SCORES_BEFORE_TRUTH, index=False)
    status = {
        "phase": "scores_frozen_before_directional_truth_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_tasks": len(tasks),
        "score_file": str(SCORES_BEFORE_TRUTH.relative_to(ROOT)),
        "score_file_sha256": sha256(SCORES_BEFORE_TRUTH),
        "model_file_sha256": sha256(MODEL_PATH),
        "target_truth_columns_read": [],
    }
    (OUT / "SCORE_FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return status


def exact_audit():
    spec = {
        "root": E138_ROOT,
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714/manifests/E136_TASK_MANIFEST.csv",
        "cache": Path("/home/yyf/data/safeconf_e112_external/Nadig_two_cellline_CONTROL_ONLY_512.npz"),
    }
    frame, source_audit = E134.audit_dataset("Nadig_two_cellline", spec)
    scores = pd.read_csv(SCORES_BEFORE_TRUTH)
    merge_keys = ["fold_id", "task_id", "setting", "context", "perturbation"]
    frame = frame.merge(scores[merge_keys + ["directional_risk_frozen", "frozen_model_sha256", "target_truth_used_for_score_or_transform"]], on=merge_keys, how="left", validate="one_to_one")
    rank_parts = []
    for endpoint in ENDPOINTS[:2]:
        rank_parts.append(frame.groupby("fold_id")[endpoint].transform(lambda values: rankdata(values) / len(values)))
    frame["direction_error_rank_target"] = np.mean(np.stack(rank_parts), axis=0)
    return frame, source_audit


def metrics(tasks: pd.DataFrame):
    rows = []
    for fold, group in tasks.groupby("fold_id", sort=True):
        for score in SCORES:
            for endpoint in ENDPOINTS:
                rows.append({"fold_id": fold, "score": score, "endpoint": endpoint, "n_tasks": len(group), "spearman": rho(group[score], group[endpoint])})
    folds = pd.DataFrame(rows)
    macro = folds.groupby(["score", "endpoint"], as_index=False).agg(n_folds=("fold_id", "nunique"), fold_macro_spearman=("spearman", "mean"), min_fold_spearman=("spearman", "min"))
    return folds, macro


def bootstrap(tasks: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    perturbations = sorted(tasks.perturbation.astype(str).unique())
    pindex = {value: index for index, value in enumerate(perturbations)}
    folds = []
    for _, fold in tasks.groupby("fold_id", sort=False):
        folds.append({
            "cluster": np.asarray([pindex[str(value)] for value in fold.perturbation], int),
            "scores": {score: fold[score].to_numpy(float) for score in SCORES},
            "endpoints": {endpoint: fold[endpoint].to_numpy(float) for endpoint in ENDPOINTS},
        })
    rows = []
    for draw in range(N_BOOTSTRAP):
        counts = rng.multinomial(len(perturbations), np.full(len(perturbations), 1 / len(perturbations)))
        row = {"draw": draw}
        estimates = {score: {endpoint: [] for endpoint in ENDPOINTS} for score in SCORES}
        for fold in folds:
            indices = np.repeat(np.arange(len(fold["cluster"])), counts[fold["cluster"]])
            for score in SCORES:
                for endpoint in ENDPOINTS:
                    estimates[score][endpoint].append(rho(fold["scores"][score][indices], fold["endpoints"][endpoint][indices]))
        for score in SCORES:
            for endpoint in ENDPOINTS:
                row[f"{score}__{endpoint}"] = float(np.nanmean(estimates[score][endpoint]))
        for endpoint in ENDPOINTS:
            row[f"delta_directional_minus_magnitude__{endpoint}"] = row[f"directional_risk_frozen__{endpoint}"] - row[f"baseline_predicted_magnitude__{endpoint}"]
        rows.append(row)
    draws = pd.DataFrame(rows)
    summary = []
    for column in draws.columns[1:]:
        values = draws[column].to_numpy(float)
        summary.append({"metric": column, "bootstrap_draws": N_BOOTSTRAP, "ci_low_2.5pct": np.nanquantile(values, .025), "median": np.nanmedian(values), "ci_high_97.5pct": np.nanquantile(values, .975), "fraction_above_zero": np.nanmean(values > 0)})
    return draws, pd.DataFrame(summary)


def write_report(tasks, macro, boot, baseline, passed):
    direction = macro[macro.score.eq("directional_risk_frozen")].set_index("endpoint")
    b = boot.set_index("metric")
    combined = b.loc["directional_risk_frozen__direction_error_rank_target"]
    lines = [
        "# E139｜Nadig 第七数据方向风险确认",
        "",
        f"## 预注册结论：{'通过' if passed else '未通过'}",
        "",
        f"冻结方向风险在 centered Pearson error 上的两 fold 宏平均 Spearman 为 **{direction.loc['error_centered_pearson_mean','fold_macro_spearman']:.3f}**，centered cosine error 为 **{direction.loc['error_centered_cosine_mean','fold_macro_spearman']:.3f}**。复合方向 rank 的 perturbation-cluster bootstrap 95% CI 为 **[{combined['ci_low_2.5pct']:.3f}, {combined['ci_high_97.5pct']:.3f}]**。",
        "",
        "## 分数对照",
        "",
        "| score | Pearson ρ | cosine ρ | combined rank ρ |",
        "|---|---:|---:|---:|",
    ]
    pivot = macro.pivot(index="score", columns="endpoint", values="fold_macro_spearman")
    for score, row in pivot.iterrows():
        lines.append(f"| {score} | {row['error_centered_pearson_mean']:.3f} | {row['error_centered_cosine_mean']:.3f} | {row['direction_error_rank_target']:.3f} |")
    lines += [
        "",
        "## 上游模型与简单质心",
        "",
        "| dataset | ensemble RMSE | training perturbed-centroid RMSE | ensemble − simple | ensemble 胜出比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.rmse_ensemble_mean:.4f} | {row.rmse_training_perturbed_mean:.4f} | {row.ensemble_minus_simple_baseline:+.4f} | {row.fraction_tasks_ensemble_beats_simple_baseline:.1%} |")
    lines += [
        "",
        "## 信息边界",
        "",
        "- E135 模型哈希在 E136 合同中冻结；E139 第一阶段只读取四个部署特征并写出风险分数，第二阶段才读取向量计算方向误差。",
        "- 两个主要终点、复合终点、竞争分数和全部 3,000 次 bootstrap 均落盘。",
        "- 若未通过，冻结模型不在 Nadig 上调参后重新宣称确认。",
    ]
    (REPORTS / "E139_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E139 先看这个\n\n先读 `reports/E139_REPORT.md`。风险分数冻结记录见 `SCORE_FREEZE_STATUS.json`。\n")


def main():
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    score_status = score_without_truth()
    tasks, source_audit = exact_audit()
    folds, macro = metrics(tasks)
    draws, boot = bootstrap(tasks)
    baseline = E134.E133.baseline_summary(tasks)
    direction = macro[macro.score.eq("directional_risk_frozen")].set_index("endpoint")
    combined = boot.set_index("metric").loc["directional_risk_frozen__direction_error_rank_target"]
    passed = bool(
        direction.loc["error_centered_pearson_mean", "fold_macro_spearman"] > 0
        and direction.loc["error_centered_cosine_mean", "fold_macro_spearman"] > 0
        and combined["ci_low_2.5pct"] > 0
    )
    tasks.to_csv(TABLES / "E139_TASK_AUDIT.csv", index=False)
    folds.to_csv(TABLES / "E139_FOLD_METRICS.csv", index=False)
    macro.to_csv(TABLES / "E139_MACRO_SUMMARY.csv", index=False)
    draws.to_csv(TABLES / "E139_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E139_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    baseline.to_csv(TABLES / "E139_SIMPLE_BASELINE_SUMMARY.csv", index=False)
    pd.DataFrame([source_audit]).to_csv(TABLES / "E139_SOURCE_TRUTH_AUDIT.csv", index=False)
    write_report(tasks, macro, boot, baseline, passed)
    status = {
        "experiment": "E139_nadig_directional_confirmation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "n_folds": int(tasks.fold_id.nunique()),
        "n_test_tasks": len(tasks),
        "bootstrap_draws": N_BOOTSTRAP,
        "score_file_sha256": score_status["score_file_sha256"],
        "model_file_sha256": score_status["model_file_sha256"],
        "all_source_truth_checks_pass": bool(source_audit["truth_reconstruction_pass_atol_1e-5"]),
        "preregistered_directional_gate_passed": passed,
        "model_refit_on_nadig": False,
        "nadig_truth_used_for_score_or_transform": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(macro.to_string(index=False))
    print(boot.to_string(index=False))
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
