#!/usr/bin/env python3
"""Evaluate E239 against the stronger context+magnitude+novelty comparator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E239_nadig_third_gene_confirmation_20260923"
DATASET = "Nadig_E239_disjoint"
TASK_ROOT = OUT / DATASET
SCORES = OUT / "E239_SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION.csv"
CACHE = Path("/home/yyf/data/safeconf_e112_external/Nadig_E239_disjoint_CONTROL_ONLY_512.npz")
KEYS = ["fold_id", "task_id", "setting", "context", "perturbation"]
STRONG = "source_plus_magnitude_plus_novelty"
METHODS = ["directional_risk_frozen", "magnitude", "novelty", "disagreement",
           "magnitude_plus_novelty", "source_seen", STRONG]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def top20(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in tasks.groupby("fold_id", sort=True):
        k = math.ceil(.2 * len(group))
        y = group.direction_error_rank_target.to_numpy(float)
        oracle = np.sort(y)[-k:].sum()
        for score in METHODS:
            order = np.argsort(-group[score].to_numpy(float), kind="stable")[:k]
            rows.append({"fold_id": fold, "score": score, "n_tasks": len(group), "k": k,
                         "normalized_directional_error_capture": float(y[order].sum() / oracle)})
    return pd.DataFrame(rows)


def sensitivity(tasks: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_csv(TASK_ROOT / "PREDICTION_RECORDS.csv",
                          usecols=["fold_id", "task_id", "true_error_cosine"])
    cosine = records.groupby(["fold_id", "task_id"], as_index=False).true_error_cosine.mean()
    absolute = pd.read_csv(TASK_ROOT / "TASK_RISK_TABLE.csv",
                           usecols=["fold_id", "task_id", "error_two_predictor_mean_rmse"])
    joined = tasks.merge(cosine, on=["fold_id", "task_id"], validate="one_to_one")
    joined = joined.merge(absolute, on=["fold_id", "task_id"], validate="one_to_one")
    rows = []
    for scope, data in (("all_predeclared_test_tasks", joined),
                        ("heldout_cell_line_only", joined[joined.setting.str.startswith("context")])):
        for endpoint in ["direction_error_rank_target", "true_error_cosine", "error_two_predictor_mean_rmse"]:
            for score in ["directional_risk_frozen", "magnitude", STRONG]:
                values = [rho(group[score], group[endpoint]) for _, group in data.groupby("fold_id")]
                rows.append({"scope": scope, "endpoint": endpoint, "score": score,
                             "n_tasks": len(data), "fold_macro_spearman": float(np.mean(values)),
                             "fold1_spearman": values[0], "fold2_spearman": values[1]})
    return pd.DataFrame(rows)


def main() -> None:
    if (OUT / "E239_EVALUATION_STATUS.json").exists():
        raise RuntimeError("refusing to overwrite E239 evaluation")
    contract = json.loads((OUT / "CONTRACT_STATUS.json").read_text())
    score_record = json.loads((OUT / "E239_SCORE_STATUS.json").read_text())
    if sha256(SCORES) != score_record["score_sha256"]:
        raise RuntimeError("E239 frozen score changed")
    if sha256(OUT / "manifests/E239_TASK_MANIFEST.csv") != contract["manifest_sha256"]:
        raise RuntimeError("E239 manifest changed")
    e134 = load_module("e134_for_e239", ROOT / "tools/scripts/run_e134_systema_exact_expression_space_audit.py")
    audit, integrity = e134.audit_dataset(DATASET, {
        "root": TASK_ROOT, "task": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv",
        "manifest": OUT / "manifests/E239_TASK_MANIFEST.csv", "cache": CACHE,
    })
    scores = pd.read_csv(SCORES)
    if len(scores) != contract["n_test_rows"] or scores.target_directional_truth_used_for_score.astype(bool).any():
        raise RuntimeError("E239 frozen score count/no-truth flag failed")
    fields = KEYS + ["directional_risk_frozen", "baseline_predicted_magnitude",
                     "risk_model_disagreement", "perturbation_novelty", "context_novelty_scaled"]
    tasks = audit[KEYS + ["error_centered_pearson_mean", "error_centered_cosine_mean"]].merge(
        scores[fields], on=KEYS, validate="one_to_one")
    if len(tasks) != len(scores) or not integrity["truth_reconstruction_pass_atol_1e-5"]:
        raise RuntimeError("E239 task alignment/truth reconstruction failed")
    tasks["dataset"] = "Nadig"
    tasks["magnitude"] = tasks.baseline_predicted_magnitude
    tasks["novelty"] = tasks.perturbation_novelty
    tasks["disagreement"] = tasks.risk_model_disagreement
    tasks["source_seen"] = -tasks.context_novelty_scaled
    pct = lambda values: rankdata(values) / len(values)
    tasks["magnitude_plus_novelty"] = (tasks.groupby("fold_id").magnitude.transform(pct)
                                         + tasks.groupby("fold_id").novelty.transform(pct)) / 2
    tasks[STRONG] = (tasks.groupby("fold_id").source_seen.transform(pct)
                     + tasks.groupby("fold_id").magnitude.transform(pct)
                     + tasks.groupby("fold_id").novelty.transform(pct)) / 3
    ranks = [tasks.groupby("fold_id")[endpoint].transform(pct)
             for endpoint in ["error_centered_pearson_mean", "error_centered_cosine_mean"]]
    tasks["direction_error_rank_target"] = np.mean(np.stack(ranks), axis=0)
    e236 = load_module("e236_for_e239", ROOT / "tools/scripts/run_e236_directional_simple_baseline_audit.py")
    folds, comparison = e236.summarize(tasks, 2000, 202609239,
                                       scores=METHODS, comparators=["magnitude", STRONG])
    primary = comparison[(comparison.endpoint == "direction_error_rank_target")
                         & (comparison.score == "directional_risk_frozen")].iloc[0]
    fold_table = folds[folds.endpoint == "direction_error_rank_target"].pivot(
        index="fold_id", columns="score", values="spearman")
    fold_differences = fold_table["directional_risk_frozen"] - fold_table[STRONG]
    passed = bool((fold_differences > 0).all()
                  and primary[f"delta_vs_{STRONG}"] > 0
                  and primary[f"delta_vs_{STRONG}_ci95_low"] > 0)
    utility = top20(tasks)
    secondary = sensitivity(tasks)
    tasks.to_csv(OUT / "E239_TASK_AUDIT.csv", index=False)
    folds.to_csv(OUT / "E239_FOLD_SCORES.csv", index=False)
    comparison.to_csv(OUT / "E239_COMPARISON.csv", index=False)
    utility.to_csv(OUT / "E239_TOP20.csv", index=False)
    secondary.to_csv(OUT / "E239_MANDATORY_SENSITIVITY.csv", index=False)
    report = ["# E239｜第三批未用基因：强简单基线确认", "",
              f"预定主判定：**{'通过' if passed else '未通过'}**；两折、{len(tasks)} 行、{tasks.perturbation.nunique()} 个此前未用基因。",
              "", "| 分数 | 混合任务折宏平均 Spearman |", "|---|---:|"]
    for score in METHODS:
        report.append(f"| {score} | {fold_table[score].mean():.3f} |")
    report += ["", f"冻结分数 − 背景见过+幅度+新颖度：{primary[f'delta_vs_{STRONG}']:+.3f}，基因簇95%区间 [{primary[f'delta_vs_{STRONG}_ci95_low']:+.3f}, {primary[f'delta_vs_{STRONG}_ci95_high']:+.3f}]。",
               "", "## 必须同时看的适用边界", "",
               "目标细胞系留出内部以及未中心化 cosine、绝对 RMSE 的完整数字见 `E239_MANDATORY_SENSITIVITY.csv`。即便混合队列主门通过，也不可称为整细胞系留出内部优于幅度或所有错误定义都有效。", "",
               "本轮与 E136/E237 基因完全不重叠，但仍来自 Nadig 同一研究。E112 图构建器提前物化测试目标，故不宣称首次读取真值前严格盲测。", ""]
    (OUT / "E239_REPORT.md").write_text("\n".join(report))
    status = {"status": "COMPLETE", "created_at": datetime.now().isoformat(timespec="seconds"),
              "preregistered_primary_gate_passed": passed, "n_test_rows": len(tasks),
              "n_gene_clusters": int(tasks.perturbation.nunique()),
              "source_truth_reconstruction_pass": bool(integrity["truth_reconstruction_pass_atol_1e-5"]),
              "score_sha256": sha256(SCORES), "strict_first_read_blindness": False}
    (OUT / "E239_EVALUATION_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(primary[["fold_macro_spearman", f"delta_vs_{STRONG}",
                   f"delta_vs_{STRONG}_ci95_low", f"delta_vs_{STRONG}_ci95_high"]])


if __name__ == "__main__":
    main()
