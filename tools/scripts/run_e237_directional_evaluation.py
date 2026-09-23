#!/usr/bin/env python3
"""Evaluate the E237 frozen score against its predeclared gene-disjoint targets."""

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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E237_nadig_disjoint_gene_confirmation_20260923"
DATASET = "Nadig_E237_disjoint"
TASK_ROOT = OUT / DATASET
SCORES_PATH = OUT / "E237_SCORES_FIXED_BEFORE_DIRECTIONAL_EVALUATION.csv"
CACHE = Path("/home/yyf/data/safeconf_e112_external/Nadig_E237_disjoint_CONTROL_ONLY_512.npz")
KEYS = ["fold_id", "task_id", "setting", "context", "perturbation"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top20(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, group in tasks.groupby("fold_id", sort=True):
        k = math.ceil(.2 * len(group))
        y = group.direction_error_rank_target.to_numpy(float)
        oracle = float(np.sort(y)[-k:].sum())
        for score in ["directional_risk_frozen", "magnitude", "novelty", "disagreement", "magnitude_plus_novelty"]:
            order = np.argsort(-group[score].to_numpy(float), kind="stable")[:k]
            rows.append({"fold_id": fold, "score": score, "n_tasks": len(group), "k": k,
                         "normalized_directional_error_capture": float(y[order].sum() / oracle)})
    return pd.DataFrame(rows)


def main() -> None:
    if (OUT / "E237_EVALUATION_STATUS.json").exists():
        raise RuntimeError("refusing to overwrite existing E237 evaluation")
    contract = json.loads((OUT / "CONTRACT_STATUS.json").read_text())
    score_record = json.loads((OUT / "E237_SCORE_STATUS.json").read_text())
    if sha256(SCORES_PATH) != score_record["score_sha256"]:
        raise RuntimeError("frozen score file changed")
    if sha256(OUT / "manifests/E237_TASK_MANIFEST.csv") != contract["manifest_sha256"]:
        raise RuntimeError("task manifest changed")
    source = load_module("e134_for_e237", ROOT / "tools/scripts/run_e134_systema_exact_expression_space_audit.py")
    audit, integrity = source.audit_dataset(DATASET, {
        "root": TASK_ROOT,
        "task": "TASK_RISK_TABLE.csv",
        "records": "PREDICTION_RECORDS.csv",
        "manifest": OUT / "manifests/E237_TASK_MANIFEST.csv",
        "cache": CACHE,
    })
    scores = pd.read_csv(SCORES_PATH)
    if len(scores) != contract["n_test_rows"] or scores.target_directional_truth_used_for_score.astype(bool).any():
        raise RuntimeError("score count or no-target-truth flag failed")
    fields = KEYS + ["directional_risk_frozen", "baseline_predicted_magnitude",
                     "risk_model_disagreement", "perturbation_novelty"]
    tasks = audit[KEYS + ["error_centered_pearson_mean", "error_centered_cosine_mean"]].merge(
        scores[fields], on=KEYS, how="inner", validate="one_to_one")
    if len(tasks) != len(scores) or not integrity["truth_reconstruction_pass_atol_1e-5"]:
        raise RuntimeError("test task join or truth reconstruction failed")
    tasks["dataset"] = "Nadig"
    tasks["magnitude"] = tasks.baseline_predicted_magnitude.astype(float)
    tasks["novelty"] = tasks.perturbation_novelty.astype(float)
    tasks["disagreement"] = tasks.risk_model_disagreement.astype(float)
    rank_parts = [tasks.groupby("fold_id")[field].transform(lambda x: rankdata(x) / len(x))
                  for field in ["error_centered_pearson_mean", "error_centered_cosine_mean"]]
    tasks["direction_error_rank_target"] = np.mean(np.stack(rank_parts), axis=0)
    tasks["magnitude_plus_novelty"] = (
        tasks.groupby("fold_id").magnitude.transform(lambda x: rankdata(x) / len(x))
        + tasks.groupby("fold_id").novelty.transform(lambda x: rankdata(x) / len(x))) / 2
    comparator = load_module("e236_for_e237", ROOT / "tools/scripts/run_e236_directional_simple_baseline_audit.py")
    folds, comparison = comparator.summarize(tasks, 2000, 20260923)
    utility = top20(tasks)
    primary = comparison[comparison.endpoint.eq("direction_error_rank_target")
                         & comparison.score.eq("directional_risk_frozen")].iloc[0]
    primary_folds = folds[folds.endpoint.eq("direction_error_rank_target")
                          & folds.score.eq("directional_risk_frozen")]
    passed = bool((primary_folds.spearman > 0).all()
                  and primary.delta_vs_magnitude_plus_novelty > 0
                  and primary.delta_vs_magnitude_plus_novelty_ci95_low > 0)
    # This is a small audit table, not a publication substitute; include all folds and comparators.
    tasks.to_csv(OUT / "E237_TASK_AUDIT.csv", index=False)
    folds.to_csv(OUT / "E237_FOLD_SCORES.csv", index=False)
    comparison.to_csv(OUT / "E237_COMPARISON.csv", index=False)
    utility.to_csv(OUT / "E237_TOP20.csv", index=False)
    point = comparison[comparison.endpoint.eq("direction_error_rank_target")][
        ["score", "fold_macro_spearman", "ci95_low", "ci95_high"]]
    lines = ["# E237｜Nadig 未用基因的方向风险复制", "",
             f"预登记主判定：**{'通过' if passed else '未通过'}**。这是同一研究内未用基因复制，不是新的外部研究。",
             "", f"测试行 {len(tasks)}；基因簇 {tasks.perturbation.nunique()}；两折；真值重建误差最大值 {integrity['max_abs_saved_truth_vs_reconstructed_truth']:.2e}。", "",
             "| 分数 | 方向误差折宏平均 ρ | 基因簇 95% 区间 |", "|---|---:|---:|"]
    for row in point.itertuples(index=False):
        lines.append(f"| {row.score} | {row.fold_macro_spearman:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] |")
    lines += ["",
              f"冻结方向风险相对 M+N 的 Δρ={primary.delta_vs_magnitude_plus_novelty:+.3f}；95%区间 [{primary.delta_vs_magnitude_plus_novelty_ci95_low:+.3f}, {primary.delta_vs_magnitude_plus_novelty_ci95_high:+.3f}]。",
              "", "## 执行边界", "",
              "场景在看过 E139/E152 后选定；128 个测试基因与 E136/E139 的 96 个基因不重叠。E112 上游实现会在生成任务图时物化测试真值；公式和名单先已远程封存、评分函数未读取方向真值，但本轮不能标为严格的先交卷后首次读取真值。全部预定任务及不利结果必须保留。", ""]
    (OUT / "E237_REPORT.md").write_text("\n".join(lines))
    status = {"status": "COMPLETE", "created_at": datetime.now().isoformat(timespec="seconds"),
              "preregistered_primary_gate_passed": passed, "n_test_rows": len(tasks),
              "n_gene_clusters": int(tasks.perturbation.nunique()),
              "source_truth_reconstruction_pass": bool(integrity["truth_reconstruction_pass_atol_1e-5"]),
              "frozen_score_sha256": sha256(SCORES_PATH),
              "strict_prior_test_expression_read_blindness": False}
    (OUT / "E237_EVALUATION_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(point.to_string(index=False))


if __name__ == "__main__":
    main()
