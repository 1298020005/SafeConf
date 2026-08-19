#!/usr/bin/env python3
"""E120: run the E112 formal scGPT/GEARS pipeline on frozen Shifrut tasks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "tools/scripts/run_e112_external_formal_dual_models.py"
CONTRACT = ROOT / "docs/实验结果/E119_shifrut_four_context_contract_20260714/manifests/E119_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714"


def load_e112():
    spec = importlib.util.spec_from_file_location("e112_for_e120", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CONTRACT = CONTRACT
    module.OUT = OUT
    module.SPECS = {"Shifrut": {"source": Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/ShifrutMarson2018.h5ad"), "context": "sample"}}
    module.ALIASES.update({"C10orf54": "VSIR", "TCEB2": "ELOB"})
    return module


def finalize():
    root = OUT / "Shifrut"
    tasks = pd.read_csv(root / "TASK_RISK_TABLE.csv")
    status = json.loads((root / "RUN_STATUS.json").read_text())
    root_status = {"experiment": "E120_shifrut_formal_dual_models", "generated_at": datetime.now().isoformat(timespec="seconds"), "status": status["status"], "source_contract": str(CONTRACT.relative_to(ROOT)), "dataset": "ShifrutMarson2018", "context_definition": "donor x TCR stimulation", "n_folds": int(tasks.fold_id.nunique()), "n_test_tasks": len(tasks), "n_prediction_records": int(status["n_records"]), "strict_issue_count": int(status["strict_issue_count"]), "test_truth_used_for_training_calibration_score_or_threshold": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(root_status, ensure_ascii=False, indent=2) + "\n")
    summary = pd.read_csv(root / "SETTING_SUMMARY.csv")
    macro = summary.groupby(["setting", "score"], as_index=False).spearman.mean()
    pooled = []
    for score in ["safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]:
        values = []
        for _, g in tasks.groupby("fold_id"):
            values.append(g[[score, "error_two_predictor_mean_rmse"]].corr(method="spearman").iloc[0, 1])
        pooled.append({"setting": "all_test_settings_pooled", "score": score, "spearman": float(pd.Series(values).mean())})
    report_table = pd.concat([macro, pd.DataFrame(pooled)], ignore_index=True)
    report_table.to_csv(OUT / "E120_SETTING_MACRO.csv", index=False)
    lines = ["# E120｜Shifrut–Marson 四背景正式双模型复制", "", "E120 使用 E119 事先冻结的两供体 × TCR 刺激/未刺激矩阵。scGPT 加载 whole-human 预训练参数并按外层 fold 微调；GEARS 的共表达图只使用训练背景 control。目标受扰动表达在预测和风险分数落盘后才用于评价。", "", f"- 外层 folds：{root_status['n_folds']}", f"- 测试任务：{root_status['n_test_tasks']}", f"- strict PredictionRecord：{root_status['n_prediction_records']}；issues={root_status['strict_issue_count']}", "", "| setting | score | macro Spearman |", "|---|---|---:|"]
    for r in report_table.itertuples(index=False):
        lines.append(f"| {r.setting} | {r.score} | {r.spearman:.3f} |")
    lines += ["", "该数据在 E115/E117 完成后才解封，适合作为新增外部效用验证和误差界修正的独立测试来源。"]
    (OUT / "E120_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E120 先看这个\n\n先读 `E120_REPORT.md`。模型与 strict arrays 在 `Shifrut/`。\n")
    print(json.dumps(root_status, ensure_ascii=False, indent=2))
    print(report_table.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.finalize_only:
        e112 = load_e112()
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(json.dumps(e112.run_dataset("Shifrut", device), ensure_ascii=False, indent=2))
    finalize()


if __name__ == "__main__":
    main()
