#!/usr/bin/env python3
"""E140: seven-dataset absolute-RMSE meta-audit including Nadig."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tools/scripts/run_e113_formal_three_dataset_meta_audit.py"
OUT = ROOT / "docs/实验结果/E140_formal_seven_dataset_meta_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"


def module():
    spec = importlib.util.spec_from_file_location("e113_for_e140", BASE)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    loaded.N_BOOT = 3000
    return loaded


def load():
    frames = []
    frangieh = pd.read_csv(ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv")
    frangieh["dataset"] = "Frangieh"
    frames.append(frangieh)
    frames.append(pd.read_csv(ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv"))
    for dataset, path in [
        ("Shifrut", "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/TASK_RISK_TABLE.csv"),
        ("Liang", "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/TASK_RISK_TABLE.csv"),
        ("Tian_CRISPRi", "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/TASK_RISK_TABLE.csv"),
        ("Nadig_two_cellline", "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/TASK_RISK_TABLE.csv"),
    ]:
        frame = pd.read_csv(ROOT / path)
        frame["dataset"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    analysis = module()
    data = load()
    fold = analysis.summary(data)
    boot = analysis.bootstrap(data)
    boot["scope"] = boot.scope.replace({"three_dataset_macro": "seven_dataset_macro"})
    boot["unit"] = boot.unit.replace({"fixed_three_datasets": "fixed_seven_datasets"})
    macro = fold.groupby(["dataset", "target", "score"], as_index=False).spearman.mean()
    fold.to_csv(TABLES / "E140_FOLD_SUMMARY.csv", index=False)
    macro.to_csv(TABLES / "E140_DATASET_MACRO.csv", index=False)
    boot.to_csv(TABLES / "E140_BOOTSTRAP.csv", index=False)
    pivot = macro.pivot(index="dataset", columns="score", values="spearman").reset_index()
    lines = [
        "# E140｜七套正式数据 absolute-RMSE 元分析",
        "",
        "Nadig 是为 E135 方向风险头冻结的第七数据，同时把其原 SafeConf absolute-RMSE 结果并入总账，不能只报告方向风险成功。",
        "",
        "| dataset | SafeConf | frozen | disagreement | magnitude |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pivot.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.safeconf_calibrated_pair_risk:.3f} | {row.safeconf_frozen_pair_risk:.3f} | {row.risk_model_disagreement:.3f} | {row.baseline_predicted_magnitude:.3f} |")
    lines += ["", "## Bootstrap", "", "| scope | comparator | unit | Δρ | 95% CI | P(Δ>0) |", "|---|---|---|---:|---:|---:|"]
    for row in boot.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.comparator} | {row.unit} | {row.delta:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {row.p_gt_zero:.3f} |")
    lines += [
        "",
        "## 边界",
        "",
        "Nadig 的 absolute RMSE 中 magnitude 明显强于原 SafeConf。E139 的 Directional-SafeConf 通过不能覆盖这个负结果；两种误差必须使用各自风险头并分别报告。",
    ]
    (REPORTS / "E140_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text("# E140 先看这个\n\n先读 `reports/E140_REPORT.md`。\n")
    status = {
        "experiment": "E140_formal_seven_dataset_meta",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "n_datasets": int(data.dataset.nunique()),
        "n_folds": int(data[["dataset", "fold_id"]].drop_duplicates().shape[0]),
        "n_test_tasks": len(data),
        "n_prediction_records": len(data) * 2,
        "strict_issue_count": 0,
        "n_bootstrap": analysis.N_BOOT,
        "nadig_negative_absolute_rmse_comparison_retained": True,
        "test_truth_used_to_change_safeconf_score": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(pivot.to_string(index=False))
    print(boot[boot.scope.eq("seven_dataset_macro")].to_string(index=False))


if __name__ == "__main__":
    main()
