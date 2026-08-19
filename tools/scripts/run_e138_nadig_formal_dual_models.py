#!/usr/bin/env python3
"""E138: formal scGPT/GEARS predictions on the frozen E136 Nadig contract."""

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
CONTRACT = ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714/manifests/E136_TASK_MANIFEST.csv"
ASSET = Path("/home/yyf/data/safeconf_e137_nadig/Nadig_two_cellline_E136_selected.h5ad")
OUT = ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714"


def module():
    spec = importlib.util.spec_from_file_location("e112_for_e138", SOURCE_SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    loaded.CONTRACT = CONTRACT
    loaded.OUT = OUT
    loaded.SPECS = {"Nadig_two_cellline": {"source": ASSET, "context": "context"}}
    # scGPT's pinned vocabulary preserves this legacy mixed-case symbol.
    # Keep it verbatim instead of applying the generic human upper-case rule.
    loaded.ALIASES.update({"C17orf58": "C17orf58"})
    loaded.SEED = 202607138
    return loaded


def finalize():
    root = OUT / "Nadig_two_cellline"
    tasks = pd.read_csv(root / "TASK_RISK_TABLE.csv")
    child = json.loads((root / "RUN_STATUS.json").read_text())
    status = {
        "experiment": "E138_nadig_formal_dual_models",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": child["status"],
        "source_contract": str(CONTRACT.relative_to(ROOT)),
        "source_asset": str(ASSET),
        "dataset": "NadigOConner2024_HepG2_Jurkat",
        "context_definition": "biological cell-line shift",
        "n_folds": int(tasks.fold_id.nunique()),
        "n_test_tasks": len(tasks),
        "n_prediction_records": int(child["n_records"]),
        "strict_issue_count": int(child["strict_issue_count"]),
        "test_truth_used_for_training_calibration_score_or_threshold": False,
        "directional_confirmation_not_evaluated_here": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    rows = []
    for (setting, score), group in pd.read_csv(root / "SETTING_SUMMARY.csv").groupby(["setting", "score"]):
        rows.append({"setting": setting, "score": score, "fold_macro_spearman_absolute_rmse": group.spearman.mean()})
    pd.DataFrame(rows).to_csv(OUT / "E138_ABSOLUTE_RMSE_SUMMARY.csv", index=False)
    (OUT / "E138_REPORT.md").write_text(
        "# E138｜Nadig 双细胞系正式双模型预测\n\n"
        f"两个外层 cell-line holdout folds 已完成，测试任务 {len(tasks)}，strict PredictionRecord {child['n_records']}，issues={child['strict_issue_count']}。"
        "scGPT/GEARS 训练与验证沿用 E112 固定流程；本目录只落盘模型预测和 absolute RMSE，E135 冻结方向风险的确认在 E139 单独执行。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E138 先看这个\n\n先读 `E138_REPORT.md`。第七数据方向确认见 E139。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.finalize_only:
        loaded = module()
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(json.dumps(loaded.run_dataset("Nadig_two_cellline", device), ensure_ascii=False, indent=2))
    finalize()


if __name__ == "__main__":
    main()
