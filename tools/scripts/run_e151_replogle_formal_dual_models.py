#!/usr/bin/env python3
"""E151: formal scGPT/GEARS run on the frozen E149 Replogle contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "tools/scripts/run_e112_external_formal_dual_models.py"
CONTRACT_ROOT = ROOT / "docs/实验结果/E149_replogle_two_cellline_contract_20260714"
CONTRACT = CONTRACT_ROOT / "manifests/E149_TASK_MANIFEST.csv"
ASSET_ROOT = ROOT / "docs/实验结果/E150_replogle_combined_asset_20260714"
ASSET = Path(
    "/home/yyf/data/safeconf_e150_replogle/"
    "Replogle_two_cellline_E149_selected_raw_counts.h5ad"
)
OUT = ROOT / "docs/实验结果/E151_replogle_formal_dual_models_20260714"
DATASET = "Replogle_two_cellline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs() -> dict[str, object]:
    contract_status = json.loads((CONTRACT_ROOT / "RUN_STATUS.json").read_text())
    asset_status = json.loads((ASSET_ROOT / "RUN_STATUS.json").read_text())
    expected_manifest = contract_status["artifact_sha256"]["manifests/E149_TASK_MANIFEST.csv"]
    if sha256(CONTRACT) != expected_manifest:
        raise RuntimeError("E149 manifest changed after freeze")
    expected_asset = asset_status["asset_sha256"]
    if sha256(ASSET) != expected_asset:
        raise RuntimeError("E150 combined asset hash mismatch")
    if asset_status["verified_frozen_artifact_sha256"]["manifests/E149_TASK_MANIFEST.csv"] != expected_manifest:
        raise RuntimeError("E150 was not built from the frozen E149 manifest")
    if not asset_status["all_selected_task_counts_match_e149"]:
        raise RuntimeError("E150 task coverage audit failed")
    return {
        "contract_status_sha256": sha256(CONTRACT_ROOT / "RUN_STATUS.json"),
        "manifest_sha256": expected_manifest,
        "asset_sha256": expected_asset,
        "asset_shape": asset_status["asset_shape"],
    }


def module():
    spec = importlib.util.spec_from_file_location("e112_for_e151", SOURCE_SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    loaded.CONTRACT = CONTRACT
    loaded.OUT = OUT
    loaded.SPECS = {DATASET: {"source": ASSET, "context": "context"}}
    loaded.SEED = 202607151
    return loaded


def finalize(input_audit: dict[str, object]) -> dict[str, object]:
    child_root = OUT / DATASET
    child_status = json.loads((child_root / "RUN_STATUS.json").read_text())
    tasks = pd.read_csv(child_root / "TASK_RISK_TABLE.csv")
    manifest = pd.read_csv(CONTRACT, keep_default_na=False)
    primary_keys = manifest.loc[
        manifest["primary_analysis"].astype(bool),
        ["fold_id", "context", "perturbation", "setting"],
    ].copy()
    primary = tasks.merge(
        primary_keys,
        on=["fold_id", "context", "perturbation", "setting"],
        how="inner",
        validate="one_to_one",
    )
    if len(primary) != 256 or primary.duplicated(["context", "perturbation"]).any():
        raise RuntimeError("E151 primary table is not the frozen 256 unique held-out-context tasks")
    primary.to_csv(child_root / "PRIMARY_TASK_RISK_TABLE.csv", index=False)
    status = {
        "experiment": "E151_replogle_formal_dual_models",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": child_status["status"],
        "source_contract": str(CONTRACT.relative_to(ROOT)),
        "source_asset": str(ASSET),
        "dataset": "ReplogleWeissman2022_K562_RPE1_CRISPRi",
        "context_definition": "control-observed cell-line shift within one study",
        "n_folds": int(tasks.fold_id.nunique()),
        "n_all_test_diagnostic_tasks": int(len(tasks)),
        "n_primary_unique_heldout_context_tasks": int(len(primary)),
        "n_prediction_records": int(child_status["n_records"]),
        "strict_issue_count": int(child_status["strict_issue_count"]),
        "test_truth_used_for_training_calibration_score_or_threshold": False,
        "directional_confirmation_not_evaluated_here": True,
        "input_audit": input_audit,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    summary = []
    for (setting, score), group in pd.read_csv(child_root / "SETTING_SUMMARY.csv").groupby(
        ["setting", "score"], sort=True
    ):
        summary.append(
            {
                "setting": setting,
                "score": score,
                "fold_macro_spearman_absolute_rmse": group.spearman.mean(),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "E151_ABSOLUTE_RMSE_SUMMARY.csv", index=False)
    (OUT / "E151_REPORT.md").write_text(
        "# E151｜Replogle K562/RPE1 正式双模型预测\n\n"
        f"两个外层 cell-line holdout folds 已完成。全部诊断测试任务 {len(tasks)} 个，"
        f"其中预注册主分析是 {len(primary)} 个唯一 held-out cell-line × perturbation 任务；"
        f"strict PredictionRecord {child_status['n_records']} 条，issues={child_status['strict_issue_count']}。\n\n"
        "scGPT 与 GEARS 完整沿用 E112/E138 的固定训练、验证和输出流程。目标细胞系的 control 可见，"
        "目标 perturbed expression 只在预测与风险分数固定后用于评价。因此本实验检验同一研究内的"
        " control-observed 跨细胞系复制，不是跨研究或完全不可见目标背景的 zero-shot。\n"
    )
    (OUT / "README_先看这个.md").write_text(
        "# E151 先看这个\n\n先读 `E151_REPORT.md`；方向风险的预注册 gate 在 E152 单独计算。\n"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    input_audit = validate_inputs()
    if not args.finalize_only:
        loaded = module()
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(json.dumps(loaded.run_dataset(DATASET, device), ensure_ascii=False, indent=2))
    print(json.dumps(finalize(input_audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
