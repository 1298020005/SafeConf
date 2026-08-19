#!/usr/bin/env python3
"""E91: freeze PRESCRIBE comparisons on the two existing Norman panels."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/NormanWeissman2019_filtered.h5ad")
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
OUT = ROOT / "docs/实验结果/E91_prescribe_norman_contract_20260712"
SPLITS = {
    "Norman_P1": ROOT / "docs/实验结果/E67_norman_scgpt_formal_fixed_panel_20260711/tables/E67_FIXED_SPLIT.csv",
    "Norman_P2": ROOT / "docs/实验结果/E76b_norman_scgpt_panel2_20260711/tables/E76b_FIXED_SPLIT.csv",
}


def normalize_condition(value: str) -> str:
    parts = str(value).replace("control", "ctrl").split("_")
    if len(parts) == 1 and parts[0] == "ctrl":
        return "ctrl"
    if len(parts) == 1:
        parts.append("ctrl")
    return "+".join(sorted(parts))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    adata = sc.read_h5ad(DATA, backed="r")
    raw_counts = adata.obs["perturbation"].astype(str).map(normalize_condition).value_counts()
    available = set(raw_counts.index)
    manifest_rows = []
    panel_sets = {}
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    for panel, split_path in SPLITS.items():
        frame = pd.read_csv(split_path)
        if set(frame["split"]) != {"train", "val", "test"}:
            raise RuntimeError(f"{panel}: invalid split labels")
        if frame["condition"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicated conditions")
        missing = sorted(set(frame["condition"]) - available)
        if missing:
            raise RuntimeError(f"{panel}: missing raw conditions: {missing[:10]}")
        split_dict = {
            role: frame.loc[frame["split"].eq(role), "condition"].astype(str).tolist()
            for role in ["train", "val", "test"]
        }
        output_pickle = OUT / "manifests" / f"{panel}_set2conditions.pkl"
        with output_pickle.open("wb") as handle:
            pickle.dump(split_dict, handle)
        panel_sets[panel] = set(split_dict["test"])
        for row in frame.itertuples(index=False):
            manifest_rows.append(
                {
                    "panel": panel,
                    "split": row.split,
                    "condition": row.condition,
                    "n_cells": int(raw_counts[row.condition]),
                    "is_single_gene": row.condition.endswith("+ctrl") and row.condition.count("+") == 1,
                    "condition_selected_before_prescribe_run": True,
                    "prescribe_prediction_or_uncertainty_used_for_selection": False,
                    "source_split_csv": str(split_path.relative_to(ROOT)),
                    "split_csv_sha256": sha256_file(split_path),
                }
            )
    overlap = sorted(panel_sets["Norman_P1"] & panel_sets["Norman_P2"])
    if overlap:
        raise RuntimeError(f"Panel test overlap: {overlap}")
    manifest = pd.DataFrame(manifest_rows)
    test = manifest.loc[manifest["split"].eq("test")]
    if len(test) != 48 or not test["is_single_gene"].all() or (test["n_cells"] < 200).any():
        raise RuntimeError("Frozen test panel contract failed")
    manifest.to_csv(OUT / "tables/E91_PRESCRIBE_SPLIT_MANIFEST.csv", index=False)
    summary = manifest.groupby(["panel", "split"], as_index=False).agg(n_conditions=("condition", "size"), min_cells=("n_cells", "min"), median_cells=("n_cells", "median"), max_cells=("n_cells", "max"))
    summary.to_csv(OUT / "tables/E91_SPLIT_SUMMARY.csv", index=False)
    status = {
        "experiment": "E91_prescribe_norman_contract",
        "phase": "contract_frozen_prescribe_outputs_unseen",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATA),
        "dataset_shape": [int(adata.n_obs), int(adata.n_vars)],
        "panels": 2,
        "test_tasks_per_panel": 24,
        "test_overlap": 0,
        "all_test_single_gene": True,
        "all_test_min_cells_200": True,
        "target_selection_uses_prescribe_output": False,
        "prescribe_commit": "6f7264a205aaff654a9594863c5c10b656f88ebe",
        "comparison_scope": "PRESCRIBE native epistemic/aleatoric uncertainty vs its own task error and predicted magnitude; SafeConf GEARS-scGPT disagreement is compared on the identical frozen tasks but not as the same predictor",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E91｜PRESCRIBE × Norman 双面板冻结合同

PRESCRIBE 将在 Norman 的两套既有不重叠面板上分别训练。面板来自 E67 与 E76b，早于本轮 PRESCRIBE 接入；本轮没有根据 PRESCRIBE 的预测、误差或不确定性挑任务。

| panel | train | val | test | test 最少细胞 | 与另一面板重叠 |
|---|---:|---:|---:|---:|---:|
| Norman_P1 | 183 | 20 | 24 | {int(test.loc[test['panel'].eq('Norman_P1'), 'n_cells'].min())} | 0 |
| Norman_P2 | 183 | 20 | 24 | {int(test.loc[test['panel'].eq('Norman_P2'), 'n_cells'].min())} | 0 |

主要比较固定为：

1. PRESCRIBE epistemic、aleatoric 与论文组合分数，对 PRESCRIBE 自身 RMSE 的排序；
2. PRESCRIBE predicted magnitude 对同一 RMSE 的排序；
3. 拒绝最高风险 10%、20%、30% 后的 remaining error 与 AURC；
4. 在相同 48 个任务上并列展示既有 GEARS–scGPT disagreement，但明确二者对应不同 predictor error，不能混成一条相关系数。

PRESCRIBE 是 integrated predictor；SafeConf 是异构 predictor 的 post-hoc pair-risk。公平比较单位是“各自的风险分数能否排序各自预测器的错误”，并同时给 magnitude 基线。不能让 PRESCRIBE uncertainty 去解释 GEARS 或 scGPT 的误差。
"""
    (OUT / "reports/E91_CONTRACT_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E91 先看这个\n\n先读 `reports/E91_CONTRACT_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
