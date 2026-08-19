#!/usr/bin/env python3
"""E84: formal CPA-RDKit runs on untouched E81 manifests.

E81_r1_p75 was inspected during E83 development and is permanently excluded.
All remaining manifests use the same frozen architecture and 20-epoch budget.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
E81 = ROOT / "docs/实验结果/E81_sciplex_cartesian_contract_20260712"
OUT = ROOT / "docs/实验结果/E84_cpa_rdkit_cartesian_formal_20260712"
CORE_PATH = ROOT / "tools/scripts/run_e83_cpa_rdkit_pilot.py"
DEVELOPMENT_MANIFEST = "E81_r1_p75"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def load_core():
    spec = importlib.util.spec_from_file_location("e83_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bootstrap_manifest_mean(values: np.ndarray, seed: int, n_boot: int = 10000):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def run(args) -> None:
    core = load_core()
    core.RUN_TYPE = "formal"
    all_manifests = sorted(pd.read_csv(E81 / "tables/E81_SPLIT_MANIFEST.csv")["manifest_id"].unique())
    manifests = [m for m in all_manifests if m != DEVELOPMENT_MANIFEST]
    status_rows = []
    for index, manifest_id in enumerate(manifests):
        manifest_out = OUT / "manifests" / manifest_id
        status_path = manifest_out / "RUN_STATUS.json"
        if args.resume and status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("strict_issue_count") == 0 and status.get("run_type") == "formal":
                status_rows.append({"manifest_id": manifest_id, "status": "reused_complete", "reason": ""})
                continue
        core.OUT = manifest_out
        run_args = SimpleNamespace(
            manifest_id=manifest_id,
            seed=args.seed + index,
            epochs=20,
            max_cells=32,
            control_cells=64,
            pseudo_cells=16,
            batch_size=64,
            eval_batch_size=128,
            device=args.device,
        )
        try:
            core.predict(run_args)
            core.evaluate(run_args)
            status_rows.append({"manifest_id": manifest_id, "status": "complete", "reason": ""})
        except Exception as exc:
            status_rows.append({"manifest_id": manifest_id, "status": "failed", "reason": repr(exc)})
            pd.DataFrame(status_rows).to_csv(OUT / "E84_PARTIAL_STATUS.csv", index=False)
            raise

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    status_frame = pd.DataFrame(status_rows)
    status_frame.to_csv(OUT / "tables/E84_MANIFEST_STATUS.csv", index=False)

    task_frames, association_frames = [], []
    total_records = 0
    strict_issues = 0
    for manifest_id in manifests:
        directory = OUT / "manifests" / manifest_id
        task = pd.read_csv(directory / "tables/E83_TASK_SCORES.csv")
        task["manifest_id"] = manifest_id
        task_frames.append(task)
        association = pd.read_csv(directory / "tables/E83_RISK_ERROR_SUMMARY.csv")
        association["manifest_id"] = manifest_id
        association_frames.append(association)
        status = json.loads((directory / "RUN_STATUS.json").read_text())
        total_records += int(status["n_prediction_records"])
        strict_issues += int(status["strict_issue_count"])
    tasks = pd.concat(task_frames, ignore_index=True)
    associations = pd.concat(association_frames, ignore_index=True)
    tasks["triangle_pair_mean_bound_holds"] = (
        tasks["pair_mean_rmse"] + 1e-7 >= tasks["cpa_ridge_disagreement_rmse"] / 2.0
    )
    tasks["triangle_pair_max_bound_holds"] = (
        tasks["pair_max_rmse"] + 1e-7 >= tasks["cpa_ridge_disagreement_rmse"] / 2.0
    )
    tasks.to_csv(OUT / "tables/E84_TASK_SCORES.csv", index=False)
    associations.to_csv(OUT / "tables/E84_MANIFEST_ASSOCIATIONS.csv", index=False)

    rows = []
    for quadrant in sorted(tasks["quadrant"].unique()):
        dis = associations.loc[
            associations["quadrant"].eq(quadrant)
            & associations["score_name"].eq("cpa_ridge_disagreement_rmse")
            & associations["target_error"].eq("pair_mean_rmse")
        ].set_index("manifest_id")["spearman"]
        mag = associations.loc[
            associations["quadrant"].eq(quadrant)
            & associations["score_name"].eq("predicted_magnitude_mean")
            & associations["target_error"].eq("pair_mean_rmse")
        ].set_index("manifest_id")["spearman"]
        joined = pd.concat([dis.rename("disagreement"), mag.rename("magnitude")], axis=1).dropna()
        delta = (joined["disagreement"] - joined["magnitude"]).to_numpy(float)
        dis_ci = bootstrap_manifest_mean(joined["disagreement"].to_numpy(float), 20268401)
        mag_ci = bootstrap_manifest_mean(joined["magnitude"].to_numpy(float), 20268402)
        delta_ci = bootstrap_manifest_mean(delta, 20268403)
        rows.append(
            {
                "quadrant": quadrant,
                "n_manifests": len(joined),
                "mean_spearman_disagreement": joined["disagreement"].mean(),
                "disagreement_manifest_bootstrap_ci95_low": dis_ci[0],
                "disagreement_manifest_bootstrap_ci95_high": dis_ci[1],
                "mean_spearman_magnitude": joined["magnitude"].mean(),
                "magnitude_manifest_bootstrap_ci95_low": mag_ci[0],
                "magnitude_manifest_bootstrap_ci95_high": mag_ci[1],
                "mean_delta_disagreement_minus_magnitude": delta.mean(),
                "delta_manifest_bootstrap_ci95_low": delta_ci[0],
                "delta_manifest_bootstrap_ci95_high": delta_ci[1],
                "positive_disagreement_manifests": int((joined["disagreement"] > 0).sum()),
                "positive_delta_manifests": int((delta > 0).sum()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUT / "tables/E84_QUADRANT_AGGREGATE.csv", index=False)

    display = aggregate.copy()
    for column in display.columns:
        if column not in {"quadrant", "n_manifests", "positive_disagreement_manifests", "positive_delta_manifests"}:
            display[column] = display[column].round(3)
    report = f"""# E84｜CPA-RDKit 化学四象限正式复核

E84 在 E81 的 8 个未查看 manifest 上固定运行 CPA 0.8.8 + RDKit Morgan embedding。E83 开发时查看过的 `{DEVELOPMENT_MANIFEST}` 永久排除。8 个正式运行统一使用 20 epochs、相同网络、log10-dose、相同细胞抽样上限和相同评价合同。

- manifest：{len(manifests)}
- manifest-task：{len(tasks)}
- strict PredictionRecord：{total_records}
- strict issues：{strict_issues}
- pair-mean 三角下界：{int(tasks['triangle_pair_mean_bound_holds'].sum())}/{len(tasks)}
- pair-max 三角下界：{int(tasks['triangle_pair_max_bound_holds'].sum())}/{len(tasks)}

## 分象限描述

{markdown_table(display)}

分歧在四个象限的平均相关都为正。相对 predicted magnitude 的增量只在随机缺失 pair 中稳定为正；新 context 和新药两个难象限由 magnitude 稳定占优，双未见的增量区间跨 0。这个结果支持“分歧能排序部分 pair risk”，不支持“它在所有难 setting 中优于简单幅度基线”。

区间通过 manifest 重采样得到，只表示同一 sciPlex3 数据内不同冻结 split 的敏感性，不能写成 8 个独立数据集的外部置信区间。所有负 delta 和失败象限保留。
"""
    (OUT / "reports/E84_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E84 先看这个\n\n先读 `reports/E84_REPORT.md`。\n")
    run_status = {
        "experiment": "E84_cpa_rdkit_cartesian_formal",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "development_manifest_excluded": DEVELOPMENT_MANIFEST,
        "formal_manifests": manifests,
        "epochs_fixed": 20,
        "n_manifest_tasks": len(tasks),
        "n_prediction_records": total_records,
        "strict_issue_count": strict_issues,
        "triangle_pair_mean_violations": int((~tasks["triangle_pair_mean_bound_holds"]).sum()),
        "triangle_pair_max_violations": int((~tasks["triangle_pair_max_bound_holds"]).sum()),
        "target_truth_used_for_scores": False,
        "target_truth_used_for_evaluation_only": True,
        "inference_scope": "within-dataset repeated frozen manifests; not independent datasets",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(run_status, ensure_ascii=False, indent=2))
    print(display.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20268400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
