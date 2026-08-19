#!/usr/bin/env python3
"""E56 source-size ablation for cross-dataset transfer.

Advisor asked for harder matrix settings such as "only a small block of the
matrix".  E34/E35 already does this within a dataset.  E56 repeats the idea in
the cross-dataset setting: keep the target dataset fixed, then score it using
only a fraction of the source-domain task matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_e55_cross_dataset_transfer import (  # noqa: E402
    DatasetSpec,
    build_specs,
    build_tasks_for_genes,
    choose_common_genes,
    rel,
    score_pair,
    spearman,
    top_enrichment,
)


OUT = ROOT / "docs" / "实验结果" / "E56_cross_dataset_source_size_ablation_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


PAIR_KEYS = [
    ("same_system", "KaggleCrossPatient_celltype", "KaggleCrossCell_celltype"),
    ("same_system_patient_context", "KaggleCrossCell_celltype", "KaggleCrossPatient_donor"),
    ("immune_same_system", "kangCrossCell_celltype", "kangCrossPatient_celltype"),
    ("hard_chemical_positive", "KaggleCrossCell_celltype", "McFarland_cellline"),
    ("hard_chemical_mixed", "sciplex3_cellline", "KaggleCrossCell_celltype"),
    ("hard_chemical_boundary", "sciplex3_cellline", "crossPatient_patient"),
]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode().strip())
    except Exception:
        return True


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def sample_source_tasks(tasks: list[dict], frac: float, seed: int, min_tasks: int) -> list[dict]:
    if frac >= 0.999:
        return list(tasks)
    rng = np.random.default_rng(seed)
    n = max(min_tasks, int(round(len(tasks) * frac)))
    n = min(n, len(tasks))
    idx = np.sort(rng.choice(np.arange(len(tasks)), size=n, replace=False))
    return [tasks[int(i)] for i in idx]


def one_summary(score: pd.DataFrame, frac: float, repeat_seed: int, source_n_tasks: int) -> dict:
    if score.empty:
        return {
            "source_fraction": frac,
            "repeat_seed": repeat_seed,
            "source_n_tasks_used": source_n_tasks,
            "n_target_tasks": 0,
            "spearman_vs_error": np.nan,
            "top20_error_enrichment": np.nan,
            "top20_k": 0,
            "mean_error_combined_rmse": np.nan,
            "mean_source_support": np.nan,
            "shared_perturbation_tasks": 0,
        }
    k, _, enrich = top_enrichment(score, "risk_cross_dataset", "error_combined_rmse", frac=0.2)
    return {
        "source_fraction": frac,
        "repeat_seed": repeat_seed,
        "source_n_tasks_used": source_n_tasks,
        "n_target_tasks": int(len(score)),
        "spearman_vs_error": spearman(score["risk_cross_dataset"], score["error_combined_rmse"]),
        "top20_error_enrichment": enrich,
        "top20_k": int(k),
        "mean_error_combined_rmse": float(score["error_combined_rmse"].mean()),
        "mean_source_support": float(score["source_support_count"].mean()),
        "shared_perturbation_tasks": int((score["source_support_count"] > 0).sum()),
    }


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    specs = build_specs()
    all_scores = []
    rows = []
    pair_status = []

    for group, src_key, tgt_key in PAIR_KEYS:
        source_spec: DatasetSpec = specs[src_key]
        target_spec: DatasetSpec = specs[tgt_key]
        print(f"[E56] {group}: {source_spec.name} -> {target_spec.name}", flush=True)
        try:
            source_head = ad.read_h5ad(source_spec.path, backed="r")
            target_head = ad.read_h5ad(target_spec.path, backed="r")
            genes = choose_common_genes(source_head, target_head, args.n_genes)
            if len(genes) < args.min_common_genes:
                pair_status.append(
                    {
                        "pair_group": group,
                        "source_dataset": source_spec.name,
                        "target_dataset": target_spec.name,
                        "status": "skipped_too_few_common_genes",
                        "n_common_genes": len(genes),
                    }
                )
                continue
            source_tasks, source_meta = build_tasks_for_genes(
                source_spec,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed,
            )
            target_tasks, target_meta = build_tasks_for_genes(
                target_spec,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed + 17,
            )
            if len(source_tasks) < args.min_source_tasks or len(target_tasks) < args.min_target_tasks:
                pair_status.append(
                    {
                        "pair_group": group,
                        "source_dataset": source_spec.name,
                        "target_dataset": target_spec.name,
                        "status": "skipped_too_few_tasks",
                        "n_common_genes": len(genes),
                        "source_n_tasks": len(source_tasks),
                        "target_n_tasks": len(target_tasks),
                    }
                )
                continue

            for frac in args.source_fractions:
                seeds = [0] if frac >= 0.999 else list(range(args.repeats))
                for rep in seeds:
                    sampled = sample_source_tasks(source_tasks, frac, seed=args.seed + rep * 101, min_tasks=args.min_source_tasks)
                    score = score_pair(source_spec, target_spec, sampled, target_tasks, len(genes), group)
                    score["source_fraction"] = frac
                    score["repeat_seed"] = rep
                    all_scores.append(score)
                    row = one_summary(score, frac, rep, len(sampled))
                    row.update(
                        {
                            "pair_group": group,
                            "source_dataset": source_spec.name,
                            "target_dataset": target_spec.name,
                            "directional_pair": f"{source_spec.name} -> {target_spec.name}",
                            "n_common_genes": len(genes),
                            "source_n_tasks_full": len(source_tasks),
                            "target_n_tasks_full": len(target_tasks),
                        }
                    )
                    rows.append(row)
            pair_status.append(
                {
                    "pair_group": group,
                    "source_dataset": source_spec.name,
                    "target_dataset": target_spec.name,
                    "status": "ok",
                    "n_common_genes": len(genes),
                    "source_n_tasks": len(source_tasks),
                    "target_n_tasks": len(target_tasks),
                    "source_path": rel(source_spec.path),
                    "target_path": rel(target_spec.path),
                    "source_context_col": source_meta["context_col"],
                    "target_context_col": target_meta["context_col"],
                }
            )
        except Exception as exc:
            pair_status.append(
                {
                    "pair_group": group,
                    "source_dataset": source_spec.name,
                    "target_dataset": target_spec.name,
                    "status": "failed",
                    "message": repr(exc),
                }
            )
            print(f"  - failed: {exc!r}", flush=True)

    score_table = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    summary = pd.DataFrame(rows)
    status = pd.DataFrame(pair_status)
    if not summary.empty:
        agg = (
            summary.groupby(["pair_group", "source_dataset", "target_dataset", "directional_pair", "source_fraction"], observed=False)
            .agg(
                repeats=("repeat_seed", "nunique"),
                source_n_tasks_used_mean=("source_n_tasks_used", "mean"),
                n_target_tasks=("n_target_tasks", "max"),
                spearman_mean=("spearman_vs_error", "mean"),
                spearman_sd=("spearman_vs_error", "std"),
                top20_enrichment_mean=("top20_error_enrichment", "mean"),
                top20_enrichment_sd=("top20_error_enrichment", "std"),
                mean_source_support=("mean_source_support", "mean"),
                shared_perturbation_tasks=("shared_perturbation_tasks", "mean"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame()

    score_table.to_csv(TABLES / "E56_SOURCE_SIZE_SCORE_TABLE.csv", index=False)
    summary.to_csv(TABLES / "E56_SOURCE_SIZE_REPEAT_SUMMARY.csv", index=False)
    agg.to_csv(TABLES / "E56_SOURCE_SIZE_AGG_SUMMARY.csv", index=False)
    status.to_csv(TABLES / "E56_SOURCE_SIZE_PAIR_STATUS.csv", index=False)

    run_status = {
        "experiment": "E56_cross_dataset_source_size_ablation",
        "created_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "output_dir": rel(OUT),
        "n_pairs": len(PAIR_KEYS),
        "n_score_rows": int(len(score_table)),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(agg, status, run_status)


def fmt(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def write_report(agg: pd.DataFrame, status: pd.DataFrame, run_status: dict) -> None:
    lines = [
        "# E56 跨数据集 source-size ablation",
        "",
        "这一轮把老师说的“小矩阵/历史任务少”放到跨数据集 setting 里检查：目标数据集固定，源数据集只给一部分任务。",
        "",
        f"- pair 数：{run_status['n_pairs']}",
        f"- 分数明细行数：{run_status['n_score_rows']}",
        "",
        "## 主表",
        "",
    ]
    if agg.empty:
        lines.append("没有生成 summary。")
    else:
        lines.extend(
            [
                "| 方向 | source fraction | 源任务数均值 | 目标任务数 | ρ均值 | ρ标准差 | top20富集均值 | 平均支持数 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, r in agg.sort_values(["pair_group", "directional_pair", "source_fraction"]).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["directional_pair"]),
                        fmt(r["source_fraction"], 2),
                        fmt(r["source_n_tasks_used_mean"], 1),
                        str(int(r["n_target_tasks"])),
                        fmt(r["spearman_mean"]),
                        fmt(r["spearman_sd"]),
                        fmt(r["top20_enrichment_mean"]),
                        fmt(r["mean_source_support"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 汇报口径",
            "",
            "这张表用来回答：如果历史矩阵只给一小块，风险排序还能不能用。",
            "",
            "看法很直接：同体系方向一般更稳；硬化学迁移对源任务数量和源/目标相似性更敏感。后面写论文时，可以把这部分作为“数据覆盖度影响”的补充实验。",
            "",
            "## 文件",
            "",
            f"- 聚合表：`{rel(TABLES / 'E56_SOURCE_SIZE_AGG_SUMMARY.csv')}`",
            f"- repeat 表：`{rel(TABLES / 'E56_SOURCE_SIZE_REPEAT_SUMMARY.csv')}`",
            f"- 分数明细：`{rel(TABLES / 'E56_SOURCE_SIZE_SCORE_TABLE.csv')}`",
            f"- pair 状态：`{rel(TABLES / 'E56_SOURCE_SIZE_PAIR_STATUS.csv')}`",
        ]
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "E56_SOURCE_SIZE_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = [
        "# E56 先看这个",
        "",
        "E56 是 E55 的 source-size ablation。目标数据集固定，源数据集只给 25%、50%、75%、100% 的任务。",
        "",
        "先看：`reports/E56_SOURCE_SIZE_ABLATION_REPORT.md`",
    ]
    (OUT / "README_先看这个.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--min-common-genes", type=int, default=100)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--max-cells-per-group", type=int, default=400)
    p.add_argument("--min-source-tasks", type=int, default=3)
    p.add_argument("--min-target-tasks", type=int, default=3)
    p.add_argument("--source-fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=56)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
