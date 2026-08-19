#!/usr/bin/env python3
"""E57 dataset expansion for advisor-requested cross-dataset audit.

E55/E56 established the setting.  E57 adds more local datasets that were
already downloaded but not yet included in the cross-dataset table:

* Lara ex vivo / in vivo / leukemia CRISPR screens;
* Dixit 7-day / 13-day TF perturbation screens;
* Tian CRISPRa / CRISPRi;
* Replogle exp6 / exp7 / exp8;
* Adamson plus Replogle/Tian genetic cross-study checks;
* SciPlex2 / SciPlex4 chemical checks.

This is still a lightweight source-only audit, not a heavy deep-model training
run.
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
    build_tasks_for_genes,
    choose_common_genes,
    rel,
    score_pair,
    summarize_scores,
)


ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
OUT = ROOT / "docs" / "实验结果" / "E57_dataset_expansion_cross_dataset_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


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


def specs() -> dict[str, DatasetSpec]:
    og = ATLAS / "official_generalization"
    sp = ATLAS / "official_scperturb"
    extra = ATLAS / "extra_official" / "cellular_context_generalization"
    return {
        "Lara_exvivo": DatasetSpec(
            name="Lara_exvivo_celltype",
            path=sp / "LaraAstiasoHuntly2023_exvivo.h5ad",
            context_col="celltype",
            perturbation_col="perturbation",
            family="genetic_lara_bone_marrow",
            note="Ex vivo mouse bone marrow CRISPR screen.",
        ),
        "Lara_invivo": DatasetSpec(
            name="Lara_invivo_celltype",
            path=sp / "LaraAstiasoHuntly2023_invivo.h5ad",
            context_col="celltype",
            perturbation_col="perturbation",
            family="genetic_lara_bone_marrow",
            note="In vivo mouse bone marrow CRISPR screen.",
        ),
        "Lara_leukemia": DatasetSpec(
            name="Lara_leukemia_celltype",
            path=sp / "LaraAstiasoHuntly2023_leukemia.h5ad",
            context_col="celltype",
            perturbation_col="perturbation",
            family="genetic_lara_bone_marrow",
            note="Mouse leukemia CRISPR screen.",
        ),
        "Dixit_7d": DatasetSpec(
            name="Dixit_7d_target",
            path=sp / "DixitRegev2016_K562_TFs_7_days.h5ad",
            context_col="cell_line",
            perturbation_col="target",
            control_col="perturbation",
            family="genetic_dixit_tf_time",
            note="K562 TF CRISPR screen, 7 days; target-level grouping.",
        ),
        "Dixit_13d": DatasetSpec(
            name="Dixit_13d_target",
            path=sp / "DixitRegev2016_K562_TFs_13_days.h5ad",
            context_col="cell_line",
            perturbation_col="target",
            control_col="perturbation",
            family="genetic_dixit_tf_time",
            note="K562 TF CRISPR screen, 13 days; target-level grouping.",
        ),
        "TianActivation": DatasetSpec(
            name="TianActivation_batch",
            path=og / "TianActivation.h5ad",
            context_col="batch",
            perturbation_col="perturbation",
            family="genetic_tian_crispr",
            note="CRISPRa screen; batch as context.",
        ),
        "TianInhibition": DatasetSpec(
            name="TianInhibition_batch",
            path=og / "TianInhibition.h5ad",
            context_col="batch",
            perturbation_col="perturbation",
            family="genetic_tian_crispr",
            note="CRISPRi screen; batch as context.",
        ),
        "Replogle_exp6": DatasetSpec(
            name="Replogle_exp6_global",
            path=og / "Replogle_exp6.h5ad",
            context_col="__global__",
            perturbation_col="perturbation",
            family="genetic_replogle_small",
            note="Replogle small experiment 6; global context.",
        ),
        "Replogle_exp7": DatasetSpec(
            name="Replogle_exp7_global",
            path=og / "Replogle_exp7.h5ad",
            context_col="__global__",
            perturbation_col="perturbation",
            family="genetic_replogle_small",
            note="Replogle small experiment 7; global context.",
        ),
        "Replogle_exp8": DatasetSpec(
            name="Replogle_exp8_global",
            path=og / "Replogle_exp8.h5ad",
            context_col="__global__",
            perturbation_col="perturbation",
            family="genetic_replogle_small",
            note="Replogle small experiment 8; global context.",
        ),
        "Adamson": DatasetSpec(
            name="Adamson_global",
            path=og / "Adamson.h5ad",
            context_col="__global__",
            perturbation_col="perturbation",
            family="genetic_k562_cross_study",
            note="Adamson K562 CRISPR screen; global context.",
        ),
        "SciPlex2": DatasetSpec(
            name="SciPlex2_cellline",
            path=sp / "SrivatsanTrapnell2020_sciplex2.h5ad",
            context_col="cell_line",
            perturbation_col="perturbation",
            family="chemical_sciplex_series",
            note="SciPlex2 drug perturbation; A549.",
        ),
        "SciPlex4": DatasetSpec(
            name="SciPlex4_cellline",
            path=sp / "SrivatsanTrapnell2020_sciplex4.h5ad",
            context_col="cell_line",
            perturbation_col="perturbation",
            family="chemical_sciplex_series",
            note="SciPlex4 drug/metabolite perturbation.",
        ),
        "sciplex3_small": DatasetSpec(
            name="sciplex3_small_cellline",
            path=extra / "sciplex3.h5ad",
            context_col="condition1",
            perturbation_col="condition2",
            family="chemical_sciplex_series",
            note="Processed sciplex3 subset already used in E55.",
        ),
    }


def pair_plan(d: dict[str, DatasetSpec]) -> list[tuple[str, DatasetSpec, DatasetSpec]]:
    out: list[tuple[str, DatasetSpec, DatasetSpec]] = []

    for a, b in [
        ("Lara_exvivo", "Lara_invivo"),
        ("Lara_invivo", "Lara_exvivo"),
        ("Lara_exvivo", "Lara_leukemia"),
        ("Lara_leukemia", "Lara_exvivo"),
        ("Lara_invivo", "Lara_leukemia"),
        ("Lara_leukemia", "Lara_invivo"),
    ]:
        out.append(("lara_bone_marrow_cross_condition", d[a], d[b]))

    out.extend(
        [
            ("dixit_timecourse", d["Dixit_7d"], d["Dixit_13d"]),
            ("dixit_timecourse", d["Dixit_13d"], d["Dixit_7d"]),
            ("tian_crispra_crispri", d["TianActivation"], d["TianInhibition"]),
            ("tian_crispra_crispri", d["TianInhibition"], d["TianActivation"]),
        ]
    )

    for a, b in [
        ("Replogle_exp6", "Replogle_exp7"),
        ("Replogle_exp7", "Replogle_exp6"),
        ("Replogle_exp6", "Replogle_exp8"),
        ("Replogle_exp8", "Replogle_exp6"),
        ("Replogle_exp7", "Replogle_exp8"),
        ("Replogle_exp8", "Replogle_exp7"),
    ]:
        out.append(("replogle_small_cross_experiment", d[a], d[b]))

    for a, b in [
        ("Adamson", "Replogle_exp7"),
        ("Replogle_exp7", "Adamson"),
        ("Adamson", "TianInhibition"),
        ("TianInhibition", "Adamson"),
    ]:
        out.append(("hard_genetic_cross_study", d[a], d[b]))

    for a, b in [
        ("SciPlex2", "SciPlex4"),
        ("SciPlex4", "SciPlex2"),
        ("SciPlex2", "sciplex3_small"),
        ("sciplex3_small", "SciPlex2"),
        ("SciPlex4", "sciplex3_small"),
        ("sciplex3_small", "SciPlex4"),
    ]:
        out.append(("sciplex_series_cross_dataset", d[a], d[b]))

    return out


def status_row(group: str, src: DatasetSpec, tgt: DatasetSpec, status: str, message: str = "", **kwargs) -> dict:
    row = {
        "pair_group": group,
        "source_dataset": src.name,
        "target_dataset": tgt.name,
        "directional_pair": f"{src.name} -> {tgt.name}",
        "status": status,
        "message": message,
        "source_path": rel(src.path),
        "target_path": rel(tgt.path),
    }
    row.update(kwargs)
    return row


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    d = specs()
    pairs = pair_plan(d)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    scores = []
    status = []
    dataset_rows = []

    for i, (group, src, tgt) in enumerate(pairs, start=1):
        print(f"[E57] {i}/{len(pairs)} {group}: {src.name} -> {tgt.name}", flush=True)
        try:
            src_head = ad.read_h5ad(src.path, backed="r")
            tgt_head = ad.read_h5ad(tgt.path, backed="r")
            genes = choose_common_genes(src_head, tgt_head, args.n_genes)
            if len(genes) < args.min_common_genes:
                msg = f"too few common genes: {len(genes)} < {args.min_common_genes}"
                status.append(status_row(group, src, tgt, "skipped_too_few_common_genes", msg, n_common_genes=len(genes)))
                print(f"  - skip: {msg}", flush=True)
                continue

            source_tasks, source_meta = build_tasks_for_genes(
                src,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed,
            )
            target_tasks, target_meta = build_tasks_for_genes(
                tgt,
                genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed + 57,
            )
            dataset_rows.extend([{**source_meta, "role": "source", "pair": f"{src.name}->{tgt.name}"}, {**target_meta, "role": "target", "pair": f"{src.name}->{tgt.name}"}])
            if len(source_tasks) < args.min_source_tasks or len(target_tasks) < args.min_target_tasks:
                msg = f"too few tasks: source={len(source_tasks)}, target={len(target_tasks)}"
                status.append(
                    status_row(
                        group,
                        src,
                        tgt,
                        "skipped_too_few_tasks",
                        msg,
                        n_common_genes=len(genes),
                        source_n_tasks=len(source_tasks),
                        target_n_tasks=len(target_tasks),
                    )
                )
                print(f"  - skip: {msg}", flush=True)
                continue

            score = score_pair(src, tgt, source_tasks, target_tasks, len(genes), group)
            scores.append(score)
            status.append(
                status_row(
                    group,
                    src,
                    tgt,
                    "ok",
                    "scored",
                    n_common_genes=len(genes),
                    source_n_tasks=len(source_tasks),
                    target_n_tasks=len(target_tasks),
                    source_n_contexts=source_meta["n_contexts"],
                    target_n_contexts=target_meta["n_contexts"],
                    source_n_perturbations=source_meta["n_perturbations"],
                    target_n_perturbations=target_meta["n_perturbations"],
                )
            )
            print(f"  - ok: source_tasks={len(source_tasks)}, target_tasks={len(target_tasks)}, genes={len(genes)}", flush=True)
        except Exception as exc:
            status.append(status_row(group, src, tgt, "failed", repr(exc)))
            print(f"  - failed: {exc!r}", flush=True)

    score_table = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    summary = summarize_scores(score_table)
    status_df = pd.DataFrame(status)
    dataset_df = pd.DataFrame(dataset_rows).drop_duplicates().reset_index(drop=True) if dataset_rows else pd.DataFrame()

    score_table.to_csv(TABLES / "E57_DATASET_EXPANSION_SCORE_TABLE.csv", index=False)
    summary.to_csv(TABLES / "E57_DATASET_EXPANSION_SUMMARY.csv", index=False)
    status_df.to_csv(TABLES / "E57_DATASET_EXPANSION_PAIR_STATUS.csv", index=False)
    dataset_df.to_csv(TABLES / "E57_DATASET_EXPANSION_DATASET_STATUS.csv", index=False)

    run_status = {
        "experiment": "E57_dataset_expansion_cross_dataset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "output_dir": rel(OUT),
        "n_pairs_planned": len(pairs),
        "n_pairs_ok": int((status_df["status"] == "ok").sum()) if not status_df.empty else 0,
        "n_score_rows": int(len(score_table)),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, status_df, dataset_df, run_status)


def fmt(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def write_report(summary: pd.DataFrame, status: pd.DataFrame, dataset_status: pd.DataFrame, run_status: dict) -> None:
    main = summary[summary["score_col"] == "risk_cross_dataset"].copy() if not summary.empty else pd.DataFrame()
    if not main.empty:
        main = main.sort_values(["pair_group", "spearman_vs_error"], ascending=[True, False])

    lines = [
        "# E57 数据集扩容：跨数据集审计",
        "",
        "E55/E56 先把老师点名的跨数据集 setting 跑通。E57 继续加本地已下载的数据集，检查信号是否只来自 Kaggle/Kang 那几组。",
        "",
        f"- 计划方向对：{run_status['n_pairs_planned']}",
        f"- 成功打分方向对：{run_status['n_pairs_ok']}",
        f"- 目标任务打分行数：{run_status['n_score_rows']}",
        "",
        "新增覆盖：Lara 骨髓 CRISPR、Dixit TF 时间点、Tian CRISPRa/i、Replogle 小实验、Adamson/Replogle/Tian 跨研究、SciPlex2/4 化学扰动。",
        "",
        "## 主结果",
        "",
    ]
    if main.empty:
        lines.append("暂无 summary。")
    else:
        lines.extend(
            [
                "| 分组 | 方向 | 任务数 | 共同基因 | 共享扰动任务 | ρ(risk,error) | top20 错误富集 | 平均误差 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, r in main.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["pair_group"]),
                        str(r["directional_pair"]),
                        str(int(r["n_target_tasks"])),
                        str(int(r["n_common_genes"])),
                        str(int(r["shared_perturbation_tasks"])),
                        fmt(r["spearman_vs_error"]),
                        fmt(r["top20_error_enrichment"]),
                        fmt(r["mean_error_combined_rmse"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 读法",
            "",
            "这批结果主要用来增加数据覆盖面。强结果可以作为证据，弱结果和跳过项照样有用：它们说明哪些生物体系、共同基因、control 结构和任务数量会限制跨数据集打分。",
            "",
            "## 文件",
            "",
            f"- 汇总表：`{rel(TABLES / 'E57_DATASET_EXPANSION_SUMMARY.csv')}`",
            f"- 分数明细：`{rel(TABLES / 'E57_DATASET_EXPANSION_SCORE_TABLE.csv')}`",
            f"- pair 状态：`{rel(TABLES / 'E57_DATASET_EXPANSION_PAIR_STATUS.csv')}`",
            f"- 数据任务状态：`{rel(TABLES / 'E57_DATASET_EXPANSION_DATASET_STATUS.csv')}`",
        ]
    )
    (REPORTS / "E57_DATASET_EXPANSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# E57 先看这个",
        "",
        "E57 是数据集扩容版跨数据集审计。它接在 E55/E56 后面，新增 Lara、Dixit、Tian、Replogle、Adamson、SciPlex2/4。",
        "",
        "先看：`reports/E57_DATASET_EXPANSION_REPORT.md`",
    ]
    (OUT / "README_先看这个.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-genes", type=int, default=500)
    p.add_argument("--min-common-genes", type=int, default=100)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--max-cells-per-group", type=int, default=300)
    p.add_argument("--min-source-tasks", type=int, default=3)
    p.add_argument("--min-target-tasks", type=int, default=3)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--max-pairs", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
