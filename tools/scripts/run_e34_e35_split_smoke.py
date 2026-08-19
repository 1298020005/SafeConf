#!/usr/bin/env python3
"""E34/E35 split smoke for advisor-requested settings.

This script does not rerun heavy predictors.  It builds the concrete task
matrices and split manifests needed for:

E34: submatrix / low-coverage task settings.
E35: row-holdout and column-holdout task settings.

The point is to trigger the next experimental chain with real data files and
real split manifests rather than a prose-only plan.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code" / "20260426_154505_perturb_transport_final_push" / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_context_splits import build_effect_tasks  # noqa: E402


OUT = ROOT / "docs" / "实验结果" / "E34_E35_split_smoke_20260709"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

DEFAULT_DATASETS = [
    "Haber",
    "Parekh",
    "kangCrossCell",
    "kangCrossPatient",
    "TCDD",
    "sciplex3",
    "ShifrutMarson2018",
    "AissaBenevolenskaya2021",
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


def load_scan(atlas_root: Path) -> pd.DataFrame:
    scan = atlas_root / "metadata" / "h5ad_scan.tsv"
    if not scan.exists():
        raise FileNotFoundError(scan)
    return pd.read_csv(scan, sep="\t")


def resolve_dataset(scan: pd.DataFrame, selector: str) -> tuple[str, Path, pd.Series]:
    sel = selector.lower()
    df = scan.copy()

    def path_tokens(x: object) -> set[str]:
        p = Path(str(x))
        tokens = {p.name.lower(), p.stem.lower(), str(p).lower()}
        # `sciplex3.h5ad.gz` should also match `sciplex3`.
        if p.suffix == ".gz":
            tokens.add(Path(p.stem).stem.lower())
        return tokens

    def score_row(row: pd.Series) -> int:
        local_tokens = path_tokens(row["local_path"])
        file_tokens = path_tokens(row["file_name"])
        if sel in local_tokens:
            return 0
        if sel in file_tokens:
            return 1
        if str(row["study_family"]).lower() == sel:
            return 2
        return 99

    df["_selector_rank"] = df.apply(score_row, axis=1)
    hit = df[df["_selector_rank"] < 99].copy()
    if hit.empty:
        raise FileNotFoundError(selector)
    row = hit.sort_values(["_selector_rank", "file_size_bytes"], ascending=[True, True]).iloc[0]
    name = str(row["study_family"]) if str(row["study_family"]) else Path(str(row["local_path"])).stem
    return name, Path(str(row["local_path"])), row


def normalize_task_ids(tasks: list[dict], dataset: str) -> list[dict]:
    out = []
    for i, t in enumerate(tasks):
        rec = dict(t)
        rec["task_id"] = i
        rec["task_key"] = f"{dataset}::task_{i:05d}"
        out.append(rec)
    return out


def task_summary(dataset: str, path: Path, tasks: list[dict], meta: dict, row: pd.Series, status: str, error: str = "") -> dict:
    contexts = sorted({str(t.get("context", "")) for t in tasks})
    perts = sorted({str(t.get("perturbation", "")) for t in tasks})
    return {
        "dataset_name": dataset,
        "path": str(path),
        "status": status,
        "error": error,
        "n_tasks": len(tasks),
        "n_contexts": len(contexts),
        "n_perturbations": len(perts),
        "context_col": meta.get("context_col", ""),
        "perturbation_col": meta.get("perturbation_col", ""),
        "perturbation_type": row.get("perturbation_type", ""),
        "has_control_like_scan": row.get("has_control_like", ""),
        "suitable_generalization_scan": row.get("suitable_perturbation_generalization", ""),
        "file_size_bytes": row.get("file_size_bytes", ""),
    }


def build_submatrix_splits(dataset: str, tasks: list[dict], coverages: list[float], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    contexts = np.array(sorted({str(t["context"]) for t in tasks}), dtype=object)
    perts = np.array(sorted({str(t["perturbation"]) for t in tasks}), dtype=object)
    if len(contexts) < 2 or len(perts) < 2 or len(tasks) < 6:
        return pd.DataFrame(), pd.DataFrame()
    for coverage in coverages:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n_ctx = max(1, min(len(contexts), int(math.ceil(len(contexts) * math.sqrt(coverage)))))
            n_pert = max(1, min(len(perts), int(math.ceil(len(perts) * math.sqrt(coverage)))))
            visible_ctx = set(rng.choice(contexts, size=n_ctx, replace=False).tolist())
            visible_pert = set(rng.choice(perts, size=n_pert, replace=False).tolist())
            train_ids = []
            test_ids = []
            region_counts: dict[str, int] = {}
            for t in tasks:
                ctx_seen = str(t["context"]) in visible_ctx
                pert_seen = str(t["perturbation"]) in visible_pert
                if ctx_seen and pert_seen:
                    split = "train_visible_block"
                    train_ids.append(int(t["task_id"]))
                else:
                    split = "test_outside_block"
                    test_ids.append(int(t["task_id"]))
                if ctx_seen and pert_seen:
                    region = "inside_visible_block"
                elif ctx_seen and not pert_seen:
                    region = "seen_context_unseen_perturbation"
                elif (not ctx_seen) and pert_seen:
                    region = "unseen_context_seen_perturbation"
                else:
                    region = "unseen_context_unseen_perturbation"
                region_counts[region] = region_counts.get(region, 0) + 1
                rows.append(
                    {
                        "dataset_name": dataset,
                        "setting": "E34_submatrix",
                        "coverage_target": coverage,
                        "seed": seed,
                        "task_id": int(t["task_id"]),
                        "task_key": t["task_key"],
                        "context": t["context"],
                        "perturbation": t["perturbation"],
                        "split": split,
                        "region": region,
                        "context_visible": ctx_seen,
                        "perturbation_visible": pert_seen,
                    }
                )
            summary.append(
                {
                    "dataset_name": dataset,
                    "setting": "E34_submatrix",
                    "coverage_target": coverage,
                    "seed": seed,
                    "n_contexts_total": len(contexts),
                    "n_perturbations_total": len(perts),
                    "n_contexts_visible": n_ctx,
                    "n_perturbations_visible": n_pert,
                    "n_train_visible_block": len(train_ids),
                    "n_test_outside_block": len(test_ids),
                    **{f"n_{k}": v for k, v in region_counts.items()},
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def build_row_column_splits(dataset: str, tasks: list[dict], min_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    contexts = sorted({str(t["context"]) for t in tasks})
    perts = sorted({str(t["perturbation"]) for t in tasks})

    split_specs = []
    for ctx in contexts:
        test = [int(t["task_id"]) for t in tasks if str(t["context"]) == ctx]
        train = [int(t["task_id"]) for t in tasks if str(t["context"]) != ctx]
        if len(test) >= min_test and len(train) >= min_test:
            shared_perts = len({str(tasks[i]["perturbation"]) for i in test} & {str(tasks[i]["perturbation"]) for i in train})
            split_specs.append(("E35_row_holdout", ctx, test, train, shared_perts))
    for pert in perts:
        test = [int(t["task_id"]) for t in tasks if str(t["perturbation"]) == pert]
        train = [int(t["task_id"]) for t in tasks if str(t["perturbation"]) != pert]
        if len(test) >= min_test and len(train) >= min_test:
            shared_ctx = len({str(tasks[i]["context"]) for i in test} & {str(tasks[i]["context"]) for i in train})
            split_specs.append(("E35_column_holdout", pert, test, train, shared_ctx))

    by_id = {int(t["task_id"]): t for t in tasks}
    for setting, heldout, test, train, shared_axis in split_specs:
        split_id = f"{dataset}::{setting}::{heldout}"
        for split_name, ids in [("train", train), ("test", test)]:
            for tid in ids:
                t = by_id[tid]
                rows.append(
                    {
                        "dataset_name": dataset,
                        "setting": setting,
                        "split_id": split_id,
                        "heldout": heldout,
                        "task_id": tid,
                        "task_key": t["task_key"],
                        "context": t["context"],
                        "perturbation": t["perturbation"],
                        "split": split_name,
                    }
                )
        summary.append(
            {
                "dataset_name": dataset,
                "setting": setting,
                "split_id": split_id,
                "heldout": heldout,
                "n_train": len(train),
                "n_test": len(test),
                "shared_opposite_axis_count": shared_axis,
                "cold_start_level": "new_context" if setting.endswith("row_holdout") else "new_perturbation",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def write_report(task_summary_df: pd.DataFrame, sub_summary: pd.DataFrame, rc_summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ok = task_summary_df[task_summary_df["status"].eq("ok")]
    text = f"""# E34/E35 split smoke

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 已触发的内容

- 已检查/构建数据集：{len(task_summary_df)}
- 成功构建任务矩阵：{len(ok)}
- E34 submatrix split rows：{len(sub_summary)}
- E35 row/column split rows：{len(rc_summary)}

## 这一步的意义

这一步把周老师说的“小矩阵、整行、整列”从口头计划变成了真实 split manifest。下一步可以在这些 split 上重算 SafeConf 分数和 predictor error。

## 输出表

- `tables/E34_E35_DATASET_TASK_SUMMARY.csv`
- `tables/E34_SUBMATRIX_SPLIT_MANIFEST.csv`
- `tables/E34_SUBMATRIX_SPLIT_SUMMARY.csv`
- `tables/E35_ROW_COLUMN_SPLIT_MANIFEST.csv`
- `tables/E35_ROW_COLUMN_SPLIT_SUMMARY.csv`

## 明天汇报口径

我已经触发了新 setting 的数据准备：现有 P1/P2/P3 数据都在本地，不需要重新下载；先把小矩阵和整行整列的 split manifest 生成出来。下一步就是在这些 split 上计算 score 与真实 predictor error。
"""
    (REPORTS / "E34_E35_SPLIT_SMOKE_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", default="/home/yyf/data/singlecell_perturbation_atlas")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n-genes", type=int, default=96)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--max-cells-per-group", type=int, default=80)
    parser.add_argument("--min-test", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--coverages", default="0.25,0.50,0.75")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    scan = load_scan(Path(args.atlas_root))
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    coverages = [float(x) for x in args.coverages.split(",") if x.strip()]

    task_summaries = []
    all_task_rows = []
    sub_rows = []
    sub_summaries = []
    rc_rows = []
    rc_summaries = []

    for selector in datasets:
        try:
            dataset, path, row = resolve_dataset(scan, selector)
            print(f"[dataset] {selector} -> {dataset}: {path}", flush=True)
            tasks, genes, meta = build_effect_tasks(
                path=path,
                dataset=dataset,
                n_genes=args.n_genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=0,
            )
            tasks = normalize_task_ids(tasks, dataset)
            status = "ok" if tasks else "no_tasks"
            err = "" if tasks else str(meta.get("error", "no eligible tasks"))
            task_summaries.append(task_summary(dataset, path, tasks, meta, row, status, err))
            for t in tasks:
                all_task_rows.append(
                    {
                        "dataset_name": dataset,
                        "task_id": t["task_id"],
                        "task_key": t["task_key"],
                        "context": t["context"],
                        "perturbation": t["perturbation"],
                        "n_cells": t.get("n_cells", ""),
                        "context_col": t.get("context_col", ""),
                        "perturbation_col": t.get("perturbation_col", ""),
                    }
                )
            if tasks:
                s_rows, s_sum = build_submatrix_splits(dataset, tasks, coverages, seeds)
                r_rows, r_sum = build_row_column_splits(dataset, tasks, args.min_test)
                if not s_rows.empty:
                    sub_rows.append(s_rows)
                if not s_sum.empty:
                    sub_summaries.append(s_sum)
                if not r_rows.empty:
                    rc_rows.append(r_rows)
                if not r_sum.empty:
                    rc_summaries.append(r_sum)
        except Exception as exc:
            task_summaries.append(
                {
                    "dataset_name": selector,
                    "path": "",
                    "status": "failed",
                    "error": repr(exc),
                    "n_tasks": 0,
                    "n_contexts": 0,
                    "n_perturbations": 0,
                }
            )
            print(f"[failed] {selector}: {exc!r}", flush=True)

    task_summary_df = pd.DataFrame(task_summaries)
    task_rows_df = pd.DataFrame(all_task_rows)
    sub_manifest = pd.concat(sub_rows, ignore_index=True) if sub_rows else pd.DataFrame()
    sub_summary = pd.concat(sub_summaries, ignore_index=True) if sub_summaries else pd.DataFrame()
    rc_manifest = pd.concat(rc_rows, ignore_index=True) if rc_rows else pd.DataFrame()
    rc_summary = pd.concat(rc_summaries, ignore_index=True) if rc_summaries else pd.DataFrame()

    task_summary_df.to_csv(TABLES / "E34_E35_DATASET_TASK_SUMMARY.csv", index=False)
    task_rows_df.to_csv(TABLES / "E34_E35_TASK_MATRIX.csv", index=False)
    sub_manifest.to_csv(TABLES / "E34_SUBMATRIX_SPLIT_MANIFEST.csv", index=False)
    sub_summary.to_csv(TABLES / "E34_SUBMATRIX_SPLIT_SUMMARY.csv", index=False)
    rc_manifest.to_csv(TABLES / "E35_ROW_COLUMN_SPLIT_MANIFEST.csv", index=False)
    rc_summary.to_csv(TABLES / "E35_ROW_COLUMN_SPLIT_SUMMARY.csv", index=False)
    write_report(task_summary_df, sub_summary, rc_summary)

    status = {
        "run_id": "E34_E35_split_smoke_20260709",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "status": "completed",
        "datasets_requested": datasets,
        "datasets_ok": task_summary_df[task_summary_df["status"].eq("ok")]["dataset_name"].tolist(),
        "outputs": [
            "tables/E34_E35_DATASET_TASK_SUMMARY.csv",
            "tables/E34_E35_TASK_MATRIX.csv",
            "tables/E34_SUBMATRIX_SPLIT_MANIFEST.csv",
            "tables/E34_SUBMATRIX_SPLIT_SUMMARY.csv",
            "tables/E35_ROW_COLUMN_SPLIT_MANIFEST.csv",
            "tables/E35_ROW_COLUMN_SPLIT_SUMMARY.csv",
            "reports/E34_E35_SPLIT_SMOKE_REPORT.md",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E34/E35 split smoke\n\n"
        "先看 `reports/E34_E35_SPLIT_SMOKE_REPORT.md`。这一步已把小矩阵、整行、整列 setting 的 split manifest 触发生成。\n",
        encoding="utf-8",
    )
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
