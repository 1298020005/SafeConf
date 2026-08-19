#!/usr/bin/env python3
"""E34/E35 scoring smoke on advisor-requested split manifests.

This is the first executable score-vs-error pass after the split-smoke step.
It uses two existing lightweight reference predictors:

* V0StrongBaseline
* ContextSimilarityBaseline

For each E34/E35 split, it computes deployable risk proxies:

* model disagreement
* predicted magnitude
* inverse support
* inverse context similarity
* a simple z-scored SafeConf smoke risk

Then it evaluates these risks against actual prediction error from the two
reference predictors.  This is a smoke result, not the final formal table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code" / "20260426_154505_perturb_transport_final_push" / "03_code"
TOOLS_DIR = ROOT / "tools" / "scripts"
for p in [CODE_DIR, TOOLS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transport_models import ContextSimilarityBaseline, V0StrongBaseline  # noqa: E402
from run_e34_e35_split_smoke import (  # noqa: E402
    DEFAULT_DATASETS,
    load_scan,
    normalize_task_ids,
    resolve_dataset,
)
from build_context_splits import build_effect_tasks  # noqa: E402


OUT = ROOT / "docs" / "实验结果" / "E34_E35_scoring_smoke_20260709"
SPLIT_OUT = ROOT / "docs" / "实验结果" / "E34_E35_split_smoke_20260709"
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def vec_l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def z(x: pd.Series) -> pd.Series:
    arr = pd.to_numeric(x, errors="coerce")
    sd = float(arr.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(arr)), index=x.index, dtype=float)
    return (arr - float(arr.mean())) / sd


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return float("nan")
    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


def top_enrichment(df: pd.DataFrame, score_col: str, error_col: str, frac: float = 0.2) -> tuple[int, float, float]:
    sub = df[[score_col, error_col]].dropna()
    if len(sub) < 5:
        return 0, float("nan"), float("nan")
    k = max(1, int(np.ceil(len(sub) * frac)))
    top = sub.sort_values(score_col, ascending=False).head(k)
    all_mean = float(sub[error_col].mean())
    top_mean = float(top[error_col].mean())
    return k, top_mean, top_mean / all_mean if all_mean > 1e-12 else float("nan")


def build_tasks(atlas_root: Path, datasets: list[str], n_genes: int, min_cells: int, max_cells_per_group: int) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    scan = load_scan(atlas_root)
    tasks_by_dataset: dict[str, list[dict]] = {}
    rows = []
    for selector in datasets:
        try:
            dataset, path, row = resolve_dataset(scan, selector)
            print(f"[tasks] {selector} -> {dataset}", flush=True)
            tasks, genes, meta = build_effect_tasks(
                path=path,
                dataset=dataset,
                n_genes=n_genes,
                min_cells=min_cells,
                max_cells_per_group=max_cells_per_group,
                seed=0,
            )
            tasks = normalize_task_ids(tasks, dataset)
            if tasks:
                tasks_by_dataset[dataset] = tasks
            rows.append(
                {
                    "dataset_name": dataset,
                    "selector": selector,
                    "path": str(path),
                    "status": "ok" if tasks else "no_tasks",
                    "n_tasks": len(tasks),
                    "n_contexts": len({t["context"] for t in tasks}),
                    "n_perturbations": len({t["perturbation"] for t in tasks}),
                    "context_col": meta.get("context_col", ""),
                    "perturbation_col": meta.get("perturbation_col", ""),
                    "error": "" if tasks else meta.get("error", "no eligible tasks"),
                    "perturbation_type": row.get("perturbation_type", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_name": selector,
                    "selector": selector,
                    "path": "",
                    "status": "failed",
                    "n_tasks": 0,
                    "n_contexts": 0,
                    "n_perturbations": 0,
                    "context_col": "",
                    "perturbation_col": "",
                    "error": repr(exc),
                    "perturbation_type": "",
                }
            )
    return tasks_by_dataset, pd.DataFrame(rows)


def score_one_split(dataset: str, setting: str, split_key: dict, tasks: list[dict], split_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = sorted(split_rows[split_rows["split"].astype(str).str.startswith("train")]["task_id"].astype(int).unique().tolist())
    test_ids = sorted(split_rows[split_rows["split"].astype(str).str.startswith("test")]["task_id"].astype(int).unique().tolist())
    if len(train_ids) < 2 or len(test_ids) < 3:
        return pd.DataFrame(), pd.DataFrame([
            {
                **split_key,
                "dataset_name": dataset,
                "setting": setting,
                "status": "skipped_too_few_tasks",
                "n_train": len(train_ids),
                "n_test": len(test_ids),
            }
        ])

    train_mask = np.zeros(len(tasks), dtype=bool)
    train_mask[train_ids] = True
    test_idx = np.asarray(test_ids, dtype=int)
    try:
        v0 = V0StrongBaseline().fit(tasks, train_mask)
        ctx_model = ContextSimilarityBaseline().fit(tasks, train_mask)
        pred_v0 = v0.predict(tasks, test_idx)
        ctx_details = ctx_model.predict_details(tasks, test_idx)
        pred_ctx = ctx_details["prediction"]
        ctx_sim = ctx_details["transportability"]["transportability_score"].to_numpy(dtype=float)
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([
            {
                **split_key,
                "dataset_name": dataset,
                "setting": setting,
                "status": "failed_predictor",
                "error": repr(exc),
                "n_train": len(train_ids),
                "n_test": len(test_ids),
            }
        ])

    train_support = {}
    for tid in train_ids:
        pert = str(tasks[tid]["perturbation"])
        train_support[pert] = train_support.get(pert, 0) + 1

    rows = []
    for pos, tid in enumerate(test_ids):
        task = tasks[tid]
        true = np.asarray(task["effect"], dtype=float)
        disagreement = rmse(pred_v0[pos], pred_ctx[pos])
        pred_mag = 0.5 * (vec_l2(pred_v0[pos]) + vec_l2(pred_ctx[pos]))
        support = int(train_support.get(str(task["perturbation"]), 0))
        row = {
            **split_key,
            "dataset_name": dataset,
            "setting": setting,
            "task_id": tid,
            "task_key": task["task_key"],
            "context": task["context"],
            "perturbation": task["perturbation"],
            "support_count": support,
            "context_similarity": float(ctx_sim[pos]),
            "model_disagreement_rmse": disagreement,
            "predicted_magnitude": pred_mag,
            "error_v0_rmse": rmse(pred_v0[pos], true),
            "error_contextsim_rmse": rmse(pred_ctx[pos], true),
        }
        row["error_mean_rmse"] = 0.5 * (row["error_v0_rmse"] + row["error_contextsim_rmse"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["risk_disagreement"] = df["model_disagreement_rmse"]
    df["risk_predicted_magnitude"] = df["predicted_magnitude"]
    df["risk_inverse_support"] = -np.log1p(df["support_count"].astype(float))
    df["risk_inverse_context_similarity"] = -df["context_similarity"]
    df["risk_safeconf_smoke"] = (
        z(df["risk_disagreement"])
        + z(df["risk_predicted_magnitude"])
        + z(df["risk_inverse_support"])
        + z(df["risk_inverse_context_similarity"])
    )
    status = pd.DataFrame([
        {
            **split_key,
            "dataset_name": dataset,
            "setting": setting,
            "status": "ok",
            "n_train": len(train_ids),
            "n_test": len(test_ids),
        }
    ])
    return df, status


def build_split_jobs(split_dir: Path) -> list[tuple[str, str, dict, pd.DataFrame]]:
    jobs = []
    sub = pd.read_csv(split_dir / "tables" / "E34_SUBMATRIX_SPLIT_MANIFEST.csv")
    for keys, g in sub.groupby(["dataset_name", "coverage_target", "seed"], dropna=False):
        dataset, coverage, seed = keys
        jobs.append(
            (
                str(dataset),
                "E34_submatrix",
                {"coverage_target": float(coverage), "seed": int(seed), "split_id": f"{dataset}::E34::{coverage}::{seed}"},
                g,
            )
        )
    rc = pd.read_csv(split_dir / "tables" / "E35_ROW_COLUMN_SPLIT_MANIFEST.csv")
    for split_id, g in rc.groupby("split_id", dropna=False):
        setting = str(g["setting"].iloc[0])
        dataset = str(g["dataset_name"].iloc[0])
        heldout = str(g["heldout"].iloc[0])
        jobs.append((dataset, setting, {"split_id": str(split_id), "heldout": heldout}, g))
    return jobs


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    risk_cols = [
        "risk_safeconf_smoke",
        "risk_disagreement",
        "risk_predicted_magnitude",
        "risk_inverse_support",
        "risk_inverse_context_similarity",
    ]
    target_cols = ["error_mean_rmse", "error_v0_rmse", "error_contextsim_rmse"]
    group_cols = ["dataset_name", "setting"]
    extra_cols = []
    if "coverage_target" in scores.columns:
        extra_cols.append("coverage_target")
    rows = []
    for keys, g in scores.groupby(group_cols + extra_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols + extra_cols, keys))
        for risk in risk_cols:
            for target in target_cols:
                k, top_mean, enrich = top_enrichment(g, risk, target, frac=0.2)
                rows.append(
                    {
                        **base,
                        "risk_score_name": risk,
                        "target_error": target,
                        "n_tasks": len(g),
                        "spearman": spearman(g[risk], g[target]),
                        "top20_k": k,
                        "top20_mean_error": top_mean,
                        "top20_enrichment": enrich,
                    }
                )
    return pd.DataFrame(rows)


def write_report(task_status: pd.DataFrame, split_status: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ok_splits = int((split_status["status"] == "ok").sum()) if not split_status.empty else 0
    total_splits = len(split_status)
    best = summary.sort_values("spearman", ascending=False).head(8) if not summary.empty else summary
    text = f"""# E34/E35 scoring smoke

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 已触发

- 数据任务构建成功：{int((task_status['status'] == 'ok').sum())}/{len(task_status)}
- split scoring 成功：{ok_splits}/{total_splits}

## 怎么理解

这不是 formal 结果，但已经完成了周老师要求的下一步实验链第一跳：在小矩阵、整行、整列 split 上，用两个参考预测器产生真实 error，再看 SafeConf smoke risk、disagreement、magnitude、support、context similarity 是否能排序错误。

## 当前最该看

- `tables/E34_E35_SCORING_SUMMARY.csv`
- `tables/E34_E35_SCORE_TABLE.csv`
- `tables/E34_E35_SPLIT_SCORE_STATUS.csv`

## Spearman 最高的前几项

{best.to_string(index=False) if not best.empty else '暂无可汇总结果'}
"""
    (REPORTS / "E34_E35_SCORING_SMOKE_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", default="/home/yyf/data/singlecell_perturbation_atlas")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n-genes", type=int, default=96)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--max-cells-per-group", type=int, default=80)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    tasks_by_dataset, task_status = build_tasks(
        Path(args.atlas_root),
        datasets,
        n_genes=args.n_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
    )
    jobs = build_split_jobs(SPLIT_OUT)
    score_frames = []
    status_frames = []
    for dataset, setting, split_key, split_rows in jobs:
        if dataset not in tasks_by_dataset:
            status_frames.append(
                pd.DataFrame([
                    {
                        **split_key,
                        "dataset_name": dataset,
                        "setting": setting,
                        "status": "skipped_dataset_not_available",
                        "n_train": 0,
                        "n_test": 0,
                    }
                ])
            )
            continue
        scores, stat = score_one_split(dataset, setting, split_key, tasks_by_dataset[dataset], split_rows)
        if not scores.empty:
            score_frames.append(scores)
        status_frames.append(stat)

    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    split_status = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    summary = summarize(scores)
    task_status.to_csv(TABLES / "E34_E35_SCORING_TASK_STATUS.csv", index=False)
    split_status.to_csv(TABLES / "E34_E35_SPLIT_SCORE_STATUS.csv", index=False)
    scores.to_csv(TABLES / "E34_E35_SCORE_TABLE.csv", index=False)
    summary.to_csv(TABLES / "E34_E35_SCORING_SUMMARY.csv", index=False)
    write_report(task_status, split_status, summary)

    status = {
        "run_id": "E34_E35_scoring_smoke_20260709",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "status": "completed",
        "n_score_rows": int(len(scores)),
        "n_split_status_rows": int(len(split_status)),
        "outputs": [
            "tables/E34_E35_SCORING_TASK_STATUS.csv",
            "tables/E34_E35_SPLIT_SCORE_STATUS.csv",
            "tables/E34_E35_SCORE_TABLE.csv",
            "tables/E34_E35_SCORING_SUMMARY.csv",
            "reports/E34_E35_SCORING_SMOKE_REPORT.md",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E34/E35 scoring smoke\n\n"
        "先看 `reports/E34_E35_SCORING_SMOKE_REPORT.md`。这是小矩阵、整行、整列 split 上的第一版 score-vs-error smoke。\n",
        encoding="utf-8",
    )
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
