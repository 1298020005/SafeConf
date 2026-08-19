#!/usr/bin/env python3
"""E41 first-batch multidimensional smoke experiments.

This script turns the E40 data-line ledger into the first executable records.
It deliberately stays lightweight:

1. OpenProblems / NeurIPS 2023 Kaggle DGE
   - uses the completed train/test/prediction h5ad files;
   - computes real prediction error from `prediction.h5ad` vs test `logFC`;
   - builds deployable risk proxies from support, SMILES similarity and
     predicted magnitude.

2. Tahoe-100M raw single-cell shards
   - audits complete raw shards only;
   - records drug / cell line / MoA / SMILES / plate coverage;
   - does not wait for the full 3388-shard download.

The output is a reproducible notebook-free experiment folder under docs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
OUT = ROOT / "docs" / "实验结果" / "E41_multidim_first_batch_smoke_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

OPENPROBLEMS = ATLAS / "mega_external" / "OpenProblems_NeurIPS2023_single_cell_perturbations" / "data"
OPENPROBLEMS_KAGGLE = OPENPROBLEMS / "workflow_resources" / "neurips-2023-kaggle"
TAHOE_RAW = ATLAS / "mega_external" / "Tahoe-100M" / "data"
TAHOE_META = ATLAS / "mega_external" / "Tahoe-100M" / "metadata"


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


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        try:
            return path.relative_to(ATLAS).as_posix()
        except Exception:
            return str(path)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def zscore(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / sd


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return float("nan")
    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


def top_enrichment(df: pd.DataFrame, score_col: str, error_col: str, frac: float = 0.2) -> tuple[int, float, float, float]:
    sub = df[[score_col, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return 0, float("nan"), float("nan"), float("nan")
    k = max(1, int(math.ceil(len(sub) * frac)))
    top = sub.sort_values(score_col, ascending=False).head(k)
    all_mean = float(sub[error_col].mean())
    top_mean = float(top[error_col].mean())
    return k, all_mean, top_mean, top_mean / all_mean if all_mean > 1e-12 else float("nan")


def smiles_ngrams(smiles: object, n: int = 3) -> set[str]:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return set()
    text = text.replace(" ", "")
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def load_layer(adata, layer: str) -> np.ndarray:
    if layer not in adata.layers:
        raise KeyError(f"missing layer: {layer}")
    return as_dense(adata.layers[layer])


def run_openproblems_dge() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_path = OPENPROBLEMS_KAGGLE / "de_train.h5ad"
    test_path = OPENPROBLEMS_KAGGLE / "de_test.h5ad"
    pred_path = OPENPROBLEMS_KAGGLE / "prediction.h5ad"

    missing = [p for p in [train_path, test_path, pred_path] if not p.exists()]
    if missing:
        status = {
            "status": "missing_files",
            "missing": [str(p) for p in missing],
        }
        return pd.DataFrame(), pd.DataFrame(), status

    train = ad.read_h5ad(train_path)
    test = ad.read_h5ad(test_path)
    pred = ad.read_h5ad(pred_path)

    y_true = load_layer(test, "logFC").astype(np.float32)
    y_pred = load_layer(pred, "prediction").astype(np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"prediction shape {y_pred.shape} != truth shape {y_true.shape}")

    if "is_de_adj" in test.layers:
        de_mask = as_dense(test.layers["is_de_adj"]).astype(bool)
    elif "is_de" in test.layers:
        de_mask = as_dense(test.layers["is_de"]).astype(bool)
    else:
        de_mask = np.zeros_like(y_true, dtype=bool)

    train_obs = train.obs.copy()
    test_obs = test.obs.copy()
    train_obs["_sm_name"] = train_obs["sm_name"].astype(str)
    train_obs["_cell_type"] = train_obs["cell_type"].astype(str)
    test_obs["_sm_name"] = test_obs["sm_name"].astype(str)
    test_obs["_cell_type"] = test_obs["cell_type"].astype(str)

    drug_support = train_obs["_sm_name"].value_counts().to_dict()
    cell_support = train_obs["_cell_type"].value_counts().to_dict()
    pair_support = train_obs.groupby(["_sm_name", "_cell_type"], observed=False).size().to_dict()

    train_smiles = (
        train_obs[["_sm_name", "SMILES"]]
        .drop_duplicates()
        .assign(_ngram=lambda d: d["SMILES"].astype(str).apply(smiles_ngrams))
    )
    train_smiles_sets = list(train_smiles["_ngram"])

    rows = []
    for i, (_, obs) in enumerate(test_obs.iterrows()):
        diff = y_pred[i] - y_true[i]
        absdiff = np.abs(diff)
        mask = de_mask[i]
        de_n = int(mask.sum())
        rmse_de = float(np.sqrt(np.mean(diff[mask] ** 2))) if de_n > 0 else float("nan")
        smiles_set = smiles_ngrams(obs.get("SMILES", ""))
        nearest_smiles = max((jaccard(smiles_set, s) for s in train_smiles_sets), default=0.0)
        sm_name = str(obs["_sm_name"])
        cell_type = str(obs["_cell_type"])
        rows.append(
            {
                "task_id": int(obs.get("id", i)),
                "row_index": i,
                "cell_type": cell_type,
                "sm_name": sm_name,
                "sm_lincs_id": str(obs.get("sm_lincs_id", "")),
                "SMILES": str(obs.get("SMILES", "")),
                "dose_uM": float(obs.get("dose_uM", np.nan)),
                "split": str(obs.get("split", "")),
                "drug_support_rows": int(drug_support.get(sm_name, 0)),
                "cell_type_support_rows": int(cell_support.get(cell_type, 0)),
                "pair_support_rows": int(pair_support.get((sm_name, cell_type), 0)),
                "nearest_train_smiles_jaccard": float(nearest_smiles),
                "predicted_l2": float(np.linalg.norm(y_pred[i])),
                "predicted_abs_mean": float(np.mean(np.abs(y_pred[i]))),
                "true_l2_diagnostic": float(np.linalg.norm(y_true[i])),
                "de_adj_gene_count": de_n,
                "rmse_all": float(np.sqrt(np.mean(diff**2))),
                "mae_all": float(np.mean(absdiff)),
                "rmse_de_adj": rmse_de,
            }
        )

    task_df = pd.DataFrame(rows)
    task_df["risk_low_drug_support"] = -zscore(np.log1p(task_df["drug_support_rows"]))
    task_df["risk_low_pair_support"] = -zscore(np.log1p(task_df["pair_support_rows"]))
    task_df["risk_low_cell_type_support"] = -zscore(np.log1p(task_df["cell_type_support_rows"]))
    task_df["risk_low_smiles_similarity"] = -zscore(task_df["nearest_train_smiles_jaccard"])
    task_df["risk_predicted_magnitude"] = zscore(task_df["predicted_l2"])
    task_df["risk_oracle_true_magnitude_diagnostic"] = zscore(task_df["true_l2_diagnostic"])
    task_df["risk_safeconf_op_smoke"] = (
        task_df["risk_low_drug_support"]
        + task_df["risk_low_pair_support"]
        + task_df["risk_low_cell_type_support"]
        + task_df["risk_low_smiles_similarity"]
        + task_df["risk_predicted_magnitude"]
    )

    risk_cols = [
        "risk_safeconf_op_smoke",
        "risk_predicted_magnitude",
        "risk_low_drug_support",
        "risk_low_pair_support",
        "risk_low_cell_type_support",
        "risk_low_smiles_similarity",
        "risk_oracle_true_magnitude_diagnostic",
    ]
    error_cols = ["rmse_all", "mae_all", "rmse_de_adj"]
    summary_rows = []
    for risk_col in risk_cols:
        for error_col in error_cols:
            k, all_mean, top_mean, enrichment = top_enrichment(task_df, risk_col, error_col)
            summary_rows.append(
                {
                    "dataset_name": "OpenProblems_NeurIPS2023_Kaggle_DGE",
                    "risk_score_name": risk_col,
                    "target_error": error_col,
                    "n_tasks": int(task_df[[risk_col, error_col]].dropna().shape[0]),
                    "spearman": spearman(task_df[risk_col], task_df[error_col]),
                    "top20_k": k,
                    "all_mean_error": all_mean,
                    "top20_mean_error": top_mean,
                    "top20_enrichment": enrichment,
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["target_error", "spearman"], ascending=[True, False], kind="stable"
    )

    status = {
        "status": "ok",
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "prediction_shape": list(pred.shape),
        "n_train_drugs": int(train_obs["_sm_name"].nunique()),
        "n_test_drugs": int(test_obs["_sm_name"].nunique()),
        "n_test_tasks": int(len(task_df)),
        "test_cell_types": sorted(test_obs["_cell_type"].dropna().unique().tolist()),
        "train_cell_types": sorted(train_obs["_cell_type"].dropna().unique().tolist()),
    }
    return task_df, summary_df, status


def complete_tahoe_shards() -> list[Path]:
    if not TAHOE_RAW.exists():
        return []
    shards = sorted(TAHOE_RAW.glob("train-*.parquet"))
    return [p for p in shards if not Path(str(p) + ".aria2").exists()]


def run_tahoe_raw_metadata(max_shards: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    complete = complete_tahoe_shards()
    selected = complete[:max_shards]
    if not selected:
        status = {"status": "no_complete_shards", "complete_shards": 0}
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), status

    needed_cols = ["drug", "cell_line_id", "moa-fine", "canonical_smiles", "pubchem_cid", "plate"]
    file_rows = []
    long_count_rows = []
    schema_rows = []

    for path in selected:
        pf = pq.ParquetFile(path)
        schema_names = pf.schema_arrow.names
        cols = [c for c in needed_cols if c in schema_names]
        schema_rows.append(
            {
                "file_name": path.name,
                "relative_path": rel(path),
                "n_rows_metadata": int(pf.metadata.num_rows),
                "schema_columns": ";".join(schema_names),
                "audited_columns": ";".join(cols),
            }
        )
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas()
        rec = {
            "file_name": path.name,
            "relative_path": rel(path),
            "n_rows": int(len(df)),
        }
        for col in cols:
            rec[f"n_unique_{col}"] = int(df[col].nunique(dropna=True))
            top = df[col].value_counts(dropna=True).head(5)
            rec[f"top5_{col}"] = " | ".join(f"{k}:{int(v)}" for k, v in top.items())
            for value, count in top.items():
                long_count_rows.append(
                    {
                        "field": col,
                        "value": str(value),
                        "count_in_audited_shards": int(count),
                        "source_file": path.name,
                    }
                )
        file_rows.append(rec)

    file_df = pd.DataFrame(file_rows)
    long_counts = pd.DataFrame(long_count_rows)
    if not long_counts.empty:
        aggregate = (
            long_counts.groupby(["field", "value"], observed=False)["count_in_audited_shards"]
            .sum()
            .reset_index()
            .sort_values(["field", "count_in_audited_shards"], ascending=[True, False], kind="stable")
        )
    else:
        aggregate = pd.DataFrame(columns=["field", "value", "count_in_audited_shards"])

    gene_meta = TAHOE_META / "gene_metadata.parquet"
    obs_meta = TAHOE_META / "obs_metadata.parquet"
    gene_rows = 0
    obs_rows = 0
    if gene_meta.exists():
        gene_rows = int(pq.ParquetFile(gene_meta).metadata.num_rows)
    if obs_meta.exists():
        obs_rows = int(pq.ParquetFile(obs_meta).metadata.num_rows)

    status = {
        "status": "ok",
        "complete_shards_seen": int(len(complete)),
        "partial_shard_markers": int(len(list(TAHOE_RAW.glob("*.aria2")))),
        "audited_shards": int(len(selected)),
        "audited_rows": int(file_df["n_rows"].sum()) if not file_df.empty else 0,
        "gene_metadata_rows": gene_rows,
        "obs_metadata_rows": obs_rows,
    }
    return file_df, aggregate, pd.DataFrame(schema_rows), status


def build_first_batch_queue(open_status: dict, tahoe_status: dict) -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "experiment_id": "E41A",
            "experiment_name": "OpenProblems Kaggle DGE smoke",
            "data_line": "PBMC chemical / cell type / SMILES",
            "input": "workflow_resources/neurips-2023-kaggle/de_train.h5ad + de_test.h5ad + prediction.h5ad",
            "current_status": "finished" if open_status.get("status") == "ok" else "blocked",
            "what_it_answers": "独立小分子 benchmark 上，风险代理能不能提前找到高误差任务。",
        },
        {
            "priority": 2,
            "experiment_id": "E41B",
            "experiment_name": "Tahoe raw shard metadata audit",
            "data_line": "large-scale raw single-cell / drug / cell line / MoA / plate",
            "input": "Tahoe-100M raw complete shards",
            "current_status": "finished" if tahoe_status.get("status") == "ok" else "blocked",
            "what_it_answers": "Tahoe raw 已下载部分是否足以支持多维度分层实验。",
        },
        {
            "priority": 3,
            "experiment_id": "E42",
            "experiment_name": "sciplex3 cell-line holdout",
            "data_line": "chemical / dose / A549-K562-MCF7",
            "input": "official_generalization/sciplex3_A549/K562/MCF7.h5ad",
            "current_status": "queued",
            "what_it_answers": "同一药物换细胞系时，SafeConf 是否能识别失败。",
        },
        {
            "priority": 4,
            "experiment_id": "E43",
            "experiment_name": "TCDD dose holdout",
            "data_line": "single compound / dose response / liver cell type",
            "input": "extra_official/cellular_context_generalization/TCDD.h5ad",
            "current_status": "queued",
            "what_it_answers": "剂量变化是否造成预测误差和风险上升。",
        },
        {
            "priority": 5,
            "experiment_id": "E44",
            "experiment_name": "KaggleCrossPatient donor holdout",
            "data_line": "PBMC chemical / donor shift",
            "input": "extra_official/cellular_context_generalization/KaggleCrossPatient.h5ad",
            "current_status": "queued",
            "what_it_answers": "不同供体/病人之间迁移是否是主要失败来源。",
        },
        {
            "priority": 6,
            "experiment_id": "E45",
            "experiment_name": "crossSpecies species holdout",
            "data_line": "species / LPS / time",
            "input": "extra_official/cellular_context_generalization/crossSpecies.h5ad",
            "current_status": "queued",
            "what_it_answers": "强跨物种 domain shift 下风险信号是否仍有效。",
        },
        {
            "priority": 7,
            "experiment_id": "E46",
            "experiment_name": "Norman single-to-combo",
            "data_line": "genetic combination",
            "input": "official_generalization/Norman.h5ad",
            "current_status": "queued",
            "what_it_answers": "单基因信息外推到组合扰动时，失败是否更集中。",
        },
        {
            "priority": 8,
            "experiment_id": "E47",
            "experiment_name": "Gasperini regulatory target holdout",
            "data_line": "regulatory / guide-level sparsity",
            "input": "official_scperturb/GasperiniShendure2019_*.h5ad",
            "current_status": "queued",
            "what_it_answers": "调控元件/guide 标签稀疏是否让 SafeConf 的 support 项变重要。",
        },
        {
            "priority": 9,
            "experiment_id": "E48",
            "experiment_name": "Papalexi RNA-protein consistency",
            "data_line": "multimodal RNA + protein",
            "input": "PapalexiSatija2021_eccite_RNA/protein.h5ad",
            "current_status": "queued",
            "what_it_answers": "RNA 层风险能否对应 protein 表型失败。",
        },
    ]
    return pd.DataFrame(rows)


def write_report(open_summary: pd.DataFrame, open_status: dict, tahoe_file: pd.DataFrame, tahoe_counts: pd.DataFrame, tahoe_status: dict, queue: pd.DataFrame) -> None:
    lines = []
    lines.append("# E41 多维数据第一批 smoke\n")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append(f"- Git：`{git_head()[:12]}`")
    lines.append(f"- 工作区 dirty：`{git_dirty()}`\n")

    lines.append("## 1. 这次实际做了什么\n")
    lines.append("- OpenProblems / NeurIPS 2023 Kaggle DGE：用官方 prediction 和 test logFC 计算真实误差，再看 support、SMILES 相似度、预测幅度这些前置风险线索。")
    lines.append("- Tahoe raw：读取已完成 shard 的字段，统计 drug、cell line、MoA、SMILES、PubChem、plate 覆盖。")
    lines.append("- 同时生成下一批实验队列，避免后续只停留在“想做”。\n")

    lines.append("## 2. OpenProblems 结果快照\n")
    if open_status.get("status") == "ok":
        lines.append(
            f"- train/test/prediction shape：{open_status['train_shape']} / {open_status['test_shape']} / {open_status['prediction_shape']}"
        )
        lines.append(f"- test tasks：{open_status['n_test_tasks']}；test cell types：{', '.join(open_status['test_cell_types'])}")
        top = open_summary.sort_values("spearman", ascending=False).head(8)
        lines.append("\nSpearman 最高的几项：\n")
        lines.append(top.to_string(index=False))
    else:
        lines.append(f"- 状态：{open_status.get('status')}")
        lines.append(f"- 缺失：{open_status.get('missing')}")

    lines.append("\n## 3. Tahoe raw 字段审计快照\n")
    if tahoe_status.get("status") == "ok":
        lines.append(
            f"- 当前可见完整 shard：{tahoe_status['complete_shards_seen']}；本次审计 shard：{tahoe_status['audited_shards']}；审计行数：{tahoe_status['audited_rows']}"
        )
        lines.append(
            f"- gene metadata rows：{tahoe_status['gene_metadata_rows']}；obs metadata rows：{tahoe_status['obs_metadata_rows']}"
        )
        lines.append("\n各字段聚合 top 值见 `tables/TAHOE_RAW_FIELD_COUNTS.csv`。")
    else:
        lines.append(f"- 状态：{tahoe_status.get('status')}")

    lines.append("\n## 4. 第一批实验队列\n")
    lines.append(queue[["priority", "experiment_id", "experiment_name", "current_status"]].to_string(index=False))

    report = "\n".join(lines)
    (REPORTS / "E41_FIRST_BATCH_SMOKE_REPORT.md").write_text(report, encoding="utf-8")

    readme = [
        "# E41 多维数据第一批 smoke",
        "",
        "先看 `reports/E41_FIRST_BATCH_SMOKE_REPORT.md`。",
        "",
        "核心输出：",
        "",
        "- `tables/OPENPROBLEMS_DGE_TASK_RISK.csv`：OpenProblems 每个测试任务的误差与风险代理。",
        "- `tables/OPENPROBLEMS_DGE_RISK_SUMMARY.csv`：风险分数与误差的 Spearman / top20 enrichment。",
        "- `tables/TAHOE_RAW_SHARD_METADATA_SUMMARY.csv`：Tahoe raw 已完成 shard 的字段覆盖。",
        "- `tables/TAHOE_RAW_FIELD_COUNTS.csv`：Tahoe raw 字段聚合计数。",
        "- `tables/FIRST_BATCH_EXPERIMENT_QUEUE.csv`：后续实验队列。",
    ]
    (OUT / "README_先看这个.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tahoe-shards", type=int, default=24)
    args = parser.parse_args()

    ensure_dirs()

    open_tasks, open_summary, open_status = run_openproblems_dge()
    tahoe_file, tahoe_counts, tahoe_schema, tahoe_status = run_tahoe_raw_metadata(args.max_tahoe_shards)
    queue = build_first_batch_queue(open_status, tahoe_status)

    if not open_tasks.empty:
        save_csv(open_tasks, TABLES / "OPENPROBLEMS_DGE_TASK_RISK.csv")
    if not open_summary.empty:
        save_csv(open_summary, TABLES / "OPENPROBLEMS_DGE_RISK_SUMMARY.csv")
    if not tahoe_file.empty:
        save_csv(tahoe_file, TABLES / "TAHOE_RAW_SHARD_METADATA_SUMMARY.csv")
    if not tahoe_counts.empty:
        save_csv(tahoe_counts, TABLES / "TAHOE_RAW_FIELD_COUNTS.csv")
    if not tahoe_schema.empty:
        save_csv(tahoe_schema, TABLES / "TAHOE_RAW_SCHEMA_BY_SHARD.csv")
    save_csv(queue, TABLES / "FIRST_BATCH_EXPERIMENT_QUEUE.csv")

    status = {
        "generated_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "openproblems_status": open_status,
        "tahoe_raw_status": tahoe_status,
        "output_dir": rel(OUT),
        "tables": [
            "tables/OPENPROBLEMS_DGE_TASK_RISK.csv",
            "tables/OPENPROBLEMS_DGE_RISK_SUMMARY.csv",
            "tables/TAHOE_RAW_SHARD_METADATA_SUMMARY.csv",
            "tables/TAHOE_RAW_FIELD_COUNTS.csv",
            "tables/TAHOE_RAW_SCHEMA_BY_SHARD.csv",
            "tables/FIRST_BATCH_EXPERIMENT_QUEUE.csv",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(open_summary, open_status, tahoe_file, tahoe_counts, tahoe_status, queue)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
