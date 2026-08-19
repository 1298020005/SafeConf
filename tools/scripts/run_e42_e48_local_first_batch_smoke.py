#!/usr/bin/env python3
"""E42-E48 local first-batch smoke experiments.

This is the executable companion to E41.  It avoids heavy model training and
uses existing lightweight reference predictors where a task matrix is well
defined.  For settings that are not a normal context x perturbation matrix
(Norman combos, Papalexi RNA/protein, Gasperini regulatory sparsity), it writes
direct audit tables instead of pretending they are the same experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
ATLAS = Path("/home/yyf/data/singlecell_perturbation_atlas")
CODE_DIR = ROOT / "code" / "20260426_154505_perturb_transport_final_push" / "03_code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from transport_models import ContextSimilarityBaseline, V0StrongBaseline  # noqa: E402


OUT = ROOT / "docs" / "实验结果" / "E42_E48_local_first_batch_smoke_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"


PATHS = {
    "sciplex3": ATLAS / "extra_official/cellular_context_generalization/sciplex3.h5ad",
    "TCDD": ATLAS / "extra_official/cellular_context_generalization/TCDD.h5ad",
    "KaggleCrossPatient": ATLAS / "extra_official/cellular_context_generalization/KaggleCrossPatient.h5ad",
    "crossSpecies": ATLAS / "extra_official/cellular_context_generalization/crossSpecies.h5ad",
    "Norman": ATLAS / "official_generalization/Norman.h5ad",
    "Gasperini_lowMOI": ATLAS / "official_scperturb/GasperiniShendure2019_lowMOI.h5ad",
    "Papalexi_RNA": ATLAS / "official_scperturb/PapalexiSatija2021_eccite_RNA.h5ad",
    "Papalexi_protein": ATLAS / "official_scperturb/PapalexiSatija2021_eccite_protein.h5ad",
}


CONTROL_VALUES = {"", "control", "ctrl", "vehicle", "dmso", "nt", "non-targeting", "nan", "none", "null", "CTRL"}


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


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        try:
            return path.relative_to(ATLAS).as_posix()
        except Exception:
            return str(path)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def is_control(x: object) -> bool:
    text = "" if pd.isna(x) else str(x).strip()
    low = text.lower()
    return text in CONTROL_VALUES or low in CONTROL_VALUES or low.startswith("control") or low.startswith("ctrl")


def as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def select_gene_idx(adata, n_genes: int) -> np.ndarray:
    hv_cols = [c for c in adata.var.columns if str(c).startswith("highly_variable")]
    for col in reversed(hv_cols):
        vals = np.asarray(adata.var[col]).astype(bool)
        idx = np.flatnonzero(vals)
        if len(idx) >= min(100, n_genes):
            return idx[:n_genes]
    return np.arange(min(n_genes, adata.n_vars))


def get_matrix(adata, gene_idx: np.ndarray) -> np.ndarray:
    layer = "logNor" if "logNor" in adata.layers else None
    x = adata[:, gene_idx].layers[layer] if layer else adata[:, gene_idx].X
    return as_dense(x).astype(np.float32)


def sample_positions(pos: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    if len(pos) > max_cells:
        return rng.choice(pos, size=max_cells, replace=False)
    return pos


def build_context_tasks(
    path: Path,
    dataset: str,
    context_col: str | None,
    perturbation_col: str,
    n_genes: int,
    min_cells: int,
    max_cells_per_group: int,
    seed: int,
    perturbation_label_col: str | None = None,
    control_col: str | None = None,
    task_filter=None,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(path)
    gene_idx = select_gene_idx(adata, n_genes)
    x = get_matrix(adata, gene_idx)
    obs = adata.obs.copy()
    obs["_context"] = "global" if context_col is None else obs[context_col].astype(str)
    obs["_pert"] = obs[perturbation_col].astype(str)
    if perturbation_label_col is not None:
        obs["_pert_label"] = obs[perturbation_label_col].astype(str)
    else:
        obs["_pert_label"] = obs["_pert"]
    if control_col is not None:
        obs["_control_label"] = obs[control_col].astype(str)
    else:
        obs["_control_label"] = obs["_pert"]
    if task_filter is not None:
        obs["_keep_task"] = task_filter(obs)
    else:
        obs["_keep_task"] = True

    group_pos: dict[tuple[str, str], np.ndarray] = {}
    group_control_like: dict[tuple[str, str], bool] = {}
    for (ctx, pert), sub in obs.groupby(["_context", "_pert"], observed=False):
        pos = obs.index.get_indexer(sub.index)
        if len(pos) < min_cells:
            continue
        group_pos[(str(ctx), str(pert))] = sample_positions(pos, max_cells_per_group, rng)
        control_labels = sub["_control_label"].astype(str)
        group_control_like[(str(ctx), str(pert))] = bool(is_control(pert) or control_labels.map(is_control).any())

    control_means: dict[str, np.ndarray] = {}
    for (ctx, pert), pos in group_pos.items():
        if group_control_like.get((ctx, pert), False):
            control_means[ctx] = x[pos].mean(axis=0)

    tasks = []
    for (ctx, pert), pos in group_pos.items():
        if group_control_like.get((ctx, pert), False):
            continue
        if ctx not in control_means:
            continue
        if not bool(obs.iloc[pos]["_keep_task"].all()):
            continue
        label_vals = obs.iloc[pos]["_pert_label"].astype(str)
        label = str(label_vals.mode().iloc[0]) if len(label_vals) else pert
        effect = x[pos].mean(axis=0) - control_means[ctx]
        tasks.append(
            {
                "dataset": dataset,
                "context": str(ctx),
                "perturbation": label,
                "raw_perturbation": str(pert),
                "effect": effect.astype(np.float32),
                "control_mean": control_means[ctx].astype(np.float32),
                "n_cells": int(len(pos)),
            }
        )

    meta = {
        "dataset": dataset,
        "path": rel(path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_genes_used": int(len(gene_idx)),
        "context_col": context_col or "global",
        "perturbation_col": perturbation_col,
        "n_tasks": int(len(tasks)),
        "n_contexts": int(len({t["context"] for t in tasks})),
        "n_perturbations": int(len({t["perturbation"] for t in tasks})),
    }
    return tasks, meta


def zscore(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce")
    sd = float(arr.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(arr)), index=s.index, dtype=float)
    return (arr - float(arr.mean())) / sd


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def vec_l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


def top_enrichment(df: pd.DataFrame, score_col: str, error_col: str, frac: float = 0.2) -> tuple[int, float, float]:
    sub = df[[score_col, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return 0, float("nan"), float("nan")
    k = max(1, int(math.ceil(len(sub) * frac)))
    top = sub.sort_values(score_col, ascending=False).head(k)
    all_mean = float(sub[error_col].mean())
    top_mean = float(top[error_col].mean())
    return k, top_mean, top_mean / all_mean if all_mean > 1e-12 else float("nan")


def leave_context_splits(tasks: list[dict], min_test: int = 2) -> list[dict]:
    contexts = sorted({str(t["context"]) for t in tasks})
    out = []
    for ctx in contexts:
        test = [i for i, t in enumerate(tasks) if str(t["context"]) == ctx]
        train = [i for i, t in enumerate(tasks) if str(t["context"]) != ctx]
        if len(train) >= min_test and len(test) >= min_test:
            out.append({"setting": "leave_context", "heldout": ctx, "train_ids": train, "test_ids": test})
    return out


def leave_perturbation_splits(tasks: list[dict], min_test: int = 2) -> list[dict]:
    perts = sorted({str(t["perturbation"]) for t in tasks})
    out = []
    for pert in perts:
        test = [i for i, t in enumerate(tasks) if str(t["perturbation"]) == pert]
        train = [i for i, t in enumerate(tasks) if str(t["perturbation"]) != pert]
        if len(train) >= min_test and len(test) >= min_test:
            out.append({"setting": "leave_perturbation", "heldout": pert, "train_ids": train, "test_ids": test})
    return out


def score_splits(dataset: str, tasks: list[dict], splits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    split_status = []
    for spec in splits:
        train_ids = sorted(set(map(int, spec["train_ids"])))
        test_ids = sorted(set(map(int, spec["test_ids"])))
        if len(train_ids) < 2 or len(test_ids) < 2:
            split_status.append({**spec, "dataset_name": dataset, "status": "skipped_too_small", "n_train": len(train_ids), "n_test": len(test_ids)})
            continue
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
            split_status.append({**spec, "dataset_name": dataset, "status": "failed", "error": repr(exc), "n_train": len(train_ids), "n_test": len(test_ids)})
            continue

        support = {}
        for tid in train_ids:
            p = str(tasks[tid]["perturbation"])
            support[p] = support.get(p, 0) + 1
        split_rows = []
        for pos, tid in enumerate(test_ids):
            task = tasks[tid]
            true = np.asarray(task["effect"], dtype=float)
            mean_pred = 0.5 * (pred_v0[pos] + pred_ctx[pos])
            split_rows.append(
                {
                    "dataset_name": dataset,
                    "setting": spec["setting"],
                    "heldout": spec["heldout"],
                    "task_id": int(tid),
                    "context": str(task["context"]),
                    "perturbation": str(task["perturbation"]),
                    "raw_perturbation": str(task.get("raw_perturbation", task["perturbation"])),
                    "n_cells": int(task.get("n_cells", 0)),
                    "support_count": int(support.get(str(task["perturbation"]), 0)),
                    "context_similarity": float(ctx_sim[pos]),
                    "predicted_l2": vec_l2(mean_pred),
                    "disagreement_rmse": rmse(pred_v0[pos], pred_ctx[pos]),
                    "error_v0_rmse": rmse(pred_v0[pos], true),
                    "error_contextsim_rmse": rmse(pred_ctx[pos], true),
                    "error_mean_rmse": rmse(mean_pred, true),
                }
            )
        df = pd.DataFrame(split_rows)
        df["risk_disagreement"] = zscore(df["disagreement_rmse"])
        df["risk_predicted_magnitude"] = zscore(df["predicted_l2"])
        df["risk_inverse_support"] = -zscore(np.log1p(df["support_count"]))
        df["risk_inverse_context_similarity"] = -zscore(df["context_similarity"])
        df["risk_safeconf_smoke"] = (
            df["risk_disagreement"]
            + df["risk_predicted_magnitude"]
            + df["risk_inverse_support"]
            + df["risk_inverse_context_similarity"]
        )
        rows.append(df)
        split_status.append({**{k: v for k, v in spec.items() if k not in {"train_ids", "test_ids"}}, "dataset_name": dataset, "status": "ok", "n_train": len(train_ids), "n_test": len(test_ids)})

    score_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    status_df = pd.DataFrame(split_status).drop(columns=["train_ids", "test_ids"], errors="ignore")
    return score_df, status_df


def summarize_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if score_df.empty:
        return pd.DataFrame()
    risks = [
        "risk_safeconf_smoke",
        "risk_disagreement",
        "risk_predicted_magnitude",
        "risk_inverse_support",
        "risk_inverse_context_similarity",
    ]
    targets = ["error_mean_rmse", "error_v0_rmse", "error_contextsim_rmse"]
    for (dataset, setting), g in score_df.groupby(["dataset_name", "setting"], observed=False):
        for risk in risks:
            for target in targets:
                k, top_mean, enrich = top_enrichment(g, risk, target)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "setting": setting,
                        "risk_score_name": risk,
                        "target_error": target,
                        "n_tasks": int(g[[risk, target]].dropna().shape[0]),
                        "spearman": spearman(g[risk], g[target]),
                        "top20_k": k,
                        "top20_mean_error": top_mean,
                        "top20_enrichment": enrich,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset_name", "setting", "spearman"], ascending=[True, True, False], kind="stable")


def run_matrix_experiments(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_specs = [
        ("sciplex3_cell_line", PATHS["sciplex3"], "cell_line", "perturbation", None, "leave_context"),
        ("TCDD_celltype", PATHS["TCDD"], "celltype", "perturbation", None, "leave_context"),
        ("TCDD_dose", PATHS["TCDD"], "celltype", "dose", None, "leave_perturbation", "perturbation"),
        ("KaggleCrossPatient_donor", PATHS["KaggleCrossPatient"], "donor_id", "perturbation", None, "leave_context"),
        ("crossSpecies_species", PATHS["crossSpecies"], "condition1", "perturbation", None, "leave_context"),
    ]
    all_scores = []
    all_status = []
    task_rows = []
    for spec in dataset_specs:
        if len(spec) == 6:
            dataset, path, context_col, pert_col, label_col, split_type = spec
            control_col = None
        else:
            dataset, path, context_col, pert_col, label_col, split_type, control_col = spec
        try:
            tasks, meta = build_context_tasks(
                path=path,
                dataset=dataset,
                context_col=context_col,
                perturbation_col=pert_col,
                perturbation_label_col=label_col,
                control_col=control_col,
                n_genes=args.n_genes,
                min_cells=args.min_cells,
                max_cells_per_group=args.max_cells_per_group,
                seed=args.seed,
            )
            task_rows.append({**meta, "status": "ok" if tasks else "no_tasks", "error": "" if tasks else "no eligible tasks"})
            if split_type == "leave_context":
                splits = leave_context_splits(tasks)
            else:
                splits = leave_perturbation_splits(tasks)
            scores, status = score_splits(dataset, tasks, splits)
            if not scores.empty:
                all_scores.append(scores)
            if not status.empty:
                all_status.append(status)
        except Exception as exc:
            task_rows.append(
                {
                    "dataset": dataset,
                    "path": rel(path),
                    "status": "failed",
                    "error": repr(exc),
                    "n_tasks": 0,
                    "n_contexts": 0,
                    "n_perturbations": 0,
                }
            )
    score_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    status_df = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    task_df = pd.DataFrame(task_rows)
    return score_df, status_df, task_df


def parse_genes(label: str) -> list[str]:
    text = str(label).replace(",", "+")
    if text.lower() in {"control", "ctrl"}:
        return []
    return [x.strip() for x in text.split("+") if x.strip()]


def run_norman_single_to_combo(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks, meta = build_context_tasks(
        path=PATHS["Norman"],
        dataset="Norman_single_to_combo",
        context_col=None,
        perturbation_col="perturbation",
        n_genes=args.n_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
    )
    by_pert = {str(t["perturbation"]): t for t in tasks}
    single_effects = {p: t["effect"] for p, t in by_pert.items() if len(parse_genes(p)) == 1}
    global_mean = np.mean([t["effect"] for t in tasks], axis=0) if tasks else np.zeros(args.n_genes, dtype=np.float32)
    rows = []
    for pert, task in by_pert.items():
        genes = parse_genes(pert)
        if len(genes) < 2:
            continue
        available = [single_effects[g] for g in genes if g in single_effects]
        if available:
            pred = np.mean(available, axis=0)
        else:
            pred = global_mean
        missing = [g for g in genes if g not in single_effects]
        rows.append(
            {
                "perturbation": pert,
                "genes": "+".join(genes),
                "combo_size": len(genes),
                "available_single_count": len(available),
                "missing_single_count": len(missing),
                "missing_genes": "+".join(missing),
                "n_cells": int(task.get("n_cells", 0)),
                "predicted_l2": vec_l2(pred),
                "true_l2_diagnostic": vec_l2(task["effect"]),
                "error_single_additive_rmse": rmse(pred, task["effect"]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["risk_missing_single"] = zscore(df["missing_single_count"])
        df["risk_combo_size"] = zscore(df["combo_size"])
        df["risk_predicted_magnitude"] = zscore(df["predicted_l2"])
        df["risk_norman_combo_smoke"] = df["risk_missing_single"] + df["risk_combo_size"] + df["risk_predicted_magnitude"]
    summary = []
    for risk in ["risk_norman_combo_smoke", "risk_missing_single", "risk_combo_size", "risk_predicted_magnitude"]:
        if risk in df:
            k, top_mean, enrich = top_enrichment(df, risk, "error_single_additive_rmse")
            summary.append(
                {
                    "dataset_name": "Norman_single_to_combo",
                    "risk_score_name": risk,
                    "target_error": "error_single_additive_rmse",
                    "n_tasks": int(df[[risk, "error_single_additive_rmse"]].dropna().shape[0]),
                    "spearman": spearman(df[risk], df["error_single_additive_rmse"]),
                    "top20_k": k,
                    "top20_mean_error": top_mean,
                    "top20_enrichment": enrich,
                }
            )
    return df, pd.DataFrame(summary)


def run_gasperini_audit() -> pd.DataFrame:
    path = PATHS["Gasperini_lowMOI"]
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    rows = []
    for col in ["perturbation", "gene", "nperts", "sample", "sample_directory"]:
        if col not in obs:
            continue
        vc = obs[col].astype(str).value_counts(dropna=False)
        rows.append(
            {
                "field": col,
                "n_unique": int(vc.shape[0]),
                "top10": " | ".join(f"{k}:{int(v)}" for k, v in vc.head(10).items()),
            }
        )
    pert_counts = obs["perturbation"].astype(str).value_counts(dropna=False)
    target_df = (
        pert_counts.rename_axis("perturbation")
        .reset_index(name="cell_count")
        .assign(
            is_control_like=lambda d: d["perturbation"].str.lower().str.contains("control|nan|scrambled", regex=True),
            is_coordinate_like=lambda d: d["perturbation"].str.contains("chr", regex=False),
            low_support_lt15=lambda d: d["cell_count"] < 15,
        )
    )
    rows.append(
        {
            "field": "perturbation_support_profile",
            "n_unique": int(len(target_df)),
            "top10": f"low_support_lt15={int(target_df['low_support_lt15'].sum())}; coordinate_like={int(target_df['is_coordinate_like'].sum())}; control_like={int(target_df['is_control_like'].sum())}",
        }
    )
    a.file.close()
    return pd.DataFrame(rows)


def run_papalexi_consistency(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rna_tasks, rna_meta = build_context_tasks(
        path=PATHS["Papalexi_RNA"],
        dataset="Papalexi_RNA",
        context_col=None,
        perturbation_col="perturbation",
        n_genes=args.n_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
    )
    protein_tasks, protein_meta = build_context_tasks(
        path=PATHS["Papalexi_protein"],
        dataset="Papalexi_protein",
        context_col=None,
        perturbation_col="perturbation",
        n_genes=4,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
    )
    rna = {str(t["perturbation"]): t for t in rna_tasks}
    prot = {str(t["perturbation"]): t for t in protein_tasks}
    rows = []
    for pert in sorted(set(rna) & set(prot)):
        rv = rna[pert]["effect"]
        pv = prot[pert]["effect"]
        rows.append(
            {
                "perturbation": pert,
                "rna_n_cells": int(rna[pert].get("n_cells", 0)),
                "protein_n_cells": int(prot[pert].get("n_cells", 0)),
                "rna_l2": vec_l2(rv),
                "protein_l2": vec_l2(pv),
                "protein_abs_mean": float(np.mean(np.abs(pv))),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["risk_rna_magnitude"] = zscore(df["rna_l2"])
        df["risk_low_cell_support"] = -zscore(np.log1p(np.minimum(df["rna_n_cells"], df["protein_n_cells"])))
    summary = pd.DataFrame(
        [
            {
                "dataset_name": "Papalexi_RNA_protein",
                "n_aligned_perturbations": int(len(df)),
                "spearman_rna_l2_vs_protein_l2": spearman(df["rna_l2"], df["protein_l2"]) if not df.empty else float("nan"),
                "spearman_rna_l2_vs_protein_abs_mean": spearman(df["rna_l2"], df["protein_abs_mean"]) if not df.empty else float("nan"),
                "rna_tasks": int(rna_meta.get("n_tasks", 0)),
                "protein_tasks": int(protein_meta.get("n_tasks", 0)),
            }
        ]
    )
    return df, summary


def write_report(score_summary: pd.DataFrame, task_status: pd.DataFrame, norman_summary: pd.DataFrame, pap_summary: pd.DataFrame, gasperini: pd.DataFrame) -> None:
    top = score_summary.sort_values("spearman", ascending=False).head(12) if not score_summary.empty else score_summary
    norman_top = norman_summary.sort_values("spearman", ascending=False).head(6) if not norman_summary.empty else norman_summary
    lines = []
    lines.append("# E42-E48 本地第一批 smoke\n")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append(f"- Git：`{git_head()[:12]}`")
    lines.append(f"- 工作区 dirty：`{git_dirty()}`\n")
    lines.append("## 1. 普通任务矩阵结果\n")
    lines.append("覆盖 sciplex3、TCDD、KaggleCrossPatient、crossSpecies。每个任务先用轻量参考预测器出 error，再看风险分数排序。")
    lines.append("\n任务构建：\n")
    lines.append(task_status.to_string(index=False))
    lines.append("\nSpearman 较高的项目：\n")
    lines.append(top.to_string(index=False) if not top.empty else "暂无结果")
    lines.append("\n## 2. Norman 单基因到组合\n")
    lines.append(norman_top.to_string(index=False) if not norman_top.empty else "暂无结果")
    lines.append("\n## 3. Papalexi RNA-protein 一致性\n")
    lines.append(pap_summary.to_string(index=False) if not pap_summary.empty else "暂无结果")
    lines.append("\n## 4. Gasperini 调控标签稀疏审计\n")
    lines.append(gasperini.to_string(index=False) if not gasperini.empty else "暂无结果")
    (REPORTS / "E42_E48_LOCAL_FIRST_BATCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E42-E48 本地第一批 smoke\n\n"
        "先看 `reports/E42_E48_LOCAL_FIRST_BATCH_REPORT.md`。\n\n"
        "这一步把 E41 队列里的本地数据线实际跑成了结果表或字段审计表。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=256)
    parser.add_argument("--min-cells", type=int, default=15)
    parser.add_argument("--max-cells-per-group", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    score_df, split_status, task_status = run_matrix_experiments(args)
    score_summary = summarize_scores(score_df)
    norman_df, norman_summary = run_norman_single_to_combo(args)
    gasperini = run_gasperini_audit()
    pap_df, pap_summary = run_papalexi_consistency(args)

    save_csv(task_status, TABLES / "LOCAL_TASK_BUILD_STATUS.csv")
    save_csv(split_status, TABLES / "LOCAL_SPLIT_STATUS.csv")
    save_csv(score_df, TABLES / "LOCAL_SCORE_TABLE.csv")
    save_csv(score_summary, TABLES / "LOCAL_SCORE_SUMMARY.csv")
    save_csv(norman_df, TABLES / "NORMAN_SINGLE_TO_COMBO.csv")
    save_csv(norman_summary, TABLES / "NORMAN_SINGLE_TO_COMBO_SUMMARY.csv")
    save_csv(gasperini, TABLES / "GASPERINI_REGULATORY_SPARSITY_AUDIT.csv")
    save_csv(pap_df, TABLES / "PAPALEXI_RNA_PROTEIN_CONSISTENCY.csv")
    save_csv(pap_summary, TABLES / "PAPALEXI_RNA_PROTEIN_SUMMARY.csv")

    status = {
        "generated_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "n_score_rows": int(len(score_df)),
        "n_score_summary_rows": int(len(score_summary)),
        "n_norman_combo_rows": int(len(norman_df)),
        "n_papalexi_aligned_rows": int(len(pap_df)),
        "output_dir": rel(OUT),
        "tables": [
            "tables/LOCAL_TASK_BUILD_STATUS.csv",
            "tables/LOCAL_SPLIT_STATUS.csv",
            "tables/LOCAL_SCORE_TABLE.csv",
            "tables/LOCAL_SCORE_SUMMARY.csv",
            "tables/NORMAN_SINGLE_TO_COMBO.csv",
            "tables/NORMAN_SINGLE_TO_COMBO_SUMMARY.csv",
            "tables/GASPERINI_REGULATORY_SPARSITY_AUDIT.csv",
            "tables/PAPALEXI_RNA_PROTEIN_CONSISTENCY.csv",
            "tables/PAPALEXI_RNA_PROTEIN_SUMMARY.csv",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(score_summary, task_status, norman_summary, pap_summary, gasperini)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
