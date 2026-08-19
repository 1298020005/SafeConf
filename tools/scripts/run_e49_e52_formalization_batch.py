#!/usr/bin/env python3
"""E49-E52 formalization batch for advisor-requested multidimensional data.

This batch follows the first smoke pass and starts turning the strongest data
lines into more formal, reproducible experiments:

E49 OpenProblems DGE
    Official Kaggle split + internal compound/cell-type holdout on DGE effects.

E50 sciplex3
    Larger gene panel cell-line holdout with lightweight reference predictors.

E51 Norman
    Single-gene-to-combination prediction with explicit additive/mean baselines.

E52 TCDD
    Dose-aware holdout using log-dose nearest-neighbour and linear trend
    predictors, instead of treating dose as a plain category.
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
TOOLS = ROOT / "tools" / "scripts"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_e42_e48_local_first_batch_smoke import (  # noqa: E402
    PATHS as LOCAL_PATHS,
    build_context_tasks,
    leave_context_splits,
    rmse,
    save_csv,
    score_splits,
    select_gene_idx,
    summarize_scores,
    top_enrichment,
    vec_l2,
    zscore,
)


OUT = ROOT / "docs" / "实验结果" / "E49_E52_formalization_batch_20260710"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

OPENPROBLEMS_KAGGLE = (
    ATLAS
    / "mega_external"
    / "OpenProblems_NeurIPS2023_single_cell_perturbations"
    / "data"
    / "workflow_resources"
    / "neurips-2023-kaggle"
)


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


def as_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def layer_array(adata, layer: str) -> np.ndarray:
    if layer not in adata.layers:
        raise KeyError(layer)
    return as_dense(adata.layers[layer]).astype(np.float32)


def smiles_ngrams(x: object, n: int = 3) -> set[str]:
    text = "" if pd.isna(x) else str(x).strip().replace(" ", "")
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


def openproblems_tasks() -> pd.DataFrame:
    train = ad.read_h5ad(OPENPROBLEMS_KAGGLE / "de_train.h5ad")
    test = ad.read_h5ad(OPENPROBLEMS_KAGGLE / "de_test.h5ad")
    train_x = layer_array(train, "logFC")
    test_x = layer_array(test, "logFC")
    rows = []
    effects = []
    for source, obs, x in [("train", train.obs.copy(), train_x), ("test", test.obs.copy(), test_x)]:
        obs = obs.reset_index(drop=True)
        for i, row in obs.iterrows():
            if bool(row.get("control", False)):
                continue
            rows.append(
                {
                    "source": source,
                    "row_index": int(i),
                    "cell_type": str(row.get("cell_type", "")),
                    "sm_name": str(row.get("sm_name", "")),
                    "sm_lincs_id": str(row.get("sm_lincs_id", "")),
                    "SMILES": str(row.get("SMILES", "")),
                    "dose_uM": float(row.get("dose_uM", np.nan)),
                    "split": str(row.get("split", source)),
                }
            )
            effects.append(np.asarray(x[i], dtype=np.float32))
    df = pd.DataFrame(rows)
    df["_effect_index"] = np.arange(len(df), dtype=int)
    return df, np.stack(effects, axis=0)


def op_predict(train_df: pd.DataFrame, train_y: np.ndarray, test_df: pd.DataFrame) -> dict[str, np.ndarray | np.ndarray]:
    global_mean = train_y.mean(axis=0)
    by_drug = {k: train_y[g.index.to_numpy()].mean(axis=0) for k, g in train_df.groupby("sm_name", observed=False)}
    by_cell = {k: train_y[g.index.to_numpy()].mean(axis=0) for k, g in train_df.groupby("cell_type", observed=False)}
    by_pair = {
        k: train_y[g.index.to_numpy()].mean(axis=0)
        for k, g in train_df.groupby(["sm_name", "cell_type"], observed=False)
    }
    drug_pred = []
    cell_pred = []
    additive_pred = []
    pair_pred = []
    for _, row in test_df.iterrows():
        drug = str(row["sm_name"])
        cell = str(row["cell_type"])
        d = by_drug.get(drug, global_mean)
        c = by_cell.get(cell, global_mean)
        p = by_pair.get((drug, cell), None)
        add = d + c - global_mean
        drug_pred.append(d)
        cell_pred.append(c)
        additive_pred.append(add)
        pair_pred.append(p if p is not None else add)
    return {
        "drug_mean": np.stack(drug_pred).astype(np.float32),
        "cell_mean": np.stack(cell_pred).astype(np.float32),
        "additive": np.stack(additive_pred).astype(np.float32),
        "pair_or_additive": np.stack(pair_pred).astype(np.float32),
    }


def op_score_split(name: str, df: pd.DataFrame, effects: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    train_y = effects[train_df["_effect_index"].to_numpy()]
    test_y = effects[test_df["_effect_index"].to_numpy()]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    preds = op_predict(train_df, train_y, test_df)
    train_smiles = train_df[["sm_name", "SMILES"]].drop_duplicates().copy()
    train_smiles["_ngram"] = train_smiles["SMILES"].astype(str).apply(smiles_ngrams)
    train_smiles_sets = list(train_smiles["_ngram"])
    drug_support = train_df["sm_name"].value_counts().to_dict()
    cell_support = train_df["cell_type"].value_counts().to_dict()
    pair_support = train_df.groupby(["sm_name", "cell_type"], observed=False).size().to_dict()

    rows = []
    for pos, (_, row) in enumerate(test_df.iterrows()):
        true = test_y[pos]
        p_drug = preds["drug_mean"][pos]
        p_cell = preds["cell_mean"][pos]
        p_add = preds["additive"][pos]
        p_pair = preds["pair_or_additive"][pos]
        sset = smiles_ngrams(row["SMILES"])
        rows.append(
            {
                "experiment": name,
                "cell_type": row["cell_type"],
                "sm_name": row["sm_name"],
                "split": row["split"],
                "drug_support": int(drug_support.get(row["sm_name"], 0)),
                "cell_type_support": int(cell_support.get(row["cell_type"], 0)),
                "pair_support": int(pair_support.get((row["sm_name"], row["cell_type"]), 0)),
                "nearest_train_smiles_jaccard": max((jaccard(sset, t) for t in train_smiles_sets), default=0.0),
                "predicted_l2_additive": vec_l2(p_add),
                "predicted_l2_pair_or_additive": vec_l2(p_pair),
                "disagreement_drug_cell_rmse": rmse(p_drug, p_cell),
                "error_drug_mean_rmse": rmse(p_drug, true),
                "error_cell_mean_rmse": rmse(p_cell, true),
                "error_additive_rmse": rmse(p_add, true),
                "error_pair_or_additive_rmse": rmse(p_pair, true),
                "true_l2_diagnostic": vec_l2(true),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame()
    out["risk_low_drug_support"] = -zscore(np.log1p(out["drug_support"]))
    out["risk_low_cell_type_support"] = -zscore(np.log1p(out["cell_type_support"]))
    out["risk_low_pair_support"] = -zscore(np.log1p(out["pair_support"]))
    out["risk_low_smiles_similarity"] = -zscore(out["nearest_train_smiles_jaccard"])
    out["risk_predicted_magnitude"] = zscore(out["predicted_l2_additive"])
    out["risk_disagreement"] = zscore(out["disagreement_drug_cell_rmse"])
    out["risk_op_formal"] = (
        out["risk_low_drug_support"]
        + out["risk_low_cell_type_support"]
        + out["risk_low_pair_support"]
        + out["risk_low_smiles_similarity"]
        + out["risk_predicted_magnitude"]
        + out["risk_disagreement"]
    )
    out["risk_oracle_true_magnitude_diagnostic"] = zscore(out["true_l2_diagnostic"])
    risks = [
        "risk_op_formal",
        "risk_predicted_magnitude",
        "risk_disagreement",
        "risk_low_drug_support",
        "risk_low_pair_support",
        "risk_low_smiles_similarity",
        "risk_oracle_true_magnitude_diagnostic",
    ]
    targets = ["error_additive_rmse", "error_pair_or_additive_rmse", "error_drug_mean_rmse", "error_cell_mean_rmse"]
    summary = []
    for risk in risks:
        for target in targets:
            k, top_mean, enrich = top_enrichment(out, risk, target)
            summary.append(
                {
                    "experiment": name,
                    "risk_score_name": risk,
                    "target_error": target,
                    "n_tasks": int(out[[risk, target]].dropna().shape[0]),
                    "spearman": spearman(out[risk], out[target]),
                    "top20_k": k,
                    "top20_mean_error": top_mean,
                    "top20_enrichment": enrich,
                }
            )
    return out, pd.DataFrame(summary)


def run_openproblems_formal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, effects = openproblems_tasks()
    frames = []
    summaries = []
    split_rows = []
    official_train = df["source"].eq("train").to_numpy()
    official_test = df["source"].eq("test").to_numpy()
    for name, train_mask, test_mask in [("official_train_to_test", official_train, official_test)]:
        scores, summary = op_score_split(name, df, effects, train_mask, test_mask)
        frames.append(scores)
        summaries.append(summary)
        split_rows.append({"experiment": name, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})

    train_only = df["source"].eq("train")
    train_df = df[train_only].copy()
    train_idx = train_df.index.to_numpy()
    # Internal cell-type holdout on training DGE.
    for cell in sorted(train_df["cell_type"].unique()):
        test_idx = train_df[train_df["cell_type"].eq(cell)].index.to_numpy()
        train_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_mask[np.setdiff1d(train_idx, test_idx)] = True
        test_mask[test_idx] = True
        if train_mask.sum() >= 10 and test_mask.sum() >= 5:
            name = f"internal_cell_type_holdout::{cell}"
            scores, summary = op_score_split(name, df, effects, train_mask, test_mask)
            frames.append(scores)
            summaries.append(summary)
            split_rows.append({"experiment": name, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})

    # Internal compound holdout: keep drugs with at least 4 train tasks to avoid tiny rows.
    for drug, n in train_df["sm_name"].value_counts().items():
        if int(n) < 4:
            continue
        test_idx = train_df[train_df["sm_name"].eq(drug)].index.to_numpy()
        train_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_mask[np.setdiff1d(train_idx, test_idx)] = True
        test_mask[test_idx] = True
        name = f"internal_compound_holdout::{drug}"
        scores, summary = op_score_split(name, df, effects, train_mask, test_mask)
        frames.append(scores)
        summaries.append(summary)
        split_rows.append({"experiment": name, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})

    score_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not score_df.empty:
        score_df["experiment_group"] = score_df["experiment"].map(openproblems_group)
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    return score_df, summary_df, split_df


def openproblems_group(experiment: object) -> str:
    text = str(experiment)
    if text.startswith("official"):
        return "official_train_to_test"
    if text.startswith("internal_cell_type"):
        return "internal_cell_type_holdout_pooled"
    if text.startswith("internal_compound"):
        return "internal_compound_holdout_pooled"
    return "other"


def summarize_openproblems_groups(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    risks = [
        "risk_op_formal",
        "risk_predicted_magnitude",
        "risk_disagreement",
        "risk_low_drug_support",
        "risk_low_pair_support",
        "risk_low_smiles_similarity",
        "risk_oracle_true_magnitude_diagnostic",
    ]
    targets = ["error_additive_rmse", "error_pair_or_additive_rmse", "error_drug_mean_rmse", "error_cell_mean_rmse"]
    rows = []
    for group, g in score_df.groupby("experiment_group", observed=False):
        for risk in risks:
            for target in targets:
                k, top_mean, enrich = top_enrichment(g, risk, target)
                rows.append(
                    {
                        "experiment_group": group,
                        "risk_score_name": risk,
                        "target_error": target,
                        "n_tasks": int(g[[risk, target]].dropna().shape[0]),
                        "spearman": spearman(g[risk], g[target]),
                        "top20_k": k,
                        "top20_mean_error": top_mean,
                        "top20_enrichment": enrich,
                    }
                )
    return pd.DataFrame(rows)


def run_sciplex3_formal(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks, meta = build_context_tasks(
        path=LOCAL_PATHS["sciplex3"],
        dataset="sciplex3_cell_line_formal",
        context_col="cell_line",
        perturbation_col="perturbation",
        n_genes=args.sciplex3_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
    )
    splits = leave_context_splits(tasks)
    scores, status = score_splits("sciplex3_cell_line_formal", tasks, splits)
    summary = summarize_scores(scores)
    meta_df = pd.DataFrame([{**meta, "status": "ok" if tasks else "no_tasks"}])
    return scores, summary, pd.concat([meta_df, status], ignore_index=True, sort=False)


def parse_genes(label: str) -> list[str]:
    text = str(label).replace(",", "+")
    if text.lower() in {"control", "ctrl"}:
        return []
    return [x.strip() for x in text.split("+") if x.strip()]


def run_norman_formal(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks, meta = build_context_tasks(
        path=LOCAL_PATHS["Norman"],
        dataset="Norman_combo_formal",
        context_col=None,
        perturbation_col="perturbation",
        n_genes=args.norman_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
    )
    by_pert = {str(t["perturbation"]): t for t in tasks}
    single = {p: t["effect"] for p, t in by_pert.items() if len(parse_genes(p)) == 1}
    global_mean = np.mean([t["effect"] for t in tasks], axis=0)
    rows = []
    for pert, task in by_pert.items():
        genes = parse_genes(pert)
        if len(genes) < 2:
            continue
        avail = [single[g] for g in genes if g in single]
        mean_pred = np.mean(avail, axis=0) if avail else global_mean
        sum_pred = np.sum(avail, axis=0) if avail else global_mean
        rows.append(
            {
                "perturbation": pert,
                "combo_size": len(genes),
                "genes": "+".join(genes),
                "available_single_count": len(avail),
                "missing_single_count": len([g for g in genes if g not in single]),
                "n_cells": int(task.get("n_cells", 0)),
                "mean_single_pred_l2": vec_l2(mean_pred),
                "sum_single_pred_l2": vec_l2(sum_pred),
                "true_l2_diagnostic": vec_l2(task["effect"]),
                "error_mean_single_rmse": rmse(mean_pred, task["effect"]),
                "error_sum_single_rmse": rmse(sum_pred, task["effect"]),
                "mean_vs_sum_disagreement_rmse": rmse(mean_pred, sum_pred),
            }
        )
    df = pd.DataFrame(rows)
    df["risk_mean_single_magnitude"] = zscore(df["mean_single_pred_l2"])
    df["risk_sum_single_magnitude"] = zscore(df["sum_single_pred_l2"])
    df["risk_additive_disagreement"] = zscore(df["mean_vs_sum_disagreement_rmse"])
    df["risk_missing_single"] = zscore(df["missing_single_count"])
    df["risk_norman_formal"] = (
        df["risk_mean_single_magnitude"]
        + df["risk_additive_disagreement"]
        + df["risk_missing_single"]
    )
    summary = []
    for risk in ["risk_norman_formal", "risk_mean_single_magnitude", "risk_sum_single_magnitude", "risk_additive_disagreement", "risk_missing_single"]:
        for target in ["error_mean_single_rmse", "error_sum_single_rmse"]:
            k, top_mean, enrich = top_enrichment(df, risk, target)
            summary.append(
                {
                    "experiment": "Norman_combo_formal",
                    "risk_score_name": risk,
                    "target_error": target,
                    "n_tasks": int(df[[risk, target]].dropna().shape[0]),
                    "spearman": spearman(df[risk], df[target]),
                    "top20_k": k,
                    "top20_mean_error": top_mean,
                    "top20_enrichment": enrich,
                }
            )
    return df, pd.DataFrame(summary)


def tcdd_effect_tasks(args) -> tuple[list[dict], pd.DataFrame]:
    path = LOCAL_PATHS["TCDD"]
    adata = ad.read_h5ad(path)
    gene_idx = select_gene_idx(adata, args.tcdd_genes)
    x = adata[:, gene_idx].layers["logNor"] if "logNor" in adata.layers else adata[:, gene_idx].X
    x = as_dense(x).astype(np.float32)
    obs = adata.obs.copy()
    obs["_celltype"] = obs["celltype"].astype(str)
    obs["_dose"] = obs["Dose"].astype(float)
    obs["_pert"] = obs["perturbation"].astype(str)
    rng = np.random.default_rng(args.seed)
    control_means = {}
    tasks = []
    for cell, sub in obs.groupby("_celltype", observed=False):
        ctrl_idx = sub[sub["_pert"].str.lower().eq("control")].index
        pos = obs.index.get_indexer(ctrl_idx)
        if len(pos) < args.min_cells:
            continue
        if len(pos) > args.max_cells_per_group:
            pos = rng.choice(pos, size=args.max_cells_per_group, replace=False)
        control_means[cell] = x[pos].mean(axis=0)
    for (cell, dose), sub in obs[obs["_pert"].str.lower().eq("tcdd")].groupby(["_celltype", "_dose"], observed=False):
        if float(dose) <= 0 or cell not in control_means:
            continue
        pos = obs.index.get_indexer(sub.index)
        if len(pos) < args.min_cells:
            continue
        if len(pos) > args.max_cells_per_group:
            pos = rng.choice(pos, size=args.max_cells_per_group, replace=False)
        effect = x[pos].mean(axis=0) - control_means[cell]
        tasks.append(
            {
                "context": str(cell),
                "dose": float(dose),
                "log_dose": float(np.log10(float(dose))),
                "effect": effect.astype(np.float32),
                "n_cells": int(len(pos)),
            }
        )
    return tasks, pd.DataFrame([{ "n_tasks": len(tasks), "n_celltypes": len({t["context"] for t in tasks}), "n_doses": len({t["dose"] for t in tasks}), "n_genes": len(gene_idx)}])


def tcdd_predict_split(tasks: list[dict], heldout_dose: float) -> pd.DataFrame:
    train = [t for t in tasks if float(t["dose"]) != float(heldout_dose)]
    test = [t for t in tasks if float(t["dose"]) == float(heldout_dose)]
    global_mean = np.mean([t["effect"] for t in train], axis=0)
    rows = []
    for task in test:
        same = [t for t in train if t["context"] == task["context"]]
        if not same:
            nearest = global_mean
            linear = global_mean
            nearest_dist = float("nan")
        else:
            nearest_task = min(same, key=lambda t: abs(float(t["log_dose"]) - float(task["log_dose"])))
            nearest = nearest_task["effect"]
            nearest_dist = abs(float(nearest_task["log_dose"]) - float(task["log_dose"]))
            xs = np.asarray([t["log_dose"] for t in same], dtype=float)
            ys = np.stack([t["effect"] for t in same], axis=0).astype(float)
            if len(np.unique(xs)) >= 2:
                x_mean = xs.mean()
                denom = float(np.sum((xs - x_mean) ** 2))
                slope = ((xs - x_mean)[:, None] * (ys - ys.mean(axis=0))).sum(axis=0) / max(denom, 1e-8)
                intercept = ys.mean(axis=0) - slope * x_mean
                linear = intercept + slope * float(task["log_dose"])
            else:
                linear = nearest
        pred_mean = 0.5 * (nearest + linear)
        true = task["effect"]
        rows.append(
            {
                "heldout_dose": float(heldout_dose),
                "celltype": task["context"],
                "log_dose": task["log_dose"],
                "n_train": len(train),
                "n_test_same_dose": len(test),
                "nearest_logdose_distance": nearest_dist,
                "predicted_l2_mean": vec_l2(pred_mean),
                "dose_predictor_disagreement": rmse(nearest, linear),
                "error_nearest_rmse": rmse(nearest, true),
                "error_linear_rmse": rmse(linear, true),
                "error_mean_rmse": rmse(pred_mean, true),
                "true_l2_diagnostic": vec_l2(true),
            }
        )
    return pd.DataFrame(rows)


def run_tcdd_dose_aware(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks, meta = tcdd_effect_tasks(args)
    frames = []
    for dose in sorted({t["dose"] for t in tasks}):
        frames.append(tcdd_predict_split(tasks, dose))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df["risk_logdose_extreme"] = zscore(np.abs(df["log_dose"]))
    df["risk_nearest_dose_distance"] = zscore(df["nearest_logdose_distance"])
    df["risk_predicted_magnitude"] = zscore(df["predicted_l2_mean"])
    df["risk_dose_disagreement"] = zscore(df["dose_predictor_disagreement"])
    df["risk_tcdd_dose_aware"] = (
        df["risk_logdose_extreme"]
        + df["risk_nearest_dose_distance"]
        + df["risk_predicted_magnitude"]
        + df["risk_dose_disagreement"]
    )
    summary = []
    for risk in ["risk_tcdd_dose_aware", "risk_dose_disagreement", "risk_predicted_magnitude", "risk_nearest_dose_distance", "risk_logdose_extreme"]:
        for target in ["error_mean_rmse", "error_nearest_rmse", "error_linear_rmse"]:
            k, top_mean, enrich = top_enrichment(df, risk, target)
            summary.append(
                {
                    "experiment": "TCDD_dose_aware",
                    "risk_score_name": risk,
                    "target_error": target,
                    "n_tasks": int(df[[risk, target]].dropna().shape[0]),
                    "spearman": spearman(df[risk], df[target]),
                    "top20_k": k,
                    "top20_mean_error": top_mean,
                    "top20_enrichment": enrich,
                }
            )
    return df, pd.DataFrame(summary), meta


def write_report(all_summary: dict[str, pd.DataFrame], statuses: dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("# E49-E52 第二阶段正式化实验\n")
    lines.append(f"- 生成时间：{now_text()}")
    lines.append(f"- Git：`{git_head()[:12]}`")
    lines.append(f"- 工作区 dirty：`{git_dirty()}`\n")
    lines.append("## 1. 已跑实验\n")
    lines.append("- E49 OpenProblems DGE：官方 train→test、训练集内部 cell-type holdout、compound holdout。")
    lines.append("- E50 sciplex3：1000 基因 cell-line holdout。")
    lines.append("- E51 Norman：单基因到组合扰动，mean/sum 两种单基因组合预测。")
    lines.append("- E52 TCDD：dose-aware 留出，用最近剂量和 log-dose 线性趋势两个预测器。\n")

    for name, df in all_summary.items():
        lines.append(f"## {name}\n")
        if df.empty:
            lines.append("暂无结果。\n")
            continue
        show = df.copy()
        if "n_tasks" in show.columns:
            show = show[pd.to_numeric(show["n_tasks"], errors="coerce").fillna(0) >= 15]
        top = show.sort_values("spearman", ascending=False).head(12)
        lines.append(top.to_string(index=False))
        lines.append("")
    lines.append("## 状态表\n")
    for name, df in statuses.items():
        lines.append(f"\n### {name}\n")
        lines.append(df.head(30).to_string(index=False) if not df.empty else "空")
    (REPORTS / "E49_E52_FORMALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E49-E52 第二阶段正式化实验\n\n"
        "先看 `reports/E49_E52_FORMALIZATION_REPORT.md`。\n\n"
        "这一步把 E41/E42-E48 的 smoke 往正式 split 推进：OpenProblems、sciplex3、Norman、TCDD。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sciplex3-genes", type=int, default=1000)
    parser.add_argument("--norman-genes", type=int, default=1000)
    parser.add_argument("--tcdd-genes", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=15)
    parser.add_argument("--max-cells-per-group", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()

    op_scores, op_summary, op_splits = run_openproblems_formal()
    op_group_summary = summarize_openproblems_groups(op_scores)
    sc_scores, sc_summary, sc_status = run_sciplex3_formal(args)
    norman_scores, norman_summary = run_norman_formal(args)
    tcdd_scores, tcdd_summary, tcdd_status = run_tcdd_dose_aware(args)

    save_csv(op_scores, TABLES / "E49_OPENPROBLEMS_SCORE_TABLE.csv")
    save_csv(op_summary, TABLES / "E49_OPENPROBLEMS_SUMMARY.csv")
    save_csv(op_group_summary, TABLES / "E49_OPENPROBLEMS_GROUP_SUMMARY.csv")
    save_csv(op_splits, TABLES / "E49_OPENPROBLEMS_SPLITS.csv")
    save_csv(sc_scores, TABLES / "E50_SCIPLEX3_SCORE_TABLE.csv")
    save_csv(sc_summary, TABLES / "E50_SCIPLEX3_SUMMARY.csv")
    save_csv(sc_status, TABLES / "E50_SCIPLEX3_STATUS.csv")
    save_csv(norman_scores, TABLES / "E51_NORMAN_COMBO_SCORE_TABLE.csv")
    save_csv(norman_summary, TABLES / "E51_NORMAN_COMBO_SUMMARY.csv")
    save_csv(tcdd_scores, TABLES / "E52_TCDD_DOSE_AWARE_SCORE_TABLE.csv")
    save_csv(tcdd_summary, TABLES / "E52_TCDD_DOSE_AWARE_SUMMARY.csv")
    save_csv(tcdd_status, TABLES / "E52_TCDD_DOSE_AWARE_STATUS.csv")

    write_report(
        {
            "E49 OpenProblems pooled": op_group_summary,
            "E49 OpenProblems split-level": op_summary,
            "E50 sciplex3": sc_summary,
            "E51 Norman": norman_summary,
            "E52 TCDD dose-aware": tcdd_summary,
        },
        {
            "OpenProblems splits": op_splits,
            "sciplex3 status": sc_status,
            "TCDD status": tcdd_status,
        },
    )
    status = {
        "generated_at": now_text(),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "output_dir": rel(OUT),
        "n_openproblems_rows": int(len(op_scores)),
        "n_sciplex3_rows": int(len(sc_scores)),
        "n_norman_combo_rows": int(len(norman_scores)),
        "n_tcdd_rows": int(len(tcdd_scores)),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
