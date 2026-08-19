#!/usr/bin/env python3
"""Tahoe-100M pseudobulk adapter smoke test.

This is intentionally not a formal benchmark. It checks whether Tahoe
pseudobulk differential-expression shards can be converted into a SafeConf-like
PredictionRecord table without downloading the 337GB raw expression matrix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TAHOE_ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas/mega_external/Tahoe-100M")
DEFAULT_OUT_DIR = Path(
    "/home/yyf/proj/code/20260426_154505_perturb_transport_final_push/outputs/"
    "safeconf_tahoe_pseudobulk_smoke_20260605"
)

PB_COLUMNS = [
    "gene_name",
    "log2FoldChange",
    "n_cells_trt",
    "n_cells_ctrl",
    "plate",
    "Cell_ID_Cellosaur",
    "Cell_ID_DepMap",
    "Cell_Name_Vevo",
    "drug",
    "concentration",
    "concentration_unit",
]


def stable_mod(text: str, modulus: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def rank_residual(values: pd.Series, control: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"v": values, "c": control}).apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(frame) < 3 or frame["c"].nunique() < 2:
        return out
    y = frame["v"].rank(method="average").to_numpy(dtype=float)
    z = frame["c"].rank(method="average").to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    out.loc[frame.index] = y - design @ beta
    return out


def partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    rx = rank_residual(x, control)
    ry = rank_residual(y, control)
    mask = rx.notna() & ry.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rx[mask].corr(ry[mask], method="pearson"))


def robust_z(values: pd.Series, ref: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    ref = pd.to_numeric(ref, errors="coerce")
    med = ref.median()
    if pd.isna(med):
        med = 0.0
    scale = ref.quantile(0.75) - ref.quantile(0.25)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = ref.std()
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (values.fillna(med) - med) / scale


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "tables": out_dir / "tables",
        "reports": out_dir / "reports",
        "arrays": out_dir / "arrays",
        "logs": out_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_task_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    depmap = out["Cell_ID_DepMap"].fillna("").astype(str)
    cellosaur = out["Cell_ID_Cellosaur"].fillna("").astype(str)
    vevo = out["Cell_Name_Vevo"].fillna("").astype(str)
    context = depmap.mask(depmap.eq("") | depmap.eq("nan"), cellosaur)
    context = context.mask(context.eq("") | context.eq("nan"), vevo)
    out["context"] = context
    out["perturbation"] = (
        out["drug"].astype(str)
        + "@"
        + out["concentration"].astype(str)
        + out["concentration_unit"].fillna("").astype(str)
    )
    out["task_key"] = out["context"] + "||" + out["perturbation"]
    return out


def load_gene_panel(tahoe_root: Path, max_genes: int) -> list[str]:
    gene_meta = pd.read_parquet(
        tahoe_root / "metadata" / "gene_metadata.parquet",
        columns=["gene_symbol", "token_id"],
    )
    gene_meta = gene_meta.dropna(subset=["gene_symbol"]).sort_values("token_id")
    return gene_meta["gene_symbol"].astype(str).drop_duplicates().head(max_genes).tolist()


def select_shards(all_shards: list[Path], max_shards: int, base_shards: int = 0) -> list[Path]:
    if max_shards >= len(all_shards):
        return all_shards
    if base_shards <= 0:
        indices = np.linspace(0, len(all_shards) - 1, max_shards, dtype=int)
        return [all_shards[int(index)] for index in indices]

    n_base = min(base_shards, max_shards)
    selected_indices = set(np.linspace(0, len(all_shards) - 1, n_base, dtype=int).tolist())
    n_extra = max_shards - len(selected_indices)
    remaining = [index for index in range(len(all_shards)) if index not in selected_indices]
    if n_extra > 0:
        extra_positions = np.linspace(0, len(remaining) - 1, n_extra, dtype=int)
        selected_indices.update(remaining[int(position)] for position in extra_positions)
    return [all_shards[index] for index in sorted(selected_indices)]


def load_sample_rows(
    pseudobulk_dir: Path,
    max_shards: int,
    max_tasks: int,
    min_genes_per_task: int,
    gene_panel: set[str] | None = None,
    base_shards: int = 0,
    progress_every: int = 10,
    progress_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    all_shards = sorted(pseudobulk_dir.glob("*.parquet"))
    shards = select_shards(all_shards, max_shards=max_shards, base_shards=base_shards)
    selected_parts: list[pd.DataFrame] = []
    task_rows: list[pd.DataFrame] = []
    skipped_rows: list[dict] = []
    selected_task_keys: set[str] = set()

    processed_shards = 0
    for shard_number, shard in enumerate(shards, start=1):
        processed_shards = shard_number
        try:
            df = pd.read_parquet(shard, columns=PB_COLUMNS)
        except Exception as exc:
            skipped_rows.append(
                {
                    "shard_path": str(shard),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "event": "shard_skipped",
                                "shard_number": shard_number,
                                "n_shards_requested": len(shards),
                                "shard_path": str(shard),
                                "error": type(exc).__name__,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            continue
        df = build_task_key(df)
        df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
        valid = df.dropna(subset=["context", "drug", "concentration", "gene_name"])
        valid = valid[~valid["context"].astype(str).isin(["", "nan"])]
        valid = valid[~valid["drug"].astype(str).str.upper().eq("DMSO_TF")]
        if gene_panel is not None:
            valid = valid[valid["gene_name"].astype(str).isin(gene_panel)]
        stats = (
            valid.groupby("task_key", observed=False)
            .agg(
                context=("context", "first"),
                perturbation=("perturbation", "first"),
                drug=("drug", "first"),
                concentration=("concentration", "first"),
                concentration_unit=("concentration_unit", "first"),
                plate=("plate", "first"),
                cell_id_depmap=("Cell_ID_DepMap", "first"),
                cell_id_cellosaur=("Cell_ID_Cellosaur", "first"),
                cell_name_vevo=("Cell_Name_Vevo", "first"),
                n_gene_rows=("gene_name", "size"),
                n_nonnull_lfc=("log2FoldChange", lambda x: int(x.notna().sum())),
                median_n_cells_trt=("n_cells_trt", "median"),
                median_n_cells_ctrl=("n_cells_ctrl", "median"),
            )
            .reset_index()
        )
        stats = stats[stats["n_nonnull_lfc"] >= min_genes_per_task].copy()
        stats = stats.sort_values(["drug", "context", "concentration", "task_key"])
        remaining = max_tasks - len(selected_task_keys)
        if remaining <= 0:
            break
        by_context = {
            str(ctx): [k for k in group["task_key"].astype(str).tolist() if k not in selected_task_keys]
            for ctx, group in stats.groupby("context", dropna=False)
        }
        new_keys: list[str] = []
        while len(new_keys) < remaining and any(by_context.values()):
            for ctx in sorted(by_context):
                if by_context[ctx] and len(new_keys) < remaining:
                    new_keys.append(by_context[ctx].pop(0))
        if not new_keys:
            if shard_number == 1 or shard_number % max(1, progress_every) == 0:
                progress = {
                    "event": "shard_progress",
                    "shard_number": shard_number,
                    "n_shards_requested": len(shards),
                    "shard_path": str(shard),
                    "selected_tasks": len(selected_task_keys),
                    "max_tasks": max_tasks,
                    "eligible_tasks_in_shard": int(len(stats)),
                    "skipped_shards": len(skipped_rows),
                }
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                if progress_path is not None:
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
            continue
        selected_task_keys.update(new_keys)
        selected = valid[valid["task_key"].astype(str).isin(new_keys)].copy()
        selected_parts.append(selected)
        task_rows.append(stats[stats["task_key"].astype(str).isin(new_keys)].copy())
        if shard_number == 1 or shard_number % max(1, progress_every) == 0 or len(selected_task_keys) >= max_tasks:
            progress = {
                "event": "shard_progress",
                "shard_number": shard_number,
                "n_shards_requested": len(shards),
                "shard_path": str(shard),
                "selected_tasks": len(selected_task_keys),
                "max_tasks": max_tasks,
                "eligible_tasks_in_shard": int(len(stats)),
                "skipped_shards": len(skipped_rows),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
        if len(selected_task_keys) >= max_tasks:
            break
        del df
        del valid
        gc.collect()

    if not selected_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(skipped_rows), processed_shards
    return (
        pd.concat(selected_parts, ignore_index=True),
        pd.concat(task_rows, ignore_index=True),
        pd.DataFrame(skipped_rows),
        processed_shards,
    )


def build_effect_matrix(
    rows: pd.DataFrame,
    task_meta: pd.DataFrame,
    max_genes: int,
    fixed_genes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    pivot = rows.pivot_table(
        index="task_key",
        columns="gene_name",
        values="log2FoldChange",
        aggfunc="mean",
    )
    if fixed_genes is None:
        gene_stats = pd.DataFrame(
            {
                "gene_name": pivot.columns.astype(str),
                "nonnull_tasks": pivot.notna().sum(axis=0).to_numpy(),
                "variance": pivot.var(axis=0, skipna=True).fillna(0.0).to_numpy(),
            }
        )
        gene_stats = gene_stats.sort_values(["nonnull_tasks", "variance", "gene_name"], ascending=[False, False, True])
        genes = gene_stats["gene_name"].head(max_genes).tolist()
    else:
        genes = [gene for gene in fixed_genes if gene in pivot.columns][:max_genes]
    effect = pivot.reindex(columns=genes).fillna(0.0).astype(float)
    meta = task_meta.drop_duplicates("task_key").set_index("task_key").loc[effect.index].reset_index()
    meta["task_id"] = np.arange(len(meta))
    return effect, genes, meta


def build_split(task_meta: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows: list[dict] = []
    contexts = task_meta["context"].astype(str)
    perturbations = task_meta["perturbation"].astype(str)
    keys = task_meta["task_key"].astype(str)

    for fold in range(n_folds):
        candidate_test = keys.map(lambda x: stable_mod(x, n_folds) == fold)
        test_mask = []
        for idx, row in task_meta.iterrows():
            if not bool(candidate_test.loc[idx]):
                test_mask.append(False)
                continue
            train = task_meta.loc[~candidate_test].copy()
            context_seen = bool(train["context"].astype(str).eq(str(row["context"])).any())
            pert_seen = bool(train["perturbation"].astype(str).eq(str(row["perturbation"])).any())
            test_mask.append(context_seen and pert_seen)
        test_mask = pd.Series(test_mask, index=task_meta.index)
        for idx, row in task_meta.iterrows():
            if bool(test_mask.loc[idx]):
                split = "test"
            else:
                split = "val" if stable_mod(f"{row['task_key']}::val::{fold}", 10) == 0 else "train"
            train_rows = task_meta.loc[~test_mask & (task_meta.index != idx)]
            pair_seen = bool(train_rows["task_key"].astype(str).eq(str(row["task_key"])).any())
            context_seen = bool(train_rows["context"].astype(str).eq(str(row["context"])).any())
            pert_seen = bool(train_rows["perturbation"].astype(str).eq(str(row["perturbation"])).any())
            rows.append(
                {
                    "dataset_name": "Tahoe100M_pseudobulk_smoke",
                    "task_id": int(row["task_id"]),
                    "task_key": row["task_key"],
                    "context": row["context"],
                    "perturbation": row["perturbation"],
                    "drug": row["drug"],
                    "concentration": row["concentration"],
                    "concentration_unit": row["concentration_unit"],
                    "plate": row["plate"],
                    "fold_id": fold,
                    "split": split,
                    "pair_seen_in_train": pair_seen if split == "test" else False,
                    "context_seen_in_train": context_seen,
                    "perturbation_seen_in_train": pert_seen,
                }
            )
    return pd.DataFrame(rows)


def assign_context_folds(task_meta: pd.DataFrame, n_folds: int) -> dict[str, int]:
    counts = (
        task_meta.groupby("context", observed=False)["task_key"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    fold_loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for context, n_tasks in counts.items():
        fold = min(range(n_folds), key=lambda index: (fold_loads[index], index))
        assignment[str(context)] = int(fold)
        fold_loads[fold] += int(n_tasks)
    return assignment


def build_context_holdout_split(task_meta: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows: list[dict] = []
    context_folds = assign_context_folds(task_meta, n_folds=n_folds)

    for fold in range(n_folds):
        test_contexts = {context for context, assigned_fold in context_folds.items() if assigned_fold == fold}
        test_mask = task_meta["context"].astype(str).isin(test_contexts)
        fold_splits = pd.Series(index=task_meta.index, dtype=object)
        fold_splits.loc[test_mask] = "test"
        for idx, row in task_meta.loc[~test_mask].iterrows():
            fold_splits.loc[idx] = (
                "val" if stable_mod(f"{row['task_key']}::val::{fold}", 10) == 0 else "train"
            )
        train_rows = task_meta.loc[fold_splits.eq("train")]
        for idx, row in task_meta.iterrows():
            split = str(fold_splits.loc[idx])
            support_rows = train_rows.loc[train_rows.index != idx]
            pair_seen = bool(support_rows["task_key"].astype(str).eq(str(row["task_key"])).any())
            context_seen = bool(support_rows["context"].astype(str).eq(str(row["context"])).any())
            pert_seen = bool(support_rows["perturbation"].astype(str).eq(str(row["perturbation"])).any())
            rows.append(
                {
                    "dataset_name": "Tahoe100M_pseudobulk_smoke",
                    "task_id": int(row["task_id"]),
                    "task_key": row["task_key"],
                    "context": row["context"],
                    "perturbation": row["perturbation"],
                    "drug": row["drug"],
                    "concentration": row["concentration"],
                    "concentration_unit": row["concentration_unit"],
                    "plate": row["plate"],
                    "fold_id": fold,
                    "split": split,
                    "pair_seen_in_train": pair_seen if split == "test" else False,
                    "context_seen_in_train": context_seen,
                    "perturbation_seen_in_train": pert_seen,
                    "heldout_context_fold": int(context_folds[str(row["context"])]),
                }
            )
    return pd.DataFrame(rows)


def build_v0_records(
    effect: pd.DataFrame,
    task_meta: pd.DataFrame,
    split_df: pd.DataFrame,
    min_exact_support: int,
    record_splits: set[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    task_to_pos = {task: i for i, task in enumerate(effect.index.astype(str))}
    task_meta = task_meta.set_index("task_key")
    pred_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    pred_by_task: dict[tuple[int, str, str], np.ndarray] = {}
    rec_i = 0

    for fold, fold_split in split_df.groupby("fold_id", dropna=False):
        train_keys = fold_split[fold_split["split"].eq("train")]["task_key"].astype(str).tolist()
        target_rows = fold_split[fold_split["split"].isin(record_splits)].copy()
        train_meta = task_meta.loc[[k for k in train_keys if k in task_meta.index]].copy()
        for _, row in target_rows.iterrows():
            task_key = str(row["task_key"])
            pert = str(row["perturbation"])
            drug = str(row["drug"])
            context = str(row["context"])
            exact_support_keys = train_meta[
                train_meta["perturbation"].astype(str).eq(pert)
                & ~train_meta["context"].astype(str).eq(context)
            ].index.astype(str).tolist()
            exact_support_keys = [k for k in exact_support_keys if k in task_to_pos and k != task_key]
            drug_support_keys = train_meta[
                train_meta["drug"].astype(str).eq(drug)
                & ~train_meta["context"].astype(str).eq(context)
            ].index.astype(str).tolist()
            drug_support_keys = [k for k in drug_support_keys if k in task_to_pos and k != task_key]
            if len(exact_support_keys) < min_exact_support or task_key not in task_to_pos:
                continue
            predictors = [
                ("V0ExactDoseMean", exact_support_keys, "same_drug_same_concentration_other_cell_lines"),
            ]
            if drug_support_keys:
                predictors.append(
                    ("V0DrugMeanAcrossDose", drug_support_keys, "same_drug_all_concentrations_other_cell_lines")
                )
            true = effect.loc[task_key].to_numpy(dtype=float)
            for predictor_name, support_keys, support_definition in predictors:
                support_mat = effect.loc[support_keys].to_numpy(dtype=float)
                pred = np.nanmean(support_mat, axis=0)
                record_id = f"tahoe_smoke_rec_{rec_i:06d}"
                pred_key = f"{record_id}::predicted_effect"
                true_key = f"{record_id}::true_effect"
                pred_arrays[pred_key] = pred
                true_arrays[true_key] = true
                pred_by_task[(int(fold), task_key, predictor_name)] = pred
                rows.append(
                    {
                        "schema_version": "safeconf_prediction_record_v1",
                        "record_id": record_id,
                        "task_id": int(row["task_id"]),
                        "task_key": task_key,
                        "dataset_name": "Tahoe100M_pseudobulk_smoke",
                        "dataset_group": "tahoe_mega_external",
                        "fold_id": int(fold),
                        "split": row["split"],
                        "context": row["context"],
                        "perturbation": pert,
                        "drug": drug,
                        "concentration": row["concentration"],
                        "concentration_unit": row["concentration_unit"],
                        "plate": row["plate"],
                        "predictor_name": predictor_name,
                        "run_type": "smoke",
                        "gene_panel_id": "tahoe_pseudobulk_smoke_gene_panel",
                        "gene_order_hash": "",
                        "effect_definition": "logFC",
                        "normalization_id": "Tahoe_pseudobulk_DESeq2_log2FoldChange",
                        "error_normalization": "raw_rmse",
                        "predicted_effect_key": pred_key,
                        "true_effect_key": true_key,
                        "true_error_rmse": rmse(pred, true),
                        "true_error_cosine": cosine_distance(pred, true),
                        "true_effect_l2_norm": float(np.linalg.norm(true) / np.sqrt(len(true))),
                        "perturbation_support_count": len(support_keys),
                        "support_definition": support_definition,
                        "exact_dose_support_count": len(exact_support_keys),
                        "drug_across_dose_support_count": len(drug_support_keys),
                    }
                )
                rec_i += 1

    records = pd.DataFrame(rows)
    disagreement_rows: list[dict] = []
    if not records.empty:
        for (fold, task_key), group in records.groupby(["fold_id", "task_key"], dropna=False):
            preds = {
                str(row["predictor_name"]): pred_by_task[(int(fold), str(task_key), str(row["predictor_name"]))]
                for _, row in group.iterrows()
            }
            if "V0ExactDoseMean" not in preds or "V0DrugMeanAcrossDose" not in preds:
                continue
            disagreement_rows.append(
                {
                    "fold_id": int(fold),
                    "task_key": task_key,
                    "model_disagreement_rmse": rmse(preds["V0ExactDoseMean"], preds["V0DrugMeanAcrossDose"]),
                    "model_disagreement_cosine": cosine_distance(
                        preds["V0ExactDoseMean"], preds["V0DrugMeanAcrossDose"]
                    ),
                }
            )
    disagreement = pd.DataFrame(disagreement_rows)
    if not records.empty and not disagreement.empty:
        records = records.merge(disagreement, on=["fold_id", "task_key"], how="left")
    return records, pred_arrays, true_arrays, disagreement


def concentration_leakage_audit(split_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fold, fold_split in split_df.groupby("fold_id", dropna=False):
        train = fold_split[fold_split["split"].eq("train")].copy()
        test = fold_split[fold_split["split"].eq("test")].copy()
        for _, row in test.iterrows():
            same_drug_train = train[train["drug"].astype(str).eq(str(row["drug"]))]
            same_drug_other_conc = same_drug_train[
                ~same_drug_train["perturbation"].astype(str).eq(str(row["perturbation"]))
            ]
            rows.append(
                {
                    "fold_id": int(fold),
                    "task_key": row["task_key"],
                    "drug": row["drug"],
                    "perturbation": row["perturbation"],
                    "same_drug_train_tasks": int(len(same_drug_train)),
                    "same_drug_other_concentration_train_tasks": int(len(same_drug_other_conc)),
                    "has_same_drug_other_concentration_in_train": bool(len(same_drug_other_conc) > 0),
                }
            )
    return pd.DataFrame(rows)


def heldout_drug_feasibility(split_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Audit whether a held-out-drug split is compatible with V0-family predictors."""
    rows: list[dict] = []
    for fold in range(n_folds):
        train = split_df[split_df["fold_id"].eq(fold)].copy()
        train = train[train["drug"].astype(str).map(lambda x: stable_mod(f"drug::{x}", n_folds) != fold)]
        test = split_df[split_df["fold_id"].eq(fold)].copy()
        test = test[test["drug"].astype(str).map(lambda x: stable_mod(f"drug::{x}", n_folds) == fold)]
        for _, row in test.drop_duplicates("task_key").iterrows():
            same_drug_train = train[train["drug"].astype(str).eq(str(row["drug"]))]
            rows.append(
                {
                    "fold_id": int(fold),
                    "task_key": row["task_key"],
                    "drug": row["drug"],
                    "perturbation": row["perturbation"],
                    "same_drug_train_tasks": int(len(same_drug_train)),
                    "v0_family_predictor_applicable": bool(len(same_drug_train) > 0),
                    "note": "held-out-drug split removes same-drug support required by V0 Tahoe predictors",
                }
            )
    return pd.DataFrame(rows)


def plate_audit(task_meta: pd.DataFrame, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_line_plate = (
        task_meta.groupby("context", observed=False)
        .agg(n_plates=("plate", "nunique"), n_tasks=("task_key", "nunique"))
        .reset_index()
        .sort_values(["n_plates", "n_tasks"], ascending=[False, False])
    )
    rows: list[dict] = []
    for fold, fold_split in split_df.groupby("fold_id", dropna=False):
        train_plates = set(fold_split.loc[fold_split["split"].eq("train"), "plate"].astype(str))
        test = fold_split[fold_split["split"].eq("test")]
        for _, row in test.iterrows():
            rows.append(
                {
                    "fold_id": int(fold),
                    "task_key": row["task_key"],
                    "context": row["context"],
                    "perturbation": row["perturbation"],
                    "plate": row["plate"],
                    "test_plate_seen_in_train": str(row["plate"]) in train_plates,
                }
            )
    return cell_line_plate, pd.DataFrame(rows)


def build_scores_and_eval(
    records: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if records.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    score_frames: list[pd.DataFrame] = []
    for (fold, predictor), group in records.groupby(["fold_id", "predictor_name"], dropna=False):
        train_ref = group[group["split"].isin(["train", "val"])].copy()
        if train_ref.empty:
            train_ref = group.copy()
        scored = group.copy()
        support_z = robust_z(np.log1p(scored["perturbation_support_count"]), np.log1p(train_ref["perturbation_support_count"]))
        disagree_z = robust_z(scored["model_disagreement_rmse"], train_ref["model_disagreement_rmse"])
        scored["score_name"] = "tahoe_protocol_v0_2_confidence"
        scored["score_type"] = "confidence"
        scored["score_value"] = support_z - disagree_z
        scored["risk_axis_value"] = -scored["score_value"]
        score_frames.append(
            scored[
                [
                    "record_id",
                    "dataset_name",
                    "dataset_group",
                    "fold_id",
                    "split",
                    "context",
                    "perturbation",
                    "drug",
                    "predictor_name",
                    "score_name",
                    "score_type",
                    "score_value",
                    "risk_axis_value",
                    "true_error_rmse",
                    "true_effect_l2_norm",
                ]
            ]
        )
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    test_scores = scores[scores["split"].eq("test")].copy()
    if test_scores.empty:
        return scores, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    group_specs: list[tuple[str, tuple | str, pd.DataFrame]] = [("overall", "overall", test_scores)]
    for fold, group in test_scores.groupby("fold_id", dropna=False):
        group_specs.append(("fold", f"fold{fold}", group))
    for predictor, group in test_scores.groupby("predictor_name", dropna=False):
        group_specs.append(("predictor", str(predictor), group))
    for (predictor, fold), group in test_scores.groupby(["predictor_name", "fold_id"], dropna=False):
        group_specs.append(("predictor_fold", f"{predictor}::fold{fold}", group))

    for level, group_id, group in group_specs:
        aligned = spearman(group["risk_axis_value"], group["true_error_rmse"])
        partial = partial_spearman(group["risk_axis_value"], group["true_error_rmse"], group["true_effect_l2_norm"])
        rows.append(
            {
                "level": level,
                "group_id": group_id,
                "dataset_name": "Tahoe100M_pseudobulk_smoke",
                "score_name": "tahoe_protocol_v0_2_confidence",
                "n_records": int(len(group)),
                "aligned_rho": aligned,
                "partial_rho_control_magnitude": partial,
                "mean_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").mean()),
                "median_rmse": float(pd.to_numeric(group["true_error_rmse"], errors="coerce").median()),
                "mean_effect_scale": float(pd.to_numeric(group["true_effect_l2_norm"], errors="coerce").mean()),
            }
        )
    eval_df = pd.DataFrame(rows)

    cov_rows: list[dict] = []
    for (predictor, fold), group in test_scores.groupby(["predictor_name", "fold_id"], dropna=False):
        group = group.sort_values("score_value", ascending=False).copy()
        full_mean = float(group["true_error_rmse"].mean()) if len(group) else math.nan
        for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            n_keep = max(1, int(math.ceil(len(group) * coverage)))
            kept = group.head(n_keep)
            mean_rmse = float(kept["true_error_rmse"].mean())
            cov_rows.append(
                {
                    "dataset_name": "Tahoe100M_pseudobulk_smoke",
                    "predictor_name": str(predictor),
                    "fold_id": int(fold),
                    "score_name": "tahoe_protocol_v0_2_confidence",
                    "coverage": float(coverage),
                    "n_kept": int(n_keep),
                    "mean_rmse": mean_rmse,
                    "median_rmse": float(kept["true_error_rmse"].median()),
                    "full_mean_rmse": full_mean,
                    "rmse_improvement_pct": float((full_mean - mean_rmse) / full_mean * 100.0)
                    if np.isfinite(full_mean) and full_mean > 0
                    else math.nan,
                }
            )
    cov_df = pd.DataFrame(cov_rows)

    rng = np.random.default_rng(seed)
    boot_rows: list[dict] = []
    for predictor, group in test_scores.groupby("predictor_name", dropna=False):
        group = group.reset_index(drop=True)
        vals: list[float] = []
        if len(group) >= 5:
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(group), len(group))
                sample = group.iloc[idx]
                vals.append(spearman(sample["risk_axis_value"], sample["true_error_rmse"]))
        clean = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        boot_rows.append(
            {
                "dataset_name": "Tahoe100M_pseudobulk_smoke",
                "predictor_name": str(predictor),
                "score_name": "tahoe_protocol_v0_2_confidence",
                "n_bootstrap": int(len(clean)),
                "aligned_rho_bootstrap_mean": float(np.mean(clean)) if len(clean) else math.nan,
                "aligned_rho_ci_low": float(np.quantile(clean, 0.025)) if len(clean) else math.nan,
                "aligned_rho_ci_high": float(np.quantile(clean, 0.975)) if len(clean) else math.nan,
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    return scores, eval_df, cov_df, boot_df


def write_report(out_dir: Path, status: dict) -> None:
    report = f"""# Tahoe pseudobulk adapter smoke report

## 结论

当前状态：`{status['decision']}`。

这不是论文正式结果，只是检查 Tahoe pseudobulk 能否转成 SafeConf PredictionRecord。

## 输入

- Tahoe root: `{status['tahoe_root']}`
- pseudobulk shards available: {status['n_available_shards']}
- shards scanned: {status['n_scanned_shards']}
- skipped/corrupt shards: {status.get('n_skipped_shards')}
- selected tasks: {status['n_tasks']}
- selected genes: {status['n_genes']}

## Smoke 输出

- PredictionRecords: {status['n_prediction_records']}
- Predictor names: {status['predictors']}
- test pair leakage rows: {status['test_pair_leakage_rows']}
- missing context support in test: {status['test_context_missing_rows']}
- missing perturbation support in test: {status['test_perturbation_missing_rows']}
- concentration leakage rows: {status['test_same_drug_other_concentration_rows']}
- test plate seen in train ratio: {status['test_plate_seen_in_train_ratio']}
- true_error_rmse CV: {status['true_error_rmse_cv']}
- mean RMSE: {status['mean_rmse']}
- median RMSE: {status['median_rmse']}
- formal eval: {status['formal_eval']}
- aligned rho: {status.get('formal_aligned_rho')}
- partial rho controlling effect magnitude: {status.get('formal_partial_rho')}
- RC@80 improvement pct: {status.get('formal_risk_cov_80_improve_pct')}
- held-out drug split note: {status.get('heldout_drug_split_note')}

## 怎么理解

如果 `PredictionRecords > 0` 且 leakage 为 0，说明 Tahoe 可以进入下一步 adapter/formal external validation。

如果这里失败，不代表 SafeConf 失败，只说明 Tahoe 的 pseudobulk 结构暂时不能直接进入当前协议。
"""
    (out_dir / "reports" / "TAHOE_PSEUDOBULK_SMOKE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tahoe pseudobulk adapter smoke test.")
    parser.add_argument("--tahoe-root", default=str(DEFAULT_TAHOE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=300)
    parser.add_argument("--max-genes", type=int, default=1000)
    parser.add_argument("--min-genes-per-task", type=int, default=1000)
    parser.add_argument("--min-exact-support", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--split-mode", choices=["heldout_pair", "heldout_context"], default="heldout_pair")
    parser.add_argument("--fixed-gene-panel", action="store_true")
    parser.add_argument("--include-train-val-records", action="store_true")
    parser.add_argument("--formal-eval", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument(
        "--base-shards",
        type=int,
        default=0,
        help="Preserve this many evenly spaced base shards, then add shards up to --max-shards.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    tahoe_root = Path(args.tahoe_root)
    out_dir = Path(args.out_dir)
    paths = ensure_dirs(out_dir)
    pseudobulk_dir = tahoe_root / "metadata" / "pseudobulk_differential_expression"
    shards = sorted(pseudobulk_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {pseudobulk_dir}")

    selected_shards = select_shards(shards, max_shards=args.max_shards, base_shards=args.base_shards)
    base_selection = set(select_shards(shards, max_shards=args.base_shards)) if args.base_shards > 0 else set()
    pd.DataFrame(
        [
            {
                "selection_order": order,
                "source_shard_index": shards.index(shard),
                "shard_path": str(shard),
                "is_base_shard": shard in base_selection,
            }
            for order, shard in enumerate(selected_shards)
        ]
    ).to_csv(paths["tables"] / "TAHOE_SHARD_SELECTION_SMOKE.csv", index=False)

    fixed_genes = load_gene_panel(tahoe_root, args.max_genes) if args.fixed_gene_panel else None
    rows, task_meta_raw, skipped_shards, processed_shards = load_sample_rows(
        pseudobulk_dir,
        max_shards=args.max_shards,
        max_tasks=args.max_tasks,
        min_genes_per_task=args.min_genes_per_task,
        gene_panel=set(fixed_genes) if fixed_genes is not None else None,
        base_shards=args.base_shards,
        progress_every=args.progress_every,
        progress_path=paths["logs"] / "TAHOE_SHARD_PROGRESS.jsonl",
    )
    if rows.empty or task_meta_raw.empty:
        raise RuntimeError("No eligible Tahoe pseudobulk tasks selected")

    effect, genes, task_meta = build_effect_matrix(
        rows,
        task_meta_raw,
        max_genes=args.max_genes,
        fixed_genes=fixed_genes,
    )
    split_df = (
        build_context_holdout_split(task_meta, n_folds=args.n_folds)
        if args.split_mode == "heldout_context"
        else build_split(task_meta, n_folds=args.n_folds)
    )
    record_splits = {"train", "val", "test"} if args.include_train_val_records else {"test"}
    records, pred_arrays, true_arrays, disagreement = build_v0_records(
        effect,
        task_meta,
        split_df,
        min_exact_support=args.min_exact_support,
        record_splits=record_splits,
    )
    conc_audit = concentration_leakage_audit(split_df)
    drug_feas = heldout_drug_feasibility(split_df, n_folds=args.n_folds)
    cell_line_plate, test_plate = plate_audit(task_meta, split_df)
    scores, eval_df, cov_df, boot_df = (
        build_scores_and_eval(records, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if args.formal_eval
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )

    task_meta.to_csv(paths["tables"] / "TAHOE_TASK_INDEX_SMOKE.csv", index=False)
    pd.DataFrame({"gene_order": np.arange(len(genes)), "gene_name": genes}).to_csv(
        paths["tables"] / "TAHOE_GENE_PANEL_SMOKE.csv", index=False
    )
    skipped_shards.to_csv(paths["tables"] / "TAHOE_SKIPPED_SHARDS_SMOKE.csv", index=False)
    split_df.to_csv(paths["tables"] / "TAHOE_HELDOUT_PAIR_SPLITS_SMOKE.csv", index=False)
    records.to_csv(paths["tables"] / "TAHOE_PREDICTION_RECORDS_SMOKE.csv", index=False)
    disagreement.to_csv(paths["tables"] / "TAHOE_PREDICTOR_DISAGREEMENT_SMOKE.csv", index=False)
    conc_audit.to_csv(paths["tables"] / "TAHOE_CONCENTRATION_LEAKAGE_AUDIT_SMOKE.csv", index=False)
    drug_feas.to_csv(paths["tables"] / "TAHOE_HELDOUT_DRUG_SPLIT_FEASIBILITY_SMOKE.csv", index=False)
    cell_line_plate.to_csv(paths["tables"] / "TAHOE_CELL_LINE_PLATE_AUDIT_SMOKE.csv", index=False)
    test_plate.to_csv(paths["tables"] / "TAHOE_TEST_PLATE_AUDIT_SMOKE.csv", index=False)
    if args.formal_eval:
        scores.to_csv(paths["tables"] / "TAHOE_PROTOCOL_SCORES_SMOKE.csv", index=False)
        eval_df.to_csv(paths["tables"] / "TAHOE_FORMAL_EVAL_SUMMARY_SMOKE.csv", index=False)
        cov_df.to_csv(paths["tables"] / "TAHOE_RISK_COVERAGE_SMOKE.csv", index=False)
        boot_df.to_csv(paths["tables"] / "TAHOE_BOOTSTRAP_CI_SMOKE.csv", index=False)
    np.savez(paths["arrays"] / "TAHOE_PREDICTED_EFFECTS_SMOKE.npz", **pred_arrays)
    np.savez(paths["arrays"] / "TAHOE_TRUE_EFFECTS_SMOKE.npz", **true_arrays)

    test = split_df[split_df["split"].eq("test")]
    err = pd.to_numeric(records["true_error_rmse"], errors="coerce") if len(records) else pd.Series(dtype=float)
    err_mean = float(err.mean()) if len(err) else math.nan
    err_std = float(err.std()) if len(err) else math.nan
    pair_leakage = int(test["pair_seen_in_train"].sum()) if "pair_seen_in_train" in test else 0
    context_seen_rows = int(test["context_seen_in_train"].sum()) if "context_seen_in_train" in test else 0
    perturbation_missing_rows = (
        int((~test["perturbation_seen_in_train"]).sum()) if "perturbation_seen_in_train" in test else 0
    )
    split_valid = pair_leakage == 0 and perturbation_missing_rows == 0
    if args.split_mode == "heldout_context":
        split_valid = split_valid and context_seen_rows == 0
    else:
        split_valid = split_valid and context_seen_rows == len(test)
    status = {
        "decision": "PASS_ADAPTER_SMOKE" if len(records) > 0 and split_valid else "CHECK_REQUIRED",
        "tahoe_root": str(tahoe_root),
        "pseudobulk_dir": str(pseudobulk_dir),
        "n_available_shards": len(shards),
        "n_selected_shards": int(len(selected_shards)),
        "n_scanned_shards": int(processed_shards),
        "shard_sampling_strategy": "nested_base_plus_expansion" if args.base_shards > 0 else "linspace",
        "base_shards": int(args.base_shards),
        "n_skipped_shards": int(len(skipped_shards)),
        "n_tasks": int(len(task_meta)),
        "n_contexts": int(task_meta["context"].nunique()),
        "n_perturbations": int(task_meta["perturbation"].nunique()),
        "max_genes": int(args.max_genes),
        "min_genes_per_task": int(args.min_genes_per_task),
        "min_task_nonnull_lfc": int(task_meta["n_nonnull_lfc"].min()),
        "median_task_nonnull_lfc": float(task_meta["n_nonnull_lfc"].median()),
        "min_exact_support": int(args.min_exact_support),
        "n_genes": int(len(genes)),
        "n_prediction_records": int(len(records)),
        "split_mode": str(args.split_mode),
        "predictors": sorted(records["predictor_name"].unique().tolist()) if len(records) else [],
        "test_pair_leakage_rows": pair_leakage,
        "test_context_seen_rows": context_seen_rows,
        "test_context_missing_rows": int((~test["context_seen_in_train"]).sum()) if "context_seen_in_train" in test else 0,
        "test_perturbation_missing_rows": perturbation_missing_rows,
        "test_same_drug_other_concentration_rows": int(conc_audit["has_same_drug_other_concentration_in_train"].sum()) if len(conc_audit) else 0,
        "test_same_drug_other_concentration_ratio": float(conc_audit["has_same_drug_other_concentration_in_train"].mean()) if len(conc_audit) else math.nan,
        "test_plate_seen_in_train_ratio": float(test_plate["test_plate_seen_in_train"].mean()) if len(test_plate) else math.nan,
        "cell_line_plate_median": float(cell_line_plate["n_plates"].median()) if len(cell_line_plate) else math.nan,
        "mean_rmse": err_mean,
        "median_rmse": float(err.median()) if len(err) else math.nan,
        "std_rmse": err_std,
        "true_error_rmse_cv": float(err_std / err_mean) if len(err) and err_mean > 0 else math.nan,
        "fixed_gene_panel": bool(args.fixed_gene_panel),
        "record_splits": sorted(record_splits),
        "formal_eval": bool(args.formal_eval),
        "formal_aligned_rho": math.nan,
        "formal_partial_rho": math.nan,
        "formal_risk_cov_80_improve_pct": math.nan,
        "heldout_drug_split_v0_applicable": False,
        "heldout_drug_split_note": "V0-family Tahoe predictors require same-drug support; held-out-drug split is reported as feasibility audit, not as the main score.",
        "effect_definition": "logFC",
        "run_type": "smoke",
    }
    if args.formal_eval and not eval_df.empty:
        overall = eval_df[eval_df["level"].eq("overall")]
        if not overall.empty:
            status["formal_aligned_rho"] = float(overall.iloc[0]["aligned_rho"])
            status["formal_partial_rho"] = float(overall.iloc[0]["partial_rho_control_magnitude"])
    if args.formal_eval and not cov_df.empty:
        cov80 = cov_df[cov_df["coverage"].eq(0.8)]
        if not cov80.empty:
            status["formal_risk_cov_80_improve_pct"] = float(cov80["rmse_improvement_pct"].mean())
    (out_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
