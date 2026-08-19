from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


CONTROL_STRINGS = {
    "",
    "control",
    "ctrl",
    "non-targeting",
    "nt",
    "mock",
    "vehicle",
    "dmso",
    "nan",
    "none",
    "null",
}
CONTEXT_CANDIDATES = ["cell_type", "cell_label", "condition1", "cell_line", "patient", "donor_id", "batch"]
PERT_CANDIDATES = ["perturbation", "gene", "condition2", "sgRNA", "GenePair"]


def read_scan_table(default_root: Path) -> pd.DataFrame:
    path = default_root / "metadata" / "h5ad_scan.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def select_gate_datasets(scan: pd.DataFrame, max_datasets: int = 4) -> pd.DataFrame:
    df = scan.copy()
    for col in ["scan_status", "modality", "perturbation_type", "suitable_perturbation_generalization"]:
        if col not in df:
            df[col] = ""
    df = df[
        (df["scan_status"] == "ok")
        & (df["modality"].astype(str).str.contains("RNA", case=False, na=False))
        & (df["perturbation_type"].astype(str).str.contains("genetic", case=False, na=False))
        & (df["has_control_like"].astype(str).str.lower() == "true")
        & (df["suitable_perturbation_generalization"].astype(str).str.lower() == "true")
    ].copy()
    preferred = ["Haber", "Parekh", "KaggleCrossCell", "kangCrossCell", "kangCrossPatient", "Norman", "Wessels"]
    df["priority"] = df["study_family"].map({k: i for i, k in enumerate(preferred)}).fillna(99)
    df = df.sort_values(["priority", "n_obs"], ascending=[True, True])
    return df.head(max_datasets)


def infer_column(obs: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in obs.columns and obs[col].nunique(dropna=True) > 1:
            return col
    low = {c.lower(): c for c in obs.columns}
    for cand in candidates:
        for key, real in low.items():
            if cand.lower() in key and obs[real].nunique(dropna=True) > 1:
                return real
    return None


def is_control_value(x: object) -> bool:
    if pd.isna(x):
        return True
    text = str(x).strip().lower()
    return text in CONTROL_STRINGS or text.startswith("ctrl") or text.startswith("control")


def is_missing_label(x: object) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() in {"", "nan", "none", "null"}


def normalize_obs_label(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def matrix_to_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def choose_gene_indices(adata, n_genes: int) -> np.ndarray:
    hv_cols = [c for c in adata.var.columns if str(c).startswith("highly_variable")]
    if hv_cols:
        col = hv_cols[-1]
        vals = np.asarray(adata.var[col]).astype(bool)
        idx = np.flatnonzero(vals)
        if len(idx) >= min(500, n_genes):
            return idx[:n_genes]
    return np.arange(min(n_genes, adata.n_vars))


def build_effect_tasks(path: Path, dataset: str, n_genes: int = 1500, min_cells: int = 15, max_cells_per_group: int = 400, seed: int = 0) -> tuple[list[dict], list[str], dict]:
    rng = np.random.default_rng(seed)
    adata = sc.read_h5ad(path)
    context_col = infer_column(adata.obs, CONTEXT_CANDIDATES)
    pert_col = infer_column(adata.obs, PERT_CANDIDATES)
    if context_col is None or pert_col is None:
        return [], [], {"error": "missing context or perturbation column", "path": str(path)}
    gene_idx = choose_gene_indices(adata, n_genes)
    genes = [str(adata.var_names[i]) for i in gene_idx]
    x = adata[:, gene_idx].layers["logNor"] if "logNor" in adata.layers else adata[:, gene_idx].X
    x = matrix_to_dense(x).astype(np.float32)
    obs = adata.obs[[context_col, pert_col]].copy()
    obs["_context"] = obs[context_col].map(normalize_obs_label)
    obs["_pert"] = obs[pert_col].map(normalize_obs_label)
    tasks = []
    control_means = {}
    group_indices = {}
    for (ctx, pert), sub in obs.groupby(["_context", "_pert"], observed=False):
        if not ctx or is_missing_label(pert):
            continue
        idx = sub.index
        pos = obs.index.get_indexer(idx)
        if len(pos) < min_cells:
            continue
        if len(pos) > max_cells_per_group:
            pos = rng.choice(pos, size=max_cells_per_group, replace=False)
        group_indices[(ctx, pert)] = pos
    for (ctx, pert), pos in group_indices.items():
        if is_control_value(pert):
            control_means[ctx] = x[pos].mean(axis=0)
    for (ctx, pert), pos in group_indices.items():
        if is_control_value(pert) or ctx not in control_means:
            continue
        effect = x[pos].mean(axis=0) - control_means[ctx]
        tasks.append(
            {
                "dataset": dataset,
                "context": str(ctx),
                "perturbation": str(pert),
                "effect": effect.astype(np.float32),
                "control_mean": control_means[ctx].astype(np.float32),
                "n_cells": int(len(pos)),
                "context_col": context_col,
                "perturbation_col": pert_col,
            }
        )
    meta = {
        "path": str(path),
        "dataset": dataset,
        "n_tasks": len(tasks),
        "n_contexts": len(set(t["context"] for t in tasks)),
        "n_perturbations": len(set(t["perturbation"] for t in tasks)),
        "context_col": context_col,
        "perturbation_col": pert_col,
        "n_genes": len(genes),
    }
    return tasks, genes, meta


def feasible_splits(tasks: list[dict], min_test_tasks: int = 2) -> list[dict]:
    rows = []
    contexts = sorted(set(t["context"] for t in tasks))
    perts = sorted(set(t["perturbation"] for t in tasks))
    for ctx in contexts:
        test = [i for i, t in enumerate(tasks) if t["context"] == ctx]
        train = [i for i, t in enumerate(tasks) if t["context"] != ctx]
        shared = set(tasks[i]["perturbation"] for i in test) & set(tasks[i]["perturbation"] for i in train)
        if len(test) >= min_test_tasks and len(train) >= min_test_tasks and shared:
            rows.append({"split_type": "leave_context", "heldout": ctx, "n_train": len(train), "n_test": len(test), "shared_perturbations": len(shared)})
    for pert in perts:
        test = [i for i, t in enumerate(tasks) if t["perturbation"] == pert]
        train = [i for i, t in enumerate(tasks) if t["perturbation"] != pert]
        shared_ctx = set(tasks[i]["context"] for i in test) & set(tasks[i]["context"] for i in train)
        if len(test) >= min_test_tasks and len(train) >= min_test_tasks and shared_ctx:
            rows.append({"split_type": "heldout_perturbation", "heldout": pert, "n_train": len(train), "n_test": len(test), "shared_contexts": len(shared_ctx)})
    return rows


def materialize_split(tasks: list[dict], split_type: str, heldout: str) -> tuple[np.ndarray, np.ndarray]:
    if split_type == "leave_context":
        test = np.array([i for i, t in enumerate(tasks) if t["context"] == heldout], dtype=int)
    elif split_type == "heldout_perturbation":
        test = np.array([i for i, t in enumerate(tasks) if t["perturbation"] == heldout], dtype=int)
    else:
        raise ValueError(split_type)
    train = np.array([i for i in range(len(tasks)) if i not in set(test.tolist())], dtype=int)
    return train, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-root", default="/home/yyf/datasets/singlecell_perturbation_atlas")
    p.add_argument("--out", required=True)
    p.add_argument("--max-datasets", type=int, default=4)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scan = read_scan_table(Path(args.atlas_root))
    selected = select_gate_datasets(scan, args.max_datasets)
    selected.to_csv(out, index=False)


if __name__ == "__main__":
    main()
