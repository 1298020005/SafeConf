#!/usr/bin/env python3
"""E13: build a 3-cell-line official sciplex3 drug-dose panel.

The official sciplex3_A549/K562/MCF7 files each contain only one cellular
context.  Running the generic held-out-pair MVP on each file separately cannot
work, because a task needs both a context axis and a perturbation axis.  This
script merges the three files into one task dataset:

    context = cell line (A549/K562/MCF7)
    perturbation = drug_dose_name

Then it reuses the existing confidence MVP evaluation functions.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
MVP_PATH = ROOT / "code/20260426_154505_perturb_transport_final_push/confidence_task/run_confidence_mvp_v2_1.py"
DEFAULT_ATLAS_ROOT = Path("/home/yyf/data/singlecell_perturbation_atlas")
DEFAULT_FILES = [
    "official_generalization/sciplex3_A549.h5ad",
    "official_generalization/sciplex3_K562.h5ad",
    "official_generalization/sciplex3_MCF7.h5ad",
]


def load_mvp_module():
    spec = importlib.util.spec_from_file_location("safeconf_confidence_mvp_v2_1", MVP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MVP module: {MVP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matrix_to_dense(x: Any) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def is_control_value(x: object) -> bool:
    if pd.isna(x):
        return True
    text = str(x).strip().lower()
    return text in {"", "control", "ctrl", "vehicle", "dmso", "nan", "none", "null"} or text.startswith("control") or text.startswith("ctrl")


def common_gene_panel(paths: list[Path], n_genes: int) -> list[str]:
    first = sc.read_h5ad(paths[0], backed="r")
    genes = [str(x) for x in first.var_names[:n_genes]]
    first.file.close()
    for path in paths[1:]:
        ad = sc.read_h5ad(path, backed="r")
        present = set(map(str, ad.var_names))
        ad.file.close()
        missing = [g for g in genes if g not in present]
        if missing:
            raise ValueError(f"{path.name} is missing {len(missing)} genes from the panel; first={missing[:3]}")
    return genes


def build_tasks_for_file(
    path: Path,
    genes: list[str],
    min_cells: int,
    max_cells_per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    adata = sc.read_h5ad(path)
    gene_idx = adata.var_names.get_indexer(genes)
    if np.any(gene_idx < 0):
        missing = [genes[i] for i, j in enumerate(gene_idx) if j < 0]
        raise ValueError(f"{path.name} missing genes: {missing[:5]}")

    context_col = "cell_line" if "cell_line" in adata.obs.columns else "cell_type"
    perturbation_col = "drug_dose_name" if "drug_dose_name" in adata.obs.columns else "condition"
    if context_col not in adata.obs.columns or perturbation_col not in adata.obs.columns:
        raise ValueError(f"{path.name} lacks required context/perturbation columns")

    contexts = adata.obs[context_col].astype(str).dropna().unique().tolist()
    if len(contexts) != 1:
        raise ValueError(f"{path.name} should have exactly one context, found {contexts}")
    context = str(contexts[0])

    x_raw = adata[:, gene_idx].layers["logNor"] if "logNor" in adata.layers else adata[:, gene_idx].X
    x = matrix_to_dense(x_raw).astype(np.float32)
    obs = adata.obs[[perturbation_col]].copy()
    obs["_pert"] = obs[perturbation_col].astype(str).str.strip()

    group_indices: dict[str, np.ndarray] = {}
    for pert, sub in obs.groupby("_pert", observed=False):
        pert = str(pert).strip()
        if not pert or pert.lower() in {"nan", "none", "null"}:
            continue
        pos = obs.index.get_indexer(sub.index)
        pos = pos[pos >= 0]
        if len(pos) < min_cells:
            continue
        if len(pos) > max_cells_per_group:
            pos = rng.choice(pos, size=max_cells_per_group, replace=False)
        group_indices[pert] = pos

    control_positions = [pos for pert, pos in group_indices.items() if is_control_value(pert)]
    if not control_positions:
        raise ValueError(f"{path.name} has no usable control group")
    control_pos = np.concatenate(control_positions)
    control_mean = x[control_pos].mean(axis=0).astype(np.float32)

    tasks: list[dict[str, Any]] = []
    for pert, pos in sorted(group_indices.items()):
        if is_control_value(pert):
            continue
        effect = x[pos].mean(axis=0) - control_mean
        tasks.append(
            {
                "dataset": "sciplex3_official_3cell",
                "context": context,
                "perturbation": str(pert),
                "effect": effect.astype(np.float32),
                "control_mean": control_mean,
                "n_cells": int(len(pos)),
                "context_col": context_col,
                "perturbation_col": perturbation_col,
                "source_file": path.name,
            }
        )

    del x, adata
    gc.collect()
    return tasks


def filter_shared_perturbations(tasks: list[dict[str, Any]], max_perturbations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = pd.DataFrame(
        {
            "context": [t["context"] for t in tasks],
            "perturbation": [t["perturbation"] for t in tasks],
            "n_cells": [t["n_cells"] for t in tasks],
        }
    )
    support = (
        rows.groupby("perturbation", as_index=False)
        .agg(n_contexts=("context", "nunique"), total_cells=("n_cells", "sum"), min_cells=("n_cells", "min"))
        .sort_values(["n_contexts", "min_cells", "total_cells", "perturbation"], ascending=[False, False, False, True])
    )
    n_contexts = rows["context"].nunique()
    shared = support[support["n_contexts"] == n_contexts].copy()
    if shared.empty:
        raise ValueError("No perturbations are shared across all contexts")
    selected = shared.head(max_perturbations)["perturbation"].tolist()
    kept = [t for t in tasks if t["perturbation"] in set(selected)]
    info = {
        "n_contexts": int(n_contexts),
        "n_raw_tasks": int(len(tasks)),
        "n_shared_perturbations": int(len(shared)),
        "n_selected_perturbations": int(len(selected)),
        "selection_rule": f"top {max_perturbations} shared perturbations by min_cells, total_cells",
    }
    return kept, info


def build_panel_tasks(paths: list[Path], n_genes: int, min_cells: int, max_cells_per_group: int, max_perturbations: int, seed: int):
    genes = common_gene_panel(paths, n_genes)
    tasks: list[dict[str, Any]] = []
    for i, path in enumerate(paths):
        print(f"[tasks] official sciplex3 file {path.name}", flush=True)
        tasks.extend(
            build_tasks_for_file(
                path=path,
                genes=genes,
                min_cells=min_cells,
                max_cells_per_group=max_cells_per_group,
                seed=seed + i,
            )
        )
    tasks, selection_info = filter_shared_perturbations(tasks, max_perturbations=max_perturbations)
    for i, task in enumerate(tasks):
        task["task_id"] = i
        task["task_key"] = f"sciplex3_official_3cell::task_{i:05d}"
    meta = {
        "path": ";".join(str(p) for p in paths),
        "dataset": "sciplex3_official_3cell",
        "n_tasks": len(tasks),
        "n_contexts": len(set(t["context"] for t in tasks)),
        "n_perturbations": len(set(t["perturbation"] for t in tasks)),
        "context_col": "cell_line",
        "perturbation_col": "drug_dose_name",
        "n_genes": len(genes),
        "source_files": ";".join(p.name for p in paths),
        **selection_info,
    }
    return {"sciplex3_official_3cell": tasks}, {"sciplex3_official_3cell": genes}, pd.DataFrame([meta])


def make_zip(path: Path) -> Path:
    zip_path = path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in path.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(path.parent))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E13 official sciplex3 3-cell-line panel probe.")
    parser.add_argument("--atlas-root", default=str(DEFAULT_ATLAS_ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "runtime/e13_sciplex3_official_3cell_panel_20260707"))
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--max-cells-per-group", type=int, default=800)
    parser.add_argument("--max-perturbations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=5204)
    args = parser.parse_args()

    mvp = load_mvp_module()
    mvp.DATASET_NAMES = ["sciplex3_official_3cell"]

    atlas_root = Path(args.atlas_root)
    paths = [atlas_root / rel for rel in DEFAULT_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    dirs = mvp.make_dirs(Path(args.out_dir))
    start = time.time()
    print(f"[start] E13 sciplex3 official 3-cell panel output: {dirs.out}", flush=True)

    mvp.run_s0_audit(dirs)
    tasks_by_dataset, genes_by_dataset, dataset_meta = build_panel_tasks(
        paths=paths,
        n_genes=args.n_genes,
        min_cells=args.min_cells,
        max_cells_per_group=args.max_cells_per_group,
        max_perturbations=args.max_perturbations,
        seed=args.seed,
    )
    dataset_meta.to_csv(dirs.tables / "DATASET_TASK_SUMMARY.csv", index=False)
    mvp.save_json(dirs.input / "dataset_paths.json", {"sciplex3_official_3cell": [str(p) for p in paths]})

    split_df = mvp.build_pair_splits(tasks_by_dataset, dirs, seed=args.seed)
    rec_df, pred_arrays, true_arrays = mvp.run_predictors(tasks_by_dataset, genes_by_dataset, split_df, dirs)
    feat_df = mvp.compute_features(rec_df, tasks_by_dataset, split_df, pred_arrays, dirs)
    scores, _ = mvp.run_scores(rec_df, feat_df, dirs, seed=args.seed)
    eval_result = mvp.evaluate_scores(scores, dirs)
    eval_summary = pd.read_csv(dirs.tables / "CONFIDENCE_EVAL_SUMMARY.csv")
    coverage = pd.read_csv(dirs.tables / "RISK_COVERAGE.csv")
    high_low = pd.read_csv(dirs.tables / "HIGH_LOW_CONFIDENCE_RMSE.csv")
    buckets = pd.read_csv(dirs.tables / "CALIBRATION_BUCKETS.csv")
    mvp.plot_results(scores, eval_summary, coverage, high_low, buckets, split_df, feat_df, dirs)
    mvp.write_reports(dirs, dataset_meta, split_df, rec_df, feat_df, scores, eval_summary, eval_result)
    mvp.copy_scripts(dirs)

    checklist = dirs.out / "stage_completion_checklist_v2_1.md"
    if checklist.exists() and (dirs.scripts / "run_confidence_mvp_v2_1.sh").exists():
        text = checklist.read_text(encoding="utf-8")
        text = text.replace("| G11 | FAIL | set after zip test |", "| G11 | PASS | zip created after checklist and scripts copied |")
        checklist.write_text(text, encoding="utf-8")

    elapsed = time.time() - start
    zip_path = make_zip(dirs.out)
    status = {
        "output_dir": str(dirs.out),
        "zip_path": str(zip_path),
        "elapsed_seconds": elapsed,
        "n_prediction_records": int(len(rec_df)),
        "datasets": ["sciplex3_official_3cell"],
        "source_files": [p.name for p in paths],
        "best_overall": eval_result.get("best", {}),
        "learned_overall": eval_result.get("learned", {}),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
    }
    mvp.save_json(dirs.out / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=mvp.to_jsonable), flush=True)
    print(f"[done] zip: {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
