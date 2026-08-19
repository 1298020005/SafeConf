#!/usr/bin/env python3
"""E107: context-aware GEARS on the E97 Frangieh outer folds.

GO edges are prior biological knowledge.  Coexpression edges are rebuilt in
each fold from control cells in source contexts only.  Training and model
selection use E97 train/validation tasks; test targets are read only after the
best validation checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad")
FULL_GEARS_DIR = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas")
CONTRACT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
E106_SCRIPT = ROOT / "tools/scripts/run_e106_frangieh_context_scgpt.py"
OUT = ROOT / "docs/实验结果/E107_frangieh_context_gears_20260713"
SEED = 202607107

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.loader import DataLoader


def import_e106() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e106_for_e107", E106_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import E106 task graph implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def edge_tensors(frame: pd.DataFrame, genes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    node = {gene: index for index, gene in enumerate(genes)}
    clean = frame[frame.source.isin(node) & frame.target.isin(node)].copy()
    clean = clean.drop_duplicates(["source", "target"], keep="first")
    existing = set(zip(clean.source.astype(str), clean.target.astype(str)))
    additions = [{"source": gene, "target": gene, "importance": 1.0} for gene in genes if (gene, gene) not in existing]
    if additions:
        clean = pd.concat([clean, pd.DataFrame(additions)], ignore_index=True)
    edge_index = torch.tensor([[node[str(a)], node[str(b)]] for a, b in zip(clean.source, clean.target)], dtype=torch.long).T
    edge_weight = torch.tensor(clean.importance.to_numpy(float), dtype=torch.float32)
    return edge_index, edge_weight


def build_go(genes: list[str], k: int) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    go = pd.read_csv(FULL_GEARS_DIR / "go.csv")
    go = go[go.source.isin(genes) & go.target.isin(genes)].copy()
    go = go.sort_values(["target", "importance"], ascending=[True, False]).groupby("target", as_index=False, group_keys=False).head(k + 1)
    edge_index, weight = edge_tensors(go, genes)
    return go, edge_index, weight


def build_coexpression(adata: Any, source_contexts: list[str], genes: list[str], k: int, threshold: float) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, int]:
    context = adata.obs["cell_type"].astype(str)
    condition = adata.obs["condition"].astype(str)
    mask = context.isin(source_contexts).to_numpy() & condition.eq("ctrl").to_numpy()
    matrix = adata[mask].X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    corr = np.corrcoef(np.asarray(matrix, dtype=np.float32), rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    for target_index, target in enumerate(genes):
        candidates = np.argsort(corr[:, target_index])[-(k + 1):][::-1]
        for source_index in candidates:
            value = float(corr[source_index, target_index])
            if value >= threshold or source_index == target_index:
                rows.append({"source": genes[int(source_index)], "target": target, "importance": value})
    frame = pd.DataFrame(rows)
    edge_index, weight = edge_tensors(frame, genes)
    return frame, edge_index, weight, int(mask.sum())


def mse_epoch(model: Any, loader: DataLoader, device: torch.device, optimizer: Any | None) -> float:
    model.train(optimizer is not None)
    losses = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            loss = torch.mean((prediction - batch.y) ** 2)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run_fold(fold: str, epochs: int, patience: int, batch_size: int, lr: float, device: torch.device) -> dict[str, Any]:
    from gears.model import GEARS_Model

    started = time.time()
    fold_dir = OUT / "folds" / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CONTRACT)
    tasks = manifest[manifest.fold_id.eq(fold)].copy()
    tasks = tasks[~tasks.split.eq("train") | tasks["in_train_fraction_100"].astype(bool)].copy()
    source_contexts = sorted(tasks.loc[tasks.split.eq("train"), "context"].astype(str).unique())
    heldout_contexts = sorted(set(tasks.context.astype(str)) - set(source_contexts))
    adata = sc.read_h5ad(SOURCE)
    genes = adata.var["gene_name"].astype(str).tolist()
    e106 = import_e106()
    graphs, provenance, control_means = e106.make_mean_graphs(adata, tasks, genes)
    provenance.to_csv(fold_dir / "TASK_INPUT_PROVENANCE.csv", index=False)
    split_graphs = {name: [graph for graph in graphs if graph.split == name] for name in ("train", "val", "test")}

    go_frame, go_index, go_weight = build_go(genes, 20)
    co_frame, co_index, co_weight, n_co_cells = build_coexpression(adata, source_contexts, genes, 10, 0.4)
    go_frame.to_csv(fold_dir / "GO_EDGES.csv", index=False)
    co_frame.to_csv(fold_dir / "TRAIN_ONLY_COEXPRESSION_EDGES.csv", index=False)
    config = {
        "hidden_size": 64,
        "num_go_gnn_layers": 1,
        "num_gene_gnn_layers": 1,
        "decoder_hidden_size": 16,
        "uncertainty": False,
        "G_go": go_index,
        "G_go_weight": go_weight,
        "G_coexpress": co_index,
        "G_coexpress_weight": co_weight,
        "device": str(device),
        "num_genes": len(genes),
    }
    model = GEARS_Model(config).to(device)
    train_loader = DataLoader(split_graphs["train"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(split_graphs["val"], batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = mse_epoch(model, train_loader, device, optimizer)
        val_loss = mse_epoch(model, val_loader, device, None)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        print(f"{fold} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no GEARS validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    pd.DataFrame(history).to_csv(fold_dir / "TRAINING_HISTORY.csv", index=False)

    predictions, truths, rows = {}, {}, []
    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            for batch in DataLoader(split_graphs[split_name], batch_size=batch_size, shuffle=False):
                batch = batch.to(device)
                output = model(batch)
                for task_id, context, condition, setting, pred_raw, true_raw in zip(
                    batch.pert, batch.context, batch.perturbation, batch.setting,
                    output.detach().cpu().numpy(), batch.y.detach().cpu().numpy(),
                ):
                    pred_effect = np.asarray(pred_raw, np.float32) - control_means[str(context)]
                    true_effect = np.asarray(true_raw, np.float32) - control_means[str(context)]
                    key = str(task_id)
                    array_key = f"{split_name}::{key}"
                    predictions[array_key], truths[array_key] = pred_effect, true_effect
                    rows.append({
                        "fold_id": fold,
                        "task_id": key,
                        "split": split_name,
                        "context": str(context),
                        "perturbation": str(condition),
                        "setting": str(setting),
                        "predictor_name": "GEARS_context_mean_trainonly_graphs",
                        "predicted_effect_key": array_key,
                        "true_effect_key": array_key,
                        "true_error_rmse": float(np.sqrt(np.mean((pred_effect - true_effect) ** 2))),
                        "predicted_effect_l2": float(np.linalg.norm(pred_effect)),
                        "true_effect_l2_diagnostic": float(np.linalg.norm(true_effect)),
                        "test_truth_used_for_training_or_selection": False,
                    })
    np.savez_compressed(fold_dir / "predicted_effects.npz", **predictions)
    np.savez_compressed(fold_dir / "true_effects.npz", **truths)
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(fold_dir / "ALL_TASK_METRICS.csv", index=False)
    all_metrics[all_metrics.split.eq("test")].to_csv(fold_dir / "TEST_TASK_METRICS.csv", index=False)
    status = {
        "fold_id": fold,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "seed": SEED,
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "n_train_tasks": len(split_graphs["train"]),
        "n_validation_tasks": len(split_graphs["val"]),
        "n_test_tasks": len(split_graphs["test"]),
        "source_contexts_for_coexpression": source_contexts,
        "heldout_contexts_excluded_from_coexpression": heldout_contexts,
        "n_source_control_cells_for_coexpression": n_co_cells,
        "n_go_edges_before_self_loop_completion": len(go_frame),
        "n_coexpression_edges_before_self_loop_completion": len(co_frame),
        "test_target_used_for_training_or_selection": False,
        "wall_seconds": time.time() - started,
    }
    (fold_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return status


def aggregate() -> dict[str, Any]:
    folds = sorted(pd.read_csv(CONTRACT).fold_id.unique())
    statuses, metrics, missing = [], [], []
    for fold in folds:
        directory = OUT / "folds" / fold
        if not (directory / "RUN_STATUS.json").exists():
            missing.append(fold)
            continue
        statuses.append(json.loads((directory / "RUN_STATUS.json").read_text()))
        metrics.append(pd.read_csv(directory / "TEST_TASK_METRICS.csv"))
    result = {
        "experiment": "E107_frangieh_context_gears",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete" if not missing else "partial",
        "completed_folds": [item["fold_id"] for item in statuses],
        "missing_folds": missing,
        "test_target_used_for_training_or_selection": False,
    }
    if metrics:
        frame = pd.concat(metrics, ignore_index=True)
        frame.to_csv(OUT / "E107_ALL_TEST_TASK_METRICS.csv", index=False)
        frame.groupby("setting", as_index=False).agg(n_tasks=("task_id", "size"), median_rmse=("true_error_rmse", "median"), mean_rmse=("true_error_rmse", "mean")).to_csv(OUT / "E107_SETTING_SUMMARY.csv", index=False)
        result["n_test_task_records"] = len(frame)
    (OUT / "RUN_STATUS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (OUT / "E107_REPORT.md").write_text(
        "# E107｜Frangieh 同背景输入 GEARS 正式实验\n\n"
        "三个外层 fold 各自重建共表达图，只读取源背景对照细胞；留出背景不参与该图。GO 图属于外部先验。"
        "任务输入为同背景 control mean 和扰动位点，验证集选择 checkpoint，测试标签在选择结束后才用于误差。"
        "逐 fold 图边、输入来源、训练曲线和结果位于 `folds/`。\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
        return
    folds = sorted(pd.read_csv(CONTRACT).fold_id.unique())
    selected = folds if args.fold == "all" else [folds[int(args.fold) - 1] if args.fold.isdigit() else args.fold]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for fold in selected:
        set_seed(SEED + folds.index(fold))
        print(json.dumps(run_fold(fold, args.epochs, args.patience, args.batch_size, args.lr, device), ensure_ascii=False, indent=2))
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
