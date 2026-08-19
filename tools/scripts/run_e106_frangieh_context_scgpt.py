#!/usr/bin/env python3
"""E106: context-aware scGPT on the frozen E97 Frangieh benchmark.

Each task is represented by the mean basal expression of its own context,
one perturbation flag, and the mean perturbed expression target.  The target
context's basal state is an allowed prediction input; held-out perturbation
expression is never used for fitting or model selection.  All task partitions
come directly from E97.
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
CONTRACT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E106_frangieh_context_scgpt_20260713"
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
SCGPT_REPO = Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/codex_scgpt_attnres_workspace/repo")
SEED = 202607106

sys.path.insert(0, str(SCGPT_REPO))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def import_e65() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e65_for_e106", E65_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the audited E65 scGPT implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dense_mean(matrix: Any) -> np.ndarray:
    if issparse(matrix):
        result = np.asarray(matrix.mean(axis=0)).reshape(-1)
    else:
        result = np.asarray(matrix, dtype=np.float32).mean(axis=0)
    return result.astype(np.float32)


def perturbation_gene(condition: str) -> str:
    genes = [part for part in str(condition).split("+") if part != "ctrl"]
    if len(genes) != 1:
        raise ValueError(condition)
    return genes[0]


def make_mean_graphs(adata: Any, tasks: pd.DataFrame, genes: list[str]) -> tuple[list[Data], pd.DataFrame, dict[str, np.ndarray]]:
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    contexts = adata.obs["cell_type"].astype(str)
    conditions = adata.obs["condition"].astype(str)
    control_means: dict[str, np.ndarray] = {}
    graphs: list[Data] = []
    rows: list[dict[str, Any]] = []
    for context in sorted(tasks["context"].astype(str).unique()):
        mask = contexts.eq(context).to_numpy() & conditions.eq("ctrl").to_numpy()
        if not mask.any():
            raise RuntimeError(f"no control cells for {context}")
        control_means[context] = dense_mean(adata[mask].X)
    for row in tasks.itertuples(index=False):
        context, condition = str(row.context), str(row.perturbation)
        mask = contexts.eq(context).to_numpy() & conditions.eq(condition).to_numpy()
        if not mask.any():
            raise RuntimeError(f"empty target task {context}::{condition}")
        target = dense_mean(adata[mask].X)
        flag = np.zeros(len(genes), dtype=np.float32)
        gene = perturbation_gene(condition)
        flag[gene_to_index[gene]] = 1.0
        task_id = f"{context}::{condition}"
        graphs.append(
            Data(
                x=torch.from_numpy(np.stack([control_means[context], flag], axis=1)),
                y=torch.from_numpy(target).unsqueeze(0),
                pert=task_id,
                context=context,
                perturbation=condition,
                split=str(row.split),
                setting=str(row.setting),
            )
        )
        rows.append({
            "task_id": task_id,
            "split": row.split,
            "setting": row.setting,
            "context": context,
            "perturbation": condition,
            "n_target_cells": int(mask.sum()),
            "n_control_cells": int((contexts.eq(context) & conditions.eq("ctrl")).sum()),
            "basal_input_definition": "same_context_control_mean",
            "target_used_for_fit": str(row.split) == "train",
            "target_used_for_model_selection": str(row.split) == "val",
        })
    return graphs, pd.DataFrame(rows), control_means


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(fold: str, epochs: int, patience: int, batch_size: int, lr: float, device: torch.device) -> dict[str, Any]:
    started = time.time()
    fold_dir = OUT / "folds" / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CONTRACT)
    tasks = manifest[manifest.fold_id.eq(fold)].copy()
    tasks = tasks[~tasks.split.eq("train") | tasks["in_train_fraction_100"].astype(bool)].copy()
    adata = sc.read_h5ad(SOURCE)
    genes = adata.var["gene_name"].astype(str).tolist()
    graphs, provenance, control_means = make_mean_graphs(adata, tasks, genes)
    provenance.to_csv(fold_dir / "TASK_INPUT_PROVENANCE.csv", index=False)
    split_graphs = {name: [graph for graph in graphs if graph.split == name] for name in ("train", "val", "test")}
    if any(set(g.pert for g in split_graphs[a]) & set(g.pert for g in split_graphs[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("task split overlap")

    e65 = import_e65()
    model, _, metadata = e65.load_model(device)
    gene_ids = e65.make_gene_ids(genes, metadata["vocab"])
    train_loader = DataLoader(split_graphs["train"], batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(split_graphs["val"], batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = e65.train_one_epoch(model, train_loader, gene_ids, optimizer, scaler, device, device.type == "cuda")
        val_loss = e65.evaluate_mse(model, val_loader, gene_ids, device, device.type == "cuda")
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
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    pd.DataFrame(history).to_csv(fold_dir / "TRAINING_HISTORY.csv", index=False)

    predictions: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            loader = DataLoader(split_graphs[split_name], batch_size=batch_size, shuffle=False)
            for batch in loader:
                output, target = e65.model_forward(model, batch, gene_ids, device, device.type == "cuda", False)
                for task_id, context, condition, setting, pred_raw, true_raw in zip(
                    batch.pert, batch.context, batch.perturbation, batch.setting,
                    output.detach().cpu().numpy(), target.detach().cpu().numpy(),
                ):
                    pred_effect = np.asarray(pred_raw, np.float32) - control_means[str(context)]
                    true_effect = np.asarray(true_raw, np.float32) - control_means[str(context)]
                    key = str(task_id)
                    array_key = f"{split_name}::{key}"
                    predictions[array_key] = pred_effect
                    truths[array_key] = true_effect
                    rows.append({
                        "fold_id": fold,
                        "task_id": key,
                        "split": split_name,
                        "context": str(context),
                        "perturbation": str(condition),
                        "setting": str(setting),
                        "predictor_name": "scGPT_context_mean_finetuned",
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
        "batch_size": batch_size,
        "learning_rate": lr,
        "n_train_tasks": len(split_graphs["train"]),
        "n_validation_tasks": len(split_graphs["val"]),
        "n_test_tasks": len(split_graphs["test"]),
        "pretrained_tensors_loaded": metadata["matched_pretrained_parameter_tensors"],
        "basal_input": "same-context control mean; available at prediction time",
        "test_target_used_for_training_or_selection": False,
        "wall_seconds": time.time() - started,
    }
    (fold_dir / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return status


def aggregate() -> dict[str, Any]:
    manifest = pd.read_csv(CONTRACT)
    folds = sorted(manifest.fold_id.unique())
    statuses = []
    metrics = []
    missing = []
    for fold in folds:
        fold_dir = OUT / "folds" / fold
        if not (fold_dir / "RUN_STATUS.json").exists():
            missing.append(fold)
            continue
        statuses.append(json.loads((fold_dir / "RUN_STATUS.json").read_text()))
        metrics.append(pd.read_csv(fold_dir / "TEST_TASK_METRICS.csv"))
    result = {
        "experiment": "E106_frangieh_context_scgpt",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete" if not missing else "partial",
        "completed_folds": [item["fold_id"] for item in statuses],
        "missing_folds": missing,
        "n_completed_folds": len(statuses),
        "test_target_used_for_training_or_selection": False,
    }
    if metrics:
        frame = pd.concat(metrics, ignore_index=True)
        frame.to_csv(OUT / "E106_ALL_TEST_TASK_METRICS.csv", index=False)
        summary = frame.groupby("setting", as_index=False).agg(n_tasks=("task_id", "size"), median_rmse=("true_error_rmse", "median"), mean_rmse=("true_error_rmse", "mean"))
        summary.to_csv(OUT / "E106_SETTING_SUMMARY.csv", index=False)
        result["n_test_task_records"] = len(frame)
    (OUT / "RUN_STATUS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = """# E106｜Frangieh 同背景输入 scGPT 正式实验

E106 按 E97 的三个外层背景留出 fold 独立微调 scGPT。每个任务的输入是该背景未扰动细胞的平均表达和扰动基因标记；测试扰动后的表达只在训练与模型选择结束后用于计算误差。新背景的基础状态属于预测问题给定的初始状态，不是测试答案。

训练使用 whole-human scGPT 预训练参数，验证集选择 epoch。逐 fold 输入来源、训练曲线、预测 effect 和测试误差位于 `folds/`。本目录完整后才能与同合同 GEARS 组成正式双预测器分歧。
"""
    (OUT / "E106_REPORT.md").write_text(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="all", help="fold id, 1-based fold index, or all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
        return
    folds = sorted(pd.read_csv(CONTRACT)["fold_id"].unique())
    selected = folds if args.fold == "all" else [folds[int(args.fold) - 1] if args.fold.isdigit() else args.fold]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for fold in selected:
        set_seed(SEED + folds.index(fold))
        print(json.dumps(run_fold(fold, args.epochs, args.patience, args.batch_size, args.lr, device), ensure_ascii=False, indent=2))
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
