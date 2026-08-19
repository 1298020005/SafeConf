#!/usr/bin/env python3
"""E105: context-safe graph construction and real model-interface smoke.

The legacy GEARS ``PertData`` implementation samples basal cells from one
global control pool.  That is invalid for a context x perturbation benchmark:
the basal cell can come from a different context than the target cell.  This
adapter constructs every graph from a control cell in the *same* context and
keeps E97 train/validation/test task identities disjoint.

This experiment is a contract and interface smoke, not a performance result.
It executes one real scGPT pretrained-model optimization step and one GEARS
architecture optimization step on training graphs only, then inference on
held-out graphs.  Formal fitting is reserved for E106+.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/"
    "frangieh_e72_fixed512/perturb_processed.h5ad"
)
CONTRACT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E105_context_safe_graph_adapter_20260713"
SCGPT_REPO = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/repo"
)
E65_SCRIPT = ROOT / "tools/scripts/run_e65_scgpt_formal_fixed_panel.py"
SEED = 202607105

sys.path.insert(0, str(SCGPT_REPO))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def stable_index(size: int, *parts: object) -> int:
    if size < 1:
        raise ValueError(f"cannot sample an empty pool: {parts}")
    value = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "little") % size


def dense_row(matrix: Any, index: int) -> np.ndarray:
    row = matrix[index]
    if issparse(row):
        row = row.toarray()
    return np.asarray(row, dtype=np.float32).reshape(-1)


def pert_gene(condition: str) -> str:
    genes = [part for part in str(condition).split("+") if part != "ctrl"]
    if len(genes) != 1:
        raise ValueError(f"E105 expects one-gene perturbations, got {condition!r}")
    return genes[0]


def load_e65_module() -> Any:
    spec = importlib.util.spec_from_file_location("safeconf_e65", E65_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the audited E65 scGPT adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_graphs(
    adata: Any,
    tasks: pd.DataFrame,
    gene_to_index: dict[str, int],
) -> tuple[list[Data], pd.DataFrame]:
    obs = adata.obs.reset_index(drop=False).rename(columns={adata.obs.index.name or "index": "cell_id"})
    pools: dict[tuple[str, str], np.ndarray] = {}
    for context in sorted(obs["cell_type"].astype(str).unique()):
        pools[(context, "ctrl")] = np.flatnonzero(
            obs["cell_type"].astype(str).eq(context).to_numpy()
            & obs["condition"].astype(str).eq("ctrl").to_numpy()
        )
    for row in tasks.itertuples(index=False):
        key = (str(row.context), str(row.perturbation))
        if key not in pools:
            pools[key] = np.flatnonzero(
                obs["cell_type"].astype(str).eq(key[0]).to_numpy()
                & obs["condition"].astype(str).eq(key[1]).to_numpy()
            )

    graphs: list[Data] = []
    audit: list[dict[str, Any]] = []
    n_genes = adata.n_vars
    for row in tasks.itertuples(index=False):
        context, condition = str(row.context), str(row.perturbation)
        task_id = f"{context}::{condition}"
        target_pool = pools[(context, condition)]
        control_pool = pools[(context, "ctrl")]
        target_index = int(target_pool[stable_index(len(target_pool), SEED, row.fold_id, task_id, "target")])
        control_index = int(control_pool[stable_index(len(control_pool), SEED, row.fold_id, task_id, "control")])
        basal = dense_row(adata.X, control_index)
        target = dense_row(adata.X, target_index)
        gene = pert_gene(condition)
        if gene not in gene_to_index:
            raise RuntimeError(f"perturbation gene {gene} is absent from the fixed panel")
        flag = np.zeros(n_genes, dtype=np.float32)
        flag[gene_to_index[gene]] = 1.0
        graph = Data(
            x=torch.from_numpy(np.stack([basal, flag], axis=1)),
            # GEARS/scGPT loaders expect one graph target as [1, n_genes]; a
            # one-dimensional target would make ``len(batch.y)`` count genes.
            y=torch.from_numpy(target).unsqueeze(0),
            pert=task_id,
            context=context,
            perturbation=condition,
            split=str(row.split),
            setting=str(row.setting),
            basal_context=context,
        )
        graphs.append(graph)
        audit.append(
            {
                "fold_id": row.fold_id,
                "task_id": task_id,
                "split": row.split,
                "setting": row.setting,
                "context": context,
                "basal_context": context,
                "perturbation": condition,
                "perturbation_gene": gene,
                "perturbation_gene_index": gene_to_index[gene],
                "perturbation_flag_sum": float(flag.sum()),
                "control_pool_size_same_context": len(control_pool),
                "target_pool_size": len(target_pool),
                "control_row_index": control_index,
                "target_row_index": target_index,
            }
        )
    return graphs, pd.DataFrame(audit)


def assert_contract(graphs: list[Data], audit: pd.DataFrame, tasks: pd.DataFrame) -> None:
    if len(graphs) != len(tasks) or len(audit) != len(tasks):
        raise AssertionError("one graph per frozen task was not produced")
    if audit["task_id"].duplicated().any():
        raise AssertionError("duplicate context-task identity")
    if not audit["context"].eq(audit["basal_context"]).all():
        raise AssertionError("cross-context basal control leakage")
    if not np.allclose(audit["perturbation_flag_sum"], 1.0):
        raise AssertionError("one-hot perturbation flag contract failed")
    task_sets = {split: set(audit.loc[audit.split.eq(split), "task_id"]) for split in ("train", "val", "test")}
    if task_sets["train"] & task_sets["val"] or task_sets["train"] & task_sets["test"] or task_sets["val"] & task_sets["test"]:
        raise AssertionError("train/validation/test task identities overlap")
    if any(graph.x.shape != (512, 2) or graph.y.shape != (1, 512) for graph in graphs):
        raise AssertionError("unexpected graph shape")


def take(graphs: list[Data], split: str, n: int, settings: bool = False) -> list[Data]:
    candidates = [graph for graph in graphs if graph.split == split]
    if not settings:
        return candidates[:n]
    selected: list[Data] = []
    for setting in sorted({str(graph.setting) for graph in candidates}):
        selected.extend([graph for graph in candidates if graph.setting == setting][:n])
    return selected


def run_scgpt_smoke(graphs: list[Data], genes: list[str], device: torch.device) -> dict[str, Any]:
    e65 = load_e65_module()
    model, _, metadata = e65.load_model(device)
    gene_ids = e65.make_gene_ids(genes, metadata["vocab"])
    train_graphs = take(graphs, "train", 4)
    test_graphs = take(graphs, "test", 1, settings=True)
    loader = DataLoader(train_graphs, batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss = e65.train_one_epoch(model, loader, gene_ids, optimizer, scaler, device, device.type == "cuda")
    nonzero_gradient_tensors = sum(
        int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item()))
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    test_loader = DataLoader(test_graphs, batch_size=2, shuffle=False)
    with torch.no_grad():
        prediction, target = e65.model_forward(model, next(iter(test_loader)), gene_ids, device, device.type == "cuda", False)
    return {
        "model": "scGPT TransformerGenerator",
        "checkpoint": metadata["checkpoint"],
        "pretrained_tensors_loaded": metadata["matched_pretrained_parameter_tensors"],
        "n_train_graphs_used": len(train_graphs),
        "n_test_graphs_inference_only": len(test_graphs),
        "train_loss_one_step": loss,
        "optimizer_steps": len(loader),
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "test_prediction_shape": list(prediction.shape),
        "test_target_shape": list(target.shape),
    }


def run_gears_smoke(graphs: list[Data], device: torch.device) -> dict[str, Any]:
    from gears.model import GEARS_Model

    n_genes = int(graphs[0].y.numel())
    nodes = torch.arange(n_genes, dtype=torch.long)
    self_loops = torch.stack([nodes, nodes])
    weights = torch.ones(n_genes, dtype=torch.float32)
    config = {
        "hidden_size": 16,
        "num_go_gnn_layers": 1,
        "num_gene_gnn_layers": 1,
        "decoder_hidden_size": 8,
        "uncertainty": False,
        "G_go": self_loops,
        "G_go_weight": weights,
        "G_coexpress": self_loops,
        "G_coexpress_weight": weights,
        "device": str(device),
        "num_genes": n_genes,
    }
    model = GEARS_Model(config).to(device)
    train_graphs = take(graphs, "train", 4)
    test_graphs = take(graphs, "test", 1, settings=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = next(iter(DataLoader(train_graphs, batch_size=2, shuffle=False))).to(device)
    model.train()
    prediction = model(batch)
    loss = torch.mean((prediction - batch.y) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.eval()
    test_batch = next(iter(DataLoader(test_graphs, batch_size=2, shuffle=False))).to(device)
    with torch.no_grad():
        test_prediction = model(test_batch)
    return {
        "model": "GEARS_Model architecture",
        "graph_network": "self-loop smoke only; formal GO/coexpression graphs are not claimed",
        "n_train_graphs_used": len(train_graphs),
        "n_test_graphs_inference_only": len(test_graphs),
        "train_loss_one_step": float(loss.detach().cpu()),
        "test_prediction_shape": list(test_prediction.shape),
        "test_target_shape": list(test_batch.y.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "smoke"), default="smoke")
    parser.add_argument("--fold", default=None)
    args = parser.parse_args()
    for directory in (OUT / "tables", OUT / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    manifest = pd.read_csv(CONTRACT)
    fold = args.fold or sorted(manifest["fold_id"].unique())[0]
    tasks = manifest[manifest.fold_id.eq(fold)].copy()
    # The smoke uses the complete 100% E97 training partition for the contract.
    tasks = tasks[~tasks.split.eq("train") | tasks["in_train_fraction_100"].astype(bool)].copy()
    adata = sc.read_h5ad(SOURCE)
    genes = adata.var["gene_name"].astype(str).tolist()
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    graphs, audit = build_graphs(adata, tasks, gene_to_index)
    assert_contract(graphs, audit, tasks)
    audit.to_csv(OUT / "tables/E105_GRAPH_PROVENANCE.csv", index=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    status: dict[str, Any] = {
        "experiment": "E105_context_safe_graph_adapter",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "fold_id": fold,
        "source_h5ad": str(SOURCE),
        "n_genes": len(genes),
        "n_graphs": len(graphs),
        "split_counts": audit["split"].value_counts().sort_index().to_dict(),
        "setting_counts": audit["setting"].value_counts().sort_index().to_dict(),
        "all_basal_controls_match_target_context": bool(audit.context.eq(audit.basal_context).all()),
        "train_validation_test_task_ids_disjoint": True,
        "test_targets_used_for_optimization": False,
        "device": str(device),
    }
    if args.mode == "smoke":
        status["scgpt_smoke"] = run_scgpt_smoke(graphs, genes, device)
        status["gears_smoke"] = run_gears_smoke(graphs, device)
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    report = f"""# E105｜同背景对照图构建合同

E105 修复的是数据入口。Frangieh 每个任务都写成“细胞背景::扰动”唯一标识；基础表达只从该背景的 `ctrl` 细胞中确定性抽取，目标表达来自同一背景的扰动细胞。{len(graphs)} 个冻结任务逐一通过检查，训练、验证和测试任务没有交集。

旧版 `PertData.create_cell_graph_dataset` 从全局 `ctrl_adata` 抽样，可能把 Control、IFNγ 和 Co-culture 混用。E105 不调用该路径。

## 当前检查

- 同背景对照：`{status['all_basal_controls_match_target_context']}`
- 任务拆分互斥：`{status['train_validation_test_task_ids_disjoint']}`
- 测试目标参与优化：`{status['test_targets_used_for_optimization']}`
- 图数量：`{len(graphs)}`；基因数：`{len(genes)}`
- 拆分：`{status['split_counts']}`

## 模型接口

`smoke` 模式会执行 scGPT whole-human 预训练权重加载、一个真实反向传播步骤和测试只读推理；GEARS 也执行一次前向/反向与测试只读推理。GEARS 在本实验只用 self-loop 图验证架构接口，不能作为正式 GEARS 性能。正式实验必须按每个外层 fold 的训练背景重新构造 GO/共表达图。

详细的逐任务控制池大小、目标池大小、行索引和扰动位点见 `tables/E105_GRAPH_PROVENANCE.csv`。
"""
    (OUT / "reports/E105_CONTEXT_GRAPH_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E105 先看这个\n\n先读 `reports/E105_CONTEXT_GRAPH_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
