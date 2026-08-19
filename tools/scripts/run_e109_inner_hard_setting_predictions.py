#!/usr/bin/env python3
"""E109: nested row/column/double simulations for risk calibration.

For each E97 outer fold, each of its two source contexts is held out once in
an inner fold.  Twenty perturbations shared by both source contexts are also
held out as columns.  The resulting predictions calibrate structural risk
without reading any outer-test target.
"""

from __future__ import annotations

import argparse
import hashlib
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
E106_SCRIPT = ROOT / "tools/scripts/run_e106_frangieh_context_scgpt.py"
E107_SCRIPT = ROOT / "tools/scripts/run_e107_frangieh_context_gears.py"
OUT = ROOT / "docs/实验结果/E109_inner_hard_setting_predictions_20260713"
SEED = 202607109
N_COLUMN = 20
N_MODEL_VAL = 10

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import rankdata
from torch_geometric.loader import DataLoader


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hash_rank(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def freeze_inner_contract(outer_fold: str, outer: pd.DataFrame) -> pd.DataFrame:
    train = outer[outer.split.eq("train")].copy()
    contexts = sorted(train.context.astype(str).unique())
    matrix = train.assign(present=1).pivot(index="perturbation", columns="context", values="present").fillna(0)
    shared = sorted(matrix.index[matrix[contexts].sum(axis=1).eq(len(contexts))].astype(str))
    rows = []
    for heldout in contexts:
        source = next(value for value in contexts if value != heldout)
        inner_id = f"{outer_fold}__inner_holdout_{heldout.replace(' ', '_')}"
        ordered = sorted(shared, key=lambda gene: hash_rank("E109", inner_id, gene))
        columns = set(ordered[:N_COLUMN])
        validation = set(ordered[N_COLUMN:N_COLUMN + N_MODEL_VAL])
        fitted = set(ordered[N_COLUMN + N_MODEL_VAL:])
        for context in contexts:
            for perturbation in shared:
                if context == source and perturbation in fitted:
                    split, setting = "train", "inner_source_train"
                elif context == source and perturbation in validation:
                    split, setting = "val", "inner_model_validation_column"
                elif context == source and perturbation in columns:
                    split, setting = "test", "perturbation_unseen_column"
                elif context == heldout and perturbation in columns:
                    split, setting = "test", "context_and_perturbation_unseen"
                elif context == heldout and perturbation in fitted:
                    split, setting = "test", "context_unseen_row"
                else:
                    # Heldout-context validation genes are not used: their
                    # targets would turn model-validation genes into risk data.
                    continue
                rows.append({
                    "outer_fold_id": outer_fold,
                    "inner_fold_id": inner_id,
                    "inner_source_context": source,
                    "inner_heldout_context": heldout,
                    "context": context,
                    "perturbation": perturbation,
                    "split": split,
                    "setting": setting,
                    "selected_without_expression": True,
                })
    return pd.DataFrame(rows)


def fit_scgpt(graphs: dict[str, list[Any]], genes: list[str], device: torch.device, e106: Any) -> tuple[Any, np.ndarray, Any, list[dict[str, float]]]:
    e65 = e106.import_e65()
    model, _, metadata = e65.load_model(device)
    gene_ids = e65.make_gene_ids(genes, metadata["vocab"])
    train = DataLoader(graphs["train"], batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val = DataLoader(graphs["val"], batch_size=16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, 11):
        train_loss = e65.train_one_epoch(model, train, gene_ids, optimizer, scaler, device, device.type == "cuda")
        val_loss = e65.evaluate_mse(model, val, gene_ids, device, device.type == "cuda")
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best - 1e-7:
            best, stale = val_loss, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3: break
    model.load_state_dict(state); model.to(device).eval()
    return model, gene_ids, e65, history


def fit_gears(graphs: dict[str, list[Any]], genes: list[str], adata: Any, source: str, device: torch.device, e107: Any) -> tuple[Any, list[dict[str, float]], dict[str, int]]:
    from gears.model import GEARS_Model
    go, go_i, go_w = e107.build_go(genes, 20)
    co, co_i, co_w, n_cells = e107.build_coexpression(adata, [source], genes, 10, .4)
    config = {"hidden_size":64,"num_go_gnn_layers":1,"num_gene_gnn_layers":1,"decoder_hidden_size":16,"uncertainty":False,
              "G_go":go_i,"G_go_weight":go_w,"G_coexpress":co_i,"G_coexpress_weight":co_w,"device":str(device),"num_genes":len(genes)}
    model = GEARS_Model(config).to(device)
    train = DataLoader(graphs["train"], batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val = DataLoader(graphs["val"], batch_size=16, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, 51):
        train_loss = e107.mse_epoch(model, train, device, optimizer)
        val_loss = e107.mse_epoch(model, val, device, None)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best - 1e-7:
            best, stale = val_loss, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6: break
    model.load_state_dict(state); model.to(device).eval()
    return model, history, {"n_go_edges":len(go),"n_coexpression_edges":len(co),"n_source_control_cells":n_cells}


def predict_scgpt(model: Any, graphs: list[Any], gene_ids: np.ndarray, e65: Any, device: torch.device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False):
            pred, truth = e65.model_forward(model, batch, gene_ids, device, device.type == "cuda", False)
            basal = batch.x[:, 0].reshape(len(batch.pert), -1).detach().cpu().numpy()
            for task, p, t, b in zip(batch.pert, pred.detach().cpu().numpy(), truth.detach().cpu().numpy(), basal):
                result[str(task)] = (np.asarray(p-b,np.float32), np.asarray(t-b,np.float32))
    return result


def predict_gears(model: Any, graphs: list[Any], device: torch.device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16, shuffle=False):
            batch = batch.to(device); pred = model(batch)
            basal = batch.x[:, 0].reshape(len(batch.pert), -1)
            for task, p, t, b in zip(batch.pert, pred.detach().cpu().numpy(), batch.y.detach().cpu().numpy(), basal.detach().cpu().numpy()):
                result[str(task)] = (np.asarray(p-b,np.float32), np.asarray(t-b,np.float32))
    return result


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(b,float))**2)))


def robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center=float(np.median(reference)); mad=float(np.median(np.abs(reference-center))); scale=max(1.4826*mad,float(np.std(reference)),1e-8)
    return np.clip((np.asarray(values)-center)/scale,-5,5)


def run_outer(outer_fold: str, device: torch.device) -> dict[str, Any]:
    start=time.time(); directory=OUT/"outer_folds"/outer_fold; directory.mkdir(parents=True,exist_ok=True)
    all_manifest=pd.read_csv(CONTRACT); outer=all_manifest[all_manifest.fold_id.eq(outer_fold)].copy()
    contract=freeze_inner_contract(outer_fold,outer); contract.to_csv(directory/"INNER_CONTRACT.csv",index=False)
    e106=load_module(f"e106_{hash_rank(outer_fold)[:8]}",E106_SCRIPT); e107=load_module(f"e107_{hash_rank(outer_fold)[:8]}",E107_SCRIPT)
    adata=sc.read_h5ad(SOURCE); genes=adata.var["gene_name"].astype(str).tolist(); outputs=[]; histories=[]
    for inner_index,(inner_id, tasks) in enumerate(contract.groupby("inner_fold_id",sort=True)):
        set_seed(SEED+inner_index)
        graphs_list,_,_=e106.make_mean_graphs(adata,tasks,genes)
        graphs={split:[g for g in graphs_list if g.split==split] for split in ("train","val","test")}
        source=str(tasks.inner_source_context.iloc[0])
        sc_model,gene_ids,e65,sc_history=fit_scgpt(graphs,genes,device,e106)
        ge_model,ge_history,graph_meta=fit_gears(graphs,genes,adata,source,device,e107)
        query=graphs["val"]+graphs["test"]
        sc_pred=predict_scgpt(sc_model,query,gene_ids,e65,device); ge_pred=predict_gears(ge_model,query,device)
        query_frame=tasks[tasks.split.isin(["val","test"])].copy()
        for row in query_frame.itertuples(index=False):
            key=f"{row.context}::{row.perturbation}"; sp,truth=sc_pred[key]; gp,gtruth=ge_pred[key]
            if not np.allclose(truth,gtruth,atol=1e-6): raise RuntimeError("inner truth mismatch")
            ensemble=(sp+gp)/2
            outputs.append({
                "outer_fold_id":outer_fold,"inner_fold_id":inner_id,"split":row.split,"setting":row.setting,
                "context":row.context,"perturbation":row.perturbation,
                "risk_model_disagreement":rmse(sp,gp),"baseline_predicted_magnitude":float(np.sqrt(np.mean(ensemble**2))),
                "error_two_predictor_mean_rmse":float(np.mean([rmse(sp,truth),rmse(gp,truth)])),
                "context_novelty_scaled":0.0 if row.context==source else 1.0,
                "training_support_count":1 if row.perturbation in set(tasks.loc[tasks.split.eq('train'),'perturbation']) else 0,
                "perturbation_novelty":0.5 if row.perturbation in set(tasks.loc[tasks.split.eq('train'),'perturbation']) else 1.0,
                "outer_test_truth_used":False,
            })
        histories += [{"inner_fold_id":inner_id,"model":"scGPT",**item} for item in sc_history]
        histories += [{"inner_fold_id":inner_id,"model":"GEARS",**item} for item in ge_history]
        (directory/f"{hash_rank(inner_id)[:12]}_GRAPH_META.json").write_text(json.dumps({"inner_fold_id":inner_id,**graph_meta},ensure_ascii=False,indent=2)+"\n")
        del sc_model,ge_model; torch.cuda.empty_cache()
    scored=pd.DataFrame(outputs)
    pieces=[]
    for inner_id,group in scored.groupby("inner_fold_id",sort=True):
        reference=group.split.eq("val"); group=group.copy()
        group["risk_disagreement_z"]=robust_z(group.risk_model_disagreement,group.loc[reference,"risk_model_disagreement"].to_numpy(float))
        group["predicted_magnitude_z"]=robust_z(group.baseline_predicted_magnitude,group.loc[reference,"baseline_predicted_magnitude"].to_numpy(float))
        pieces.append(group)
    scored=pd.concat(pieces,ignore_index=True)
    scored.to_csv(directory/"INNER_CALIBRATION_ROWS.csv",index=False); pd.DataFrame(histories).to_csv(directory/"TRAINING_HISTORY.csv",index=False)
    status={"outer_fold_id":outer_fold,"status":"complete","generated_at":datetime.now().isoformat(timespec="seconds"),
            "n_inner_folds":int(scored.inner_fold_id.nunique()),"n_calibration_test_rows":int(scored.split.eq('test').sum()),
            "settings":sorted(scored.loc[scored.split.eq('test'),'setting'].unique()),"outer_test_truth_used":False,"wall_seconds":time.time()-start}
    (directory/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n"); return status


def aggregate() -> dict[str, Any]:
    folds=sorted(pd.read_csv(CONTRACT).fold_id.unique()); frames=[]; missing=[]
    for fold in folds:
        path=OUT/"outer_folds"/fold/"INNER_CALIBRATION_ROWS.csv"
        if path.exists(): frames.append(pd.read_csv(path))
        else: missing.append(fold)
    status={"experiment":"E109_inner_hard_setting_predictions","generated_at":datetime.now().isoformat(timespec="seconds"),
            "status":"complete" if not missing else "partial","missing_outer_folds":missing,"outer_test_truth_used":False}
    if frames:
        all_rows=pd.concat(frames,ignore_index=True); all_rows.to_csv(OUT/"E109_ALL_INNER_ROWS.csv",index=False)
        summary=all_rows[all_rows.split.eq('test')].groupby(["setting"],as_index=False).agg(n_rows=("perturbation","size"),mean_error=("error_two_predictor_mean_rmse","mean"))
        summary.to_csv(OUT/"E109_SETTING_SUMMARY.csv",index=False); status["n_calibration_test_rows"]=len(all_rows[all_rows.split.eq('test')])
    (OUT/"RUN_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n")
    (OUT/"E109_REPORT.md").write_text("# E109｜内层困难设置校准样本\n\n每个外层 fold 仅用其两个源背景再做两次内层行留出，并同时冻结 20 个整列新扰动。正式 scGPT 与 GEARS 在内层训练任务上重新拟合，得到新背景、新扰动和双未见的校准误差。外层测试标签未被读取。\n")
    return status


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--outer",default="all"); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--aggregate-only",action="store_true"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.aggregate_only: print(json.dumps(aggregate(),ensure_ascii=False,indent=2)); return
    folds=sorted(pd.read_csv(CONTRACT).fold_id.unique()); selected=folds if args.outer=="all" else [folds[int(args.outer)-1] if args.outer.isdigit() else args.outer]
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    for fold in selected: print(json.dumps(run_outer(fold,device),ensure_ascii=False,indent=2))
    print(json.dumps(aggregate(),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
