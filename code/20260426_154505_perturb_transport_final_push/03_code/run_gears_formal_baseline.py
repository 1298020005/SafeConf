from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch


LOCAL_ATLAS_FILES = {
    "norman": Path("/home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Norman.h5ad"),
    "adamson": Path("/home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad"),
    "dixit": Path("/home/yyf/datasets/singlecell_perturbation_atlas/official_scperturb/DixitRegev2016_K562_TFs_7_days.h5ad"),
}


def _load_local_adata(data_name: str, max_genes: int = 6000):
    import scanpy as sc
    import numpy as np
    import scipy.sparse as sp

    path = LOCAL_ATLAS_FILES[data_name]
    if not path.exists():
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    if "perturbation" not in adata.obs.columns:
        raise ValueError(f"local atlas file missing perturbation column: {path}")

    def _is_ctrl_like(values):
        return (
            values.str.lower().isin({"control", "ctrl", "non-targeting", "nt", "mock", "vehicle", "dmso"})
            | values.str.contains("control", case=False, na=False)
            | values.str.contains("ctrl", case=False, na=False)
            | values.eq("nan")
        )

    gene_name_set = set(adata.var_names.astype(str))
    pert = adata.obs["perturbation"].astype(str).fillna("control")
    source = pert
    if "target" in adata.obs.columns:
        target = adata.obs["target"].astype(str).fillna("control")
        target_matches = int(target.isin(gene_name_set).sum())
        perturbation_matches = int(pert.isin(gene_name_set).sum())
        if target_matches > perturbation_matches:
            keep = _is_ctrl_like(target) | target.isin(gene_name_set)
            if not bool(keep.all()):
                adata = adata[keep.values].copy()
                target = target.loc[adata.obs_names]
            source = target

    if max_genes and adata.n_vars > max_genes:
        gene_name_set = set(adata.var_names.astype(str))
        required_genes = set()
        for condition in source.astype(str).unique():
            for gene in str(condition).split("+"):
                if gene not in {"ctrl", "nan"} and gene in gene_name_set:
                    required_genes.add(gene)
        if "ncounts" in adata.var.columns:
            rank_values = pd.to_numeric(adata.var["ncounts"], errors="coerce").fillna(0)
        elif "ncells" in adata.var.columns:
            rank_values = pd.to_numeric(adata.var["ncells"], errors="coerce").fillna(0)
        else:
            rank_values = pd.Series(range(adata.n_vars, 0, -1), index=adata.var_names)
        ranked_genes = rank_values.sort_values(ascending=False).index.astype(str).tolist()
        keep_genes = set(required_genes)
        for gene in ranked_genes:
            if len(keep_genes) >= max_genes:
                break
            keep_genes.add(gene)
        adata = adata[:, adata.var_names.astype(str).isin(keep_genes)].copy()

    pert = source.astype(str).fillna("control")
    ctrl_like = (
        _is_ctrl_like(pert)
    )
    single_like = (~ctrl_like) & (~pert.str.contains(r"\+", regex=True, na=False))
    pert = pert.copy()
    pert.loc[single_like] = pert.loc[single_like] + "+ctrl"
    adata.obs = adata.obs.copy()
    adata.obs["condition"] = np.where(ctrl_like, "ctrl", pert)
    if "celltype" in adata.obs.columns:
        cell_type = adata.obs["celltype"].astype(str).fillna("cell")
    elif "cell_line" in adata.obs.columns:
        cell_type = adata.obs["cell_line"].astype(str).fillna("cell")
    else:
        cell_type = pd.Series(["cell"] * adata.n_obs, index=adata.obs_names)
    adata.obs["cell_type"] = cell_type.astype(str).values
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    return adata


def run_one(data_name: str, args, out: Path) -> dict:
    import gears
    from gears.inference import compute_metrics, evaluate

    row = {"dataset": data_name, "split": args.split, "seed": args.seed, "status": "started"}
    try:
        data_root = Path(args.data_path)
        data_root.mkdir(parents=True, exist_ok=True)
        pert_data = gears.PertData(str(data_root))
        if args.use_local_atlas and data_name in LOCAL_ATLAS_FILES:
            adata = _load_local_adata(data_name, max_genes=args.max_genes)
            local_dataset_name = f"{data_name}_local_atlas"
            pert_data.new_data_process(local_dataset_name, adata=adata)
        else:
            pert_data.load(data_name=data_name)
        pert_data.prepare_split(split=args.split, seed=args.seed, train_gene_set_size=args.train_gene_set_size)
        if args.split != "no_test" and "val" not in getattr(pert_data, "set2conditions", {}):
            train_conditions = list(pert_data.set2conditions.get("train", []))
            if not train_conditions:
                raise ValueError("GEARS split missing train conditions")
            val_pool = [c for c in train_conditions if c != "ctrl"] or train_conditions
            n_val = max(1, min(len(val_pool) // 5 or 1, len(val_pool)))
            val_conditions = val_pool[:n_val]
            remaining_train = [c for c in train_conditions if c not in val_conditions]
            if not remaining_train:
                remaining_train = train_conditions
            pert_data.set2conditions["train"] = remaining_train
            pert_data.set2conditions["val"] = val_conditions
        pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = gears.GEARS(pert_data, device=device, weight_bias_track=False)
        model.model_initialize(
            hidden_size=args.hidden_size,
            num_go_gnn_layers=args.num_go_gnn_layers,
            num_gene_gnn_layers=args.num_gene_gnn_layers,
            decoder_hidden_size=args.decoder_hidden_size,
            num_similar_genes_go_graph=args.num_similar_genes_go_graph,
            num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,
            coexpress_threshold=args.coexpress_threshold,
            uncertainty=args.uncertainty,
            direction_lambda=args.direction_lambda,
        )
        model.train(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        test_res = evaluate(pert_data.dataloader["test_loader"], model.best_model, args.uncertainty, device)
        metrics, pert_metrics = compute_metrics(test_res)
        row.update({f"test_{k}": float(v) for k, v in metrics.items()})
        row["n_test_perturbations"] = int(len(pert_metrics))
        row["status"] = "ok"
        pd.DataFrame([{"perturbation": k, **v} for k, v in pert_metrics.items()]).to_csv(out / f"GEARS_{data_name}_{args.split}_PERT_METRICS.csv", index=False)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc(limit=8)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-path", default="/home/yyf/datasets/gears_formal_baselines")
    parser.add_argument("--datasets", default="norman,dixit,adamson")
    parser.add_argument("--split", default="single")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-gene-set-size", type=float, default=0.75)
    parser.add_argument("--use-local-atlas", action="store_true", default=True)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--decoder-hidden-size", type=int, default=16)
    parser.add_argument("--num-go-gnn-layers", type=int, default=1)
    parser.add_argument("--num-gene-gnn-layers", type=int, default=1)
    parser.add_argument("--num-similar-genes-go-graph", type=int, default=10)
    parser.add_argument("--num-similar-genes-co-express-graph", type=int, default=10)
    parser.add_argument("--coexpress-threshold", type=float, default=0.4)
    parser.add_argument("--direction-lambda", type=float, default=0.1)
    parser.add_argument("--uncertainty", action="store_true")
    args = parser.parse_args()

    out = Path(args.root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        row = run_one(name, args, out)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "GEARS_FORMAL_BASELINE_STATUS.csv", index=False)
    (out / "GEARS_FORMAL_BASELINE_STATUS.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
