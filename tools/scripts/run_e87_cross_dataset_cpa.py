#!/usr/bin/env python3
"""E87: sciPlex3 -> OpenProblems cross-dataset CPA-RDKit audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(PACKAGE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402

SOURCE = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
TARGET = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/mega_external/"
    "OpenProblems_NeurIPS2023_single_cell_perturbations/data/raw/sc_counts_processed.h5ad"
)
E86 = ROOT / "docs/实验结果/E86_sciplex_to_openproblems_contract_20260712"
OUT = ROOT / "docs/实验结果/E87_sciplex_to_openproblems_cpa_20260712"
SOURCE_SMILES = Path("/home/yyf/archive/external/chemCPA/embeddings/trapnell_drugs_smiles.csv")


def norm_name(value: str) -> str:
    return re.sub("[^a-z0-9]", "", str(value).lower())


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def dense_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(x, dtype=np.float32).mean(axis=0)


def backed_columns(adata: ad.AnnData, positions: np.ndarray):
    """Read backed HDF5 columns in increasing order, then restore panel order."""
    positions = np.asarray(positions, dtype=int)
    read_order = np.argsort(positions)
    restore_order = np.argsort(read_order)
    matrix = adata.X[:, positions[read_order]]
    return matrix[:, restore_order]


def take(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return np.sort(indices)
    return np.sort(np.random.default_rng(seed).choice(indices, n, replace=False))


def load_data():
    panel = pd.read_csv(E86 / "tables/E86_GENE_PANEL.csv")
    contract = pd.read_csv(E86 / "tables/E86_CROSS_DATASET_MANIFEST.csv")
    source = sc.read_h5ad(SOURCE, backed="r")
    target = sc.read_h5ad(TARGET, backed="r")
    genes = panel["gene_id"].astype(str)
    sp = source.var_names.astype(str).get_indexer(genes)
    tp = target.var_names.astype(str).get_indexer(genes)
    if (sp < 0).any() or (tp < 0).any():
        raise RuntimeError("E86 gene panel mismatch")
    # h5py requires fancy column indices to be increasing.  E86 freezes a
    # biologically determined gene order, so read sorted columns and restore it.
    sx = backed_columns(source, sp)
    tx = backed_columns(target, tp)
    so = source.obs.copy()
    so["context_raw"] = so["cell_line"].astype(str)
    so["context_cpa"] = "sciPlex3::" + so["context_raw"]
    so["drug_raw"] = so["condition2"].astype(str)
    so["dose_nM"] = so["dose"].astype(float)
    so["task_key"] = (
        "sciPlex3::" + so["context_raw"] + "::" + so["drug_raw"] + "::dose_nM=" + so["dose_nM"].astype(str)
    )
    to = target.obs.copy()
    to["context_raw"] = to["cell_type"].astype(str)
    to["context_cpa"] = "OpenProblems::" + to["context_raw"]
    to["drug_raw"] = to["sm_name"].astype(str)
    to["dose_nM"] = to["dose_uM"].astype(float) * 1000.0
    to["task_key"] = (
        "OpenProblems2023::" + to["context_raw"] + "::" + to["drug_raw"] + "::dose_nM=" + to["dose_nM"].astype(str)
    )
    return panel, contract, source, target, sx, tx, so, to


def smiles_maps(so: pd.DataFrame, to: pd.DataFrame):
    external = pd.read_csv(SOURCE_SMILES, header=None, names=["drug", "smiles", "pathway"])
    ext = {norm_name(r.drug): str(r.smiles) for r in external.itertuples(index=False)}
    source_map = {drug: ext[norm_name(drug)] for drug in so.loc[so["perturbation"].astype(str).ne("control"), "drug_raw"].unique()}
    target_pairs = to.loc[~to["control"].astype(bool), ["drug_raw", "SMILES"]].drop_duplicates()
    counts = target_pairs.groupby("drug_raw")["SMILES"].nunique()
    if (counts > 1).any():
        raise RuntimeError("A target drug maps to multiple SMILES")
    target_map = target_pairs.set_index("drug_raw")["SMILES"].astype(str).to_dict()
    return source_map, target_map


def control_centroids(sx, tx, so, to):
    result = {}
    for context in sorted(so["context_cpa"].unique()):
        mask = so["context_cpa"].eq(context).to_numpy() & so["perturbation"].astype(str).eq("control").to_numpy()
        result[context] = dense_mean(sx[mask])
    for context in sorted(to["context_cpa"].unique()):
        mask = to["context_cpa"].eq(context).to_numpy() & to["control"].astype(bool).to_numpy()
        result[context] = dense_mean(tx[mask])
    return result


def build_combined(args):
    panel, contract, source, target, sx, tx, so, to = load_data()
    source_tasks = contract.loc[contract["role"].eq("source_train")]
    target_tasks = contract.loc[contract["role"].eq("target_test")]
    source_map, target_map = smiles_maps(so, to)
    x_parts, rows = [], []
    # Cell-level validation keeps every source task represented in training.
    for task in source_tasks.itertuples(index=False):
        idx = np.flatnonzero(so["task_key"].eq(task.task_key).to_numpy())
        chosen = take(idx, args.max_cells, stable_seed(args.seed, task.task_key))
        n_val = max(1, round(0.125 * len(chosen)))
        for j, original in enumerate(chosen):
            x_parts.append(sx[original])
            rows.append(
                {
                    "context_cpa": "sciPlex3::" + task.context,
                    "drug_cpa": task.drug,
                    "dose_cpa": str(np.log10(max(float(task.dose_nM), 1.0))),
                    "smiles_cpa": source_map[task.drug],
                    "split_cpa": "valid" if j < n_val else "train",
                    "pseudo_test": False,
                    "prediction_task_key": "",
                }
            )
    for context in sorted(so["context_raw"].unique()):
        idx = np.flatnonzero(
            so["context_raw"].eq(context).to_numpy()
            & so["perturbation"].astype(str).eq("control").to_numpy()
        )
        for original in take(idx, args.control_cells, stable_seed(args.seed, "source_control", context)):
            x_parts.append(sx[original])
            rows.append({"context_cpa": "sciPlex3::" + context, "drug_cpa": "control", "dose_cpa": "0.0", "smiles_cpa": "C", "split_cpa": "train", "pseudo_test": False, "prediction_task_key": ""})
    for context in sorted(to["context_raw"].unique()):
        idx = np.flatnonzero(to["context_raw"].eq(context).to_numpy() & to["control"].astype(bool).to_numpy())
        for original in take(idx, args.control_cells, stable_seed(args.seed, "target_control", context)):
            x_parts.append(tx[original])
            rows.append({"context_cpa": "OpenProblems::" + context, "drug_cpa": "control", "dose_cpa": "0.0", "smiles_cpa": "C", "split_cpa": "train", "pseudo_test": False, "prediction_task_key": ""})
    for task in target_tasks.itertuples(index=False):
        ctrl = np.flatnonzero(
            to["context_raw"].eq(task.context).to_numpy() & to["control"].astype(bool).to_numpy()
        )
        chosen = take(ctrl, args.pseudo_cells, stable_seed(args.seed, "pseudo", task.task_key))
        for original in chosen:
            x_parts.append(tx[original])
            rows.append(
                {
                    "context_cpa": "OpenProblems::" + task.context,
                    "drug_cpa": task.drug,
                    "dose_cpa": str(np.log10(max(float(task.dose_nM), 1.0))),
                    "smiles_cpa": target_map[task.drug],
                    "split_cpa": "test",
                    "pseudo_test": True,
                    "prediction_task_key": task.task_key,
                }
            )
    # sciPlex3 is stored dense while OpenProblems is sparse.  Normalize every
    # selected cell to one row before stacking so the two storage formats can
    # coexist without changing values or gene order.
    combined_x = np.vstack(
        [part.toarray().ravel() if sparse.issparse(part) else np.asarray(part).ravel() for part in x_parts]
    ).astype(np.float32, copy=False)
    combined = ad.AnnData(X=combined_x, obs=pd.DataFrame(rows))
    combined.var_names = panel["gene_id"].astype(str).to_numpy()
    return combined, panel, contract, sx, tx, so, to, source_map, target_map


def fingerprint(smiles: str, bits: int = 256) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    return np.asarray(fp, dtype=np.float32)


def ridge_predictions(contract, sx, so, centroids, source_map, target_map):
    contexts = sorted(centroids)
    context_matrix = np.vstack([centroids[c] for c in contexts])
    c_scaled = StandardScaler().fit_transform(context_matrix)
    context_pca = PCA(n_components=min(6, len(contexts) - 1), random_state=20260712).fit_transform(c_scaled)
    c_embed = {context: context_pca[i].astype(np.float32) for i, context in enumerate(contexts)}

    def vector(context, drug, dose_nm, smiles):
        c = c_embed[context]
        d = fingerprint(smiles)
        dose = np.float32(np.log10(max(float(dose_nm), 1.0)) / 5.0)
        return np.concatenate([c, d, [dose], d * dose, (c[:, None] * d[None, :]).ravel()]).astype(np.float32)

    x_train, y_train = [], []
    for task in contract.loc[contract["role"].eq("source_train")].itertuples(index=False):
        context = "sciPlex3::" + task.context
        mask = so["task_key"].eq(task.task_key).to_numpy()
        effect = dense_mean(sx[mask]) - centroids[context]
        x_train.append(vector(context, task.drug, task.dose_nM, source_map[task.drug]))
        y_train.append(effect)
    scaler = StandardScaler().fit(np.vstack(x_train))
    ridge = Ridge(alpha=10.0, solver="lsqr").fit(scaler.transform(np.vstack(x_train)), np.vstack(y_train))
    output = {}
    for task in contract.loc[contract["role"].eq("target_test")].itertuples(index=False):
        context = "OpenProblems::" + task.context
        features = vector(context, task.drug, task.dose_nM, target_map[task.drug])[None, :]
        output[task.task_key] = ridge.predict(scaler.transform(features))[0].astype(np.float32)
    return output


def predict(args) -> None:
    import cpa
    import torch

    combined, panel, contract, sx, tx, so, to, source_map, target_map = build_combined(args)
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(combined, perturbation_key="drug_cpa", dosage_key="dose_cpa", control_group="control", smiles_key="smiles_cpa", is_count_data=False, categorical_covariate_keys=["context_cpa"], max_comb_len=1)
    model = cpa.CPA(
        combined, split_key="split_cpa", train_split="train", valid_split="valid", test_split="test",
        use_rdkit_embeddings=True, n_latent=32, recon_loss="gauss", doser_type="linear",
        n_hidden_encoder=128, n_layers_encoder=2, n_hidden_decoder=128, n_layers_decoder=2,
        use_batch_norm_encoder=True, use_layer_norm_encoder=False, use_batch_norm_decoder=True,
        use_layer_norm_decoder=False, dropout_rate_encoder=0.1, dropout_rate_decoder=0.1,
        variational=False, seed=args.seed,
    )
    model.train(
        max_epochs=20, use_gpu=args.device, batch_size=64,
        plan_kwargs={"n_epochs_pretrain_ae": 5, "n_epochs_adv_warmup": 5, "adv_steps": 1, "n_hidden_adv": 64, "n_layers_adv": 2, "n_epochs_verbose": 1, "lr": 3e-4},
        save_path=False, check_val_every_n_epoch=1, early_stopping_patience=5,
        enable_progress_bar=False, logger=False,
    )
    test = combined[combined.obs["split_cpa"].eq("test")].copy()
    model.predict(test, batch_size=128, n_samples=1)
    cpa_expression = np.asarray(test.obsm["CPA_pred"], dtype=np.float32)
    centroids = control_centroids(sx, tx, so, to)
    cpa_pred = {}
    test_obs = test.obs.reset_index(drop=True)
    for task_key, indices in test_obs.groupby("prediction_task_key").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        context = str(test_obs.loc[idx[0], "context_cpa"])
        cpa_pred[task_key] = (cpa_expression[idx].mean(axis=0) - centroids[context]).astype(np.float32)
    ridge_pred = ridge_predictions(contract, sx, so, centroids, source_map, target_map)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arrays").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "raw_cpa").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "arrays/E87_CPA_PREDICTED_EFFECTS.npz", **cpa_pred)
    np.savez_compressed(OUT / "arrays/E87_RIDGE_PREDICTED_EFFECTS.npz", **ridge_pred)
    model.epoch_history.to_csv(OUT / "tables/E87_TRAINING_HISTORY.csv", index=False)
    torch.save(model.module.state_dict(), OUT / "raw_cpa/cpa_cross_dataset_state.pt")
    status = {
        "experiment": "E87_sciplex_to_openproblems_cpa",
        "phase": "prediction_complete_truth_unread",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_tasks": int(contract["role"].eq("source_train").sum()),
        "target_tasks": len(cpa_pred),
        "cpa_train_cells": int(combined.obs["split_cpa"].eq("train").sum()),
        "cpa_val_cells": int(combined.obs["split_cpa"].eq("valid").sum()),
        "pseudo_target_control_cells": int(combined.obs["split_cpa"].eq("test").sum()),
        "source_target_exact_drug_overlap": 0,
        "target_perturbed_truth_used_for_prediction": False,
        "gene_order_hash": panel["gene_order_hash"].iloc[0],
        "cpa_source_commit": "fbd7c0250edc23eff003a10c99655579c53afd63",
        "environment": "cpa_runtime_env compatibility environment; pinned rerun pending",
    }
    (OUT / "PREDICT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def bootstrap_rho(x, y, seed, n_boot=2000):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        vals.append(stats.spearmanr(x[idx], y[idx]).statistic)
    return tuple(np.nanquantile(vals, [0.025, 0.975]))


def bootstrap_delta_rho(score_a, score_b, error, seed, n_boot=2000):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(error), len(error))
        vals.append(
            stats.spearmanr(score_a[idx], error[idx]).statistic
            - stats.spearmanr(score_b[idx], error[idx]).statistic
        )
    return tuple(np.nanquantile(vals, [0.025, 0.975]))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def evaluate(args) -> None:
    panel, contract, source, target, sx, tx, so, to = load_data()
    cpa_pred = np.load(OUT / "arrays/E87_CPA_PREDICTED_EFFECTS.npz")
    ridge_pred = np.load(OUT / "arrays/E87_RIDGE_PREDICTED_EFFECTS.npz")
    centroids = control_centroids(sx, tx, so, to)
    rows, pred_arrays, true_arrays, records = [], {}, {}, []
    target_tasks = contract.loc[contract["role"].eq("target_test")]
    for task in target_tasks.itertuples(index=False):
        context = "OpenProblems::" + task.context
        truth = dense_mean(tx[to["task_key"].eq(task.task_key).to_numpy()]) - centroids[context]
        pc = cpa_pred[task.task_key]
        pr = ridge_pred[task.task_key]
        ec = float(np.sqrt(np.mean(np.square(pc - truth))))
        er = float(np.sqrt(np.mean(np.square(pr - truth))))
        dis = float(np.sqrt(np.mean(np.square(pc - pr))))
        true_key = f"E87::{task.task_key}::true"
        true_arrays[true_key] = truth.astype(np.float32)
        for predictor, prediction, error in [("CPA_0.8.8_RDKIT_cross_dataset", pc, ec), ("inductive_ridge_cross_dataset_v1", pr, er)]:
            record_id = f"E87::{task.task_key}::{predictor}"
            pred_key = record_id + "::pred"
            pred_arrays[pred_key] = prediction.astype(np.float32)
            denom = max(float(np.linalg.norm(prediction) * np.linalg.norm(truth)), 1e-12)
            records.append({
                "schema_version": "safeconf_prediction_record_v1", "record_id": record_id,
                "task_id": task.task_key, "task_key": task.task_key, "dataset_name": "OpenProblems2023_cross_from_sciPlex3",
                "dataset_group": "cross_dataset_chemical", "fold_id": "sciPlex3_to_OpenProblems2023", "split": "test",
                "context": task.context, "perturbation": f"{task.drug}::dose_nM={task.dose_nM}",
                "predictor_name": predictor, "run_type": "formal", "gene_panel_id": "E86_control_only_common1000",
                "gene_order_hash": panel["gene_order_hash"].iloc[0], "effect_definition": "mean_diff",
                "normalization_id": "cross_dataset_log_expression_minus_dataset_context_control_v1",
                "error_normalization": "raw_rmse", "predicted_effect_key": pred_key, "true_effect_key": true_key,
                "true_error_rmse": error, "true_error_cosine": 1.0 - float(np.dot(prediction, truth) / denom), "n_cells": task.n_cells,
            })
        rows.append({
            "task_key": task.task_key, "context": task.context, "drug": task.drug, "dose_nM": task.dose_nM,
            "n_cells": task.n_cells, "predicted_magnitude_cpa": float(np.sqrt(np.mean(np.square(pc)))),
            "predicted_magnitude_ridge": float(np.sqrt(np.mean(np.square(pr)))),
            "predicted_magnitude_mean": (float(np.sqrt(np.mean(np.square(pc)))) + float(np.sqrt(np.mean(np.square(pr))))) / 2,
            "model_disagreement_rmse": dis, "error_cpa_rmse": ec, "error_ridge_rmse": er,
            "pair_mean_rmse": (ec + er) / 2, "pair_max_rmse": max(ec, er),
            "zero_effect_rmse": float(np.sqrt(np.mean(np.square(truth)))),
            "true_magnitude_oracle": float(np.sqrt(np.mean(np.square(truth)))), "target_truth_used_for_scores": False,
        })
    scores = pd.DataFrame(rows)
    scores["triangle_mean_bound_holds"] = scores["pair_mean_rmse"] + 1e-7 >= scores["model_disagreement_rmse"] / 2
    scores["triangle_max_bound_holds"] = scores["pair_max_rmse"] + 1e-7 >= scores["model_disagreement_rmse"] / 2
    scores.to_csv(OUT / "tables/E87_TASK_SCORES.csv", index=False)
    record_frame = pd.DataFrame(records)
    record_frame.to_csv(OUT / "tables/PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT / "arrays/predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT / "arrays/true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=record_frame, strict=True)
    pd.DataFrame({"issue": issues}).to_csv(OUT / "tables/E87_STRICT_CONTRACT_ISSUES.csv", index=False)
    if issues:
        raise RuntimeError("E87 strict contract: " + "; ".join(issues))

    summary_rows = []
    for group_name, group in [("pooled", scores)] + [(f"context::{c}", g) for c, g in scores.groupby("context")]:
        for score, target_col in [("model_disagreement_rmse", "pair_mean_rmse"), ("model_disagreement_rmse", "pair_max_rmse"), ("predicted_magnitude_mean", "pair_mean_rmse")]:
            x = group[score].to_numpy(float); y = group[target_col].to_numpy(float)
            rho = float(stats.spearmanr(x, y).statistic)
            ci = bootstrap_rho(x, y, stable_seed(group_name, score, target_col))
            k = max(1, int(np.ceil(0.2 * len(group))))
            top = np.argsort(-x)[:k]
            summary_rows.append({"group": group_name, "score_name": score, "target_error": target_col, "n_tasks": len(group), "spearman": rho, "bootstrap_ci95_low": ci[0], "bootstrap_ci95_high": ci[1], "top20_error_enrichment": float(y[top].mean() / y.mean())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tables/E87_RISK_ERROR_SUMMARY.csv", index=False)
    pooled = summary.loc[summary["group"].eq("pooled")]
    diagnostics = []
    for group_name, group in [("pooled", scores)] + [(f"context::{c}", g) for c, g in scores.groupby("context")]:
        disagreement_rho = float(stats.spearmanr(group["model_disagreement_rmse"], group["pair_mean_rmse"]).statistic)
        magnitude_rho = float(stats.spearmanr(group["predicted_magnitude_mean"], group["pair_mean_rmse"]).statistic)
        delta_ci = bootstrap_delta_rho(
            group["model_disagreement_rmse"].to_numpy(float),
            group["predicted_magnitude_mean"].to_numpy(float),
            group["pair_mean_rmse"].to_numpy(float),
            stable_seed("delta", group_name),
        )
        diagnostics.append({
            "group": group_name,
            "n_tasks": len(group),
            "rho_disagreement_pair_mean": disagreement_rho,
            "rho_magnitude_pair_mean": magnitude_rho,
            "delta_rho_disagreement_minus_magnitude": disagreement_rho - magnitude_rho,
            "bootstrap_delta_ci95_low": delta_ci[0],
            "bootstrap_delta_ci95_high": delta_ci[1],
            "cpa_beats_zero_fraction": float((group["error_cpa_rmse"] < group["zero_effect_rmse"]).mean()),
            "ridge_beats_zero_fraction": float((group["error_ridge_rmse"] < group["zero_effect_rmse"]).mean()),
            "median_cpa_error_over_zero": float(np.median(group["error_cpa_rmse"] / group["zero_effect_rmse"])),
            "median_ridge_error_over_zero": float(np.median(group["error_ridge_rmse"] / group["zero_effect_rmse"])),
            "rho_disagreement_vs_ridge_magnitude": float(stats.spearmanr(group["model_disagreement_rmse"], group["predicted_magnitude_ridge"]).statistic),
        })
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(OUT / "tables/E87_PREDICTOR_DIAGNOSTICS.csv", index=False)
    main = diagnostic_frame.loc[diagnostic_frame["group"].eq("pooled")].iloc[0]
    pooled_display = pooled[["score_name", "target_error", "n_tasks", "spearman", "bootstrap_ci95_low", "bootstrap_ci95_high", "top20_error_enrichment"]].copy()
    diagnostic_display = diagnostic_frame[["group", "n_tasks", "rho_disagreement_pair_mean", "rho_magnitude_pair_mean", "delta_rho_disagreement_minus_magnitude", "bootstrap_delta_ci95_low", "bootstrap_delta_ci95_high", "cpa_beats_zero_fraction", "ridge_beats_zero_fraction"]].copy()
    for frame in (pooled_display, diagnostic_display):
        for column in frame.select_dtypes(include=[np.number]).columns:
            frame[column] = frame[column].round(3)
    report = f"""# E87｜sciPlex3 → OpenProblems 跨数据集 CPA 审计

CPA-RDKit 与 inductive ridge 只在 sciPlex3 的 108 个扰动任务上学习；OpenProblems 只提供 4 类 PBMC 的 control 表达、141 个药物的 SMILES 和剂量。553 个目标 perturbed truth 在预测文件落盘后才读取。

- strict PredictionRecord：{len(record_frame)}，issues=0
- source/target 同名药：0
- pair mean/max 下界违反：{int((~scores['triangle_mean_bound_holds']).sum())}/{int((~scores['triangle_max_bound_holds']).sum())}

## 风险排序

{markdown_table(pooled_display)}

## 预测器与幅度诊断

{markdown_table(diagnostic_display)}

分歧与 pair-mean error 的相关性很高，但预测幅度得到几乎相同的排序。两者的差值为 {main['delta_rho_disagreement_minus_magnitude']:.4f}，bootstrap 95% CI 为 [{main['bootstrap_delta_ci95_low']:.4f}, {main['bootstrap_delta_ci95_high']:.4f}]。CPA 和 ridge 分别只在 {main['cpa_beats_zero_fraction']:.1%}、{main['ridge_beats_zero_fraction']:.1%} 的目标任务上优于零效应预测；分歧与 ridge 预测幅度的相关性为 {main['rho_disagreement_vs_ridge_magnitude']:.3f}。

因此，E87 证明了整个跨数据集预测—落盘—解封—评估链路可以运行，也保留了 pair-risk 下界；它没有证明分歧稳定优于幅度。两个源域预测器在 PBMC 目标域严重失准，当前的高相关主要来自 ridge 外推幅度同时主导分歧与误差。这个结果按负面边界保留，不能作为 SafeConf 跨域独立增益的主证据。
"""
    (OUT / "reports/E87_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E87 先看这个\n\n先读 `reports/E87_REPORT.md`。\n")
    status = {
        "experiment": "E87_sciplex_to_openproblems_cpa", "phase": "evaluation_complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"), "source_tasks": 108,
        "target_tasks": len(scores), "n_prediction_records": len(record_frame), "strict_issue_count": len(issues),
        "triangle_mean_violations": int((~scores["triangle_mean_bound_holds"]).sum()),
        "triangle_max_violations": int((~scores["triangle_max_bound_holds"]).sum()),
        "target_truth_used_for_scores": False, "target_truth_used_for_evaluation_only": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(pooled.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "evaluate", "full"], default="full")
    parser.add_argument("--seed", type=int, default=20268700)
    parser.add_argument("--max-cells", type=int, default=32)
    parser.add_argument("--control-cells", type=int, default=64)
    parser.add_argument("--pseudo-cells", type=int, default=4)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    if args.mode in {"predict", "full"}: predict(args)
    if args.mode in {"evaluate", "full"}: evaluate(args)


if __name__ == "__main__":
    main()
