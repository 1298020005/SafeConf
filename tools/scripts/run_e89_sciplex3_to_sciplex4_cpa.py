#!/usr/bin/env python3
"""E89: formal sciPlex3 -> sciPlex4 CPA-RDKit transfer after E88 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(PACKAGE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402

SOURCE = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex3.h5ad")
TARGET = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/SrivatsanTrapnell2020_sciplex4.h5ad")
E88 = ROOT / "docs/实验结果/E88_sciplex3_to_sciplex4_contract_20260712"
OUT = ROOT / "docs/实验结果/E89_sciplex3_to_sciplex4_cpa_20260712"
SMILES_FILE = Path("/home/yyf/archive/external/chemCPA/embeddings/trapnell_drugs_smiles.csv")
RAW_DRUG = {"Abexinostat": "Abexinostat (PCI-24781)", "Pracinostat": "Pracinostat (SB939)"}


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def take(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return np.sort(indices)
    return np.sort(np.random.default_rng(seed).choice(indices, n, replace=False))


def read_normalized(adata, rows: np.ndarray, positions: np.ndarray) -> sparse.csr_matrix:
    rows = np.sort(np.asarray(rows, dtype=int))
    positions = np.asarray(positions, dtype=int)
    order = np.argsort(positions)
    restore = np.argsort(order)
    matrix = sparse.csr_matrix(adata.X[rows][:, positions[order]][:, restore], dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scales = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    matrix = sparse.diags(scales).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def mean_vector(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def load_metadata():
    panel = pd.read_csv(E88 / "tables/E88_GENE_PANEL.csv")
    manifest = pd.read_csv(E88 / "tables/E88_TRANSFER_MANIFEST.csv")
    source = sc.read_h5ad(SOURCE, backed="r")
    target = sc.read_h5ad(TARGET, backed="r")
    positions_source = source.var_names.astype(str).get_indexer(panel["gene_id"].astype(str))
    positions_target = target.var_names.astype(str).get_indexer(panel["gene_id"].astype(str))
    if (positions_source < 0).any() or (positions_target < 0).any():
        raise RuntimeError("E88 gene panel mismatch")
    return panel, manifest, source, target, source.obs.copy(), target.obs.copy(), positions_source, positions_target


def source_indices(obs: pd.DataFrame, task) -> np.ndarray:
    mask = (
        obs["cell_line"].astype(str).eq(task.context)
        & obs["perturbation"].astype(str).eq(RAW_DRUG[task.drug])
        & obs["dose_value"].astype(float).eq(float(task.dose_nM))
        & obs["time"].astype(float).eq(24.0)
    )
    return np.flatnonzero(mask.to_numpy())


def target_indices(obs: pd.DataFrame, task) -> np.ndarray:
    mask = (
        obs["cell_line"].astype(str).eq(task.context)
        & obs["perturbation"].astype(str).eq("control")
        & obs["perturbation_2"].astype(str).eq(task.drug)
        & obs["dose_value_2"].astype(float).eq(float(task.dose_nM) / 1000.0)
    )
    return np.flatnonzero(mask.to_numpy())


def control_indices(obs: pd.DataFrame, dataset: str, context: str) -> np.ndarray:
    if dataset == "source":
        mask = (
            obs["cell_line"].astype(str).eq(context)
            & obs["perturbation"].astype(str).eq("control")
            & obs["time"].astype(float).eq(24.0)
        )
    else:
        mask = (
            obs["cell_line"].astype(str).eq(context)
            & obs["perturbation"].astype(str).eq("control")
            & obs["perturbation_2"].astype(str).eq("control")
        )
    return np.flatnonzero(mask.to_numpy())


def smiles_map() -> dict[str, str]:
    frame = pd.read_csv(SMILES_FILE, header=None, names=["drug", "smiles", "pathway"])
    result = {}
    for drug in RAW_DRUG:
        hit = frame.loc[frame["drug"].astype(str).str.lower().eq(drug.lower()), "smiles"]
        if hit.empty:
            raise RuntimeError(f"Missing SMILES for {drug}")
        result[drug] = str(hit.iloc[0])
    return result


def control_centroids(source, target, so, to, sp, tp):
    source_centroids, target_centroids = {}, {}
    for context in ["A549", "MCF7", "K562"]:
        source_centroids[context] = mean_vector(read_normalized(source, control_indices(so, "source", context), sp))
    for context in ["A549", "MCF7"]:
        target_centroids[context] = mean_vector(read_normalized(target, control_indices(to, "target", context), tp))
    return source_centroids, target_centroids


def source_interpolator(manifest, source, so, sp, source_centroids):
    effects = {}
    source_rows = manifest.loc[manifest["role"].eq("source_train")]
    for task in source_rows.itertuples(index=False):
        effects[(task.context, task.drug, float(task.dose_nM))] = (
            mean_vector(read_normalized(source, source_indices(so, task), sp)) - source_centroids[task.context]
        )

    def predict(context: str, drug: str, dose_nm: float) -> np.ndarray:
        doses = np.array(sorted(d for c, p, d in effects if c == context and p == drug), dtype=float)
        log_doses = np.log10(doses)
        value = np.log10(float(dose_nm))
        if value <= log_doses[0]:
            return effects[(context, drug, float(doses[0]))].copy()
        if value >= log_doses[-1]:
            return effects[(context, drug, float(doses[-1]))].copy()
        upper = int(np.searchsorted(log_doses, value))
        lower = upper - 1
        weight = (value - log_doses[lower]) / (log_doses[upper] - log_doses[lower])
        low = effects[(context, drug, float(doses[lower]))]
        high = effects[(context, drug, float(doses[upper]))]
        return ((1.0 - weight) * low + weight * high).astype(np.float32)

    return predict


def build_cpa_data(args, panel, manifest, source, target, so, to, sp, tp):
    smiles = smiles_map()
    matrices, rows = [], []
    for task in manifest.loc[manifest["role"].eq("source_train")].itertuples(index=False):
        chosen = take(source_indices(so, task), args.max_cells, stable_seed(args.seed, task.task_key))
        matrix = read_normalized(source, chosen, sp)
        n_val = max(1, round(0.125 * len(chosen)))
        valid = set(np.random.default_rng(stable_seed(args.seed, "valid", task.task_key)).choice(len(chosen), n_val, replace=False))
        matrices.append(matrix)
        for index in range(len(chosen)):
            rows.append({"context_cpa": task.context, "drug_cpa": task.drug, "dose_cpa": str(np.log10(task.dose_nM)), "smiles_cpa": smiles[task.drug], "split_cpa": "valid" if index in valid else "train", "prediction_task_key": ""})
    for context in ["A549", "MCF7", "K562"]:
        chosen = take(control_indices(so, "source", context), args.control_cells, stable_seed(args.seed, "source_control", context))
        matrices.append(read_normalized(source, chosen, sp))
        rows.extend([{"context_cpa": context, "drug_cpa": "control", "dose_cpa": "0.0", "smiles_cpa": "C", "split_cpa": "train", "prediction_task_key": ""}] * len(chosen))
    target_controls = {}
    for context in ["A549", "MCF7"]:
        all_controls = control_indices(to, "target", context)
        target_controls[context] = all_controls
        chosen = take(all_controls, args.control_cells, stable_seed(args.seed, "target_control", context))
        matrices.append(read_normalized(target, chosen, tp))
        rows.extend([{"context_cpa": context, "drug_cpa": "control", "dose_cpa": "0.0", "smiles_cpa": "C", "split_cpa": "train", "prediction_task_key": ""}] * len(chosen))
    for task in manifest.loc[manifest["role"].eq("target_test")].itertuples(index=False):
        chosen = take(target_controls[task.context], args.pseudo_cells, stable_seed(args.seed, "pseudo", task.task_key))
        matrices.append(read_normalized(target, chosen, tp))
        rows.extend([{"context_cpa": task.context, "drug_cpa": task.drug, "dose_cpa": str(np.log10(task.dose_nM)), "smiles_cpa": smiles[task.drug], "split_cpa": "test", "prediction_task_key": task.task_key}] * len(chosen))
    combined = ad.AnnData(X=sparse.vstack(matrices).tocsr(), obs=pd.DataFrame(rows))
    combined.var_names = panel["gene_id"].astype(str).to_numpy()
    return combined


def predict(args) -> None:
    import cpa
    import torch

    panel, manifest, source, target, so, to, sp, tp = load_metadata()
    combined = build_cpa_data(args, panel, manifest, source, target, so, to, sp, tp)
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(combined, perturbation_key="drug_cpa", dosage_key="dose_cpa", control_group="control", smiles_key="smiles_cpa", is_count_data=False, categorical_covariate_keys=["context_cpa"], max_comb_len=1)
    model = cpa.CPA(combined, split_key="split_cpa", train_split="train", valid_split="valid", test_split="test", use_rdkit_embeddings=True, n_latent=32, recon_loss="gauss", doser_type="linear", n_hidden_encoder=128, n_layers_encoder=2, n_hidden_decoder=128, n_layers_decoder=2, use_batch_norm_encoder=True, use_layer_norm_encoder=False, use_batch_norm_decoder=True, use_layer_norm_decoder=False, dropout_rate_encoder=0.1, dropout_rate_decoder=0.1, variational=False, seed=args.seed)
    model.train(max_epochs=20, use_gpu=args.device, batch_size=64, plan_kwargs={"n_epochs_pretrain_ae": 5, "n_epochs_adv_warmup": 5, "adv_steps": 1, "n_hidden_adv": 64, "n_layers_adv": 2, "n_epochs_verbose": 1, "lr": 3e-4}, save_path=False, check_val_every_n_epoch=1, early_stopping_patience=5, enable_progress_bar=False, logger=False)
    test = combined[combined.obs["split_cpa"].eq("test")].copy()
    model.predict(test, batch_size=128, n_samples=1)
    cpa_expression = np.asarray(test.obsm["CPA_pred"], dtype=np.float32)
    source_centroids, target_centroids = control_centroids(source, target, so, to, sp, tp)
    test_obs = test.obs.reset_index(drop=True)
    cpa_effects = {}
    for task_key, indices in test_obs.groupby("prediction_task_key").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        context = str(test_obs.loc[idx[0], "context_cpa"])
        cpa_effects[task_key] = (cpa_expression[idx].mean(axis=0) - target_centroids[context]).astype(np.float32)
    interpolate = source_interpolator(manifest, source, so, sp, source_centroids)
    interpolation_effects = {task.task_key: interpolate(task.context, task.drug, task.dose_nM) for task in manifest.loc[manifest["role"].eq("target_test")].itertuples(index=False)}
    (OUT / "arrays").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "raw_cpa").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "arrays/E89_CPA_PREDICTED_EFFECTS.npz", **cpa_effects)
    np.savez_compressed(OUT / "arrays/E89_INTERPOLATION_PREDICTED_EFFECTS.npz", **interpolation_effects)
    model.epoch_history.to_csv(OUT / "tables/E89_TRAINING_HISTORY.csv", index=False)
    torch.save(model.module.state_dict(), OUT / "raw_cpa/cpa_state.pt")
    status = {"experiment": "E89_sciplex3_to_sciplex4_cpa", "phase": "prediction_complete_truth_unread", "generated_at": datetime.now().isoformat(timespec="seconds"), "source_tasks": 24, "target_tasks": len(cpa_effects), "target_perturbed_truth_used_for_prediction": False, "gene_order_hash": panel["gene_order_hash"].iloc[0], "cpa_source_commit": "fbd7c0250edc23eff003a10c99655579c53afd63", "environment": "cpa_runtime_env compatibility environment; pinned rerun pending"}
    (OUT / "PREDICT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def bootstrap_delta(a, b, y, seed, n_boot=5000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        values.append(stats.spearmanr(a[idx], y[idx]).statistic - stats.spearmanr(b[idx], y[idx]).statistic)
    return np.nanquantile(values, [0.025, 0.975])


def evaluate(args) -> None:
    panel, manifest, source, target, so, to, sp, tp = load_metadata()
    cpa = np.load(OUT / "arrays/E89_CPA_PREDICTED_EFFECTS.npz")
    interp = np.load(OUT / "arrays/E89_INTERPOLATION_PREDICTED_EFFECTS.npz")
    _, target_centroids = control_centroids(source, target, so, to, sp, tp)
    scores, records, pred_arrays, true_arrays = [], [], {}, {}
    for task in manifest.loc[manifest["role"].eq("target_test")].itertuples(index=False):
        truth = mean_vector(read_normalized(target, target_indices(to, task), tp)) - target_centroids[task.context]
        pc, pi = cpa[task.task_key], interp[task.task_key]
        ec = float(np.sqrt(np.mean(np.square(pc - truth))))
        ei = float(np.sqrt(np.mean(np.square(pi - truth))))
        disagreement = float(np.sqrt(np.mean(np.square(pc - pi))))
        true_key = f"E89::{task.task_key}::true"
        true_arrays[true_key] = truth.astype(np.float32)
        for name, prediction, error in [("CPA_0.8.8_RDKIT_sciPlex3", pc, ec), ("source_dose_interpolation_v1", pi, ei)]:
            record_id = f"E89::{task.task_key}::{name}"
            pred_key = record_id + "::pred"
            pred_arrays[pred_key] = prediction.astype(np.float32)
            denom = max(float(np.linalg.norm(prediction) * np.linalg.norm(truth)), 1e-12)
            records.append({"schema_version": "safeconf_prediction_record_v1", "record_id": record_id, "task_id": task.task_key, "task_key": task.task_key, "dataset_name": "sciPlex4_cross_from_sciPlex3", "dataset_group": "same_family_cross_dataset_chemical", "fold_id": "sciPlex3_to_sciPlex4", "split": "test", "context": task.context, "perturbation": f"{task.drug}::dose_nM={task.dose_nM}", "predictor_name": name, "run_type": "formal", "gene_panel_id": "E88_control_only_common1000", "gene_order_hash": panel["gene_order_hash"].iloc[0], "effect_definition": "mean_diff", "normalization_id": "per_cell_total1e4_log1p_minus_dataset_context_control", "error_normalization": "raw_rmse", "predicted_effect_key": pred_key, "true_effect_key": true_key, "true_error_rmse": error, "true_error_cosine": 1.0 - float(np.dot(prediction, truth) / denom), "n_cells": task.n_cells})
        mc = float(np.sqrt(np.mean(np.square(pc)))); mi = float(np.sqrt(np.mean(np.square(pi)))); zero = float(np.sqrt(np.mean(np.square(truth))))
        scores.append({"task_key": task.task_key, "context": task.context, "drug": task.drug, "dose_nM": task.dose_nM, "dose_seen_in_source": bool(task.dose_seen_in_source), "n_cells": task.n_cells, "model_disagreement_rmse": disagreement, "predicted_magnitude_mean": (mc + mi) / 2, "predicted_magnitude_cpa": mc, "predicted_magnitude_interpolation": mi, "error_cpa_rmse": ec, "error_interpolation_rmse": ei, "pair_mean_rmse": (ec + ei) / 2, "pair_max_rmse": max(ec, ei), "zero_effect_rmse": zero, "target_truth_used_for_scores": False})
    scores = pd.DataFrame(scores)
    scores["triangle_mean_bound_holds"] = scores["pair_mean_rmse"] + 1e-7 >= scores["model_disagreement_rmse"] / 2
    scores["triangle_max_bound_holds"] = scores["pair_max_rmse"] + 1e-7 >= scores["model_disagreement_rmse"] / 2
    record_frame = pd.DataFrame(records)
    scores.to_csv(OUT / "tables/E89_TASK_SCORES.csv", index=False)
    record_frame.to_csv(OUT / "tables/PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT / "arrays/predicted_effects.npz", **pred_arrays)
    np.savez_compressed(OUT / "arrays/true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=record_frame, strict=True)
    pd.DataFrame({"issue": issues}).to_csv(OUT / "tables/E89_STRICT_CONTRACT_ISSUES.csv", index=False)
    if issues:
        raise RuntimeError("E89 strict contract: " + "; ".join(issues))
    rows = []
    for group_name, group in [("pooled", scores), ("exact_source_dose", scores[scores["dose_seen_in_source"]]), ("interpolated_dose", scores[~scores["dose_seen_in_source"]]), *[(f"context::{c}", g) for c, g in scores.groupby("context")], *[(f"drug::{d}", g) for d, g in scores.groupby("drug")]]:
        disagreement_rho = float(stats.spearmanr(group["model_disagreement_rmse"], group["pair_mean_rmse"]).statistic)
        magnitude_rho = float(stats.spearmanr(group["predicted_magnitude_mean"], group["pair_mean_rmse"]).statistic)
        delta_ci = bootstrap_delta(group["model_disagreement_rmse"].to_numpy(float), group["predicted_magnitude_mean"].to_numpy(float), group["pair_mean_rmse"].to_numpy(float), stable_seed("delta", group_name))
        k = max(1, int(np.ceil(0.2 * len(group))))
        top = np.argsort(-group["model_disagreement_rmse"].to_numpy())[:k]
        rows.append({"group": group_name, "n_tasks": len(group), "rho_disagreement_pair_mean": disagreement_rho, "rho_magnitude_pair_mean": magnitude_rho, "delta_rho": disagreement_rho - magnitude_rho, "bootstrap_delta_ci95_low": delta_ci[0], "bootstrap_delta_ci95_high": delta_ci[1], "top20_error_enrichment": float(group["pair_mean_rmse"].to_numpy()[top].mean() / group["pair_mean_rmse"].mean()), "cpa_beats_zero_fraction": float((group["error_cpa_rmse"] < group["zero_effect_rmse"]).mean()), "interpolation_beats_zero_fraction": float((group["error_interpolation_rmse"] < group["zero_effect_rmse"]).mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "tables/E89_SUMMARY.csv", index=False)
    main = summary.iloc[0]
    display = summary.round(3)
    headers = list(display.columns)
    md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    report = f"""# E89｜sciPlex3 → sciPlex4 CPA 同族外部验证

CPA-RDKit 与 source-dose interpolation 只使用 sciPlex3 的 24 个源任务以及两个数据集的 control 细胞。sciPlex4 的 28 个 perturbed truth 在预测文件落盘后才读取。

- strict PredictionRecord：{len(record_frame)}，issues=0
- pair mean/max 下界违反：{int((~scores['triangle_mean_bound_holds']).sum())}/{int((~scores['triangle_max_bound_holds']).sum())}
- pooled disagreement ρ={main['rho_disagreement_pair_mean']:.3f}；magnitude ρ={main['rho_magnitude_pair_mean']:.3f}
- Δρ={main['delta_rho']:.3f}，bootstrap 95% CI [{main['bootstrap_delta_ci95_low']:.3f},{main['bootstrap_delta_ci95_high']:.3f}]
- CPA / interpolation 胜过零效应比例：{main['cpa_beats_zero_fraction']:.1%} / {main['interpolation_beats_zero_fraction']:.1%}

{md}

E89 只有两个共享药物、28 个目标任务，属于独立批次复核。解释时同时报告 pooled、精确剂量、插值剂量、细胞系和药物分层；不以单个分层的点估计替代总体区间。
"""
    (OUT / "reports/E89_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E89 先看这个\n\n先读 `reports/E89_REPORT.md`。\n")
    status = {"experiment": "E89_sciplex3_to_sciplex4_cpa", "phase": "evaluation_complete", "generated_at": datetime.now().isoformat(timespec="seconds"), "source_tasks": 24, "target_tasks": len(scores), "n_prediction_records": len(record_frame), "strict_issue_count": len(issues), "triangle_mean_violations": int((~scores["triangle_mean_bound_holds"]).sum()), "triangle_max_violations": int((~scores["triangle_max_bound_holds"]).sum()), "target_truth_used_for_scores": False, "target_truth_used_for_evaluation_only": True}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2)); print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "evaluate", "full"], default="full")
    parser.add_argument("--seed", type=int, default=20268900)
    parser.add_argument("--max-cells", type=int, default=32)
    parser.add_argument("--control-cells", type=int, default=64)
    parser.add_argument("--pseudo-cells", type=int, default=4)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    if args.mode in {"predict", "full"}: predict(args)
    if args.mode in {"evaluate", "full"}: evaluate(args)


if __name__ == "__main__":
    main()
