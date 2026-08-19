#!/usr/bin/env python3
"""E83: CPA-RDKit pilot on one frozen E81 Cartesian manifest.

Prediction uses E81 training-task perturbed cells, controls from all contexts,
and pseudo-test rows built only from the target context controls.  Target
perturbed cells are read only by the separate evaluation phase.
"""

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
from scipy import sparse, stats


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(PACKAGE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402

DATA = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
E81 = ROOT / "docs/实验结果/E81_sciplex_cartesian_contract_20260712"
E82 = ROOT / "docs/实验结果/E82_sciplex_inductive_reference_20260712"
OUT = ROOT / "docs/实验结果/E83_cpa_rdkit_pilot_20260712"
SMILES = Path("/home/yyf/archive/external/chemCPA/embeddings/trapnell_drugs_smiles.csv")
RUN_TYPE = "smoke"


def stable_u01(*parts: object) -> float:
    value = "||".join(map(str, parts)).encode()
    return int(hashlib.sha256(value).hexdigest()[:16], 16) / float(16**16)


def norm_name(value: str) -> str:
    return re.sub("[^a-z0-9]", "", str(value).lower())


def load_smiles() -> dict[str, str]:
    frame = pd.read_csv(SMILES, header=None, names=["drug", "smiles", "pathway"])
    return {norm_name(r.drug): str(r.smiles) for r in frame.itertuples(index=False)}


def dense_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(x, dtype=np.float32).mean(axis=0)


def sample_indices(indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, size=n, replace=False))


def base_inputs(manifest_id: str):
    manifest = pd.read_csv(E81 / "tables/E81_SPLIT_MANIFEST.csv")
    manifest = manifest.loc[manifest["manifest_id"].eq(manifest_id)].copy()
    if manifest.empty:
        raise RuntimeError(f"Unknown manifest {manifest_id}")
    panel = pd.read_csv(E81 / "tables/E81_GENE_PANEL.csv")
    source = sc.read_h5ad(DATA)
    positions = source.var_names.astype(str).get_indexer(panel["gene_id"].astype(str))
    if (positions < 0).any():
        raise RuntimeError("E81 gene panel mismatch")
    x = source.X[:, positions]
    obs = source.obs.copy()
    obs["context"] = obs["cell_line"].astype(str)
    obs["drug"] = obs["condition2"].astype(str)
    obs["dose_key"] = obs["dose"].astype(str)
    obs["task_key"] = obs["context"] + "::" + obs["drug"] + "::dose=" + obs["dose_key"]
    return source, x, obs, manifest, panel


def build_cpa_adata(manifest_id: str, max_cells: int, control_cells: int, pseudo_cells: int, seed: int):
    source, x, obs, manifest, panel = base_inputs(manifest_id)
    train_tasks = manifest.loc[manifest["role"].eq("train"), "task_key"].tolist()
    ranked = sorted(train_tasks, key=lambda task: (stable_u01(seed, "val", task), task))
    n_val = max(4, round(0.20 * len(ranked)))
    val_tasks = set(ranked[:n_val])
    train_tasks = set(ranked[n_val:])

    selected, split_by_index = [], {}
    for task in sorted(train_tasks | val_tasks):
        idx = np.flatnonzero(obs["task_key"].eq(task).to_numpy())
        keep = sample_indices(idx, max_cells, int(stable_u01(seed, task) * (2**32 - 1)))
        selected.extend(keep.tolist())
        for i in keep:
            split_by_index[i] = "valid" if task in val_tasks else "train"
    for context in sorted(manifest["context"].unique()):
        idx = np.flatnonzero(
            obs["context"].eq(context).to_numpy()
            & obs["perturbation"].astype(str).eq("control").to_numpy()
        )
        keep = sample_indices(idx, control_cells, int(stable_u01(seed, "control", context) * (2**32 - 1)))
        selected.extend(keep.tolist())
        for i in keep:
            split_by_index[i] = "train"
    selected = np.asarray(sorted(set(selected)), dtype=int)
    train_x = x[selected]
    train_obs = obs.iloc[selected].copy()
    train_obs["split_cpa"] = [split_by_index[i] for i in selected]
    train_obs["pseudo_test"] = False
    train_obs["prediction_task_key"] = ""

    pseudo_x, pseudo_rows = [], []
    test = manifest.loc[manifest["role"].eq("test")].copy()
    for row in test.itertuples(index=False):
        ctrl_idx = np.flatnonzero(
            obs["context"].eq(row.context).to_numpy()
            & obs["perturbation"].astype(str).eq("control").to_numpy()
        )
        keep = sample_indices(
            ctrl_idx, pseudo_cells, int(stable_u01(seed, "pseudo", row.task_key) * (2**32 - 1))
        )
        pseudo_x.append(x[keep])
        for i in keep:
            record = obs.iloc[i].copy()
            record["drug"] = row.perturbation_key
            record["dose_key"] = str(row.dose_key)
            record["context"] = row.context
            record["cell_line"] = row.context
            record["split_cpa"] = "test"
            record["pseudo_test"] = True
            record["prediction_task_key"] = row.task_key
            pseudo_rows.append(record)
    pseudo_x = sparse.vstack(pseudo_x) if sparse.issparse(x) else np.vstack(pseudo_x)
    pseudo_obs = pd.DataFrame(pseudo_rows)
    combined_x = sparse.vstack([train_x, pseudo_x]) if sparse.issparse(x) else np.vstack([train_x, pseudo_x])
    combined_obs = pd.concat([train_obs, pseudo_obs], ignore_index=True)

    control_mask = combined_obs["perturbation"].astype(str).eq("control") & ~combined_obs["pseudo_test"]
    combined_obs["drug_cpa"] = combined_obs["drug"].astype(str)
    combined_obs.loc[control_mask, "drug_cpa"] = "control"
    combined_obs["dose_cpa"] = combined_obs["dose_key"].astype(float).map(
        lambda value: str(np.log10(max(value, 1.0)))
    )
    combined_obs.loc[control_mask, "dose_cpa"] = "0.0"
    smiles = load_smiles()
    combined_obs["smiles_cpa"] = combined_obs["drug_cpa"].map(
        lambda value: "C" if value == "control" else smiles[norm_name(value)]
    )
    combined = ad.AnnData(X=combined_x, obs=combined_obs)
    combined.var_names = panel["gene_id"].astype(str).to_numpy()
    return combined, manifest, panel, source, x, obs, train_tasks, val_tasks


def predict(args) -> None:
    import cpa
    import torch

    combined, manifest, panel, source, x, obs, train_tasks, val_tasks = build_cpa_adata(
        args.manifest_id, args.max_cells, args.control_cells, args.pseudo_cells, args.seed
    )
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None
    cpa.CPA.setup_anndata(
        combined,
        perturbation_key="drug_cpa",
        dosage_key="dose_cpa",
        control_group="control",
        smiles_key="smiles_cpa",
        is_count_data=False,
        categorical_covariate_keys=["cell_line"],
        max_comb_len=1,
    )
    model = cpa.CPA(
        combined,
        split_key="split_cpa",
        train_split="train",
        valid_split="valid",
        test_split="test",
        use_rdkit_embeddings=True,
        n_latent=32,
        recon_loss="gauss",
        doser_type="linear",
        n_hidden_encoder=128,
        n_layers_encoder=2,
        n_hidden_decoder=128,
        n_layers_decoder=2,
        use_batch_norm_encoder=True,
        use_layer_norm_encoder=False,
        use_batch_norm_decoder=True,
        use_layer_norm_decoder=False,
        dropout_rate_encoder=0.1,
        dropout_rate_decoder=0.1,
        variational=False,
        seed=args.seed,
    )
    model.train(
        max_epochs=args.epochs,
        use_gpu=args.device,
        batch_size=args.batch_size,
        plan_kwargs={
            "n_epochs_pretrain_ae": min(5, args.epochs),
            "n_epochs_adv_warmup": min(5, args.epochs),
            "adv_steps": 1,
            "n_hidden_adv": 64,
            "n_layers_adv": 2,
            "n_epochs_verbose": 1,
            "lr": 3e-4,
        },
        save_path=False,
        check_val_every_n_epoch=1,
        early_stopping_patience=5,
        enable_progress_bar=False,
        logger=False,
    )
    test_indices = np.flatnonzero(combined.obs["split_cpa"].eq("test").to_numpy())
    pseudo = combined[test_indices].copy()
    model.predict(pseudo, batch_size=args.eval_batch_size, n_samples=1)
    predicted_expression = np.asarray(pseudo.obsm["CPA_pred"], dtype=np.float32)

    controls = {}
    for context in sorted(manifest["context"].unique()):
        mask = obs["context"].eq(context).to_numpy() & obs["perturbation"].astype(str).eq("control").to_numpy()
        controls[context] = dense_mean(x[mask])
    predictions, rows = {}, []
    pseudo_obs = pseudo.obs.reset_index(drop=True)
    for task_key, idx in pseudo_obs.groupby("prediction_task_key").groups.items():
        positions = np.asarray(list(idx), dtype=int)
        context = str(pseudo_obs.loc[positions[0], "context"])
        effect = predicted_expression[positions].mean(axis=0) - controls[context]
        key = f"E83::{args.manifest_id}::{task_key}::CPA_RDKIT::pred"
        predictions[key] = effect.astype(np.float32)
        source_row = manifest.loc[manifest["task_key"].eq(task_key)].iloc[0]
        rows.append(
            {
                "manifest_id": args.manifest_id,
                "task_key": task_key,
                "context": context,
                "perturbation_key": source_row["perturbation_key"],
                "dose_key": source_row["dose_key"],
                "quadrant": source_row["quadrant"],
                "n_cells": int(source_row["n_cells"]),
                "predicted_effect_key": key,
                "predicted_magnitude_cpa": float(np.sqrt(np.mean(np.square(effect)))),
                "target_truth_used_for_prediction": False,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arrays").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "raw_cpa").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "arrays/E83_CPA_PREDICTED_EFFECTS.npz", **predictions)
    pd.DataFrame(rows).to_csv(OUT / "tables/E83_PREDICTION_MANIFEST.csv", index=False)
    model.epoch_history.to_csv(OUT / "tables/E83_TRAINING_HISTORY.csv", index=False)
    torch.save(model.module.state_dict(), OUT / "raw_cpa/cpa_rdkit_state.pt")
    status = {
        "experiment": "E83_cpa_rdkit_pilot",
        "phase": "prediction_complete_truth_unread",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_id": args.manifest_id,
        "predictor": "CPA_0.8.8_RDKIT_Morgan_frozen_logdose",
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": len(val_tasks),
        "n_test_tasks": len(rows),
        "n_train_cells_including_controls": int(combined.obs["split_cpa"].eq("train").sum()),
        "n_val_cells": int(combined.obs["split_cpa"].eq("valid").sum()),
        "n_pseudo_test_control_cells": int(combined.obs["split_cpa"].eq("test").sum()),
        "dose_input": "log10(max(nM,1)); control=0",
        "drug_input": "frozen RDKit Morgan fingerprint from external SMILES",
        "context_input": "target and source vehicle-control cells available",
        "target_perturbed_truth_used_for_prediction": False,
        "epochs_requested": args.epochs,
        "epochs_recorded": int(model.epoch_history["epoch"].max() + 1),
        "gene_order_hash": panel["gene_order_hash"].iloc[0],
        "cpa_source_commit": "fbd7c0250edc23eff003a10c99655579c53afd63",
        "cpa_compat_patch": "optional Ray tuner import only; core model unchanged",
    }
    (OUT / "PREDICT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def spearman(x, y) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def evaluate(args) -> None:
    source, x, obs, manifest, panel = base_inputs(args.manifest_id)
    pred_meta = pd.read_csv(OUT / "tables/E83_PREDICTION_MANIFEST.csv")
    cpa_npz = np.load(OUT / "arrays/E83_CPA_PREDICTED_EFFECTS.npz")
    ridge_npz = np.load(E82 / "arrays/E82_PRED_RIDGE.npz")
    controls = {}
    for context in sorted(manifest["context"].unique()):
        mask = obs["context"].eq(context).to_numpy() & obs["perturbation"].astype(str).eq("control").to_numpy()
        controls[context] = dense_mean(x[mask])
    rows, strict_pred, strict_true, records = [], {}, {}, []
    for row in pred_meta.itertuples(index=False):
        truth = dense_mean(x[obs["task_key"].eq(row.task_key).to_numpy()]) - controls[row.context]
        cpa_pred = cpa_npz[row.predicted_effect_key]
        ridge_key = f"{args.manifest_id}||{row.task_key}"
        ridge_pred = ridge_npz[ridge_key]
        cpa_error = float(np.sqrt(np.mean(np.square(cpa_pred - truth))))
        ridge_error = float(np.sqrt(np.mean(np.square(ridge_pred - truth))))
        disagreement = float(np.sqrt(np.mean(np.square(cpa_pred - ridge_pred))))
        true_key = f"E83::{args.manifest_id}::{row.task_key}::true"
        strict_true[true_key] = truth.astype(np.float32)
        for predictor_name, prediction, error in [
            ("CPA_0.8.8_RDKIT_logdose", cpa_pred, cpa_error),
            ("inductive_ridge_reference_v1", ridge_pred, ridge_error),
        ]:
            record_id = f"E83::{args.manifest_id}::{row.task_key}::{predictor_name}"
            pred_key = record_id + "::pred"
            strict_pred[pred_key] = prediction.astype(np.float32)
            denom = max(float(np.linalg.norm(prediction) * np.linalg.norm(truth)), 1e-12)
            records.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": row.task_key,
                    "task_key": f"{args.manifest_id}::{row.task_key}",
                    "dataset_name": "sciPlex3_E81_cartesian",
                    "dataset_group": "sciplex3_chemical_cartesian",
                    "fold_id": args.manifest_id,
                    "split": "test",
                    "context": row.context,
                    "perturbation": f"{row.perturbation_key}::dose={row.dose_key}",
                    "predictor_name": predictor_name,
                    "run_type": RUN_TYPE,
                    "gene_panel_id": "sciplex3_vehicle_control_topvar1000",
                    "gene_order_hash": panel["gene_order_hash"].iloc[0],
                    "effect_definition": "mean_diff",
                    "normalization_id": "sciplex3_log_expression_minus_context_vehicle_control_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": error,
                    "true_error_cosine": 1.0 - float(np.dot(prediction, truth) / denom),
                    "n_cells": row.n_cells,
                }
            )
        rows.append(
            {
                **row._asdict(),
                "predicted_magnitude_ridge": float(np.sqrt(np.mean(np.square(ridge_pred)))),
                "cpa_ridge_disagreement_rmse": disagreement,
                "error_cpa_rmse": cpa_error,
                "error_ridge_rmse": ridge_error,
                "pair_mean_rmse": (cpa_error + ridge_error) / 2.0,
                "pair_max_rmse": max(cpa_error, ridge_error),
                "true_magnitude_oracle": float(np.sqrt(np.mean(np.square(truth)))),
                "target_truth_used_for_scores": False,
            }
        )
    scores = pd.DataFrame(rows)
    scores["predicted_magnitude_mean"] = (
        scores["predicted_magnitude_cpa"] + scores["predicted_magnitude_ridge"]
    ) / 2.0
    scores["predicted_magnitude_max"] = scores[
        ["predicted_magnitude_cpa", "predicted_magnitude_ridge"]
    ].max(axis=1)
    scores.to_csv(OUT / "tables/E83_TASK_SCORES.csv", index=False)
    record_frame = pd.DataFrame(records)
    record_frame.to_csv(OUT / "tables/PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT / "arrays/predicted_effects.npz", **strict_pred)
    np.savez_compressed(OUT / "arrays/true_effects.npz", **strict_true)
    issues = validate_prediction_record_artifacts(OUT, records=record_frame, strict=True)
    pd.DataFrame({"issue": issues}).to_csv(OUT / "tables/E83_STRICT_CONTRACT_ISSUES.csv", index=False)
    if issues:
        raise RuntimeError("E83 strict contract failed: " + "; ".join(issues))

    summary_rows = []
    for quadrant, group in scores.groupby("quadrant"):
        for score, target in [
            ("cpa_ridge_disagreement_rmse", "pair_mean_rmse"),
            ("cpa_ridge_disagreement_rmse", "pair_max_rmse"),
            ("predicted_magnitude_cpa", "error_cpa_rmse"),
            ("predicted_magnitude_ridge", "error_ridge_rmse"),
            ("predicted_magnitude_mean", "pair_mean_rmse"),
            ("predicted_magnitude_max", "pair_max_rmse"),
        ]:
            summary_rows.append(
                {
                    "quadrant": quadrant,
                    "score_name": score,
                    "target_error": target,
                    "n_tasks": len(group),
                    "spearman": spearman(group[score].to_numpy(float), group[target].to_numpy(float)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tables/E83_RISK_ERROR_SUMMARY.csv", index=False)
    primary = summary.loc[
        summary["score_name"].eq("cpa_ridge_disagreement_rmse")
        & summary["target_error"].eq("pair_mean_rmse")
    ]
    report = f"""# E83｜CPA-RDKit 化学四象限 pilot

E83 在 E81 的 `{args.manifest_id}` 上训练官方 CPA 0.8.8。模型只读取冻结训练任务、各 context 的 vehicle control、外部 SMILES 和 log10-dose；测试任务的 perturbed expression 在预测文件落盘后才用于评价。本轮用于开发管线，run_type=`{RUN_TYPE}`，不进入正式汇总。

- CPA 与 ridge 共享任务、1000 基因顺序和 true effect
- strict PredictionRecord：{len(record_frame)}，问题数 0
- target truth 进入 score：否
- 训练输入修正：原始 nM 会使 linear doser 数值溢出，因此预先固定为 log10-dose；该规则未读取测试误差

## CPA–ridge 分歧对 pair mean error

{primary.to_csv(index=False)}

这是单 manifest pilot。测试结果已经查看，因此该 manifest 永久保留为开发集，不再通过增加 epoch 或改参数升级成正式证据。正式 E84 固定参数后只运行其余 8 个未查看 manifest。
"""
    (OUT / "reports/E83_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E83 先看这个\n\n先读 `reports/E83_REPORT.md`。\n")
    status = {
        "experiment": "E83_cpa_rdkit_pilot",
        "phase": "evaluation_complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_id": args.manifest_id,
        "n_tasks": len(scores),
        "n_prediction_records": len(record_frame),
        "strict_issue_count": len(issues),
        "run_type": RUN_TYPE,
        "target_truth_used_for_scores": False,
        "target_truth_used_for_evaluation_only": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(primary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "evaluate", "full"], default="full")
    parser.add_argument("--manifest-id", default="E81_r1_p75")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-cells", type=int, default=32)
    parser.add_argument("--control-cells", type=int, default=64)
    parser.add_argument("--pseudo-cells", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.mode in {"predict", "full"}:
        predict(args)
    if args.mode in {"evaluate", "full"}:
        evaluate(args)


if __name__ == "__main__":
    main()
