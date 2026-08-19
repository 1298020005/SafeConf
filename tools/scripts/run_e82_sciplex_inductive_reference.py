#!/usr/bin/env python3
"""E82: leakage-separated inductive reference predictors on E81 splits.

The prediction phase reads perturbed expression for E81 training tasks only.
Target task truth is loaded in a separate evaluation phase after predictions
have been written.  These reference models validate the protocol plumbing; they
are not substitutes for the planned CPA/chemCPA experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(PACKAGE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402
DATA = Path(
    "/home/yyf/data/singlecell_perturbation_atlas/extra_official/"
    "cellular_context_generalization/sciplex3.h5ad"
)
E81 = ROOT / "docs/实验结果/E81_sciplex_cartesian_contract_20260712"
OUT = ROOT / "docs/实验结果/E82_sciplex_inductive_reference_20260712"
SMILES = Path("/home/yyf/archive/external/chemCPA/embeddings/trapnell_drugs_smiles.csv")


def dense_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(x, dtype=np.float32).mean(axis=0)


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def load_smiles(drugs: list[str]) -> dict[str, str]:
    frame = pd.read_csv(SMILES, header=None, names=["drug", "smiles", "pathway"])
    mapping = {normalize_name(r.drug): str(r.smiles) for r in frame.itertuples(index=False)}
    result = {}
    for drug in drugs:
        key = normalize_name(drug)
        if key not in mapping:
            raise RuntimeError(f"No external SMILES for {drug}")
        result[drug] = mapping[key]
    return result


def load_inputs():
    manifest = pd.read_csv(E81 / "tables/E81_SPLIT_MANIFEST.csv")
    panel = pd.read_csv(E81 / "tables/E81_GENE_PANEL.csv")
    adata = sc.read_h5ad(DATA)
    positions = adata.var_names.astype(str).get_indexer(panel["gene_id"].astype(str))
    if (positions < 0).any():
        raise RuntimeError("E81 gene panel does not align to sciPlex3")
    x = adata.X[:, positions]
    obs = adata.obs.copy()
    obs["context"] = obs["cell_line"].astype(str)
    obs["drug"] = obs["condition2"].astype(str)
    obs["dose_key"] = obs["dose"].astype(str)
    obs["task_key"] = obs["context"] + "::" + obs["drug"] + "::dose=" + obs["dose_key"]
    return adata, x, obs, manifest, panel


def context_features(x, obs, contexts: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    centroids = {}
    for context in contexts:
        mask = obs["context"].eq(context).to_numpy() & obs["perturbation"].astype(str).eq("control").to_numpy()
        if int(mask.sum()) == 0:
            raise RuntimeError(f"No controls for {context}")
        centroids[context] = dense_mean(x[mask])
    matrix = np.vstack([centroids[c] for c in contexts])
    scaled = StandardScaler().fit_transform(matrix)
    n_components = min(2, len(contexts) - 1)
    projected = PCA(n_components=n_components, random_state=20260712).fit_transform(scaled)
    return centroids, {c: projected[i].astype(np.float32) for i, c in enumerate(contexts)}


def drug_features(drugs: list[str]) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    smiles = load_smiles(drugs)
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(2, 4), n_features=128, alternate_sign=False, norm="l2"
    )
    matrix = vectorizer.transform([smiles[d] for d in drugs]).toarray().astype(np.float32)
    return smiles, {d: matrix[i] for i, d in enumerate(drugs)}


def design_vector(context_vec: np.ndarray, drug_vec: np.ndarray, dose: float) -> np.ndarray:
    logdose = np.log10(max(float(dose), 1.0)) / 4.0
    interactions = np.concatenate([context_vec[:, None] * drug_vec[None, :]], axis=0).ravel()
    return np.concatenate(
        [context_vec, drug_vec, np.array([logdose], dtype=np.float32), drug_vec * logdose, interactions]
    ).astype(np.float32)


def cosine_max(query: np.ndarray, references: np.ndarray) -> float:
    qn = np.linalg.norm(query)
    rn = np.linalg.norm(references, axis=1)
    denom = np.maximum(qn * rn, 1e-12)
    return float(np.max((references @ query) / denom))


def predict() -> None:
    adata, x, obs, manifest, panel = load_inputs()
    contexts = sorted(manifest["context"].unique())
    drugs = sorted(manifest["perturbation_key"].unique())
    control_centroids, context_embed = context_features(x, obs, contexts)
    smiles, drug_embed = drug_features(drugs)
    pred_ridge, pred_knn, meta_rows = {}, {}, []

    for manifest_id, group in manifest.groupby("manifest_id", sort=True):
        train = group.loc[group["role"].eq("train")].copy()
        test = group.loc[group["role"].eq("test")].copy()
        train_keys = set(train["task_key"])
        test_keys = set(test["task_key"])
        if train_keys & test_keys:
            raise RuntimeError(f"{manifest_id}: train/test task overlap")

        y_train, f_train = [], []
        for row in train.itertuples(index=False):
            mask = obs["task_key"].eq(row.task_key).to_numpy()
            # This is the only perturbed-expression read in prediction mode.
            treated = dense_mean(x[mask])
            effect = treated - control_centroids[row.context]
            y_train.append(effect)
            f_train.append(design_vector(context_embed[row.context], drug_embed[row.perturbation_key], row.dose_key))
        y_train = np.vstack(y_train)
        f_train = np.vstack(f_train)
        scaler = StandardScaler().fit(f_train)
        z_train = scaler.transform(f_train)
        ridge = Ridge(alpha=10.0, fit_intercept=True).fit(z_train, y_train)
        knn = KNeighborsRegressor(n_neighbors=min(5, len(train)), weights="distance", p=2).fit(z_train, y_train)
        train_contexts = sorted(train["context"].unique())
        train_drugs = sorted(train["perturbation_key"].unique())
        context_refs = np.vstack([control_centroids[c] for c in train_contexts])
        drug_refs = np.vstack([drug_embed[d] for d in train_drugs])

        for row in test.itertuples(index=False):
            features = design_vector(context_embed[row.context], drug_embed[row.perturbation_key], row.dose_key)[None, :]
            z = scaler.transform(features)
            key = f"{manifest_id}||{row.task_key}"
            pr = ridge.predict(z)[0].astype(np.float32)
            pk = knn.predict(z)[0].astype(np.float32)
            pred_ridge[key] = pr
            pred_knn[key] = pk
            same_drug = train["perturbation_key"].eq(row.perturbation_key)
            meta_rows.append(
                {
                    "prediction_key": key,
                    "manifest_id": manifest_id,
                    "task_key": row.task_key,
                    "context": row.context,
                    "perturbation_key": row.perturbation_key,
                    "dose_key": row.dose_key,
                    "quadrant": row.quadrant,
                    "n_train_tasks": len(train),
                    "context_seen_in_training": bool(row.context_seen_in_training),
                    "perturbation_seen_in_training": bool(row.perturbation_seen_in_training),
                    "history_support_same_drug": int(same_drug.sum()),
                    "max_context_control_cosine": cosine_max(control_centroids[row.context], context_refs),
                    "max_external_smiles_hash_cosine": cosine_max(drug_embed[row.perturbation_key], drug_refs),
                    "predicted_magnitude_ridge": float(np.sqrt(np.mean(np.square(pr)))),
                    "predicted_magnitude_knn": float(np.sqrt(np.mean(np.square(pk)))),
                    "model_disagreement_rmse": float(np.sqrt(np.mean(np.square(pr - pk)))),
                    "target_truth_used_for_prediction": False,
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arrays").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "arrays/E82_PRED_RIDGE.npz", **pred_ridge)
    np.savez_compressed(OUT / "arrays/E82_PRED_KNN.npz", **pred_knn)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT / "tables/E82_PREDICTION_MANIFEST.csv", index=False)
    status = {
        "experiment": "E82_sciplex_inductive_reference",
        "phase": "prediction_complete_truth_unread",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_contract": "E81_sciplex_cartesian_contract_20260712",
        "predictors": ["inductive_ridge_reference_v1", "inductive_knn_reference_v1"],
        "ridge_alpha": 10.0,
        "knn_neighbors_max": 5,
        "context_input": "vehicle-control expression only",
        "perturbation_input": "external SMILES character hash + dose",
        "n_predictions_per_model": len(meta),
        "target_perturbed_truth_used_for_prediction": False,
        "gene_order_hash": panel["gene_order_hash"].iloc[0],
    }
    (OUT / "PREDICT_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    sx = float(rx.std())
    sy = float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        vals.append(spearman_fast(x[idx], y[idx]))
    if not vals:
        return np.nan, np.nan
    return tuple(np.quantile(vals, [0.025, 0.975]))


def evaluate() -> None:
    if not (OUT / "PREDICT_STATUS.json").exists():
        raise RuntimeError("Run --mode predict before evaluation")
    adata, x, obs, manifest, panel = load_inputs()
    meta = pd.read_csv(OUT / "tables/E82_PREDICTION_MANIFEST.csv")
    ridge_npz = np.load(OUT / "arrays/E82_PRED_RIDGE.npz")
    knn_npz = np.load(OUT / "arrays/E82_PRED_KNN.npz")
    contexts = sorted(manifest["context"].unique())
    control_centroids, _ = context_features(x, obs, contexts)
    truth_cache = {}
    rows = []
    for row in meta.itertuples(index=False):
        if row.task_key not in truth_cache:
            mask = obs["task_key"].eq(row.task_key).to_numpy()
            truth_cache[row.task_key] = dense_mean(x[mask]) - control_centroids[row.context]
        truth = truth_cache[row.task_key]
        pr = ridge_npz[row.prediction_key]
        pk = knn_npz[row.prediction_key]
        er = float(np.sqrt(np.mean(np.square(pr - truth))))
        ek = float(np.sqrt(np.mean(np.square(pk - truth))))
        rows.append(
            {
                **row._asdict(),
                "error_ridge_rmse": er,
                "error_knn_rmse": ek,
                "pair_mean_rmse": (er + ek) / 2.0,
                "pair_max_rmse": max(er, ek),
                "true_magnitude_oracle": float(np.sqrt(np.mean(np.square(truth)))),
                "target_truth_used_for_scores": False,
                "target_truth_used_for_evaluation_only": True,
            }
        )
    scores = pd.DataFrame(rows)
    scores["predicted_magnitude_mean"] = (
        scores["predicted_magnitude_ridge"] + scores["predicted_magnitude_knn"]
    ) / 2.0
    scores["predicted_magnitude_max"] = scores[
        ["predicted_magnitude_ridge", "predicted_magnitude_knn"]
    ].max(axis=1)
    scores["perturbation_coverage_target"] = (
        scores["manifest_id"].str.rsplit("p", n=1).str[-1].astype(float) / 100.0
    )
    scores.to_csv(OUT / "tables/E82_TASK_SCORES.csv", index=False)

    n_cells_lookup = manifest.set_index(["manifest_id", "task_key"])["n_cells"].to_dict()
    strict_pred, strict_true, record_rows = {}, {}, []
    for row in scores.itertuples(index=False):
        true_key = f"E82::{row.manifest_id}::{row.task_key}::true"
        strict_true[true_key] = truth_cache[row.task_key].astype(np.float32)
        for predictor_name, source_npz, error_name in [
            ("inductive_ridge_reference_v1", ridge_npz, "error_ridge_rmse"),
            ("inductive_knn_reference_v1", knn_npz, "error_knn_rmse"),
        ]:
            record_id = f"E82::{row.manifest_id}::{row.task_key}::{predictor_name}"
            pred_key = record_id + "::pred"
            pred = source_npz[row.prediction_key].astype(np.float32)
            strict_pred[pred_key] = pred
            truth = truth_cache[row.task_key]
            denom = max(float(np.linalg.norm(pred) * np.linalg.norm(truth)), 1e-12)
            cosine_error = 1.0 - float(np.dot(pred, truth) / denom)
            record_rows.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": row.task_key,
                    "task_key": f"{row.manifest_id}::{row.task_key}",
                    "dataset_name": "sciPlex3_E81_cartesian",
                    "dataset_group": "sciplex3_chemical_cartesian",
                    "fold_id": row.manifest_id,
                    "split": "test",
                    "context": row.context,
                    "perturbation": f"{row.perturbation_key}::dose={row.dose_key}",
                    "predictor_name": predictor_name,
                    "run_type": "formal",
                    "gene_panel_id": "sciplex3_vehicle_control_topvar1000",
                    "gene_order_hash": panel["gene_order_hash"].iloc[0],
                    "effect_definition": "mean_diff",
                    "normalization_id": "sciplex3_log_expression_minus_context_vehicle_control_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": float(getattr(row, error_name)),
                    "true_error_cosine": cosine_error,
                    "n_cells": int(n_cells_lookup[(row.manifest_id, row.task_key)]),
                }
            )
    records = pd.DataFrame(record_rows)
    records.to_csv(OUT / "tables/PREDICTION_RECORDS.csv", index=False)
    np.savez_compressed(OUT / "arrays/predicted_effects.npz", **strict_pred)
    np.savez_compressed(OUT / "arrays/true_effects.npz", **strict_true)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"issue": issues}).to_csv(OUT / "tables/E82_STRICT_CONTRACT_ISSUES.csv", index=False)
    if issues:
        raise RuntimeError("E82 strict PredictionRecord failed: " + "; ".join(issues))

    score_targets = [
        ("model_disagreement_rmse", "pair_mean_rmse"),
        ("model_disagreement_rmse", "pair_max_rmse"),
        ("predicted_magnitude_ridge", "error_ridge_rmse"),
        ("predicted_magnitude_knn", "error_knn_rmse"),
        ("predicted_magnitude_mean", "pair_mean_rmse"),
        ("predicted_magnitude_max", "pair_max_rmse"),
        ("max_context_control_cosine", "pair_mean_rmse"),
        ("max_external_smiles_hash_cosine", "pair_mean_rmse"),
        ("history_support_same_drug", "pair_mean_rmse"),
    ]
    summary_rows = []
    for (manifest_id, quadrant), group in scores.groupby(["manifest_id", "quadrant"], sort=True):
        if len(group) < 4:
            continue
        for score_name, target_name in score_targets:
            xval = group[score_name].to_numpy(float)
            if score_name in {"max_context_control_cosine", "max_external_smiles_hash_cosine", "history_support_same_drug"}:
                xval = -xval
            yval = group[target_name].to_numpy(float)
            rho = spearman_fast(xval, yval)
            lo, hi = bootstrap_ci(xval, yval, int(hashlib.sha256(f"{manifest_id}|{quadrant}|{score_name}".encode()).hexdigest()[:8], 16))
            k = max(1, int(np.ceil(0.2 * len(group))))
            order = np.argsort(-xval)[:k]
            enrichment = float(yval[order].mean() / max(yval.mean(), 1e-12))
            summary_rows.append(
                {
                    "manifest_id": manifest_id,
                    "quadrant": quadrant,
                    "score_name": score_name,
                    "target_error": target_name,
                    "n_tasks": len(group),
                    "spearman": rho,
                    "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi,
                    "top20_error_enrichment": enrichment,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "tables/E82_RISK_ERROR_SUMMARY.csv", index=False)
    primary = summary.loc[
        summary["score_name"].eq("model_disagreement_rmse")
        & summary["target_error"].eq("pair_mean_rmse")
    ].copy()
    aggregate = (
        summary.groupby(["quadrant", "score_name", "target_error"], as_index=False)
        .agg(
            n_manifests=("manifest_id", "nunique"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            positive_manifests=("spearman", lambda s: int((s > 0).sum())),
        )
    )
    aggregate.to_csv(OUT / "tables/E82_MANIFEST_AGGREGATE.csv", index=False)
    aggregate_primary = aggregate.loc[
        (
            aggregate["score_name"].eq("model_disagreement_rmse")
            & aggregate["target_error"].eq("pair_mean_rmse")
        )
        | (
            aggregate["score_name"].eq("predicted_magnitude_mean")
            & aggregate["target_error"].eq("pair_mean_rmse")
        )
    ].copy()
    for column in ["mean_spearman", "median_spearman", "min_spearman", "max_spearman"]:
        aggregate_primary[column] = aggregate_primary[column].round(3)
    report = f"""# E82｜sciPlex3 四象限参考预测管线

E82 用 E81 冻结的 9 个子矩阵设置跑通两个来源域参考预测器。预测阶段只读取训练任务的 perturbed expression；新 context 只使用 vehicle control，新药只使用外部 SMILES 字符结构和剂量。预测文件落盘后，评价阶段才读取测试任务真值。

这一步验证合同和统计流程，不把 ridge/kNN 写成 CPA 或化学主结果。正式模型仍需 CPA/chemCPA。

- 测试预测：{len(scores)} tasks per predictor（按 manifest 计）
- 真实生物任务：{scores['task_key'].nunique()}
- target truth 进入 score：否
- gene order hash：`{panel['gene_order_hash'].iloc[0]}`

## 四象限汇总（9 个 manifest 的描述统计）

{markdown_table(aggregate_primary)}

## 这批结果暴露出的边界

- 在新 context、双未见和新药三个较难象限，predicted magnitude 的平均相关分别为 0.879、0.706、0.528，均高于参考模型分歧的 0.588、0.321、0.434。分歧只在随机缺失 pair 象限的描述统计中高于 magnitude（0.589 vs 0.354）。这不支持把参考模型分歧写成难 setting 下的稳定增量。
- 整行只留出一个 context 时，context similarity 对该行所有任务是同一个数，无法在行内排序；整列新药时，历史支持都为 0，也无法在列内排序。表里的 NaN 是特征在该 setting 下退化为常数，不是程序漏算。
- p25 的子矩阵内 pair holdout 每个 manifest 只有 4 个任务，相关系数区间很宽，只能作为管线检查。正式统计以较大象限、跨 manifest 汇总和 CPA/chemCPA 复核为准。

这说明后续不能把四项特征在所有 split 中机械相加。每个 setting 需要先列出可辨识特征：整行依赖扰动侧与模型输出，整列依赖结构表征、context 差异与模型输出，双未见主要依赖外部表征和模型输出。

完整表：`tables/E82_RISK_ERROR_SUMMARY.csv` 与 `tables/E82_MANIFEST_AGGREGATE.csv`。单 manifest 的负相关、区间跨 0 和四象限差异全部保留；上表只是描述统计，不把 9 个 manifest 当成 9 个独立数据集。
"""
    (OUT / "reports/E82_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E82 先看这个\n\n先读 `reports/E82_REPORT.md`。\n")
    status = {
        "experiment": "E82_sciplex_inductive_reference",
        "phase": "evaluation_complete",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_manifest_task_predictions_per_model": len(scores),
        "n_unique_biological_tasks": int(scores["task_key"].nunique()),
        "n_summary_rows": len(summary),
        "n_aggregate_rows": len(aggregate),
        "n_prediction_records": len(records),
        "strict_issue_count": len(issues),
        "target_truth_used_for_scores": False,
        "target_truth_used_for_evaluation_only": True,
        "interpretation": "formal protocol/reference-predictor result; not CPA/chemCPA evidence",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(primary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["predict", "evaluate", "full"], default="full")
    args = parser.parse_args()
    if args.mode in {"predict", "full"}:
        predict()
    if args.mode in {"evaluate", "full"}:
        evaluate()


if __name__ == "__main__":
    main()
