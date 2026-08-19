#!/usr/bin/env python3
"""E100: external gene-matrix replication on Lara ex vivo and Santinha.

The E99 manifest is authoritative.  Gene panels are chosen from control cells
only.  Test perturbed expression is unavailable to predictors and deployable
risk scores; validation truth is used for predictor/risk calibration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402


def load_e98_core():
    path = ROOT / "tools/scripts/run_e98_frangieh_cartesian_predictions.py"
    spec = importlib.util.spec_from_file_location("e98_cartesian_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_e98_core()
CONTRACT_ROOT = ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713"
CONTRACT = CONTRACT_ROOT / "manifests/E99_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E100_gene_external_cartesian_predictions_20260713"
TABLES, ARRAYS, REPORTS, FIGURES = OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures"
CACHE_ROOT = Path("/home/yyf/data/safeconf_e100_gene_external")
CHECKPOINT = CORE.CHECKPOINT
DATASETS = ("Lara_exvivo", "Santinha")
FRACTIONS = (25, 50, 75, 100)
N_GENES = 3000
HUMAN_ORTHOLOG_ALIAS = {"Gltscr1": "BICRA", "Dgcr14": "ESS2"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def hash_order(items: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(items).encode()).hexdigest()


def normalize_log1p(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0)
    matrix = sp.diags(scale) @ matrix
    matrix = matrix.tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def prepare_assets(dataset_name: str, source: Path, context_column: str, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = CACHE_ROOT / f"E100_{dataset_name}_CONTROL_PANEL_EFFECTS.npz"
    contexts = sorted(manifest["context"].astype(str).unique())
    perturbations = sorted(manifest["perturbation"].astype(str).unique())
    if cache.exists():
        with np.load(cache, allow_pickle=False) as store:
            assets = {key: np.asarray(store[key]) for key in store.files}
        if assets["contexts"].astype(str).tolist() == contexts and assets["perturbations"].astype(str).tolist() == perturbations:
            return assets

    data = ad.read_h5ad(source)
    obs_context = data.obs[context_column].astype(str).to_numpy()
    obs_perturbation = data.obs["perturbation"].astype(str).to_numpy()
    keep = np.isin(obs_context, contexts) & (np.isin(obs_perturbation, perturbations) | (obs_perturbation == "control"))
    x = data.X[keep]
    if not sp.issparse(x):
        x = sp.csr_matrix(np.asarray(x))
    x = normalize_log1p(x)
    kept_context = obs_context[keep]
    kept_perturbation = obs_perturbation[keep]
    control_mask = kept_perturbation == "control"
    control_mean = np.asarray(x[control_mask].mean(axis=0)).ravel()
    var_names = data.var_names.astype(str).tolist()
    var_index = {gene: index for index, gene in enumerate(var_names)}
    required = sorted({var_index[gene] for gene in perturbations if gene in var_index})
    ranked = np.argsort(-control_mean, kind="stable")
    selected = list(required)
    selected_set = set(selected)
    for index in ranked:
        if int(index) not in selected_set:
            selected.append(int(index)); selected_set.add(int(index))
        if len(selected) == N_GENES:
            break
    selected = np.asarray(selected, dtype=int)
    genes = np.asarray([var_names[index] for index in selected], dtype=str)
    x = x[:, selected]
    labels = np.asarray([f"{c}\x1f{p}" for c, p in zip(kept_context, kept_perturbation)])
    groups, codes = np.unique(labels, return_inverse=True)
    membership = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    sums = membership @ x
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    means = np.asarray(sums.multiply((1.0 / counts)[:, None]).toarray(), dtype=np.float32)
    mean_map = {label: means[index] for index, label in enumerate(groups)}
    controls = np.stack([mean_map[f"{context}\x1fcontrol"] for context in contexts]).astype(np.float32)
    effects = np.stack(
        [mean_map[f"{context}\x1f{perturbation}"] - mean_map[f"{context}\x1fcontrol"]
         for context in contexts for perturbation in perturbations]
    ).astype(np.float32)
    assets = {
        "contexts": np.asarray(contexts, dtype=str),
        "perturbations": np.asarray(perturbations, dtype=str),
        "genes": genes,
        "controls": controls,
        "effects": effects,
        "panel_selected_from_control_only": np.asarray([1], dtype=np.int8),
    }
    np.savez_compressed(cache, **assets)
    return assets


def perturbation_embeddings(perturbations: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    vocab = json.loads((CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(CHECKPOINT / "best_model.pt", map_location="cpu")
    weights = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    rows, vectors = [], []
    for perturbation in perturbations:
        human = HUMAN_ORTHOLOG_ALIAS.get(perturbation, perturbation.upper())
        if human not in vocab:
            raise RuntimeError(f"No audited human/scGPT mapping for {perturbation} -> {human}")
        vector = weights[int(vocab[human])]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
        vectors.append(vector)
        rows.append(
            {"mouse_perturbation": perturbation, "human_scgpt_token": human,
             "mapping_rule": "NCBI_audited_alias" if perturbation in HUMAN_ORTHOLOG_ALIAS else "uppercase_symbol_match",
             "scgpt_vocab_id": int(vocab[human])}
        )
    return np.stack(vectors).astype(np.float32), pd.DataFrame(rows)


def attach_test_truth(
    dataset_name: str,
    store,
    scored: pd.DataFrame,
    pred_a: dict,
    pred_b: dict,
    fold: str,
    fraction: int,
    predicted_arrays: dict[str, np.ndarray],
    true_arrays: dict[str, np.ndarray],
    dataset_group: str = "external_gene_context_cartesian",
) -> tuple[pd.DataFrame, list[dict]]:
    task_rows, records = [], []
    gene_hash = hash_order(store.genes)
    model_names = ("SourceEffect_scGPTKNN", "scGPTEmbedding_ContextRidge")
    for row in scored.itertuples(index=False):
        pair = (str(row.context), str(row.perturbation))
        truth = store.effect(*pair)
        predictions = (pred_a[pair], pred_b[pair])
        task_key = f"E100::{dataset_name}::{fold}::train{fraction}::{pair[0]}::{pair[1]}"
        true_key = task_key + "::truth"
        true_arrays[true_key] = truth.astype(np.float32)
        errors = [CORE.rmse(prediction, truth) for prediction in predictions]
        task = row._asdict()
        task.update(
            {"dataset": dataset_name, "fold_id": fold, "train_fraction": fraction / 100.0,
             "error_source_knn_rmse": errors[0], "error_context_ridge_rmse": errors[1],
             "error_two_predictor_mean_rmse": float(np.mean(errors)),
             "error_two_predictor_max_rmse": float(np.max(errors))}
        )
        task_rows.append(task)
        for name, prediction, error in zip(model_names, predictions, errors):
            record_id = task_key + f"::{name}"
            pred_key = record_id + "::prediction"
            predicted_arrays[pred_key] = prediction.astype(np.float32)
            records.append(
                {"schema_version": "safeconf_prediction_record_v1", "record_id": record_id,
                 "task_id": f"{pair[0]}::{pair[1]}", "task_key": task_key,
                 "dataset_name": f"{dataset_name}_E99_Cartesian", "dataset_group": dataset_group,
                 "fold_id": f"{fold}__train{fraction}", "split": str(row.split), "context": pair[0],
                 "perturbation": pair[1], "predictor_name": name, "run_type": "formal",
                 "gene_panel_id": f"{dataset_name}_control_selected_{N_GENES}", "gene_order_hash": gene_hash,
                 "effect_definition": "mean_diff",
                 "normalization_id": f"{dataset_name}_normalize_total_1e4_log1p_context_ctrl_mean_diff_v1",
                 "error_normalization": "raw_rmse", "predicted_effect_key": pred_key,
                 "true_effect_key": true_key, "true_error_rmse": error,
                 "true_error_cosine": CORE.cosine_error(prediction, truth), "n_cells": int(row.n_cells)}
            )
    return pd.DataFrame(task_rows), records


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scores = ["safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]
    test = tasks[tasks["split"].eq("test")]
    for (dataset, fold, fraction), fold_group in test.groupby(["dataset", "fold_id", "train_fraction"], sort=True):
        groups = list(fold_group.groupby("setting", sort=True)) + [("all_test_settings_pooled", fold_group)]
        for setting, group in groups:
            error = group["error_two_predictor_mean_rmse"].to_numpy(float)
            for score in scores:
                value = group[score].to_numpy(float)
                rows.append(
                    {"dataset": dataset, "fold_id": fold, "train_fraction": fraction, "setting": setting,
                     "score": score, "n_tasks": len(group), "mean_error": float(error.mean()),
                     "spearman": CORE.spearman(value, error),
                     "risk_coverage80_improve_pct": CORE.risk_coverage_improvement(value, error),
                     "accepted_by_validation_q80_fraction": float(group["accepted_by_validation_q80"].mean()),
                     "accepted_by_validation_q80_error": float(group.loc[group["accepted_by_validation_q80"], "error_two_predictor_mean_rmse"].mean())
                     if group["accepted_by_validation_q80"].any() else float("nan")}
                )
    return pd.DataFrame(rows)


def write_dataset_figure(dataset: str, summary: pd.DataFrame) -> None:
    original = CORE.FIGURES
    CORE.FIGURES = FIGURES
    CORE.write_svg(summary[summary["dataset"].eq(dataset)].drop(columns="dataset"))
    source = FIGURES / "F1_four_setting_risk_spearman.svg"
    target = FIGURES / f"F1_{dataset}_four_setting_risk_spearman.svg"
    source.replace(target)
    text = target.read_text(encoding="utf-8").replace("E98｜Frangieh 三背景矩阵", f"E100｜{dataset} 外部矩阵")
    target.write_text(text, encoding="utf-8")
    CORE.FIGURES = original


def write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame, status: dict) -> None:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["dataset", "setting", "score"], as_index=False)["spearman"].mean()
    macro.to_csv(TABLES / "E100_FULL_FRACTION_MACRO_SUMMARY.csv", index=False)
    lines = [
        "# E100｜两套独立多背景遗传扰动矩阵外部复制", "",
        "Lara ex vivo 与 Santinha 的任务、背景和切分来自 E99 冻结合同。每个数据集先只用所选背景的 control 细胞确定 3,000 基因面板，再统一做 library-size 10,000、log1p 和背景内 mean-difference。测试扰动细胞不参与基因选择、预测、风险特征、校准或阈值。", "",
        "## 100% 训练量", "",
        "| dataset | setting | calibrated pair risk ρ | disagreement ρ | magnitude ρ |", "|---|---|---:|---:|---:|",
    ]
    pivot = macro.pivot(index=["dataset", "setting"], columns="score", values="spearman").reset_index()
    for row in pivot.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.setting} | {row.safeconf_calibrated_pair_risk:.3f} | "
            f"{row.risk_model_disagreement:.3f} | {row.baseline_predicted_magnitude:.3f} |"
        )
    lines += ["", "## 分层聚类 bootstrap", "", "| dataset | comparator | Δρ | 95% CI | P(Δ>0) |", "|---|---|---:|---:|---:|"]
    cluster = bootstrap[bootstrap["bootstrap_unit"].eq("outer_fold_plus_perturbation_cluster")]
    for row in cluster.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.comparator} | {row.observed_macro_delta_spearman:.3f} | "
            f"[{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] | {row.bootstrap_probability_delta_gt_zero:.3f} |"
        )
    lines += [
        "", "## 合同边界", "",
        "scGPT embedding 的常规映射采用小鼠符号大写后匹配人类词表，该步骤是 symbol match，不单独声称每个基因都完成一对一同源证明。Gltscr1→BICRA 与 Dgcr14→ESS2 使用 NCBI Gene 记录核对后的别名。模型仍是 embedding/transfer predictor，不写成端到端 scGPT 或 GEARS。任何 cluster CI 跨 0 的增量都只按趋势解释。", "",
        f"strict PredictionRecord issue_count = {status['strict_issue_count']}；任务行 {status['n_task_rows']}，预测记录 {status['n_prediction_records']}。", "",
        "- `tables/E100_TASK_RISK_TABLE.csv`", "- `tables/PREDICTION_RECORDS.csv`",
        "- `tables/E100_RISK_SUMMARY.csv`", "- `tables/E100_CLUSTER_BOOTSTRAP.csv`",
        "- `tables/E100_ORTHOLOG_MAPPING.csv`", "- `figures/F1_*_four_setting_risk_spearman.svg`",
    ]
    (REPORTS / "E100_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E100 先看这个\n\n先读 `reports/E100_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    for path in (TABLES, ARRAYS, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CONTRACT, keep_default_na=False)
    sources = pd.read_csv(CONTRACT_ROOT / "tables/E99_SOURCE_AUDIT.csv", keep_default_na=False).set_index("dataset")
    predicted_arrays, true_arrays = {}, {}
    all_tasks, all_records, ridge_rows, calibration_rows, mapping_rows, asset_rows = [], [], [], [], [], []

    for dataset_name in DATASETS:
        dataset_manifest = manifest[manifest["dataset"].eq(dataset_name)].copy()
        source_row = sources.loc[dataset_name]
        assets = prepare_assets(dataset_name, Path(source_row["source_h5ad"]), str(source_row["context_column"]), dataset_manifest)
        store = CORE.EffectStore(assets)
        embeddings, mapping = perturbation_embeddings(store.perts)
        mapping.insert(0, "dataset", dataset_name); mapping_rows.append(mapping)
        features, feature_meta = CORE.make_features(store, embeddings)
        asset_rows.append(
            {"dataset": dataset_name, "n_contexts": len(store.contexts), "n_perturbations": len(store.perts),
             "n_genes": len(store.genes), "gene_order_hash": hash_order(store.genes),
             "control_only_panel": True, "feature_meta": json.dumps(feature_meta, ensure_ascii=False)}
        )

        for fold, fold_frame in dataset_manifest.groupby("fold_id", sort=True):
            validation_frame = fold_frame[fold_frame["split"].eq("val")]
            validation_pairs = list(zip(validation_frame["context"].astype(str), validation_frame["perturbation"].astype(str)))
            for fraction in FRACTIONS:
                selected = fold_frame["split"].eq("train") & fold_frame[f"in_train_fraction_{fraction}"].astype(bool)
                train_frame = fold_frame[selected]
                train_pairs = list(zip(train_frame["context"].astype(str), train_frame["perturbation"].astype(str)))
                query_frame = fold_frame[fold_frame["split"].isin(["val", "test"])].copy()
                query_pairs = list(zip(query_frame["context"].astype(str), query_frame["perturbation"].astype(str)))
                pred_a = CORE.source_knn_predictions(store, embeddings, train_pairs, query_pairs)
                ridge, alpha, alpha_table = CORE.fit_context_ridge(store, features, train_pairs, validation_pairs)
                query_x = np.stack([features[store.cix[c], store.pix[p]] for c, p in query_pairs])
                pred_b = {pair: vector.astype(np.float32) for pair, vector in zip(query_pairs, ridge.predict(query_x))}
                for row in alpha_table.itertuples(index=False):
                    ridge_rows.append(
                        {"dataset": dataset_name, "fold_id": fold, "train_fraction": fraction / 100.0,
                         "alpha": row.alpha, "validation_profile_rmse": row.validation_profile_rmse,
                         "selected": row.alpha == alpha}
                    )
                scored = CORE.score_queries_without_truth(store, train_pairs, query_frame, pred_a, pred_b)
                scored, calibration = CORE.calibrate_pair_risk_with_validation_truth(store, scored, pred_a, pred_b)
                calibration_rows.append({"dataset": dataset_name, "fold_id": fold, "train_fraction": fraction / 100.0, **calibration})
                evaluated, records = attach_test_truth(
                    dataset_name, store, scored, pred_a, pred_b, str(fold), fraction, predicted_arrays, true_arrays
                )
                all_tasks.append(evaluated); all_records.extend(records)
                print(f"[E100] {dataset_name} {fold} train={fraction}% tasks={len(evaluated)} alpha={alpha}", flush=True)

    tasks = pd.concat(all_tasks, ignore_index=True)
    records = pd.DataFrame(all_records)
    summary = summarize(tasks)
    boot_rows = []
    for dataset_name, group in tasks.groupby("dataset"):
        table = CORE.pooled_bootstrap(group, n_bootstrap=2000)
        table.insert(0, "dataset", dataset_name); boot_rows.append(table)
    bootstrap = pd.concat(boot_rows, ignore_index=True)
    tasks.to_csv(TABLES / "E100_TASK_RISK_TABLE.csv", index=False)
    records.to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    summary.to_csv(TABLES / "E100_RISK_SUMMARY.csv", index=False)
    bootstrap.to_csv(TABLES / "E100_CLUSTER_BOOTSTRAP.csv", index=False)
    pd.DataFrame(ridge_rows).to_csv(TABLES / "E100_RIDGE_VALIDATION.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(TABLES / "E100_RISK_CALIBRATION.csv", index=False)
    pd.concat(mapping_rows, ignore_index=True).to_csv(TABLES / "E100_ORTHOLOG_MAPPING.csv", index=False)
    pd.DataFrame(asset_rows).to_csv(TABLES / "E100_ASSET_AUDIT.csv", index=False)
    np.savez_compressed(ARRAYS / "predicted_effects.npz", **predicted_arrays)
    np.savez_compressed(ARRAYS / "true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E100_STRICT_CONTRACT_ISSUES.csv", index=False)
    for dataset_name in DATASETS:
        write_dataset_figure(dataset_name, summary)
    status = {
        "experiment": "E100_gene_external_cartesian_predictions", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "contract": str(CONTRACT.relative_to(ROOT)), "contract_sha256": file_sha256(CONTRACT),
        "datasets": list(DATASETS), "normalization": "normalize_total_1e4_log1p_context_specific_control_mean_difference",
        "gene_panel_selection": "top control-only mean expression plus perturbed genes; no test perturbation expression",
        "n_genes": N_GENES, "training_fractions": list(FRACTIONS),
        "n_task_rows": len(tasks), "n_prediction_records": len(records),
        "strict_issue_count": len(issues), "strict_issues": issues,
        "target_test_perturbed_truth_used_in_prediction_or_risk": False,
        "validation_truth_used_for_predictor_and_risk_calibration": True,
        "scgpt_checkpoint_sha256": file_sha256(CHECKPOINT / "best_model.pt"),
        "ortholog_aliases": HUMAN_ORTHOLOG_ALIAS,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, bootstrap, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
