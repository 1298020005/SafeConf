#!/usr/bin/env python3
"""E103: run the E98/E100 predictor-risk contract on Cui direct mappings."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E100 = load_module("e100_external_core", ROOT / "tools/scripts/run_e100_gene_external_cartesian_predictions.py")
CORE = E100.CORE
CONTRACT_ROOT = ROOT / "docs/实验结果/E102_cui_direct_mapping_contract_20260713"
CONTRACT = CONTRACT_ROOT / "manifests/E102_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E103_cui_cartesian_predictions_20260713"
TABLES, ARRAYS, REPORTS, FIGURES = OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures"
SOURCE = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/CuiHacohen2023.h5ad")
DATASET = "Cui_direct41"
FRACTIONS = (25, 50, 75, 100)


def load_embeddings(perturbations: list[str], mapping: pd.DataFrame) -> np.ndarray:
    lookup = mapping.set_index("cytokine_label")["scgpt_token"].astype(str).to_dict()
    vocab = json.loads((E100.CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(E100.CHECKPOINT / "best_model.pt", map_location="cpu")
    weights = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    vectors = []
    for perturbation in perturbations:
        token = lookup[perturbation]
        if token not in vocab:
            raise RuntimeError(f"contract token absent from checkpoint vocab: {perturbation}->{token}")
        vector = weights[int(vocab[token])]
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-8))
    return np.stack(vectors).astype(np.float32)


def write_report(summary: pd.DataFrame, bootstrap: pd.DataFrame, status: dict) -> None:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["setting", "score"], as_index=False)["spearman"].mean()
    macro.to_csv(TABLES / "E103_FULL_FRACTION_MACRO_SUMMARY.csv", index=False)
    pivot = macro.pivot(index="setting", columns="score", values="spearman").reset_index()
    lines = [
        "# E103｜Cui 六背景细胞因子刺激矩阵", "",
        "E103 使用 E102 事先冻结的 41 个直接 scGPT-token 映射刺激。目标背景 control、训练 pair 效应和预训练 token embedding 可用于预测；测试刺激后的表达在预测、风险特征和 validation 校准阶段锁定。", "",
        "## 100% 训练量", "",
        "| setting | calibrated pair risk ρ | frozen pair risk ρ | disagreement ρ | magnitude ρ |", "|---|---:|---:|---:|---:|",
    ]
    for row in pivot.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.safeconf_calibrated_pair_risk:.3f} | {row.safeconf_frozen_pair_risk:.3f} | "
            f"{row.risk_model_disagreement:.3f} | {row.baseline_predicted_magnitude:.3f} |"
        )
    lines += ["", "## 聚类 bootstrap", "", "| primary | comparator | Δρ | 95% CI | P(Δ>0) |", "|---|---|---:|---:|---:|"]
    cluster = bootstrap[bootstrap["bootstrap_unit"].eq("outer_fold_plus_perturbation_cluster")]
    for row in cluster.itertuples(index=False):
        lines.append(
            f"| {row.primary_score} | {row.comparator} | {row.observed_macro_delta_spearman:.3f} | "
            f"[{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] | {row.bootstrap_probability_delta_gt_zero:.3f} |"
        )
    lines += [
        "", "## 边界", "",
        "E103 是 cytokine stimulus，不与 gene knockout 混成同一生物主表；它回答周老师提出的“不同扰动类型都看看”。仅 41/86 个刺激有无需手工别名的直接词表映射，结果只代表该可审计子集。预测器使用 scGPT embedding，不是端到端 scGPT 或 GEARS。", "",
        f"strict PredictionRecord issue_count = {status['strict_issue_count']}；任务行 {status['n_task_rows']}，预测记录 {status['n_prediction_records']}。", "",
        "- `tables/E103_TASK_RISK_TABLE.csv`", "- `tables/PREDICTION_RECORDS.csv`",
        "- `tables/E103_RISK_SUMMARY.csv`", "- `tables/E103_CLUSTER_BOOTSTRAP.csv`",
        "- `figures/F1_Cui_direct41_four_setting_risk_spearman.svg`",
    ]
    (REPORTS / "E103_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text("# E103 先看这个\n\n先读 `reports/E103_REPORT.md`。\n", encoding="utf-8")


def main() -> None:
    for path in (TABLES, ARRAYS, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CONTRACT, keep_default_na=False)
    mapping = pd.read_csv(CONTRACT_ROOT / "tables/E102_CYTOKINE_MAPPING_AUDIT.csv", keep_default_na=False)
    assets = E100.prepare_assets(DATASET, SOURCE, "celltype", manifest)
    store = CORE.EffectStore(assets)
    embeddings = load_embeddings(store.perts, mapping)
    features, feature_meta = CORE.make_features(store, embeddings)
    predicted_arrays, true_arrays = {}, {}
    task_frames, all_records, ridge_rows, calibration_rows = [], [], [], []
    for fold, fold_frame in manifest.groupby("fold_id", sort=True):
        validation = fold_frame[fold_frame["split"].eq("val")]
        validation_pairs = list(zip(validation["context"].astype(str), validation["perturbation"].astype(str)))
        for fraction in FRACTIONS:
            selected = fold_frame["split"].eq("train") & fold_frame[f"in_train_fraction_{fraction}"].astype(bool)
            train = fold_frame[selected]
            train_pairs = list(zip(train["context"].astype(str), train["perturbation"].astype(str)))
            query = fold_frame[fold_frame["split"].isin(["val", "test"])].copy()
            query_pairs = list(zip(query["context"].astype(str), query["perturbation"].astype(str)))
            pred_a = CORE.source_knn_predictions(store, embeddings, train_pairs, query_pairs)
            ridge, alpha, alpha_table = CORE.fit_context_ridge(store, features, train_pairs, validation_pairs)
            x_query = np.stack([features[store.cix[c], store.pix[p]] for c, p in query_pairs])
            pred_b = {pair: vector.astype(np.float32) for pair, vector in zip(query_pairs, ridge.predict(x_query))}
            for row in alpha_table.itertuples(index=False):
                ridge_rows.append(
                    {"fold_id": fold, "train_fraction": fraction / 100.0, "alpha": row.alpha,
                     "validation_profile_rmse": row.validation_profile_rmse, "selected": row.alpha == alpha}
                )
            scored = CORE.score_queries_without_truth(store, train_pairs, query, pred_a, pred_b)
            scored, calibration = CORE.calibrate_pair_risk_with_validation_truth(store, scored, pred_a, pred_b)
            calibration_rows.append({"fold_id": fold, "train_fraction": fraction / 100.0, **calibration})
            evaluated, records = E100.attach_test_truth(
                DATASET, store, scored, pred_a, pred_b, str(fold), fraction, predicted_arrays, true_arrays,
                dataset_group="external_cytokine_context_cartesian",
            )
            task_frames.append(evaluated); all_records.extend(records)
            print(f"[E103] {fold} train={fraction}% tasks={len(evaluated)} alpha={alpha}", flush=True)
    tasks = pd.concat(task_frames, ignore_index=True)
    records = pd.DataFrame(all_records)
    summary = E100.summarize(tasks)
    bootstrap = CORE.pooled_bootstrap(tasks, n_bootstrap=2000)
    tasks.to_csv(TABLES / "E103_TASK_RISK_TABLE.csv", index=False)
    records.to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    summary.to_csv(TABLES / "E103_RISK_SUMMARY.csv", index=False)
    bootstrap.to_csv(TABLES / "E103_CLUSTER_BOOTSTRAP.csv", index=False)
    pd.DataFrame(ridge_rows).to_csv(TABLES / "E103_RIDGE_VALIDATION.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(TABLES / "E103_RISK_CALIBRATION.csv", index=False)
    np.savez_compressed(ARRAYS / "predicted_effects.npz", **predicted_arrays)
    np.savez_compressed(ARRAYS / "true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E103_STRICT_CONTRACT_ISSUES.csv", index=False)
    E100.FIGURES = FIGURES
    E100.write_dataset_figure(DATASET, summary)
    status = {
        "experiment": "E103_cui_cartesian_predictions", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "contract": str(CONTRACT.relative_to(ROOT)), "dataset": DATASET, "modality": "cytokine_stimulus",
        "n_contexts": len(store.contexts), "n_perturbations": len(store.perts), "n_genes": len(store.genes),
        "features": feature_meta, "n_task_rows": len(tasks), "n_prediction_records": len(records),
        "strict_issue_count": len(issues), "strict_issues": issues,
        "target_test_perturbed_truth_used_in_prediction_or_risk": False,
        "validation_truth_used_for_predictor_and_risk_calibration": True,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, bootstrap, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
