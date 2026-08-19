#!/usr/bin/env python3
"""E142: sealed Frangieh RNA-to-CITE-seq orthogonal mechanism audit.

Risk scores and all analysis choices are frozen before the protein matrix is
opened.  A train-only RNA->protein decoder translates formal scGPT/GEARS RNA
predictions into the 20 measured surface-protein readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E142_frangieh_cite_orthogonal_20260714"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
E108 = ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713"
E106 = ROOT / "docs/实验结果/E106_frangieh_context_scgpt_20260713/folds"
MANIFEST = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
PROTEIN = Path("/home/yyf/data/singlecell_perturbation_atlas/official_scperturb/FrangiehIzar2021_protein.h5ad")
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
SEED = 202607142
N_BOOTSTRAP = 3000
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.unique(a[mask]).size < 2 or np.unique(b[mask]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a[mask]), rankdata(b[mask]))[0, 1])


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def cosine_error(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1 - np.dot(a, b) / denominator) if denominator > 1e-12 else 1.0


def score_snapshot() -> pd.DataFrame:
    model = json.loads(MODEL.read_text())
    tasks = pd.read_csv(E108 / "tables/E108_TEST_TASK_RISK_TABLE.csv")
    keep = ["fold_id", "task_id", "setting", "context", "perturbation",
            "safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk",
            "risk_model_disagreement", "baseline_predicted_magnitude", *model["features_in_order"]]
    missing = sorted(set(keep) - set(tasks.columns))
    if missing:
        raise RuntimeError(f"missing deployable E108 columns: {missing}")
    result = tasks[keep].copy()
    matrix = result[model["features_in_order"]].to_numpy(float)
    result["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(model["coefficients_in_order"], float)
    result["protein_values_used_for_score_or_transform"] = False
    return result


def freeze() -> None:
    for directory in [OUT, TABLES, REPORTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    scores = score_snapshot()
    score_file = TABLES / "E142_RISK_SCORES_BEFORE_PROTEIN.csv"
    scores.to_csv(score_file, index=False)
    records = E108 / "tables/PREDICTION_RECORDS.csv"
    status = {
        "experiment": "E142_frangieh_cite_orthogonal_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_protein_matrix_opened",
        "n_test_tasks": len(scores),
        "n_folds": int(scores.fold_id.nunique()),
        "risk_score_snapshot_sha256": sha256(score_file),
        "e108_prediction_records_sha256": sha256(records),
        "e97_manifest_sha256": sha256(MANIFEST),
        "protein_h5ad_sha256": sha256(PROTEIN),
        "protein_source_shape_from_scperturb_metadata": [218331, 24],
        "protein_matrix_opened_during_freeze": False,
        "normalization": "Frangieh published isotype normalization max(ln((antibody+1)/(matched_isotype+1)),0)",
        "biological_markers": 20,
        "isotype_controls": 4,
        "decoder": "train-only StandardScaler + multi-output Ridge; alpha selected on source-context validation tasks",
        "ridge_alphas": RIDGE_ALPHAS,
        "primary_endpoints": ["protein_rmse_two_predictor_mean", "protein_cosine_error_two_predictor_mean"],
        "primary_score": "safeconf_calibrated_pair_risk",
        "comparators": ["baseline_predicted_magnitude", "risk_model_disagreement", "directional_risk_frozen"],
        "bootstrap_unit": "outer_fold_plus_perturbation_cluster",
        "n_bootstrap": N_BOOTSTRAP,
    }
    (OUT / "FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "PREREG.md").write_text(
        "# E142 预注册｜Frangieh CITE-seq 封存式跨模态验证\n\n"
        "E108 风险分数、scGPT/GEARS RNA 预测和切分已经完成；冻结本合同后才打开同批细胞的蛋白矩阵。\n\n"
        "## 固定处理\n\n"
        "- 采用原论文的 isotype normalization：`max(ln((antibody+1)/(matched IgG+1)), 0)`。\n"
        "- 4 个 IgG isotype 只作归一化/负控，20 个生物表面蛋白构成评价轴。\n"
        "- 同一 context 内，protein effect = perturbation mean − control mean；扰动标签去掉 E108 的 `+ctrl` 后匹配。\n"
        "- 每个外层 fold 仅用 train RNA 真值与 train 蛋白效应拟合 decoder；alpha 在 val 蛋白上从固定网格选择，再用 train+val 重拟合。\n"
        "- decoder 分别接收 scGPT 与 GEARS 的测试 RNA 预测，目标蛋白从不进入 SafeConf 评分。\n\n"
        "## 主终点与 gate\n\n"
        "主终点是两预测器平均 protein RMSE 与 protein cosine error。每折算 Spearman 后等权平均，按 perturbation 整簇 bootstrap 3,000 次。"
        "Gate 要求两个相关方向均为正、至少一个 95% CI 下界大于 0，且 true-RNA decoder 的 protein RMSE 优于 train-mean baseline。"
        "所有 predicted magnitude、disagreement、isotype 负控和失败 setting 原样保留。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E142 先看这个\n\n先读 `PREREG.md`；完成后读 `reports/E142_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def protein_effects():
    data = ad.read_h5ad(PROTEIN)
    var = data.var.copy()
    isotype = var["Isotype_control"].astype(str)
    biological = ~isotype.eq("nan")
    marker_names = var.index[biological].astype(str).tolist()
    all_names = var.index.astype(str).tolist()
    name_to_index = {name: index for index, name in enumerate(all_names)}
    antibody_indices = np.asarray([name_to_index[name] for name in marker_names], int)
    control_indices = np.asarray([name_to_index[value] for value in isotype[biological]], int)
    matrix = data.X
    matrix = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    antibody = matrix[:, antibody_indices].astype(np.float32)
    controls = matrix[:, control_indices].astype(np.float32)
    normalized = np.maximum(np.log((antibody + 1.0) / (controls + 1.0)), 0.0).astype(np.float32)
    raw_log = np.log1p(antibody).astype(np.float32)
    contexts = data.obs["perturbation_2"].astype(str).to_numpy()
    perturbations = data.obs["perturbation"].astype(str).to_numpy()
    labels = np.asarray([f"{context}\x1f{perturbation}" for context, perturbation in zip(contexts, perturbations)])
    groups, codes = np.unique(labels, return_inverse=True)
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    sums = np.zeros((len(groups), len(marker_names)), np.float64)
    raw_sums = np.zeros_like(sums)
    np.add.at(sums, codes, normalized)
    np.add.at(raw_sums, codes, raw_log)
    means, raw_means = sums / counts[:, None], raw_sums / counts[:, None]
    mean_map = {label: means[index].astype(np.float32) for index, label in enumerate(groups)}
    raw_map = {label: raw_means[index].astype(np.float32) for index, label in enumerate(groups)}
    count_map = {label: int(counts[index]) for index, label in enumerate(groups)}
    effects, raw_effects, cell_counts = {}, {}, {}
    for label in groups:
        context, perturbation = label.split("\x1f")
        if perturbation == "control":
            continue
        control = f"{context}\x1fcontrol"
        effects[(context, perturbation)] = mean_map[label] - mean_map[control]
        raw_effects[(context, perturbation)] = raw_map[label] - raw_map[control]
        cell_counts[(context, perturbation)] = count_map[label]
    data.file.close()
    return marker_names, effects, raw_effects, cell_counts


def fit_decoder(x_train, y_train, x_val, y_val):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    xs, ys = x_scaler.fit_transform(x_train), y_scaler.fit_transform(y_train)
    xv = x_scaler.transform(x_val)
    rows = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(xs, ys)
        predicted = y_scaler.inverse_transform(model.predict(xv))
        rows.append({"alpha": alpha, "validation_protein_rmse": rmse(predicted, y_val)})
    audit = pd.DataFrame(rows)
    best = float(audit.sort_values(["validation_protein_rmse", "alpha"]).iloc[0].alpha)
    x_all, y_all = np.vstack([x_train, x_val]), np.vstack([y_train, y_val])
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    model = Ridge(alpha=best).fit(x_scaler.fit_transform(x_all), y_scaler.fit_transform(y_all))
    return x_scaler, y_scaler, model, best, audit


def analyze_tasks(scores, marker_names, protein, raw_protein, cell_counts):
    manifest = pd.read_csv(MANIFEST)
    records = pd.read_csv(E108 / "tables/PREDICTION_RECORDS.csv")
    prediction_store = np.load(E108 / "arrays/predicted_effects.npz")
    true_store = np.load(E108 / "arrays/true_effects.npz")
    record_groups = records.groupby([records.fold_id.astype(str), records.task_id.astype(str)], sort=False)
    rows, decoder_rows = [], []
    for fold, fold_scores in scores.groupby("fold_id", sort=True):
        metrics = pd.read_csv(E106 / fold / "ALL_TASK_METRICS.csv")
        truth_vectors = np.load(E106 / fold / "true_effects.npz")
        train, val = metrics[metrics.split.eq("train")], metrics[metrics.split.eq("val")]

        def matrices(frame):
            xs, ys = [], []
            for task in frame.itertuples(index=False):
                gene = str(task.perturbation).removesuffix("+ctrl")
                xs.append(np.asarray(truth_vectors[f"{task.split}::{task.task_id}"], float))
                ys.append(protein[(str(task.context), gene)])
            return np.stack(xs), np.stack(ys)

        x_train, y_train = matrices(train); x_val, y_val = matrices(val)
        x_scaler, y_scaler, decoder, alpha, audit = fit_decoder(x_train, y_train, x_val, y_val)
        audit.insert(0, "fold_id", fold); decoder_rows.append(audit)
        train_val_mean = np.vstack([y_train, y_val]).mean(axis=0)
        for task in fold_scores.itertuples(index=False):
            gene = str(task.perturbation).removesuffix("+ctrl")
            truth_protein = protein[(str(task.context), gene)]
            truth_raw = raw_protein[(str(task.context), gene)]
            block = record_groups.get_group((str(fold), str(task.task_id)))
            predicted_proteins, errors, cosines = [], [], []
            for record in block.itertuples(index=False):
                rna = np.asarray(prediction_store[str(record.predicted_effect_key)], float).reshape(1, -1)
                pred = y_scaler.inverse_transform(decoder.predict(x_scaler.transform(rna)))[0]
                predicted_proteins.append(pred); errors.append(rmse(pred, truth_protein)); cosines.append(cosine_error(pred, truth_protein))
            true_rna = np.asarray(true_store[str(block.true_effect_key.iloc[0])], float).reshape(1, -1)
            oracle = y_scaler.inverse_transform(decoder.predict(x_scaler.transform(true_rna)))[0]
            ensemble = np.mean(np.stack(predicted_proteins), axis=0)
            top_true = int(np.argmax(np.abs(truth_protein)))
            row = task._asdict()
            row.update({
                "protein_cells": cell_counts[(str(task.context), gene)],
                "decoder_alpha": alpha,
                "protein_rmse_two_predictor_mean": float(np.mean(errors)),
                "protein_cosine_error_two_predictor_mean": float(np.mean(cosines)),
                "protein_rmse_ensemble": rmse(ensemble, truth_protein),
                "protein_rmse_true_rna_oracle": rmse(oracle, truth_protein),
                "protein_rmse_train_mean_baseline": rmse(train_val_mean, truth_protein),
                "protein_excess_rmse_over_true_rna_oracle": float(np.mean(errors) - rmse(oracle, truth_protein)),
                "strongest_true_marker": marker_names[top_true],
                "strongest_marker_match": float(np.argmax(np.abs(ensemble)) == top_true),
                "strongest_marker_sign_correct": float(np.sign(ensemble[top_true]) == np.sign(truth_protein[top_true])),
                "true_protein_effect_l2": float(np.linalg.norm(truth_protein)),
                "raw_log_protein_effect_l2_sensitivity": float(np.linalg.norm(truth_raw)),
                "protein_truth_used_for_risk_score": False,
            })
            rows.append(row)
        truth_vectors.close()
    prediction_store.close(); true_store.close()
    return pd.DataFrame(rows), pd.concat(decoder_rows, ignore_index=True)


def fold_metrics(tasks):
    scores = ["safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk", "directional_risk_frozen",
              "risk_model_disagreement", "baseline_predicted_magnitude"]
    endpoints = ["protein_rmse_two_predictor_mean", "protein_cosine_error_two_predictor_mean",
                 "protein_excess_rmse_over_true_rna_oracle"]
    rows = []
    for fold, group in tasks.groupby("fold_id", sort=True):
        for score in scores:
            for endpoint in endpoints:
                rows.append({"fold_id": fold, "score": score, "endpoint": endpoint, "n_tasks": len(group),
                             "spearman": rho(group[score], group[endpoint])})
    return pd.DataFrame(rows)


def bootstrap(tasks):
    rng = np.random.default_rng(SEED)
    primary = "safeconf_calibrated_pair_risk"
    endpoints = ["protein_rmse_two_predictor_mean", "protein_cosine_error_two_predictor_mean"]
    comparators = ["baseline_predicted_magnitude", "risk_model_disagreement", "directional_risk_frozen"]
    folds = []
    perturbations = sorted(tasks.perturbation.astype(str).unique())
    pindex = {value: index for index, value in enumerate(perturbations)}
    for _, group in tasks.groupby("fold_id", sort=False):
        folds.append({"cluster": np.asarray([pindex[value] for value in group.perturbation.astype(str)], int), "data": group.reset_index(drop=True)})
    rows = []
    for draw in range(N_BOOTSTRAP):
        counts = rng.multinomial(len(perturbations), np.full(len(perturbations), 1 / len(perturbations)))
        row = {"draw": draw}
        for endpoint in endpoints:
            estimates = {score: [] for score in [primary, *comparators]}
            for fold in folds:
                index = np.repeat(np.arange(len(fold["cluster"])), counts[fold["cluster"]])
                sample = fold["data"].iloc[index]
                for score in estimates:
                    estimates[score].append(rho(sample[score], sample[endpoint]))
            for score, values in estimates.items():
                row[f"{score}__{endpoint}"] = float(np.nanmean(values))
            for comparator in comparators:
                row[f"delta_vs_{comparator}__{endpoint}"] = row[f"{primary}__{endpoint}"] - row[f"{comparator}__{endpoint}"]
        rows.append(row)
    draws = pd.DataFrame(rows)
    summary = []
    for column in draws.columns[1:]:
        values = draws[column].to_numpy(float)
        summary.append({"metric": column, "n_bootstrap": N_BOOTSTRAP, "median": np.nanmedian(values),
                        "ci95_low": np.nanquantile(values, .025), "ci95_high": np.nanquantile(values, .975),
                        "fraction_above_zero": np.nanmean(values > 0)})
    return draws, pd.DataFrame(summary)


def report(tasks, folds, boot, markers):
    macro = folds.groupby(["score", "endpoint"], as_index=False).spearman.mean()
    b = boot.set_index("metric")
    rmse_key = "safeconf_calibrated_pair_risk__protein_rmse_two_predictor_mean"
    cosine_key = "safeconf_calibrated_pair_risk__protein_cosine_error_two_predictor_mean"
    oracle_advantage = float((tasks.protein_rmse_train_mean_baseline - tasks.protein_rmse_true_rna_oracle).mean())
    primary_positive = all(float(macro[(macro.score == "safeconf_calibrated_pair_risk") & (macro.endpoint == endpoint)].spearman.iloc[0]) > 0 for endpoint in ["protein_rmse_two_predictor_mean", "protein_cosine_error_two_predictor_mean"])
    passed = bool(primary_positive and (b.loc[[rmse_key, cosine_key], "ci95_low"] > 0).any() and oracle_advantage > 0)
    lines = [
        "# E142｜Frangieh RNA→CITE-seq 封存式跨模态验证", "", f"## 预注册 gate：{'通过' if passed else '未通过'}", "",
        f"同批 **218,331** 个细胞、20 个生物表面蛋白、{len(tasks)} 个正式测试任务。蛋白矩阵在风险分数和分析合同冻结后才打开。", "",
        "| endpoint | SafeConf fold-macro ρ | 95% CI | Δ vs magnitude (95% CI) | Δ vs disagreement (95% CI) |", "|---|---:|---:|---:|---:|",
    ]
    for endpoint, key in [("protein RMSE", rmse_key), ("protein cosine error", cosine_key)]:
        raw_endpoint = "protein_rmse_two_predictor_mean" if endpoint == "protein RMSE" else "protein_cosine_error_two_predictor_mean"
        value = float(macro[(macro.score == "safeconf_calibrated_pair_risk") & (macro.endpoint == raw_endpoint)].spearman.iloc[0])
        ci = b.loc[key]; dm = b.loc[f"delta_vs_baseline_predicted_magnitude__{raw_endpoint}"]; dd = b.loc[f"delta_vs_risk_model_disagreement__{raw_endpoint}"]
        lines.append(
            f"| {endpoint} | {value:.3f} | [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] | "
            f"{dm['median']:+.3f} [{dm['ci95_low']:+.3f}, {dm['ci95_high']:+.3f}] | "
            f"{dd['median']:+.3f} [{dd['ci95_low']:+.3f}, {dd['ci95_high']:+.3f}] |"
        )
    lines += ["", "## RNA→protein decoder 可用性", "",
              f"输入真实 RNA 效应时，decoder 相对训练蛋白均值基线的平均 RMSE 优势为 **{oracle_advantage:.4f}**（正值表示 decoder 更好）。这一步检查蛋白误差是否只是不可预测的翻译噪声。", "",
              "## 生物标志物", "", ", ".join(markers), "",
              "## 边界", "",
              "这是同一公开实验中的正交蛋白读出，不是新采集湿实验。RNA→protein decoder 只在外层 train/val 任务拟合，但蛋白丰度仍受翻译后调控和抗体噪声影响；失败 setting 和更强基线必须逐项保留。"]
    (REPORTS / "E142_REPORT.md").write_text("\n".join(lines) + "\n")
    return passed, macro, oracle_advantage


def analyze():
    freeze_status = json.loads((OUT / "FREEZE_STATUS.json").read_text())
    score_file = TABLES / "E142_RISK_SCORES_BEFORE_PROTEIN.csv"
    if sha256(score_file) != freeze_status["risk_score_snapshot_sha256"] or sha256(PROTEIN) != freeze_status["protein_h5ad_sha256"]:
        raise RuntimeError("frozen score or protein source hash changed")
    scores = pd.read_csv(score_file)
    markers, effects, raw_effects, cell_counts = protein_effects()
    expected = {(str(row.context), str(row.perturbation).removesuffix("+ctrl")) for row in scores.itertuples(index=False)}
    missing = sorted(expected - set(effects))
    if missing:
        raise RuntimeError(f"protein effects missing for {len(missing)} frozen test tasks: {missing[:5]}")
    tasks, decoder_audit = analyze_tasks(scores, markers, effects, raw_effects, cell_counts)
    folds = fold_metrics(tasks)
    draws, boot = bootstrap(tasks)
    passed, macro, oracle_advantage = report(tasks, folds, boot, markers)
    tasks.to_csv(TABLES / "E142_TASK_PROTEIN_ERRORS.csv", index=False)
    decoder_audit.to_csv(TABLES / "E142_DECODER_SELECTION.csv", index=False)
    folds.to_csv(TABLES / "E142_FOLD_METRICS.csv", index=False)
    macro.to_csv(TABLES / "E142_MACRO_SUMMARY.csv", index=False)
    draws.to_csv(TABLES / "E142_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E142_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    pd.DataFrame({"marker": markers}).to_csv(TABLES / "E142_BIOLOGICAL_PROTEIN_PANEL.csv", index=False)
    status = {**freeze_status, "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete",
              "preregistered_gate_passed": passed, "n_aligned_test_tasks": len(tasks), "n_biological_markers": len(markers),
              "true_rna_decoder_advantage_over_train_mean": oracle_advantage,
              "protein_truth_used_to_refit_or_change_safeconf": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(macro.to_string(index=False)); print(boot.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-only", action="store_true"); args = parser.parse_args()
    freeze() if args.freeze_only else analyze()


if __name__ == "__main__":
    main()
