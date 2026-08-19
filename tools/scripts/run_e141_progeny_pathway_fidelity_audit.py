#!/usr/bin/env python3
"""E141: frozen seven-dataset PROGENy pathway-fidelity audit.

The freeze phase snapshots deployable scores and the pathway resource before any
saved prediction/truth vector is opened.  The analysis phase projects the saved
RNA effects into signed PROGENy activities and evaluates pathway-level errors.
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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E141_progeny_pathway_fidelity_20260714"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
RESOURCE = Path("/home/yyf/data/safeconf_mechanism/progeny_omnipath_20260714.tsv")
RESOURCE_URL = "https://omnipathdb.org/annotations?resources=PROGENy&format=tsv"
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
SEED = 202607141
N_BOOTSTRAP = 3000
TOP_GENES_PER_PATHWAY = 500
MIN_PANEL_GENES = 5


DATASETS = {
    "Frangieh": {
        "root": ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713",
        "tasks": "tables/E108_TEST_TASK_RISK_TABLE.csv",
        "records": "tables/PREDICTION_RECORDS.csv",
        "panel_h5ad": Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad"),
    },
    "Lara_exvivo": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Santinha": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Shifrut": {
        "root": ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Liang": {
        "root": ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Tian_CRISPRi": {
        "root": ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Nadig_two_cellline": {
        "root": ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
}


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


def cosine_error(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1 - np.dot(a, b) / denominator) if denominator > 1e-12 else 1.0


def panel_genes(spec: dict) -> list[str]:
    if "panel_h5ad" in spec:
        data = ad.read_h5ad(spec["panel_h5ad"], backed="r")
        genes = data.var["gene_name"].astype(str).tolist()
        data.file.close()
        return genes
    panel = pd.read_csv(spec["root"] / spec["panel"])
    return panel["scgpt_token"].astype(str).tolist()


def pathway_resource() -> pd.DataFrame:
    raw = pd.read_csv(RESOURCE, sep="\t")
    wide = raw.pivot_table(index=["record_id", "genesymbol"], columns="label", values="value", aggfunc="first").reset_index()
    wide["p_value"] = pd.to_numeric(wide["p_value"], errors="coerce")
    wide["weight"] = pd.to_numeric(wide["weight"], errors="coerce")
    wide = wide.dropna(subset=["pathway", "p_value", "weight"])
    return wide.sort_values(["pathway", "p_value", "genesymbol"]).groupby("pathway", sort=True).head(TOP_GENES_PER_PATHWAY)


def score_snapshot() -> pd.DataFrame:
    model = json.loads(MODEL.read_text())
    frames = []
    for dataset, spec in DATASETS.items():
        columns = ["fold_id", "task_id", "setting", "context", "perturbation",
                   "safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk",
                   "risk_model_disagreement", "baseline_predicted_magnitude", *model["features_in_order"]]
        task = pd.read_csv(spec["root"] / spec["tasks"])
        missing = sorted(set(columns) - set(task.columns))
        if missing:
            raise RuntimeError(f"{dataset}: missing deployable columns {missing}")
        task = task[columns].copy()
        matrix = task[model["features_in_order"]].to_numpy(float)
        task["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(model["coefficients_in_order"], float)
        task["dataset"] = dataset
        task["target_truth_used_for_score_or_transform"] = False
        frames.append(task)
    return pd.concat(frames, ignore_index=True, sort=False)


def freeze() -> None:
    for directory in [OUT, TABLES, REPORTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    if not RESOURCE.exists():
        raise FileNotFoundError(f"download the frozen PROGENy snapshot first: {RESOURCE_URL}")
    scores = score_snapshot()
    score_file = TABLES / "E141_SCORES_BEFORE_VECTOR_TRUTH.csv"
    scores.to_csv(score_file, index=False)
    status = {
        "experiment": "E141_progeny_pathway_fidelity",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_prediction_or_truth_vectors_opened",
        "n_datasets": int(scores.dataset.nunique()),
        "n_tasks": len(scores),
        "progeny_source_url": RESOURCE_URL,
        "progeny_snapshot_path": str(RESOURCE),
        "progeny_snapshot_sha256": sha256(RESOURCE),
        "frozen_direction_model_sha256": sha256(MODEL),
        "score_snapshot_sha256": sha256(score_file),
        "top_genes_per_pathway": TOP_GENES_PER_PATHWAY,
        "minimum_panel_overlap": MIN_PANEL_GENES,
        "primary_pairs": [
            ["safeconf_calibrated_pair_risk", "progeny_activity_rmse_mean"],
            ["directional_risk_frozen", "progeny_activity_cosine_error_mean"],
        ],
        "comparators": ["baseline_predicted_magnitude", "risk_model_disagreement"],
        "bootstrap_unit": "dataset_then_perturbation_cluster_with_fold_macro",
        "n_bootstrap": N_BOOTSTRAP,
        "prediction_or_truth_vector_files_opened": [],
    }
    (OUT / "FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "PREREG.md").write_text(
        "# E141 预注册｜七数据 PROGENy 通路忠实度\n\n"
        "冻结顺序：PROGENy 资源快照与部署分数 → 本合同 → 打开预测/真值向量 → 通路误差评价。\n\n"
        "- PROGENy 每条通路按资源 p value 固定取前 500 个响应基因；与 512 基因面板重叠不少于 5 个才纳入。\n"
        "- 权重在每个数据集面板内做 L2 归一化，通路活性为 signed weighted projection。\n"
        "- absolute 主终点：原 SafeConf 与两预测器平均通路活性 RMSE 的 fold→dataset 等权 Spearman。\n"
        "- direction 主终点：冻结 Directional-SafeConf 与两预测器平均通路活性 cosine error 的同层级 Spearman。\n"
        "- 以 perturbation 为整簇、dataset 为总体层做 3,000 次 bootstrap，同时报告 predicted magnitude 与 disagreement。\n"
        "- 通过标准：两个主相关方向均为正，至少一个 95% CI 下界大于 0；不通过也不得改分数后重报。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E141 先看这个\n\n冻结合同见 `PREREG.md`；分析完成后读 `reports/E141_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def pathway_matrix(genes: list[str], resource: pd.DataFrame):
    gene_index = {str(gene).upper(): index for index, gene in enumerate(genes)}
    rows, names, coverage = [], [], []
    for pathway, group in resource.groupby("pathway", sort=True):
        vector = np.zeros(len(genes), np.float64)
        for row in group.itertuples(index=False):
            index = gene_index.get(str(row.genesymbol).upper())
            if index is not None:
                vector[index] = float(row.weight)
        overlap = int(np.count_nonzero(vector))
        if overlap < MIN_PANEL_GENES:
            continue
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        rows.append(vector); names.append(str(pathway)); coverage.append(overlap)
    return np.stack(rows), names, coverage


def analyze_dataset(dataset: str, spec: dict, scores: pd.DataFrame, resource: pd.DataFrame):
    genes = panel_genes(spec)
    weights, pathways, coverage = pathway_matrix(genes, resource)
    root = spec["root"]
    records = pd.read_csv(root / spec["records"])
    predictions = np.load(root / "arrays/predicted_effects.npz")
    truths = np.load(root / "arrays/true_effects.npz")
    lookup = records.groupby([records.fold_id.astype(str), records.task_id.astype(str)], sort=False)
    rows, contributions = [], []
    selected = scores[scores.dataset.eq(dataset)]
    for task in selected.itertuples(index=False):
        block = lookup.get_group((str(task.fold_id), str(task.task_id)))
        truth_key = str(block.true_effect_key.iloc[0])
        truth_activity = weights @ np.asarray(truths[truth_key], float)
        rmses, cosines, top_matches, signs = [], [], [], []
        prediction_activities = []
        for record in block.itertuples(index=False):
            activity = weights @ np.asarray(predictions[str(record.predicted_effect_key)], float)
            prediction_activities.append(activity)
            rmses.append(float(np.sqrt(np.mean((activity - truth_activity) ** 2))))
            cosines.append(cosine_error(activity, truth_activity))
            top_true, top_pred = int(np.argmax(np.abs(truth_activity))), int(np.argmax(np.abs(activity)))
            top_matches.append(float(top_true == top_pred))
            signs.append(float(np.sign(truth_activity[top_true]) == np.sign(activity[top_true])))
        row = task._asdict()
        row.update({
            "n_progeny_pathways": len(pathways),
            "progeny_activity_rmse_mean": float(np.mean(rmses)),
            "progeny_activity_cosine_error_mean": float(np.mean(cosines)),
            "strongest_true_pathway_match_fraction": float(np.mean(top_matches)),
            "strongest_true_pathway_sign_fraction": float(np.mean(signs)),
            "strongest_true_pathway": pathways[int(np.argmax(np.abs(truth_activity)))],
            "truth_pathway_activity_l2": float(np.linalg.norm(truth_activity)),
        })
        rows.append(row)
        ensemble_activity = np.mean(np.stack(prediction_activities), axis=0)
        residual = np.abs(ensemble_activity - truth_activity)
        for index, pathway in enumerate(pathways):
            contributions.append({"dataset": dataset, "fold_id": task.fold_id, "task_id": task.task_id,
                                  "perturbation": task.perturbation, "pathway": pathway,
                                  "panel_overlap_genes": coverage[index], "true_activity": truth_activity[index],
                                  "ensemble_predicted_activity": ensemble_activity[index], "absolute_residual": residual[index]})
    predictions.close(); truths.close()
    return pd.DataFrame(rows), pd.DataFrame(contributions), pd.DataFrame({"dataset": dataset, "pathway": pathways, "panel_overlap_genes": coverage})


def fold_metrics(tasks: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("safeconf_calibrated_pair_risk", "progeny_activity_rmse_mean"),
        ("directional_risk_frozen", "progeny_activity_cosine_error_mean"),
    ]
    scores = ["safeconf_calibrated_pair_risk", "directional_risk_frozen", "baseline_predicted_magnitude", "risk_model_disagreement"]
    endpoints = sorted({endpoint for _, endpoint in pairs})
    rows = []
    for (dataset, fold), group in tasks.groupby(["dataset", "fold_id"], sort=True):
        for score in scores:
            for endpoint in endpoints:
                rows.append({"dataset": dataset, "fold_id": fold, "score": score, "endpoint": endpoint,
                             "n_tasks": len(group), "spearman": rho(group[score], group[endpoint])})
    return pd.DataFrame(rows)


def hierarchical_bootstrap(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    datasets = sorted(tasks.dataset.unique())
    pairs = [
        ("safeconf_calibrated_pair_risk", "progeny_activity_rmse_mean"),
        ("directional_risk_frozen", "progeny_activity_cosine_error_mean"),
    ]
    cache = {}
    for dataset in datasets:
        data = tasks[tasks.dataset.eq(dataset)]
        perturbations = sorted(data.perturbation.astype(str).unique())
        perturbation_index = {value: index for index, value in enumerate(perturbations)}
        folds = []
        for _, fold in data.groupby("fold_id", sort=False):
            folds.append({
                "cluster": np.asarray([perturbation_index[value] for value in fold.perturbation.astype(str)], int),
                "safeconf": fold["safeconf_calibrated_pair_risk"].to_numpy(float),
                "directional": fold["directional_risk_frozen"].to_numpy(float),
                "magnitude": fold["baseline_predicted_magnitude"].to_numpy(float),
                "rmse": fold["progeny_activity_rmse_mean"].to_numpy(float),
                "cosine": fold["progeny_activity_cosine_error_mean"].to_numpy(float),
            })
        cache[dataset] = {"n_clusters": len(perturbations), "folds": folds}
    draws = []
    for draw in range(N_BOOTSTRAP):
        row = {"draw": draw}
        sampled_datasets = rng.choice(datasets, len(datasets), replace=True)
        for score, endpoint in pairs:
            dataset_values, magnitude_values = [], []
            for dataset in sampled_datasets:
                item = cache[str(dataset)]
                n_clusters = item["n_clusters"]
                counts = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters))
                fold_values, fold_magnitude = [], []
                score_key = "safeconf" if score == "safeconf_calibrated_pair_risk" else "directional"
                endpoint_key = "rmse" if endpoint == "progeny_activity_rmse_mean" else "cosine"
                for fold in item["folds"]:
                    indices = np.repeat(np.arange(len(fold["cluster"])), counts[fold["cluster"]])
                    fold_values.append(rho(fold[score_key][indices], fold[endpoint_key][indices]))
                    fold_magnitude.append(rho(fold["magnitude"][indices], fold[endpoint_key][indices]))
                dataset_values.append(float(np.nanmean(fold_values)))
                magnitude_values.append(float(np.nanmean(fold_magnitude)))
            key = f"{score}__{endpoint}"
            row[key] = float(np.nanmean(dataset_values))
            row[f"delta_vs_magnitude__{key}"] = float(np.nanmean(np.asarray(dataset_values) - np.asarray(magnitude_values)))
        draws.append(row)
    frame = pd.DataFrame(draws)
    summary = []
    for column in frame.columns[1:]:
        values = frame[column].to_numpy(float)
        summary.append({"metric": column, "n_bootstrap": N_BOOTSTRAP, "median": np.nanmedian(values),
                        "ci95_low": np.nanquantile(values, .025), "ci95_high": np.nanquantile(values, .975),
                        "fraction_above_zero": np.nanmean(values > 0)})
    return frame, pd.DataFrame(summary)


def write_report(tasks, folds, coverage, pathway_contrib, boot):
    dataset_macro = folds.groupby(["dataset", "score", "endpoint"], as_index=False).spearman.mean()
    overall = dataset_macro.groupby(["score", "endpoint"], as_index=False).spearman.mean()
    primary = overall[((overall.score == "safeconf_calibrated_pair_risk") & (overall.endpoint == "progeny_activity_rmse_mean")) |
                      ((overall.score == "directional_risk_frozen") & (overall.endpoint == "progeny_activity_cosine_error_mean"))]
    b = boot.set_index("metric")
    absolute_key = "safeconf_calibrated_pair_risk__progeny_activity_rmse_mean"
    direction_key = "directional_risk_frozen__progeny_activity_cosine_error_mean"
    passed = bool((primary.spearman > 0).all() and (b.loc[[absolute_key, direction_key], "ci95_low"] > 0).any())
    lines = [
        "# E141｜七数据 PROGENy 通路忠实度审计", "", f"## 预注册 gate：{'通过' if passed else '未通过'}", "",
        "该分析把每条 RNA 预测投影到 14 类有符号 PROGENy 通路活动；资源、风险分数和分析规则在打开预测/真值向量前冻结。", "",
        "| 风险分数 | 通路误差 | 七数据等权 ρ | 95% CI | 相对 magnitude Δρ（95% CI） |", "|---|---|---:|---:|---:|",
    ]
    for score, endpoint, key in [("safeconf_calibrated_pair_risk", "progeny_activity_rmse_mean", absolute_key),
                                 ("directional_risk_frozen", "progeny_activity_cosine_error_mean", direction_key)]:
        value = float(overall[(overall.score == score) & (overall.endpoint == endpoint)].spearman.iloc[0])
        ci = b.loc[key]; delta = b.loc[f"delta_vs_magnitude__{key}"]
        lines.append(
            f"| {score} | {endpoint} | {value:.3f} | [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] | "
            f"{delta['median']:+.3f} [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}] |"
        )
    lines += ["", "## 每数据集", "", "| dataset | SafeConf→pathway RMSE | Directional→pathway cosine | pathways |", "|---|---:|---:|---:|"]
    cov_count = coverage.groupby("dataset").pathway.nunique()
    for dataset in sorted(tasks.dataset.unique()):
        a = dataset_macro[(dataset_macro.dataset == dataset) & (dataset_macro.score == "safeconf_calibrated_pair_risk") & (dataset_macro.endpoint == "progeny_activity_rmse_mean")].spearman.iloc[0]
        d = dataset_macro[(dataset_macro.dataset == dataset) & (dataset_macro.score == "directional_risk_frozen") & (dataset_macro.endpoint == "progeny_activity_cosine_error_mean")].spearman.iloc[0]
        lines.append(f"| {dataset} | {a:.3f} | {d:.3f} | {int(cov_count[dataset])} |")
    high = tasks.assign(high=lambda x: x.groupby(["dataset", "fold_id"])["safeconf_calibrated_pair_risk"].rank(pct=True) >= .8)
    case_ids = set(high.loc[high.high, "task_id"])
    cases = pathway_contrib[pathway_contrib.task_id.isin(case_ids)].groupby("pathway", as_index=False).absolute_residual.mean().sort_values("absolute_residual", ascending=False).head(8)
    lines += ["", "## 高风险任务中残差较大的通路", ""]
    for row in cases.itertuples(index=False):
        lines.append(f"- {row.pathway}: mean absolute activity residual={row.absolute_residual:.4f}")
    lines += ["", "## 边界", "", "PROGENy 是由外部扰动实验得到的转录响应足迹，不等于蛋白磷酸化或因果通路真值。512 基因固定面板使部分通路只能由少量响应基因覆盖；覆盖数逐数据集完整落盘，不能把本分析写成湿实验因果证明。"]
    (REPORTS / "E141_REPORT.md").write_text("\n".join(lines) + "\n")
    return passed, dataset_macro, overall


def analyze() -> None:
    freeze_status = json.loads((OUT / "FREEZE_STATUS.json").read_text())
    score_path = TABLES / "E141_SCORES_BEFORE_VECTOR_TRUTH.csv"
    if sha256(score_path) != freeze_status["score_snapshot_sha256"] or sha256(RESOURCE) != freeze_status["progeny_snapshot_sha256"]:
        raise RuntimeError("frozen score or PROGENy resource hash changed")
    scores, resource = pd.read_csv(score_path), pathway_resource()
    task_parts, contribution_parts, coverage_parts = [], [], []
    for dataset, spec in DATASETS.items():
        print(f"[E141] {dataset}", flush=True)
        task, contribution, coverage = analyze_dataset(dataset, spec, scores, resource)
        task_parts.append(task); contribution_parts.append(contribution); coverage_parts.append(coverage)
    tasks = pd.concat(task_parts, ignore_index=True, sort=False)
    contributions = pd.concat(contribution_parts, ignore_index=True, sort=False)
    coverage = pd.concat(coverage_parts, ignore_index=True, sort=False)
    folds = fold_metrics(tasks)
    draws, boot = hierarchical_bootstrap(tasks)
    passed, dataset_macro, overall = write_report(tasks, folds, coverage, contributions, boot)
    tasks.to_csv(TABLES / "E141_TASK_PATHWAY_ERRORS.csv", index=False)
    contributions.to_csv(TABLES / "E141_TASK_PATHWAY_CONTRIBUTIONS.csv", index=False)
    coverage.to_csv(TABLES / "E141_PATHWAY_COVERAGE.csv", index=False)
    folds.to_csv(TABLES / "E141_FOLD_METRICS.csv", index=False)
    dataset_macro.to_csv(TABLES / "E141_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E141_SEVEN_DATASET_MACRO.csv", index=False)
    draws.to_csv(TABLES / "E141_HIERARCHICAL_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E141_HIERARCHICAL_BOOTSTRAP_SUMMARY.csv", index=False)
    status = {
        **freeze_status, "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete",
        "preregistered_gate_passed": passed, "n_tasks": len(tasks), "n_pathway_task_rows": len(contributions),
        "score_or_predictor_refit": False, "test_truth_used_to_change_score": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(boot.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze_only else analyze()


if __name__ == "__main__":
    main()
