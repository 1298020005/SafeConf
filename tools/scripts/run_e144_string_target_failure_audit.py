#!/usr/bin/env python3
"""E144: STRING knowledge-distance and perturbed-target recovery audit.

Freeze mode builds deployable graph features without opening saved target truth
vectors.  Analyze mode then measures whether SafeConf identifies failure to
recover the perturbed gene itself and whether distance from training targets is
associated with whole-vector prediction error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E144_string_target_failure_20260714"
TABLES, REPORTS = OUT / "tables", OUT / "reports"
STRING_ROOT = Path("/home/yyf/data/safeconf_mechanism/string_v12_human")
STRING_INFO = STRING_ROOT / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS = STRING_ROOT / "9606.protein.links.v12.0.txt.gz"
STRING_VERSION = "12.0"
STRING_SCORE_THRESHOLD = 700
STRING_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
STRING_LINKS_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
SEED = 202607144
N_BOOTSTRAP = 3000
N_LABEL_PERMUTATIONS = 1000


DATASETS = {
    "Frangieh": {
        "root": ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713",
        "tasks": "tables/E108_TEST_TASK_RISK_TABLE.csv", "records": "tables/PREDICTION_RECORDS.csv",
        "manifest": ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv",
        "panel_h5ad": Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad"),
    },
    "Lara_exvivo": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv",
        "manifest_dataset": "Lara_exvivo",
    },
    "Santinha": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E99_multicontext_external_contract_20260713/manifests/E99_TASK_MANIFEST.csv",
        "manifest_dataset": "Santinha",
    },
    "Shifrut": {
        "root": ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E119_shifrut_four_context_contract_20260714/manifests/E119_TASK_MANIFEST.csv",
    },
    "Liang": {
        "root": ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E122_liang_nine_context_contract_20260714/manifests/E122_TASK_MANIFEST.csv",
    },
    "Tian_CRISPRi": {
        "root": ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E128_tian_crispri_four_batch_contract_20260714/manifests/E128_TASK_MANIFEST.csv",
    },
    "Nadig_two_cellline": {
        "root": ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline",
        "tasks": "TASK_RISK_TABLE.csv", "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
        "manifest": ROOT / "docs/实验结果/E136_nadig_two_cellline_contract_20260714/manifests/E136_TASK_MANIFEST.csv",
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


def target_symbol(value: str) -> str | None:
    parts = [part.strip() for part in str(value).replace(",", "+").split("+")]
    genes = [part for part in parts if part and part.lower() not in {"ctrl", "control", "nt", "non-targeting"}]
    return genes[0].upper() if len(genes) == 1 else None


def panel_genes(spec: dict) -> list[str]:
    if "panel_h5ad" in spec:
        data = ad.read_h5ad(spec["panel_h5ad"], backed="r")
        genes = data.var["gene_name"].astype(str).tolist()
        data.file.close()
        return genes
    return pd.read_csv(spec["root"] / spec["panel"])["scgpt_token"].astype(str).tolist()


def load_string_graph():
    info = pd.read_csv(STRING_INFO, sep="\t")
    protein_ids = info["#string_protein_id"].astype(str).tolist()
    id_to_index = {protein: index for index, protein in enumerate(protein_ids)}
    symbol_to_nodes: dict[str, list[int]] = {}
    for index, symbol in enumerate(info["preferred_name"].astype(str)):
        symbol_to_nodes.setdefault(symbol.upper(), []).append(index)
    adjacency: list[list[int]] = [[] for _ in protein_ids]
    retained = 0
    with gzip.open(STRING_LINKS, "rt") as handle:
        next(handle)
        for line in handle:
            protein1, protein2, score = line.rstrip().split()
            if int(score) < STRING_SCORE_THRESHOLD:
                continue
            left, right = id_to_index.get(protein1), id_to_index.get(protein2)
            if left is None or right is None or left == right:
                continue
            adjacency[left].append(right)
            adjacency[right].append(left)
            retained += 1
    return info, symbol_to_nodes, adjacency, retained


def multi_source_distance(adjacency: list[list[int]], sources: set[int]) -> np.ndarray:
    distance = np.full(len(adjacency), -1, np.int16)
    queue: deque[int] = deque()
    for source in sources:
        distance[source] = 0
        queue.append(source)
    while queue:
        node = queue.popleft()
        next_distance = distance[node] + 1
        for neighbour in adjacency[node]:
            if distance[neighbour] < 0:
                distance[neighbour] = next_distance
                queue.append(neighbour)
    return distance


def manifest_for_dataset(spec: dict) -> pd.DataFrame:
    manifest = pd.read_csv(spec["manifest"])
    if "manifest_dataset" in spec:
        manifest = manifest[manifest["dataset"].astype(str).eq(spec["manifest_dataset"])]
    return manifest


def build_deployable_network_features() -> tuple[pd.DataFrame, dict]:
    info, symbol_to_nodes, adjacency, retained_edges = load_string_graph()
    degree = np.asarray([len(set(neighbours)) for neighbours in adjacency], int)
    parts = []
    for dataset, spec in DATASETS.items():
        task = pd.read_csv(spec["root"] / spec["tasks"])
        columns = ["fold_id", "task_id", "split", "setting", "context", "perturbation",
                   "safeconf_calibrated_pair_risk", "safeconf_frozen_pair_risk",
                   "risk_model_disagreement", "baseline_predicted_magnitude"]
        missing = sorted(set(columns) - set(task.columns))
        if missing:
            raise RuntimeError(f"{dataset}: missing deployable columns {missing}")
        task = task[columns].copy()
        task["dataset"] = dataset
        task["target_symbol"] = task["perturbation"].map(target_symbol)
        manifest = manifest_for_dataset(spec)
        rows = []
        for fold_id, fold in task.groupby("fold_id", sort=True):
            fold_manifest = manifest[manifest["fold_id"].astype(str).eq(str(fold_id))]
            train_symbols = {target_symbol(value) for value in fold_manifest.loc[fold_manifest["split"].eq("train"), "perturbation"]}
            train_symbols.discard(None)
            sources = {node for symbol in train_symbols for node in symbol_to_nodes.get(symbol, [])}
            distances = multi_source_distance(adjacency, sources) if sources else np.full(len(adjacency), -1, np.int16)
            for row in fold.itertuples(index=False):
                nodes = symbol_to_nodes.get(row.target_symbol, []) if row.target_symbol else []
                finite_distances = [int(distances[node]) for node in nodes if distances[node] >= 0]
                network_distance = min(finite_distances) if finite_distances else np.nan
                network_degree = max((degree[node] for node in nodes), default=np.nan)
                item = row._asdict()
                item.update({
                    "target_in_string": bool(nodes),
                    "string_degree_high_confidence": network_degree,
                    "string_log1p_degree": float(np.log1p(network_degree)) if np.isfinite(network_degree) else np.nan,
                    "string_distance_to_training_target": network_distance,
                    "target_seen_in_training": row.target_symbol in train_symbols if row.target_symbol else False,
                    "n_training_target_symbols": len(train_symbols),
                    "n_training_target_nodes_in_string": len(sources),
                    "target_truth_opened_for_features": False,
                })
                rows.append(item)
        parts.append(pd.DataFrame(rows))
    metadata = {
        "string_proteins": len(info), "string_high_confidence_edges": retained_edges,
        "string_symbols": len(symbol_to_nodes),
    }
    return pd.concat(parts, ignore_index=True, sort=False), metadata


def freeze() -> None:
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    for path in [STRING_INFO, STRING_LINKS]:
        if not path.exists():
            raise FileNotFoundError(path)
    features, graph_metadata = build_deployable_network_features()
    feature_file = TABLES / "E144_NETWORK_FEATURES_BEFORE_TARGET_TRUTH.csv"
    features.to_csv(feature_file, index=False)
    manifest_hashes = {dataset: sha256(spec["manifest"]) for dataset, spec in DATASETS.items()}
    status = {
        "experiment": "E144_string_target_failure",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_target_gene_truth_opened",
        "n_datasets": int(features.dataset.nunique()), "n_tasks": len(features),
        "string_version": STRING_VERSION, "string_combined_score_threshold": STRING_SCORE_THRESHOLD,
        "string_info_url": STRING_INFO_URL, "string_links_url": STRING_LINKS_URL,
        "string_info_sha256": sha256(STRING_INFO), "string_links_sha256": sha256(STRING_LINKS),
        "network_feature_snapshot_sha256": sha256(feature_file), "manifest_sha256": manifest_hashes,
        **graph_metadata,
        "primary_pair": ["safeconf_calibrated_pair_risk", "target_gene_absolute_error_mean"],
        "supportive_pair": ["string_distance_to_training_target", "error_two_predictor_mean_rmse"],
        "comparators": ["baseline_predicted_magnitude", "risk_model_disagreement"],
        "bootstrap_unit": "dataset_then_perturbation_cluster_with_fold_macro",
        "n_bootstrap": N_BOOTSTRAP, "n_gene_label_permutations": N_LABEL_PERMUTATIONS,
        "saved_target_truth_vectors_opened": [],
        "independence_note": "literature-motivated audit frozen after prior whole-vector analyses; not a new independent dataset",
    }
    (OUT / "FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "PREREG.md").write_text(
        "# E144 预注册｜STRING 知识距离与靶基因自身恢复失败\n\n"
        "本分析由 TxPert 2026 报告的两个失败因素触发：知识图谱位置和未见扰动靶基因自身下调。"
        "STRING v12.0、combined score≥700、训练目标集合、网络特征和 SafeConf 分数先冻结，再打开保存的目标基因真值。\n\n"
        "- 主终点：原 SafeConf 与两个预测器对被扰动基因自身的平均绝对误差。\n"
        "- 支持终点：只在训练未见目标中，STRING 到训练目标的最短距离与全向量平均 RMSE。\n"
        "- 统计单位为 perturbation；先 fold 等权，再 dataset 等权；3,000 次 dataset→perturbation 整簇 bootstrap。\n"
        "- 主 gate：七数据主相关为正，且 95% CI 下界大于 0。若失败，保留失败；不改阈值或分数。\n"
        "- degree 和 distance 的基因标签置换为机制负对照；这不是 TxPert 的重训练或因果网络证明。\n"
        "- 该假设是在已有数据完成整体误差分析后由新文献触发，属于冻结后的二级机制审计，不冒充全新独立验证。\n\n"
        "依据：[TxPert, Nature Biotechnology (2026)](https://www.nature.com/articles/s41587-026-03113-4)；"
        "[STRING v12 download](https://string-db.org/cgi/download.pl)。\n"
    )
    (OUT / "README_先看这个.md").write_text("# E144 先看这个\n\n先读 `PREREG.md`；完成后读 `reports/E144_REPORT.md`。\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def add_target_errors(dataset: str, spec: dict, features: pd.DataFrame) -> pd.DataFrame:
    genes = panel_genes(spec)
    index = {gene.upper(): position for position, gene in enumerate(genes)}
    records = pd.read_csv(spec["root"] / spec["records"])
    predictions = np.load(spec["root"] / "arrays/predicted_effects.npz")
    truths = np.load(spec["root"] / "arrays/true_effects.npz")
    lookup = records.groupby([records.fold_id.astype(str), records.task_id.astype(str)], sort=False)
    rows = []
    for task in features[features.dataset.eq(dataset)].itertuples(index=False):
        item = task._asdict()
        target_index = index.get(str(task.target_symbol).upper()) if task.target_symbol else None
        if target_index is None:
            item.update({"target_in_prediction_panel": False, "true_target_effect": np.nan,
                         "predicted_target_effect_mean": np.nan, "target_gene_absolute_error_mean": np.nan,
                         "target_gene_ensemble_absolute_error": np.nan, "target_effect_sign_recovered_fraction": np.nan})
            rows.append(item)
            continue
        block = lookup.get_group((str(task.fold_id), str(task.task_id)))
        truth = float(np.asarray(truths[str(block.true_effect_key.iloc[0])], float)[target_index])
        predicted = np.asarray([float(np.asarray(predictions[str(record.predicted_effect_key)], float)[target_index])
                                for record in block.itertuples(index=False)])
        item.update({
            "target_in_prediction_panel": True, "true_target_effect": truth,
            "predicted_target_effect_mean": float(predicted.mean()),
            "target_gene_absolute_error_mean": float(np.mean(np.abs(predicted - truth))),
            "target_gene_ensemble_absolute_error": float(abs(predicted.mean() - truth)),
            "target_effect_sign_recovered_fraction": float(np.mean(np.sign(predicted) == np.sign(truth))),
        })
        rows.append(item)
    predictions.close(); truths.close()
    result = pd.DataFrame(rows)
    task_errors = pd.read_csv(spec["root"] / spec["tasks"])[["fold_id", "task_id", "error_two_predictor_mean_rmse"]]
    return result.merge(task_errors, on=["fold_id", "task_id"], how="left", validate="one_to_one")


def fold_correlations(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("safeconf_calibrated_pair_risk", "target_gene_absolute_error_mean", "all_mappable"),
        ("baseline_predicted_magnitude", "target_gene_absolute_error_mean", "all_mappable"),
        ("risk_model_disagreement", "target_gene_absolute_error_mean", "all_mappable"),
        ("string_distance_to_training_target", "error_two_predictor_mean_rmse", "unseen_string_mappable"),
        ("string_log1p_degree", "error_two_predictor_mean_rmse", "unseen_string_mappable"),
    ]
    for (dataset, fold_id), group in tasks.groupby(["dataset", "fold_id"], sort=True):
        for score, endpoint, subset in pairs:
            selected = group if subset == "all_mappable" else group[(~group.target_seen_in_training) & group.target_in_string]
            rows.append({"dataset": dataset, "fold_id": fold_id, "score": score, "endpoint": endpoint,
                         "subset": subset, "n_tasks": int(selected[[score, endpoint]].dropna().shape[0]),
                         "spearman": rho(selected[score], selected[endpoint])})
    return pd.DataFrame(rows)


def hierarchical_bootstrap(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    datasets = sorted(tasks.dataset.unique())
    cache = {}
    for dataset in datasets:
        data = tasks[tasks.dataset.eq(dataset)]
        perturbations = sorted(data.perturbation.astype(str).unique())
        cluster_index = {value: index for index, value in enumerate(perturbations)}
        folds = []
        for _, fold in data.groupby("fold_id", sort=False):
            folds.append({
                "cluster": np.asarray([cluster_index[value] for value in fold.perturbation.astype(str)], int),
                "safe": fold.safeconf_calibrated_pair_risk.to_numpy(float),
                "magnitude": fold.baseline_predicted_magnitude.to_numpy(float),
                "disagreement": fold.risk_model_disagreement.to_numpy(float),
                "target_error": fold.target_gene_absolute_error_mean.to_numpy(float),
            })
        cache[dataset] = {"n_clusters": len(perturbations), "folds": folds}
    draws = []
    for draw in range(N_BOOTSTRAP):
        sampled = rng.choice(datasets, len(datasets), replace=True)
        score_values = {key: [] for key in ["safe", "magnitude", "disagreement"]}
        for dataset in sampled:
            item = cache[str(dataset)]
            counts = rng.multinomial(item["n_clusters"], np.full(item["n_clusters"], 1 / item["n_clusters"]))
            dataset_values = {key: [] for key in score_values}
            for fold in item["folds"]:
                indices = np.repeat(np.arange(len(fold["cluster"])), counts[fold["cluster"]])
                for key in score_values:
                    dataset_values[key].append(rho(fold[key][indices], fold["target_error"][indices]))
            for key in score_values:
                score_values[key].append(float(np.nanmean(dataset_values[key])))
        safe = float(np.nanmean(score_values["safe"]))
        magnitude = float(np.nanmean(score_values["magnitude"]))
        disagreement = float(np.nanmean(score_values["disagreement"]))
        draws.append({"draw": draw, "safeconf__target_gene_absolute_error_mean": safe,
                      "magnitude__target_gene_absolute_error_mean": magnitude,
                      "disagreement__target_gene_absolute_error_mean": disagreement,
                      "delta_safeconf_minus_magnitude": safe - magnitude,
                      "delta_safeconf_minus_disagreement": safe - disagreement})
    frame = pd.DataFrame(draws)
    summary = []
    for column in frame.columns[1:]:
        values = frame[column].to_numpy(float)
        summary.append({"metric": column, "n_bootstrap": N_BOOTSTRAP, "median": np.nanmedian(values),
                        "ci95_low": np.nanquantile(values, .025), "ci95_high": np.nanquantile(values, .975),
                        "fraction_above_zero": np.nanmean(values > 0)})
    return frame, pd.DataFrame(summary)


def label_permutation_null(tasks: pd.DataFrame, observed: float) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED + 1)
    eligible = tasks[(~tasks.target_seen_in_training) & tasks.target_in_string].copy()
    values = []
    for draw in range(N_LABEL_PERMUTATIONS):
        dataset_values = []
        for _, dataset in eligible.groupby("dataset", sort=False):
            fold_values = []
            for _, fold in dataset.groupby("fold_id", sort=False):
                permuted = rng.permutation(fold.string_distance_to_training_target.to_numpy(float))
                fold_values.append(rho(permuted, fold.error_two_predictor_mean_rmse))
            dataset_values.append(float(np.nanmean(fold_values)))
        values.append(float(np.nanmean(dataset_values)))
    frame = pd.DataFrame({"draw": np.arange(N_LABEL_PERMUTATIONS), "permuted_fold_macro_spearman": values})
    summary = {"observed_fold_macro_spearman": observed, "null_median": float(np.nanmedian(values)),
               "two_sided_empirical_p": float((1 + np.sum(np.abs(values) >= abs(observed))) / (N_LABEL_PERMUTATIONS + 1))}
    return frame, summary


def analyze() -> None:
    status = json.loads((OUT / "FREEZE_STATUS.json").read_text())
    feature_file = TABLES / "E144_NETWORK_FEATURES_BEFORE_TARGET_TRUTH.csv"
    if sha256(feature_file) != status["network_feature_snapshot_sha256"] or sha256(STRING_LINKS) != status["string_links_sha256"]:
        raise RuntimeError("frozen network features or STRING resource changed")
    features = pd.read_csv(feature_file)
    tasks = pd.concat([add_target_errors(dataset, spec, features) for dataset, spec in DATASETS.items()], ignore_index=True)
    folds = fold_correlations(tasks)
    dataset_macro = folds.groupby(["dataset", "score", "endpoint", "subset"], as_index=False).spearman.mean()
    overall = dataset_macro.groupby(["score", "endpoint", "subset"], as_index=False).spearman.mean()
    draws, bootstrap = hierarchical_bootstrap(tasks)
    b = bootstrap.set_index("metric")
    primary_key = "safeconf__target_gene_absolute_error_mean"
    primary = overall[(overall.score == "safeconf_calibrated_pair_risk") & (overall.endpoint == "target_gene_absolute_error_mean")].spearman.iloc[0]
    passed = bool(primary > 0 and b.loc[primary_key, "ci95_low"] > 0)
    network_row = overall[(overall.score == "string_distance_to_training_target") & (overall.endpoint == "error_two_predictor_mean_rmse")]
    network_observed = float(network_row.spearman.iloc[0]) if len(network_row) else float("nan")
    permutation, permutation_summary = label_permutation_null(tasks, network_observed)
    coverage = tasks.groupby("dataset", as_index=False).agg(n_tasks=("task_id", "size"),
        n_target_in_panel=("target_in_prediction_panel", "sum"), n_target_in_string=("target_in_string", "sum"),
        n_unseen_targets=("target_seen_in_training", lambda value: int((~value.astype(bool)).sum())))
    primary_ci = b.loc[primary_key]
    magnitude_delta = b.loc["delta_safeconf_minus_magnitude"]
    disagreement_delta = b.loc["delta_safeconf_minus_disagreement"]
    lines = [
        "# E144｜STRING 知识距离与靶基因自身恢复失败", "", f"## 预注册主 gate：{'通过' if passed else '未通过'}", "",
        f"原 SafeConf 对被扰动基因自身绝对预测误差的七数据等权 ρ={primary:.3f}；"
        f"bootstrap 中位数 {primary_ci['median']:.3f}，95% CI [{primary_ci['ci95_low']:.3f}, {primary_ci['ci95_high']:.3f}]。", "",
        f"相对 predicted magnitude 的 Δρ={magnitude_delta['median']:+.3f} "
        f"[{magnitude_delta['ci95_low']:+.3f}, {magnitude_delta['ci95_high']:+.3f}]；"
        f"相对 disagreement 的 Δρ={disagreement_delta['median']:+.3f} "
        f"[{disagreement_delta['ci95_low']:+.3f}, {disagreement_delta['ci95_high']:+.3f}]。", "",
        "## 网络知识距离", "",
        f"训练未见目标中，STRING 最短距离与全向量 RMSE 的 fold→dataset 等权 ρ={network_observed:.3f}。"
        f"折内基因标签置换的中位数为 {permutation_summary['null_median']:.3f}，"
        f"双侧经验 P={permutation_summary['two_sided_empirical_p']:.4f}。", "",
        "该结果只说明网络知识位置与错误存在统计联系；STRING 是功能关联网络，不能据此宣称直接调控或因果机制。", "",
        "## 覆盖", "", "| dataset | tasks | target in 512 panel | target in STRING | unseen target tasks |", "|---|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.n_tasks} | {row.n_target_in_panel} | {row.n_target_in_string} | {row.n_unseen_targets} |")
    lines += ["", "## 解释边界", "",
              "目标基因自身误差是一个严格、易解释但很窄的终点；它不能替代全转录组、通路或蛋白层评价。"
              "本分析由 2026 年 TxPert 论文触发，并在同一批既有数据上完成，因此属于机制审计，不计作新增独立数据集。"
              "若主 gate 失败，不能靠改 STRING 阈值或只保留个别数据集挽救。"]
    (REPORTS / "E144_REPORT.md").write_text("\n".join(lines) + "\n")
    tasks.to_csv(TABLES / "E144_TASK_TARGET_ERRORS.csv", index=False)
    folds.to_csv(TABLES / "E144_FOLD_METRICS.csv", index=False)
    dataset_macro.to_csv(TABLES / "E144_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E144_SEVEN_DATASET_MACRO.csv", index=False)
    coverage.to_csv(TABLES / "E144_COVERAGE.csv", index=False)
    draws.to_csv(TABLES / "E144_HIERARCHICAL_BOOTSTRAP_DRAWS.csv", index=False)
    bootstrap.to_csv(TABLES / "E144_HIERARCHICAL_BOOTSTRAP_SUMMARY.csv", index=False)
    permutation.to_csv(TABLES / "E144_NETWORK_LABEL_PERMUTATION.csv", index=False)
    (TABLES / "E144_NETWORK_LABEL_PERMUTATION_SUMMARY.json").write_text(json.dumps(permutation_summary, indent=2) + "\n")
    run_status = {**status, "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete",
                  "preregistered_gate_passed": passed, "n_target_panel_tasks": int(tasks.target_in_prediction_panel.sum()),
                  "n_string_mapped_tasks": int(tasks.target_in_string.sum()), "score_refit": False,
                  "truth_used_to_change_score_or_network_threshold": False}
    (OUT / "RUN_STATUS.json").write_text(json.dumps(run_status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(run_status, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False)); print(bootstrap.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze_only else analyze()


if __name__ == "__main__":
    main()
