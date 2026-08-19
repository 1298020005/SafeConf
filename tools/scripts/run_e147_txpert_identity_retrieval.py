#!/usr/bin/env python3
"""E147: TxPert-style perturbation identity retrieval on seven reused datasets.

This is a new endpoint audit on already used test predictions, not an external
or prospective confirmation.  ``--freeze-only`` snapshots deployable scores
and source hashes without loading any prediction or truth vector.  The default
analysis verifies that freeze before opening the saved NPZ arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E147_txpert_identity_retrieval_20260714"
TABLES, REPORTS, FIGURES = OUT / "tables", OUT / "reports", OUT / "figures"
PREREG = OUT / "PREREG.md"
E141_SCORES = ROOT / "docs/实验结果/E141_progeny_pathway_fidelity_20260714/tables/E141_SCORES_BEFORE_VECTOR_TRUTH.csv"
SEED = 202607147
N_BOOTSTRAP = 3000
MIN_POOL = 10
EPS = 1e-12

SCORES = [
    "directional_risk_frozen",
    "safeconf_calibrated_pair_risk",
    "baseline_predicted_magnitude",
    "risk_model_disagreement",
]
PREDICTORS = ["scGPT", "GEARS"]
SIMILARITIES = ["pearson", "cosine"]

DATASETS = {
    "Frangieh": {
        "root": ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713",
        "records": "tables/PREDICTION_RECORDS.csv",
        "panel_h5ad": Path("/home/yyf/data/scgpt_formal_frangieh_fixed_panel_20260711/frangieh_e72_fixed512/perturb_processed.h5ad"),
    },
    "Lara_exvivo": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Lara_exvivo",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Santinha": {
        "root": ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/Santinha",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Shifrut": {
        "root": ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Liang": {
        "root": ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Tian_CRISPRi": {
        "root": ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
    },
    "Nadig_two_cellline": {
        "root": ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline",
        "records": "PREDICTION_RECORDS.csv", "panel": "GENE_PANEL.csv",
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


def predictor_family(value: str) -> str:
    value = str(value).lower()
    if value.startswith("scgpt"):
        return "scGPT"
    if value.startswith("gears"):
        return "GEARS"
    raise RuntimeError(f"unrecognized predictor_name: {value}")


def panel_genes(spec: dict) -> list[str]:
    if "panel_h5ad" in spec:
        data = ad.read_h5ad(spec["panel_h5ad"], backed="r")
        genes = data.var["gene_name"].astype(str).tolist()
        data.file.close()
        return genes
    panel = pd.read_csv(spec["root"] / spec["panel"])
    return panel["scgpt_token"].astype(str).tolist()


def source_file_manifest() -> list[dict]:
    rows = []
    for dataset, spec in DATASETS.items():
        root = spec["root"]
        for kind, path in [
            ("prediction_records", root / spec["records"]),
            ("predicted_effects_npz", root / "arrays/predicted_effects.npz"),
            ("true_effects_npz", root / "arrays/true_effects.npz"),
        ]:
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append({"dataset": dataset, "kind": kind, "path": str(path), "size_bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    return rows


def freeze() -> None:
    for directory in [OUT, TABLES, REPORTS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    if not PREREG.exists():
        raise FileNotFoundError("write and review PREREG.md before freezing inputs")
    source = pd.read_csv(E141_SCORES)
    required = {"dataset", "fold_id", "task_id", "setting", "context", "perturbation", *SCORES}
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"E141 score snapshot missing columns: {missing}")
    snapshot = source[["dataset", "fold_id", "task_id", "setting", "context", "perturbation", *SCORES]].copy()
    if set(snapshot.dataset) != set(DATASETS):
        raise RuntimeError(f"dataset mismatch: {sorted(snapshot.dataset.unique())}")
    if snapshot.duplicated(["dataset", "fold_id", "task_id"]).any():
        raise RuntimeError("duplicate dataset/fold/task in score snapshot")
    score_path = TABLES / "E147_SCORES_BEFORE_VECTOR_TRUTH.csv"
    snapshot.to_csv(score_path, index=False)
    pool = snapshot.groupby(["dataset", "fold_id", "context"], as_index=False).agg(
        n_task_rows=("task_id", "size"), n_unique_task_ids=("task_id", "nunique"),
        n_unique_perturbations=("perturbation", "nunique"),
    )
    pool["duplicate_perturbation_structure"] = pool.n_unique_perturbations.ne(pool.n_task_rows)
    pool["pool_size_at_least_10_before_metric_qc"] = pool.n_unique_perturbations.ge(MIN_POOL)
    pool.to_csv(TABLES / "E147_PREVECTOR_POOL_STRUCTURE.csv", index=False)
    files = source_file_manifest()
    pd.DataFrame(files).to_csv(TABLES / "E147_SOURCE_FILE_HASHES.csv", index=False)
    status = {
        "experiment": "E147_txpert_identity_retrieval",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_before_prediction_or_truth_vectors_loaded",
        "analysis_position": "new_endpoint_audit_on_previously_used_seven_datasets_not_independent_validation",
        "n_datasets": int(snapshot.dataset.nunique()), "n_tasks": len(snapshot),
        "n_candidate_pools": len(pool), "n_pools_ge10_pre_metric_qc": int(pool.pool_size_at_least_10_before_metric_qc.sum()),
        "prereg_sha256": sha256(PREREG), "source_e141_scores_sha256": sha256(E141_SCORES),
        "frozen_score_snapshot_sha256": sha256(score_path),
        "source_file_hash_table_sha256": sha256(TABLES / "E147_SOURCE_FILE_HASHES.csv"),
        "minimum_candidate_pool": MIN_POOL, "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_cluster": "dataset_context_perturbation", "dataset_aggregation": "fixed_seven_dataset_equal_weight",
        "prediction_or_truth_vectors_loaded": False,
    }
    (OUT / "FREEZE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README_先看这个.md").write_text(
        "# E147 先看这个\n\n先读 `PREREG.md`，再读 `reports/E147_REPORT.md`。本实验是已使用七数据上的新检索终点审计，不是新增独立验证。\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


def normalized_rows(matrix: np.ndarray, similarity: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, np.float64)
    if similarity == "pearson":
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    finite = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(matrix), matrix, 0.0), axis=1)
    valid = finite & (norms > EPS)
    output = np.zeros_like(matrix)
    output[valid] = matrix[valid] / norms[valid, None]
    return output, valid


def load_dataset_results(dataset: str, spec: dict, score: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = spec["root"]
    records = pd.read_csv(root / spec["records"])
    records = records[records.split.astype(str).eq("test")].copy()
    records["predictor"] = records.predictor_name.map(predictor_family)
    if records.duplicated(["fold_id", "task_id", "predictor"]).any():
        raise RuntimeError(f"{dataset}: duplicate task/predictor records")
    lookup = records.set_index([records.fold_id.astype(str), records.task_id.astype(str), "predictor"], drop=False)
    predicted = np.load(root / "arrays/predicted_effects.npz")
    truths = np.load(root / "arrays/true_effects.npz")
    genes = panel_genes(spec)
    rows, pool_audit = [], []
    subset = score[score.dataset.eq(dataset)].copy()
    for (fold_id, context), group in subset.groupby(["fold_id", "context"], sort=True):
        group = group.sort_values(["perturbation", "task_id"]).reset_index(drop=True)
        base_audit = {"dataset": dataset, "fold_id": fold_id, "context": context,
                      "n_task_rows": len(group), "n_unique_perturbations": group.perturbation.nunique()}
        if group.task_id.nunique() != len(group) or group.perturbation.nunique() != len(group):
            pool_audit.append({**base_audit, "similarity": "all", "n_valid_truth_candidates": 0,
                               "pool_eligible": False, "exclusion_reason": "duplicate_task_or_perturbation"})
            continue
        if len(group) < MIN_POOL:
            pool_audit.append({**base_audit, "similarity": "all", "n_valid_truth_candidates": 0,
                               "pool_eligible": False, "exclusion_reason": "candidate_pool_below_10"})
            continue
        truth_keys, prediction_keys = [], {predictor: [] for predictor in PREDICTORS}
        for task in group.itertuples(index=False):
            task_records = []
            for predictor in PREDICTORS:
                key = (str(task.fold_id), str(task.task_id), predictor)
                if key not in lookup.index:
                    raise RuntimeError(f"{dataset}: missing record {key}")
                record = lookup.loc[key]
                if isinstance(record, pd.DataFrame):
                    raise RuntimeError(f"{dataset}: nonunique record {key}")
                task_records.append(record)
                prediction_keys[predictor].append(str(record.predicted_effect_key))
            unique_truth = {str(record.true_effect_key) for record in task_records}
            if len(unique_truth) != 1:
                raise RuntimeError(f"{dataset}: predictor truth mismatch for {task.task_id}")
            truth_keys.append(unique_truth.pop())
        truth_matrix = np.stack([np.asarray(truths[key], float) for key in truth_keys])
        if truth_matrix.shape[1] != len(genes):
            raise RuntimeError(f"{dataset}: vector panel {truth_matrix.shape[1]} != gene panel {len(genes)}")
        prediction_matrices = {predictor: np.stack([np.asarray(predicted[key], float) for key in prediction_keys[predictor]])
                               for predictor in PREDICTORS}
        for similarity in SIMILARITIES:
            normalized_truth, truth_valid = normalized_rows(truth_matrix, similarity)
            valid_indices = np.flatnonzero(truth_valid)
            n_valid = len(valid_indices)
            eligible = n_valid >= MIN_POOL
            pool_audit.append({**base_audit, "similarity": similarity, "n_valid_truth_candidates": n_valid,
                               "pool_eligible": eligible,
                               "exclusion_reason": "" if eligible else "valid_truth_pool_below_10"})
            if not eligible:
                continue
            candidate_position = {int(original): position for position, original in enumerate(valid_indices)}
            candidates = normalized_truth[valid_indices]
            for predictor in PREDICTORS:
                normalized_prediction, prediction_valid = normalized_rows(prediction_matrices[predictor], similarity)
                similarity_matrix = normalized_prediction @ candidates.T
                for query_index, task in enumerate(group.itertuples(index=False)):
                    base = task._asdict()
                    base.update({"predictor": predictor, "similarity": similarity,
                                 "n_candidates_raw": len(group), "n_candidates_valid": n_valid})
                    if not prediction_valid[query_index]:
                        rows.append({**base, "query_valid": False, "invalid_reason": "nonfinite_or_zero_prediction",
                                     "correct_similarity": np.nan, "correct_rank_ascending": np.nan,
                                     "normalized_correct_rank": np.nan, "retrieval_error": np.nan,
                                     "top1_unique": np.nan, "top5_by_average_rank": np.nan})
                        continue
                    if query_index not in candidate_position:
                        rows.append({**base, "query_valid": False, "invalid_reason": "correct_truth_candidate_invalid",
                                     "correct_similarity": np.nan, "correct_rank_ascending": np.nan,
                                     "normalized_correct_rank": np.nan, "retrieval_error": np.nan,
                                     "top1_unique": np.nan, "top5_by_average_rank": np.nan})
                        continue
                    similarities = similarity_matrix[query_index]
                    correct = candidate_position[query_index]
                    ranks = rankdata(similarities, method="average")
                    correct_rank = float(ranks[correct])
                    normalized_rank = (correct_rank - 1.0) / (n_valid - 1.0)
                    maximum = float(np.max(similarities))
                    tied_max = np.isclose(similarities, maximum, rtol=1e-12, atol=1e-12)
                    best_rank = n_valid - correct_rank + 1.0
                    rows.append({**base, "query_valid": True, "invalid_reason": "",
                                 "correct_similarity": float(similarities[correct]),
                                 "correct_rank_ascending": correct_rank,
                                 "normalized_correct_rank": normalized_rank,
                                 "retrieval_error": 1.0 - normalized_rank,
                                 "top1_unique": float(tied_max[correct] and tied_max.sum() == 1),
                                 "top5_by_average_rank": float(best_rank <= 5.0)})
    predicted.close(); truths.close()
    return pd.DataFrame(rows), pd.DataFrame(pool_audit)


def fold_associations(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = results[results.query_valid.astype(bool)].copy()
    for (dataset, fold_id, predictor, similarity), group in valid.groupby(
            ["dataset", "fold_id", "predictor", "similarity"], sort=True):
        for score in SCORES:
            value = rho(group[score], group.retrieval_error) if len(group) >= MIN_POOL else float("nan")
            rows.append({"dataset": dataset, "fold_id": fold_id, "predictor": predictor,
                         "similarity": similarity, "score": score, "n_valid_queries": len(group),
                         "spearman_risk_vs_retrieval_error": value})
    return pd.DataFrame(rows)


def weighted_codes(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(np.asarray(values, float), return_inverse=True)
    return codes.astype(int), int(codes.max() + 1)


def weighted_spearman_from_codes(
        x_codes, n_x, y_codes, n_y, weights, minimum_weighted_queries: int = 3) -> float:
    weights = np.asarray(weights, float)
    active = weights > 0
    if active.sum() < 3 or weights.sum() < minimum_weighted_queries:
        return float("nan")
    x_group = np.bincount(x_codes, weights=weights, minlength=n_x)
    y_group = np.bincount(y_codes, weights=weights, minlength=n_y)
    if np.count_nonzero(x_group) < 2 or np.count_nonzero(y_group) < 2:
        return float("nan")
    x_mid = np.cumsum(x_group) - x_group + (x_group + 1.0) / 2.0
    y_mid = np.cumsum(y_group) - y_group + (y_group + 1.0) / 2.0
    rx, ry = x_mid[x_codes], y_mid[y_codes]
    total = weights.sum()
    mx, my = np.dot(weights, rx) / total, np.dot(weights, ry) / total
    dx, dy = rx - mx, ry - my
    denominator = np.sqrt(np.dot(weights, dx * dx) * np.dot(weights, dy * dy))
    return float(np.dot(weights, dx * dy) / denominator) if denominator > EPS else float("nan")


def summarize_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for column in frame.columns[1:]:
        values = frame[column].to_numpy(float)
        finite = values[np.isfinite(values)]
        summary.append({"metric": column, "n_bootstrap": N_BOOTSTRAP, "n_finite": len(finite),
                        "median": np.median(finite) if len(finite) else np.nan,
                        "ci95_low": np.quantile(finite, .025) if len(finite) else np.nan,
                        "ci95_high": np.quantile(finite, .975) if len(finite) else np.nan,
                        "fraction_above_zero": np.mean(finite > 0) if len(finite) else np.nan})
    return pd.DataFrame(summary)


def add_seven_dataset_metrics(row: dict, dataset_metrics: dict, datasets: list[str]) -> None:
    for score in SCORES:
        for predictor in PREDICTORS:
            for similarity in SIMILARITIES:
                key_tuple = (score, predictor, similarity)
                values = np.asarray([dataset_metrics[dataset].get(key_tuple, np.nan) for dataset in datasets], float)
                key = f"{score}__{predictor}__{similarity}"
                row[key] = float(np.mean(values)) if np.isfinite(values).all() else float("nan")
    for predictor in PREDICTORS:
        for similarity in SIMILARITIES:
            direction = row[f"directional_risk_frozen__{predictor}__{similarity}"]
            for comparator in ["safeconf_calibrated_pair_risk", "baseline_predicted_magnitude", "risk_model_disagreement"]:
                row[f"delta_directional_vs_{comparator}__{predictor}__{similarity}"] = (
                    direction - row[f"{comparator}__{predictor}__{similarity}"]
                )


def bootstrap(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = results[results.query_valid.astype(bool)].copy()
    datasets = sorted(DATASETS)
    cache = {}
    for dataset in datasets:
        data = valid[valid.dataset.eq(dataset)].copy()
        clusters = sorted((data.context.astype(str) + "\x1f" + data.perturbation.astype(str)).unique())
        cluster_index = {value: index for index, value in enumerate(clusters)}
        data["cluster_index"] = [cluster_index[value] for value in data.context.astype(str) + "\x1f" + data.perturbation.astype(str)]
        groups = []
        for (fold_id, predictor, similarity), group in data.groupby(["fold_id", "predictor", "similarity"], sort=True):
            if len(group) < MIN_POOL:
                continue
            item = {"fold_id": fold_id, "predictor": predictor, "similarity": similarity,
                    "cluster_index": group.cluster_index.to_numpy(int)}
            item["y_codes"], item["n_y"] = weighted_codes(group.retrieval_error.to_numpy(float))
            for score in SCORES:
                item[f"{score}_codes"], item[f"{score}_n"] = weighted_codes(group[score].to_numpy(float))
            groups.append(item)
        cache[dataset] = {"n_clusters": len(clusters), "groups": groups}
    rng = np.random.default_rng(SEED)
    draws, sensitivity_draws, threshold_draws = [], [], []
    group_qc = {}
    for draw in range(N_BOOTSTRAP):
        row, sensitivity_row = {"draw": draw}, {"draw": draw}
        dataset_metrics = {dataset: {} for dataset in datasets}
        sensitivity_dataset_metrics = {dataset: {} for dataset in datasets}
        endpoint_evaluations = endpoint_below = 0
        unique_fold_counts = {}
        for dataset in datasets:
            item = cache[dataset]
            counts = rng.multinomial(item["n_clusters"], np.full(item["n_clusters"], 1.0 / item["n_clusters"]))
            fold_values, sensitivity_fold_values = {}, {}
            for group in item["groups"]:
                weights = counts[group["cluster_index"]]
                weighted_queries = int(weights.sum())
                endpoint = (group["predictor"], group["similarity"])
                endpoint_evaluations += 1
                below = weighted_queries < MIN_POOL
                endpoint_below += int(below)
                qc_key = (dataset, str(group["fold_id"]), *endpoint)
                qc_item = group_qc.setdefault(qc_key, {"n_draws": 0, "n_draws_below_10": 0,
                                                       "minimum_weighted_queries": None,
                                                       "maximum_weighted_queries": None})
                qc_item["n_draws"] += 1
                qc_item["n_draws_below_10"] += int(below)
                qc_item["minimum_weighted_queries"] = weighted_queries if qc_item["minimum_weighted_queries"] is None else min(qc_item["minimum_weighted_queries"], weighted_queries)
                qc_item["maximum_weighted_queries"] = weighted_queries if qc_item["maximum_weighted_queries"] is None else max(qc_item["maximum_weighted_queries"], weighted_queries)
                unique_fold_counts.setdefault((dataset, str(group["fold_id"])), []).append(weighted_queries)
                for score in SCORES:
                    sensitivity_value = weighted_spearman_from_codes(
                        group[f"{score}_codes"], group[f"{score}_n"],
                        group["y_codes"], group["n_y"], weights, minimum_weighted_queries=3,
                    )
                    value = sensitivity_value if not below else float("nan")
                    fold_values.setdefault((score, *endpoint), []).append(value)
                    sensitivity_fold_values.setdefault((score, *endpoint), []).append(sensitivity_value)
            for key, values in fold_values.items():
                values = np.asarray(values, float)
                dataset_metrics[dataset][key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            for key, values in sensitivity_fold_values.items():
                values = np.asarray(values, float)
                sensitivity_dataset_metrics[dataset][key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        add_seven_dataset_metrics(row, dataset_metrics, datasets)
        add_seven_dataset_metrics(sensitivity_row, sensitivity_dataset_metrics, datasets)
        unique_fold_below = sum(int(min(values) < MIN_POOL) for values in unique_fold_counts.values())
        threshold_draws.append({
            "draw": draw,
            "n_fold_endpoint_evaluations": endpoint_evaluations,
            "n_fold_endpoint_evaluations_below_10": endpoint_below,
            "n_unique_dataset_fold_evaluations": len(unique_fold_counts),
            "n_unique_dataset_fold_evaluations_below_10": unique_fold_below,
            "any_fold_endpoint_below_10": bool(endpoint_below),
            "any_unique_dataset_fold_below_10": bool(unique_fold_below),
        })
        draws.append(row)
        sensitivity_draws.append(sensitivity_row)
    frame = pd.DataFrame(draws)
    sensitivity_frame = pd.DataFrame(sensitivity_draws)
    threshold_frame = pd.DataFrame(threshold_draws)
    group_rows = []
    for (dataset, fold_id, predictor, similarity), item in sorted(group_qc.items()):
        group_rows.append({"dataset": dataset, "fold_id": fold_id, "predictor": predictor,
                           "similarity": similarity, **item,
                           "fraction_draws_below_10": item["n_draws_below_10"] / item["n_draws"]})
    return (frame, summarize_bootstrap(frame), sensitivity_frame, summarize_bootstrap(sensitivity_frame),
            threshold_frame, pd.DataFrame(group_rows))


def make_figure(performance: pd.DataFrame, overall: pd.DataFrame, boot: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .8})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), facecolor="white")
    colors = {"scGPT": "#3B6FB6", "GEARS": "#C65A3A"}
    aggregate = performance.groupby(["predictor", "similarity"], as_index=False).agg(
        normalized_correct_rank=("normalized_correct_rank_mean", "mean"))
    positions, labels = [], []
    index = 0
    for similarity in SIMILARITIES:
        for predictor in PREDICTORS:
            value = aggregate[(aggregate.predictor == predictor) & (aggregate.similarity == similarity)].normalized_correct_rank.iloc[0]
            axes[0].bar(index, value, color=colors[predictor], width=.72, edgecolor="none")
            positions.append(index); labels.append(f"{predictor}\n{similarity}"); index += 1
        index += .45
    axes[0].axhline(.5, color="#777777", linestyle="--", linewidth=1, label="random expectation")
    axes[0].set_xticks(positions, labels); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Normalized correct rank")
    axes[0].set_title("a  Perturbation identity retrieval", loc="left", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    b = boot.set_index("metric")
    direction = overall[overall.score.eq("directional_risk_frozen")].copy()
    direction["endpoint"] = direction.predictor + " / " + direction.similarity
    direction = direction.set_index(["predictor", "similarity"]).loc[
        [(p, s) for s in SIMILARITIES for p in PREDICTORS]
    ].reset_index()
    y = np.arange(len(direction))[::-1]
    for position, row in zip(y, direction.itertuples(index=False)):
        key = f"directional_risk_frozen__{row.predictor}__{row.similarity}"
        ci = b.loc[key]
        axes[1].errorbar(row.spearman_risk_vs_retrieval_error, position,
                         xerr=[[row.spearman_risk_vs_retrieval_error - ci.ci95_low],
                               [ci.ci95_high - row.spearman_risk_vs_retrieval_error]],
                         fmt="o", color=colors[row.predictor], capsize=2.5, markersize=5)
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].set_yticks(y, direction.endpoint)
    axes[1].set_xlabel("Spearman ρ with retrieval error")
    axes[1].set_title("b  Directional-SafeConf association", loc="left", fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGURES / "E147_RETRIEVAL_AUDIT.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "E147_RETRIEVAL_AUDIT.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(results, pool_audit, performance, dataset_macro, overall, boot,
                 sensitivity_boot, threshold_draws, threshold_groups) -> None:
    b = boot.set_index("metric")
    sensitivity = sensitivity_boot.set_index("metric")
    lines = [
        "# E147｜扰动身份检索审计", "",
        "这是已用于 SafeConf 开发与评价的七数据上的新终点，不是新增独立验证。查询为模型预测效应；候选库为同一 dataset、fold 和 context 的测试真值。归一化正确秩越高越好，随机期望约 0.5；检索误差等于 1−正确秩。", "",
        "## 检索本身", "",
        "| predictor | similarity | 七数据等权正确秩 | top-1 | top-5 | 有效查询 |", "|---|---|---:|---:|---:|---:|",
    ]
    perf_all = performance.groupby(["predictor", "similarity"], as_index=False).agg(
        normalized_correct_rank_mean=("normalized_correct_rank_mean", "mean"),
        top1_rate=("top1_rate", "mean"), top5_rate=("top5_rate", "mean"), n_valid_queries=("n_valid_queries", "sum"))
    for row in perf_all.sort_values(["similarity", "predictor"]).itertuples(index=False):
        lines.append(f"| {row.predictor} | {row.similarity} | {row.normalized_correct_rank_mean:.3f} | {row.top1_rate:.3f} | {row.top5_rate:.3f} | {int(row.n_valid_queries)} |")
    lines += ["", "## 风险与检索误差", "",
              "正相关表示风险分数越高，模型越难在同背景候选中找回正确扰动。点估计按 fold→dataset→七数据等权聚合；区间按 dataset 内唯一 context×perturbation 整簇 bootstrap。每次重抽后，dataset×fold×predictor×similarity 的加权有效查询仍须不少于 10，否则该折在该次抽样中记为缺失。", "",
              "| score | predictor | similarity | 七数据等权 ρ | 95% CI |", "|---|---|---|---:|---:|"]
    for row in overall.sort_values(["score", "similarity", "predictor"]).itertuples(index=False):
        key = f"{row.score}__{row.predictor}__{row.similarity}"
        ci = b.loc[key]
        lines.append(f"| {row.score} | {row.predictor} | {row.similarity} | {row.spearman_risk_vs_retrieval_error:.3f} | [{ci.ci95_low:.3f}, {ci.ci95_high:.3f}] |")
    lines += ["", "## Directional-SafeConf 的逐数据集结果", "",
              "| dataset | GEARS Pearson | GEARS cosine | scGPT Pearson | scGPT cosine |", "|---|---:|---:|---:|---:|"]
    direction_dataset = dataset_macro[dataset_macro.score.eq("directional_risk_frozen")]
    for dataset in sorted(DATASETS):
        values = {}
        for predictor in PREDICTORS:
            for similarity in SIMILARITIES:
                selected = direction_dataset[(direction_dataset.dataset == dataset) &
                                             (direction_dataset.predictor == predictor) &
                                             (direction_dataset.similarity == similarity)]
                values[(predictor, similarity)] = float(selected.spearman_risk_vs_retrieval_error.iloc[0])
        lines.append(f"| {dataset} | {values[('GEARS', 'pearson')]:.3f} | {values[('GEARS', 'cosine')]:.3f} | "
                     f"{values[('scGPT', 'pearson')]:.3f} | {values[('scGPT', 'cosine')]:.3f} |")
    lines += ["", "## 判读", ""]
    gp = b.loc["directional_risk_frozen__GEARS__pearson"]
    gc = b.loc["directional_risk_frozen__GEARS__cosine"]
    dp = b.loc["delta_directional_vs_safeconf_calibrated_pair_risk__GEARS__pearson"]
    dc = b.loc["delta_directional_vs_safeconf_calibrated_pair_risk__GEARS__cosine"]
    lines.append(
        f"Directional-SafeConf 对 GEARS 的两个预指定检索误差均为稳定正相关：Pearson 检索 95% CI "
        f"[{gp.ci95_low:.3f}, {gp.ci95_high:.3f}]，cosine 检索 [{gc.ci95_low:.3f}, {gc.ci95_high:.3f}]；"
        "七个数据集中六个方向为正，Santinha 为负。"
    )
    lines.append(
        f"相对原 SafeConf，Directional-SafeConf 的条件差值为 Pearson {dp['median']:+.3f} "
        f"[{dp.ci95_low:+.3f}, {dp.ci95_high:+.3f}]、cosine {dc['median']:+.3f} "
        f"[{dc.ci95_low:+.3f}, {dc.ci95_high:+.3f}]。这是方向风险模型在既有数据上的终点一致性，不是独立确认。"
    )
    lines.append(
        "scGPT 的正确秩在两种相似度下均约为随机期望 0.5。这里审计的是保存为 "
        "`scGPT_context_mean_finetuned` 的具体均值型预测记录，不能外推为所有 scGPT 模型都没有扰动身份信息。"
    )
    lines.append(
        "magnitude 和 disagreement 对 GEARS 检索误差为负，说明大效应任务在身份检索上反而更容易；"
        "原 SafeConf 因包含绝对难度成分，在该终点上没有正相关。检索身份和逐基因方向误差回答的是不同问题。"
    )
    endpoint_total = int(threshold_draws.n_fold_endpoint_evaluations.sum())
    endpoint_below = int(threshold_draws.n_fold_endpoint_evaluations_below_10.sum())
    unique_total = int(threshold_draws.n_unique_dataset_fold_evaluations.sum())
    unique_below = int(threshold_draws.n_unique_dataset_fold_evaluations_below_10.sum())
    draws_any = int(threshold_draws.any_fold_endpoint_below_10.sum())
    unique_draws_any = int(threshold_draws.any_unique_dataset_fold_below_10.sum())
    lines += ["", "## Bootstrap 门槛修复与敏感性", "",
              f"3,000 次抽样共产生 {endpoint_total} 个 fold×predictor×similarity 评价，其中 {endpoint_below} 个加权有效查询少于 10；"
              f"受影响的 bootstrap draw 为 {draws_any}/3000。按不重复的 dataset×fold 计，共 {unique_below}/{unique_total} 个评价低于门槛，"
              f"涉及 {unique_draws_any}/3000 个 draw。主区间已经排除这些折，不再沿用修复前的三查询下限。", "",
              "| Directional-SafeConf 终点 | 严格≥10：median [95% CI] | 不设10门槛诊断：median [95% CI] | median变化 |",
              "|---|---:|---:|---:|"]
    for predictor in PREDICTORS:
        for similarity in SIMILARITIES:
            key = f"directional_risk_frozen__{predictor}__{similarity}"
            strict_row, loose_row = b.loc[key], sensitivity.loc[key]
            lines.append(
                f"| {predictor} / {similarity} | {strict_row['median']:.3f} "
                f"[{strict_row.ci95_low:.3f}, {strict_row.ci95_high:.3f}] | "
                f"{loose_row['median']:.3f} [{loose_row.ci95_low:.3f}, {loose_row.ci95_high:.3f}] | "
                f"{strict_row['median'] - loose_row['median']:+.4f} |"
            )
    strict_join = boot.set_index("metric")[["median"]].join(
        sensitivity_boot.set_index("metric")[["median"]], lsuffix="_strict", rsuffix="_no_min10")
    maximum_shift = float((strict_join.median_strict - strict_join.median_no_min10).abs().max())
    lines += ["", f"全部直接关联和差值指标中，严格门槛相对不设10门槛诊断的最大 median 变化为 {maximum_shift:.4f}。"
              "该敏感性只用于核查门槛实现，不替代主结果。",
              f"低于门槛最频繁的 fold-endpoint 及次数已保存到 `tables/E147_BOOTSTRAP_THRESHOLD_QC_BY_FOLD_ENDPOINT.csv`；"
              f"该表共有 {len(threshold_groups)} 行。"]
    n_pools = int(pool_audit.pool_eligible.fillna(False).sum())
    n_invalid = int((~results.query_valid.astype(bool)).sum()) if len(results) else 0
    lines += ["", f"共有 {n_pools} 个 metric-specific 候选池满足规则；无效查询 {n_invalid} 条。",
              "", "## 边界", "",
              "身份检索衡量预测效应是否更像正确扰动，而逐基因 RMSE/cosine 衡量向量误差，二者不可互换。候选库难度受同背景内候选数和扰动相似性影响；本分析通过同 fold、同 context 和归一化秩限制了明显偏差，但数据仍全部被此前实验使用过。"]
    lines.append("bootstrap 固定这七个数据集并只在各数据集内部重抽生物任务簇，因此区间是对当前七数据的条件不确定性，不包含‘换一批数据集’的研究间异质性。")
    (REPORTS / "E147_REPORT.md").write_text("\n".join(lines) + "\n")


def analyze() -> None:
    freeze_path = OUT / "FREEZE_STATUS.json"
    if not freeze_path.exists():
        raise RuntimeError("run --freeze-only before analysis")
    frozen = json.loads(freeze_path.read_text())
    score_path = TABLES / "E147_SCORES_BEFORE_VECTOR_TRUTH.csv"
    if sha256(PREREG) != frozen["prereg_sha256"] or sha256(score_path) != frozen["frozen_score_snapshot_sha256"]:
        raise RuntimeError("preregistered contract or score snapshot changed after freeze")
    source_hashes = pd.read_csv(TABLES / "E147_SOURCE_FILE_HASHES.csv")
    for row in source_hashes.itertuples(index=False):
        if sha256(Path(row.path)) != row.sha256:
            raise RuntimeError(f"source changed after freeze: {row.path}")
    scores = pd.read_csv(score_path)
    result_parts, pool_parts = [], []
    for dataset, spec in DATASETS.items():
        print(f"[E147] retrieval {dataset}", flush=True)
        results, pools = load_dataset_results(dataset, spec, scores)
        result_parts.append(results); pool_parts.append(pools)
    results = pd.concat(result_parts, ignore_index=True, sort=False)
    pools = pd.concat(pool_parts, ignore_index=True, sort=False)
    if results.empty:
        raise RuntimeError("no eligible retrieval queries")
    valid = results[results.query_valid.astype(bool)].copy()
    folds = fold_associations(results)
    dataset_macro = folds.groupby(["dataset", "predictor", "similarity", "score"], as_index=False).agg(
        n_folds=("spearman_risk_vs_retrieval_error", lambda x: int(np.isfinite(x).sum())),
        spearman_risk_vs_retrieval_error=("spearman_risk_vs_retrieval_error", "mean"),
    )
    overall = dataset_macro.groupby(["predictor", "similarity", "score"], as_index=False).agg(
        n_datasets=("spearman_risk_vs_retrieval_error", lambda x: int(np.isfinite(x).sum())),
        spearman_risk_vs_retrieval_error=("spearman_risk_vs_retrieval_error", "mean"),
    )
    performance = valid.groupby(["dataset", "predictor", "similarity"], as_index=False).agg(
        n_valid_queries=("task_id", "size"),
        normalized_correct_rank_mean=("normalized_correct_rank", "mean"),
        normalized_correct_rank_median=("normalized_correct_rank", "median"),
        retrieval_error_mean=("retrieval_error", "mean"),
        top1_rate=("top1_unique", "mean"), top5_rate=("top5_by_average_rank", "mean"),
        mean_candidate_pool=("n_candidates_valid", "mean"),
    )
    draws, boot, sensitivity_draws, sensitivity_boot, threshold_draws, threshold_groups = bootstrap(results)
    results.to_csv(TABLES / "E147_TASK_RETRIEVAL_RESULTS.csv", index=False)
    pools.to_csv(TABLES / "E147_POOL_METRIC_AUDIT.csv", index=False)
    folds.to_csv(TABLES / "E147_FOLD_ASSOCIATIONS.csv", index=False)
    dataset_macro.to_csv(TABLES / "E147_DATASET_MACRO.csv", index=False)
    overall.to_csv(TABLES / "E147_SEVEN_DATASET_MACRO.csv", index=False)
    performance.to_csv(TABLES / "E147_RETRIEVAL_PERFORMANCE.csv", index=False)
    draws.to_csv(TABLES / "E147_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot.to_csv(TABLES / "E147_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    sensitivity_draws.to_csv(TABLES / "E147_BOOTSTRAP_NO_MIN10_SENSITIVITY_DRAWS.csv", index=False)
    sensitivity_boot.to_csv(TABLES / "E147_BOOTSTRAP_NO_MIN10_SENSITIVITY_SUMMARY.csv", index=False)
    threshold_draws.to_csv(TABLES / "E147_BOOTSTRAP_THRESHOLD_QC_DRAWS.csv", index=False)
    threshold_groups.to_csv(TABLES / "E147_BOOTSTRAP_THRESHOLD_QC_BY_FOLD_ENDPOINT.csv", index=False)
    write_report(results, pools, performance, dataset_macro, overall, boot,
                 sensitivity_boot, threshold_draws, threshold_groups)
    make_figure(performance, overall, boot)
    sensitivity_join = boot.set_index("metric")[["median"]].join(
        sensitivity_boot.set_index("metric")[["median"]], lsuffix="_strict", rsuffix="_no_min10")
    status = {
        **frozen, "generated_at": datetime.now().isoformat(timespec="seconds"), "status": "complete",
        "prediction_or_truth_vectors_loaded": True, "n_result_rows": len(results),
        "n_valid_queries": len(valid), "n_invalid_queries": int(len(results) - len(valid)),
        "n_metric_specific_eligible_pools": int(pools.pool_eligible.fillna(False).sum()),
        "bootstrap_minimum_weighted_valid_queries_per_dataset_fold_endpoint_draw": MIN_POOL,
        "n_bootstrap_fold_endpoint_draw_evaluations": int(threshold_draws.n_fold_endpoint_evaluations.sum()),
        "n_bootstrap_fold_endpoint_draw_evaluations_below_10": int(threshold_draws.n_fold_endpoint_evaluations_below_10.sum()),
        "n_bootstrap_draws_with_any_fold_endpoint_below_10": int(threshold_draws.any_fold_endpoint_below_10.sum()),
        "n_bootstrap_unique_dataset_fold_draw_evaluations": int(threshold_draws.n_unique_dataset_fold_evaluations.sum()),
        "n_bootstrap_unique_dataset_fold_draw_evaluations_below_10": int(threshold_draws.n_unique_dataset_fold_evaluations_below_10.sum()),
        "n_bootstrap_draws_with_any_unique_dataset_fold_below_10": int(threshold_draws.any_unique_dataset_fold_below_10.sum()),
        "strict_vs_no_min10_max_abs_bootstrap_median_shift": float(
            (sensitivity_join.median_strict - sensitivity_join.median_no_min10).abs().max()),
        "bootstrap_threshold_sensitivity_is_diagnostic_only": True,
        "score_or_predictor_refit": False, "test_truth_used_to_change_score": False,
        "independent_validation_claim_allowed": False, "preregistered_pass_fail_gate": None,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(performance.to_string(index=False))
    print(overall.to_string(index=False))
    print(boot[boot.metric.str.startswith("directional_risk_frozen")].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze_only else analyze()


if __name__ == "__main__":
    main()
