#!/usr/bin/env python3
"""E98: honest predictors and task-risk audit on the E97 Frangieh matrix.

The E97 manifest is the only split authority.  Predictor fitting receives only
training-pair effects.  Query metadata, target-context control profiles and
pretrained scGPT gene embeddings are available at prediction time.  Held-out
perturbed expression is attached only after predictions and deployable risk
features have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code/20260426_154505_perturb_transport_final_push"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from safetrans_confidence.data.records import validate_prediction_record_artifacts  # noqa: E402


SOURCE = Path("/home/yyf/data/gears_formal_baselines_v2/frangieh_local_atlas/perturb_processed.h5ad")
CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
CONTRACT = ROOT / "docs/实验结果/E97_frangieh_gene_cartesian_contract_20260713/manifests/E97_TASK_MANIFEST.csv"
OUT = ROOT / "docs/实验结果/E98_frangieh_gene_cartesian_predictions_20260713"
TABLES, ARRAYS, REPORTS, FIGURES = OUT / "tables", OUT / "arrays", OUT / "reports", OUT / "figures"
CACHE = Path("/home/yyf/data/safeconf_e98_frangieh_cartesian")
ASSET_CACHE = CACHE / "E98_EFFECT_ASSETS.npz"
FRACTIONS = (25, 50, 75, 100)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
K_NEIGHBORS = 8
SEED = 20260713


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def hash_order(items: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(items).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def cosine_error(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - np.dot(a, b) / denominator) if denominator > 1e-12 else 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    ranked_a = rankdata(a[mask], method="average")
    ranked_b = rankdata(b[mask], method="average")
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(1.4826 * mad, float(np.std(reference)), 1e-8)
    return np.clip((np.asarray(values, dtype=float) - center) / scale, -5.0, 5.0)


def prepare_assets(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute task means once; no split or error participates in aggregation."""
    CACHE.mkdir(parents=True, exist_ok=True)
    expected_contexts = sorted(manifest["context"].astype(str).unique())
    expected_perts = sorted(manifest["perturbation"].astype(str).unique())
    if ASSET_CACHE.exists():
        try:
            with np.load(ASSET_CACHE, allow_pickle=False) as store:
                assets = {key: np.asarray(store[key]) for key in store.files}
            if assets["contexts"].astype(str).tolist() == expected_contexts and assets["perturbations"].astype(str).tolist() == expected_perts:
                return assets
        except ValueError:
            # The first development cache used object-dtype string indexes.
            # Rebuild instead of weakening the loader with allow_pickle=True.
            ASSET_CACHE.unlink()

    dataset = ad.read_h5ad(SOURCE)
    context = dataset.obs["cell_type"].astype(str).to_numpy()
    condition = dataset.obs["condition"].astype(str).to_numpy()
    keep = np.isin(context, expected_contexts) & (np.isin(condition, expected_perts) | (condition == "ctrl"))
    x = dataset.X[keep]
    if not sp.issparse(x):
        x = sp.csr_matrix(np.asarray(x))
    else:
        x = x.tocsr()
    labels = np.asarray([f"{c}\x1f{p}" for c, p in zip(context[keep], condition[keep])])
    groups, codes = np.unique(labels, return_inverse=True)
    membership = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    sums = membership @ x
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    means = np.asarray(sums.multiply((1.0 / counts)[:, None]).toarray(), dtype=np.float32)
    mean_map = {label: means[index] for index, label in enumerate(groups)}
    controls = np.stack([mean_map[f"{c}\x1fctrl"] for c in expected_contexts]).astype(np.float32)
    effects = np.stack(
        [
            mean_map[f"{c}\x1f{p}"] - mean_map[f"{c}\x1fctrl"]
            for c in expected_contexts
            for p in expected_perts
        ]
    ).astype(np.float32)
    genes = np.asarray(dataset.var_names.astype(str).tolist(), dtype=str)
    assets = {
        "contexts": np.asarray(expected_contexts, dtype=str),
        "perturbations": np.asarray(expected_perts, dtype=str),
        "genes": genes,
        "controls": controls,
        "effects": effects,
    }
    np.savez_compressed(ASSET_CACHE, **assets)
    return assets


def load_scgpt_embeddings(perturbations: list[str]) -> tuple[np.ndarray, dict]:
    vocab = json.loads((CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(CHECKPOINT / "best_model.pt", map_location="cpu")
    weights = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    genes = [pert.split("+")[0] for pert in perturbations]
    missing = [gene for gene in genes if gene not in vocab]
    if missing:
        raise RuntimeError(f"scGPT vocabulary misses {missing}")
    embedding = np.stack([weights[int(vocab[gene])] for gene in genes])
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norm, 1e-8)
    return embedding.astype(np.float32), {
        "checkpoint": str(CHECKPOINT / "best_model.pt"),
        "checkpoint_sha256": file_sha256(CHECKPOINT / "best_model.pt"),
        "embedding_tensor": "encoder.embedding.weight",
        "embedding_dimension": int(embedding.shape[1]),
        "missing_perturbation_genes": missing,
    }


class EffectStore:
    def __init__(self, assets: dict[str, np.ndarray]):
        self.contexts = assets["contexts"].astype(str).tolist()
        self.perts = assets["perturbations"].astype(str).tolist()
        self.genes = assets["genes"].astype(str).tolist()
        self.controls = np.asarray(assets["controls"], dtype=np.float32)
        self.effects = np.asarray(assets["effects"], dtype=np.float32).reshape(
            len(self.contexts), len(self.perts), -1
        )
        self.cix = {value: i for i, value in enumerate(self.contexts)}
        self.pix = {value: i for i, value in enumerate(self.perts)}

    def effect(self, context: str, perturbation: str) -> np.ndarray:
        return self.effects[self.cix[context], self.pix[perturbation]]

    def control(self, context: str) -> np.ndarray:
        return self.controls[self.cix[context]]


def make_features(store: EffectStore, embeddings: np.ndarray) -> tuple[np.ndarray, dict]:
    gene_components = min(24, embeddings.shape[0] - 1)
    gene_pca = PCA(n_components=gene_components, random_state=SEED).fit_transform(embeddings)
    context_components = min(2, len(store.contexts) - 1)
    context_pca = PCA(n_components=context_components, random_state=SEED).fit_transform(store.controls)
    gene_pca = (gene_pca - gene_pca.mean(0)) / np.maximum(gene_pca.std(0), 1e-8)
    context_pca = (context_pca - context_pca.mean(0)) / np.maximum(context_pca.std(0), 1e-8)
    rows = []
    for ci in range(len(store.contexts)):
        for pi in range(len(store.perts)):
            interaction = np.outer(gene_pca[pi, :12], context_pca[ci]).ravel()
            rows.append(np.concatenate([gene_pca[pi], context_pca[ci], interaction]))
    return np.asarray(rows, dtype=np.float32).reshape(len(store.contexts), len(store.perts), -1), {
        "gene_embedding_pca_components": gene_components,
        "context_control_pca_components": context_components,
        "interaction_components": int(12 * context_components),
        "feature_dimension": int(len(rows[0])),
        "target_context_control_is_available": True,
        "target_perturbed_expression_is_available": False,
    }


def source_knn_predictions(
    store: EffectStore,
    embeddings: np.ndarray,
    train_pairs: list[tuple[str, str]],
    queries: list[tuple[str, str]],
) -> dict[tuple[str, str], np.ndarray]:
    by_pert: dict[str, list[np.ndarray]] = {}
    for context, pert in train_pairs:
        by_pert.setdefault(pert, []).append(store.effect(context, pert))
    prototypes = {pert: np.mean(np.stack(vectors), axis=0) for pert, vectors in by_pert.items()}
    available = sorted(prototypes)
    available_ix = np.asarray([store.pix[pert] for pert in available], dtype=int)
    available_embeddings = embeddings[available_ix]
    result = {}
    for query in queries:
        pert = query[1]
        if pert in prototypes:
            result[query] = prototypes[pert].astype(np.float32)
            continue
        similarity = available_embeddings @ embeddings[store.pix[pert]]
        k = min(K_NEIGHBORS, len(available))
        nearest = np.argsort(-similarity, kind="stable")[:k]
        weight = np.exp((similarity[nearest] - similarity[nearest].max()) / 0.10)
        weight /= weight.sum()
        result[query] = np.sum(
            np.stack([prototypes[available[index]] for index in nearest]) * weight[:, None], axis=0
        ).astype(np.float32)
    return result


def fit_context_ridge(
    store: EffectStore,
    features: np.ndarray,
    train_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
) -> tuple[Ridge, float, pd.DataFrame]:
    def design(pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.stack([features[store.cix[c], store.pix[p]] for c, p in pairs])

    x_train = design(train_pairs)
    y_train = np.stack([store.effect(c, p) for c, p in train_pairs])
    x_val = design(validation_pairs)
    y_val = np.stack([store.effect(c, p) for c, p in validation_pairs])
    rows = []
    models = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        error = rmse(model.predict(x_val), y_val)
        rows.append({"alpha": alpha, "validation_profile_rmse": error})
        models.append(model)
    best = int(np.argmin([row["validation_profile_rmse"] for row in rows]))
    return models[best], float(RIDGE_ALPHAS[best]), pd.DataFrame(rows)


def context_novelty_scale(store: EffectStore) -> float:
    distances = []
    for i in range(len(store.contexts)):
        for j in range(i + 1, len(store.contexts)):
            distances.append(1.0 - cosine_similarity(store.controls[i], store.controls[j]))
    positive = [value for value in distances if value > 1e-10]
    return float(np.median(positive)) if positive else 1.0


def score_queries_without_truth(
    store: EffectStore,
    train_pairs: list[tuple[str, str]],
    query_frame: pd.DataFrame,
    pred_a: dict[tuple[str, str], np.ndarray],
    pred_b: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    train_contexts = sorted({context for context, _ in train_pairs})
    support = pd.Series([pert for _, pert in train_pairs]).value_counts().to_dict()
    scale = context_novelty_scale(store)
    rows = []
    for row in query_frame.itertuples(index=False):
        pair = (str(row.context), str(row.perturbation))
        a, b = pred_a[pair], pred_b[pair]
        max_similarity = max(cosine_similarity(store.control(pair[0]), store.control(c)) for c in train_contexts)
        context_novelty = 0.0 if pair[0] in train_contexts else min((1.0 - max_similarity) / scale, 5.0)
        n_support = int(support.get(pair[1], 0))
        rows.append(
            {
                "context": pair[0],
                "perturbation": pair[1],
                "split": row.split,
                "setting": row.setting,
                "n_cells": int(row.n_cells),
                "context_max_control_cosine": max_similarity,
                "context_novelty_scaled": context_novelty,
                "training_support_count": n_support,
                "perturbation_novelty": 1.0 / (1.0 + n_support),
                "risk_model_disagreement": rmse(a, b),
                "baseline_predicted_magnitude": float(np.sqrt(np.mean(((a + b) / 2.0) ** 2))),
            }
        )
    scored = pd.DataFrame(rows)
    val = scored["split"].eq("val")
    disagreement_reference = scored.loc[val, "risk_model_disagreement"].to_numpy(float)
    magnitude_reference = scored.loc[val, "baseline_predicted_magnitude"].to_numpy(float)
    scored["risk_disagreement_z"] = robust_z(
        scored["risk_model_disagreement"].to_numpy(float), disagreement_reference
    )
    scored["predicted_magnitude_z"] = robust_z(
        scored["baseline_predicted_magnitude"].to_numpy(float), magnitude_reference
    )
    scored["safeconf_frozen_pair_risk"] = (
        scored["risk_disagreement_z"]
        + scored["context_novelty_scaled"]
        + scored["perturbation_novelty"]
    )
    scored["deployable_features_use_target_perturbed_truth"] = False
    return scored


def calibrate_pair_risk_with_validation_truth(
    store: EffectStore,
    scored: pd.DataFrame,
    pred_a: dict[tuple[str, str], np.ndarray],
    pred_b: dict[tuple[str, str], np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    """Fit the pair-risk combiner on val tasks; test effects remain unopened."""
    val = scored["split"].eq("val")
    validation_error = []
    for row in scored.loc[val].itertuples(index=False):
        pair = (str(row.context), str(row.perturbation))
        truth = store.effect(*pair)
        validation_error.append(float(np.mean([rmse(pred_a[pair], truth), rmse(pred_b[pair], truth)])))
    x_val = scored.loc[val, ["risk_disagreement_z", "predicted_magnitude_z"]].to_numpy(float)
    y_val = np.asarray(validation_error, dtype=float)
    calibrator = Ridge(alpha=1.0, positive=True).fit(x_val, y_val)
    x_all = scored[["risk_disagreement_z", "predicted_magnitude_z"]].to_numpy(float)
    base_error = calibrator.predict(x_all)
    structural_scale = max(float(np.std(y_val)), 0.05 * float(np.mean(y_val)), 1e-6)
    support_reference = float(scored.loc[val, "perturbation_novelty"].median())
    structural_novelty = (
        scored["context_novelty_scaled"].to_numpy(float)
        + np.maximum(scored["perturbation_novelty"].to_numpy(float) - support_reference, 0.0)
    )
    scored = scored.copy()
    scored["safeconf_calibrated_pair_risk"] = base_error + structural_scale * structural_novelty
    threshold = float(scored.loc[val, "safeconf_calibrated_pair_risk"].quantile(0.80))
    scored["accepted_by_validation_q80"] = scored["safeconf_calibrated_pair_risk"].le(threshold)
    scored["validation_q80_threshold"] = threshold
    scored["validation_truth_used_for_calibration_only"] = True
    return scored, {
        "ridge_alpha": 1.0,
        "feature_order": ["risk_disagreement_z", "predicted_magnitude_z"],
        "coefficients": calibrator.coef_.astype(float).tolist(),
        "intercept": float(calibrator.intercept_),
        "structural_scale_from_validation_error_sd": structural_scale,
        "validation_support_novelty_reference": support_reference,
        "validation_q80_threshold": threshold,
        "n_validation_tasks": int(val.sum()),
    }


def attach_truth_and_records(
    store: EffectStore,
    scored: pd.DataFrame,
    pred_a: dict[tuple[str, str], np.ndarray],
    pred_b: dict[tuple[str, str], np.ndarray],
    fold: str,
    fraction: int,
    predicted_arrays: dict[str, np.ndarray],
    true_arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[dict]]:
    """This is the first evaluation function allowed to read test-task truth."""
    task_rows, records = [], []
    panel_hash = hash_order(store.genes)
    model_names = ("SourceEffect_scGPTKNN", "scGPTEmbedding_ContextRidge")
    for row in scored.itertuples(index=False):
        pair = (str(row.context), str(row.perturbation))
        truth = store.effect(*pair)
        predictions = (pred_a[pair], pred_b[pair])
        task_key = f"E98::{fold}::train{fraction}::{pair[0]}::{pair[1]}"
        true_key = task_key + "::truth"
        true_arrays[true_key] = truth.astype(np.float32)
        errors = [rmse(prediction, truth) for prediction in predictions]
        task = row._asdict()
        task.update(
            {
                "fold_id": fold,
                "train_fraction": fraction / 100.0,
                "error_source_knn_rmse": errors[0],
                "error_context_ridge_rmse": errors[1],
                "error_two_predictor_mean_rmse": float(np.mean(errors)),
                "error_two_predictor_max_rmse": float(np.max(errors)),
            }
        )
        task_rows.append(task)
        for name, prediction, error in zip(model_names, predictions, errors):
            record_id = task_key + f"::{name}"
            pred_key = record_id + "::prediction"
            predicted_arrays[pred_key] = prediction.astype(np.float32)
            records.append(
                {
                    "schema_version": "safeconf_prediction_record_v1",
                    "record_id": record_id,
                    "task_id": f"{pair[0]}::{pair[1]}",
                    "task_key": task_key,
                    "dataset_name": "Frangieh2019_E97_Cartesian",
                    "dataset_group": "frangieh_gene_context_cartesian",
                    "fold_id": f"{fold}__train{fraction}",
                    "split": str(row.split),
                    "context": pair[0],
                    "perturbation": pair[1],
                    "predictor_name": name,
                    "run_type": "formal",
                    "gene_panel_id": "frangieh_processed_3000",
                    "gene_order_hash": panel_hash,
                    "effect_definition": "mean_diff",
                    "normalization_id": "frangieh_processed_mean_minus_context_specific_ctrl_v1",
                    "error_normalization": "raw_rmse",
                    "predicted_effect_key": pred_key,
                    "true_effect_key": true_key,
                    "true_error_rmse": error,
                    "true_error_cosine": cosine_error(prediction, truth),
                    "n_cells": int(row.n_cells),
                }
            )
    return pd.DataFrame(task_rows), records


def risk_coverage_improvement(score: np.ndarray, error: np.ndarray, coverage: float = 0.80) -> float:
    keep = max(1, int(math.ceil(len(score) * coverage)))
    selected = np.argsort(score, kind="stable")[:keep]
    return float(100.0 * (error.mean() - error[selected].mean()) / max(error.mean(), 1e-12))


def summarize(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scores = [
        "safeconf_calibrated_pair_risk",
        "safeconf_frozen_pair_risk",
        "risk_model_disagreement",
        "baseline_predicted_magnitude",
    ]
    test = tasks[tasks["split"].eq("test")]
    for (fold, fraction), fold_group in test.groupby(["fold_id", "train_fraction"], sort=True):
        groups = [(setting, group) for setting, group in fold_group.groupby("setting", sort=True)]
        groups.append(("all_test_settings_pooled", fold_group))
        for setting, group in groups:
            error = group["error_two_predictor_mean_rmse"].to_numpy(float)
            for score in scores:
                value = group[score].to_numpy(float)
                rows.append(
                    {
                        "fold_id": fold,
                        "train_fraction": fraction,
                        "setting": setting,
                        "score": score,
                        "n_tasks": len(group),
                        "mean_error": float(error.mean()),
                        "spearman": spearman(value, error),
                        "risk_coverage80_improve_pct": risk_coverage_improvement(value, error),
                        "accepted_by_validation_q80_fraction": float(group["accepted_by_validation_q80"].mean()),
                        "accepted_by_validation_q80_error": float(
                            group.loc[group["accepted_by_validation_q80"], "error_two_predictor_mean_rmse"].mean()
                        ) if group["accepted_by_validation_q80"].any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def pooled_bootstrap(tasks: pd.DataFrame, n_bootstrap: int = 2000) -> pd.DataFrame:
    full = tasks[tasks["split"].eq("test") & np.isclose(tasks["train_fraction"], 1.0)]
    rng = np.random.default_rng(SEED + 98)
    primary = "safeconf_calibrated_pair_risk"
    rows = []
    fold_cache = {}
    for fold, group in full.groupby("fold_id"):
        group = group.reset_index(drop=True)
        cluster_indexes = [
            np.flatnonzero(group["perturbation"].astype(str).to_numpy() == perturbation)
            for perturbation in sorted(group["perturbation"].astype(str).unique())
        ]
        fold_cache[str(fold)] = (group, cluster_indexes)
    fold_names = sorted(fold_cache)
    for comparator in ["safeconf_frozen_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]:
        observed = []
        for _, group in full.groupby("fold_id"):
            error = group["error_two_predictor_mean_rmse"].to_numpy(float)
            observed.append(spearman(group[primary].to_numpy(float), error) - spearman(group[comparator].to_numpy(float), error))
        for bootstrap_unit in ["task_row", "outer_fold_plus_perturbation_cluster"]:
            samples = []
            for _ in range(n_bootstrap):
                fold_deltas = []
                sampled_folds = (
                    fold_names
                    if bootstrap_unit == "task_row"
                    else rng.choice(fold_names, len(fold_names), replace=True).tolist()
                )
                for fold in sampled_folds:
                    group, cluster_indexes = fold_cache[str(fold)]
                    if bootstrap_unit == "task_row":
                        index = rng.integers(0, len(group), len(group))
                    else:
                        draws = rng.integers(0, len(cluster_indexes), len(cluster_indexes))
                        index = np.concatenate([cluster_indexes[int(draw)] for draw in draws])
                    error = group["error_two_predictor_mean_rmse"].to_numpy(float)[index]
                    fold_deltas.append(
                        spearman(group[primary].to_numpy(float)[index], error)
                        - spearman(group[comparator].to_numpy(float)[index], error)
                    )
                samples.append(float(np.nanmean(fold_deltas)))
            values = np.asarray(samples, dtype=float)
            rows.append(
                {
                    "primary_score": primary,
                    "comparator": comparator,
                    "setting": "all_test_settings_pooled",
                    "train_fraction": 1.0,
                    "bootstrap_unit": bootstrap_unit,
                    "n_folds": int(full["fold_id"].nunique()),
                    "n_unique_perturbations": int(full["perturbation"].nunique()),
                    "n_task_rows": len(full),
                    "observed_macro_delta_spearman": float(np.nanmean(observed)),
                    "bootstrap_ci95_low": float(np.nanquantile(values, 0.025)),
                    "bootstrap_ci95_high": float(np.nanquantile(values, 0.975)),
                    "bootstrap_probability_delta_gt_zero": float(np.nanmean(values > 0)),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def write_svg(summary: pd.DataFrame) -> None:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["setting", "score"], as_index=False)["spearman"].mean()
    settings = ["random_missing_pair", "context_unseen_row", "perturbation_unseen_column", "context_and_perturbation_unseen"]
    labels = ["随机缺失 pair", "整行新背景", "整列新扰动", "背景与扰动双未见"]
    scores = ["safeconf_calibrated_pair_risk", "risk_model_disagreement", "baseline_predicted_magnitude"]
    colors = {"safeconf_calibrated_pair_risk": "#2f6f8f", "risk_model_disagreement": "#7f9f8b", "baseline_predicted_magnitude": "#b78655"}
    width, height = 1120, 670
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#26343d}.title{font-size:26px;font-weight:700}.sub{font-size:15px;fill:#5d6870}.lab{font-size:15px}.tick{font-size:13px;fill:#63717a}</style>',
        '<text x="54" y="45" class="title">E98｜Frangieh 三背景矩阵：风险排序在四类缺失任务上的表现</text>',
        '<text x="54" y="73" class="sub">100% 训练子矩阵；每根柱为 3 个背景留出 fold 的 Spearman 均值。白底图可直接放入组会材料。</text>',
    ]
    x0, y0, plot_w, plot_h = 110, 115, 930, 455
    minimum, maximum = -0.4, 1.0
    zero_y = y0 + (maximum / (maximum - minimum)) * plot_h
    for tick in [-0.4, 0.0, 0.4, 0.8, 1.0]:
        yy = y0 + (maximum - tick) / (maximum - minimum) * plot_h
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+plot_w}" y2="{yy:.1f}" stroke="#dfe5e8"/>')
        parts.append(f'<text x="{x0-16}" y="{yy+5:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    group_w = plot_w / len(settings)
    bar_w = 46
    for gi, (setting, label) in enumerate(zip(settings, labels)):
        center = x0 + group_w * (gi + 0.5)
        for si, score in enumerate(scores):
            hit = macro[(macro["setting"] == setting) & (macro["score"] == score)]
            value = float(hit["spearman"].iloc[0]) if len(hit) else float("nan")
            if not np.isfinite(value):
                continue
            xx = center + (si - 1) * (bar_w + 8) - bar_w / 2
            yy = y0 + (maximum - value) / (maximum - minimum) * plot_h
            top, bottom = min(yy, zero_y), max(yy, zero_y)
            parts.append(f'<rect x="{xx:.1f}" y="{top:.1f}" width="{bar_w}" height="{max(bottom-top,1):.1f}" fill="{colors[score]}"/>')
            parts.append(f'<text x="{xx+bar_w/2:.1f}" y="{top-7:.1f}" text-anchor="middle" class="tick">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="610" text-anchor="middle" class="lab">{label}</text>')
    legend = [("safeconf_calibrated_pair_risk", "SafeConf 校准 pair risk"), ("risk_model_disagreement", "双模型分歧"), ("baseline_predicted_magnitude", "预测幅度")]
    for i, (score, label) in enumerate(legend):
        xx = 260 + i * 245
        parts.append(f'<rect x="{xx}" y="638" width="18" height="12" fill="{colors[score]}"/><text x="{xx+27}" y="650" class="tick">{label}</text>')
    parts.append('</svg>')
    (FIGURES / "F1_four_setting_risk_spearman.svg").write_text("\n".join(parts), encoding="utf-8")


def write_report(tasks: pd.DataFrame, summary: pd.DataFrame, bootstrap: pd.DataFrame, status: dict) -> None:
    full = summary[np.isclose(summary["train_fraction"], 1.0)]
    macro = full.groupby(["setting", "score"], as_index=False).agg(
        folds=("fold_id", "nunique"),
        n_tasks_per_fold=("n_tasks", "median"),
        mean_spearman=("spearman", "mean"),
        mean_rc80_improve_pct=("risk_coverage80_improve_pct", "mean"),
        mean_error=("mean_error", "mean"),
    )
    macro.to_csv(TABLES / "E98_FULL_FRACTION_MACRO_SUMMARY.csv", index=False)
    pivot = macro.pivot(index="setting", columns="score", values="mean_spearman").reset_index()
    lines = [
        "# E98｜Frangieh 三背景遗传扰动矩阵正式预测与风险审计",
        "",
        "## 回答周老师的输入问题",
        "",
        "每个测试任务的预测输入只有：训练子矩阵内已观测的扰动效应、目标背景的未扰动 control 表达、扰动基因的 scGPT 预训练 embedding。SourceEffect-scGPTKNN 先查同一扰动在训练背景的效应；整列新扰动没有历史效应时，按 scGPT embedding 找相邻训练基因。ContextRidge 使用 scGPT embedding、目标背景 control 和两者交互项，由训练 pair 拟合。两个预测向量的分歧和预测幅度在 30 个 validation pair 上校准，训练支持数与背景 control 距离提供结构性新颖度。测试扰动后的真实表达没有进入预测、分数或阈值，只在这些量冻结后计算 RMSE。",
        "",
        "## 100% 训练子矩阵结果",
        "",
        "| setting | SafeConf 校准 pair risk ρ | disagreement ρ | magnitude ρ |",
        "|---|---:|---:|---:|",
    ]
    for row in pivot.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {getattr(row, 'safeconf_calibrated_pair_risk'):.3f} | "
            f"{getattr(row, 'risk_model_disagreement'):.3f} | "
            f"{getattr(row, 'baseline_predicted_magnitude'):.3f} |"
        )
    lines += [
        "",
        "表中数值是三个整行留出 fold 的 Spearman 宏平均。`all_test_settings_pooled` 把四类任务放回同一个待质检队列，同时检查任务类型之间和同类型内部的排序。完整的 25%/50%/75%/100% 训练量、各 fold、风险覆盖率和验证阈值结果保存在 `tables/E98_RISK_SUMMARY.csv`。",
        "",
        "## 与强基线的配对 bootstrap",
        "",
        "| comparator | bootstrap unit | Δρ（SafeConf−基线） | 95% CI | P(Δ>0) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.comparator} | {row.bootstrap_unit} | {row.observed_macro_delta_spearman:.3f} | "
            f"[{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}] | "
            f"{row.bootstrap_probability_delta_gt_zero:.3f} |"
        )
    acceptance = full[full["score"].eq("safeconf_calibrated_pair_risk")].groupby("setting", as_index=False).agg(
        accepted_fraction=("accepted_by_validation_q80_fraction", "mean"),
        accepted_error=("accepted_by_validation_q80_error", "mean"),
        all_error=("mean_error", "mean"),
    )
    lines += [
        "",
        "## validation q80 阈值审计",
        "",
        "| setting | 实际接受比例 | 接受任务 RMSE | 全部任务 RMSE |",
        "|---|---:|---:|---:|",
    ]
    for row in acceptance.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {row.accepted_fraction:.3f} | {row.accepted_error:.4f} | {row.all_error:.4f} |"
        )
    lines += [
        "",
        "q80 是 validation 分位阈值，不是覆盖率保证。尤其在双未见任务上，validation 的任务类型不匹配，接受集误差没有下降；该结果保留为下一轮分层校准的直接依据。",
        "",
        "## 解释边界",
        "",
        "两个预测器是可复现的训练数据预测器：一个做来源效应迁移与 scGPT embedding 邻域插值，一个做 scGPT embedding 和背景 control 的监督 Ridge。第二个使用了 scGPT 预训练表示，但不是 scGPT Transformer 的端到端微调；第一个也不是 GEARS。E98 因而直接验证矩阵 setting、验证集校准和输入防泄漏，不替代后续同合同下的 GEARS/scGPT 正式重训。配对 bootstrap 若跨过 0，只能写成趋势。跨数据集结果仍引用已完成的 sciPlex 压力测试，不把其负结果改写为成功。",
        "",
        f"strict PredictionRecord issue_count = {status['strict_issue_count']}；共 {status['n_task_rows']} 个任务行、{status['n_prediction_records']} 条双预测器记录。",
        "",
        "## 文件",
        "",
        "- 任务级分数与误差：`tables/E98_TASK_RISK_TABLE.csv`",
        "- 严格预测合同：`tables/PREDICTION_RECORDS.csv`",
        "- 每折每 setting 汇总：`tables/E98_RISK_SUMMARY.csv`",
        "- Ridge 验证选参：`tables/E98_RIDGE_VALIDATION.csv`",
        "- 校准风险配对 bootstrap：`tables/E98_POOLED_BOOTSTRAP.csv`",
        "- 白底结果图：`figures/F1_four_setting_risk_spearman.svg`",
    ]
    (REPORTS / "E98_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E98 先看这个\n\n先读 `reports/E98_REPORT.md`。E98 接 E97 冻结矩阵，正式执行两个训练数据预测器、四类缺失 setting、四档训练量和严格 PredictionRecord 审计。\n",
        encoding="utf-8",
    )


def main() -> None:
    for path in (TABLES, ARRAYS, REPORTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CONTRACT)
    assets = prepare_assets(manifest)
    store = EffectStore(assets)
    embeddings, embedding_meta = load_scgpt_embeddings(store.perts)
    features, feature_meta = make_features(store, embeddings)
    predicted_arrays: dict[str, np.ndarray] = {}
    true_arrays: dict[str, np.ndarray] = {}
    all_tasks, all_records, ridge_rows, calibration_rows = [], [], [], []

    for fold, fold_frame in manifest.groupby("fold_id", sort=True):
        validation_frame = fold_frame[fold_frame["split"].eq("val")].copy()
        validation_pairs = list(zip(validation_frame["context"].astype(str), validation_frame["perturbation"].astype(str)))
        for fraction in FRACTIONS:
            selected = fold_frame["split"].eq("train") & fold_frame[f"in_train_fraction_{fraction}"].astype(bool)
            train_frame = fold_frame[selected].copy()
            train_pairs = list(zip(train_frame["context"].astype(str), train_frame["perturbation"].astype(str)))
            query_frame = fold_frame[fold_frame["split"].isin(["val", "test"])].copy()
            query_pairs = list(zip(query_frame["context"].astype(str), query_frame["perturbation"].astype(str)))

            pred_a = source_knn_predictions(store, embeddings, train_pairs, query_pairs)
            ridge, alpha, alpha_table = fit_context_ridge(store, features, train_pairs, validation_pairs)
            query_x = np.stack([features[store.cix[c], store.pix[p]] for c, p in query_pairs])
            ridge_values = ridge.predict(query_x).astype(np.float32)
            pred_b = {pair: vector for pair, vector in zip(query_pairs, ridge_values)}
            for row in alpha_table.itertuples(index=False):
                ridge_rows.append(
                    {"fold_id": fold, "train_fraction": fraction / 100.0, "alpha": row.alpha,
                     "validation_profile_rmse": row.validation_profile_rmse, "selected": row.alpha == alpha}
                )

            scored = score_queries_without_truth(store, train_pairs, query_frame, pred_a, pred_b)
            scored, calibration = calibrate_pair_risk_with_validation_truth(store, scored, pred_a, pred_b)
            calibration_rows.append({"fold_id": fold, "train_fraction": fraction / 100.0, **calibration})
            evaluated, records = attach_truth_and_records(
                store, scored, pred_a, pred_b, str(fold), fraction, predicted_arrays, true_arrays
            )
            all_tasks.append(evaluated)
            all_records.extend(records)
            print(f"[E98] {fold} train={fraction}% tasks={len(evaluated)} alpha={alpha}", flush=True)

    tasks = pd.concat(all_tasks, ignore_index=True)
    records = pd.DataFrame(all_records)
    ridge_table = pd.DataFrame(ridge_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    tasks.to_csv(TABLES / "E98_TASK_RISK_TABLE.csv", index=False)
    records.to_csv(TABLES / "PREDICTION_RECORDS.csv", index=False)
    ridge_table.to_csv(TABLES / "E98_RIDGE_VALIDATION.csv", index=False)
    calibration_table.to_csv(TABLES / "E98_RISK_CALIBRATION.csv", index=False)
    np.savez_compressed(ARRAYS / "predicted_effects.npz", **predicted_arrays)
    np.savez_compressed(ARRAYS / "true_effects.npz", **true_arrays)
    issues = validate_prediction_record_artifacts(OUT, records=records, strict=True)
    pd.DataFrame({"strict_issue": issues}).to_csv(TABLES / "E98_STRICT_CONTRACT_ISSUES.csv", index=False)
    summary = summarize(tasks)
    summary.to_csv(TABLES / "E98_RISK_SUMMARY.csv", index=False)
    bootstrap = pooled_bootstrap(tasks)
    bootstrap.to_csv(TABLES / "E98_POOLED_BOOTSTRAP.csv", index=False)
    write_svg(summary)
    status = {
        "experiment": "E98_frangieh_gene_cartesian_predictions",
        "generated_at": now(),
        "git_head_at_run": git_head(),
        "source_h5ad": str(SOURCE),
        "source_h5ad_sha256": json.loads((CONTRACT.parent.parent / "RUN_STATUS.json").read_text())["source_sha256"],
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": file_sha256(CONTRACT),
        "effect_asset_cache": str(ASSET_CACHE),
        "effect_asset_cache_sha256": file_sha256(ASSET_CACHE),
        "contexts": store.contexts,
        "n_perturbations": len(store.perts),
        "n_genes": len(store.genes),
        "training_fractions": list(FRACTIONS),
        "predictors": ["SourceEffect_scGPTKNN", "scGPTEmbedding_ContextRidge"],
        "embedding": embedding_meta,
        "features": feature_meta,
        "ridge_alpha_candidates": list(RIDGE_ALPHAS),
        "knn_neighbors": K_NEIGHBORS,
        "n_task_rows": len(tasks),
        "n_prediction_records": len(records),
        "strict_issue_count": len(issues),
        "strict_issues": issues,
        "target_test_perturbed_truth_used_in_prediction_or_risk": False,
        "validation_truth_used_for_ridge_alpha_risk_calibration_and_q80_threshold": True,
        "risk_calibration_rows": len(calibration_table),
        "pooled_bootstrap_rows": len(bootstrap),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(tasks, summary, bootstrap, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
