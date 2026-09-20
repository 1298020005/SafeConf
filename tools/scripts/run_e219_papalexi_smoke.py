#!/usr/bin/env python3
"""Run a sealed three-fold Papalexi technical-replicate smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import run_e219_adamson_smoke as core


SOURCE = Path(
    "/home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Papalexi.h5ad"
)
DEFAULT_RUN_ROOT = Path("/home/yyf/data/safeconf_e219/papalexi_smoke")
AUTHORIZATION = core.E219 / "E219_PAPALEXI_TRUTH_AUTHORIZATION.json"
ALIASES = {"MARCH8": "MARCHF8"}


def embeddings(perturbations: list[str]) -> dict[str, np.ndarray]:
    vocab = json.loads((core.CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(core.CHECKPOINT / "best_model.pt", map_location="cpu")
    weights = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    result = {}
    for perturbation in perturbations:
        token = ALIASES.get(perturbation.upper(), perturbation.upper())
        if token not in vocab:
            raise core.SmokeFailure(f"scGPT vocabulary missing {perturbation}->{token}")
        vector = weights[int(vocab[token])]
        result[perturbation] = vector / max(float(np.linalg.norm(vector)), 1e-8)
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def load_fold_source(
    train_pairs: set[tuple[str, str]], val_pairs: set[tuple[str, str]]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[tuple[str, str], np.ndarray], dict]:
    """Read controls and fold train/validation pairs, never fold test pairs."""
    allowed = train_pairs | val_pairs
    data = ad.read_h5ad(SOURCE, backed="r")
    contexts = data.obs["replicate"].astype(str).to_numpy()
    perturbations = data.obs["perturbation"].astype(str).to_numpy()
    pair_labels = np.asarray(
        [f"{context}\x1f{perturbation}" for context, perturbation in zip(contexts, perturbations)]
    )
    allowed_labels = {f"{context}\x1f{perturbation}" for context, perturbation in allowed}
    control_labels = {f"{context}\x1fcontrol" for context in sorted(set(contexts))}
    keep = np.isin(pair_labels, sorted(allowed_labels | control_labels))
    matrix = data.X[np.flatnonzero(keep), :]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    means = core.group_means(matrix, pair_labels[keep])
    genes = np.asarray(data.var_names.astype(str), dtype=str)
    data.file.close()
    controls = {
        context: means[f"{context}\x1fcontrol"] for context in sorted(set(contexts))
    }
    effects = {
        pair: means[f"{pair[0]}\x1f{pair[1]}"] - controls[pair[0]]
        for pair in sorted(allowed)
    }
    return genes, controls, effects, {
        "source_rows_read": int(keep.sum()),
        "target_test_expression_rows_read": 0,
    }


def feature_map(
    contexts: list[str],
    perturbations: list[str],
    controls: dict[str, np.ndarray],
    embedding: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    gene_matrix = np.stack([embedding[gene] for gene in perturbations])
    gene_pca = PCA(
        n_components=min(12, len(perturbations) - 1), random_state=20260920
    ).fit_transform(gene_matrix)
    control_matrix = np.stack([controls[context] for context in contexts])
    context_pca = PCA(
        n_components=min(2, len(contexts) - 1), random_state=20260920
    ).fit_transform(control_matrix)
    gene_pca = (gene_pca - gene_pca.mean(0)) / np.maximum(gene_pca.std(0), 1e-8)
    context_pca = (context_pca - context_pca.mean(0)) / np.maximum(
        context_pca.std(0), 1e-8
    )
    result = {}
    for ci, context in enumerate(contexts):
        for pi, perturbation in enumerate(perturbations):
            interaction = np.outer(gene_pca[pi, :6], context_pca[ci]).ravel()
            result[(context, perturbation)] = np.concatenate(
                [gene_pca[pi], context_pca[ci], interaction]
            ).astype(np.float32)
    return result


def prepare(run_root: Path) -> None:
    if (run_root / "PRETRUTH_STATUS.json").exists():
        raise core.SmokeFailure("pretruth run already exists; refusing overwrite")
    manifest = pd.read_csv(core.MANIFEST)
    tasks = manifest.loc[manifest.dataset.eq("Papalexi")].copy()
    all_perturbations = sorted(tasks.perturbation.astype(str).unique())
    all_contexts = sorted(tasks.context.astype(str).unique())
    embedding = embeddings(all_perturbations)
    table_parts, array_keys, arrays_a, arrays_b = [], [], [], []
    ridge_rows, source_rows = [], []
    genes_reference: np.ndarray | None = None

    for fold, fold_frame in tasks.groupby("fold_id", sort=True):
        train_frame = fold_frame.loc[fold_frame.split.eq("train")]
        val_frame = fold_frame.loc[fold_frame.split.eq("val")]
        query_frame = fold_frame.loc[fold_frame.split.isin(["val", "test"])].copy()
        train_pairs = set(zip(train_frame.context.astype(str), train_frame.perturbation.astype(str)))
        val_pairs = set(zip(val_frame.context.astype(str), val_frame.perturbation.astype(str)))
        query_pairs = list(zip(query_frame.context.astype(str), query_frame.perturbation.astype(str)))
        genes, controls, effects, source_status = load_fold_source(train_pairs, val_pairs)
        source_rows.append({"fold_id": fold, **source_status})
        if genes_reference is None:
            genes_reference = genes
        elif not np.array_equal(genes_reference, genes):
            raise core.SmokeFailure("Papalexi gene axis changed across folds")

        features = feature_map(all_contexts, all_perturbations, controls, embedding)
        train_x = np.stack([features[pair] for pair in sorted(train_pairs)])
        train_y = np.stack([effects[pair] for pair in sorted(train_pairs)])
        val_x = np.stack([features[pair] for pair in sorted(val_pairs)])
        val_y = np.stack([effects[pair] for pair in sorted(val_pairs)])
        models, validation_rows = [], []
        for alpha in core.ALPHAS:
            model = Ridge(alpha=alpha).fit(train_x, train_y)
            error = core.rmse(model.predict(val_x), val_y)
            models.append(model)
            validation_rows.append({"alpha": alpha, "validation_rmse": error})
        best = int(np.argmin([row["validation_rmse"] for row in validation_rows]))
        ridge = models[best]
        for row in validation_rows:
            ridge_rows.append(
                {"fold_id": fold, **row, "selected": row["alpha"] == core.ALPHAS[best]}
            )

        by_perturbation: dict[str, list[np.ndarray]] = {}
        for context, perturbation in sorted(train_pairs):
            by_perturbation.setdefault(perturbation, []).append(effects[(context, perturbation)])
        prototypes = {
            perturbation: np.mean(np.stack(vectors), axis=0)
            for perturbation, vectors in by_perturbation.items()
        }
        available = sorted(prototypes)
        available_embedding = np.stack([embedding[gene] for gene in available])

        def source_prediction(pair: tuple[str, str]) -> np.ndarray:
            perturbation = pair[1]
            if perturbation in prototypes:
                return prototypes[perturbation].astype(np.float32)
            similarity = available_embedding @ embedding[perturbation]
            nearest = np.argsort(-similarity, kind="stable")[: core.K_NEIGHBORS]
            weights = np.exp((similarity[nearest] - similarity[nearest].max()) / 0.10)
            weights /= weights.sum()
            return np.sum(
                np.stack([prototypes[available[index]] for index in nearest])
                * weights[:, None],
                axis=0,
            ).astype(np.float32)

        pred_a = np.stack([source_prediction(pair) for pair in query_pairs]).astype(np.float32)
        pred_b = ridge.predict(np.stack([features[pair] for pair in query_pairs])).astype(np.float32)
        mean_prediction = (pred_a + pred_b) / 2.0
        magnitude = np.sqrt(np.mean(np.square(mean_prediction), axis=1))
        disagreement = np.sqrt(np.mean(np.square(pred_a - pred_b), axis=1))
        train_contexts = sorted({pair[0] for pair in train_pairs})
        support = pd.Series([pair[1] for pair in train_pairs]).value_counts().to_dict()
        context_novelty = np.asarray(
            [
                1.0
                - max(cosine(controls[pair[0]], controls[source]) for source in train_contexts)
                for pair in query_pairs
            ],
            dtype=float,
        )
        perturbation_novelty = np.asarray(
            [1.0 / (1.0 + int(support.get(pair[1], 0))) for pair in query_pairs],
            dtype=float,
        )
        scored = query_frame[["context", "perturbation", "split", "setting", "n_cells"]].copy()
        scored.insert(0, "fold_id", fold)
        scored["predicted_magnitude"] = magnitude
        scored["model_disagreement"] = disagreement
        scored["context_novelty"] = context_novelty
        scored["perturbation_novelty"] = perturbation_novelty
        for split, indices in scored.groupby("split", sort=False).groups.items():
            ix = np.asarray(list(indices), dtype=int)
            m = core.percentile(scored.loc[ix, "predicted_magnitude"].to_numpy(float))
            d = core.percentile(scored.loc[ix, "model_disagreement"].to_numpy(float))
            c = core.percentile(scored.loc[ix, "context_novelty"].to_numpy(float))
            p = core.percentile(scored.loc[ix, "perturbation_novelty"].to_numpy(float))
            structural = (d + c + p) / 3.0
            combined = m + 0.50 * np.maximum(structural - m, 0.0) + 0.125 * np.maximum(d - m, 0.0)
            scored.loc[ix, "magnitude_percentile"] = m
            scored.loc[ix, "disagreement_percentile"] = d
            scored.loc[ix, "context_novelty_percentile"] = c
            scored.loc[ix, "perturbation_novelty_percentile"] = p
            scored.loc[ix, "safeconf_structural_proxy"] = structural
            scored.loc[ix, "e218_transfer_router"] = combined
        scored["test_truth_used_for_features"] = False
        table_parts.append(scored)
        for pair, left, right in zip(query_pairs, pred_a, pred_b):
            array_keys.append(f"{fold}|{pair[0]}|{pair[1]}")
            arrays_a.append(left)
            arrays_b.append(right)

    table = pd.concat(table_parts, ignore_index=True)
    run_root.mkdir(parents=True, exist_ok=True)
    array_path = run_root / "PRETRUTH_PREDICTIONS.npz"
    np.savez_compressed(
        array_path,
        task_keys=np.asarray(array_keys, dtype=str),
        genes=genes_reference,
        pred_source=np.stack(arrays_a).astype(np.float32),
        pred_ridge=np.stack(arrays_b).astype(np.float32),
    )
    table_path = run_root / "PRETRUTH_RISK_FEATURES.csv"
    ridge_path = run_root / "RIDGE_VALIDATION.csv"
    core.atomic_csv(table_path, table)
    core.atomic_csv(ridge_path, pd.DataFrame(ridge_rows))
    status = {
        "experiment": "E219_Papalexi_technical_replicate_smoke",
        "stage": "PRETRUTH_SEAL",
        "status": "PASS",
        "generated_at": core.now(),
        "n_folds": int(tasks.fold_id.nunique()),
        "n_train_rows": int((tasks.split == "train").sum()),
        "n_validation_rows": int((tasks.split == "val").sum()),
        "n_test_rows": int((tasks.split == "test").sum()),
        "n_output_genes": int(len(genes_reference)),
        "predictors": ["source_effect_scGPT_KNN", "scGPT_embedding_context_ridge"],
        "validation_truth_used_for_model_selection": True,
        "test_truth_used": False,
        "target_test_expression_rows_read": 0,
        "source_rows_by_fold": source_rows,
        "normalization": "source_h5ad_X_logNor_as_provided",
        "symbol_aliases": ALIASES,
        "prediction_sha256": core.sha256(array_path),
        "risk_table_sha256": core.sha256(table_path),
        "ridge_validation_sha256": core.sha256(ridge_path),
        "context_scope": "technical replicates; no claim of new biological cell context",
        "method_scope": "exploratory external adapter smoke",
    }
    core.atomic_json(run_root / "PRETRUTH_STATUS.json", status)
    core.atomic_json(core.E219 / "E219_PAPALEXI_PRETRUTH_SEAL.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def evaluate(run_root: Path, authorization: Path) -> None:
    status_path = run_root / "PRETRUTH_STATUS.json"
    table_path = run_root / "PRETRUTH_RISK_FEATURES.csv"
    arrays_path = run_root / "PRETRUTH_PREDICTIONS.npz"
    if not all(path.is_file() for path in (status_path, table_path, arrays_path, authorization)):
        raise core.SmokeFailure("pretruth artifacts or authorization missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    if auth.get("status") != "AUTHORIZED" or auth.get("pretruth_status_sha256") != core.sha256(status_path):
        raise core.SmokeFailure("evaluation authorization does not match pretruth seal")
    table = pd.read_csv(table_path)
    test = table.loc[table.split.eq("test")].copy().reset_index(drop=True)
    with np.load(arrays_path, allow_pickle=False) as store:
        keys = store["task_keys"].astype(str).tolist()
        pred_a, pred_b = store["pred_source"], store["pred_ridge"]
    array_index = {key: index for index, key in enumerate(keys)}

    data = ad.read_h5ad(SOURCE, backed="r")
    contexts = data.obs["replicate"].astype(str).to_numpy()
    perturbations = data.obs["perturbation"].astype(str).to_numpy()
    pair_labels = np.asarray(
        [f"{context}\x1f{perturbation}" for context, perturbation in zip(contexts, perturbations)]
    )
    required = {f"{row.context}\x1f{row.perturbation}" for row in test.itertuples(index=False)}
    required |= {f"{context}\x1fcontrol" for context in sorted(set(contexts))}
    keep = np.isin(pair_labels, sorted(required))
    matrix = data.X[np.flatnonzero(keep), :]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    means = core.group_means(matrix, pair_labels[keep])
    data.file.close()

    rows = []
    for row in test.itertuples(index=False):
        key = f"{row.fold_id}|{row.context}|{row.perturbation}"
        index = array_index[key]
        control = means[f"{row.context}\x1fcontrol"]
        truth = means[f"{row.context}\x1f{row.perturbation}"] - control
        left, right = pred_a[index], pred_b[index]
        rows.append(
            {
                **row._asdict(),
                "error_source_rmse": core.rmse(left, truth),
                "error_ridge_rmse": core.rmse(right, truth),
                "error_two_predictor_mean_rmse": (core.rmse(left, truth) + core.rmse(right, truth)) / 2.0,
                "error_no_change_rmse": core.rmse(np.zeros_like(truth), truth),
            }
        )
    evaluated = pd.DataFrame(rows)
    score_names = (
        "predicted_magnitude",
        "model_disagreement",
        "safeconf_structural_proxy",
        "e218_transfer_router",
    )
    metric_rows = []
    for (fold, setting), group in evaluated.groupby(["fold_id", "setting"], sort=True):
        if len(group) < 3:
            continue
        error = group.error_two_predictor_mean_rmse.to_numpy(float)
        for score in score_names:
            association = spearmanr(group[score].to_numpy(float), error)
            n_review = max(1, int(np.ceil(0.20 * len(group))))
            selected = np.argsort(-group[score].to_numpy(float), kind="stable")[:n_review]
            metric_rows.append(
                {
                    "fold_id": fold,
                    "setting": setting,
                    "score": score,
                    "n_tasks": len(group),
                    "spearman": float(association.statistic),
                    "top20_error_enrichment": float(error[selected].mean() / error.mean() - 1.0),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(["setting", "score"], as_index=False).agg(
        n_folds=("fold_id", "nunique"),
        mean_spearman=("spearman", "mean"),
        mean_top20_error_enrichment=("top20_error_enrichment", "mean"),
    )
    competence = evaluated.assign(
        gain=lambda frame: frame.error_no_change_rmse - frame.error_two_predictor_mean_rmse
    ).groupby("setting", as_index=False).agg(
        n_tasks=("perturbation", "size"),
        mean_model_error=("error_two_predictor_mean_rmse", "mean"),
        mean_no_change_error=("error_no_change_rmse", "mean"),
        mean_absolute_gain=("gain", "mean"),
    )
    competence["relative_error_reduction"] = 1.0 - competence.mean_model_error / competence.mean_no_change_error
    competence["upstream_gate"] = np.where(competence.mean_absolute_gain > 0, "PASS", "NOT_SUPPORTED")
    overall_model = float(evaluated.error_two_predictor_mean_rmse.mean())
    overall_no_change = float(evaluated.error_no_change_rmse.mean())
    output = core.E219 / "papalexi_smoke_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    core.atomic_csv(output / "E219_PAPALEXI_TASK_RESULTS.csv", evaluated)
    core.atomic_csv(output / "E219_PAPALEXI_FOLD_METRICS.csv", metrics)
    core.atomic_csv(output / "E219_PAPALEXI_MACRO_METRICS.csv", macro)
    core.atomic_csv(output / "E219_PAPALEXI_COMPETENCE_BY_SETTING.csv", competence)
    result = {
        "experiment": "E219_Papalexi_technical_replicate_smoke",
        "stage": "POSTSEAL_EVALUATION",
        "status": "PASS",
        "evaluated_at": core.now(),
        "n_test_rows": len(evaluated),
        "n_unique_context_perturbation_pairs": int(evaluated[["context", "perturbation"]].drop_duplicates().shape[0]),
        "mean_two_predictor_error": overall_model,
        "mean_no_change_error": overall_no_change,
        "relative_error_reduction_vs_no_change": float(1.0 - overall_model / overall_no_change),
        "overall_upstream_competence_gate": "PASS" if overall_model < overall_no_change else "NOT_SUPPORTED",
        "context_scope": "technical replicate stability, not biological context generalization",
        "claim_rule": "only settings with upstream_gate PASS may contribute routing evidence",
    }
    core.atomic_json(output / "E219_PAPALEXI_EVALUATION_STATUS.json", result)
    lines = [
        "# E219 Papalexi 三技术重复冒烟结果",
        "",
        f"- 测试行：{len(evaluated)}；不同背景—扰动对：{result['n_unique_context_perturbation_pairs']}。",
        f"- 两预测器平均 RMSE：{overall_model:.6f}；no-change：{overall_no_change:.6f}。",
        f"- 整体上游能力门：{result['overall_upstream_competence_gate']}。",
        "- replicate 是技术重复，本实验不写成跨生物细胞背景验证。",
        "",
        "## 各任务类型的上游能力",
        "",
        competence.to_markdown(index=False),
        "",
        "## 三折宏平均风险结果",
        "",
        macro.to_markdown(index=False),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run_root.resolve())
    else:
        evaluate(args.run_root.resolve(), args.authorization.resolve())


if __name__ == "__main__":
    main()
