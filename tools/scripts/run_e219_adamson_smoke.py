#!/usr/bin/env python3
"""Run a sealed Adamson unseen-perturbation smoke in prepare/evaluate phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
E219 = ROOT / "docs/实验结果/E219_database_expansion_contract_20260920"
MANIFEST = E219 / "tables/E219_TASK_MANIFEST.csv"
SOURCE = Path(
    "/home/yyf/datasets/singlecell_perturbation_atlas/official_generalization/Adamson.h5ad"
)
CHECKPOINT = Path(
    "/home/yyf/archive/code/20260519_0958_home_cleanup/"
    "moved_top_level/codex_scgpt_attnres_workspace/checkpoints/whole-human"
)
DEFAULT_RUN_ROOT = Path("/home/yyf/data/safeconf_e219/adamson_smoke")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
K_NEIGHBORS = 8


class SmokeFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(left, dtype=np.float64)
                    - np.asarray(right, dtype=np.float64)
                )
            )
        )
    )


def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / len(values)


def paired_bootstrap_summary(evaluated: pd.DataFrame, n_bootstrap: int = 20_000) -> dict:
    """Quantify fixed-score increment and upstream improvement without refitting."""
    rng = np.random.default_rng(20260920)
    error = evaluated.error_two_predictor_mean_rmse.to_numpy(float)
    no_change = evaluated.error_no_change_rmse.to_numpy(float)
    magnitude = evaluated.predicted_magnitude.to_numpy(float)
    combined = evaluated.e218_transfer_router.to_numpy(float)
    delta_rho, upstream_gain = [], []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(evaluated), len(evaluated))
        rho_magnitude = spearmanr(magnitude[index], error[index]).statistic
        rho_combined = spearmanr(combined[index], error[index]).statistic
        if np.isfinite(rho_magnitude) and np.isfinite(rho_combined):
            delta_rho.append(float(rho_combined - rho_magnitude))
        upstream_gain.append(float(np.mean(no_change[index] - error[index])))
    delta_rho = np.asarray(delta_rho, dtype=float)
    upstream_gain = np.asarray(upstream_gain, dtype=float)
    return {
        "bootstrap_seed": 20260920,
        "n_bootstrap": n_bootstrap,
        "router_delta_spearman": float(
            spearmanr(combined, error).statistic
            - spearmanr(magnitude, error).statistic
        ),
        "router_delta_spearman_ci95": np.quantile(
            delta_rho, [0.025, 0.975]
        ).astype(float).tolist(),
        "router_probability_delta_gt_zero": float(np.mean(delta_rho > 0)),
        "upstream_absolute_rmse_gain": float(np.mean(no_change - error)),
        "upstream_absolute_rmse_gain_ci95": np.quantile(
            upstream_gain, [0.025, 0.975]
        ).astype(float).tolist(),
        "upstream_probability_gain_gt_zero": float(np.mean(upstream_gain > 0)),
    }


def group_means(matrix: sp.csr_matrix, labels: np.ndarray) -> dict[str, np.ndarray]:
    groups, codes = np.unique(labels.astype(str), return_inverse=True)
    membership = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    sums = membership @ matrix
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float32)
    means = np.asarray(sums.multiply((1.0 / counts)[:, None]).toarray(), dtype=np.float32)
    return {label: means[index] for index, label in enumerate(groups)}


def load_embeddings(perturbations: list[str]) -> np.ndarray:
    vocab = json.loads((CHECKPOINT / "vocab.json").read_text(encoding="utf-8"))
    state = torch.load(CHECKPOINT / "best_model.pt", map_location="cpu")
    weights = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
    missing = [gene for gene in perturbations if gene.upper() not in vocab]
    if missing:
        raise SmokeFailure(f"scGPT vocabulary missing Adamson genes: {missing}")
    vectors = np.stack([weights[int(vocab[gene.upper()])] for gene in perturbations])
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)


def source_effects(
    allowed: set[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Read control/train/validation X only; test perturbation X stays unopened."""
    data = ad.read_h5ad(SOURCE, backed="r")
    labels = data.obs["perturbation"].astype(str).to_numpy()
    keep = np.isin(labels, sorted(allowed | {"control"}))
    indices = np.flatnonzero(keep)
    matrix = data.X[indices, :]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    means = group_means(matrix, labels[indices])
    genes = np.asarray(data.var_names.astype(str), dtype=str)
    data.file.close()
    control = means["control"]
    effects = {gene: means[gene] - control for gene in sorted(allowed)}
    return genes, effects, {
        "source_rows_read": int(keep.sum()),
        "target_test_expression_rows_read": 0,
        "normalization": "source_h5ad_X_logNor_as_provided",
    }


def prepare(run_root: Path) -> None:
    if (run_root / "PRETRUTH_STATUS.json").exists():
        raise SmokeFailure("pretruth run already exists; refusing overwrite")
    manifest = pd.read_csv(MANIFEST)
    tasks = manifest.loc[manifest.dataset.eq("Adamson")].copy()
    train = sorted(tasks.loc[tasks.split.eq("train"), "perturbation"].astype(str))
    validation = sorted(tasks.loc[tasks.split.eq("val"), "perturbation"].astype(str))
    test = sorted(tasks.loc[tasks.split.eq("test"), "perturbation"].astype(str))
    all_perturbations = train + validation + test
    if len(train) != 52 or len(validation) != 8 or len(test) != 16:
        raise SmokeFailure("Adamson frozen split changed")
    genes, observed_effects, source_status = source_effects(set(train + validation))
    embeddings = load_embeddings(all_perturbations)
    embedding = dict(zip(all_perturbations, embeddings))
    train_x = np.stack([embedding[gene] for gene in train])
    train_y = np.stack([observed_effects[gene] for gene in train])
    val_x = np.stack([embedding[gene] for gene in validation])
    val_y = np.stack([observed_effects[gene] for gene in validation])

    ridge_models, ridge_rows = [], []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        error = rmse(model.predict(val_x), val_y)
        ridge_models.append(model)
        ridge_rows.append({"alpha": alpha, "validation_rmse": error})
    best_index = int(np.argmin([row["validation_rmse"] for row in ridge_rows]))
    ridge = ridge_models[best_index]
    selected_alpha = float(ALPHAS[best_index])

    def knn_prediction(gene: str) -> np.ndarray:
        similarities = train_x @ embedding[gene]
        nearest = np.argsort(-similarities, kind="stable")[:K_NEIGHBORS]
        weights = np.exp((similarities[nearest] - similarities[nearest].max()) / 0.10)
        weights /= weights.sum()
        return np.sum(train_y[nearest] * weights[:, None], axis=0).astype(np.float32)

    query = validation + test
    pred_knn = np.stack([knn_prediction(gene) for gene in query]).astype(np.float32)
    pred_ridge = ridge.predict(np.stack([embedding[gene] for gene in query])).astype(np.float32)
    mean_prediction = (pred_knn + pred_ridge) / 2.0
    magnitude = np.sqrt(np.mean(np.square(mean_prediction), axis=1))
    disagreement = np.sqrt(np.mean(np.square(pred_knn - pred_ridge), axis=1))
    novelty = np.asarray(
        [1.0 - float(np.max(train_x @ embedding[gene])) for gene in query], dtype=float
    )

    table = pd.DataFrame(
        {
            "perturbation": query,
            "split": ["val"] * len(validation) + ["test"] * len(test),
            "predicted_magnitude": magnitude,
            "model_disagreement": disagreement,
            "embedding_novelty": novelty,
        }
    )
    for split, indices in table.groupby("split", sort=False).groups.items():
        ix = np.asarray(list(indices), dtype=int)
        m = percentile(table.loc[ix, "predicted_magnitude"].to_numpy(float))
        d = percentile(table.loc[ix, "model_disagreement"].to_numpy(float))
        n = percentile(table.loc[ix, "embedding_novelty"].to_numpy(float))
        structural = (d + n) / 2.0
        combined = m + 0.50 * np.maximum(structural - m, 0.0) + 0.125 * np.maximum(d - m, 0.0)
        table.loc[ix, "magnitude_percentile"] = m
        table.loc[ix, "disagreement_percentile"] = d
        table.loc[ix, "novelty_percentile"] = n
        table.loc[ix, "safeconf_structural_proxy"] = structural
        table.loc[ix, "e218_transfer_router"] = combined
    table["test_truth_used_for_features"] = False

    run_root.mkdir(parents=True, exist_ok=True)
    arrays = run_root / "PRETRUTH_PREDICTIONS.npz"
    np.savez_compressed(
        arrays,
        perturbations=np.asarray(query, dtype=str),
        genes=genes,
        pred_knn=pred_knn,
        pred_ridge=pred_ridge,
    )
    table_path = run_root / "PRETRUTH_RISK_FEATURES.csv"
    atomic_csv(table_path, table)
    ridge_path = run_root / "RIDGE_VALIDATION.csv"
    atomic_csv(ridge_path, pd.DataFrame(ridge_rows))
    status = {
        "experiment": "E219_Adamson_unseen_perturbation_smoke",
        "stage": "PRETRUTH_SEAL",
        "status": "PASS",
        "generated_at": now(),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "n_output_genes": len(genes),
        "predictors": ["scGPT_embedding_effect_KNN", "scGPT_embedding_ridge"],
        "ridge_alpha_selected_on_validation": selected_alpha,
        "validation_truth_used_for_model_selection": True,
        "test_truth_used": False,
        **source_status,
        "prediction_sha256": sha256(arrays),
        "risk_table_sha256": sha256(table_path),
        "ridge_validation_sha256": sha256(ridge_path),
        "method_scope": (
            "exploratory external adapter smoke; the structural proxy is not claimed "
            "to be identical to the E201 five-component SafeConf score"
        ),
    }
    atomic_json(run_root / "PRETRUTH_STATUS.json", status)
    atomic_json(E219 / "E219_ADAMSON_PRETRUTH_SEAL.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def evaluate(run_root: Path, authorization: Path) -> None:
    status_path = run_root / "PRETRUTH_STATUS.json"
    table_path = run_root / "PRETRUTH_RISK_FEATURES.csv"
    arrays_path = run_root / "PRETRUTH_PREDICTIONS.npz"
    if not all(path.is_file() for path in (status_path, table_path, arrays_path, authorization)):
        raise SmokeFailure("pretruth artifacts or authorization missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    if auth.get("status") != "AUTHORIZED" or auth.get("pretruth_status_sha256") != sha256(status_path):
        raise SmokeFailure("evaluation authorization does not match pretruth seal")
    if status.get("test_truth_used") is not False:
        raise SmokeFailure("pretruth status is not sealed")
    table = pd.read_csv(table_path)
    test_table = table.loc[table.split.eq("test")].copy().reset_index(drop=True)
    with np.load(arrays_path, allow_pickle=False) as store:
        perturbations = store["perturbations"].astype(str).tolist()
        pred_knn = store["pred_knn"]
        pred_ridge = store["pred_ridge"]
    index = {gene: i for i, gene in enumerate(perturbations)}

    data = ad.read_h5ad(SOURCE, backed="r")
    labels = data.obs["perturbation"].astype(str).to_numpy()
    needed = set(test_table.perturbation.astype(str)) | {"control"}
    keep = np.isin(labels, sorted(needed))
    matrix = data.X[np.flatnonzero(keep), :]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    else:
        matrix = matrix.tocsr().astype(np.float32)
    means = group_means(matrix, labels[keep])
    data.file.close()
    control = means["control"]

    rows = []
    for record in test_table.itertuples(index=False):
        gene = str(record.perturbation)
        truth = means[gene] - control
        a, b = pred_knn[index[gene]], pred_ridge[index[gene]]
        rows.append(
            {
                **record._asdict(),
                "error_knn_rmse": rmse(a, truth),
                "error_ridge_rmse": rmse(b, truth),
                "error_two_predictor_mean_rmse": (rmse(a, truth) + rmse(b, truth)) / 2.0,
                "error_no_change_rmse": rmse(np.zeros_like(truth), truth),
            }
        )
    evaluated = pd.DataFrame(rows)
    error = evaluated.error_two_predictor_mean_rmse.to_numpy(float)
    metric_rows = []
    for score in (
        "predicted_magnitude",
        "model_disagreement",
        "embedding_novelty",
        "safeconf_structural_proxy",
        "e218_transfer_router",
    ):
        values = evaluated[score].to_numpy(float)
        association = spearmanr(values, error)
        rho = float(association.statistic)
        n_review = max(1, int(np.ceil(0.20 * len(values))))
        selected = np.argsort(-values, kind="stable")[:n_review]
        utility = float(error[selected].mean() / error.mean() - 1.0)
        metric_rows.append(
            {
                "score": score,
                "n_tasks": len(values),
                "spearman_vs_mean_model_error": rho,
                "spearman_pvalue": float(association.pvalue),
                "top20_error_enrichment": utility,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    model_error = float(evaluated.error_two_predictor_mean_rmse.mean())
    no_change = float(evaluated.error_no_change_rmse.mean())
    competence = model_error < no_change
    bootstrap = paired_bootstrap_summary(evaluated)
    output = E219 / "adamson_smoke_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "E219_ADAMSON_TASK_RESULTS.csv", evaluated)
    atomic_csv(output / "E219_ADAMSON_RISK_METRICS.csv", metrics)
    result = {
        "experiment": "E219_Adamson_unseen_perturbation_smoke",
        "stage": "POSTSEAL_EVALUATION",
        "status": "PASS",
        "evaluated_at": now(),
        "n_test_tasks": len(evaluated),
        "mean_two_predictor_error": model_error,
        "mean_no_change_error": no_change,
        "relative_error_reduction_vs_no_change": float(1.0 - model_error / no_change),
        "upstream_competence_gate": "PASS" if competence else "NOT_SUPPORTED",
        "risk_evaluation_allowed_for_claim": competence,
        "interpretation": (
            "risk metrics may be used as external evidence"
            if competence
            else "upstream predictors did not beat no-change; risk metrics are diagnostic only"
        ),
        **bootstrap,
    }
    atomic_json(output / "E219_ADAMSON_EVALUATION_STATUS.json", result)
    lines = [
        "# E219 Adamson 未见扰动冒烟结果",
        "",
        f"- 测试任务：{len(evaluated)}。",
        f"- 两预测器平均 RMSE：{model_error:.6f}。",
        f"- no-change RMSE：{no_change:.6f}。",
        f"- 相对 no-change 的平均误差降低：{100.0 * (1.0 - model_error / no_change):.1f}%。",
        f"- 上游能力门：{result['upstream_competence_gate']}。",
        f"- 固定组合相对幅度的 ΔSpearman：{bootstrap['router_delta_spearman']:+.3f}，"
        f"配对自举 95% 区间 [{bootstrap['router_delta_spearman_ci95'][0]:+.3f}, "
        f"{bootstrap['router_delta_spearman_ci95'][1]:+.3f}]。",
        "",
        "上游能力门已通过，因此可以检查风险排序；但只有 16 个测试扰动，组合相对幅度的区间跨 0，本结果属于外部冒烟，不单独承担确证结论。",
        "",
        metrics.to_markdown(index=False),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=E219 / "E219_ADAMSON_TRUTH_AUTHORIZATION.json",
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run_root.resolve())
    else:
        evaluate(args.run_root.resolve(), args.authorization.resolve())


if __name__ == "__main__":
    main()
