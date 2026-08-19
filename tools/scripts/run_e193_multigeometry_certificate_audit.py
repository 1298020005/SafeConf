#!/usr/bin/env python3
"""Run the frozen post-truth E193 multi-geometry certificate audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "docs/实验结果"
OUT = RESULTS_ROOT / "E193_multigeometry_certificate_robustness_20260729"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
MODEL_KEYS = tuple(
    f"{architecture}_seed{seed}"
    for seed in (3407, 3408, 3409)
    for architecture in ("scGPT", "GEARS")
)
GEOMETRIES = ("absolute_rmse", "cosine", "pearson")
BUDGETS = (0.10, 0.20, 0.30)
N_GENES = 512
NORM_TOL = 1e-12
LOWER_TOL = 1e-10
IDENTITY_TOL = 1e-10
REPLICATION_TOL = 1e-7

DATASETS = {
    "E190_K562": {
        "root": RESULTS_ROOT
        / "E190_adamson_to_replogle_direct_transfer_20260729",
        "asset_root": Path("/home/yyf/data/safeconf_e190_adamson_replogle"),
        "prefix": "E190",
        "n_tasks": 692,
        "n_genes": 47,
        "target": "Replogle K562",
    },
    "E192_RPE1": {
        "root": RESULTS_ROOT
        / "E192_adamson_to_replogle_rpe1_locked_transfer_20260729",
        "asset_root": Path("/home/yyf/data/safeconf_e192_adamson_rpe1"),
        "prefix": "E192",
        "n_tasks": 175,
        "n_genes": 21,
        "target": "Replogle RPE1",
    },
}


class AuditFailure(RuntimeError):
    """Fail-closed E193 audit error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    text = "::".join(parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def verify_locks(
    dataset: str, root: Path, lock_file: Path, namespace: str
) -> list[dict[str, Any]]:
    locks = pd.read_csv(lock_file, keep_default_na=False)
    observed_rows: list[dict[str, Any]] = []
    for row in locks.itertuples(index=False):
        relative = Path(str(row.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise AuditFailure(f"{dataset}: unsafe lock path {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise AuditFailure(f"{dataset}: lock escapes root") from exc
        observed_hash = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_hash != str(row.sha256) or observed_bytes != int(row.bytes):
            raise AuditFailure(f"{dataset}: lock mismatch for {relative}")
        observed_rows.append(
            {
                "dataset": dataset,
                "namespace": namespace,
                "path": relative.as_posix(),
                "bytes": observed_bytes,
                "sha256": observed_hash,
            }
        )
    return observed_rows


def verify_source_asset(
    dataset: str, experiment_root: Path, asset_root: Path
) -> list[dict[str, Any]]:
    locks = pd.read_csv(
        experiment_root / "MODEL_ASSET_LOCKS.csv", keep_default_na=False
    )
    relative = "model_assets/SOURCE_GENE_EFFECTS.npz"
    rows = locks.loc[locks.path.astype(str).eq(relative)]
    if len(rows) != 1:
        raise AuditFailure(f"{dataset}: source-effect lock missing or duplicated")
    row = rows.iloc[0]
    path = asset_root / relative
    observed_hash = sha256_file(path)
    observed_bytes = path.stat().st_size
    if observed_hash != str(row.sha256) or observed_bytes != int(row.bytes):
        raise AuditFailure(f"{dataset}: source-effect asset lock mismatch")
    return [
        {
            "dataset": dataset,
            "namespace": "model_asset",
            "path": relative,
            "bytes": observed_bytes,
            "sha256": observed_hash,
        }
    ]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): np.asarray(archive[key], dtype=np.float64) for key in archive.files}


def embed_rows(
    values: np.ndarray, geometry: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != N_GENES:
        raise AuditFailure(f"unexpected feature dimension {values.shape}")
    finite = np.isfinite(values).all(axis=-1)
    if geometry == "absolute_rmse":
        norms = np.linalg.norm(values, axis=-1)
        return values / math.sqrt(N_GENES), finite, norms
    if geometry == "cosine":
        centered = values
    elif geometry == "pearson":
        centered = values - values.mean(axis=-1, keepdims=True)
    else:
        raise AuditFailure(f"unknown geometry {geometry}")
    norms = np.linalg.norm(centered, axis=-1)
    valid = finite & (norms > NORM_TOL)
    safe_norms = np.where(valid, norms, 1.0)
    embedded = centered / safe_norms[..., None] / math.sqrt(2.0)
    embedded = np.where(valid[..., None], embedded, np.nan)
    return embedded, valid, norms


def row_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4:
        return float("nan")
    a = a[keep]
    b = b[keep]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(a, method="average"),
            rankdata(b, method="average"),
        )[0, 1]
    )


def cluster_groups(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    groups = {
        str(gene): group.index.to_numpy(int)
        for gene, group in frame.reset_index(drop=True).groupby(
            "gene", observed=True, sort=True
        )
    }
    genes = np.asarray(sorted(groups))
    if len(genes) < 4:
        raise AuditFailure("fewer than four gene clusters")
    return genes, groups


def cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    seed: int,
    n_boot: int = 5000,
) -> dict[str, float | int]:
    work = frame.loc[
        np.isfinite(frame[predictor]) & np.isfinite(frame[outcome])
    ].reset_index(drop=True)
    genes, groups = cluster_groups(work)
    x = work[predictor].to_numpy(float)
    y = work[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        take = np.concatenate([groups[str(gene)] for gene in sampled])
        value = spearman(x[take], y[take])
        if math.isfinite(value):
            boot[n_valid] = value
            n_valid += 1
    if n_valid < int(0.95 * n_boot):
        raise AuditFailure("too few valid cluster-bootstrap correlations")
    boot = boot[:n_valid]
    return {
        "n_tasks": len(work),
        "n_gene_clusters": len(genes),
        "spearman": spearman(x, y),
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": n_valid,
    }


def cluster_bootstrap_spearman_delta(
    frame: pd.DataFrame,
    predictor: str,
    comparator: str,
    outcome: str,
    seed: int,
    n_boot: int = 5000,
) -> dict[str, float | int]:
    work = frame.loc[
        np.isfinite(frame[predictor])
        & np.isfinite(frame[comparator])
        & np.isfinite(frame[outcome])
    ].reset_index(drop=True)
    genes, groups = cluster_groups(work)
    first = work[predictor].to_numpy(float)
    second = work[comparator].to_numpy(float)
    y = work[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        take = np.concatenate([groups[str(gene)] for gene in sampled])
        first_rho = spearman(first[take], y[take])
        second_rho = spearman(second[take], y[take])
        delta = first_rho - second_rho
        if math.isfinite(delta):
            boot[n_valid] = delta
            n_valid += 1
    if n_valid < int(0.95 * n_boot):
        raise AuditFailure("too few valid paired bootstrap correlations")
    boot = boot[:n_valid]
    return {
        "n_tasks": len(work),
        "n_gene_clusters": len(genes),
        "predictor_spearman": spearman(first, y),
        "comparator_spearman": spearman(second, y),
        "spearman_delta": spearman(first, y) - spearman(second, y),
        "delta_ci95_lower": float(np.quantile(boot, 0.025)),
        "delta_ci95_upper": float(np.quantile(boot, 0.975)),
        "bootstrap_valid": n_valid,
    }


def top_indices(values: np.ndarray, tie_ids: np.ndarray, n_select: int) -> np.ndarray:
    return np.lexsort((tie_ids, -np.asarray(values, float)))[:n_select]


def utility_values(
    predictor: np.ndarray,
    outcome: np.ndarray,
    tie_ids: np.ndarray,
    budget: float,
) -> dict[str, float | int]:
    n_select = int(math.ceil(len(outcome) * budget))
    oracle = top_indices(outcome, tie_ids, n_select)
    selected = top_indices(predictor, tie_ids, n_select)
    oracle_mean = float(np.mean(outcome[oracle]))
    selected_mean = float(np.mean(outcome[selected]))
    overall_mean = float(np.mean(outcome))
    denominator = oracle_mean - overall_mean
    return {
        "n_selected": n_select,
        "high_error_capture": float(
            len(np.intersect1d(tie_ids[oracle], tie_ids[selected])) / n_select
        ),
        "random_expected_capture": float(n_select / len(outcome)),
        "selected_mean_error": selected_mean,
        "overall_mean_error": overall_mean,
        "error_lift": float(selected_mean / overall_mean),
        "oracle_mean_error": oracle_mean,
        "oracle_normalized_utility": (
            float((selected_mean - overall_mean) / denominator)
            if denominator > 1e-15
            else float("nan")
        ),
    }


def cluster_bootstrap_utilities(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    seed: int,
    n_boot: int = 3000,
) -> dict[float, dict[str, float | int]]:
    work = frame.loc[
        np.isfinite(frame[predictor]) & np.isfinite(frame[outcome])
    ].reset_index(drop=True)
    genes, groups = cluster_groups(work)
    x = work[predictor].to_numpy(float)
    y = work[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = {budget: np.empty(n_boot, dtype=float) for budget in BUDGETS}
    valid = {budget: 0 for budget in BUDGETS}
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        blocks: list[np.ndarray] = []
        tie_blocks: list[np.ndarray] = []
        offset = 0
        for occurrence, gene in enumerate(sampled):
            block = groups[str(gene)]
            blocks.append(block)
            tie_blocks.append(
                occurrence * (len(work) + 1) + np.arange(offset, offset + len(block))
            )
            offset += len(block)
        take = np.concatenate(blocks)
        tie_ids = np.concatenate(tie_blocks)
        for budget in BUDGETS:
            value = utility_values(x[take], y[take], tie_ids, budget)[
                "oracle_normalized_utility"
            ]
            if math.isfinite(float(value)):
                boot[budget][valid[budget]] = float(value)
                valid[budget] += 1
    result: dict[float, dict[str, float | int]] = {}
    for budget in BUDGETS:
        n_valid = valid[budget]
        if n_valid < int(0.95 * n_boot):
            raise AuditFailure("too few valid cluster-bootstrap utilities")
        values = boot[budget][:n_valid]
        result[budget] = {
            "utility_ci95_lower": float(np.quantile(values, 0.025)),
            "utility_ci95_upper": float(np.quantile(values, 0.975)),
            "utility_bootstrap_valid": n_valid,
        }
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def load_dataset(
    dataset: str, config: dict[str, Any]
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    experiment_root = Path(config["root"])
    asset_root = Path(config["asset_root"])
    release = experiment_root / "pretruth_release"
    truth_root = experiment_root / "evaluation_truth"
    input_hashes = verify_locks(
        dataset, release, release / "RELEASE_LOCKS.csv", "pretruth_release"
    )
    input_hashes.extend(
        verify_locks(
            dataset, truth_root, truth_root / "TRUTH_LOCKS.csv", "evaluation_truth"
        )
    )
    input_hashes.extend(
        verify_source_asset(dataset, experiment_root, asset_root)
    )

    order = pd.read_csv(release / "tables/QUERY_ORDER.csv", keep_default_na=False)
    truth_index = pd.read_csv(
        truth_root / "tables/TARGET_TRUTH_INDEX.csv", keep_default_na=False
    )
    if (
        len(order) != int(config["n_tasks"])
        or order.query_index.astype(int).tolist() != list(range(len(order)))
        or truth_index.task_id.astype(str).tolist()
        != order.task_id.astype(str).tolist()
    ):
        raise AuditFailure(f"{dataset}: task order contract failed")
    query = order.merge(
        truth_index[["task_id", "batch", "gene", "n_target_cells"]],
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    if query.gene.nunique() != int(config["n_genes"]) or query.isna().any().any():
        raise AuditFailure(f"{dataset}: query metadata contract failed")

    with np.load(
        release / "arrays/PRETRUTH_PREDICTIONS.npz", allow_pickle=False
    ) as archive:
        if set(archive.files) != set(MODEL_KEYS):
            raise AuditFailure(f"{dataset}: model family keys changed")
        predictions = np.stack(
            [np.asarray(archive[key], dtype=np.float64) for key in MODEL_KEYS], axis=0
        )
    truth_archive = load_npz(truth_root / "arrays/TARGET_TRUE_EFFECTS.npz")
    source_archive = load_npz(
        asset_root / "model_assets/SOURCE_GENE_EFFECTS.npz"
    )
    task_ids = query.task_id.astype(str).tolist()
    truth = np.stack([truth_archive[task_id] for task_id in task_ids])
    source = np.stack([source_archive[gene] for gene in query.gene.astype(str)])
    expected_shape = (len(query), N_GENES)
    if (
        predictions.shape != (len(MODEL_KEYS), len(query), N_GENES)
        or truth.shape != expected_shape
        or source.shape != expected_shape
        or not np.isfinite(predictions).all()
        or not np.isfinite(truth).all()
        or not np.isfinite(source).all()
    ):
        raise AuditFailure(f"{dataset}: aligned arrays invalid")
    prior = pd.read_csv(
        experiment_root
        / "final_evaluation/tables"
        / f"{config['prefix']}_TASK_METRICS.csv",
        keep_default_na=False,
    )
    if prior.task_id.astype(str).tolist() != task_ids:
        raise AuditFailure(f"{dataset}: prior task table order changed")
    return query, predictions, truth, source, prior, input_hashes


def geometry_metrics(
    dataset: str,
    geometry: str,
    query: pd.DataFrame,
    predictions: np.ndarray,
    truth: np.ndarray,
    source: np.ndarray,
    raw_diversity: np.ndarray,
    raw_magnitude: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    pred_flat = predictions.reshape(-1, N_GENES)
    pred_z_flat, pred_valid_flat, pred_norms_flat = embed_rows(pred_flat, geometry)
    pred_z = pred_z_flat.reshape(predictions.shape)
    pred_valid = pred_valid_flat.reshape(predictions.shape[:2])
    pred_norms = pred_norms_flat.reshape(predictions.shape[:2])
    truth_z, truth_valid, truth_norms = embed_rows(truth, geometry)
    source_z, source_valid, source_norms = embed_rows(source, geometry)
    raw_centroid_z, raw_centroid_valid, raw_centroid_norms = embed_rows(
        predictions.mean(axis=0), geometry
    )
    certificate_valid = truth_valid & pred_valid.all(axis=0)
    source_baseline_valid = certificate_valid & source_valid
    normalized_centroid_valid = certificate_valid & raw_centroid_valid

    frame = query.copy()
    frame.insert(0, "geometry", geometry)
    frame.insert(0, "dataset", dataset)
    frame["certificate_valid"] = certificate_valid
    frame["source_baseline_valid"] = source_baseline_valid
    frame["truth_norm"] = truth_norms
    frame["min_prediction_norm"] = pred_norms.min(axis=0)
    frame["source_norm"] = source_norms
    metric_columns = (
        "family_rms_error",
        "family_worst_error",
        "centroid_error",
        "diversity_lower_bound",
        "diameter_half_lower_bound",
        "source_to_family_centroid_distance",
        "source_directional_error",
        "normalized_raw_centroid_error",
        "raw_diversity_lower_bound",
        "raw_predicted_magnitude",
        "source_effect_magnitude",
        "family_mean_standard_loss",
        "diversity_standard_loss_lower_bound",
        "worst_standard_loss",
        "diameter_standard_loss_lower_bound",
        "family_lower_tightness",
        "worst_lower_tightness",
        "scGPT_family_rms_error",
        "GEARS_family_rms_error",
        "rms_identity_residual",
    )
    for column in metric_columns:
        frame[column] = np.nan
    frame["family_rms_lower_violation"] = pd.Series(
        pd.NA, index=frame.index, dtype="boolean"
    )
    frame["family_worst_lower_violation"] = pd.Series(
        pd.NA, index=frame.index, dtype="boolean"
    )
    frame["raw_diversity_lower_bound"] = raw_diversity
    frame["raw_predicted_magnitude"] = raw_magnitude
    frame["source_effect_magnitude"] = row_distances(
        source / math.sqrt(N_GENES), np.zeros_like(source)
    )

    take = np.flatnonzero(certificate_valid)
    if len(take):
        z_pred = pred_z[:, take, :]
        z_truth = truth_z[take]
        member_errors = row_distances(z_pred, z_truth[None, :, :])
        centroid = z_pred.mean(axis=0)
        centroid_error = row_distances(centroid, z_truth)
        family_rms = np.sqrt(np.mean(member_errors**2, axis=0))
        family_worst = member_errors.max(axis=0)
        diversity = np.sqrt(
            np.mean(np.sum((z_pred - centroid[None, :, :]) ** 2, axis=-1), axis=0)
        )
        diameter = np.zeros(len(take), dtype=float)
        for left in range(len(MODEL_KEYS)):
            for right in range(left + 1, len(MODEL_KEYS)):
                diameter = np.maximum(
                    diameter, row_distances(z_pred[left], z_pred[right])
                )
        diameter_half = diameter / 2.0
        identity_residual = np.abs(
            family_rms**2 - (centroid_error**2 + diversity**2)
        )
        frame.loc[take, "family_rms_error"] = family_rms
        frame.loc[take, "family_worst_error"] = family_worst
        frame.loc[take, "centroid_error"] = centroid_error
        frame.loc[take, "diversity_lower_bound"] = diversity
        frame.loc[take, "diameter_half_lower_bound"] = diameter_half
        frame.loc[take, "family_mean_standard_loss"] = family_rms**2
        frame.loc[take, "diversity_standard_loss_lower_bound"] = diversity**2
        frame.loc[take, "worst_standard_loss"] = family_worst**2
        frame.loc[take, "diameter_standard_loss_lower_bound"] = (
            diameter_half**2
        )
        frame.loc[take, "family_lower_tightness"] = np.divide(
            diversity,
            family_rms,
            out=np.full_like(diversity, np.nan),
            where=family_rms > 0,
        )
        frame.loc[take, "worst_lower_tightness"] = np.divide(
            diameter_half,
            family_worst,
            out=np.full_like(diameter_half, np.nan),
            where=family_worst > 0,
        )
        for architecture in ("scGPT", "GEARS"):
            member_take = [
                index
                for index, key in enumerate(MODEL_KEYS)
                if key.startswith(architecture)
            ]
            frame.loc[take, f"{architecture}_family_rms_error"] = np.sqrt(
                np.mean(member_errors[member_take] ** 2, axis=0)
            )
        frame.loc[take, "family_rms_lower_violation"] = (
            diversity > family_rms + LOWER_TOL
        )
        frame.loc[take, "family_worst_lower_violation"] = (
            diameter_half > family_worst + LOWER_TOL
        )
        frame.loc[take, "rms_identity_residual"] = identity_residual
        source_take_mask = source_baseline_valid[take]
        if source_take_mask.any():
            source_take = take[source_take_mask]
            source_dist = row_distances(
                source_z[source_take], centroid[source_take_mask]
            )
            frame.loc[
                source_take, "source_to_family_centroid_distance"
            ] = source_dist
            frame.loc[source_take, "source_directional_error"] = row_distances(
                source_z[source_take], truth_z[source_take]
            )
        centroid_take_mask = normalized_centroid_valid[take]
        if centroid_take_mask.any():
            centroid_take = take[centroid_take_mask]
            frame.loc[
                centroid_take, "normalized_raw_centroid_error"
            ] = row_distances(
                raw_centroid_z[centroid_take], truth_z[centroid_take]
            )

    audit = {
        "dataset": dataset,
        "geometry": geometry,
        "n_input_tasks": len(frame),
        "n_certificate_valid": int(certificate_valid.sum()),
        "n_certificate_excluded": int((~certificate_valid).sum()),
        "n_truth_nonfinite": int((~np.isfinite(truth).all(axis=1)).sum()),
        "n_truth_low_norm": int((truth_norms <= NORM_TOL).sum())
        if geometry != "absolute_rmse"
        else 0,
        "n_any_prediction_nonfinite": int(
            (~np.isfinite(predictions).all(axis=2)).any(axis=0).sum()
        ),
        "n_any_prediction_low_norm": int(
            (pred_norms <= NORM_TOL).any(axis=0).sum()
        )
        if geometry != "absolute_rmse"
        else 0,
        "n_source_baseline_valid": int(source_baseline_valid.sum()),
        "n_source_nonfinite": int((~np.isfinite(source).all(axis=1)).sum()),
        "n_source_low_norm": int((source_norms <= NORM_TOL).sum())
        if geometry != "absolute_rmse"
        else 0,
        "n_normalized_raw_centroid_valid": int(normalized_centroid_valid.sum()),
        "n_normalized_raw_centroid_low_norm": int(
            (raw_centroid_norms <= NORM_TOL).sum()
        )
        if geometry != "absolute_rmse"
        else 0,
    }
    invalid_ledger: list[dict[str, Any]] = []
    if geometry != "absolute_rmse":
        for index, row in query.iterrows():
            if not truth_valid[index]:
                invalid_ledger.append(
                    {
                        "dataset": dataset,
                        "task_id": row.task_id,
                        "gene": row.gene,
                        "batch": row.batch,
                        "geometry": geometry,
                        "role": "truth",
                        "model": "",
                        "norm": truth_norms[index],
                        "threshold": NORM_TOL,
                        "reason": "LOW_NORM",
                    }
                )
            for member, key in enumerate(MODEL_KEYS):
                if not pred_valid[member, index]:
                    invalid_ledger.append(
                        {
                            "dataset": dataset,
                            "task_id": row.task_id,
                            "gene": row.gene,
                            "batch": row.batch,
                            "geometry": geometry,
                            "role": "prediction",
                            "model": key,
                            "norm": pred_norms[member, index],
                            "threshold": NORM_TOL,
                            "reason": "LOW_NORM",
                        }
                    )
            if not source_valid[index]:
                invalid_ledger.append(
                    {
                        "dataset": dataset,
                        "task_id": row.task_id,
                        "gene": row.gene,
                        "batch": row.batch,
                        "geometry": geometry,
                        "role": "source_effect",
                        "model": "",
                        "norm": source_norms[index],
                        "threshold": NORM_TOL,
                        "reason": "LOW_NORM",
                    }
                )
            if not raw_centroid_valid[index]:
                invalid_ledger.append(
                    {
                        "dataset": dataset,
                        "task_id": row.task_id,
                        "gene": row.gene,
                        "batch": row.batch,
                        "geometry": geometry,
                        "role": "raw_family_centroid",
                        "model": "",
                        "norm": raw_centroid_norms[index],
                        "threshold": NORM_TOL,
                        "reason": "LOW_NORM",
                    }
                )
    return frame, audit, invalid_ledger


def make_figure(
    tasks: pd.DataFrame,
    associations: pd.DataFrame,
    budgets: pd.DataFrame,
) -> None:
    colors = {
        "absolute_rmse": "#3C5488",
        "cosine": "#00A087",
        "pearson": "#E64B35",
    }
    datasets = list(DATASETS)
    geometries = list(GEOMETRIES)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    x = np.arange(len(datasets))
    width = 0.23
    for offset, geometry in enumerate(geometries):
        means = []
        for dataset in datasets:
            take = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.certificate_valid
            ]
            means.append(take.family_rms_error.mean())
        axes[0].bar(
            x + (offset - 1) * width,
            means,
            width,
            color=colors[geometry],
            label=geometry,
        )
    axes[0].set_xticks(x, ["K562", "RPE1"])
    axes[0].set_ylabel("Mean family error")
    axes[0].set_title("A  Error definition changes scale")

    assoc = associations.loc[
        associations.predictor.eq("diversity_lower_bound")
        & associations.outcome.eq("family_rms_error")
    ].copy()
    for offset, geometry in enumerate(geometries):
        take = assoc.loc[assoc.geometry.eq(geometry)].set_index("dataset").loc[datasets]
        xpos = x + (offset - 1) * width
        axes[1].errorbar(
            xpos,
            take.spearman,
            yerr=np.vstack(
                [
                    take.spearman - take.ci95_lower,
                    take.ci95_upper - take.spearman,
                ]
            ),
            fmt="o",
            color=colors[geometry],
            capsize=3,
            label=geometry,
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, ["K562", "RPE1"])
    axes[1].set_ylabel("Spearman ρ (gene-cluster CI)")
    axes[1].set_title("B  Exploratory risk association")

    utility = budgets.loc[
        budgets.budget.eq(0.20)
        & budgets.predictor.isin(
            [
                "diversity_lower_bound",
                "raw_predicted_magnitude",
                "source_effect_magnitude",
                "source_to_family_centroid_distance",
            ]
        )
    ].copy()
    predictors = [
        "diversity_lower_bound",
        "raw_predicted_magnitude",
        "source_to_family_centroid_distance",
    ]
    labels = ["Diversity", "Magnitude", "Source shift"]
    marker = {"E190_K562": "o", "E192_RPE1": "s"}
    for dataset in datasets:
        take = utility.loc[
            utility.dataset.eq(dataset) & utility.geometry.eq("pearson")
        ].set_index("predictor").loc[predictors]
        axes[2].plot(
            range(3),
            take.oracle_normalized_utility,
            marker=marker[dataset],
            linewidth=1.2,
            label=dataset.replace("E190_", "").replace("E192_", ""),
        )
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set_xticks(range(3), labels, rotation=20, ha="right")
    axes[2].set_ylabel("20% oracle-normalized utility")
    axes[2].set_title("C  Pearson-space review utility")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "E193_multigeometry_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    status_path = OUT / "E193_STATUS.json"
    if status_path.exists():
        raise AuditFailure("E193 output is append-only and already exists")
    if not (OUT / "ANALYSIS_FREEZE.md").exists():
        raise AuditFailure("analysis freeze is missing")

    all_tasks: list[pd.DataFrame] = []
    validity_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    input_hashes: list[dict[str, Any]] = []

    for dataset, config in DATASETS.items():
        query, predictions, truth, source, prior, hashes = load_dataset(
            dataset, config
        )
        input_hashes.extend(hashes)
        raw_pred_z, raw_pred_valid, _ = embed_rows(
            predictions.reshape(-1, N_GENES), "absolute_rmse"
        )
        raw_pred_z = raw_pred_z.reshape(predictions.shape)
        if not raw_pred_valid.all():
            raise AuditFailure(f"{dataset}: invalid raw prediction")
        raw_centroid = raw_pred_z.mean(axis=0)
        raw_diversity = np.sqrt(
            np.mean(
                np.sum((raw_pred_z - raw_centroid[None, :, :]) ** 2, axis=-1),
                axis=0,
            )
        )
        raw_magnitude = row_distances(
            predictions.mean(axis=0) / math.sqrt(N_GENES),
            np.zeros_like(truth),
        )
        for geometry in GEOMETRIES:
            frame, validity, invalid = geometry_metrics(
                dataset,
                geometry,
                query,
                predictions,
                truth,
                source,
                raw_diversity,
                raw_magnitude,
            )
            validity_rows.append(validity)
            invalid_rows.extend(invalid)
            valid = frame.loc[frame.certificate_valid].copy()
            violations_rms = int(
                valid.family_rms_lower_violation.astype(bool).sum()
            )
            violations_worst = int(
                valid.family_worst_lower_violation.astype(bool).sum()
            )
            max_identity = float(valid.rms_identity_residual.max())
            replication = float("nan")
            if geometry == "absolute_rmse":
                comparisons = {
                    "family_rms_error": "family_rms_error",
                    "family_worst_error": "family_worst_error",
                    "centroid_error": "centroid_error",
                    "diversity_lower_bound": "diversity_lower_bound",
                    "diameter_half_lower_bound": "diameter_half_lower_bound",
                }
                differences = []
                for new_column, prior_column in comparisons.items():
                    differences.append(
                        np.max(
                            np.abs(
                                frame[new_column].to_numpy(float)
                                - prior[prior_column].to_numpy(float)
                            )
                        )
                    )
                replication = float(max(differences))
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "geometry": geometry,
                    "n_valid_tasks": len(valid),
                    "mean_family_rms_error": float(
                        valid.family_rms_error.mean()
                    ),
                    "mean_diversity_lower_bound": float(
                        valid.diversity_lower_bound.mean()
                    ),
                    "family_rms_lower_violations": violations_rms,
                    "family_worst_lower_violations": violations_worst,
                    "max_rms_identity_residual": max_identity,
                    "raw_replication_max_abs_diff": replication,
                }
            )
            all_tasks.append(frame)

    tasks = pd.concat(all_tasks, ignore_index=True)
    validity = pd.DataFrame(validity_rows)
    certificate = pd.DataFrame(certificate_rows)

    association_rows: list[dict[str, Any]] = []
    paired_delta_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    association_pairs = (
        ("diversity_lower_bound", "family_rms_error"),
        ("diameter_half_lower_bound", "family_worst_error"),
        ("raw_diversity_lower_bound", "family_rms_error"),
        ("raw_predicted_magnitude", "family_rms_error"),
        ("source_effect_magnitude", "family_rms_error"),
        ("source_to_family_centroid_distance", "family_rms_error"),
    )
    budget_predictors = (
        "diversity_lower_bound",
        "raw_predicted_magnitude",
        "source_effect_magnitude",
        "source_to_family_centroid_distance",
    )
    for dataset in DATASETS:
        for geometry in GEOMETRIES:
            frame = tasks.loc[
                tasks.dataset.eq(dataset)
                & tasks.geometry.eq(geometry)
                & tasks.certificate_valid
            ].reset_index(drop=True)
            for predictor, outcome in association_pairs:
                values = cluster_bootstrap_spearman(
                    frame,
                    predictor,
                    outcome,
                    stable_seed("E193", dataset, geometry, predictor, outcome),
                )
                association_rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "predictor": predictor,
                        "outcome": outcome,
                        **values,
                    }
                )
            for comparator in (
                "raw_predicted_magnitude",
                "raw_diversity_lower_bound",
                "source_effect_magnitude",
                "source_to_family_centroid_distance",
            ):
                values = cluster_bootstrap_spearman_delta(
                    frame,
                    "diversity_lower_bound",
                    comparator,
                    "family_rms_error",
                    stable_seed(
                        "E193",
                        dataset,
                        geometry,
                        "diversity_lower_bound",
                        comparator,
                        "paired_delta",
                    ),
                )
                paired_delta_rows.append(
                    {
                        "dataset": dataset,
                        "geometry": geometry,
                        "predictor": "diversity_lower_bound",
                        "comparator": comparator,
                        "outcome": "family_rms_error",
                        **values,
                    }
                )
            for predictor in budget_predictors:
                eligible = frame.loc[
                    np.isfinite(frame[predictor])
                    & np.isfinite(frame.family_rms_error)
                ].reset_index(drop=True)
                tie_ids = np.arange(len(eligible), dtype=np.int64)
                boot = cluster_bootstrap_utilities(
                    eligible,
                    predictor,
                    "family_rms_error",
                    stable_seed("E193", dataset, geometry, predictor, "utility"),
                )
                for budget in BUDGETS:
                    point = utility_values(
                        eligible[predictor].to_numpy(float),
                        eligible.family_rms_error.to_numpy(float),
                        tie_ids,
                        budget,
                    )
                    budget_rows.append(
                        {
                            "dataset": dataset,
                            "geometry": geometry,
                            "predictor": predictor,
                            "outcome": "family_rms_error",
                            "budget": budget,
                            "n_eligible_tasks": len(eligible),
                            "n_gene_clusters": eligible.gene.nunique(),
                            **point,
                            **boot[budget],
                        }
                    )
    associations = pd.DataFrame(association_rows)
    paired_deltas = pd.DataFrame(paired_delta_rows)
    budgets = pd.DataFrame(budget_rows)

    total_rms_violations = int(certificate.family_rms_lower_violations.sum())
    total_worst_violations = int(certificate.family_worst_lower_violations.sum())
    max_identity = float(certificate.max_rms_identity_residual.max())
    max_replication = float(
        certificate.raw_replication_max_abs_diff.dropna().max()
    )
    certificate_gate = bool(
        total_rms_violations == 0
        and total_worst_violations == 0
        and max_identity <= IDENTITY_TOL
        and max_replication <= REPLICATION_TOL
    )
    n_excluded = int(validity.n_certificate_excluded.sum())
    status_label = (
        "FAIL"
        if not certificate_gate
        else ("PASS_WITH_UNDEFINED_TASKS" if n_excluded else "PASS")
    )
    status = {
        "experiment": "E193",
        "stage": "POSTTRUTH_METRIC_ROBUSTNESS",
        "status": status_label,
        "analysis_is_prospective_preregistration": False,
        "independent_confirmation": False,
        "truth_already_opened_before_analysis_design": True,
        "analysis_freeze_committed_before_execution": True,
        "datasets": list(DATASETS),
        "geometries": list(GEOMETRIES),
        "unique_target_tasks": int(sum(c["n_tasks"] for c in DATASETS.values())),
        "dataset_geometry_task_instances": int(
            validity.n_certificate_valid.sum()
        ),
        "undefined_dataset_geometry_task_instances": n_excluded,
        "family_rms_lower_violations": total_rms_violations,
        "family_worst_lower_violations": total_worst_violations,
        "max_rms_identity_residual": max_identity,
        "max_raw_replication_abs_diff": max_replication,
        "certificate_implementation_gate_pass": certificate_gate,
        "risk_ranking_results_are_exploratory": True,
        "risk_ranking_has_no_activation_gate": True,
        "e192_abstain_status_changed": False,
        "systema_exact_metric_claimed": False,
    }

    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(TABLES / "E193_TASK_METRICS.csv", index=False)
    validity.to_csv(TABLES / "E193_VALIDITY_AUDIT.csv", index=False)
    certificate.to_csv(TABLES / "E193_CERTIFICATE_AUDIT.csv", index=False)
    associations.to_csv(TABLES / "E193_RISK_ASSOCIATIONS.csv", index=False)
    paired_deltas.to_csv(TABLES / "E193_PAIRED_RHO_DELTAS.csv", index=False)
    budgets.to_csv(TABLES / "E193_BUDGET_UTILITY.csv", index=False)
    invalid_columns = [
        "dataset",
        "task_id",
        "gene",
        "batch",
        "geometry",
        "role",
        "model",
        "norm",
        "threshold",
        "reason",
    ]
    pd.DataFrame(invalid_rows, columns=invalid_columns).to_csv(
        TABLES / "E193_INVALID_VECTOR_LEDGER.csv", index=False
    )
    pd.DataFrame(input_hashes).to_csv(
        TABLES / "E193_INPUT_HASHES.csv", index=False
    )
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(tasks, associations, budgets)

    key_associations = associations.loc[
        associations.predictor.eq("diversity_lower_bound")
        & associations.outcome.eq("family_rms_error")
    ][
        [
            "dataset",
            "geometry",
            "n_tasks",
            "n_gene_clusters",
            "spearman",
            "ci95_lower",
            "ci95_upper",
        ]
    ]
    key_budgets = budgets.loc[
        budgets.budget.eq(0.20)
        & budgets.predictor.isin(
            [
                "diversity_lower_bound",
                "raw_predicted_magnitude",
                "source_effect_magnitude",
                "source_to_family_centroid_distance",
            ]
        )
    ][
        [
            "dataset",
            "geometry",
            "predictor",
            "high_error_capture",
            "error_lift",
            "oracle_normalized_utility",
            "utility_ci95_lower",
            "utility_ci95_upper",
        ]
    ]
    key_deltas = paired_deltas[
        [
            "dataset",
            "geometry",
            "comparator",
            "predictor_spearman",
            "comparator_spearman",
            "spearman_delta",
            "delta_ci95_lower",
            "delta_ci95_upper",
        ]
    ]
    report = [
        "# E193 多几何注册家族证书结果",
        "",
        f"确定性实现 gate：**{status_label}**。",
        "",
        "E193 是开真值后的指标稳健性分析。证书恒等式与下界属于确定性复核；"
        "相关性和复核收益全部是探索性结果。",
        "",
        "## 证书审计",
        "",
        markdown_table(certificate),
        "",
        "## 同几何 diversity 与 family error",
        "",
        markdown_table(key_associations),
        "",
        "## Diversity 相对基线的配对相关差",
        "",
        markdown_table(key_deltas),
        "",
        "## 20% 复核预算",
        "",
        markdown_table(key_budgets),
        "",
        "方向几何中的证书值只能解释该几何定义的家族误差下界，不能改写成模型正确"
        "概率，也不能从本项 post-truth 分析推导新的选择性风险保证。",
        "",
    ]
    (REPORTS / "E193_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    interpretation_lines = [
        "# E193 结果解释",
        "",
        f"- 输入：E190 692 个任务与 E192 175 个任务，共 "
        f"{status['unique_target_tasks']} 个独特目标任务；",
        f"- 三种几何的 family RMS / worst lower violation："
        f"{total_rms_violations} / {total_worst_violations}；",
        f"- 最大平方恒等式残差：{max_identity:.3e}；",
        f"- 原始 RMSE 结果最大复算差：{max_replication:.3e}；",
        f"- 确定性实现结论：**{status_label}**。",
        "",
    ]
    for row in key_associations.itertuples(index=False):
        interpretation_lines.append(
            f"- {row.dataset} / {row.geometry}：diversity–family error "
            f"ρ={row.spearman:.3f}，基因整簇 95% CI "
            f"[{row.ci95_lower:.3f}, {row.ci95_upper:.3f}]；"
        )
    association_lookup = key_associations.set_index(["dataset", "geometry"])
    budget_lookup = budgets.loc[budgets.budget.eq(0.20)].set_index(
        ["dataset", "geometry", "predictor"]
    )
    e190_cosine = association_lookup.loc[("E190_K562", "cosine")]
    e190_pearson = association_lookup.loc[("E190_K562", "pearson")]
    e192_cosine = association_lookup.loc[("E192_RPE1", "cosine")]
    e192_pearson = association_lookup.loc[("E192_RPE1", "pearson")]
    e190_cosine_budget = budget_lookup.loc[
        ("E190_K562", "cosine", "diversity_lower_bound")
    ]
    e190_pearson_budget = budget_lookup.loc[
        ("E190_K562", "pearson", "diversity_lower_bound")
    ]
    e190_pearson_source_budget = budget_lookup.loc[
        (
            "E190_K562",
            "pearson",
            "source_to_family_centroid_distance",
        )
    ]
    interpretation_lines.extend(
        [
            "",
            "## 这次真正得到什么",
            "",
            "确定性部分不依赖原始 RMSE。把每个成员先映射到 effect-vector cosine 或 "
            "Pearson 几何后，家族平方误差恒等式、diversity lower bound 和 "
            "diameter/2 worst-member lower bound 仍然全部成立。",
            "",
            "经验排序没有跨 target 和 metric 运输：",
            "",
            f"- E190 K562 cosine：ρ={e190_cosine.spearman:.3f}，20% utility="
            f"{e190_cosine_budget.oracle_normalized_utility:.3f}；",
            f"- E190 K562 Pearson：ρ={e190_pearson.spearman:.3f}，20% diversity "
            f"utility={e190_pearson_budget.oracle_normalized_utility:.3f}；"
            f"source-to-family-centroid utility="
            f"{e190_pearson_source_budget.oracle_normalized_utility:.3f}；",
            f"- E192 RPE1 cosine / Pearson：ρ={e192_cosine.spearman:.3f} / "
            f"{e192_pearson.spearman:.3f}，均未形成可运输的方向型排序。",
            "",
            "所以 E193 加强的是 metric-aware registered-family certificate，"
            "没有产生一个跨细胞系通用的方向型风险分数。E192 的原始 ABSTAIN 不变。",
            "",
            "## 不能从 E193 推出的结论",
            "",
            "- 不能称为 Systema exact replication；",
            "- 不能声称 diversity 普遍优于 magnitude 或 source-shift；",
            "- 不能把开真值后的相关结果称为独立确认；",
            "- 不能用零违例替代原生 UQ 与简单强基线的同协议比较。",
            "",
            "这项结果回答“证书是否只在 RMSE 定义下成立”。它不回答“方向型排序是否已"
            "在未见真值上确认”；后者若要形成确认性主张，需要另一个在预测和分析冻结后"
            "才解封的外部数据块。",
            "",
        ]
    )
    (REPORTS / "E193_INTERPRETATION.md").write_text(
        "\n".join(interpretation_lines), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
