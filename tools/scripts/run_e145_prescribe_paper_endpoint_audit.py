#!/usr/bin/env python3
"""E145: correct E96 to the paper-exact PRESCRIBE calibration endpoint.

This is a post-unblinding metric correction.  It reuses frozen E95 predictions,
does not retrain a model, and must not be described as independent confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
E95 = ROOT / "docs/实验结果/E95_prescribe_norman_native_20260712"
OUT = ROOT / "docs/实验结果/E145_prescribe_paper_endpoint_20260714"
PRESCRIBE = Path("/home/yyf/archive/external/PRESCRIBE")
PAPER_URL = (
    "https://papers.nips.cc/paper_files/paper/2025/file/"
    "d6383e7643415842b48a5077a1b09c98-Paper-Conference.pdf"
)
N_BOOT = 10_000
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)
FOCAL_COVERAGES = (0.95, 0.90)

PANELS = {
    "Norman_P1": {
        "run": E95 / "norman_p1_formal_seed3407",
        "processed": PRESCRIBE / "data/norman_p1/perturb_processed.h5ad",
        "raw_sha256": "98c0c57e755dc18f5a325bad657e1a850f8868e13fea956de5b76849acbb0831",
        "table_sha256": "c17af2fee00f7694e435de8ecce9aa81cb0f5a5d44edbcaee72dff0e49b06d05",
    },
    "Norman_P2": {
        "run": E95 / "norman_p2_formal_seed3407",
        "processed": PRESCRIBE / "data/norman_p2/perturb_processed.h5ad",
        "raw_sha256": "5ab3bdf59f82bd637d16f401f805a3a5cb981b50aa40f87b6cfea3738c59b4f3",
        "table_sha256": "a11fcd538647f5ed7e7a0afdebc0ee21daeab201aaa548f93efd7dc41b6c0fcf",
    },
}

UPSTREAM_FORMULA_FILES = {
    PRESCRIBE / "Step3_test.py": "b3c2f0e243824e4c8f27352b401c0709193361b5697287812bfda67dc96f989d",
    PRESCRIBE / "src/model/lightening_module.py": "5add09fb0ca49dd43f9caac892140c3d8a31add3bdd492aef2ea4668b0acb3cd",
}

SCORES = {
    "epistemic_confidence": "epistemic_confidence",
    "aleatoric_confidence": "aleatoric_confidence",
    "combined_confidence": "combined_confidence",
    "predicted_magnitude": "predicted_magnitude_rms",
}

TARGETS = {
    "pearson_effect_accuracy": {
        "column": "pearson_effect_accuracy",
        "kind": "accuracy",
        "favorable_sign": 1.0,
    },
    "cosine_effect_accuracy": {
        "column": "cosine_effect_accuracy",
        "kind": "accuracy",
        "favorable_sign": 1.0,
    },
    "rmse_effect_error": {
        "column": "rmse_effect_error",
        "kind": "error",
        "favorable_sign": -1.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[mask], np.asarray(y, float)[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    value = np.corrcoef(rx, ry)[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def vector_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = np.corrcoef(x, y)[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def vector_cosine(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return float("nan")
    value = float(np.dot(x, y) / denom)
    return value if math.isfinite(value) else float("nan")


def percentile_interval(values: list[float]) -> tuple[float, float, int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], float)
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def association_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x), len(x))
        values.append(safe_spearman(x[idx], y[idx]))
    return percentile_interval(values)


def macro_association_bootstrap(
    groups: list[pd.DataFrame],
    score: str,
    target: str,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    arrays = [
        (group[score].to_numpy(float), group[target].to_numpy(float))
        for group in groups
    ]
    for _ in range(N_BOOT):
        panel_values = []
        for x, y in arrays:
            idx = rng.integers(0, len(x), len(x))
            panel_values.append(safe_spearman(x[idx], y[idx]))
        if all(math.isfinite(value) for value in panel_values):
            values.append(float(np.mean(panel_values)))
    return percentile_interval(values)


def delta_bootstrap(
    group: pd.DataFrame,
    score: str,
    baseline: str,
    target: str,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    x = group[score].to_numpy(float)
    b = group[baseline].to_numpy(float)
    y = group[target].to_numpy(float)
    values: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(group), len(group))
        value = safe_spearman(x[idx], y[idx]) - safe_spearman(b[idx], y[idx])
        values.append(value)
    return percentile_interval(values)


def macro_delta_bootstrap(
    groups: list[pd.DataFrame],
    score: str,
    baseline: str,
    target: str,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    arrays = [
        (
            group[score].to_numpy(float),
            group[baseline].to_numpy(float),
            group[target].to_numpy(float),
        )
        for group in groups
    ]
    values: list[float] = []
    for _ in range(N_BOOT):
        panel_values = []
        for x, b, y in arrays:
            idx = rng.integers(0, len(x), len(x))
            value = safe_spearman(x[idx], y[idx]) - safe_spearman(b[idx], y[idx])
            panel_values.append(value)
        if all(math.isfinite(value) for value in panel_values):
            values.append(float(np.mean(panel_values)))
    return percentile_interval(values)


def retained_mean(score: np.ndarray, target: np.ndarray, coverage: float) -> tuple[float, int]:
    n_keep = len(score) if coverage >= 1 else max(2, int(np.floor(len(score) * coverage)))
    order = np.argsort(-score, kind="stable")
    return float(np.mean(target[order[:n_keep]])), n_keep


def coverage_delta_bootstrap(
    group: pd.DataFrame,
    score: str,
    baseline: str,
    target: str,
    coverage: float,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    x = group[score].to_numpy(float)
    b = group[baseline].to_numpy(float)
    y = group[target].to_numpy(float)
    values: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(group), len(group))
        x_mean, _ = retained_mean(x[idx], y[idx], coverage)
        b_mean, _ = retained_mean(b[idx], y[idx], coverage)
        values.append(x_mean - b_mean)
    return percentile_interval(values)


def macro_coverage_delta_bootstrap(
    groups: list[pd.DataFrame],
    score: str,
    baseline: str,
    target: str,
    coverage: float,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    arrays = [
        (
            group[score].to_numpy(float),
            group[baseline].to_numpy(float),
            group[target].to_numpy(float),
        )
        for group in groups
    ]
    values: list[float] = []
    for _ in range(N_BOOT):
        panel_values = []
        for x, b, y in arrays:
            idx = rng.integers(0, len(x), len(x))
            x_mean, _ = retained_mean(x[idx], y[idx], coverage)
            b_mean, _ = retained_mean(b[idx], y[idx], coverage)
            panel_values.append(x_mean - b_mean)
        values.append(float(np.mean(panel_values)))
    return percentile_interval(values)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate_frozen_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    for panel, config in PANELS.items():
        raw = config["run"] / "test_predictions_raw.npz"
        table = config["run"] / "task_prediction_records.csv"
        for path in [raw, table, config["processed"]]:
            if not path.exists():
                raise FileNotFoundError(path)
        raw_hash, table_hash = sha256(raw), sha256(table)
        if raw_hash != config["raw_sha256"]:
            raise RuntimeError(f"{panel}: frozen raw SHA256 changed: {raw_hash}")
        if table_hash != config["table_sha256"]:
            raise RuntimeError(f"{panel}: frozen task table SHA256 changed: {table_hash}")
        observed[f"{panel}_raw"] = raw_hash
        observed[f"{panel}_table"] = table_hash
    for path, expected in UPSTREAM_FORMULA_FILES.items():
        observed_hash = sha256(path)
        if observed_hash != expected:
            raise RuntimeError(f"Frozen upstream formula source changed: {path}: {observed_hash}")
        observed[str(path)] = observed_hash
    return observed


def load_panel(panel: str, config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    run = Path(config["run"])
    prior = pd.read_csv(run / "task_prediction_records.csv").sort_values("task_id")
    if len(prior) != 24 or prior["task_id"].nunique() != 24:
        raise RuntimeError(f"{panel}: expected 24 unique tasks, observed {len(prior)}")

    raw = np.load(run / "test_predictions_raw.npz", allow_pickle=False)
    required = {"pert_cat", "pred", "truth", "epistemic_conf", "aleatoric_conf"}
    missing = required - set(raw.files)
    if missing:
        raise RuntimeError(f"{panel}: raw arrays missing {sorted(missing)}")
    labels = raw["pert_cat"].astype(str)
    pred, truth = raw["pred"], raw["truth"]
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[0] != len(labels):
        raise RuntimeError(f"{panel}: incompatible prediction/truth shapes")
    if set(labels) != set(prior["task_id"].astype(str)):
        raise RuntimeError(f"{panel}: raw labels do not match frozen task table")

    adata = sc.read_h5ad(Path(config["processed"]), backed="r")
    try:
        if pred.shape[1] != adata.n_vars:
            raise RuntimeError(
                f"{panel}: prediction genes {pred.shape[1]} != processed genes {adata.n_vars}"
            )
        control_mask = np.asarray(adata.obs["condition"].astype(str) == "ctrl")
        if not control_mask.any():
            raise RuntimeError(f"{panel}: no control cells")
        control_x = adata[control_mask].to_memory().X
        control = np.asarray(control_x.mean(axis=0)).reshape(-1).astype(float)
        gene_hash = "sha256:" + hashlib.sha256(
            "\n".join(map(str, adata.var_names)).encode()
        ).hexdigest()
    finally:
        adata.file.close()

    rows: list[dict[str, object]] = []
    for task in sorted(set(labels)):
        mask = labels == task
        pred_mean = np.asarray(pred[mask].mean(axis=0), float)
        truth_mean = np.asarray(truth[mask].mean(axis=0), float)
        pred_effect = pred_mean - control
        truth_effect = truth_mean - control
        epistemic = float(np.asarray(raw["epistemic_conf"])[mask].mean())
        aleatoric = float(np.asarray(raw["aleatoric_conf"])[mask].mean())
        rows.append(
            {
                "panel": panel,
                "task_id": task,
                "n_cells": int(mask.sum()),
                "n_genes": int(len(pred_effect)),
                "gene_order_hash": gene_hash,
                "pearson_effect_accuracy": vector_pearson(pred_effect, truth_effect),
                "cosine_effect_accuracy": vector_cosine(pred_effect, truth_effect),
                "rmse_effect_error": float(np.sqrt(np.mean((pred_effect - truth_effect) ** 2))),
                "predicted_magnitude_rms": float(np.sqrt(np.mean(pred_effect**2))),
                "true_magnitude_rms_diagnostic_only": float(np.sqrt(np.mean(truth_effect**2))),
                "epistemic_confidence": epistemic,
                "aleatoric_confidence": aleatoric,
                "combined_confidence": 2 * epistemic + aleatoric,
            }
        )
    table = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)

    check = table.merge(prior, on="task_id", suffixes=("_new", "_e95"), validate="one_to_one")
    comparisons = {
        "predicted_magnitude_rms": "magnitude_pred_rms",
        "epistemic_confidence": "epistemic_confidence_e95",
        "aleatoric_confidence": "aleatoric_confidence_e95",
        "combined_confidence": "combined_confidence_official",
        "rmse_effect_error": "rmse_mean_profile",
    }
    for new_column, old_column in comparisons.items():
        candidate = new_column
        if candidate not in check and f"{new_column}_new" in check:
            candidate = f"{new_column}_new"
        if old_column not in check and f"{old_column}_e95" in check:
            old_column = f"{old_column}_e95"
        if not np.allclose(
            check[candidate].to_numpy(float),
            check[old_column].to_numpy(float),
            rtol=2e-5,
            atol=2e-7,
        ):
            max_diff = float(
                np.max(np.abs(check[candidate].to_numpy(float) - check[old_column].to_numpy(float)))
            )
            raise RuntimeError(f"{panel}: E95 consistency failed for {new_column}, max diff {max_diff}")

    metadata = {
        "n_cells": int(len(labels)),
        "n_tasks": int(len(table)),
        "n_genes": int(pred.shape[1]),
        "n_control_cells": int(control_mask.sum()),
        "gene_order_hash": gene_hash,
        "control_vector_sha256": hashlib.sha256(control.tobytes()).hexdigest(),
        "n_nonfinite_primary": int((~np.isfinite(table["pearson_effect_accuracy"])).sum()),
    }
    return table, metadata


def build_associations(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [group.reset_index(drop=True) for _, group in tasks.groupby("panel", sort=True)]
    for panel, group in tasks.groupby("panel", sort=True):
        for score_name, score in SCORES.items():
            for target_name, target_config in TARGETS.items():
                target = str(target_config["column"])
                x, y = group[score].to_numpy(float), group[target].to_numpy(float)
                rho = safe_spearman(x, y)
                low, high, valid = association_bootstrap(
                    x, y, fixed_seed(f"E145|association|{panel}|{score_name}|{target_name}")
                )
                point = spearmanr(x, y, nan_policy="omit")
                rows.append(
                    {
                        "scope": panel,
                        "score": score_name,
                        "score_class": "magnitude_baseline" if score_name == "predicted_magnitude" else "prescribe_confidence",
                        "target": target_name,
                        "target_kind": target_config["kind"],
                        "expected_favorable_rho_sign": int(target_config["favorable_sign"]),
                        "n_tasks": int(len(group)),
                        "spearman_rho": rho,
                        "spearman_pvalue_descriptive": float(point.pvalue),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "bootstrap_valid": valid,
                    }
                )
    for score_name, score in SCORES.items():
        for target_name, target_config in TARGETS.items():
            target = str(target_config["column"])
            panel_rhos = [
                safe_spearman(group[score].to_numpy(float), group[target].to_numpy(float))
                for group in groups
            ]
            low, high, valid = macro_association_bootstrap(
                groups,
                score,
                target,
                fixed_seed(f"E145|association|two_panel_macro|{score_name}|{target_name}"),
            )
            rows.append(
                {
                    "scope": "two_panel_macro",
                    "score": score_name,
                    "score_class": "magnitude_baseline" if score_name == "predicted_magnitude" else "prescribe_confidence",
                    "target": target_name,
                    "target_kind": target_config["kind"],
                    "expected_favorable_rho_sign": int(target_config["favorable_sign"]),
                    "n_tasks": int(len(tasks)),
                    "spearman_rho": float(np.mean(panel_rhos)),
                    "spearman_pvalue_descriptive": float("nan"),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_valid": valid,
                }
            )
    return pd.DataFrame(rows)


def build_incremental_deltas(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [group.reset_index(drop=True) for _, group in tasks.groupby("panel", sort=True)]
    confidence_scores = [score for score in SCORES if score != "predicted_magnitude"]
    baseline = SCORES["predicted_magnitude"]
    for score_name in confidence_scores:
        score = SCORES[score_name]
        for target_name, target_config in TARGETS.items():
            target = str(target_config["column"])
            sign = float(target_config["favorable_sign"])
            panel_deltas = []
            for panel, group in tasks.groupby("panel", sort=True):
                observed = safe_spearman(
                    group[score].to_numpy(float), group[target].to_numpy(float)
                ) - safe_spearman(
                    group[baseline].to_numpy(float), group[target].to_numpy(float)
                )
                low, high, valid = delta_bootstrap(
                    group,
                    score,
                    baseline,
                    target,
                    fixed_seed(f"E145|delta|{panel}|{score_name}|{target_name}"),
                )
                panel_deltas.append(observed)
                rows.append(
                    {
                        "scope": panel,
                        "score": score_name,
                        "baseline": "predicted_magnitude",
                        "target": target_name,
                        "raw_delta_rho": observed,
                        "favorable_delta_rho": sign * observed,
                        "raw_bootstrap_ci95_low": low,
                        "raw_bootstrap_ci95_high": high,
                        "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
                        "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
                        "bootstrap_valid": valid,
                    }
                )
            low, high, valid = macro_delta_bootstrap(
                groups,
                score,
                baseline,
                target,
                fixed_seed(f"E145|delta|two_panel_macro|{score_name}|{target_name}"),
            )
            observed = float(np.mean(panel_deltas))
            rows.append(
                {
                    "scope": "two_panel_macro",
                    "score": score_name,
                    "baseline": "predicted_magnitude",
                    "target": target_name,
                    "raw_delta_rho": observed,
                    "favorable_delta_rho": sign * observed,
                    "raw_bootstrap_ci95_low": low,
                    "raw_bootstrap_ci95_high": high,
                    "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
                    "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
                    "bootstrap_valid": valid,
                }
            )
    return pd.DataFrame(rows)


def build_score_redundancy(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [group.reset_index(drop=True) for _, group in tasks.groupby("panel", sort=True)]
    baseline = SCORES["predicted_magnitude"]
    for score_name in [score for score in SCORES if score != "predicted_magnitude"]:
        score = SCORES[score_name]
        panel_rhos = []
        for panel, group in tasks.groupby("panel", sort=True):
            x, y = group[score].to_numpy(float), group[baseline].to_numpy(float)
            rho = safe_spearman(x, y)
            low, high, valid = association_bootstrap(
                x, y, fixed_seed(f"E145|redundancy|{panel}|{score_name}")
            )
            panel_rhos.append(rho)
            rows.append(
                {
                    "scope": panel,
                    "score": score_name,
                    "comparison": "predicted_magnitude",
                    "n_tasks": int(len(group)),
                    "spearman_rho": rho,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_valid": valid,
                }
            )
        low, high, valid = macro_association_bootstrap(
            groups,
            score,
            baseline,
            fixed_seed(f"E145|redundancy|two_panel_macro|{score_name}"),
        )
        rows.append(
            {
                "scope": "two_panel_macro",
                "score": score_name,
                "comparison": "predicted_magnitude",
                "n_tasks": int(len(tasks)),
                "spearman_rho": float(np.mean(panel_rhos)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_valid": valid,
            }
        )
    return pd.DataFrame(rows)


def build_coverage(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    groups = [group.reset_index(drop=True) for _, group in tasks.groupby("panel", sort=True)]
    for panel, group in tasks.groupby("panel", sort=True):
        for score_name, score in SCORES.items():
            score_values = group[score].to_numpy(float)
            for target_name, target_config in TARGETS.items():
                target = str(target_config["column"])
                target_values = group[target].to_numpy(float)
                full_mean = float(np.mean(target_values))
                for coverage in COVERAGES:
                    mean_value, n_keep = retained_mean(score_values, target_values, float(coverage))
                    raw_change = mean_value - full_mean
                    curve_rows.append(
                        {
                            "panel": panel,
                            "score": score_name,
                            "target": target_name,
                            "target_kind": target_config["kind"],
                            "coverage": float(coverage),
                            "n_retained": int(n_keep),
                            "n_rejected": int(len(group) - n_keep),
                            "retained_mean": mean_value,
                            "full_set_mean": full_mean,
                            "raw_change_vs_full": raw_change,
                            "favorable_change_vs_full": float(target_config["favorable_sign"]) * raw_change,
                        }
                    )

    baseline = SCORES["predicted_magnitude"]
    for score_name in [score for score in SCORES if score != "predicted_magnitude"]:
        score = SCORES[score_name]
        for target_name, target_config in TARGETS.items():
            target = str(target_config["column"])
            sign = float(target_config["favorable_sign"])
            for coverage in FOCAL_COVERAGES:
                panel_observed = []
                for panel, group in tasks.groupby("panel", sort=True):
                    score_mean, n_keep = retained_mean(
                        group[score].to_numpy(float), group[target].to_numpy(float), coverage
                    )
                    baseline_mean, _ = retained_mean(
                        group[baseline].to_numpy(float), group[target].to_numpy(float), coverage
                    )
                    observed = score_mean - baseline_mean
                    panel_observed.append(observed)
                    low, high, valid = coverage_delta_bootstrap(
                        group,
                        score,
                        baseline,
                        target,
                        coverage,
                        fixed_seed(
                            f"E145|coverage_delta|{panel}|{score_name}|{target_name}|{coverage}"
                        ),
                    )
                    delta_rows.append(
                        {
                            "scope": panel,
                            "score": score_name,
                            "baseline": "predicted_magnitude",
                            "target": target_name,
                            "coverage": coverage,
                            "n_retained": n_keep,
                            "raw_delta_retained_mean": observed,
                            "favorable_delta_retained_mean": sign * observed,
                            "raw_bootstrap_ci95_low": low,
                            "raw_bootstrap_ci95_high": high,
                            "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
                            "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
                            "bootstrap_valid": valid,
                        }
                    )
                low, high, valid = macro_coverage_delta_bootstrap(
                    groups,
                    score,
                    baseline,
                    target,
                    coverage,
                    fixed_seed(
                        f"E145|coverage_delta|two_panel_macro|{score_name}|{target_name}|{coverage}"
                    ),
                )
                observed = float(np.mean(panel_observed))
                delta_rows.append(
                    {
                        "scope": "two_panel_macro",
                        "score": score_name,
                        "baseline": "predicted_magnitude",
                        "target": target_name,
                        "coverage": coverage,
                        "n_retained": int(sum(max(2, int(np.floor(len(group) * coverage))) for group in groups)),
                        "raw_delta_retained_mean": observed,
                        "favorable_delta_retained_mean": sign * observed,
                        "raw_bootstrap_ci95_low": low,
                        "raw_bootstrap_ci95_high": high,
                        "favorable_bootstrap_ci95_low": min(sign * low, sign * high),
                        "favorable_bootstrap_ci95_high": max(sign * low, sign * high),
                        "bootstrap_valid": valid,
                    }
                )
    return pd.DataFrame(curve_rows), pd.DataFrame(delta_rows)


def make_figure(
    associations: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    colors = {
        "epistemic_confidence": "#3B6FB6",
        "aleatoric_confidence": "#7E9BC5",
        "combined_confidence": "#C75B39",
        "predicted_magnitude": "#6B6B6B",
    }
    primary = associations[associations["target"] == "pearson_effect_accuracy"].copy()
    scopes = ["Norman_P1", "Norman_P2", "two_panel_macro"]
    scores = list(SCORES)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    x = np.arange(len(scopes))
    width = 0.19
    for offset, score in enumerate(scores):
        subset = primary.set_index(["scope", "score"])
        values = [subset.loc[(scope, score), "spearman_rho"] for scope in scopes]
        axes[0].bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            color=colors[score],
            label=score.replace("_", " "),
        )
    axes[0].axhline(0, color="#555555", lw=0.8)
    axes[0].set_xticks(x, ["P1", "P2", "macro"])
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("Confidence vs Pearson effect accuracy")

    primary_curves = curves[curves["target"] == "pearson_effect_accuracy"]
    macro_curve = (
        primary_curves.groupby(["score", "coverage"], as_index=False)["retained_mean"].mean()
    )
    for score in scores:
        subset = macro_curve[macro_curve["score"] == score]
        axes[1].plot(
            subset["coverage"],
            subset["retained_mean"],
            marker="o",
            ms=3,
            lw=1.8,
            color=colors[score],
            label=score.replace("_", " "),
        )
    axes[1].set_xlabel("Coverage retained")
    axes[1].set_ylabel("Mean Pearson effect accuracy")
    axes[1].set_title("Retrospective filtering (panel macro mean)")

    for axis in axes:
        axis.grid(axis="y", color="#E5E5E5", lw=0.7)
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "figures/F1_paper_endpoint_audit.svg", facecolor="white")
    plt.close(fig)


def write_report(
    associations: pd.DataFrame,
    deltas: pd.DataFrame,
    redundancy: pd.DataFrame,
    coverage_deltas: pd.DataFrame,
) -> None:
    primary = associations[associations["target"] == "pearson_effect_accuracy"]
    primary_display = primary[
        [
            "scope",
            "score",
            "spearman_rho",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ]
    ].copy()
    primary_display = primary_display.round(4)

    primary_delta = deltas[
        (deltas["target"] == "pearson_effect_accuracy")
        & (deltas["scope"] == "two_panel_macro")
    ][
        [
            "score",
            "raw_delta_rho",
            "raw_bootstrap_ci95_low",
            "raw_bootstrap_ci95_high",
        ]
    ].round(4)
    redundancy_display = redundancy[
        redundancy["scope"].isin(["Norman_P1", "Norman_P2", "two_panel_macro"])
    ][
        [
            "scope",
            "score",
            "spearman_rho",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
        ]
    ].round(4)
    filtering = coverage_deltas[
        (coverage_deltas["scope"] == "two_panel_macro")
        & (coverage_deltas["target"] == "pearson_effect_accuracy")
    ][
        [
            "score",
            "coverage",
            "raw_delta_retained_mean",
            "raw_bootstrap_ci95_low",
            "raw_bootstrap_ci95_high",
        ]
    ].round(4)

    combined_macro = primary[
        (primary["scope"] == "two_panel_macro")
        & (primary["score"] == "combined_confidence")
    ].iloc[0]
    combined_panels = primary[
        (primary["scope"].isin(["Norman_P1", "Norman_P2"]))
        & (primary["score"] == "combined_confidence")
    ].set_index("scope")
    magnitude_macro = primary[
        (primary["scope"] == "two_panel_macro")
        & (primary["score"] == "predicted_magnitude")
    ].iloc[0]
    combined_delta = deltas[
        (deltas["scope"] == "two_panel_macro")
        & (deltas["score"] == "combined_confidence")
        & (deltas["target"] == "pearson_effect_accuracy")
    ].iloc[0]
    combined_redundancy = redundancy[
        (redundancy["score"] == "combined_confidence")
        & (redundancy["scope"].isin(["Norman_P1", "Norman_P2"]))
    ].set_index("scope")

    if (
        combined_macro["bootstrap_ci95_low"] > 0
        and combined_panels.loc["Norman_P1", "spearman_rho"] > 0
        and combined_panels.loc["Norman_P2", "spearman_rho"] > 0
    ):
        association_text = "组合置信度在两个面板同向，且双面板宏平均区间高于 0。"
    else:
        association_text = "组合置信度未达到合同规定的双面板稳定正关联标准。"
    if combined_delta["raw_bootstrap_ci95_low"] > 0:
        delta_text = "组合置信度相对 magnitude 的增量区间高于 0。"
    else:
        delta_text = "组合置信度相对 magnitude 的增量区间未高于 0，不能宣称独立排序增益。"

    report = f"""# E145｜PRESCRIBE 论文终点口径纠正

## 结论

PRESCRIBE 原论文用预测与真实扰动效应的 Pearson 相关作为置信度校准的默认准确度终点。E145 按这一口径重新分析已有 E95 Norman P1/P2 formal 输出。组合置信度与 Pearson 准确度的 Spearman 为：P1 `{combined_panels.loc['Norman_P1', 'spearman_rho']:.4f}`，P2 `{combined_panels.loc['Norman_P2', 'spearman_rho']:.4f}`，双面板等权宏平均 `{combined_macro['spearman_rho']:.4f}`（任务 bootstrap 95% CI `{combined_macro['bootstrap_ci95_low']:.4f}` 至 `{combined_macro['bootstrap_ci95_high']:.4f}`）。{association_text}

同一终点上，predicted magnitude 的双面板宏 Spearman 为 `{magnitude_macro['spearman_rho']:.4f}`。组合置信度相对 magnitude 的 Δρ 为 `{combined_delta['raw_delta_rho']:.4f}`（95% CI `{combined_delta['raw_bootstrap_ci95_low']:.4f}` 至 `{combined_delta['raw_bootstrap_ci95_high']:.4f}`）。{delta_text}

组合置信度与 magnitude 的面板内 Spearman 分别为 P1 `{combined_redundancy.loc['Norman_P1', 'spearman_rho']:.4f}`、P2 `{combined_redundancy.loc['Norman_P2', 'spearman_rho']:.4f}`。因此，E96 中“PRESCRIBE 在当前 setting 完全没有可靠性信号”的表述需要收窄；E145 能回答的是论文定义的方向准确度信号，不能据此宣称它稳定超过幅度基线。

## 论文口径核对

论文式（5）定义 `pseudo E-distance = 2 × normalized posterior evidence − normalized predictive entropy`。官方测试代码对每个扰动取 `epistemic_conf` 和 `aleatoric_conf` 的细胞均值，再计算 `2 × epistemic + aleatoric`。论文第 4.2 节把置信度校准默认定义为置信度与预测—真实 log-fold-change Pearson 准确度之间的相关。E145 的预测效应和真实效应均由同一 log-normalized 表达空间减去同一 control 均值得到。

来源：<{PAPER_URL}>。

## Pearson 主终点

{markdown_table(primary_display)}

## 相对 predicted magnitude 的增量

{markdown_table(primary_delta)}

## 与 predicted magnitude 的排序重合

{markdown_table(redundancy_display)}

## 论文式 5%/10% 过滤的回顾性比较

下表是置信度过滤与 magnitude 过滤在同一 coverage 下的保留集平均 Pearson 差值；正值有利于置信度。

{markdown_table(filtering)}

## 边界

- E145 使用已查看真实结果的 P1/P2 数据，只是 post-unblinding metric correction，不是独立确认。
- 过滤曲线在相同测试任务上排序并评价，只能描述回顾性选择性表现。
- cosine、RMSE 和所有逐任务数据保存在 `tables/`，没有因结果删除任务。
- E145 不把 SafeConf 与 PRESCRIBE 混成同预测器的直接比较，也不改动 E96 已报告的 RMSE 结果。
"""
    (OUT / "reports/E145_REPORT.md").write_text(report)


def main() -> None:
    contract = OUT / "ANALYSIS_CONTRACT.md"
    if not contract.exists():
        raise RuntimeError("ANALYSIS_CONTRACT.md must exist before E145 is run")
    for name in ["tables", "figures", "reports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)

    validated_hashes = validate_frozen_files()
    loaded = [load_panel(panel, config) for panel, config in PANELS.items()]
    tasks = pd.concat([item[0] for item in loaded], ignore_index=True)
    panel_metadata = {
        panel: loaded[index][1] for index, panel in enumerate(PANELS)
    }
    if len(tasks) != 48 or tasks["task_id"].nunique() != 48:
        raise RuntimeError("Expected 48 distinct tasks across the two frozen panels")
    tasks.to_csv(OUT / "tables/E145_TASK_METRICS.csv", index=False)

    associations = build_associations(tasks)
    deltas = build_incremental_deltas(tasks)
    redundancy = build_score_redundancy(tasks)
    curves, coverage_deltas = build_coverage(tasks)
    associations.to_csv(OUT / "tables/E145_ASSOCIATIONS.csv", index=False)
    deltas.to_csv(OUT / "tables/E145_INCREMENTAL_VS_MAGNITUDE.csv", index=False)
    redundancy.to_csv(OUT / "tables/E145_SCORE_REDUNDANCY.csv", index=False)
    curves.to_csv(OUT / "tables/E145_COVERAGE_CURVES.csv", index=False)
    coverage_deltas.to_csv(OUT / "tables/E145_COVERAGE_VS_MAGNITUDE_BOOTSTRAP.csv", index=False)

    make_figure(associations, curves)
    write_report(associations, deltas, redundancy, coverage_deltas)
    (OUT / "README_先看这个.md").write_text(
        "# E145 先看这个\n\n"
        "先读 `ANALYSIS_CONTRACT.md`，再读 `reports/E145_REPORT.md`。\n"
        "本实验是已解封 E95/E96 的论文终点口径纠正，不是独立确认。\n"
    )

    primary = associations[
        (associations["scope"] == "two_panel_macro")
        & (associations["target"] == "pearson_effect_accuracy")
    ].set_index("score")
    primary_delta = deltas[
        (deltas["scope"] == "two_panel_macro")
        & (deltas["score"] == "combined_confidence")
        & (deltas["target"] == "pearson_effect_accuracy")
    ].iloc[0]
    status = {
        "experiment": "E145_prescribe_paper_endpoint_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_nature": "post_unblinding_metric_correction_not_independent_confirmation",
        "paper_url": PAPER_URL,
        "paper_exact_combined_formula": "2 * mean(epistemic_conf) + mean(aleatoric_conf)",
        "primary_endpoint": "pearson_effect_accuracy_on_2037_log_normalized_effect_genes",
        "n_panels": 2,
        "n_tasks": int(len(tasks)),
        "n_bootstrap": N_BOOT,
        "coverage_grid": [float(value) for value in COVERAGES],
        "focal_filtering": ["bottom_5_percent", "bottom_10_percent"],
        "panel_metadata": panel_metadata,
        "validated_frozen_hashes": validated_hashes,
        "analysis_contract_sha256": sha256(contract),
        "primary_macro_spearman": {
            score: {
                "rho": float(primary.loc[score, "spearman_rho"]),
                "ci95_low": float(primary.loc[score, "bootstrap_ci95_low"]),
                "ci95_high": float(primary.loc[score, "bootstrap_ci95_high"]),
            }
            for score in SCORES
        },
        "combined_delta_rho_vs_magnitude": {
            "delta": float(primary_delta["raw_delta_rho"]),
            "ci95_low": float(primary_delta["raw_bootstrap_ci95_low"]),
            "ci95_high": float(primary_delta["raw_bootstrap_ci95_high"]),
        },
        "target_truth_used_to_change_task_set_or_score": False,
        "target_truth_used_for_posthoc_metric_correction": True,
        "independent_confirmation": False,
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("\nPrimary panel/macro associations")
    print(
        associations[associations["target"] == "pearson_effect_accuracy"][
            ["scope", "score", "spearman_rho", "bootstrap_ci95_low", "bootstrap_ci95_high"]
        ].to_string(index=False)
    )
    print("\nPrimary macro deltas vs magnitude")
    print(
        deltas[
            (deltas["scope"] == "two_panel_macro")
            & (deltas["target"] == "pearson_effect_accuracy")
        ][
            ["score", "raw_delta_rho", "raw_bootstrap_ci95_low", "raw_bootstrap_ci95_high"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
