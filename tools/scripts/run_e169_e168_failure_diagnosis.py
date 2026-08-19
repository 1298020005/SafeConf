#!/usr/bin/env python3
"""E169: reproducible post-unseal diagnosis of the E168 no-confirmation result.

This script is deliberately diagnostic.  It never changes the frozen E168
endpoint, score, thresholds, or decision.  It consumes only the committed E168
pretruth/postgate releases plus the physically isolated F2/F3 assets.  Every
analysis conceived after unsealing is marked exploratory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
E168 = ROOT / "docs/实验结果/E168_primary_human_cd4_fresh_confirmation_20260716"
PRE = E168 / "pretruth_release"
POST = E168 / "postgate_release"
F2 = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/F2_pretruth")
F3 = Path("/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/F3_postgate")
MODEL = ROOT / "docs/实验结果/E135_directional_risk_lodo_20260714/E135_FROZEN_DIRECTION_MODEL.json"
OUT = ROOT / "docs/实验结果/E169_e168_failure_diagnosis_20260718"
TABLES, FIGURES, REPORTS = OUT / "tables", OUT / "figures", OUT / "reports"

SEED = 202607169
N_BOOTSTRAP = 3000
STATES = ("Rest", "Stim8hr", "Stim48hr")
STRATA = {
    "all_200": None,
    "seen_160": "DONOR_UNSEEN_ONLY",
    "column_unseen_40_descriptive": "COLUMN_UNSEEN",
}
SCORES = {
    "SafeConf": "safeconf_risk",
    "magnitude": "magnitude_risk",
    "disagreement": "disagreement_only_risk",
}


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    lines = (root / "MANIFEST.sha256").read_text().splitlines()
    checked = 0
    for line in lines:
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        observed = sha256(root / relative)
        if observed != expected:
            raise IntegrityError(f"manifest mismatch: {relative}")
        checked += 1
    return checked


def rho(a: Any, b: Any) -> float:
    left, right = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 4 or np.unique(left[keep]).size < 2 or np.unique(right[keep]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(left[keep]), rankdata(right[keep]))[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return cosine(a - np.mean(a), b - np.mean(b))


def robust_z(values: Any, reference: Any) -> np.ndarray:
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(1.4826 * mad, float(np.std(reference)), 1e-8)
    return np.clip((values - center) / scale, -5.0, 5.0)


def weak_order_identical(a: Any, b: Any, tolerance: float = 1e-10) -> bool:
    left = np.rint(np.asarray(a, float) / tolerance).astype(np.int64)
    right = np.rint(np.asarray(b, float) / tolerance).astype(np.int64)
    return bool(np.array_equal(rankdata(left), rankdata(right)))


def tie_aware_aurc(score: Any, loss: Any) -> float:
    score, loss = np.asarray(score, float), np.asarray(loss, float)
    labels = np.rint(score / 1e-10).astype(np.int64)
    levels = np.sort(np.unique(labels))
    n = len(score)
    cumulative_sum = 0.0
    cumulative_count = 0
    coverages, risks = [], []
    for level in levels:
        block = loss[labels == level]
        start_count, start_sum = cumulative_count, cumulative_sum
        for slot in range(1, len(block) + 1):
            cumulative_count = start_count + slot
            cumulative_sum = start_sum + slot * float(np.mean(block))
            coverages.append(cumulative_count / n)
            risks.append(cumulative_sum / cumulative_count)
    if coverages[0] > 0:
        coverages.insert(0, 0.0)
        risks.insert(0, risks[0])
    return float(np.trapezoid(np.asarray(risks), np.asarray(coverages)))


def input_audit() -> dict[str, Any]:
    snapshot = json.loads((PRE / "PRETRUTH_GATE_SNAPSHOT.json").read_text())
    status = json.loads((POST / "RUN_STATUS.json").read_text())
    if snapshot.get("status") != "PASS" or status.get("decision") != "NO_CONFIRMATION":
        raise IntegrityError("E169 requires the immutable E168 PASS gate and NO_CONFIRMATION result")
    for relative, expected in snapshot["pretruth_files_sha256"].items():
        if sha256(PRE / relative) != expected:
            raise IntegrityError(f"pretruth snapshot mismatch: {relative}")
    f2_manifest = sha256(F2 / "MANIFEST.sha256")
    f3_manifest = sha256(F3 / "MANIFEST.sha256")
    if f2_manifest != status["f2_manifest_sha256"] or f3_manifest != status["f3_manifest_sha256"]:
        raise IntegrityError("isolated asset manifest changed")
    return {
        "pretruth_bound_files": len(snapshot["pretruth_files_sha256"]),
        "postgate_bound_files": verify_manifest(POST),
        "gate_commit": status["gate_commit"],
        "gate_snapshot_sha256": status["gate_snapshot_sha256"],
        "f2_manifest_sha256": f2_manifest,
        "f3_manifest_sha256": f3_manifest,
        "prediction_record_contract_issues": status["prediction_record_contract_issues"],
    }


def identifiability_tables(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows, score_rows = [], []
    features = {
        "context_similarity": "context_similarity_max",
        "support_count": "perturbation_support_count",
        "model_disagreement": "model_disagreement_rmse",
        "predicted_magnitude": "predicted_magnitude",
        "SafeConf_risk": "safeconf_risk",
    }
    for state in STATES:
        state_frame = tasks[tasks.culture_condition.eq(state)]
        for stratum_name, stratum in STRATA.items():
            block = state_frame if stratum is None else state_frame[state_frame.target_stratum.eq(stratum)]
            for feature, column in features.items():
                values = block[column].to_numpy(float)
                feature_rows.append({
                    "culture_condition": state,
                    "stratum": stratum_name,
                    "feature": feature,
                    "n_tasks": len(block),
                    "n_unique_operational_levels": int(len(np.unique(np.rint(values / 1e-10)))),
                    "population_std": float(np.std(values)),
                    "spearman_with_true_error": rho(values, block.true_error_rmse),
                    "exploratory_post_unseal": True,
                })
            for score_name, column in SCORES.items():
                score_rows.append({
                    "culture_condition": state,
                    "stratum": stratum_name,
                    "score": score_name,
                    "n_tasks": len(block),
                    "spearman_with_true_error": rho(block[column], block.true_error_rmse),
                    "tie_aware_aurc": tie_aware_aurc(block[column], block.true_error_rmse),
                    "weak_order_identical_to_disagreement": weak_order_identical(
                        block[column], block.disagreement_only_risk
                    ),
                    "weak_order_identical_to_magnitude": weak_order_identical(
                        block[column], block.magnitude_risk
                    ),
                    "exploratory_post_unseal": True,
                })
    return pd.DataFrame(feature_rows), pd.DataFrame(score_rows)


def derive_pretruth_history(pre: pd.DataFrame) -> pd.DataFrame:
    with np.load(PRE / "arrays/PRETRUTH_PREDICTIONS.npz") as predictions, np.load(
        F2 / "SEEN_TARGET_EFFECTS.npz"
    ) as truths:
        ensemble = predictions["ensemble_seed_family_mean"]
        errors = []
        truth_keys = set(truths.files)
        for index, row in enumerate(pre.itertuples(index=False)):
            if row.task_id not in truth_keys:
                errors.append(float("nan"))
            else:
                errors.append(float(np.sqrt(np.mean((ensemble[index] - truths[row.task_id]) ** 2))))
    result = pre.copy()
    result["derived_pretruth_error"] = errors
    return result


def history_diagnostics(pre: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    keys = ["culture_condition", "perturbed_gene_id"]
    validation = pre[
        pre.donor_role.eq("validation") & pre.derived_pretruth_error.notna()
    ][keys + ["derived_pretruth_error"]].rename(
        columns={"derived_pretruth_error": "validation_donor_error"}
    )
    training = (
        pre[pre.donor_role.eq("train") & pre.derived_pretruth_error.notna()]
        .groupby(keys, as_index=False)
        .agg(
            train_donor_error_mean=("derived_pretruth_error", "mean"),
            train_donor_error_sd=("derived_pretruth_error", "std"),
        )
    )
    seen = (
        tasks[tasks.target_stratum.eq("DONOR_UNSEEN_ONLY")]
        .merge(validation, on=keys, how="left", validate="one_to_one")
        .merge(training, on=keys, how="left", validate="one_to_one")
    )
    candidates = {
        "frozen_SafeConf": "safeconf_risk",
        "magnitude": "magnitude_risk",
        "disagreement": "disagreement_only_risk",
        "validation_donor_error": "validation_donor_error",
        "train_donor_error_mean": "train_donor_error_mean",
        "train_donor_error_sd": "train_donor_error_sd",
    }
    rows = []
    for state, block in seen.groupby("culture_condition", sort=True):
        for name, column in candidates.items():
            rows.append({
                "culture_condition": state,
                "feature_or_score": name,
                "n_tasks": len(block),
                "spearman_with_test_error": rho(block[column], block.true_error_rmse),
                "tie_aware_aurc": tie_aware_aurc(block[column], block.true_error_rmse),
                "analysis_role": "exploratory_post_unseal_no_model_selection",
            })
    return pd.DataFrame(rows)


def guide_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    guides = pd.read_csv(POST / "tables/GUIDE_CONSISTENCY.csv")
    merged = guides.merge(
        tasks[["task_id", "true_error_rmse"]], on="task_id", how="left", validate="one_to_one"
    )
    rows = []
    for (state, stratum), block in merged.groupby(["culture_condition", "target_stratum"]):
        rows.append({
            "culture_condition": state,
            "target_stratum": stratum,
            "n_tasks": len(block),
            "median_guide_effect_rmse": float(block.guide_effect_rmse.median()),
            "median_guide_effect_cosine_similarity": float(
                block.guide_effect_cosine_similarity.median()
            ),
            "median_guide_effect_spearman": float(block.guide_effect_spearman.median()),
            "guide_rmse_vs_task_error_spearman": rho(
                block.guide_effect_rmse, block.true_error_rmse
            ),
            "truth_derived_diagnostic_only": True,
        })
    return pd.DataFrame(rows)


def trim_generated_svg(path: Path) -> None:
    """Normalize Matplotlib path whitespace so git's whitespace audit stays clean."""
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def directional_secondary(pre: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the pre-existing E135 model without refitting; this is post-hoc for E168."""
    model = json.loads(MODEL.read_text())
    test = pre[pre.donor_role.eq("test")].copy().reset_index().rename(
        columns={"index": "prediction_row"}
    )
    validation = pre.donor_role.eq("validation")
    test["risk_disagreement_z"] = robust_z(
        test.model_disagreement_rmse, pre.loc[validation, "model_disagreement_rmse"]
    )
    test["predicted_magnitude_z"] = robust_z(
        test.predicted_magnitude, pre.loc[validation, "predicted_magnitude"]
    )
    with np.load(F2 / "CONTROL_PROFILES.npz") as bundle:
        controls = {key: np.asarray(bundle[key], float) for key in bundle.files}
    control_keys = sorted(controls)
    distances = []
    for index, left in enumerate(control_keys):
        for right in control_keys[index + 1 :]:
            distance = 1.0 - cosine(controls[left], controls[right])
            if distance > 1e-10:
                distances.append(distance)
    context_scale = max(float(np.median(distances)), 1e-8)
    train_contexts = [
        f"{donor}::{state}"
        for donor in ("CE0006864", "CE0008162")
        for state in STATES
    ]
    novelty = []
    for row in test.itertuples(index=False):
        key = f"{row.donor_id}::{row.culture_condition}"
        similarity = max(cosine(controls[key], controls[item]) for item in train_contexts)
        novelty.append(min((1.0 - similarity) / context_scale, 5.0))
    test["context_novelty_scaled"] = novelty
    test["perturbation_novelty"] = 1.0 / (
        1.0 + test.perturbation_support_count.to_numpy(float)
    )
    matrix = test[model["features_in_order"]].to_numpy(float)
    test["directional_risk_frozen"] = float(model["intercept"]) + matrix @ np.asarray(
        model["coefficients_in_order"], float
    )

    train = pre[pre.donor_role.eq("train")]
    with np.load(F2 / "SEEN_TARGET_EFFECTS.npz") as truths:
        states = [
            controls[f"{row.donor_id}::{row.culture_condition}"] + truths[row.task_id]
            for row in train.itertuples(index=False)
        ]
    centroid = np.mean(np.stack(states), axis=0)
    with np.load(PRE / "arrays/PRETRUTH_PREDICTIONS.npz") as predictions, np.load(
        F3 / "TEST_TARGET_EFFECTS.npz"
    ) as truths:
        scgpt = predictions["scGPT_seed_mean"]
        gears = predictions["GEARS_seed_mean"]
        pearson_error, cosine_error = [], []
        for row in test.itertuples(index=False):
            control = controls[f"{row.donor_id}::{row.culture_condition}"]
            true_centered = control + truths[row.task_id] - centroid
            task_pearson, task_cosine = [], []
            for values in (scgpt[row.prediction_row], gears[row.prediction_row]):
                predicted_centered = control + values - centroid
                task_pearson.append(1.0 - pearson(predicted_centered, true_centered))
                task_cosine.append(1.0 - cosine(predicted_centered, true_centered))
            pearson_error.append(float(np.mean(task_pearson)))
            cosine_error.append(float(np.mean(task_cosine)))
    test["error_centered_pearson_mean"] = pearson_error
    test["error_centered_cosine_mean"] = cosine_error
    rank_parts = []
    for endpoint in ("error_centered_pearson_mean", "error_centered_cosine_mean"):
        rank_parts.append(
            test.groupby("culture_condition")[endpoint].transform(
                lambda values: rankdata(values) / len(values)
            )
        )
    test["direction_error_rank_target"] = np.mean(np.stack(rank_parts), axis=0)
    test["directional_model_refit_on_e168"] = False
    test["analysis_conceived_after_e168_primary_unseal"] = True
    test["frozen_direction_model_sha256"] = sha256(MODEL)

    score_columns = {
        "directional_risk_frozen": "directional_risk_frozen",
        "predicted_magnitude": "predicted_magnitude",
        "SafeConf_absolute_head": "safeconf_risk",
        "model_disagreement": "model_disagreement_rmse",
    }
    endpoints = (
        "error_centered_pearson_mean",
        "error_centered_cosine_mean",
        "direction_error_rank_target",
    )
    metric_rows = []
    for stratum_name, stratum in STRATA.items():
        target = test if stratum is None else test[test.target_stratum.eq(stratum)]
        for state, block in target.groupby("culture_condition", sort=True):
            for score_name, score_column in score_columns.items():
                for endpoint in endpoints:
                    metric_rows.append({
                        "stratum": stratum_name,
                        "culture_condition": state,
                        "score": score_name,
                        "endpoint": endpoint,
                        "n_tasks": len(block),
                        "spearman": rho(block[score_column], block[endpoint]),
                        "analysis_role": "secondary_post_unseal_frozen_model_no_refit",
                    })
    metrics = pd.DataFrame(metric_rows)

    # Pre-pack state × target arrays.  This is exactly the same target-cluster
    # resampling as a repeated pandas concat, but it avoids rebuilding 200 tiny
    # DataFrames in every draw and keeps the multi-panel extension tractable.
    packed: dict[str, dict[str, Any]] = {}
    for stratum_name, stratum in {
        "all_200": None,
        "seen_160": "DONOR_UNSEEN_ONLY",
    }.items():
        target = test if stratum is None else test[test.target_stratum.eq(stratum)]
        genes = sorted(target.perturbed_gene_id.unique())
        score_arrays, endpoint_arrays = {}, {}
        for score_name, score_column in score_columns.items():
            score_arrays[score_name] = np.stack([
                target[target.culture_condition.eq(state)]
                .set_index("perturbed_gene_id")
                .loc[genes, score_column]
                .to_numpy(float)
                for state in STATES
            ])
        for endpoint in endpoints:
            endpoint_arrays[endpoint] = np.stack([
                target[target.culture_condition.eq(state)]
                .set_index("perturbed_gene_id")
                .loc[genes, endpoint]
                .to_numpy(float)
                for state in STATES
            ])
        packed[stratum_name] = {
            "n_genes": len(genes),
            "scores": score_arrays,
            "endpoints": endpoint_arrays,
        }

    rng = np.random.default_rng(SEED)
    draw_rows = []
    for draw in range(N_BOOTSTRAP):
        row: dict[str, Any] = {"draw": draw}
        for stratum_name, bundle in packed.items():
            take = rng.integers(0, bundle["n_genes"], bundle["n_genes"])
            for score_name, score_values in bundle["scores"].items():
                for endpoint, endpoint_values in bundle["endpoints"].items():
                    estimates = [
                        rho(score_values[state_index, take], endpoint_values[state_index, take])
                        for state_index in range(len(STATES))
                    ]
                    row[f"{stratum_name}::{score_name}::{endpoint}"] = float(
                        np.nanmean(estimates)
                    )
        draw_rows.append(row)
    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for column in draws.columns[1:]:
        stratum, score, endpoint = column.split("::")
        values = draws[column].to_numpy(float)
        summary_rows.append({
            "stratum": stratum,
            "score": score,
            "endpoint": endpoint,
            "bootstrap_draws": N_BOOTSTRAP,
            "ci95_lower": float(np.nanquantile(values, 0.025)),
            "median": float(np.nanmedian(values)),
            "ci95_upper": float(np.nanquantile(values, 0.975)),
            "fraction_above_zero": float(np.nanmean(values > 0)),
            "analysis_role": "secondary_post_unseal_frozen_model_no_refit",
        })
    return test, metrics, pd.DataFrame(summary_rows)


def plot_state_delta(state_deltas: pd.DataFrame) -> None:
    data = state_deltas[state_deltas.stratum.isin(["all_200", "seen_160"])].copy()
    labels = [f"{row.stratum} · {row.culture_condition}" for row in data.itertuples()]
    values = data.delta_magnitude_minus_safeconf.to_numpy(float)
    colors = ["#2B6F6D" if value > 0 else "#B24A45" for value in values]
    fig, axis = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    axis.set_facecolor("white")
    y = np.arange(len(data))
    axis.barh(y, values, color=colors, height=0.58)
    axis.axvline(0, color="#4B555B", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("ΔAURC = magnitude − SafeConf（正值表示 SafeConf 更好）")
    axis.set_title("E168：三种状态的冻结主终点", loc="left", weight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#DDE3E5", linewidth=0.7)
    fig.tight_layout()
    svg = FIGURES / "E169_FIG1_STATE_DELTAS.svg"
    fig.savefig(svg, facecolor="white")
    trim_generated_svg(svg)
    fig.savefig(FIGURES / "E169_FIG1_STATE_DELTAS.png", dpi=220, facecolor="white")
    plt.close(fig)


def plot_feature_levels(features: pd.DataFrame) -> None:
    data = features[
        features.stratum.eq("seen_160")
        & features.feature.isin(["context_similarity", "support_count", "model_disagreement", "predicted_magnitude"])
    ]
    pivot = data.pivot(index="feature", columns="culture_condition", values="n_unique_operational_levels")
    pivot = pivot.reindex(["context_similarity", "support_count", "model_disagreement", "predicted_magnitude"])
    fig, axis = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    pivot.plot.bar(ax=axis, color=["#3B6EA8", "#6E9F72", "#C58B3A"], width=0.72)
    axis.set_yscale("symlog", linthresh=2)
    axis.set_ylabel("可区分取值数（symlog）")
    axis.set_xlabel("")
    axis.set_title("seen-160 内部：结构特征没有排序分辨率", loc="left", weight="bold")
    axis.legend(title="状态", frameon=False, ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDE3E5", linewidth=0.7)
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    svg = FIGURES / "E169_FIG2_FEATURE_RESOLUTION.svg"
    fig.savefig(svg, facecolor="white")
    trim_generated_svg(svg)
    fig.savefig(FIGURES / "E169_FIG2_FEATURE_RESOLUTION.png", dpi=220, facecolor="white")
    plt.close(fig)


def plot_directional(metrics: pd.DataFrame) -> None:
    data = metrics[
        metrics.stratum.eq("all_200")
        & metrics.endpoint.eq("direction_error_rank_target")
    ].copy()
    order = ["directional_risk_frozen", "predicted_magnitude", "model_disagreement", "SafeConf_absolute_head"]
    pivot = data.pivot(index="score", columns="culture_condition", values="spearman").reindex(order)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    pivot.plot.bar(ax=axis, color=["#3B6EA8", "#6E9F72", "#C58B3A"], width=0.72)
    axis.axhline(0, color="#4B555B", linewidth=0.9)
    axis.set_ylabel("Spearman ρ")
    axis.set_xlabel("")
    axis.set_title("冻结方向风险头：E168 事后二级审计", loc="left", weight="bold")
    axis.legend(title="状态", frameon=False, ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDE3E5", linewidth=0.7)
    axis.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    svg = FIGURES / "E169_FIG3_DIRECTIONAL_SECONDARY.svg"
    fig.savefig(svg, facecolor="white")
    trim_generated_svg(svg)
    fig.savefig(FIGURES / "E169_FIG3_DIRECTIONAL_SECONDARY.png", dpi=220, facecolor="white")
    plt.close(fig)


def write_report(
    state: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    history: pd.DataFrame,
    guides: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    direction_bootstrap: pd.DataFrame,
) -> None:
    all_primary = state[state.stratum.eq("all_200")]
    seen_primary = state[state.stratum.eq("seen_160")]
    direction = (
        direction_metrics[
            direction_metrics.stratum.eq("all_200")
            & direction_metrics.endpoint.eq("direction_error_rank_target")
        ]
        .groupby("score")
        .spearman.mean()
    )
    direction_ci = direction_bootstrap[
        direction_bootstrap.stratum.eq("all_200")
        & direction_bootstrap.endpoint.eq("direction_error_rank_target")
    ].set_index("score")
    seen_resolution = features[
        features.stratum.eq("seen_160")
        & features.feature.isin(["context_similarity", "support_count"])
    ]
    collapse = scores[
        scores.stratum.isin(["seen_160", "column_unseen_40_descriptive"])
        & scores.score.eq("SafeConf")
    ].weak_order_identical_to_disagreement.all()
    lines = [
        "# E169｜E168 未确认结果的可复现诊断",
        "",
        "## 结论",
        "",
        "E168 的正式判定保持 **NO_CONFIRMATION**，本轮没有改公式、换终点或重写门槛。全 200 targets 的 ΔAURC 为 "
        f"{all_primary.delta_magnitude_minus_safeconf.mean():.6f}；seen 160 为 "
        f"{seen_primary.delta_magnitude_minus_safeconf.mean():.6f}。两个区间均跨 0，不能写成独立确认。",
        "",
        "## 为什么没有形成增量",
        "",
        f"在 seen-160 和 column-unseen-40 各自内部，SafeConf 与 disagreement 的弱序完全一致：**{collapse}**。"
        "同一 test donor、同一状态内，context similarity 只有 1 个取值；分层后 support 也只有 1 个取值。"
        "因此这两个结构特征只能区分 seen/unseen 两层，不能在每层内部给 160 或 40 个基因排序。真正承担排序的只剩模型分歧。",
        "",
        f"seen-160 的 context/support 可区分层级范围为 {int(seen_resolution.n_unique_operational_levels.min())}–"
        f"{int(seen_resolution.n_unique_operational_levels.max())}。这不是代码退化：24 个预测器证书、6 个风险证书和三种子稳定证书都已通过；问题在于可部署特征的信息量。",
        "",
        "## 三状态结果",
        "",
        "| stratum | state | SafeConf AURC | magnitude AURC | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in state.itertuples(index=False):
        lines.append(
            f"| {row.stratum} | {row.culture_condition} | {row.safeconf_aurc:.6f} | "
            f"{row.magnitude_aurc:.6f} | {row.delta_magnitude_minus_safeconf:+.6f} |"
        )
    lines += [
        "",
        "column-unseen-40 在三个状态的 Δ 都为负；seen-160 三个状态点估计都为正，但效应很小，层级推断未通过。",
        "",
        "## 另外两项检查",
        "",
        "validation donor 的历史误差、两个 train donors 的平均误差和跨 donor 标准差全部原样检查，没有只保留最好的组合。"
        "这些线索在已解封 160 个 seen targets 上信号偏弱，只能用于下一轮开发集特征筛选，不能拿来重算 E168 主结论。",
        "",
        f"guide 复现审计覆盖 {int(guides.n_tasks.sum())} 个任务。guide 间差异是真值解封后才能得到的实验噪声指标，"
        "可解释测量不稳定性，但不能作为部署分数输入。",
        "",
        "## 冻结方向风险头的二级结果",
        "",
        f"E135 在 E168 之前冻结的 Directional-SafeConf 未重拟合。其三状态宏平均方向 rank ρ="
        f"{direction['directional_risk_frozen']:.3f}，bootstrap 95% CI "
        f"[{direction_ci.loc['directional_risk_frozen','ci95_lower']:.3f}, "
        f"{direction_ci.loc['directional_risk_frozen','ci95_upper']:.3f}]；magnitude="
        f"{direction['predicted_magnitude']:.3f}，disagreement={direction['model_disagreement']:.3f}。",
        "",
        "这项分析是在 E168 absolute 主结果解封后提出，只能标为事后二级审计。即使相关为正，也不能替代失败的主终点，更不能冒充第二次前瞻确认。",
        "",
        "## 下一步实验约束",
        "",
        "1. E168 已解封 200 targets 永久退出模型选择和阈值调整；",
        "2. 从 5,310 个尚未读取 targeting X 的合格 targets 中，按预先冻结哈希建立多个不重叠 200-target 面板；",
        "3. 先冻结所有面板、代码、分数和联合统计，再统一解封，检验当前微弱正效应能否靠更大样本得到精确结论；",
        "4. perturbation similarity、历史跨 donor 误差和模型特异风险只能在 train/validation 或其他开发数据中开发，之后另找未解封面板确认；",
        "5. 同一供体的多面板属于靶点复制，不冒充多供体或多研究复制。真正冲一区仍需要新研究/新背景，优先是 E143 前瞻湿实验。",
        "",
        "## 图",
        "",
        "- `figures/E169_FIG1_STATE_DELTAS.png`：absolute 主终点；",
        "- `figures/E169_FIG2_FEATURE_RESOLUTION.png`：结构特征分辨率；",
        "- `figures/E169_FIG3_DIRECTIONAL_SECONDARY.png`：方向风险二级审计。",
    ]
    (REPORTS / "E169_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "README_先看这个.md").write_text(
        "# E169 先看这个\n\n先读 `reports/E169_REPORT.md`。本实验解释 E168 的失败，不改变 E168 的正式 NO_CONFIRMATION。\n"
    )


def main() -> None:
    for directory in (OUT, TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    audit = input_audit()
    pre = pd.read_csv(PRE / "tables/PRETRUTH_SCORING_INTERFACE.csv")
    tasks = pd.read_csv(POST / "tables/TASK_METRICS.csv")
    if len(pre) != 2160 or len(tasks) != 600:
        raise IntegrityError("E168 task cardinality changed")
    state = pd.read_csv(POST / "tables/STATE_DELTAS.csv")
    features, scores = identifiability_tables(tasks)
    pre_with_history = derive_pretruth_history(pre)
    history = history_diagnostics(pre_with_history, tasks)
    guides = guide_summary(tasks)
    direction_tasks, direction_metrics, direction_bootstrap = directional_secondary(pre)

    state.to_csv(TABLES / "E169_STATE_DELTAS.csv", index=False)
    features.to_csv(TABLES / "E169_FEATURE_RESOLUTION.csv", index=False)
    scores.to_csv(TABLES / "E169_SCORE_EQUIVALENCE.csv", index=False)
    history.to_csv(TABLES / "E169_HISTORY_FEATURE_DIAGNOSTICS.csv", index=False)
    guides.to_csv(TABLES / "E169_GUIDE_CONSISTENCY_SUMMARY.csv", index=False)
    direction_tasks.to_csv(TABLES / "E169_DIRECTIONAL_TASK_AUDIT.csv", index=False)
    direction_metrics.to_csv(TABLES / "E169_DIRECTIONAL_STATE_METRICS.csv", index=False)
    direction_bootstrap.to_csv(TABLES / "E169_DIRECTIONAL_CLUSTER_BOOTSTRAP.csv", index=False)
    pd.DataFrame(
        [{
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
         for path in [
             PRE / "PRETRUTH_GATE_SNAPSHOT.json",
             POST / "RUN_STATUS.json",
             POST / "MANIFEST.sha256",
             F2 / "MANIFEST.sha256",
             F3 / "MANIFEST.sha256",
             MODEL,
             Path(__file__).resolve(),
         ]]
    ).to_csv(TABLES / "E169_INPUT_HASHES.csv", index=False)
    plot_state_delta(state)
    plot_feature_levels(features)
    plot_directional(direction_metrics)
    write_report(state, features, scores, history, guides, direction_metrics, direction_bootstrap)

    status = {
        "schema": "safeconf_e169_failure_diagnosis_v1",
        "experiment": "E169_e168_failure_diagnosis",
        "status": "COMPLETE",
        "e168_primary_decision_unchanged": "NO_CONFIRMATION",
        "analysis_role": "post_unseal_diagnostic",
        "score_or_threshold_refit_on_e168_test": False,
        "frozen_direction_model_refit": False,
        "n_test_tasks": len(tasks),
        "n_direction_bootstrap_draws": N_BOOTSTRAP,
        "safeconf_collapses_to_disagreement_within_seen_and_unseen_strata": bool(
            scores[
                scores.stratum.isin(["seen_160", "column_unseen_40_descriptive"])
                & scores.score.eq("SafeConf")
            ].weak_order_identical_to_disagreement.all()
        ),
        "input_audit": audit,
        "deployment_authorized": False,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    manifest_lines = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(OUT)}")
    (OUT / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(
        direction_metrics[direction_metrics.stratum.eq("all_200")]
        .groupby(["score", "endpoint"])
        .spearman.mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
