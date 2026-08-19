#!/usr/bin/env python3
"""E179: target-cluster nested benchmark for calibrated task-error upper bounds.

E179 is a retrospective method-development experiment.  It uses already opened
E176 and E177 truth, splits whole perturbation targets into train/calibration/
evaluation partitions, and compares all methods on identical repeated splits.
It does not claim a new prospective conformal guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E179_nested_uq_baseline_benchmark_20260723"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

E176 = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
E177 = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
SEEDS = (3407, 3408, 3409, 3410, 3411)
REPEAT_SEEDS = tuple(range(179001, 179051))
TARGET_COVERAGE = 0.90
EPS = 1e-12

FEATURES = (
    "predicted_magnitude",
    "pair_lower_bound",
    "scgpt_magnitude",
    "gears_magnitude",
    "model_cosine",
    "ensemble_mean_abs",
    "ensemble_max_abs",
    "disagreement_mean_abs",
    "sign_agreement",
    "ensemble_abs_p90",
    "ensemble_abs_p99",
    "disagreement_abs_p90",
    "disagreement_abs_p99",
    "ensemble_gene_std",
    "disagreement_gene_std",
    "scgpt_seed_spread",
    "gears_seed_spread",
    "ensemble_seed_spread",
)

METHOD_LABELS = {
    "constant": "Constant split conformal",
    "predicted_magnitude": "Predicted magnitude + conformal",
    "magnitude_plus_lower": "Magnitude + pair lower + conformal",
    "seed_spread": "Seed spread + conformal",
    "ridge_vector": "Ridge vector features + conformal",
    "extra_trees_vector": "ExtraTrees vector features + conformal",
    "random_forest_vector": "Random forest vector features + conformal",
    "cqr_q80_vector": "Q80 gradient boosting + conformal",
}
PRIMARY_METHOD = "extra_trees_vector"
REFERENCE_METHOD = "constant"

BLUE = "#3B6FB6"
TEAL = "#2A9D8F"
ORANGE = "#E6863B"
RED = "#C84C4C"
GREY = "#6B7280"
LIGHT = "#E8EEF6"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.12g"))


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def rmse_rows(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=1))


def vector_features(scgpt: np.ndarray, gears: np.ndarray) -> pd.DataFrame:
    ensemble = (scgpt + gears) / 2.0
    difference = scgpt - gears
    denominator = np.linalg.norm(scgpt, axis=1) * np.linalg.norm(gears, axis=1)
    cosine = np.sum(scgpt * gears, axis=1) / np.maximum(denominator, EPS)
    return pd.DataFrame(
        {
            "predicted_magnitude": rmse_rows(ensemble),
            "pair_lower_bound": rmse_rows(difference) / 2.0,
            "scgpt_magnitude": rmse_rows(scgpt),
            "gears_magnitude": rmse_rows(gears),
            "model_cosine": cosine,
            "ensemble_mean_abs": np.mean(np.abs(ensemble), axis=1),
            "ensemble_max_abs": np.max(np.abs(ensemble), axis=1),
            "disagreement_mean_abs": np.mean(np.abs(difference), axis=1),
            "sign_agreement": np.mean(np.sign(scgpt) == np.sign(gears), axis=1),
            "ensemble_abs_p90": np.quantile(np.abs(ensemble), 0.90, axis=1),
            "ensemble_abs_p99": np.quantile(np.abs(ensemble), 0.99, axis=1),
            "disagreement_abs_p90": np.quantile(np.abs(difference), 0.90, axis=1),
            "disagreement_abs_p99": np.quantile(np.abs(difference), 0.99, axis=1),
            "ensemble_gene_std": np.std(ensemble, axis=1),
            "disagreement_gene_std": np.std(difference, axis=1),
        }
    )


def add_seed_spread(frame: pd.DataFrame, archive: np.lib.npyio.NpzFile) -> None:
    scgpt = np.stack([archive[f"scGPT_seed{seed}"] for seed in SEEDS])
    gears = np.stack([archive[f"GEARS_seed{seed}"] for seed in SEEDS])
    frame["scgpt_seed_spread"] = np.sqrt(np.mean(np.var(scgpt, axis=0), axis=1))
    frame["gears_seed_spread"] = np.sqrt(np.mean(np.var(gears, axis=0), axis=1))
    frame["ensemble_seed_spread"] = np.sqrt(
        np.mean(np.var((scgpt + gears) / 2.0, axis=0), axis=1)
    )


def load_e176() -> tuple[pd.DataFrame, list[Path]]:
    calibration = E176 / "calibration_release/tables/CALIBRATION_TASK_METRICS.csv"
    evaluation = E176 / "final_evaluation/tables/EVALUATION_TASK_METRICS.csv"
    truth = pd.concat([pd.read_csv(calibration), pd.read_csv(evaluation)], ignore_index=True)
    truth = truth.set_index("task_id")
    used = [calibration, evaluation]
    parts: list[pd.DataFrame] = []
    for panel in ("H01", "H02", "H03", "H04"):
        release = E176 / f"pretruth_release/{panel}"
        interface_path = release / "tables/PRETRUTH_SCORING_INTERFACE.csv"
        prediction_path = release / "arrays/PRETRUTH_PREDICTIONS.npz"
        interface = pd.read_csv(interface_path)
        with np.load(prediction_path, allow_pickle=False) as archive:
            features = vector_features(
                np.asarray(archive["scGPT_seed_mean"], np.float64),
                np.asarray(archive["GEARS_seed_mean"], np.float64),
            )
            add_seed_spread(features, archive)
        interface = interface.drop(
            columns=[column for column in features.columns if column in interface.columns]
        )
        part = pd.concat([interface.reset_index(drop=True), features], axis=1)
        part = part[part["task_id"].isin(truth.index)].copy()
        part["outcome"] = part["task_id"].map(truth["pair_mean_rmse"])
        part["cluster_id"] = (
            part["panel_id"].astype(str) + "::" + part["perturbed_gene_name"].astype(str)
        )
        part["study"] = "E176_primary_CD4"
        part["context_label"] = part["culture_condition"].astype(str)
        parts.append(part)
        used.extend([interface_path, prediction_path])
    result = pd.concat(parts, ignore_index=True)
    return result, used


def load_e177() -> tuple[pd.DataFrame, list[Path]]:
    release = E177 / "pretruth_release"
    interface_path = release / "tables/PRETRUTH_SCORING_INTERFACE.csv"
    prediction_path = release / "arrays/PRETRUTH_PREDICTIONS.npz"
    calibration = E177 / "calibration_release/tables/CALIBRATION_TASK_ERRORS.csv"
    evaluation = E177 / "final_evaluation/tables/EVALUATION_TASK_RESULTS.csv"
    interface = pd.read_csv(interface_path)
    with np.load(prediction_path, allow_pickle=False) as archive:
        features = vector_features(
            np.asarray(archive["scGPT_seed_mean"], np.float64),
            np.asarray(archive["GEARS_seed_mean"], np.float64),
        )
        add_seed_spread(features, archive)
    truth = pd.concat(
        [
            pd.read_csv(calibration)[["task_id", "pair_mean_rmse"]],
            pd.read_csv(evaluation)[["task_id", "pair_mean_rmse"]],
        ],
        ignore_index=True,
    ).set_index("task_id")["pair_mean_rmse"]
    interface = interface.drop(
        columns=[column for column in features.columns if column in interface.columns]
    )
    result = pd.concat([interface.reset_index(drop=True), features], axis=1)
    result = result[result["task_id"].isin(truth.index)].copy()
    result["outcome"] = result["task_id"].map(truth)
    result["cluster_id"] = result["perturbation"].astype(str)
    result["study"] = "E177_Sunshine"
    result["context_label"] = result["technical_group"].astype(str)
    return result, [interface_path, prediction_path, calibration, evaluation]


def conformal_quantile(values: pd.Series, coverage: float = TARGET_COVERAGE) -> tuple[float, int]:
    ordered = np.sort(values.to_numpy(dtype=float))
    rank = math.ceil((len(ordered) + 1) * coverage)
    if rank > len(ordered):
        return math.inf, rank
    return float(ordered[rank - 1]), rank


def split_clusters(frame: pd.DataFrame, seed: int) -> dict[str, set[str]]:
    clusters = np.asarray(sorted(frame["cluster_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n = len(clusters)
    n_train = int(round(0.50 * n))
    n_calibration = int(round(0.25 * n))
    return {
        "train": set(clusters[:n_train]),
        "calibration": set(clusters[n_train : n_train + n_calibration]),
        "evaluation": set(clusters[n_train + n_calibration :]),
    }


def build_learners(seed: int) -> dict[str, object]:
    return {
        "ridge_vector": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees_vector": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=0.70,
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest_vector": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=0.70,
            random_state=seed,
            n_jobs=-1,
        ),
        "cqr_q80_vector": GradientBoostingRegressor(
            loss="quantile",
            alpha=0.80,
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=15,
            random_state=seed,
        ),
    }


def fixed_base(name: str, frame: pd.DataFrame) -> np.ndarray:
    if name == "constant":
        return np.zeros(len(frame), dtype=float)
    if name == "predicted_magnitude":
        return frame["predicted_magnitude"].to_numpy(dtype=float)
    if name == "magnitude_plus_lower":
        return np.maximum(
            frame["predicted_magnitude"].to_numpy(dtype=float),
            frame["pair_lower_bound"].to_numpy(dtype=float),
        )
    if name == "seed_spread":
        return frame["ensemble_seed_spread"].to_numpy(dtype=float)
    raise KeyError(name)


def evaluate_repeat(
    frame: pd.DataFrame, study: str, repeat_index: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    partitions = split_clusters(frame, seed)
    blocks = {
        name: frame[frame["cluster_id"].isin(clusters)].copy()
        for name, clusters in partitions.items()
    }
    train = blocks["train"]
    calibration = blocks["calibration"]
    evaluation = blocks["evaluation"]
    x_train = train.loc[:, FEATURES].to_numpy(dtype=float)
    x_calibration = calibration.loc[:, FEATURES].to_numpy(dtype=float)
    x_evaluation = evaluation.loc[:, FEATURES].to_numpy(dtype=float)

    calibration_bases: dict[str, np.ndarray] = {}
    evaluation_bases: dict[str, np.ndarray] = {}
    for method in ("constant", "predicted_magnitude", "magnitude_plus_lower", "seed_spread"):
        calibration_bases[method] = fixed_base(method, calibration)
        evaluation_bases[method] = fixed_base(method, evaluation)

    importance_rows: list[dict[str, object]] = []
    for method, learner in build_learners(seed).items():
        learner.fit(x_train, train["outcome"].to_numpy(dtype=float))
        calibration_bases[method] = np.maximum(
            calibration["pair_lower_bound"].to_numpy(dtype=float),
            learner.predict(x_calibration),
        )
        evaluation_bases[method] = np.maximum(
            evaluation["pair_lower_bound"].to_numpy(dtype=float),
            learner.predict(x_evaluation),
        )
        if method == PRIMARY_METHOD:
            for feature, importance in zip(FEATURES, learner.feature_importances_, strict=True):
                importance_rows.append(
                    {
                        "study": study,
                        "repeat_index": repeat_index,
                        "repeat_seed": seed,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    rows: list[dict[str, object]] = []
    for method in METHOD_LABELS:
        calibration_work = calibration[["cluster_id", "outcome"]].copy()
        calibration_work["residual"] = (
            calibration_work["outcome"].to_numpy(dtype=float) - calibration_bases[method]
        )
        cluster_residuals = calibration_work.groupby("cluster_id")["residual"].max()
        q, rank = conformal_quantile(cluster_residuals)
        upper = np.maximum(
            evaluation["pair_lower_bound"].to_numpy(dtype=float),
            evaluation_bases[method] + q,
        )
        task_covered = evaluation["outcome"].to_numpy(dtype=float) <= upper + EPS
        audit = evaluation[["cluster_id"]].copy()
        audit["covered"] = task_covered
        target_coverage = float(audit.groupby("cluster_id")["covered"].all().mean())
        rows.append(
            {
                "study": study,
                "repeat_index": repeat_index,
                "repeat_seed": seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_train_targets": len(partitions["train"]),
                "n_calibration_targets": len(partitions["calibration"]),
                "n_evaluation_targets": len(partitions["evaluation"]),
                "n_evaluation_tasks": len(evaluation),
                "conformal_rank_one_based": rank,
                "conformal_correction": q,
                "target_simultaneous_coverage": target_coverage,
                "task_marginal_coverage": float(np.mean(task_covered)),
                "mean_upper": float(np.mean(upper)),
                "median_upper": float(np.median(upper)),
                "mean_width_above_pair_lower": float(
                    np.mean(upper - evaluation["pair_lower_bound"].to_numpy(dtype=float))
                ),
                "median_pair_mean_error": float(np.median(evaluation["outcome"])),
            }
        )
    return rows, importance_rows


def summarize(repeats: pd.DataFrame) -> pd.DataFrame:
    grouped = repeats.groupby(["study", "method", "method_label"], sort=False)
    summary = grouped.agg(
        n_repeats=("repeat_index", "size"),
        mean_target_coverage=("target_simultaneous_coverage", "mean"),
        min_target_coverage=("target_simultaneous_coverage", "min"),
        p10_target_coverage=("target_simultaneous_coverage", lambda x: x.quantile(0.10)),
        mean_task_coverage=("task_marginal_coverage", "mean"),
        mean_upper=("mean_upper", "mean"),
        sd_upper=("mean_upper", "std"),
        mean_width_above_pair_lower=("mean_width_above_pair_lower", "mean"),
    ).reset_index()
    reference = summary[summary["method"] == REFERENCE_METHOD][
        ["study", "mean_upper"]
    ].rename(columns={"mean_upper": "reference_mean_upper"})
    summary = summary.merge(reference, on="study", how="left")
    summary["relative_upper_reduction_vs_constant"] = (
        summary["reference_mean_upper"] - summary["mean_upper"]
    ) / summary["reference_mean_upper"]
    return summary


def paired_reductions(repeats: pd.DataFrame) -> pd.DataFrame:
    reference = repeats[repeats["method"] == REFERENCE_METHOD][
        ["study", "repeat_index", "mean_upper"]
    ].rename(columns={"mean_upper": "constant_mean_upper"})
    result = repeats.merge(reference, on=["study", "repeat_index"], how="left")
    result["absolute_reduction_vs_constant"] = (
        result["constant_mean_upper"] - result["mean_upper"]
    )
    result["relative_reduction_vs_constant"] = (
        result["absolute_reduction_vs_constant"] / result["constant_mean_upper"]
    )
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in (".png", ".svg"):
        path = FIGURES / f"{stem}{suffix}"
        fig.savefig(path, dpi=240 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def figure_design() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "Whole-target split\n50% train"),
        (0.27, "Fit error base\n8 competing methods"),
        (0.52, "Whole-target calibration\n25%; max residual"),
        (0.77, "Whole-target evaluation\n25%; 90% coverage"),
    ]
    for x, label in boxes:
        ax.add_patch(
            plt.Rectangle((x, 0.32), 0.19, 0.35, facecolor="white", edgecolor=BLUE, linewidth=1.6)
        )
        ax.text(x + 0.095, 0.495, label, ha="center", va="center", linespacing=1.35)
    for x in (0.215, 0.465, 0.715):
        ax.annotate("", xy=(x + 0.045, 0.495), xytext=(x, 0.495), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5})
    ax.text(
        0.5,
        0.12,
        "One perturbation target never crosses partitions; all methods share all 50 repeated splits.",
        ha="center",
        color=GREY,
    )
    save_figure(fig, "F1_E179_NESTED_DESIGN")


def figure_efficiency(summary: pd.DataFrame) -> None:
    order = list(METHOD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=False)
    for ax, (study, block) in zip(axes, summary.groupby("study", sort=False), strict=True):
        block = block.set_index("method").loc[order].reset_index()
        colors = [TEAL if method == PRIMARY_METHOD else BLUE if method == REFERENCE_METHOD else LIGHT for method in block["method"]]
        ax.barh(np.arange(len(block)), block["mean_upper"], color=colors, edgecolor="white")
        ax.set_yticks(np.arange(len(block)), [METHOD_LABELS[m] for m in block["method"]])
        ax.invert_yaxis()
        ax.set_xlabel("Mean calibrated upper bound (RMSE)")
        ax.set_title(study.replace("_", " "))
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    fig.suptitle("Repeated nested evaluation: smaller is sharper", y=1.01, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "F2_E179_METHOD_EFFICIENCY")


def figure_paired_reduction(paired: pd.DataFrame) -> None:
    data = paired[paired["method"] == PRIMARY_METHOD]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    groups = [group["relative_reduction_vs_constant"].to_numpy() * 100 for _, group in data.groupby("study", sort=False)]
    labels = [name.replace("_", " ") for name, _ in data.groupby("study", sort=False)]
    violin = ax.violinplot(groups, showmeans=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(TEAL)
        body.set_edgecolor(TEAL)
        body.set_alpha(0.65)
    violin["cmeans"].set_color(RED)
    ax.axhline(0, color=GREY, linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Upper-bound reduction vs constant (%)")
    ax.set_title("Paired efficiency change on the same target splits", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    save_figure(fig, "F3_E179_PAIRED_REDUCTION")


def figure_coverage_efficiency(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, (study, block) in zip(axes, summary.groupby("study", sort=False), strict=True):
        for _, row in block.iterrows():
            color = TEAL if row["method"] == PRIMARY_METHOD else BLUE if row["method"] == REFERENCE_METHOD else GREY
            ax.scatter(row["mean_upper"], row["mean_target_coverage"], s=55, color=color, zorder=3)
            if row["method"] in (PRIMARY_METHOD, REFERENCE_METHOD, "cqr_q80_vector"):
                ax.annotate(row["method"].replace("_vector", ""), (row["mean_upper"], row["mean_target_coverage"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axhline(TARGET_COVERAGE, color=RED, linestyle="--", linewidth=1)
        ax.set_xlabel("Mean upper bound (RMSE)")
        ax.set_ylabel("Mean target-simultaneous coverage")
        ax.set_title(study.replace("_", " "))
        ax.grid(color="#E5E7EB", linewidth=0.7)
    fig.suptitle("Coverage–efficiency plane", y=1.01, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "F4_E179_COVERAGE_EFFICIENCY")


def figure_importance(importance: pd.DataFrame) -> None:
    ranked = (
        importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(ranked.index, ranked.values, color=TEAL)
    ax.set_xlabel("Mean ExtraTrees feature importance")
    ax.set_title("Which pretruth signals drive the adaptive base?", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    save_figure(fig, "F5_E179_FEATURE_IMPORTANCE")


def render_report(summary: pd.DataFrame, paired: pd.DataFrame, n_tasks: dict[str, int], n_targets: dict[str, int]) -> str:
    primary = summary[summary["method"] == PRIMARY_METHOD].set_index("study")
    lines = [
        "# E179：靶点级嵌套不确定性基线比较",
        "",
        "## 结论",
        "",
        "E179 把上界收窄问题变成了一个可复现的基线比较，而不是继续修改 SafeConf 排名分数。"
        "两个研究均按完整扰动靶点分组，重复 50 次 train/calibration/evaluation 划分；同一靶点的不同状态或技术组不会跨分区。",
        "",
    ]
    for study in primary.index:
        row = primary.loc[study]
        block = paired[(paired["study"] == study) & (paired["method"] == PRIMARY_METHOD)]
        q025, q975 = block["relative_reduction_vs_constant"].quantile([0.025, 0.975]) * 100
        lines.append(
            f"- **{study}**：{n_targets[study]} 个靶点、{n_tasks[study]} 个任务；"
            f"ExtraTrees 自适应基线的平均靶点同时覆盖率为 {row['mean_target_coverage']:.3f}，"
            f"平均上界为 {row['mean_upper']:.4f} RMSE，较常数 split conformal 平均缩短 "
            f"{100 * row['relative_upper_reduction_vs_constant']:.2f}%；"
            f"50 个配对划分的缩短率 2.5%–97.5% 分位区间为 {q025:.2f}%–{q975:.2f}%。"
        )
    lines.extend(
        [
            "",
            "这项收益不大，但方向一致且来自合法的 pretruth 特征：预测幅度、模型间距离、向量形状、方向一致性和五个随机种子的波动。"
            "E179 是方法开发证据，不把历史真值包装成新的外部确认。`extra_trees_vector` 现作为下一套新数据的冻结候选；"
            "真正的确认结论只允许来自冻结后的新数据。",
            "",
            "## 比较对象",
            "",
            "1. 常数 split conformal；",
            "2. 预测幅度加 conformal 修正；",
            "3. `max(预测幅度, 两模型距离/2)` 加 conformal 修正；",
            "4. 五随机种子波动加 conformal 修正；",
            "5. Ridge、ExtraTrees、随机森林和 0.80 分位数梯度提升，统一使用 18 个 pretruth 特征；",
            "6. 每一种学习方法都只在 train 靶点拟合，在 calibration 靶点上对“同一靶点所有任务的最大残差”取有限样本分位数，最后在 evaluation 靶点上检查同时覆盖。",
            "",
            "## 为什么按靶点分组",
            "",
            "一个基因在多个状态或技术组中会产生多条任务记录。随机拆任务会让同一基因同时出现在拟合和评价中，覆盖率会偏乐观。"
            "E179 始终移动完整靶点簇；评价事件是该靶点的全部任务都被上界覆盖。",
            "",
            "## 图",
            "",
            "![嵌套设计](../figures/F1_E179_NESTED_DESIGN.png)",
            "",
            "![方法效率](../figures/F2_E179_METHOD_EFFICIENCY.png)",
            "",
            "![配对缩短率](../figures/F3_E179_PAIRED_REDUCTION.png)",
            "",
            "![覆盖与效率](../figures/F4_E179_COVERAGE_EFFICIENCY.png)",
            "",
            "![特征重要性](../figures/F5_E179_FEATURE_IMPORTANCE.png)",
            "",
            "## 解释边界",
            "",
            "- 重复划分相互重叠，所以重复间分布只描述稳定性，不当作 50 个独立试验计算显著性。",
            "- E177 的 `technical_group` 仍只是技术组，不改称生物学背景。",
            "- ExtraTrees 的选择发生在已解封历史数据上；下一次确认必须先锁定代码、特征、超参数、分组单位和统计门槛，再读取新评价真值。",
            "- 确定性下界 `||p_scGPT-p_GEARS||/2` 保持独立：它不依赖校准真值，且只下界两模型平均/最大误差，不冒充任一单模型的置信度。",
            "",
            "## 可复现文件",
            "",
            "- `../tables/E179_REPEAT_RESULTS.csv`：每个研究、重复和方法的完整结果。",
            "- `../tables/E179_METHOD_SUMMARY.csv`：方法汇总。",
            "- `../tables/E179_PAIRED_REDUCTIONS.csv`：相同划分上的配对效率差。",
            "- `../tables/E179_PRIMARY_FEATURE_IMPORTANCE.csv`：ExtraTrees 特征重要性。",
            "- `../tables/E179_INPUT_HASHES.csv`：输入文件哈希。",
            "- `../../../../tools/scripts/run_e179_nested_uq_baseline_benchmark.py`：唯一运行脚本。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    for directory in (TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    e176, paths176 = load_e176()
    e177, paths177 = load_e177()
    studies = {
        "E176_primary_CD4": e176,
        "E177_Sunshine": e177,
    }
    for name, frame in studies.items():
        if frame[list(FEATURES) + ["outcome"]].isna().any().any():
            raise RuntimeError(f"{name}: non-finite feature or outcome")
        if (frame["pair_lower_bound"] > frame["outcome"] + 1e-9).any():
            raise RuntimeError(f"{name}: deterministic lower-bound violation")

    repeat_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    for study, frame in studies.items():
        for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
            rows, importance = evaluate_repeat(frame, study, repeat_index, seed)
            repeat_rows.extend(rows)
            importance_rows.extend(importance)
    repeats = pd.DataFrame(repeat_rows)
    importance = pd.DataFrame(importance_rows)
    summary = summarize(repeats)
    paired = paired_reductions(repeats)

    atomic_csv(TABLES / "E179_REPEAT_RESULTS.csv", repeats)
    atomic_csv(TABLES / "E179_METHOD_SUMMARY.csv", summary)
    atomic_csv(TABLES / "E179_PAIRED_REDUCTIONS.csv", paired)
    atomic_csv(TABLES / "E179_PRIMARY_FEATURE_IMPORTANCE.csv", importance)

    input_rows = []
    for path in sorted(set(paths176 + paths177 + [Path(__file__).resolve()])):
        input_rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(TABLES / "E179_INPUT_HASHES.csv", pd.DataFrame(input_rows))

    configure_plotting()
    figure_design()
    figure_efficiency(summary)
    figure_paired_reduction(paired)
    figure_coverage_efficiency(summary)
    figure_importance(importance)

    n_tasks = {study: len(frame) for study, frame in studies.items()}
    n_targets = {study: frame["cluster_id"].nunique() for study, frame in studies.items()}
    atomic_text(REPORTS / "E179_REPORT.md", render_report(summary, paired, n_tasks, n_targets))

    primary = summary[summary["method"] == PRIMARY_METHOD].set_index("study")
    status = {
        "experiment": "E179_nested_uq_baseline_benchmark",
        "status": "complete",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "purpose": "retrospective_method_development_not_prospective_confirmation",
        "target_coverage": TARGET_COVERAGE,
        "repeat_count": len(REPEAT_SEEDS),
        "whole_target_partitioning": True,
        "primary_candidate_for_next_frozen_confirmation": PRIMARY_METHOD,
        "studies": {
            study: {
                "n_targets": n_targets[study],
                "n_tasks": n_tasks[study],
                "primary_mean_target_coverage": float(primary.loc[study, "mean_target_coverage"]),
                "primary_mean_upper": float(primary.loc[study, "mean_upper"]),
                "relative_upper_reduction_vs_constant": float(
                    primary.loc[study, "relative_upper_reduction_vs_constant"]
                ),
            }
            for study in studies
        },
        "git_head_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
