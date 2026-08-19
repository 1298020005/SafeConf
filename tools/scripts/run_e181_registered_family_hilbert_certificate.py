#!/usr/bin/env python3
"""E181: retrospective registered-family Hilbert error certificate audit.

This audit consolidates frozen prediction vectors and already opened evaluation
truth from E176, E177, and E180.  It does not tune a predictor or claim a new
prospective experiment.  Its purpose is to test an exact, prediction-family
certificate and to carry the original calibrated centroid upper bounds into a
two-sided family-error envelope.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E181_registered_family_hilbert_certificate_20260724"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

E176 = ROOT / "docs/实验结果/E176_four_donor_fresh_confirmation_20260719"
E177 = ROOT / "docs/实验结果/E177_sunshine_external_certificate_20260719"
E180 = ROOT / "docs/实验结果/E180_xucao_fresh_guide_certificate_20260723"
E176_EXTERNAL = Path(
    "/home/yyf/data/safeconf_external/primary_cd4_perturbseq_2025/isolated/E176"
)

SEEDS = (3407, 3408, 3409, 3410, 3411)
BOOTSTRAP_SEED = 181024
N_BOOTSTRAP = 2000
TOLERANCE = 1e-10
SOURCE_REPRODUCTION_TOLERANCE = 5e-9

FAMILY_LABELS = {
    "architecture_pair": "2个架构均值",
    "frozen_10_seed_family": "冻结10模型家族",
    "nochange_negative_control": "加入不变预测（反例）",
}
STUDY_LABELS = {
    "E176_primary_CD4": "E176 四供体原代CD4",
    "E177_Sunshine": "E177 独立公开研究",
    "E180_XuCao": "E180 XuCao独立数据",
}

BLUE = "#3B6FB6"
TEAL = "#2A9D8F"
ORANGE = "#D9822B"
RED = "#C94C4C"
GREY = "#6B7280"
LIGHT_BLUE = "#EAF1F8"
LIGHT_TEAL = "#E7F4F1"
LIGHT_GREY = "#F2F4F7"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.12g"))


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_output_manifest() -> None:
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(OUT)}")
    atomic_text(OUT / "MANIFEST.sha256", "\n".join(entries) + "\n")


def rmse(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(vector))))


def configure_plotting() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = FIGURES / f"{stem}.svg"
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg = svg_path.read_text(encoding="utf-8")
    atomic_text(svg_path, "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    plt.close(figure)


def family_metrics(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predictions.ndim != 2 or truth.ndim != 1:
        raise ValueError("predictions must be M×G and truth must be G")
    if predictions.shape[1] != truth.shape[0]:
        raise ValueError("prediction and truth gene dimensions do not match")
    centroid = predictions.mean(axis=0)
    member_mse = np.mean(np.square(predictions - truth[None, :]), axis=1)
    centroid_mse = float(np.mean(np.square(centroid - truth)))
    diversity_mse = float(np.mean(np.square(predictions - centroid[None, :])))
    family_mean_mse = float(member_mse.mean())
    radius = max(rmse(member - centroid) for member in predictions)
    diameter = max(
        rmse(predictions[left] - predictions[right])
        for left in range(len(predictions))
        for right in range(left + 1, len(predictions))
    )
    family_rms_error = float(np.sqrt(family_mean_mse))
    diversity_lower = float(np.sqrt(diversity_mse))
    worst_member_error = float(np.sqrt(member_mse.max()))
    diameter_lower = float(diameter / 2.0)
    return {
        "n_members": int(len(predictions)),
        "family_rms_error": family_rms_error,
        "centroid_rmse": float(np.sqrt(centroid_mse)),
        "diversity_lower": diversity_lower,
        "worst_member_error": worst_member_error,
        "diameter_lower": diameter_lower,
        "family_radius": float(radius),
        "family_identity_abs_residual": float(
            abs(family_mean_mse - centroid_mse - diversity_mse)
        ),
        "family_lower_violation": bool(diversity_lower > family_rms_error + TOLERANCE),
        "worst_lower_violation": bool(diameter_lower > worst_member_error + TOLERANCE),
        "family_lower_tightness": (
            diversity_lower / family_rms_error if family_rms_error > 0 else np.nan
        ),
        "worst_lower_tightness": (
            diameter_lower / worst_member_error if worst_member_error > 0 else np.nan
        ),
    }


def registered_families(
    prediction_arrays: dict[str, np.ndarray], row: int
) -> dict[str, np.ndarray]:
    scgpt_mean = np.asarray(prediction_arrays["scGPT_seed_mean"][row], dtype=np.float64)
    gears_mean = np.asarray(prediction_arrays["GEARS_seed_mean"][row], dtype=np.float64)
    return {
        "architecture_pair": np.stack([scgpt_mean, gears_mean]),
        "frozen_10_seed_family": np.stack(
            [prediction_arrays[f"scGPT_seed{seed}"][row] for seed in SEEDS]
            + [prediction_arrays[f"GEARS_seed{seed}"][row] for seed in SEEDS]
        ),
        "nochange_negative_control": np.stack(
            [scgpt_mean, gears_mean, np.zeros_like(scgpt_mean)]
        ),
    }


def append_task_rows(
    rows: list[dict[str, object]],
    study: str,
    task_id: str,
    target_cluster: str,
    technical_context: str,
    truth: np.ndarray,
    upper_centroid: float,
    prediction_arrays: dict[str, np.ndarray],
    prediction_row: int,
) -> None:
    reference_centroid = (
        np.asarray(prediction_arrays["scGPT_seed_mean"][prediction_row], dtype=np.float64)
        + np.asarray(prediction_arrays["GEARS_seed_mean"][prediction_row], dtype=np.float64)
    ) / 2.0
    for family, predictions in registered_families(
        prediction_arrays, prediction_row
    ).items():
        metrics = family_metrics(predictions, truth)
        family_centroid = np.asarray(predictions, dtype=np.float64).mean(axis=0)
        centroid_reference_shift = rmse(family_centroid - reference_centroid)
        transported_centroid_upper = float(upper_centroid + centroid_reference_shift)
        family_upper = float(
            np.sqrt(
                np.square(transported_centroid_upper)
                + np.square(metrics["diversity_lower"])
            )
        )
        worst_upper = float(transported_centroid_upper + metrics["family_radius"])
        rows.append(
            {
                "study": study,
                "study_cn": STUDY_LABELS[study],
                "task_id": task_id,
                "target_cluster": target_cluster,
                "technical_context": technical_context,
                "family": family,
                "family_cn": FAMILY_LABELS[family],
                **metrics,
                "reference_centroid_upper": float(upper_centroid),
                "centroid_reference_shift": centroid_reference_shift,
                "transported_centroid_upper": transported_centroid_upper,
                "family_upper": family_upper,
                "worst_upper": worst_upper,
                "family_upper_covered": bool(
                    metrics["family_rms_error"] <= family_upper + TOLERANCE
                ),
                "worst_upper_covered": bool(
                    metrics["worst_member_error"] <= worst_upper + TOLERANCE
                ),
                "family_interval_width": float(family_upper - metrics["diversity_lower"]),
                "worst_interval_width": float(worst_upper - metrics["diameter_lower"]),
            }
        )


def load_e176() -> tuple[pd.DataFrame, list[Path], list[dict[str, object]]]:
    evaluation_path = E176 / "final_evaluation/tables/EVALUATION_TASK_METRICS.csv"
    evaluation = pd.read_csv(evaluation_path).set_index("task_id")
    rows: list[dict[str, object]] = []
    inputs = [evaluation_path]
    reproduction: list[dict[str, object]] = []
    for panel in ("H01", "H02", "H03", "H04"):
        release = E176 / f"pretruth_release/{panel}"
        interface_path = release / "tables/PRETRUTH_SCORING_INTERFACE.csv"
        prediction_path = release / "arrays/PRETRUTH_PREDICTIONS.npz"
        truth_path = E176_EXTERNAL / f"{panel}/F4_evaluation/EVALUATION_TARGET_EFFECTS.npz"
        interface = pd.read_csv(interface_path)
        row_lookup = dict(zip(interface["task_id"], interface.index, strict=True))
        panel_diffs: dict[str, list[float]] = {
            "scgpt_rmse": [],
            "gears_rmse": [],
            "centroid_rmse": [],
            "pair_lower": [],
        }
        with np.load(prediction_path, allow_pickle=False) as archive:
            predictions = {
                key: np.asarray(archive[key], dtype=np.float64)
                for key in (
                    *[f"scGPT_seed{seed}" for seed in SEEDS],
                    *[f"GEARS_seed{seed}" for seed in SEEDS],
                    "scGPT_seed_mean",
                    "GEARS_seed_mean",
                )
            }
        with np.load(truth_path, allow_pickle=False) as truths:
            for task_id in truths.files:
                if task_id not in row_lookup or task_id not in evaluation.index:
                    raise KeyError(f"E176 task missing from frozen interface or result: {task_id}")
                row_number = int(row_lookup[task_id])
                source = evaluation.loc[task_id]
                truth = np.asarray(truths[task_id], dtype=np.float64)
                scgpt = np.asarray(predictions["scGPT_seed_mean"][row_number], dtype=np.float64)
                gears = np.asarray(predictions["GEARS_seed_mean"][row_number], dtype=np.float64)
                panel_diffs["scgpt_rmse"].append(abs(rmse(scgpt - truth) - source["scgpt_rmse"]))
                panel_diffs["gears_rmse"].append(abs(rmse(gears - truth) - source["gears_rmse"]))
                panel_diffs["centroid_rmse"].append(
                    abs(rmse((scgpt + gears) / 2.0 - truth) - source["ensemble_rmse"])
                )
                panel_diffs["pair_lower"].append(
                    abs(rmse(scgpt - gears) / 2.0 - source["pair_lower_bound_rmse"])
                )
                append_task_rows(
                    rows=rows,
                    study="E176_primary_CD4",
                    task_id=task_id,
                    target_cluster=f"{panel}::{source['perturbed_gene_id']}",
                    technical_context=str(source["culture_condition"]),
                    truth=truth,
                    upper_centroid=float(
                        source["upper_ensemble_rmse__state_stratum_constant"]
                    ),
                    prediction_arrays=predictions,
                    prediction_row=row_number,
                )
        reproduction.append(
            {
                "study": "E176_primary_CD4",
                "partition": panel,
                "n_tasks": len(panel_diffs["scgpt_rmse"]),
                **{
                    f"max_abs_difference__{metric}": max(values)
                    for metric, values in panel_diffs.items()
                },
            }
        )
        inputs.extend([interface_path, prediction_path, truth_path])
    return pd.DataFrame(rows), inputs, reproduction


def load_single_release(
    study: str,
    experiment: Path,
    evaluation_path: Path,
    truth_path: Path,
    upper_column: str,
    target_column: str,
    context_column: str,
    source_pair_lower_column: str,
) -> tuple[pd.DataFrame, list[Path], list[dict[str, object]]]:
    interface_path = experiment / "pretruth_release/tables/PRETRUTH_SCORING_INTERFACE.csv"
    prediction_path = experiment / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz"
    interface = pd.read_csv(interface_path)
    evaluation = pd.read_csv(evaluation_path).set_index("task_id")
    row_lookup = dict(zip(interface["task_id"], interface.index, strict=True))
    rows: list[dict[str, object]] = []
    diffs: dict[str, list[float]] = {
        "scgpt_rmse": [],
        "gears_rmse": [],
        "pair_lower": [],
    }
    if "ensemble_rmse" in evaluation.columns:
        diffs["centroid_rmse"] = []
    with np.load(prediction_path, allow_pickle=False) as archive:
        predictions = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in (
                *[f"scGPT_seed{seed}" for seed in SEEDS],
                *[f"GEARS_seed{seed}" for seed in SEEDS],
                "scGPT_seed_mean",
                "GEARS_seed_mean",
            )
        }
    with np.load(truth_path, allow_pickle=False) as truths:
        for task_id in truths.files:
            if task_id not in row_lookup or task_id not in evaluation.index:
                raise KeyError(f"{study} task missing from frozen interface or result: {task_id}")
            row_number = int(row_lookup[task_id])
            source = evaluation.loc[task_id]
            truth = np.asarray(truths[task_id], dtype=np.float64)
            scgpt = np.asarray(predictions["scGPT_seed_mean"][row_number], dtype=np.float64)
            gears = np.asarray(predictions["GEARS_seed_mean"][row_number], dtype=np.float64)
            diffs["scgpt_rmse"].append(abs(rmse(scgpt - truth) - source["scgpt_rmse"]))
            diffs["gears_rmse"].append(abs(rmse(gears - truth) - source["gears_rmse"]))
            diffs["pair_lower"].append(
                abs(rmse(scgpt - gears) / 2.0 - source[source_pair_lower_column])
            )
            if "centroid_rmse" in diffs:
                diffs["centroid_rmse"].append(
                    abs(rmse((scgpt + gears) / 2.0 - truth) - source["ensemble_rmse"])
                )
            append_task_rows(
                rows=rows,
                study=study,
                task_id=task_id,
                target_cluster=str(source[target_column]),
                technical_context=str(source[context_column]),
                truth=truth,
                upper_centroid=float(source[upper_column]),
                prediction_arrays=predictions,
                prediction_row=row_number,
            )
    reproduction = [
        {
            "study": study,
            "partition": "all",
            "n_tasks": len(diffs["scgpt_rmse"]),
            **{
                f"max_abs_difference__{metric}": max(values)
                for metric, values in diffs.items()
            },
        }
    ]
    return (
        pd.DataFrame(rows),
        [interface_path, prediction_path, evaluation_path, truth_path],
        reproduction,
    )


def bootstrap_median_interval(
    frame: pd.DataFrame, value_column: str, seed: int
) -> tuple[float, float]:
    groups = {
        cluster: part[value_column].to_numpy(dtype=float)
        for cluster, part in frame.groupby("target_cluster", sort=True)
    }
    clusters = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([groups[cluster] for cluster in sample])
        draws[index] = np.median(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, ((study, family), part) in enumerate(
        tasks.groupby(["study", "family"], sort=False)
    ):
        family_ci = bootstrap_median_interval(
            part, "family_lower_tightness", BOOTSTRAP_SEED + index
        )
        worst_ci = bootstrap_median_interval(
            part, "worst_lower_tightness", BOOTSTRAP_SEED + 100 + index
        )
        cluster_covered_family = part.groupby("target_cluster")[
            "family_upper_covered"
        ].all()
        cluster_covered_worst = part.groupby("target_cluster")["worst_upper_covered"].all()
        rows.append(
            {
                "study": study,
                "study_cn": STUDY_LABELS[study],
                "family": family,
                "family_cn": FAMILY_LABELS[family],
                "n_tasks": len(part),
                "n_target_clusters": part["target_cluster"].nunique(),
                "family_lower_violations": int(part["family_lower_violation"].sum()),
                "worst_lower_violations": int(part["worst_lower_violation"].sum()),
                "family_upper_task_coverage": float(part["family_upper_covered"].mean()),
                "family_upper_target_simultaneous_coverage": float(
                    cluster_covered_family.mean()
                ),
                "worst_upper_task_coverage": float(part["worst_upper_covered"].mean()),
                "worst_upper_target_simultaneous_coverage": float(
                    cluster_covered_worst.mean()
                ),
                "median_family_rms_error": float(part["family_rms_error"].median()),
                "median_diversity_lower": float(part["diversity_lower"].median()),
                "median_family_upper": float(part["family_upper"].median()),
                "median_centroid_reference_shift": float(
                    part["centroid_reference_shift"].median()
                ),
                "max_centroid_reference_shift": float(
                    part["centroid_reference_shift"].max()
                ),
                "median_family_interval_width": float(
                    part["family_interval_width"].median()
                ),
                "median_family_lower_tightness": float(
                    part["family_lower_tightness"].median()
                ),
                "family_lower_tightness_boot_ci95_low": family_ci[0],
                "family_lower_tightness_boot_ci95_high": family_ci[1],
                "median_worst_lower_tightness": float(
                    part["worst_lower_tightness"].median()
                ),
                "worst_lower_tightness_boot_ci95_low": worst_ci[0],
                "worst_lower_tightness_boot_ci95_high": worst_ci[1],
                "max_family_identity_abs_residual": float(
                    part["family_identity_abs_residual"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study, study_frame in tasks.groupby("study", sort=False):
        pair = study_frame[study_frame["family"] == "architecture_pair"].set_index(
            "task_id"
        )
        for family in ("frozen_10_seed_family", "nochange_negative_control"):
            other = study_frame[study_frame["family"] == family].set_index("task_id")
            common = pair.index.intersection(other.index)
            diversity_difference = (
                other.loc[common, "diversity_lower"]
                - pair.loc[common, "diversity_lower"]
            )
            tightness_difference = (
                other.loc[common, "family_lower_tightness"]
                - pair.loc[common, "family_lower_tightness"]
            )
            rows.append(
                {
                    "study": study,
                    "study_cn": STUDY_LABELS[study],
                    "comparison_family": family,
                    "comparison_family_cn": FAMILY_LABELS[family],
                    "reference_family": "architecture_pair",
                    "n_paired_tasks": len(common),
                    "fraction_higher_diversity_lower": float(
                        (diversity_difference > 0).mean()
                    ),
                    "fraction_higher_family_tightness": float(
                        (tightness_difference > 0).mean()
                    ),
                    "median_diversity_lower_difference": float(
                        diversity_difference.median()
                    ),
                    "median_family_tightness_difference": float(
                        tightness_difference.median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def theorem_tests() -> pd.DataFrame:
    rng = np.random.default_rng(181)
    rows = []
    for members in (2, 3, 10, 25):
        predictions = rng.normal(size=(members, 127))
        truth = rng.normal(size=127)
        values = family_metrics(predictions, truth)
        rows.append(
            {
                "test": f"synthetic_hilbert_identity_M{members}",
                "n_members": members,
                "identity_abs_residual": values["family_identity_abs_residual"],
                "family_lower_valid": not values["family_lower_violation"],
                "worst_lower_valid": not values["worst_lower_violation"],
                "pass": bool(
                    values["family_identity_abs_residual"] <= TOLERANCE
                    and not values["family_lower_violation"]
                    and not values["worst_lower_violation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def figure_method() -> None:
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4.5)
    axis.axis("off")
    boxes = [
        (0.25, 1.35, 2.2, 1.6, "冻结预测家族", "scGPT × 5\nGEARS × 5", LIGHT_BLUE, BLUE),
        (3.05, 1.35, 2.35, 1.6, "真值不可见部分", "多样性 D\n直径 Δ / 2", LIGHT_TEAL, TEAL),
        (
            6.0,
            1.35,
            2.35,
            1.6,
            "校准并搬移",
            "参考质心上界 U\n新质心上界 U+s",
            LIGHT_GREY,
            GREY,
        ),
        (
            8.95,
            0.85,
            2.75,
            2.6,
            "双侧误差证书",
            "D ≤ 家族RMS\n≤ √((U+s)²+D²)\n\nΔ/2 ≤ 最坏模型\n≤ U+s+r",
            "#FFF7E8",
            ORANGE,
        ),
    ]
    for x, y, width, height, title, body, fill, edge in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height - 0.38, title, ha="center", va="center",
                  fontsize=12, weight="bold", color=edge)
        axis.text(x + width / 2, y + height / 2 - 0.22, body, ha="center", va="center",
                  fontsize=10.5, color="#263238", linespacing=1.5)
    for left, right in ((2.45, 3.05), (5.4, 6.0), (8.35, 8.95)):
        axis.add_patch(
            FancyArrowPatch(
                (left, 2.15),
                (right, 2.15),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=GREY,
            )
        )
    axis.text(
        0.25,
        4.05,
        "E181｜注册模型家族的 Hilbert 双侧误差证书",
        fontsize=16,
        weight="bold",
        color="#1F2937",
    )
    axis.text(
        0.25,
        3.62,
        "预测几何给出确定性下界；已有 conformal 上界通过可计算的质心距离严格搬移。",
        fontsize=10.5,
        color=GREY,
    )
    save_figure(figure, "F1_E181_METHOD")


def figure_envelopes(summary: pd.DataFrame) -> None:
    data = summary[summary["family"] == "frozen_10_seed_family"].copy()
    data = data.set_index("study").loc[list(STUDY_LABELS)].reset_index()
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(data))
    for position, (_, row) in zip(y, data.iterrows()):
        axis.plot(
            [row["median_diversity_lower"], row["median_family_upper"]],
            [position, position],
            color=GREY,
            linewidth=2.2,
            zorder=1,
        )
    axis.scatter(
        data["median_diversity_lower"], y, s=72, color=TEAL, label="确定性下界 D", zorder=3
    )
    axis.scatter(
        data["median_family_rms_error"], y, s=72, color=BLUE, label="实际家族RMS", zorder=3
    )
    axis.scatter(
        data["median_family_upper"], y, s=72, color=ORANGE, label="继承后的上界", zorder=3
    )
    axis.set_yticks(y, data["study_cn"])
    axis.invert_yaxis()
    axis.set_xlabel("任务级数值的中位数（RMSE）")
    axis.set_title("冻结10模型家族：下界、实际误差与上界", loc="left", weight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save_figure(figure, "F2_E181_TWO_SIDED_ENVELOPES")


def figure_tightness(tasks: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=True)
    families = list(FAMILY_LABELS)
    colors = [BLUE, TEAL, ORANGE]
    for axis, (study, label) in zip(axes, STUDY_LABELS.items()):
        frame = tasks[tasks["study"] == study]
        values = [
            frame.loc[frame["family"] == family, "family_lower_tightness"].to_numpy()
            for family in families
        ]
        box = axis.boxplot(
            values,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            medianprops={"color": "#111827", "linewidth": 1.5},
            whiskerprops={"color": GREY},
            capprops={"color": GREY},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("white")
        axis.set_xticks(range(1, 4), ["2架构", "10模型", "加不变\n反例"])
        axis.set_title(label, fontsize=11, weight="bold")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_ylim(0, 0.75)
    axes[0].set_ylabel("下界紧致度 D / 家族RMS误差")
    figure.suptitle(
        "冻结10模型家族在三套独立评估中均提高下界紧致度",
        x=0.06,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(figure, "F3_E181_LOWER_TIGHTNESS")


def figure_validity(summary: pd.DataFrame) -> None:
    primary = summary[summary["family"] == "frozen_10_seed_family"].copy()
    primary = primary.set_index("study").loc[list(STUDY_LABELS)].reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    y = np.arange(len(primary))
    axes[0].barh(y, primary["n_tasks"], color=BLUE, alpha=0.88)
    axes[0].set_yticks(y, primary["study_cn"])
    axes[0].invert_yaxis()
    for position, row in zip(y, primary.itertuples()):
        axes[0].text(
            row.n_tasks + max(primary["n_tasks"]) * 0.02,
            position,
            f"{row.n_tasks:,}个任务｜0次下界违反",
            va="center",
            fontsize=9,
        )
    axes[0].set_xlim(0, max(primary["n_tasks"]) * 1.42)
    axes[0].set_xlabel("评估任务数")
    axes[0].set_title("确定性下界审计", loc="left", weight="bold")
    residual = primary["max_family_identity_abs_residual"].to_numpy() * 1e16
    axes[1].barh(y, residual, color=TEAL, alpha=0.88)
    axes[1].set_yticks(y, primary["study_cn"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(1.1, residual.max() * 1.12))
    axes[1].set_xlabel("最大恒等式绝对残差（单位：1e-16）")
    axes[1].set_title("数值恒等式复核", loc="left", weight="bold")
    for position, value in zip(y, residual):
        axes[1].text(value + 0.025, position, f"{value:.3f}", va="center", fontsize=9)
    for axis in axes:
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    figure.suptitle(
        "2,393个任务：下界零违反，恒等式残差处于浮点误差量级",
        x=0.05,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.91])
    save_figure(figure, "F4_E181_VALIDITY_AUDIT")


def figure_composition(comparisons: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    studies = list(STUDY_LABELS)
    x = np.arange(len(studies))
    width = 0.34
    for offset, family, color, label in (
        (-width / 2, "frozen_10_seed_family", TEAL, "冻结10模型"),
        (width / 2, "nochange_negative_control", ORANGE, "加入不变预测反例"),
    ):
        part = comparisons[comparisons["comparison_family"] == family].set_index("study")
        values = [part.loc[study, "fraction_higher_family_tightness"] for study in studies]
        axes[0].bar(x + offset, values, width=width, color=color, label=label)
        differences = [
            part.loc[study, "median_family_tightness_difference"] for study in studies
        ]
        axes[1].bar(x + offset, differences, width=width, color=color, label=label)
    for axis in axes:
        axis.axhline(0, color="#9CA3AF", linewidth=0.9)
        axis.set_xticks(x, ["E176", "E177", "E180"])
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("相对2架构家族，提高紧致度的任务比例")
    axes[0].set_title("逐任务改善比例", loc="left", weight="bold")
    axes[1].set_ylabel("紧致度中位数变化")
    axes[1].set_title("改善幅度与反例", loc="left", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    figure.suptitle(
        "家族组成必须预先固定：真实种子家族稳定增强，随意加入基线并不稳定",
        x=0.05,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    figure.tight_layout(rect=[0, 0.1, 1, 0.9])
    save_figure(figure, "F5_E181_FAMILY_COMPOSITION_STRESS")


def write_report(
    tasks: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    reproduction: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    primary = summary[summary["family"] == "frozen_10_seed_family"].set_index("study")
    comparison = comparisons[
        comparisons["comparison_family"] == "frozen_10_seed_family"
    ].set_index("study")
    total_tasks = int(
        tasks[tasks["family"] == "frozen_10_seed_family"]["task_id"].nunique()
    )
    total_clusters = int(primary["n_target_clusters"].sum())
    maximum_residual = float(primary["max_family_identity_abs_residual"].max())
    maximum_centroid_shift = float(primary["max_centroid_reference_shift"].max())
    report = f"""# E181｜注册模型家族的 Hilbert 双侧误差证书

## 结论

E181 将 E176、E177 和 E180 的冻结预测重新放进同一个数学对象中审计。主分析使用预先保存的 10 个预测器：scGPT 五个种子和 GEARS 五个种子。三套评估合计 {total_tasks:,} 个任务、{total_clusters:,} 个靶点簇。

- 家族平均平方误差的确定性下界共出现 **0 次违反**；
- 最坏成员误差的直径下界共出现 **0 次违反**；
- Hilbert 分解的最大数值残差为 `{maximum_residual:.3e}`；
- 冻结 10 模型家族相对原参考质心的最大搬移仅 `{maximum_centroid_shift:.3e}` RMSE；
- 原实验的质心 conformal 上界可直接转换成家族 RMS 上界，不重新拟合、不改变校准分位数；
- 学习型自适应上界在 E180 失效后不再进入主方法，保留稳定的既有 conformal 上界。

## 方法

对一个在冻结阶段注册的模型家族 `F={{p₁,…,pₘ}}`，令 `p̄` 为所有预测向量的均值，`y` 为实验真值。基因维度上的 RMSE 范数记为 `||·||`：

`R_F² = (1/M) Σᵢ ||pᵢ-y||²`

`D_F² = (1/M) Σᵢ ||pᵢ-p̄||²`

Hilbert 空间的平方范数分解给出：

`R_F² = ||p̄-y||² + D_F²`

因此，在不读取目标任务真值时，`D_F ≤ R_F` 恒成立。若校准阶段已经得到参考质心误差上界 `U`，正式家族质心与参考质心相同，则同一覆盖事件下：

`D_F ≤ R_F ≤ √(U²+D_F²)`

更一般地，令正式家族质心与参考质心之间的可计算距离为 `s`，三角不等式给出：

`D_F ≤ R_F ≤ √((U+s)²+D_F²)`

对最坏成员，设 `Δ=maxᵢⱼ||pᵢ-pⱼ||`，`r=maxᵢ||pᵢ-p̄||`，则：

`Δ/2 ≤ maxᵢ||pᵢ-y|| ≤ U+s+r`

下界是确定性的；上界继承原 conformal 参考质心上界的覆盖事件。冻结 10 模型家族的质心与两个架构种子均值的质心只有浮点误差，`s` 近似为零；反例家族的质心发生变化，必须支付明确的搬移代价。

## 三套独立评估

| 数据 | 任务 | 靶点簇 | 家族下界违反 | 靶点同时上界覆盖 | 下界紧致度中位数（95%簇自助法区间） |
|---|---:|---:|---:|---:|---:|
| E176 四供体原代CD4 | {int(primary.loc['E176_primary_CD4','n_tasks']):,} | {int(primary.loc['E176_primary_CD4','n_target_clusters']):,} | {int(primary.loc['E176_primary_CD4','family_lower_violations'])} | {primary.loc['E176_primary_CD4','family_upper_target_simultaneous_coverage']:.1%} | {primary.loc['E176_primary_CD4','median_family_lower_tightness']:.3f} [{primary.loc['E176_primary_CD4','family_lower_tightness_boot_ci95_low']:.3f}, {primary.loc['E176_primary_CD4','family_lower_tightness_boot_ci95_high']:.3f}] |
| E177 独立公开研究 | {int(primary.loc['E177_Sunshine','n_tasks']):,} | {int(primary.loc['E177_Sunshine','n_target_clusters']):,} | {int(primary.loc['E177_Sunshine','family_lower_violations'])} | {primary.loc['E177_Sunshine','family_upper_target_simultaneous_coverage']:.1%} | {primary.loc['E177_Sunshine','median_family_lower_tightness']:.3f} [{primary.loc['E177_Sunshine','family_lower_tightness_boot_ci95_low']:.3f}, {primary.loc['E177_Sunshine','family_lower_tightness_boot_ci95_high']:.3f}] |
| E180 XuCao独立数据 | {int(primary.loc['E180_XuCao','n_tasks']):,} | {int(primary.loc['E180_XuCao','n_target_clusters']):,} | {int(primary.loc['E180_XuCao','family_lower_violations'])} | {primary.loc['E180_XuCao','family_upper_target_simultaneous_coverage']:.1%} | {primary.loc['E180_XuCao','median_family_lower_tightness']:.3f} [{primary.loc['E180_XuCao','family_lower_tightness_boot_ci95_low']:.3f}, {primary.loc['E180_XuCao','family_lower_tightness_boot_ci95_high']:.3f}] |

冻结 10 模型家族相对两个架构均值，在 E176、E177、E180 中均有 100% 的任务获得更高下界紧致度；紧致度中位数分别增加 {comparison.loc['E176_primary_CD4','median_family_tightness_difference']:.3f}、{comparison.loc['E177_Sunshine','median_family_tightness_difference']:.3f} 和 {comparison.loc['E180_XuCao','median_family_tightness_difference']:.3f}。

## 反例与边界

把“预测不发生变化”的零向量临时加入家族，有时会提高下界，有时会降低紧致度；E180 中所有任务都没有改善。它还会改变家族质心，使上界增加搬移代价 `s`。这说明家族证书不能靠事后加入任意模型来包装结果。正式方法必须在读取评估真值前固定模型成员、种子、基因面板和聚合规则。

E181 是已经打开真值后的方法整合实验。它证明代数恒等式、复核数据索引并量化紧致度，不替代新的前瞻性确认。它也不恢复已经失败的 SafeConf 排序主张；任务排序只保留为诊断分析。

## 可复现文件

- `tables/E181_TASK_CERTIFICATES.csv`：每个任务、每个家族的完整双侧证书；
- `tables/E181_DATASET_SUMMARY.csv`：数据集级覆盖、紧致度和恒等式审计；
- `tables/E181_FAMILY_COMPARISONS.csv`：家族组成的配对比较；
- `tables/E181_SOURCE_REPRODUCTION_AUDIT.csv`：原实验汇总与向量重算的一致性；
- `tables/E181_THEOREM_TESTS.csv`：合成向量单元测试；
- `tables/INPUT_HASHES.csv`：代码、预测、真值和原上界文件的 SHA-256。

## 状态

`PASS`：三套数据的确定性下界均为零违反，恒等式残差低于 `{TOLERANCE:.0e}`，原始结果重算与 CSV 舍入值的差异低于 `{SOURCE_REPRODUCTION_TOLERANCE:.0e}`，全部合成测试通过。这个状态只表示 E181 的预设数值审计通过，不代表期刊录用结论。
"""
    atomic_text(REPORTS / "E181_REPORT.md", report)


def write_overview() -> None:
    overview = """# E181｜先看这个

E181 是 E176、E177、E180 已冻结预测的跨研究方法整合，不是新的前瞻性实验。

主结果：scGPT 五个种子与 GEARS 五个种子组成的注册家族，在 2,393 个任务上出现 0 次家族 RMS 下界违反和 0 次最坏成员下界违反。原 conformal 参考质心上界通过可计算的质心距离搬移，得到严格的双侧误差证书。

入口：

- 完整说明：`reports/E181_REPORT.md`
- 方法图：`figures/F1_E181_METHOD.svg`
- 数据集汇总：`tables/E181_DATASET_SUMMARY.csv`
- 逐任务证书：`tables/E181_TASK_CERTIFICATES.csv`
- 原结果复算审计：`tables/E181_SOURCE_REPRODUCTION_AUDIT.csv`
- 输出完整性：`MANIFEST.sha256`

边界：E181 使用的是已经打开的评估真值，作用是理论整合、索引复核与定量审计。SafeConf 排序没有因本实验恢复为主张；E180 中失败的学习型上界也没有进入正式证书。
"""
    atomic_text(OUT / "README_先看这个.md", overview)


def main() -> None:
    for directory in (OUT, TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    e176, inputs176, reproduction176 = load_e176()
    e177, inputs177, reproduction177 = load_single_release(
        study="E177_Sunshine",
        experiment=E177,
        evaluation_path=E177 / "final_evaluation/tables/EVALUATION_TASK_RESULTS.csv",
        truth_path=E177 / "final_evaluation/arrays/EVALUATION_TRUE_EFFECTS.npz",
        upper_column="ensemble_upper_bound",
        target_column="perturbation",
        context_column="technical_group",
        source_pair_lower_column="pair_lower_bound_rmse",
    )
    e180, inputs180, reproduction180 = load_single_release(
        study="E180_XuCao",
        experiment=E180,
        evaluation_path=E180 / "final_evaluation/tables/EVALUATION_TASK_RESULTS.csv",
        truth_path=E180 / "final_evaluation/arrays/EVALUATION_TRUE_EFFECTS.npz",
        upper_column="upper__constant",
        target_column="perturbation",
        context_column="guide_id",
        source_pair_lower_column="pair_lower_bound",
    )
    tasks = pd.concat([e176, e177, e180], ignore_index=True)
    summary = summarize_tasks(tasks)
    comparisons = paired_comparisons(tasks)
    reproduction = pd.DataFrame(
        reproduction176 + reproduction177 + reproduction180
    )
    tests = theorem_tests()

    expected_task_counts = {
        "E176_primary_CD4": 1920,
        "E177_Sunshine": 400,
        "E180_XuCao": 73,
    }
    primary = tasks[tasks["family"] == "frozen_10_seed_family"]
    observed_task_counts = primary.groupby("study")["task_id"].nunique().to_dict()
    if observed_task_counts != expected_task_counts:
        raise AssertionError(
            f"unexpected task counts: {observed_task_counts} != {expected_task_counts}"
        )
    if tasks["family_lower_violation"].any() or tasks["worst_lower_violation"].any():
        raise AssertionError("deterministic lower-bound violation detected")
    if tasks["family_identity_abs_residual"].max() > TOLERANCE:
        raise AssertionError("Hilbert identity residual exceeded tolerance")
    reproduction_numeric = reproduction.select_dtypes(include=[np.number]).drop(
        columns=["n_tasks"]
    )
    if np.nanmax(reproduction_numeric.to_numpy()) > SOURCE_REPRODUCTION_TOLERANCE:
        raise AssertionError("source reproduction mismatch exceeded tolerance")
    if not tests["pass"].all():
        raise AssertionError("synthetic theorem test failed")
    ten_comparisons = comparisons[
        comparisons["comparison_family"] == "frozen_10_seed_family"
    ]
    if not np.allclose(ten_comparisons["fraction_higher_family_tightness"], 1.0):
        raise AssertionError("registered 10-model family did not improve every task")

    atomic_csv(TABLES / "E181_TASK_CERTIFICATES.csv", tasks)
    atomic_csv(TABLES / "E181_DATASET_SUMMARY.csv", summary)
    atomic_csv(TABLES / "E181_FAMILY_COMPARISONS.csv", comparisons)
    atomic_csv(TABLES / "E181_SOURCE_REPRODUCTION_AUDIT.csv", reproduction)
    atomic_csv(TABLES / "E181_THEOREM_TESTS.csv", tests)

    all_inputs = [Path(__file__).resolve(), *inputs176, *inputs177, *inputs180]
    hashes = pd.DataFrame(
        [
            {
                "role": path.name,
                "path": (
                    str(path.relative_to(ROOT))
                    if path.is_relative_to(ROOT)
                    else str(path)
                ),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in dict.fromkeys(all_inputs)
        ]
    )
    atomic_csv(TABLES / "INPUT_HASHES.csv", hashes)

    figure_method()
    figure_envelopes(summary)
    figure_tightness(tasks)
    figure_validity(summary)
    figure_composition(comparisons)
    write_report(tasks, summary, comparisons, reproduction, tests)
    write_overview()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "UNKNOWN"
    status = {
        "experiment": "E181_registered_family_hilbert_certificate",
        "analysis_type": "retrospective_method_consolidation",
        "status": "PASS",
        "git_commit_at_run": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "n_evaluation_tasks": int(primary["task_id"].nunique()),
        "n_target_clusters": int(
            summary[summary["family"] == "frozen_10_seed_family"][
                "n_target_clusters"
            ].sum()
        ),
        "family_lower_violations": int(primary["family_lower_violation"].sum()),
        "worst_lower_violations": int(primary["worst_lower_violation"].sum()),
        "max_identity_abs_residual": float(
            primary["family_identity_abs_residual"].max()
        ),
        "source_reproduction_tolerance": SOURCE_REPRODUCTION_TOLERANCE,
        "registered_family": [
            *[f"scGPT_seed{seed}" for seed in SEEDS],
            *[f"GEARS_seed{seed}" for seed in SEEDS],
        ],
        "adaptive_upper_model_in_primary_method": False,
        "notes": [
            "Evaluation truth had already been opened before E181.",
            "The original calibrated reference-centroid upper bound is transported without refitting.",
            "The no-change member is a negative-control composition stress test only.",
        ],
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    write_output_manifest()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
