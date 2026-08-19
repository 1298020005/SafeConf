#!/usr/bin/env python3
"""Open E182 evaluation truth once and produce the final family certificate."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
COMMON_PATH = ROOT / "tools/scripts/e182_registered_family_common.py"
OUT = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
RELEASE = OUT / "final_evaluation"
CALIBRATION = OUT / "calibration_release"
COMMITTED_INPUTS = (
    RUNNER,
    COMMON_PATH,
    ROOT / "tools/scripts/run_e180_xucao_pretruth.py",
    ROOT / "tools/scripts/build_e180_xucao_pretruth_assets.py",
    ROOT / "tools/scripts/run_e182_gse225807_calibration.py",
    ROOT / "tools/scripts/run_e182_gse225807_pretruth.py",
    ROOT / "tools/scripts/build_e182_gse225807_pretruth_assets.py",
    OUT / "SOURCE_LOCK.json",
    OUT / "MODEL_INPUT_LOCK.json",
    OUT / "STATISTICAL_ANALYSIS_LOCK.json",
    OUT / "PREREG_ANALYSIS_PLAN.md",
    OUT / "manifests/E182_SELECTED_TARGETS.csv",
    OUT / "manifests/E182_GUIDE_TASK_MANIFEST.csv",
    OUT / "pretruth_release/PRETRUTH_GATE_SNAPSHOT.json",
    OUT / "pretruth_release/tables/PRETRUTH_SCORING_INTERFACE.csv",
    OUT / "pretruth_release/tables/INPUT_HASHES.csv",
    OUT / "pretruth_release/arrays/PRETRUTH_PREDICTIONS.npz",
    CALIBRATION / "CALIBRATION_LOCK.json",
    CALIBRATION / "tables/CALIBRATION_TASK_ERRORS.csv",
    CALIBRATION / "tables/CALIBRATION_TARGET_CLUSTERS.csv",
    CALIBRATION / "tables/CALIBRATION_X_ACCESS_AUDIT.csv",
    CALIBRATION / "tables/INPUT_HASHES.csv",
)

BLUE = "#3A6EA5"
TEAL = "#2A8C82"
ORANGE = "#D97732"
RED = "#B84A4A"
GREY = "#6B7280"
LIGHT = "#E9EEF3"
INK = "#20262E"


def import_common() -> Any:
    spec = importlib.util.spec_from_file_location("e182_final_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def proportion_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )
    return [lower, upper]


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def method_figure(directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 2.7))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 2.7)
    ax.axis("off")
    boxes = [
        (0.2, "Frozen family\n5 scGPT + 5 GEARS", BLUE),
        (2.35, "Prediction geometry\nD, radius, diameter", TEAL),
        (4.65, "19 calibration targets\nmax over 2 guides", ORANGE),
        (6.95, "90% conformal\ncentroid upper U", ORANGE),
        (9.05, "Two-sided certificate\nD ≤ family error ≤ √(U²+D²)", BLUE),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.78),
            1.55,
            1.12,
            boxstyle="round,pad=0.08,rounding_size=0.07",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + 0.775, 1.34, label, ha="center", va="center", fontsize=8.5)
    for x in (1.83, 4.02, 6.32, 8.62):
        ax.annotate(
            "",
            xy=(x + 0.38, 1.34),
            xytext=(x, 1.34),
            arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.3},
        )
    ax.text(
        0.2,
        2.35,
        "Registered-family error certificate on a new CRISPRi study",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.2,
        0.28,
        "Evaluation target expression is opened once after the calibration lock is committed.",
        color=GREY,
    )
    save_figure(fig, directory, "F1_E182_METHOD")


def data_figures(
    tasks: pd.DataFrame,
    clusters: pd.DataFrame,
    access: pd.DataFrame,
    correction: float,
    f2_attestation: dict[str, Any],
    calibration_access_rows: int,
    directory: Path,
) -> None:
    ordered = clusters.sort_values("max_centroid_rmse").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    colors = np.where(ordered["centroid_simultaneous_covered"], TEAL, RED)
    ax.scatter(
        np.arange(1, len(ordered) + 1),
        ordered["max_centroid_rmse"],
        c=colors,
        s=34,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axhline(correction, color=ORANGE, lw=1.5, label=f"Frozen U = {correction:.3f}")
    ax.set_xlabel("Evaluation target, sorted by maximum guide error")
    ax.set_ylabel("Maximum centroid RMSE across two guides")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save_figure(fig, directory, "F2_E182_TARGET_COVERAGE")

    display = tasks.sort_values("family_rms_error").reset_index(drop=True)
    x = np.arange(len(display))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.vlines(
        x,
        display["family_diversity_lower"],
        display["family_rms_upper"],
        color=LIGHT,
        lw=2.2,
        label="Certificate interval",
    )
    ax.scatter(
        x,
        display["family_rms_error"],
        color=BLUE,
        s=18,
        label="Observed family RMS error",
        zorder=3,
    )
    ax.scatter(
        x,
        display["family_diversity_lower"],
        color=TEAL,
        s=12,
        label="Deterministic lower",
        zorder=3,
    )
    ax.set_xlabel("Evaluation guide task, sorted by observed error")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, directory, "F3_E182_TWO_SIDED_CERTIFICATES")

    family_tightness = (
        tasks["family_diversity_lower"] / tasks["family_rms_error"].clip(lower=1e-12)
    )
    worst_tightness = (
        tasks["worst_member_lower"] / tasks["worst_member_error"].clip(lower=1e-12)
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    parts = ax.violinplot(
        [family_tightness, worst_tightness],
        positions=[1, 2],
        widths=0.72,
        showmeans=False,
        showmedians=True,
    )
    if len(parts["bodies"]) != 2:
        raise RuntimeError("E182 lower-tightness plot expected two violin bodies")
    for body, color in zip(parts["bodies"], (TEAL, BLUE)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.72)
    parts["cmedians"].set_color(INK)
    ax.set_xticks([1, 2], ["Family RMS lower /\nobserved", "Worst-member lower /\nobserved"])
    ax.set_ylabel("Lower-bound tightness")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save_figure(fig, directory, "F4_E182_LOWER_TIGHTNESS")

    phase_counts = f2_attestation["logical_x_rows_read_by_phase"]
    labels = [
        "F2 control",
        "F2 train",
        "F2 validation",
        "F3 calibration",
        "F4 evaluation",
    ]
    values = [
        int(phase_counts.get("PRETRUTH_CONTROL_X", 0)),
        int(phase_counts.get("PRETRUTH_TRAIN_X", 0)),
        int(phase_counts.get("PRETRUTH_VALIDATION_X", 0)),
        int(calibration_access_rows),
        int(len(access)),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    bars = ax.barh(labels, values, color=[GREY, BLUE, BLUE, ORANGE, TEAL])
    ax.bar_label(bars, padding=4, fmt="%d")
    ax.set_xlabel("Expression rows opened in the registered phase")
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    save_figure(fig, directory, "F5_E182_ACCESS_AUDIT")


def main() -> None:
    common = import_common()
    if RELEASE.exists():
        raise common.IntegrityError("E182 final release is append-only")
    audit = common.require_committed(COMMITTED_INPUTS)
    scores, arrays, external_hashes = common.load_pretruth()
    calibration_lock = json.loads(
        (CALIBRATION / "CALIBRATION_LOCK.json").read_text()
    )
    if (
        calibration_lock.get("status") != "PASS"
        or calibration_lock.get("evaluation_target_x_rows_read") != 0
        or calibration_lock.get("learned_or_adaptive_upper_fitted") is not False
    ):
        raise common.IntegrityError("E182 calibration lock is invalid")
    correction = float(calibration_lock["constant_centroid_upper"])

    truth, access, source_hashes = common.read_split_truth(
        "prospective_evaluation"
    )
    tasks = common.evaluate_tasks(
        scores, arrays, truth, "prospective_evaluation"
    )
    tasks["centroid_covered"] = tasks["centroid_rmse"].le(correction + 1e-10)
    tasks["family_rms_upper"] = np.sqrt(
        correction**2 + np.square(tasks["family_diversity_lower"])
    )
    tasks["worst_member_upper"] = correction + tasks["family_radius"]
    tasks["family_upper_covered"] = tasks["family_rms_error"].le(
        tasks["family_rms_upper"] + 1e-10
    )
    tasks["worst_member_upper_covered"] = tasks["worst_member_error"].le(
        tasks["worst_member_upper"] + 1e-10
    )
    tasks["family_lower_tightness"] = (
        tasks["family_diversity_lower"]
        / tasks["family_rms_error"].clip(lower=1e-12)
    )
    tasks["worst_member_lower_tightness"] = (
        tasks["worst_member_lower"]
        / tasks["worst_member_error"].clip(lower=1e-12)
    )

    clusters = (
        tasks.groupby("perturbation", observed=True)
        .agg(
            n_guides=("guide_id", "nunique"),
            max_centroid_rmse=("centroid_rmse", "max"),
            centroid_simultaneous_covered=("centroid_covered", "all"),
            family_rms_simultaneous_covered=("family_upper_covered", "all"),
            worst_member_simultaneous_covered=("worst_member_upper_covered", "all"),
            mean_family_lower_tightness=("family_lower_tightness", "mean"),
            mean_worst_member_lower_tightness=(
                "worst_member_lower_tightness",
                "mean",
            ),
        )
        .reset_index()
    )
    if clusters["n_guides"].ne(2).any():
        raise common.IntegrityError("E182 evaluation target lost a guide")
    successes = int(clusters["centroid_simultaneous_covered"].sum())
    total = len(clusters)
    coverage = successes / total
    coverage_ci = proportion_interval(successes, total)

    lower_violations = int(tasks["family_lower_violation"].sum())
    worst_lower_violations = int(tasks["worst_member_lower_violation"].sum())
    identity_residual = float(tasks["hilbert_identity_residual"].abs().max())
    access_violations = int(
        (~access["target_split"].eq("prospective_evaluation")).sum()
    )
    gates = {
        "family_rms_lower_violations_zero": lower_violations == 0,
        "worst_member_lower_violations_zero": worst_lower_violations == 0,
        "hilbert_identity_residual_at_most_1e_10": identity_residual <= 1e-10,
        "target_simultaneous_coverage_at_least_0_85": coverage >= 0.85,
        "truth_access_violations_zero": access_violations == 0,
        "no_adaptive_upper_used": calibration_lock[
            "learned_or_adaptive_upper_fitted"
        ]
        is False,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    staging = OUT / f".final_evaluation.staging.{os.getpid()}"
    try:
        for subdirectory in ("tables", "arrays", "figures", "reports"):
            (staging / subdirectory).mkdir(parents=True, exist_ok=False)
        common.atomic_csv(staging / "tables/E182_EVALUATION_TASKS.csv", tasks)
        common.atomic_csv(staging / "tables/E182_EVALUATION_TARGETS.csv", clusters)
        common.atomic_csv(staging / "tables/E182_EVALUATION_X_ACCESS_AUDIT.csv", access)
        common.atomic_csv(
            staging / "tables/E182_FINAL_GATES.csv",
            pd.DataFrame(
                [{"gate": name, "passed": value} for name, value in gates.items()]
            ),
        )
        common.atomic_csv(
            staging / "tables/INPUT_HASHES.csv",
            pd.DataFrame(
                audit["input_hashes"] + external_hashes + source_hashes
            ),
        )
        common.atomic_npz(staging / "arrays/E182_EVALUATION_TRUTH.npz", truth)
        f2_attestation = json.loads(
            (common.F2_ROOT / "ACCESS_ATTESTATION.json").read_text()
        )
        calibration_access_rows = len(
            pd.read_csv(
                CALIBRATION / "tables/CALIBRATION_X_ACCESS_AUDIT.csv"
            )
        )
        configure_plots()
        method_figure(staging / "figures")
        data_figures(
            tasks,
            clusters,
            access,
            correction,
            f2_attestation,
            calibration_access_rows,
            staging / "figures",
        )
        summary = {
            "schema": "safeconf_e182_final_summary_v1",
            "status": status,
            "experiment": "E182_gse225807_registered_family",
            "study": "GSE225807",
            "git_head": audit["head"],
            "git_branch": audit["branch"],
            "remote_heads": audit["remote_heads"],
            "registered_family_size": 10,
            "n_evaluation_targets": total,
            "n_evaluation_guide_tasks": len(tasks),
            "constant_centroid_upper": correction,
            "target_simultaneous_coverage": coverage,
            "covered_targets": successes,
            "target_coverage_clopper_pearson_ci95": coverage_ci,
            "family_rms_lower_violations": lower_violations,
            "worst_member_lower_violations": worst_lower_violations,
            "max_hilbert_identity_absolute_residual": identity_residual,
            "median_family_lower_tightness": float(
                tasks["family_lower_tightness"].median()
            ),
            "median_worst_member_lower_tightness": float(
                tasks["worst_member_lower_tightness"].median()
            ),
            "family_upper_task_coverage": float(
                tasks["family_upper_covered"].mean()
            ),
            "worst_member_upper_task_coverage": float(
                tasks["worst_member_upper_covered"].mean()
            ),
            "evaluation_x_rows_read": len(access),
            "truth_access_violations": access_violations,
            "learned_or_adaptive_upper_used": False,
            "gates": gates,
        }
        common.atomic_json(staging / "E182_FINAL_SUMMARY.json", summary)
        report = f"""# E182 GSE225807 最终评价

## 结论

E182 在此前未进入项目结果的人源 K562 CRISPRi 研究 GSE225807 上，完成了靶基因级事前划分和一次性最终评价。最终评价包含 {total} 个新靶基因、{len(tasks)} 条 guide 任务。

- 10 模型家族 RMS 误差的确定性下界：**{lower_violations} 个违反**；
- 最坏家族成员误差的确定性下界：**{worst_lower_violations} 个违反**；
- Hilbert 恒等式最大绝对残差：`{identity_residual:.3e}`；
- 冻结质心上界：**{correction:.4f} RMSE**；
- 两条 guide 同时覆盖：**{successes}/{total} = {coverage:.1%}**，Clopper–Pearson 95% 区间 **[{coverage_ci[0]:.1%}, {coverage_ci[1]:.1%}]**；
- 家族 RMS 下界紧致度中位数：**{summary['median_family_lower_tightness']:.3f}**；
- 最坏成员下界紧致度中位数：**{summary['median_worst_member_lower_tightness']:.3f}**。

E182 没有训练或选择学习型上界。校准阶段只用 19 个靶基因冻结一个常数 target-cluster conformal 阈值；20 个评价靶基因的表达值在阈值提交并双远端留存后只打开一次。

## 图

![方法](../figures/F1_E182_METHOD.png)

![靶基因同时覆盖](../figures/F2_E182_TARGET_COVERAGE.png)

![双侧证书](../figures/F3_E182_TWO_SIDED_CERTIFICATES.png)

![下界紧致度](../figures/F4_E182_LOWER_TIGHTNESS.png)

![访问审计](../figures/F5_E182_ACCESS_AUDIT.png)
"""
        common.atomic_bytes(
            staging / "reports/E182_FINAL_REPORT.md", report.encode()
        )
        readme = """# E182 final evaluation

先读 `reports/E182_FINAL_REPORT.md`，机器可读结论在 `E182_FINAL_SUMMARY.json`。所有图同时提供 PNG 和 SVG；表格保留 task、target cluster、gate 和表达行访问记录。
"""
        common.atomic_bytes(staging / "README_先看这个.md", readme.encode())
        os.replace(staging, RELEASE)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
