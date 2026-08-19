#!/usr/bin/env python3
"""Synthesize all frozen-family evaluation certificates through E182."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
E181 = (
    ROOT
    / "docs/实验结果/E181_registered_family_hilbert_certificate_20260724"
)
E182 = (
    ROOT
    / "docs/实验结果/E182_gse225807_registered_family_20260724"
    / "final_evaluation"
)
E181_TASKS = E181 / "tables/E181_TASK_CERTIFICATES.csv"
E182_TASKS = E182 / "tables/E182_EVALUATION_TASKS.csv"
E182_TARGETS = E182 / "tables/E182_EVALUATION_TARGETS.csv"
E182_SUMMARY = E182 / "E182_FINAL_SUMMARY.json"
E182_GATES = E182 / "tables/E182_FINAL_GATES.csv"
OUT = ROOT / "docs/实验结果/E183_all_study_family_synthesis_20260724"

BLUE = "#3A6EA5"
TEAL = "#2A8C82"
ORANGE = "#D97732"
RED = "#B84A4A"
GREY = "#6B7280"
LIGHT = "#E9EEF3"
INK = "#20262E"


class IntegrityError(RuntimeError):
    """A frozen input or a registered-family identity changed."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def exact_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return lower, upper


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9,
            "axes.unicode_minus": False,
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


def load_tasks() -> pd.DataFrame:
    prior = pd.read_csv(E181_TASKS, keep_default_na=False)
    prior = prior[prior["family"].eq("frozen_10_seed_family")].copy()
    if (
        len(prior) != 2393
        or prior.groupby(["study", "target_cluster"], observed=True).ngroups != 717
    ):
        raise IntegrityError("E181 frozen-family task population changed")

    new = pd.read_csv(E182_TASKS, keep_default_na=False)
    summary = json.loads(E182_SUMMARY.read_text())
    gates = pd.read_csv(E182_GATES, keep_default_na=False)
    if (
        len(new) != 40
        or summary.get("status") != "FAIL"
        or int(summary.get("covered_targets", -1)) != 16
    ):
        raise IntegrityError("E182 one-shot result changed")
    gate_passed = gates["passed"].astype(str).str.strip().str.lower().eq("true")
    failed = set(gates.loc[~gate_passed, "gate"])
    if failed != {"target_simultaneous_coverage_at_least_0_85"}:
        raise IntegrityError(f"E182 failure set changed: {sorted(failed)}")

    mapped = pd.DataFrame(
        {
            "study": "E182_GSE225807",
            "study_cn": "E182 独立RBP CRISPRi",
            "task_id": new["task_id"],
            "target_cluster": new["perturbation"],
            "technical_context": new["guide_id"],
            "family": "frozen_10_seed_family",
            "family_cn": "冻结10模型家族",
            "n_members": 10,
            "family_rms_error": new["family_rms_error"],
            "centroid_rmse": new["centroid_rmse"],
            "diversity_lower": new["family_diversity_lower"],
            "worst_member_error": new["worst_member_error"],
            "diameter_lower": new["worst_member_lower"],
            "family_radius": new["family_radius"],
            "family_identity_abs_residual": new[
                "hilbert_identity_residual"
            ].abs(),
            "family_lower_violation": new["family_lower_violation"],
            "worst_lower_violation": new["worst_member_lower_violation"],
            "family_lower_tightness": new["family_lower_tightness"],
            "worst_lower_tightness": new["worst_member_lower_tightness"],
            "reference_centroid_upper": summary["constant_centroid_upper"],
            "centroid_reference_shift": 0.0,
            "transported_centroid_upper": summary["constant_centroid_upper"],
            "family_upper": new["family_rms_upper"],
            "worst_upper": new["worst_member_upper"],
            "family_upper_covered": new["family_upper_covered"],
            "worst_upper_covered": new["worst_member_upper_covered"],
            "family_interval_width": (
                new["family_rms_upper"] - new["family_diversity_lower"]
            ),
            "worst_interval_width": (
                new["worst_member_upper"] - new["worst_member_lower"]
            ),
        }
    )
    combined = pd.concat([prior, mapped], ignore_index=True)
    if len(combined) != 2433:
        raise IntegrityError("E183 combined task count changed")
    if combined["family_lower_violation"].astype(bool).any():
        raise IntegrityError("E183 family lower certificate has a violation")
    if combined["worst_lower_violation"].astype(bool).any():
        raise IntegrityError("E183 worst-member lower certificate has a violation")
    if combined["family_identity_abs_residual"].max() > 1e-10:
        raise IntegrityError("E183 Hilbert identity numeric gate failed")
    return combined


def summarize_targets(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = (
        tasks.groupby(["study", "study_cn", "target_cluster"], observed=True)
        .agg(
            n_tasks=("task_id", "size"),
            family_upper_simultaneous_covered=("family_upper_covered", "all"),
            worst_upper_simultaneous_covered=("worst_upper_covered", "all"),
            mean_family_lower_tightness=("family_lower_tightness", "mean"),
            mean_worst_lower_tightness=("worst_lower_tightness", "mean"),
        )
        .reset_index()
    )
    expected = {
        "E176_primary_CD4": 640,
        "E177_Sunshine": 50,
        "E180_XuCao": 27,
        "E182_GSE225807": 20,
    }
    if targets.groupby("study").size().to_dict() != expected:
        raise IntegrityError("E183 target-cluster counts changed")

    rows: list[dict[str, object]] = []
    for (study, study_cn), block in targets.groupby(
        ["study", "study_cn"], sort=False, observed=True
    ):
        n_targets = len(block)
        family_covered = int(block["family_upper_simultaneous_covered"].sum())
        worst_covered = int(block["worst_upper_simultaneous_covered"].sum())
        family_ci = exact_interval(family_covered, n_targets)
        worst_ci = exact_interval(worst_covered, n_targets)
        task_block = tasks[tasks["study"].eq(study)]
        rows.append(
            {
                "study": study,
                "study_cn": study_cn,
                "n_tasks": len(task_block),
                "n_target_clusters": n_targets,
                "family_lower_violations": int(
                    task_block["family_lower_violation"].sum()
                ),
                "worst_lower_violations": int(
                    task_block["worst_lower_violation"].sum()
                ),
                "family_upper_tasks_covered": int(
                    task_block["family_upper_covered"].sum()
                ),
                "family_upper_task_coverage": float(
                    task_block["family_upper_covered"].mean()
                ),
                "family_upper_targets_covered": family_covered,
                "family_upper_target_coverage": family_covered / n_targets,
                "family_upper_target_ci95_low": family_ci[0],
                "family_upper_target_ci95_high": family_ci[1],
                "worst_upper_targets_covered": worst_covered,
                "worst_upper_target_coverage": worst_covered / n_targets,
                "worst_upper_target_ci95_low": worst_ci[0],
                "worst_upper_target_ci95_high": worst_ci[1],
                "median_family_lower_tightness": float(
                    task_block["family_lower_tightness"].median()
                ),
                "median_worst_lower_tightness": float(
                    task_block["worst_lower_tightness"].median()
                ),
                "max_identity_abs_residual": float(
                    task_block["family_identity_abs_residual"].max()
                ),
            }
        )
    studies = pd.DataFrame(rows)
    return targets, studies


def make_figures(
    tasks: pd.DataFrame,
    studies: pd.DataFrame,
    pooled_coverage: float,
    directory: Path,
) -> None:
    configure_plots()
    display_names = {
        "E176_primary_CD4": "E176 primary CD4",
        "E177_Sunshine": "E177 Sunshine",
        "E180_XuCao": "E180 XuCao",
        "E182_GSE225807": "E182 GSE225807",
    }
    labels = studies["study"].map(display_names).tolist() + [
        "Pooled (descriptive)"
    ]
    coverage = studies["family_upper_target_coverage"].tolist() + [pooled_coverage]
    lows = studies["family_upper_target_ci95_low"].tolist()
    highs = studies["family_upper_target_ci95_high"].tolist()
    pooled_n = int(studies["n_target_clusters"].sum())
    pooled_k = int(studies["family_upper_targets_covered"].sum())
    pooled_ci = exact_interval(pooled_k, pooled_n)
    lows.append(pooled_ci[0])
    highs.append(pooled_ci[1])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [BLUE, BLUE, BLUE, RED, TEAL]
    ax.errorbar(
        coverage,
        y,
        xerr=[
            np.asarray(coverage) - np.asarray(lows),
            np.asarray(highs) - np.asarray(coverage),
        ],
        fmt="none",
        ecolor=GREY,
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(coverage, y, c=colors, s=46, zorder=2)
    ax.axvline(0.9, color=ORANGE, lw=1.3, ls="--", label="Nominal 90%")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Target-simultaneous family-upper coverage")
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, directory, "F1_E183_CROSS_STUDY_TARGET_COVERAGE")

    study_order = studies["study"].tolist()
    values = [
        tasks.loc[tasks["study"].eq(study), "family_lower_tightness"].to_numpy()
        for study in study_order
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    boxes = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": INK, "lw": 1.2},
    )
    for patch, color in zip(boxes["boxes"], (BLUE, TEAL, ORANGE, RED)):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor(color)
    ax.set_xticks(
        np.arange(1, 5),
        studies["study"].map(display_names),
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("Deterministic lower / observed family RMS error")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save_figure(fig, directory, "F2_E183_LOWER_TIGHTNESS_BY_STUDY")

    support = np.arange(0, 21)
    probability = betabinom.pmf(support, 20, 18, 2)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = np.where(support <= 16, RED, BLUE)
    ax.bar(support, probability, color=colors, width=0.78)
    ax.axvline(16.5, color=GREY, lw=1.0, ls="--")
    ax.text(
        16.25,
        probability.max() * 0.92,
        "P(K ≤ 16) = 18.7%",
        ha="right",
        color=RED,
        fontweight="bold",
    )
    ax.set_xlim(7.5, 20.7)
    ax.set_xticks(np.arange(8, 21, 2))
    ax.set_xlabel("Covered targets among 20 future targets")
    ax.set_ylabel("Beta-binomial reference probability")
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    save_figure(fig, directory, "F3_E183_FINITE_CALIBRATION_VARIATION")


def main() -> None:
    if OUT.exists():
        raise IntegrityError(f"append-only E183 output exists: {OUT}")
    inputs = (RUNNER, E181_TASKS, E182_TASKS, E182_TARGETS, E182_SUMMARY, E182_GATES)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E183 inputs missing: {missing}")

    tasks = load_tasks()
    targets, studies = summarize_targets(tasks)
    total_targets = int(studies["n_target_clusters"].sum())
    family_targets_covered = int(studies["family_upper_targets_covered"].sum())
    worst_targets_covered = int(studies["worst_upper_targets_covered"].sum())
    total_tasks = len(tasks)
    family_tasks_covered = int(tasks["family_upper_covered"].sum())
    pooled_coverage = family_targets_covered / total_targets
    pooled_ci = exact_interval(family_targets_covered, total_targets)
    beta_binomial_tail = float(betabinom.cdf(16, 20, 18, 2))

    for subdirectory in ("tables", "figures", "reports"):
        (OUT / subdirectory).mkdir(parents=True, exist_ok=False)
    atomic_csv(OUT / "tables/E183_COMBINED_TASK_CERTIFICATES.csv", tasks)
    atomic_csv(OUT / "tables/E183_TARGET_CERTIFICATES.csv", targets)
    atomic_csv(OUT / "tables/E183_STUDY_SUMMARY.csv", studies)
    make_figures(tasks, studies, pooled_coverage, OUT / "figures")

    input_hashes = pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
        ]
    )
    atomic_csv(OUT / "tables/INPUT_HASHES.csv", input_hashes)
    summary = {
        "schema": "safeconf_e183_all_study_family_synthesis_v1",
        "status": "PASS",
        "analysis_type": "retrospective_cross_study_synthesis_after_e182",
        "git_head_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "n_studies": 4,
        "n_evaluation_tasks": total_tasks,
        "n_target_clusters": total_targets,
        "family_lower_violations": int(tasks["family_lower_violation"].sum()),
        "worst_lower_violations": int(tasks["worst_lower_violation"].sum()),
        "max_hilbert_identity_absolute_residual": float(
            tasks["family_identity_abs_residual"].max()
        ),
        "family_upper_tasks_covered": family_tasks_covered,
        "family_upper_task_coverage": family_tasks_covered / total_tasks,
        "family_upper_targets_covered": family_targets_covered,
        "family_upper_target_coverage": pooled_coverage,
        "family_upper_target_descriptive_ci95": list(pooled_ci),
        "worst_upper_targets_covered": worst_targets_covered,
        "worst_upper_target_coverage": worst_targets_covered / total_targets,
        "e182_standalone_preregistered_status": "FAIL",
        "e182_covered_targets": 16,
        "e182_total_targets": 20,
        "e182_beta_binomial_reference_probability_k_at_most_16": beta_binomial_tail,
        "beta_binomial_reference_parameters": {
            "calibration_targets": 19,
            "selected_order_rank": 18,
            "future_targets": 20,
            "alpha": 18,
            "beta": 2,
        },
        "interpretation_limits": [
            "E183 was run after E182 truth was opened and is not a new prospective test.",
            "The pooled confidence interval is descriptive because study target units are heterogeneous.",
            "E182 remains FAIL against its registered empirical gate; no threshold or model was changed.",
        ],
    }
    atomic_json(OUT / "RUN_STATUS.json", summary)
    report = f"""# E183 四项研究合并审计

E182 的事前门槛仍记为 **FAIL**：20 个靶点覆盖 16 个，少于注册要求的 17 个。没有换阈值、删靶点或重跑划分。

把 E181 的三项评价与 E182 按同一个冻结 10 模型家族定义合并后，共有 **{total_tasks:,} 条评价任务、{total_targets:,} 个靶点簇**。家族 RMS 确定性下界和最坏成员确定性下界均为 **0 违反**；Hilbert 恒等式最大绝对残差为 `{summary['max_hilbert_identity_absolute_residual']:.3e}`。

家族上界在任务层覆盖 **{family_tasks_covered:,}/{total_tasks:,} = {summary['family_upper_task_coverage']:.2%}**，在靶点同时覆盖层面为 **{family_targets_covered}/{total_targets} = {pooled_coverage:.2%}**，回到注册的 90% 附近。最坏成员上界的靶点同时覆盖为 **{worst_targets_covered}/{total_targets} = {summary['worst_upper_target_coverage']:.2%}**。

E182 使用 19 个校准靶点的第 18 顺序统计量。若靶点分数连续且交换，冻结阈值的覆盖概率本身服从 `Beta(18, 2)`；对随后 20 个靶点，覆盖数不超过 16 的 beta-binomial 参考概率为 **{beta_binomial_tail:.2%}**。因此 E182 的 16/20 属于可预期的有限校准波动，不能被写成 E182 单项通过，也不能据此事后修改方法。

E183 是看到 E182 后的合并解释，不是新的前瞻验证。合并置信区间只作描述，因为四项研究的靶点和技术结构并非完全同质。
"""
    atomic_text(OUT / "reports/E183_SYNTHESIS_REPORT.md", report)
    readme = """# E183 先看这个

本目录回答一个明确问题：E182 单项少覆盖 1 个靶点，是否意味着已经冻结的家族证书整体失效。

先看 `RUN_STATUS.json` 和 `reports/E183_SYNTHESIS_REPORT.md`。E182 的 FAIL 原样保留；E183 只做跨研究合并与有限样本解释，不重写历史结果。
"""
    atomic_text(OUT / "README_先看这个.md", readme)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
