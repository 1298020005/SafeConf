#!/usr/bin/env python3
"""Build the five manuscript figures and submission source-data tables.

Every plotted empirical value is read from the committed E145, E178--E187 releases.
Conceptual panels contain only the equations and access sequence defined in the
registered analysis.  The script is intentionally deterministic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import betabinom


HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
OUT = HERE / "figures"
TABLES = HERE / "tables"

E181 = ROOT / "docs/实验结果/E181_registered_family_hilbert_certificate_20260724"
E182 = ROOT / "docs/实验结果/E182_gse225807_registered_family_20260724"
E183 = ROOT / "docs/实验结果/E183_all_study_family_synthesis_20260724"
E185 = ROOT / "docs/实验结果/E185_minimal_release_validation_20260724"
E186 = ROOT / "docs/实验结果/E186_presubmission_integrity_audit_20260724"
E178 = ROOT / "docs/实验结果/E178_crossstudy_bilateral_certificate_audit_20260722"
E179 = ROOT / "docs/实验结果/E179_nested_uq_baseline_benchmark_20260723"
E145 = ROOT / "docs/实验结果/E145_prescribe_paper_endpoint_20260714"
E187 = ROOT / "docs/实验结果/E187_advisor_difficulty_certificate_20260726"

BLUE = "#2F6B8A"
LIGHT_BLUE = "#DCEAF1"
PALE_BLUE = "#F3F8FA"
TEAL = "#4C8C8A"
GRAY = "#606A70"
LIGHT_GRAY = "#E7ECEF"
DARK = "#1F2933"
RED = "#B94A48"
LIGHT_RED = "#F7E5E3"
GREEN = "#4F7D63"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9AA4AA",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    svg_path = OUT / f"{stem}.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.12,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=DARK,
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    face: str = PALE_BLUE,
    edge: str = BLUE,
    title_color: str = DARK,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=6.1,
        color=GRAY,
        linespacing=1.25,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def figure_1() -> None:
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.22, 0.92], hspace=0.34)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a")

    box(
        ax,
        (0.015, 0.56),
        0.175,
        0.30,
        "Registered family",
        "5 scGPT + 5 GEARS\nseeds 3407–3411\nfrozen before truth",
    )
    box(
        ax,
        (0.245, 0.56),
        0.185,
        0.30,
        "Family geometry",
        r"$\bar{p}$,  $D_F$,  $\Delta_F$,  $r_F$"
        "\nshared 512-gene scale\ncomputed without truth",
    )
    box(
        ax,
        (0.485, 0.56),
        0.205,
        0.30,
        "Reference calibration",
        r"target event: $\|c-y\|\leq U$"
        "\nobservable shift"
        "\n"
        r"$s=\|\bar{p}-c\|$",
    )
    box(
        ax,
        (0.745, 0.56),
        0.235,
        0.30,
        "Two-sided certificate",
        r"$D_F\leq R_F\leq\sqrt{(U+s)^2+D_F^2}$"
        "\n"
        r"$\Delta_F/2\leq W_F\leq U+s+r_F$",
        face="#EEF5F1",
        edge=GREEN,
    )
    arrow(ax, (0.19, 0.71), (0.245, 0.71))
    arrow(ax, (0.43, 0.71), (0.485, 0.71))
    arrow(ax, (0.69, 0.71), (0.745, 0.71))

    ax.text(
        0.337,
        0.36,
        r"$R_F^2=\|\bar{p}-y\|^2+D_F^2$",
        ha="center",
        va="center",
        fontsize=10,
        color=BLUE,
        fontweight="bold",
    )
    ax.text(
        0.337,
        0.22,
        "Classical squared-error ambiguity identity",
        ha="center",
        va="center",
        fontsize=7.4,
        color=GRAY,
    )
    ax.text(
        0.755,
        0.32,
        "Output is an error interval for a fixed model family,\n"
        "not a probability that one biological prediction is correct.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=DARK,
        linespacing=1.4,
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    panel_label(ax2, "b")
    stages = [
        (
            "F1  Metadata",
            "eligibility + salted\nidentity split\ntruth closed",
            BLUE,
            PALE_BLUE,
        ),
        (
            "F2  Predictions",
            "train permitted targets\nrelease frozen predictions\nhidden truth closed",
            BLUE,
            PALE_BLUE,
        ),
        (
            "F3  Calibration",
            "open calibration targets\nfreeze conformal threshold\nevaluation truth closed",
            TEAL,
            "#EEF6F5",
        ),
        (
            "F4  Evaluation",
            "open evaluation once\nno model or gate changes\nretain pass/fail",
            RED,
            LIGHT_RED,
        ),
    ]
    xs = [0.015, 0.265, 0.515, 0.765]
    for x, (title, body, edge, face) in zip(xs, stages):
        box(ax2, (x, 0.43), 0.21, 0.36, title, body, edge=edge, face=face)
    for x in xs[:-1]:
        arrow(ax2, (x + 0.21, 0.61), (x + 0.25, 0.61))
    ax2.text(
        0.5,
        0.18,
        "Target identity is the exchangeability unit: guides, states, or technical groups stay together.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=DARK,
    )
    save_figure(fig, "Figure_1_method_and_protocol")


def read_release() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(E183 / "tables/E183_STUDY_SUMMARY.csv")
    tasks = pd.read_csv(E183 / "tables/E183_COMBINED_TASK_CERTIFICATES.csv")
    targets = pd.read_csv(E183 / "tables/E183_TARGET_CERTIFICATES.csv")
    comparisons = pd.read_csv(E181 / "tables/E181_FAMILY_COMPARISONS.csv")
    return summary, tasks, targets, comparisons


DISPLAY = {
    "E176_primary_CD4": "Primary CD4",
    "E177_Sunshine": "Sunshine",
    "E180_XuCao": "XuCao",
    "E182_GSE225807": "GSE225807",
}


def figure_2() -> None:
    summary, tasks, _, comparisons = read_release()
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.44, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    estimates = summary["family_upper_target_coverage"].to_numpy()
    lows = summary["family_upper_target_ci95_low"].to_numpy()
    highs = summary["family_upper_target_ci95_high"].to_numpy()
    labels = [DISPLAY[x] for x in summary["study"]]
    pooled_n = int(summary["n_target_clusters"].sum())
    pooled_k = int(summary["family_upper_targets_covered"].sum())
    # Exact pooled interval is already frozen in the E183 report.
    pooled = pooled_k / pooled_n
    pooled_low, pooled_high = 0.880038, 0.923993
    estimates = np.r_[estimates, pooled]
    lows = np.r_[lows, pooled_low]
    highs = np.r_[highs, pooled_high]
    labels += ["Pooled (descriptive)"]
    y = np.arange(len(labels))[::-1]
    colors = [BLUE, BLUE, BLUE, RED, GRAY]
    for value, low, high, yy, color in zip(estimates, lows, highs, y, colors):
        ax.errorbar(
            value,
            yy,
            xerr=[[value - low], [high - value]],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            zorder=1,
        )
    ax.scatter(estimates, y, c=colors, s=28, zorder=2, edgecolor="white", linewidth=0.5)
    ax.axvline(0.90, color="#9AA4AA", linestyle="--", linewidth=0.9)
    for value, yy in zip(estimates, y):
        ax.text(value + 0.012, yy, f"{100*value:.1f}%", va="center", fontsize=7.2, color=DARK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.50, 1.04)
    ax.set_xlabel("Target-simultaneous family-upper coverage")
    ax.set_title("Coverage by study", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    data = [
        tasks.loc[tasks["study"] == code, "family_lower_tightness"].to_numpy()
        for code in DISPLAY
    ]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.2},
        whiskerprops={"color": GRAY, "linewidth": 0.8},
        capprops={"color": GRAY, "linewidth": 0.8},
        boxprops={"edgecolor": BLUE, "linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_RED]):
        patch.set_facecolor(color)
    medians = [np.median(x) for x in data]
    for i, value in enumerate(medians, start=1):
        ax.text(i, min(value + 0.055, 0.94), f"{value:.3f}", ha="center", fontsize=7.1, color=DARK)
    ax.set_xticks(range(1, 5), ["Primary\nCD4", "Sunshine", "XuCao", "GSE225807"])
    ax.set_ylabel(r"Lower-bound tightness  $D_F/R_F$")
    ax.set_ylim(0, 1)
    ax.set_title("Truth-free lower bound", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)

    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "c")
    comp = comparisons.loc[
        comparisons["comparison_family"].eq("frozen_10_seed_family")
    ].copy()
    comp["label"] = comp["study"].map(DISPLAY)
    x = np.arange(len(comp))
    values = comp["median_family_tightness_difference"].to_numpy()
    ax.bar(x, values, width=0.52, color=BLUE, edgecolor="white", linewidth=0.5)
    for xx, value in zip(x, values):
        ax.text(xx, value + 0.004, f"+{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.set_xticks(x, comp["label"])
    ax.set_ylabel("Median tightness difference")
    ax.set_title(
        "Ten seed-level members versus two architecture-level seed means",
        loc="left",
    )
    ax.text(
        0.99,
        0.92,
        "Positive in every paired task",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GREEN,
        fontsize=7.5,
        fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    save_figure(fig, "Figure_2_cross_study_results")


def figure_3() -> None:
    task_path = E182 / "final_evaluation/tables/E182_EVALUATION_TASKS.csv"
    target_path = E182 / "final_evaluation/tables/E182_EVALUATION_TARGETS.csv"
    tasks = pd.read_csv(task_path).sort_values("family_rms_error").reset_index(drop=True)
    targets = pd.read_csv(target_path).sort_values("max_centroid_rmse").reset_index(drop=True)
    with (E182 / "final_evaluation/E182_FINAL_SUMMARY.json").open() as handle:
        final = json.load(handle)
    threshold = float(final["constant_centroid_upper"])

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.44, wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a")
    x = np.arange(len(tasks))
    covered = tasks["family_upper_covered"].astype(bool).to_numpy()
    for xx, lower, upper, good in zip(
        x,
        tasks["family_diversity_lower"],
        tasks["family_rms_upper"],
        covered,
    ):
        ax.plot([xx, xx], [lower, upper], color=BLUE if good else RED, linewidth=1.0, alpha=0.9)
    ax.scatter(
        x[covered],
        tasks.loc[covered, "family_rms_error"],
        s=14,
        color=DARK,
        zorder=3,
        label="Observed family RMS",
    )
    ax.scatter(
        x[~covered],
        tasks.loc[~covered, "family_rms_error"],
        s=24,
        facecolor=RED,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Upper failure",
    )
    ax.set_xlabel("Evaluation guide task (sorted by observed error)")
    ax.set_ylabel("RMSE")
    ax.set_title("Forty one-time evaluation tasks", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b")
    k = np.arange(21)
    pmf = betabinom.pmf(k, 20, 18, 2)
    colors = np.where(k <= 16, RED, LIGHT_BLUE)
    ax.bar(k, pmf, color=colors, edgecolor="white", width=0.84, linewidth=0.4)
    ax.axvline(16.5, color=RED, linestyle="--", linewidth=0.8)
    ax.text(
        0.04,
        0.94,
        r"$P(K\leq16)=0.186813$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=RED,
        fontweight="bold",
    )
    ax.set_xlabel("Covered evaluation targets, K")
    ax.set_ylabel("Beta-binomial probability")
    ax.set_title("Finite-calibration reference", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c")
    y = np.arange(len(targets))
    family_ok = targets["family_rms_simultaneous_covered"].astype(bool).to_numpy()
    worst_ok = targets["worst_member_simultaneous_covered"].astype(bool).to_numpy()
    bar_colors = np.where(family_ok, LIGHT_BLUE, RED)
    ax.barh(
        y,
        targets["max_centroid_rmse"],
        color=bar_colors,
        edgecolor="white",
        height=0.75,
        linewidth=0.3,
    )
    ax.axvline(threshold, color=DARK, linestyle="--", linewidth=0.9)
    failed = targets.loc[~family_ok]
    for idx, row in failed.iterrows():
        suffix = " *" if not bool(worst_ok[idx]) else ""
        ax.text(
            row["max_centroid_rmse"] + 0.005,
            idx,
            str(row["perturbation"]) + suffix,
            va="center",
            fontsize=6.8,
            color=RED,
            fontweight="bold",
        )
    ax.text(
        threshold + 0.003,
        0.5,
        "calibrated threshold",
        rotation=90,
        va="bottom",
        fontsize=6.5,
        color=GRAY,
    )
    ax.text(
        0.99,
        0.02,
        "* also fails worst-member upper",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=RED,
    )
    ax.set_yticks([])
    ax.set_xlabel("Maximum centroid RMSE across two guides")
    ax.set_title("Target-simultaneous result: 16/20", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    save_figure(fig, "Figure_3_gse225807_confirmation")


def figure_4() -> None:
    source = E187 / "figures/Figure_E187_difficulty_ladder"
    for suffix in ("png", "pdf", "svg"):
        shutil.copy2(
            source.with_suffix(f".{suffix}"),
            OUT / f"Figure_4_difficulty_ladder.{suffix}",
        )


def figure_5() -> None:
    with (E185 / "RUN_STATUS.json").open() as handle:
        validation = json.load(handle)
    checklist = pd.read_csv(E186 / "tables/E186_CHECKLIST.csv")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 0.9], hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("Metadata freeze", "b4701ac", "F1"),
        ("Runtime hardening", "fc0de74", "F1"),
        ("Pretruth release", "2f4f148", "F2"),
        ("Calibration lock", "420e536", "F3"),
        ("Final evaluation", "593f663", "F4"),
        ("Four-study synthesis", "744b5ef", "post-F4"),
    ]
    xs = np.linspace(0.06, 0.94, len(stages))
    ax.plot([xs[0], xs[-1]], [0.56, 0.56], color="#A7B1B7", linewidth=1.2, zorder=0)
    for idx, (xx, (title, commit, phase)) in enumerate(zip(xs, stages)):
        color = RED if phase == "F4" else (GRAY if phase == "post-F4" else BLUE)
        ax.scatter(xx, 0.56, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=2)
        va = "bottom" if idx % 2 == 0 else "top"
        y = 0.69 if idx % 2 == 0 else 0.43
        ax.text(xx, y, title, ha="center", va=va, fontsize=7.1, fontweight="bold", color=DARK)
        ax.text(
            xx,
            y + (0.06 if idx % 2 == 0 else -0.06),
            f"{phase} · {commit}",
            ha="center",
            va=va,
            fontsize=6.4,
            color=GRAY,
        )
    ax.set_title("Committed before each truth boundary opened", loc="left", pad=4)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b")
    phases = ["F2 control", "F2 train", "F2 validation", "F3 calibration", "F4 evaluation"]
    rows = [91, 1348, 456, 926, 1051]
    colors = [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, TEAL, RED]
    x = np.arange(len(rows))
    bars = ax.bar(x, rows, color=colors, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,}", ha="center", fontsize=7)
    ax.set_xticks(x, ["control", "train", "validation", "calibration", "evaluation"], rotation=25, ha="right")
    ax.set_ylabel("Expression rows opened")
    ax.set_title("Phase-limited truth access", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.set_ylim(0, 1500)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    values = [
        ("4", "independent\nstudies"),
        ("2,433", "evaluation\ntasks"),
        ("737", "target\nclusters"),
        (f"{int(validation['checks_total']):,}", "release\nchecks"),
        ("0", "failed\nchecks"),
        ("16/20", "registered E182\ngate: FAIL"),
    ]
    coords = [(0.02, 0.58), (0.35, 0.58), (0.68, 0.58), (0.02, 0.12), (0.35, 0.12), (0.68, 0.12)]
    for (value, label), (xx, yy) in zip(values, coords):
        is_fail = "FAIL" in label
        face = LIGHT_RED if is_fail else PALE_BLUE
        edge = RED if is_fail else BLUE
        patch = FancyBboxPatch(
            (xx, yy),
            0.29,
            0.30,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
        ax.add_patch(patch)
        ax.text(xx + 0.145, yy + 0.19, value, ha="center", va="center", fontsize=12, color=edge, fontweight="bold")
        ax.text(
            xx + 0.145,
            yy + 0.075,
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            color=DARK,
            linespacing=1.05,
        )
    audit_pass = int(checklist["status"].eq("PASS").sum())
    ax.set_title(
        f"Independent reconstruction: {audit_pass}/{len(checklist)} integrity checks passed",
        loc="left",
    )
    save_figure(fig, "Figure_5_reproducibility_chain")


def write_tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary, _, _, comparisons = read_release()
    table_1 = pd.DataFrame(
        [
            ["Primary CD4", "Primary CD4 T cells; 4 donors; 3 states", 160, 640, 1920, "retrospective family reformulation"],
            ["Sunshine", "Calu-3 CRISPRi; 8 technical groups", 30, 50, 400, "retrospective family reformulation"],
            ["XuCao", "PerturbSci-Kinetics; guide-level effects", 29, 27, 73, "retrospective family reformulation"],
            ["GSE225807", "K562 RBP CRISPRi; 2 guides", 19, 20, 40, "fully preregistered confirmation"],
        ],
        columns=[
            "study",
            "biological_system",
            "calibration_targets_total",
            "evaluation_targets",
            "evaluation_tasks",
            "analysis_status",
        ],
    )
    table_1.to_csv(TABLES / "Table_1_study_design.csv", index=False)

    table_2 = summary.copy()
    table_2["study"] = table_2["study"].map(DISPLAY)
    table_2 = table_2[
        [
            "study",
            "n_tasks",
            "n_target_clusters",
            "family_lower_violations",
            "worst_lower_violations",
            "family_upper_tasks_covered",
            "family_upper_task_coverage",
            "family_upper_targets_covered",
            "family_upper_target_coverage",
            "worst_upper_targets_covered",
            "worst_upper_target_coverage",
            "median_family_lower_tightness",
            "median_worst_lower_tightness",
            "max_identity_abs_residual",
        ]
    ]
    table_2.to_csv(TABLES / "Table_2_certificate_results.csv", index=False)

    comparisons.to_csv(TABLES / "Table_S1_family_comparisons.csv", index=False)
    pd.read_csv(
        E182 / "final_evaluation/tables/E182_EVALUATION_TARGETS.csv"
    ).to_csv(TABLES / "Table_S2_gse225807_targets.csv", index=False)
    pd.read_csv(
        E182 / "final_evaluation/tables/E182_EVALUATION_TASKS.csv"
    ).to_csv(TABLES / "Table_S3_gse225807_tasks.csv", index=False)
    copy_tables = (
        (
            E187 / "tables/E187_CARTESIAN_SETTING_SUMMARY.csv",
            "Table_S4_difficulty_setting_summary.csv",
        ),
        (
            E187 / "tables/E187_MACRO_BOOTSTRAP.csv",
            "Table_S5_difficulty_macro_bootstrap.csv",
        ),
        (
            E187 / "tables/E187_CROSS_DATASET_SUMMARY.csv",
            "Table_S6_cross_dataset_summary.csv",
        ),
        (
            E187 / "tables/E187_CARTESIAN_TASK_CERTIFICATES.csv",
            "Table_S7_difficulty_task_certificates.csv",
        ),
        (
            E187 / "tables/E187_CROSS_DATASET_TASK_CERTIFICATES.csv",
            "Table_S8_cross_dataset_task_certificates.csv",
        ),
        (
            E178 / "tables/E178_SHARED_DIFFICULTY.csv",
            "Table_S9_model_error_concordance.csv",
        ),
        (
            E178 / "tables/E178_MODEL_SPECIFICITY.csv",
            "Table_S10_score_model_specificity.csv",
        ),
        (
            E179 / "tables/E179_METHOD_SUMMARY.csv",
            "Table_S11_nested_upper_baselines.csv",
        ),
        (
            E145 / "tables/E145_ASSOCIATIONS.csv",
            "Table_S12_prescribe_native_endpoint.csv",
        ),
        (
            E145 / "tables/E145_INCREMENTAL_VS_MAGNITUDE.csv",
            "Table_S13_prescribe_incremental.csv",
        ),
        (
            E145 / "tables/E145_SCORE_REDUNDANCY.csv",
            "Table_S14_prescribe_redundancy.csv",
        ),
    )
    for source, destination in copy_tables:
        shutil.copy2(source, TABLES / destination)


def main() -> None:
    for suffix in ("png", "pdf", "svg"):
        (OUT / f"Figure_4_reproducibility_chain.{suffix}").unlink(missing_ok=True)
    set_style()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    write_tables()
    print(f"Wrote figures to {OUT}")
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
