#!/usr/bin/env python3
"""Create audit-friendly figures for the completed E207 stress test.

The script only visualizes committed E207 tables.  It does not select weights,
drop studies, recompute outcomes, or change any statistical decision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#315A7D"
TEAL = "#16877A"
CORAL = "#D86654"
GOLD = "#C89B3C"
GREY = "#7A8791"
LIGHT = "#EEF2F4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "docs/实验结果/E207_fixed_formula_multimodal_stress_20260912"
        ),
    )
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B1B8")
    ax.spines["bottom"].set_color("#A8B1B8")
    ax.tick_params(color="#A8B1B8", length=3)


def save(fig: plt.Figure, root: Path, stem: str) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci_summary(root: Path) -> None:
    tables = root / "tables"
    point = pd.read_csv(tables / "E207_POINT_DELTAS.csv")
    boot = pd.read_csv(tables / "E207_BOOTSTRAP_SUMMARY.csv")
    data = point.merge(boot, on=["analysis", "metric"], validate="one_to_one")
    order = [
        ("E153_genetic_8study", "delta_spearman"),
        ("E153_genetic_8study", "delta_review_utility"),
        ("E187_genetic_difficulty", "delta_spearman"),
        ("E187_genetic_difficulty", "delta_review_utility"),
        ("E187_cytokine_difficulty", "delta_spearman"),
        ("E187_cytokine_difficulty", "delta_review_utility"),
    ]
    labels = [
        "8 genetic studies · rank association",
        "8 genetic studies · review utility",
        "3 genetic datasets · rank association",
        "3 genetic datasets · review utility",
        "Cytokine dataset · rank association",
        "Cytokine dataset · review utility",
    ]
    indexed = data.set_index(["analysis", "metric"])
    rows = indexed.loc[order].reset_index()
    y = np.arange(len(rows))[::-1]
    colors = [TEAL, TEAL, NAVY, NAVY, CORAL, CORAL]

    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.axvline(0, color="#525C63", lw=0.9, ls="--", zorder=0)
    for yi, (_, row), color in zip(y, rows.iterrows(), colors):
        estimate = float(row["point_estimate"])
        lower = float(row["ci95_lower"])
        upper = float(row["ci95_upper"])
        ax.plot([lower, upper], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter(estimate, yi, s=31, color=color, edgecolor="white", lw=0.7, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("SafeConf-M minus predicted magnitude")
    ax.set_xlim(-0.024, 0.058)
    ax.grid(axis="x", color=LIGHT, lw=0.8)
    clean_axis(ax)
    save(fig, root, "E207_ci_summary")


def cross_study(root: Path) -> None:
    raw = pd.read_csv(root / "tables/E207_E153_BATCH_METRICS.csv")
    raw = raw[np.isclose(raw["budget"], 0.20)]
    metrics = {
        "spearman": "Rank association",
        "review_utility": "Review utility at 20% budget",
    }
    datasets = list(dict.fromkeys(raw["dataset"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    for ax, (metric, xlabel) in zip(axes, metrics.items()):
        wide = raw.groupby(["dataset", "score"], sort=False)[metric].mean().unstack()
        delta = (wide["safeconf_m"] - wide["magnitude"]).reindex(datasets)
        y = np.arange(len(delta))[::-1]
        ax.axvline(0, color="#525C63", lw=0.9, ls="--", zorder=0)
        for yi, value in zip(y, delta):
            color = TEAL if value >= 0 else CORAL
            ax.plot([0, value], [yi, yi], color=color, lw=1.7)
            ax.scatter(value, yi, s=30, color=color, edgecolor="white", lw=0.7, zorder=3)
        ax.set_xlabel(f"Δ {xlabel}")
        ax.grid(axis="x", color=LIGHT, lw=0.8)
        clean_axis(ax)
    dataset_labels = [name.replace("_", " ") for name in datasets]
    axes[0].set_yticks(np.arange(len(datasets))[::-1], dataset_labels)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(wspace=0.14)
    save(fig, root, "E207_eight_study_deltas")


def scenario_heatmap(root: Path) -> None:
    raw = pd.read_csv(root / "tables/E207_E187_SCENARIO_SUMMARY.csv")
    raw = raw[np.isclose(raw["budget"], 0.20)]
    datasets = ["Frangieh", "Lara_exvivo", "Santinha", "Cui_direct41"]
    settings = [
        "random_missing_pair",
        "perturbation_unseen_column",
        "context_unseen_row",
        "context_and_perturbation_unseen",
    ]
    setting_labels = [
        "Missing pair",
        "Unseen perturbation",
        "Unseen context",
        "Both unseen",
    ]
    dataset_labels = ["Frangieh", "Lara ex vivo", "Santinha", "Cui · cytokine"]
    wide = raw.pivot_table(
        index=["dataset", "setting"],
        columns="score",
        values="spearman",
        aggfunc="mean",
    )
    wide["delta"] = wide["safeconf_m"] - wide["magnitude"]
    matrix = np.array(
        [
            [wide.loc[(dataset, setting), "delta"] for setting in settings]
            for dataset in datasets
        ]
    )
    limit = max(0.03, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "safeconf_diverging", [CORAL, "#FAFAFA", TEAL]
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(settings)), setting_labels)
    ax.set_yticks(np.arange(len(datasets)), dataset_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    ax.axhline(2.5, color="white", lw=3)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    bar.set_label("Δ rank association")
    bar.outline.set_visible(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, root, "E207_scenario_heatmap")


def modality_boundary(root: Path) -> None:
    point = pd.read_csv(root / "tables/E207_POINT_DELTAS.csv")
    boot = pd.read_csv(root / "tables/E207_BOOTSTRAP_SUMMARY.csv")
    merged = point.merge(boot, on=["analysis", "metric"], validate="one_to_one")
    key = merged.set_index(["analysis", "metric"])
    cards = [
        ("Genetic\n8 studies", "E153_genetic_8study"),
        ("Genetic\ndifficulty ladder", "E187_genetic_difficulty"),
        ("Cytokine\ndifficulty ladder", "E187_cytokine_difficulty"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 2.55))
    ax.set_xlim(0, 6.25)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (label, analysis) in enumerate(cards):
        x = 0.10 + index * 1.55
        rank = key.loc[(analysis, "delta_spearman")]
        utility = key.loc[(analysis, "delta_review_utility")]
        rank_pass = float(rank["ci95_lower"]) > 0
        utility_pass = float(utility["ci95_lower"]) > 0
        edge = TEAL if rank_pass else GREY
        box = mpl.patches.FancyBboxPatch(
            (x, 0.42),
            1.36,
            2.02,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor="white",
            edgecolor=edge,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + 0.68, 2.14, label, ha="center", va="center", weight="bold", fontsize=9)
        ax.text(x + 0.68, 1.63, "Rank association", ha="center", color=GREY, fontsize=7.5)
        ax.text(
            x + 0.68,
            1.42,
            "confirmed" if rank_pass else "not confirmed",
            ha="center",
            color=TEAL if rank_pass else CORAL,
            weight="bold",
            fontsize=7.5,
        )
        ax.text(x + 0.68, 1.10, "20% review utility", ha="center", color=GREY, fontsize=7.5)
        ax.text(
            x + 0.68,
            0.89,
            "confirmed" if utility_pass else "not confirmed",
            ha="center",
            color=TEAL if utility_pass else CORAL,
            weight="bold",
            fontsize=7.5,
        )
        ax.text(
            x + 0.68,
            0.60,
            "use conditionally" if rank_pass else "keep separate",
            ha="center",
            va="center",
            fontsize=8,
            color=edge,
        )
    x = 4.75
    box = mpl.patches.FancyBboxPatch(
        (x, 0.42),
        1.36,
        2.02,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor="white",
        edgecolor=GREY,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x + 0.68, 2.14, "Chemical", ha="center", va="center", weight="bold", fontsize=9)
    ax.text(x + 0.68, 1.58, "Magnitude baseline", ha="center", color=NAVY, fontsize=8)
    ax.text(x + 0.68, 1.10, "No matched fusion input", ha="center", color=GREY, fontsize=7.5)
    ax.text(x + 0.68, 0.60, "do not merge", ha="center", color=CORAL, weight="bold", fontsize=8)
    save(fig, root, "E207_modality_boundary")


def main() -> None:
    args = parse_args()
    root = args.result_dir.resolve()
    style()
    ci_summary(root)
    cross_study(root)
    scenario_heatmap(root)
    modality_boundary(root)
    print(root / "figures")


if __name__ == "__main__":
    main()
