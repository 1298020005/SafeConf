#!/usr/bin/env python3
"""Plot Phase 4c draft figures from figure-ready tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "agents").is_dir() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root")


ROOT = find_repo_root()
OUT = ROOT / "docs" / "实验结果" / "Evidence_to_Claim_20260615"
TABLES = OUT / "figure_ready_tables"
FIGS = OUT / "figures"


DATASET_LABELS = {
    "CuiHacohen2023": "Cui",
    "Frangieh": "Frangieh",
    "LaraAstiasoHuntly2023_exvivo": "Lara ex vivo",
    "LaraAstiasoHuntly2023_invivo": "Lara in vivo",
    "McFarlandTsherniak2020": "McFarland",
    "SantinhaPlatt2023": "Santinha",
    "SrivatsanTrapnell2020_sciplex3": "Srivatsan",
}


def save(fig: plt.Figure, stem: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def plot_fig1() -> None:
    var = pd.read_csv(TABLES / "FIG1_A1_VARIANCE_DECOMPOSITION.csv")
    paired = pd.read_csv(TABLES / "FIG1_A1_ERROR_SCATTER.csv")
    overall = var[var["dataset_name"].eq("__overall__")].iloc[0]

    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 0.9], wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    boxes = [
        ("single-cell\nperturbation data", 0.50, 0.84),
        ("train-side\nrisk features", 0.50, 0.63),
        ("frozen / learned\nrisk score", 0.50, 0.42),
        ("triage high-risk\npredictions", 0.50, 0.21),
    ]
    for text, x, y in boxes:
        ax0.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#555555", lw=1),
            transform=ax0.transAxes,
        )
    arrows = [((0.50, 0.78), (0.50, 0.70)), ((0.50, 0.57), (0.50, 0.49)), ((0.50, 0.36), (0.50, 0.28))]
    for start, end in arrows:
        ax0.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
    ax0.set_title("A  SafeConf workflow", loc="left")

    ax1 = fig.add_subplot(gs[0, 1])
    sample = paired.sample(min(len(paired), 2500), random_state=5201)
    ax1.scatter(
        sample["v0_error_rmse"],
        sample["contextsim_error_rmse"],
        s=8,
        alpha=0.32,
        color="#4c78a8",
        edgecolors="none",
    )
    lim = max(sample["v0_error_rmse"].quantile(0.995), sample["contextsim_error_rmse"].quantile(0.995))
    ax1.plot([0, lim], [0, lim], ls="--", lw=1, color="#777777")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("V0 error RMSE")
    ax1.set_ylabel("ContextSim error RMSE")
    ax1.set_title(f"B  Task errors align (rho={overall['spearman_v0_vs_contextsim_error']:.3f})", loc="left")

    ax2 = fig.add_subplot(gs[0, 2])
    labels = ["task", "predictor", "residual"]
    vals = [overall["frac_task"], overall["frac_predictor"], overall["frac_residual"]]
    colors = ["#59a14f", "#e15759", "#bab0ac"]
    ax2.bar(labels, vals, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction of error variance")
    ax2.set_title("C  Variance decomposition", loc="left")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.025, f"{100*v:.1f}%", ha="center", va="bottom")

    fig.suptitle("Fig 1 draft: task-risk motivation and SafeConf workflow", y=1.02, fontsize=11, fontweight="bold")
    save(fig, "FIG1_task_risk_motivation")


def plot_fig2() -> None:
    df = pd.read_csv(TABLES / "FIG2_FORMAL_MAIN_FOREST.csv")
    order = [
        "CuiHacohen2023",
        "Frangieh",
        "LaraAstiasoHuntly2023_exvivo",
        "LaraAstiasoHuntly2023_invivo",
        "SantinhaPlatt2023",
        "SrivatsanTrapnell2020_sciplex3",
        "McFarlandTsherniak2020",
    ]
    df["order"] = df["dataset_name"].map({d: i for i, d in enumerate(order)})
    df = df.sort_values("order")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvline(0, color="#888888", lw=1)
    ax.scatter(df["aligned_rho"], y - 0.18, marker="o", color="#4c78a8", label="aligned rho", zorder=3)
    ax.errorbar(
        df["partial_rho"],
        y,
        xerr=[
            df["partial_rho"] - df["partial_rho_ci_low"],
            df["partial_rho_ci_high"] - df["partial_rho"],
        ],
        fmt="s",
        color="#f28e2b",
        ecolor="#f28e2b",
        capsize=3,
        label="partial rho (95% CI)",
        zorder=4,
    )
    ax.scatter(df["magnitude_only_rho"], y + 0.18, marker="^", color="#59a14f", label="magnitude-only rho", zorder=3)
    labels = [DATASET_LABELS.get(d, d) for d in df["dataset_name"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for tick, is_fail in zip(ax.get_yticklabels(), df["is_failure_boundary"]):
        if is_fail:
            tick.set_color("#d62728")
            tick.set_fontweight("bold")
    ax.set_xlabel("Spearman rho")
    ax.set_xlim(-0.18, 0.92)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Fig 2 draft: frozen v0.2 main-table risk ranking", loc="left")
    save(fig, "FIG2_formal_main_forest")


def plot_fig3() -> None:
    e2 = pd.read_csv(TABLES / "FIG3_E2_MAGNITUDE_RESIDUAL.csv")
    panel = pd.read_csv(TABLES / "FIG3_LOPO_LEARNED_PANEL.csv")
    order = e2.sort_values("aurc_diff_magnitude_minus_combined", ascending=False)["dataset_name"]
    e2 = e2.set_index("dataset_name").loc[order].reset_index()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(e2))
    y = e2["aurc_diff_magnitude_minus_combined"]
    yerr = np.vstack([y - e2["aurc_diff_ci_low"], e2["aurc_diff_ci_high"] - y])
    colors = ["#d62728" if d == "McFarlandTsherniak2020" else "#4c78a8" for d in e2["dataset_name"]]
    ax0.bar(x, y, color=colors, alpha=0.85)
    ax0.errorbar(x, y, yerr=yerr, fmt="none", color="#333333", lw=1, capsize=3)
    ax0.axhline(0, color="#777777", lw=1)
    ax0.set_xticks(x)
    ax0.set_xticklabels([DATASET_LABELS.get(d, d) for d in e2["dataset_name"]], rotation=35, ha="right")
    ax0.set_ylabel("AURC improvement\n(magnitude-only minus combined)")
    ax0.set_title("A  Beyond magnitude-only (E2)", loc="left")

    ax1.axline((0, 0), slope=1, ls="--", color="#999999", lw=1)
    for _, row in panel.iterrows():
        color = "#d62728" if row["is_mcfarland"] else "#4c78a8"
        size = 70 if row["is_mcfarland"] else 42
        ax1.scatter(row["frozen_partial_rho"], row["learned_partial_rho"], s=size, color=color, alpha=0.9)
        label = DATASET_LABELS.get(row["dataset_name"], row["dataset_name"])
        ax1.text(row["frozen_partial_rho"] + 0.012, row["learned_partial_rho"] + 0.012, label, fontsize=7)
    ax1.axvline(0, color="#dddddd", lw=1)
    ax1.axhline(0, color="#dddddd", lw=1)
    ax1.set_xlabel("frozen v0.2 partial rho")
    ax1.set_ylabel("learned LOPO partial rho")
    ax1.set_xlim(-0.12, 0.72)
    ax1.set_ylim(-0.02, 0.9)
    ax1.set_title("B  Learned risk captures missed signal", loc="left")
    fig.suptitle("Fig 3 draft: magnitude defense and learned-risk extension", y=1.02, fontsize=11, fontweight="bold")
    save(fig, "FIG3_magnitude_residual_and_learned")


def plot_fig4() -> None:
    raw = pd.read_csv(TABLES / "FIG4_E8B_EXTERNAL_BENCHMARK.csv")
    paired = pd.read_csv(TABLES / "FIG4_E8B_PARTIAL_PER_METHOD.csv")
    raw = raw.sort_values("spearman_rho", ascending=True).reset_index(drop=True)
    paired = paired.set_index("method_name").loc[raw["method_name"]].reset_index()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(raw))
    lo = raw["shuffled_null_ci_low"].iloc[0]
    hi = raw["shuffled_null_ci_high"].iloc[0]
    ax0.axhspan(lo, hi, color="#d9d9d9", alpha=0.8, label="shuffled null 95% range")
    colors = ["#d62728" if v < 0.25 else "#4c78a8" for v in raw["spearman_rho"]]
    ax0.bar(x, raw["spearman_rho"], color=colors)
    ax0.axhline(0, color="#777777", lw=1)
    ax0.set_xticks(x)
    ax0.set_xticklabels(raw["method_name"], rotation=55, ha="right")
    ax0.set_ylabel("Spearman rho")
    ax0.set_title("A  Frangieh scPerturBench MSE association", loc="left")
    ax0.legend(frameon=False, loc="upper left")

    y_pos = np.arange(len(paired))
    ax1.axvline(paired["sample_size_baseline_median_spearman"].iloc[0], color="#e15759", ls=":", lw=1.5, label="Nstim baseline")
    for i, row in paired.iterrows():
        ax1.plot(
            [row["partial_spearman_control_log_nstimulated"], row["raw_spearman_rho"]],
            [i, i],
            color="#bbbbbb",
            lw=1,
            zorder=1,
        )
    ax1.scatter(paired["raw_spearman_rho"], y_pos, color="#4c78a8", label="raw", zorder=3)
    ax1.scatter(
        paired["partial_spearman_control_log_nstimulated"],
        y_pos,
        color="#f28e2b",
        label="partial, control log(Nstim)",
        zorder=3,
    )
    ax1.axvline(0, color="#888888", lw=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(paired["method_name"], fontsize=7)
    ax1.set_xlabel("Spearman rho")
    ax1.set_title("B  Sample-size diagnostic (post hoc)", loc="left")
    ax1.legend(frameon=False, loc="lower right")
    fig.suptitle("Fig 4 draft: external benchmark association with caveats", y=1.02, fontsize=11, fontweight="bold")
    save(fig, "FIG4_e8b_external_benchmark")


def _fig5_heatmap_matrix(top_fracs: list[float]) -> pd.DataFrame:
    heat = pd.read_csv(TABLES / "FIG5_COST_EFFECTIVENESS_HEATMAP.csv")
    heat = heat[heat["top_fraction"].isin(top_fracs)].copy()
    heat = heat[heat["score_name"].isin([
        "random",
        "predicted_magnitude",
        "protocol_v0_2_family_confidence",
        "safeconf_lodo_risk",
        "safeconf_perdataset_risk",
    ])]
    heat["dataset_label"] = heat["dataset_name"].map(DATASET_LABELS).fillna(heat["dataset_name"])
    if len(top_fracs) == 1:
        heat["column"] = heat["strategy_label"]
    else:
        heat["column"] = heat["strategy_label"] + "\n@" + heat["top_percent"].astype(str) + "%"
    dataset_order = [
        "Cui",
        "Frangieh",
        "Lara ex vivo",
        "Lara in vivo",
        "McFarland",
        "Santinha",
        "Srivatsan",
    ]
    strategies = ["Random", "Magnitude-only", "Frozen v0.2", "LODO risk", "Per-dataset risk"]
    if len(top_fracs) == 1:
        col_order = strategies
    else:
        pct_order = [int(round(f * 100)) for f in top_fracs]
        col_order = [f"{strategy}\n@{pct}%" for strategy in strategies for pct in pct_order]
    return (
        heat.pivot_table(index="dataset_label", columns="column", values="enrichment_fold", aggfunc="first")
        .reindex(index=dataset_order, columns=col_order)
    )


def _draw_fig5_heatmap(ax: plt.Axes, mat: pd.DataFrame, title: str, fontsize: float = 7.0):
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35 if mat.shape[1] <= 5 else 55, ha="right", fontsize=fontsize)
    ax.set_title(title, loc="left")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=fontsize, color="#222222")
    return im


def plot_fig5() -> None:
    macro = pd.read_csv(TABLES / "FIG5_COST_EFFECTIVENESS_MACRO_TOP10.csv")
    macro_order = [
        "random",
        "predicted_magnitude",
        "protocol_v0_2_family_confidence",
        "safeconf_lodo_risk",
        "safeconf_perdataset_risk",
        "oracle_magnitude_diagnostic",
    ]
    macro = macro.set_index("score_name").loc[macro_order].reset_index()

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )

    colors = {
        "random": "#bab0ac",
        "predicted_magnitude": "#59a14f",
        "protocol_v0_2_family_confidence": "#4c78a8",
        "safeconf_lodo_risk": "#9c755f",
        "safeconf_perdataset_risk": "#f28e2b",
        "oracle_magnitude_diagnostic": "#7f7f7f",
    }
    x = np.arange(len(macro))
    ax0.bar(x, macro["enrichment_fold"], color=[colors[s] for s in macro["score_name"]])
    ax0.axhline(1, color="#777777", lw=1, ls="--")
    ax0.set_xticks(x)
    ax0.set_xticklabels(macro["strategy_label"], rotation=35, ha="right")
    ax0.set_ylabel("enrichment over random")
    ax0.set_title("A  Top 10% high-error retrieval", loc="left")
    ax0.axvline(4.5, color="#888888", lw=1, ls="--")
    ax0.text(
        5,
        max(macro["enrichment_fold"]) * 0.58,
        "non-deployable\nreference",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )
    for i, v in enumerate(macro["enrichment_fold"]):
        ax0.text(i, v + 0.12, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)

    mat = _fig5_heatmap_matrix([0.10])
    im = _draw_fig5_heatmap(ax1, mat, "B  Per-dataset enrichment at top 10%", fontsize=7.0)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("enrichment fold")

    fig.suptitle("Fig 5 draft: prediction triage as high-error retrieval", y=1.02, fontsize=11, fontweight="bold")
    save(fig, "FIG5_cost_effectiveness")


def plot_sfig5() -> None:
    mat = _fig5_heatmap_matrix([0.05, 0.10, 0.20])
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    im = _draw_fig5_heatmap(ax, mat, "SFig 5 draft: per-dataset enrichment across triage thresholds", fontsize=5.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("enrichment fold")
    save(fig, "SFIG5_cost_effectiveness_thresholds")


def main() -> None:
    setup()
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_sfig5()
    print(f"Wrote draft figures to {FIGS}")


if __name__ == "__main__":
    main()
