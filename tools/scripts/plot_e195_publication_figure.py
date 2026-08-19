#!/usr/bin/env python3
"""Render a concise same-prediction summary for completed E195 tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "docs"
    / "实验结果"
    / "E195_native_gears_uq_norman_p1p2_20260730"
)
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

COLORS = {"P1": "#3C5488", "P2": "#E64B35"}
SCORE_LABELS = {
    "native_logvar_mean": "Native log-variance",
    "seed_disagreement": "Seed disagreement",
    "predicted_magnitude": "Predicted magnitude",
}
LINESTYLES = {
    "native_logvar_mean": "-",
    "seed_disagreement": "--",
    "predicted_magnitude": ":",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source_paths = [
        TABLES / "E195_FAMILY_TASKS.csv",
        TABLES / "E195_ASSOCIATION.csv",
        TABLES / "E195_ROUTING_METRICS.csv",
        TABLES / "E195_RISK_COVERAGE.csv",
    ]
    family = pd.read_csv(source_paths[0])
    association = pd.read_csv(source_paths[1])
    routing = pd.read_csv(source_paths[2])
    coverage = pd.read_csv(source_paths[3])
    expected_panels = {"P1", "P2"}
    if set(family.panel.astype(str)) != expected_panels or len(family) != 48:
        raise RuntimeError("E195 family table is not the frozen 2 × 24 task set")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a | Native score and its own prediction-family error.
    for panel in ("P1", "P2"):
        take = family[family.panel == panel]
        rho = association[
            (association.track == "A_family")
            & (association.panel == panel)
            & (association.score_name == "native_logvar_mean")
        ].spearman.iloc[0]
        ax_a.scatter(
            take.native_logvar_mean,
            take.family_rms_error,
            s=31,
            alpha=0.78,
            color=COLORS[panel],
            edgecolor="white",
            linewidth=0.45,
            label=f"{panel}  ρ={rho:.2f}",
        )
    ax_a.set(
        xlabel="Native mean log-variance (higher = riskier)",
        ylabel="GEARS-UQ family RMS error",
    )
    ax_a.legend(frameon=False, loc="upper left")

    # b | Panel-specific associations for the three scores on identical outputs.
    scores = list(SCORE_LABELS)
    x = np.arange(len(scores))
    for panel, offset in (("P1", -0.07), ("P2", 0.07)):
        values = []
        for score in scores:
            take = association[
                (association.track == "A_family")
                & (association.panel == panel)
                & (association.score_name == score)
            ]
            if len(take) != 1:
                raise RuntimeError(f"missing association: {panel}/{score}")
            values.append(float(take.spearman.iloc[0]))
        ax_b.plot(
            x + offset,
            values,
            marker="o",
            markersize=6,
            linewidth=1.25,
            color=COLORS[panel],
            label=panel,
        )
    ax_b.axhline(0, color="#777777", linewidth=0.7)
    ax_b.set_xticks(x, [SCORE_LABELS[score] for score in scores], rotation=14)
    ax_b.set(ylabel="Spearman ρ with family RMS error")
    ax_b.legend(frameon=False)

    # c | Fixed-budget routing, kept panel-specific rather than silently pooled.
    width = 0.35
    for panel, offset in (("P1", -width / 2), ("P2", width / 2)):
        values = []
        for score in scores:
            take = routing[
                (routing.system == "GEARS-UQ")
                & (routing.arm == "same_prediction_family_rms")
                & (routing.panel == panel)
                & (routing.score_name == score)
                & np.isclose(routing.budget, 0.20)
            ]
            if len(take) != 1:
                raise RuntimeError(f"missing routing result: {panel}/{score}")
            values.append(float(take.oracle_normalized_utility.iloc[0]))
        ax_c.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[panel],
            alpha=0.9,
            label=panel,
        )
    ax_c.axhline(0, color="#777777", linewidth=0.7)
    ax_c.set_xticks(x, [SCORE_LABELS[score] for score in scores], rotation=14)
    ax_c.set(
        ylabel="20% oracle-normalized routing utility",
        ylim=(-0.05, 1.02),
    )
    ax_c.legend(frameon=False)

    # d | Low-risk-first selective error curves.
    take = coverage[
        (coverage.system == "GEARS-UQ")
        & (coverage.arm == "same_prediction_family_rms")
    ]
    for panel in ("P1", "P2"):
        for score in scores:
            curve = take[
                (take.panel == panel) & (take.score_name == score)
            ].sort_values("coverage")
            ax_d.plot(
                curve.coverage,
                curve.normalized_selective_error,
                color=COLORS[panel],
                linestyle=LINESTYLES[score],
                linewidth=1.55,
                label=f"{panel} · {SCORE_LABELS[score]}",
            )
    ax_d.axhline(1, color="#777777", linewidth=0.7)
    ax_d.set(
        xlabel="Coverage retained (low risk first)",
        ylabel="Selective error / full error",
        xlim=(0.49, 1.01),
    )
    ax_d.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower right")

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.14,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)

    fig.suptitle(
        "Native uncertainty on the same GEARS-UQ prediction family",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.012,
        "P1 and P2 are separate frozen 24-task Norman panels; "
        "post-truth direct-competitor replication.",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.955), h_pad=2.2, w_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / "E195_same_prediction_summary.png"
    pdf_path = FIGURES / "E195_same_prediction_summary.pdf"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)
    provenance = {
        "schema": "safeconf_e195_visualization_v1",
        "scope": (
            "Same-prediction GEARS-UQ summary only; no new statistical analysis"
        ),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sources": {
            str(path.relative_to(OUT)): sha256_file(path) for path in source_paths
        },
        "outputs": {
            str(png_path.relative_to(OUT)): sha256_file(png_path),
            str(pdf_path.relative_to(OUT)): sha256_file(pdf_path),
        },
    }
    (FIGURES / "E195_VISUALIZATION_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
