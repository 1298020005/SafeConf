"""Make clean, paper-style figures for the 2026-08-20 supervisor update.

All numerical values are copied from frozen E189/E191/E192/E198/E199/E200 records.
E201 is shown only as an execution-status panel: its target truth remains sealed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures_nature"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
DOUBLE = 183 * MM
INK = "#1F2933"
NAVY = "#244A68"
GREEN = "#16856B"
AMBER = "#D99500"
RED = "#C9553D"
BLUE = "#6D9FC7"
GREY = "#7A8790"
LIGHT = "#F3F6F8"
PALE_GREEN = "#E9F5EF"
PALE_AMBER = "#FFF4D9"
PALE_RED = "#FCEDEA"
PALE_BLUE = "#EAF2F8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def rounded_box(ax, x, y, w, h, text, *, fc=LIGHT, ec=NAVY, fs=8.0, color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=fc, edgecolor=ec, linewidth=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=color, linespacing=1.25)


def arrow(ax, a, b, color=NAVY, lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=color))


def fig1_contract() -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.25))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(0.05, 5.12, "Figure 1  |  SafeConf is an audit after prediction", fontsize=11,
            fontweight="bold", color=NAVY, ha="left")

    boxes = [
        (0.25, 3.08, 2.45, 1.22, "Predictors\nTxPert · scGPT\nGEARS", PALE_BLUE, NAVY),
        (3.05, 3.08, 2.45, 1.22, "Frozen predictions\nhash +\nremote seal", PALE_AMBER, AMBER),
        (5.85, 2.78, 3.25, 1.82, "SafeConf audit\nvalid inputs only:\ndisagreement\nmagnitude · support\n· dispersion", PALE_GREEN, GREEN),
        (9.50, 3.35, 2.15, 0.95, "Validated\nROUTE", PALE_GREEN, GREEN),
        (9.50, 2.05, 2.15, 0.95, "Unvalidated\nABSTAIN", PALE_RED, RED),
    ]
    for x, y, w, h, text, fc, ec in boxes:
        fs = 7.4 if "SafeConf" in text else (7.7 if "Frozen" in text else 8.0)
        rounded_box(ax, x, y, w, h, text, fc=fc, ec=ec, fs=fs)
    arrow(ax, (2.70, 3.69), (3.05, 3.69))
    arrow(ax, (5.50, 3.69), (5.85, 3.69))
    arrow(ax, (9.10, 3.82), (9.50, 3.82))
    arrow(ax, (9.10, 3.05), (9.50, 2.52))

    rounded_box(ax, 0.25, 0.52, 13.35, 0.88,
                "Truth-access order: train 16 models → seal predictions, risks and baselines\n"
                "→ commit to GitHub + Gitee → release target truth → adjudicate all four cell lines",
                fc=LIGHT, ec=GREY, fs=7.8)
    ax.text(0.25, 1.78,
            "Allowed before truth: control similarity, training support, frozen predictions and predicted magnitude.",
            fontsize=7.6, color=INK, ha="left")
    ax.text(0.25, 1.48,
            "Forbidden before truth: target perturbation expression.",
            fontsize=7.6, color=RED, ha="left")
    save(fig, "Fig01_contract")


def fig2_footprint() -> None:
    rows = [
        ("E189 random pair", "0.37–0.41", "0.24", "both positive", GREEN),
        ("E189 unseen column", "0.21–0.25", "−0.09", "risk signal only", GREEN),
        ("E189 unseen row", "−0.10–−0.01", "0.10*", "risk negative", RED),
        ("E189 double unseen", "−0.35–−0.24", "−0.08", "ABSTAIN", RED),
        ("E190 cross-study K562", "0.42", "0.42", "similar", GREY),
        ("E192 cross-study RPE1", "0.30 (CI×0)", "0.34 (CI×0)", "ABSTAIN", AMBER),
        ("E199 unseen genes · K562", "0.395 ✓", "0.096 (CI×0)", "disagreement valid", GREEN),
        ("E200 context holdout · K562", "0.424", "0.880 ✓", "magnitude dominates", AMBER),
        ("E158 strict unseen", "undefined", "undefined", "PRESCRIBE saturated", GREY),
        ("E201 four contexts", "SEALED", "SEALED", "evaluation pending", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE, 5.45))
    ax.axis("off")
    ax.text(0.0, 1.035, "Figure 2  |  A signal is usable only inside its validated setting",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color=NAVY, ha="left")
    x0, y0 = 0.01, 0.92
    widths = [0.34, 0.17, 0.17, 0.26]
    headers = ["Setting", "Risk signal\nSpearman ρ", "Magnitude\nSpearman ρ", "Interpretation"]
    xpos = np.cumsum([x0] + widths[:-1])
    for x, w, h in zip(xpos, widths, headers):
        ax.add_patch(plt.Rectangle((x, y0), w - 0.006, 0.055, facecolor=NAVY, edgecolor="white"))
        ax.text(x + (w - 0.006) / 2, y0 + 0.028, h, ha="center", va="center", color="white", fontsize=7.0)
    row_h = 0.075
    for i, (label, risk, mag, verdict, color) in enumerate(rows):
        y = y0 - (i + 1) * row_h
        base = "#FFFFFF" if i % 2 == 0 else "#F6F8FA"
        for x, w in zip(xpos, widths):
            ax.add_patch(plt.Rectangle((x, y), w - 0.006, row_h - 0.004, facecolor=base, edgecolor="white"))
        ax.text(xpos[0] + 0.01, y + row_h / 2, label, ha="left", va="center", fontsize=7.0)
        ax.text(xpos[1] + widths[1] / 2, y + row_h / 2, risk, ha="center", va="center", fontsize=7.0,
                color=GREEN if "✓" in risk or (risk.startswith("0.") and "CI×0" not in risk) else color)
        ax.text(xpos[2] + widths[2] / 2, y + row_h / 2, mag, ha="center", va="center", fontsize=7.0,
                color=AMBER if "0.880" in mag else (RED if mag.startswith("−") else INK))
        ax.text(xpos[3] + 0.01, y + row_h / 2, verdict, ha="left", va="center", fontsize=7.0, color=color)
    ax.text(0.01, 0.035, "* point estimate only; CI crossing 0 means the system must abstain. E201 is sealed, not a result.",
            transform=ax.transAxes, fontsize=7.2, color=GREY, ha="left")
    save(fig, "Fig02_validation_footprint")


def interval(ax, y, value, lo, hi, color, marker, label=None):
    ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt=marker, markersize=6,
                color=color, ecolor=color, elinewidth=1.5, capsize=3, label=label, zorder=3)


def fig3_flip() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.8), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.77, bottom=0.22, wspace=0.25)
    groups = [("E199 · K562 unseen genes", 1.35, (0.3948, 0.2835, 0.4969), (0.0955, -0.0256, 0.2187)),
              ("E200 · K562 context holdout", 0.55, (0.4240, 0.3506, 0.4953), (0.8797, 0.8437, 0.9095))]
    for ax, utility in zip(axes, [False, True]):
        ax.axvline(0, color=GREY, lw=0.8, ls="--")
        for name, y, risk, mag in groups:
            if utility:
                risk = (0.3648, 0.2356, 0.4813) if y < 1 else (0.2084, 0.1033, 0.3755)
                mag = (0.9133, 0.8748, 0.9520) if y < 1 else (0.0397, -0.0830, 0.2255)
            interval(ax, y + 0.13, *risk, GREEN, "o", "disagreement / risk" if y > 1 else None)
            interval(ax, y - 0.13, *mag, AMBER, "s", "predicted magnitude" if y > 1 else None)
        ax.set_yticks([1.35, 0.55])
        ax.set_yticklabels(["K562 unseen genes\nE199 · n=263", "K562 context holdout\nE200 · n=566"])
        ax.set_ylim(0.1, 1.8)
        ax.set_xlim(-0.15, 1.03)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("20% review utility (95% CI)" if utility else "Spearman ρ with task error (95% CI)")
    axes[0].set_title("a  Association with task error", loc="left", fontweight="bold")
    axes[1].set_title("b  Fixed 20% review budget", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Figure 3  |  The leading signal reverses across settings", fontsize=11,
                 fontweight="bold", color=NAVY, x=0.18, ha="left")
    save(fig, "Fig03_signal_flip")


def fig4_e201() -> None:
    fig = plt.figure(figsize=(DOUBLE, 4.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax.set_title("a  E201 training status", loc="left", fontweight="bold")
    ax.set_xlim(-0.8, 4.6); ax.set_ylim(-0.6, 4.9); ax.axis("off")
    targets = ["K562", "RPE1", "HepG2", "Jurkat"]
    for i, target in enumerate(targets):
        ax.text(-0.58, 3.75 - i, target, ha="right", va="center", fontsize=7.4)
        for j in range(4):
            ax.add_patch(plt.Rectangle((j, 3.45 - i), 0.78, 0.58, facecolor=PALE_GREEN,
                                       edgecolor=GREEN, linewidth=1.0))
            ax.text(j + 0.39, 3.74 - i, "80 ep", ha="center", va="center", fontsize=6.5, color=GREEN)
    ax.set_xticks([0.39, 1.39, 2.39, 3.39]); ax.set_xticklabels(["seed 1", "seed 2", "seed 3", "seed 4"], fontsize=7)
    ax.text(1.55, -0.2, "16 / 16 models trained and sealed", ha="center", fontsize=8.5, fontweight="bold", color=GREEN)
    ax.text(1.55, -0.5, "target perturbation expression accessed: 0 rows", ha="center", fontsize=7.2, color=RED)

    ax = fig.add_subplot(gs[1])
    ax.set_title("b  Frozen evaluation order", loc="left", fontweight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
    steps = [
        ("16/16 training", "complete", PALE_GREEN, GREEN),
        ("family seal", "complete", PALE_BLUE, BLUE),
        ("prediction", "K562 seed 1 done; 15 left", PALE_BLUE, BLUE),
        ("risk + baseline seal", "pending", LIGHT, GREY),
        ("target truth release", "blocked until seal", PALE_RED, RED),
        ("formal evaluation", "pending", LIGHT, GREY),
    ]
    y = 4.95
    for idx, (name, status, fc, ec) in enumerate(steps):
        rounded_box(ax, 0.55, y, 8.9, 0.58, f"{idx + 1}. {name}   |   {status}", fc=fc, ec=ec, fs=7.4)
        if idx < len(steps) - 1:
            arrow(ax, (5.0, y), (5.0, y - 0.28), color=GREY, lw=0.8)
        y -= 0.84
    ax.text(0.55, 0.05, "E201 figures must report status, not invent target-test curves.", fontsize=7.3, color=RED)
    fig.suptitle("Figure 4  |  E201 supplies the prospective whole-context test", fontsize=11,
                 fontweight="bold", color=NAVY, x=0.06, ha="left")
    save(fig, "Fig04_E201_status")


def fig5_mechanism() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.65), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    labels = ["E199\nD | M", "E199\nM | D", "E200\nM | R", "E200\nR | M"]
    values = [0.386, 0.005, 0.875, 0.301]
    colors = [GREEN, GREY, AMBER, BLUE]
    x = np.arange(4)
    ax.bar(x, values, color=colors, width=0.58, edgecolor=INK, linewidth=0.4)
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.025, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel("rank-partial correlation with task error")
    ax.set_title("a  The information axis switches", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    items = [("E199 Δρ", 0.299, 0.161, 0.432, GREEN), ("E199 Δutility", 0.169, -0.028, 0.353, GREEN),
             ("E200 Δρ", -0.456, -0.536, -0.378, RED), ("E200 Δutility", -0.548, -0.689, -0.434, RED)]
    y = np.arange(len(items))[::-1]
    for yi, (label, value, lo, hi, color) in zip(y, items):
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round")
        ax.plot(value, yi, "o", color=color, markersize=6)
        ax.text(-0.72, yi, label, ha="right", va="center", fontsize=7.1)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xlim(-0.75, 0.5); ax.set_yticks([])
    ax.set_xlabel("increment over predicted magnitude (95% CI)")
    ax.set_title("b  Fixed score cannot win both settings", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 5  |  Why the contract is conditional, not a universal score", fontsize=11,
                 fontweight="bold", color=NAVY, x=0.05, ha="left")
    save(fig, "Fig05_mechanism")


def main() -> None:
    fig1_contract()
    fig2_footprint()
    fig3_flip()
    fig4_e201()
    fig5_mechanism()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
