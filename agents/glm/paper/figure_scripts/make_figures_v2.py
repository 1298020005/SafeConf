#!/usr/bin/env python3
"""Alternative versions (V2) of main figures 1-4. GLM 2026-08-17.
Same frozen numbers as make_figures.py; different layouts so the author and
supervisor can choose. V1 files remain untouched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OI = {"green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "blue": "#0072B2",
      "vermillion": "#D55E00", "grey": "#7F7F7F", "yellow": "#F0E442"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.facecolor": "white",
})
MM = 1 / 25.4
DOUBLE = 183 * MM


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, text, fc, ec, fs=6.5, tc="black", lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, lw=1.0, ls="-", color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                 color=color, lw=lw, linestyle=ls, shrinkA=1, shrinkB=1))


# ─────────────── Fig1 V2: single-row horizontal flow ───────────────
def fig1_v2():
    fig, ax = plt.subplots(figsize=(DOUBLE, DOUBLE * 0.30))
    ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.2, 5.2, 2.9, 2.6, "Perturbation\npredictors\n(TxPert ×4 seeds;\nscGPT/GEARS)", "#EAF3FB", OI["blue"])
    box(ax, 3.6, 5.2, 2.9, 2.6, "Frozen prediction\nrelease\n(hashed, dual-\nremote)", "#FFF7E6", OI["orange"])
    box(ax, 7.0, 3.6, 5.6, 5.6, "SafeConf audit\ndeployment-time signals only:\n· family disagreement\n· predicted magnitude\n· source support / dispersion\n· context deficit", "#EBF7F1", OI["green"], fs=6.6)
    box(ax, 13.0, 6.1, 3.2, 3.1, "3 pre-registered gates\ncertificate · routing\nmagnitude-increment", "#F5F5F5", OI["grey"], fs=6.2)
    box(ax, 13.0, 3.7, 1.5, 1.7, "ROUTE\n(validated)", "#EBF7F1", OI["green"], fs=6.2)
    box(ax, 14.7, 3.7, 1.5, 1.7, "ABSTAIN\n(fail closed)", "#FDEBE6", OI["vermillion"], fs=6.2)
    box(ax, 0.2, 0.4, 16.0, 2.2, "Truth-access timeline (frozen):   train 16 models (target expression zeroed)   →   seal checkpoints + predictions + risk + baseline\n→   commit to GitHub + Gitee (commitment point)   →   only then release target truth   →   three-gate adjudication, all four cell lines reported", "#F5F5F5", OI["grey"], fs=6.0)
    arrow(ax, 3.1, 6.5, 3.6, 6.5); arrow(ax, 6.5, 6.5, 7.0, 6.5)
    arrow(ax, 12.6, 7.6, 13.0, 7.6)
    arrow(ax, 14.6, 6.1, 14.0, 5.4); arrow(ax, 14.9, 6.1, 15.4, 5.4)
    ax.text(10.0, 9.7, "SafeConf: post-prediction, fail-closed reliability contract — V2 (horizontal flow)",
            ha="center", fontsize=8.4, fontweight="bold")
    save(fig, "Fig1V2_horizontal_flow")


# ─────────────── Fig2 V2: verdict heatmap ───────────────
def fig2_v2():
    rows = [
        "E189 random pair", "E189 unseen column", "E189 unseen row", "E189 double unseen",
        "E190 cross-study K562", "E192 cross-study RPE1",
        "E199 unseen genes (K562)", "E200 context holdout (K562)",
        "E158 strict unseen (PRESCRIBE)", "E201 four contexts (sealed)",
    ]
    cols = ["disagreement /\nrisk score", "predicted\nmagnitude", "source-effect\ndispersion", "model–baseline\ngap", "PRESCRIBE\nofficial"]
    # codes: 2=VALID(CI>0), 1=weak/point, 0=degraded CI×0, -1=negative, -2=undefined(saturated), -3=sealed
    M = np.array([
        [2, 2, -9, -9, -9],   # random: both positive (point estimates)
        [2, -1, -9, -9, -9],  # column: div +, mag −
        [-1, 1, -9, -9, -9],  # row: div −, mag weak+
        [-1, -1, -9, -9, -9], # double: both negative → ABSTAIN
        [2, 2, 2, -9, -9],    # E190: div/mag/src-mag all ≈0.44
        [0, 0, -9, -9, -9],   # E192: both correlation CIs cross 0 → ABSTAIN
        [2, 0, -9, 0, -9],    # E199: div VALID, mag CI×0, gap weak
        [2, 2, 2, 2, -9],     # E200: mag dominates; all four valid
        [-9, -9, -9, -9, -2], # E158: PRESCRIBE saturated
        [-3, -3, -3, -3, -3], # E201 sealed
    ])
    rho = [
        ["0.37–0.41", "0.24", "", "", ""],
        ["0.21–0.25", "−0.09", "", "", ""],
        ["−0.10–−0.01", "0.10*", "", "", ""],
        ["−0.35–−0.24", "−0.08", "", "", ""],
        ["0.42", "0.42", "0.47", "", ""],
        ["0.30 (CI×0)", "0.34 (CI×0)", "", "", ""],
        ["0.395 ✓", "0.096 (CI×0)", "", "0.16", ""],
        ["0.424 ✓", "0.880 ✓", "0.664 ✓", "0.16", ""],
        ["", "", "", "", "saturated"],
        ["SEALED", "SEALED", "SEALED", "SEALED", ""],
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.72, DOUBLE * 0.46))
    cmap = {-1: "#D55E00", 0: "#BFBFBF", 1: "#E8D06A", 2: "#009E73",
            -2: "#666666", -3: "#8FB8D8", -9: "#F2F2F2"}
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.add_patch(plt.Rectangle((j, len(rows) - 1 - i), 0.96, 0.96,
                         fc=cmap[M[i, j]], ec="white", lw=1.2))
            t = rho[i][j]
            if t:
                ax.text(j + 0.48, len(rows) - 1 - i + 0.48, t, ha="center", va="center",
                        fontsize=5.5, color="black")
    ax.set_xticks(np.arange(len(cols)) + 0.48); ax.set_xticklabels(cols, fontsize=6.2)
    ax.set_yticks(np.arange(len(rows)) + 0.48); ax.set_yticklabels(rows[::-1], fontsize=6.2)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rows))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap[k]) for k in (2, 1, 0, -1, -2, -3)]
    ax.legend(handles, ["VALID (CI>0)", "positive point only*", "degraded (CI×0) → ABSTAIN",
                        "negative → ABSTAIN", "undefined (saturated)", "sealed (E201)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=5.9, frameon=False)
    ax.set_title("Validation footprint heatmap — signal validity by setting (V2)\nρ values where defined; * = point estimate only (E191 setting means)",
                 loc="left", fontweight="bold")
    save(fig, "Fig2V2_footprint_heatmap")


# ─────────────── Fig3 V2: dumbbell flip ───────────────
def fig3_v2():
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.55, DOUBLE * 0.26))
    sets = [
        ("E199 · K562 unseen genes (n=263)", 0.208, 0.103, 0.376, 0.040, -0.083, 0.226),
        ("E200 · K562 context holdout (n=566)", 0.365, 0.236, 0.481, 0.913, 0.875, 0.952),
    ]
    for i, (name, dv, dl, dh, mv, ml, mh) in enumerate(sets):
        y = 1 - i
        ax.plot([dl, dh], [y + 0.13, y + 0.13], color=OI["green"], lw=1.6)
        ax.plot([dv], [y + 0.13], "o", color=OI["green"], ms=5, label="disagreement / risk score" if i == 0 else None)
        ax.plot([ml, mh], [y - 0.13, y - 0.13], color=OI["orange"], lw=1.6)
        ax.plot([mv], [y - 0.13], "s", color=OI["orange"], ms=5, label="predicted magnitude" if i == 0 else None)
        winner = "disagreement wins" if dv > mv else "magnitude dominates"
        ax.text(max(dh, mh) + 0.06, y, winner, fontsize=6.6, va="center",
                color=OI["green"] if dv > mv else OI["orange"], fontweight="bold")
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks([1, 0]); ax.set_yticklabels([s[0] for s in sets], fontsize=6.8)
    ax.set_xlabel("20% review-budget utility (95% cluster-bootstrap CI)")
    ax.set_xlim(-0.15, 1.35); ax.set_ylim(-0.55, 1.55)
    ax.legend(loc="lower right", fontsize=6.2, frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("The winner flips with the setting (V2 dumbbell)", loc="left", fontweight="bold")
    save(fig, "Fig3V2_dumbbell_flip")


# ─────────────── Fig4 V2: complete version (simulation preview in c) ───────────────
def fig4_v2():
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.36))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.3], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    targets = ["K562", "RPE1", "HepG2", "Jurkat"]
    for i in range(4):
        for j in range(4):
            fc = OI["green"] if (i, j) != (3, 3) else OI["yellow"]
            ax.add_patch(plt.Rectangle((j, 3 - i), 0.92, 0.92, fc=fc, ec="black", lw=0.6, alpha=0.85))
    ax.text(3.46, 0.46, "running\n08-17", fontsize=5.6, va="center")
    ax.set_xticks(np.arange(4) + 0.46); ax.set_xticklabels(["s1", "s2", "s3", "s4"])
    ax.set_yticks(np.arange(4) + 0.46); ax.set_yticklabels(targets[::-1])
    ax.set_xlim(-0.05, 4.6); ax.set_ylim(-0.6, 4.0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("a  Blind training grid\n(leave-one-context-out × 4 seeds)", loc="left", fontweight="bold")
    ax.text(2.3, -0.45, "target perturbed expression accessed 0 rows", ha="center", fontsize=5.9, style="italic")

    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    ax.set_title("b  Sealed release pipeline\n(frozen 2026-08-02)", loc="left", fontweight="bold")
    steps = [("16/16 training complete", "#EBF7F1", OI["green"]),
             ("family seal (SHA-256)", "#EAF3FB", OI["blue"]),
             ("zero-truth predictions", "#EAF3FB", OI["blue"]),
             ("risk features + baseline\n(E200 equivalence ≤5e-6)", "#EAF3FB", OI["blue"]),
             ("commit GitHub + Gitee", "#FFF7E6", OI["orange"]),
             ("release truth (irreversible)", "#FDEBE6", OI["vermillion"]),
             ("3-gate adjudication ×4 targets", "#F5F5F5", OI["grey"])]
    y = 10.8
    for i, (t, fc, ec) in enumerate(steps):
        box(ax, 0.3, y - 1.3, 9.4, 1.3, t, fc, ec, fs=6.3)
        if i < 6:
            arrow(ax, 5.0, y - 1.35, 5.0, y - 1.6)
        y -= 1.62

    axc = fig.add_subplot(gs[2])
    strategies = ["always\ndisagreement", "always\nmagnitude", "conditional\ncontract"]
    means = [0.262, 0.254, 0.232]
    mins = [-0.127, -0.080, 0.0]
    colors = [OI["sky"], OI["orange"], OI["green"]]
    x = np.arange(3)
    axc.bar(x, means, width=0.5, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
    axc.scatter(x, mins, marker="_", s=200, color=OI["vermillion"], lw=1.8, zorder=3)
    axc.axhline(0, color="#555555", lw=0.8)
    axc.set_xticks(x); axc.set_xticklabels(strategies, fontsize=6.2)
    axc.set_ylabel("20% review utility")
    for xi, m in zip(x, means):
        axc.text(xi, m + 0.012, f"{m:.3f}", ha="center", fontsize=6.2)
    axc.set_ylim(-0.18, 0.40)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.set_title("c  Why a contract, not a fixed score\n(mean vs worst setting across 9 frozen settings;\nillustrative — prospective test = E201, sealed)",
                  loc="left", fontweight="bold")
    save(fig, "Fig4V2_E201_complete_preview")


if __name__ == "__main__":
    fig1_v2(); fig2_v2(); fig3_v2(); fig4_v2()
    print("V2 figures done ->", OUT)
