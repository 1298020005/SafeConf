#!/usr/bin/env python3
"""SafeConf paper figures v1 (GLM, 2026-08-17).

All numbers are copied verbatim from frozen evaluation tables:
  E199: docs/实验结果/E199_txpert_public_k562_20260802/formal_evaluation/tables/
        E199_RISK_ASSOCIATIONS.csv, E199_REVIEW_UTILITY.csv
  E200: docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/
        E200_RISK_ASSOCIATIONS.csv, E200_REVIEW_UTILITY.csv
  E192/E189/E158/E201 summary: docs/实验结果/E202_q1_blocker_closure_20260815/tables/
        E202A_SETTING_COMPARISON.csv
No E201 evaluation numbers exist yet (blind); E201 panels are explicit placeholders.
Output: agents/glm/paper/figures/*.png (300 dpi, white background) + .pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "reddish": "#CC79A7", "grey": "#7F7F7F"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 8.0,
    "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
})
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ============================= Figure 1: concept =============================
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, DOUBLE * 0.36))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    def box(ax, x, y, w, h, text, fc, ec, fs=6.6, tc="black", lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                     fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.3)

    def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=9, color=color, lw=lw, linestyle=ls,
                     shrinkA=1, shrinkB=1))

    # (a) deployment scenario
    ax = axes[0]
    ax.set_title("a  Post-prediction audit,\n    not another predictor", loc="left", fontweight="bold")
    box(ax, 0.3, 9.2, 4.2, 2.3, "Perturbation predictors\n(TxPert STRING-GAT ×4 seeds;\nscGPT/GEARS family)", "#EAF3FB", OI["blue"])
    box(ax, 5.0, 9.2, 4.7, 2.3, "Frozen prediction\nrelease (hashed,\ndual-remote)", "#FFF7E6", OI["orange"])
    box(ax, 0.3, 5.6, 9.4, 2.6, "SafeConf audit — deployment-time signals only:\nfamily disagreement · predicted magnitude\nsource support · source-effect dispersion · context deficit", "#EBF7F1", OI["green"], fs=6.4)
    box(ax, 0.3, 2.6, 4.4, 2.0, "ROUTE\nrank tasks for review\nor wet-lab\n(validated settings)", "#EBF7F1", OI["green"])
    box(ax, 5.3, 2.6, 4.4, 2.0, "ABSTAIN\nsignal unvalidated,\ndegraded or saturated\n(fail closed)", "#FDEBE6", OI["vermillion"])
    box(ax, 0.3, 0.3, 9.4, 1.5, "Target truth stays sealed until predictions, risk table\nand baselines are committed to GitHub + Gitee", "#F5F5F5", OI["grey"], fs=6.2)
    arrow(ax, 2.5, 9.2, 2.5, 8.2); arrow(ax, 4.5, 10.35, 5.0, 10.35)
    arrow(ax, 2.8, 5.6, 2.5, 4.6); arrow(ax, 7.2, 5.6, 7.5, 4.6)
    arrow(ax, 2.5, 2.6, 2.5, 1.8, ls=":")

    # (b) three pre-registered gates
    ax = axes[1]
    ax.set_title("b  Three separated\n    adjudication gates", loc="left", fontweight="bold")
    gates = [
        ("Certificate gate", "family RMS² = centroid RMSE² + disagreement²\nidentity → integrity check, not a discovery", "#EAF3FB", OI["blue"]),
        ("Routing gate", "risk–error pooled Spearman CI lower bound > 0\nAND 20% review-utility CI lower bound > 0", "#EBF7F1", OI["green"]),
        ("Magnitude-increment gate", "partial Spearman | magnitude > 0\nOR paired utility increment > 0", "#FFF7E6", OI["orange"]),
    ]
    y = 9.6
    for name, body, fc, ec in gates:
        box(ax, 0.3, y - 2.3, 9.4, 2.6, "", fc, ec)
        ax.text(5.0, y + 0.1, name, ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.text(5.0, y - 1.0, body, ha="center", va="center", fontsize=6.1, linespacing=1.4)
        y -= 3.2
    ax.text(5.0, 1.1, "fail any gate → negative result retained;\nno re-tuning of seeds, weights, tasks or metrics",
            ha="center", va="center", fontsize=6.2, style="italic", color=OI["vermillion"], linespacing=1.4)

    # (c) validation footprint
    ax = axes[2]
    ax.set_title("c  Validation footprint\n    of every signal", loc="left", fontweight="bold")
    rows = [
        ("K562 unseen genes (E199)", "VALID · magnitude not (CI crosses 0)", OI["green"]),
        ("K562 context hold-out (E200)", "magnitude DOMINATES (ρ = 0.88)", OI["vermillion"]),
        ("Cross-study RPE1 (E192)", "ABSTAIN (preregistered)", OI["orange"]),
        ("Strict unseen genes (E158)", "saturated → undefined", OI["grey"]),
        ("Double unseen (E189/E191)", "NEGATIVE assoc. & utility", OI["vermillion"]),
        ("4-context blind test (E201)", "SEALED — pending", OI["blue"]),
    ]
    y = 9.9
    for label, status, color in rows:
        box(ax, 0.2, y - 0.95, 4.7, 1.45, label, "white", "#999999", fs=6.1)
        box(ax, 5.1, y - 0.95, 4.7, 1.45, status, "white", color, fs=6.0, tc=color)
        y -= 1.75
    ax.text(5.0, 0.6, "the setting decides whether a\nsignal may be used at all",
            ha="center", fontsize=6.3, style="italic", linespacing=1.4)
    save(fig, "Fig1_concept_contract")


# ===================== Figure 2: forest plot across settings =====================
def fig2():
    # (label, rho, lo, hi, status, status_color, group)
    rows = [
        ("K562 unseen genes · E199 · n=263", [
            ("family disagreement (diversity lb)", 0.3948, 0.2835, 0.4969, "ROUTING SUPPORTED", OI["green"]),
            ("predicted magnitude", 0.0955, -0.0256, 0.2187, "CI crosses 0", OI["grey"]),
            ("graph-isolated risk", -0.1067, -0.1961, -0.0029, "negative assoc.", OI["vermillion"]),
        ]),
        ("K562 full context hold-out · E200 · n=566", [
            ("transfer risk (SafeConf)", 0.4240, 0.3506, 0.4953, "correlates", OI["green"]),
            ("predicted magnitude", 0.8797, 0.8437, 0.9095, "DOMINATES", OI["orange"]),
            ("source-effect dispersion", 0.6639, 0.6075, 0.7143, "valid, weaker", OI["sky"]),
            ("model–baseline gap", 0.1597, 0.0751, 0.2415, "weak", OI["grey"]),
        ]),
        ("Cross-study RPE1 · E192 · n=175", [
            ("family diversity", 0.300, -0.040, 0.580, "PREREGISTERED ABSTAIN", OI["orange"]),
        ]),
        ("Strict unseen genes · E158 · n=48", [
            ("PRESCRIBE official combined", np.nan, np.nan, np.nan, "UNDEFINED — score saturated", OI["grey"]),
            ("PRESCRIBE epistemic", np.nan, np.nan, np.nan, "UNDEFINED — score saturated", OI["grey"]),
        ]),
        ("Double unseen · E189/E191", [
            ("disagreement–error association (Spearman)", np.nan, np.nan, np.nan, "NEGATIVE ρ −0.349..−0.241; utility −0.127 — abstain", OI["vermillion"]),
        ]),
        ("Four contexts leave-one-out · E201 · n=2008", [
            ("SafeConf risk / magnitude", np.nan, np.nan, np.nan, "SEALED — evaluation pending", OI["blue"]),
        ]),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.78, DOUBLE * 0.62))
    y = 0
    yticks, ylabels, group_spans = [], [], []
    for gname, items in rows:
        gstart = y
        for (lab, rho, lo, hi, status, color) in items:
            y += 1
            yticks.append(y); ylabels.append(lab)
            if not np.isnan(rho):
                ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round")
                ax.plot([rho], [y], "o", color=color, ms=4.5)
            else:
                ax.text(0.02, y, status, va="center", fontsize=6.2, color=color, style="italic")
                continue
            ax.text(1.06, y, status, va="center", fontsize=6.2, color=color)
        group_spans.append((gname, gstart + 1, y))
        y += 0.6
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_xlim(-0.35, 1.42); ax.set_ylim(0.3, y + 0.4)
    ax.set_xlabel("Spearman ρ vs task error (95% cluster-bootstrap CI)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for gname, y1, y2 in group_spans:
        ax.text(-0.33, y2 + 0.35, gname, fontsize=6.8, fontweight="bold", color="#222222")
        ax.axhline(y1 - 0.3, color="#DDDDDD", lw=0.5)
    ax.set_title("Signal validity flips with the evaluation setting — hence a fail-closed contract",
                 loc="left", fontweight="bold")
    save(fig, "Fig2_setting_forest")


# ===================== Figure 3: the flip, review utility =====================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.72, DOUBLE * 0.30))
    panels = [
        ("a  E199 · K562 unseen genes (n=263)\nmagnitude fails, disagreement works", [
            ("predicted\nmagnitude", 0.0397, -0.0830, 0.2255, OI["grey"]),
            ("family\ndisagreement", 0.2084, 0.1033, 0.3755, OI["green"]),
        ]),
        ("b  E200 · K562 full context hold-out (n=566)\nmagnitude dominates every risk signal", [
            ("transfer risk\n(SafeConf)", 0.3648, 0.2356, 0.4813, OI["green"]),
            ("source-effect\ndispersion", 0.6483, 0.5441, 0.7339, OI["sky"]),
            ("predicted\nmagnitude", 0.9133, 0.8748, 0.9520, OI["orange"]),
        ]),
    ]
    for ax, (title, bars) in zip(axes, panels):
        labels = [b[0] for b in bars]
        vals = [b[1] for b in bars]; los = [b[2] for b in bars]; his = [b[3] for b in bars]
        colors = [b[4] for b in bars]
        x = np.arange(len(bars))
        ax.bar(x, vals, width=0.55, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                    fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("20% review-budget utility\n(normalized, 95% CI)")
        ax.set_title(title, loc="left", fontweight="bold")
        for xi, v in zip(x, vals):
            ax.text(xi, max(his) + 0.05, f"{v:.3f}", ha="center", fontsize=6.4)
        ax.set_ylim(min(0, min(los) - 0.08), 1.08)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    save(fig, "Fig3_utility_flip")


# ===================== Figure 4: E201 design + placeholders =====================
def fig4():
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.34))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.25], wspace=0.28)

    # (a) 4x4 training grid, live status
    ax = fig.add_subplot(gs[0])
    ax.set_title("a  E201 blind training grid\n(leave-one-cell-line-out × 4 seeds)", loc="left", fontweight="bold")
    targets = ["K562", "RPE1", "HepG2", "Jurkat"]
    done = {(i, j) for i in range(4) for j in range(4)} - {(3, 3)}
    for i in range(4):
        for j in range(4):
            fc = OI["green"] if (i, j) in done else OI["yellow"]
            ax.add_patch(plt.Rectangle((j, 3 - i), 0.92, 0.92, fc=fc, ec="black", lw=0.6, alpha=0.85))
    ax.text(3.46, 3 - 3 + 0.46, "running\n(2026-08-17)", fontsize=5.8, va="center")
    ax.set_xticks(np.arange(4) + 0.46); ax.set_xticklabels(["s1", "s2", "s3", "s4"])
    ax.set_yticks(np.arange(4) + 0.46); ax.set_yticklabels(targets[::-1])
    ax.set_xlim(-0.05, 4.6); ax.set_ylim(-0.05, 4.0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(2.3, -0.55, "each cell = 80-epoch TxPert STRING-GAT retraining;\ntarget perturbed expression accessed 0 rows",
            ha="center", fontsize=6.0, style="italic")

    # (b) frozen release pipeline
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("b  Sealed release pipeline (frozen 2026-08-02)", loc="left", fontweight="bold")
    steps = [
        ("16/16 training complete", "#EBF7F1", OI["green"]),
        ("family seal (16 checkpoints, SHA-256)", "#EAF3FB", OI["blue"]),
        ("zero-truth predictions (batch.x ≡ 0 audited)", "#EAF3FB", OI["blue"]),
        ("risk features + general baseline\n+ E200 equivalence ≤ 5e-6", "#EAF3FB", OI["blue"]),
        ("commit to GitHub + Gitee (commitment point)", "#FFF7E6", OI["orange"]),
        ("release target truth (irreversible)", "#FDEBE6", OI["vermillion"]),
        ("frozen evaluation: 3 gates × 4 targets", "#F5F5F5", OI["grey"]),
    ]
    y = 9.0
    for i, (text, fc, ec) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((0.3, y - 1.05), 9.2, 1.05,
                     boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=0.8))
        ax.text(4.9, y - 0.52, text, ha="center", va="center", fontsize=6.6, linespacing=1.25)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((4.9, y - 1.1), (4.9, y - 1.35),
                         arrowstyle="-|>", mutation_scale=8, color="#333333", lw=0.9))
        y -= 1.35

    # (c) placeholder result panels
    axc = fig.add_subplot(gs[2])
    axc.set_title("c  Four-target adjudication —\nresults sealed until pipeline completes", loc="left", fontweight="bold")
    sub = axc.inset_axes([0.02, 0.30, 0.97, 0.66])
    sub.set_xlim(0, 2); sub.set_ylim(0, 2)
    for k, t in enumerate(targets):
        px, py = (k % 2) * 1.0 + 0.08, (k // 2) * 1.0 + 0.08
        sub.add_patch(plt.Rectangle((px, py), 0.84, 0.84, fc="#F7F7F7", ec=OI["blue"], lw=0.8, ls="--"))
        sub.text(px + 0.42, py + 0.42, t, ha="center", va="center", fontsize=7.5, color=OI["blue"])
        sub.text(px + 0.42, py + 0.18, "blind", ha="center", fontsize=5.8, color=OI["grey"], style="italic")
    sub.axis("off")
    axc.text(0.5, 0.12, "main analysis: 1,808 tasks (≥30 cells) of 2,008 total;\ncluster bootstrap 5,000 × by perturbation condition;\nnegative results will be reported in full",
             ha="center", fontsize=6.2, linespacing=1.4, transform=axc.transAxes)
    axc.axis("off")
    save(fig, "Fig4_E201_design")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("all figures done ->", OUT)
