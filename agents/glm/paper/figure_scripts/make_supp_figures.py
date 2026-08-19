#!/usr/bin/env python3
"""Supplementary Figure S1: the abstention ledger (GLM, 2026-08-17).

Numbers verbatim from E202A_SETTING_COMPARISON.csv and E199/E200 frozen tables.
E158 saturation is drawn as a status ledger (no raw score values are published
in the summary table, so no synthetic curve is drawn).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OI = {"green": "#009E73", "orange": "#E69F00", "vermillion": "#D55E00",
      "grey": "#7F7F7F", "blue": "#0072B2"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.facecolor": "white",
})
MM = 1 / 25.4
DOUBLE = 183 * MM

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, DOUBLE * 0.26))

# (a) E192: preregistered abstention despite positive utility
ax = axes[0]
ax.set_title("a  E192 cross-study RPE1 (n=175)\ncorrelation CI crosses 0 → ABSTAIN", loc="left", fontweight="bold")
ax.plot([-0.040, 0.580], [1, 1], color=OI["orange"], lw=2, solid_capstyle="round")
ax.plot([0.300], [1], "o", color=OI["orange"], ms=5)
ax.axvline(0, color="#555555", lw=0.8, ls="--")
ax.set_yticks([1]); ax.set_yticklabels(["family diversity ρ"])
ax.set_xlim(-0.25, 0.85); ax.set_ylim(0.5, 1.6)
ax.plot([0.113, 0.872], [0.72, 0.72], color=OI["grey"], lw=1.4)
ax.plot([0.696], [0.72], "s", color=OI["grey"], ms=4)
ax.text(-0.22, 0.72, "20% utility\n(point est. positive)", fontsize=6.0, va="center")
ax.text(0.30, 1.38, "ABSTAIN\n(preregistered dual gate)", fontsize=6.6,
        ha="center", color=OI["vermillion"], fontweight="bold")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# (b) E189/E191 double-unseen: sourced numbers only
ax = axes[1]
ax.set_title("b  E189/E191 double-unseen tasks\nassociation flips negative; utilities below random", loc="left", fontweight="bold")
x = np.arange(3)
vals = [-0.295, -0.127, -0.080]
colors = [OI["vermillion"], OI["vermillion"], OI["grey"]]
ax.bar(x, vals, width=0.5, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
ax.errorbar([0], [-0.295], yerr=[[0.054], [0.054]], fmt="none",
            ecolor="black", elinewidth=0.9, capsize=2.5)
ax.axhline(0, color="#555555", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Spearman ρ, disagreement\n(E189, range −0.349..−0.241)",
                    "20% review utility,\ndisagreement (E191)",
                    "20% review utility,\nmagnitude (E191)"], fontsize=5.7)
ax.set_ylabel("value (negative = worse than random)")
for xi, v in zip(x, vals):
    ax.text(xi, v - 0.035, f"{v:.3f}", ha="center", fontsize=6.2)
ax.set_ylim(-0.45, 0.10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (c) E158/E159: score saturation ledger (no synthetic curves)
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("c  E158/E159 strict unseen genes (Norman P3/P4)\ncompetitor scores saturate → undefined", loc="left", fontweight="bold")
rows = [
    ("PRESCRIBE official combined", "1 distinct value in panel", "ρ undefined", OI["grey"]),
    ("PRESCRIBE epistemic", "1 distinct value in panel", "ρ undefined", OI["grey"]),
    ("predicted magnitude (same panel)", "also saturated on this panel", "ρ undefined", OI["grey"]),
]
y = 6.4
for label, obs, verdict, color in rows:
    ax.add_patch(FancyBboxPatch((0.2, y - 1.2), 9.6, 1.5, boxstyle="round,pad=0.1",
                 fc="white", ec=color, lw=0.8))
    ax.text(0.5, y - 0.45, label, fontsize=6.4, va="center")
    ax.text(5.4, y - 0.45, obs, fontsize=6.0, va="center", color="#555555")
    ax.text(8.0, y - 0.45, verdict, fontsize=6.2, va="center", color=color, fontweight="bold")
    y -= 2.0
ax.text(5.0, 1.6, "correct system behaviour: ABSTAIN;\ncorrect scientific statement: predictor-intrinsic confidence\ndid not survive this shift — not 'we won'",
        fontsize=6.2, ha="center", style="italic", linespacing=1.4)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"FigS1_abstention_ledger.{ext}", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "FigS1_abstention_ledger")
