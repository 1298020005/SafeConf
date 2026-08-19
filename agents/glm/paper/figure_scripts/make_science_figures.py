#!/usr/bin/env python3
"""Science figures: component footprint (Fig5), incremental+mechanism (Fig6),
conditional-contract simulation (Fig7). GLM 2026-08-17.

Sources (all frozen tables or GLM-recomputed values documented in
agents/glm/08_SCIENCE_CRITIQUE_AND_FIXES.md):
- E199_RISK_ASSOCIATIONS.csv / E199_REVIEW_UTILITY.csv (n=263)
- E200_RISK_ASSOCIATIONS.csv / E200_REVIEW_UTILITY.csv (n=566)
- E199/E200_INCREMENTAL_TESTS.csv (paired deltas)
- E191_INTERPRETATION.md §3, E189_INTERPRETATION.md §4 (setting utilities)
- E192_RISK_ASSOCIATIONS.csv / E192_BUDGET_UTILITY.csv (gate data)
- partial correlations & composites & simulation: GLM recomputation (see 08 doc)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OI = {"green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9", "blue": "#0072B2",
      "vermillion": "#D55E00", "grey": "#7F7F7F", "yellow": "#F0E442"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.8,
    "axes.labelsize": 7.0, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.linewidth": 0.6,
})
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def verdict_color(lo, hi):
    if lo > 0:
        return OI["green"]
    if hi < 0:
        return OI["vermillion"]
    return OI["grey"]


# ───────────────────────── Fig5: component footprint ─────────────────────────
def fig5():
    e199 = [  # label, rho, lo, hi, util, ulo, uhi
        ("family disagreement (lb)", 0.3948, 0.2835, 0.4969, 0.2084, 0.1033, 0.3755),
        ("predicted magnitude", 0.0955, -0.0256, 0.2187, 0.0397, -0.0830, 0.2255),
        ("model–baseline gap", -0.0064, -0.1332, 0.1185, -0.0150, -0.1278, 0.1781),
        ("STRING-neighbor count (−)", -0.0822, -0.2052, 0.0388, -0.0835, -0.2408, 0.0418),
        ("GO-neighbor count (−)", -0.1018, -0.2200, 0.0165, -0.1537, -0.2578, 0.0024),
        ("graph-isolated flag", -0.1067, -0.1961, -0.0029, -0.2500, -0.1790, 0.1222),
    ]
    e200 = [
        ("predicted magnitude", 0.8797, 0.8437, 0.9095, 0.9133, 0.8748, 0.9520),
        ("source-effect dispersion", 0.6639, 0.6075, 0.7143, 0.6483, 0.5441, 0.7339),
        ("transfer risk (5-comp score)", 0.4240, 0.3506, 0.4953, 0.3648, 0.2356, 0.4813),
        ("neg. log source cells", 0.2149, 0.1309, 0.2955, 0.1913, 0.0670, 0.3180),
        ("model–baseline gap", 0.1597, 0.0751, 0.2415, 0.2468, 0.1183, 0.3641),
        ("support-context deficit", 0.0170, -0.0662, 0.1034, 0.0344, -0.0476, 0.2046),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.8, DOUBLE * 0.42))
    for ax, rows, title in (
        (axes[0], e199, "a  E199 · K562 unseen genes (n=263)"),
        (axes[1], e200, "b  E200 · K562 full-context holdout (n=566)"),
    ):
        labels = [r[0] for r in rows][::-1]
        y = np.arange(len(rows))
        ax.barh(y, [r[1] for r in rows][::-1], xerr=[np.array([r[1]-r[2] for r in rows][::-1]),
                np.array([r[3]-r[1] for r in rows][::-1])],
                color=[verdict_color(r[2], r[3]) for r in rows][::-1], alpha=0.85,
                edgecolor="black", lw=0.5, error_kw=dict(elinewidth=0.8, capsize=2))
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlabel("Spearman ρ with task error (95% CI)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel("20% review utility →", color=OI["blue"], fontsize=6.4)
        for yi, r in zip(y, rows[::-1]):
            ax2.plot([r[4]], [yi], "d", color=OI["blue"], ms=3.5)
            ax2.plot([r[5], r[6]], [yi, yi], color=OI["blue"], lw=0.8, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Component footprint: no component is valid everywhere; the composite is diluted by weak components",
                 x=0.02, ha="left", fontsize=8.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "Fig5_component_footprint")


# ─────────────────── Fig6: incremental + mechanism + composite ───────────────────
def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, DOUBLE * 0.32))

    # (a) paired increments vs magnitude
    ax = axes[0]
    ax.set_title("a  Paired increment of the risk score\nover predicted magnitude", loc="left", fontweight="bold")
    items = [
        ("E199 Δρ", 0.299, 0.161, 0.432, OI["green"]),
        ("E199 Δutility", 0.169, -0.028, 0.353, OI["green"]),
        ("E200 Δρ", -0.456, -0.536, -0.378, OI["vermillion"]),
        ("E200 Δutility", -0.548, -0.689, -0.434, OI["vermillion"]),
    ]
    for i, (lab, v, lo, hi, c) in enumerate(items):
        ax.plot([lo, hi], [i, i], color=c, lw=1.8, solid_capstyle="round")
        ax.plot([v], [i], "o", color=c, ms=4.5)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel("paired difference (95% cluster-bootstrap CI)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) mechanism: information axis switches
    ax = axes[1]
    ax.set_title("b  The information axis switches\n(rank-partial correlations)", loc="left", fontweight="bold")
    groups = [
        ("E199\nunseen genes", [("disagreement\n| magnitude", 0.386, OI["green"]),
                                 ("magnitude\n| disagreement", 0.005, OI["grey"])]),
        ("E200\ncontext holdout", [("magnitude\n| risk score", 0.875, OI["orange"]),
                                     ("risk score\n| magnitude", 0.301, OI["sky"])]),
    ]
    x = np.arange(2); w = 0.36
    for gi, (gname, bars) in enumerate(groups):
        for bi, (lab, v, c) in enumerate(bars):
            xi = x[gi] + (bi - 0.5) * w
            ax.bar(xi, v, width=w * 0.9, color=c, alpha=0.85, edgecolor="black", lw=0.5)
            ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=6.2)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=6.6)
    ax.set_ylabel("partial ρ with error (controlling the other signal)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c) composites dilute magnitude where it dominates
    ax = axes[2]
    ax.set_title("c  Fixed composites cannot win both\n(exploratory; not for E201 main analysis)", loc="left", fontweight="bold")
    bars = [
        ("magnitude", 0.916, OI["orange"]), ("mag+risk", 0.822, OI["grey"]),
        ("mag+disp", 0.890, OI["grey"]), ("risk score", 0.363, OI["sky"]),
        ("divergence", 0.208, OI["green"]), ("div+mag", 0.249, OI["grey"]),
        ("magnitude", 0.040, OI["orange"]),
    ]
    xpos = [0, 1, 2, 3, 4.6, 5.6, 6.6]
    for xi, (lab, v, c) in zip(xpos, bars):
        ax.bar(xi, v, width=0.75, color=c, alpha=0.85, edgecolor="black", lw=0.5)
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.0)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([b[0] for b in bars], rotation=32, ha="right", fontsize=6.0)
    ax.set_ylabel("20% review utility")
    ax.text(1.5, 1.02, "E200 (n=566)", fontsize=6.6, ha="center", fontweight="bold")
    ax.text(5.6, 1.02, "E199 (n=263)", fontsize=6.6, ha="center", fontweight="bold")
    ax.axvline(3.8, color="#CCCCCC", lw=0.6)
    ax.set_ylim(0, 1.12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "Fig6_increment_mechanism")


# ─────────────────── Fig7: conditional-contract simulation ───────────────────
def fig7():
    settings = ["E189 random pair", "E189 unseen column", "E189 unseen row",
                "E189 double unseen", "E190 cross-study K562", "E192 cross-study RPE1 (locked)",
                "E199 unseen genes", "E200 context holdout", "E158 strict unseen (PRESCRIBE)"]
    contract = [0.310, 0.117, 0.101, 0.0, 0.443, 0.0, 0.208, 0.913, 0.0]
    always_div = [0.310, 0.117, 0.046, -0.127, 0.443, 0.696, 0.208, 0.365, 0.0]
    always_mag = [0.239, -0.090, 0.101, -0.080, 0.441, 0.725, 0.040, 0.913, 0.0]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(DOUBLE * 0.82, DOUBLE * 0.44),
                                  gridspec_kw={"width_ratios": [2.1, 1.0]})
    # (a) matrix
    data = np.array([always_div, always_mag, contract])
    ylabels = ["always disagreement\n(fixed score, no footprint)",
               "always magnitude\n(fixed score, no footprint)",
               "conditional contract\n(best gate-passing signal; else ABSTAIN=0)"]
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.35, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([s.replace(" (locked)", "\n(locked)").replace(" (PRESCRIBE)", "\n(PRESCRIBE)") for s in settings],
                       rotation=38, ha="right", fontsize=5.8)
    ax.set_yticks(range(3)); ax.set_yticklabels(ylabels, fontsize=6.2)
    for i in range(3):
        for j in range(len(settings)):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}" if v != 0 else "ABSTAIN", ha="center", va="center",
                    fontsize=5.8, color="black" if abs(v) < 0.75 else "white")
    # gate stamps under E192 and E158 columns
    ax.annotate("both signals fail\ncorrelation gate", xy=(5, -0.62), xycoords=("data", "data"),
                fontsize=5.6, ha="center", color=OI["vermillion"])
    ax.annotate("scores saturated\n(undefined)", xy=(8, -0.62), xycoords=("data", "data"),
                fontsize=5.6, ha="center", color=OI["grey"])
    ax.set_title("a  20%-budget utility across nine frozen settings\n(contract = per-setting best validated signal; illustrative, in-sample)",
                 loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=5.5); cb.set_label("utility", fontsize=6)

    # (b) mean vs worst-setting
    axb.set_title("b  Mean vs worst setting", loc="left", fontweight="bold")
    strategies = ["oracle\n(best fixed\nper setting)", "always\ndisagreement",
                  "always\nmagnitude", "conditional\ncontract"]
    means = [0.316, 0.262, 0.254, 0.232]
    mins = [0.0, -0.127, -0.080, 0.0]
    colors = [OI["yellow"], OI["sky"], OI["orange"], OI["green"]]
    x = np.arange(4)
    axb.bar(x, means, width=0.55, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
    axb.scatter(x, mins, marker="_", s=220, color=OI["vermillion"], lw=1.8, zorder=3,
                label="worst-setting utility")
    axb.axhline(0, color="#555555", lw=0.8)
    axb.set_xticks(x); axb.set_xticklabels(strategies, fontsize=6.0)
    axb.set_ylabel("20% review utility")
    for xi, m in zip(x, means):
        axb.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=6.2)
    axb.legend(loc="lower left", fontsize=5.8, frameon=False)
    axb.set_ylim(-0.2, 0.42)
    axb.text(0.5, -0.30, "means are close; the contract guarantees a floor of 0\n(fixed scores go negative in double-unseen tasks);\nprospective test of the footprint = E201 (sealed)",
             transform=axb.transAxes, ha="center", fontsize=5.9, style="italic")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    save(fig, "Fig7_conditional_contract")


if __name__ == "__main__":
    fig5(); fig6(); fig7()
    print("science figures done ->", OUT)
