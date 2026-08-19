#!/usr/bin/env python3
"""Nature-style main figures for SafeConf. English (paper) + Chinese (supervisor).

Numbers are from frozen E189/E191/E192/E199/E200 reports only.
E201 results are not plotted. 16/16 training is complete; evaluation is sealed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_EN = Path(__file__).resolve().parent.parent / "figures_nature"
OUT_CN = Path("/home/yyf/Desktop/论文主图_20260819")
OUT_EN.mkdir(parents=True, exist_ok=True)
OUT_CN.mkdir(parents=True, exist_ok=True)

NAVY = "#1f3a5f"
INK = "#222222"
MUTED = "#5a6570"
GREEN = "#2f6b4f"
AMBER = "#8a6a2d"
RED = "#8a3b2d"
BLUE = "#2c6e8a"
GREY = "#7a7a7a"
FILL = "#f4f6f8"
WHITE = "#ffffff"

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT_REG)
font_manager.fontManager.addfont(FONT_B)
CN = font_manager.FontProperties(fname=FONT_REG)
CNB = font_manager.FontProperties(fname=FONT_B)

plt.rcParams.update(
    {
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
MM = 1 / 25.4
DOUBLE = 183 * MM
SINGLE = 89 * MM


def save(fig, stem: str) -> None:
    for out in (OUT_EN, OUT_CN):
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def box(ax, xy, w, h, text, *, fc=FILL, ec=NAVY, fs=8.5, color=INK, fp=None):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        fontproperties=fp,
        linespacing=1.35,
    )


def arrow(ax, a, b):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
        )
    )


# Frozen numbers
E199_DIV = (0.3948, 0.2835, 0.4969, 0.2084, 0.1033, 0.3755)
E199_MAG = (0.0955, -0.0256, 0.2187, 0.0397, -0.0830, 0.2255)
E200_RISK = (0.4240, 0.3506, 0.4953, 0.3648, 0.2356, 0.4813)
E200_MAG = (0.8797, 0.8437, 0.9095, 0.9133, 0.8748, 0.9520)
E200_DISP = (0.6639, 0.6075, 0.7143, 0.6483, 0.5441, 0.7339)


def fig1_en():
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(0.15, 4.85, "Figure 1  |  SafeConf is an audit after prediction, not a new predictor",
            fontsize=11, color=NAVY, fontweight="bold", ha="left")
    box(ax, (0.2, 2.7), 2.15, 1.45, "Existing predictors\nTxPert / scGPT / GEARS", fs=8.8)
    box(ax, (2.7, 2.7), 2.25, 1.45, "Frozen predictions\nhashed, dual-remote", fs=8.8, fc="#eef4f7")
    box(ax, (5.3, 2.55), 3.15, 1.75,
        "SafeConf audit\nuses only deploy-time inputs\ndisagreement · magnitude\nsupport · dispersion",
        fs=8.5, fc="#eef4f7")
    box(ax, (8.8, 3.45), 2.9, 1.05, "Validated setting\nROUTE  — rank for review", fs=8.5, fc="#e8f2ea", ec=GREEN)
    box(ax, (8.8, 2.15), 2.9, 1.05, "Unvalidated / degraded\nABSTAIN  — do not rank", fs=8.5, fc="#f7eee9", ec=RED)
    arrow(ax, (2.35, 3.4), (2.7, 3.4))
    arrow(ax, (4.95, 3.4), (5.3, 3.4))
    arrow(ax, (8.45, 3.7), (8.8, 3.9))
    arrow(ax, (8.45, 2.95), (8.8, 2.7))
    ax.text(0.2, 1.55, "Legal inputs: control similarity, training support, frozen predictions, disagreement, predicted magnitude.",
            fontsize=8.5, color=INK, ha="left")
    ax.text(0.2, 1.1, "Illegal input: target perturbation expression. Truth is opened only after predictions, risk scores and baselines are committed.",
            fontsize=8.5, color=RED, ha="left")
    ax.text(0.2, 0.45, "E201 (in progress): 4 cell lines × 4 seeds = 16 models, leave-one-context-out. Training complete; truth still sealed.",
            fontsize=8.5, color=MUTED, ha="left")
    save(fig, "Fig1_contract")


def fig1_cn():
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(0.15, 4.85, "图1  SafeConf 是预测之后的质检，不是新的预测器",
            fontsize=12, color=NAVY, ha="left", fontproperties=CNB)
    box(ax, (0.2, 2.7), 2.15, 1.45, "已有预测器\nTxPert / scGPT / GEARS", fs=9, fp=CN)
    box(ax, (2.7, 2.7), 2.25, 1.45, "冻结预测\n哈希，双远程提交", fs=9, fc="#eef4f7", fp=CN)
    box(ax, (5.3, 2.55), 3.15, 1.75,
        "SafeConf 风险审计\n只用部署时能拿到的信息\n分歧 · 幅度 · 支持量 · 离散度",
        fs=9, fc="#eef4f7", fp=CN)
    box(ax, (8.8, 3.45), 2.9, 1.05, "已验证的场景\n排序，优先复核", fs=9, fc="#e8f2ea", ec=GREEN, fp=CN)
    box(ax, (8.8, 2.15), 2.9, 1.05, "未验证或已退化\n明确停止排序", fs=9, fc="#f7eee9", ec=RED, fp=CN)
    arrow(ax, (2.35, 3.4), (2.7, 3.4))
    arrow(ax, (4.95, 3.4), (5.3, 3.4))
    arrow(ax, (8.45, 3.7), (8.8, 3.9))
    arrow(ax, (8.45, 2.95), (8.8, 2.7))
    ax.text(0.2, 1.5, "打分可用：细胞 control 相似度、训练历史支持、冻结预测、模型分歧、预测变化幅度。",
            fontsize=9, color=INK, ha="left", fontproperties=CN)
    ax.text(0.2, 1.05, "打分不可用：目标扰动后的真实表达。必须先提交预测和风险分，再打开真值。",
            fontsize=9, color=RED, ha="left", fontproperties=CN)
    ax.text(0.2, 0.4, "E201：四种细胞轮流整组留出，每种训练 4 次。16 次训练已完成，真实结果仍未打开。",
            fontsize=9, color=MUTED, ha="left", fontproperties=CN)
    save(fig, "图1_系统合同")


def _errorbar_pair(ax, y, rho, lo, hi, color, marker="o", label=None):
    ax.errorbar(
        rho,
        y,
        xerr=[[rho - lo], [hi - rho]],
        fmt=marker,
        color=color,
        ecolor=color,
        elinewidth=1.4,
        capsize=3.5,
        markersize=8,
        label=label,
        zorder=3,
    )


def fig2_flip():
    """The money figure: same two signals reverse roles."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 4.35), sharey=True)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.18, left=0.22, right=0.98)
    ys = [1.35, 0.55]
    # left: Spearman
    ax = axes[0]
    ax.axvline(0, color="#888888", lw=0.8, zorder=0)
    _errorbar_pair(ax, 1.48, *E199_DIV[:3], GREEN, "o", "disagreement / risk")
    _errorbar_pair(ax, 1.22, *E199_MAG[:3], AMBER, "s", "predicted magnitude")
    _errorbar_pair(ax, 0.68, *E200_RISK[:3], GREEN, "o")
    _errorbar_pair(ax, 0.42, *E200_MAG[:3], AMBER, "s")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [
            "K562 unseen genes\nE199  n = 263",
            "K562 whole-context\nhold-out  E200  n = 566",
        ]
    )
    ax.set_xlabel("Spearman ρ with task error  (95% CI)")
    ax.set_xlim(-0.15, 1.02)
    ax.set_ylim(0.15, 1.75)
    ax.set_title("a   Association with error", loc="left", fontsize=10, color=NAVY, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    # right: utility
    ax = axes[1]
    ax.axvline(0, color="#888888", lw=0.8, zorder=0)
    _errorbar_pair(ax, 1.48, *E199_DIV[3:], GREEN, "o")
    _errorbar_pair(ax, 1.22, *E199_MAG[3:], AMBER, "s")
    _errorbar_pair(ax, 0.68, *E200_RISK[3:], GREEN, "o")
    _errorbar_pair(ax, 0.42, *E200_MAG[3:], AMBER, "s")
    ax.set_xlabel("20% review-budget utility  (95% CI)")
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(0.15, 1.75)
    ax.set_title("b   Fixed 20% review budget", loc="left", fontsize=10, color=NAVY, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        1.48,
        "disagreement\nwins",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=GREEN,
    )
    ax.text(
        0.98,
        0.42,
        "magnitude\ndominates",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=AMBER,
    )
    fig.suptitle(
        "Figure 2  |  The same two signals reverse roles across settings",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
        x=0.22,
        ha="left",
    )
    save(fig, "Fig2_signal_flip")


def fig2_cn():
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.18, left=0.24, right=0.98)
    ys = [1.35, 0.55]
    ax = axes[0]
    ax.axvline(0, color="#888888", lw=0.8, zorder=0)
    _errorbar_pair(ax, 1.48, *E199_DIV[:3], GREEN, "o", "模型分歧 / 风险分")
    _errorbar_pair(ax, 1.22, *E199_MAG[:3], AMBER, "s", "预测变化幅度")
    _errorbar_pair(ax, 0.68, *E200_RISK[:3], GREEN, "o")
    _errorbar_pair(ax, 0.42, *E200_MAG[:3], AMBER, "s")
    ax.set_yticks(ys)
    ax.set_yticklabels(["K562 内未见基因\n263 个任务", "K562 整个细胞背景留出\n566 个任务"], fontproperties=CN)
    ax.set_xlabel("与误差的相关（95% 区间）", fontproperties=CN)
    ax.set_xlim(-0.15, 1.02)
    ax.set_ylim(0.15, 1.75)
    ax.set_title("a   和误差的关系", loc="left", fontsize=10, color=NAVY, pad=8, fontproperties=CNB)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", prop=CN, fontsize=8)
    ax = axes[1]
    ax.axvline(0, color="#888888", lw=0.8, zorder=0)
    _errorbar_pair(ax, 1.48, *E199_DIV[3:], GREEN, "o")
    _errorbar_pair(ax, 1.22, *E199_MAG[3:], AMBER, "s")
    _errorbar_pair(ax, 0.68, *E200_RISK[3:], GREEN, "o")
    _errorbar_pair(ax, 0.42, *E200_MAG[3:], AMBER, "s")
    ax.set_xlabel("固定 20% 复核预算的收益（95% 区间）", fontproperties=CN)
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(0.15, 1.75)
    ax.set_title("b   固定复核预算", loc="left", fontsize=10, color=NAVY, pad=8, fontproperties=CNB)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("图2  同一套信号会随场景翻转", fontsize=12, color=NAVY, fontproperties=CNB, x=0.24, ha="left")
    save(fig, "图2_信号翻转")


def fig3_footprint():
    """Clean footprint table as colored cells. No overlapping text."""
    settings = [
        "Random missing pair",
        "Unseen column",
        "Unseen row",
        "Double unseen",
        "Cross-study K562",
        "Cross-study RPE1",
        "Unseen genes (K562)",
        "Whole-context hold-out",
        "PRESCRIBE unseen-gene",
        "Four cell lines (E201)",
    ]
    cols = ["Disagreement\n/ risk score", "Predicted\nmagnitude"]
    # 1 valid, 0 abstain (CI crosses 0), -1 negative, 2 saturated, 3 sealed, None empty
    grid = [
        [1, 1],
        [1, -1],
        [0, 0],
        [-1, -1],
        [1, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    labels = [
        ["0.37–0.41", "pos."],
        ["0.21–0.25", "neg."],
        ["CI × 0", "CI × 0"],
        ["−0.35–−0.24", "neg. util."],
        ["0.42", "0.42"],
        ["ABSTAIN", "ABSTAIN"],
        ["0.40", "CI × 0"],
        ["0.42", "0.88"],
        ["saturated", "saturated"],
        ["sealed", "sealed"],
    ]
    colors = {
        1: ("#d9ead3", GREEN),
        0: ("#eeeeee", GREY),
        -1: ("#f4d6d0", RED),
        2: ("#dddddd", INK),
        3: ("#d6e3f0", BLUE),
    }
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.72, 6.2))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.6, 10.6)
    ax.axis("off")
    ax.text(-0.4, 10.25, "Figure 3  |  Where each signal may be used", fontsize=11, color=NAVY, fontweight="bold")
    for j, lab in enumerate(cols):
        ax.text(j + 0.5, 9.75, lab, ha="center", va="center", fontsize=8.5, color=NAVY, fontweight="bold")
    for i, row in enumerate(settings):
        y = 8.9 - i * 0.85
        ax.text(-0.08, y, row, ha="right", va="center", fontsize=8.3, color=INK)
        for j, val in enumerate(grid[i]):
            fc, ec = colors[val]
            ax.add_patch(
                FancyBboxPatch(
                    (j + 0.08, y - 0.32),
                    0.84,
                    0.64,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    linewidth=0.9,
                    edgecolor=ec,
                    facecolor=fc,
                )
            )
            ax.text(j + 0.5, y, labels[i][j], ha="center", va="center", fontsize=8, color=ec, fontweight="bold")
    ax.text(
        1.0,
        -0.35,
        "Green = CI excludes 0 (may rank).  Grey = CI crosses 0 (ABSTAIN).  Red = negative.  Blue = E201 truth still sealed.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    save(fig, "Fig3_footprint")


def fig3_cn():
    settings = [
        "随机缺一格",
        "整列未见",
        "整行未见",
        "双未见",
        "跨研究 K562",
        "跨研究 RPE1",
        "K562 内未见基因",
        "整个细胞背景留出",
        "PRESCRIBE 严格未见基因",
        "四细胞整列留出（E201）",
    ]
    cols = ["模型分歧 / 风险分", "预测变化幅度"]
    grid = [
        [1, 1],
        [1, -1],
        [0, 0],
        [-1, -1],
        [1, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    labels = [
        ["可用", "可用"],
        ["可用", "变负"],
        ["停止", "停止"],
        ["变负", "变负"],
        ["可用", "可用"],
        ["停止", "停止"],
        ["可用", "停止"],
        ["可用但较弱", "更强"],
        ["分数饱和", "分数饱和"],
        ["尚未开真值", "尚未开真值"],
    ]
    colors = {
        1: ("#d9ead3", GREEN),
        0: ("#eeeeee", GREY),
        -1: ("#f4d6d0", RED),
        2: ("#dddddd", INK),
        3: ("#d6e3f0", BLUE),
    }
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.78, 6.4))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.6, 10.6)
    ax.axis("off")
    ax.text(-0.4, 10.25, "图3  每种信号在哪些场景能用", fontsize=12, color=NAVY, fontproperties=CNB)
    for j, lab in enumerate(cols):
        ax.text(j + 0.5, 9.75, lab, ha="center", va="center", fontsize=9, color=NAVY, fontproperties=CNB)
    for i, row in enumerate(settings):
        y = 8.9 - i * 0.85
        ax.text(-0.08, y, row, ha="right", va="center", fontsize=9, color=INK, fontproperties=CN)
        for j, val in enumerate(grid[i]):
            fc, ec = colors[val]
            ax.add_patch(
                FancyBboxPatch(
                    (j + 0.08, y - 0.32),
                    0.84,
                    0.64,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    linewidth=0.9,
                    edgecolor=ec,
                    facecolor=fc,
                )
            )
            ax.text(j + 0.5, y, labels[i][j], ha="center", va="center", fontsize=8.5, color=ec, fontproperties=CNB)
    ax.text(1.0, -0.35, "绿：可以排序    灰：按规则停止    红：会帮倒忙    蓝：E201 还没打开真值",
            ha="center", fontsize=8.5, color=MUTED, fontproperties=CN)
    save(fig, "图3_使用足迹")


def fig4_e201():
    fig = plt.figure(figsize=(DOUBLE, 4.8))
    ax1 = fig.add_axes([0.06, 0.14, 0.34, 0.72])
    ax2 = fig.add_axes([0.48, 0.14, 0.48, 0.72])
    fig.text(0.06, 0.93, "Figure 4  |  E201 leave-one-cell-line-out, still blinded", fontsize=11, color=NAVY, fontweight="bold")

    ax1.set_xlim(-0.6, 4.6)
    ax1.set_ylim(-0.7, 4.6)
    ax1.axis("off")
    ax1.set_title("a   Training grid  (16 / 16 complete)", loc="left", fontsize=10, color=NAVY, pad=6)
    lines = ["K562", "RPE1", "HepG2", "Jurkat"]
    for i, name in enumerate(lines):
        ax1.text(-0.15, 3.4 - i, name, ha="right", va="center", fontsize=9, color=INK)
        for j in range(4):
            ax1.add_patch(
                FancyBboxPatch(
                    (j + 0.15, 3.4 - i - 0.35),
                    0.7,
                    0.7,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    linewidth=0.8,
                    edgecolor=GREEN,
                    facecolor="#d9ead3",
                )
            )
    ax1.text(2.0, -0.35, "s1      s2      s3      s4\ntarget perturbation expression accessed: 0 rows",
             ha="center", fontsize=8, color=MUTED)

    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("b   Frozen release order  (do not skip)", loc="left", fontsize=10, color=NAVY, pad=6)
    steps = [
        (GREEN, "1  Sixteen last.ckpt sealed  (done)"),
        (BLUE, "2  Zero-truth predictions  (running)"),
        (NAVY, "3  Risk scores + general baseline"),
        (NAVY, "4  Commit hashes to GitHub and Gitee"),
        (RED, "5  Only then release target truth"),
        (INK, "6  Three gates × four cell lines, all reported"),
    ]
    for i, (ec, text) in enumerate(steps):
        y = 8.4 - i * 1.35
        ax2.add_patch(
            FancyBboxPatch(
                (0.2, y - 0.45),
                9.4,
                1.05,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                linewidth=1.0,
                edgecolor=ec,
                facecolor="#f7f8fa",
            )
        )
        ax2.text(0.5, y + 0.08, text, ha="left", va="center", fontsize=9, color=ec)
    save(fig, "Fig4_E201_protocol")


def fig4_cn():
    fig = plt.figure(figsize=(DOUBLE, 4.8))
    ax1 = fig.add_axes([0.08, 0.14, 0.34, 0.72])
    ax2 = fig.add_axes([0.50, 0.14, 0.46, 0.72])
    fig.text(0.08, 0.93, "图4  E201：四种细胞整组留出，真值仍未打开", fontsize=12, color=NAVY, fontproperties=CNB)

    ax1.set_xlim(-0.6, 4.6)
    ax1.set_ylim(-0.7, 4.6)
    ax1.axis("off")
    ax1.set_title("a   16 次训练全部完成", loc="left", fontsize=10, color=NAVY, pad=6, fontproperties=CNB)
    lines = ["K562 血液肿瘤", "RPE1 视网膜上皮", "HepG2 肝脏肿瘤", "Jurkat T 细胞"]
    for i, name in enumerate(lines):
        ax1.text(-0.15, 3.4 - i, name, ha="right", va="center", fontsize=8.5, color=INK, fontproperties=CN)
        for j in range(4):
            ax1.add_patch(
                FancyBboxPatch(
                    (j + 0.15, 3.4 - i - 0.35),
                    0.7,
                    0.7,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    linewidth=0.8,
                    edgecolor=GREEN,
                    facecolor="#d9ead3",
                )
            )
    ax1.text(2.0, -0.35, "每种细胞独立训练 4 次\n目标扰动真值访问：0 行",
             ha="center", fontsize=8.5, color=MUTED, fontproperties=CN)

    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("b   打开真值前必须按这个顺序", loc="left", fontsize=10, color=NAVY, pad=6, fontproperties=CNB)
    steps = [
        (GREEN, "1  16 个模型已封存"),
        (BLUE, "2  正在做不看答案的预测"),
        (NAVY, "3  计算风险分和简单基线"),
        (NAVY, "4  推到 GitHub 和 Gitee"),
        (RED, "5  这时才打开真实实验结果"),
        (INK, "6  四个细胞的结果无论好坏都报告"),
    ]
    for i, (ec, text) in enumerate(steps):
        y = 8.4 - i * 1.35
        ax2.add_patch(
            FancyBboxPatch(
                (0.2, y - 0.45),
                9.4,
                1.05,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                linewidth=1.0,
                edgecolor=ec,
                facecolor="#f7f8fa",
            )
        )
        ax2.text(0.5, y + 0.08, text, ha="left", va="center", fontsize=9.5, color=ec, fontproperties=CN)
    save(fig, "图4_E201协议")


def fig5_components():
    e199 = [
        ("family disagreement", 0.3948, 0.2835, 0.4969),
        ("predicted magnitude", 0.0955, -0.0256, 0.2187),
        ("model–baseline gap", -0.0064, -0.1332, 0.1185),
        ("STRING neighbors (−)", -0.0822, -0.2052, 0.0388),
        ("GO neighbors (−)", -0.1018, -0.2200, 0.0165),
        ("graph-isolated flag", -0.1067, -0.1961, -0.0029),
    ]
    e200 = [
        ("predicted magnitude", 0.8797, 0.8437, 0.9095),
        ("source-effect dispersion", 0.6639, 0.6075, 0.7143),
        ("transfer risk (5-part)", 0.4240, 0.3506, 0.4953),
        ("neg. log source cells", 0.2149, 0.1309, 0.2955),
        ("model–baseline gap", 0.1597, 0.0751, 0.2415),
        ("support-context deficit", 0.0170, -0.0662, 0.1034),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 4.6))
    fig.subplots_adjust(wspace=0.45, top=0.84, bottom=0.14, left=0.22, right=0.98)
    for ax, rows, title in (
        (axes[0], e199, "a   Unseen genes in K562"),
        (axes[1], e200, "b   Whole K562 context held out"),
    ):
        n = len(rows)
        y = np.arange(n)[::-1]
        ax.axvline(0, color="#888888", lw=0.8, zorder=0)
        for yi, row in zip(y, rows):
            _, rho, lo, hi = row
            col = GREEN if lo > 0 else (RED if hi < 0 else GREY)
            ax.errorbar(rho, yi, xerr=[[rho - lo], [hi - rho]], fmt="o", color=col,
                        ecolor=col, elinewidth=1.3, capsize=3, markersize=6)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlabel("Spearman ρ with task error (95% CI)")
        ax.set_title(title, loc="left", fontsize=10, color=NAVY, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Figure 5  |  No single component is valid in both settings",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
        x=0.22,
        ha="left",
    )
    save(fig, "Fig5_components")


def fig5_cn():
    e199 = [
        ("模型分歧", 0.3948, 0.2835, 0.4969),
        ("预测变化幅度", 0.0955, -0.0256, 0.2187),
        ("模型与基线差距", -0.0064, -0.1332, 0.1185),
        ("STRING 邻域（取负）", -0.0822, -0.2052, 0.0388),
        ("GO 邻域（取负）", -0.1018, -0.2200, 0.0165),
        ("图上孤立基因", -0.1067, -0.1961, -0.0029),
    ]
    e200 = [
        ("预测变化幅度", 0.8797, 0.8437, 0.9095),
        ("源效应离散度", 0.6639, 0.6075, 0.7143),
        ("五分量风险分", 0.4240, 0.3506, 0.4953),
        ("源支持细胞数（取负）", 0.2149, 0.1309, 0.2955),
        ("模型与基线差距", 0.1597, 0.0751, 0.2415),
        ("缺少源背景数", 0.0170, -0.0662, 0.1034),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 4.7))
    fig.subplots_adjust(wspace=0.48, top=0.84, bottom=0.14, left=0.24, right=0.98)
    for ax, rows, title in (
        (axes[0], e199, "a   K562 内未见基因"),
        (axes[1], e200, "b   K562 整个背景留出"),
    ):
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color="#888888", lw=0.8, zorder=0)
        for yi, row in zip(y, rows):
            _, rho, lo, hi = row
            col = GREEN if lo > 0 else (RED if hi < 0 else GREY)
            ax.errorbar(rho, yi, xerr=[[rho - lo], [hi - rho]], fmt="o", color=col,
                        ecolor=col, elinewidth=1.3, capsize=3, markersize=6)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontproperties=CN, fontsize=9)
        ax.set_xlabel("与误差的相关（95% 区间）", fontproperties=CN)
        ax.set_title(title, loc="left", fontsize=10, color=NAVY, pad=6, fontproperties=CNB)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("图5  没有一个分量在两种场景里都好用", fontsize=12, color=NAVY, fontproperties=CNB, x=0.24, ha="left")
    save(fig, "图5_分量足迹")


def main() -> None:
    fig1_en()
    fig1_cn()
    fig2_flip()
    fig2_cn()
    fig3_footprint()
    fig3_cn()
    fig4_e201()
    fig4_cn()
    fig5_components()
    fig5_cn()
    readme = OUT_CN / "先看这个.md"
    readme.write_text(
        """# 论文主图（2026-08-19 重做）

GLM 原先那套图不能投稿：标题叠字、框体重叠、E201 还画着“running 08-17”。
这一套按 Nature 白底重画，中英各一份。数字只来自已经解盲的正式报告。
E201 的结果数字还没有，图4只画协议，不画假结果。

## 投稿用（英文，Briefings / Nature 栏宽）

同一目录里的 `Fig1_contract` 到 `Fig5_components`（png + pdf）。

| 图 | 讲什么 | 贴哪里 |
|---|---|---|
| Fig1_contract | 质检不是新预测器；验证了才排序，否则停止 | 引言后 / 方法总览 |
| Fig2_signal_flip | 最重要的一张：同一信号会翻转 | 主结果 |
| Fig3_footprint | 每种信号在哪些场景能用 | 主结果 / 讨论 |
| Fig4_E201_protocol | 四细胞整列留出，16 次训练完，真值未开 | 前瞻实验 |
| Fig5_components | 没有一个分量两边都好用 | 补充主文或附录 |

## 给周老师 / PPT（中文）

`图1_系统合同` 到 `图5_分量足迹`。白底，可直接贴。

不要用 `agents/glm/paper/figures/` 里带 V2、叠字的旧图。
""",
        encoding="utf-8",
    )
    print("wrote", OUT_EN)
    print("wrote", OUT_CN)


if __name__ == "__main__":
    main()
