#!/usr/bin/env python3
"""White, Nature-style briefing figures for the 2026-08-13 meeting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

NAVY = "#1f3a5f"
INK = "#222222"
MUTED = "#5a6570"
LINE = "#c8cdd3"
BOX = "#f7f8fa"
ACCENT = "#2c6e8a"
WARN = "#8a3b2d"
OK = "#2f6b4f"
AMBER = "#8a6a2d"
WHITE = "#ffffff"


def setup() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    font_manager.fontManager.addfont(FONT)
    font_manager.fontManager.addfont(FONT_B)
    regular = font_manager.FontProperties(fname=FONT)
    bold = font_manager.FontProperties(fname=FONT_B)
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
        }
    )
    return regular, bold


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / name
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def box(ax, xy, w, h, text, regular, *, fc=BOX, ec=NAVY, lw=0.9, fs=10, color=INK, weight=None, va="center"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va=va,
        fontproperties=regular,
        fontsize=fs,
        color=color,
        linespacing=1.35,
        wrap=False,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig_system(regular, bold):
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(0.15, 5.08, "SafeConf 在做什么", fontproperties=bold, fontsize=15, color=NAVY, ha="left")
    ax.text(
        0.15,
        4.72,
        "不是再训练一个预测器，而是在预测之后做质检：真值未知时，哪些任务应优先复核。",
        fontproperties=regular,
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    box(ax, (0.2, 2.35), 2.35, 1.7, "已有扰动预测器\nscGPT / GEARS / TxPert", regular, fs=10.5)
    box(ax, (3.05, 2.35), 2.45, 1.7, "输出：某个基因或药物\n扰动后的表达变化", regular, fs=10.5)
    box(ax, (6.00, 2.35), 2.45, 1.7, "SafeConf 风险审计\n只用部署时能拿到的信息", regular, fc="#eef4f7", fs=10.5)
    box(ax, (8.95, 3.25), 2.05, 1.15, "风险高\n优先复核 / 湿实验", regular, fc="#f6eeea", ec=WARN, fs=10)
    box(ax, (8.95, 1.85), 2.05, 1.15, "未验证 setting\n明确 abstain", regular, fc="#f3f1ea", ec=AMBER, fs=10)
    arrow(ax, (2.55, 3.2), (3.05, 3.2))
    arrow(ax, (5.50, 3.2), (6.00, 3.2))
    arrow(ax, (8.45, 3.45), (8.95, 3.75))
    arrow(ax, (8.45, 2.85), (8.95, 2.45))
    ax.text(
        0.2,
        1.35,
        "打分可用：细胞 control 相似度、训练历史支持、冻结模型预测、模型分歧、predicted magnitude。",
        fontproperties=regular,
        fontsize=10,
        color=INK,
        ha="left",
    )
    ax.text(
        0.2,
        0.85,
        "打分不可用：目标扰动后的真实表达。真值只用于事后评价，且必须在预测和分数封存之后打开。",
        fontproperties=regular,
        fontsize=10,
        color=WARN,
        ha="left",
    )
    ax.text(
        0.2,
        0.35,
        "今天要强调：分数对应明确预测器/家族的误差，不是模糊的“模型置信度”。",
        fontproperties=regular,
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    save(fig, "01_系统定位.png")


def fig_questions(regular, bold):
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(0.15, 5.45, "上次周老师的三个核心问题", fontproperties=bold, fontsize=15, color=NAVY)
    headers = ["老师的问题", "现在的回答", "证据"]
    rows = [
        [
            "分数跟谁的误差比？",
            "每条结果绑定预测器、任务和误差定义。\n分歧是冻结家族的 family 风险，\n不能说某一个模型一定错。",
            "E189–E197、E194",
        ],
        [
            "predicted magnitude\n有没有偷看真值？",
            "没有。它只算冻结预测的变化幅度。\n真实表达在预测和打分留存后才打开。\n它也是必须保留的强基线。",
            "E190、E192\nprediction-first 合同",
        ],
        [
            "完全没见过的任务\n怎么打分？",
            "可用 control、历史支持、冻结预测\n和模型差异。没有合法输入时\n返回未验证，不伪装成低风险。",
            "E189、E192\nE201 检验整背景留出",
        ],
    ]
    xs = [0.2, 3.55, 8.55]
    ws = [3.2, 4.85, 2.4]
    ax.text(xs[0] + ws[0] / 2, 5.05, headers[0], ha="center", fontproperties=bold, fontsize=10.5, color=NAVY)
    ax.text(xs[1] + ws[1] / 2, 5.05, headers[1], ha="center", fontproperties=bold, fontsize=10.5, color=NAVY)
    ax.text(xs[2] + ws[2] / 2, 5.05, headers[2], ha="center", fontproperties=bold, fontsize=10.5, color=NAVY)
    ys = [3.45, 1.85, 0.25]
    for i, row in enumerate(rows):
        y = ys[i]
        box(ax, (xs[0], y), ws[0], 1.45, row[0], regular, fc="#eef4f7", fs=10.5)
        box(ax, (xs[1], y), ws[1], 1.45, row[1], regular, fs=10)
        box(ax, (xs[2], y), ws[2], 1.45, row[2], regular, fc="#f3f6f4", ec=OK, fs=10)
    save(fig, "02_老师三问.png")


def fig_settings(regular, bold):
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(0.15, 5.45, "老师要的三个更难 setting：已经跑完，但不是全部变好", fontproperties=bold, fontsize=14.5, color=NAVY)
    ax.text(
        0.15,
        5.08,
        "老师原话：如果这三个都解决了，感觉写一个小文章应该可以了。今天要把“能算”和“算得更好”分开讲。",
        fontproperties=regular,
        fontsize=10,
        color=MUTED,
    )
    cards = [
        ("小训练矩阵", "E189\nsupport 1/2/3/5", "能算，已闭合", "训练可见背景变少时，\n误差随支持量变化。", OK),
        ("整行 / 整列 / 双未见", "E189 gene\nE84 chemical", "能算，有负结果", "random 偏容易；双未见时\n经验排序可为负。", AMBER),
        ("跨数据集预测", "E190 K562\nE192 RPE1", "流程闭合，效能弱", "防泄漏成立；RPE1 按\n事前规则返回 ABSTAIN。", AMBER),
    ]
    for i, (title, ev, status, note, color) in enumerate(cards):
        x = 0.25 + i * 3.65
        box(ax, (x, 2.55), 3.4, 2.25, "", regular, fc=WHITE, ec=color, lw=1.15)
        ax.text(x + 1.7, 4.5, title, ha="center", fontproperties=bold, fontsize=12, color=color)
        ax.text(x + 1.7, 3.85, ev, ha="center", va="center", fontproperties=regular, fontsize=10.5, color=INK)
        ax.text(x + 1.7, 3.25, status, ha="center", fontproperties=bold, fontsize=11, color=color)
        ax.text(x + 1.7, 2.8, note, ha="center", va="center", fontproperties=regular, fontsize=9.5, color=MUTED)
    box(
        ax,
        (0.25, 0.25),
        10.7,
        2.05,
        "补充边界，不要和上面三档混成一句成功：\n"
        "chemical（E84/E87/E89）能做，但多处是 predicted magnitude 更强，只作边界章节。\n"
        "公开图模型：E199 在 K562 未见基因扰动上有可用风险信号；E200 整背景留出时 magnitude 更强。\n"
        "E201 正在把整背景留出扩到 K562 / RPE1 / HepG2 / Jurkat × 4 个种子，今天还没有真值结论。",
        regular,
        fc="#f7f8fa",
        fs=10.2,
    )
    save(fig, "03_三个更难setting.png")


def fig_e201(regular, bold):
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.15, 5.25, "E201：四个细胞背景的整列留出，仍在盲测", fontproperties=bold, fontsize=15, color=NAVY)
    steps = [
        "三个 source\n可见扰动数据",
        "target 只留\ncontrol",
        "TxPert 训练\n4 背景 × 4 种子",
        "封存预测、\n风险、基线",
        "双远程 Git\n留存后开真值",
        "统一评价\n误差与复核收益",
    ]
    for i, text in enumerate(steps):
        x = 0.2 + i * 1.83
        box(ax, (x, 3.35), 1.7, 1.35, text, regular, fc="#eef4f7" if i < 3 else "#f6eeea", fs=9.5)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.7, 4.02), (x + 1.83, 4.02))
    ax.text(0.2, 2.9, "截至今天中午可汇报的训练账", fontproperties=bold, fontsize=11, color=NAVY)
    table = [
        ("已完成 80 轮", "7 / 16", "K562、RPE1、HepG2 的 seed 1–2；Jurkat seed 1"),
        ("正在训练", "1 / 16", "Jurkat seed 2，约第 79 / 80 轮"),
        ("已排队未跑", "8 / 16", "四个背景的 seed 3–4，顺序已冻结"),
        ("target 扰动真值", "0 行", "没有正式预测表，也没有相对 magnitude 的结论"),
    ]
    y = 2.35
    for title, num, note in table:
        box(ax, (0.2, y - 0.05), 2.3, 0.48, title, regular, fc=BOX, fs=9.5)
        box(ax, (2.6, y - 0.05), 1.5, 0.48, num, regular, fc="#eef4f7", fs=9.5)
        box(ax, (4.2, y - 0.05), 6.75, 0.48, note, regular, fs=9.5)
        y -= 0.55
    ax.text(
        0.2,
        0.18,
        "今天只能说：训练和盲测合同已经落实。不能说 E201 成功或失败。",
        fontproperties=regular,
        fontsize=10.5,
        color=WARN,
    )
    save(fig, "04_E201盲测协议.png")


def fig_publish(regular, bold):
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(0.15, 5.45, "现在能不能写论文：建议这样问老师", fontproperties=bold, fontsize=15, color=NAVY)
    box(
        ax,
        (0.2, 3.55),
        5.3,
        1.65,
        "现在不要说“已经可以投稿”。\n最关键的多背景盲测 E201 还没开真值。\n但问题、协议和已有正负结果，\n已经可以开始搭文章结构。",
        regular,
        fc="#f6eeea",
        ec=WARN,
        fs=10.5,
    )
    box(
        ax,
        (5.7, 3.55),
        5.3,
        1.65,
        "更稳的主张不是“永远优于 magnitude”。\n而是：预测后风险审计在哪些 setting\n可以路由，哪些 setting 必须 abstain。\n这和老师说的小文章条件一致。",
        regular,
        fc="#eef4f7",
        fs=10.5,
    )
    ax.text(0.2, 3.15, "想请老师当场拍板的三件事", fontproperties=bold, fontsize=12, color=NAVY)
    asks = [
        "1  “风险审计 + 困难 setting + 不适用就 abstain” 能否作为论文主线？",
        "2  E201 四背景评价完后，是直接开始组织正文，还是必须再加一套独立数据？",
        "3  chemical 是否只保留为边界和讨论，不再硬塞进 gene 的主结论？",
    ]
    y = 2.5
    for text in asks:
        box(ax, (0.2, y), 10.8, 0.5, text, regular, fs=11)
        y -= 0.62
    ax.text(
        0.2,
        0.28,
        "如果这三件事没拿到，继续加数据和换模型不会让论文更清楚。",
        fontproperties=regular,
        fontsize=10.5,
        color=MUTED,
    )
    save(fig, "05_论文判断与请老师拍板.png")


def main() -> None:
    regular, bold = setup()
    fig_system(regular, bold)
    fig_questions(regular, bold)
    fig_settings(regular, bold)
    fig_e201(regular, bold)
    fig_publish(regular, bold)
    print("wrote", sorted(p.name for p in OUT.glob("0*.png")))


if __name__ == "__main__":
    main()
