from __future__ import annotations

import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")
OUT = ROOT / "22_methodology_story_pack"
FIG = OUT / "figures"
TABLES = OUT / "tables"
DOCS = OUT / "docs"
PACKAGE = OUT / "package"


def setup_plot() -> None:
    for font_path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", context="talk")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "top20_delta",
        "deg_precision_delta",
        "program_consistency_delta",
        "pearson_delta",
        "spearman_delta",
    ]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in df.groupby(["source", "phase", "split_type"], dropna=False):
        row = dict(zip(["source", "phase", "split_type"], keys))
        row["n"] = len(sub)
        for col in cols:
            if col in sub:
                row[col] = float(sub[col].mean())
        if {"top20_delta", "deg_precision_delta", "program_consistency_delta"}.issubset(sub.columns):
            row["effect_positive_fraction"] = float(
                ((sub[["top20_delta", "deg_precision_delta", "program_consistency_delta"]] > 0).sum(axis=1) >= 2).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def box(ax, xy, wh, text, color, fontsize=12, lw=1.5):
    x, y = xy
    w, h = wh
    rect = plt.Rectangle((x, y), w, h, fc=color, ec="#263238", lw=lw, joinstyle="round")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, start, end, text=None):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#263238"))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, text, ha="center", fontsize=10)


def fig_problem_shift() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.94, "从“预测扰动后表达量”到“判断扰动效应能否安全迁移”", ha="center", fontsize=18, weight="bold")
    box(ax, (0.04, 0.58), (0.25, 0.20), "已有主线\nResponse prediction\n扰动响应预测", "#e3f2fd")
    box(ax, (0.37, 0.58), (0.25, 0.20), "真实难点\nContext shift\n细胞环境变化", "#fff3e0")
    box(ax, (0.70, 0.58), (0.25, 0.20), "我们的主线\nSafe transport\n安全迁移", "#e8f5e9")
    arrow(ax, (0.29, 0.68), (0.37, 0.68), "不够")
    arrow(ax, (0.62, 0.68), (0.70, 0.68), "需要判断风险")
    ax.text(
        0.17,
        0.30,
        "问题：模型可能会“硬预测”，\n但不知道什么时候不该迁移。",
        ha="center",
        fontsize=13,
    )
    ax.text(
        0.50,
        0.30,
        "本质：同一个 perturbation effect\n在不同 cellular context 中可能改变。",
        ha="center",
        fontsize=13,
    )
    ax.text(
        0.82,
        0.30,
        "目标：能迁移就预测，\n不能迁移就标记 unsafe。",
        ha="center",
        fontsize=13,
    )
    fig.savefig(FIG / "01_problem_shift_safe_transport.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_method_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "SafeTrans-PT 方法流程", ha="center", fontsize=19, weight="bold")
    steps = [
        ("Datasets\n数据集", "多队列单细胞扰动数据", "#f1f8e9"),
        ("Hard splits\n困难划分", "held-out perturbation\nleave-context\nexternal validation", "#e8eaf6"),
        ("Effect space\n扰动效应空间", "基因差值 + gene program\n基因程序", "#e0f7fa"),
        ("Transportability\n可迁移性评分", "先判断能不能迁移", "#fff8e1"),
        ("Prediction / Abstention\n预测或拒绝", "safe: 迁移预测\nunsafe: 保守/拒判", "#fce4ec"),
    ]
    xs = [0.03, 0.23, 0.43, 0.63, 0.82]
    for i, (title, desc, color) in enumerate(steps):
        box(ax, (xs[i], 0.52), (0.15, 0.24), f"{title}\n\n{desc}", color, fontsize=10.5)
        if i < len(steps) - 1:
            arrow(ax, (xs[i] + 0.15, 0.64), (xs[i + 1], 0.64))
    box(ax, (0.18, 0.14), (0.26, 0.22), "Biological prior\n生物先验\npathway / graph\n通路 / 图结构", "#ede7f6", fontsize=11)
    box(ax, (0.56, 0.14), (0.26, 0.22), "Network module\n共表达网络模块\nhdWGCNA-inspired\n借鉴 hdWGCNA 思想", "#e0f2f1", fontsize=11)
    arrow(ax, (0.31, 0.36), (0.50, 0.52), "帮助解释")
    arrow(ax, (0.69, 0.36), (0.70, 0.52), "帮助拒绝 unsafe")
    fig.savefig(FIG / "02_method_pipeline.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_evidence(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    plot["setting"] = plot["source"] + "\n" + plot["phase"] + " / " + plot["split_type"]
    metrics = ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"]
    heat = plot.set_index("setting")[metrics]
    heat.columns = ["top20\n前20基因", "DEG\n差异基因", "program\n基因程序", "Pearson\n相关", "Spearman\n秩相关"]
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.62 * len(heat))))
    sns.heatmap(heat, center=0, cmap="vlag", annot=True, fmt=".3f", cbar_kws={"label": "delta / 提升幅度"}, ax=ax)
    ax.set_title("当前证据热图：哪些设置更稳", fontsize=16, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(FIG / "03_current_evidence_heatmap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_claim_boundary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    plot["effect_score"] = plot[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean(axis=1)
    colors = {"heldout_perturbation": "#2e7d32", "leave_context": "#c62828"}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for split, sub in plot.groupby("split_type"):
        ax.scatter(
            sub["effect_positive_fraction"],
            sub["effect_score"],
            s=150,
            label=f"{split}（{'留出扰动' if split == 'heldout_perturbation' else '留出环境'}）",
            color=colors.get(split, "#455a64"),
            alpha=0.78,
            edgecolor="white",
            linewidth=1.5,
        )
        for _, row in sub.iterrows():
            ax.text(row["effect_positive_fraction"] + 0.012, row["effect_score"], row["source"], fontsize=8)
    ax.axhline(0, color="#37474f", lw=1)
    ax.axvline(0.7, color="#6d4c41", lw=1, ls="--")
    ax.set_xlabel(">=2 个 effect 指标为正的比例\nEffect-positive fraction")
    ax.set_ylabel("effect 指标平均提升\nMean effect delta")
    ax.set_title("主张边界：held-out perturbation 是主战场，leave-context 是风险边界", fontsize=15, weight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(FIG / "04_claim_boundary.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fig_network_story() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.94, "为什么加入共表达网络模块", ha="center", fontsize=18, weight="bold")
    box(ax, (0.05, 0.55), (0.22, 0.22), "Gene-level effect\n基因层面效应\n很多基因一起变化", "#e3f2fd", fontsize=11)
    box(ax, (0.39, 0.55), (0.22, 0.22), "Co-expression module\n共表达模块\n一组基因形成网络", "#e0f2f1", fontsize=11)
    box(ax, (0.73, 0.55), (0.22, 0.22), "Module preservation\n模块保守性\n跨 context 是否相似", "#fff8e1", fontsize=11)
    arrow(ax, (0.27, 0.66), (0.39, 0.66), "压缩成模块")
    arrow(ax, (0.61, 0.66), (0.73, 0.66), "判断可迁移")
    ax.text(
        0.50,
        0.27,
        "本质：如果 target context 中同一组基因网络仍然存在，扰动效应更可能安全迁移；\n如果网络结构变了，强行迁移就容易错，所以要标记 unsafe transport。",
        ha="center",
        fontsize=13,
    )
    fig.savefig(FIG / "05_network_module_rationale.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def docs(summary: pd.DataFrame, network: pd.DataFrame) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    best = summary.copy()
    enough = "强二区雏形已成，但一区还需要继续补强 community baselines（社区强基线）和 network explanation（网络解释）"
    if not best.empty:
        held = best[best["split_type"] == "heldout_perturbation"]
        leave = best[best["split_type"] == "leave_context"]
        held_msg = held[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean().round(4).to_dict()
        leave_msg = leave[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean().round(4).to_dict()
    else:
        held_msg = {}
        leave_msg = {}

    write(
        DOCS / "01_ONE_PAGE_ANSWER_CN.md",
        f"""
        # 一页回答：我们到底在解决什么问题

        生成时间：{now}

        ## 最短答案

        我们不是在做“又一个 perturbation prediction（扰动预测）模型”。

        我们在做：

        **safe cross-context perturbation effect transport（跨细胞环境的扰动效应安全迁移）**。

        人话版：

        > 一个基因扰动在 A 细胞环境里产生的影响，能不能拿去预测 B 细胞环境？如果不能，模型能不能提前知道“别乱迁移”？

        ## 为什么这个问题有意义

        单细胞扰动实验很贵，不可能每个 cell type（细胞类型）、patient（病人）、cell state（细胞状态）、perturbation（扰动）都做一遍。

        如果模型能判断哪些扰动效应可以迁移，就能减少实验成本；如果模型能指出哪些不能迁移，就能避免错误推广。

        ## 目前够不够

        当前判断：**{enough}**。

        已经比较稳的是 held-out perturbation（留出扰动）场景。

        仍然需要谨慎的是 leave-context（留出细胞环境）场景。

        ## 最适合讲的结论

        当前结果支持一个 focused claim（聚焦主张）：

        > 对于部分未见扰动，SafeTrans-PT / DeepSafeTransGPU 能提升扰动效应预测；对于更困难的跨细胞环境迁移，模型需要识别 unsafe transport（不安全迁移）边界。
        """,
    )

    write(
        DOCS / "02_METHOD_TRANSLATION_CN.md",
        """
        # 英文名词翻译表

        - perturbation（扰动）：人为改变一个基因、药物、通路或实验条件。
        - perturbation effect（扰动效应）：扰动前后基因表达变化的方向和幅度。
        - single-cell perturbation prediction（单细胞扰动预测）：预测细胞受到扰动后会怎样变化。
        - cellular context（细胞环境）：细胞类型、病人、细胞状态、数据集或批次等背景。
        - cross-context（跨细胞环境）：从一个细胞环境迁移到另一个细胞环境。
        - transport（迁移）：把一个环境中学到的扰动效应用到另一个环境。
        - safe transport（安全迁移）：迁移后仍然可信。
        - unsafe transport（不安全迁移）：迁移风险高，模型不应该硬预测。
        - transportability score（可迁移性评分）：模型判断“这个效应能不能迁移”的分数。
        - abstention（拒判/保守输出）：模型认为风险高时，不强行给激进预测。
        - baseline（基线模型）：用来比较的已有或简单强方法。
        - held-out perturbation（留出扰动）：训练时不看某些扰动，测试时预测它们。
        - leave-context（留出细胞环境）：训练时不看某个细胞环境，测试时预测它。
        - external validation（外部验证）：换一个独立数据集验证。
        - top20 overlap（前20关键基因重合度）：预测出的最重要20个基因和真实最重要20个基因重合多少。
        - DEG precision（差异基因准确度）：预测出的差异基因是否真的重要。
        - program consistency（基因程序一致性）：扰动影响的基因程序是否一致。
        - co-expression module（共表达模块）：一组经常一起变化的基因。
        - hub gene（核心基因）：共表达模块中最关键、连接最多的基因。
        - hdWGCNA（高维共表达网络分析）：在单细胞/空间转录组中找共表达模块的方法。
        - module preservation（模块保守性）：同一个基因模块在不同细胞环境中是否仍然存在。
        """,
    )

    write(
        DOCS / "03_WHY_CAN_BE_PAPER_CN.md",
        f"""
        # 为什么这不是“普通调参”

        ## 以前别人主要做什么

        scGen（单细胞生成模型）、CPA（组合扰动自编码器）、GEARS（基因关系扰动预测模型）、CellOT（最优传输扰动预测）主要都在回答：

        > 扰动后表达量能不能预测准？

        这当然重要，但还不够。

        ## 我们切入的问题

        真实实验里更常见的问题是：

        > A 细胞环境里的扰动效应，能不能推广到 B 细胞环境？

        如果能，说明这个效应具有 transportability（可迁移性）。

        如果不能，模型应该识别 unsafe transport（不安全迁移），而不是硬预测。

        ## 我们为什么有论文空间

        因为我们把任务从 response prediction（响应预测）推进到 safe effect transport（安全效应迁移）。

        这带来三个贡献点：

        1. Problem（问题）：明确提出跨细胞环境扰动效应是否可迁移。
        2. Method（方法）：提出 transportability score（可迁移性评分）和 safe/unsafe 判断。
        3. Evidence（证据）：用 held-out perturbation（留出扰动）、leave-context（留出环境）、external validation（外部验证）来验证。

        ## 当前证据状态

        held-out perturbation（留出扰动）平均结果：

        {held_msg}

        leave-context（留出环境）平均结果：

        {leave_msg}

        所以当前最稳的论文主张应该是：

        > 我们能在未见扰动上得到稳定提升，同时识别更困难跨环境迁移的风险边界。
        """,
    )

    write(
        DOCS / "04_METHOD_AS_STORY_CN.md",
        """
        # 方法怎么讲才像论文

        ## 第一步：别盲目预测

        普通方法：

        > 输入 control cell（对照细胞）和 perturbation（扰动），直接预测结果。

        我们的方法：

        > 先判断这个扰动效应能不能迁移，再决定预测强度。

        ## 第二步：把基因变化压缩成 gene program（基因程序）

        单个基因太多、太噪声。

        所以我们把 effect（效应）压缩成 program（程序），看一组基因整体怎么变。

        ## 第三步：加入 pathway / graph prior（通路/图先验）

        基因不是孤立的。

        如果扰动基因在通路或图结构上相近，它们可能有类似效应。

        ## 第四步：加入 transportability score（可迁移性评分）

        模型根据这些信息打分：

        - source support（源环境支持）：训练集中有没有类似扰动？
        - context similarity（环境相似度）：目标环境和训练环境像不像？
        - perturbation consistency（扰动一致性）：同一扰动在不同环境中是否稳定？
        - pathway similarity（通路相似度）：生物先验是否支持？
        - model disagreement（模型分歧）：baseline 和 transport 分歧大不大？

        ## 第五步：加入 co-expression module（共表达模块）

        这是借鉴 hdWGCNA（高维共表达网络分析）。

        如果一个扰动影响的是某个 gene module（基因模块），并且这个模块在目标环境中也存在，那么迁移更可信。

        如果模块不保守，模型应该更保守。
        """,
    )

    write(
        DOCS / "05_CURRENT_STATUS_AND_ACTION_CN.md",
        """
        # 当前状态和后续动作

        ## 当前最强主线

        DeepSafeTransGPU（GPU增强安全迁移模型）目前是最能支撑结果的版本。

        它在 held-out perturbation（留出扰动）场景中，对 V0（强基线）和 V2（更强图先验基线）都有正向结果。

        ## 当前边界

        leave-context（留出细胞环境）还不够稳定。

        这不能硬说已经解决所有跨环境泛化，但可以作为 unsafe transport boundary（不安全迁移边界）来讲。

        ## 为什么加 NetworkSafeTransPT

        它不是为了立刻把所有指标刷高，而是为了给论文加 biological explanation（生物解释）：

        > 为什么这个扰动能迁移？因为相关 gene module（基因模块）在两个 context 中保守。

        > 为什么这个扰动不能迁移？因为模块结构变了。

        ## 接下来继续补强

        1. 继续跑 `safetrans_network_hdWGCNA`，看网络模块是否稳定提升 program/module 指标。
        2. 继续保留 DeepSafeTransGPU 作为主结果。
        3. 汇报时少讲模型堆叠，多讲 safe transport（安全迁移）这个问题。
        4. 投稿前补更强 community baseline（社区基线），如 GEARS-style / CPA-style / CellOT-style 对照。
        """,
    )

    write(
        DOCS / "06_TEACHER_SCRIPT_SIMPLE_CN.md",
        """
        # 给老师汇报的人话稿

        老师，我现在把方向收敛成一个更具体的问题：单细胞扰动效应能不能跨细胞环境安全迁移。

        以前很多工作主要关注扰动后表达量能不能预测准，但真实实验里还有一个问题：如果我只在某一种细胞类型或某一个数据集里做了扰动实验，这个扰动效应能不能推广到另一个细胞环境？

        如果可以推广，就说明这个扰动效应比较稳定；如果不能推广，模型应该识别出风险，而不是强行预测。

        所以我现在做的 SafeTrans-PT，核心不是直接预测，而是先判断可迁移性，再进行扰动效应预测。

        实验上，我用了多个单细胞扰动数据集，构造了留出扰动、留出细胞环境和外部验证这些更难的测试场景。现在比较稳定的结果是在留出扰动场景下，模型相对 baseline 有提升；而留出细胞环境场景更难，目前更适合作为不安全迁移边界来分析。

        最近我又加入了共表达网络模块的想法，借鉴 hdWGCNA。原因是扰动效应不是单个基因孤立变化，而是一组基因模块一起变化。后续我希望用模块是否保守来解释为什么某些扰动效应可以迁移，某些不适合迁移。
        """,
    )

    write(
        DOCS / "07_FIGURE_GUIDE_CN.md",
        """
        # 图片怎么讲

        ## 01_problem_shift_safe_transport.png

        讲问题转变：从普通 response prediction（扰动响应预测）转向 safe transport（安全迁移）。

        ## 02_method_pipeline.png

        讲方法流程：数据集、困难划分、扰动效应空间、可迁移性评分、预测或拒判。

        ## 03_current_evidence_heatmap.png

        讲当前结果：绿色/正值代表模型比 baseline 好。重点看 held-out perturbation（留出扰动）。

        ## 04_claim_boundary.png

        讲边界：held-out perturbation 是主结果；leave-context 是风险边界。

        ## 05_network_module_rationale.png

        讲为什么加入 hdWGCNA 思想：从单基因变化上升到 gene module（基因模块）和 module preservation（模块保守性）。
        """,
    )


def main() -> None:
    setup_plot()
    for path in [FIG, TABLES, DOCS, PACKAGE]:
        path.mkdir(parents=True, exist_ok=True)

    main_v2 = read_csv(ROOT / "17_may_resume_full_push/reports/ALL_GPU_DEEPSAFE_VS_V2.csv")
    q1_v2 = read_csv(ROOT / "20_q1_strong_push/reports/ALL_GPU_DEEPSAFE_VS_V2.csv")
    network_v2 = read_csv(ROOT / "21_network_hdWGCNA_push/reports/ALL_NETWORK_SAFE_VS_V2.csv")
    frames = []
    if not main_v2.empty:
        x = main_v2.copy()
        x["source"] = "DeepSafe main"
        frames.append(x)
    if not q1_v2.empty:
        x = q1_v2.copy()
        x["source"] = "DeepSafe q1push"
        frames.append(x)
    if not network_v2.empty:
        x = network_v2.copy()
        x["source"] = "NetworkSafe"
        frames.append(x)
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = metric_summary(all_df)
    all_df.to_csv(TABLES / "all_vs_v2_for_story.csv", index=False)
    summary.to_csv(TABLES / "setting_summary_for_story.csv", index=False)
    network_v2.to_csv(TABLES / "network_safe_vs_v2_current.csv", index=False)

    fig_problem_shift()
    fig_method_pipeline()
    fig_evidence(summary)
    fig_claim_boundary(summary)
    fig_network_story()
    docs(summary, network_v2)

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIG, PACKAGE / "figures")
    shutil.copytree(TABLES, PACKAGE / "tables")
    shutil.copytree(DOCS, PACKAGE / "docs")
    zip_target = OUT / "SafeTransPT_Methodology_Story_Pack"
    old_zip = zip_target.with_suffix(".zip")
    if old_zip.exists():
        old_zip.unlink()
    zip_path = shutil.make_archive(str(zip_target), "zip", PACKAGE)
    print(zip_path)


if __name__ == "__main__":
    main()
