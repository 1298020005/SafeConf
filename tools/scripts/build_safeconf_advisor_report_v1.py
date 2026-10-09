#!/usr/bin/env python3
"""Build advisor-facing SafeConf diagrams and a factual meeting report."""
from __future__ import annotations
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/研究推进/20261010_导师汇报_v1"
OUT.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_EN = FontProperties(family="DejaVu Sans")

TEAL = "#197C78"
TEAL_LIGHT = "#DCEFEB"
BLUE = "#2F6FA3"
BLUE_LIGHT = "#E5EFF8"
CORAL = "#C96A54"
CORAL_LIGHT = "#F7E5DF"
PURPLE = "#6E5A9E"
PURPLE_LIGHT = "#EEE9F8"
GRAY = "#68757A"
GRAY_LIGHT = "#EEF1F2"
DARK = "#1D2A30"
GOLD = "#BE8B2C"


def box(ax, xy, wh, title, body="", fc="white", ec=DARK, lw=1.5, dashed=False, title_color=DARK, fontsize=11):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(x + w/2, y + h*0.69, title, ha="center", va="center", color=title_color,
            fontsize=fontsize, fontweight="bold", fontproperties=FONT)
    if body:
        ax.text(x + w/2, y + h*0.30, body, ha="center", va="center", color=DARK,
                fontsize=fontsize-2.1, linespacing=1.35, fontproperties=FONT)
    return patch


def arrow(ax, start, end, color=DARK, lw=1.8, dashed=False, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=connectionstyle))


def save(fig, stem):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(.35, 8.58, "SafeConf：公共实验如何启动预测风险审核", fontsize=25, fontweight="bold", color=DARK, fontproperties=FONT)
    ax.text(.38, 8.18, "核心逻辑：先用真实历史建立可解释的冷启动排序；只有在监督增量被证实后，才接入 Source / Target 学习。", fontsize=12.5, color=GRAY, fontproperties=FONT)

    # Main flow.
    box(ax, (.55, 5.55), (2.2, 1.32), "冻结预测器", "p：当前预测\n幅度、输出轴、版本", BLUE_LIGHT, BLUE, title_color=BLUE)
    box(ax, (3.45, 5.15), (3.0, 2.12), "公共实验历史", "协议兼容的历史响应\ncontext / support / conflict\n独立实验单位与来源", TEAL_LIGHT, TEAL, title_color=TEAL)
    box(ax, (7.15, 5.55), (2.55, 1.32), "PublicRule", "预测—历史距离\n+支持与冲突字段", TEAL, TEAL, title_color="white")
    box(ax, (10.35, 5.55), (2.45, 1.32), "统一风险排序", "同一 CDF 尺度\n无历史也进入队列", PURPLE_LIGHT, PURPLE, title_color=PURPLE)
    box(ax, (13.45, 5.55), (2.0, 1.32), "有限复核", "优先检查高风险\n剩余预测更可靠", "#F3F0E5", GOLD, title_color=GOLD)
    arrow(ax, (2.75, 6.2), (3.42, 6.2), BLUE)
    arrow(ax, (6.48, 6.2), (7.12, 6.2), TEAL)
    arrow(ax, (9.72, 6.2), (10.32, 6.2), PURPLE)
    arrow(ax, (12.82, 6.2), (13.42, 6.2), GOLD)

    # Data treatment mini cards.
    box(ax, (3.72, 3.48), (2.22, .88), "有历史", "公共距离", "#F4FBF9", TEAL, lw=1.2, title_color=TEAL, fontsize=10)
    box(ax, (6.25, 3.48), (2.22, .88), "无历史", "幅度 fallback", "#F7F8F8", GRAY, lw=1.2, title_color=GRAY, fontsize=10)
    box(ax, (8.78, 3.48), (2.22, .88), "有反馈", "候选学习器", "#FFF5F1", CORAL, lw=1.2, dashed=True, title_color=CORAL, fontsize=10)
    ax.text(2.95, 4.65, "推理时状态", fontsize=11, fontweight="bold", color=GRAY, fontproperties=FONT)
    arrow(ax, (4.82, 5.12), (4.82, 4.38), TEAL, lw=1.2)
    arrow(ax, (7.35, 5.12), (7.35, 4.38), GRAY, lw=1.2)
    arrow(ax, (9.88, 5.12), (9.88, 4.38), CORAL, lw=1.2, dashed=True)

    # Optional supervised lanes.
    ax.text(.55, 2.68, "可选监督扩展（必须通过同任务、同预算、同簇 bootstrap 验证）", fontsize=11.5, fontweight="bold", color=GRAY, fontproperties=FONT)
    box(ax, (1.05, 1.02), (3.45, 1.28), "Source error memory", "其他预测器的合法留出错误\n当前结果：压力证据，未进默认系统", GRAY_LIGHT, GRAY, dashed=True, title_color=GRAY, fontsize=10)
    box(ax, (5.32, 1.02), (3.45, 1.28), "Target feedback", "当前预测器真实反馈\n当前结果：部分 global 增益，非普遍替代", CORAL_LIGHT, CORAL, dashed=True, title_color=CORAL, fontsize=10)
    box(ax, (9.62, 1.02), (3.45, 1.28), "版本更新", "新旧错误共同训练候选\n发布门通过后才替换旧版", BLUE_LIGHT, BLUE, dashed=True, title_color=BLUE, fontsize=10)
    arrow(ax, (2.78, 2.31), (8.12, 5.48), GRAY, lw=1.5, dashed=True, connectionstyle="arc3,rad=0.12")
    arrow(ax, (7.05, 2.31), (9.72, 5.48), CORAL, lw=1.5, dashed=True, connectionstyle="arc3,rad=-0.12")
    arrow(ax, (11.34, 2.31), (11.58, 5.48), BLUE, lw=1.5, dashed=True)
    ax.text(14.62, 1.55, "默认\nPublicRule", ha="center", va="center", fontsize=11, fontweight="bold", color=TEAL, fontproperties=FONT)
    ax.add_patch(Circle((14.62, 1.55), .48, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.5))
    ax.text(14.62, 1.55, "默认\nPublicRule", ha="center", va="center", fontsize=9, fontweight="bold", color=TEAL, fontproperties=FONT)
    ax.text(.55, .35, "证据状态：McFaline 主结果（SEEN）  |  KOLF 冻结独立评价  |  Frangieh 跨家族压力证据  |  Source/Target 条件性扩展", fontsize=9.2, color=GRAY, fontproperties=FONT)
    save(fig, "ARCHITECTURE_NATURE_STYLE")


def evolution():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(.35, 8.1, "研究演变：从“学习错误”到“公共证据启动风险审核”", fontsize=24, fontweight="bold", color=DARK, fontproperties=FONT)
    ax.text(.38, 7.68, "每一步都由前一步的实验结果推动，而不是预先堆叠三个模块。", fontsize=12.5, color=GRAY, fontproperties=FONT)
    stages = [
        ("1", "Target error learning", "问题起点\n模型错在哪里？", "PertEMA 类方法\n需要当前 screen 的 OOF error", GRAY_LIGHT, GRAY),
        ("2", "Public cold start", "新模型还没有错误标签\n先能不能审核？", "公共历史 → 距离\nMcFaline 22/43、24/43", TEAL_LIGHT, TEAL),
        ("3", "Information separation", "不同信息来源含义不同\n不能混成一个分数", "Public / Source / Target\n分别记账、分别验收", BLUE_LIGHT, BLUE),
        ("4", "Component tests", "复杂模块是否真的增加信息？", "DeepSets / mapping / Source gate\n未通过就保留简单方案", CORAL_LIGHT, CORAL),
        ("5", "Current SafeConf", "可解释、可复核、可扩展", "PublicRule 默认\nTarget 条件增强、外部边界明确", PURPLE_LIGHT, PURPLE),
    ]
    xs = [0.45, 3.65, 6.85, 10.05, 13.25]
    for i,(num,title,q,ans,fc,ec) in enumerate(stages):
        x=xs[i]
        box(ax, (x, 4.05), (2.35, 2.2), title, q, fc, ec, title_color=ec, fontsize=10.8)
        ax.text(x+1.175, 3.57, ans, ha="center", va="top", fontsize=9.1, color=DARK, linespacing=1.35, fontproperties=FONT)
        ax.add_patch(Circle((x+.22, 6.0), .22, facecolor=ec, edgecolor="white", linewidth=1.2))
        ax.text(x+.22, 6.0, num, ha="center", va="center", color="white", fontsize=10, fontweight="bold", fontproperties=FONT_EN)
        if i < len(stages)-1:
            arrow(ax, (x+2.42, 5.15), (xs[i+1]-.10, 5.15), GRAY, lw=1.8)
    ax.text(.65, 1.82, "当前主线", fontsize=12, fontweight="bold", color=TEAL, fontproperties=FONT)
    ax.add_patch(FancyBboxPatch((.65, .72), 14.7, .72, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.5))
    ax.text(8.0, 1.08, "公共实验历史先启动审核 → 目标反馈有了再比较增量 → Source 只在明确互补时启用 → 最终输出统一风险排序", ha="center", va="center", fontsize=12, color=TEAL, fontweight="bold", fontproperties=FONT)
    save(fig, "EVOLUTION_NATURE_STYLE")


def results():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), gridspec_kw={"width_ratios":[1.15,1.15,1.3]})
    fig.patch.set_facecolor("white")
    fig.suptitle("当前真实结果：公共证据是最稳定的起点", fontsize=22, fontweight="bold", color=DARK, fontproperties=FONT)
    # McFaline.
    ax=axes[0]
    labels=["DecoderOnly", "SAMS-VAE"]
    public=[22,24]; amp=[4,5]
    x=np.arange(2); w=.34
    ax.bar(x-w/2, amp, w, color="#8A989E", label="Magnitude")
    ax.bar(x+w/2, public, w, color=TEAL, label="PublicRule")
    ax.set_xticks(x,labels,rotation=20,ha="right",fontproperties=FONT)
    ax.set_ylabel("Top-error tasks found / 43",fontproperties=FONT)
    ax.set_ylim(0,28); ax.set_title("McFaline · 20% review",fontproperties=FONT,fontweight="bold")
    for i,(a,p) in enumerate(zip(amp,public)):
        ax.text(i-w/2,a+.8,f"{a}",ha="center",fontsize=10,fontproperties=FONT_EN)
        ax.text(i+w/2,p+.8,f"{p}",ha="center",fontsize=10,fontproperties=FONT_EN)
    ax.legend(frameon=False,prop=FONT)
    # KOLF.
    ax=axes[1]
    ax.axhline(0,color="#777",lw=1)
    ax.errorbar([0],[.136],yerr=[[.136-(-.048)],[.366-.136]],fmt='o',color=TEAL,ecolor=TEAL,capsize=5,ms=8)
    ax.set_xticks([0],["Public − Magnitude"],fontproperties=FONT)
    ax.set_ylabel("Δ U20 (95% cluster CI)",fontproperties=FONT)
    ax.set_ylim(-.12,.42); ax.set_title("KOLF · frozen evaluation",fontproperties=FONT,fontweight="bold")
    ax.text(0,.29,"17/60 vs 17/60 hits\nPublic coverage 228/300",ha="center",va="center",fontsize=10,color=DARK,fontproperties=FONT)
    # decision panel.
    ax=axes[2]; ax.axis('off'); ax.set_title("Adoption decision",fontproperties=FONT,fontweight="bold")
    rows=[("PublicRule", "DEFAULT", TEAL), ("Target feedback", "CONDITIONAL", CORAL), ("Source transfer", "STRESS / OPTIONAL", GRAY), ("DeepSets / mapping", "STOPPED", GOLD)]
    y=.84
    for name,status,color in rows:
        ax.add_patch(FancyBboxPatch((.08,y-.08),.84,.15,boxstyle="round,pad=.01,rounding_size=.02",facecolor="white",edgecolor=color,lw=1.5))
        ax.text(.13,y,name,va='center',fontsize=11,fontweight='bold',color=DARK,fontproperties=FONT)
        ax.text(.87,y,status,va='center',ha='right',fontsize=9.5,fontweight='bold',color=color,fontproperties=FONT)
        y-=.2
    ax.text(.08,.10,"结论不是“所有模块都成功”，\n而是确定哪一层真正提供稳定增量。",fontsize=10.5,color=DARK,linespacing=1.4,fontproperties=FONT)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
    save(fig, "RESULTS_AT_A_GLANCE")


def report():
    report = '''# SafeConf 小导师汇报材料（2026-10-10）

## 先给结论

SafeConf 目前不是“再换一个模型”的项目，而是一个已经形成清晰演变逻辑的**预测风险审核系统**：

> 新的扰动预测器在刚开始工作时没有自己的错误记录，公共真实实验能否先帮助我们找到最值得复核的高风险预测？当目标模型逐渐积累反馈后，反馈能否在公共证据之上继续改善审核？

当前采用的默认方案是 **PublicRule**。Source 错误迁移和 Target 反馈保留为有边界的扩展，不把三个模块强行拼成一个复杂网络。

## 一、为什么研究问题会这样演变

### 1. 起点：预测准确不等于知道谁会错

扰动预测器通常给出一个整体误差，但实验人员实际面对的是一个队列：只能复核其中一小部分任务。因此真正需要的是风险排序，而不只是平均 RMSE。

### 2. 发现 PertEMA 后，问题被具体化

PertEMA 说明：已有预测器错误记录可以训练后置可靠性层。但它依赖当前 screen 的 OOF error。新预测器刚上线时，正是最需要审核的时候，却还没有自己的错误标签。

于是问题从“如何让预测器更准”转成：

> 在目标错误标签为零或很少时，系统能不能先启动审核？

### 3. 公共实验成为冷启动证据

我们把历史扰动实验视为模型无关的外部证据：它不直接告诉系统当前预测一定错，而是提供一个真实响应参照，帮助判断当前预测和过去实验是否不一致。

### 4. Source 和 Target 被拆开

其他预测器的错误、当前预测器自己的反馈，与公共实验的生物响应含义不同。因此三类信息分开记账、分别比较：

- Public：没有目标模型错误标签，也可以启动；
- Source：只有跨模型错误规律真的增加信息时才启用；
- Target：当前模型有反馈后，按反馈预算评价，而不是预设一定优于 Public。

## 二、当前 SafeConf 架构

架构图见 `ARCHITECTURE_NATURE_STYLE.svg/png/pdf`。

### 推理流程

1. 输入冻结预测、预测幅度和任务元数据；
2. 查询协议兼容的公共历史；
3. 对有历史任务计算预测—历史距离、支持量和冲突；
4. 无历史任务使用固定幅度回退；
5. 通过训练侧 CDF 将不同状态放进同一个风险排序；
6. 在固定复核预算下优先检查高风险任务。

Target feedback 和 Source error 不是默认必经层，而是虚线候选：只有在同任务、同反馈预算、同生物簇 bootstrap 下显示额外价值，才进入扩展系统。

## 三、实际做过什么

### McFaline 主结果

- 212 个任务，152 个扰动基因簇；
- 20% 复核预算对应 43 个任务；
- DecoderOnly：PublicRule 命中 22/43 个最高误差任务，Magnitude 命中 4/43；
- SAMS-VAE：PublicRule 命中 24/43，Magnitude 命中 5/43；
- 复核后剩余平均误差分别降低 7.07% 和 6.84%。

这说明公共实验可以在没有当前模型错误标签时启动风险审核。

### Target feedback

在 McFaline 的 global pooled ranking 中，Native+Public 在充分反馈时相对 Public 仍有正的点增量；但 context-macro 主端点和外部 KOLF 没有形成普遍稳定的替代证据。因此 Target 被保留为条件性扩展，而不是默认方法。

### Frangieh 跨模型压力测试

GEARS→scGPT 和 scGPT→GEARS 两个方向中，PublicHGB 相对 Magnitude 的 paired U20 增量分别为 +0.460 和 +0.309。但六个上游 predictor competence gate 全部失败，所以这组结果用于说明压力场景下公共信号仍可能有用，不作为合格独立确认。

### KOLF 冻结独立评价

- 600 个上游训练基因；
- 300 个开发/反馈基因；
- 300 个评价基因；
- Ridge 通过预测器能力门，MLP 因方差退化失败；
- Public 覆盖 228/300 个评价任务，72 个任务使用固定幅度回退；
- Public 相对 Magnitude 的 U20 点增量为 +0.136，95% 基因簇区间为约 [-0.048, 0.366]；
- 两者在 20% 复核下均命中 17/60 个最高误差任务。

KOLF 最重要的价值是验证了完整冻结流程、混合历史覆盖和无历史回退，而不是强行证明 Public 在所有外部场景都显著优于幅度。

## 四、当前故事为什么有意义

### 对实验人员

实验资源有限时，系统不需要等待当前模型积累足够错误记录，就能先把最值得复核的任务排到前面。

### 对模型开发

模型的平均误差、模型自己的错误记忆和公共实验历史被分开，能看清收益来自哪类信息，而不是把所有信息拼接后只报告一个分数。

### 对方法学

SafeConf 的贡献不是发明一个更深的网络，而是提出一个可复核的信息协议：谁提供监督、什么时候可用、如何防止泄漏、在多少复核预算下带来实际收益。

## 五、目前不能混淆的边界

- PublicRule 的主要证据来自 McFaline，KOLF 是点估计正但区间较宽的冻结独立评价；
- Frangieh 是 stress evidence，不是合格外部 confirmation；
- Target feedback 有部分 global 增益，但不能写成普遍超过 Public；
- PertEMA 只完成了输入适配比较，没有宣称完整官方 conformal 流程复现；
- DeepSets、背景映射和 Source gate 没有稳定额外价值，已停止继续扩展。

## 六、给导师汇报时建议这样讲

> 我最初关注的是如何学习预测模型会在哪里出错。后来发现，现有后置方法需要当前模型自己的错误记录，而新模型上线时恰恰没有这些记录。因此我把问题改成：能不能用已经存在的真实扰动实验先启动风险审核。现在的系统先用公共实验形成模型无关的风险参照；目标模型有反馈后再比较是否增加信息；其他模型错误只作为单独的跨模型证据。McFaline 上 PublicRule 在两个预测器上都明显优于只看预测幅度，KOLF 上完成了冻结独立评价，Frangieh 则给出了跨模型压力边界。现在主方法已经收敛为 PublicRule，下一步主要是听取老师对主线和适用范围的意见，而不是继续堆网络。

## 七、汇报后再做什么

1. 根据导师意见决定是否保留“Source/Target 扩展”在主图还是补充图；
2. 如果 E208 保护队列完成，按既定冻结合同接入独立外部结果；
3. 填写作者和基金信息，套 TCBB 官方模板；
4. 不因一次负结果重新开启无边界模型搜索。

## 文件入口

- `ARCHITECTURE_NATURE_STYLE.svg/png/pdf`：主架构图；
- `EVOLUTION_NATURE_STYLE.svg/png/pdf`：研究演变图；
- `RESULTS_AT_A_GLANCE.svg/png/pdf`：真实结果总览；
- `SPEAKER_NOTES.md`：口头汇报稿；
- `evidence_freeze_v1/`：全部结果、哈希和复现入口。
'''
    (OUT / "ADVISOR_REPORT.md").write_text(report)


def speaker_notes():
    notes = '''# 小导师汇报口头稿（建议 8–10 分钟）

## 开场（约 1 分钟）

我这次不先讲模型名字，先讲问题变化。原来想研究的是预测器会在哪里出错，后来发现 PertEMA 这类方法需要当前 screen 的错误标签，而新模型刚开始预测时并没有这些标签。所以我把问题改成：能不能先用已经存在的真实扰动实验启动风险审核。

## 架构图（约 2 分钟）

左边是冻结预测器，输出当前预测。中间是公共真实实验库，经过资格筛选后形成响应参照。PublicRule 计算当前预测和真实历史之间的距离，并把有历史、无历史任务放进同一个风险排序。没有历史时不伪造零效应，而是使用幅度回退。

图下方的 Source 和 Target 是虚线，因为它们不是默认必经层。Source 要证明跨模型错误有额外价值，Target 要在相同反馈预算下证明增量。当前真实结果支持 PublicRule 作为默认起点。

## 结果（约 3 分钟）

McFaline 上，20% 复核时 DecoderOnly 的 PublicRule 命中 22/43 个最高误差任务，幅度只有 4/43；SAMS-VAE 是 24/43 对 5/43。这个结果说明公共历史不需要等待当前模型错误标签，就能帮助决定优先复核谁。

随后做了跨模型和外部检查。Frangieh 两个方向有正的压力结果，但预测器能力门没过，因此我把它叫 stress evidence。KOLF 则按 600/300/300 做了能力门、风险冻结和一次性评价；Ridge 通过，Public 覆盖 228/300，点增量是正的，但区间宽，所以我把它解释成流程和适用条件验证，不夸大成普遍显著优越。

## 贡献和边界（约 2 分钟）

这项工作真正的贡献不是再造一个复杂预测网络，而是把公共实验、Source 错误和 Target 反馈按信息来源拆开，形成一个能在冷启动阶段工作的风险审核协议。它回答的是实验资源如何分配，而不是只比较平均预测误差。

目前默认采用 PublicRule；Source 和 Target 保留为有条件的扩展。DeepSets、背景映射等没有稳定增量的分支已经停止，不再为了让所有模块成功而继续加复杂度。

## 收尾（约 1 分钟）

我希望老师主要帮我判断两件事：第一，主线是否应该强调“冷启动风险审核”还是“公共实验减少错误监督需求”；第二，Source/Target 应该放在主结果还是扩展结果。实验和证据已经按这个分工整理好，后续根据意见完成最终模板和外部队列接入。
'''
    (OUT / "SPEAKER_NOTES.md").write_text(notes)


if __name__ == "__main__":
    architecture(); evolution(); results(); report(); speaker_notes()
