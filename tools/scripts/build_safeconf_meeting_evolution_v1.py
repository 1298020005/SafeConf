#!/usr/bin/env python3
"""Build meeting figures from released tables; no fitting or new truth access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/review_repair_20261007_v1"
OUT = ROOT / "docs/组会汇报/20261008_演变与当前架构"
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables"
COLORS = dict(ink="#223342", muted="#596b78", blue="#2b738d", green="#52775c", gold="#b37d2a", red="#aa5d5d", grey="#98a7b6")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                         "svg.fonttype": "none", "font.size": 11, "figure.facecolor": "white"})


def canvas(w=16, h=9):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=12, color=None, weight="normal", **kwargs):
    ax.text(x, y, value, fontsize=size, color=color or COLORS["ink"],
            fontweight=weight, va="center", **kwargs)


def box(ax, x, y, w, h, title, body, color="blue", title_size=15, body_size=12, face=None):
    c = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.12",
                               linewidth=1.2, edgecolor=c, facecolor=face or "#f7f9fa"))
    text(ax, x+.18, y+h-.32, title, size=title_size, weight="bold", color=c)
    text(ax, x+.18, y+(h-.62)/2, body, size=body_size, linespacing=1.6)


def arrow(ax, start, end, color="muted", dashed=False, curved=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                               linewidth=1.25, color=COLORS[color],
                               linestyle="--" if dashed else "-",
                               connectionstyle=f"arc3,rad={curved}", shrinkA=3, shrinkB=3))


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.svg")
    p = output / f"{name}.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)


def evolution(output):
    fig, ax = canvas(16, 8.6)
    text(ax, .2, 8.17, "SafeConf 的演变：问题和结果怎样改变了方法", 24, weight="bold")
    text(ax, .2, 7.57, "周老师 7 月 9 日追问：谁的误差？分歧代表谁？输入能否提前获得？整行、整列和跨数据集呢？", 14)
    stages = [
        ("早期", "固定经验风险分", "支持 / 背景相似 / 分歧\n在部分任务上有信号\n\n待验证：能否普遍排序？", "grey"),
        ("7 月", "明确对象与困难设置", "单模型误差与家族误差分开\n整行 / 整列 / 双未见\n\n分歧的方向依赖设置", "blue"),
        ("8—9 月", "把预测幅度作为起点", "E201：幅度强于旧总分\n旧分仍有条件关联\n\nSafeConf-M：80% 幅度秩", "gold"),
        ("9—10 月", "PertEMA 后重定问题", "自身 OOF 错误学习已有方法\n重点转向公共真实实验\n与 A/B 错误 → 新预测器 C", "blue"),
        ("当前", "用证据决定配置", "Public：McFaline 默认\nSource：同体系有明确增量\nTarget：加入 Public 有收益", "green"),
    ]
    xs = [.2, 3.4, 6.6, 9.8, 13.0]
    for x, (date, title, body, color) in zip(xs, stages):
        text(ax, x+.03, 6.98, date, 13, weight="bold", color=COLORS[color])
        box(ax, x, 4.22, 2.8, 2.35, title, body, color, title_size=13, body_size=11)
    for x in xs[:-1]:
        arrow(ax, (x+2.84, 5.42), (x+3.16, 5.42))
    text(ax, 6.7, 3.87, "E201：ρ 0.408 vs 幅度 0.619；控制幅度后 ρ 0.250", 10, color=COLORS["muted"])
    box(ax, 3.4, 1.70, 6.0, 1.55, "7 月已开展的并行线：注册家族证书",
        "E181：2,393 任务，两类下界零违例\n回答固定家族的误差下界；E182 上界门未通过仍保留 FAIL", "grey", 14, 11)
    arrow(ax, (4.75, 4.18), (4.75, 3.30), dashed=True)
    box(ax, 9.8, 1.70, 6.0, 1.55, "当前研究判断",
        "公共参照跨预测机制有用；错误经验的迁移范围需分开验证\n同 family 的正结果与跨研究的负结果共同约束采用范围", "green", 14, 11)
    arrow(ax, (14.40, 4.18), (14.40, 3.30), color="green")
    text(ax, .2, .85, "证书与经验排序一直并行。这里按研究阶段归纳，阶段名不是仓库正式版本号。", 12, color=COLORS["muted"])
    text(ax, .2, .34, "E201 / SafeConf-M / 当前 Source 的任务和评价合同不同；数值分别回答各阶段问题。", 11, color=COLORS["muted"])
    save(fig, output, "01_RESEARCH_EVOLUTION")


def architecture(output):
    fig, ax = canvas(16, 10.4)
    text(ax, .2, 10.02, "当前 SafeConf：公共参照、可迁移错误经验与目标反馈", 23, weight="bold")
    text(ax, .2, 9.47, "三类信息分别训练；推理时使用开发侧冻结的完整配置，输出一个风险分数。", 13)
    x = [.25, 5.6, 10.95]; w = 4.8
    box(ax, x[0], 7.64, w, 1.36, "Public Memory · 真实实验", "效应向量 / 对照 / 实验身份 / 背景\n独立支持与质量分别登记", "green", 15, 12)
    box(ax, x[1], 7.64, w, 1.36, "Source Error Memory · A/B", "冻结预测器的合法留出错误\n不使用当前 C 任务的错误", "blue", 15, 12)
    box(ax, x[2], 7.64, w, 1.36, "Target Error Memory · C", "已开放预算内的 C 自身错误\n开发、CDF、拟合和选型全部计账", "gold", 15, 12)
    for xx in x: arrow(ax, (xx+w/2, 7.6), (xx+w/2, 7.16))
    box(ax, x[0], 5.58, w, 1.52, "资格筛选 → 公共参照", "排除当前实验及禁止重复；对齐基因轴\n支持加权均值 μ、历史分散度 V\n原始冲突与缺失状态另存", "green", 15, 11)
    box(ax, x[1], 5.58, w, 1.52, "训练共享风险模型", "训练区错误 CDF → 风险监督标签\n内部留出公共特征 → 固定 HGB\nA/B 的错误进入模型参数", "blue", 15, 11)
    box(ax, x[2], 5.58, w, 1.52, "训练目标风险候选", "预算内 C 错误 CDF → 目标标签\nHGB / 官方 XGBoost 配方适配\n组合训练使用折外 Shared 分数", "gold", 15, 11)
    for xx in x: arrow(ax, (xx+w/2, 5.53), (xx+w/2, 5.03))
    box(ax, .25, 3.99, 15.50, .98, "推理输入：C 的冻结预测、预测幅度、任务元数据 ＋ 合法公共证据",
        "计算 D = RMSE(预测, μ) 及同合同特征；Source / Target 只调用适用的已训练参数。", "grey", 14, 11)
    for xx in x: arrow(ax, (xx+w/2, 3.95), (xx+w/2, 3.51))
    box(ax, x[0], 2.16, w, 1.29, "PublicRule · 当前 McFaline 默认", "R = √(D² + V)\n单历史退化为直接距离；无需错误标签", "green", 14, 11, "#edf5ef")
    box(ax, x[1], 2.16, w, 1.29, "SourceRisk · 有适用范围的增强", "当前预测＋公共特征 → 共享 HGB\nTxPert 双向正增量；跨研究另行验收", "blue", 14, 11)
    box(ax, x[2], 2.16, w, 1.29, "TargetRisk · 有反馈时的候选", "P＋Public；可加合法 Shared 分数\n当前尚未稳定替换强 PublicRule", "gold", 14, 11)
    for xx in x: arrow(ax, (xx+w/2, 2.11), (xx+w/2, 1.63))
    box(ax, .25, .42, 15.50, 1.15, "DEV 冻结的评分入口 → 一个分数、证据状态和版本 → 唯一排序 → 复核前 20%",
        "无历史：幅度回退，通道尺度仅由训练区分数确定。有反馈：候选经训练侧验收后更新，否则保留旧版。", "grey", 14, 11)
    text(ax, .25, .09, "当前查询真值只进入评分器。三路不强制串联或相加。家族证书是独立研究线；风险秩不等于校准概率。", 10, color=COLORS["muted"])
    save(fig, output, "02_CURRENT_ARCHITECTURE")


def e201_turn(output):
    assoc = pd.read_csv(E201/"E201_RISK_ASSOCIATIONS.csv")
    partial = pd.read_csv(E201/"E201_PARTIAL_ASSOCIATIONS.csv")
    util = pd.read_csv(E201/"E201_REVIEW_UTILITY.csv")
    inc = pd.read_csv(E201/"E201_INCREMENTAL_TESTS.csv")
    a = assoc[(assoc.scope=="pooled") & (assoc.predictor=="safeconf_e201_risk")].iloc[0]
    b = assoc[(assoc.scope=="pooled") & (assoc.predictor=="predicted_magnitude")].iloc[0]
    c = partial[partial.scope=="pooled"].iloc[0]
    fig, ax = canvas(14, 6.2)
    text(ax, .2, 5.80, "E201：有幅度之外的信息，旧固定分数却未带来更好的复核排序", 21, weight="bold")
    text(ax, .2, 5.22, "1,808 任务 / 575 扰动簇 / 四个完整背景留出；误差目标：四个 GAT 种子的 family RMS", 12)
    for xx, row, title, color in zip([.25,4.85,9.45],[a,b,c],
                                    ["原 SafeConf · Spearman","预测幅度 · Spearman","SafeConf | 幅度 · partial ρ"],
                                    ["blue","gold","grey"]):
        box(ax, xx, 2.62, 4.25, 2.03, title, "", color, 14, 11)
        text(ax, xx+2.12, 3.57, f"{row.estimate:.3f}", 37, weight="bold", ha="center", color=COLORS[color])
        text(ax, xx+2.12, 2.98, f"95% CI [{row.ci95_lower:.3f}, {row.ci95_upper:.3f}]", 11, ha="center")
    ua=util[(util.scope=="pooled")&(util.predictor=="safeconf_e201_risk")].iloc[0]
    ub=util[(util.scope=="pooled")&(util.predictor=="predicted_magnitude")].iloc[0]
    diff=inc[(inc.scope=="pooled")&(inc.measure=="delta_oracle_normalized_utility")].iloc[0]
    text(ax, .3, 1.92, f"同样复核 362 / 1,808：U20 旧分 {ua.oracle_normalized_utility:.3f}，幅度 {ub.oracle_normalized_utility:.3f}", 17, weight="bold")
    text(ax, .3, 1.37, f"旧分 − 幅度：{diff.estimate:+.3f}，95% CI [{diff.ci95_lower:.3f}, {diff.ci95_upper:.3f}]", 14)
    text(ax, .3, .71, "后续动作：保留幅度作为起点，研究额外信息怎样使用；9 月先形成 SafeConf-M，再进入信息条件比较。", 12)
    text(ax, .3, .26, "partial ρ 是条件关联，和前两项边际相关不同；它不直接证明跨模型迁移或复核增益。", 11, color=COLORS["muted"])
    save(fig, output, "APPENDIX_E201_TURNING_POINT")


def hard_settings(output):
    fig, ax = canvas(14, 6.5)
    text(ax, .2, 6.09, "困难设置改变了模型分歧的含义", 23, weight="bold")
    text(ax, .2, 5.53, "E189：3 scGPT ＋ 3 GEARS；每扰动提供 1 / 2 / 3 / 5 个训练背景；分歧 vs family RMS", 12)
    titles=["随机缺格","扰动整列未见","背景整行未见","背景＋扰动双未见"]
    ranges=["ρ = 0.368—0.412","ρ = 0.210—0.247","ρ = −0.095—−0.013","ρ = −0.349—−0.241"]
    labels=["正信号","仍有正信号","未见稳定正信号","关系方向反转"]
    for k,(title,ran,label) in enumerate(zip(titles,ranges,labels)):
        xx=.55+k*3.45
        text(ax, xx+1.13, 4.80, title, 14, weight="bold", ha="center")
        for i in range(6):
            for j in range(6):
                held = (i,j) in {(1,2),(3,4),(5,0)} if k==0 else (j==5 if k==1 else (i==5 if k==2 else i==5 or j==5))
                ax.add_patch(Rectangle((xx+j*.36, 2.1+i*.36), .32,.32,
                                       facecolor="#b8ced8" if not held else "#d5a9a2", linewidth=0))
        text(ax, xx+1.1, 1.64, ran, 13, ha="center", color=COLORS["blue"] if k<2 else COLORS["red"])
        text(ax, xx+1.1, 1.11, label, 12, ha="center")
    text(ax, .25, .56, "图中矩阵为划分示意；ρ 范围来自四个支持预算，不是置信区间。旧模型家族能力较弱，结果用于解释边界。", 11, color=COLORS["muted"])
    text(ax, .25, .15, "纵轴：背景；横轴：扰动。蓝色为训练区域，红色为留出区域。小分歧不能直接推出预测安全。", 11, color=COLORS["muted"])
    save(fig, output, "APPENDIX_HARD_SETTINGS")


def separate_results(output):
    """Same data as original three panels, enlarged separately for slide use."""
    src=pd.read_csv(RECEIPT/"SOURCE_REAL_COMPARISON.csv")
    bud=pd.read_csv(RECEIPT/"NATIVE_PUBLIC_BUDGET_TABLE.csv")
    rev=pd.read_csv(RECEIPT/"GLOBAL_20_PERCENT_REVIEW.csv")
    fig, ax=plt.subplots(figsize=(9.2,5.6))
    for i,(method,color,label) in enumerate(zip(["always_public","always_source","selected_gate"],
            ["#98a7b6","#2b738d","#e5a64a"],["Public rank","Source HGB","Selected gate"])):
        vals=[src[(src.source==s)&(src.target==t)&(src.method==method)].u20.iloc[0]
              for s,t in [("TxPert_Exphormer","TxPert_GAT"),("TxPert_GAT","TxPert_Exphormer")]]
        bars=ax.bar(np.arange(2)+(i-1)*.23, vals, width=.22,color=color,label=label)
        ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=13)
    ax.set_xticks(np.arange(2),["Exphormer → GAT","GAT → Exphormer"])
    ax.set_ylim(0,1); ax.set_ylabel("U20（越高越好）")
    ax.set_title("旧模型错误能迁移吗？\n1,808 tasks / 575 gene clusters · same-family DEV",fontsize=17)
    ax.legend(loc="lower left",fontsize=11)
    finish_axis(fig, ax, output, "03_SOURCE_RESULT")
    fig, ax=plt.subplots(figsize=(9.2,5.6))
    ax.plot(bud.budget*100,bud.Native61,"o-",color="#98a7b6",label="Native61 adaptation",linewidth=2)
    ax.plot(bud.budget*100,bud.Native61_Public,"o-",color="#2b738d",label="Native61 + Public",linewidth=2)
    ax.axhline(bud.public_rule_no_feedback.iloc[0],color="#5b7956",linestyle="--",label="PublicRule · 无新增反馈")
    ax.set_ylim(-.05,1); ax.set_xticks([10,25,50,75,100]); ax.set_xlabel("反馈基因预算（%）")
    ax.set_ylabel("Context-macro U20 · 三个种子均值")
    ax.set_title("PertEMA 输入适配增加 Public 后的收益\n212 tasks / 152 gene clusters · DEV/SEEN",fontsize=17)
    ax.legend(loc="lower right",fontsize=11)
    finish_axis(fig, ax, output, "04_TARGET_RESULT")
    fig, ax=plt.subplots(figsize=(9.2,5.6))
    vals=[rev[rev.method==m].high_error_found.iloc[0] for m in
          ["Amplitude","PublicRule","PertEMA_Native61_Public_50_corrected"]]
    bars=ax.bar(np.arange(3),vals,color=["#98a7b6","#5b7956","#2b738d"],width=.62)
    ax.bar_label(bars,fmt="%d",padding=5,fontsize=20)
    ax.set_xticks(np.arange(3),["Amplitude","PublicRule","Native + Public\n50% feedback"])
    ax.set_ylim(0,30); ax.set_ylabel("发现的真实最高误差任务数")
    ax.set_title("同样只复核 43 / 212 个预测\n真实最高误差 43 项的命中数 · 固定首种子",fontsize=17)
    finish_axis(fig, ax, output, "05_REVIEW_RESULT")


def finish_axis(fig, ax, output, name):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, output, name)


def evidence(output):
    rows=[]
    def record(topic, source, selector, result, meaning):
        p=Path(source)
        if not p.is_absolute(): p=ROOT/p
        if not p.is_file(): raise FileNotFoundError(p)
        rows.append(dict(topic=topic,source_file=str(p),selector=selector,
                         result=result,interpretation=meaning,source_sha256=sha(p)))
    teacher=Path("/home/yyf/.codex/attachments/3d36bc69-c573-44c2-821b-6795c42ac3fc/pasted-text.txt")
    record("周老师原始追问",teacher,"12:23-12:25; 12:32; 12:35", "误差对应模型、分歧、幅度可获得性、输入、小矩阵、整行整列、跨数据集", "聊天原文核准；不把后续整理当作逐字原话")
    assoc=pd.read_csv(E201/"E201_RISK_ASSOCIATIONS.csv")
    for _,r in assoc[assoc.scope=="pooled"].iterrows():
        record("E201 边际关联",E201/"E201_RISK_ASSOCIATIONS.csv",f"scope=pooled; predictor={r.predictor}",
               f"rho={r.estimate}; CI=[{r.ci95_lower},{r.ci95_upper}]", "family_rms_error; 1808 tasks/575 clusters")
    p=pd.read_csv(E201/"E201_PARTIAL_ASSOCIATIONS.csv").query("scope=='pooled'").iloc[0]
    record("E201 条件关联",E201/"E201_PARTIAL_ASSOCIATIONS.csv","scope=pooled",f"partial_rho={p.estimate}; CI=[{p.ci95_lower},{p.ci95_upper}]", "控制幅度的条件关联；不等同复核增益或迁移")
    u=pd.read_csv(E201/"E201_REVIEW_UTILITY.csv")
    for _,r in u[u.scope=="pooled"].iterrows():
        record("E201 固定复核",E201/"E201_REVIEW_UTILITY.csv",f"scope=pooled; predictor={r.predictor}",f"U20={r.oracle_normalized_utility}; selected={r.n_selected}", "旧分与幅度的同任务比较")
    historical=[
        ("困难设置", "docs/实验结果/E189_primary_cd4_formal_cartesian_20260729/reports/E189_INTERPRETATION.md", "第4节", "随机ρ .368-.412; 整列 .210-.247; 整行 -.095至-.013; 双未见 -.349至-.241", "四个支持预算的范围，不是CI；旧家族能力较弱"),
        ("早期未见基因正信号", "docs/实验结果/E90_gene_hard_setting_matrix_20260712/reports/E90_REPORT.md", "第1节池化6面板", "144 tasks; delta rho=.120 CI[.007,.238]", "未见基因面板的局部正结果；不是全部设置普适"),
        ("新靶点未确认", "docs/实验结果/E172_primary_cd4_fresh_targets_20260718/postgate_release/reports/E172_JOINT_POSTGATE_REPORT.md", "NO_TARGET_REPLICATION", "delta AURC=-.000284818; CI[-.00239393,.00180507]", "800新靶点；同一study/test donor"),
        ("并行家族证书", "docs/实验结果/E181_registered_family_hilbert_certificate_20260724/reports/E181_REPORT.md", "结论", "2393 tasks/717 clusters; 两类下界0违反", "7月24日；回顾性代数整合；不是排序确认"),
        ("SafeConf-M", "docs/组会汇报_20260913.md", "候选公式与开发结果", "0.8Q(M)+0.2Q(S); delta rho=.0183 CI[.0091,.0271]; delta U20=.0134 CI[-.0139,.0520]", "E201留背景DEV宏平均；不是当前默认"),
        ("PertEMA近邻流程", "/home/yyf/runtime_artifacts/official_pertema_43c09a/README.md", "new screen / OOF errors", "screen-specific refit on own OOF errors", "固定本地官方版本；无需错误学习首创主张"),
        ("公共网络采用决定", "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/publicset_execution_v1/FINAL_EXPERIMENTAL_DECISION.md", "FINAL_EXPERIMENTAL_DECISION", "停止PublicSet新架构；保留各合同更有效公共方案", "数据聚合修复收益与新算法收益分开"),
    ]
    for r in historical: record(*r)
    src=pd.read_csv(RECEIPT/"SOURCE_REAL_COMPARISON.csv")
    for _,r in src.iterrows():
        record("当前 Source",RECEIPT/"SOURCE_REAL_COMPARISON.csv",f"{r.source}->{r.target}; {r.method}",f"U20={r.u20}; tasks={r.rows}; clusters={r.genes}","5 outer gene folds; same-family DEV")
    ci=pd.read_csv(RECEIPT/"source_gate/SOURCE_GATE_PAIRED_BOOTSTRAP.csv")
    for _,r in ci[ci.method=="always_source"].iterrows():
        record("Source配对增量",RECEIPT/"source_gate/SOURCE_GATE_PAIRED_BOOTSTRAP.csv",f"{r.source}->{r.target}; always_source vs public",f"delta={r.delta_utility20}; CI=[{r.ci95_lower},{r.ci95_upper}]", "5000 perturbation-cluster bootstrap; 点差与bootstrap均值分列")
    sh=pd.read_csv(RECEIPT/"SOURCE_SHUFFLED_SUMMARY.csv")
    for _,r in sh.iterrows():
        record("Source 整簇置乱",RECEIPT/"SOURCE_SHUFFLED_SUMMARY.csv",f"{r.source}->{r.target}",f"mean={r['mean']}; min={r['min']}; max={r['max']}", "五固定种子；标签块关系置乱")
    budgets=pd.read_csv(RECEIPT/"NATIVE_PUBLIC_BUDGET_TABLE.csv")
    for _,r in budgets.iterrows():
        record("Native增加Public",RECEIPT/"NATIVE_PUBLIC_BUDGET_TABLE.csv",f"budget={r.budget}",f"Native61={r.Native61}; Native61_Public={r.Native61_Public}; PublicRule={r.public_rule_no_feedback}","212 tasks/152 clusters; context-macro U20, three-seed mean")
    nci=pd.read_csv(RECEIPT/"native_public/PAIRED_BOOTSTRAP.csv")
    selected=nci[(nci.method_a=="Native61_Public")&(nci.method_b=="Native61")]
    assert len(selected)==5 and (selected.ci95_lower>0).all()
    record("Native增加Public配对CI",RECEIPT/"native_public/PAIRED_BOOTSTRAP.csv","Native61_Public vs Native61; seed20260930", "五预算CI下界均>0", "主种子CI；不冒充三个种子平均的CI")
    review=pd.read_csv(RECEIPT/"GLOBAL_20_PERCENT_REVIEW.csv")
    for _,r in review.iterrows():
        record("全局20%复核",RECEIPT/"GLOBAL_20_PERCENT_REVIEW.csv",f"method={r.method}",f"review={r.review_budget}; found={r.high_error_found}; remaining_error={r.remaining_mean_error}","固定首种子; 212 tasks; global ordering, not context-macro")
    a=review[review.method=="Amplitude"].iloc[0]; b=review[review.method=="PublicRule"].iloc[0]
    improvement=1-b.remaining_mean_error/a.remaining_mean_error
    assert b.high_error_found-a.high_error_found==18
    assert round(improvement*100,2)==7.07
    record("Public实际复核收益",RECEIPT/"GLOBAL_20_PERCENT_REVIEW.csv","Amplitude/PublicRule",f"extra_hits=18; retained_error_reduction={improvement*100:.6f}%", "相对幅度方案留存误差；不代表上游预测改进")
    record("当前采用及跨研究边界",RECEIPT/"REPORT_FOR_20261008.md","当前结论/第5节", "McFaline默认PublicRule; SAMS跨family未稳定超过强公共规则; E182回顾边界", "不同研究、不同合同分别报告")
    with (output/"EVIDENCE_BINDINGS.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    return [Path(r["source_file"]) for r in rows]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args(); output=args.output; output.mkdir(parents=True,exist_ok=True)
    setup()
    originals={n:sha(RECEIPT/n) for n in ["MEETING_RESULTS.png","MEETING_RESULTS.svg"]}
    for n in originals:
        shutil.copyfile(RECEIPT/n, output/n)
        assert sha(output/n)==originals[n]
    evolution(output); architecture(output); e201_turn(output); hard_settings(output); separate_results(output)
    evidence_inputs=evidence(output)
    inputs=[RECEIPT/n for n in ["SOURCE_REAL_COMPARISON.csv","NATIVE_PUBLIC_BUDGET_TABLE.csv","GLOBAL_20_PERCENT_REVIEW.csv"]]
    inputs += [E201/n for n in ["E201_RISK_ASSOCIATIONS.csv","E201_PARTIAL_ASSOCIATIONS.csv","E201_REVIEW_UTILITY.csv","E201_INCREMENTAL_TESTS.csv"]]
    inputs += evidence_inputs
    # Recheck originals after rendering; this script never writes to their directory.
    assert all(sha(RECEIPT/n)==digest for n,digest in originals.items())
    manifest={"source_revision": "3314f0752ff9a8dabe4c836a4b945bccd9114116",
              "purpose": "meeting evolution and architecture; no new fitting or truth opening",
              "original_result_figures_unchanged": originals,
              "input_files": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in inputs},
              "figures": {p.name:sha(p) for p in sorted(output.iterdir()) if p.suffix in {".png",".svg"}},
              "script_sha256":sha(Path(__file__)), "evidence_contracts_separate":True}
    (output/"FIGURE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(output),"figures":len(manifest["figures"]),"originals_preserved":True},ensure_ascii=False))


if __name__=="__main__":
    main()
