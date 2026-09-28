#!/usr/bin/env python3
"""Assemble the frozen SafeConf-v4 evidence into paper-facing artifacts.

This script does not train or select a method.  It only reads released DEV/SEEN
outputs and post-freeze competence/status records.  It never opens still-sealed
test perturbation truth for E208, E247 or E258.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
DEV = OUT / "safeconf_v4_development"
PERTEMA = OUT / "pertema_fair_comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def build_confirmation_audit() -> pd.DataFrame:
    rows = [
        {
            "candidate": "E208_Jiang24",
            "data_role_after_freeze": "SEALED_CONFIRMATION",
            "n_tasks": 224,
            "n_contexts": 12,
            "test_truth_opened": False,
            "test_prediction_ready": False,
            "upstream_competence": "BLOCKED",
            "support_available": True,
            "relevance_available": True,
            "quality_available": False,
            "conflict_available": True,
            "content_available": True,
            "eligible_for_v4_confirmation": False,
            "reason": "E208 latent and linear validation MSE were 53.31x and 32.91x the no-change MSE; test predictions never started.",
            "evidence": "/home/yyf/data/perturbench_e208/posttraining_20260921/validation_competence/E208_VALIDATION_COMPETENCE_STATUS.json",
        },
        {
            "candidate": "E216_Jiang24_resource",
            "data_role_after_freeze": "SEEN",
            "n_tasks": 224,
            "n_contexts": 12,
            "test_truth_opened": True,
            "test_prediction_ready": True,
            "upstream_competence": "BLOCKED",
            "support_available": True,
            "relevance_available": False,
            "quality_available": False,
            "conflict_available": True,
            "content_available": True,
            "eligible_for_v4_confirmation": False,
            "reason": "Formal result was released on 2026-09-20 before v4; both upstreams lost to no-change and it cannot be relabelled pristine.",
            "evidence": "/home/yyf/runtime_worktrees/publication_sprint_20260919/docs/实验结果/E216_jiang24_resource_bounded_confirmation_20260920/E216_FORMAL_EVALUATION_STATUS.json",
        },
        {
            "candidate": "E247_Kaden_CRISPRa",
            "data_role_after_freeze": "SEALED_CONFIRMATION",
            "n_tasks": 367,
            "n_contexts": 1,
            "test_truth_opened": False,
            "test_prediction_ready": False,
            "upstream_competence": "BLOCKED",
            "support_available": False,
            "relevance_available": False,
            "quality_available": False,
            "conflict_available": False,
            "content_available": False,
            "eligible_for_v4_confirmation": False,
            "reason": "Two-seed GEARS ensemble was 1.70% worse than the strongest train-mean-effect baseline on validation; the single-context asset cannot instantiate v4 history evidence.",
            "evidence": "docs/实验结果/E247_kaden_crispra_unseen_tf_20260924/E247_GEARS_ENSEMBLE_STATUS.json",
        },
        {
            "candidate": "E258_Feng2025",
            "data_role_after_freeze": "SEALED_CONFIRMATION",
            "n_tasks": 2895,
            "n_contexts": 7,
            "test_truth_opened": False,
            "test_prediction_ready": False,
            "upstream_competence": "BLOCKED",
            "support_available": True,
            "relevance_available": True,
            "quality_available": True,
            "conflict_available": True,
            "content_available": True,
            "eligible_for_v4_confirmation": False,
            "reason": "Best residual MLP improved only 0.071% over the strongest simple predictor, below the preregistered 2% gate; four-donor test truth remains sealed.",
            "evidence": "docs/实验结果/E258_feng2025_independent_confirmation_20260924/DEV_REPORT_20260925_历史何时有用.md",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "CONFIRMATION_FIELD_AUDIT.csv", index=False)
    return frame


def load_results() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    dev = {}
    for name in ("txpert_gat", "txpert_exphormer"):
        dev[name] = pd.read_csv(DEV / name / "SUMMARY.csv")
    pertema = pd.read_csv(PERTEMA / "SUMMARY.csv")
    boot = pd.read_csv(PERTEMA / "PAIRED_BOOTSTRAP.csv")
    return dev, pertema, boot


def build_master(dev: dict[str, pd.DataFrame], pertema: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    labels = {"txpert_gat": "TxPert GAT", "txpert_exphormer": "TxPert Exphormer"}
    for upstream, frame in dev.items():
        mag = float(frame.loc[frame.method.eq("Magnitude_raw"), "utility20"].iloc[0])
        for row in frame.itertuples():
            rows.append({
                "evidence_stage": "DEV_nested_OOF",
                "dataset": "TxPert four-context panel",
                "upstream_family": "TxPert",
                "upstream_model": labels[upstream],
                "perturbation_type": "genetic",
                "method": row.method,
                "n_tasks": 1808,
                "n_strata": row.n_strata,
                "utility20": row.utility20,
                "delta_utility20_vs_magnitude": row.utility20 - mag,
                "spearman": row.spearman,
                "aurc": row.aurc,
                "formal_confirmation": False,
            })
    for row in pertema[pertema.method.eq("PertEMA_adapted")].itertuples():
        label = labels[row.upstream]
        mag = float(pertema[(pertema.upstream.eq(row.upstream)) & (pertema.method.eq("Magnitude_raw"))].utility20.iloc[0])
        rows.append({
            "evidence_stage": "DEV_fair_comparator",
            "dataset": "TxPert four-context panel",
            "upstream_family": "TxPert",
            "upstream_model": label,
            "perturbation_type": "genetic",
            "method": "PertEMA_official_algorithm_adaptation",
            "n_tasks": 1808,
            "n_strata": row.n_strata,
            "utility20": row.utility20,
            "delta_utility20_vs_magnitude": row.utility20 - mag,
            "spearman": row.spearman,
            "aurc": row.aurc,
            "formal_confirmation": False,
        })
    gears = pd.read_csv(OUT / "gears_risk_transfer/SUMMARY.csv")
    gmag = float(gears.loc[gears.method.eq("Magnitude_raw"), "utility20"].iloc[0])
    for row in gears.itertuples():
        rows.append({
            "evidence_stage": "SEEN_small_transfer",
            "dataset": "Adamson+Dixit+Norman formal subset",
            "upstream_family": "GEARS",
            "upstream_model": "GEARS formal released",
            "perturbation_type": "genetic",
            "method": row.method,
            "n_tasks": row.n,
            "n_strata": 3,
            "utility20": row.utility20,
            "delta_utility20_vs_magnitude": row.utility20 - gmag,
            "spearman": row.spearman,
            "aurc": np.nan,
            "formal_confirmation": False,
        })
    master = pd.DataFrame(rows)
    master.to_csv(OUT / "EXPERIMENT_MASTER_TABLE.csv", index=False)
    return master


def build_figures(dev: dict[str, pd.DataFrame], pertema: pd.DataFrame, boot: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    names = ["txpert_gat", "txpert_exphormer"]
    pretty = ["GAT", "Exphormer"]
    methods = ["Magnitude_raw", "PertEMA_adapted", "Ridge_USR", "V2_nested"]
    mpretty = ["Magnitude", "PertEMA*", "V1", "V2"]
    colors = ["#777777", "#D95F02", "#1B9E77", "#377EB8"]
    x = np.arange(2)
    width = 0.19
    for j, (method, label, color) in enumerate(zip(methods, mpretty, colors)):
        vals = []
        for name in names:
            source = pertema[(pertema.upstream.eq(name)) & (pertema.method.eq(method))]
            vals.append(float(source.utility20.iloc[0]))
        axes[0].bar(x + (j - 1.5) * width, vals, width, label=label, color=color)
    axes[0].set_xticks(x, pretty)
    axes[0].set_ylim(0.68, 0.82)
    axes[0].set_ylabel("Utility@20")
    axes[0].set_title("a  Development ranking utility")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    comparisons = []
    for name in names:
        inc = pd.read_csv(DEV / name / "INCREMENTAL_RESULTS.csv")
        for comp, short in [("V1_vs_magnitude", "V1−Magnitude"), ("V2_vs_V1", "V2−V1")]:
            r = inc[inc.comparison.eq(comp)].iloc[0]
            comparisons.append((name, short, r.delta_utility20, r.utility_ci95_lower, r.utility_ci95_upper))
        r = boot[(boot.upstream.eq(name)) & (boot.comparison.eq("SafeConf_V2_vs_PertEMA"))].iloc[0]
        comparisons.append((name, "V2−PertEMA*", r.delta_utility20, r.utility_ci95_lower, r.utility_ci95_upper))
    ypos = np.arange(len(comparisons))
    vals = np.array([r[2] for r in comparisons])
    lower = vals - np.array([r[3] for r in comparisons])
    upper = np.array([r[4] for r in comparisons]) - vals
    axes[1].errorbar(vals, ypos, xerr=np.vstack([lower, upper]), fmt="o", color="#333333", capsize=3)
    axes[1].axvline(0, color="#999999", lw=1)
    axes[1].set_yticks(ypos, [f"{pretty[names.index(r[0])]} {r[1]}" for r in comparisons], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ΔUtility@20 (95% cluster bootstrap CI)")
    axes[1].set_title("b  Incremental evidence")

    for name, label, color in zip(names, pretty, ["#377EB8", "#984EA3"]):
        rc = pd.read_csv(DEV / name / "RISK_COVERAGE.csv")
        for method, ls in [("Magnitude_raw", "--"), ("V2_nested", "-")]:
            line = rc[rc.method.eq(method)].groupby("coverage", as_index=False).mean(numeric_only=True)
            axes[2].plot(line.coverage, line.mean_true_rmse, ls=ls, marker="o", color=color,
                         label=f"{label} {'V2' if method == 'V2_nested' else 'Magnitude'}")
    axes[2].set_xlabel("Accepted low-risk fraction")
    axes[2].set_ylabel("Mean true RMSE")
    axes[2].set_title("c  Risk–coverage")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "SAFECONF_V4_MAIN_RESULTS.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        OUT / "SAFECONF_V4_MAIN_RESULTS.pdf", bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "SafeConf-v4"},
    )
    plt.close(fig)


def build_reports(dev: dict[str, pd.DataFrame], pertema: pd.DataFrame, boot: pd.DataFrame,
                  master: pd.DataFrame, audit: pd.DataFrame) -> None:
    labels = {"txpert_gat": "TxPert GAT", "txpert_exphormer": "TxPert Exphormer"}
    main_rows = []
    for upstream in ("txpert_gat", "txpert_exphormer"):
        frame = dev[upstream].set_index("method")
        main_rows.append([
            labels[upstream], 1808,
            frame.loc["Magnitude_raw", "utility20"],
            frame.loc["Ridge_USR", "utility20"],
            frame.loc["V2_nested", "utility20"],
            frame.loc["V2_nested", "utility20"] - frame.loc["Magnitude_raw", "utility20"],
            frame.loc["V2_nested", "spearman"],
        ])
    pertema_rows = []
    for upstream in ("txpert_gat", "txpert_exphormer"):
        point = pertema[pertema.upstream.eq(upstream)].set_index("method")
        b = boot[(boot.upstream.eq(upstream)) & (boot.comparison.eq("SafeConf_V2_vs_PertEMA"))].iloc[0]
        pertema_rows.append([
            labels[upstream], point.loc["PertEMA_adapted", "utility20"],
            point.loc["V2_nested", "utility20"], b.delta_utility20,
            f"[{b.utility_ci95_lower:.4f}, {b.utility_ci95_upper:.4f}]",
        ])
    quality = pd.read_csv(OUT / "FIELD_AVAILABILITY_MATRIX.csv")
    txq = quality[quality.dataset_id.astype(str).str.contains("TxPert")]
    quality_cov = txq[txq.field_name.astype(str).str.contains("quality", case=False, na=False)].eligible_coverage.max() if len(txq) else 0.0
    freeze = json.loads((OUT / "FINAL_CANDIDATE_FREEZE.json").read_text())

    report = f"""# SafeConf v4 论文结果包

生成日期：2026-09-29。证据角色：TxPert 为开发数据；E208/E247/E258 测试真值保持封存；E216 为此前已经揭盲的负结果。

## 主结果

{md_table(["上游", "任务", "Magnitude U20", "V1 U20", "V2 U20", "V2−Magnitude", "V2 Spearman"], main_rows)}

开发阶段的 Final Candidate 已按机器门冻结为 `V2_nested_evidence_shrinkage`。合并 8 个评价 strata 的 `V2−V1` 为 **{freeze['aggregate']['delta_u20_macro']:.4f}**，{freeze['aggregate']['nonnegative_strata_fraction']:.1%} strata 非负；所有安全退化门均通过。

## PertEMA 公平比较

{md_table(["上游", "PertEMA* U20", "SafeConf V2 U20", "差值", "95% CI"], pertema_rows)}

`PertEMA*` 是官方算法与原超参数在同一 TxPert 任务、同一 gene-disjoint outer split、同一错误标签预算上的适配，不是把官方 CD4 冻结权重直接搬到 TxPert。SafeConf 多使用合法历史证据，因此这里检验的是“历史证据在同标签预算下能否提供增量”，不声称覆盖一切 PertEMA 特征工程。

## 能支持的结论

1. 在两个通过能力门的 TxPert 架构上，Magnitude 是强基线，V1/V2 的开发点估计均进一步提高 Utility@20 和 Spearman。
2. Support 后加入 Relevance，在两个架构上的 Spearman 增量均为正且簇自举区间下限大于 0；历史字段打乱后，V1 的 Spearman 明显下降。
3. 固定 Ridge 比 HGB 和小 MLP 稳定；增加复杂度没有收益。
4. V2 的 40 个 outer fits 中，32 个（80%）产生随任务变化的 evidence weight；K562 多数折退化成常数混合，必须保留为边界。
5. Quality 的合法 eligible coverage 为 **{quality_cov:.1%}**。当前结果只能叫 Support/Relevance-aware，不能写成已经证明 Quality-aware。

## 尚未获得的证据

- Gate A/B 尚未通过：没有一个同时满足“未见、上游能力合格、V2 字段可用”的 sealed confirmation。
- 跨模型家族强主张尚未成立：54-task GEARS 只能作小样本压力线；E208/E247/E258 均在打开测试真值前被能力门阻断。
- 外部公共历史尚未验证；TxPert 的有效历史是同研究、目标背景排除后的 internal historical evidence。
- V2 相对 V1 的单架构 Utility@20 自举区间均跨 0；不能把开发门通过写成统计确认。

## 当前论文决策

本轮选择 **V2 作为冻结候选，不宣布最终方法已获外部确认**。现阶段最有价值的论文骨架是：

> 一个带上游能力门和输出合同的黑盒扰动预测风险审计框架；在合格 TxPert 上，Support/Relevance 历史证据和嵌套收缩提高开发期风险排序；在不合格上游上，系统拒绝生成看似漂亮但无部署意义的风险结论。

这已经形成真实的方法与结果包，但投稿前的最高优先级仍是一个能力合格的独立上游 family 或新的 sealed context。不能用已揭盲 E216、弱上游或重新切分看过的数据补这个空位。
"""
    (OUT / "PAPER_RESULT_TABLES.md").write_text(report, encoding="utf-8")

    method = f"""# SafeConf v4 方法决策

## 决策

- Final Candidate：`V2_nested_evidence_shrinkage`。
- 冻结依据：合并开发 `ΔUtility@20(V2−V1)={freeze['aggregate']['delta_u20_macro']:.6f}`，非负 strata 比例 {freeze['aggregate']['nonnegative_strata_fraction']:.1%}，有效 strata 100%。
- 证据等级：**开发候选**。Gate A/B 均等待合法 confirmation。
- V1 保留为预指定 secondary benchmark，confirmation 失败后不得用同一结果把 V1 事后改封为主方法。

## 方法结构

1. `rP`：Universal Prediction Evidence 的 Ridge 风险。
2. `rPQ`：Prediction + Support + Relevance + Conflict proxy + Content 的 Ridge 风险。
3. 两个分数只在 inner OOF 上校准到相同误差尺度。
4. 单调低容量 gate 仅使用预测前 evidence，得到 `rP + w(rPQ-rP)`。
5. outer-test 只执行 outer-train 冻结的标准化、校准和 gate。

## 独立判断

当前真正成立的是 Support/Relevance-aware shrinkage；Quality 未进入正式实证。V2 的价值来自在历史模块整体不稳时学习收缩，而不是因为更大的网络。小 MLP 明显落后，因此不扩 Transformer 或 set encoder。

正式投稿主张暂定为 `architecture-agnostic within TxPert`。取得独立 family 正结果后才升级为跨模型家族；取得 external history 增量后才使用 `public experimental history`。
"""
    (OUT / "METHOD_DECISION.md").write_text(method, encoding="utf-8")

    failures = """# SafeConf v4 失败与修复日志

| 资产/环节 | 失败诊断 | 一次修复或替换 | 结果 | 决策 |
| --- | --- | --- | --- | --- |
| E208 Jiang24 | Latent/Linear 验证 MSE 分别为 no-change 的 53.31/32.91 倍 | E233 修复 LinearAdditive 的训练—推理 control 合同，并比较 softplus/linear | 最佳 linear 仍差 54.97%，0/12 context 不劣 | 停止该修复；测试真值继续封存 |
| E245 Jiang24 来源迁移 | 成熟上游缺失 | 用合法 train 历史做质量门控来源迁移 | 最佳仅改善 1.25%，未达 2% 门 | 不生成测试预测 |
| E216 资源模型 | 上游未超过 no-change | 结果完整保留 | SafeConf-M 外部确认失败 | 登记为 SEEN 负结果，不复用为 confirmation |
| E247 Kaden CRISPRa | GEARS 虽略胜 no-change，却弱于最强 train-mean-effect 基线 | 两 seed ensemble + 图特征阶段 | 相对最强简单基线差 1.70%，CI 稳定为负 | 测试表达继续封存 |
| E258 Feng | 学习型上游没有实质超过收缩历史均值 | residual MLP、历史距离、分半噪声诊断 | 最佳 MLP 只多 0.071%；历史增量可被支持度/噪声解释 | 测试供体真值继续封存 |
| V4 risk learner | 复杂模型可能过拟合 1,808 tasks | 同 split 比 Ridge/HGB/small MLP | Ridge 稳定最好，MLP 明显落后 | 不扩大型网络 |
| Quality evidence | 无合法 replicate/split-half 字段 | 不用 conflict proxy 冒充 Quality | eligible coverage 0 | 收缩主张为 Support/Relevance-aware |
"""
    (OUT / "FAILURE_AND_REPAIR_LOG.md").write_text(failures, encoding="utf-8")

    claims = """# SafeConf v4 Claim–Evidence Matrix

| 论文主张 | 证据 | 当前裁决 | 论文措辞 |
| --- | --- | --- | --- |
| 输出接口统一 | Prediction/Error Contract；direct-effect 与 treated-state 两路径 | 支持 | common black-box adapter contract |
| 幅度是强风险基线 | TxPert 两架构 U20 0.759–0.768，Spearman 0.728–0.742 | 支持 | strong mandatory baseline |
| Support/Relevance 有额外信息 | V1 超幅度；Relevance 的 Spearman 增量两架构 CI 下限为正；shuffle 下降 | 开发支持 | development evidence, pending external confirmation |
| V2 可安全收缩历史 | 开发门通过；8/8 valid strata，7/8 非负；risk guards 通过 | 开发支持 | frozen candidate, not confirmed |
| Quality-aware | TxPert legal Quality coverage 0 | 不支持 | 不使用该主张 |
| External public history | 只有 H_internal 已验证 | 不支持 | historical experimental evidence |
| 跨结构 | TxPert GAT 与 Exphormer 同方向 | 支持 | cross-architecture within one family |
| 跨模型家族/model-agnostic | GEARS 仅 54 tasks；其他 family 能力门失败 | 不支持强主张 | black-box interface; family generalization pending |
| 胜过 PertEMA | 同任务适配中 V2 高 0.066–0.069，CI 下限 >0 | 支持限定比较 | beats official-algorithm adaptation under registered contract |
| 独立 confirmation | E208/E247/E258 因上游能力门阻断；E216 已揭盲且失败 | 未完成 | 不声称 confirmed |
| 化学通用性 | CPA 多数资产弱于简单基线 | 压力测试 | chemical stress test only |
"""
    (OUT / "CLAIM_EVIDENCE_MATRIX.md").write_text(claims, encoding="utf-8")

    novelty = """# SafeConf v4 创新定位

## 与 PertEMA 的关系

| 能力 | SafeConf v4 | PertEMA |
| --- | --- | --- |
| 黑盒预测器后置层 | 是 | 是 |
| 预测侧特征 | Universal P | prediction-time features |
| 历史实验 Support/Relevance | 是 | 官方核心不要求 |
| 历史 Conflict/Missingness 收缩 | V2 候选 | 无同一机制 |
| 无同任务 truth 推理 | 是 | 是 |
| 训练风险层需要 OOF error | 是 | 是 |
| isotonic/conformal interval | 本版不作为核心输出 | 是 |
| selective review/risk-coverage | 是 | 是 |
| 模型错误反馈 | 扩展线 | OOF error 是核心监督来源 |

SafeConf 不能把“黑盒后置可靠性”或“GBDT/Ridge 预测误差”写成首创。当前可辨识的方法差异是：**把合法历史证据拆成 Support、Relevance、Conflict 与 Missingness，并以嵌套、单调、低容量收缩决定历史修正的使用强度；上游能力门失败时拒绝路由。**

## 仍需防守的审稿问题

1. 当前 Quality 没有实证，名称必须收缩。
2. GAT/Exphormer 同属 TxPert，不能据此声称广泛 model-agnostic。
3. PertEMA 是官方算法适配，不是其 CD4 冻结模型复现；比较合同要在正文写清。
4. V2−V1 的 Utility CI 跨 0；真正方法确认需要新 family/context。
5. 旧 SafeConf 证书线与本次历史风险线是不同研究对象，稿件不能混成一个模糊贡献。
"""
    (OUT / "NOVELTY_POSITIONING.md").write_text(novelty, encoding="utf-8")

    outline = """# SafeConf v4 稿件大纲

## 暂定题目

**SafeConf: Evidence-Aware Post-hoc Risk Auditing for Single-Cell Perturbation Predictions**

## 1. Introduction

- 扰动预测准确率不足以告诉实验人员先复核哪些任务。
- 预测幅度是强而被低估的风险基线。
- 历史实验有帮助潜力，但会稀疏、无关、冲突；直接拼接会失效。
- 贡献：统一输出合同、能力门、历史证据分层、nested shrinkage、同预算 selective review。

## 2. Methods

1. PredictionRecord 与 effect/error contract。
2. History Eligibility 与 internal/external history。
3. Universal P、Support、Relevance、Conflict、Missingness。
4. V1 与 V2 nested cross-fitting。
5. Utility@20、risk-coverage、AURC 和 paired cluster bootstrap。
6. PertEMA 同合同适配与能力门。

## 3. Results

1. TxPert 两架构均通过上游能力门。
2. Magnitude 强，但 Support/Relevance 提供开发增量。
3. V2 通过预注册开发门；复杂模型没有收益。
4. 历史 shuffle、within-context 与 matched-support 排除明显 shortcut。
5. 同合同 PertEMA 适配比较。
6. E208/E216/E247/E258 展示能力门和适用边界。

## 4. Discussion

- 当前证据限于基因扰动与 TxPert family。
- Quality/external history/chemical 仍是扩展，不包装为已完成。
- 上游 competence 是风险审计的前置条件。
- 部署流程：冻结上游→生成 PredictionRecord→冷启动风险→积累合法反馈→周期重训。

## 5. 必须在投稿前补齐

- 一个能力合格且此前未用于设计的独立 family/context confirmation。
- 对应 confirmation 的一次性 Gate A/B 表。
- 若无法取得，则把稿件转为 Gate C，并在独立 split/context 复现至少两条边界规律。
"""
    (OUT / "MANUSCRIPT_OUTLINE.md").write_text(outline, encoding="utf-8")

    figure_plan = """# SafeConf v4 论文主图规划

1. **Figure 1：方法架构。** Frozen upstream → Universal P；History Support/Relevance/Conflict/Missingness → nested evidence weight → task risk → selective review。
2. **Figure 2：主结果。** 已生成 `SAFECONF_V4_MAIN_RESULTS.pdf/png`，包含 U20、成对增量和 risk-coverage。
3. **Figure 3：Evidence weight。** 各 context/fold 的 gate weight 分布与系数；突出 K562 常数退化边界。
4. **Figure 4：历史来源与 shortcut。** history shuffle、matched-support、within-context 结果。
5. **Figure 5：能力门与失败边界。** 合格 TxPert 对比 E208/E216/E247/E258 的 fail-closed 流程。
6. **Supplement：** 全 strata、bootstrap、MLP/HGB 消融、Error Memory budget、化学压力测试。
"""
    (OUT / "PAPER_FIGURE_PLAN.md").write_text(figure_plan, encoding="utf-8")

    repro_files = [
        OUT / "FINAL_METHOD_CONFIG.json", OUT / "FINAL_CANDIDATE_FREEZE.json",
        OUT / "FIELD_AVAILABILITY_MATRIX.csv", OUT / "HISTORY_SOURCE_AUDIT.csv",
        OUT / "UPSTREAM_COMPETENCE_V4.csv", OUT / "EXPERIMENT_MASTER_TABLE.csv",
        PERTEMA / "RUN_STATUS.json", PERTEMA / "SUMMARY.csv",
    ]
    hashes = [[str(p.relative_to(ROOT)), sha256(p)] for p in repro_files if p.exists()]
    repro = f"""# SafeConf v4 复现报告

## 运行合同

- 开发：4 contexts × 5 gene-disjoint folds；outer task 从未进入本折风险模型训练。
- V2：inner OOF 生成 rP/rPQ、outer-train 内尺度校准和单调 gate。
- 主指标：`ceil(0.2*n)`、risk 降序、task_id 字典序处理 ties。
- Bootstrap：5,000 次，perturbation gene cluster 为重采样单位。
- SEALED：Final Candidate 冻结前未读取结果、字段覆盖或 truth；E216 后验识别为此前已揭盲并降级为 SEEN。
- PertEMA：官方软件 commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`，官方 XGB 超参数，同 outer split/label budget。

## 关键文件哈希

{md_table(["文件", "SHA-256"], hashes)}

## 复跑命令

```bash
python tools/scripts/build_safeconf_governance_artifacts.py
python tools/scripts/run_safeconf_v4_development.py --upstream e201
python tools/scripts/run_safeconf_v4_development.py --upstream e205
python tools/scripts/audit_safeconf_competence_gate.py
/tmp/pertema-venv/bin/python tools/scripts/run_safeconf_pertema_fair_comparison.py
python tools/scripts/freeze_safeconf_final_candidate.py
python tools/scripts/build_safeconf_v4_paper_package.py
```

结果脚本默认拒绝覆盖已经存在的 RUN_STATUS，正式复跑应使用新输出目录或先保留原证据快照。
"""
    (OUT / "REPRODUCIBILITY_REPORT.md").write_text(repro, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = build_confirmation_audit()
    dev, pertema, boot = load_results()
    master = build_master(dev, pertema)
    build_figures(dev, pertema, boot)
    build_reports(dev, pertema, boot, master, audit)
    status = {
        "status": "COMPLETE",
        "sealed_test_truth_opened_by_this_script": False,
        "final_candidate": "V2_nested_evidence_shrinkage",
        "gate_a": "PENDING_ELIGIBLE_CONFIRMATION",
        "gate_b": "PENDING_ELIGIBLE_CONFIRMATION",
        "gate_c": "NOT_YET_INVOKED",
        "n_master_rows": int(len(master)),
        "n_confirmation_candidates_audited": int(len(audit)),
        "main_figure_sha256": sha256(OUT / "SAFECONF_V4_MAIN_RESULTS.pdf"),
    }
    (OUT / "PAPER_PACKAGE_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
