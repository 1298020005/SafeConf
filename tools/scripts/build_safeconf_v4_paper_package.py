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
CONFIRM = OUT / "confirmation/e170_primary_cd4_four_panel"
ERROR_MEMORY = OUT / "error_memory_budget"
CPA = OUT / "cpa_risk_batch_v2"


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
            "candidate": "E170_primary_CD4_four_panel",
            "data_role_after_freeze": "SEALED_CONFIRMATION_OPENED_ONCE",
            "n_tasks": 2400,
            "n_contexts": 12,
            "test_truth_opened": True,
            "test_prediction_ready": True,
            "upstream_competence": "PASS",
            "support_available": True,
            "relevance_available": True,
            "quality_available": False,
            "conflict_available": True,
            "content_available": True,
            "eligible_for_v4_confirmation": True,
            "reason": "All four panels were opened together after V2 freeze and remote authorization; frozen V2 passed preregistered Gate A on 2,400 new-perturbation/held-out-donor tasks.",
            "evidence": "docs/实验结果/Stage2_mature_upstream_20260928/confirmation/e170_primary_cd4_four_panel/RUN_STATUS.json",
        },
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
    frame.to_csv(OUT / "CONFIRMATION_CANDIDATE_AUDIT.csv", index=False)
    return frame


def build_confirmation_field_audit() -> pd.DataFrame:
    """Release the frozen E170 field audit only after one-shot opening.

    The source audit was created after candidate freeze and before truth access;
    copying it here keeps the field definitions independent of the outcome.
    """
    frame = pd.read_csv(OUT / "E170_PREOPEN_FIELD_AUDIT.csv")
    frame["confirmation_opened_once"] = True
    frame["field_definition_changed_after_open"] = False
    frame["confirmation_result"] = "GATE_A_PASS"
    frame.to_csv(OUT / "CONFIRMATION_FIELD_AUDIT.csv", index=False)
    return frame


def load_results() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    dev = {}
    for name in ("txpert_gat", "txpert_exphormer", "e190_gears_crossfamily"):
        dev[name] = pd.read_csv(DEV / name / "SUMMARY.csv")
    pertema = pd.read_csv(PERTEMA / "SUMMARY.csv")
    boot = pd.read_csv(PERTEMA / "PAIRED_BOOTSTRAP.csv")
    return dev, pertema, boot


def build_master(dev: dict[str, pd.DataFrame], pertema: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    labels = {
        "txpert_gat": "TxPert GAT",
        "txpert_exphormer": "TxPert Exphormer",
        "e190_gears_crossfamily": "GEARS 3-seed centroid",
    }
    for upstream, frame in dev.items():
        mag = float(frame.loc[frame.method.eq("Magnitude_raw"), "utility20"].iloc[0])
        is_e190 = upstream == "e190_gears_crossfamily"
        for row in frame.itertuples():
            rows.append({
                "evidence_stage": "SEEN_crossfamily_nested_OOF" if is_e190 else "DEV_nested_OOF",
                "dataset": "Adamson→Replogle K562 cross-study" if is_e190 else "TxPert four-context panel",
                "upstream_family": "GEARS" if is_e190 else "TxPert",
                "upstream_model": labels[upstream],
                "perturbation_type": "genetic",
                "method": row.method,
                "n_tasks": 692 if is_e190 else 1808,
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
    confirmation = pd.read_csv(CONFIRM / "SUMMARY.csv")
    cmag = float(confirmation.loc[confirmation.method.eq("Magnitude_raw"), "utility20"].iloc[0])
    for row in confirmation.itertuples():
        rows.append({
            "evidence_stage": "SEALED_CONFIRMATION_ONE_SHOT",
            "dataset": "E170 primary CD4 four-panel held-out donor",
            "upstream_family": "scGPT+GEARS ensemble",
            "upstream_model": "Frozen 6-member scGPT+GEARS ensemble",
            "perturbation_type": "genetic",
            "method": row.method,
            "n_tasks": 2400,
            "n_strata": row.n_strata,
            "utility20": row.utility20,
            "delta_utility20_vs_magnitude": row.utility20 - cmag,
            "spearman": row.spearman,
            "aurc": row.aurc,
            "formal_confirmation": True,
        })
    memory = pd.read_csv(ERROR_MEMORY / "SUMMARY.csv")
    for row in memory.itertuples():
        rows.append({
            "evidence_stage": "DEV_feedback_budget",
            "dataset": "TxPert four-context panel",
            "upstream_family": "TxPert",
            "upstream_model": row.upstream,
            "perturbation_type": "genetic",
            "method": f"ErrorMemory_budget_{row.budget:.2f}",
            "n_tasks": 1808,
            "n_strata": row.n_folds,
            "utility20": row.utility20,
            "delta_utility20_vs_magnitude": np.nan,
            "spearman": row.spearman,
            "aurc": np.nan,
            "formal_confirmation": False,
        })
    cpa = pd.read_csv(CPA / "SUMMARY.csv")
    cmag = float(cpa.loc[cpa.method.eq("Magnitude_raw"), "utility20"].iloc[0])
    for row in cpa.itertuples():
        rows.append({
            "evidence_stage": "SEEN_chemical_stress",
            "dataset": "sciPlex3 eight fixed splits",
            "upstream_family": "CPA",
            "upstream_model": "CPA 0.8.8 RDKIT log-dose",
            "perturbation_type": "chemical",
            "method": row.method,
            "n_tasks": 629,
            "n_strata": row.n_manifests,
            "utility20": row.utility20,
            "delta_utility20_vs_magnitude": row.utility20 - cmag,
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

    summary = pd.read_csv(CONFIRM / "SUMMARY.csv").set_index("method")
    strata = pd.read_csv(CONFIRM / "STRATUM_RESULTS.csv")
    coverage = pd.read_csv(CONFIRM / "RISK_COVERAGE.csv")
    history = pd.read_csv(CONFIRM / "HISTORY_AVAILABILITY_RESULTS.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1))
    methods = ["Magnitude_raw", "Ridge_USR", "V2_nested"]
    labels = ["Magnitude", "V1", "V2"]
    colors = ["#777777", "#1B9E77", "#377EB8"]
    axes[0].bar(labels, [summary.loc[m, "utility20"] for m in methods], color=colors)
    axes[0].set_ylabel("Utility@20")
    axes[0].set_title("a  One-shot confirmation")

    pivot = strata.pivot(index=["panel_id", "culture_condition"], columns="method", values="utility20")
    delta = (pivot["V2_nested"] - pivot["Magnitude_raw"]).sort_values()
    axes[1].barh(np.arange(len(delta)), delta.values,
                 color=np.where(delta.values >= 0, "#377EB8", "#D95F02"))
    axes[1].axvline(0, color="#555555", lw=1)
    axes[1].set_yticks(np.arange(len(delta)), [f"{a}-{b}" for a, b in delta.index], fontsize=7)
    axes[1].set_xlabel("ΔUtility@20")
    axes[1].set_title("b  10/12 strata nonnegative")

    for method, color, label in [("Magnitude_raw", "#777777", "Magnitude"), ("V2_nested", "#377EB8", "V2")]:
        line = coverage[coverage.method.eq(method)].groupby("coverage", as_index=False).mean(numeric_only=True)
        axes[2].plot(line.coverage, line.mean_true_rmse, marker="o", color=color, label=label)
    axes[2].set_xlabel("Accepted low-risk fraction")
    axes[2].set_ylabel("Mean true RMSE")
    axes[2].set_title("c  Risk–coverage")
    axes[2].legend(frameon=False)

    xpos = np.arange(2)
    for j, (method, color, label) in enumerate(zip(methods, colors, labels)):
        part = history[history.method.eq(method)].set_index("target_stratum")
        axes[3].bar(xpos + (j - 1) * 0.24,
                    [part.loc["DONOR_UNSEEN_ONLY", "utility20"], part.loc["COLUMN_UNSEEN", "utility20"]],
                    0.24, color=color, label=label)
    axes[3].axhline(0, color="#555555", lw=1)
    axes[3].set_xticks(xpos, ["History\navailable", "No history"])
    axes[3].set_ylabel("Utility@20")
    axes[3].set_title("d  Availability boundary")
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "E170_CONFIRMATION_RESULTS.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "E170_CONFIRMATION_RESULTS.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Creator": "SafeConf-v4"})
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
    e190 = dev["e190_gears_crossfamily"].set_index("method")
    e190_inc = pd.read_csv(DEV / "e190_gears_crossfamily/INCREMENTAL_RESULTS.csv").set_index("comparison")
    confirm = pd.read_csv(CONFIRM / "SUMMARY.csv").set_index("method")
    confirm_boot = pd.read_csv(CONFIRM / "PAIRED_CLUSTER_BOOTSTRAP.csv").set_index("comparison")
    confirm_gate = json.loads((CONFIRM / "GATE_A_RESULT.json").read_text())
    confirm_history = pd.read_csv(CONFIRM / "HISTORY_AVAILABILITY_RESULTS.csv")
    memory = pd.read_csv(ERROR_MEMORY / "SUMMARY.csv")
    cpa = pd.read_csv(CPA / "SUMMARY.csv").set_index("method")
    cpa_inc = pd.read_csv(CPA / "INCREMENTAL_RESULTS.csv").set_index("comparison")

    report = f"""# SafeConf v4 论文结果包

生成日期：2026-09-29。证据角色：TxPert/E190 为开发或已见数据；E170 四面板在方法冻结、远端授权和字节级核验后一次性打开；E208/E247/E258 测试真值继续封存；E216 为此前已经揭盲的负结果。

## 主结果

{md_table(["上游", "任务", "Magnitude U20", "V1 U20", "V2 U20", "V2−Magnitude", "V2 Spearman"], main_rows)}

开发阶段的 Final Candidate 已按机器门冻结为 `V2_nested_evidence_shrinkage`。合并 8 个评价 strata 的 `V2−V1` 为 **{freeze['aggregate']['delta_u20_macro']:.4f}**，{freeze['aggregate']['nonnegative_strata_fraction']:.1%} strata 非负；所有安全退化门均通过。

## 一次性确认：E170 四面板

{md_table(["方法", "Utility@20", "Spearman", "AURC", "risk@20", "高风险漏检率"], [[m, confirm.loc[m, 'utility20'], confirm.loc[m, 'spearman'], confirm.loc[m, 'aurc'], confirm.loc[m, 'risk_at_20'], confirm.loc[m, 'high_risk_miss_rate']] for m in ['Magnitude_raw', 'Ridge_USR', 'V2_nested']])}

- 冻结 V2 相对 Magnitude：ΔU20=**{confirm_gate['delta_u20_macro']:+.4f}**，12 个预注册 panel×state strata 中 **{confirm_gate['nonnegative_strata_fraction']:.1%}** 非负，Gate A **PASS**。
- 成对 target-cluster bootstrap：ΔU20 95% CI **[{confirm_boot.loc['FinalV2_vs_Magnitude', 'utility_ci95_lower']:+.4f}, {confirm_boot.loc['FinalV2_vs_Magnitude', 'utility_ci95_upper']:+.4f}]**；区间跨 0，说明估计仍不精确，不能写成传统显著性确认。
- safety：risk@10/20 无恶化，risk@50 相对恶化 {confirm_gate['risk50_relative_degradation']:.2%}，AURC 和高风险漏检率均未恶化，全部通过预注册阈值。
- 证据等级：同研究内的新 perturbation + held-out donor confirmation；不是 external-study confirmation。
- 有合法两供体历史的 1,920 tasks：Magnitude/V1/V2 U20 为 **0.2383/0.2718/0.2857**。无历史的 480 column-unseen tasks 三者均接近 0；这是必须保留的适用边界，不把它写成成功的冷启动 fallback。

## PertEMA 公平比较

{md_table(["上游", "PertEMA* U20", "SafeConf V2 U20", "差值", "95% CI"], pertema_rows)}

`PertEMA*` 是官方算法与原超参数在同一 TxPert 任务、同一 gene-disjoint outer split、同一错误标签预算上的适配，不是把官方 CD4 冻结权重直接搬到 TxPert。SafeConf 多使用合法历史证据，因此这里检验的是“历史证据在同标签预算下能否提供增量”，不声称覆盖一切 PertEMA 特征工程。

## 独立 GEARS 家族压力结果

E190 在运行 SafeConf 前通过既定能力门：三 seed GEARS centroid 相对最强 Adamson source-effect baseline 的误差差为 **+0.115%**，48/48 batch strata 在 +2% margin 内，gene-cluster bootstrap 95% CI 为 **[-0.607%, +0.893%]**。该资产此前已开封，因此属于 SEEN cross-family development evidence。

{md_table(["方法", "Utility@20", "Spearman", "AURC"], [[m, e190.loc[m, "utility20"], e190.loc[m, "spearman"], e190.loc[m, "aurc"]] for m in ["Magnitude_raw", "Ridge_U", "Ridge_USR", "Ridge_USRC", "V2_nested"]])}

- V1−Magnitude：ΔU20={e190_inc.loc['V1_vs_magnitude', 'delta_utility20']:+.4f}，95% CI [{e190_inc.loc['V1_vs_magnitude', 'utility_ci95_lower']:+.4f}, {e190_inc.loc['V1_vs_magnitude', 'utility_ci95_upper']:+.4f}]。
- Relevance given Support：ΔU20={e190_inc.loc['Relevance_given_support', 'delta_utility20']:+.4f}；ΔSpearman={e190_inc.loc['Relevance_given_support', 'delta_spearman']:+.4f}。
- V2−V1：ΔU20={e190_inc.loc['V2_vs_V1', 'delta_utility20']:+.4f}。V2 没有跨家族复现 V1 的点增量，作为正式失败边界保留。
- Conflict proxy 带来 ΔSpearman={e190_inc.loc['Conflict_given_relevance', 'delta_spearman']:+.4f}，95% CI [{e190_inc.loc['Conflict_given_relevance', 'spearman_ci95_lower']:+.4f}, {e190_inc.loc['Conflict_given_relevance', 'spearman_ci95_upper']:+.4f}]；该信号来自单一 source study 的 fold dispersion，不能升级成完整 Quality 结论。

## Error Memory 部署增强

{md_table(["上游", "反馈预算", "平均反馈基因", "Utility@20", "Spearman"], [[r.upstream, f"{r.budget:.0%}", r.n_feedback_genes, r.utility20, r.spearman] for r in memory.itertuples()])}

反馈按 gene cluster 从 outer-train 逐步提供，评价基因保持留出。10% 反馈在两个上游都下降；GAT 到 75–100% 后 U20 超过零反馈，Exphormer 在 50–75% 达到平台。该实验支持“按批次周期重训”的部署增强，不冒充真实时间顺序的在线学习。

## 化学扰动压力线

CPA 八个固定 sciPlex3 划分中，Magnitude / P+Q / P+Q+H 的 U20 为 **{cpa.loc['Magnitude_raw', 'utility20']:.4f} / {cpa.loc['Ridge_PQ', 'utility20']:.4f} / {cpa.loc['Ridge_PQH', 'utility20']:.4f}**。P+Q+H 相对 Magnitude 的 ΔU20 为 {cpa_inc.loc['Ridge_PQH_minus_Magnitude_raw', 'delta_utility20']:+.4f}，95% CI [{cpa_inc.loc['Ridge_PQH_minus_Magnitude_raw', 'utility_ci95_lower']:+.4f}, {cpa_inc.loc['Ridge_PQH_minus_Magnitude_raw', 'utility_ci95_upper']:+.4f}]。CPA 本身未通过上游能力门，因此该结果只证明统一接口可以运行并暴露剂量/来源异质性，定位为 chemical stress test。

## 能支持的结论

1. 在两个通过能力门的 TxPert 架构上，Magnitude 是强基线，V1/V2 的开发点估计均进一步提高 Utility@20 和 Spearman。
2. Support 后加入 Relevance，在两个 TxPert 架构上的 Spearman 增量均为正且簇自举区间下限大于 0；历史字段打乱后，V1 的 Spearman 明显下降。E190 GEARS 上同方向但区间宽。
3. 固定 Ridge 比 HGB 和小 MLP 稳定；增加复杂度没有收益。
4. V2 的 40 个 outer fits 中，32 个（80%）产生随任务变化的 evidence weight；K562 多数折退化成常数混合，必须保留为边界。
5. E170 的一次性确认满足预注册 Gate A，证明冻结候选不是只在 TxPert 开发集上成立；其统计区间和同研究证据等级必须完整报告。
6. E190 提供一个能力合格的独立 GEARS family 和真正跨 study 的 Adamson history；V1 点估计超过 magnitude，但 V2 失败且 cluster CI 宽，因此它加强“跨家族可运行/有信号”，不构成确认。
7. Error Memory 在足量同模型反馈后提供增量，但 10% 反馈不稳定，因此只作为周期重训扩展。
8. 化学接口已在 CPA 压力线上运行；CPA 未通过能力门，不能承担化学主确认。
9. Quality 的合法 eligible coverage 为 **{quality_cov:.1%}**。当前结果只能叫 Support/Relevance-aware，不能写成已经证明 Quality-aware。

## 尚未获得的证据

- Gate B 无法评价：冻结时没有合法 Quality 字段，不能事后补入确认。
- external-study confirmation 尚未成立：E170 属于同研究 held-out donor；E190 GEARS 是 692-task 已开封开发压力线；E208/E247/E258 均在打开测试真值前被能力门阻断。
- 外部历史在 E190 中合法存在且有开发点增量，但只有一个 source/target cell context，区间不足以支持广泛 external-public-history 主张。
- V2 相对 V1 的单架构 Utility@20 自举区间均跨 0；不能把开发门通过写成统计确认。

## 当前论文决策

本轮维持 **V2 为最终主方法**：它在 E170 一次性确认中通过预注册 Gate A；没有在看到结果后切换到 V1。当前最有价值的论文骨架是：

> 一个带上游能力门和输出合同的黑盒扰动预测风险审计框架；在合格 TxPert 上开发 Support/Relevance 历史证据与嵌套收缩，在此前封存的 E170 新扰动/留出供体上一次性通过 Gate A，并对无历史与跨家族失效条件实行完整披露。

据此项目进入 **Route A：SafeConf 主方法论文**。投稿增强项是 external-study/cross-family confirmation，而不是重新选择方法；不能用已揭盲 E190/E216、弱上游或重新切分看过的数据伪装该证据。
"""
    (OUT / "PAPER_RESULT_TABLES.md").write_text(report, encoding="utf-8")

    method = f"""# SafeConf v4 方法决策

## 决策

- Final Candidate：`V2_nested_evidence_shrinkage`。
- 冻结依据：合并开发 `ΔUtility@20(V2−V1)={freeze['aggregate']['delta_u20_macro']:.6f}`，非负 strata 比例 {freeze['aggregate']['nonnegative_strata_fraction']:.1%}，有效 strata 100%。
- 证据等级：**同研究 held-out-donor/new-perturbation confirmation**。E170 一次性 Gate A 通过；Gate B 因无合法 Quality 字段不可评价。
- V1 保留为预指定 secondary benchmark，confirmation 失败后不得用同一结果把 V1 事后改封为主方法。

## 方法结构

1. `rP`：Universal Prediction Evidence 的 Ridge 风险。
2. `rPQ`：Prediction + Support + Relevance + Conflict proxy + Content 的 Ridge 风险。
3. 两个分数只在 inner OOF 上校准到相同误差尺度。
4. 单调低容量 gate 仅使用预测前 evidence，得到 `rP + w(rPQ-rP)`。
5. outer-test 只执行 outer-train 冻结的标准化、校准和 gate。

## 独立判断

当前真正成立的是 Support/Relevance-aware shrinkage；Quality 未进入正式实证。V2 的价值来自在历史模块整体不稳时学习收缩，而不是因为更大的网络。小 MLP 明显落后，因此不扩 Transformer 或 set encoder。

E170 在方法冻结后一次性打开全部四面板，V2 相对 Magnitude 的 ΔU20 为 +0.0504，10/12 strata 非负并通过所有安全护栏，因此 V2 升格为最终主方法。其 target-cluster bootstrap CI 跨 0，证据属于同研究新扰动/留出供体确认，不扩写为 external-study confirmation。E190 表明 V1 的 Support/Relevance 点增量可延伸到独立 GEARS family，但冻结 V2 没有复现，因此跨 family 仍按开发压力证据表述。
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
| E190 GEARS V2 | 独立 family 仅 47 个 gene clusters，gate 在一折退回 prediction-only | 核验输入哈希、同尺度校准、inner/outer 隔离与 fold gate；未发现执行错误 | V1 点增量为正但区间宽；V2−V1 为负 | 保留为跨家族失败边界，不修改冻结候选 |
| E170 truth builder | 旧 E168 Git ancestry 在当前快照分支不可证明，构建器在读取 test X 前 fail-closed | 在揭盲前登记一次代码合同修复：改为核验冻结 RUN_STATUS、wrapper/helper 与 F2 manifest 的精确字节哈希 | 42GB 源哈希一致，四面板同开，未在修复时读取 truth | 合法修复；授权与修复均先提交远端 |
| E170 confirmation | 冻结 V2 需要一次性确认 | 所有 2,400 tasks、12 strata 完整运行，不删无历史列 | Gate A PASS；ΔU20=+0.0504，10/12 strata 非负；bootstrap CI 跨 0 | Route A 成立，同时保留统计不精确和 no-history 边界 |
"""
    (OUT / "FAILURE_AND_REPAIR_LOG.md").write_text(failures, encoding="utf-8")

    claims = """# SafeConf v4 Claim–Evidence Matrix

| 论文主张 | 证据 | 当前裁决 | 论文措辞 |
| --- | --- | --- | --- |
| 输出接口统一 | Prediction/Error Contract；direct-effect 与 treated-state 两路径 | 支持 | common black-box adapter contract |
| 幅度是强风险基线 | TxPert 两架构 U20 0.759–0.768，Spearman 0.728–0.742 | 支持 | strong mandatory baseline |
| Support/Relevance 有额外信息 | TxPert 开发增量；E170 history-available 组 U20 0.2383→0.2718/0.2857 | 同研究确认支持 | confirmed on new perturbations/held-out donor; external confirmation pending |
| V2 可安全收缩历史 | 开发门通过；E170 Gate A PASS，10/12 strata 非负，risk guards 通过 | 支持，带边界 | confirmed under registered same-study contract; no-history remains weak |
| Quality-aware | TxPert legal Quality coverage 0 | 不支持 | 不使用该主张 |
| External cross-study history | E190 使用 target-truth 前冻结的 Adamson history；V1 点增量为正但 CI 宽 | 初步支持 | cross-study historical evidence; not confirmed broadly |
| 跨结构 | TxPert GAT 与 Exphormer 同方向 | 支持 | cross-architecture within one family |
| 跨模型家族/model-agnostic | E190 GEARS 692 tasks 通过能力门；V1 点增量为正，V2 失败，且资产已开封 | 开发压力支持，未确认 | black-box interface with cross-family development evidence |
| 胜过 PertEMA | 同任务适配中 V2 高 0.066–0.069，CI 下限 >0 | 支持限定比较 | beats official-algorithm adaptation under registered contract |
| 独立 confirmation | E170 全四面板在方法冻结后同开；2,400 tasks；Gate A PASS | 同研究确认完成 | one-shot held-out-donor/new-perturbation confirmation, not external-study |
| Error Memory 部署增强 | gene-disjoint 反馈预算曲线；GAT 75–100%、Exphormer 50–75% 后改善 | 开发支持 | periodic batch adaptation; not online temporal learning |
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
4. E170 的 V2−Magnitude bootstrap CI 跨 0，且 column-unseen 子集接近无信息；应完整报告，不把 Gate A PASS 写成统计显著或无历史普适性。
5. E170 是同研究确认；external-study、跨独立 family 的强 model-agnostic 主张仍需额外证据。
6. 旧 SafeConf 证书线与本次历史风险线是不同研究对象，稿件不能混成一个模糊贡献。
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
5. E190 GEARS 跨 study 压力线：V1 信号与 V2 gate 失效边界。
6. 同合同 PertEMA 适配比较。
7. E170 一次性确认：Gate A、风险护栏、history-available/no-history 边界。
8. E208/E216/E247/E258 展示能力门和适用边界。

## 4. Discussion

- 主确认限于基因扰动与同研究 held-out donor；跨 family 只有开发压力证据。
- Quality/external history/chemical 仍是扩展，不包装为已完成。
- 上游 competence 是风险审计的前置条件。
- 部署流程：冻结上游→生成 PredictionRecord→冷启动风险→积累合法反馈→周期重训。

## 5. 投稿增强项

- 外部研究或独立 family 的 sealed confirmation；E190 只能作为已开封开发证据。
- 合法 Quality 字段与化学成熟上游属于扩展，不阻塞当前 Route A 稿件。
- 继续保留无历史任务和 E190 gate 失败，不以子集筛选删除负结果。
"""
    (OUT / "MANUSCRIPT_OUTLINE.md").write_text(outline, encoding="utf-8")

    figure_plan = """# SafeConf v4 论文主图规划

1. **Figure 1：方法架构。** Frozen upstream → Universal P；History Support/Relevance/Conflict/Missingness → nested evidence weight → task risk → selective review。
2. **Figure 2：主结果。** 已生成 `SAFECONF_V4_MAIN_RESULTS.pdf/png`，包含 U20、成对增量和 risk-coverage。
3. **Figure 3：Evidence weight。** 各 context/fold 的 gate weight 分布与系数；突出 K562 常数退化边界。
4. **Figure 4：历史来源与 shortcut。** history shuffle、matched-support、within-context 结果。
5. **Figure 5：一次性确认。** 已生成 `E170_CONFIRMATION_RESULTS.pdf/png`，含主结果、12 strata、risk-coverage 和 history availability 边界。
6. **Figure 6：跨家族与失败边界。** E190 GEARS 的 V1 正向点增量、V2 gate 失败，以及 E208/E216/E247/E258 的 fail-closed 流程。
7. **Supplement：** 全 strata、bootstrap、MLP/HGB 消融、Error Memory budget、化学压力测试。
"""
    (OUT / "PAPER_FIGURE_PLAN.md").write_text(figure_plan, encoding="utf-8")

    repro_files = [
        OUT / "FINAL_METHOD_CONFIG.json", OUT / "FINAL_CANDIDATE_FREEZE.json",
        OUT / "FIELD_AVAILABILITY_MATRIX.csv", OUT / "HISTORY_SOURCE_AUDIT.csv",
        OUT / "UPSTREAM_COMPETENCE_V4.csv", OUT / "EXPERIMENT_MASTER_TABLE.csv",
        PERTEMA / "RUN_STATUS.json", PERTEMA / "SUMMARY.csv",
        DEV / "e190_gears_crossfamily/RUN_STATUS.json",
        DEV / "e190_gears_crossfamily/SUMMARY.csv",
        OUT / "E170_V4_CONFIRMATION_AUTHORIZATION.json",
        CONFIRM / "RUN_STATUS.json", CONFIRM / "SUMMARY.csv",
        CONFIRM / "GATE_A_RESULT.json", CONFIRM / "PAIRED_CLUSTER_BOOTSTRAP.csv",
        ERROR_MEMORY / "RUN_STATUS.json", ERROR_MEMORY / "SUMMARY.csv",
        CPA / "RUN_STATUS.json", CPA / "SUMMARY.csv", CPA / "INCREMENTAL_RESULTS.csv",
    ]
    hashes = [[str(p.relative_to(ROOT)), sha256(p)] for p in repro_files if p.exists()]
    repro = f"""# SafeConf v4 复现报告

## 运行合同

- 开发：4 contexts × 5 gene-disjoint folds；outer task 从未进入本折风险模型训练。
- V2：inner OOF 生成 rP/rPQ、outer-train 内尺度校准和单调 gate。
- 主指标：`ceil(0.2*n)`、risk 降序、task_id 字典序处理 ties。
- Bootstrap：5,000 次，perturbation gene cluster 为重采样单位。
- SEALED：Final Candidate 冻结前未读取 E170 test 结果、字段覆盖或 truth；四面板在授权提交远端后一次性打开。E216 后验识别为此前已揭盲并降级为 SEEN。
- PertEMA：官方软件 commit `43c09a32e23d0ee2ae5dfbab21b2deeab27f1803`，官方 XGB 超参数，同 outer split/label budget。

## 关键文件哈希

{md_table(["文件", "SHA-256"], hashes)}

## 复跑命令

```bash
python tools/scripts/build_safeconf_governance_artifacts.py
python tools/scripts/run_safeconf_v4_development.py --upstream e201
python tools/scripts/run_safeconf_v4_development.py --upstream e205
python tools/scripts/audit_safeconf_competence_gate.py
python tools/scripts/run_e190_gears_safeconf_v4.py
/tmp/pertema-venv/bin/python tools/scripts/run_safeconf_pertema_fair_comparison.py
python tools/scripts/freeze_safeconf_final_candidate.py
python tools/scripts/register_e170_v4_confirmation.py
python tools/scripts/build_e170_v4_confirmation_truth.py
python tools/scripts/run_e170_safeconf_v4_confirmation.py
python tools/scripts/build_safeconf_v4_paper_package.py
```

结果脚本默认拒绝覆盖已经存在的 RUN_STATUS，正式复跑应使用新输出目录或先保留原证据快照。
"""
    (OUT / "REPRODUCIBILITY_REPORT.md").write_text(repro, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = build_confirmation_audit()
    build_confirmation_field_audit()
    dev, pertema, boot = load_results()
    master = build_master(dev, pertema)
    build_figures(dev, pertema, boot)
    build_reports(dev, pertema, boot, master, audit)
    status = {
        "status": "COMPLETE",
        "sealed_test_truth_opened_by_this_script": False,
        "final_candidate": "V2_nested_evidence_shrinkage",
        "gate_a": "PASS_E170_ONE_SHOT_HELDOUT_DONOR_NEW_PERTURBATIONS",
        "gate_b": "NOT_EVALUABLE_NO_FROZEN_QUALITY_FIELDS",
        "gate_c": "NOT_YET_INVOKED",
        "paper_route": "A_SAFECONF_MAIN_METHOD",
        "n_master_rows": int(len(master)),
        "n_confirmation_candidates_audited": int(len(audit)),
        "main_figure_sha256": sha256(OUT / "SAFECONF_V4_MAIN_RESULTS.pdf"),
        "confirmation_figure_sha256": sha256(OUT / "E170_CONFIRMATION_RESULTS.pdf"),
        "confirmation_result_sha256": sha256(CONFIRM / "RUN_STATUS.json"),
    }
    (OUT / "PAPER_PACKAGE_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
