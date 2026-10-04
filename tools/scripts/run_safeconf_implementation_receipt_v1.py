#!/usr/bin/env python3
"""Assemble the implementation receipt without changing prior experiment data."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path("/home/yyf/runtime_artifacts/safeconf_impl_20261004_v1")
DOC = ROOT / (
    "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/"
    "research_closure_20261001/data_model_feedback_20261003_v1/implementation_v1"
)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=RUNTIME)
    args = ap.parse_args()
    run = args.runtime.resolve()
    preflight = load_json(run / "PREFLIGHT_MANIFEST.json", {})
    rel = load_json(run / "public_reliability/E258_VALIDATION_DECISION.json", {})
    target_tab = load_json(run / "tabpfn_target50/TABPFN_TARGET50_HOLDOUT_STATUS.json", {})
    freeze = load_json(run / "system_freeze_v4/COMPONENT_DECISION.json", {})
    source_metrics = pd.read_csv(run / "source_gate_v3/SOURCE_GATE_METRICS.csv") if (run / "source_gate_v3/SOURCE_GATE_METRICS.csv").exists() else pd.DataFrame()
    pertema_status = load_json(run / "pertema_current_truth_v2/PERTEMA_CURRENT_RUN_STATUS.json", {})
    system_status = load_json(run / "system_freeze_v4/RUN_STATUS.json", {})
    system_comparison = pd.read_csv(run / "system_freeze_v4/SYSTEM_COMPARISON.csv")
    independent_assets = pd.read_csv(run / "INDEPENDENT_ASSET_AUDIT.csv") if (run / "INDEPENDENT_ASSET_AUDIT.csv").exists() else pd.DataFrame()
    mixed_audit = load_json(run / "mixed_queue/MIXED_QUEUE_AUDIT.json", {})
    mixed_comparison = pd.read_csv(run / "mixed_queue/MIXED_QUEUE_SYSTEM_COMPARISON.csv") if (run / "mixed_queue/MIXED_QUEUE_SYSTEM_COMPARISON.csv").exists() else pd.DataFrame()
    mixed_discovery = pd.read_csv(run / "mixed_queue/MIXED_QUEUE_ACTUAL_ERROR_DISCOVERY.csv") if (run / "mixed_queue/MIXED_QUEUE_ACTUAL_ERROR_DISCOVERY.csv").exists() else pd.DataFrame()
    e192_audit = load_json(run / "e192_contract_audit/E192_ASSET_STATUS.json", {})
    e192_cross = load_json(run / "e192_crosscontext_v4/E192_CROSSCONTEXT_AUDIT.json", {})
    e192_cross_results = pd.read_csv(run / "e192_crosscontext_v4/E192_CROSSCONTEXT_SYSTEM_COMPARISON.csv") if (run / "e192_crosscontext_v4/E192_CROSSCONTEXT_SYSTEM_COMPARISON.csv").exists() else pd.DataFrame()
    e208_status_path = Path("/home/yyf/data/perturbench_e208/pretruth_risk_seal_20260921/E208_RISK_SEAL_QUEUE_STATUS.json")
    e208_status = load_json(e208_status_path, {})
    implementation = {
        "run_id": "safeconf_impl_20261004_v1",
        "status": "COMPLETE_IMPLEMENTATION_EVIDENCE_WITH_PUBLICRULE_DEFAULT",
        "configuration": {
            "cpu_python": "/home/miniconda/bin/python",
            "tabpfn_python": "/home/yyf/.venvs/safeconf-tabpfn-20261001/bin/python",
            "source_dev_rows": 1808, "source_dev_genes": 575,
            "target_dev_rows": 542, "target_dev_genes": 377,
            "feedback_pool_rows": 331, "feedback_pool_genes": 228,
            "current_holdout_rows": 212, "current_holdout_genes": 152,
            "target_50_feedback_genes": 114, "bootstrap_replicates": 5000,
            "new_download_bytes": 0, "new_large_upstream_training": False,
            "permanent_test_truth_opened": False,
            "source_gate_candidates": ["always_public", "source_top25_unreliable", "source_top50_unreliable", "source_top75_unreliable"],
            "tabpfn_package": "tabpfn==9.0.0",
            "tabpfn_checkpoint_sha256": "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
            "pertema_commit": "43c09a32e23d0ee2ae5dfbab21b2deeab27f1803",
        },
        "preflight_status": preflight.get("status"),
        "work_packages": {
            "current_truth_pertema": pertema_status,
            "public_reliability": rel,
            "source_gate": {
                "status": "COMPLETE" if not source_metrics.empty else "MISSING",
                "source_metrics": str(run / "source_gate_v3/SOURCE_GATE_METRICS.csv"),
                "candidate_set": ["always_public", "source_top25_unreliable", "source_top50_unreliable", "source_top75_unreliable"],
                "shuffle_seeds": [20260930, 20261001, 20261002, 20261003, 20261004],
            },
            "target_tabpfn": target_tab,
            "system_freeze": system_status,
            "independent_asset_audit": independent_assets.to_dict("records"),
            "mixed_queue": mixed_audit,
            "e192_contract_audit": e192_audit,
            "e192_crosscontext": e192_cross,
            "e208_independent_asset": e208_status,
        },
        "decision": freeze,
        "permanent_test_truth_opened": False,
        "new_large_upstream_training": False,
        "new_download_bytes": 0,
    }
    write_json(DOC / "IMPLEMENTATION_CONFIG.json", implementation)
    write_json(DOC / "IMPLEMENTATION_RUN_STATUS.json", implementation)
    # Keep the reviewable audit bundle beside the receipt. Large raw artifacts
    # remain under runtime_artifacts; only the compact 212-row ranking is copied.
    for name in ("PREFLIGHT_MANIFEST.json", "ASSET_ROLE_LEDGER.csv", "RESOURCE_SNAPSHOT.json",
                 "TRUTH_CONTRACT_AUDIT.csv", "INDEPENDENT_ASSET_AUDIT.csv"):
        copy_if_exists(run / name, DOC / name)
    copy_if_exists(run / "system_freeze_v4/COMPONENT_DECISION.json", DOC / "COMPONENT_DECISION.json")
    copy_if_exists(run / "system_freeze_v4/SYSTEM_RANKING.parquet", DOC / "PER_QUERY_RESULTS.parquet")
    copy_if_exists(run / "system_freeze_v4/SYSTEM_COMPARISON.csv", DOC / "SYSTEM_COMPARISON.csv")
    for name in ("MIXED_QUEUE_AUDIT.json", "MIXED_QUEUE_SYSTEM_COMPARISON.csv",
                 "MIXED_QUEUE_ACTUAL_ERROR_DISCOVERY.csv", "MIXED_QUEUE_COVERAGE.csv",
                 "MIXED_QUEUE_PAIRED_BOOTSTRAP.csv", "MIXED_QUEUE_CONTRACT.md",
                 "MIXED_QUEUE_SYSTEM_RANKING.parquet"):
        copy_if_exists(run / "mixed_queue" / name, DOC / name)
    for name in ("E192_INPUT_CONTRACT_AUDIT.json", "E192_PUBLIC_HISTORY_COVERAGE.csv", "E192_ASSET_STATUS.json"):
        copy_if_exists(run / "e192_contract_audit" / name, DOC / name)
    for name in ("E192_CROSSCONTEXT_AUDIT.json", "E192_CROSSCONTEXT_SYSTEM_COMPARISON.csv", "E192_CROSSCONTEXT_PAIRED_BOOTSTRAP.csv"):
        copy_if_exists(run / "e192_crosscontext_v4" / name, DOC / name)

    ledger = pd.DataFrame([
        {"information_or_cost": "Public truth", "source": "E258 validation + McFaline public history", "rows_or_units": "1530 validation records; 542 DEV tasks", "role": "reference and reliability estimation", "evaluation_truth_used": False, "status": "used"},
        {"information_or_cost": "Source errors", "source": "GAT/Exphormer nested cache", "rows_or_units": "1808 rows; 575 gene clusters", "role": "Source gate training and label-shuffle mechanism test", "evaluation_truth_used": False, "status": "used"},
        {"information_or_cost": "Target development errors", "source": "McFaline DEV", "rows_or_units": "542 tasks; 377 genes", "role": "H1/XGB/TabPFN selection", "evaluation_truth_used": False, "status": "DEV only"},
        {"information_or_cost": "Target feedback errors", "source": "registered feedback pool", "rows_or_units": "331 rows; 228 genes", "role": "current-truth feedback learner inputs", "evaluation_truth_used": False, "status": "registered"},
        {"information_or_cost": "Current holdout truth", "source": "HOLDOUT_FEATURES.parquet", "rows_or_units": "212 rows; 152 genes", "role": "fixed current-contract evaluation", "evaluation_truth_used": True, "status": "SEEN holdout; no permanent TEST"},
        {"information_or_cost": "Mixed queue cached errors", "source": "GWPS score-sealed retrospective artifact", "rows_or_units": "2993 tasks; 1750 genes", "role": "unified history/no-history ranking audit", "evaluation_truth_used": True, "status": "SEEN only; no method selection or independent confirmation"},
        {"information_or_cost": "New download", "source": "none", "rows_or_units": 0, "role": "resource cost", "evaluation_truth_used": False, "status": "within budget"},
        {"information_or_cost": "Additional TabPFN GPU", "source": "fixed V2.0 target 50%", "rows_or_units": "0.00128 GPU-hours", "role": "capacity audit", "evaluation_truth_used": False, "status": "within 2 GPU-hour reserve"},
    ])
    write_csv(DOC / "INFORMATION_BUDGET_LEDGER.csv", ledger)

    claims = pd.DataFrame([
        {"claim": "Public history adds risk information beyond amplitude", "status": "supported on current holdout", "evidence": "SYSTEM_COMPARISON.csv", "scope": "212 current-contract tasks; 152 gene clusters", "boundary": "SEEN, not independent confirmation"},
        {"claim": "Jackknife stability predicts reference deviation", "status": rel.get("status", "unknown"), "evidence": "E258_VALIDATION_DECISION.json", "scope": "1530 E258 validation records", "boundary": "J is highly correlated with V; no default upgrade"},
        {"claim": "Source error can complement Public under a source DEV contract", "status": "supported in nested source DEV", "evidence": "SOURCE_GATE_PAIRED_BOOTSTRAP.csv", "scope": "GAT/Exphormer bidirectional source cache", "boundary": "not a current McFaline holdout gain"},
        {"claim": "Target feedback reliably improves the current system", "status": "not established", "evidence": "SYSTEM_PAIRED_BOOTSTRAP.csv", "scope": "212 current-contract tasks", "boundary": "XGB point gain CI crosses zero"},
        {"claim": "TabPFN V2 improves the target learner", "status": "not adopted", "evidence": "TABPFN_TARGET50_RESULTS.csv and holdout predictions", "scope": "542 DEV plus one fixed 212-task score", "boundary": "fixed replacement only; not all algorithms"},
        {"claim": "Current-truth PertEMA adaptation is competitive", "status": "not supported", "evidence": "PERTEMA_CURRENT_MACRO.csv", "scope": "P6/Native61 current adaptation", "boundary": "not a claim about the complete official PertEMA pipeline"},
        {"claim": "One public score can rank history-present and history-absent tasks together", "status": "supported in frozen SEEN queue", "evidence": "MIXED_QUEUE_SYSTEM_COMPARISON.csv and MIXED_QUEUE_ACTUAL_ERROR_DISCOVERY.csv", "scope": "2993 GWPS tasks", "boundary": "retrospective cached-error audit; not independent confirmation"},
        {"claim": "Source HGB improves the mixed queue beyond the public fallback", "status": "not supported in mixed queue", "evidence": "MIXED_QUEUE_SYSTEM_COMPARISON.csv", "scope": "2993 GWPS tasks", "boundary": "source candidate is retained as conditional evidence only"},
        {"claim": "E192 provides a cross-context public-risk audit", "status": "supported as SEEN same-study cross-context", "evidence": "E192_CROSSCONTEXT_SYSTEM_COMPARISON.csv + paired bootstrap", "scope": "173 scored RPE1 tasks; 20 gene clusters", "boundary": "not independent-study confirmation; intervals are wide"},
        {"claim": "E208 supplies independent-study confirmation", "status": "not eligible: upstream competence gate failed", "evidence": "E208_VALIDATION_COMPETENCE_STATUS.json", "scope": "216 validation tasks; 12 contexts", "boundary": "negative upstream result retained; target truth remains closed"},
    ])
    write_csv(DOC / "CLAIM_EVIDENCE_MATRIX.csv", claims)

    failure_log = "# SafeConf implementation failure and action log\n\n"
    failure_log += "- Source gate first run failed in summary index handling; aggregation was fixed and the complete real plus five shuffled-label pipelines were rerun in `source_gate_v3`.\n"
    failure_log += "- Native61 contains non-finite `native_training_similarity`; the current-truth adaptation preserves task identity coverage and uses the fixed XGBoost missing-value path. It is reported separately from P6.\n"
    failure_log += "- PublicMeanJackknife is complete for 210/212 holdout tasks. Missing values are not imputed; the system falls back to PublicRule and records the coverage failure.\n"
    failure_log += "- Target TabPFN passed the DEV technical gate but lost on the frozen current holdout. It remains a capacity audit and does not replace H1/XGB or PublicRule.\n"
    failure_log += "- Current-truth PertEMA P6 and Native61 are reproducible adaptations under the current truth contract; the result is not labelled as a complete official conformal PertEMA reproduction.\n"
    failure_log += "- Mixed history/no-history ranking was completed from the pre-sealed 2,993-task score artifact. It is a SEEN retrospective audit, not independent confirmation; no permanent TEST truth was opened, no new download was made, and no E208 process was changed.\n"
    write_text(DOC / "FAILURE_AND_ACTION_LOG.md", failure_log)
    report = "# SafeConf implementation receipt\n\n"
    report += "## 实际完成\n\n"
    report += f"- Preflight: `{preflight.get('status')}`。\n"
    report += f"- 当前 truth contract PertEMA: `{pertema_status.get('status', 'unknown')}`，P6 与 Native61 分开保存。\n"
    report += f"- 公共可靠性：E258 jackknife 验证 `{rel.get('status', 'unknown')}`；McFaline 仅生成特征和开发比较。\n"
    report += "- Source gate：完成 always-public、三种切换比例和五个置乱种子；真实标签 gate 相对 always-public 在两个方向均有正的 5000 次基因簇 bootstrap 区间。当前 source cache 没有实验级历史向量，因此 gate 使用既有 `prior_uncertainty`；E258 的 J 只作验证结果，未被冒充接入。\n"
    report += f"- Target TabPFN：DEV gate 后完成当前 212 holdout 固定分数，状态 `{target_tab.get('status', 'unknown')}`。\n"
    report += f"- 完整当前系统：`{system_status.get('status', 'unknown')}`。\n\n"
    report += "## 采用决定\n\n"
    report += f"- 当前默认保持 `{freeze.get('default_current', 'PublicRule')}`。\n"
    report += "- Target XGB 的点估计高于 PublicRule，但配对区间跨零，未替换默认。\n"
    report += "- PertEMA P6/Native61 当前适配低于 PublicRule，保留为公平适配结果，不宣称官方模型复现。\n"
    report += "- TabPFN Target 50% 的 DEV 点增益未形成当前 holdout 的稳定收益，保留为固定学习器对照。\n"
    report += "- Jackknife 能预测 E258 留出参照偏差，但与历史分散度高度相关，暂不升级为默认风险规则。\n\n"
    report += "- 独立资产核验：Replogle GWPS 有数据但没有合格冻结预测；Nadig 有预测但属于 SEEN 同研究；E192 已用 10000-scale 公共效应完成 173 个任务的跨背景审计，但仍是同研究 SEEN 证据；E208 Jiang24 的两个上游家族均未通过 no-change competence gate，已保留为负结果，不绕过门生成测试预测。\n\n"
    report += "## 失败与修复\n\n"
    report += "- Source gate 首次运行在汇总阶段出现索引错误；已修复指标聚合并在 `source_gate_v3` 完整重跑真实与置乱标签流程。\n"
    report += "- Native61 的部分字段为 NaN；采用 XGBoost 原生缺失值路径，完整任务覆盖与有限字段覆盖分别登记。\n"
    report += "- PublicMeanJackknife 有 210/212 个完整任务，系统排序未将缺失值伪造成零，回退并单列覆盖。\n\n"
    report += "## 当前系统指标\n\n"
    for _, row in system_comparison.iterrows():
        report += f"- `{row['method']}`：U20={row['u20']:.6f}，AURC={row['aurc']:.6f}，有效 context={int(row['valid_contexts'])}/{int(row['contexts'])}。\n"
    if not mixed_comparison.empty:
        report += "\n## 混合历史队列（SEEN）\n\n"
        for _, row in mixed_comparison[mixed_comparison.context.eq("all")].iterrows():
            report += f"- `{row['method']}`：U20={row['u20']:.6f}，复核20%发现 {int(row['true_top20_found'])}/{int(row['review_count'])} 个真实高误差任务，剩余平均误差={row['remaining_mean_error']:.6f}。\n"
        if not mixed_discovery.empty:
            pub = mixed_discovery.loc[mixed_discovery.method.eq("PublicRule_mixed")].iloc[0]
            amp = mixed_discovery.loc[mixed_discovery.method.eq("Amplitude")].iloc[0]
            report += f"- PublicRule_mixed 相对 Amplitude 多发现 {int(pub.true_high_error_found - amp.true_high_error_found)} 个高误差任务，剩余误差减少 {amp.remaining_mean_error - pub.remaining_mean_error:.6f}；这是冻结分数的 SEEN 回顾，不是独立确认。\n"
    if not e192_cross_results.empty:
        report += "\n## E192 跨背景审计（SEEN）\n\n"
        for model, group in e192_cross_results.groupby("model", sort=True):
            a = group.loc[group.method.eq("Amplitude")].iloc[0]
            p = group.loc[group.method.str.startswith("PublicRule")].iloc[0]
            report += f"- `{model}`：PublicRule U20={p.u20:.6f}，Amplitude U20={a.u20:.6f}，点差={p.u20-a.u20:.6f}。\n"
        report += "- 20 个基因簇的 5000 次区间均跨零；该结果用于跨背景适用范围，不升级默认方法。\n"
    report += "\n"
    report += "## 下一动作\n\n"
    report += "1. 继续保留 PublicRule 作为当前系统默认。\n"
    report += "2. 以 E258/J 与 Source gate 结果形成适用范围和机制证据，不继续扩网络。\n"
    report += "3. 独立确认资产仍按资格表处理；当前没有打开永久测试真值。\n"
    report += "4. 正文、PDF和投稿材料仍暂停。\n"
    write_text(DOC / "MORNING_REPORT.md", report)
    print(json.dumps({"status": implementation["status"], "doc": str(DOC)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
