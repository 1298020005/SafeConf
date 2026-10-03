#!/usr/bin/env python3
"""Assemble a non-paper SafeConf evidence package from versioned results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001"
DATA = BASE / "data_model_feedback_20261003_v1"
DEFAULT_OUT = DATA / "system_evidence_v1"


def run(out: Path, source_dir: Path | None = None, target_dir: Path | None = None):
    out.mkdir(parents=False, exist_ok=False)
    source_dir = source_dir or (DATA / "unified_fusion_v4")
    target_dir = target_dir or (DATA / "target_ridge_fusion_v8")
    rows = []
    # Current-contract public evidence; retain cohorts separately.
    pub = pd.read_csv(DATA / "gwps/ALL_METRICS.csv")
    for _, r in pub[pub.metric.eq("utility20")].iterrows():
        rows.append({
            "line": "Public",
            "contract": "GWPS_current",
            "cohort": r.cohort,
            "method": r.method,
            "budget": "none",
            "n_tasks": int(r.n_tasks),
            "n_genes": int(r.n_genes),
            "u20": r.point,
            "ci95_lower": r.ci95_lower,
            "ci95_upper": r.ci95_upper,
            "source_errors_used": False,
            "target_errors_used_for_fit": False,
            "status": "current_public_evidence",
        })
    # Source-supervised candidate and its fixed target evaluation.
    src = pd.read_csv(source_dir / "METRICS.csv")
    for _, r in src.iterrows():
        rows.append({
            "line": "SourceFusion",
            "contract": "E201_source_to_McFaline_current_external",
            "cohort": "all_target_rows",
            "method": r.method,
            "budget": "source_full",
            "n_tasks": int(r.n_tasks),
            "n_genes": None,
            "u20": r.u20,
            "ci95_lower": None,
            "ci95_upper": None,
            "source_errors_used": True,
            "target_errors_used_for_fit": False,
            "status": "current_candidate_point_estimate",
        })
    # Target feedback candidate and locked 212-task evaluation.
    tgt = pd.read_csv(target_dir / "METRICS.csv")
    for _, r in tgt.iterrows():
        rows.append({
            "line": "TargetFusion",
            "contract": "McFaline_DEV_to_current_212_holdout",
            "cohort": "212_holdout",
            "method": r.method,
            "budget": r.budget,
            "n_tasks": int(r.n_tasks),
            "n_genes": 152,
            "u20": r.u20,
            "ci95_lower": None,
            "ci95_upper": None,
            "source_errors_used": False,
            "target_errors_used_for_fit": r.method == "TargetRidge_fusion",
            "status": "current_candidate_point_estimate",
        })
    evidence = pd.DataFrame(rows)
    evidence.to_csv(out / "SYSTEM_COMPARISON.csv", index=False)

    # The bootstrap references are deliberately attached separately: no
    # cross-contract comparison is silently promoted to one table.
    src_boot = pd.read_csv(source_dir / "bootstrap/BOOTSTRAP_SUMMARY.csv")
    tgt_boot = pd.read_csv(target_dir / "bootstrap_b050/BOOTSTRAP_SUMMARY.csv")
    src_boot.insert(0, "line", "SourceFusion")
    src_boot.insert(1, "contract", "E201_source_to_McFaline_current_external")
    tgt_boot.insert(0, "line", "TargetFusion")
    tgt_boot.insert(1, "contract", "McFaline_DEV_to_current_212_holdout")
    pd.concat([src_boot, tgt_boot], ignore_index=True).to_csv(out / "BOOTSTRAP_SUMMARY.csv", index=False)

    claims = pd.DataFrame([
        {"claim": "Public history provides information beyond amplitude", "evidence": "gwps/ALL_METRICS.csv; public content null outputs", "status": "SUPPORTED_WITH_SCOPE", "decision": "retain Public module"},
        {"claim": "GWPS increases legal coverage", "evidence": "gwps/ACTUAL_COVERAGE.csv; gwps/ALL_METRICS.csv", "status": "SUPPORTED_COVERAGE_SEPARATE_FROM_ACCURACY", "decision": "retain coverage extension"},
        {"claim": "Source-supervised Ridge adds beyond current public rule", "evidence": str(source_dir / "METRICS.csv") + "; " + str(source_dir / "bootstrap/BOOTSTRAP_SUMMARY.csv"), "status": "POINT_GAIN_CI_CROSSES_ZERO", "decision": "conditional candidate; do not promote"},
        {"claim": "Target Ridge is useful at a fixed feedback budget", "evidence": str(target_dir / "METRICS.csv") + "; " + str(target_dir / "bootstrap_b050/BOOTSTRAP_SUMMARY.csv"), "status": "NO_STABLE_GAIN_AT_50_PERCENT", "decision": "do not make default"},
        {"claim": "Legacy matrix/PertEMA scores are a same-task baseline", "evidence": str(source_dir / "LEGACY_ALIGNMENT_AUDIT.csv") + "; " + str(target_dir / "LEGACY_ALIGNMENT_AUDIT.csv"), "status": "REJECTED_TRUTH_CONTRACT_MISMATCH", "decision": "keep as SEEN audit only"},
        {"claim": "Full SafeConf is ready for one combined ranking", "evidence": "unified scoring interface plus current module outputs", "status": "NOT_YET_CLOSED", "decision": "next action: regenerate all channels under one current truth contract"},
    ])
    claims.to_csv(out / "CLAIM_EVIDENCE_MATRIX.csv", index=False)

    ledger = pd.DataFrame([
        {"information": "Public truth", "purpose": "GWPS reference and coverage", "rows": "see GWPS manifest", "used_for_fit": True},
        {"information": "Source errors", "purpose": "SharedRisk and Source Ridge fusion", "rows": 3616, "used_for_fit": True},
        {"information": "Target development errors", "purpose": "Target Ridge selection and fitting", "rows": 542, "used_for_fit": True},
        {"information": "Target holdout errors", "purpose": "evaluation only", "rows": 212, "used_for_fit": False},
        {"information": "Legacy matrix/PertEMA errors", "purpose": "alignment audit only", "rows": "excluded after truth mismatch", "used_for_fit": False},
    ])
    ledger.to_csv(out / "INFORMATION_BUDGET_LEDGER.csv", index=False)

    decision = {
        "status": "PARTIAL_EXECUTION_COMPLETE",
        "default_public": "current legal Public rule / weighted history contract",
        "source_ridge": {"decision": "CONDITIONAL_NOT_DEFAULT", "reason": "point gain over current simple rule but 5000-cluster CI crosses zero"},
        "target_ridge": {"decision": "CONDITIONAL_NOT_DEFAULT", "reason": "50% point gain over current simple rule but paired cluster CI crosses zero; no stable gain across budgets"},
        "deep_sets": "not promoted; prior decision retained",
        "sams": "existing training/postprocess remains separate and protected",
        "same_task_full_system": "not closed because legacy scores fail current truth-contract alignment; regenerate before paper stage",
        "paper_work": "DEFERRED",
        "next_action": "build one current-contract combined score table from regenerated Public/Shared/Target channels, then run paired 5000-cluster system comparison",
        "source_artifact": str(source_dir),
        "target_artifact": str(target_dir),
    }
    (out / "COMPONENT_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    report = "# SafeConf 当前执行回执\n\n"
    report += "## 已完成\n\n"
    report += "- 实现统一评分入口，但保留规则型公共评分与监督型Ridge候选的不同信息合同。\n"
    report += "- 完成E201 Source错误监督的折外Shared分数和Source-supervised Ridge候选。\n"
    report += "- 完成McFaline五个反馈预算的Target-supervised Ridge候选。\n"
    report += "- 完成5000次按基因簇配对bootstrap。\n"
    report += "- 检出旧矩阵/PertEMA结果与当前冻结特征的真实错误合同不一致，已排除出同任务指标。\n\n"
    report += "## 采用决定\n\n"
    report += "Source Ridge和Target Ridge均暂不替换强公共规则：当前点估计有局部增量，但bootstrap区间跨零，且Target增量不跨反馈预算稳定。保留为条件候选。\n\n"
    report += "## 未完成\n\n"
    report += "完整SafeConf尚未形成同一当前真值合同下的单一全量排序。下一步必须重新生成与当前Public、Source、Target完全一致的任务表后，再做完整系统比较。SAMS作业继续按原合同运行。\n"
    (out / "MORNING_REPORT.md").write_text(report)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    print(run(args.out, args.source_dir, args.target_dir).to_string(index=False))


if __name__ == "__main__":
    main()
