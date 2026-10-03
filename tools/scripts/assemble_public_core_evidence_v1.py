#!/usr/bin/env python3
"""Assemble the Public Biological Evidence core claim with its boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001"
DATA = BASE / "data_model_feedback_20261003_v1"
DEFAULT_OUT = DATA / "system_evidence_v4/public_core"


def run(out: Path):
    out.mkdir(parents=False, exist_ok=False)
    rows = []
    full_boot = pd.read_csv(DATA / "full_system_current_v1/bootstrap_vs_amplitude/BOOTSTRAP_SUMMARY.csv")
    for _, r in full_boot[full_boot.method.isin(["PublicRule", "Amplitude"])].iterrows():
        rows.append({
            "evidence": "current_same_truth_212",
            "direction": "DecoderOnly_McFaline",
            "scope": "current_212_holdout",
            "method": r.method,
            "baseline": "Amplitude" if r.method == "PublicRule" else "oracle_reference",
            "point_or_mean_u20": r.u20_mean,
            "ci95_lower": r.u20_ci95_lower,
            "ci95_upper": r.u20_ci95_upper,
            "delta_vs_baseline": r.delta_mean_vs_reference,
            "delta_ci95_lower": r.delta_ci95_lower,
            "delta_ci95_upper": r.delta_ci95_upper,
            "n_clusters": 152,
            "status": "same_truth_primary",
        })
    fr_macro = pd.read_csv(BASE / "frangieh_cross_family_v1/MACRO_RESULTS.csv")
    fr_boot = pd.read_csv(BASE / "frangieh_cross_family_v1/PAIRED_BOOTSTRAP.csv")
    for _, r in fr_macro[(fr_macro.method == "PublicHGB") &
                         (fr_macro.scope.isin(["all_test", "heldout_context"])) &
                         (fr_macro.aggregation == "risk_fold")].iterrows():
        b = fr_boot[(fr_boot.direction == r.direction) & (fr_boot.scope == r.scope) &
                    (fr_boot.method == "PublicHGB") & (fr_boot.baseline == "Magnitude")]
        if len(b) != 1:
            continue
        b = b.iloc[0]
        rows.append({
            "evidence": "frangieh_native512_cross_family_seen",
            "direction": r.direction,
            "scope": r.scope,
            "method": "PublicHGB",
            "baseline": "Magnitude",
            "point_or_mean_u20": r.utility20,
            "ci95_lower": b.ci95_lower,
            "ci95_upper": b.ci95_upper,
            "delta_vs_baseline": b.delta_utility20,
            "delta_ci95_lower": b.ci95_lower,
            "delta_ci95_upper": b.ci95_upper,
            "n_clusters": int(b.n_paired_gene_clusters),
            "status": "stress_test_upstream_competence_failed",
        })
    public = pd.DataFrame(rows)
    public.to_csv(out / "PUBLIC_CORE_EVIDENCE.csv", index=False)

    competence = pd.read_csv(BASE / "frangieh_cross_family_v1/UPSTREAM_COMPETENCE.csv")
    competence.to_csv(out / "CROSS_FAMILY_COMPETENCE_BOUNDARY.csv", index=False)
    content = pd.read_csv(BASE / "common_gene_axis/results/SUPPORT_CONTENT_ATTRIBUTION_COMPARISONS.csv")
    content = content[(content.metric == "utility20") &
                      content.method_a.astype(str).str.contains("WeightedHistoryDistance")]
    content.to_csv(out / "PUBLIC_CONTENT_MECHANISM_EVIDENCE.csv", index=False)

    decision = {
        "core_contribution": "Public Biological Evidence for risk auditing",
        "primary_same_truth_result": "PublicRule exceeds Amplitude on current 212-task contract; 5000 gene-cluster delta CI is strictly positive",
        "cross_family_result": "PublicHGB exceeds Magnitude in both scGPT/GEARS directions in native512 SEEN stress test",
        "boundary": "all six native upstream competence checks failed; cross-family result is stress evidence, not independent qualified confirmation",
        "source_and_target": "Source/Target Ridge remain conditional and do not replace Public default",
        "paper_ready_claim": "real public perturbation evidence provides model-agnostic risk signal, with strongest support on legally covered tasks and a competence-limited cross-family stress test",
        "not_allowed": ["claim universal cross-family validation", "claim Source/Target fusion is default", "claim background mapping succeeded"],
    }
    (out / "PUBLIC_CORE_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    (out / "PUBLIC_CORE_DECISION.md").write_text(
        "# Public Biological Evidence 核心证据\n\n"
        "当前212任务同一真值合同中，PublicRule显著超过Amplitude；5000次按基因簇bootstrap的差值区间严格高于零。\n\n"
        "Frangieh native512的GEARS/scGPT双向压力测试中，PublicHGB超过Magnitude，但六个上游能力门均未通过，因此该部分只承担模型无关公共证据的压力测试和适用边界。\n\n"
        "Source/Target Ridge不进入默认配置。\n"
    )
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.out).to_string(index=False))


if __name__ == "__main__":
    main()
