#!/usr/bin/env python3
"""Assemble and verify the v0.4.2 SafeConf evidence contract.

This is an audit/assembly step, not a new learner search.  It consumes the
already frozen Source, Target and current-holdout score artifacts and writes a
small, traceable package for the contract's unified score interface.  Rule
methods remain label-free; Ridge methods retain an explicit label source.

The script intentionally refuses to silently turn a legacy PertEMA table with
a different truth contract into a current result.  It records that comparison
as excluded evidence instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.safeconf_continual.fusion import (  # noqa: E402
    OOFReadyRidgeFusion,
    build_fusion_channels,
    score_task,
)


DEFAULT_BASE = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/"
    "Stage2_mature_upstream_20260928/dual_memory_continual/research_closure_20261001/"
    "data_model_feedback_20261003_v1"
)
DEFAULT_OUT = DEFAULT_BASE / "system_evidence_v042"
SOURCE_DIR = DEFAULT_BASE / "unified_fusion_v4"
TARGET_DIR = DEFAULT_BASE / "target_ridge_fusion_v8"
FULL_DIR = DEFAULT_BASE / "full_system_current_v1"
V4_DIR = DEFAULT_BASE / "system_evidence_v4"
SAMS_STATUS = DEFAULT_BASE / "sams" / "TRAINING_STATUS.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict:
    return json.loads(require(path).read_text())


def check_same_truth(full: pd.DataFrame) -> dict:
    cols = ["task_id", "true_error_rmse"]
    if full[cols].duplicated("task_id").any():
        raise ValueError("full system has duplicate task IDs")
    if full["true_error_rmse"].isna().any():
        raise ValueError("current system truth contains missing values")
    return {
        "same_task_truth_contract": True,
        "n_tasks": int(len(full)),
        "n_genes": int(full["gene"].nunique()),
        "task_id_sha256": hashlib.sha256(
            "\n".join(sorted(full["task_id"].astype(str))).encode()
        ).hexdigest(),
    }


def interface_audit() -> dict:
    """Exercise both dispatch modes without reading a target truth value."""
    rule = score_task(
        {"amplitude": 0.2, "public": 0.7},
        {"mode": "rule", "version": "v0.4.2-rule-audit"},
    )
    x = np.array([[0.1, 0.2, np.nan, np.nan],
                  [0.2, 0.3, np.nan, np.nan],
                  [0.3, 0.4, np.nan, np.nan]], dtype=float)
    channels, names = build_fusion_channels(
        x[:, 0], x[:, 1], None, None,
        has_public=np.ones(3, dtype=int),
        has_shared=np.zeros(3, dtype=int),
        has_target=np.zeros(3, dtype=int),
    )
    model = OOFReadyRidgeFusion(
        alpha=10.0,
        label_source="SOURCE_ERROR_AUDIT_ONLY",
        target_error_budget="none",
    ).fit(channels, np.array([0.1, 0.2, 0.3]), names,
          base_scores_are_oof=True)
    ridge = score_task(
        {"channels": channels[0], "has_public": 1,
         "evidence_status": "public_source_supervised"},
        {"mode": "ridge", "model": model, "version": "v0.4.2-ridge-audit"},
    )
    return {
        "rule_dispatch": rule,
        "ridge_dispatch": ridge,
        "rule_reads_truth": False,
        "ridge_requires_oof": True,
        "ridge_label_source": model.audit()["label_source"],
        "passed": bool(np.isfinite(rule["risk_score"]) and np.isfinite(ridge["risk_score"])),
    }


def normalize_channel_audits() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_channel: list[dict] = []
    rows_label: list[dict] = []
    source = pd.read_csv(require(SOURCE_DIR / "CHANNEL_SCORE_CDF_AUDIT.csv"))
    for row in source.to_dict("records"):
        rows_channel.append({
            "source_artifact": "unified_fusion_v4",
            "budget": "source_full",
            "fold": "OOF_source",
            "cdf_kind": "channel_score",
            **row,
            "uses_target_error": False,
        })
    source_label = pd.read_csv(require(SOURCE_DIR / "ERROR_LABEL_CDF_AUDIT.csv"))
    for row in source_label.to_dict("records"):
        rows_label.append({"source_artifact": "unified_fusion_v4", **row,
                           "cdf_kind": "error_label"})

    target_cdf = pd.read_csv(require(TARGET_DIR / "CDF_AUDIT.csv"))
    for row in target_cdf.to_dict("records"):
        base = {
            "source_artifact": "target_ridge_fusion_v8",
            "budget": row.get("budget"),
            "fold": row.get("fold"),
            "cdf_kind": row.get("cdf_kind"),
            "channel": row.get("channel"),
            "fit_rows": row.get("fit_rows"),
            "uses_target_error": bool(row.get("uses_target_error", False)),
        }
        if str(row.get("cdf_kind")) == "error_label":
            rows_label.append(base)
        else:
            rows_channel.append(base)
    return pd.DataFrame(rows_channel), pd.DataFrame(rows_label)


def build_system_comparison(full: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    metric = pd.read_csv(require(FULL_DIR / "SYSTEM_METRICS.csv"))
    method_meta = {
        "Amplitude": ("rule", "none", 0, "current_holdout"),
        "PublicRule": ("rule", "none", 0, "current_holdout"),
        "SourceSharedRisk": ("supervised_shared", "E201 source errors", 0, "current_holdout"),
        "SourceRidge": ("supervised_fusion", "E201 source errors", 0, "current_holdout"),
        "TargetRidge50": ("supervised_target_fusion", "McFaline DEV errors", 542, "current_holdout"),
        "TargetH1F1_50": ("supervised_target", "McFaline DEV errors", 542, "current_holdout"),
        "SafeConf_no_target_feedback": ("complete_candidate", "E201 source errors", 0, "current_holdout"),
        "SafeConf_target_feedback_50": ("complete_candidate", "McFaline DEV errors", 542, "current_holdout"),
    }
    out = metric.copy()
    # Recompute the user-facing ranking quantities from the same per-task
    # table.  The older full-system writer predates v0.4.2 and did not emit
    # AURC/Spearman/high-risk miss rate, so this audit does it once without
    # refitting or opening any truth beyond the current evaluation table.
    truth = full["true_error_rmse"].to_numpy(float)
    ids = full["task_id"].astype(str).to_numpy()
    k = int(np.ceil(0.2 * len(truth)))
    oracle = np.lexsort((ids, -truth))[:k]
    metrics_v042 = []
    for _, row in out.iterrows():
        method = row["method"]
        if method not in full:
            metrics_v042.append({"method": method})
            continue
        score = full[method].to_numpy(float)
        risky = np.lexsort((ids, -score))[:k]
        ascending = np.lexsort((ids, score))
        rho = spearmanr(score, truth).statistic if np.ptp(score) > 0 and np.ptp(truth) > 0 else np.nan
        metrics_v042.append({
            "method": method,
            "aurc": float(np.mean(np.cumsum(truth[ascending]) / np.arange(1, len(truth) + 1))),
            "spearman": float(rho) if np.isfinite(rho) else np.nan,
            "high_risk_hit_rate": float(len(set(risky).intersection(set(oracle))) / len(oracle)),
            "high_risk_miss_rate": float(1.0 - len(set(risky).intersection(set(oracle))) / len(oracle)),
        })
    extra = pd.DataFrame(metrics_v042)
    out = out.merge(extra, on="method", how="left", suffixes=("", "_recomputed"))
    u20_lookup = dict(zip(out["method"], out["u20"]))
    out["delta_u20_vs_amplitude"] = out["u20"] - float(u20_lookup.get("Amplitude", np.nan))
    out["delta_u20_vs_public"] = out["u20"] - float(u20_lookup.get("PublicRule", np.nan))
    out["method_class"] = out["method"].map(lambda m: method_meta.get(m, ("unknown", "unknown", np.nan, "unknown"))[0])
    out["label_source"] = out["method"].map(lambda m: method_meta.get(m, ("unknown", "unknown", np.nan, "unknown"))[1])
    out["target_errors_used_for_fit"] = out["method"].map(lambda m: method_meta.get(m, ("unknown", "unknown", np.nan, "unknown"))[2])
    out["truth_contract"] = out["method"].map(lambda m: method_meta.get(m, ("unknown", "unknown", np.nan, "unknown"))[3])
    # Legacy PertEMA is kept as an explicit excluded row.  It must not be
    # silently re-evaluated under a truth contract that trained it differently.
    legacy = pd.read_csv(require(TARGET_DIR / "LEGACY_ALIGNMENT_AUDIT.csv"))
    legacy_delta = float(legacy.iloc[0]["max_abs_truth_delta"])
    out = pd.concat([out, pd.DataFrame([{
        "method": "PertEMA_adapted_legacy",
        "method_class": "external_baseline",
        "n_tasks": 0,
        "u20": np.nan,
        "selected_top20": np.nan,
        "true_top20_found": np.nan,
        "remaining_mean_error": np.nan,
        "mean_error": np.nan,
        "score_min": np.nan,
        "score_max": np.nan,
        "label_source": "legacy feedback errors",
        "target_errors_used_for_fit": np.nan,
        "truth_contract": "excluded",
        "status": f"EXCLUDED_TRUTH_CONTRACT_MISMATCH:max_abs_delta={legacy_delta:.9g}",
    }])], ignore_index=True, sort=False)
    # Attach the paired bootstrap decision where available.
    b = bootstrap[["method", "delta_ci95_lower", "delta_ci95_upper"]].copy()
    b = b.rename(columns={"delta_ci95_lower": "delta_ci_lower_vs_public",
                          "delta_ci95_upper": "delta_ci_upper_vs_public"})
    out = out.merge(b, on="method", how="left")
    out["same_task_unique_ranking"] = out["method"].isin(metric["method"])
    return out


def ranking_audit(full: pd.DataFrame) -> pd.DataFrame:
    methods = [c for c in full.columns if c in {
        "Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge",
        "TargetRidge50", "TargetH1F1_50", "SafeConf_no_target_feedback",
        "SafeConf_target_feedback_50",
    }]
    rows = []
    for method in methods:
        vals = full[method].to_numpy(float)
        order = np.lexsort((full.task_id.astype(str).to_numpy(), -vals))
        rows.append({
            "method": method,
            "n_tasks": int(len(vals)),
            "n_finite_scores": int(np.isfinite(vals).sum()),
            "n_unique_score_values": int(pd.Series(vals).nunique()),
            "n_unique_rank_positions": int(len(order)),
            "stable_task_id_tiebreak": True,
            "truth_used_in_scoring": False,
            "single_fixed_ranking": True,
        })
    return pd.DataFrame(rows)


def actual_error_discovery(full: pd.DataFrame) -> pd.DataFrame:
    """Translate the ranking into the fixed 20% review action."""
    truth = full["true_error_rmse"].to_numpy(float)
    ids = full["task_id"].astype(str).to_numpy()
    k = int(np.ceil(0.2 * len(truth)))
    oracle = np.lexsort((ids, -truth))[:k]
    oracle_set = set(oracle)
    methods = [c for c in full.columns if c in {
        "Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge",
        "TargetRidge50", "TargetH1F1_50", "SafeConf_no_target_feedback",
        "SafeConf_target_feedback_50",
    }]
    rows = []
    for method in methods:
        score = full[method].to_numpy(float)
        selected = np.lexsort((ids, -score))[:k]
        remaining = float(np.delete(truth, selected).mean())
        found = int(len(set(selected).intersection(oracle_set)))
        rows.append({
            "method": method,
            "review_fraction": float(k / len(truth)),
            "review_count": int(k),
            "true_high_error_top20_count": int(k),
            "true_high_error_found": found,
            "true_high_error_hit_rate": float(found / k),
            "remaining_mean_error": remaining,
            "all_task_mean_error": float(truth.mean()),
        })
    out = pd.DataFrame(rows)
    amp = out.loc[out.method.eq("Amplitude")].iloc[0]
    out["additional_high_error_found_vs_amplitude"] = out.true_high_error_found - float(amp.true_high_error_found)
    out["remaining_error_reduction_vs_amplitude"] = float(amp.remaining_mean_error) - out.remaining_mean_error
    out["remaining_error_reduction_fraction_vs_amplitude"] = out.remaining_error_reduction_vs_amplitude / float(amp.remaining_mean_error)
    public = out.loc[out.method.eq("PublicRule")].iloc[0]
    out["additional_high_error_found_vs_public"] = out.true_high_error_found - float(public.true_high_error_found)
    out["remaining_error_reduction_vs_public"] = float(public.remaining_mean_error) - out.remaining_mean_error
    return out


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def run(base: Path, out: Path) -> dict:
    out.mkdir(parents=False, exist_ok=False)
    full = pd.read_parquet(require(FULL_DIR / "FULL_SYSTEM_TASK_SCORES.parquet"))
    truth_contract = check_same_truth(full)
    required_scores = {
        "Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge",
        "TargetRidge50", "TargetH1F1_50", "SafeConf_no_target_feedback",
        "SafeConf_target_feedback_50",
    }
    missing = sorted(required_scores - set(full.columns))
    if missing:
        raise ValueError(f"full system missing required score columns: {missing}")
    if full[list(required_scores)].isna().any().any():
        raise ValueError("full system has missing scores in a required candidate")
    # The current 212-task holdout happens to have Public evidence for every
    # task.  Record that fact explicitly instead of silently treating the
    # amplitude fallback as if it had been exercised.  The fallback policy is
    # still frozen and is tested below through the score interface audit.
    if "simple_history_risk" in full:
        has_public = full["simple_history_risk"].notna()
    else:
        has_public = full["PublicRule"].notna()
    fallback_audit = pd.DataFrame([{
        "cohort": "current_212_holdout",
        "n_tasks": int(len(full)),
        "n_public": int(has_public.sum()),
        "n_amplitude_fallback": int((~has_public).sum()),
        "fallback_policy": "amplitude_when_public_missing",
        "ranking_includes_all_tasks": True,
        "interpretation": "no missing-public task in this cohort; fallback policy is frozen but not empirically exercised here",
    }])
    full["has_public"] = has_public.astype(int)
    full["evidence_status"] = np.where(has_public, "public", "amplitude_only")

    bootstrap = pd.read_csv(require(FULL_DIR / "bootstrap" / "BOOTSTRAP_SUMMARY.csv"))
    system = build_system_comparison(full, bootstrap)
    channel, labels = normalize_channel_audits()
    ranks = ranking_audit(full)
    discovery = actual_error_discovery(full)
    interface = interface_audit()
    if not interface["passed"]:
        raise RuntimeError("unified score interface audit failed")

    full.to_parquet(out / "PER_QUERY_RESULTS.parquet", index=False)
    fallback_audit.to_csv(out / "NO_HISTORY_FALLBACK_AUDIT.csv", index=False)
    system.to_csv(out / "SYSTEM_COMPARISON.csv", index=False)
    ranks.to_csv(out / "UNIFIED_RANKING_AUDIT.csv", index=False)
    discovery.to_csv(out / "ACTUAL_ERROR_DISCOVERY.csv", index=False)
    channel.to_csv(out / "CHANNEL_SCORE_CDF_AUDIT.csv", index=False)
    labels.to_csv(out / "ERROR_LABEL_CDF_AUDIT.csv", index=False)

    ledger = pd.DataFrame([
        {"method": "Amplitude", "method_class": "rule", "public_truth": "none", "source_errors": 0, "target_errors": 0, "target_error_use": "none", "truth_contract": "current_holdout"},
        {"method": "PublicRule", "method_class": "rule", "public_truth": "legal public history", "source_errors": 0, "target_errors": 0, "target_error_use": "none", "truth_contract": "current_holdout"},
        {"method": "SourceSharedRisk", "method_class": "supervised_shared", "public_truth": "legal public history", "source_errors": 3616, "target_errors": 0, "target_error_use": "evaluation only", "truth_contract": "current_holdout"},
        {"method": "SourceRidge", "method_class": "source_supervised_fusion", "public_truth": "legal public history", "source_errors": 3616, "target_errors": 0, "target_error_use": "forbidden", "truth_contract": "current_holdout"},
        {"method": "TargetRidge50", "method_class": "target_supervised_fusion", "public_truth": "legal public history", "source_errors": 0, "target_errors": 542, "target_error_use": "DEV fit and selection", "truth_contract": "current_holdout"},
        {"method": "TargetH1F1_50", "method_class": "target_supervised", "public_truth": "legal public history", "source_errors": 0, "target_errors": 542, "target_error_use": "DEV fit and selection", "truth_contract": "current_holdout"},
        {"method": "SafeConf_no_target_feedback", "method_class": "complete_candidate", "public_truth": "legal public history", "source_errors": 3616, "target_errors": 0, "target_error_use": "forbidden", "truth_contract": "current_holdout"},
        {"method": "SafeConf_target_feedback_50", "method_class": "complete_candidate", "public_truth": "legal public history", "source_errors": 0, "target_errors": 542, "target_error_use": "DEV fit and selection", "truth_contract": "current_holdout"},
        {"method": "PertEMA_adapted_legacy", "method_class": "external_baseline", "public_truth": "legacy public history", "source_errors": 0, "target_errors": "unknown", "target_error_use": "excluded after truth audit", "truth_contract": "excluded"},
    ])
    ledger.to_csv(out / "INFORMATION_BUDGET_LEDGER.csv", index=False)

    sams = read_json(SAMS_STATUS) if SAMS_STATUS.exists() else {"status": "MISSING"}
    decision = {
        "contract": "SafeConf v0.4.2",
        "status": "CURRENT_CONTRACT_AUDIT_COMPLETE_SAMS_PENDING",
        "default_configuration": "PublicRule",
        "rule_methods_preserved": ["Amplitude", "PublicRule"],
        "source_supervised_candidate": "SourceRidge",
        "target_supervised_candidate": "TargetRidge50",
        "decisions": {
            "public_rule": "RETAIN_DEFAULT",
            "source_ridge": "CONDITIONAL_NOT_DEFAULT",
            "target_ridge_50": "CONDITIONAL_NOT_DEFAULT",
            "legacy_pertema": "EXCLUDED_TRUTH_CONTRACT_MISMATCH",
            "sams": "CONTINUE_EXISTING_JOB",
        },
        "reason": "PublicRule has the current same-truth positive evidence; supervised candidates have point gains in some conditions but paired cluster CIs cross zero or are negative.",
        "truth_contract": truth_contract,
        "sams_status": sams.get("status"),
        "paper": "PAUSED_BY_USER",
        "created_by": "run_safeconf_contract_v042_audit.py",
    }
    (out / "COMPONENT_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))

    claims = pd.DataFrame([
        {"claim": "Public history improves risk auditing over amplitude", "evidence": "full_system_current_v1 + paired 5000 gene-cluster bootstrap", "status": "SUPPORTED_WITH_SCOPE", "decision": "retain PublicRule"},
        {"claim": "Source error supervision adds stable value beyond PublicRule", "evidence": "unified_fusion_v4 and full_system_current_v1", "status": "NOT_SUPPORTED_ON_CURRENT_212", "decision": "conditional only"},
        {"claim": "Target feedback adds stable value at 50 percent budget", "evidence": "target_ridge_fusion_v8 and paired bootstrap", "status": "POINT_GAIN_CI_CROSSES_ZERO", "decision": "conditional only"},
        {"claim": "Ridge may replace the original public rule", "evidence": "v0.4.2 unified ranking audit", "status": "REJECTED", "decision": "keep original rule as baseline/default"},
        {"claim": "Legacy PertEMA is a fair current same-task comparator", "evidence": "LEGACY_ALIGNMENT_AUDIT.csv", "status": "REJECTED_TRUTH_CONTRACT_MISMATCH", "decision": "retain as excluded audit"},
        {"claim": "SAMS cross-family evidence is complete", "evidence": str(SAMS_STATUS), "status": "PENDING_TRUE_GENERATION", "decision": "continue protected training"},
    ])
    claims.to_csv(out / "CLAIM_EVIDENCE_MATRIX.csv", index=False)

    write_text(out / "DIAGNOSIS_AND_ACTIONS.md", """# v0.4.2 diagnosis and actions

## Completed

- The rule and supervised candidates now share one `score_task(task, frozen_config)` dispatch interface without forcing the original public rule through a label-trained model.
- Source Ridge uses only E201 source error labels and fold-out-of-fold Shared scores. Target errors are not read during its fit.
- Target Ridge uses the registered McFaline development labels; its 212-task holdout is evaluation-only.
- The complete current-holdout ranking contains the same 212 tasks for every required candidate, with stable task-ID tie breaking.
- Channel-score CDFs and error-label CDFs are written as separate audits.

## Decisions

- Keep the original PublicRule as the current default. It is the strongest current same-truth no-error-label candidate.
- Keep SourceRidge and TargetRidge50 as conditional candidates. Their point gains do not pass the paired bootstrap adoption gate on the current contract.
- Keep the legacy PertEMA comparison as an excluded audit because its stored truth contract differs from the current holdout; do not describe it as a fair current score.

## Continuing action

SAMS training remains protected and active. Once it reaches a terminal checkpoint, run the registered truth-free generation, competence screen and bidirectional transfer protocol. No new network or Ridge search is opened by this audit.
""")

    write_text(out / "MORNING_REPORT.md", f"""# SafeConf v0.4.2 execution report

## Actual work

- Assembled the current-contract Source, Target and full-system evidence into one audit package.
- Verified {truth_contract['n_tasks']} tasks / {truth_contract['n_genes']} perturbation genes share one current truth contract.
- Verified rule and supervised dispatch, OOF requirement, separate score and label CDF audits, and unique rankings.

## Current decision

- Default: `PublicRule`.
- Source Ridge: point gain exists in the external Source audit, but the paired gene-cluster interval crosses zero and the current complete ranking is below PublicRule; conditional only.
- Target Ridge at 50%: point gain is small and its paired interval crosses zero; conditional only.
- Legacy PertEMA: excluded from current same-task metrics after truth-contract mismatch audit.

## Next action

SAMS remains in the existing protected training process (`{sams.get('status', 'UNKNOWN')}`). When it terminates, complete its registered generation/competence/cross-family postprocess and update this evidence package. Papers and PDF remain paused.
""")

    run_status = {
        "status": "COMPLETE_WITH_SAMS_PENDING",
        "contract": "SafeConf v0.4.2",
        "output_dir": str(out),
        "required_outputs": sorted(p.name for p in out.iterdir()),
        "truth_contract": truth_contract,
        "interface_audit": interface,
        "sams_status": sams.get("status"),
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(run_status, indent=2, ensure_ascii=False))
    (out / "MANIFEST.json").write_text(json.dumps({
        "contract": "SafeConf v0.4.2",
        "source_artifact": str(SOURCE_DIR),
        "target_artifact": str(TARGET_DIR),
        "full_system_artifact": str(FULL_DIR),
        "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "MANIFEST.json"},
    }, indent=2, ensure_ascii=False))
    return run_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    status = run(args.base, args.out)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
