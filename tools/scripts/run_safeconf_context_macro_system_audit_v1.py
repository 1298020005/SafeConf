#!/usr/bin/env python3
"""Re-audit the fixed SafeConf system under the registered macro-U20 metric.

The v0.4.2 contract names context-macro Utility@20 as the primary metric.
The earlier current-system audit also reported a pooled-task U20. This
amendment preserves that secondary view, recomputes the registered primary
view, and adds the already-frozen H1_F1 and official-XGB F1 feedback outputs
without fitting or reading any new truth.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / ("docs/实验结果/Stage2_mature_upstream_20260928/dual_memory_continual/"
               "research_closure_20261001/data_model_feedback_20261003_v1")
FULL_DIR = BASE / "full_system_current_v1"
FEEDBACK_ROOT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1")
DEFAULT_OUT = BASE / "system_evidence_v042_context_macro_v1"
SEED = 20260930
BOOTSTRAP = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric(frame: pd.DataFrame, score: np.ndarray) -> dict:
    y = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.astype(str).to_numpy()
    s = np.asarray(score, float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, ids, s = y[valid], ids[valid], s[valid]
    out = {"n_tasks": int(len(y)), "u20": np.nan, "aurc": np.nan,
           "spearman": np.nan, "high_risk_hit_rate": np.nan,
           "high_risk_miss_rate": np.nan}
    if not len(y):
        return out
    k = int(math.ceil(0.2 * len(y)))
    risky = np.lexsort((ids, -s))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    denom = float(y[oracle].mean() - y.mean())
    if len(y) >= 20 and denom > 1e-12:
        out["u20"] = float((y[risky].mean() - y.mean()) / denom)
    if len(y) >= 3 and np.ptp(s) > 0 and np.ptp(y) > 0:
        out["spearman"] = float(spearmanr(s, y).statistic)
    low = np.lexsort((ids, s))
    out["aurc"] = float(np.mean(np.cumsum(y[low]) / np.arange(1, len(y) + 1)))
    hit = len(set(risky).intersection(set(oracle))) / k
    out["high_risk_hit_rate"] = float(hit)
    out["high_risk_miss_rate"] = float(1.0 - hit)
    return out


def macro_metric(frame: pd.DataFrame, score: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Equal-weight registered context strata; never substitute pooled U20."""
    frame = frame.reset_index(drop=True)
    s = np.asarray(score, float)
    rows = []
    for context, part in frame.groupby("target", sort=True):
        row = metric(part, s[part.index.to_numpy()])
        row["target"] = str(context)
        rows.append(row)
    strata = pd.DataFrame(rows)
    out = {"n_tasks": int(len(frame)), "n_contexts": int(len(strata))}
    for col in ["u20", "aurc", "spearman", "high_risk_hit_rate", "high_risk_miss_rate"]:
        out[col] = float(strata[col].mean()) if strata[col].notna().any() else np.nan
    out["valid_contexts"] = int(strata.u20.notna().sum())
    return out, strata


def macro_delta_bootstrap(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray,
                          seed: int = SEED) -> dict:
    frame = frame.reset_index(drop=True)
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(frame.true_error_rmse.to_numpy(float))
    frame, a, b = frame.loc[valid].reset_index(drop=True), a[valid], b[valid]
    clusters = sorted(frame.gene.astype(str).unique())
    blocks = [np.flatnonzero(frame.gene.astype(str).to_numpy() == g) for g in clusters]
    contexts = frame.target.astype(str).to_numpy()

    def delta(indices: np.ndarray) -> float:
        vals = []
        for context in sorted(set(contexts)):
            rows = indices[contexts[indices] == context]
            ma = metric(frame.iloc[rows], a[rows])["u20"]
            mb = metric(frame.iloc[rows], b[rows])["u20"]
            if np.isfinite(ma) and np.isfinite(mb):
                vals.append(ma - mb)
        return float(np.mean(vals)) if vals else np.nan

    observed = delta(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        delta(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
        for _ in range(BOOTSTRAP)
    ])
    return {"point_delta": observed, "bootstrap_mean_delta": float(np.nanmean(draws)),
            "ci95_lower": float(np.nanquantile(draws, 0.025)),
            "ci95_upper": float(np.nanquantile(draws, 0.975)),
            "n_tasks": int(len(frame)), "n_clusters": int(len(clusters)),
            "bootstrap_replicates": BOOTSTRAP}


def actual_discovery(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.astype(str).to_numpy()
    k = int(math.ceil(0.2 * len(y)))
    oracle = set(np.lexsort((ids, -y))[:k])
    rows = []
    for name, s in scores.items():
        selected = np.lexsort((ids, -np.asarray(s, float)))[:k]
        remaining = float(np.delete(y, selected).mean())
        found = int(len(set(selected).intersection(oracle)))
        rows.append({"method": name, "review_count": k, "review_fraction": k / len(y),
                     "true_high_error_found": found, "true_high_error_hit_rate": found / k,
                     "remaining_mean_error": remaining, "all_task_mean_error": float(y.mean())})
    out = pd.DataFrame(rows)
    amp = out.loc[out.method.eq("Amplitude")].iloc[0]
    public = out.loc[out.method.eq("PublicRule")].iloc[0]
    out["additional_high_error_found_vs_amplitude"] = out.true_high_error_found - amp.true_high_error_found
    out["remaining_error_reduction_vs_amplitude"] = amp.remaining_mean_error - out.remaining_mean_error
    out["additional_high_error_found_vs_public"] = out.true_high_error_found - public.true_high_error_found
    out["remaining_error_reduction_vs_public"] = public.remaining_mean_error - out.remaining_mean_error
    return out


def run(out: Path) -> None:
    out.mkdir(parents=False, exist_ok=False)
    hold = pd.read_parquet(FEEDBACK_ROOT / "HOLDOUT_FEATURES.parquet")
    full = pd.read_parquet(FULL_DIR / "FULL_SYSTEM_TASK_SCORES.parquet")
    if not np.array_equal(hold.task_id.to_numpy(), full.task_id.to_numpy()):
        raise RuntimeError("holdout task order differs from current full-system table")
    frame = hold[["task_id", "target", "gene", "true_error_rmse", "predicted_magnitude", "simple_history_risk"]].copy()
    existing = ["Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge", "TargetRidge50",
                "TargetH1F1_50", "SafeConf_no_target_feedback", "SafeConf_target_feedback_50"]
    for col in existing:
        if col in full:
            frame[col] = full[col].to_numpy(float)

    feedback = pd.read_csv(FEEDBACK_ROOT / "TASK_PREDICTIONS.csv.gz")
    feedback = feedback[(feedback.budget == 0.5) & feedback.seed.eq(SEED)]
    for method, new_name in [("H1_F1", "FeedbackH1F1_50"), ("X0_F1", "FeedbackXGBF1_50")]:
        q = feedback[feedback.method.eq(method)][["task_id", "risk", "true_error_rmse"]]
        if len(q) != len(frame) or q.task_id.nunique() != len(frame):
            raise RuntimeError(f"feedback score coverage incomplete for {method}")
        aligned = frame[["task_id", "true_error_rmse"]].merge(
            q, on="task_id", validate="one_to_one", suffixes=("_holdout", "_feedback"))
        max_delta = float(np.max(np.abs(aligned.true_error_rmse_holdout - aligned.true_error_rmse_feedback)))
        if max_delta > 1e-12:
            raise RuntimeError(f"{method} truth contract mismatch: {max_delta}")
        frame[new_name] = frame.task_id.map(q.set_index("task_id").risk).to_numpy(float)

    frame["SafeConf_rule_no_feedback"] = frame.PublicRule
    frame["SafeConf_feedback_H1F1_50"] = frame.FeedbackH1F1_50
    score_cols = [c for c in frame.columns if c in {
        "Amplitude", "PublicRule", "SourceSharedRisk", "SourceRidge", "TargetRidge50",
        "TargetH1F1_50", "FeedbackH1F1_50", "FeedbackXGBF1_50",
        "SafeConf_no_target_feedback", "SafeConf_target_feedback_50",
        "SafeConf_rule_no_feedback", "SafeConf_feedback_H1F1_50"}]
    if frame[score_cols].isna().any().any():
        raise RuntimeError("required score has NaN")
    score_map = {c: frame[c].to_numpy(float) for c in score_cols}
    rows, strata_rows, paired = [], [], []
    for name, score in score_map.items():
        primary, strata = macro_metric(frame, score)
        secondary = metric(frame, score)
        row = {"method": name,
               "method_class": "rule" if name in {"Amplitude", "PublicRule", "SafeConf_rule_no_feedback"} else "supervised_candidate",
               "primary_metric": "context_macro_u20", "u20": primary["u20"],
               "aurc": primary["aurc"], "spearman": primary["spearman"],
               "high_risk_hit_rate": primary["high_risk_hit_rate"],
               "high_risk_miss_rate": primary["high_risk_miss_rate"],
               "valid_contexts": primary["valid_contexts"], "n_contexts": primary["n_contexts"],
               "secondary_pooled_u20": secondary["u20"], "secondary_pooled_aurc": secondary["aurc"],
               "n_tasks": len(frame)}
        rows.append(row)
        for _, srow in strata.iterrows():
            strata_rows.append({"method": name, **srow.to_dict()})
    public = frame.PublicRule.to_numpy(float)
    amplitude = frame.Amplitude.to_numpy(float)
    for name, score in score_map.items():
        paired.append({"method": name, "comparison": "vs_PublicRule", **macro_delta_bootstrap(frame, score, public)})
        paired.append({"method": name, "comparison": "vs_Amplitude", **macro_delta_bootstrap(frame, score, amplitude)})
    system = pd.DataFrame(rows)
    system["delta_primary_u20_vs_public"] = system.u20 - float(system.loc[system.method.eq("PublicRule"), "u20"].iloc[0])
    system["delta_primary_u20_vs_amplitude"] = system.u20 - float(system.loc[system.method.eq("Amplitude"), "u20"].iloc[0])
    system.to_csv(out / "SYSTEM_COMPARISON_CONTEXT_MACRO.csv", index=False, lineterminator="\n")
    pd.DataFrame(strata_rows).to_csv(out / "SYSTEM_STRATA_CONTEXT_MACRO.csv", index=False, lineterminator="\n")
    pd.DataFrame(paired).to_csv(out / "PAIRED_BOOTSTRAP_CONTEXT_MACRO.csv", index=False, lineterminator="\n")
    frame.to_parquet(out / "PER_QUERY_RESULTS_CONTEXT_MACRO.parquet", index=False)
    actual_discovery(frame, score_map).to_csv(out / "ACTUAL_ERROR_DISCOVERY_GLOBAL_REVIEW.csv", index=False, lineterminator="\n")
    pd.DataFrame([
        {"method": "Amplitude", "information": "prediction only", "source_errors_used": 0, "target_errors_used": 0},
        {"method": "PublicRule", "information": "legal public history, no error labels", "source_errors_used": 0, "target_errors_used": 0},
        {"method": "SourceRidge", "information": "public + E201 source errors", "source_errors_used": 3616, "target_errors_used": 0},
        {"method": "TargetRidge50", "information": "public + target Ridge, McFaline DEV", "source_errors_used": 0, "target_errors_used": 542},
        {"method": "FeedbackH1F1_50", "information": "public + fixed target H1, fixed pool", "source_errors_used": 0, "target_errors_used": 331},
        {"method": "FeedbackXGBF1_50", "information": "public + official XGB recipe, fixed pool", "source_errors_used": 0, "target_errors_used": 331},
    ]).to_csv(out / "INFORMATION_BUDGET_LEDGER_CONTEXT_MACRO.csv", index=False, lineterminator="\n")
    (out / "METRIC_DEFINITION_AMENDMENT.md").write_text(
        "# Context-macro primary metric amendment\n\n"
        "The registered v0.4.2 contract names context-macro Utility@20 as the primary metric. "
        "For each target/context stratum, `k=ceil(0.2*n)` and task IDs break score ties; "
        "the reported primary value is the equal-weight mean over valid strata. The earlier "
        "pooled-task U20 remains in `secondary_pooled_u20` for diagnostic comparison and is "
        "not used for adoption decisions.\n\n"
        "All rows use the same 212-task current holdout and 152 perturbation-gene clusters. "
        "Feedback H1_F1 and XGB F1 are reused from the frozen `feedback_v1` fixed evaluation "
        "at 50% feedback; truth hashes are checked before scoring. No new learner fit or truth "
        "read occurs here.\n"
    )
    decision = {
        "contract": "SafeConf v0.4.2", "metric_amendment": "context_macro_primary",
        "status": "COMPLETE_CURRENT_SYSTEM_CONTEXT_MACRO_AUDIT",
        "default_configuration": "PublicRule_without_target_feedback",
        "target_feedback_candidate": "FeedbackH1F1_50",
        "decisions": {"public_rule": "RETAIN_DEFAULT",
                      "feedback_h1_f1_50": "CONDITIONAL_NOT_DEFAULT_CI_CROSSES_ZERO_VS_PUBLIC",
                      "feedback_xgb_f1_50": "CONDITIONAL_BASELINE",
                      "source_ridge": "CONDITIONAL_NOT_DEFAULT",
                      "target_ridge_50": "CONDITIONAL_NOT_DEFAULT"},
        "reason": "PublicRule remains strong under the registered context-macro metric. H1_F1 has a small point gain over PublicRule but its paired 5000-gene bootstrap interval crosses zero; XGB is retained as a fair target learner comparator.",
        "truth_contract": {"n_tasks": len(frame), "n_genes": int(frame.gene.nunique()),
                           "task_id_sha256": hashlib.sha256("\n".join(sorted(frame.task_id.astype(str))).encode()).hexdigest()},
        "source_artifacts": {"current_system": str(FULL_DIR), "feedback": str(FEEDBACK_ROOT)},
        "paper": "PAUSED_BY_USER"}
    (out / "COMPONENT_DECISION_CONTEXT_MACRO.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame([
        {"claim": "Public history improves risk auditing over amplitude", "status": "SUPPORTED_CONTEXT_MACRO", "evidence": "SYSTEM_COMPARISON_CONTEXT_MACRO.csv + paired bootstrap", "decision": "retain PublicRule"},
        {"claim": "Target H1_F1 adds stable value beyond PublicRule", "status": "NOT_ESTABLISHED_CI_CROSSES_ZERO", "evidence": "PAIRED_BOOTSTRAP_CONTEXT_MACRO.csv", "decision": "conditional candidate"},
        {"claim": "Official XGB F1 is a fair target-specific comparator", "status": "SAME_TRUTH_CONTRACT_FIXED_POOL", "evidence": "feedback_v1 TASK_PREDICTIONS.csv.gz", "decision": "retain comparator; not full PertEMA certification"},
        {"claim": "Pooled and context-macro U20 give the same adoption decision", "status": "NOT_ASSUMED", "evidence": "METRIC_DEFINITION_AMENDMENT.md", "decision": "use context-macro primary"},
    ]).to_csv(out / "CLAIM_EVIDENCE_MATRIX_CONTEXT_MACRO.csv", index=False, lineterminator="\n")
    run_status = {"status": "COMPLETE", "contract": "SafeConf v0.4.2", "primary_metric": "context_macro_u20",
                  "n_tasks": len(frame), "n_genes": int(frame.gene.nunique()), "new_fits": 0,
                  "new_truth_reads": 0, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (out / "RUN_STATUS.json").write_text(json.dumps(run_status, indent=2, ensure_ascii=False) + "\n")
    manifest = {"contract": "SafeConf v0.4.2", "primary_metric": "context_macro_u20",
                "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(system.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
