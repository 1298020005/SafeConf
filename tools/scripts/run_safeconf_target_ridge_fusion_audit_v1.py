#!/usr/bin/env python3
"""Evaluate Target-supervised Ridge fusion on the locked feedback contract.

The 542 McFaline development tasks provide the only fitting labels. Their
H1/F1 scores are already held-out within the three development folds. The 212
task table is read only after each fusion candidate is frozen. This script is
separate from the Source-supervised fusion audit so the two error sources
cannot be accidentally pooled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.safeconf_continual.fusion import OOFReadyRidgeFusion, TrainOnlyQuantileScale, build_fusion_channels, rule_score

DEFAULT_ROOT = Path("/home/yyf/runtime_artifacts/safeconf_research_20261003/feedback_v1")
DEFAULT_OUT = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/"
    "dual_memory_continual/research_closure_20261001/data_model_feedback_20261003_v1/target_ridge_fusion_v1"
)
STRICT_PATH = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/"
    "dual_memory_continual/research_closure_20261001/STRICT_FEEDBACK_TASK_PREDICTIONS.csv.gz"
)
STRONG_PUBLIC_PATH = Path(
    "/home/yyf/runtime_worktrees/e220_reviewer_closure_20260921/docs/实验结果/Stage2_mature_upstream_20260928/"
    "dual_memory_continual/research_closure_20261001/FEEDBACK_STRONG_BASELINE_TASK_PREDICTIONS.csv.gz"
)
BUDGETS = {23: 0.10, 57: 0.25, 114: 0.50, 171: 0.75, 228: 1.00}
SEED = 20260930


def cdf_fit_transform(train, query, name):
    s = TrainOnlyQuantileScale.fit(np.asarray(train, float), name)
    return s, s.transform(query)


def labels_cdf(train_labels, query_labels):
    s = TrainOnlyQuantileScale.fit(np.asarray(train_labels, float), "target_error_label")
    return s, s.transform(query_labels)


def metric_frame(frame, values, method):
    truth = frame.true_error_rmse.to_numpy(float)
    ids = frame.task_id.astype(str).to_numpy()
    values = np.asarray(values, float)
    valid = np.isfinite(truth) & np.isfinite(values)
    truth, ids, values = truth[valid], ids[valid], values[valid]
    result = {"budget": np.nan, "method": method, "n_tasks": len(truth), "u20": np.nan,
              "aurc": np.nan, "spearman": np.nan, "selected_top20": np.nan,
              "true_top20_found": np.nan, "remaining_mean_error": np.nan}
    if len(truth) == 0:
        return result
    k = max(1, int(np.ceil(.2 * len(truth))))
    risky = np.lexsort((ids, -values))[:k]
    oracle = np.lexsort((ids, -truth))[:k]
    denom = truth[oracle].mean() - truth.mean()
    if len(truth) >= 20 and denom > 1e-12:
        result["u20"] = float((truth[risky].mean() - truth.mean()) / denom)
    if len(truth) >= 3 and np.ptp(values) > 0 and np.ptp(truth) > 0:
        result["spearman"] = float(spearmanr(values, truth).statistic)
    ascending = np.lexsort((ids, values))
    result["aurc"] = float(np.mean(np.cumsum(truth[ascending]) / np.arange(1, len(truth) + 1)))
    result["selected_top20"] = k
    result["true_top20_found"] = int(len(set(risky).intersection(set(oracle))))
    result["remaining_mean_error"] = float(np.delete(truth, risky).mean()) if len(truth) > k else np.nan
    return result


def run(root: Path, out: Path):
    out.mkdir(parents=False, exist_ok=False)
    dev = pd.read_parquet(root / "DEV_FEATURES.parquet")
    hold = pd.read_parquet(root / "HOLDOUT_FEATURES.parquet")
    dev_pred = pd.read_csv(root / "DEV_PREDICTIONS.csv.gz")
    fixed_pred = pd.read_csv(root / "TASK_PREDICTIONS.csv.gz")
    fixed_pred = fixed_pred[fixed_pred.seed == SEED]
    strict_pred = pd.read_csv(STRICT_PATH)
    strict_pred = strict_pred[strict_pred.seed == SEED]
    strong_public = pd.read_csv(STRONG_PUBLIC_PATH)
    strong_public = strong_public[(strong_public.seed == SEED) &
                                  (strong_public.method == "Learned_WeightedHistoryDistance")]
    strong_alignment = hold[["task_id", "true_error_rmse"]].merge(
        strong_public[["task_id", "true_error_rmse"]].drop_duplicates("task_id"),
        on="task_id", suffixes=("_current", "_legacy")
    )
    strong_max_truth_delta = float(np.max(np.abs(
        strong_alignment.true_error_rmse_current.to_numpy(float) - strong_alignment.true_error_rmse_legacy.to_numpy(float)
    ))) if len(strong_alignment) else np.inf
    for required in ["task_id", "dev_fold", "predicted_magnitude", "prediction_prior_rmse", "true_error_rmse"]:
        if required not in dev:
            raise RuntimeError(f"development feature contract missing {required}")
    candidate_rows, audit_rows, cdf_rows, dev_oof_exports = [], [], [], []
    for n_clusters, budget in BUDGETS.items():
        p = dev_pred[(dev_pred.feature_set == "F1") & (dev_pred.learner == "H1") &
                     (dev_pred.budget_clusters == n_clusters) & (dev_pred.seed == SEED)][
            ["task_id", "heldout_fold", "risk"]
        ].rename(columns={"risk": "target_oof_score"})
        if len(p) != len(dev) or p.task_id.nunique() != len(dev):
            raise RuntimeError(f"development OOF score coverage incomplete for budget {budget}")
        d = dev.merge(p, on="task_id", how="left", validate="one_to_one")
        oof_fused = np.full(len(d), np.nan, dtype=float)
        for fold in sorted(d.dev_fold.unique()):
            train = d.dev_fold != fold
            query = d.dev_fold == fold
            amp_s = TrainOnlyQuantileScale.fit(d.loc[train, "predicted_magnitude"], "amplitude_q")
            public_s = TrainOnlyQuantileScale.fit(d.loc[train, "prediction_prior_rmse"], "public_q")
            target_s = TrainOnlyQuantileScale.fit(d.loc[train, "target_oof_score"], "target_q")
            _, y_s = labels_cdf(d.loc[train, "true_error_rmse"], d.loc[train, "true_error_rmse"])
            x_train, names = build_fusion_channels(
                amp_s.transform(d.loc[train, "predicted_magnitude"]),
                public_s.transform(d.loc[train, "prediction_prior_rmse"]),
                None,
                target_s.transform(d.loc[train, "target_oof_score"]),
                has_public=d.loc[train, "prediction_prior_rmse"].notna().astype(int),
                has_shared=np.zeros(train.sum(), dtype=int),
                has_target=np.ones(train.sum(), dtype=int),
            )
            x_query, _ = build_fusion_channels(
                amp_s.transform(d.loc[query, "predicted_magnitude"]),
                public_s.transform(d.loc[query, "prediction_prior_rmse"]),
                None,
                target_s.transform(d.loc[query, "target_oof_score"]),
                has_public=d.loc[query, "prediction_prior_rmse"].notna().astype(int),
                has_shared=np.zeros(query.sum(), dtype=int),
                has_target=np.ones(query.sum(), dtype=int),
            )
            model = OOFReadyRidgeFusion(alpha=10, label_source="McFaline_DEV_true_error_rmse",
                                        target_error_budget=f"{budget:.2f}")
            model.fit(x_train, y_s, names, base_scores_are_oof=True)
            oof_fused[query] = model.predict(x_query)
            audit_rows.append({"budget": budget, "fold": int(fold), "n_fit": int(train.sum()),
                               "n_query": int(query.sum()), "label_source": "McFaline_DEV_true_error_rmse",
                               "target_feedback_budget": budget, "base_scores_are_oof": True})
            cdf_rows.extend([
                {"budget": budget, "fold": int(fold), "cdf_kind": "channel_score", "channel": name,
                 "fit_rows": int(train.sum()), "uses_target_error": False}
                for name in ["amplitude_q", "public_q", "target_q"]
            ])
            cdf_rows.append({"budget": budget, "fold": int(fold), "cdf_kind": "error_label",
                             "channel": "target_error", "fit_rows": int(train.sum()),
                             "uses_target_error": True})
        # Fit the selected DEV-side candidate on all 542 tasks and score the
        # fixed 212-task holdout. Holdout errors are never passed to fit().
        amp_s = TrainOnlyQuantileScale.fit(d.predicted_magnitude, "amplitude_q")
        public_s = TrainOnlyQuantileScale.fit(d.prediction_prior_rmse, "public_q")
        target_s = TrainOnlyQuantileScale.fit(d.target_oof_score, "target_q")
        _, y_all = labels_cdf(d.true_error_rmse, d.true_error_rmse)
        x_all, names = build_fusion_channels(
            amp_s.transform(d.predicted_magnitude), public_s.transform(d.prediction_prior_rmse), None,
            target_s.transform(d.target_oof_score),
            has_public=d.prediction_prior_rmse.notna().astype(int),
            has_shared=np.zeros(len(d), dtype=int), has_target=np.ones(len(d), dtype=int))
        model = OOFReadyRidgeFusion(alpha=10, label_source="McFaline_DEV_true_error_rmse",
                                    target_error_budget=f"{budget:.2f}").fit(
            x_all, y_all, names, base_scores_are_oof=True)
        q = fixed_pred[(fixed_pred.method == "H1_F1") & (fixed_pred.budget == budget)][
            ["task_id", "risk"]
        ].rename(columns={"risk": "target_score"})
        if len(q) != len(hold) or q.task_id.nunique() != len(hold):
            raise RuntimeError(f"fixed target score coverage incomplete for budget {budget}")
        h = hold.merge(q, on="task_id", how="left", validate="one_to_one")
        h_amp = amp_s.transform(h.predicted_magnitude)
        h_public = public_s.transform(h.prediction_prior_rmse)
        h_target = target_s.transform(h.target_score)
        x_h, _ = build_fusion_channels(
            h_amp, h_public, None, h_target,
            has_public=h.prediction_prior_rmse.notna().astype(int),
            has_shared=np.zeros(len(h), dtype=int), has_target=np.ones(len(h), dtype=int))
        h_fused = model.predict(x_h)
        h_rule, _ = rule_score(h_amp, h_public)
        dev_oof_exports.append(d[["task_id", "gene", "dev_fold", "true_error_rmse"]].assign(
            target_feedback_budget=budget, target_oof_score=d.target_oof_score.to_numpy(float),
            target_ridge_oof=oof_fused,
        ))
        # Keep pre-existing candidate outputs as fixed references.
        references = {
            "Amplitude": h.predicted_magnitude.to_numpy(float),
            "PublicRule_simple_history": h.simple_history_risk.to_numpy(float),
            "TargetRidge_fusion": h_fused,
        }
        strong_q = strong_public[strong_public.budget == 0]
        if strong_max_truth_delta <= 1e-12 and len(strong_q) == len(h):
            references["StrongPublic_WeightedHistoryDistance"] = h.task_id.map(
                strong_q.set_index("task_id").risk
            ).to_numpy(float)
        for method in ["TargetOnly_HGB", "PublicTarget_HGB", "SharedTarget_HGB", "ResidualHGB", "Shared"]:
            qmethod = strict_pred[(strict_pred.budget == budget) & (strict_pred.method == method)]
            if len(qmethod) == len(h):
                references[method] = h.task_id.map(qmethod.set_index("task_id").risk).to_numpy(float)
        for method in ["H1_F1", "X0_F1", "H1_F2", "X0_F2"]:
            qmethod = fixed_pred[(fixed_pred.budget == budget) & (fixed_pred.method == method)]
            if len(qmethod) == len(h):
                references[method] = h.task_id.map(qmethod.set_index("task_id").risk).to_numpy(float)
        for method, values in references.items():
            row = metric_frame(h, values, method)
            row["budget"] = budget
            candidate_rows.append(row)
        pd.DataFrame([model.audit()]).to_json(out / f"FUSION_AUDIT_B{int(budget*100):03d}.json", orient="records", indent=2)
        score_export = h[["task_id", "gene", "target", "true_error_rmse", "predicted_magnitude",
                          "prediction_prior_rmse", "target_score"]].copy()
        score_export["TargetRidge_fusion"] = h_fused
        score_export["PublicRule_no_error_labels"] = h_rule
        for method, values in references.items():
            if method not in score_export:
                score_export[method] = values
        score_export["target_feedback_budget"] = budget
        score_export.to_parquet(out / f"HOLDOUT_SCORES_B{int(budget*100):03d}.parquet", index=False)
    pd.DataFrame(candidate_rows).to_csv(out / "METRICS.csv", index=False)
    pd.concat(dev_oof_exports, ignore_index=True).to_parquet(out / "DEV_OOF_FUSION.parquet", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "SPLIT_AND_FIT_LEDGER.csv", index=False)
    pd.DataFrame(cdf_rows).to_csv(out / "CDF_AUDIT.csv", index=False)
    pd.DataFrame([{
        "label_source": "McFaline_DEV_true_error_rmse",
        "target_feedback_budget": "0.10/0.25/0.50/0.75/1.00",
        "evaluation_errors_used_for_fit": 0,
        "evaluation_tasks": len(hold),
        "development_tasks": len(dev),
        "channel_scores_oof": True,
        "legacy_strong_public_max_abs_truth_delta": strong_max_truth_delta,
        "legacy_strong_public_reused": bool(strong_max_truth_delta <= 1e-12),
    }]).to_csv(out / "INFORMATION_BUDGET.csv", index=False)
    pd.DataFrame([{
        "asset": str(STRONG_PUBLIC_PATH), "rows": len(strong_alignment),
        "max_abs_truth_delta": strong_max_truth_delta,
        "same_truth_contract": bool(strong_max_truth_delta <= 1e-12),
        "action": "reuse" if strong_max_truth_delta <= 1e-12 else "exclude_from_same_task_metrics",
    }]).to_csv(out / "LEGACY_ALIGNMENT_AUDIT.csv", index=False)
    with (out / "RUN_STATUS.json").open("w") as f:
        json.dump({"status": "COMPLETE", "development_tasks": len(dev), "holdout_tasks": len(hold),
                   "budgets": list(BUDGETS.values()), "target_errors_used_for_holdout_fit": 0,
                   "outputs": [p.name for p in sorted(out.iterdir())]}, f, indent=2)
    return pd.DataFrame(candidate_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.root, args.out).to_string(index=False))


if __name__ == "__main__":
    main()
