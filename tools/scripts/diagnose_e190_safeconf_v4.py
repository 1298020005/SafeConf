#!/usr/bin/env python3
"""Localize E190 SafeConf-v4 failures without modifying the frozen method.

The script replays the released five outer folds and exposes the prediction
branch, history branch, calibration, and evidence gate separately.  Three
post-hoc calibration variants are diagnostic only; none replaces the frozen
v4 candidate or constitutes a new confirmation result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import run_safeconf_v4_development as core  # noqa: E402


STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
SOURCE = STAGE / "safeconf_v4_development/e190_gears_crossfamily"
DEFAULT_OUT = STAGE / "e190_failure_localization"
SEED = 20260930
PRACTICAL_MARGIN = 0.005
CALIBRATORS = ("frozen_linear", "identity", "positive_affine", "isotonic")


def metrics(task_ids: np.ndarray, risk: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    return core.utility20(task_ids.astype(str), risk.astype(float), truth.astype(float)), core.rho(
        risk.astype(float), truth.astype(float)
    )


def classify_abnormalities(
    raw_delta_u20: float,
    calibrated_delta_u20: float,
    p_slope: float,
    h_slope: float,
    v2_loss_from_best: float,
    mean_weight: float,
    margin: float = PRACTICAL_MARGIN,
) -> dict[str, bool | str]:
    """Classify observed pipeline abnormalities without assigning root cause."""
    history_underperformance = bool(raw_delta_u20 < -margin)
    rank_reversal = bool(
        (raw_delta_u20 >= margin and calibrated_delta_u20 < -margin)
        or (raw_delta_u20 <= -margin and calibrated_delta_u20 > margin)
    )
    calibration_abnormality = bool(
        (np.isfinite(p_slope) and p_slope <= 0)
        or (np.isfinite(h_slope) and h_slope <= 0)
        or rank_reversal
    )
    history_is_better = calibrated_delta_u20 >= margin
    prediction_is_better = calibrated_delta_u20 <= -margin
    gate_favors_history = mean_weight > 0.5
    gate_misrouting = bool(
        v2_loss_from_best > margin
        and (
            (history_is_better and not gate_favors_history)
            or (prediction_is_better and gate_favors_history)
        )
    )
    labels = []
    if history_underperformance:
        labels.append("History branch underperformance")
    if calibration_abnormality:
        labels.append("Calibration abnormality")
    if gate_misrouting:
        labels.append("Gate misrouting")
    if not labels:
        labels.append("No registered abnormality")
    return {
        "history_branch_underperformance": history_underperformance,
        "calibration_abnormality": calibration_abnormality,
        "gate_misrouting": gate_misrouting,
        "abnormality_class": "Mixed abnormality" if len(labels) > 1 else labels[0],
        "abnormality_labels": "; ".join(labels),
    }


def fit_calibrator(name: str, score: np.ndarray, truth: np.ndarray):
    score = np.asarray(score, float)
    truth = np.asarray(truth, float)
    if name == "identity":
        return lambda x: np.maximum(0.0, np.asarray(x, float)), 1.0, 0.0
    if name in {"frozen_linear", "positive_affine"}:
        model = LinearRegression(positive=name == "positive_affine").fit(score[:, None], truth)
        return (
            lambda x: np.maximum(0.0, model.predict(np.asarray(x, float)[:, None])),
            float(model.coef_[0]),
            float(model.intercept_),
        )
    if name == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0).fit(score, truth)
        return lambda x: np.asarray(model.predict(np.asarray(x, float)), float), math.nan, math.nan
    raise ValueError(name)


def fit_gate(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    p_oof: np.ndarray,
    h_oof: np.ndarray,
    p_test: np.ndarray,
    h_test: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ev_train, ev_query = core.evidence_matrix(fit, query)

    def objective(theta: np.ndarray) -> float:
        logits = theta[0] + ev_train @ theta[1:]
        weight = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        pred = p_oof + weight * (h_oof - p_oof)
        scale = max(float(truth.std()), 1e-8)
        return float(np.mean(((pred - truth) / scale) ** 2) + 1e-4 * np.sum(theta[1:] ** 2))

    bounds = [(-10, 10), (0, 10), (0, 10), (-10, 0), (-10, 0)]
    fitted = minimize(objective, np.zeros(5), method="L-BFGS-B", bounds=bounds)
    if not fitted.success:
        raise RuntimeError(f"diagnostic gate optimization failed: {fitted.message}")
    logits = fitted.x[0] + ev_query @ fitted.x[1:]
    weight = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    pred = p_test + weight * (h_test - p_test)
    return pred, {
        "gate_intercept": float(fitted.x[0]),
        "coef_support": float(fitted.x[1]),
        "coef_relevance": float(fitted.x[2]),
        "coef_conflict": float(fitted.x[3]),
        "coef_missingness": float(fitted.x[4]),
        "mean_weight": float(weight.mean()),
        "min_weight": float(weight.min()),
        "max_weight": float(weight.max()),
        "inner_objective": float(fitted.fun),
    }


def inner_predictions(fit: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p_oof = np.full(len(fit), np.nan)
    h_oof = np.full(len(fit), np.nan)
    for inner, (tr, va) in enumerate(GroupKFold(4).split(fit, groups=fit.gene)):
        p_oof[va] = core.fit_model(fit.iloc[tr], fit.iloc[va], core.GROUPS["U"], "Ridge", seed + inner)
        h_oof[va] = core.fit_model(
            fit.iloc[tr], fit.iloc[va], core.GROUPS["USRCH"], "Ridge", seed + 20 + inner
        )
    if not np.isfinite(p_oof).all() or not np.isfinite(h_oof).all():
        raise RuntimeError("inner OOF branch predictions are incomplete")
    return p_oof, h_oof


def diagnose(frame: pd.DataFrame, released: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []
    variant_rows: list[dict] = []
    task_rows: list[dict] = []
    released_gates = pd.read_csv(SOURCE / "V2_GATE_FITS.csv").set_index("fold")
    part = frame.sort_values(["gene", "task_id"]).reset_index(drop=True)
    for fold, (tr, va) in enumerate(GroupKFold(5).split(part, groups=part.gene)):
        fit = part.iloc[tr].copy()
        query = part.iloc[va].copy()
        # The released E190 runner calls the frozen core directly; therefore
        # model fitting uses core.SEED.  E190's local SEED is bootstrap-only.
        seed = core.SEED + 1000 * (fold + 1)
        p_oof, h_oof = inner_predictions(fit, seed)
        raw_p = core.fit_model(fit, query, core.GROUPS["U"], "Ridge", seed + 100)
        raw_h = core.fit_model(fit, query, core.GROUPS["USRCH"], "Ridge", seed + 200)
        v1 = core.fit_model(fit, query, core.GROUPS["USR"], "Ridge", core.SEED + fold)
        truth_train = fit.true_error_rmse.to_numpy(float)
        truth = query.true_error_rmse.to_numpy(float)
        ids = query.task_id.to_numpy(str)
        ev_train, ev_query = core.evidence_matrix(fit, query)
        raw_p_u20, raw_p_rho = metrics(ids, raw_p, truth)
        raw_h_u20, raw_h_rho = metrics(ids, raw_h, truth)
        v1_u20, v1_rho = metrics(ids, v1, truth)
        frozen: dict | None = None
        for calibrator in CALIBRATORS:
            p_apply, p_slope, p_intercept = fit_calibrator(calibrator, p_oof, truth_train)
            h_apply, h_slope, h_intercept = fit_calibrator(calibrator, h_oof, truth_train)
            pc_oof, hc_oof = p_apply(p_oof), h_apply(h_oof)
            pc_test, hc_test = p_apply(raw_p), h_apply(raw_h)
            v2, gate = fit_gate(fit, query, pc_oof, hc_oof, pc_test, hc_test, truth_train)
            pc_u20, pc_rho = metrics(ids, pc_test, truth)
            hc_u20, hc_rho = metrics(ids, hc_test, truth)
            replay_maxdiff = math.nan
            v2_for_metrics = v2
            if calibrator == "frozen_linear":
                released_fold = released[(released.fold == fold) & released.method.eq("V2_nested")]
                released_map = released_fold.set_index("task_id").predicted_risk.to_dict()
                if set(released_map) != set(ids):
                    raise RuntimeError(f"fold {fold}: released task IDs differ from diagnostic replay")
                released_risk = np.asarray([released_map[task_id] for task_id in ids], float)
                replay_maxdiff = float(np.max(np.abs(released_risk - v2)))
                # scipy optimizer stopping can vary slightly by runtime version.
                # The released scores remain the source of truth for frozen v4.
                if replay_maxdiff > 1e-3:
                    raise RuntimeError(f"fold {fold}: frozen V2 replay differs by {replay_maxdiff}")
                v2_for_metrics = released_risk
                gate.update(
                    {
                        key: float(value)
                        for key, value in released_gates.loc[fold].to_dict().items()
                        if key not in {"target"}
                    }
                )
            v2_u20, v2_rho = metrics(ids, v2_for_metrics, truth)
            variant = {
                "fold": fold,
                "calibrator": calibrator,
                "n_train_tasks": len(fit),
                "n_test_tasks": len(query),
                "n_train_gene_clusters": int(fit.gene.nunique()),
                "n_test_gene_clusters": int(query.gene.nunique()),
                "p_slope": p_slope,
                "p_intercept": p_intercept,
                "h_slope": h_slope,
                "h_intercept": h_intercept,
                "raw_rP_U20": raw_p_u20,
                "raw_rP_Spearman": raw_p_rho,
                "raw_rPQ_U20": raw_h_u20,
                "raw_rPQ_Spearman": raw_h_rho,
                "calibrated_rP_U20": pc_u20,
                "calibrated_rP_Spearman": pc_rho,
                "calibrated_rPQ_U20": hc_u20,
                "calibrated_rPQ_Spearman": hc_rho,
                "V1_U20": v1_u20,
                "V1_Spearman": v1_rho,
                "V2_U20": v2_u20,
                "V2_Spearman": v2_rho,
                "frozen_replay_max_absdiff": replay_maxdiff,
                **gate,
            }
            variant_rows.append(variant)
            for task_id, risk in zip(ids, v2_for_metrics):
                row_index = int(np.flatnonzero(ids == task_id)[0])
                task_rows.append(
                    {
                        "task_id": task_id,
                        "gene": str(query.loc[query.task_id.eq(task_id), "gene"].iloc[0]),
                        "fold": fold,
                        "calibrator": calibrator,
                        "true_error_rmse": float(query.loc[query.task_id.eq(task_id), "true_error_rmse"].iloc[0]),
                        "raw_rP": float(raw_p[row_index]),
                        "raw_rPQ": float(raw_h[row_index]),
                        "calibrated_rP": float(pc_test[row_index]),
                        "calibrated_rPQ": float(hc_test[row_index]),
                        "V1": float(v1[row_index]),
                        "predicted_risk": float(risk),
                    }
                )
            if calibrator == "frozen_linear":
                frozen = variant
        assert frozen is not None
        calibrated_delta = frozen["calibrated_rPQ_U20"] - frozen["calibrated_rP_U20"]
        best = max(frozen["calibrated_rP_U20"], frozen["calibrated_rPQ_U20"])
        classes = classify_abnormalities(
            raw_delta_u20=raw_h_u20 - raw_p_u20,
            calibrated_delta_u20=calibrated_delta,
            p_slope=float(frozen["p_slope"]),
            h_slope=float(frozen["h_slope"]),
            v2_loss_from_best=best - frozen["V2_U20"],
            mean_weight=float(frozen["mean_weight"]),
        )
        fold_rows.append(
            {
                **frozen,
                "raw_rPQ_minus_rP_U20": raw_h_u20 - raw_p_u20,
                "calibrated_rPQ_minus_rP_U20": calibrated_delta,
                "V2_loss_from_best_calibrated_branch": best - frozen["V2_U20"],
                "support_test_smd": float(ev_query[:, 0].mean() - ev_train[:, 0].mean()),
                "relevance_test_smd": float(ev_query[:, 1].mean() - ev_train[:, 1].mean()),
                "conflict_test_smd": float(ev_query[:, 2].mean() - ev_train[:, 2].mean()),
                "missingness_test_minus_train": float(ev_query[:, 3].mean() - ev_train[:, 3].mean()),
                **classes,
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(variant_rows), pd.DataFrame(task_rows)


def aggregate_variants(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calibrator, group in tasks.groupby("calibrator", sort=False):
        ids = group.task_id.to_numpy(str)
        truth = group.true_error_rmse.to_numpy(float)
        row = {"calibrator": calibrator, "n_tasks": len(group)}
        for label, column in (
            ("raw_rP", "raw_rP"),
            ("raw_rPQ", "raw_rPQ"),
            ("calibrated_rP", "calibrated_rP"),
            ("calibrated_rPQ", "calibrated_rPQ"),
            ("V1", "V1"),
            ("V2", "predicted_risk"),
        ):
            u20, rho = metrics(ids, group[column].to_numpy(float), truth)
            row[f"{label}_U20"] = u20
            row[f"{label}_Spearman"] = rho
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_variants(tasks: pd.DataFrame, n_bootstrap: int = 5000) -> pd.DataFrame:
    wide = tasks.pivot(
        index=["task_id", "gene", "true_error_rmse"], columns="calibrator", values="predicted_risk"
    ).reset_index()
    v1 = tasks[tasks.calibrator.eq("frozen_linear")][["task_id", "V1"]]
    wide = wide.merge(v1, on="task_id", validate="one_to_one")
    comparisons = (
        ("identity_vs_frozen", "identity", "frozen_linear"),
        ("isotonic_vs_frozen", "isotonic", "frozen_linear"),
        ("positive_affine_vs_frozen", "positive_affine", "frozen_linear"),
        ("identity_vs_V1", "identity", "V1"),
        ("isotonic_vs_V1", "isotonic", "V1"),
    )
    genes = np.asarray(sorted(wide.gene.unique()))
    gene_rows = [np.flatnonzero(wide.gene.to_numpy(str) == gene) for gene in genes]
    ids = wide.task_id.to_numpy(str)
    truth = wide.true_error_rmse.to_numpy(float)
    risks = {name: wide[name].to_numpy(float) for _, a, b in comparisons for name in (a, b)}
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name, _, _ in comparisons}
    for _ in range(n_bootstrap):
        take = np.concatenate([gene_rows[index] for index in rng.integers(0, len(genes), len(genes))])
        values = {method: core.utility20(ids[take], risk[take], truth[take]) for method, risk in risks.items()}
        for name, left, right in comparisons:
            draws[name].append(values[left] - values[right])
    rows = []
    for name, left, right in comparisons:
        delta = core.utility20(ids, risks[left], truth) - core.utility20(ids, risks[right], truth)
        values = np.asarray(draws[name], float)
        rows.append(
            {
                "comparison": name,
                "method_a": left,
                "method_b": right,
                "delta_utility20": float(delta),
                "utility_ci95_lower": float(np.nanquantile(values, 0.025)),
                "utility_ci95_upper": float(np.nanquantile(values, 0.975)),
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_unit": "gene_cluster",
            }
        )
    return pd.DataFrame(rows)


def report_markdown(
    folds: pd.DataFrame, variants: pd.DataFrame, aggregate: pd.DataFrame, bootstrap: pd.DataFrame
) -> str:
    gate_count = int(folds.gate_misrouting.sum())
    calibration_count = int(folds.calibration_abnormality.sum())
    history_count = int(folds.history_branch_underperformance.sum())
    lines = [
        "# E190 SafeConf-v4 failure localization",
        "",
        "This is post-hoc diagnosis on a SEEN development asset. It does not modify frozen v4.",
        "",
        "## Frozen pipeline by fold",
        "",
        "| fold | raw P U20 | raw PQ U20 | p slope | pq slope | weight | V2 U20 | classification |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in folds.itertuples():
        lines.append(
            f"| {row.fold} | {row.raw_rP_U20:.4f} | {row.raw_rPQ_U20:.4f} | "
            f"{row.p_slope:.4f} | {row.h_slope:.4f} | {row.mean_weight:.4f} | "
            f"{row.V2_U20:.4f} | {row.abnormality_labels} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic calibration variants",
            "",
            "| calibration | raw P | raw PQ | calibrated P | calibrated PQ | V2 | V2 rho |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate.itertuples():
        lines.append(
            f"| {row.calibrator} | {row.raw_rP_U20:.6f} | {row.raw_rPQ_U20:.6f} | "
            f"{row.calibrated_rP_U20:.6f} | {row.calibrated_rPQ_U20:.6f} | "
            f"{row.V2_U20:.6f} | {row.V2_Spearman:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired gene-cluster bootstrap for diagnostic variants",
            "",
            "| comparison | delta U20 | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples():
        lines.append(
            f"| {row.comparison} | {row.delta_utility20:+.6f} | "
            f"[{row.utility_ci95_lower:+.6f}, {row.utility_ci95_upper:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Calibration abnormality occurs in {calibration_count}/5 folds.",
            f"- History branch underperformance occurs in {history_count}/5 folds.",
            f"- Gate misrouting occurs in {gate_count}/5 folds.",
            "- Frozen fold-specific affine calibration reduces pooled history-branch U20 from "
            f"{aggregate.set_index('calibrator').loc['frozen_linear', 'raw_rPQ_U20']:.6f} to "
            f"{aggregate.set_index('calibrator').loc['frozen_linear', 'calibrated_rPQ_U20']:.6f}.",
            "- Identity and isotonic calibration recover pooled V2 U20 to "
            f"{aggregate.set_index('calibrator').loc['identity', 'V2_U20']:.6f} and "
            f"{aggregate.set_index('calibrator').loc['isotonic', 'V2_U20']:.6f}, respectively.",
            "- Cluster-count downsampling is authorized only when gate misrouting appears in at least 3/5 folds.",
            f"- Downsampling decision: **{'RUN' if gate_count >= 3 else 'STOP'}**.",
            "- Distribution summaries are diagnostic and are not used as an automatic causal verdict.",
            "",
            "The calibration variants are explanatory controls. Their performance cannot replace the frozen v4 result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.out}")
    started = time.time()
    frame = pd.read_csv(SOURCE / "FEATURE_TABLE.csv.gz")
    released = pd.read_csv(SOURCE / "OOF_PREDICTIONS.csv.gz")
    folds, variants, tasks = diagnose(frame, released)
    aggregate = aggregate_variants(tasks)
    bootstrap = bootstrap_variants(tasks)
    args.out.mkdir(parents=True)
    folds.to_csv(args.out / "E190_FAILURE_LOCALIZATION.csv", index=False)
    variants.to_csv(args.out / "E190_CALIBRATION_VARIANTS_BY_FOLD.csv", index=False)
    aggregate.to_csv(args.out / "E190_CALIBRATION_VARIANTS_SUMMARY.csv", index=False)
    bootstrap.to_csv(args.out / "E190_CALIBRATION_VARIANT_BOOTSTRAP.csv", index=False)
    (args.out / "E190_FAILURE_LOCALIZATION.md").write_text(
        report_markdown(folds, variants, aggregate, bootstrap), encoding="utf-8"
    )
    status = {
        "status": "COMPLETE",
        "data_role": "SEEN",
        "frozen_v4_changed": False,
        "n_folds": int(len(folds)),
        "calibration_abnormality_folds": int(folds.calibration_abnormality.sum()),
        "history_branch_underperformance_folds": int(folds.history_branch_underperformance.sum()),
        "gate_misrouting_folds": int(folds.gate_misrouting.sum()),
        "run_cluster_downsampling": bool(folds.gate_misrouting.sum() >= 3),
        "cross_fold_calibration_abnormality": bool(
            aggregate.set_index("calibrator").loc["identity", "V2_U20"]
            - aggregate.set_index("calibrator").loc["frozen_linear", "V2_U20"]
            > PRACTICAL_MARGIN
        ),
        "elapsed_seconds": time.time() - started,
    }
    (args.out / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(folds[["fold", "abnormality_labels"]].to_string(index=False))
    print(aggregate.to_string(index=False))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
