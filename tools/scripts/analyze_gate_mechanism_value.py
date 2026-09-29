#!/usr/bin/env python3
"""Test whether SafeConf's learned gate beats fixed branch mixtures.

All inputs are DEV/SEEN.  The experiment replays the frozen branch models and
calibration, varies only the diagnostic mixing weight, and never changes v4.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scripts import run_e190_gears_safeconf_v4 as e190  # noqa: E402
from tools.scripts import run_safeconf_v4_development as core  # noqa: E402


STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
OUT = STAGE / "gate_mechanism_value"
UPSTREAMS = ("txpert_gat", "txpert_exphormer", "e190_gears")
FIXED_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_BOOTSTRAP = 5000
SEED = 20261001


def load_asset(upstream: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if upstream == "txpert_gat":
        frame, _ = core.build_frame(core.configure("e201"))
        path = STAGE / "safeconf_v4_development/txpert_gat/OOF_PREDICTIONS.csv.gz"
    elif upstream == "txpert_exphormer":
        frame, _ = core.build_frame(core.configure("e205"))
        path = STAGE / "safeconf_v4_development/txpert_exphormer/OOF_PREDICTIONS.csv.gz"
    elif upstream == "e190_gears":
        e190.verify_inputs()
        frame, _ = e190.build_frame()
        path = STAGE / "safeconf_v4_development/e190_gears_crossfamily/OOF_PREDICTIONS.csv.gz"
    else:
        raise ValueError(upstream)
    return frame, pd.read_csv(path)


def inner_branches(fit: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.full(len(fit), np.nan)
    h = np.full(len(fit), np.nan)
    for inner, (tr, va) in enumerate(GroupKFold(4).split(fit, groups=fit.gene)):
        p[va] = core.fit_model(fit.iloc[tr], fit.iloc[va], core.GROUPS["U"], "Ridge", seed + inner)
        h[va] = core.fit_model(
            fit.iloc[tr], fit.iloc[va], core.GROUPS["USRCH"], "Ridge", seed + 20 + inner
        )
    if not np.isfinite(p).all() or not np.isfinite(h).all():
        raise RuntimeError("inner branch predictions are incomplete")
    return p, h


def replay_fold(fit: pd.DataFrame, query: pd.DataFrame, seed: int) -> dict[str, object]:
    p_oof, h_oof = inner_branches(fit, seed)
    truth_train = fit.true_error_rmse.to_numpy(float)
    p_cal = LinearRegression().fit(p_oof[:, None], truth_train)
    h_cal = LinearRegression().fit(h_oof[:, None], truth_train)
    p_oof_cal = np.maximum(0.0, p_cal.predict(p_oof[:, None]))
    h_oof_cal = np.maximum(0.0, h_cal.predict(h_oof[:, None]))
    p_raw = core.fit_model(fit, query, core.GROUPS["U"], "Ridge", seed + 100)
    h_raw = core.fit_model(fit, query, core.GROUPS["USRCH"], "Ridge", seed + 200)
    p_test = np.maximum(0.0, p_cal.predict(p_raw[:, None]))
    h_test = np.maximum(0.0, h_cal.predict(h_raw[:, None]))
    ev_train, ev_test = core.evidence_matrix(fit, query)

    def objective(theta: np.ndarray) -> float:
        weight = 1.0 / (1.0 + np.exp(-np.clip(theta[0] + ev_train @ theta[1:], -30, 30)))
        pred = p_oof_cal + weight * (h_oof_cal - p_oof_cal)
        scale = max(float(truth_train.std()), 1e-8)
        return float(np.mean(((pred - truth_train) / scale) ** 2) + 1e-4 * np.sum(theta[1:] ** 2))

    fitted = minimize(
        objective,
        np.zeros(5),
        method="L-BFGS-B",
        bounds=[(-10, 10), (0, 10), (0, 10), (-10, 0), (-10, 0)],
    )
    if not fitted.success:
        raise RuntimeError(f"gate optimization failed: {fitted.message}")
    weight = 1.0 / (1.0 + np.exp(-np.clip(fitted.x[0] + ev_test @ fitted.x[1:], -30, 30)))
    return {
        "p": p_test,
        "h": h_test,
        "weight": weight,
        "learned_replay": p_test + weight * (h_test - p_test),
        "p_slope": float(p_cal.coef_[0]),
        "h_slope": float(h_cal.coef_[0]),
        "theta": fitted.x,
    }


def build_predictions(upstream: str, frame: pd.DataFrame, released: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    released_v2 = released[released.method.eq("V2_nested")].set_index("task_id").predicted_risk.to_dict()
    for target, part in frame.groupby("target", sort=True):
        part = part.sort_values(["gene", "task_id"]).reset_index(drop=True)
        for fold, (tr, va) in enumerate(GroupKFold(5).split(part, groups=part.gene)):
            fit, query = part.iloc[tr].copy(), part.iloc[va].copy()
            replay = replay_fold(fit, query, core.SEED + 1000 * (fold + 1))
            ids = query.task_id.to_numpy(str)
            truth = query.true_error_rmse.to_numpy(float)
            released_score = np.asarray([released_v2[x] for x in ids], float)
            maxdiff = float(np.max(np.abs(np.asarray(replay["learned_replay"]) - released_score)))
            if maxdiff > 1e-3:
                raise RuntimeError(f"{upstream}/{target}/fold{fold}: replay difference {maxdiff}")
            methods = {f"fixed_w_{weight:.2f}": replay["p"] + weight * (replay["h"] - replay["p"]) for weight in FIXED_WEIGHTS}
            methods["learned_gate"] = released_score
            v1 = core.fit_model(fit, query, core.GROUPS["USR"], "Ridge", core.SEED + fold)
            methods["V1"] = v1
            for method, risk in methods.items():
                for index, task_id in enumerate(ids):
                    rows.append(
                        {
                            "upstream": upstream,
                            "target": target,
                            "fold": fold,
                            "task_id": task_id,
                            "gene": str(query.iloc[index].gene),
                            "true_error_rmse": float(truth[index]),
                            "method": method,
                            "predicted_risk": float(risk[index]),
                            "learned_weight": float(np.asarray(replay["weight"])[index]),
                        }
                    )
            branch_u20 = {
                name: core.utility20(ids, np.asarray(risk), truth)
                for name, risk in methods.items()
                if name in {"fixed_w_0.00", "fixed_w_1.00", "learned_gate", "V1"}
            }
            better = "history" if branch_u20["fixed_w_1.00"] > branch_u20["fixed_w_0.00"] else "prediction"
            favors = "history" if float(np.mean(replay["weight"])) > 0.5 else "prediction"
            fold_rows.append(
                {
                    "upstream": upstream,
                    "target": target,
                    "fold": fold,
                    "n_tasks": len(query),
                    "n_gene_clusters": int(query.gene.nunique()),
                    "p_slope": replay["p_slope"],
                    "h_slope": replay["h_slope"],
                    "mean_weight": float(np.mean(replay["weight"])),
                    "min_weight": float(np.min(replay["weight"])),
                    "max_weight": float(np.max(replay["weight"])),
                    "prediction_branch_u20": branch_u20["fixed_w_0.00"],
                    "history_branch_u20": branch_u20["fixed_w_1.00"],
                    "learned_gate_u20": branch_u20["learned_gate"],
                    "v1_u20": branch_u20["V1"],
                    "oracle_better_branch": better,
                    "gate_favored_branch": favors,
                    "routing_correct": better == favors,
                    "replay_max_absdiff": maxdiff,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def metric_rows(predictions: pd.DataFrame, level: str) -> pd.DataFrame:
    group_cols = ["upstream", "target", "method"] if level == "context" else ["upstream", "target", "fold", "method"]
    rows = []
    for keys, group in predictions.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(group_cols, keys))
        ids = group.task_id.to_numpy(str)
        risk = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        item.update(
            {
                "level": level,
                "n_tasks": len(group),
                "utility20": core.utility20(ids, risk, truth),
                "spearman": core.rho(risk, truth),
                **core.selective_metrics(ids, risk, truth),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    result = []
    rng = np.random.default_rng(SEED)
    for upstream, data in predictions.groupby("upstream", sort=True):
        wide = data.pivot(
            index=["task_id", "target", "gene", "true_error_rmse"], columns="method", values="predicted_risk"
        ).reset_index()
        fixed = [f"fixed_w_{weight:.2f}" for weight in FIXED_WEIGHTS]
        fixed_points = {}
        for method in fixed:
            values = []
            for _, group in wide.groupby("target"):
                values.append(core.utility20(group.task_id.to_numpy(str), group[method].to_numpy(float), group.true_error_rmse.to_numpy(float)))
            fixed_points[method] = float(np.nanmean(values))
        best_fixed = max(fixed_points, key=fixed_points.get)
        comparisons = (("learned_vs_best_fixed", "learned_gate", best_fixed), ("learned_vs_V1", "learned_gate", "V1"))
        genes = np.asarray(sorted(wide.gene.unique()))
        gene_rows = [np.flatnonzero(wide.gene.to_numpy(str) == gene) for gene in genes]
        draws = {name: [] for name, _, _ in comparisons}
        for _ in range(N_BOOTSTRAP):
            take = np.concatenate([gene_rows[i] for i in rng.integers(0, len(genes), len(genes))])
            sample = wide.iloc[take]
            values = {}
            for method in {m for _, a, b in comparisons for m in (a, b)}:
                per_target = []
                for _, group in sample.groupby("target"):
                    per_target.append(core.utility20(group.task_id.to_numpy(str), group[method].to_numpy(float), group.true_error_rmse.to_numpy(float)))
                values[method] = float(np.nanmean(per_target))
            for name, left, right in comparisons:
                draws[name].append(values[left] - values[right])
        for name, left, right in comparisons:
            values = np.asarray(draws[name], float)
            point_left = metric_rows(data[data.method.eq(left)], "context").utility20.mean()
            point_right = metric_rows(data[data.method.eq(right)], "context").utility20.mean()
            result.append(
                {
                    "upstream": upstream,
                    "comparison": name,
                    "method_a": left,
                    "method_b": right,
                    "delta_u20": float(point_left - point_right),
                    "ci95_lower": float(np.nanquantile(values, 0.025)),
                    "ci95_upper": float(np.nanquantile(values, 0.975)),
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "bootstrap_unit": "gene_cluster",
                }
            )
    return pd.DataFrame(result)


def make_report(summary: pd.DataFrame, folds: pd.DataFrame, boots: pd.DataFrame) -> str:
    lines = [
        "# SafeConf learned-gate mechanism test",
        "",
        "DEV/SEEN diagnostic only. Frozen v4 and external confirmation are unchanged.",
        "",
        "## Fixed mixtures and learned gate",
        "",
        "| upstream | method | U20 | Spearman | AURC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.upstream} | {row.method} | {row.utility20:.6f} | {row.spearman:.6f} | {row.aurc:.6f} |")
    lines.extend(["", "## Mechanism decision", ""])
    for upstream, part in summary.groupby("upstream"):
        fixed = part[part.method.str.startswith("fixed_w_")].sort_values("utility20", ascending=False).iloc[0]
        learned = part[part.method.eq("learned_gate")].iloc[0]
        v1 = part[part.method.eq("V1")].iloc[0]
        routing = folds[folds.upstream.eq(upstream)].routing_correct.mean()
        boot = boots[(boots.upstream.eq(upstream)) & (boots.comparison.eq("learned_vs_best_fixed"))].iloc[0]
        lines.append(
            f"- **{upstream}**: best fixed `{fixed.method}` U20={fixed.utility20:.6f}; learned={learned.utility20:.6f}; "
            f"V1={v1.utility20:.6f}; correct branch routing={routing:.1%}; learned-best fixed "
            f"delta={boot.delta_u20:+.6f} (95% CI [{boot.ci95_lower:+.6f}, {boot.ci95_upper:+.6f}])."
        )
    lines.extend(
        [
            "",
            "The gate is considered a core method contribution only if it improves on the best fixed mixture across assets and does not rely on a single family. Otherwise it remains a conditional extension.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    started = time.time()
    all_predictions, all_folds = [], []
    for upstream in UPSTREAMS:
        frame, released = load_asset(upstream)
        predictions, folds = build_predictions(upstream, frame, released)
        all_predictions.append(predictions)
        all_folds.append(folds)
        print(f"[gate-mechanism] {upstream} complete", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    context = metric_rows(predictions, "context")
    fold_metrics = metric_rows(predictions, "fold")
    summary = context.groupby(["upstream", "method"], as_index=False).agg(
        n_contexts=("target", "nunique"),
        utility20=("utility20", "mean"),
        spearman=("spearman", "mean"),
        aurc=("aurc", "mean"),
        risk_at_10=("risk_at_10", "mean"),
        risk_at_20=("risk_at_20", "mean"),
        risk_at_50=("risk_at_50", "mean"),
        high_risk_miss_rate=("high_risk_miss_rate", "mean"),
    )
    boots = bootstrap(predictions)
    OUT.mkdir(parents=True)
    predictions.to_csv(OUT / "GATE_MECHANISM_TASK_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    folds.to_csv(OUT / "GATE_ORACLE_ROUTING_BY_FOLD.csv", index=False)
    fold_metrics.to_csv(OUT / "GATE_METHODS_BY_FOLD.csv", index=False)
    summary.to_csv(OUT / "GATE_METHODS_SUMMARY.csv", index=False)
    boots.to_csv(OUT / "GATE_METHOD_BOOTSTRAP.csv", index=False)
    (OUT / "REPORT.md").write_text(make_report(summary, folds, boots))
    status = {
        "status": "COMPLETE",
        "data_role": "DEV_SEEN_ONLY",
        "external_test_truth_opened": False,
        "frozen_v4_changed": False,
        "fixed_weights": FIXED_WEIGHTS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(boots.to_string(index=False))


if __name__ == "__main__":
    main()
