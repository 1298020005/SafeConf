#!/usr/bin/env python3
"""Run the frozen SafeConf-v4 development protocol on released TxPert assets.

This script only reads DEV/SEEN E201/E205 assets. It never opens a sealed
confirmation asset. V1 is Universal-P + Support + Relevance. Quality is absent
in these assets; source dispersion is treated only as a conflict proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928"
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
DATA = Path("/home/yyf/data/txpert_official_20260802/e201")
SUPPORT_PATH = E201 / "tables/E201_SOURCE_CONTEXT_SUPPORT.csv"
TRUTH_PATH = DATA / "evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy"
SEED = 20260929
N_BOOTSTRAP = 5000
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")

UNIVERSAL_P = [
    "predicted_magnitude",
    "prediction_abs_mean",
    "prediction_signed_mean",
    "prediction_std",
    "prediction_abs_q95",
    "prediction_sparsity",
]
SUPPORT = ["n_source_cells", "n_source_contexts", "n_source_batches", "min_source_cells"]
RELEVANCE = ["prediction_source_cosine", "negative_model_source_gap"]
CONFLICT = ["source_delta_dispersion", "conflict_missing"]
CONTENT = ["source_transfer_magnitude"]
GROUPS = {
    "U": UNIVERSAL_P,
    "US": UNIVERSAL_P + SUPPORT,
    "USR": UNIVERSAL_P + SUPPORT + RELEVANCE,
    "USRC": UNIVERSAL_P + SUPPORT + RELEVANCE + CONFLICT,
    "USRCH": UNIVERSAL_P + SUPPORT + RELEVANCE + CONFLICT + CONTENT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure(upstream: str) -> dict:
    if upstream == "e201":
        return {
            "label": "TxPert STRING-GAT",
            "features": E201 / "tables/E201_PRETRUTH_RISK_FEATURES.csv",
            "metrics": E201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv",
            "vectors": DATA / "pretruth_vectors",
            "prefix": "E201",
            "out": STAGE / "safeconf_v4_development" / "txpert_gat",
        }
    if upstream == "e205":
        e205 = ROOT / "docs/实验结果/E205_cross_family_disagreement_20260830"
        return {
            "label": "TxPert Exphormer",
            "features": e205 / "tables/E205_PRETRUTH_RISK_FEATURES.csv",
            "metrics": e205 / "formal_evaluation/E205_TASK_METRICS.csv",
            "vectors": Path("/home/yyf/data/txpert_official_20260802/e205/pretruth_vectors"),
            "prefix": "E205",
            "out": STAGE / "safeconf_v4_development" / "txpert_exphormer",
        }
    raise ValueError(upstream)


def utility20(task_ids: np.ndarray, score: np.ndarray, outcome: np.ndarray) -> float:
    n = len(outcome)
    if n < 20 or not np.isfinite(score).all() or not np.isfinite(outcome).all():
        return float("nan")
    k = int(math.ceil(0.2 * n))
    ids = np.asarray(task_ids, dtype=str)
    chosen = np.lexsort((ids, -score))[:k]
    oracle = np.lexsort((ids, -outcome))[:k]
    denominator = float(outcome[oracle].mean() - outcome.mean())
    if denominator <= 1e-12:
        return float("nan")
    return float((outcome[chosen].mean() - outcome.mean()) / denominator)


def rho(score: np.ndarray, outcome: np.ndarray) -> float:
    if len(score) < 3 or np.ptp(score) <= 1e-15 or np.ptp(outcome) <= 1e-15:
        return float("nan")
    return float(spearmanr(score, outcome).statistic)


def selective_metrics(task_ids: np.ndarray, score: np.ndarray, outcome: np.ndarray) -> dict:
    ids = np.asarray(task_ids, dtype=str)
    order = np.lexsort((ids, score))  # lowest risk accepted first
    n = len(outcome)
    values = {}
    curve = []
    for k in range(1, n + 1):
        curve.append(float(outcome[order[:k]].mean()))
    values["aurc"] = float(np.mean(curve))
    for coverage in (0.1, 0.2, 0.5):
        k = max(1, int(math.ceil(coverage * n)))
        values[f"risk_at_{int(coverage*100)}"] = float(outcome[order[:k]].mean())
    k = int(math.ceil(0.2 * n))
    predicted_high = set(np.lexsort((ids, -score))[:k])
    actual_high = set(np.lexsort((ids, -outcome))[:k])
    values["high_risk_miss_rate"] = float(len(actual_high - predicted_high) / max(len(actual_high), 1))
    return values


def prepare(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]):
    x = fit[columns].to_numpy(float)
    z = query[columns].to_numpy(float)
    mx = ~np.isfinite(x)
    mz = ~np.isfinite(z)
    med = np.array([np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0 for j in range(x.shape[1])])
    x = np.where(mx, med, x)
    z = np.where(mz, med, z)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return np.c_[(x - mean) / scale, mx], np.c_[(z - mean) / scale, mz]


def fit_model(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str], kind: str, seed: int) -> np.ndarray:
    x, z = prepare(fit, query, columns)
    y = fit.true_error_rmse.to_numpy(float)
    center, scale = float(y.mean()), max(float(y.std()), 1e-8)
    target = (y - center) / scale
    if kind == "Ridge":
        model = Ridge(alpha=10.0).fit(x, target)
    elif kind == "HGB":
        model = HistGradientBoostingRegressor(
            max_iter=80, max_depth=3, max_leaf_nodes=7, min_samples_leaf=5,
            learning_rate=0.05, l2_regularization=1.0, early_stopping=False,
            random_state=seed,
        ).fit(x, target)
    elif kind == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8), activation="relu", alpha=0.01,
            learning_rate_init=0.005, max_iter=300, early_stopping=False,
            random_state=seed,
        ).fit(x, target)
    else:
        raise ValueError(kind)
    return np.maximum(0.0, np.asarray(model.predict(z), float) * scale + center)


def build_frame(config: dict) -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(config["features"])
    observed = pd.read_csv(config["metrics"])
    support = (
        pd.read_csv(SUPPORT_PATH)
        .groupby(["target", "condition"], as_index=False)
        .agg(
            support_n_source_cells=("n_source_perturbed_cells", "sum"),
            n_source_contexts_check=("source_context", "nunique"),
            n_source_batches=("n_source_batches", "sum"),
            min_source_cells=("n_source_perturbed_cells", "min"),
        )
    )
    frame = base.merge(support, on=["target", "condition"], how="left", validate="one_to_one")
    frame = frame.merge(
        observed[["task_id", "family_centroid_rmse"]], on="task_id", how="left", validate="one_to_one"
    )
    frame["true_error_rmse"] = frame["family_centroid_rmse"]
    frame = frame[frame.analysis_stratum.eq("primary_ge30")].copy().reset_index(drop=True)
    assert len(frame) == 1808
    assert (frame.n_source_cells == frame.support_n_source_cells).all()
    assert (frame.n_source_contexts == frame.n_source_contexts_check).all()

    vectors = config["vectors"]
    prefix = config["prefix"]
    family_all = np.load(vectors / f"{prefix}_FAMILY_CENTROIDS.npy", mmap_mode="r")
    control_all = np.load(vectors / f"{prefix}_CONTROL_CENTROIDS.npy", mmap_mode="r")
    source_all = np.load(vectors / f"{prefix}_SOURCE_TRANSFER_CENTROIDS.npy", mmap_mode="r")
    truth_all = np.load(TRUTH_PATH, mmap_mode="r")
    original = base.index[base.analysis_stratum.eq("primary_ge30")].to_numpy()
    family = np.asarray(family_all[original], np.float64)
    control = np.asarray(control_all[original], np.float64)
    source = np.asarray(source_all[original], np.float64)
    truth = np.asarray(truth_all[original], np.float64)
    effect = family - control
    source_effect = source - control
    recomputed_error = np.sqrt(np.mean((family - truth) ** 2, axis=1))
    recomputed_magnitude = np.sqrt(np.mean(effect**2, axis=1))
    assert float(np.max(np.abs(recomputed_error - frame.family_centroid_rmse))) < 1e-6
    assert float(np.max(np.abs(recomputed_magnitude - frame.predicted_magnitude))) < 1e-6

    abs_effect = np.abs(effect)
    frame["prediction_abs_mean"] = abs_effect.mean(axis=1)
    frame["prediction_signed_mean"] = effect.mean(axis=1)
    frame["prediction_std"] = effect.std(axis=1)
    frame["prediction_abs_q95"] = np.quantile(abs_effect, 0.95, axis=1)
    # Relative near-zero shape statistic is comparable within an output contract.
    row_scale = np.maximum(np.quantile(abs_effect, 0.95, axis=1), 1e-12)
    frame["prediction_sparsity"] = np.mean(abs_effect <= row_scale[:, None] * 0.01, axis=1)
    numerator = np.sum(effect * source_effect, axis=1)
    denominator = np.linalg.norm(effect, axis=1) * np.linalg.norm(source_effect, axis=1)
    frame["prediction_source_cosine"] = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    frame["negative_model_source_gap"] = -frame.model_source_gap.astype(float)
    frame["conflict_missing"] = frame.source_delta_dispersion_observed.isna().astype(float)
    assert np.isfinite(frame[sum(GROUPS.values(), [])].to_numpy(float)).all()
    return frame, {
        "n_tasks": len(frame),
        "n_targets": int(frame.target.nunique()),
        "error_max_absdiff": float(np.max(np.abs(recomputed_error - frame.family_centroid_rmse))),
        "magnitude_max_absdiff": float(np.max(np.abs(recomputed_magnitude - frame.predicted_magnitude))),
        "quality_fields_available": False,
        "conflict_proxy_coverage": float(1.0 - frame.conflict_missing.mean()),
        "relevance_fields": RELEVANCE,
    }


def evidence_matrix(fit: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def score(cols: list[str], signs: list[float]) -> tuple[np.ndarray, np.ndarray]:
        x = fit[cols].to_numpy(float) * np.asarray(signs)[None, :]
        z = query[cols].to_numpy(float) * np.asarray(signs)[None, :]
        med = np.array([np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0.0 for j in range(x.shape[1])])
        x = np.where(np.isfinite(x), x, med)
        z = np.where(np.isfinite(z), z, med)
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        return ((x - mean) / scale).mean(axis=1), ((z - mean) / scale).mean(axis=1)

    sx, sz = score(SUPPORT, [1, 1, 1, 1])
    rx, rz = score(RELEVANCE, [1, 1])
    cx, cz = score(["source_delta_dispersion"], [1])
    mx = fit.conflict_missing.to_numpy(float)
    mz = query.conflict_missing.to_numpy(float)
    return np.c_[sx, rx, cx, mx], np.c_[sz, rz, cz, mz]


def nested_v2(fit: pd.DataFrame, query: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    n = len(fit)
    p_oof = np.full(n, np.nan)
    h_oof = np.full(n, np.nan)
    splitter = GroupKFold(4)
    for inner, (tr, va) in enumerate(splitter.split(fit, groups=fit.gene)):
        inner_fit = fit.iloc[tr]
        inner_query = fit.iloc[va]
        p_oof[va] = fit_model(inner_fit, inner_query, GROUPS["U"], "Ridge", seed + inner)
        h_oof[va] = fit_model(inner_fit, inner_query, GROUPS["USRCH"], "Ridge", seed + 20 + inner)
    assert np.isfinite(p_oof).all() and np.isfinite(h_oof).all()
    y = fit.true_error_rmse.to_numpy(float)
    p_cal = LinearRegression().fit(p_oof[:, None], y)
    h_cal = LinearRegression().fit(h_oof[:, None], y)
    pc_oof = np.maximum(0.0, p_cal.predict(p_oof[:, None]))
    hc_oof = np.maximum(0.0, h_cal.predict(h_oof[:, None]))
    ev_train, ev_query = evidence_matrix(fit, query)

    def objective(theta: np.ndarray) -> float:
        logits = theta[0] + ev_train @ theta[1:]
        weight = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        pred = pc_oof + weight * (hc_oof - pc_oof)
        scale = max(float(y.std()), 1e-8)
        return float(np.mean(((pred - y) / scale) ** 2) + 1e-4 * np.sum(theta[1:] ** 2))

    bounds = [(-10, 10), (0, 10), (0, 10), (-10, 0), (-10, 0)]
    fit_gate = minimize(objective, np.zeros(5), method="L-BFGS-B", bounds=bounds)
    if not fit_gate.success:
        raise RuntimeError(f"gate optimization failed: {fit_gate.message}")
    p_test = fit_model(fit, query, GROUPS["U"], "Ridge", seed + 100)
    h_test = fit_model(fit, query, GROUPS["USRCH"], "Ridge", seed + 200)
    pc_test = np.maximum(0.0, p_cal.predict(p_test[:, None]))
    hc_test = np.maximum(0.0, h_cal.predict(h_test[:, None]))
    logits = fit_gate.x[0] + ev_query @ fit_gate.x[1:]
    weight = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    result = pc_test + weight * (hc_test - pc_test)
    return result, {
        "gate_intercept": float(fit_gate.x[0]),
        "coef_support": float(fit_gate.x[1]),
        "coef_relevance": float(fit_gate.x[2]),
        "coef_conflict": float(fit_gate.x[3]),
        "coef_missingness": float(fit_gate.x[4]),
        "mean_weight": float(weight.mean()),
        "min_weight": float(weight.min()),
        "max_weight": float(weight.max()),
        "inner_objective": float(fit_gate.fun),
    }


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    gate_rows = []
    for target, part in frame.groupby("target", sort=True):
        part = part.sort_values(["gene", "task_id"]).reset_index(drop=True)
        for row in part.itertuples():
            prediction_rows.append({
                "task_id": row.task_id, "target": target, "gene": row.gene,
                "fold": -1, "method": "Magnitude_raw",
                "true_error_rmse": row.family_centroid_rmse,
                "predicted_risk": row.predicted_magnitude,
            })
        split = GroupKFold(5)
        for fold, (tr, va) in enumerate(split.split(part, groups=part.gene)):
            fit = part.iloc[tr].copy()
            query = part.iloc[va].copy()
            assert not set(fit.gene) & set(query.gene)
            methods = {}
            for group in ("U", "US", "USR", "USRC", "USRCH"):
                methods[f"Ridge_{group}"] = fit_model(fit, query, GROUPS[group], "Ridge", SEED + fold)
            for kind in ("HGB", "MLP"):
                methods[f"{kind}_USR"] = fit_model(fit, query, GROUPS["USR"], kind, SEED + fold)
                methods[f"{kind}_USRCH"] = fit_model(fit, query, GROUPS["USRCH"], kind, SEED + 20 + fold)
            v2, gate = nested_v2(fit, query, SEED + 1000 * (fold + 1))
            methods["V2_nested"] = v2
            gate_rows.append({"target": target, "fold": fold, **gate})

            # Deterministic history permutation checks whether result survives
            # when support/relevance is detached from its biological task.
            shuffled_fit = fit.copy()
            shuffled_query = query.copy()
            rng = np.random.default_rng(SEED + 10000 + fold)
            for data in (shuffled_fit, shuffled_query):
                perm = rng.permutation(len(data))
                data.loc[:, SUPPORT + RELEVANCE] = data[SUPPORT + RELEVANCE].to_numpy()[perm]
            methods["Ridge_USR_history_shuffled"] = fit_model(
                shuffled_fit, shuffled_query, GROUPS["USR"], "Ridge", SEED + fold
            )
            for method, scores in methods.items():
                for row, score in zip(query.itertuples(), scores):
                    prediction_rows.append({
                        "task_id": row.task_id, "target": target, "gene": row.gene,
                        "fold": fold, "method": method,
                        "true_error_rmse": row.family_centroid_rmse,
                        "predicted_risk": float(score),
                    })
        print(f"[SafeConf-v4] {target} complete", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    return predictions, pd.DataFrame(gate_rows)


def score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strata = []
    coverage = []
    for (target, method), group in predictions.groupby(["target", "method"], sort=True):
        ids = group.task_id.to_numpy(str)
        risks = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        select = selective_metrics(ids, risks, truth)
        strata.append({
            "target": target, "method": method, "n_tasks": len(group),
            "utility20": utility20(ids, risks, truth), "spearman": rho(risks, truth), **select,
        })
        for fraction in (0.1, 0.2, 0.5, 1.0):
            order = np.lexsort((ids, risks))
            k = max(1, int(math.ceil(fraction * len(group))))
            coverage.append({
                "target": target, "method": method, "coverage": fraction,
                "n_accepted": k, "mean_true_rmse": float(truth[order[:k]].mean()),
            })
    strata_df = pd.DataFrame(strata)
    summary = strata_df.groupby("method", as_index=False).agg(
        n_strata=("target", "nunique"), utility20=("utility20", "mean"),
        spearman=("spearman", "mean"), aurc=("aurc", "mean"),
        risk_at_10=("risk_at_10", "mean"), risk_at_20=("risk_at_20", "mean"),
        risk_at_50=("risk_at_50", "mean"), high_risk_miss_rate=("high_risk_miss_rate", "mean"),
    ).sort_values(["utility20", "spearman"], ascending=False)
    return strata_df, summary, pd.DataFrame(coverage)


def bootstrap(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("V1_vs_magnitude", "Ridge_USR", "Magnitude_raw"),
        ("Relevance_given_support", "Ridge_USR", "Ridge_US"),
        ("Conflict_given_relevance", "Ridge_USRC", "Ridge_USR"),
        ("Content_given_conflict", "Ridge_USRCH", "Ridge_USRC"),
        ("V2_vs_V1", "V2_nested", "Ridge_USR"),
        ("V1_vs_shuffled_history", "Ridge_USR", "Ridge_USR_history_shuffled"),
    ]
    wide = predictions.pivot(
        index=["task_id", "target", "gene", "true_error_rmse"], columns="method", values="predicted_risk"
    ).reset_index()
    assert len(wide) == 1808 and not wide.isna().any().any()
    genes = np.asarray(sorted(wide.gene.unique()))
    gene_rows = [np.flatnonzero(wide.gene.to_numpy() == gene) for gene in genes]
    targets = wide.target.to_numpy()
    ids = wide.task_id.to_numpy(str)
    truth = wide.true_error_rmse.to_numpy(float)
    risks = {name: wide[name].to_numpy(float) for _, a, b in comparisons for name in (a, b)}
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name, _, _ in comparisons}
    rho_draws = {name: [] for name, _, _ in comparisons}
    for _ in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(genes), len(genes))
        rows = np.concatenate([gene_rows[i] for i in selected])
        values = {}
        for method in risks:
            us, rs = [], []
            for target in TARGETS:
                use = rows[targets[rows] == target]
                us.append(utility20(ids[use], risks[method][use], truth[use]))
                rs.append(rho(risks[method][use], truth[use]))
            values[method] = (float(np.nanmean(us)), float(np.nanmean(rs)))
        for name, a, b in comparisons:
            draws[name].append(values[a][0] - values[b][0])
            rho_draws[name].append(values[a][1] - values[b][1])
    point = summary.set_index("method")
    rows = []
    for name, a, b in comparisons:
        u = np.asarray(draws[name])
        r = np.asarray(rho_draws[name])
        rows.append({
            "comparison": name, "method_a": a, "method_b": b,
            "delta_utility20": float(point.loc[a, "utility20"] - point.loc[b, "utility20"]),
            "utility_ci95_lower": float(np.nanquantile(u, 0.025)),
            "utility_ci95_upper": float(np.nanquantile(u, 0.975)),
            "delta_spearman": float(point.loc[a, "spearman"] - point.loc[b, "spearman"]),
            "spearman_ci95_lower": float(np.nanquantile(r, 0.025)),
            "spearman_ci95_upper": float(np.nanquantile(r, 0.975)),
            "bootstrap_replicates": N_BOOTSTRAP,
        })
    return pd.DataFrame(rows)


def development_gate(strata: pd.DataFrame, summary: pd.DataFrame) -> dict:
    s = summary.set_index("method")
    st = strata.pivot(index="target", columns="method", values="utility20")
    delta = float(s.loc["V2_nested", "utility20"] - s.loc["Ridge_USR", "utility20"])
    nonnegative = float((st["V2_nested"] - st["Ridge_USR"] >= 0).mean())
    def relative_degradation(column: str) -> float:
        baseline = float(s.loc["Ridge_USR", column])
        candidate = float(s.loc["V2_nested", column])
        return max(0.0, (candidate - baseline) / max(abs(baseline), 1e-12))
    miss = max(0.0, float(s.loc["V2_nested", "high_risk_miss_rate"] - s.loc["Ridge_USR", "high_risk_miss_rate"]))
    result = {
        "delta_u20_macro": delta,
        "nonnegative_strata_fraction": nonnegative,
        "risk10_relative_degradation": relative_degradation("risk_at_10"),
        "risk20_relative_degradation": relative_degradation("risk_at_20"),
        "risk50_relative_degradation": relative_degradation("risk_at_50"),
        "high_risk_miss_rate_degradation": miss,
        "aurc_relative_degradation": relative_degradation("aurc"),
        "valid_strata_fraction": float(st[["V2_nested", "Ridge_USR"]].notna().all(axis=1).mean()),
    }
    result["v2_passes_development_gate"] = bool(
        result["delta_u20_macro"] >= 0.005
        and result["nonnegative_strata_fraction"] >= 0.60
        and max(result["risk10_relative_degradation"], result["risk20_relative_degradation"], result["risk50_relative_degradation"]) <= 0.05
        and result["high_risk_miss_rate_degradation"] <= 0.02
        and result["aurc_relative_degradation"] <= 0.05
        and result["valid_strata_fraction"] >= 0.80
    )
    return result


def matched_support(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    v1 = predictions[predictions.method.eq("Ridge_US")][["task_id", "predicted_risk"]]
    work = frame.merge(v1, on="task_id", validate="one_to_one")
    work["residual_error"] = work.true_error_rmse - work.predicted_risk
    rows = []
    for target, part in work.groupby("target"):
        support_score = np.log1p(part.n_source_cells) + part.n_source_contexts
        part = part.assign(support_bin=pd.qcut(support_score.rank(method="first"), 5, labels=False))
        for support_bin, group in part.groupby("support_bin"):
            relevance = (group.prediction_source_cosine.rank(pct=True) + group.negative_model_source_gap.rank(pct=True)) / 2
            rows.append({
                "target": target, "support_bin": int(support_bin), "n_tasks": len(group),
                "spearman_relevance_vs_residual_error": rho(relevance.to_numpy(float), group.residual_error.to_numpy(float)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", choices=["e201", "e205"], required=True)
    args = parser.parse_args()
    config = configure(args.upstream)
    out: Path = config["out"]
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "RUN_STATUS.json"
    if status_path.exists():
        raise FileExistsError(f"refusing to overwrite {status_path}")
    started = time.time()
    status = {
        "status": "RUNNING", "upstream": args.upstream, "label": config["label"],
        "seed": SEED, "bootstrap_replicates": N_BOOTSTRAP,
        "data_role": "SEEN", "sealed_confirmation_opened": False,
        "quality_fields_available": False,
        "history_layers": ["Support", "Relevance", "ConflictProxy", "Content"],
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    frame, audit = build_frame(config)
    predictions, gates = evaluate(frame)
    strata, summary, coverage = score(predictions)
    increments = bootstrap(predictions, summary)
    matched = matched_support(frame, predictions)
    gate_result = development_gate(strata, summary)

    frame.to_csv(out / "FEATURE_TABLE.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    predictions.to_csv(out / "OOF_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    gates.to_csv(out / "V2_GATE_FITS.csv", index=False)
    strata.to_csv(out / "STRATUM_RESULTS.csv", index=False)
    summary.to_csv(out / "SUMMARY.csv", index=False)
    coverage.to_csv(out / "RISK_COVERAGE.csv", index=False)
    increments.to_csv(out / "INCREMENTAL_RESULTS.csv", index=False)
    matched.to_csv(out / "MATCHED_SUPPORT.csv", index=False)
    (out / "DEVELOPMENT_GATE.json").write_text(json.dumps(gate_result, indent=2) + "\n")
    status.update({
        "status": "COMPLETE", "elapsed_seconds": time.time() - started,
        "audit": audit, "n_methods": int(predictions.method.nunique()),
        "n_oof_rows": int(len(predictions)), "development_gate": gate_result,
        "summary_sha256": sha256(out / "SUMMARY.csv"),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(increments.to_string(index=False), flush=True)
    print(json.dumps(gate_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
