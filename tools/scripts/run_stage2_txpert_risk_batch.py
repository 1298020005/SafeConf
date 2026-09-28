#!/usr/bin/env python3
"""Run the preregistered TxPert magnitude/public-history risk comparison."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
OUT = ROOT / "docs/实验结果/Stage2_mature_upstream_20260928/txpert_risk_batch"
DATA = Path("/home/yyf/data/txpert_official_20260802/e201")
CONTRACT = OUT.parent / "TXPERT_RISK_BATCH_CONTRACT.md"
FEATURES = E201 / "tables/E201_PRETRUTH_RISK_FEATURES.csv"
SUPPORT = E201 / "tables/E201_SOURCE_CONTEXT_SUPPORT.csv"
METRICS = E201 / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
VECTORS = DATA / "pretruth_vectors"
TRUTH = DATA / "evaluation_vectors/E201_TARGET_TRUTH_CENTROIDS.npy"

SEED = 20260928
N_BOOTSTRAP = 2000
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
P = [
    "predicted_magnitude",
    "family_disagreement",
    "family_radius",
    "prediction_abs_mean",
    "prediction_signed_mean",
    "prediction_std",
    "prediction_abs_q95",
]
Q = [
    "n_source_cells",
    "n_source_contexts",
    "n_source_batches",
    "min_source_cells",
]
H = [
    "source_transfer_magnitude",
    "source_delta_dispersion",
    "model_source_gap",
    "prediction_source_cosine",
]
GROUPS = {"M": P[:1], "P": P, "PQ": P + Q, "PQH": P + Q + H}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utility20(score: np.ndarray, outcome: np.ndarray) -> float:
    score = np.asarray(score, float)
    outcome = np.asarray(outcome, float)
    if len(score) < 5 or not np.isfinite(score).all() or not np.isfinite(outcome).all():
        return float("nan")
    n_select = int(math.ceil(0.2 * len(score)))
    order = np.lexsort((np.arange(len(score)), -score))[:n_select]
    oracle = np.lexsort((np.arange(len(score)), -outcome))[:n_select]
    selected_mean = float(outcome[order].mean())
    overall_mean = float(outcome.mean())
    oracle_mean = float(outcome[oracle].mean())
    denominator = oracle_mean - overall_mean
    return (selected_mean - overall_mean) / denominator if denominator > 1e-15 else float("nan")


def spearman(score: np.ndarray, outcome: np.ndarray) -> float:
    if np.ptp(score) <= 0 or np.ptp(outcome) <= 0:
        return float("nan")
    return float(spearmanr(score, outcome).statistic)


def make_features(fit: pd.DataFrame, query: pd.DataFrame, columns: list[str]):
    x = fit[columns].to_numpy(float)
    z = query[columns].to_numpy(float)
    missing_x = ~np.isfinite(x)
    missing_z = ~np.isfinite(z)
    medians = np.asarray(
        [np.median(x[~missing_x[:, j], j]) if (~missing_x[:, j]).any() else 0.0 for j in range(x.shape[1])]
    )
    x = np.where(missing_x, medians, x)
    z = np.where(missing_z, medians, z)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    x = np.c_[(x - mean) / std, missing_x].astype("float32")
    z = np.c_[(z - mean) / std, missing_z].astype("float32")
    return x, z


def learn(x: np.ndarray, z: np.ndarray, y: np.ndarray, kind: str, device: str):
    center = float(y.mean())
    scale = max(float(y.std()), 1e-8)
    yy = ((y - center) / scale).astype("float32")
    if kind == "Ridge":
        model = Ridge(alpha=10.0).fit(x, yy)
        pred = model.predict(z)
        parameters = x.shape[1] + 1
    elif kind == "HGB":
        model = HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=3,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED,
        ).fit(x, yy)
        pred = model.predict(z)
        parameters = None
    elif kind == "MLP":
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = nn.Sequential(
            nn.Linear(x.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        ).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        tx = torch.as_tensor(x, device=device)
        tz = torch.as_tensor(z, device=device)
        ty = torch.as_tensor(yy, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
        model.train()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx).squeeze(-1) - ty) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MLP loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(tz).squeeze(-1).cpu().numpy()
    else:
        raise ValueError(kind)
    return np.maximum(0.0, np.asarray(pred, float) * scale + center), parameters


def build_frame() -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(FEATURES)
    observed = pd.read_csv(METRICS)
    support = (
        pd.read_csv(SUPPORT)
        .groupby(["target", "condition"], as_index=False)
        .agg(
            support_n_source_cells=("n_source_perturbed_cells", "sum"),
            support_n_source_contexts=("source_context", "nunique"),
            n_source_batches=("n_source_batches", "sum"),
            min_source_cells=("n_source_perturbed_cells", "min"),
        )
    )
    frame = base.merge(
        support, on=["target", "condition"], how="left", validate="one_to_one"
    ).merge(
        observed[["task_id", "family_centroid_rmse"]],
        on="task_id",
        how="left",
        validate="one_to_one",
    )
    assert len(frame) == 2008 and frame.task_id.nunique() == 2008
    assert (frame.n_source_cells == frame.support_n_source_cells).all()
    assert (frame.n_source_contexts == frame.support_n_source_contexts).all()
    frame = frame[frame.analysis_stratum.eq("primary_ge30")].copy().reset_index(drop=True)
    assert len(frame) == 1808 and tuple(sorted(frame.target.unique())) == tuple(sorted(TARGETS))

    seed = np.load(VECTORS / "E201_SEED_CENTROIDS.npy", mmap_mode="r")[:, :]
    family = np.load(VECTORS / "E201_FAMILY_CENTROIDS.npy", mmap_mode="r")
    control = np.load(VECTORS / "E201_CONTROL_CENTROIDS.npy", mmap_mode="r")
    source = np.load(VECTORS / "E201_SOURCE_TRANSFER_CENTROIDS.npy", mmap_mode="r")
    truth = np.load(TRUTH, mmap_mode="r")
    original = base.index[base.analysis_stratum.eq("primary_ge30")].to_numpy()
    family = np.asarray(family[original], dtype=np.float64)
    control = np.asarray(control[original], dtype=np.float64)
    source = np.asarray(source[original], dtype=np.float64)
    truth = np.asarray(truth[original], dtype=np.float64)
    seed = np.asarray(seed[:, original], dtype=np.float64)
    effect = family - control
    source_effect = source - control
    recomputed_magnitude = np.sqrt(np.mean(effect**2, axis=1))
    recomputed_error = np.sqrt(np.mean((family - truth) ** 2, axis=1))
    recomputed_disagreement = np.sqrt(np.mean((seed - family[None, :]) ** 2, axis=(0, 2)))
    magnitude_diff = float(np.max(np.abs(recomputed_magnitude - frame.predicted_magnitude)))
    error_diff = float(np.max(np.abs(recomputed_error - frame.family_centroid_rmse)))
    disagreement_diff = float(np.max(np.abs(recomputed_disagreement - frame.family_disagreement)))
    assert magnitude_diff < 1e-6 and error_diff < 1e-6 and disagreement_diff < 1e-6
    frame["prediction_abs_mean"] = np.mean(np.abs(effect), axis=1)
    frame["prediction_signed_mean"] = np.mean(effect, axis=1)
    frame["prediction_std"] = np.std(effect, axis=1)
    frame["prediction_abs_q95"] = np.quantile(np.abs(effect), 0.95, axis=1)
    numerator = np.sum(effect * source_effect, axis=1)
    denominator = np.linalg.norm(effect, axis=1) * np.linalg.norm(source_effect, axis=1)
    frame["prediction_source_cosine"] = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    assert np.isfinite(frame[P + Q + H].to_numpy(float)).all()
    del seed, family, control, source, truth, effect, source_effect
    return frame, {
        "magnitude_max_absdiff": magnitude_diff,
        "error_max_absdiff": error_diff,
        "disagreement_max_absdiff": disagreement_diff,
    }


def evaluate(frame: pd.DataFrame, device: str):
    predictions = []
    folds = []
    fit_count = 0
    parameter_counts = {}
    for target, target_frame in frame.groupby("target", sort=True):
        target_frame = target_frame.sort_values(["gene", "task_id"]).reset_index(drop=True)
        predictions.extend(
            dict(
                task_id=r.task_id,
                target=target,
                gene=r.gene,
                fold=-1,
                method="Magnitude_raw",
                true_error_rmse=r.family_centroid_rmse,
                predicted_risk=r.predicted_magnitude,
            )
            for r in target_frame.itertuples()
        )
        splitter = GroupKFold(5)
        for fold, (fit_index, eval_index) in enumerate(
            splitter.split(target_frame, groups=target_frame.gene)
        ):
            fit = target_frame.iloc[fit_index].reset_index(drop=True)
            query = target_frame.iloc[eval_index].reset_index(drop=True)
            assert not set(fit.gene) & set(query.gene)
            y = fit.family_centroid_rmse.to_numpy(float)
            for r in fit.itertuples():
                folds.append(dict(target=target, fold=fold, role="fit", task_id=r.task_id, gene=r.gene))
            for r in query.itertuples():
                folds.append(dict(target=target, fold=fold, role="eval", task_id=r.task_id, gene=r.gene))
            for group, columns in GROUPS.items():
                x, z = make_features(fit, query, columns)
                for kind in ("Ridge", "HGB", "MLP"):
                    risk, parameters = learn(x, z, y, kind, device)
                    method = f"{kind}_{group}"
                    parameter_counts.setdefault(method, parameters)
                    fit_count += 1
                    for r, value in zip(query.itertuples(), risk):
                        predictions.append(
                            dict(
                                task_id=r.task_id,
                                target=target,
                                gene=r.gene,
                                fold=fold,
                                method=method,
                                true_error_rmse=r.family_centroid_rmse,
                                predicted_risk=float(value),
                            )
                        )
        print(f"[TxPert risk] {target} complete; fits={fit_count}", flush=True)
    pred = pd.DataFrame(predictions)
    assert fit_count == 240
    counts = pred.groupby(["target", "method"]).size()
    expected = frame.groupby("target").size()
    for (target, _), count in counts.items():
        assert count == expected[target]
    assert pred.groupby(["target", "method"]).task_id.nunique().eq(counts).all()
    return pred, pd.DataFrame(folds), fit_count, parameter_counts


def score_predictions(pred: pd.DataFrame):
    target_rows = []
    for (target, method), group in pred.groupby(["target", "method"], sort=True):
        score = group.predicted_risk.to_numpy(float)
        truth = group.true_error_rmse.to_numpy(float)
        target_rows.append(
            dict(
                target=target,
                method=method,
                n_tasks=len(group),
                utility20=utility20(score, truth),
                spearman=spearman(score, truth),
                risk_rmse=float(np.sqrt(np.mean((score - truth) ** 2))) if method != "Magnitude_raw" else np.nan,
            )
        )
    target = pd.DataFrame(target_rows)
    summary = (
        target.groupby("method", as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            utility20=("utility20", "mean"),
            spearman=("spearman", "mean"),
            risk_rmse=("risk_rmse", "mean"),
        )
        .sort_values(["utility20", "spearman"], ascending=False)
    )
    return target, summary


def macro_metrics(sample: pd.DataFrame, method: str):
    values_u, values_rho = [], []
    for target in TARGETS:
        group = sample[(sample.target == target) & (sample.method == method)]
        if len(group) < 5:
            return float("nan"), float("nan")
        values_u.append(utility20(group.predicted_risk.to_numpy(), group.true_error_rmse.to_numpy()))
        values_rho.append(spearman(group.predicted_risk.to_numpy(), group.true_error_rmse.to_numpy()))
    return float(np.nanmean(values_u)), float(np.nanmean(values_rho))


def bootstrap_increments(pred: pd.DataFrame, summary: pd.DataFrame):
    methods = set(summary.method)
    comparisons = []
    for kind in ("Ridge", "HGB", "MLP"):
        comparisons += [
            (kind, "P_given_M", f"{kind}_P", f"{kind}_M"),
            (kind, "Q_given_P", f"{kind}_PQ", f"{kind}_P"),
            (kind, "H_given_PQ", f"{kind}_PQH", f"{kind}_PQ"),
            (kind, "final_vs_raw_magnitude", f"{kind}_PQH", "Magnitude_raw"),
        ]
    assert all(a in methods and b in methods for _, _, a, b in comparisons)
    point = summary.set_index("method")
    genes = sorted(pred.gene.unique())
    by_gene = {gene: pred.index[pred.gene.eq(gene)].to_numpy() for gene in genes}
    rng = np.random.default_rng(SEED)
    draws = {item[1] + "::" + item[0]: [] for item in comparisons}
    rho_draws = {key: [] for key in draws}
    for _ in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(genes), len(genes))
        pieces = []
        for occurrence, index in enumerate(chosen):
            block = pred.loc[by_gene[genes[index]]].copy()
            block["bootstrap_occurrence"] = occurrence
            pieces.append(block)
        sample = pd.concat(pieces, ignore_index=True)
        needed = sorted({method for _, _, a, b in comparisons for method in (a, b)})
        stats = {method: macro_metrics(sample, method) for method in needed}
        for kind, name, a, b in comparisons:
            key = name + "::" + kind
            draws[key].append(stats[a][0] - stats[b][0])
            rho_draws[key].append(stats[a][1] - stats[b][1])
    rows = []
    for kind, name, a, b in comparisons:
        key = name + "::" + kind
        u = np.asarray(draws[key], float)
        rho = np.asarray(rho_draws[key], float)
        by_target = []
        target_table, _ = score_predictions(pred[pred.method.isin([a, b])])
        wide = target_table.pivot(index="target", columns="method", values="utility20")
        by_target = (wide[a] - wide[b]).reindex(TARGETS)
        rows.append(
            dict(
                algorithm=kind,
                comparison=name,
                method_a=a,
                method_b=b,
                delta_utility20=float(point.loc[a, "utility20"] - point.loc[b, "utility20"]),
                utility_ci95_lower=float(np.nanquantile(u, 0.025)),
                utility_ci95_upper=float(np.nanquantile(u, 0.975)),
                delta_spearman=float(point.loc[a, "spearman"] - point.loc[b, "spearman"]),
                spearman_ci95_lower=float(np.nanquantile(rho, 0.025)),
                spearman_ci95_upper=float(np.nanquantile(rho, 0.975)),
                positive_targets=int((by_target > 0).sum()),
                target_deltas=";".join(f"{t}:{by_target[t]:.6f}" for t in TARGETS),
                bootstrap_draws=N_BOOTSTRAP,
            )
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, increments: pd.DataFrame):
    order = ["Magnitude_raw", "Ridge_M", "Ridge_P", "Ridge_PQ", "Ridge_PQH"]
    view = summary.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(range(len(order)), view.utility20, color=["#777777", "#5677A6", "#4C956C", "#E09F3E", "#A23B72"])
    axes[0].set_xticks(range(len(order)), ["Raw M", "Learned M", "P", "P+Q", "P+Q+H"], rotation=20)
    axes[0].set_ylabel("Macro Utility@20")
    axes[0].set_title("TxPert risk-information staircase (Ridge)")
    axes[0].axhline(0, color="black", lw=0.7)
    inc = increments[increments.comparison.isin(["P_given_M", "Q_given_P", "H_given_PQ"])]
    pivot = inc.pivot(index="comparison", columns="algorithm", values="delta_utility20").reindex(["P_given_M", "Q_given_P", "H_given_PQ"])
    x = np.arange(3)
    for offset, kind, color in [(-0.24, "Ridge", "#5677A6"), (0, "HGB", "#4C956C"), (0.24, "MLP", "#A23B72")]:
        axes[1].bar(x + offset, pivot[kind], width=0.22, label=kind, color=color)
    axes[1].set_xticks(x, ["M→P", "P→P+Q", "P+Q→P+Q+H"])
    axes[1].set_ylabel("Δ macro Utility@20")
    axes[1].set_title("Conditional information gain")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "TXPERT_RISK_RESULTS.png", dpi=220)
    fig.savefig(OUT / "TXPERT_RISK_RESULTS.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "RUN_STATUS.json").exists():
        raise FileExistsError("refusing to overwrite an existing TxPert risk batch")
    if not torch.cuda.is_available():
        raise RuntimeError("existing cu124 GPU environment required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = "cuda:0"
    started = time.time()
    status = {
        "status": "RUNNING",
        "started_unix": started,
        "git_head": os.popen(f"git -C '{ROOT}' rev-parse HEAD").read().strip(),
        "contract_sha256": sha256(CONTRACT),
        "feature_groups": GROUPS,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "new_upstream_training_runs": 0,
        "evidence_level": "RELEASED_RETROSPECTIVE_DEVELOPMENT",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    frame, vector_audit = build_frame()
    pred, folds, fit_count, parameters = evaluate(frame, device)
    target, summary = score_predictions(pred)
    increments = bootstrap_increments(pred, summary)
    frame.to_csv(OUT / "FEATURE_TABLE.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pred.to_csv(OUT / "OOF_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    folds.to_csv(OUT / "SPLIT_MANIFEST.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    target.to_csv(OUT / "TARGET_RESULTS.csv", index=False)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    increments.to_csv(OUT / "INCREMENTAL_RESULTS.csv", index=False)
    make_figure(summary, increments)
    status.update(
        status="COMPLETE_STOPPED_AFTER_REGISTERED_BATCH",
        completed_unix=time.time(),
        elapsed_seconds=time.time() - started,
        n_tasks=len(frame),
        n_targets=frame.target.nunique(),
        n_unique_genes=frame.gene.nunique(),
        n_model_fits=fit_count,
        n_methods=pred.method.nunique(),
        n_oof_rows=len(pred),
        label_rows_used=len(frame),
        final_sealed_external_test_rows_used=0,
        error_memory_used=False,
        vector_audit=vector_audit,
        parameter_counts=parameters,
        result_sha256=sha256(OUT / "SUMMARY.csv"),
    )
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(increments.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
