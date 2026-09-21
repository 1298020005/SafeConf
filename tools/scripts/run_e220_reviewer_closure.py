#!/usr/bin/env python3
"""Post-release E205 controls: scale matching, centroid endpoints and ablations.

Reads a single released task CSV. It cannot open E208 truth or model checkpoints.
All candidate results are retained; no candidate is promoted to a main method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs/实验结果/E205_cross_family_disagreement_20260830/formal_evaluation/E205_TASK_METRICS.csv"
DEFAULT_OUTPUT = ROOT / "docs/实验结果/E220_reviewer_closure_20260921"
COMPONENTS = {
    "D": "z_family_disagreement", "G": "z_model_source_gap",
    "H": "z_source_delta_dispersion", "N": "z_negative_log_source_cells",
    "C": "z_support_context_deficit",
}
TARGETS = ("K562", "RPE1", "hepg2", "jurkat")
METRICS = ("spearman", "utility_20", "error_capture_20", "retained_mean_error_80", "normalized_aurc")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within_rank(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.empty(len(frame), dtype=float)
    for idx in frame.groupby("target", sort=False).indices.values():
        result[idx] = rankdata(np.asarray(values)[idx], method="average") / len(idx)
    return result


def score_matrices(frame: pd.DataFrame) -> dict[str, tuple[list[str], np.ndarray]]:
    if frame.task_id.duplicated().any() or len(frame) != 2008:
        raise ValueError("E205 task identity contract failed")
    z = frame[list(COMPONENTS.values())].to_numpy(float)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite pretruth feature")
    if not np.allclose(z.mean(axis=1), frame.safeconf_e205_risk, rtol=0, atol=1e-12):
        raise ValueError("Released five-component score does not reproduce")
    s = within_rank(frame, z.mean(axis=1))
    m = within_rank(frame, frame.predicted_magnitude.to_numpy())
    if not np.allclose(0.8 * m + 0.2 * s, frame.safeconf_m_4to1, rtol=0, atol=1e-12):
        raise ValueError("Released 4:1 score does not reproduce on full ranking batches")
    scores = {
        "magnitude_raw": frame.predicted_magnitude.to_numpy(),
        "magnitude_percentile": m,
        "safeconf_original": s,
        "safeconf_m_4to1": frame.safeconf_m_4to1.to_numpy(),
        "disagreement_only": within_rank(frame, frame.family_disagreement.to_numpy()),
        "source_effect_magnitude": within_rank(frame, frame.source_transfer_magnitude.to_numpy()),
    }
    for i, name in enumerate(COMPONENTS):
        scores[f"m_plus_{name}"] = 0.8 * m + 0.2 * within_rank(frame, z[:, i])
        scores[f"drop_{name}"] = 0.8 * m + 0.2 * within_rank(frame, np.delete(z, i, axis=1).mean(axis=1))
    scores["m_plus_source_GHNC"] = 0.8 * m + 0.2 * within_rank(frame, z[:, 1:].mean(axis=1))
    scores["m_plus_prediction_DG"] = 0.8 * m + 0.2 * within_rank(frame, z[:, :2].mean(axis=1))
    for weight in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        scores[f"weight_{weight:.2f}"] = (1 - weight) * m + weight * s
    registered_m = within_rank(frame, frame.registered_predicted_magnitude.to_numpy())
    cross = within_rank(frame, frame.cross_family_disagreement.to_numpy())
    registered = {
        "magnitude_raw": frame.registered_predicted_magnitude.to_numpy(),
        "magnitude_percentile": registered_m,
        "context_holdout_router": frame.context_holdout_router.to_numpy(),
        "architecture_aware_router": frame.architecture_aware_router.to_numpy(),
        "historical_nonnegative_router": frame.historical_nonnegative_router.to_numpy(),
        "certificate_priority_q80": frame.certificate_priority_q80.to_numpy(),
        "m_plus_cross_disagreement": 0.8 * registered_m + 0.2 * cross,
    }
    return {key: (list(values), np.asarray(list(values.values()), dtype=float))
            for key, values in (("exphormer", scores), ("registered", registered))}


def metric_matrix(scores: np.ndarray, outcome: np.ndarray, ties: np.ndarray) -> np.ndarray:
    """Rows are methods; positive risk scores select larger errors first."""
    scores = np.asarray(scores, float)
    y = np.asarray(outcome, float)
    n = len(y)
    if n < 5 or scores.ndim != 2 or scores.shape[1] != n:
        raise ValueError("Bad metric dimensions")
    if not np.isfinite(scores).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Metrics require finite nonnegative errors")
    ranked_s = rankdata(scores, method="average", axis=1)
    ranked_s -= ranked_s.mean(axis=1, keepdims=True)
    ranked_y = rankdata(y, method="average")
    ranked_y -= ranked_y.mean()
    denom = np.linalg.norm(ranked_s, axis=1) * np.linalg.norm(ranked_y)
    rho = np.divide(ranked_s @ ranked_y, denom, out=np.full(len(scores), np.nan), where=denom > 0)
    order_high = np.lexsort((np.broadcast_to(ties, scores.shape), -scores), axis=1)
    k = math.ceil(0.20 * n)
    selected_sum = y[order_high[:, :k]].sum(axis=1)
    average = y.mean()
    oracle = np.sort(y)[-k:].mean()
    utility = (selected_sum / k - average) / (oracle - average) if oracle > average else np.full(len(scores), np.nan)
    capture = selected_sum / y.sum()
    remaining = (y.sum() - selected_sum) / (n - k)
    order_low = np.lexsort((np.broadcast_to(ties, scores.shape), scores), axis=1)
    aurc = (np.cumsum(y[order_low], axis=1) / np.arange(1, n + 1)).mean(axis=1) / average
    return np.column_stack((rho, utility, capture, remaining, aurc))


def cluster_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(labels)
    members = [np.flatnonzero(labels == x) for x in unique]
    return np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])


def evaluate_job(args: tuple) -> tuple[list[dict], list[dict]]:
    suite, outcome_name, names, scores, y, targets, conditions, ties, n_bootstrap, seed = args
    scopes = (*TARGETS, "pooled", "macro")
    point = np.empty((len(scopes), len(names), len(METRICS)))
    for i, target in enumerate(TARGETS):
        idx = np.flatnonzero(targets == target)
        point[i] = metric_matrix(scores[:, idx], y[idx], ties[idx])
    point[4] = metric_matrix(scores, y, ties)
    point[5] = point[:4].mean(axis=0)
    baseline = names.index("magnitude_percentile")
    draws = np.empty((n_bootstrap, len(scopes), len(names), len(METRICS)))
    rng = np.random.default_rng(seed)
    unique = np.unique(conditions)
    members = [np.flatnonzero(conditions == x) for x in unique]
    for b in range(n_bootstrap):
        selected = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        for i, target in enumerate(TARGETS):
            idx = selected[targets[selected] == target]
            draws[b, i] = metric_matrix(scores[:, idx], y[idx], ties[idx])
        draws[b, 4] = metric_matrix(scores[:, selected], y[selected], ties[selected])
        draws[b, 5] = draws[b, :4].mean(axis=0)
    differences = draws - draws[:, :, baseline:baseline + 1, :]
    lo, hi = np.nanquantile(differences, [0.025, 0.975], axis=0)
    rows, intervals = [], []
    for i, scope in enumerate(scopes):
        for j, name in enumerate(names):
            row = {"suite": suite, "outcome": outcome_name, "scope": scope, "method": name,
                   "stratum": "primary_ge30", "n_tasks": int((targets == scope).sum()) if scope in TARGETS else len(y)}
            row.update(dict(zip(METRICS, point[i, j])))
            rows.append(row)
            for k, metric in enumerate(METRICS):
                intervals.append({**{key: row[key] for key in ("suite", "outcome", "scope", "method")},
                                  "metric": metric, "comparator": "magnitude_percentile",
                                  "delta": point[i, j, k] - point[i, baseline, k],
                                  "ci95_lower": lo[i, j, k], "ci95_upper": hi[i, j, k],
                                  "n_bootstrap": n_bootstrap, "seed": seed,
                                  "invalid_draws": int(np.isnan(differences[:, i, j, k]).sum()),
                                  "inference": "exploratory_paired_condition_cluster"})
    return rows, intervals


def plot_results(summary: pd.DataFrame, intervals: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    methods = ("magnitude_raw", "magnitude_percentile", "safeconf_m_4to1")
    labels = ("Raw magnitude", "Within-context rank", "SafeConf-M")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, scope in zip(axes, ("pooled", "macro")):
        block = summary.query("suite == 'exphormer' and outcome == 'family_rms_error' and stratum == 'primary_ge30'")
        block = block[block.scope == scope].set_index("method")
        ax.bar(np.arange(3), [block.loc[m, "utility_20"] for m in methods], color=("#acb4ba", "#497398", "#25877c"), width=0.64)
        ax.set_xticks(np.arange(3), labels, rotation=15, ha="right")
        ax.set_ylabel("Review utility at 20% budget")
        ax.set_title("Pooled tasks" if scope == "pooled" else "Equal-context mean")
    save(fig, "scale_matched_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, outcome in zip(axes, ("family_rms_error", "family_centroid_rmse")):
        block = intervals.query("method == 'safeconf_m_4to1' and metric == 'utility_20'")
        block = block[block.outcome == outcome].set_index("scope")
        selected = block.loc[[*TARGETS, "macro"]]
        y = np.arange(5)
        ax.errorbar(selected.delta, y, xerr=np.vstack((selected.delta - selected.ci95_lower,
                     selected.ci95_upper - selected.delta)), fmt="o", color="#25877c", capsize=3)
        ax.axvline(0, color="#81878b", linewidth=0.8)
        ax.set_yticks(y, [*TARGETS, "Equal-context mean"])
        ax.invert_yaxis()
        ax.set_xlabel("Utility difference vs matched magnitude")
        ax.set_title("Member RMS error" if outcome == "family_rms_error" else "Mean prediction error")
    save(fig, "context_and_centroid_controls")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    methods = ["safeconf_m_4to1", *[f"m_plus_{x}" for x in COMPONENTS], *[f"drop_{x}" for x in COMPONENTS]]
    block = summary.query("scope == 'macro' and outcome == 'family_centroid_rmse' and stratum == 'primary_ge30'").set_index("method")
    baseline = block.loc["magnitude_percentile", "utility_20"]
    axes[0].barh(np.arange(len(methods)), [block.loc[m, "utility_20"] - baseline for m in methods], color="#497398")
    axes[0].set_yticks(np.arange(len(methods)), ["Full score", *[f"Magnitude + {x}" for x in COMPONENTS], *[f"Without {x}" for x in COMPONENTS]])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#81878b", linewidth=0.8)
    axes[0].set_xlabel("Mean utility gain (centroid error)")
    for target in TARGETS:
        b = summary[(summary.scope == target) & (summary.outcome == "family_centroid_rmse") & (summary.stratum == "primary_ge30")].set_index("method")
        weights = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
        axes[1].plot(weights, [b.loc[f"weight_{w:.2f}", "utility_20"] - b.loc["magnitude_percentile", "utility_20"] for w in weights], marker="o", markersize=3, label=target)
    axes[1].axhline(0, color="#81878b", linewidth=0.8)
    axes[1].axvline(0.2, color="#aaa", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("SafeConf weight (all values retained)")
    axes[1].set_ylabel("Review utility difference")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "ablation_and_weight_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.n_bootstrap < 100 or not 1 <= args.workers <= 4:
        raise ValueError("Use >=100 bootstraps and 1--4 workers")
    started = time.monotonic()
    frame = pd.read_csv(args.input)
    source_sha = digest(args.input)
    all_scores = score_matrices(frame)
    mask = frame.analysis_stratum.eq("primary_ge30").to_numpy()
    if mask.sum() != 1808 or set(frame.target) != set(TARGETS):
        raise ValueError("E205 primary contract failed")
    ties = np.array([int(hashlib.sha256(f"E205\0{x}\00".encode()).hexdigest()[:16], 16) for x in frame.task_id], dtype=np.uint64)
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for suite, outcomes in (("exphormer", ("family_rms_error", "family_centroid_rmse")),
                            ("registered", ("registered_family_rms_error", "registered_family_centroid_rmse"))):
        names, scores = all_scores[suite]
        for outcome in outcomes:
            tasks.append((suite, outcome, names, scores[:, mask], frame.loc[mask, outcome].to_numpy(),
                          frame.loc[mask, "target"].to_numpy(), frame.loc[mask, "condition"].to_numpy(), ties[mask],
                          args.n_bootstrap, 20260921))
    summaries, intervals = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, cis in pool.map(evaluate_job, tasks):
            summaries.extend(rows)
            intervals.extend(cis)
            print(f"Finished {rows[0]['suite']} / {rows[0]['outcome']}", flush=True)
    for suite, outcome, names, _, _, _, _, _, _, _ in tasks:
        scores = all_scores[suite][1]
        y = frame[outcome].to_numpy()
        points = []
        for scope in (*TARGETS, "pooled"):
            idx = np.arange(len(frame)) if scope == "pooled" else np.flatnonzero(frame.target.to_numpy() == scope)
            values = metric_matrix(scores[:, idx], y[idx], ties[idx])
            points.append(values)
            for j, name in enumerate(names):
                summaries.append({"suite": suite, "outcome": outcome, "scope": scope, "method": name,
                                  "stratum": "all_2008", "n_tasks": len(idx), **dict(zip(METRICS, values[j]))})
        for j, name in enumerate(names):
            summaries.append({"suite": suite, "outcome": outcome, "scope": "macro", "method": name,
                              "stratum": "all_2008", "n_tasks": len(frame),
                              **dict(zip(METRICS, np.mean(points[:4], axis=0)[j]))})
    summary, ci = pd.DataFrame(summaries), pd.DataFrame(intervals)
    summary.to_csv(args.output / "SUMMARY.csv", index=False)
    ci.to_csv(args.output / "PAIRED_INTERVALS.csv", index=False)
    plot_results(summary, ci, args.output)
    rows = ci[(ci.method == "safeconf_m_4to1") & ci.scope.isin(["pooled", "macro"]) & ci.metric.isin(["spearman", "utility_20"])]
    report = ["# E220 同口径基线与实际预测误差复核", "",
              "本分析只使用已公开 E205 表；所有结论属于事后稳健性分析。原始任务、误差和 E208 协议未改。",
              "", "## 固定组合相对同口径幅度", "",
              "基线在每个背景内先转百分位，与 SafeConf-M 使用相同的排名批次。宏平均表示每个背景各复核 20%，合并表示所有背景共用一份预算。", "",
              "| 误差对象 | 汇总方式 | 指标 | 增量 | 配对簇自助 95% 区间 |", "|---|---|---|---:|---|"]
    for row in rows.itertuples():
        report.append(f"| {row.outcome} | {row.scope} | {row.metric} | {row.delta:+.5f} | [{row.ci95_lower:+.5f}, {row.ci95_upper:+.5f}] |")
    report += ["", "## 图件与完整记录", "",
               "- `scale_matched_comparison.svg`：原始幅度、同背景幅度百分位和固定组合，区分合并与背景等权。",
               "- `context_and_centroid_controls.svg`：四个背景、两种误差对象和配对区间。",
               "- `ablation_and_weight_sensitivity.svg`：单项对照、删除分量和全部候选权重。",
               "- 每张图提供 PDF、PNG；图内无 Figure 编号。",
               "- `SUMMARY.csv`：全部方法、全部背景、主任务与全任务敏感性分析。",
               "- `PAIRED_INTERVALS.csv`：每个方法相对同口径幅度的探索性区间。",
               "", "## 解释边界", "",
               "这些配对区间固定四个观察背景；同一扰动跨背景一起抽样。不能将其解释为新背景总体保证。",
               "消融和权重结果用于理解机制，不按最好的结果重定义主方法。当前确认性外部协议仍为 E208。",
               "E205 沿用 E201 已公开的真实标签，因此属于新增预测结构检验，不是全新独立标签数据的盲测。",
               "真实预测向量的误差没有被风险排序降低；效用指固定预算内优先找到错误预测。", ""]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if digest(args.input) != source_sha:
        raise ValueError("Input CSV changed during analysis")
    result = {
        "experiment": "E220_reviewer_closure", "status": "COMPLETE", "evidence_class": "post_release_sensitivity",
        "created_at": datetime.now().astimezone().isoformat(), "source_sha256": source_sha,
        "source": str(args.input.relative_to(ROOT)), "script_sha256": digest(Path(__file__)),
        "n_all_tasks": len(frame), "n_primary_tasks": int(mask.sum()),
        "all_target_counts": {k: int(v) for k, v in frame.groupby("target").size().items()},
        "primary_target_counts": {k: int(v) for k, v in frame.loc[mask].groupby("target").size().items()},
        "n_bootstrap": args.n_bootstrap, "workers": args.workers, "elapsed_seconds": time.monotonic() - started,
        "e208_test_rows_read": 0, "new_weights_selected": False, "source_data_modified": False,
        "summary_rows": len(summary), "interval_rows": len(ci), "seed": 20260921,
    }
    (args.output / "STATUS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
