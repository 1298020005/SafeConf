#!/usr/bin/env python3
"""E209: retrospective magnitude-adjusted SafeConf increment audit.

The analysis never treats historical truth as a deployable feature.  Truth is
used only to evaluate a score that was already available before truth access.
E209 is post-hoc mechanism development, not an external confirmation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E209_magnitude_matched_increment_20260912"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
E201 = ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802/formal_core_evaluation/tables/E201_TASK_METRICS.csv"
E153 = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv"
E187 = ROOT / "docs/实验结果/E187_advisor_difficulty_certificate_20260726/tables/E187_CARTESIAN_TASK_CERTIFICATES.csv"
EXPECTED = {
    E201: ("4a02d132cae1605a2f6f54bee8912f45e835d1fc3936e6cda0fcc98e2c6bcd35", 2008),
    E153: ("b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a", 3465),
    E187: ("c84cb0f2b8c36c27b33d62cfbad7e98d2228288a85937e18351ab13d069d7ba0", 8196),
}
SEED = 202609209
DEFAULT_BOOTSTRAP = 2000
N_MAGNITUDE_BINS = 5


class ContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 4:
        return float("nan")
    left_rank = rankdata(left[keep], method="average")
    right_rank = rankdata(right[keep], method="average")
    if np.unique(left_rank).size < 2 or np.unique(right_rank).size < 2:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def partial_spearman(safeconf: np.ndarray, error: np.ndarray, magnitude: np.ndarray) -> float:
    safeconf = np.asarray(safeconf, dtype=float)
    error = np.asarray(error, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    keep = np.isfinite(safeconf) & np.isfinite(error) & np.isfinite(magnitude)
    if keep.sum() < 5:
        return float("nan")
    ranked_safe = rankdata(safeconf[keep], method="average")
    ranked_error = rankdata(error[keep], method="average")
    ranked_magnitude = rankdata(magnitude[keep], method="average")
    if min(np.unique(ranked_safe).size, np.unique(ranked_error).size, np.unique(ranked_magnitude).size) < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(ranked_magnitude)), ranked_magnitude])
    residual_safe = ranked_safe - design @ np.linalg.lstsq(design, ranked_safe, rcond=None)[0]
    residual_error = ranked_error - design @ np.linalg.lstsq(design, ranked_error, rcond=None)[0]
    if np.std(residual_safe) <= 0 or np.std(residual_error) <= 0:
        return float("nan")
    return float(np.corrcoef(residual_safe, residual_error)[0, 1])


def magnitude_matched_spearman(block: pd.DataFrame) -> tuple[float, int]:
    n = len(block)
    if n < 8:
        return float("nan"), 0
    magnitude_rank = rankdata(block["magnitude"].to_numpy(float), method="average") / n
    bins = np.minimum(N_MAGNITUDE_BINS, np.ceil(N_MAGNITUDE_BINS * magnitude_rank)).astype(int)
    values = []
    for label in sorted(np.unique(bins)):
        part = block.loc[bins == label]
        value = spearman(part["safeconf"].to_numpy(float), part["error"].to_numpy(float))
        if math.isfinite(value):
            values.append(value)
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def batch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, batch), block in frame.groupby(["study", "batch"], sort=True):
        partial = partial_spearman(block.safeconf, block.error, block.magnitude)
        matched, n_bins = magnitude_matched_spearman(block)
        rows.append({
            "analysis": str(block.analysis.iloc[0]),
            "study": study,
            "batch": batch,
            "n_tasks": len(block),
            "n_clusters": block.cluster.nunique(),
            "partial_spearman": partial,
            "magnitude_matched_spearman": matched,
            "matched_bins_evaluable": n_bins,
            "safeconf_spearman": spearman(block.safeconf, block.error),
            "magnitude_spearman": spearman(block.magnitude, block.error),
        })
    return pd.DataFrame(rows)


def study_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["analysis", "study"], as_index=False).agg(
        n_batches=("batch", "size"),
        n_tasks=("n_tasks", "sum"),
        partial_spearman=("partial_spearman", "mean"),
        magnitude_matched_spearman=("magnitude_matched_spearman", "mean"),
        safeconf_spearman=("safeconf_spearman", "mean"),
        magnitude_spearman=("magnitude_spearman", "mean"),
    )


def load_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    manifest = []
    raw = {}
    for path, (expected_hash, expected_rows) in EXPECTED.items():
        observed = sha256(path)
        if observed != expected_hash:
            raise ContractError(f"frozen input hash changed: {path}")
        frame = pd.read_csv(path)
        if len(frame) != expected_rows:
            raise ContractError(f"frozen input row count changed: {path}")
        manifest.append({
            "path": path.relative_to(ROOT).as_posix(),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": observed,
        })
        raw[path.name] = frame

    e201 = raw[E201.name]
    e201 = e201.loc[e201.analysis_stratum.eq("primary_ge30")].copy()
    if len(e201) != 1808 or e201.target.nunique() != 4:
        raise ContractError("E201 primary stratum dimensions changed")
    e201_standard = pd.DataFrame({
        "analysis": "E201_TxPert_whole_context",
        "study": e201.target.astype(str),
        "batch": e201.target.astype(str),
        "cluster": e201.gene.astype(str),
        "safeconf": e201.safeconf_e201_risk.astype(float),
        "magnitude": e201.predicted_magnitude.astype(float),
        "error": e201.family_rms_error.astype(float),
    })

    e153 = raw[E153.name]
    if e153.dataset.nunique() != 8 or e153.fold_id.nunique() != 34:
        raise ContractError("E153 dimensions changed")
    e153_standard = pd.DataFrame({
        "analysis": "E153_genetic_8study",
        "study": e153.dataset.astype(str),
        "batch": e153.dataset.astype(str) + "|" + e153.fold_id.astype(str),
        "cluster": e153.perturbation.astype(str),
        "safeconf": e153.safeconf_calibrated_pair_risk.astype(float),
        "magnitude": e153.baseline_predicted_magnitude.astype(float),
        "error": e153.error_two_predictor_mean_rmse.astype(float),
    })

    e187 = raw[E187.name]
    if set(e187.train_fraction.unique()) != {0.25, 0.50, 0.75, 1.00}:
        raise ContractError("E187 training fractions changed")
    e187 = e187.dropna(subset=["safeconf_calibrated_pair_risk"]).copy()
    e187["analysis"] = np.where(
        e187.dataset.eq("Cui_direct41"),
        "E187_cytokine_difficulty",
        "E187_genetic_difficulty",
    )
    e187_standard = pd.DataFrame({
        "analysis": e187.analysis,
        "study": e187.dataset.astype(str),
        "batch": (
            e187.dataset.astype(str) + "|" + e187.fold_id.astype(str) + "|"
            + e187.train_fraction.astype(str) + "|" + e187.setting.astype(str)
        ),
        "cluster": e187.perturbation.astype(str),
        "safeconf": e187.safeconf_calibrated_pair_risk.astype(float),
        "magnitude": e187.baseline_predicted_magnitude.astype(float),
        "error": e187.error_two_predictor_mean_rmse.astype(float),
    })

    analyses = {}
    for frame in (e201_standard, e153_standard, e187_standard):
        for name, part in frame.groupby("analysis", sort=True):
            if not np.isfinite(part[["safeconf", "magnitude", "error"]].to_numpy(float)).all():
                raise ContractError(f"non-finite score/outcome in {name}")
            analyses[name] = part.reset_index(drop=True)
    return analyses, pd.DataFrame(manifest)


def resample_study(block: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clusters = sorted(block.cluster.astype(str).unique())
    members = {name: block.loc[block.cluster.astype(str).eq(name)] for name in clusters}
    selected = rng.choice(clusters, size=len(clusters), replace=True)
    output = []
    for occurrence, name in enumerate(selected):
        part = members[str(name)].copy()
        part["cluster"] = part.cluster.astype(str) + f"|boot{occurrence}"
        output.append(part)
    return pd.concat(output, ignore_index=True)


def bootstrap_chunk(frame: pd.DataFrame, analysis: str, indices: list[int]) -> pd.DataFrame:
    studies = sorted(frame.study.unique())
    label_seed = int(hashlib.sha256(analysis.encode()).hexdigest()[:12], 16)
    rows = []
    for replicate in indices:
        rng = np.random.default_rng(SEED + label_seed + replicate * 1_000_003)
        sampled = pd.concat(
            [resample_study(frame.loc[frame.study.eq(study)], rng) for study in studies],
            ignore_index=True,
        )
        summary = study_summary(batch_metrics(sampled))
        rows.append({
            "analysis": analysis,
            "replicate": replicate,
            "partial_spearman": float(summary.partial_spearman.mean()),
            "magnitude_matched_spearman": float(summary.magnitude_matched_spearman.mean()),
        })
    return pd.DataFrame(rows)


def run_bootstrap(frame: pd.DataFrame, analysis: str, n_bootstrap: int, workers: int) -> pd.DataFrame:
    chunks = [list(map(int, x)) for x in np.array_split(np.arange(n_bootstrap), workers) if len(x)]
    if len(chunks) == 1:
        return bootstrap_chunk(frame, analysis, chunks[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(bootstrap_chunk, frame, analysis, chunk) for chunk in chunks]
        result = pd.concat([future.result() for future in futures], ignore_index=True)
    return result.sort_values("replicate", kind="stable").reset_index(drop=True)


def bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis, block in draws.groupby("analysis", sort=True):
        for metric in ("partial_spearman", "magnitude_matched_spearman"):
            values = block[metric].dropna().to_numpy(float)
            rows.append({
                "analysis": analysis,
                "metric": metric,
                "ci95_lower": float(np.quantile(values, 0.025)),
                "bootstrap_median": float(np.median(values)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "fraction_positive": float(np.mean(values > 0)),
                "n_valid": len(values),
            })
    return pd.DataFrame(rows)


def overall_summary(studies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis, block in studies.groupby("analysis", sort=True):
        rows.append({
            "analysis": analysis,
            "n_studies_or_targets": len(block),
            "n_positive_partial": int((block.partial_spearman > 0).sum()),
            "partial_spearman": float(block.partial_spearman.mean()),
            "magnitude_matched_spearman": float(block.magnitude_matched_spearman.mean()),
            "safeconf_spearman": float(block.safeconf_spearman.mean()),
            "magnitude_spearman": float(block.magnitude_spearman.mean()),
        })
    return pd.DataFrame(rows)


def plot_forest(studies: pd.DataFrame, bootstrap: pd.DataFrame, output: Path) -> None:
    focus = studies.loc[studies.analysis.eq("E153_genetic_8study")].sort_values("partial_spearman")
    overall = bootstrap.loc[
        bootstrap.analysis.eq("E153_genetic_8study") & bootstrap.metric.eq("partial_spearman")
    ].iloc[0]
    labels = focus.study.tolist() + ["Eight-study mean"]
    values = focus.partial_spearman.tolist() + [float(focus.partial_spearman.mean())]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(values[:-1], y[:-1], s=48, color="#287C8E", zorder=3)
    ax.errorbar(
        values[-1], y[-1],
        xerr=[[values[-1] - float(overall.ci95_lower)], [float(overall.ci95_upper) - values[-1]]],
        fmt="o", markersize=7, color="#D95F59", capsize=4, linewidth=2.0, zorder=4,
    )
    ax.axvline(0, color="#60666B", linestyle="--", linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Partial Spearman of SafeConf with error, controlling magnitude")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E8ECEF", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=320, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_report(overall: pd.DataFrame, studies: pd.DataFrame, ci: pd.DataFrame) -> None:
    lookup = overall.set_index("analysis")
    cilook = ci.loc[ci.metric.eq("partial_spearman")].set_index("analysis")
    names = {
        "E201_TxPert_whole_context": "E201 TxPert 整背景留出",
        "E153_genetic_8study": "E153 八个遗传研究",
        "E187_genetic_difficulty": "E187 遗传难度网格",
        "E187_cytokine_difficulty": "E187 细胞因子边界",
    }
    lines = [
        "# E209｜预测幅度相近时，SafeConf 还能提供什么",
        "",
        "证据身份：**已解封历史数据的回顾性机制分析，不是新的外部确认。**",
        "",
        "## 主结果",
        "",
        "| 分析 | 研究/目标数 | 正向数 | 控制幅度后的偏 Spearman | 95% 簇自举区间 | 幅度分层内 Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in names:
        row = lookup.loc[key]
        interval = cilook.loc[key]
        lines.append(
            f"| {names[key]} | {int(row.n_studies_or_targets)} | {int(row.n_positive_partial)} | "
            f"{row.partial_spearman:.4f} | [{interval.ci95_lower:.4f}, {interval.ci95_upper:.4f}] | "
            f"{row.magnitude_matched_spearman:.4f} |"
        )
    e153 = lookup.loc["E153_genetic_8study"]
    e153_ci = cilook.loc["E153_genetic_8study"]
    gate = bool(e153_ci.ci95_lower > 0 and e153.n_positive_partial >= 6)
    lines += [
        "",
        "## 解释",
        "",
        f"E153 的预设回顾性判据为 `{'PASS' if gate else 'FAIL'}`：八研究等权偏 Spearman 为 "
        f"{e153.partial_spearman:.4f}，95% 区间 [{e153_ci.ci95_lower:.4f}, {e153_ci.ci95_upper:.4f}]，"
        f"逐研究 {int(e153.n_positive_partial)}/{int(e153.n_studies_or_targets)} 为正。",
        "",
        "这说明预测幅度解释了任务误差的一大部分，但没有吸收 SafeConf 的全部信息。"
        "在预测幅度相近的任务之间，训练支持、背景覆盖和模型稳定性仍能提供额外的风险排序信息。",
        "",
        "该结果支持“双因素”叙述：预测幅度描述响应规模，SafeConf 描述证据质量。"
        "它不证明 SafeConf 单独优于幅度，也不把回顾性偏相关当成节省实验成本的证据。"
        "固定 SafeConf-M 的最终效果仍由 E205 和 E208 按冻结合同确认。",
        "",
        "E201 此处的 0.2827 是先在四个目标内分别计算、再取宏平均；E201 原正式报告的 "
        "0.2503 是把 1,808 个任务合并计算。两者回答的聚合问题不同，不能互相替换。",
        "",
        "## 逐研究结果",
        "",
        "逐研究与逐批次值见 `tables/E209_STUDY_SUMMARY.csv` 和 `tables/E209_BATCH_RESULTS.csv`；"
        "2,000 次簇自助原始抽样见 `tables/E209_BOOTSTRAP_DRAWS.csv`。",
    ]
    atomic_text(OUT / "E209_REPORT.md", "\n".join(lines) + "\n")


def run_self_test() -> None:
    rng = np.random.default_rng(9)
    magnitude = np.arange(40, dtype=float)
    safe = rng.normal(size=40)
    error = magnitude + 5 * safe
    value = partial_spearman(safe, error, magnitude)
    assert value > 0.80
    frame = pd.DataFrame({"safeconf": safe, "magnitude": magnitude, "error": error})
    matched, bins = magnitude_matched_spearman(frame)
    assert math.isfinite(matched) and bins == 5
    independent = partial_spearman(rng.normal(size=40), magnitude, magnitude)
    assert abs(independent) < 0.5
    print("E209 self-test: PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstrap < 100 or args.workers < 1:
        raise ContractError("bootstrap must be >=100 and workers >=1")
    if not CONTRACT.is_file():
        raise ContractError("analysis contract missing")

    analyses, manifest = load_inputs()
    batches = pd.concat([batch_metrics(frame) for frame in analyses.values()], ignore_index=True)
    studies = study_summary(batches)
    overall = overall_summary(studies)
    draws = []
    for analysis, frame in analyses.items():
        draws.append(run_bootstrap(frame, analysis, args.bootstrap, args.workers))
    draws = pd.concat(draws, ignore_index=True)
    ci = bootstrap_summary(draws)

    tables = OUT / "tables"
    figures = OUT / "figures"
    atomic_csv(tables / "E209_INPUT_MANIFEST.csv", manifest)
    atomic_csv(tables / "E209_BATCH_RESULTS.csv", batches)
    atomic_csv(tables / "E209_STUDY_SUMMARY.csv", studies)
    atomic_csv(tables / "E209_OVERALL_SUMMARY.csv", overall)
    atomic_csv(tables / "E209_BOOTSTRAP_DRAWS.csv", draws)
    atomic_csv(tables / "E209_BOOTSTRAP_SUMMARY.csv", ci)
    figures.mkdir(parents=True, exist_ok=True)
    plot_forest(studies, ci, figures / "E209_partial_forest")
    write_report(overall, studies, ci)

    e153 = overall.loc[overall.analysis.eq("E153_genetic_8study")].iloc[0]
    e153_ci = ci.loc[
        ci.analysis.eq("E153_genetic_8study") & ci.metric.eq("partial_spearman")
    ].iloc[0]
    status = {
        "experiment": "E209_magnitude_matched_increment",
        "status": "PASS" if e153_ci.ci95_lower > 0 and e153.n_positive_partial >= 6 else "FAIL",
        "evidence_identity": "retrospective_posthoc_mechanism_analysis",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "bootstrap_replicates": args.bootstrap,
        "workers": args.workers,
        "primary_analysis": "E153_genetic_8study",
        "primary_partial_spearman": float(e153.partial_spearman),
        "primary_ci95": [float(e153_ci.ci95_lower), float(e153_ci.ci95_upper)],
        "primary_positive_studies": int(e153.n_positive_partial),
        "primary_total_studies": int(e153.n_studies_or_targets),
        "target_truth_used_as_deployment_feature": False,
        "external_confirmation": False,
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
