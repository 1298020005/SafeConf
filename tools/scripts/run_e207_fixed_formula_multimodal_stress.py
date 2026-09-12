#!/usr/bin/env python3
"""E207 retrospective stress test of the frozen SafeConf-M 0.8/0.2 rule.

All inputs in this analysis were released before E207.  The runner therefore
produces retrospective robustness evidence only; it is not an independent
confirmation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E207_fixed_formula_multimodal_stress_20260912"
CONTRACT = OUT / "ANALYSIS_CONTRACT.md"
E153 = ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables/E153_ABSOLUTE_TASK_INPUT.csv"
E187 = ROOT / "docs/实验结果/E187_advisor_difficulty_certificate_20260726/tables/E187_CARTESIAN_TASK_CERTIFICATES.csv"
E118_SUMMARY = ROOT / "docs/实验结果/E118_chemical_contract_meta_20260713/tables/E118_SOURCE_SUMMARY.csv"
E118_BOOT = ROOT / "docs/实验结果/E118_chemical_contract_meta_20260713/tables/E118_BOOTSTRAP.csv"

EXPECTED = {
    E153: ("b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a", 3465),
    E187: ("c84cb0f2b8c36c27b33d62cfbad7e98d2228288a85937e18351ab13d069d7ba0", 8196),
    E118_SUMMARY: ("737b3279d48d2181d939afaeae31a03d30dbb5b9815e0fec198c9d583c324be1", 8),
    E118_BOOT: ("5b1df82a92f9082579f4fe52bf23d21b567028aeaeba491e8e4ca78baac870c6", 3),
}
SEED = 202609207
N_BOOT = 2000
BUDGETS = (0.05, 0.10, 0.20, 0.30)
ALPHA = 0.80
ERROR = "error_two_predictor_mean_rmse"
MAG = "baseline_predicted_magnitude"
SAFE = "safeconf_calibrated_pair_risk"
DIS = "risk_model_disagreement"
SCORES = ("magnitude", "safeconf", "disagreement", "safeconf_m")


class ContractError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, float_format="%.17g"))


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ContractError("percentile input is invalid")
    return rankdata(array, method="average") / len(array)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 4 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def tie_aware_selected_mean(score: np.ndarray, error: np.ndarray, budget: float) -> float:
    score = np.asarray(score, dtype=float)
    error = np.asarray(error, dtype=float)
    k = max(1, int(math.ceil(budget * len(score))))
    threshold = np.sort(score)[-k]
    above = score > threshold
    tied = score == threshold
    remaining = k - int(above.sum())
    return float((error[above].sum() + remaining * error[tied].mean()) / k)


def review_utility(score: np.ndarray, error: np.ndarray, budget: float) -> float:
    selected = tie_aware_selected_mean(score, error, budget)
    oracle = tie_aware_selected_mean(error, error, budget)
    random = float(np.mean(error))
    denominator = oracle - random
    return float((selected - random) / denominator) if denominator > 1e-15 else float("nan")


def validate_and_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = []
    frames = []
    for path, (expected_hash, expected_rows) in EXPECTED.items():
        observed = sha256(path)
        if observed != expected_hash:
            raise ContractError(f"input hash changed: {path}")
        frame = pd.read_csv(path)
        if len(frame) != expected_rows:
            raise ContractError(f"input row count changed: {path}")
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "rows": len(frame), "bytes": path.stat().st_size, "sha256": observed})
        frames.append(frame)
    e153, e187, chemical_summary, chemical_boot = frames
    if "model_disagreement_rmse" in e187.columns and DIS not in e187.columns:
        e187 = e187.rename(columns={"model_disagreement_rmse": DIS})
    required = {"dataset", "fold_id", "perturbation", ERROR, MAG, SAFE, DIS}
    for name, frame in (("E153", e153), ("E187", e187)):
        missing = required.difference(frame.columns)
        if missing:
            raise ContractError(f"{name} missing columns: {sorted(missing)}")
        if not np.isfinite(frame[[ERROR, MAG, SAFE, DIS]].to_numpy(float)).all():
            raise ContractError(f"{name} has non-finite score/outcome")
    if e153.dataset.nunique() != 8 or e153.fold_id.nunique() != 34:
        raise ContractError("E153 dimensions changed")
    if set(e187.train_fraction.unique()) != {0.25, 0.5, 0.75, 1.0}:
        raise ContractError("E187 fractions changed")
    return e153, e187, chemical_summary, chemical_boot, pd.DataFrame(manifest)


def add_scores(frame: pd.DataFrame, batch_columns: list[str]) -> pd.DataFrame:
    blocks = []
    for _, block in frame.groupby(batch_columns, sort=True, dropna=False):
        block = block.copy()
        block["magnitude"] = percentile(block[MAG])
        block["safeconf"] = percentile(block[SAFE])
        block["disagreement"] = percentile(block[DIS])
        block["safeconf_m"] = ALPHA * block["magnitude"] + (1.0 - ALPHA) * block["safeconf"]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def batch_metrics(frame: pd.DataFrame, batch_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(batch_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(batch_columns, keys))
        for score in SCORES:
            for budget in BUDGETS:
                rows.append({
                    **base,
                    "score": score,
                    "budget": budget,
                    "n_tasks": len(block),
                    "spearman": spearman(block[score], block[ERROR]),
                    "review_utility": review_utility(block[score].to_numpy(float), block[ERROR].to_numpy(float), budget),
                })
    return pd.DataFrame(rows)


def macro_table(metrics: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    group = [*dimensions, "score", "budget"]
    return metrics.groupby(group, as_index=False, dropna=False).agg(
        n_batches=("fold_id", "size"),
        spearman=("spearman", "mean"),
        review_utility=("review_utility", "mean"),
    )


def overall_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    study = metrics.groupby(["dataset", "score", "budget"], as_index=False).agg(
        spearman=("spearman", "mean"),
        review_utility=("review_utility", "mean"),
    )
    return study.groupby(["score", "budget"], as_index=False).agg(
        n_datasets=("dataset", "size"),
        spearman=("spearman", "mean"),
        review_utility=("review_utility", "mean"),
    )


def resample_clusters(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clusters = sorted(frame.perturbation.astype(str).unique())
    members = {name: frame.loc[frame.perturbation.astype(str).eq(name)] for name in clusters}
    selected = rng.choice(clusters, size=len(clusters), replace=True)
    blocks = []
    for occurrence, name in enumerate(selected):
        block = members[str(name)].copy()
        block["_cluster_occurrence"] = occurrence
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def bootstrap_delta(frame: pd.DataFrame, batch_columns: list[str], n_bootstrap: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap < 100:
        raise ContractError("bootstrap must be at least 100")
    datasets = sorted(frame.dataset.unique())
    rng = np.random.default_rng(SEED + sum(map(ord, label)))
    rows = []
    for replicate in range(n_bootstrap):
        study_values = []
        for dataset in datasets:
            sampled = resample_clusters(frame.loc[frame.dataset.eq(dataset)], rng)
            # Ranking must occur in the operational batch, not separately per bootstrap occurrence.
            scored = add_scores(sampled, batch_columns)
            metrics = batch_metrics(scored, batch_columns)
            primary = metrics.loc[metrics.budget.eq(0.20)]
            wide = primary.pivot_table(index=batch_columns, columns="score", values=["spearman", "review_utility"], aggfunc="first")
            study_values.append({
                "spearman": float((wide["spearman"]["safeconf_m"] - wide["spearman"]["magnitude"]).mean()),
                "review_utility": float((wide["review_utility"]["safeconf_m"] - wide["review_utility"]["magnitude"]).mean()),
            })
        rows.append({
            "analysis": label,
            "replicate": replicate,
            "delta_spearman": float(np.mean([x["spearman"] for x in study_values])),
            "delta_review_utility": float(np.mean([x["review_utility"] for x in study_values])),
        })
    draws = pd.DataFrame(rows)
    summary = []
    for metric in ("delta_spearman", "delta_review_utility"):
        values = draws[metric].dropna().to_numpy(float)
        summary.append({
            "analysis": label,
            "metric": metric,
            "ci95_lower": float(np.quantile(values, 0.025)),
            "bootstrap_median": float(np.median(values)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "fraction_positive": float(np.mean(values > 0)),
            "n_valid": len(values),
        })
    return draws, pd.DataFrame(summary)


def point_delta(metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    primary = metrics.loc[metrics.budget.eq(0.20)]
    study = primary.groupby(["dataset", "score"], as_index=False)[["spearman", "review_utility"]].mean()
    wide = study.pivot(index="dataset", columns="score", values=["spearman", "review_utility"])
    rows = []
    for metric in ("spearman", "review_utility"):
        rows.append({
            "analysis": label,
            "metric": f"delta_{metric}",
            "point_estimate": float((wide[metric]["safeconf_m"] - wide[metric]["magnitude"]).mean()),
            "positive_datasets": int((wide[metric]["safeconf_m"] > wide[metric]["magnitude"]).sum()),
            "n_datasets": len(wide),
        })
    return pd.DataFrame(rows)


def run_self_test() -> None:
    frame = pd.DataFrame({
        "dataset": ["d"] * 10,
        "fold_id": ["f"] * 10,
        "perturbation": [f"p{i}" for i in range(10)],
        ERROR: np.arange(10, dtype=float),
        MAG: np.arange(10, dtype=float),
        SAFE: np.arange(10, dtype=float)[::-1],
        DIS: np.arange(10, dtype=float),
    })
    scored = add_scores(frame, ["dataset", "fold_id"])
    assert np.isclose(scored.safeconf_m.iloc[-1], 0.82)
    assert np.isclose(review_utility(scored.magnitude, scored[ERROR], 0.20), 1.0)
    tied = np.ones(10)
    assert np.isclose(tie_aware_selected_mean(tied, frame[ERROR].to_numpy(), 0.20), 4.5)
    print("E207 self-test: PASS")


def write_report(e153_overall: pd.DataFrame, e187_genetic: pd.DataFrame, e187_cytokine: pd.DataFrame, point: pd.DataFrame, ci: pd.DataFrame, chemical: pd.DataFrame) -> None:
    def primary_rows(table: pd.DataFrame) -> pd.DataFrame:
        return table.loc[table.budget.eq(0.20), ["score", "n_datasets", "spearman", "review_utility"]]

    lines = [
        "# E207｜固定 SafeConf-M 跨场景压力测试", "",
        "证据身份：**历史数据回顾性压力测试，不是新的独立确认。**", "",
        "固定公式：`SafeConf-M = 0.80 × 批次内幅度百分位 + 0.20 × 批次内 SafeConf 百分位`。本实验没有重新搜索权重。", "",
    ]
    for title, table in (("八个遗传研究（E153）", e153_overall), ("难度阶梯中的遗传扰动（E187）", e187_genetic), ("细胞因子扰动（E187-Cui）", e187_cytokine)):
        lines += [f"## {title}", "", "| 分数 | 数据集数 | Spearman | 20% 复核效用 |", "|---|---:|---:|---:|"]
        for row in primary_rows(table).itertuples(index=False):
            lines.append(f"| {row.score} | {row.n_datasets} | {row.spearman:.4f} | {row.review_utility:.4f} |")
        lines.append("")
    lines += ["## 固定融合式相对幅度", "", "| 分析 | 指标 | 点估计 | 95% 区间 | 正向数据集 |", "|---|---|---:|---:|---:|"]
    merged = point.merge(ci, on=["analysis", "metric"], how="left")
    for row in merged.itertuples(index=False):
        lines.append(f"| {row.analysis} | {row.metric} | {row.point_estimate:.4f} | [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}] | {row.positive_datasets}/{row.n_datasets} |")
    lines += ["", "## 化学扰动边界", "", "E118 没有与 E206 同定义的完整 SafeConf 输入，E207 没有补造融合分数。三来源等权结果如下：", "", "| 分数 | Spearman | 20% 错误富集 | 20% 错误总量抓取 |", "|---|---:|---:|---:|"]
    for row in chemical.loc[chemical.source.eq("three_source_equal_macro")].itertuples(index=False):
        lines.append(f"| {row.score} | {row.spearman:.4f} | {row.top20_error_enrichment:.4f} | {row.top20_total_error_capture:.4f} |")
    lines += [
        "", "## 解释", "",
        "E207 只判定固定公式在历史场景中的稳健程度。若某个区间跨 0，就保留为未确认；不会通过删除数据集、调整权重或更换端点把它改成通过。E205 才是当前用于确认第二种模型结构的冻结实验。", "",
    ]
    atomic_text(OUT / "E207_REPORT.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    e153, e187, chemical, chemical_boot, manifest = validate_and_load()
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    e153_scored = add_scores(e153, ["dataset", "fold_id"])
    e153_metrics = batch_metrics(e153_scored, ["dataset", "fold_id"])

    e187_scored = add_scores(e187, ["dataset", "fold_id", "train_fraction", "setting"])
    e187_metrics = batch_metrics(e187_scored, ["dataset", "fold_id", "train_fraction", "setting"])
    genetic = e187_metrics.loc[e187_metrics.dataset.ne("Cui_direct41")].copy()
    cytokine = e187_metrics.loc[e187_metrics.dataset.eq("Cui_direct41")].copy()

    e153_overall = overall_from_metrics(e153_metrics)
    genetic_overall = overall_from_metrics(genetic)
    cytokine_overall = overall_from_metrics(cytokine)
    scenario = macro_table(e187_metrics, ["dataset", "setting", "train_fraction"])

    point = pd.concat([
        point_delta(e153_metrics, "E153_genetic_8study"),
        point_delta(genetic, "E187_genetic_difficulty"),
        point_delta(cytokine, "E187_cytokine_difficulty"),
    ], ignore_index=True)

    draws, summaries = [], []
    for source, frame, batches, label in (
        ("E153", e153, ["dataset", "fold_id"], "E153_genetic_8study"),
        ("E187-genetic", e187.loc[e187.dataset.ne("Cui_direct41")], ["dataset", "fold_id", "train_fraction", "setting"], "E187_genetic_difficulty"),
        ("E187-cytokine", e187.loc[e187.dataset.eq("Cui_direct41")], ["dataset", "fold_id", "train_fraction", "setting"], "E187_cytokine_difficulty"),
    ):
        d, s = bootstrap_delta(frame, batches, args.bootstrap, label)
        draws.append(d)
        summaries.append(s)

    ci = pd.concat(summaries, ignore_index=True)
    # Align bootstrap metric names with point_delta names.
    ci["metric"] = ci.metric.str.replace("delta_review_utility", "delta_review_utility", regex=False)
    all_draws = pd.concat(draws, ignore_index=True)

    atomic_csv(tables / "E207_INPUT_MANIFEST.csv", manifest)
    atomic_csv(tables / "E207_E153_BATCH_METRICS.csv", e153_metrics)
    atomic_csv(tables / "E207_E187_BATCH_METRICS.csv", e187_metrics)
    atomic_csv(tables / "E207_E187_SCENARIO_SUMMARY.csv", scenario)
    atomic_csv(tables / "E207_OVERALL_SUMMARY.csv", pd.concat([
        e153_overall.assign(analysis="E153_genetic_8study"),
        genetic_overall.assign(analysis="E187_genetic_difficulty"),
        cytokine_overall.assign(analysis="E187_cytokine_difficulty"),
    ], ignore_index=True))
    atomic_csv(tables / "E207_POINT_DELTAS.csv", point)
    atomic_csv(tables / "E207_BOOTSTRAP_SUMMARY.csv", ci)
    atomic_csv(tables / "E207_BOOTSTRAP_DRAWS.csv", all_draws)
    atomic_csv(tables / "E207_CHEMICAL_BOUNDARY.csv", chemical)

    write_report(e153_overall, genetic_overall, cytokine_overall, point, ci, chemical)
    status = {
        "experiment": "E207_fixed_formula_multimodal_stress",
        "status": "COMPLETE_RETROSPECTIVE",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "formula": "0.80*within_batch_rank(magnitude)+0.20*within_batch_rank(safeconf)",
        "weights_retuned": False,
        "independent_confirmation": False,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "perturbation cluster within dataset",
        "truth_role": "retrospective evaluation only; outcomes were previously released",
        "e205_confirmation_status": "PENDING_TRAINING",
    }
    atomic_json(OUT / "RUN_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(point.to_string(index=False))
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
