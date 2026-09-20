#!/usr/bin/env python3
"""Compare multiple SafeConf routing hypotheses without target-study tuning.

E218 is a retrospective development audit.  Each learned candidate is trained
on seven E153 studies and evaluated on the eighth study.  E201 is used only as
a scenario-transport stress test.  E205/E208 target truth is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
E153_DEFAULT = (
    ROOT / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables"
    / "E153_ABSOLUTE_TASK_INPUT.csv"
)
E201_DEFAULT = (
    ROOT / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
)
OUTPUT_DEFAULT = ROOT / "docs/实验结果/E218_multi_hypothesis_validation_20260920"
E153_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
E201_SHA256 = "4a02d132cae1605a2f6f54bee8912f45e835d1fc3936e6cda0fcc98e2c6bcd35"
GRID = (0.0, 0.125, 0.25, 0.50, 0.75, 1.0)
BUDGETS = (0.05, 0.10, 0.20, 0.30)
SEED = 218_202_609


class AuditFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e153", type=Path, default=E153_DEFAULT)
    parser.add_argument("--e201", type=Path, default=E201_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
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
    atomic_text(path, frame.to_csv(index=False))


def percentile(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise AuditFailure("rank input is invalid")
    return rankdata(values, method="average") / len(values)


def spearman(score: np.ndarray, outcome: np.ndarray) -> float:
    a, b = percentile(score), percentile(outcome)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def utility(score: np.ndarray, outcome: np.ndarray, budget: float) -> float:
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    k = max(1, int(math.ceil(budget * len(score))))
    selected = np.argsort(-score, kind="mergesort")[:k]
    oracle = np.argsort(-outcome, kind="mergesort")[:k]
    baseline = float(outcome.mean())
    denominator = float(outcome[oracle].mean()) - baseline
    if denominator <= 1e-15:
        return float("nan")
    return float((outcome[selected].mean() - baseline) / denominator)


def canonical_setting(value: str) -> str:
    mapping = {
        "random_missing_pair": "random_pair",
        "random_seen_pair": "random_pair",
        "perturbation_unseen_column": "perturbation_unseen",
        "perturbation_unseen": "perturbation_unseen",
        "context_unseen_row": "context_unseen",
        "context_unseen": "context_unseen",
        "context_and_perturbation_unseen": "context_and_perturbation_unseen",
    }
    if value not in mapping:
        raise AuditFailure(f"unknown setting {value}")
    return mapping[value]


def load_e153(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != E153_SHA256:
        raise AuditFailure("E153 input missing or changed")
    frame = pd.read_csv(path)
    required = {
        "dataset", "fold_id", "task_id", "setting", "perturbation",
        "error_two_predictor_mean_rmse", "safeconf_calibrated_pair_risk",
        "baseline_predicted_magnitude", "risk_model_disagreement",
    }
    if not required.issubset(frame.columns) or len(frame) != 3465:
        raise AuditFailure("E153 schema changed")
    frame = frame.copy()
    frame["setting_canonical"] = frame.setting.astype(str).map(canonical_setting)
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["m"] = percentile(block.baseline_predicted_magnitude)
        block["s"] = percentile(block.safeconf_calibrated_pair_risk)
        block["d"] = percentile(block.risk_model_disagreement)
        block["y"] = percentile(block.error_two_predictor_mean_rmse)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if result.dataset.nunique() != 8 or result.groupby(["dataset", "fold_id"]).ngroups != 34:
        raise AuditFailure("E153 identity changed")
    return result


def load_e201(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != E201_SHA256:
        raise AuditFailure("E201 input missing or changed")
    frame = pd.read_csv(path)
    frame = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    blocks = []
    for _, block in frame.groupby("target", sort=True):
        block = block.copy()
        block["m"] = percentile(block.predicted_magnitude)
        block["s"] = percentile(block.safeconf_e201_risk)
        block["d"] = percentile(block.family_disagreement)
        block["y"] = percentile(block.family_rms_error)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if len(result) != 1808 or result.target.nunique() != 4:
        raise AuditFailure("E201 identity changed")
    return result


def upward_score(block: pd.DataFrame, lambda_s: float, lambda_d: float) -> np.ndarray:
    m = block.m.to_numpy(float)
    return m + lambda_s * np.maximum(block.s.to_numpy(float) - m, 0.0) + lambda_d * np.maximum(block.d.to_numpy(float) - m, 0.0)


def fixed_candidates(block: pd.DataFrame) -> dict[str, np.ndarray]:
    m, s, d = (block[c].to_numpy(float) for c in ("m", "s", "d"))
    return {
        "magnitude": m,
        "safeconf": s,
        "disagreement": d,
        "fixed_80m_20s": 0.8 * m + 0.2 * s,
        "one_sided_s025": m + 0.25 * np.maximum(s - m, 0.0),
        "maximum_alarm": np.maximum.reduce([m, s, d]),
        "e217_cross_arch": upward_score(block, 0.50, 0.125),
        "e217_context": upward_score(block, 0.125, 0.125),
        "e217_same_arch": upward_score(block, 0.125, 0.0),
    }


def block_metrics(score: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    row = {"spearman": spearman(score, outcome)}
    for budget in BUDGETS:
        row[f"utility_{int(100 * budget):02d}"] = utility(score, outcome, budget)
    return row


def aggregate_study(frame: pd.DataFrame, score_column: str, method: str) -> dict[str, float | str | int]:
    rows = []
    for _, block in frame.groupby("fold_id", sort=True):
        rows.append(block_metrics(block[score_column].to_numpy(float), block.error_two_predictor_mean_rmse.to_numpy(float)))
    values = pd.DataFrame(rows)
    return {
        "dataset": str(frame.dataset.iloc[0]),
        "method": method,
        "n_tasks": len(frame),
        "n_folds": frame.fold_id.nunique(),
        **{column: float(values[column].mean()) for column in values.columns},
    }


def training_objective(frame: pd.DataFrame, lambda_s: float, lambda_d: float, setting: str | None = None) -> tuple:
    subset = frame if setting is None else frame.loc[frame.setting_canonical.eq(setting)]
    rows = []
    for dataset, study in subset.groupby("dataset", sort=True):
        fold_rows = []
        for _, block in study.groupby("fold_id", sort=True):
            if len(block) < 4:
                continue
            metrics = block_metrics(upward_score(block, lambda_s, lambda_d), block.error_two_predictor_mean_rmse.to_numpy(float))
            base = block_metrics(block.m.to_numpy(float), block.error_two_predictor_mean_rmse.to_numpy(float))
            fold_rows.append({key: metrics[key] - base[key] for key in metrics})
        if fold_rows:
            values = pd.DataFrame(fold_rows)
            rows.append({"dataset": dataset, **{c: float(values[c].mean()) for c in values}})
    studies = pd.DataFrame(rows)
    if studies.empty:
        return (-1, -1, -math.inf, -math.inf, -lambda_s, -lambda_d)
    utility_columns = [f"utility_{int(100*b):02d}" for b in BUDGETS]
    return (
        int((studies.spearman > 0).sum()),
        int((studies[utility_columns].to_numpy(float) > 0).sum()),
        float(studies[utility_columns].to_numpy(float).mean()),
        float(studies.spearman.mean()),
        -lambda_s,
        -lambda_d,
    )


def select_grid(training: pd.DataFrame, setting: str | None = None) -> tuple[float, float]:
    candidates = []
    for lambda_s in GRID:
        for lambda_d in GRID:
            candidates.append((training_objective(training, lambda_s, lambda_d, setting), lambda_s, lambda_d))
    _, lambda_s, lambda_d = max(candidates, key=lambda item: item[0])
    return lambda_s, lambda_d


def fit_nonnegative(training: pd.DataFrame) -> tuple[float, float]:
    """Fit upward corrections with equal total weight for every study."""
    m, s, d, y = (training[c].to_numpy(float) for c in ("m", "s", "d", "y"))
    x = np.column_stack([np.maximum(s - m, 0.0), np.maximum(d - m, 0.0)])
    residual = y - m
    counts = training.groupby("dataset").dataset.transform("size").to_numpy(float)
    weights = 1.0 / np.sqrt(counts)
    fit = lsq_linear(x * weights[:, None], residual * weights, bounds=(0.0, 1.0))
    return float(fit.x[0]), float(fit.x[1])


def materialize_e153(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_blocks = []
    weight_rows = []
    for heldout in sorted(frame.dataset.unique()):
        training = frame.loc[frame.dataset.ne(heldout)]
        held = frame.loc[frame.dataset.eq(heldout)].copy()
        global_weights = select_grid(training)
        nnls_weights = fit_nonnegative(training)
        setting_weights = {
            setting: select_grid(training, setting)
            for setting in sorted(held.setting_canonical.unique())
        }
        held["lodo_global_grid"] = upward_score(held, *global_weights)
        held["lodo_nonnegative"] = upward_score(held, *nnls_weights)
        pieces = []
        for setting, block in held.groupby("setting_canonical", sort=False):
            values = pd.Series(upward_score(block, *setting_weights[setting]), index=block.index)
            pieces.append(values)
        held["lodo_setting_grid"] = pd.concat(pieces).sort_index()
        for _, block in held.groupby("fold_id", sort=False):
            for name, values in fixed_candidates(block).items():
                held.loc[block.index, name] = values
        scored_blocks.append(held)
        for label, weights in [("global_grid", global_weights), ("nonnegative", nnls_weights)]:
            weight_rows.append({"heldout_dataset": heldout, "method": label, "setting": "ALL", "lambda_safeconf": weights[0], "lambda_disagreement": weights[1], "heldout_truth_used": False})
        for setting, weights in setting_weights.items():
            weight_rows.append({"heldout_dataset": heldout, "method": "setting_grid", "setting": setting, "lambda_safeconf": weights[0], "lambda_disagreement": weights[1], "heldout_truth_used": False})
    return pd.concat(scored_blocks, ignore_index=True), pd.DataFrame(weight_rows)


def e153_results(frame: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "magnitude", "safeconf", "disagreement", "fixed_80m_20s",
        "one_sided_s025", "maximum_alarm", "e217_cross_arch",
        "e217_context", "e217_same_arch", "lodo_global_grid",
        "lodo_setting_grid", "lodo_nonnegative",
    ]
    rows = []
    for _, study in frame.groupby("dataset", sort=True):
        for method in methods:
            rows.append(aggregate_study(study, method, method))
    result = pd.DataFrame(rows)
    base = result.loc[result.method.eq("magnitude"), ["dataset", "spearman", *[f"utility_{int(100*b):02d}" for b in BUDGETS]]].copy()
    base = base.rename(columns={c: f"magnitude_{c}" for c in base.columns if c != "dataset"})
    result = result.merge(base, on="dataset", validate="many_to_one")
    result["delta_spearman"] = result.spearman - result.magnitude_spearman
    for budget in BUDGETS:
        label = f"utility_{int(100*budget):02d}"
        result[f"delta_{label}"] = result[label] - result[f"magnitude_{label}"]
    return result


def bootstrap_summary(results: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    if n_bootstrap < 1000:
        raise AuditFailure("bootstrap must be >=1000")
    metrics = ["delta_spearman", *[f"delta_utility_{int(100*b):02d}" for b in BUDGETS]]
    rng = np.random.default_rng(SEED)
    summary = []
    for method, block in results.groupby("method", sort=True):
        values = block[metrics].to_numpy(float)
        method_draws = np.empty((n_bootstrap, len(metrics)), float)
        for draw in range(n_bootstrap):
            chosen = rng.integers(0, len(values), len(values))
            method_draws[draw] = values[chosen].mean(axis=0)
        for row_index, metric in enumerate(metrics):
            vector = method_draws[:, row_index]
            summary.append({
                "method": method,
                "metric": metric,
                "estimate": float(block[metric].mean()),
                "ci95_lower": float(np.quantile(vector, 0.025)),
                "ci95_upper": float(np.quantile(vector, 0.975)),
                "positive_studies": int((block[metric] > 0).sum()),
                "n_studies": len(block),
            })
    return pd.DataFrame(summary)


def e201_stress(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, block in frame.groupby("target", sort=True):
        outcome = block.family_rms_error.to_numpy(float)
        for method, score in fixed_candidates(block).items():
            rows.append({"target": target, "method": method, "n_tasks": len(block), **block_metrics(score, outcome)})
    result = pd.DataFrame(rows)
    base = result.loc[result.method.eq("magnitude"), ["target", "spearman", *[f"utility_{int(100*b):02d}" for b in BUDGETS]]].copy()
    base = base.rename(columns={c: f"magnitude_{c}" for c in base.columns if c != "target"})
    result = result.merge(base, on="target", validate="many_to_one")
    result["delta_spearman"] = result.spearman - result.magnitude_spearman
    for budget in BUDGETS:
        label = f"utility_{int(100*budget):02d}"
        result[f"delta_{label}"] = result[label] - result[f"magnitude_{label}"]
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise AuditFailure(f"refusing to overwrite {output}")
    e153 = load_e153(args.e153.resolve())
    e201 = load_e201(args.e201.resolve())
    scored, weights = materialize_e153(e153)
    study_results = e153_results(scored)
    summary = bootstrap_summary(study_results, args.bootstrap)
    stress = e201_stress(e201)
    tables = output / "tables"
    atomic_csv(tables / "E218_E153_STUDY_RESULTS.csv", study_results)
    atomic_csv(tables / "E218_E153_BOOTSTRAP_SUMMARY.csv", summary)
    atomic_csv(tables / "E218_LODO_WEIGHTS.csv", weights)
    atomic_csv(tables / "E218_E201_SCENARIO_STRESS.csv", stress)

    key = summary.loc[summary.metric.eq("delta_spearman")].sort_values("estimate", ascending=False)
    best_lodo = key.loc[key.method.str.startswith("lodo_")].iloc[0]
    fixed = key.loc[key.method.eq("e217_cross_arch")].iloc[0]
    same = stress.loc[stress.method.eq("e217_same_arch")]
    cross_on_same = stress.loc[stress.method.eq("e217_cross_arch")]
    status = {
        "experiment": "E218_multi_hypothesis_validation",
        "evidence_identity": "retrospective_nested_leave_one_study_out_development",
        "n_e153_tasks": len(e153),
        "n_e153_studies": int(e153.dataset.nunique()),
        "n_e153_folds": int(e153.groupby(["dataset", "fold_id"]).ngroups),
        "n_e153_perturbation_clusters": int(e153.perturbation.nunique()),
        "n_e201_tasks": len(e201),
        "n_e201_targets": int(e201.target.nunique()),
        "candidate_methods": int(study_results.method.nunique()),
        "best_lodo_method": str(best_lodo.method),
        "best_lodo_mean_delta_spearman": float(best_lodo.estimate),
        "best_lodo_ci95": [float(best_lodo.ci95_lower), float(best_lodo.ci95_upper)],
        "fixed_cross_arch_delta_spearman": float(fixed.estimate),
        "fixed_cross_arch_ci95": [float(fixed.ci95_lower), float(fixed.ci95_upper)],
        "e201_same_arch_weak_mean_delta_spearman": float(same.delta_spearman.mean()),
        "e201_cross_arch_rule_mean_delta_spearman": float(cross_on_same.delta_spearman.mean()),
        "e205_truth_used": False,
        "e208_truth_used": False,
        "input_hashes": {"e153": E153_SHA256, "e201": E201_SHA256},
    }
    atomic_text(output / "RUN_STATUS.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    report = [
        "# E218｜多假设统一验证\n",
        "本实验在 E153 八研究上每次留出整个研究，学习型方法只用其余七个研究选参。E201 只用于检查跨结构规则误用到同结构种子时会发生什么。E205/E208 真值未读取。\n",
        "## 关键结果\n",
        f"- 候选方法：{study_results.method.nunique()} 类；E153：{len(e153)} 任务、{e153.dataset.nunique()} 研究、{e153.groupby(['dataset','fold_id']).ngroups} 个 fold。\n",
        f"- 最好的留一研究方法：`{best_lodo.method}`，平均 ΔSpearman={best_lodo.estimate:+.4f}，95% 研究自举区间 [{best_lodo.ci95_lower:+.4f}, {best_lodo.ci95_upper:+.4f}]。\n",
        f"- 冻结跨结构规则：平均 ΔSpearman={fixed.estimate:+.4f}，95% 区间 [{fixed.ci95_lower:+.4f}, {fixed.ci95_upper:+.4f}]。\n",
        f"- 同结构 E201 弱修正：平均 ΔSpearman={same.delta_spearman.mean():+.4f}；误用跨结构强修正：{cross_on_same.delta_spearman.mean():+.4f}。\n",
        "\n## 证据边界\n",
        "E218 用于候选方法比较和机制检查，不是新外部确证。最终确认由真值仍封存的 E205 完成；Jiang24 E216 因上游预测未超过 no-change，不用于声称路由有实用价值。\n",
    ]
    atomic_text(output / "E218_REPORT.md", "".join(report))
    atomic_text(output / "ANALYSIS_CONTRACT.md", __doc__.strip() + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
