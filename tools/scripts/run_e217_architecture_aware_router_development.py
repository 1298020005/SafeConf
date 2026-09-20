#!/usr/bin/env python3
"""Develop a low-complexity architecture-aware router on released data.

E217 is explicitly method development.  It uses the released E153
cross-architecture scGPT/GEARS task table and the released E201 homogeneous
TxPert seed-family table.  E205 target truth is not an input.
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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
E201_DEFAULT = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
)
OUTPUT_DEFAULT = ROOT / "docs/实验结果/E217_architecture_aware_router_20260920"
E153_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
E201_SHA256 = "4a02d132cae1605a2f6f54bee8912f45e835d1fc3936e6cda0fcc98e2c6bcd35"
GRID = (0.0, 0.125, 0.25, 0.50, 0.75, 1.0)
BUDGETS = (0.05, 0.10, 0.20, 0.30)
MASTER_SEED = 217_202_609


class DevelopmentFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e153-input", type=Path, required=True)
    parser.add_argument("--e201-input", type=Path, default=E201_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--overwrite-development-output", action="store_true")
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
    values = np.asarray(values, dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        raise DevelopmentFailure("percentile input is invalid")
    return rankdata(values, method="average") / len(values)


def spearman(score: np.ndarray, outcome: np.ndarray) -> float:
    left, right = percentile(score), percentile(outcome)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def review_utility(score: np.ndarray, outcome: np.ndarray, budget: float) -> float:
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    k = max(1, int(math.ceil(budget * len(score))))
    selected = np.argsort(-score, kind="mergesort")[:k]
    oracle = np.argsort(-outcome, kind="mergesort")[:k]
    baseline = float(outcome.mean())
    denominator = float(outcome[oracle].mean()) - baseline
    if denominator <= 1e-15:
        return float("nan")
    return float((outcome[selected].mean() - baseline) / denominator)


def score(magnitude: np.ndarray, safeconf: np.ndarray, disagreement: np.ndarray,
          lambda_safeconf: float, lambda_disagreement: float) -> np.ndarray:
    """Magnitude anchor with upward-only evidence corrections."""
    magnitude = np.asarray(magnitude, float)
    safeconf = np.asarray(safeconf, float)
    disagreement = np.asarray(disagreement, float)
    return (
        magnitude
        + lambda_safeconf * np.maximum(safeconf - magnitude, 0.0)
        + lambda_disagreement * np.maximum(disagreement - magnitude, 0.0)
    )


def load_e153(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != E153_SHA256:
        raise DevelopmentFailure("E153 task input is missing or changed")
    frame = pd.read_csv(path)
    required = {
        "dataset", "fold_id", "perturbation",
        "error_two_predictor_mean_rmse", "safeconf_calibrated_pair_risk",
        "risk_model_disagreement", "baseline_predicted_magnitude",
    }
    if not required.issubset(frame.columns) or len(frame) != 3_465:
        raise DevelopmentFailure("E153 task contract changed")
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["m"] = percentile(block.baseline_predicted_magnitude)
        block["s"] = percentile(block.safeconf_calibrated_pair_risk)
        block["d"] = percentile(block.risk_model_disagreement)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if result.dataset.nunique() != 8 or result.fold_id.nunique() != 34:
        raise DevelopmentFailure("E153 study or fold count changed")
    return result


def load_e201(path: Path) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != E201_SHA256:
        raise DevelopmentFailure("E201 task input is missing or changed")
    frame = pd.read_csv(path)
    required = {
        "target", "analysis_stratum", "predicted_magnitude",
        "safeconf_e201_risk", "family_disagreement", "family_rms_error",
    }
    if not required.issubset(frame.columns):
        raise DevelopmentFailure("E201 task contract changed")
    frame = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    blocks = []
    for _, block in frame.groupby("target", sort=True):
        block = block.copy()
        block["m"] = percentile(block.predicted_magnitude)
        block["s"] = percentile(block.safeconf_e201_risk)
        block["d"] = percentile(block.family_disagreement)
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if len(result) != 1_808 or result.target.nunique() != 4:
        raise DevelopmentFailure("E201 primary task identity changed")
    return result


def rerank_e153_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute deployment percentiles after a pretruth setting restriction."""
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["m"] = percentile(block.baseline_predicted_magnitude)
        block["s"] = percentile(block.safeconf_calibrated_pair_risk)
        block["d"] = percentile(block.risk_model_disagreement)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def block_metrics(block: pd.DataFrame, lambda_s: float, lambda_d: float,
                  outcome_column: str) -> dict[str, float]:
    magnitude = block.m.to_numpy(float)
    candidate = score(
        magnitude,
        block.s.to_numpy(float),
        block.d.to_numpy(float),
        lambda_s,
        lambda_d,
    )
    outcome = block[outcome_column].to_numpy(float)
    row = {
        "delta_spearman": spearman(candidate, outcome) - spearman(magnitude, outcome)
    }
    for budget in BUDGETS:
        label = f"delta_utility_{int(100 * budget):02d}"
        row[label] = review_utility(candidate, outcome, budget) - review_utility(
            magnitude, outcome, budget
        )
    return row


def e153_study_metrics(frame: pd.DataFrame, lambda_s: float,
                       lambda_d: float) -> pd.DataFrame:
    rows = []
    for dataset, study in frame.groupby("dataset", sort=True):
        folds = [
            block_metrics(block, lambda_s, lambda_d, "error_two_predictor_mean_rmse")
            for _, block in study.groupby("fold_id", sort=True)
        ]
        values = pd.DataFrame(folds)
        row = {
            "dataset": dataset,
            "lambda_safeconf": lambda_s,
            "lambda_disagreement": lambda_d,
            "n_tasks": len(study),
            "n_folds": study.fold_id.nunique(),
        }
        row.update({column: float(values[column].mean()) for column in values.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_landscape(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    rows = []
    utility_columns = [f"delta_utility_{int(100 * value):02d}" for value in BUDGETS]
    for lambda_s in GRID:
        for lambda_d in GRID:
            studies = e153_study_metrics(frame, lambda_s, lambda_d)
            row = {
                "lambda_safeconf": lambda_s,
                "lambda_disagreement": lambda_d,
                "mean_delta_spearman": float(studies.delta_spearman.mean()),
                "positive_spearman_studies": int((studies.delta_spearman > 0).sum()),
                "mean_delta_utility_all_budgets": float(
                    studies[utility_columns].to_numpy(float).mean()
                ),
                "positive_study_budget_cells": int(
                    (studies[utility_columns].to_numpy(float) > 0).sum()
                ),
                "minimum_study_delta_spearman": float(studies.delta_spearman.min()),
            }
            for column in utility_columns:
                row[f"mean_{column}"] = float(studies[column].mean())
                row[f"positive_{column}_studies"] = int((studies[column] > 0).sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    selected = table.sort_values(
        [
            "positive_spearman_studies", "positive_study_budget_cells",
            "mean_delta_utility_all_budgets", "mean_delta_spearman",
            "lambda_safeconf", "lambda_disagreement",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    return table, (
        float(selected.lambda_safeconf), float(selected.lambda_disagreement)
    )


def lodo_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    utility_columns = [f"delta_utility_{int(100 * value):02d}" for value in BUDGETS]
    for heldout in sorted(frame.dataset.unique()):
        landscape, selected = parameter_landscape(frame.loc[frame.dataset.ne(heldout)])
        held = e153_study_metrics(
            frame.loc[frame.dataset.eq(heldout)], *selected
        ).iloc[0]
        rows.append(
            {
                "heldout_dataset": heldout,
                "lambda_safeconf": selected[0],
                "lambda_disagreement": selected[1],
                "heldout_truth_used_for_selection": False,
                "delta_spearman": float(held.delta_spearman),
                **{column: float(held[column]) for column in utility_columns},
            }
        )
    return pd.DataFrame(rows)


def e201_contrast(frame: pd.DataFrame, cross_parameters: tuple[float, float]) -> pd.DataFrame:
    rows = []
    formulas = {
        "homogeneous_seed_rule": (0.125, 0.0),
        "mixed_cross_architecture_rule": cross_parameters,
    }
    for target, block in frame.groupby("target", sort=True):
        for name, parameters in formulas.items():
            rows.append(
                {
                    "target": target,
                    "formula": name,
                    "lambda_safeconf": parameters[0],
                    "lambda_disagreement": parameters[1],
                    **block_metrics(block, *parameters, "family_rms_error"),
                }
            )
    return pd.DataFrame(rows)


def study_bootstrap(studies: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    if n_bootstrap < 1_000:
        raise DevelopmentFailure("study bootstrap requires at least 1000 draws")
    rng = np.random.default_rng(MASTER_SEED)
    metrics = ["delta_spearman"] + [
        f"delta_utility_{int(100 * value):02d}" for value in BUDGETS
    ]
    values = studies[metrics].to_numpy(float)
    rows = []
    for draw in range(n_bootstrap):
        chosen = rng.integers(0, len(values), len(values))
        mean = values[chosen].mean(axis=0)
        rows.append({"draw": draw, **dict(zip(metrics, mean))})
    draws = pd.DataFrame(rows)
    summary = []
    for metric in metrics:
        summary.append(
            {
                "metric": metric,
                "estimate": float(studies[metric].mean()),
                "ci95_lower": float(draws[metric].quantile(0.025)),
                "ci95_upper": float(draws[metric].quantile(0.975)),
                "positive_studies": int((studies[metric] > 0).sum()),
                "n_studies": len(studies),
            }
        )
    return pd.DataFrame(summary)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    result_markers = (
        output / "RUN_STATUS.json",
        output / "E217_REPORT.md",
        output / "tables/E217_PARAMETER_LANDSCAPE.csv",
    )
    if any(path.exists() for path in result_markers) and not args.overwrite_development_output:
        raise DevelopmentFailure(f"refusing to overwrite existing E217 results: {output}")
    e153, e201 = load_e153(args.e153_input), load_e201(args.e201_input)
    landscape, selected = parameter_landscape(e153)
    studies = e153_study_metrics(e153, *selected)
    context_e153 = rerank_e153_blocks(
        e153.loc[
            e153.setting.astype(str).isin(
                ("context_unseen", "context_unseen_row")
            )
        ].copy()
    )
    if context_e153.dataset.nunique() != 8:
        raise DevelopmentFailure("E153 context-holdout subset lost a study")
    context_landscape, context_selected = parameter_landscape(context_e153)
    context_studies = e153_study_metrics(context_e153, *context_selected)
    lodo = lodo_selection(e153)
    contrast = e201_contrast(e201, selected)
    bootstrap = study_bootstrap(studies, args.bootstrap)
    context_bootstrap = study_bootstrap(context_studies, args.bootstrap)

    tables = output / "tables"
    atomic_csv(tables / "E217_PARAMETER_LANDSCAPE.csv", landscape)
    atomic_csv(
        tables / "E217_CONTEXT_HOLDOUT_PARAMETER_LANDSCAPE.csv",
        context_landscape,
    )
    atomic_csv(tables / "E217_CROSS_ARCH_STUDY_RESULTS.csv", studies)
    atomic_csv(
        tables / "E217_CONTEXT_HOLDOUT_STUDY_RESULTS.csv", context_studies
    )
    atomic_csv(tables / "E217_LODO_SELECTION.csv", lodo)
    atomic_csv(tables / "E217_E201_ARCHITECTURE_CONTRAST.csv", contrast)
    atomic_csv(tables / "E217_STUDY_BOOTSTRAP_SUMMARY.csv", bootstrap)
    atomic_csv(
        tables / "E217_CONTEXT_HOLDOUT_BOOTSTRAP_SUMMARY.csv",
        context_bootstrap,
    )

    utility_columns = [f"delta_utility_{int(100 * value):02d}" for value in BUDGETS]
    same_arch = contrast.loc[contrast.formula.eq("homogeneous_seed_rule")]
    cross_on_same = contrast.loc[
        contrast.formula.eq("mixed_cross_architecture_rule")
    ]
    status = {
        "experiment": "E217_architecture_aware_router_development",
        "evidence_identity": "released_data_method_development",
        "cross_architecture_formula": (
            "m + 0.50*max(s-m,0) + 0.125*max(d-m,0)"
        ),
        "cross_architecture_context_holdout_formula": (
            "m + 0.125*max(s-m,0) + 0.125*max(d-m,0)"
        ),
        "homogeneous_seed_formula": "m + 0.125*max(s-m,0)",
        "selected_cross_parameters": {
            "lambda_safeconf": selected[0],
            "lambda_disagreement": selected[1],
        },
        "selected_cross_context_parameters": {
            "lambda_safeconf": context_selected[0],
            "lambda_disagreement": context_selected[1],
        },
        "cross_architecture_positive_spearman_studies": int(
            (studies.delta_spearman > 0).sum()
        ),
        "cross_architecture_n_studies": len(studies),
        "cross_architecture_mean_delta_spearman": float(
            studies.delta_spearman.mean()
        ),
        "cross_architecture_mean_delta_utilities": {
            column: float(studies[column].mean()) for column in utility_columns
        },
        "cross_context_positive_spearman_studies": int(
            (context_studies.delta_spearman > 0).sum()
        ),
        "cross_context_mean_delta_spearman": float(
            context_studies.delta_spearman.mean()
        ),
        "cross_context_mean_delta_utilities": {
            column: float(context_studies[column].mean())
            for column in utility_columns
        },
        "lodo_positive_spearman_studies": int((lodo.delta_spearman > 0).sum()),
        "lodo_mean_delta_spearman": float(lodo.delta_spearman.mean()),
        "e201_homogeneous_mean_delta_spearman": float(
            same_arch.delta_spearman.mean()
        ),
        "e201_cross_rule_mean_delta_spearman": float(
            cross_on_same.delta_spearman.mean()
        ),
        "e205_truth_used": False,
        "e208_truth_used": False,
        "claim_role": "pretruth formula development; E205 confirmation required",
        "input_hashes": {
            "e153": sha256_file(args.e153_input),
            "e201": sha256_file(args.e201_input),
            "script": sha256_file(Path(__file__)),
        },
    }
    atomic_text(
        output / "RUN_STATUS.json",
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
    )
    report = rf"""# E217｜架构感知的单向风险修正

证据身份：已解封历史数据上的方法开发。E205、E208 目标真值均未使用。

## 发现

E153 的八个研究使用跨结构预测器，E201 使用同一 TxPert 结构的四个随机种子。
两类模型家族需要不同修正强度，不能继续假设一组权重适用于所有家族。

跨结构开发公式为：

\[
R_{{cross}}=m+0.50[s-m]_+ +0.125[d-m]_+ .
\]

其中 E205 对应的“整个细胞背景未见”使用更窄的历史子场景，冻结为：

\[
R_{{cross,context}}=m+0.125[s-m]_+ +0.125[d-m]_+ .
\]

同结构种子家族保留弱修正：

\[
R_{{seed}}=m+0.125[s-m]_+ .
\]

其中 \(m\) 为批次内预测幅度百分位，\(s\) 为 SafeConf 百分位，\(d\) 为模型分歧
百分位，\([x]_+=\max(x,0)\)。SafeConf 或分歧低于幅度时不下调已有风险。

## 历史开发结果

- 跨结构公式相对幅度的八研究平均 Spearman 增量：{status['cross_architecture_mean_delta_spearman']:+.4f}；
- Spearman 正向研究：{status['cross_architecture_positive_spearman_studies']}/8；
- 整背景留出专用公式平均 Spearman 增量：{status['cross_context_mean_delta_spearman']:+.4f}，正向 {status['cross_context_positive_spearman_studies']}/8；
- 留一研究重新选参后的平均增量：{status['lodo_mean_delta_spearman']:+.4f}，正向 {status['lodo_positive_spearman_studies']}/8；
- 同结构 E201 使用弱修正时平均 Spearman 增量：{status['e201_homogeneous_mean_delta_spearman']:+.4f}；
- 把跨结构强修正规则误用到 E201 时平均增量：{status['e201_cross_rule_mean_delta_spearman']:+.4f}。

完整参数地形、多预算效用、逐研究结果和 study-level bootstrap 区间见 `tables/`。

## 机制解释与下一步

预测幅度保留为所有场景的锚点。跨结构模型提供更独立的证据来源，因此允许 SafeConf
和分歧作较强的向上修正；同结构随机种子的误差相关性更高，只允许弱修正，避免重复
证据被多次计权。

以上规律来自历史开发数据，不能作为独立确认。E205 恰好包含 TxPert-STRING-GAT 与
TxPert-Exphormer 两类结构，应在目标真值打开前冻结本公式并一次性评价。E205 若不能
复现，公式不得按 E205 结果重新调权。
"""
    atomic_text(output / "E217_REPORT.md", report)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
