#!/usr/bin/env python3
"""E212: audit one-sided SafeConf correction parameters on released data.

This is method development.  It compares the same parameter family on E201
whole-context holdout and the eight released E153 studies, then performs
leave-one-study-out parameter selection.  It cannot replace E205/E208.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
E201 = (
    ROOT
    / "docs/实验结果/E201_txpert_multitarget_retraining_20260802"
    / "formal_core_evaluation/tables/E201_TASK_METRICS.csv"
)
E153 = (
    ROOT
    / "docs/实验结果/E153_eight_study_formal_meta_20260714/tables"
    / "E153_ABSOLUTE_TASK_INPUT.csv"
)
OUTPUT = ROOT / "docs/实验结果/E212_parameter_landscape_20260917"
E201_SHA256 = "4a02d132cae1605a2f6f54bee8912f45e835d1fc3936e6cda0fcc98e2c6bcd35"
E153_SHA256 = "b75f5edae0bb585ba5ff18aecafcc2389b0f05fd5cc86b36960afb4b62e4a15a"
LAMBDAS = tuple(float(value) for value in np.round(np.arange(0.0, 1.0001, 0.025), 3))


class ParameterAuditError(RuntimeError):
    pass


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
        raise ParameterAuditError("percentile input is invalid")
    return rankdata(values, method="average") / len(values)


def spearman(score: np.ndarray, outcome: np.ndarray) -> float:
    left, right = percentile(score), percentile(outcome)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def tie_aware_selected_mean(
    score: np.ndarray, outcome: np.ndarray, budget: float = 0.20
) -> float:
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    k = max(1, int(math.ceil(budget * len(score))))
    threshold = np.sort(score)[-k]
    above, tied = score > threshold, score == threshold
    remaining = k - int(above.sum())
    return float((outcome[above].sum() + remaining * outcome[tied].mean()) / k)


def review_utility(score: np.ndarray, outcome: np.ndarray) -> float:
    selected = tie_aware_selected_mean(score, outcome)
    oracle = tie_aware_selected_mean(outcome, outcome)
    random = float(np.mean(outcome))
    denominator = oracle - random
    return float((selected - random) / denominator) if denominator > 1e-15 else float("nan")


def one_sided(magnitude: np.ndarray, safeconf: np.ndarray, value: float) -> np.ndarray:
    return magnitude + value * np.maximum(safeconf - magnitude, 0.0)


def validate_e201() -> pd.DataFrame:
    if sha256_file(E201) != E201_SHA256:
        raise ParameterAuditError("E201 input hash changed")
    frame = pd.read_csv(E201)
    required = {
        "target",
        "analysis_stratum",
        "predicted_magnitude",
        "safeconf_e201_risk",
        "family_rms_error",
    }
    if not required.issubset(frame.columns):
        raise ParameterAuditError("E201 columns changed")
    frame = frame.loc[frame.analysis_stratum.eq("primary_ge30")].copy()
    if len(frame) != 1_808 or frame.target.nunique() != 4:
        raise ParameterAuditError("E201 task identity changed")
    blocks = []
    for _, block in frame.groupby("target", sort=True):
        block = block.copy()
        block["m"] = percentile(block.predicted_magnitude)
        block["s"] = percentile(block.safeconf_e201_risk)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


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
        raise ParameterAuditError(f"unknown setting: {value}")
    return mapping[value]


def validate_e153() -> pd.DataFrame:
    if sha256_file(E153) != E153_SHA256:
        raise ParameterAuditError("E153 input hash changed")
    frame = pd.read_csv(E153)
    required = {
        "dataset",
        "fold_id",
        "setting",
        "baseline_predicted_magnitude",
        "safeconf_calibrated_pair_risk",
        "error_two_predictor_mean_rmse",
    }
    if not required.issubset(frame.columns):
        raise ParameterAuditError("E153 columns changed")
    if len(frame) != 3_465 or frame.dataset.nunique() != 8:
        raise ParameterAuditError("E153 task identity changed")
    frame = frame.copy()
    frame["setting_canonical"] = frame.setting.astype(str).map(canonical_setting)
    blocks = []
    for _, block in frame.groupby(["dataset", "fold_id"], sort=True):
        block = block.copy()
        block["m"] = percentile(block.baseline_predicted_magnitude)
        block["s"] = percentile(block.safeconf_calibrated_pair_risk)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def batch_delta(block: pd.DataFrame, value: float, outcome: str) -> tuple[float, float]:
    magnitude = block.m.to_numpy(float)
    safeconf = block.s.to_numpy(float)
    truth = block[outcome].to_numpy(float)
    candidate = one_sided(magnitude, safeconf, value)
    return (
        spearman(candidate, truth) - spearman(magnitude, truth),
        review_utility(candidate, truth) - review_utility(magnitude, truth),
    )


def e201_sweep(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, target_rows = [], []
    for value in LAMBDAS:
        values = []
        for target, block in frame.groupby("target", sort=True):
            delta_s, delta_u = batch_delta(block, value, "family_rms_error")
            values.append((delta_s, delta_u))
            target_rows.append(
                {
                    "lambda": value,
                    "target": target,
                    "delta_spearman": delta_s,
                    "delta_utility20": delta_u,
                }
            )
        values = np.asarray(values)
        rows.append(
            {
                "lambda": value,
                "mean_delta_spearman": float(values[:, 0].mean()),
                "mean_delta_utility20": float(values[:, 1].mean()),
                "positive_targets_utility20": int((values[:, 1] > 0).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(target_rows)


def dataset_deltas(frame: pd.DataFrame, value: float) -> pd.DataFrame:
    rows = []
    for dataset, study in frame.groupby("dataset", sort=True):
        values = [
            batch_delta(block, value, "error_two_predictor_mean_rmse")
            for _, block in study.groupby("fold_id", sort=True)
        ]
        values = np.asarray(values)
        rows.append(
            {
                "dataset": dataset,
                "delta_spearman": float(values[:, 0].mean()),
                "delta_utility20": float(values[:, 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def e153_sweep(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, study_rows = [], []
    for value in LAMBDAS:
        studies = dataset_deltas(frame, value)
        studies.insert(0, "lambda", value)
        study_rows.append(studies)
        rows.append(
            {
                "lambda": value,
                "mean_delta_spearman": float(studies.delta_spearman.mean()),
                "mean_delta_utility20": float(studies.delta_utility20.mean()),
                "positive_studies_spearman": int((studies.delta_spearman > 0).sum()),
                "positive_studies_utility20": int((studies.delta_utility20 > 0).sum()),
            }
        )
    return pd.DataFrame(rows), pd.concat(study_rows, ignore_index=True)


def choose_lambda(sweep: pd.DataFrame, unit_column: str) -> float:
    ranked = sweep.sort_values(
        [unit_column, "mean_delta_utility20", "lambda"],
        ascending=[False, False, True],
    )
    return float(ranked.iloc[0]["lambda"])


def lodo_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heldout in sorted(frame.dataset.unique()):
        training = frame.loc[frame.dataset.ne(heldout)]
        candidates = []
        for value in LAMBDAS:
            studies = dataset_deltas(training, value)
            candidates.append(
                {
                    "lambda": value,
                    "mean_delta_utility20": float(studies.delta_utility20.mean()),
                    "positive_training_studies": int((studies.delta_utility20 > 0).sum()),
                }
            )
        candidates = pd.DataFrame(candidates).sort_values(
            ["positive_training_studies", "mean_delta_utility20", "lambda"],
            ascending=[False, False, True],
        )
        selected = float(candidates.iloc[0]["lambda"])
        held = dataset_deltas(frame.loc[frame.dataset.eq(heldout)], selected).iloc[0]
        rows.append(
            {
                "heldout_dataset": heldout,
                "selected_lambda": selected,
                "training_positive_studies": int(candidates.iloc[0].positive_training_studies),
                "training_mean_delta_utility20": float(candidates.iloc[0].mean_delta_utility20),
                "heldout_delta_spearman": float(held.delta_spearman),
                "heldout_delta_utility20": float(held.delta_utility20),
                "heldout_truth_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT.exists():
        raise ParameterAuditError(f"refusing to overwrite: {OUTPUT}")
    e201, e153 = validate_e201(), validate_e153()
    sweep201, targets201 = e201_sweep(e201)
    sweep153, studies153 = e153_sweep(e153)
    lodo = lodo_selection(e153)

    selected201 = choose_lambda(sweep201, "positive_targets_utility20")
    selected153 = choose_lambda(sweep153, "positive_studies_utility20")
    row201 = sweep201.loc[sweep201["lambda"].eq(selected201)].iloc[0]
    row153 = sweep153.loc[sweep153["lambda"].eq(selected153)].iloc[0]

    tables = OUTPUT / "tables"
    atomic_csv(tables / "E212_E201_PARAMETER_SWEEP.csv", sweep201)
    atomic_csv(tables / "E212_E201_TARGET_RESULTS.csv", targets201)
    atomic_csv(tables / "E212_E153_PARAMETER_SWEEP.csv", sweep153)
    atomic_csv(tables / "E212_E153_STUDY_RESULTS.csv", studies153)
    atomic_csv(tables / "E212_E153_LODO_SELECTION.csv", lodo)

    status = {
        "experiment": "E212_parameter_landscape",
        "evidence_identity": "released_data_method_development",
        "formula_family": "m + lambda * max(s-m, 0)",
        "lambda_grid": list(LAMBDAS),
        "selection_objective": (
            "maximize number of positive units at 20% review utility; "
            "then mean utility; then smaller lambda"
        ),
        "e201_selected_lambda": selected201,
        "e153_selected_lambda": selected153,
        "e153_lodo_mean_delta_utility20": float(lodo.heldout_delta_utility20.mean()),
        "e153_lodo_positive_utility_studies": int((lodo.heldout_delta_utility20 > 0).sum()),
        "e153_lodo_mean_delta_spearman": float(lodo.heldout_delta_spearman.mean()),
        "e153_lodo_positive_spearman_studies": int((lodo.heldout_delta_spearman > 0).sum()),
        "external_confirmation": False,
        "e205_or_e208_truth_used": False,
        "target_truth_modified": False,
    }
    atomic_text(OUTPUT / "RUN_STATUS.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    report = rf"""# E212｜单向修正参数地形与过拟合审计

证据身份：已解封数据上的方法开发；不是新的外部确认。

考察公式：

\[
R_{{\uparrow}}=m+\lambda\max(s-m,0),\qquad
\lambda\in\{{0,0.025,\ldots,1\}}.
\]

参数先按 20% 复核效用中正向数据单元数选择，再比较宏平均效用，并列时取更小
\(\lambda\)。这套规则在运行前写入脚本，未读取 E205 或 E208 真值。

## 两类开发数据给出的参数不同

| 开发环境 | 选中 λ | 平均 Spearman 增量 | 平均 20% 效用增量 | 正向效用单元 |
|---|---:|---:|---:|---:|
| E201 四个整背景留出 | {selected201:.3f} | {row201.mean_delta_spearman:+.4f} | {row201.mean_delta_utility20:+.4f} | {int(row201.positive_targets_utility20)}/4 |
| E153 八个历史研究 | {selected153:.3f} | {row153.mean_delta_spearman:+.4f} | {row153.mean_delta_utility20:+.4f} | {int(row153.positive_studies_utility20)}/8 |

整背景留出只支持很弱的上调；八研究混合开发倾向更强上调。这说明 λ 不是一个跨场景
通用常数，难度结构和缺失方式会改变 SafeConf 应占的权重。

## 完整留一研究检查

每次只用另外七个研究选择 λ，再应用于完整留出的第八个研究。平均 Spearman 增量为
{status['e153_lodo_mean_delta_spearman']:+.4f}，{status['e153_lodo_positive_spearman_studies']}/8
个研究为正；平均 20% 复核效用增量为 {status['e153_lodo_mean_delta_utility20']:+.4f}，
{status['e153_lodo_positive_utility_studies']}/8 个研究为正。

这一步显示，多试参数可以提高开发集平均值，但实际复核效用仍未达到跨研究稳定。
因此 E208 的固定 4∶1 主要公式不因本实验改变；E210 的 0.25 单向修正仍是预登记
次要比较器。E212 用于说明参数规律与过拟合风险，不允许挑 λ=0.90 后把历史数据重写成
独立验证。

## 结论

当前可复现规律是：预测幅度始终是锚点，SafeConf 只应作受约束修正；修正强度依赖
部署场景。能否获得具有实际大小的效用提高，仍由未参与本次搜索的 E208 一次性裁决。
"""
    atomic_text(OUTPUT / "E212_REPORT.md", report)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
