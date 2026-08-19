#!/usr/bin/env python3
"""E146: re-audit E140 after clustering repeated biological tasks.

This is deliberately a *post-unblinding statistical dependence audit*.  It
does not create a new confirmatory experiment and it must not be used to
relabel E140 as prospectively confirmed.  ``--phase freeze`` materialises and
hashes the exact E140 task table and writes the analysis contract.  Only a
subsequent ``--phase analyze`` call is allowed to inspect the endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, t


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E146_unique_biological_task_meta_20260714"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
CONTRACT = OUT / "E146_ANALYSIS_CONTRACT.md"
STATUS = OUT / "RUN_STATUS.json"
FREEZE_STATUS = OUT / "FREEZE_STATUS.json"
INPUT_SNAPSHOT = TABLES / "E146_E140_TASK_INPUT.csv"
DIRECTION_SNAPSHOT = TABLES / "E146_E139_DIRECTIONAL_INPUT.csv"
INPUT_MANIFEST = TABLES / "E146_INPUT_MANIFEST.csv"
THIS_SCRIPT = Path(__file__).resolve()

E140_SCRIPT = ROOT / "tools/scripts/run_e140_formal_seven_dataset_meta.py"
E139_TASKS = ROOT / "docs/实验结果/E139_nadig_directional_confirmation_20260714/tables/E139_TASK_AUDIT.csv"
SOURCE_FILES = [
    ROOT / "docs/实验结果/E108_formal_dual_model_risk_audit_20260713/tables/E108_TEST_TASK_RISK_TABLE.csv",
    ROOT / "docs/实验结果/E112_external_formal_dual_models_20260713/E112_ALL_TASKS.csv",
    ROOT / "docs/实验结果/E120_shifrut_formal_dual_models_20260714/Shifrut/TASK_RISK_TABLE.csv",
    ROOT / "docs/实验结果/E123_liang_formal_dual_models_20260714/Liang/TASK_RISK_TABLE.csv",
    ROOT / "docs/实验结果/E129_tian_crispri_formal_dual_models_20260714/Tian_CRISPRi/TASK_RISK_TABLE.csv",
    ROOT / "docs/实验结果/E138_nadig_formal_dual_models_20260714/Nadig_two_cellline/TASK_RISK_TABLE.csv",
]

TARGET = "error_two_predictor_mean_rmse"
PRIMARY = "safeconf_calibrated_pair_risk"
COMPARATORS = ["risk_model_disagreement", "baseline_predicted_magnitude"]
SCORES = [PRIMARY, *COMPARATORS]
CONTEXT_TASK_CLUSTER = ["dataset", "context", "perturbation"]
PERTURBATION_CLUSTER = ["dataset", "perturbation"]
FOLD_CONTEXT_ESTIMAND = "e140_fold_macro_context_task_cluster"
FOLD_PERTURBATION_ESTIMAND = "e140_fold_macro_perturbation_cluster"
POOLED_MEDIAN_ESTIMAND = "pooled_context_task_median_sensitivity"
N_BOOT = 3000
SEED = 202607146


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def deterministic_seed(label: str) -> int:
    token = hashlib.sha256(f"{SEED}|{label}".encode()).hexdigest()[:16]
    return int(token, 16) % (2**32 - 1)


def rho(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.unique(a[keep]).size < 2 or np.unique(b[keep]).size < 2:
        return float("nan")
    ra, rb = rankdata(a[keep], method="average"), rankdata(b[keep], method="average")
    value = np.corrcoef(ra, rb)[0, 1]
    return float(value) if math.isfinite(value) else float("nan")


def fisher_z(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))


def load_e140_table() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("e140_for_e146_freeze", E140_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frame = module.load()
    required = {"dataset", "fold_id", "task_id", "context", "perturbation", TARGET, *SCORES}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"E140 input lacks columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ["dataset", "fold_id", "task_id", "context", "perturbation"]:
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values(
        ["dataset", "fold_id", "context", "perturbation", "task_id"], kind="stable"
    ).reset_index(drop=True)
    if frame.dataset.nunique() != 7 or len(frame) != 3209:
        raise RuntimeError(f"E140 identity failed: datasets={frame.dataset.nunique()}, rows={len(frame)}")
    if frame[[TARGET, *SCORES]].isna().any().any():
        raise RuntimeError("E140 primary endpoint/score contains missing values")
    if frame.duplicated(["dataset", "fold_id", "context", "perturbation"]).any():
        raise RuntimeError("Duplicate task occurs within the same dataset/fold/context/perturbation")
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无）"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("NA" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def freeze_contract() -> None:
    for directory in [OUT, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    frame = load_e140_table()
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
    INPUT_SNAPSHOT.write_bytes(csv_bytes)

    direction = pd.read_csv(E139_TASKS)
    direction_required = {
        "dataset", "fold_id", "context", "perturbation", TARGET,
        "directional_risk_frozen", "error_centered_pearson_mean",
        "error_centered_cosine_mean", "direction_error_rank_target",
    }
    direction_missing = direction_required.difference(direction.columns)
    if direction_missing:
        raise RuntimeError(f"E139 directional input lacks columns: {sorted(direction_missing)}")
    for column in ["dataset", "fold_id", "task_id", "context", "perturbation"]:
        if column in direction:
            direction[column] = direction[column].astype(str)
    direction = direction.sort_values(
        ["dataset", "fold_id", "context", "perturbation"], kind="stable"
    ).reset_index(drop=True)
    direction_bytes = direction.to_csv(index=False, lineterminator="\n").encode()
    DIRECTION_SNAPSHOT.write_bytes(direction_bytes)

    counts = frame.groupby("dataset", as_index=False).agg(
        n_rows=("task_id", "size"),
        n_folds=("fold_id", "nunique"),
        n_unique_context_tasks=("task_id", lambda _: 0),
        n_unique_perturbations=("perturbation", "nunique"),
    )
    unique_counts = frame.groupby("dataset").apply(
        lambda group: group[["context", "perturbation"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    counts["n_unique_context_tasks"] = counts.dataset.map(unique_counts).astype(int)
    counts["row_to_context_task_ratio"] = counts.n_rows / counts.n_unique_context_tasks
    contract = f"""# E146｜唯一生物任务统计依赖审计合同

## 审计性质

E140 的任务误差和风险结果已经解封。本分析是在看到 E140 结果之后增加的统计依赖再审计，**不是新的预注册确认实验**，不得用于把既有证据改写成前瞻性验证。

本轮不重新拟合风险分数，不筛选任务，不更换端点。E140 冻结快照为 `{INPUT_SNAPSHOT.relative_to(ROOT)}`，SHA-256 为 `{sha256_bytes(csv_bytes)}`；E139 方向快照为 `{DIRECTION_SNAPSHOT.relative_to(ROOT)}`，SHA-256 为 `{sha256_bytes(direction_bytes)}`。

## 冻结定义

- 主端点：`{TARGET}`。
- 原 SafeConf absolute 风险：`{PRIMARY}`。
- 主比较器：`risk_model_disagreement`、`baseline_predicted_magnitude`。
- E140 主估计量：保持原 fold 内 Spearman、再对 fold 等权宏平均；bootstrap 按 `(dataset, perturbation)` 整簇重抽，使同一扰动的所有 context 和 outer-fold 记录使用同一个重抽权重。
- context-task 依赖敏感性：另以 `(dataset, context, perturbation)` 整簇重抽，但不替代更严格的 perturbation-cluster 主区间。
- pooled-median 仅为敏感性：每个 `(dataset, context, perturbation)` 输出一行；重复 outer-fold 记录的分数和端点分别取中位数。它不是 E140 原 estimand，也不是新的 pooled 主分析。
- 两个 estimand 分别进行七研究随机效应：研究效应经 Fisher z 变换；tau² 用 REML，I² 用 Cochran Q，均值区间用 modified Knapp–Hartung，prediction interval 使用 t 分布。
- E140 fold-macro 的研究内标准误来自 perturbation-cluster bootstrap；pooled-median sensitivity 的关联标准误使用经典 Fisher-z 近似、配对差值标准误使用 context-task bootstrap。
- SafeConf 与比较器的效应为 `atanh(r_safeconf)-atanh(r_comparator)`，不可解释为原始 Δrho。
- LODO：对两个 estimand 每次删除一个完整研究后重算相同随机效应模型。
- Nadig 方向风险按 perturbation 整簇同步 HepG2、Jurkat 及其 outer-fold 记录；context-task pooled median 仅作单独敏感性。方向结果绝不并入七研究 absolute 元分析。
- 所有 bootstrap 固定 `{N_BOOT}` 次，主随机种子 `{SEED}`。

## 冻结规模

{markdown_table(counts)}
"""
    CONTRACT.write_text(contract)

    manifest_rows = []
    for path in [THIS_SCRIPT, E140_SCRIPT, *SOURCE_FILES, E139_TASKS, INPUT_SNAPSHOT, DIRECTION_SNAPSHOT, CONTRACT]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path == THIS_SCRIPT:
            role = "E146_analysis_script"
        elif path == E140_SCRIPT:
            role = "E140_loader"
        elif path == E139_TASKS:
            role = "E139_live_source"
        elif path == INPUT_SNAPSHOT:
            role = "frozen_E140_task_snapshot"
        elif path == DIRECTION_SNAPSHOT:
            role = "frozen_E139_directional_snapshot"
        elif path == CONTRACT:
            role = "frozen_analysis_contract"
        else:
            role = "E140_source"
        manifest_rows.append(
            {
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(INPUT_MANIFEST, index=False)
    status = {
        "experiment": "E146_unique_biological_task_meta",
        "phase": "contract_frozen_before_e146_statistics",
        "generated_at": now(),
        "upstream_E140_results_already_unblinded": True,
        "confirmatory_claim_allowed": False,
        "input_snapshot": str(INPUT_SNAPSHOT.relative_to(ROOT)),
        "input_snapshot_sha256": sha256_bytes(csv_bytes),
        "direction_snapshot": str(DIRECTION_SNAPSHOT.relative_to(ROOT)),
        "direction_snapshot_sha256": sha256_bytes(direction_bytes),
        "contract_sha256": sha256_file(CONTRACT),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "analysis_script_sha256": sha256_file(THIS_SCRIPT),
        "n_rows": len(frame),
        "n_datasets": int(frame.dataset.nunique()),
        "n_unique_context_tasks": int(frame[CONTEXT_TASK_CLUSTER].drop_duplicates().shape[0]),
        "n_unique_dataset_perturbations": int(frame[PERTURBATION_CLUSTER].drop_duplicates().shape[0]),
        "bootstrap_cluster_primary": PERTURBATION_CLUSTER,
        "bootstrap_cluster_sensitivity": CONTEXT_TASK_CLUSTER,
        "n_bootstrap": N_BOOT,
        "primary_endpoint": TARGET,
        "primary_score": PRIMARY,
        "comparators": COMPARATORS,
        "truth_used_to_change_scores_or_tasks": False,
    }
    FREEZE_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def verify_frozen_input() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = [CONTRACT, FREEZE_STATUS, INPUT_SNAPSHOT, DIRECTION_SNAPSHOT, INPUT_MANIFEST]
    if any(not path.exists() for path in required):
        raise RuntimeError("Run --phase freeze before --phase analyze")
    status = json.loads(FREEZE_STATUS.read_text())
    if status.get("phase") != "contract_frozen_before_e146_statistics":
        raise RuntimeError(f"Unexpected pre-analysis phase: {status.get('phase')}")
    checks = {
        "frozen E140 input": (INPUT_SNAPSHOT, status["input_snapshot_sha256"]),
        "frozen E139 directional input": (DIRECTION_SNAPSHOT, status["direction_snapshot_sha256"]),
        "analysis contract": (CONTRACT, status["contract_sha256"]),
        "input manifest": (INPUT_MANIFEST, status["input_manifest_sha256"]),
        "analysis script": (THIS_SCRIPT, status["analysis_script_sha256"]),
    }
    for label, (path, expected) in checks.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} hash changed after freeze")
    manifest = pd.read_csv(INPUT_MANIFEST)
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.path)
        if not path.exists() or path.stat().st_size != int(row.bytes) or sha256_file(path) != str(row.sha256):
            raise RuntimeError(f"Frozen manifest source changed: {row.path}")
    frame = pd.read_csv(INPUT_SNAPSHOT)
    for column in ["dataset", "fold_id", "task_id", "context", "perturbation"]:
        frame[column] = frame[column].astype(str)
    if len(frame) != status["n_rows"] or frame.dataset.nunique() != status["n_datasets"]:
        raise RuntimeError("Frozen E140 input dimensions changed")
    direction = pd.read_csv(DIRECTION_SNAPSHOT)
    for column in ["dataset", "fold_id", "task_id", "context", "perturbation"]:
        if column in direction:
            direction[column] = direction[column].astype(str)
    return frame, direction, status


def fold_macro(frame: pd.DataFrame, score: str) -> float:
    values = [rho(group[score], group[TARGET]) for _, group in frame.groupby("fold_id", sort=True)]
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def deterministic_dedup(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in data.groupby(CONTEXT_TASK_CLUSTER, sort=True, dropna=False):
        row = dict(zip(CONTEXT_TASK_CLUSTER, key))
        row.update(
            {
                "n_outer_fold_occurrences": len(group),
                "n_distinct_folds": group.fold_id.nunique(),
                "source_fold_ids": ";".join(sorted(group.fold_id.astype(str).unique())),
                TARGET: float(group[TARGET].median()),
            }
        )
        for score in SCORES:
            row[score] = float(group[score].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(CONTEXT_TASK_CLUSTER, kind="stable").reset_index(drop=True)


def original_cluster_draws(
    data: pd.DataFrame,
    dataset: str,
    cluster_columns: list[str],
    estimand: str,
) -> pd.DataFrame:
    """Resample whole task clusters and preserve the E140 fold-macro estimand."""
    group = data[data.dataset.eq(dataset)].reset_index(drop=True)
    cluster_rows = [
        sub.index.to_numpy(int)
        for _, sub in group.groupby(cluster_columns, sort=True, dropna=False)
    ]
    observed = {score: fold_macro(group, score) for score in SCORES}
    rng = np.random.default_rng(deterministic_seed(f"{estimand}|{dataset}"))
    rows = []
    for draw in range(N_BOOT):
        sampled = rng.integers(0, len(cluster_rows), len(cluster_rows))
        indices = np.concatenate([cluster_rows[index] for index in sampled])
        boot = group.iloc[indices]
        values = {score: fold_macro(boot, score) for score in SCORES}
        rows.append(
            {
                "dataset": dataset,
                "estimand": estimand,
                "draw": draw,
                **{f"rho__{score}": values[score] for score in SCORES},
                "delta_rho__safe_minus_disagreement": values[PRIMARY] - values[COMPARATORS[0]],
                "delta_rho__safe_minus_magnitude": values[PRIMARY] - values[COMPARATORS[1]],
                "delta_z__safe_minus_disagreement": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[0]]),
                "delta_z__safe_minus_magnitude": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[1]]),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["observed"] = observed
    return result


def dedup_task_draws(dedup: pd.DataFrame, dataset: str) -> pd.DataFrame:
    group = dedup[dedup.dataset.eq(dataset)].reset_index(drop=True)
    observed = {score: rho(group[score], group[TARGET]) for score in SCORES}
    rng = np.random.default_rng(deterministic_seed(f"{POOLED_MEDIAN_ESTIMAND}|{dataset}"))
    rows = []
    for draw in range(N_BOOT):
        index = rng.integers(0, len(group), len(group))
        values = {score: rho(group[score].to_numpy(float)[index], group[TARGET].to_numpy(float)[index]) for score in SCORES}
        rows.append(
            {
                "dataset": dataset,
                "estimand": POOLED_MEDIAN_ESTIMAND,
                "draw": draw,
                **{f"rho__{score}": values[score] for score in SCORES},
                "delta_rho__safe_minus_disagreement": values[PRIMARY] - values[COMPARATORS[0]],
                "delta_rho__safe_minus_magnitude": values[PRIMARY] - values[COMPARATORS[1]],
                "delta_z__safe_minus_disagreement": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[0]]),
                "delta_z__safe_minus_magnitude": fisher_z(values[PRIMARY]) - fisher_z(values[COMPARATORS[1]]),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["observed"] = observed
    return result


def bootstrap_summary(draws: pd.DataFrame, observed_by_key: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    rows = []
    metric_columns = [column for column in draws.columns if column.startswith("rho__") or column.startswith("delta_")]
    for (dataset, estimand), group in draws.groupby(["dataset", "estimand"], sort=True):
        observed_scores = observed_by_key[(dataset, estimand)]
        observed_metrics = {f"rho__{score}": value for score, value in observed_scores.items()}
        observed_metrics.update(
            {
                "delta_rho__safe_minus_disagreement": observed_scores[PRIMARY] - observed_scores[COMPARATORS[0]],
                "delta_rho__safe_minus_magnitude": observed_scores[PRIMARY] - observed_scores[COMPARATORS[1]],
                "delta_z__safe_minus_disagreement": fisher_z(observed_scores[PRIMARY]) - fisher_z(observed_scores[COMPARATORS[0]]),
                "delta_z__safe_minus_magnitude": fisher_z(observed_scores[PRIMARY]) - fisher_z(observed_scores[COMPARATORS[1]]),
            }
        )
        for metric in metric_columns:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "dataset": dataset,
                    "estimand": estimand,
                    "metric": metric,
                    "observed": observed_metrics[metric],
                    "n_valid_draws": len(values),
                    "bootstrap_median": float(np.median(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "p_gt_zero": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def random_effects(yi: np.ndarray, sei: np.ndarray) -> dict[str, float]:
    y, se = np.asarray(yi, float), np.asarray(sei, float)
    keep = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[keep], se[keep]
    k = len(y)
    if k < 3:
        return {key: float("nan") for key in ["k", "pooled_z", "se_mkh", "ci95_low_z", "ci95_high_z", "tau2_reml", "Q", "I2_percent", "prediction_low_z", "prediction_high_z"]}
    variance = se**2

    def objective(tau2: float) -> float:
        weights = 1.0 / (variance + tau2)
        mean = float(np.sum(weights * y) / np.sum(weights))
        return 0.5 * float(np.sum(np.log(variance + tau2)) + np.log(np.sum(weights)) + np.sum(weights * (y - mean) ** 2))

    upper = max(1.0, float(np.var(y, ddof=1) * 20))
    fit = minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12})
    candidates = [(0.0, objective(0.0)), (float(fit.x), float(fit.fun))]
    tau2 = min(candidates, key=lambda item: item[1])[0]
    weights = 1.0 / (variance + tau2)
    mean = float(np.sum(weights * y) / np.sum(weights))
    q_hk = float(np.sum(weights * (y - mean) ** 2) / (k - 1))
    se_mkh = float(np.sqrt(max(1.0, q_hk) / np.sum(weights)))
    critical_mean = float(t.ppf(0.975, k - 1))
    ci_low, ci_high = mean - critical_mean * se_mkh, mean + critical_mean * se_mkh
    fixed_weights = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    Q = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    critical_prediction = float(t.ppf(0.975, max(1, k - 2)))
    prediction_se = float(np.sqrt(tau2 + se_mkh**2))
    return {
        "k": k,
        "pooled_z": mean,
        "se_mkh": se_mkh,
        "ci95_low_z": ci_low,
        "ci95_high_z": ci_high,
        "tau2_reml": tau2,
        "Q": Q,
        "I2_percent": I2,
        "prediction_low_z": mean - critical_prediction * prediction_se,
        "prediction_high_z": mean + critical_prediction * prediction_se,
    }


def build_study_effects(data: pd.DataFrame, dedup: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Create study effects for E140 fold-macro and pooled-median sensitivity."""
    rows = []
    for dataset in sorted(data.dataset.unique()):
        raw = data[data.dataset.eq(dataset)]
        collapsed = dedup[dedup.dataset.eq(dataset)]
        specifications = [
            {
                "estimand": FOLD_PERTURBATION_ESTIMAND,
                "correlations": {score: fold_macro(raw, score) for score in SCORES},
                "draws": draws[
                    draws.dataset.eq(dataset) & draws.estimand.eq(FOLD_PERTURBATION_ESTIMAND)
                ],
                "n_clusters": int(raw.perturbation.nunique()),
                "association_se_source": "perturbation_cluster_bootstrap_fisher_z",
            },
            {
                "estimand": POOLED_MEDIAN_ESTIMAND,
                "correlations": {score: rho(collapsed[score], collapsed[TARGET]) for score in SCORES},
                "draws": draws[
                    draws.dataset.eq(dataset) & draws.estimand.eq(POOLED_MEDIAN_ESTIMAND)
                ],
                "n_clusters": len(collapsed),
                "association_se_source": "classical_fisher_z_context_tasks_sensitivity",
            },
        ]
        for specification in specifications:
            correlations = specification["correlations"]
            study_draws = specification["draws"]
            n_clusters = int(specification["n_clusters"])
            for score in SCORES:
                if specification["estimand"] == FOLD_PERTURBATION_ESTIMAND:
                    z_draws = study_draws[f"rho__{score}"].map(fisher_z).to_numpy(float)
                    z_draws = z_draws[np.isfinite(z_draws)]
                    sei = float(np.std(z_draws, ddof=1))
                else:
                    sei = 1.0 / np.sqrt(n_clusters - 3)
                rows.append(
                    {
                        "dataset": dataset,
                        "estimand": specification["estimand"],
                        "analysis": "score_association",
                        "effect": score,
                        "n_independent_clusters": n_clusters,
                        "rho_safeconf": correlations[PRIMARY],
                        "rho_comparator": correlations[score],
                        "delta_rho_safe_minus_comparator": correlations[PRIMARY] - correlations[score],
                        "yi_fisher_z": fisher_z(correlations[score]),
                        "sei": sei,
                        "sei_source": specification["association_se_source"],
                    }
                )
            for comparator, suffix in [(COMPARATORS[0], "disagreement"), (COMPARATORS[1], "magnitude")]:
                values = study_draws[f"delta_z__safe_minus_{suffix}"].to_numpy(float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "estimand": specification["estimand"],
                        "analysis": "safeconf_minus_comparator",
                        "effect": f"safeconf_minus_{suffix}",
                        "n_independent_clusters": n_clusters,
                        "rho_safeconf": correlations[PRIMARY],
                        "rho_comparator": correlations[comparator],
                        "delta_rho_safe_minus_comparator": correlations[PRIMARY] - correlations[comparator],
                        "yi_fisher_z": fisher_z(correlations[PRIMARY]) - fisher_z(correlations[comparator]),
                        "sei": float(np.std(values, ddof=1)),
                        "sei_source": "paired_cluster_bootstrap_delta_fisher_z",
                    }
                )
    return pd.DataFrame(rows)


def meta_table(studies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (estimand, analysis, effect), group in studies.groupby(["estimand", "analysis", "effect"], sort=True):
        result = random_effects(group.yi_fisher_z.to_numpy(float), group.sei.to_numpy(float))
        row = {"estimand": estimand, "analysis": analysis, "effect": effect, **result}
        for name in ["pooled_z", "ci95_low_z", "ci95_high_z", "prediction_low_z", "prediction_high_z"]:
            row[name.replace("_z", "_rho_equivalent")] = float(np.tanh(row[name])) if math.isfinite(row[name]) else np.nan
        row["backtransform_interpretation"] = "pooled Spearman" if analysis == "score_association" else "rho-equivalent of Fisher-z difference; not raw delta-rho"
        rows.append(row)
    return pd.DataFrame(rows)


def lodo_table(studies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = sorted(studies.dataset.unique())
    for (estimand, analysis, effect), group in studies.groupby(["estimand", "analysis", "effect"], sort=True):
        for removed in datasets:
            keep = group[~group.dataset.eq(removed)]
            result = random_effects(keep.yi_fisher_z.to_numpy(float), keep.sei.to_numpy(float))
            rows.append(
                {
                    "estimand": estimand,
                    "analysis": analysis,
                    "effect": effect,
                    "removed_dataset": removed,
                    **result,
                    "pooled_rho_equivalent": float(np.tanh(result["pooled_z"])) if math.isfinite(result["pooled_z"]) else np.nan,
                    "prediction_low_rho_equivalent": float(np.tanh(result["prediction_low_z"])) if math.isfinite(result["prediction_low_z"]) else np.nan,
                    "prediction_high_rho_equivalent": float(np.tanh(result["prediction_high_z"])) if math.isfinite(result["prediction_high_z"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def directional_secondary(
    absolute: pd.DataFrame,
    direction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit Nadig direction separately with perturbations synchronized across contexts."""
    nadig = absolute[absolute.dataset.eq("Nadig_two_cellline")].copy()
    keys = ["dataset", "fold_id", "context", "perturbation"]
    if direction.duplicated(keys).any() or nadig.duplicated(keys).any():
        raise RuntimeError("Nadig direction/absolute key is not one-to-one")
    aligned = nadig[keys + [TARGET]].merge(
        direction,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_e140", "_e139"),
    )
    if not aligned._merge.eq("both").all() or len(aligned) != len(nadig):
        raise RuntimeError("E139 direction rows do not exactly align with E140 Nadig rows")
    difference = float(np.max(np.abs(aligned[f"{TARGET}_e140"] - aligned[f"{TARGET}_e139"])))
    if difference > 1e-12:
        raise RuntimeError(f"E139/E140 absolute endpoint mismatch: {difference}")
    endpoints = ["error_centered_pearson_mean", "error_centered_cosine_mean", "direction_error_rank_target"]
    scores = ["directional_risk_frozen", "baseline_predicted_magnitude", "risk_model_disagreement", "safeconf_calibrated_pair_risk"]
    collapsed = aligned.groupby(["dataset", "context", "perturbation"], as_index=False)[endpoints + scores].median()

    def direction_fold_macro(frame: pd.DataFrame, score: str, endpoint: str) -> float:
        values = [rho(group[score], group[endpoint]) for _, group in frame.groupby("fold_id", sort=True)]
        values = np.asarray(values, float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    estimand_data = [
        {
            "estimand": "nadig_direction_fold_macro_perturbation_cluster",
            "frame": aligned,
            "cluster_columns": ["perturbation"],
            "statistic": direction_fold_macro,
            "n_clusters": int(aligned.perturbation.nunique()),
        },
        {
            "estimand": "nadig_direction_pooled_context_task_median_sensitivity",
            "frame": collapsed,
            "cluster_columns": ["context", "perturbation"],
            "statistic": lambda frame, score, endpoint: rho(frame[score], frame[endpoint]),
            "n_clusters": len(collapsed),
        },
    ]
    draw_frames = []
    observed_by_estimand: dict[str, dict[str, float]] = {}
    for specification in estimand_data:
        frame = specification["frame"].reset_index(drop=True)
        cluster_rows = [
            group.index.to_numpy(int)
            for _, group in frame.groupby(specification["cluster_columns"], sort=True, dropna=False)
        ]
        statistic = specification["statistic"]
        observed = {
            f"rho__{score}__{endpoint}": statistic(frame, score, endpoint)
            for score in scores for endpoint in endpoints
        }
        for comparator in scores[1:]:
            for endpoint in endpoints:
                observed[f"delta_rho__directional_minus_{comparator}__{endpoint}"] = (
                    observed[f"rho__directional_risk_frozen__{endpoint}"]
                    - observed[f"rho__{comparator}__{endpoint}"]
                )
        observed_by_estimand[specification["estimand"]] = observed
        rng = np.random.default_rng(deterministic_seed(specification["estimand"]))
        rows = []
        for draw in range(N_BOOT):
            sampled = rng.integers(0, len(cluster_rows), len(cluster_rows))
            index = np.concatenate([cluster_rows[position] for position in sampled])
            boot = frame.iloc[index]
            values = {
                f"rho__{score}__{endpoint}": statistic(boot, score, endpoint)
                for score in scores for endpoint in endpoints
            }
            for comparator in scores[1:]:
                for endpoint in endpoints:
                    values[f"delta_rho__directional_minus_{comparator}__{endpoint}"] = (
                        values[f"rho__directional_risk_frozen__{endpoint}"]
                        - values[f"rho__{comparator}__{endpoint}"]
                    )
            rows.append({"estimand": specification["estimand"], "draw": draw, **values})
        draw_frames.append(pd.DataFrame(rows))
    directional_draws = pd.concat(draw_frames, ignore_index=True)
    summary_rows = []
    for estimand, group in directional_draws.groupby("estimand", sort=True):
        for metric in [column for column in group.columns if column not in {"estimand", "draw"}]:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if metric.startswith("rho__"):
                _, score, endpoint = metric.split("__", 2)
                metric_type, comparator = "association", ""
            else:
                prefix, endpoint = metric.rsplit("__", 1)
                score, comparator = "directional_risk_frozen", prefix.removeprefix("delta_rho__directional_minus_")
                metric_type = "directional_minus_comparator"
            summary_rows.append(
                {
                    "dataset": "Nadig_two_cellline",
                    "scope": "directional_single_study_only_not_in_absolute_meta",
                    "estimand": estimand,
                    "metric_type": metric_type,
                    "score": score,
                    "comparator": comparator,
                    "endpoint": endpoint,
                    "n_independent_clusters": (
                        int(aligned.perturbation.nunique())
                        if estimand.endswith("perturbation_cluster") else len(collapsed)
                    ),
                    "observed": observed_by_estimand[estimand][metric],
                    "bootstrap_median": float(np.median(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "p_gt_zero": float(np.mean(values > 0)),
                }
            )
    audit = {
        "aligned_rows": len(aligned),
        "unique_context_tasks": len(collapsed),
        "unique_perturbations_synchronized_across_contexts": int(aligned.perturbation.nunique()),
        "max_abs_e139_e140_absolute_endpoint_difference": difference,
        "included_in_seven_study_absolute_meta": False,
    }
    return pd.DataFrame(summary_rows), directional_draws, audit


def analyze() -> None:
    data, direction_snapshot, frozen_status = verify_frozen_input()
    dedup = deterministic_dedup(data)
    multiplicity = data.groupby("dataset", as_index=False).agg(
        n_rows=("task_id", "size"), n_folds=("fold_id", "nunique"),
        n_unique_perturbations=("perturbation", "nunique"),
    )
    per_cluster = data.groupby(CONTEXT_TASK_CLUSTER, as_index=False).size().rename(columns={"size": "n_occurrences"})
    cluster_stats = per_cluster.groupby("dataset").n_occurrences.agg(["count", "mean", "median", "max"]).reset_index()
    cluster_stats = cluster_stats.rename(columns={"count": "n_unique_context_tasks", "mean": "mean_occurrences", "median": "median_occurrences", "max": "max_occurrences"})
    duplicated = per_cluster.assign(duplicated=per_cluster.n_occurrences.gt(1)).groupby("dataset", as_index=False).duplicated.sum().rename(columns={"duplicated": "n_tasks_repeated_across_folds"})
    multiplicity = multiplicity.merge(cluster_stats, on="dataset", validate="one_to_one").merge(duplicated, on="dataset", validate="one_to_one")
    multiplicity["row_to_unique_context_task_ratio"] = multiplicity.n_rows / multiplicity.n_unique_context_tasks

    all_draws = []
    observed_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in sorted(data.dataset.unique()):
        context_cluster = original_cluster_draws(
            data, dataset, ["context", "perturbation"], FOLD_CONTEXT_ESTIMAND
        )
        perturbation_cluster = original_cluster_draws(
            data, dataset, ["perturbation"], FOLD_PERTURBATION_ESTIMAND
        )
        dedup_draw = dedup_task_draws(dedup, dataset)
        observed_by_key[(dataset, FOLD_CONTEXT_ESTIMAND)] = context_cluster.attrs["observed"]
        observed_by_key[(dataset, FOLD_PERTURBATION_ESTIMAND)] = perturbation_cluster.attrs["observed"]
        observed_by_key[(dataset, POOLED_MEDIAN_ESTIMAND)] = dedup_draw.attrs["observed"]
        all_draws.extend([context_cluster, perturbation_cluster, dedup_draw])
    draws = pd.concat(all_draws, ignore_index=True)
    boot_summary = bootstrap_summary(draws, observed_by_key)
    dedup_correlations = []
    for dataset, group in dedup.groupby("dataset", sort=True):
        for score in SCORES:
            dedup_correlations.append(
                {
                    "estimand": POOLED_MEDIAN_ESTIMAND,
                    "dataset": dataset, "score": score,
                    "n_unique_context_tasks": len(group),
                    "spearman": rho(group[score], group[TARGET]),
                }
            )
    dedup_correlations = pd.DataFrame(dedup_correlations)
    fold_correlations = []
    for dataset, group in data.groupby("dataset", sort=True):
        for score in SCORES:
            fold_correlations.append(
                {
                    "estimand": FOLD_PERTURBATION_ESTIMAND,
                    "dataset": dataset, "score": score,
                    "n_folds": group.fold_id.nunique(),
                    "n_unique_perturbations": group.perturbation.nunique(),
                    "spearman_fold_macro": fold_macro(group, score),
                }
            )
    fold_correlations = pd.DataFrame(fold_correlations)
    studies = build_study_effects(data, dedup, draws)
    meta = meta_table(studies)
    lodo = lodo_table(studies)
    directional, directional_draws, directional_audit = directional_secondary(data, direction_snapshot)

    multiplicity.to_csv(TABLES / "E146_CLUSTER_MULTIPLICITY.csv", index=False)
    dedup.to_csv(TABLES / "E146_UNIQUE_TASK_MEDIAN_TABLE.csv", index=False)
    dedup_correlations.to_csv(TABLES / "E146_DEDUP_CORRELATIONS.csv", index=False)
    fold_correlations.to_csv(TABLES / "E146_FOLD_MACRO_CORRELATIONS.csv", index=False)
    draws.to_csv(TABLES / "E146_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)
    boot_summary.to_csv(TABLES / "E146_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    studies.to_csv(TABLES / "E146_STUDY_EFFECTS.csv", index=False)
    meta.to_csv(TABLES / "E146_RANDOM_EFFECTS_META.csv", index=False)
    lodo.to_csv(TABLES / "E146_LODO.csv", index=False)
    directional.to_csv(TABLES / "E146_DIRECTIONAL_NADIG_ONLY.csv", index=False)
    directional_draws.to_csv(TABLES / "E146_DIRECTIONAL_CLUSTER_BOOTSTRAP_DRAWS.csv", index=False)

    main_meta = meta[meta.analysis.eq("safeconf_minus_comparator")][
        ["estimand", "effect", "k", "pooled_z", "ci95_low_z", "ci95_high_z", "tau2_reml", "I2_percent", "prediction_low_z", "prediction_high_z"]
    ].copy()
    main_meta.to_csv(TABLES / "E146_ESTIMAND_COMPARISON.csv", index=False)
    association_meta = meta[
        meta.analysis.eq("score_association") & meta.effect.isin(SCORES)
    ][
        ["estimand", "effect", "pooled_rho_equivalent", "ci95_low_rho_equivalent",
         "ci95_high_rho_equivalent", "prediction_low_rho_equivalent",
         "prediction_high_rho_equivalent", "tau2_reml", "I2_percent"]
    ].copy()
    main_boot = boot_summary[
        boot_summary.metric.isin(["delta_rho__safe_minus_disagreement", "delta_rho__safe_minus_magnitude"])
        & boot_summary.estimand.isin([FOLD_PERTURBATION_ESTIMAND, POOLED_MEDIAN_ESTIMAND])
    ][["dataset", "estimand", "metric", "observed", "ci95_low", "ci95_high", "p_gt_zero"]]
    directional_main = directional[
        directional.estimand.eq("nadig_direction_fold_macro_perturbation_cluster")
        & directional.metric_type.eq("association")
        & directional.score.isin(["directional_risk_frozen", "baseline_predicted_magnitude"])
        & directional.endpoint.eq("direction_error_rank_target")
    ][["estimand", "score", "n_independent_clusters", "observed", "ci95_low", "ci95_high"]]
    liang_lodo = lodo[
        lodo.analysis.eq("safeconf_minus_comparator")
        & lodo.effect.eq("safeconf_minus_magnitude")
        & lodo.removed_dataset.eq("Liang")
    ][["estimand", "pooled_z", "ci95_low_z", "ci95_high_z"]].copy()
    sensitivity_liang = liang_lodo[liang_lodo.estimand.eq(POOLED_MEDIAN_ESTIMAND)]
    if len(sensitivity_liang) != 1 or float(sensitivity_liang.iloc[0].pooled_z) >= 0:
        raise RuntimeError("Expected pooled-median SafeConf-minus-magnitude sign reversal after deleting Liang")
    fold_liang = liang_lodo[liang_lodo.estimand.eq(FOLD_PERTURBATION_ESTIMAND)]
    if len(fold_liang) != 1:
        raise RuntimeError("Missing E140 fold-macro Liang LODO result")
    report = f"""# E146｜唯一生物任务统计依赖再审计

E146 在 E140 结果已经解封之后执行。它回答重复 outer-fold 记录是否让区间过窄，不构成新的独立确认。

## 重复规模

{markdown_table(multiplicity.round(4))}

3209 行记录对应 {len(dedup)} 个 `(dataset, context, perturbation)` context-task，以及 {data[PERTURBATION_CLUSTER].drop_duplicates().shape[0]} 个 `(dataset, perturbation)` 簇。E140 fold-macro 主区间以 perturbation 整簇重抽，因此同一扰动的所有 context 与 outer-fold 记录同步出现；context-task 聚类和 pooled median 都只作敏感性。

## SafeConf 相对比较器：聚类 bootstrap

{markdown_table(main_boot.round(4))}

`{FOLD_PERTURBATION_ESTIMAND}` 保持 E140 的原 fold-macro 点估计，并使用更严格的 perturbation-cluster 区间。`{POOLED_MEDIAN_ESTIMAND}` 把每个 context-task 跨 fold 取中位数，它明确是 pooled-median sensitivity，不是 E140 原估计量。

## 七研究随机效应：E140 fold-macro 与 pooled-median sensitivity 并列

{markdown_table(main_meta.round(4))}

{markdown_table(association_meta.round(4))}

两个 estimand 均使用 Fisher z、REML tau²、modified Knapp–Hartung 均值区间和研究预测区间。差值行位于 Fisher-z 尺度，不能当成原始 Δrho。

## LODO 的 Liang 依赖

{markdown_table(liang_lodo.round(4))}

删除 Liang 后，pooled-median sensitivity 的 SafeConf−magnitude 合并效应由全七研究的正值变为 {float(sensitivity_liang.iloc[0].pooled_z):+.4f}，发生符号反转。E140 fold-macro estimand 删除 Liang 后为 {float(fold_liang.iloc[0].pooled_z):+.4f}。这一敏感性必须明写，不能只报告全数据平均。

## 方向风险：Nadig 单研究附录

E139 与 E140 Nadig 共 {directional_audit['aligned_rows']} 行一一对齐，absolute 端点最大差为 {directional_audit['max_abs_e139_e140_absolute_endpoint_difference']:.3g}。主方向 bootstrap 以 {directional_audit['unique_perturbations_synchronized_across_contexts']} 个 perturbation 整簇同步 HepG2、Jurkat；{directional_audit['unique_context_tasks']} 个 context-task pooled median 另作敏感性。方向结果没有并入七研究 absolute 元分析。

{markdown_table(directional_main.round(4))}

## 解释边界

- E146 没有改变 SafeConf 分数、任务或端点。
- perturbation-cluster 同步同一扰动的所有 context 和 fold；context-task 与 pooled-median 结果仅用于敏感性。
- 七个研究仍然只有七个研究；prediction interval 比均值区间更接近未来新研究的不确定性。
- 删除 Liang 后 pooled-median SafeConf−magnitude 符号反转；完整 LODO 见 `tables/E146_LODO.csv`。
"""
    (REPORTS / "E146_REPORT.md").write_text(report)
    (OUT / "README_先看这个.md").write_text("# E146 先看这个\n\n先读 `E146_ANALYSIS_CONTRACT.md`，再读 `reports/E146_REPORT.md`。\n")
    completed = {
        **frozen_status,
        "phase": "complete_post_unblinding_dependence_audit",
        "completed_at": now(),
        "upstream_E140_results_already_unblinded": True,
        "confirmatory_claim_allowed": False,
        "n_input_rows": len(data),
        "n_unique_context_tasks": len(dedup),
        "n_unique_dataset_perturbations": int(data[PERTURBATION_CLUSTER].drop_duplicates().shape[0]),
        "n_datasets": int(data.dataset.nunique()),
        "n_bootstrap_per_dataset_per_estimand": N_BOOT,
        "bootstrap_estimands": [FOLD_CONTEXT_ESTIMAND, FOLD_PERTURBATION_ESTIMAND, POOLED_MEDIAN_ESTIMAND],
        "random_effects_method": "Fisher z + REML tau2 + modified Knapp-Hartung + t prediction interval",
        "random_effects_estimands": [FOLD_PERTURBATION_ESTIMAND, POOLED_MEDIAN_ESTIMAND],
        "safeconf_minus_magnitude_sign_reversal_after_deleting_liang_in_pooled_median_sensitivity": True,
        "directional_analysis_scope": "Nadig single-study only; excluded from seven-study absolute meta",
        "directional_alignment_audit": directional_audit,
        "truth_used_to_change_scores_or_tasks": False,
    }
    STATUS.write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(completed, ensure_ascii=False, indent=2))
    print(main_meta.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["freeze", "analyze", "all"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase in {"freeze", "all"}:
        freeze_contract()
    if args.phase in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
