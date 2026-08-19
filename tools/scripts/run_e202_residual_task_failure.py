#!/usr/bin/env python3
"""Run the frozen E202 magnitude-adjusted TxPert failure diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E202_residual_task_failure_20260802"
FREEZE = OUT / "ANALYSIS_FREEZE.md"
SCRIPT = Path(__file__).resolve()
RELEASE = OUT / "formal_evaluation"
STAGING = OUT / ".formal_evaluation.staging"
TASKS = (
    ROOT
    / "docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_TASK_METRICS.csv"
)
SCPERT = (
    ROOT
    / "docs/实验结果/E200_txpert_cross_context_k562_20260802/formal_evaluation/tables/E200_SCPERTEVAL_TASK_METRICS.csv"
)
EXPECTED_HASHES = {
    TASKS: "91fc71c767ed4742bc794c1d55e6fd00dbfe77294f59201717233c702aff9062",
    SCPERT: "5003410d21eef7f7fa3ffac53131a6ddf49885187435f33e94cc35c0f60f83a3",
}
N_TASKS = 566
N_BOOTSTRAP = 5_000
ENDPOINTS = (
    "mse",
    "pearson_pert",
    "rank",
    "energy_distance_pca_k=50",
    "de_auprc",
)
PREDICTORS = (
    "training_delta_dispersion",
    "transfer_risk",
    "model_baseline_gap",
    "negative_log_train_cells",
    "support_context_deficit",
)


class EvaluationFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"E202::{label}".encode()).hexdigest()[:8], 16)


def repo_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def require_frozen_sources() -> None:
    for path in (FREEZE, SCRIPT):
        if not path.is_file():
            raise EvaluationFailure(f"missing frozen source: {path}")
        try:
            repo_text("ls-files", "--error-unmatch", str(path.relative_to(ROOT)))
        except subprocess.CalledProcessError as exc:
            raise EvaluationFailure(f"frozen source is not tracked: {path}") from exc
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--quiet",
                "--",
                str(path.relative_to(ROOT)),
            ]
        )
        if dirty.returncode != 0:
            raise EvaluationFailure(f"frozen source has uncommitted edits: {path}")


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 4:
        return float("nan")
    x, y = left[keep], right[keep]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return corr(rankdata(left, method="average"), rankdata(right, method="average"))


def residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    y = rankdata(np.asarray(values, dtype=float), method="average")
    z = rankdata(np.asarray(control, dtype=float), method="average")
    design = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def partial_spearman(
    predictor: np.ndarray, outcome: np.ndarray, control: np.ndarray
) -> float:
    arrays = [np.asarray(v, dtype=float) for v in (predictor, outcome, control)]
    keep = np.logical_and.reduce([np.isfinite(v) for v in arrays])
    if keep.sum() < 4:
        return float("nan")
    x, y, z = [v[keep] for v in arrays]
    return corr(residualize(x, z), residualize(y, z))


def bootstrap_association(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    control: str,
    label: str,
) -> dict[str, float | int | str]:
    x = frame[predictor].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    z = frame[control].to_numpy(float)
    point_raw = spearman(x, y)
    point_partial = partial_spearman(x, y, z)
    rng = np.random.default_rng(stable_seed(label))
    values = []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, len(frame), len(frame))
        value = partial_spearman(x[take], y[take], z[take])
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise EvaluationFailure(f"no valid bootstrap values: {label}")
    return {
        "predictor": predictor,
        "outcome": outcome,
        "control": control,
        "n_tasks": len(frame),
        "raw_spearman": point_raw,
        "partial_spearman": point_partial,
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "bootstrap_valid": len(values),
    }


def endpoint_regrets(scpert: pd.DataFrame) -> pd.DataFrame:
    block = scpert.loc[
        scpert.stratum.eq("primary_ge30")
        & scpert.predictor.isin(["gat", "general_baseline"])
        & scpert.endpoint.isin(ENDPOINTS)
    ].copy()
    wide = block.pivot(
        index=["task_id", "condition_label", "endpoint"],
        columns="predictor",
        values="oriented_error",
    ).reset_index()
    if wide[["gat", "general_baseline"]].isna().any().any():
        raise EvaluationFailure("missing GAT/general endpoint pair")
    wide["gat_excess_error_vs_general"] = wide.gat - wide.general_baseline
    return wide


def magnitude_strata(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = frame.copy()
    work["magnitude_quintile"] = pd.qcut(
        work.predicted_magnitude.rank(method="first"), 5, labels=False
    ) + 1
    work["dispersion_within_quintile"] = work.training_delta_dispersion - work.groupby(
        "magnitude_quintile"
    ).training_delta_dispersion.transform("mean")
    work["regret_within_quintile"] = work.gat_excess_rmse_vs_general - work.groupby(
        "magnitude_quintile"
    ).gat_excess_rmse_vs_general.transform("mean")
    summary = (
        work.groupby("magnitude_quintile", observed=True)
        .agg(
            n_tasks=("task_id", "size"),
            magnitude_median=("predicted_magnitude", "median"),
            dispersion_median=("training_delta_dispersion", "median"),
            regret_median=("gat_excess_rmse_vs_general", "median"),
        )
        .reset_index()
    )
    pooled = spearman(
        work.dispersion_within_quintile.to_numpy(float),
        work.regret_within_quintile.to_numpy(float),
    )
    return summary, {"within_quintile_centered_spearman": pooled}


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def make_figure(
    frame: pd.DataFrame,
    associations: pd.DataFrame,
    endpoint_assoc: pd.DataFrame,
    strata: pd.DataFrame,
    path_png: Path,
    path_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    scatter = ax.scatter(
        frame.predicted_magnitude,
        frame.gat_excess_rmse_vs_general,
        c=frame.training_delta_dispersion,
        cmap="viridis",
        s=14,
        alpha=0.72,
        linewidths=0,
    )
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.set_xlabel("Predicted magnitude")
    ax.set_ylabel("GAT RMSE − general RMSE")
    ax.set_title("A  Model regret and effect size", loc="left", fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Training-context dispersion")

    ax = axes[0, 1]
    display = associations.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    ax.errorbar(
        display.partial_spearman,
        y,
        xerr=np.vstack(
            [
                display.partial_spearman - display.ci95_lower,
                display.ci95_upper - display.partial_spearman,
            ]
        ),
        fmt="o",
        color="#31688e",
        ecolor="#31688e",
        capsize=2.5,
        linewidth=0.9,
    )
    ax.axvline(0, color="#666666", linewidth=0.7)
    ax.set_yticks(y, [x.replace("_", " ") for x in display.predictor])
    ax.set_xlabel("Partial Spearman (95% bootstrap interval)")
    ax.set_title("B  Adjusted associations", loc="left", fontweight="bold")

    ax = axes[1, 0]
    ax.plot(
        strata.magnitude_quintile,
        strata.regret_median,
        marker="o",
        color="#35b779",
        linewidth=1.2,
    )
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.set_xticks(strata.magnitude_quintile)
    ax.set_xlabel("Predicted-magnitude quintile")
    ax.set_ylabel("Median GAT RMSE − general RMSE")
    ax.set_title("C  Regret across magnitude strata", loc="left", fontweight="bold")

    ax = axes[1, 1]
    endpoint = endpoint_assoc.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(endpoint))
    ax.errorbar(
        endpoint.partial_spearman,
        y,
        xerr=np.vstack(
            [
                endpoint.partial_spearman - endpoint.ci95_lower,
                endpoint.ci95_upper - endpoint.partial_spearman,
            ]
        ),
        fmt="o",
        color="#440154",
        ecolor="#440154",
        capsize=2.5,
        linewidth=0.9,
    )
    ax.axvline(0, color="#666666", linewidth=0.7)
    ax.set_yticks(y, endpoint.endpoint)
    ax.set_xlabel("Partial Spearman (95% bootstrap interval)")
    ax.set_title("D  Endpoint sensitivity", loc="left", fontweight="bold")
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> None:
    if RELEASE.exists() or STAGING.exists():
        raise EvaluationFailure("formal output or staging directory already exists")
    require_frozen_sources()
    input_rows = []
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256_file(path)
        if observed != expected:
            raise EvaluationFailure(f"input hash mismatch: {path}")
        input_rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )

    tasks_all = pd.read_csv(TASKS)
    tasks = tasks_all.loc[tasks_all.analysis_stratum.eq("primary_ge30")].copy()
    if len(tasks) != N_TASKS or tasks.task_id.nunique() != N_TASKS:
        raise EvaluationFailure("primary task contract failed")
    tasks["gat_excess_rmse_vs_general"] = (
        tasks.gat_centroid_rmse - tasks.general_baseline_centroid_rmse
    )
    tasks["gat_excess_rmse_vs_control"] = (
        tasks.gat_centroid_rmse - tasks.control_centroid_rmse
    )
    required = set(PREDICTORS) | {
        "predicted_magnitude",
        "gat_excess_rmse_vs_general",
        "gat_excess_rmse_vs_control",
    }
    if tasks[list(required)].isna().any().any():
        raise EvaluationFailure("non-finite primary task input")

    scpert = pd.read_csv(SCPERT)
    expected_rows = N_TASKS * 3 * len(ENDPOINTS)
    primary_scpert = scpert.loc[scpert.stratum.eq("primary_ge30")].copy()
    keys = ["task_id", "predictor", "endpoint"]
    if len(primary_scpert) != expected_rows or primary_scpert.duplicated(keys).any():
        raise EvaluationFailure("scPertEval task contract failed")
    regrets = endpoint_regrets(primary_scpert)
    if len(regrets) != N_TASKS * len(ENDPOINTS):
        raise EvaluationFailure("endpoint regret contract failed")

    assoc_rows = [
        bootstrap_association(
            tasks,
            predictor,
            "gat_excess_rmse_vs_general",
            "predicted_magnitude",
            f"main::{predictor}",
        )
        for predictor in PREDICTORS
    ]
    associations = pd.DataFrame(assoc_rows)
    baseline_sensitivity = pd.DataFrame(
        [
            bootstrap_association(
                tasks,
                "training_delta_dispersion",
                "gat_excess_rmse_vs_control",
                "predicted_magnitude",
                "baseline-sensitivity::control",
            )
        ]
    )
    endpoint_input = regrets.merge(
        tasks[
            ["task_id", "training_delta_dispersion", "predicted_magnitude"]
        ],
        on="task_id",
        how="left",
        validate="many_to_one",
    )
    endpoint_rows = []
    for endpoint in ENDPOINTS:
        block = endpoint_input.loc[endpoint_input.endpoint.eq(endpoint)].copy()
        row = bootstrap_association(
            block,
            "training_delta_dispersion",
            "gat_excess_error_vs_general",
            "predicted_magnitude",
            f"endpoint::{endpoint}",
        )
        row["endpoint"] = endpoint
        endpoint_rows.append(row)
    endpoint_assoc = pd.DataFrame(endpoint_rows)
    strata, pooled = magnitude_strata(tasks)

    primary = associations.loc[
        associations.predictor.eq("training_delta_dispersion")
    ].iloc[0]
    supported = bool(primary.ci95_lower > 0)
    gates = pd.DataFrame(
        [
            {
                "gate": "input_integrity",
                "passed": True,
                "observed": f"{N_TASKS} tasks; {expected_rows} endpoint rows",
                "criterion": "all hashes, row counts and uniqueness checks pass",
            },
            {
                "gate": "magnitude_adjusted_independent_signal",
                "passed": supported,
                "observed": (
                    f"partial_rho={primary.partial_spearman:.6g}; "
                    f"CI=[{primary.ci95_lower:.6g},{primary.ci95_upper:.6g}]"
                ),
                "criterion": "primary 95% bootstrap interval lower bound > 0",
            },
        ]
    )

    (STAGING / "tables").mkdir(parents=True)
    (STAGING / "reports").mkdir()
    (STAGING / "figures").mkdir()
    pd.DataFrame(input_rows).to_csv(STAGING / "tables/E202_INPUT_HASHES.csv", index=False)
    tasks.to_csv(STAGING / "tables/E202_TASK_REGRETS.csv", index=False)
    associations.to_csv(STAGING / "tables/E202_PARTIAL_ASSOCIATIONS.csv", index=False)
    baseline_sensitivity.to_csv(
        STAGING / "tables/E202_BASELINE_SENSITIVITY.csv", index=False
    )
    regrets.to_csv(STAGING / "tables/E202_ENDPOINT_REGRETS.csv", index=False)
    endpoint_assoc.to_csv(
        STAGING / "tables/E202_ENDPOINT_PARTIAL_ASSOCIATIONS.csv", index=False
    )
    strata.to_csv(STAGING / "tables/E202_MAGNITUDE_STRATA.csv", index=False)
    gates.to_csv(STAGING / "tables/E202_GATES.csv", index=False)
    make_figure(
        tasks,
        associations,
        endpoint_assoc,
        strata,
        STAGING / "figures/E202_residual_task_failure.png",
        STAGING / "figures/E202_residual_task_failure.pdf",
    )
    report = "\n".join(
        [
            "# E202 控制扰动幅度后的模型失败诊断",
            "",
            f"- 主结论：**{'SUPPORTED' if supported else 'NOT SUPPORTED'}**。",
            f"- 主 partial Spearman：`{primary.partial_spearman:.4f}` "
            f"（95% bootstrap 区间 `{primary.ci95_lower:.4f}` 至 "
            f"`{primary.ci95_upper:.4f}`）。",
            f"- 幅度五分位内中心化描述性 Spearman："
            f"`{pooled['within_quintile_centered_spearman']:.4f}`。",
            "",
            "## 主结局上的风险成分",
            "",
            markdown_table(
                associations[
                    [
                        "predictor",
                        "raw_spearman",
                        "partial_spearman",
                        "ci95_lower",
                        "ci95_upper",
                    ]
                ]
            ),
            "",
            "## 五个评价端点",
            "",
            markdown_table(
                endpoint_assoc[
                    [
                        "endpoint",
                        "raw_spearman",
                        "partial_spearman",
                        "ci95_lower",
                        "ci95_upper",
                    ]
                ]
            ),
            "",
            "## 解释边界",
            "",
            "E202 是 E200 之后登记的 K562 诊断。主门槛只判断训练背景分歧在控制"
            "预测幅度后，是否仍与 GAT 相对 general baseline 的额外 RMSE 正相关。"
            "它不构成其他细胞系、模型家族或数据集上的确认性证据。",
            "",
        ]
    )
    (STAGING / "reports/E202_REPORT.md").write_text(report, encoding="utf-8")
    status = {
        "experiment": "E202_residual_task_failure",
        "status": "COMPLETE",
        "finished_at": now(),
        "git_head": repo_text("rev-parse", "HEAD"),
        "n_tasks": N_TASKS,
        "n_bootstrap": N_BOOTSTRAP,
        "primary_partial_spearman": float(primary.partial_spearman),
        "primary_ci95_lower": float(primary.ci95_lower),
        "primary_ci95_upper": float(primary.ci95_upper),
        "independent_signal_status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "post_e200_diagnostic": True,
    }
    (STAGING / "E202_FINAL_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_rows = []
    for path in sorted(STAGING.rglob("*")):
        if path.is_file():
            output_rows.append(
                {
                    "path": path.relative_to(STAGING).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(output_rows).to_csv(STAGING / "E202_OUTPUT_HASHES.csv", index=False)
    STAGING.replace(RELEASE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise
