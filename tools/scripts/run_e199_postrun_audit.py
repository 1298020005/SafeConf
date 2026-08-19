#!/usr/bin/env python3
"""Run the explicitly post-hoc E199 robustness audit.

The formal E199 result remains immutable.  This audit addresses two
interpretation issues documented after formal evaluation: the algebraic
appearance of diversity in family RMS error, and the denominator of the
target-gene direction diagnostic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "docs/实验结果/E199_txpert_public_k562_20260802"
FORMAL = EXPERIMENT / "formal_evaluation"
PROTOCOL = EXPERIMENT / "POSTRUN_AUDIT_PROTOCOL.md"
OUTPUT = EXPERIMENT / "postrun_audit"
TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"
FIGURES = OUTPUT / "figures"

FORMAL_MANIFEST_SHA = (
    "6c87c9e8d7eaff19b177e74fbc64e3b51e7dd94f54a9688a74b65bfdadb23f8c"
)
PROTOCOL_SHA = (
    "e24db60dc93dedb1b50b924c05b86bf3245949430ae865958d5ba7991083b621"
)
FORMAL_SCRIPT = ROOT / "tools/scripts/run_e199_txpert_evaluation.py"

OUTCOMES = (
    "centroid_error",
    "family_worst_error",
    "gat_rmse",
    "exphormer_rmse",
    "exphormer_mg_rmse",
)
PREDICTORS = ("diversity_pretruth", "predicted_magnitude")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_formal_module():
    spec = importlib.util.spec_from_file_location("e199_formal", FORMAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import E199 formal runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs(formal) -> tuple[str, pd.DataFrame]:
    if OUTPUT.exists():
        raise RuntimeError("postrun_audit is append-only and already exists")
    if sha256_file(FORMAL / "E199_OUTPUT_HASHES.csv") != FORMAL_MANIFEST_SHA:
        raise RuntimeError("formal output manifest changed")
    if sha256_file(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("postrun audit protocol changed")
    manifest = pd.read_csv(FORMAL / "E199_OUTPUT_HASHES.csv")
    for row in manifest.itertuples(index=False):
        path = FORMAL / row.path
        if (
            not path.is_file()
            or path.stat().st_size != row.bytes
            or sha256_file(path) != row.sha256
        ):
            raise RuntimeError(f"formal output changed: {row.path}")
    status = json.loads((FORMAL / "E199_FINAL_STATUS.json").read_text())
    if status.get("status") != "PASS" or status.get("n_primary_tasks") != 263:
        raise RuntimeError("formal status contract failed")
    return formal.verify_git_release(), manifest


def risk_outcomes(formal, tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for predictor in PREDICTORS:
            association = formal.bootstrap_spearman(tasks, predictor, outcome)
            utility = formal.bootstrap_utility(tasks, predictor, outcome)
            rows.append(
                {
                    "analysis_status": "POST_HOC_EXPLORATORY",
                    "outcome": outcome,
                    "predictor": predictor,
                    "n_tasks": len(tasks),
                    **association,
                    "review_budget": utility["budget"],
                    "n_selected": utility["n_selected"],
                    "high_error_capture": utility["high_error_capture"],
                    "error_lift": utility["error_lift"],
                    "oracle_normalized_utility": utility[
                        "oracle_normalized_utility"
                    ],
                    "utility_ci95_lower": utility["utility_ci95_lower"],
                    "utility_ci95_upper": utility["utility_ci95_upper"],
                    "utility_bootstrap_valid": utility[
                        "utility_bootstrap_valid"
                    ],
                }
            )
    return pd.DataFrame(rows)


def incremental_tests(formal, tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for row in formal.paired_increment(
            tasks,
            "diversity_pretruth",
            "predicted_magnitude",
            outcome,
        ):
            rows.append(
                {"analysis_status": "POST_HOC_EXPLORATORY", **row}
            )
    return pd.DataFrame(rows)


def endpoint_audit(
    formal, tasks: pd.DataFrame, endpoint_metrics: pd.DataFrame
) -> pd.DataFrame:
    family = endpoint_metrics.loc[
        endpoint_metrics.stratum.eq("primary_ge30")
        & endpoint_metrics.predictor.eq("family_centroid")
    ]
    rows = []
    for endpoint, block in family.groupby("endpoint", observed=True):
        merged = tasks.merge(
            block[["task_id", "oriented_error"]],
            on="task_id",
            validate="one_to_one",
        )
        for predictor in PREDICTORS:
            association = formal.bootstrap_spearman(
                merged, predictor, "oriented_error"
            )
            utility = formal.bootstrap_utility(
                merged, predictor, "oriented_error"
            )
            rows.append(
                {
                    "analysis_status": "POST_HOC_EXPLORATORY",
                    "endpoint": endpoint,
                    "predictor": predictor,
                    "n_tasks": len(merged),
                    **association,
                    "oracle_normalized_utility": utility[
                        "oracle_normalized_utility"
                    ],
                    "utility_ci95_lower": utility["utility_ci95_lower"],
                    "utility_ci95_upper": utility["utility_ci95_upper"],
                }
            )
    return pd.DataFrame(rows)


def target_direction(tasks_all: pd.DataFrame) -> pd.DataFrame:
    strata = {
        "all_ge10": tasks_all.n_cells.ge(10),
        "primary_ge30": tasks_all.n_cells.ge(30),
        "sensitivity_10_29": tasks_all.n_cells.between(10, 29),
    }
    rows = []
    for stratum, mask in strata.items():
        block = tasks_all.loc[mask]
        valid = block.target_direction_correct.dropna()
        rows.append(
            {
                "analysis_status": "POST_HOC_DENOMINATOR_CLARIFICATION",
                "stratum": stratum,
                "n_total_tasks": len(block),
                "n_evaluable_target_genes": len(valid),
                "n_direction_correct": int(valid.sum()),
                "direction_accuracy": float(valid.mean()),
                "n_direction_incorrect": int((1 - valid).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(risk: pd.DataFrame, endpoints: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    risk_div = risk.loc[risk.predictor.eq("diversity_pretruth")].set_index(
        "outcome"
    ).loc[list(OUTCOMES)]
    y = np.arange(len(risk_div))
    axes[0].errorbar(
        risk_div.spearman,
        y,
        xerr=np.vstack(
            [
                risk_div.spearman - risk_div.ci95_lower,
                risk_div.ci95_upper - risk_div.spearman,
            ]
        ),
        fmt="o",
        color="#4C78A8",
        ecolor="#555555",
        capsize=2,
    )
    axes[0].axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set_yticks(
        y,
        [
            "Family centroid",
            "Worst member",
            "GAT",
            "Exphormer",
            "Exphormer-MG",
        ],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Spearman: diversity vs RMSE")
    axes[0].set_title("A  Non-algebraic error outcomes")

    endpoint_order = list(formal_endpoint_order())
    end_div = endpoints.loc[
        endpoints.predictor.eq("diversity_pretruth")
    ].set_index("endpoint").loc[endpoint_order]
    y2 = np.arange(len(end_div))
    axes[1].errorbar(
        end_div.spearman,
        y2,
        xerr=np.vstack(
            [
                end_div.spearman - end_div.ci95_lower,
                end_div.ci95_upper - end_div.spearman,
            ]
        ),
        fmt="o",
        color="#E45756",
        ecolor="#555555",
        capsize=2,
    )
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(
        y2, [name.replace("_pca_k=50", " PCA50") for name in endpoint_order]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Spearman: diversity vs oriented error")
    axes[1].set_title("B  Endpoint dependence")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.55)
        axis.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def formal_endpoint_order() -> tuple[str, ...]:
    return (
        "mse",
        "pearson_pert",
        "rank",
        "energy_distance_pca_k=50",
        "de_auprc",
    )


def output_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in OUTPUT.rglob("*") if p.is_file()):
        if path.name == "E199_POSTRUN_OUTPUT_HASHES.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(OUTPUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    formal = load_formal_module()
    head, input_manifest = verify_inputs(formal)
    tasks_all = pd.read_csv(FORMAL / "tables/E199_TASK_CERTIFICATE.csv")
    tasks = tasks_all.loc[tasks_all.n_cells.ge(30)].copy()
    if len(tasks) != 263:
        raise RuntimeError("primary task count changed")
    tasks["diversity_pretruth"] = tasks[
        "diversity_lower_bound_pretruth"
    ].astype(float)
    endpoint_metrics = pd.read_csv(
        FORMAL / "tables/E199_SCPERTEVAL_TASK_METRICS.csv"
    )

    risk = risk_outcomes(formal, tasks)
    increments = incremental_tests(formal, tasks)
    endpoints = endpoint_audit(formal, tasks, endpoint_metrics)
    target = target_direction(tasks_all)

    for directory in (TABLES, REPORTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    formal.write_csv(TABLES / "E199_POSTHOC_RISK_OUTCOMES.csv", risk)
    formal.write_csv(TABLES / "E199_POSTHOC_INCREMENTAL_TESTS.csv", increments)
    formal.write_csv(TABLES / "E199_POSTHOC_ENDPOINT_AUDIT.csv", endpoints)
    formal.write_csv(TABLES / "E199_TARGET_GENE_DIRECTION_DENOMINATOR.csv", target)
    make_figure(risk, endpoints, FIGURES / "E199_postrun_robustness.png")

    diversity = risk.loc[risk.predictor.eq("diversity_pretruth")]
    centroid = diversity.loc[diversity.outcome.eq("centroid_error")].iloc[0]
    target_primary = target.loc[target.stratum.eq("primary_ge30")].iloc[0]
    endpoint_div = endpoints.loc[
        endpoints.predictor.eq("diversity_pretruth"),
        ["endpoint", "spearman", "ci95_lower", "ci95_upper"],
    ]
    report = [
        "# E199 运行后稳健性审计",
        "",
        "> **POST HOC / EXPLORATORY。** 本报告不改变正式 gate。",
        "",
        "## 不含 diversity 代数项的误差",
        "",
        formal.markdown_table(
            diversity[
                [
                    "outcome",
                    "spearman",
                    "ci95_lower",
                    "ci95_upper",
                    "oracle_normalized_utility",
                    "utility_ci95_lower",
                    "utility_ci95_upper",
                ]
            ]
        ),
        "",
        f"对等权均值真实 RMSE，diversity 的 Spearman 为 {centroid.spearman:.3f} "
        f"（95% CI {centroid.ci95_lower:.3f}–{centroid.ci95_upper:.3f}）；"
        f"20% 复核效用为 {centroid.oracle_normalized_utility:.3f} "
        f"（95% CI {centroid.utility_ci95_lower:.3f}–{centroid.utility_ci95_upper:.3f}）。",
        "",
        "## 五个端点的依赖性",
        "",
        formal.markdown_table(endpoint_div),
        "",
        "MSE 和群体距离上的方向较清楚；Pearson、检索 rank 与 DE-AUPRC 不显示同样的"
        "一致性。因此 E199 支持的是特定误差口径下的风险路由，不支持跨端点通用声明。",
        "",
        "## 目标基因方向分母",
        "",
        formal.markdown_table(target),
        "",
        f"主分析 263 个任务中只有 {int(target_primary.n_evaluable_target_genes)} 个目标基因"
        f"位于 5,000 基因面板；其中 {int(target_primary.n_direction_correct)} 个方向正确，"
        f"命中率 {target_primary.direction_accuracy:.3f}。",
        "",
        "## 仍未回答",
        "",
        "整个细胞背景留出和跨数据集迁移仍未回答，不能由 E199 外推。",
        "",
    ]
    formal.write_text(REPORTS / "E199_POSTRUN_AUDIT.md", "\n".join(report))
    status = {
        "experiment": "E199_txpert_public_k562",
        "stage": "POSTRUN_ROBUSTNESS_AUDIT",
        "analysis_status": "POST_HOC_EXPLORATORY",
        "status": "PASS",
        "git_head": head,
        "formal_manifest_sha256": FORMAL_MANIFEST_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "formal_manifest_rows_verified": len(input_manifest),
        "n_primary_tasks": len(tasks),
        "n_target_direction_evaluable": int(
            target_primary.n_evaluable_target_genes
        ),
        "n_target_direction_correct": int(target_primary.n_direction_correct),
        "cross_context_answered": False,
        "cross_dataset_transfer_answered": False,
        "formal_gate_changed": False,
    }
    formal.write_json(OUTPUT / "E199_POSTRUN_STATUS.json", status)
    formal.write_csv(
        OUTPUT / "E199_POSTRUN_OUTPUT_HASHES.csv", output_manifest()
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
