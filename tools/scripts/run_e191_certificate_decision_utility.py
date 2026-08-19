#!/usr/bin/env python3
"""Measure review-budget utility of E189/E190 frozen risk quantities."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/实验结果/E191_certificate_decision_utility_20260729"
E189 = (
    ROOT
    / "docs/实验结果/E189_primary_cd4_formal_cartesian_20260729"
    / "tables/E189_TASK_METRICS.csv"
)
E190 = (
    ROOT
    / "docs/实验结果/E190_adamson_to_replogle_direct_transfer_20260729"
    / "final_evaluation/tables/E190_TASK_METRICS.csv"
)
BUDGETS = (0.10, 0.20, 0.30)


def stable_key(value: str) -> str:
    return hashlib.sha256(f"E191\0{value}".encode()).hexdigest()


def choose_top(frame: pd.DataFrame, column: str, n_select: int) -> set[str]:
    ordered = frame.assign(
        tie_key=[stable_key(task_id) for task_id in frame.e191_task_id.astype(str)]
    ).sort_values([column, "tie_key"], ascending=[False, True])
    return set(ordered.head(n_select).e191_task_id.astype(str))


def analyze_stratum(
    frame: pd.DataFrame,
    experiment: str,
    stratum: str,
    outcome: str,
    predictors: list[tuple[str, str, bool]],
) -> list[dict[str, Any]]:
    rows = []
    for budget in BUDGETS:
        n_select = int(math.ceil(len(frame) * budget))
        oracle_ids = choose_top(frame, outcome, n_select)
        oracle_mean = frame.loc[
            frame.e191_task_id.astype(str).isin(oracle_ids), outcome
        ].mean()
        overall = frame[outcome].mean()
        denominator = oracle_mean - overall
        for label, predictor, is_lower_bound in predictors:
            selected_ids = choose_top(frame, predictor, n_select)
            selected = frame.loc[
                frame.e191_task_id.astype(str).isin(selected_ids)
            ]
            selected_mean = selected[outcome].mean()
            intersection = len(selected_ids & oracle_ids)
            utility = (
                (selected_mean - overall) / denominator
                if denominator > 1e-15
                else np.nan
            )
            rows.append(
                {
                    "experiment": experiment,
                    "stratum": stratum,
                    "outcome": outcome,
                    "predictor": label,
                    "budget": budget,
                    "n_tasks": len(frame),
                    "n_selected": n_select,
                    "high_error_capture": intersection / n_select,
                    "random_expected_capture": n_select / len(frame),
                    "selected_mean_error": selected_mean,
                    "overall_mean_error": overall,
                    "error_lift": selected_mean / overall,
                    "oracle_mean_error": oracle_mean,
                    "oracle_normalized_utility": utility,
                    "selected_mean_bound_to_error_ratio": (
                        float(np.mean(selected[predictor] / selected[outcome]))
                        if is_lower_bound
                        else np.nan
                    ),
                }
            )
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_figure(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    e189 = results.loc[
        results.experiment.eq("E189")
        & results.outcome.eq("family_rms_error")
        & results.budget.eq(0.20)
    ].copy()
    settings = [
        "random_missing_pair",
        "unseen_context_row",
        "unseen_perturbation_column",
        "double_unseen",
    ]
    labels = ["Random pair", "Unseen row", "Unseen column", "Both unseen"]
    for predictor, color, marker in (
        ("diversity_lower_bound", "#3C5488", "o"),
        ("predicted_magnitude", "#E64B35", "s"),
    ):
        values = []
        for setting in settings:
            group = e189.loc[
                e189.stratum.str.endswith(f"::{setting}")
                & e189.predictor.eq(predictor)
            ]
            values.append(group.oracle_normalized_utility.mean())
        axes[0].plot(
            range(4), values, marker=marker, color=color, label=predictor
        )
    axes[0].set_xticks(range(4), labels, rotation=28, ha="right")
    axes[0].set_ylabel("Oracle-normalized utility")
    axes[0].set_title("A  E189, 20% review")
    axes[0].legend(frameon=False, fontsize=7)

    e190 = results.loc[
        results.experiment.eq("E190")
        & results.outcome.eq("family_rms_error")
        & results.budget.eq(0.20)
    ].sort_values("predictor")
    axes[1].bar(
        range(len(e190)),
        e190.high_error_capture,
        color=["#3C5488", "#E64B35", "#7E6148"][: len(e190)],
    )
    axes[1].axhline(0.20, linestyle="--", color="#777777", linewidth=1)
    axes[1].set_xticks(
        range(len(e190)),
        e190.predictor.str.replace("_", " "),
        rotation=28,
        ha="right",
    )
    axes[1].set_ylabel("High-error capture")
    axes[1].set_title("B  E190, 20% review")

    lower = results.loc[
        results.predictor.isin(
            ["diversity_lower_bound", "diameter_half_lower_bound"]
        )
        & results.budget.eq(0.20)
    ]
    groups = (
        lower.groupby(["experiment", "predictor"], observed=True)
        .selected_mean_bound_to_error_ratio.mean()
        .reset_index()
    )
    axes[2].bar(
        range(len(groups)),
        groups.selected_mean_bound_to_error_ratio,
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(groups)],
    )
    axes[2].set_xticks(
        range(len(groups)),
        groups.experiment + "\n" + groups.predictor.str.replace("_lower_bound", ""),
        rotation=20,
        ha="right",
    )
    axes[2].set_ylabel("Mean lower bound / error")
    axes[2].set_title("C  Certificate tightness")

    for axis in axes:
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.6)
        axis.tick_params(labelsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    e189 = pd.read_csv(E189, keep_default_na=False)
    e190 = pd.read_csv(E190, keep_default_na=False)
    rows = []
    for (support, setting), group in e189.groupby(
        ["support", "e189_setting"], observed=True, sort=True
    ):
        group = group.copy()
        group["e191_task_id"] = (
            group.panel_id.astype(str)
            + "::"
            + group.task_id.astype(str)
        )
        stratum = f"support_{support}::{setting}"
        rows.extend(
            analyze_stratum(
                group,
                "E189",
                stratum,
                "family_rms_error",
                [
                    ("diversity_lower_bound", "diversity_lower_bound", True),
                    (
                        "predicted_magnitude",
                        "centroid_predicted_magnitude",
                        False,
                    ),
                ],
            )
        )
        rows.extend(
            analyze_stratum(
                group,
                "E189",
                stratum,
                "family_worst_error",
                [
                    (
                        "diameter_half_lower_bound",
                        "diameter_half_lower_bound",
                        True,
                    ),
                    (
                        "predicted_magnitude",
                        "centroid_predicted_magnitude",
                        False,
                    ),
                ],
            )
        )
    e190 = e190.copy()
    e190["e191_task_id"] = e190.task_id.astype(str)
    rows.extend(
        analyze_stratum(
            e190,
            "E190",
            "adamson_to_replogle",
            "family_rms_error",
            [
                ("diversity_lower_bound", "diversity_lower_bound", True),
                ("predicted_magnitude", "predicted_magnitude", False),
                ("source_effect_magnitude", "source_effect_magnitude", False),
            ],
        )
    )
    rows.extend(
        analyze_stratum(
            e190,
            "E190",
            "adamson_to_replogle",
            "family_worst_error",
            [
                (
                    "diameter_half_lower_bound",
                    "diameter_half_lower_bound",
                    True,
                ),
                ("predicted_magnitude", "predicted_magnitude", False),
                ("source_effect_magnitude", "source_effect_magnitude", False),
            ],
        )
    )
    results = pd.DataFrame(rows)
    if len(results) != 210:
        raise RuntimeError(f"E191 registered row count changed: {len(results)}")

    tables = OUT / "tables"
    figures = OUT / "figures"
    reports = OUT / "reports"
    for directory in (tables, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables / "E191_BUDGET_UTILITY.csv", index=False)
    make_figure(results, figures / "E191_decision_utility.png")

    primary = results.loc[
        results.budget.eq(0.20) & results.outcome.eq("family_rms_error")
    ].copy()
    e189_pair = primary.loc[
        primary.experiment.eq("E189")
        & primary.predictor.isin(
            ["diversity_lower_bound", "predicted_magnitude"]
        )
    ].pivot(
        index="stratum",
        columns="predictor",
        values="oracle_normalized_utility",
    )
    e189_pair["diversity_minus_magnitude"] = (
        e189_pair.diversity_lower_bound - e189_pair.predicted_magnitude
    )
    e190_primary = primary.loc[primary.experiment.eq("E190")][
        [
            "predictor",
            "high_error_capture",
            "error_lift",
            "oracle_normalized_utility",
            "selected_mean_bound_to_error_ratio",
        ]
    ]
    summary = {
        "experiment": "E191",
        "status": "PASS",
        "registered_rows": len(results),
        "e189_strata_at_primary_budget": len(e189_pair),
        "e189_diversity_beats_magnitude_strata": int(
            (e189_pair.diversity_minus_magnitude > 0).sum()
        ),
        "e189_diversity_loses_to_magnitude_strata": int(
            (e189_pair.diversity_minus_magnitude < 0).sum()
        ),
        "e190_primary_budget": 0.20,
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# E191 有限复核预算结果",
        "",
        f"状态：**PASS**。E189 的 20% 主预算共有 {len(e189_pair)} 个 setting×support "
        f"层；diversity utility 高于 magnitude 的层数为 "
        f"{summary['e189_diversity_beats_magnitude_strata']}，低于的层数为 "
        f"{summary['e189_diversity_loses_to_magnitude_strata']}。",
        "",
        "## E190：20% 复核预算",
        "",
        markdown_table(e190_primary),
        "",
        "下界始终正确不代表排序收益始终更高。完整的 10%/20%/30% 结果见 "
        "`../tables/E191_BUDGET_UTILITY.csv`。",
        "",
    ]
    (reports / "E191_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
