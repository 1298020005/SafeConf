#!/usr/bin/env python3
"""E58: quantify the stability of E55/E57 cross-dataset risk rankings.

The advisor asked for harder generalization settings and for a precise answer
to the question "what error is the score correlated with?".  E55/E57 score a
target task *before* accessing its truth, then compare the deployable
``risk_cross_dataset`` with ``error_combined_rmse`` afterwards.  This script
does not create a new score.  It measures how stable that already-fixed
comparison is for every source -> target direction.

For each direction it records:
  * observed Spearman(risk_cross_dataset, error_combined_rmse);
  * non-parametric bootstrap 95% CI for that correlation and top-20% error
    enrichment;
  * a two-sided permutation p-value for the correlation;
  * a conservative reporting label based on task count and the CI.

Small directions are retained as exploratory evidence, rather than silently
being presented as equally strong as the larger Lara directions.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E58_cross_dataset_statistical_stability_20260711"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
INPUTS = {
    "E55": ROOT / "docs" / "实验结果" / "E55_cross_dataset_transfer_20260710" / "tables" / "E55_CROSS_DATASET_SCORE_TABLE.csv",
    "E57": ROOT / "docs" / "实验结果" / "E57_dataset_expansion_cross_dataset_20260710" / "tables" / "E57_DATASET_EXPANSION_SCORE_TABLE.csv",
}
SCORE = "risk_cross_dataset"
ERROR = "error_combined_rmse"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def top20_enrichment(score: np.ndarray, error: np.ndarray) -> tuple[int, float]:
    if len(score) < 5:
        return 0, float("nan")
    k = max(1, int(math.ceil(len(score) * 0.2)))
    order = np.argsort(-score, kind="stable")
    overall = float(error.mean())
    value = float(error[order[:k]].mean()) / overall if overall > 1e-12 else float("nan")
    return k, value


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan"), float("nan")
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def analyse_one(group: pd.DataFrame, rng: np.random.Generator, n_boot: int, n_perm: int) -> tuple[dict, pd.DataFrame]:
    clean = group[[SCORE, ERROR]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    score = clean[SCORE].to_numpy(float)
    error = clean[ERROR].to_numpy(float)
    n = len(clean)
    rho = spearman(score, error)
    k, enrich = top20_enrichment(score, error)

    boot_rho = np.full(n_boot, np.nan, dtype=float)
    boot_enrich = np.full(n_boot, np.nan, dtype=float)
    if n >= 3:
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_rho[i] = spearman(score[idx], error[idx])
            _, boot_enrich[i] = top20_enrichment(score[idx], error[idx])
    rho_lo, rho_hi = percentile_ci(boot_rho)
    enrich_lo, enrich_hi = percentile_ci(boot_enrich)

    perm_p = float("nan")
    if n >= 3 and np.isfinite(rho):
        perm = np.full(n_perm, np.nan, dtype=float)
        for i in range(n_perm):
            perm[i] = spearman(score, rng.permutation(error))
        valid = perm[np.isfinite(perm)]
        if len(valid):
            perm_p = float((np.sum(np.abs(valid) >= abs(rho)) + 1) / (len(valid) + 1))

    if n < 10:
        label = "样本不足：保留记录，不作方向性结论"
    elif n < 30:
        label = "探索性：样本量较小，需独立复现"
    elif np.isfinite(rho_lo) and rho_lo > 0 and np.isfinite(perm_p) and perm_p < 0.05:
        label = "稳定正信号：可作为主汇报证据"
    elif np.isfinite(rho_hi) and rho_hi < 0 and np.isfinite(perm_p) and perm_p < 0.05:
        label = "稳定负信号：明确边界"
    else:
        label = "不稳定或无明显排序信号"

    base = group.iloc[0]
    row = {
        "input_batch": base["input_batch"],
        "pair_group": base["pair_group"],
        "directional_pair": base["directional_pair"],
        "source_dataset": base["source_dataset"],
        "target_dataset": base["target_dataset"],
        "n_tasks": n,
        "risk_column": SCORE,
        "error_column": ERROR,
        "spearman_risk_vs_error": rho,
        "bootstrap_rho_ci95_low": rho_lo,
        "bootstrap_rho_ci95_high": rho_hi,
        "permutation_p_two_sided": perm_p,
        "top20_k": k,
        "top20_error_enrichment": enrich,
        "bootstrap_top20_enrichment_ci95_low": enrich_lo,
        "bootstrap_top20_enrichment_ci95_high": enrich_hi,
        "reporting_label": label,
    }
    detail = pd.DataFrame(
        {
            "directional_pair": base["directional_pair"],
            "bootstrap_id": np.arange(n_boot),
            "bootstrap_spearman": boot_rho,
            "bootstrap_top20_enrichment": boot_enrich,
        }
    )
    return row, detail


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "（无行）"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_svg(summary: pd.DataFrame) -> None:
    """White-background confidence-interval figure for direct PPT use."""
    view = summary[summary["n_tasks"] >= 10].sort_values(
        ["spearman_risk_vs_error", "n_tasks"], ascending=[False, False]
    ).copy()
    if view.empty:
        return
    width, row_h, left, right, top = 1760, 46, 500, 120, 118
    height = top + len(view) * row_h + 80
    x0, x1 = left, width - right
    def sx(v: float) -> float:
        return x0 + ((max(-1.0, min(1.0, v)) + 1.0) / 2.0) * (x1 - x0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#253341}.title{font-size:27px;font-weight:700}.sub{font-size:16px;fill:#586773}.lab{font-size:14px}.small{font-size:12px;fill:#586773}</style>',
        '<text class="title" x="56" y="46">E58｜跨数据集风险排序的统计稳定性</text>',
        '<text class="sub" x="56" y="76">点为 Spearman(risk_cross_dataset, error_combined_rmse)，横线为 bootstrap 95% CI；仅显示任务数 ≥ 10 的方向。</text>',
    ]
    for tick in [-1, -0.5, 0, 0.5, 1]:
        x = sx(tick)
        stroke = "#94a3b8" if tick == 0 else "#e2e8f0"
        lines.append(f'<line x1="{x:.1f}" y1="{top-20}" x2="{x:.1f}" y2="{height-46}" stroke="{stroke}" stroke-width="{2 if tick == 0 else 1}"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{top-30}" text-anchor="middle">{tick:g}</text>')
    for i, (_, r) in enumerate(view.iterrows()):
        y = top + i * row_h
        label = str(r["directional_pair"])
        if len(label) > 57:
            label = label[:54] + "…"
        color = "#137c8b" if str(r["reporting_label"]).startswith("稳定正") else ("#b45309" if str(r["reporting_label"]).startswith("稳定负") else "#64748b")
        lo, hi, val = float(r["bootstrap_rho_ci95_low"]), float(r["bootstrap_rho_ci95_high"]), float(r["spearman_risk_vs_error"])
        if not np.isfinite(lo): lo = val
        if not np.isfinite(hi): hi = val
        lines.extend([
            f'<text class="lab" x="{left-14}" y="{y+5}" text-anchor="end">{escape(label)}</text>',
            f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',
            f'<circle cx="{sx(val):.1f}" cy="{y}" r="7" fill="{color}"/>',
            f'<text class="small" x="{width-107}" y="{y+5}" text-anchor="middle">n={int(r["n_tasks"])}，p={float(r["permutation_p_two_sided"]):.3f}</text>',
        ])
    lines.append('</svg>')
    (FIGURES / "F1_cross_dataset_risk_error_bootstrap_ci.svg").write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame, source_rows: int, args: argparse.Namespace) -> None:
    stable = summary[summary["reporting_label"].eq("稳定正信号：可作为主汇报证据")].sort_values("spearman_risk_vs_error", ascending=False)
    negative = summary[summary["reporting_label"].eq("稳定负信号：明确边界")].sort_values("spearman_risk_vs_error")
    exploratory = summary[summary["reporting_label"].str.startswith("探索性")].sort_values("n_tasks")
    lines = [
        "# E58｜跨数据集风险排序：统计稳定性审计",
        "",
        "## 这轮在回答什么",
        "",
        "E55/E57 每个目标任务先用源数据集构造 `risk_cross_dataset`，随后才读取该任务真实效应并计算 `error_combined_rmse`。本轮固定这两个量，逐方向做 bootstrap 与置换检验，防止小任务数方向被过度解读。",
        "",
        "- 输入任务行数：" + f"{source_rows:,}",
        "- 方向数：" + str(len(summary)),
        "- bootstrap 次数：" + str(args.n_boot),
        "- 置换次数：" + str(args.n_perm),
        "- 统计图：`figures/F1_cross_dataset_risk_error_bootstrap_ci.svg`",
        "",
        "## 可以放进主汇报的稳定正信号",
        "",
        markdown_table(stable.head(12), ["directional_pair", "n_tasks", "spearman_risk_vs_error", "bootstrap_rho_ci95_low", "bootstrap_rho_ci95_high", "permutation_p_two_sided", "top20_error_enrichment"]),
        "",
        "这些方向同时满足：任务数至少 30、bootstrap 95% CI 下界大于 0、双侧置换 p < 0.05。",
        "",
        "## 必须保留的边界",
        "",
        markdown_table(negative, ["directional_pair", "n_tasks", "spearman_risk_vs_error", "bootstrap_rho_ci95_low", "bootstrap_rho_ci95_high", "permutation_p_two_sided", "reporting_label"]),
        "",
        "## 探索性方向",
        "",
        markdown_table(exploratory, ["directional_pair", "n_tasks", "spearman_risk_vs_error", "bootstrap_rho_ci95_low", "bootstrap_rho_ci95_high", "reporting_label"]),
        "",
        "## 汇报口径",
        "",
        "可以直接说：分数与固定参考预测器的 `error_combined_rmse` 对照；打分输入不含目标真实效应。跨数据集结果存在稳定正方向，也存在不稳定和负方向，因此结论是风险排序受源—目标相似性、覆盖度和任务数影响，不能把一个方向的好结果外推到所有场景。",
    ]
    (REPORTS / "E58_STATISTICAL_STABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    if args.n_boot < 100 or args.n_perm < 100:
        raise ValueError("Use at least 100 bootstrap and permutation draws.")

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"directional_pair", "pair_group", "source_dataset", "target_dataset", SCORE, ERROR}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"{path}: missing columns {sorted(missing)}")
        df["input_batch"] = name
        frames.append(df)
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(TABLES / "E58_INPUT_TASK_ROWS.csv", index=False)

    rng = np.random.default_rng(args.seed)
    rows, details = [], []
    groups = all_scores.groupby(["input_batch", "pair_group", "directional_pair"], sort=True, dropna=False)
    for i, (_, group) in enumerate(groups, start=1):
        row, detail = analyse_one(group, rng, args.n_boot, args.n_perm)
        rows.append(row)
        details.append(detail)
        print(f"[E58] {i}/{len(groups)} {row['directional_pair']}: n={row['n_tasks']}, rho={row['spearman_risk_vs_error']:.3f}", flush=True)
    summary = pd.DataFrame(rows).sort_values(["input_batch", "pair_group", "directional_pair"], kind="stable")
    bootstrap = pd.concat(details, ignore_index=True)
    summary.to_csv(TABLES / "E58_DIRECTIONAL_STABILITY_SUMMARY.csv", index=False)
    bootstrap.to_csv(TABLES / "E58_BOOTSTRAP_DRAWS.csv", index=False)

    write_svg(summary)
    write_report(summary, len(all_scores), args)
    status = {
        "experiment": "E58_cross_dataset_statistical_stability",
        "created_at": now(),
        "git_head_before_run": git_head(),
        "input_files": {k: str(v.relative_to(ROOT)) for k, v in INPUTS.items()},
        "n_input_task_rows": int(len(all_scores)),
        "n_directional_pairs": int(len(summary)),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "score_column": SCORE,
        "error_column": ERROR,
        "true_effect_used_at_scoring": False,
        "outputs": [
            "tables/E58_INPUT_TASK_ROWS.csv",
            "tables/E58_DIRECTIONAL_STABILITY_SUMMARY.csv",
            "tables/E58_BOOTSTRAP_DRAWS.csv",
            "figures/F1_cross_dataset_risk_error_bootstrap_ci.svg",
            "reports/E58_STATISTICAL_STABILITY_REPORT.md",
        ],
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E58 先看这个\n\n"
        "先读 `reports/E58_STATISTICAL_STABILITY_REPORT.md`。\n\n"
        "图 `figures/F1_cross_dataset_risk_error_bootstrap_ci.svg` 可直接放到组会 PPT；白底，点为相关系数，线为 bootstrap 95% CI。\n\n"
        "本轮不重算预测，也不把目标真实效应加回打分。它只检验 E55/E57 已固定的风险—误差关系是否稳定。\n",
        encoding="utf-8",
    )
    print(f"[E58] wrote {OUT}")


if __name__ == "__main__":
    main()
