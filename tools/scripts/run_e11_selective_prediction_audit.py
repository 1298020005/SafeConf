#!/usr/bin/env python3
"""E11: selective prediction / risk-coverage audit from frozen SafeConf curves.

This script packages existing risk-coverage curves into a method-upgrade audit.
It is deliberately named "audit" rather than "conformal guarantee": the current
data show selective prediction value, but do not yet provide formal risk-control
guarantees.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E11_selective_prediction_audit_20260707"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

RISK_COVERAGE = ROOT / "docs/实验结果/Evidence_to_Claim_20260615/figure_ready_tables/SFIG_RISK_COVERAGE.csv"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_svg_clean(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in df.groupby("dataset_name"):
        g = g.sort_values("coverage", ascending=False)
        row = {
            "dataset_name": dataset,
            "full_mean_rmse": float(g[g["coverage"].eq(1.0)]["full_mean_rmse"].iloc[0]),
            "best_improve_pct": float(g["improve_pct"].max()),
            "worst_improve_pct": float(g["improve_pct"].min()),
            "positive_coverage_points": int((g["improve_pct"] > 0).sum()),
            "n_coverage_points": int(len(g)),
        }
        for cov in [0.9, 0.8, 0.7, 0.6, 0.5]:
            sub = g[g["coverage"].eq(cov)]
            row[f"improve_at_{int(cov*100)}pct_coverage"] = (
                float(sub["improve_pct"].iloc[0]) if len(sub) else np.nan
            )
            row[f"mean_rmse_at_{int(cov*100)}pct_coverage"] = (
                float(sub["mean_rmse"].iloc[0]) if len(sub) else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["selective_prediction_call"] = np.where(
        out["improve_at_80pct_coverage"] > 0,
        "selective prediction reduces RMSE at 80% coverage",
        "boundary / no improvement at 80% coverage",
    )
    return out.sort_values("improve_at_80pct_coverage")


def build_macro(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("coverage", as_index=False)
        .agg(
            macro_mean_rmse=("mean_rmse", "mean"),
            macro_full_mean_rmse=("full_mean_rmse", "mean"),
            macro_improve_pct=("improve_pct", "mean"),
            median_improve_pct=("improve_pct", "median"),
            min_improve_pct=("improve_pct", "min"),
            max_improve_pct=("improve_pct", "max"),
            positive_dataset_count=("improve_pct", lambda s: int((s > 0).sum())),
            n_datasets=("dataset_name", "nunique"),
        )
        .sort_values("coverage", ascending=False)
    )


def plot_curves(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for dataset, g in df.groupby("dataset_name"):
        g = g.sort_values("coverage")
        ax.plot(g["coverage"], g["improve_pct"], marker="o", linewidth=1.7, label=dataset)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Mean RMSE improvement vs full coverage (%)")
    ax.set_title("Selective prediction audit: risk-coverage curves")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    path = FIGURES / "E11_fig1_risk_coverage_curves.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_summary(summary: pd.DataFrame) -> Path:
    df = summary.sort_values("improve_at_80pct_coverage")
    colors = np.where(df["improve_at_80pct_coverage"] > 0, "#047857", "#b45309")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(df["dataset_name"], df["improve_at_80pct_coverage"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Improvement at 80% coverage (%)")
    ax.set_title("At 80% retained predictions, most datasets improve")
    fig.tight_layout()
    path = FIGURES / "E11_fig2_improvement_at_80pct_coverage.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_macro(macro: pd.DataFrame) -> Path:
    df = macro.sort_values("coverage")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(df["coverage"], df["macro_improve_pct"], marker="o", color="#047857", label="mean")
    ax.plot(df["coverage"], df["median_improve_pct"], marker="s", color="#2563eb", label="median")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Improvement vs full coverage (%)")
    ax.set_title("Macro selective-prediction gain")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGURES / "E11_fig3_macro_selective_gain.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def write_report(df: pd.DataFrame, summary: pd.DataFrame, macro: pd.DataFrame, figs: list[Path]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row80 = macro[macro["coverage"].eq(0.8)].iloc[0]
    row50 = macro[macro["coverage"].eq(0.5)].iloc[0]
    pos80 = int(row80["positive_dataset_count"])
    total = int(row80["n_datasets"])
    mc = summary[summary["dataset_name"].eq("McFarlandTsherniak2020")].iloc[0]

    report = f"""# E11 Selective prediction / risk-coverage 审计报告

生成时间：{now}

## 核心结论

现有 SafeConf risk-coverage 曲线显示：当只保留低风险预测、拒绝最高风险的 20% 预测时，{pos80}/{total} 个数据集的平均 RMSE 下降；宏平均 improvement = {row80['macro_improve_pct']:.2f}%，中位数 improvement = {row80['median_improve_pct']:.2f}%。

保留 50% 低风险预测时，宏平均 improvement = {row50['macro_improve_pct']:.2f}%，但这更像高强度分诊，实际 wet-lab 场景要结合复核成本。

McFarland 是边界：80% coverage improvement = {mc['improve_at_80pct_coverage']:.2f}%，50% coverage improvement = {mc['improve_at_50pct_coverage']:.2f}%。论文中需要把它保留为 failure/boundary case。

## 投稿价值

E11 把问题从“风险分数与误差是否相关”推进到“模型在有限覆盖率下是否能降低平均错误”。这更接近 selective prediction，也更适合作为 CCF-A 方法升级的入口。

当前仍缺 formal risk guarantee。下一步如果要冲 CCF-A，需要引入 calibration split，对每个 coverage / risk budget 给出可复现阈值，并报告 held-out risk control 是否满足目标。

## 自动生成图

{chr(10).join(f'- `{p.relative_to(OUT).as_posix()}`' for p in figs)}
"""
    (REPORTS / "E11_SELECTIVE_PREDICTION_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E11 Selective prediction audit</title>
<style>
body{{margin:0;background:#f7f8f6;color:#17202a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Noto Sans SC",sans-serif;line-height:1.7}}
.top{{background:#12312b;color:white;padding:28px 42px}}.wrap{{max-width:1120px;margin:0 auto;padding:28px}}
.card{{background:white;border:1px solid #d8e0dc;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 22px rgba(15,23,42,.06);overflow-x:auto}}
h2{{border-bottom:3px solid #0f766e;padding-bottom:8px}}.ok{{border-left:6px solid #047857;background:#f1fbf6}}.warn{{border-left:6px solid #b45309;background:#fff8eb}}
img{{max-width:100%;background:white;border:1px solid #d8e0dc;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dbe3df;padding:7px;vertical-align:top}}th{{background:#eef6f3}}
</style></head><body>
<div class="top"><h1>E11 Selective prediction audit</h1><p>把 SafeConf 从风险相关性推进到 risk-coverage / selective verification。</p></div>
<div class="wrap">
<div class="card ok"><h2>结论</h2><p>80% coverage 下，{pos80}/{total} 个数据集平均 RMSE 下降；宏平均 improvement = {row80['macro_improve_pct']:.2f}%。这支持 selective prediction 方向，但还不是 conformal guarantee。</p></div>
<div class="card warn"><h2>边界</h2><p>McFarland 在部分 coverage 下不稳定，需要保留为 failure/boundary case。CCF-A 版本还要补 calibration split 和 risk-control guarantee。</p></div>
<div class="card"><h2>Risk-coverage curves</h2><img src="../figures/E11_fig1_risk_coverage_curves.svg"></div>
<div class="card"><h2>80% coverage improvement</h2><img src="../figures/E11_fig2_improvement_at_80pct_coverage.svg"></div>
<div class="card"><h2>Macro gain</h2><img src="../figures/E11_fig3_macro_selective_gain.svg"></div>
<div class="card"><h2>Dataset summary</h2>{summary.to_html(index=False, escape=True)}</div>
</div></body></html>
"""
    (REPORTS / "E11_SELECTIVE_PREDICTION_AUDIT.html").write_text(html, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(RISK_COVERAGE)
    summary = build_summary(df)
    macro = build_macro(df)

    df.to_csv(TABLES / "E11_RISK_COVERAGE_CURVES.csv", index=False)
    summary.to_csv(TABLES / "E11_DATASET_SELECTIVE_SUMMARY.csv", index=False)
    macro.to_csv(TABLES / "E11_MACRO_SELECTIVE_SUMMARY.csv", index=False)

    figs = [plot_curves(df), plot_summary(summary), plot_macro(macro)]
    write_report(df, summary, macro, figs)

    readme = """# E11 selective prediction audit

入口：

- `reports/E11_SELECTIVE_PREDICTION_AUDIT.html`
- `reports/E11_SELECTIVE_PREDICTION_AUDIT_REPORT.md`

运行命令：

```bash
python3 tools/scripts/run_e11_selective_prediction_audit.py
```
"""
    (OUT / "README_先看这个.md").write_text(readme, encoding="utf-8")
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "input_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip(),
        "source_files": {
            "risk_coverage": str(RISK_COVERAGE.relative_to(ROOT)),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote E11 audit to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
