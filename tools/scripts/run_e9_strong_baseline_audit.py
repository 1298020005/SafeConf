#!/usr/bin/env python3
"""E9: strong baseline audit from frozen SafeConf result tables.

This is a lightweight, reproducible audit.  It intentionally does not claim to
rerun the full perturbation prediction pipeline; instead it consolidates frozen
summary tables to answer the reviewer-facing question:

"Is SafeConf genuinely useful beyond the very strong predicted-magnitude baseline?"

Outputs:
    docs/实验结果/E9_strong_baseline_audit_20260707/
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
OUT = ROOT / "docs" / "实验结果" / "E9_strong_baseline_audit_20260707"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

BASELINE = ROOT / "docs/实验结果/Reliability_model_corrected_20260610/tables/RELIABILITY_BASELINE_LADDER.csv"
E2 = ROOT / "docs/实验结果/E1_E4_preregistered_20260614/E2_magnitude_residual/E2_MAGNITUDE_RESIDUAL_SUMMARY.csv"
TAHOE = ROOT / "docs/实验结果/Tahoe_D5_combined_triage_20260627/tables/TAHOE_D5_POINT_SUMMARY.csv"


DEPLOYABLE = [
    "predicted_magnitude",
    "protocol_v0_2_family_confidence",
    "safeconf_lodo_risk",
    "safeconf_lodo_linear_risk",
]

LABEL = {
    "predicted_magnitude": "Magnitude-only",
    "protocol_v0_2_family_confidence": "Frozen v0.2",
    "safeconf_lodo_risk": "SafeConf LODO",
    "safeconf_lodo_linear_risk": "SafeConf LODO linear",
    "safeconf_perdataset_risk": "SafeConf per-dataset reference",
    "oracle_magnitude_diagnostic": "Oracle true magnitude",
    "random": "Random",
    "safeconf_full": "SafeConf full",
    "combined_equal": "Combined equal",
    "combined_magnitude75": "Combined 75% magnitude",
    "combined_safeconf75": "Combined 75% SafeConf",
}


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


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_deployable_ladder(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["score_name"].isin(DEPLOYABLE)].copy()
    out["score_label"] = out["score_name"].map(LABEL)
    out["rank_aligned_rho"] = out.groupby("dataset_name")["aligned_rho"].rank(
        ascending=False, method="min"
    )
    out["rank_aurc_reduction"] = out.groupby("dataset_name")[
        "aurc_reduction_vs_random_pct"
    ].rank(ascending=False, method="min")
    out["rank_partial_after_magnitude"] = out.groupby("dataset_name")[
        "partial_rho_control_magnitude"
    ].rank(ascending=False, method="min")
    return out.sort_values(["dataset_name", "rank_aligned_rho"])


def build_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "aligned_rho",
        "partial_rho_control_magnitude",
        "aurc_reduction_vs_random_pct",
        "avoidable_gap_captured_pct",
    ]
    pivot = df[df["score_name"].isin(["predicted_magnitude", "protocol_v0_2_family_confidence"])].pivot(
        index="dataset_name", columns="score_name", values=metrics
    )
    rows = []
    for dataset in pivot.index:
        row = {"dataset_name": dataset}
        for metric in metrics:
            mag = float(pivot.loc[dataset, (metric, "predicted_magnitude")])
            safe = float(pivot.loc[dataset, (metric, "protocol_v0_2_family_confidence")])
            row[f"magnitude_{metric}"] = mag
            row[f"frozen_v02_{metric}"] = safe
            row[f"delta_frozen_minus_magnitude_{metric}"] = safe - mag
        row["frozen_beats_magnitude_aligned"] = row[
            "delta_frozen_minus_magnitude_aligned_rho"
        ] > 0
        row["frozen_beats_magnitude_aurc"] = row[
            "delta_frozen_minus_magnitude_aurc_reduction_vs_random_pct"
        ] > 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_frozen_minus_magnitude_aligned_rho")


def build_winner_table(ladder: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in ladder.groupby("dataset_name"):
        for metric, col in [
            ("aligned rho", "aligned_rho"),
            ("AURC reduction", "aurc_reduction_vs_random_pct"),
            ("partial rho after magnitude control", "partial_rho_control_magnitude"),
        ]:
            winner = g.sort_values(col, ascending=False).iloc[0]
            rows.append(
                {
                    "dataset_name": dataset,
                    "metric": metric,
                    "winner_score_name": winner["score_name"],
                    "winner_label": winner["score_label"],
                    "winner_value": float(winner[col]),
                }
            )
    return pd.DataFrame(rows)


def build_e2_incremental() -> pd.DataFrame:
    df = pd.read_csv(E2)
    df = df[df["calibration_method"].eq("isotonic")].copy()
    df["incremental_positive"] = df["residual_partial_rho_ci_low"] > 0
    df["aurc_combined_better_than_magnitude"] = (
        df["aurc_improvement_magnitude_minus_combined_ci_low"] > 0
    )
    return df


def build_tahoe_audit() -> pd.DataFrame:
    df = pd.read_csv(TAHOE)
    df["score_label"] = df["score_name"].map(LABEL).fillna(df["score_name"])
    mag = (
        df[df["score_name"].eq("predicted_magnitude")]
        .set_index("top_fraction")[["aligned_rho", "enrichment", "precision"]]
        .rename(
            columns={
                "aligned_rho": "magnitude_aligned_rho",
                "enrichment": "magnitude_enrichment",
                "precision": "magnitude_precision",
            }
        )
    )
    out = df.join(mag, on="top_fraction")
    out["delta_enrichment_vs_magnitude"] = out["enrichment"] - out["magnitude_enrichment"]
    out["delta_precision_vs_magnitude"] = out["precision"] - out["magnitude_precision"]
    out["chemical_boundary_call"] = np.where(
        out["score_name"].eq("predicted_magnitude"),
        "dominant chemical baseline",
        np.where(
            out["delta_enrichment_vs_magnitude"] >= 0,
            "matches_or_beats_magnitude",
            "below_magnitude_boundary",
        ),
    )
    return out


def plot_head_to_head(h2h: pd.DataFrame) -> Path:
    df = h2h.sort_values("delta_frozen_minus_magnitude_aligned_rho")
    colors = np.where(df["delta_frozen_minus_magnitude_aligned_rho"] >= 0, "#047857", "#b45309")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(df["dataset_name"], df["delta_frozen_minus_magnitude_aligned_rho"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Frozen v0.2 aligned rho - magnitude aligned rho")
    ax.set_title("Direct head-to-head: frozen SafeConf vs magnitude")
    fig.tight_layout()
    path = FIGURES / "E9_fig1_frozen_vs_magnitude_aligned_delta.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_e2(e2: pd.DataFrame) -> Path:
    df = e2.sort_values("residual_partial_rho_point")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = df["residual_partial_rho_point"].to_numpy()
    lo = df["residual_partial_rho_ci_low"].to_numpy()
    hi = df["residual_partial_rho_ci_high"].to_numpy()
    y = np.arange(len(df))
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="o",
        color="#047857",
        ecolor="#94a3b8",
        capsize=3,
    )
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset_name"])
    ax.set_xlabel("Residual partial rho")
    ax.set_title("Incremental value after magnitude control")
    fig.tight_layout()
    path = FIGURES / "E9_fig2_incremental_value_after_magnitude_control.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def plot_tahoe(tahoe: pd.DataFrame) -> Path:
    df = tahoe[tahoe["top_fraction"].eq(0.10)].sort_values("enrichment")
    colors = np.where(df["score_name"].eq("predicted_magnitude"), "#b45309", "#047857")
    colors = np.where(df["score_name"].str.startswith("combined"), "#2563eb", colors)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(df["score_label"], df["enrichment"], color=colors)
    ax.axvline(1, color="#111827", linewidth=0.8)
    ax.set_xlabel("Top-10% enrichment over random")
    ax.set_title("Tahoe chemical: magnitude is the boundary baseline")
    fig.tight_layout()
    path = FIGURES / "E9_fig3_tahoe_top10_boundary.svg"
    save_svg_clean(fig, path)
    plt.close(fig)
    return path


def write_report(
    ladder: pd.DataFrame,
    h2h: pd.DataFrame,
    winners: pd.DataFrame,
    e2: pd.DataFrame,
    tahoe: pd.DataFrame,
    figs: list[Path],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    aligned_wins = int(h2h["frozen_beats_magnitude_aligned"].sum())
    aurc_wins = int(h2h["frozen_beats_magnitude_aurc"].sum())
    total = len(h2h)
    e2_pos = int(e2["incremental_positive"].sum())
    tahoe_top10 = tahoe[tahoe["top_fraction"].eq(0.10)]
    safeconf_top10 = float(tahoe_top10.loc[tahoe_top10["score_name"].eq("safeconf_full"), "enrichment"].iloc[0])
    mag_top10 = float(tahoe_top10.loc[tahoe_top10["score_name"].eq("predicted_magnitude"), "enrichment"].iloc[0])
    combined75_top10 = float(
        tahoe_top10.loc[tahoe_top10["score_name"].eq("combined_magnitude75"), "enrichment"].iloc[0]
    )

    report = f"""# E9 强基线统一审计报告

生成时间：{now}

## 核心结论

E9 的结果给论文定位划了边界：Frozen v0.2 不能被包装成稳定超过 magnitude 的方法。七主数据集中，Frozen v0.2 直接按 aligned rho 对比 magnitude 只赢 {aligned_wins}/{total}，按 AURC reduction 只赢 {aurc_wins}/{total}。

这并不否定 SafeConf。E2 显示 {e2_pos}/{total} 个数据集在控制 magnitude 后仍有正的 residual signal。更稳的论文主张是：SafeConf 提供 magnitude 之外的风险信息，并可用于 risk triage / selective verification。

Tahoe chemical 是必须保留的边界：top-10 enrichment 中，magnitude = {mag_top10:.2f}，SafeConf full = {safeconf_top10:.2f}，combined 75% magnitude = {combined75_top10:.2f}。这说明 chemical 场景中 magnitude 是主导强基线，SafeConf 更适合作为补充信号或失败边界解释。

## 对投稿叙事的影响

1. 摘要中不要写“SafeConf outperforms magnitude across datasets”。
2. 可以写“SafeConf adds residual risk information beyond magnitude in seven benchmark datasets”。
3. 主图中要同时放 magnitude、SafeConf、combined 和 oracle/reference。
4. Tahoe chemical 放入 Results 的 boundary subsection，语气诚实；这会比藏结果更抗审稿。
5. 下一轮实验要围绕 selective prediction / risk budget 展开，让任务从“谁相关更高”转到“有限复核预算下谁更有用”。

## 自动生成图

{chr(10).join(f'- `{p.relative_to(OUT).as_posix()}`' for p in figs)}

## 自动生成表

- `tables/E9_DEPLOYABLE_BASELINE_LADDER.csv`
- `tables/E9_FROZEN_VS_MAGNITUDE_HEAD_TO_HEAD.csv`
- `tables/E9_DEPLOYABLE_WINNER_TABLE.csv`
- `tables/E9_E2_INCREMENTAL_VALUE.csv`
- `tables/E9_TAHOE_CHEMICAL_BOUNDARY.csv`
"""
    (REPORTS / "E9_STRONG_BASELINE_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E9 强基线统一审计</title>
<style>
body{{margin:0;background:#f7f8f6;color:#17202a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Noto Sans SC",sans-serif;line-height:1.7}}
.top{{background:#12312b;color:white;padding:28px 42px}}.wrap{{max-width:1120px;margin:0 auto;padding:28px}}
.card{{background:white;border:1px solid #d8e0dc;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 22px rgba(15,23,42,.06);overflow-x:auto}}
h2{{border-bottom:3px solid #0f766e;padding-bottom:8px}}.warn{{border-left:6px solid #b45309;background:#fff8eb}}.ok{{border-left:6px solid #047857;background:#f1fbf6}}
img{{max-width:100%;background:white;border:1px solid #d8e0dc;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dbe3df;padding:7px;vertical-align:top}}th{{background:#eef6f3}}
</style></head><body>
<div class="top"><h1>E9 强基线统一审计</h1><p>审稿人最可能追问的问题：SafeConf 是否只是 magnitude 的影子？</p></div>
<div class="wrap">
<div class="card warn"><h2>结论</h2><p>Frozen v0.2 直接对比 magnitude 只赢 {aligned_wins}/{total} 个数据集；但 E2 显示 {e2_pos}/{total} 个数据集在控制 magnitude 后仍有正的 residual signal。论文主张应转向“增量风险信息”和“复核决策价值”。</p></div>
<div class="card"><h2>图 1：直接对比 magnitude</h2><img src="../figures/E9_fig1_frozen_vs_magnitude_aligned_delta.svg"></div>
<div class="card"><h2>图 2：控制 magnitude 后的增量价值</h2><img src="../figures/E9_fig2_incremental_value_after_magnitude_control.svg"></div>
<div class="card"><h2>图 3：Tahoe chemical 边界</h2><img src="../figures/E9_fig3_tahoe_top10_boundary.svg"></div>
<div class="card"><h2>Head-to-head 表</h2>{h2h.to_html(index=False, escape=True)}</div>
</div></body></html>
"""
    (REPORTS / "E9_STRONG_BASELINE_AUDIT.html").write_text(html, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    style()

    baseline = pd.read_csv(BASELINE)
    ladder = build_deployable_ladder(baseline)
    h2h = build_head_to_head(baseline)
    winners = build_winner_table(ladder)
    e2 = build_e2_incremental()
    tahoe = build_tahoe_audit()

    save(ladder, TABLES / "E9_DEPLOYABLE_BASELINE_LADDER.csv")
    save(h2h, TABLES / "E9_FROZEN_VS_MAGNITUDE_HEAD_TO_HEAD.csv")
    save(winners, TABLES / "E9_DEPLOYABLE_WINNER_TABLE.csv")
    save(e2, TABLES / "E9_E2_INCREMENTAL_VALUE.csv")
    save(tahoe, TABLES / "E9_TAHOE_CHEMICAL_BOUNDARY.csv")

    figs = [plot_head_to_head(h2h), plot_e2(e2), plot_tahoe(tahoe)]
    write_report(ladder, h2h, winners, e2, tahoe, figs)

    readme = """# E9 strong baseline audit

入口：

- `reports/E9_STRONG_BASELINE_AUDIT.html`
- `reports/E9_STRONG_BASELINE_AUDIT_REPORT.md`

运行命令：

```bash
python3 tools/scripts/run_e9_strong_baseline_audit.py
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
            "baseline_ladder": str(BASELINE.relative_to(ROOT)),
            "e2_magnitude_residual": str(E2.relative_to(ROOT)),
            "tahoe_d5": str(TAHOE.relative_to(ROOT)),
        },
    }
    (OUT / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote E9 audit to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
