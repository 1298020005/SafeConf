from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")
LATEST = Path("/home/yyf/codex_cout/SAFE_TRANS_PT_20260514_FINAL_USE_THIS")


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def collect(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in root.rglob(pattern):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def metric_by_setting(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [
        "top20_delta",
        "deg_precision_delta",
        "program_consistency_delta",
        "pearson_delta",
        "spearman_delta",
    ]
    rows = []
    for keys, sub in df.groupby(["phase", "split_type"], dropna=False):
        row = {"source": source, "phase": keys[0], "split_type": keys[1], "n": len(sub)}
        for col in metric_cols:
            if col in sub:
                row[col] = float(sub[col].mean())
        if {"top20_delta", "deg_precision_delta", "program_consistency_delta"}.issubset(sub.columns):
            row["effect_positive_fraction"] = float((sub[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].gt(0).sum(axis=1) >= 2).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(reports: Path, figures: Path, summary: pd.DataFrame, risk: pd.DataFrame, contrast: pd.DataFrame) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    if not summary.empty:
        metrics = ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"]
        heat = summary.copy()
        heat["setting"] = heat["source"] + "\n" + heat["phase"] + " / " + heat["split_type"]
        mat = heat.set_index("setting")[[c for c in metrics if c in heat]]
        if not mat.empty:
            plt.figure(figsize=(11, max(5, 0.55 * len(mat))))
            sns.heatmap(mat, center=0, cmap="vlag", annot=True, fmt=".3f", cbar_kws={"label": "delta"})
            plt.title("Weekend evidence summary: effect metrics")
            plt.tight_layout()
            plt.savefig(figures / "01_weekend_effect_metric_heatmap.png", dpi=220)
            plt.close()
    if not risk.empty and {"coverage", "top20_overlap", "deg_precision_top50", "model"}.issubset(risk.columns):
        keep = risk[risk["model"].astype(str).str.contains("SafeTransPT", na=False)].copy()
        if not keep.empty:
            plot = keep.groupby(["model", "coverage"], dropna=False)[["top20_overlap", "deg_precision_top50", "rmse"]].mean().reset_index()
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, col, title in zip(axes, ["top20_overlap", "deg_precision_top50", "rmse"], ["top20", "DEG precision", "RMSE"]):
                sns.lineplot(data=plot, x="coverage", y=col, hue="model", marker="o", ax=ax)
                ax.set_title(title)
            plt.tight_layout()
            plt.savefig(figures / "02_risk_coverage_curves.png", dpi=220)
            plt.close()
    if not contrast.empty and "unsafe_minus_safe_rmse" in contrast:
        ok = contrast[contrast.get("status", "ok") == "ok"].copy()
        if not ok.empty:
            plt.figure(figsize=(9, 5))
            sns.barplot(data=ok, x="split_type", y="unsafe_minus_safe_rmse", hue="phase")
            plt.axhline(0, color="black", lw=1)
            plt.title("Unsafe tasks should have higher error")
            plt.tight_layout()
            plt.savefig(figures / "03_unsafe_contrast_rmse.png", dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", required=True)
    args = parser.parse_args()
    push = Path(args.push)
    reports = push / "reports"
    figures = push / "figures"
    final = push / "final"
    reports.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)

    existing_main = read_csv(PROJECT / "17_may_resume_full_push/reports/ALL_GPU_DEEPSAFE_VS_V2.csv")
    existing_q1 = read_csv(PROJECT / "20_q1_strong_push/reports/ALL_GPU_DEEPSAFE_VS_V2.csv")
    existing_network = read_csv(PROJECT / "21_network_hdWGCNA_push/reports/ALL_NETWORK_SAFE_VS_V2.csv")
    existing_comm = read_csv(PROJECT / "23_unique_design_push/reports/COMMUNITY_BASELINE_BY_SETTING.csv")
    weekend_safety = collect(push / "results", "SafeTransPT_VS_V2.csv")
    weekend_gpu = collect(push / "results", "GPU_DEEPSAFE_VS_V2_*.csv")
    gears_status = collect(push / "results", "GEARS_FORMAL_BASELINE_STATUS.csv")
    gears_formal_final = collect(push / "results/gears_formal_runs", "GEARS_FORMAL_BASELINE_STATUS.csv")
    risk = collect(push / "results", "RISK_COVERAGE.csv")
    contrast = collect(push / "results", "SAFE_UNSAFE_CONTRAST.csv")

    blocks = [
        metric_by_setting(existing_main, "DeepSafe existing long-run"),
        metric_by_setting(existing_q1, "DeepSafe confirmation"),
        metric_by_setting(existing_network, "Network explanation"),
        metric_by_setting(weekend_safety, "Weekend SafeTrans abstention"),
        metric_by_setting(weekend_gpu, "Weekend official-data GPU"),
    ]
    summary = pd.concat([b for b in blocks if not b.empty], ignore_index=True) if any(not b.empty for b in blocks) else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(reports / "WEEKEND_EVIDENCE_BY_SETTING.csv", index=False)
    if not existing_comm.empty:
        existing_comm.to_csv(reports / "COMMUNITY_BASELINE_BY_SETTING.csv", index=False)
    if not gears_status.empty:
        gears_status.to_csv(reports / "GEARS_FORMAL_BASELINE_STATUS_ALL.csv", index=False)
    if not gears_formal_final.empty:
        gears_formal_final.to_csv(reports / "GEARS_FORMAL_BASELINE_STATUS_FINAL_RUNS.csv", index=False)
    if not risk.empty:
        risk.to_csv(reports / "ALL_RISK_COVERAGE.csv", index=False)
    if not contrast.empty:
        contrast.to_csv(reports / "ALL_SAFE_UNSAFE_CONTRAST.csv", index=False)

    write_figures(reports, figures, summary, risk, contrast)
    now = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    lines = [
        "# Weekend Q1 Evidence Push Status",
        "",
        f"Updated: {now}",
        "",
        "## What is done",
        "",
        f"- Existing DeepSafe long-run rows: {len(existing_main)}",
        f"- Existing strong confirmation rows: {len(existing_q1)}",
        f"- Network explanation rows: {len(existing_network)}",
        f"- Community baseline setting rows: {len(existing_comm)}",
        f"- Weekend SafeTrans abstention delta rows: {len(weekend_safety)}",
        f"- Weekend GPU official-data delta rows: {len(weekend_gpu)}",
        f"- Formal GEARS final-run status rows: {len(gears_formal_final)}",
        f"- Formal GEARS all status rows including debugging attempts: {len(gears_status)}",
        f"- Risk-coverage rows: {len(risk)}",
        f"- Safe/unsafe contrast rows: {len(contrast)}",
        "",
        "## Current interpretation",
        "",
        "The project has a real paper-shaped direction, but the final Q1-level claim still depends on three evidence upgrades:",
        "",
        "1. stronger formal baseline comparison beyond proxy baselines;",
        "2. risk-coverage evidence showing unsafe transport can be detected;",
        "3. external validation across more independent perturbation datasets.",
        "",
        "## Current strongest claim",
        "",
        "DeepSafe/SafeTrans is strongest on held-out perturbation. Leave-context remains the hardest setting and should be framed as an unsafe-transport boundary, not as a solved case.",
        "",
    ]
    if not summary.empty:
        lines += ["## Mean deltas by setting", "", "```", summary.to_string(index=False, max_colwidth=32), "```", ""]
    if not gears_formal_final.empty:
        status_counts = gears_formal_final.groupby(["dataset", "status"], dropna=False).size().reset_index(name="n")
        ok = gears_formal_final[gears_formal_final["status"].eq("ok")].copy()
        metric_cols = ["test_mse", "test_mse_de", "test_pearson", "test_pearson_de", "n_test_perturbations"]
        metric_summary = ok.groupby("dataset")[metric_cols].agg(["mean", "std"]).reset_index() if not ok.empty else pd.DataFrame()
        lines += ["## Formal GEARS final-run status", "", "```", status_counts.to_string(index=False), "```", ""]
        if not metric_summary.empty:
            lines += ["## Formal GEARS final-run metric summary", "", "```", metric_summary.to_string(index=False), "```", ""]
    if not gears_status.empty and len(gears_status) != len(gears_formal_final):
        debug_counts = gears_status.groupby(["dataset", "status"], dropna=False).size().reset_index(name="n")
        lines += ["## Formal GEARS all attempts including debugging history", "", "```", debug_counts.to_string(index=False), "```", ""]
    (reports / "WEEKEND_Q1_PUSH_STATUS.md").write_text("\n".join(lines), encoding="utf-8")

    # Keep the user's flat entry directory current.
    if LATEST.exists():
        shutil.copy2(reports / "WEEKEND_Q1_PUSH_STATUS.md", LATEST / "WEEKEND_Q1_PUSH_STATUS.md")
        for fig in figures.glob("*.png"):
            shutil.copy2(fig, LATEST / fig.name)
    print(reports / "WEEKEND_Q1_PUSH_STATUS.md")


if __name__ == "__main__":
    main()
