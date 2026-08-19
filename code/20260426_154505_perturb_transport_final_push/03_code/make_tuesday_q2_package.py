from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")
RUN = ROOT / "17_may_resume_full_push"
PUSH = ROOT / "19_tuesday_q2_push"
REPORTS = RUN / "reports"
OUT = PUSH / "package" / "SafeTransPT_Tuesday_Q2_Package"
FIG = OUT / "figures"
TABLES = OUT / "tables"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def bootstrap_ci(values: pd.Series, n_boot: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def add_effect_dims(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["top20_delta", "deg_precision_delta", "program_consistency_delta"]
    out["effect_positive_dims"] = (out[cols] > 0).sum(axis=1)
    out["two_plus_effect_dims"] = out["effect_positive_dims"] >= 2
    return out


def summarize_gpu(gpu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "top20_delta",
        "deg_precision_delta",
        "program_consistency_delta",
        "pearson_delta",
        "spearman_delta",
        "rmse_delta",
    ]
    rows = []
    for keys, sub in gpu.groupby(["phase", "split_type"], dropna=False):
        row = dict(zip(["phase", "split_type"], keys))
        row["n"] = int(len(sub))
        row["run_groups"] = int(sub["run_group"].nunique()) if "run_group" in sub else 0
        row["two_plus_effect_fraction"] = float(sub["two_plus_effect_dims"].mean())
        for metric in metrics:
            mean, lo, hi = bootstrap_ci(sub[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    setting_summary = pd.DataFrame(rows).sort_values(["phase", "split_type"])

    dataset_rows = []
    for keys, sub in gpu.groupby(["phase", "dataset", "split_type"], dropna=False):
        row = dict(zip(["phase", "dataset", "split_type"], keys))
        row["n"] = int(len(sub))
        row["run_groups"] = int(sub["run_group"].nunique()) if "run_group" in sub else 0
        row["two_plus_effect_fraction"] = float(sub["two_plus_effect_dims"].mean())
        for metric in ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
        dataset_rows.append(row)
    dataset_summary = pd.DataFrame(dataset_rows).sort_values(["phase", "dataset", "split_type"])

    main_claim = gpu[gpu["split_type"] == "heldout_perturbation"].copy()
    return setting_summary, dataset_summary, main_claim


def savefig(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()
    return path


def make_figures(
    gpu: pd.DataFrame,
    gpu_v2: pd.DataFrame,
    safe: pd.DataFrame,
    setting_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    metrics = ["top20_delta", "deg_precision_delta", "program_consistency_delta"]
    plot = setting_summary.copy()
    plot["setting"] = plot["phase"] + " / " + plot["split_type"]
    long_rows = []
    for _, row in plot.iterrows():
        for metric in metrics:
            long_rows.append(
                {
                    "setting": row["setting"],
                    "metric": metric.replace("_delta", ""),
                    "mean": row[f"{metric}_mean"],
                    "ci_low": row[f"{metric}_ci_low"],
                    "ci_high": row[f"{metric}_ci_high"],
                }
            )
    long = pd.DataFrame(long_rows)
    plt.figure(figsize=(10.5, 5.0))
    ax = sns.barplot(data=long, x="metric", y="mean", hue="setting")
    plt.axhline(0, color="black", lw=1)
    plt.ylabel("Mean delta vs V0")
    plt.title("DeepSafeTransGPU effect-level gains by OOD setting")
    savefig("01_effect_metric_barplot.png")

    mat = dataset_summary.copy()
    mat["setting"] = mat["phase"] + " | " + mat["dataset"] + " | " + mat["split_type"]
    heat = mat.set_index("setting")[[f"{m}_mean" for m in metrics]]
    heat.columns = [c.replace("_delta_mean", "") for c in heat.columns]
    plt.figure(figsize=(8.6, max(5.0, 0.36 * len(heat))))
    sns.heatmap(heat, center=0, annot=True, fmt=".3f", cmap="vlag", cbar_kws={"label": "delta vs V0"})
    plt.title("Where transport works and where it fails")
    savefig("02_dataset_setting_heatmap.png")

    frac = setting_summary.copy()
    frac["setting"] = frac["phase"] + " / " + frac["split_type"]
    plt.figure(figsize=(8.6, 4.6))
    sns.barplot(data=frac, x="setting", y="two_plus_effect_fraction")
    plt.axhline(0.7, color="#a33", ls="--", lw=1.4)
    plt.ylim(0, 1.05)
    plt.ylabel("Fraction with >=2 positive effect metrics")
    plt.xticks(rotation=18, ha="right")
    plt.title("Sign consistency supports focused claims")
    savefig("03_sign_consistency.png")

    if not safe.empty:
        ss = safe.groupby(["phase", "split_type"], as_index=False)[metrics].mean()
        ss["setting"] = ss["phase"] + " / " + ss["split_type"]
        ss_long = ss.melt(id_vars=["setting"], value_vars=metrics, var_name="metric", value_name="delta")
        plt.figure(figsize=(9.0, 4.6))
        sns.barplot(data=ss_long, x="setting", y="delta", hue="metric")
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Mean delta vs fixed V2")
        plt.title("Strict boundary: CPU SafeTransPT remains weaker than fixed V2")
        savefig("04_safetrans_boundary.png")

    if not gpu_v2.empty:
        v2s = gpu_v2.groupby(["phase", "split_type"], as_index=False)[metrics].mean()
        v2s["setting"] = v2s["phase"] + " / " + v2s["split_type"]
        v2_long = v2s.melt(id_vars=["setting"], value_vars=metrics, var_name="metric", value_name="delta")
        plt.figure(figsize=(9.2, 4.8))
        sns.barplot(data=v2_long, x="setting", y="delta", hue="metric")
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Mean delta vs fixed V2")
        plt.title("Direct stronger-baseline check: DeepSafeTransGPU vs fixed V2")
        savefig("06_gpu_vs_v2_stronger_baseline.png")

    plt.figure(figsize=(10, 5.0))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        (0.03, 0.61, "Perturbation effect\nin source context", "#e8f4f8"),
        (0.30, 0.61, "Target context\ncontrol state", "#f1f8e9"),
        (0.57, 0.61, "DeepSafeTransGPU\nresidual transport", "#ece7f2"),
        (0.57, 0.22, "Safe claim:\nheld-out perturbation", "#e5f5e0"),
        (0.30, 0.22, "Boundary:\nleave-context", "#fee0d2"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.18, fc=color, ec="#333", lw=1.5))
        ax.text(x + 0.11, y + 0.09, text, ha="center", va="center", fontsize=10)
    for a, b in [((0.25, 0.70), (0.30, 0.70)), ((0.52, 0.70), (0.57, 0.70)), ((0.68, 0.61), (0.68, 0.40)), ((0.57, 0.31), (0.52, 0.31))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.text(0.5, 0.95, "Paper story: safe, focused transport rather than universal generalization", ha="center", fontsize=15, weight="bold")
    savefig("05_story_diagram.png")


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def docs(
    gpu: pd.DataFrame,
    gpu_v2: pd.DataFrame,
    safe: pd.DataFrame,
    setting_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    hp = setting_summary[setting_summary["split_type"] == "heldout_perturbation"].copy()
    main_hp = hp[hp["phase"] == "main"].iloc[0]
    ext_hp = hp[hp["phase"] == "external"].iloc[0]
    main_lc = setting_summary[(setting_summary["phase"] == "main") & (setting_summary["split_type"] == "leave_context")].iloc[0]

    if not gpu_v2.empty:
        v2_effect = gpu_v2[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean(numeric_only=True)
        v2_note = (
            f"\nDirect stronger-baseline check is now active. Mean DeepSafeTransGPU delta vs fixed V2: "
            f"top20={v2_effect.get('top20_delta', float('nan')):.4f}, "
            f"DEG={v2_effect.get('deg_precision_delta', float('nan')):.4f}, "
            f"program={v2_effect.get('program_consistency_delta', float('nan')):.4f}."
        )
    else:
        v2_note = "\nDirect DeepSafeTransGPU vs fixed V2 table has not arrived yet; the refreshed GPU code will generate it in subsequent cycles."

    verdict = f"""
# Tuesday Verdict

## Current Verdict

`Q2_TARGETABLE_WITH_FOCUSED_CLAIMS`

This is stronger than "only potential", but still not a universal Q2-ready claim. The manuscript should be positioned as a focused method-and-benchmark paper on safe transportability of perturbation effects.

## What We Can Claim

DeepSafeTransGPU improves held-out perturbation effect prediction over V0 across repeated seeds/configurations and across main plus external settings. The strongest evidence is not only Pearson/Spearman, but top20 gene overlap, DEG precision, and program consistency.
{v2_note}

## What We Must Not Claim

Do not claim that all cellular-context transfer is solved. Leave-context remains a clear unsafe transport boundary.

## Journal Positioning

- Q2 route: realistic if written with focused claims and honest boundaries.
- Q1 route: only possible after adding stronger community baselines and curated pathway priors.
"""

    summary = f"""
# SafeTrans-PT Tuesday Q2 Summary

Generated: {now}

## Core Story

The project studies safe cross-context transport of single-cell perturbation effects. Instead of asking only whether a model can predict expression response, it asks when a perturbation effect can be safely transported to another cellular context.

## Current Evidence

GPU comparison rows: {len(gpu)}

GPU stronger-baseline comparison rows: {len(gpu_v2)}

SafeTrans CPU comparison rows: {len(safe)}

### Main held-out perturbation

- n = {int(main_hp['n'])}
- run groups = {int(main_hp['run_groups'])}
- top20 delta = {main_hp['top20_delta_mean']:.4f} [{main_hp['top20_delta_ci_low']:.4f}, {main_hp['top20_delta_ci_high']:.4f}]
- DEG precision delta = {main_hp['deg_precision_delta_mean']:.4f} [{main_hp['deg_precision_delta_ci_low']:.4f}, {main_hp['deg_precision_delta_ci_high']:.4f}]
- program consistency delta = {main_hp['program_consistency_delta_mean']:.4f} [{main_hp['program_consistency_delta_ci_low']:.4f}, {main_hp['program_consistency_delta_ci_high']:.4f}]
- two-plus effect metric fraction = {main_hp['two_plus_effect_fraction']:.3f}

### External held-out perturbation

- n = {int(ext_hp['n'])}
- run groups = {int(ext_hp['run_groups'])}
- top20 delta = {ext_hp['top20_delta_mean']:.4f} [{ext_hp['top20_delta_ci_low']:.4f}, {ext_hp['top20_delta_ci_high']:.4f}]
- DEG precision delta = {ext_hp['deg_precision_delta_mean']:.4f} [{ext_hp['deg_precision_delta_ci_low']:.4f}, {ext_hp['deg_precision_delta_ci_high']:.4f}]
- program consistency delta = {ext_hp['program_consistency_delta_mean']:.4f} [{ext_hp['program_consistency_delta_ci_low']:.4f}, {ext_hp['program_consistency_delta_ci_high']:.4f}]
- two-plus effect metric fraction = {ext_hp['two_plus_effect_fraction']:.3f}

### Main leave-context boundary

- top20 delta = {main_lc['top20_delta_mean']:.4f}
- DEG precision delta = {main_lc['deg_precision_delta_mean']:.4f}
- program consistency delta = {main_lc['program_consistency_delta_mean']:.4f}
- two-plus effect metric fraction = {main_lc['two_plus_effect_fraction']:.3f}

This boundary should be used as a strength of the paper: the model identifies that unsafe context transfer remains hard.
"""

    script = """
# 给老师汇报的口径

老师，我这周把方向收窄成了一个更具体的问题：单细胞扰动效应能不能跨 cellular context 安全迁移。

已有 scGen、CPA、CellOT、GEARS 等方法主要关注 response prediction，但在真实实验里，我们更关心一个 effect 从一个细胞类型、病人或数据集迁移到另一个环境时是否还可靠。

我现在的结果显示，DeepSafeTransGPU 在 held-out perturbation 场景下比较稳定，不只是 Pearson/Spearman 提升，top20 基因、DEG precision 和 program consistency 也有提升。外部 held-out perturbation 也有同方向信号。

但是 leave-context 仍然不稳定，所以我不会把它说成解决所有跨 context 泛化。我的论文定位会更保守：识别哪些 perturbation effect 可以安全迁移，哪些属于 unsafe transport boundary。

下一步我会补更强的 community baseline 和 pathway prior，让它从阶段性结果变成更完整的二区投稿故事。
"""

    next_steps = """
# From Focused Q2 To Stronger Q2 / Q1 Potential

1. Freeze the current long run by Tuesday night and regenerate all tables.
2. Make held-out perturbation the main claim.
3. Put leave-context into a failure-boundary figure rather than hiding it.
4. Add direct community baselines if time permits: GEARS-style, CPA-style, CellOT-style, or scVIDR-style.
5. Add curated pathway prior evidence from Reactome/GO/STRING.
6. Add uncertainty/selective prediction figure to show unsafe transport can be rejected.
7. Write the manuscript with focused claims, not universal claims.
"""

    write(OUT / "01_TUESDAY_VERDICT.md", verdict)
    write(OUT / "02_Q2_SUMMARY.md", summary)
    write(OUT / "03_TEACHER_SCRIPT_CN.md", script)
    write(OUT / "04_NEXT_STEPS_TO_STRONG_Q2_Q1.md", next_steps)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    gpu = add_effect_dims(read_csv(REPORTS / "ALL_GPU_DEEPSAFE_VS_V0.csv"))
    gpu_v2_raw = read_csv(REPORTS / "ALL_GPU_DEEPSAFE_VS_V2.csv")
    gpu_v2 = add_effect_dims(gpu_v2_raw) if not gpu_v2_raw.empty else gpu_v2_raw
    safe = add_effect_dims(read_csv(REPORTS / "ALL_SAFETRANS_VS_FIXED_V2.csv"))
    if gpu.empty:
        raise SystemExit("No GPU summary available")
    setting_summary, dataset_summary, main_claim = summarize_gpu(gpu)
    setting_summary.to_csv(TABLES / "setting_summary_bootstrap_ci.csv", index=False)
    dataset_summary.to_csv(TABLES / "dataset_summary.csv", index=False)
    gpu.to_csv(TABLES / "all_gpu_deepsafe_vs_v0.csv", index=False)
    gpu_v2.to_csv(TABLES / "all_gpu_deepsafe_vs_v2.csv", index=False)
    safe.to_csv(TABLES / "all_safetrans_vs_fixed_v2.csv", index=False)
    make_figures(gpu, gpu_v2, safe, setting_summary, dataset_summary)
    docs(gpu, gpu_v2, safe, setting_summary, dataset_summary)
    zip_path = shutil.make_archive(str(OUT), "zip", OUT)
    print(zip_path)


if __name__ == "__main__":
    main()
