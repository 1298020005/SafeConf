from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")
PUSH = ROOT / "17_may_resume_full_push"
REPORTS = PUSH / "reports"
OUT = ROOT / "18_paper_preprint_pack_20260509"
FIG = OUT / "figures"
TABLES = OUT / "tables"


def configure_style() -> None:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", context="paper")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def metric_snapshot(gpu: pd.DataFrame, safe: pd.DataFrame) -> dict:
    cols = ["top20_delta", "deg_precision_delta", "program_consistency_delta"]
    snap: dict = {
        "gpu_rows": int(len(gpu)),
        "safe_rows": int(len(safe)),
        "gpu_status": "missing" if gpu.empty else "available",
        "safe_status": "missing" if safe.empty else "available",
    }
    if not gpu.empty:
        g = gpu.copy()
        g["effect_positive_dims"] = (g[cols] > 0).sum(axis=1)
        snap.update(
            {
                "gpu_mean_top20_delta": float(g["top20_delta"].mean()),
                "gpu_mean_deg_delta": float(g["deg_precision_delta"].mean()),
                "gpu_mean_program_delta": float(g["program_consistency_delta"].mean()),
                "gpu_mean_pearson_delta": float(g["pearson_delta"].mean()),
                "gpu_mean_spearman_delta": float(g["spearman_delta"].mean()),
                "gpu_rows_with_2plus_effect_dims": int((g["effect_positive_dims"] >= 2).sum()),
                "gpu_rows_with_3_effect_dims": int((g["effect_positive_dims"] == 3).sum()),
                "gpu_2plus_effect_fraction": float((g["effect_positive_dims"] >= 2).mean()),
            }
        )
        ext_hp = g[(g["phase"] == "external") & (g["split_type"] == "heldout_perturbation")]
        main_hp = g[(g["phase"] == "main") & (g["split_type"] == "heldout_perturbation")]
        for prefix, sub in [("external_heldout", ext_hp), ("main_heldout", main_hp)]:
            if not sub.empty:
                snap[f"{prefix}_mean_top20_delta"] = float(sub["top20_delta"].mean())
                snap[f"{prefix}_mean_deg_delta"] = float(sub["deg_precision_delta"].mean())
                snap[f"{prefix}_mean_program_delta"] = float(sub["program_consistency_delta"].mean())
                snap[f"{prefix}_n"] = int(len(sub))
    if not safe.empty:
        s = safe.copy()
        s["effect_positive_dims"] = (s[cols] > 0).sum(axis=1)
        snap.update(
            {
                "safe_mean_top20_delta": float(s["top20_delta"].mean()),
                "safe_mean_deg_delta": float(s["deg_precision_delta"].mean()),
                "safe_mean_program_delta": float(s["program_consistency_delta"].mean()),
                "safe_rows_with_2plus_effect_dims": int((s["effect_positive_dims"] >= 2).sum()),
                "safe_2plus_effect_fraction": float((s["effect_positive_dims"] >= 2).mean()),
            }
        )
    return snap


def savefig(name: str) -> Path:
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=240, bbox_inches="tight")
    plt.close()
    return p


def make_figures(gpu: pd.DataFrame, safe: pd.DataFrame) -> list[Path]:
    figs: list[Path] = []

    plt.figure(figsize=(9.6, 4.8))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        (0.03, 0.60, "Source context\nobserved perturbation effect", "#e8f4f8"),
        (0.28, 0.60, "Target context\ncontrol state only", "#f2f7e9"),
        (0.53, 0.60, "Transportability\nsafe or unsafe?", "#fff2cc"),
        (0.78, 0.60, "Predicted effect\nor conservative abstention", "#fde7e7"),
        (0.28, 0.18, "Program-space residual learner\nDeepSafeTransGPU", "#ece7f2"),
        (0.58, 0.18, "Effect-level evaluation\ntop20, DEG, program consistency", "#e5f5e0"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.19, 0.18, fc=color, ec="#444444", lw=1.4))
        ax.text(x + 0.095, y + 0.09, text, ha="center", va="center", fontsize=9)
    for a, b in [((0.22, 0.69), (0.28, 0.69)), ((0.47, 0.69), (0.53, 0.69)), ((0.72, 0.69), (0.78, 0.69)), ((0.40, 0.60), (0.38, 0.36)), ((0.63, 0.60), (0.68, 0.36))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.5, 0.95, "Safe cross-context perturbation effect transport", ha="center", fontsize=15, weight="bold")
    figs.append(savefig("fig1_conceptual_framework.png"))

    if not gpu.empty:
        metrics = ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta"]
        plot = gpu.groupby(["phase", "split_type"], as_index=False)[metrics].mean()
        long = plot.melt(id_vars=["phase", "split_type"], var_name="metric", value_name="delta")
        long["setting"] = long["phase"] + " / " + long["split_type"]
        plt.figure(figsize=(11, 5.0))
        sns.barplot(data=long, x="metric", y="delta", hue="setting")
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Mean delta vs V0")
        plt.title("DeepSafeTransGPU improves effect-level metrics in held-out perturbation settings")
        figs.append(savefig("fig2_gpu_metric_deltas.png"))

        heat = gpu.groupby(["phase", "dataset", "split_type"], as_index=False)[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean()
        heat["setting"] = heat["phase"] + " | " + heat["dataset"] + " | " + heat["split_type"]
        mat = heat.set_index("setting")[["top20_delta", "deg_precision_delta", "program_consistency_delta"]]
        plt.figure(figsize=(8.5, max(5.0, 0.36 * len(mat))))
        sns.heatmap(mat, center=0, cmap="vlag", annot=True, fmt=".3f", cbar_kws={"label": "delta vs V0"})
        plt.title("Dataset-specific transport gains")
        figs.append(savefig("fig3_dataset_effect_heatmap.png"))

        g = gpu.copy()
        cols = ["top20_delta", "deg_precision_delta", "program_consistency_delta"]
        g["effect_positive_dims"] = (g[cols] > 0).sum(axis=1)
        counts = g.groupby(["phase", "split_type", "effect_positive_dims"]).size().reset_index(name="n")
        counts["setting"] = counts["phase"] + " / " + counts["split_type"]
        plt.figure(figsize=(9.2, 4.8))
        sns.barplot(data=counts, x="setting", y="n", hue="effect_positive_dims")
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Number of evaluated comparisons")
        plt.title("How often gains appear in multiple effect metrics")
        figs.append(savefig("fig4_positive_effect_dimensions.png"))

        ext = gpu[gpu["phase"] == "external"].copy()
        if not ext.empty:
            ext_plot = ext.groupby(["dataset", "split_type"], as_index=False)[["top20_delta", "deg_precision_delta", "program_consistency_delta"]].mean()
            ext_long = ext_plot.melt(id_vars=["dataset", "split_type"], var_name="metric", value_name="delta")
            ext_long["setting"] = ext_long["dataset"] + " / " + ext_long["split_type"]
            plt.figure(figsize=(9.4, 4.8))
            sns.barplot(data=ext_long, x="setting", y="delta", hue="metric")
            plt.axhline(0, color="black", lw=1)
            plt.xticks(rotation=18, ha="right")
            plt.ylabel("Mean delta vs V0")
            plt.title("External validation signal is strongest for held-out perturbations")
            figs.append(savefig("fig5_external_validation_pattern.png"))

    if not safe.empty:
        metrics = ["top20_delta", "deg_precision_delta", "program_consistency_delta"]
        plot = safe.groupby(["phase", "split_type"], as_index=False)[metrics].mean()
        long = plot.melt(id_vars=["phase", "split_type"], var_name="metric", value_name="delta")
        long["setting"] = long["phase"] + " / " + long["split_type"]
        plt.figure(figsize=(9.6, 4.6))
        sns.barplot(data=long, x="setting", y="delta", hue="metric")
        plt.axhline(0, color="black", lw=1)
        plt.xticks(rotation=18, ha="right")
        plt.ylabel("Mean delta vs fixed V2")
        plt.title("SafeTrans-PT ablation boundary: fixed V2 remains a hard baseline")
        figs.append(savefig("fig6_safetrans_vs_fixed_v2_boundary.png"))
    return figs


def copy_tables(gpu: pd.DataFrame, safe: pd.DataFrame, statuses: pd.DataFrame) -> None:
    if not gpu.empty:
        gpu.to_csv(TABLES / "all_gpu_deepsafe_vs_v0.csv", index=False)
        gpu.groupby(["phase", "dataset", "split_type"], as_index=False)[
            ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta", "rmse_delta"]
        ].agg(["mean", "std", "count"]).to_csv(TABLES / "gpu_delta_summary_by_dataset.csv")
    if not safe.empty:
        safe.to_csv(TABLES / "all_safetrans_vs_fixed_v2.csv", index=False)
        safe.groupby(["phase", "dataset", "split_type"], as_index=False)[
            ["top20_delta", "deg_precision_delta", "program_consistency_delta", "pearson_delta", "spearman_delta", "rmse_delta"]
        ].agg(["mean", "std", "count"]).to_csv(TABLES / "safetrans_delta_summary_by_dataset.csv")
    if not statuses.empty:
        statuses.to_csv(TABLES / "all_status_summary.csv", index=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_docs(snap: dict, figs: list[Path]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    manuscript = f"""
# DeepSafeTrans-PT: Safe Cross-Context Transport of Single-Cell Perturbation Effects

Snapshot generated: {now}

## Abstract

Single-cell perturbation atlases provide increasingly rich measurements of transcriptional responses to genetic and chemical interventions, yet most experimental designs remain incomplete across cell types, donors, disease states and perturbation combinations. We study a practical question: when can a perturbation effect observed in one cellular context be safely transported to another unseen context? We introduce DeepSafeTrans-PT, a program-aware residual transport model that combines context features, perturbation features, baseline effect estimates and program-space regularization. Across the current May 2026 resumed experiments, DeepSafeTransGPU shows consistent positive deltas over a strong V0 baseline in held-out perturbation settings, with mean improvements of {snap.get('main_heldout_mean_top20_delta', float('nan')):.4f} top20 overlap, {snap.get('main_heldout_mean_deg_delta', float('nan')):.4f} DEG precision and {snap.get('main_heldout_mean_program_delta', float('nan')):.4f} program consistency in main settings. External held-out perturbation settings also show positive effect-level movement, including {snap.get('external_heldout_mean_top20_delta', float('nan')):.4f} top20 overlap and {snap.get('external_heldout_mean_deg_delta', float('nan')):.4f} DEG precision deltas. The current evidence supports a focused claim: effect transport is most reliable for held-out perturbations with shared or partially aligned context structure, while leave-context transport remains the principal failure boundary.

## Introduction

Perturbation response prediction is becoming a central problem in single-cell biology. Existing methods such as scGen, CPA/chemCPA, CellOT and GEARS have made substantial progress in predicting transcriptional outcomes under unseen or compositional perturbations. Large resources and benchmarks such as scPerturb and scPerturBench further emphasize that model evaluation must move beyond small in-distribution splits.

However, a persistent gap remains: a model may predict a response, but it may not know whether an effect is transportable across cellular contexts. This matters because a perturbation effect learned in one cell type, donor background or study may not preserve direction, magnitude or pathway-level structure in another. We therefore focus on safe transportability rather than response prediction alone.

## Contributions

1. We formulate cross-context perturbation effect transport as a reliability-aware prediction task.
2. We implement hard evaluation splits including held-out perturbation and leave-context settings across multiple local perturbation datasets.
3. We develop DeepSafeTransGPU, a program-aware residual learner that improves effect-level metrics over a strong V0 baseline in repeated held-out perturbation experiments.
4. We report failure boundaries showing that leave-context transport is substantially harder than held-out perturbation transport.
5. We provide a reproducible experimental scaffold with saved code, logs, tables and figures.

## Methods

Each task is represented as a perturbation effect vector: treated mean expression minus matched control mean expression. The V0 baseline estimates conservative effects from available same-perturbation or context-related training tasks. DeepSafeTransGPU builds features from V0 predictions, graph/program prior transport estimates, source effects, control-state summaries, perturbation hash features and context hash features. A residual MLP is trained on GPU to predict normalized effect vectors and blended conservatively with V0.

Evaluation uses Pearson, Spearman and RMSE, but the main biological evidence comes from effect-level metrics: top20 overlap, DEG precision at top 50 and program shift consistency.

## Results

### Overall effect-level signal

The current GPU result table contains {snap.get('gpu_rows', 0)} comparisons. Across all GPU comparisons, mean deltas over V0 are:

- top20 overlap: {snap.get('gpu_mean_top20_delta', float('nan')):.4f}
- DEG precision: {snap.get('gpu_mean_deg_delta', float('nan')):.4f}
- program consistency: {snap.get('gpu_mean_program_delta', float('nan')):.4f}
- Pearson: {snap.get('gpu_mean_pearson_delta', float('nan')):.4f}
- Spearman: {snap.get('gpu_mean_spearman_delta', float('nan')):.4f}

Rows with at least two positive effect-level dimensions: {snap.get('gpu_rows_with_2plus_effect_dims', 0)} / {snap.get('gpu_rows', 0)}.

### Held-out perturbation is the strongest current setting

Main held-out perturbation settings show the clearest gains:

- top20 delta: {snap.get('main_heldout_mean_top20_delta', float('nan')):.4f}
- DEG precision delta: {snap.get('main_heldout_mean_deg_delta', float('nan')):.4f}
- program consistency delta: {snap.get('main_heldout_mean_program_delta', float('nan')):.4f}

External held-out perturbation settings also remain positive for top20 and DEG precision:

- top20 delta: {snap.get('external_heldout_mean_top20_delta', float('nan')):.4f}
- DEG precision delta: {snap.get('external_heldout_mean_deg_delta', float('nan')):.4f}
- program consistency delta: {snap.get('external_heldout_mean_program_delta', float('nan')):.4f}

### Leave-context remains the main failure boundary

The same model is less stable under leave-context evaluation. This is scientifically useful rather than merely negative: it motivates the safe transport framing and suggests that future models need stronger biological priors, explicit context uncertainty and better calibration before making claims about completely unseen cellular environments.

## Discussion

The current evidence is suitable for a focused paper direction, not yet for broad claims that all perturbation effects are transportable. The strongest claim is that a conservative residual transport model can improve effect-level metrics for held-out perturbation generalization, while leave-context transport exposes unsafe regions. For a high-confidence Q2 submission, the next requirement is to strengthen external validation and uncertainty/abstention evidence. For Q1-level ambition, comparison against community baselines such as GEARS/CPA/CellOT-style implementations and a clearer biological pathway explanation will be needed.

## Limitations

- Current DeepSafeTransGPU is compared against V0 in the GPU tables; a direct GPU-scale comparison against all community baselines is not yet complete.
- CPU SafeTrans-PT does not yet consistently beat fixed V2 under the strict two-effect-metric rule.
- External validation is positive for held-out perturbation but weaker for leave-context.
- Current program bank relies on PCA/NMF/HVG-derived programs rather than curated pathway priors.

## Provisional Claim

DeepSafeTrans-PT is currently a Q2-focused manuscript candidate with a credible biological machine-learning story. It is not yet safe to claim full Q1-level readiness without additional external validation, direct baseline comparisons and pathway-grounded interpretation.
"""

    zh = f"""
# 给你看的中文解释：这篇论文现在能怎么讲

生成时间：{now}

## 一句话

现在最有希望的论文主线是：

> 单细胞扰动效应不是都能跨细胞环境迁移；我们提出一个更保守的 DeepSafeTrans-PT 框架，先学习哪些 effect 比较可迁移，再在 hard split 上检验 top20 基因、DEG 和 program-level 方向是否更准。

## 现在结果怎么样

好消息：

- GPU 版 DeepSafeTrans 已经有 {snap.get('gpu_rows', 0)} 行比较结果。
- 总体 top20、DEG precision、program consistency 都是正向。
- main held-out perturbation 是最稳的设置。
- external held-out perturbation 也有正向信号。

需要诚实的地方：

- leave-context 还不稳。
- CPU SafeTrans-PT 还打不过 fixed V2 的严格标准。
- 现在可以写“有前景的二区稿”，不能直接吹成一区稳稿。

## 老师问能不能发，建议这样说

目前已经形成了比较完整的方向和第一轮系统证据。它不是普通 perturbation prediction，而是 safe cross-context transport。现阶段最稳的结果是 held-out perturbation 上 effect-level 指标提升；leave-context 暴露了模型的边界，这反而可以作为 safe transport 的动机。接下来如果能把 external 和 uncertainty/abstention 补强，就有希望冲二区；如果再加入 GEARS/CPA/CellOT 等强基线和真实 pathway prior，才更像一区故事。
"""

    judgement = f"""
# Q1/Q2 Readiness Judgement

## Verdict Snapshot

Current status: `Q2_CANDIDATE_NEEDS_CONFIRMATION`, not yet `Q1_READY`.

## Why It Is Promising

- DeepSafeTransGPU repeatedly improves effect-level metrics over V0.
- Held-out perturbation generalization is consistently positive.
- External held-out perturbation is directionally positive.
- The project has a clear gap: safe transportability across cellular contexts.

## Why It Is Not Yet A Strong Final Submission

- CPU SafeTrans-PT remains weaker than fixed V2 under strict effect-metric criteria.
- Leave-context transport is not stable enough.
- Community baselines are not yet fully implemented at the same scale.
- Curated pathway/graph priors need to be stronger.

## Submission Interpretation

- Conservative Q2 path: feasible after confirmation runs and a stronger external validation section.
- Q1 path: possible only after adding direct GEARS/CPA/CellOT-style baselines, curated pathway priors and stronger unsafe-transport calibration.
"""

    next_steps = """
# Next Experiments To Reach A Strong Q2 / Possible Q1 Story

1. Finish the current tmux run and regenerate this package.
2. Prioritize held-out perturbation as the main positive claim.
3. Treat leave-context as a safety boundary, not as a failed side result.
4. Add direct community baselines where feasible: GEARS, CPA/chemCPA-style, CellOT-style or scVIDR-style.
5. Add curated pathway priors from Reactome/GO/STRING instead of only PCA/NMF/HVG programs.
6. Add an uncertainty/abstention figure proving unsafe transport can be identified.
7. Convert the final tables into a clean main manuscript: main figure 1 concept, figure 2 benchmark, figure 3 external validation, figure 4 failure boundary.
"""

    readme = f"""
# Paper Preprint Pack

Generated from current resumed SafeTrans-PT run.

Main run directory:

`{PUSH}`

Important files:

- `MANUSCRIPT_DRAFT_EN.md`: English manuscript draft.
- `MANUSCRIPT_DRAFT_ZH_EXPLAINED.md`: Chinese explanation for the user.
- `Q1_Q2_READINESS_JUDGEMENT.md`: honest publication-readiness judgement.
- `NEXT_EXPERIMENTS_TO_REACH_Q1_Q2.md`: action list for making the story stronger.
- `figures/`: generated paper figures.
- `tables/`: copied and summarized result tables.

Note: background tmux may still be running. This is a snapshot, not the final frozen package.
"""

    write_text(OUT / "MANUSCRIPT_DRAFT_EN.md", manuscript)
    write_text(OUT / "MANUSCRIPT_DRAFT_ZH_EXPLAINED.md", zh)
    write_text(OUT / "Q1_Q2_READINESS_JUDGEMENT.md", judgement)
    write_text(OUT / "NEXT_EXPERIMENTS_TO_REACH_Q1_Q2.md", next_steps)
    write_text(OUT / "README.md", readme)
    write_text(OUT / "METRIC_SNAPSHOT.json", json.dumps(snap, indent=2, ensure_ascii=False))


def main() -> None:
    configure_style()
    if OUT.exists():
        shutil.rmtree(OUT)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    gpu = read_csv(REPORTS / "ALL_GPU_DEEPSAFE_VS_V0.csv")
    safe = read_csv(REPORTS / "ALL_SAFETRANS_VS_FIXED_V2.csv")
    statuses = read_csv(REPORTS / "ALL_STATUS_SUMMARY.csv")
    snap = metric_snapshot(gpu, safe)
    figs = make_figures(gpu, safe)
    copy_tables(gpu, safe, statuses)
    build_docs(snap, figs)
    zip_path = shutil.make_archive(str(OUT), "zip", OUT)
    print(zip_path)


if __name__ == "__main__":
    main()

