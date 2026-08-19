#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_context_splits import build_effect_tasks, feasible_splits, materialize_split, read_scan_table


ROOT = Path("/home/yyf/codex_cout/20260426_154505_perturb_transport_final_push")
OUT = Path("/home/yyf/codex_cout/20260521_safetrans_mentor_audit")
FIG = OUT / "figures"
ATLAS = Path("/home/yyf/datasets/singlecell_perturbation_atlas")
CODE = ROOT / "03_code"

COMPLETED_SAFETY = ROOT / "46_q1_cpu_push_20260520" / "results"
LATEST_SMOKE = ROOT / "54_policy_calibrated_smoke3_20260520" / "results"
RUN_CPU = ROOT / "51_policy_calibrated_q1_20260520" / "results"
RUN_GPU_MAIN = ROOT / "52_gpu_policy_fix_main_20260520" / "results"
RUN_GPU_EXT = ROOT / "53_gpu_policy_fix_external_20260520" / "results"


def sh(cmd: str, timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"COMMAND_FAILED: {cmd}\n{exc!r}\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def read(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return f"[missing] {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def csv_head(path: Path, n: int = 12) -> str:
    if not path.exists():
        return f"[missing] {path}"
    try:
        return pd.read_csv(path).head(n).to_csv(index=False)
    except Exception as exc:
        return f"[read failed] {path}: {exc!r}"


def load_scan_row(study: str) -> pd.Series | None:
    scan = read_scan_table(ATLAS)
    sub = scan[scan["study_family"].astype(str).eq(study)].copy()
    if sub.empty:
        return None
    sub = sub[sub["local_path"].map(lambda p: Path(str(p)).exists())]
    if sub.empty:
        return None
    return sub.iloc[0]


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if mask.sum() < 3:
        return float("nan")
    return float(pd.Series(x.to_numpy(dtype=float)[mask]).corr(pd.Series(y.to_numpy(dtype=float)[mask]), method=method))


def summarize_safety(results_dir: Path, model: str = "PolicySafeTransPT") -> dict:
    out: dict = {"results_dir": str(results_dir), "model": model, "status": "missing"}
    metrics_path = results_dir / "SAFETY_TASK_METRICS.csv"
    if not metrics_path.exists():
        metrics_path = results_dir / "SAFETY_TASK_METRICS_INCREMENTAL.csv"
    risk_path = results_dir / "RISK_COVERAGE.csv"
    if not risk_path.exists():
        risk_path = results_dir / "RISK_COVERAGE_INCREMENTAL.csv"
    contrast_path = results_dir / "SAFE_UNSAFE_CONTRAST.csv"
    if not contrast_path.exists():
        contrast_path = results_dir / "SAFE_UNSAFE_CONTRAST_INCREMENTAL.csv"
    if not metrics_path.exists():
        out["missing"] = f"{metrics_path.name}"
        return out
    df = pd.read_csv(metrics_path)
    sub = df[df.get("model", "").astype(str).eq(model)].copy() if "model" in df else pd.DataFrame()
    if sub.empty:
        out["missing"] = f"model {model} not found"
        return out
    out.update(
        {
            "status": "ok",
            "n_task_rows": int(len(sub)),
            "full_rmse": float(sub["rmse"].mean()) if "rmse" in sub else float("nan"),
            "has_confidence": bool("confidence" in sub),
            "has_unsafe_flag": bool("unsafe_flag" in sub),
        }
    )
    if "unsafe_flag" in sub:
        safe = sub[sub["unsafe_flag"] == 0]
        unsafe = sub[sub["unsafe_flag"] == 1]
        out["n_safe"] = int(len(safe))
        out["n_unsafe"] = int(len(unsafe))
        out["safe_rmse"] = float(safe["rmse"].mean()) if len(safe) else float("nan")
        out["unsafe_rmse"] = float(unsafe["rmse"].mean()) if len(unsafe) else float("nan")
        out["unsafe_minus_safe_rmse"] = out["unsafe_rmse"] - out["safe_rmse"] if np.isfinite(out["unsafe_rmse"]) and np.isfinite(out["safe_rmse"]) else float("nan")
    if "confidence" in sub:
        sub["predicted_risk"] = 1.0 - sub["confidence"].astype(float)
        out["risk_error_pearson"] = safe_corr(sub["predicted_risk"], sub["rmse"], "pearson")
        out["risk_error_spearman"] = safe_corr(sub["predicted_risk"], sub["rmse"], "spearman")
    if risk_path.exists():
        r = pd.read_csv(risk_path)
        rs = r[r.get("model", "").astype(str).eq(model)].copy() if "model" in r else pd.DataFrame()
        if not rs.empty and {"coverage", "rmse"}.issubset(rs.columns):
            full = rs[rs["coverage"] >= 0.99]
            cov80 = rs[rs["coverage"].between(0.75, 0.85)]
            out["risk_full_rmse"] = float(full["rmse"].mean()) if len(full) else float("nan")
            out["risk_80cov_rmse"] = float(cov80["rmse"].mean()) if len(cov80) else float("nan")
            if np.isfinite(out["risk_full_rmse"]) and np.isfinite(out["risk_80cov_rmse"]):
                out["rmse_gain_at_80cov"] = (out["risk_full_rmse"] - out["risk_80cov_rmse"]) / max(out["risk_full_rmse"], 1e-8)
    if contrast_path.exists():
        c = pd.read_csv(contrast_path)
        cs = c[(c.get("model", "").astype(str) == model) & (c.get("status", "").astype(str) == "ok")] if "model" in c else pd.DataFrame()
        if not cs.empty and "unsafe_minus_safe_rmse" in cs:
            out["contrast_ok_frac"] = float((cs["unsafe_minus_safe_rmse"] > 0).mean())
    return out


def make_split_summary() -> pd.DataFrame:
    rows: list[dict] = []
    source_csv = RUN_CPU / "SAFETY_MAIN_SELECTED.csv"
    if source_csv.exists():
        selected = pd.read_csv(source_csv)
    else:
        selected = pd.DataFrame()
    if selected.empty:
        scan = read_scan_table(ATLAS)
        selected = scan[scan["study_family"].isin(["KaggleCrossCell", "Haber", "Parekh"])].copy()
    # Keep this audit bounded; no training is done, but some h5ad files are large.
    for _, ds in selected.head(6).iterrows():
        dataset = str(ds["study_family"])
        path = Path(str(ds["local_path"]))
        try:
            tasks, genes, meta = build_effect_tasks(path, dataset, n_genes=300, seed=20260521)
            split_df = pd.DataFrame(feasible_splits(tasks))
            if split_df.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "split_type": "NO_FEASIBLE_SPLIT",
                        "heldout": "",
                        "train_task_n": 0,
                        "validation_task_n": 0,
                        "test_task_n": 0,
                        "train_context_n": 0,
                        "test_context_n": 0,
                        "train_perturbation_n": 0,
                        "test_perturbation_n": 0,
                        "train_test_context_overlap_n": 0,
                        "train_test_perturbation_overlap_n": 0,
                        "same_context_perturbation_pair_overlap_n": 0,
                        "test_seen_perturbation_n": 0,
                        "test_seen_context_n": 0,
                        "test_unseen_pair_n": 0,
                        "supports_cross_context_same_perturbation": False,
                        "supports_unseen_perturbation_in_seen_context": False,
                        "context_col": meta.get("context_col"),
                        "perturbation_col": meta.get("perturbation_col"),
                        "n_total_tasks": meta.get("n_tasks"),
                        "note": "No feasible leave_context or heldout_perturbation split under current min_test_tasks/shared-overlap rules.",
                    }
                )
            for _, sp in split_df.iterrows():
                train_idx, test_idx = materialize_split(tasks, str(sp["split_type"]), str(sp["heldout"]))
                train_ctx = {tasks[int(i)]["context"] for i in train_idx}
                test_ctx = {tasks[int(i)]["context"] for i in test_idx}
                train_pert = {tasks[int(i)]["perturbation"] for i in train_idx}
                test_pert = {tasks[int(i)]["perturbation"] for i in test_idx}
                train_pairs = {(tasks[int(i)]["context"], tasks[int(i)]["perturbation"]) for i in train_idx}
                test_pairs = {(tasks[int(i)]["context"], tasks[int(i)]["perturbation"]) for i in test_idx}
                rows.append(
                    {
                        "dataset": dataset,
                        "split_type": sp["split_type"],
                        "heldout": sp["heldout"],
                        "train_task_n": int(len(train_idx)),
                        "validation_task_n": 0,
                        "test_task_n": int(len(test_idx)),
                        "train_context_n": int(len(train_ctx)),
                        "test_context_n": int(len(test_ctx)),
                        "train_perturbation_n": int(len(train_pert)),
                        "test_perturbation_n": int(len(test_pert)),
                        "train_test_context_overlap_n": int(len(train_ctx & test_ctx)),
                        "train_test_perturbation_overlap_n": int(len(train_pert & test_pert)),
                        "same_context_perturbation_pair_overlap_n": int(len(train_pairs & test_pairs)),
                        "test_seen_perturbation_n": int(len(test_pert & train_pert)),
                        "test_seen_context_n": int(len(test_ctx & train_ctx)),
                        "test_unseen_pair_n": int(len(test_pairs - train_pairs)),
                        "supports_cross_context_same_perturbation": bool(sp["split_type"] == "leave_context" and len(test_pert & train_pert) > 0),
                        "supports_unseen_perturbation_in_seen_context": bool(sp["split_type"] == "heldout_perturbation" and len(test_ctx & train_ctx) > 0),
                        "context_col": meta.get("context_col"),
                        "perturbation_col": meta.get("perturbation_col"),
                        "n_total_tasks": meta.get("n_tasks"),
                    }
                )
        except Exception as exc:
            rows.append({"dataset": dataset, "split_type": "ERROR", "heldout": "", "error": repr(exc)})
    return pd.DataFrame(rows)


def load_matrix_tasks(dataset: str = "KaggleCrossCell") -> tuple[list[dict], dict]:
    row = load_scan_row(dataset)
    if row is None:
        return [], {"error": f"{dataset} not found"}
    tasks, genes, meta = build_effect_tasks(Path(str(row["local_path"])), dataset, n_genes=300, seed=20260521)
    return tasks, meta


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0b4c8c")
    ax.tick_params(colors="#333333", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0b4c8c")
        spine.set_linewidth(0.8)


def fig_problem_cartoon() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.axis("off")
    ax.text(0.5, 6.25, "Cross-context perturbation effect transport", fontsize=20, fontweight="bold", color="#0b4c8c")
    boxes = [
        (0.8, 3.4, 3.0, 1.7, "Source context A\nperturbation X observed\ntrue effect measured"),
        (8.2, 3.4, 3.0, 1.7, "Target context B\nperturbation X unobserved\neffect needs prediction"),
        (4.55, 1.05, 2.9, 1.35, "SafeTrans-PT\nsafe or unsafe?"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#eef6ff", edgecolor="#0b4c8c", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12, color="#123")
    ax.annotate("", xy=(8.2, 4.25), xytext=(3.8, 4.25), arrowprops=dict(arrowstyle="->", lw=2, color="#0b4c8c"))
    ax.text(5.0, 4.55, "transport effect vector", fontsize=11, color="#0b4c8c")
    ax.annotate("", xy=(5.95, 2.4), xytext=(5.95, 3.4), arrowprops=dict(arrowstyle="->", lw=2, color="#0b4c8c"))
    ax.text(0.5, 0.35, "Question: not only \"what is the prediction?\" but \"should we trust this transport?\"", fontsize=13, color="#444")
    fig.savefig(FIG / "problem_cartoon.png", bbox_inches="tight")
    plt.close(fig)


def fig_context_perturbation_matrix(tasks: list[dict]) -> bool:
    if not tasks:
        return False
    df = pd.DataFrame([{"context": t["context"], "perturbation": t["perturbation"]} for t in tasks])
    ctx = df["context"].value_counts().head(20).index.tolist()
    pert = df["perturbation"].value_counts().head(30).index.tolist()
    mat = pd.crosstab(df["context"], df["perturbation"]).reindex(index=ctx, columns=pert).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=160)
    ax.imshow(mat.to_numpy() > 0, aspect="auto", cmap=plt.cm.Blues, interpolation="nearest")
    ax.set_yticks(range(len(ctx)))
    ax.set_yticklabels(ctx)
    ax.set_xticks(range(len(pert)))
    ax.set_xticklabels(pert, rotation=60, ha="right")
    ax.set_xlabel("perturbation")
    ax.set_ylabel("context")
    style_axes(ax, "Observed context x perturbation task matrix")
    fig.tight_layout()
    fig.savefig(FIG / "context_perturbation_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=160)
    ax.axis("off")
    steps = [
        "raw h5ad",
        "build context-perturbation task",
        "effect = perturbed - control",
        "V0 / ContextSim / SafeTrans",
        "risk features",
        "risk score",
        "safe / unsafe",
        "evaluation",
    ]
    x0, y = 0.4, 3.0
    for i, s in enumerate(steps):
        x = x0 + i * 1.65
        ax.add_patch(plt.Rectangle((x, y), 1.35, 1.0, facecolor="#f7fbff", edgecolor="#0b4c8c", linewidth=1.6))
        ax.text(x + 0.675, y + 0.5, s, ha="center", va="center", fontsize=9.5, wrap=True)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.55, y + 0.5), xytext=(x + 1.35, y + 0.5), arrowprops=dict(arrowstyle="->", color="#0b4c8c", lw=1.5))
    ax.text(0.4, 5.0, "Current minimum pipeline in code", fontsize=18, fontweight="bold", color="#0b4c8c")
    ax.text(0.4, 1.1, "Main check: safe predictions should have lower true RMSE than unsafe predictions.", fontsize=12, color="#444")
    fig.savefig(FIG / "pipeline.png")
    plt.close(fig)


def get_safety_plot_df() -> tuple[pd.DataFrame, str]:
    for label, res in [
        ("current_policy_calibrated_run_51", RUN_CPU),
        ("latest_smoke_after_fix", LATEST_SMOKE),
        ("completed_strict_before_fix", COMPLETED_SAFETY),
    ]:
        path = res / "SAFETY_TASK_METRICS.csv"
        if not path.exists():
            path = res / "SAFETY_TASK_METRICS_INCREMENTAL.csv"
        if path.exists():
            df = pd.read_csv(path)
            sub = df[df.get("model", "").astype(str).eq("PolicySafeTransPT")].copy()
            if {"rmse", "confidence", "unsafe_flag"}.issubset(sub.columns) and len(sub) >= 5:
                sub["predicted_risk"] = 1.0 - sub["confidence"].astype(float)
                return sub, label
    return pd.DataFrame(), ""


def fig_risk_coverage(df: pd.DataFrame, label: str) -> bool:
    if df.empty or not {"predicted_risk", "rmse"}.issubset(df.columns):
        return False
    rows = []
    order = df.sort_values("predicted_risk", ascending=True)
    for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = max(1, int(math.ceil(len(order) * cov)))
        rows.append({"coverage": k / len(order), "rmse": order.head(k)["rmse"].mean()})
    cur = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(cur["coverage"], cur["rmse"], marker="o", color="#0b4c8c")
    ax.set_xlabel("coverage")
    ax.set_ylabel("mean true RMSE")
    style_axes(ax, f"Risk-coverage ({label})")
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig_safe_unsafe(df: pd.DataFrame, label: str) -> bool:
    if df.empty or not {"unsafe_flag", "rmse"}.issubset(df.columns):
        return False
    grp = df.groupby("unsafe_flag")["rmse"].mean()
    if grp.empty:
        return False
    names = ["safe" if int(k) == 0 else "unsafe" for k in grp.index]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.bar(names, grp.values, color=["#4c9be8" if n == "safe" else "#d95f02" for n in names])
    ax.set_ylabel("mean true RMSE")
    style_axes(ax, f"Safe vs unsafe RMSE ({label})")
    fig.tight_layout()
    fig.savefig(FIG / "safe_unsafe_rmse.png", bbox_inches="tight")
    plt.close(fig)
    return True


def fig_risk_error(df: pd.DataFrame, label: str) -> bool:
    if df.empty or not {"predicted_risk", "rmse"}.issubset(df.columns):
        return False
    pear = safe_corr(df["predicted_risk"], df["rmse"], "pearson")
    spear = safe_corr(df["predicted_risk"], df["rmse"], "spearman")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(df["predicted_risk"], df["rmse"], s=16, alpha=0.55, color="#0b4c8c")
    ax.set_xlabel("predicted risk = 1 - confidence")
    ax.set_ylabel("true task RMSE")
    style_axes(ax, f"Risk vs true error ({label})")
    ax.text(0.02, 0.98, f"Pearson={pear:.3f}\nSpearman={spear:.3f}", transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="#ccc"))
    fig.tight_layout()
    fig.savefig(FIG / "risk_error_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    split_summary = make_split_summary()
    split_summary.to_csv(OUT / "split_summary.csv", index=False)

    tasks, matrix_meta = load_matrix_tasks("KaggleCrossCell")
    safety_old = summarize_safety(COMPLETED_SAFETY)
    safety_smoke = summarize_safety(LATEST_SMOKE)
    safety_current = summarize_safety(RUN_CPU)

    figure_missing: list[str] = []
    fig_problem_cartoon()
    if not fig_context_perturbation_matrix(tasks):
        figure_missing.append("context_perturbation_matrix.png: no task matrix could be built.")
    fig_pipeline()
    safety_df, safety_label = get_safety_plot_df()
    if not fig_risk_coverage(safety_df, safety_label):
        figure_missing.append("risk_coverage.png: missing confidence/rmse columns.")
    if not fig_safe_unsafe(safety_df, safety_label):
        figure_missing.append("safe_unsafe_rmse.png: missing unsafe_flag/rmse columns.")
    if not fig_risk_error(safety_df, safety_label):
        figure_missing.append("risk_error_scatter.png: missing predicted risk/true error data.")
    write(OUT / "figure_missing.md", "\n".join(f"- {x}" for x in figure_missing) if figure_missing else "All requested figures were generated.")

    tmux = sh("tmux ls || true")
    gpu = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true")

    write(OUT / "repo_map.md", f"""
    # repo_map.md

    ## 1. 当前项目主入口脚本

    项目根目录：`{ROOT}`

    主要入口在 `{CODE}`：

    - `run_safety_abstention_evidence.py`
      - `main()`：第 216 行左右，CPU 版 safety/abstention 主实验入口。
      - `run_dataset()`：第 120 行左右，逐数据集、seed、split 训练并评估。
    - `run_deep_gpu_transport.py`
      - GPU 深度残差模型入口；当前后台 `52_gpu_policy_fix_main_20260520` 和 `53_gpu_policy_fix_external_20260520` 正在跑它。
    - `evaluate_q1_readiness.py`
      - `evaluate()`：第 114 行左右，读取结果表并给 Q1/Q2-top readiness。
    - `build_context_splits.py`
      - `build_effect_tasks()`：第 76 行左右，构造 context-perturbation effect task。
      - `feasible_splits()`：第 133 行左右，枚举 leave_context / heldout_perturbation。
      - `materialize_split()`：第 152 行左右，把 split 变成 train/test index。
    - 当前新加的运行脚本：
      - `run_policy_calibrated_push_20260520.sh`
      - `run_gpu_main_policy_fix_20260520.sh`
      - `run_gpu_external_policy_fix_20260520.sh`

    ## 2. 数据从哪里读入

    - 数据 atlas：`{ATLAS}`
    - 扫描表：`{ATLAS}/metadata/h5ad_scan.tsv`
    - h5ad 路径来自扫描表里的 `local_path` 列。
    - `build_context_splits.py:17` 的 `read_scan_table(default_root)` 读取 `metadata/h5ad_scan.tsv`。
    - `build_context_splits.py:78` 的 `sc.read_h5ad(path)` 读取实际 h5ad。

    ## 3. context / perturbation / control / treated 在代码里的名字

    - context 候选列：`CONTEXT_CANDIDATES`，`build_context_splits.py:13`
      - `cell_type`, `cell_label`, `condition1`, `cell_line`, `patient`, `donor_id`, `batch`
    - perturbation 候选列：`PERT_CANDIDATES`，`build_context_splits.py:14`
      - `perturbation`, `gene`, `condition2`, `sgRNA`, `GenePair`
    - control 判断：`CONTROL_STRINGS` 和 `is_control_value()`，`build_context_splits.py:12,54`
    - 代码内部统一列：
      - `obs["_context"]`
      - `obs["_pert"]`
    - control 表达均值变量：
      - `control_means[ctx]`
    - treated/perturbed 组：
      - groupby 后 `(ctx, pert)` 对应的 `x[pos].mean(axis=0)`。

    ## 4. 每个 task 怎么构造

    在 `build_effect_tasks()`：

    1. 选 context_col / pert_col：第 79-80 行。
    2. 选 gene_idx：第 83 行。
    3. 读取表达矩阵：优先 `adata.layers["logNor"]`，否则 `adata.X`，第 85 行。
    4. 按 `["_context", "_pert"]` 分组：第 93 行。
    5. control 组按 context 存入 `control_means[ctx]`：第 101-103 行。
    6. treated 组 effect：
       - `effect = x[pos].mean(axis=0) - control_means[ctx]`，第 107 行。
    7. task 字典包括：
       - `dataset`
       - `context`
       - `perturbation`
       - `effect`
       - `control_mean`
       - `n_cells`
       - `context_col`
       - `perturbation_col`

    ## 5. 预测目标到底是什么

    预测目标是 `effect` 向量，不是直接预测 treated expression。

    代码里 `y = np.stack([tasks[int(i)]["effect"] for i in test_idx])`：

    - `run_safety_abstention_evidence.py:138`
    - GPU 脚本里也围绕 effect/residual 训练。

    ## 6. 当前模型和 baseline

    在 `transport_models.py` 和 `safetrans_models.py`：

    - `V0StrongBaseline`：`transport_models.py:37`
    - `V1ProgramTransport`：`transport_models.py:65`
    - `V2GraphPriorTransport`：`transport_models.py:120`
    - `ContextSimilarityBaseline`：`transport_models.py:175`
    - `SafeTransPT`：`safetrans_models.py:39`
    - `NetworkSafeTransPT`：`safetrans_models.py:422`
    - `PolicySafeTransPT`：`safetrans_models.py:584`
    - GPU:
      - `DeepSafeTransport`
      - `DeepCalibratedSafeTransport`
      - `TopRankGraftV2`
      - `EffectBlendV2`

    ## 7. 当前实验输出文件

    已完成/可读结果：

    - `{RUN_CPU}/SAFETY_SUMMARY.csv`
    - `{RUN_CPU}/SAFETY_TASK_METRICS.csv`
    - `{RUN_CPU}/RISK_COVERAGE.csv`
    - `{RUN_CPU}/SAFE_UNSAFE_CONTRAST.csv`
    - `{RUN_CPU}/Q1_READINESS_REPORT.json`
    - `{RUN_GPU_MAIN}/GPU_DEEP_SUMMARY.csv`
    - `{RUN_GPU_EXT}/GPU_DEEP_SUMMARY.csv`
    - older reference: `{COMPLETED_SAFETY}/Q1_READINESS_REPORT.json`

    最新 GPU 结果：

    - `{RUN_GPU_MAIN}/GPU_DEEP_TASK_METRICS.csv`
    - `{RUN_GPU_EXT}/GPU_DEEP_TASK_METRICS.csv`

    后台状态：

    ```text
    {tmux.strip()}
    ```

    GPU 状态：

    ```text
    {gpu.strip()}
    ```

    ## 8. 是否有 safety / risk / unsafe / risk-coverage

    有。

    - `safetrans_models.py`：
      - `transportability_score`
      - `unsafe_flag`
      - `PolicySafeTransPT._calibrated_confidence()`：第 1011 行左右。
      - `predicted_selected_rmse` / `predicted_v0_rmse`：第 1189-1190 行左右。
    - `risk_coverage.py`：
      - `risk_coverage_curve()`：第 23 行左右。
    - `run_safety_abstention_evidence.py`：
      - `_risk_rows()`：第 93 行左右。
      - `_unsafe_contrast()`：第 100 行左右。
      - 输出 `RISK_COVERAGE.csv` / `SAFE_UNSAFE_CONTRAST.csv`：第 270-272 行。
    """)

    write(OUT / "effect_audit.md", f"""
    # effect_audit.md

    ## 1. true_effect 是否等于 perturbed_mean - control_mean？

    是。

    具体代码：

    - 文件：`build_context_splits.py`
    - 函数：`build_effect_tasks()`
    - 行号：第 107 行左右

    ```python
    effect = x[pos].mean(axis=0) - control_means[ctx]
    ```

    ## 2. control_mean 按什么分组算？

    按 context 分组。

    具体逻辑：

    - 先按 `("_context", "_pert")` 分组；
    - 如果 `pert` 被 `is_control_value(pert)` 判定为 control；
    - 则 `control_means[ctx] = x[pos].mean(axis=0)`。

    代码位置：

    - `build_context_splits.py:93`
    - `build_context_splits.py:101-103`

    所以 control_mean 是每个 context 自己的 control，不是全局 control。

    ## 3. perturbed_mean 按什么分组算？

    按 `context + perturbation` 分组。

    代码先做：

    ```python
    for (ctx, pert), sub in obs.groupby(["_context", "_pert"], observed=False):
    ```

    对每个非 control 的 `(ctx, pert)`：

    ```python
    x[pos].mean(axis=0)
    ```

    ## 4. effect 是在 normalization / log transform / gene selection 前还是后？

    effect 是在 gene selection 和读取表达层之后计算的。

    具体顺序：

    1. `choose_gene_indices(adata, n_genes)` 先选 gene index，`build_context_splits.py:83`。
    2. 表达矩阵优先读取 `adata.layers["logNor"]`，否则读 `adata.X`，`build_context_splits.py:85`。
    3. 对这个选好基因、已经是 `logNor` 或 `X` 的矩阵算 mean difference。

    所以：

    - 如果 h5ad 有 `logNor` layer：effect 是 log-normalized expression 上的差值。
    - 如果没有 `logNor`：effect 是 `adata.X` 上的差值。
    - 不确定每个数据集 `X` 是否都已经标准化，需要逐数据集确认。

    ## 5. 一个 task 的伪代码

    ```python
    adata = read_h5ad(path)
    context_col = infer_column(adata.obs, CONTEXT_CANDIDATES)
    pert_col = infer_column(adata.obs, PERT_CANDIDATES)
    gene_idx = choose_gene_indices(adata, n_genes)
    X = adata[:, gene_idx].layers["logNor"] if "logNor" in adata.layers else adata[:, gene_idx].X

    obs["_context"] = obs[context_col]
    obs["_pert"] = obs[pert_col]

    for each context ctx:
        control_cells = cells where _context == ctx and _pert is control
        control_mean[ctx] = mean(X[control_cells])

    for each (ctx, pert):
        if pert is not control and ctx has control_mean:
            perturbed_mean = mean(X[cells where _context == ctx and _pert == pert])
            true_effect = perturbed_mean - control_mean[ctx]
            task = {{
                "context": ctx,
                "perturbation": pert,
                "control_mean": control_mean[ctx],
                "effect": true_effect,
            }}
    ```

    ## 6. 是否预测 treated expression？

    当前主流程不是直接预测 treated expression。

    预测目标是 `effect`，也就是 `perturbed_mean - control_mean`。
    """)

    write(OUT / "v0_audit.md", """
    # v0_audit.md

    ## 1. V0 在哪里实现？

    - 文件：`03_code/transport_models.py`
    - 类：`V0StrongBaseline`
    - 起始行：第 37 行左右

    ## 2. V0 公式是什么？

    训练时：

    ```python
    self.global_mean = mean(train effects)
    self.by_pert[perturbation] = mean(train effects with same perturbation)
    self.by_context[context] = mean(train effects with same context)
    ```

    预测时：

    ```python
    pred = by_pert.get(task["perturbation"], global_mean)
    if task["context"] in by_context:
        pred = 0.85 * pred + 0.15 * by_context[task["context"]]
    ```

    ## 3. 它是不是主要用同一个 perturbation 在其他 context 下的平均 effect？

    是。

    主体是：

    ```python
    self.by_pert.get(task["perturbation"], self.global_mean)
    ```

    如果 train 里见过同一个 perturbation，就用 train 里该 perturbation 的平均 effect。

    ## 4. 它有没有混合同一个 context 的平均 effect？权重是多少？

    有。

    如果 test task 的 context 在 train 中出现过：

    - 85% same-perturbation mean
    - 15% same-context mean

    代码：`transport_models.py:57-58`

    ```python
    pred = 0.85 * pred + 0.15 * self.by_context[task["context"]]
    ```

    ## 5. 预测 test task 时有没有误用 test task 自己的 true effect？

    从代码看，V0 的 `fit()` 只遍历 `train_mask == True` 的 task。

    ```python
    for t, keep in zip(tasks, train_mask):
        if not keep:
            continue
    ```

    所以在当前实现中，V0 不会直接把 test task 自己的 true effect 放进 `by_pert` 或 `by_context`。

    ## 6. held-out context / held-out pair 下有没有泄漏风险？

    直接泄漏：目前没看到。

    但有两个需要谨慎解释的点：

    1. `heldout_perturbation` 下：
       - test perturbation 不在 train，V0 会退回 global_mean，再加 seen context 的 15% context mean；
       - 这不是泄漏，但 context 信息会让 V0 不弱。

    2. `leave_context` 下：
       - test context 不在 train，所以不会加 context mean；
       - 但如果同一个 perturbation 在其他 context 出现，V0 能用 same-perturbation mean。

    当前没有 `held-out context-perturbation pair` 的专门 split；只有 leave_context 和 heldout_perturbation。

    ## 7. 为什么 V0 简单但可能很强？

    因为很多扰动有“平均效应”。

    通俗说：

    > 如果一个 perturbation 在很多细胞背景里大体都会影响一批相似基因，那么直接拿它过去的平均 effect，就已经很难打败。

    这也是当前方法最难的地方：复杂模型必须证明它不只是绕一圈回到 V0，而是真的在某些 context 下比 V0 更懂“什么时候能迁移、迁移哪些基因”。
    """)

    split_head = split_summary.head(20).to_csv(index=False) if not split_summary.empty else "No split summary generated."
    write(OUT / "split_audit.md", f"""
    # split_audit.md

    ## 1. 当前有哪些 split 类型？

    代码里明确实现了两类：

    - `leave_context`
    - `heldout_perturbation`

    位置：

    - `build_context_splits.py:133` 的 `feasible_splits()`
    - `build_context_splits.py:152` 的 `materialize_split()`

    ## 2. 是否有 random split？

    没找到当前主流程中的 random split。

    ## 3. 是否有 held-out context？

    有。对应 `leave_context`。

    `materialize_split()` 里：

    ```python
    test = [i for i, t in enumerate(tasks) if t["context"] == heldout]
    ```

    ## 4. 是否有 held-out perturbation？

    有。对应 `heldout_perturbation`。

    ```python
    test = [i for i, t in enumerate(tasks) if t["perturbation"] == heldout]
    ```

    ## 5. 是否有 held-out context-perturbation pair？

    没找到专门实现。

    当前没有类似 `heldout_pair` / `leave_pair` 的 split 类型。

    ## 6-8. train/test overlap

    我已经生成了 `split_summary.csv`。关键列：

    - `train_test_context_overlap_n`
    - `train_test_perturbation_overlap_n`
    - `same_context_perturbation_pair_overlap_n`
    - `test_seen_perturbation_n`
    - `test_seen_context_n`
    - `test_unseen_pair_n`

    前几行：

    ```csv
    {split_head}
    ```

    ## 9. 当前 split 能不能支持 cross-cell perturbation effect transfer？

    部分支持。

    - `leave_context` 支持“同一个 perturbation 从其他 context 迁移到未见 context”的问题。
    - `heldout_perturbation` 支持“未见 perturbation 在已见 context 中泛化”的问题。
    - 但当前缺少显式 `heldout context-perturbation pair`，所以还没有完全覆盖“某个 context 和 perturbation 都见过，但这个组合没见过”的矩阵补全问题。

    ## 10. 统计结果

    详见：

    - `split_summary.csv`
    """)

    write(OUT / "safety_audit.md", """
    # safety_audit.md

    ## 1. safety score / risk score / unsafe flag 在哪里实现？

    主要在：

    - `03_code/safetrans_models.py`
    - `03_code/risk_coverage.py`
    - `03_code/run_safety_abstention_evidence.py`

    具体：

    - `SafeTransPT._feature_table()`：生成 `transportability_score`, `adaptive_blend`, `unsafe_flag`
    - `PolicySafeTransPT._calibrated_confidence()`：新版本风险校准分
    - `risk_coverage_curve()`：按 confidence 从高到低保留样本，计算 coverage 下 RMSE
    - `_unsafe_contrast()`：比较 safe vs unsafe 的真实 RMSE

    ## 2. safety score 用了哪些特征？

    `SafeTransPT` 的特征包括：

    - `support_score`
    - `context_similarity`
    - `perturbation_consistency`
    - `perturbation_variance_score`
    - `pathway_prior_similarity`
    - `transport_baseline_disagreement`
    - `disagreement_score`

    `PolicySafeTransPT` 新版本还加入：

    - expert utility regression
    - expert RMSE regression
    - expert agreement
    - retrieval confidence
    - router probability

    ## 3. 是手写规则、router confidence，还是训练出来的 risk model？

    现在是混合：

    - `SafeTransPT`：手写 heuristic + 可选 learned gate。
    - `PolicySafeTransPT`：router + out-of-fold utility regressor + out-of-fold RMSE regressor + heuristic calibration。

    ## 4. predicted risk 预测的是什么？

    新版 `PolicySafeTransPT` 中有 `error_reg_`，它拟合每个专家的 task-level RMSE。

    输出列：

    - `predicted_selected_rmse`
    - `predicted_v0_rmse`

    但旧结果表未必都有这两列，因为旧结果是在改代码前生成的。

    ## 5. unsafe threshold 怎么定？

    - `SafeTransPT`：默认 `unsafe_threshold=0.42`，手动阈值。
    - `PolicySafeTransPT`：会用 out-of-fold confidence 分布校准 `confidence_threshold_`，并在 predict 时结合低分位数阈值。

    所以不是纯手动，但也不是严格 conformal guarantee。

    ## 6. 有没有 risk-coverage curve？

    有。

    - 实现：`risk_coverage.py:23`
    - 输出：`RISK_COVERAGE.csv`

    ## 7. 有没有 safe vs unsafe 真实误差对比？

    有。

    - 实现：`run_safety_abstention_evidence.py:_unsafe_contrast()`
    - 输出：`SAFE_UNSAFE_CONTRAST.csv`

    ## 8. 有没有 predicted risk vs true error 相关性？

    没有单独固定输出文件，但可以从 `SAFETY_TASK_METRICS.csv` 的 `confidence` 和 `rmse` 计算：

    - predicted risk = `1 - confidence`
    - true error = `rmse`

    本次已在 `safety_summary.md` 中补算。

    ## 9-10. 结果和缺失

    见：

    - `safety_summary.md`
    - `figure_missing.md`
    """)

    def fmt_safety(s: dict) -> str:
        return "\n".join([f"- `{k}`: {v}" for k, v in s.items()])

    write(OUT / "safety_summary.md", f"""
    # safety_summary.md

    ## 使用的结果源

    1. 当前修复后完整 CPU 结果：`{RUN_CPU}`
    2. 完整旧结果：`{COMPLETED_SAFETY}`
    3. 改代码后的 smoke：`{LATEST_SMOKE}`

    ## 当前修复后完整 CPU 结果，PolicySafeTransPT

    {fmt_safety(safety_current)}

    ## 完整旧结果，PolicySafeTransPT

    {fmt_safety(safety_old)}

    ## 改代码后的 smoke，PolicySafeTransPT

    {fmt_safety(safety_smoke)}

    ## 解释

    - `full_rmse`：所有 task 的平均 RMSE。
    - `safe_rmse`：`unsafe_flag == 0` 的平均 RMSE。
    - `unsafe_rmse`：`unsafe_flag == 1` 的平均 RMSE。
    - `risk_error_pearson/spearman`：`1 - confidence` 与真实 `rmse` 的相关性。
    - `rmse_gain_at_80cov`：只保留约 80% 高 confidence 样本后，RMSE 是否下降。

    ## 当前是否站得住？

    还没有完全站住。

    当前 51 结果已经完整落盘，但 strict readiness 仍需按 `Q1_READINESS_REPORT.json` 判断。
    如果 safe/unsafe RMSE、risk-error correlation、80% coverage RMSE gain 仍不稳定，就不能说 safety 已经站住。

    最关键要补：

    - 新版 `predicted_selected_rmse` vs true `rmse` 的相关性是否已稳定输出
    - safe vs unsafe 的稳定 RMSE 对比
    """)

    write(OUT / "mentor_discussion_outline.md", """
    # mentor_discussion_outline.md

    ## 第 1 页：我为什么看单细胞扰动预测

    左边图：`figures/problem_cartoon.png`

    右边短句：

    - 单细胞扰动数据可以告诉我们“干预一个基因/药物后，细胞状态怎么变”。
    - 但真实实验不可能把所有细胞类型、病人、扰动组合都测完。
    - 所以我想看：能不能把一个 context 中观察到的扰动效应，迁移到另一个 context。
    - 我现在更关心“能不能信”，不是只追求预测数值。

    ## 第 2 页：文献给我的感觉

    左边图：简单列 scGen / CPA / CellOT / GEARS / scGPT / Virtual Cell Challenge

    右边短句：

    - 扰动预测已经有人做，不是空白。
    - 跨 context 泛化也有人在碰。
    - uncertainty / abstention 在机器学习里也很成熟。
    - 我的感觉是：可以把问题收缩到“跨 context 迁移是否安全”。

    ## 第 3 页：我想切的小问题

    左边图：source context A → target context B，中间 safe/unsafe。

    右边短句：

    - 一个 perturbation effect 不一定哪里都能迁移。
    - 有些 perturbation 可能跨 context 比较稳定。
    - 有些可能强依赖细胞状态。
    - 我想让模型先判断这次迁移是否可靠。

    ## 第 4 页：数据结构是否支持这个问题

    左边图：`figures/context_perturbation_matrix.png`

    右边短句：

    - 代码把数据整理成 context × perturbation task。
    - 每个格子对应一个 effect：perturbed mean - control mean。
    - 当前 split 有 leave-context 和 held-out perturbation。
    - 但还缺一个明确的 held-out pair split，后面应该补。

    ## 第 5 页：代码里的最小逻辑

    左边图：`figures/pipeline.png`

    右边短句：

    - 从 h5ad 读表达矩阵。
    - 按 context 和 perturbation 分组。
    - effect = treated mean - control mean。
    - V0 用同扰动平均 effect 做强 baseline。
    - safety 模块输出 risk score 和 unsafe flag。

    ## 第 6 页：现在结果怎么看

    左边图：`figures/risk_coverage.png` 或 `figures/safe_unsafe_rmse.png`

    右边短句：

    - V0 很强，所以复杂模型不能只和弱 baseline 比。
    - 旧结果里 safety 还没完全站住。
    - 我改了代码后，smoke 有一点 risk-coverage 正信号。
    - 但现在不能夸大，完整长跑还在进行。

    ## 第 7 页：我现在担心的问题

    左边图：`figures/risk_error_scatter.png`

    右边短句：

    - 这个角度会不会和已有工作太接近？
    - split 是否真的体现 cross-cell transport？
    - safety score 是否真的能识别高风险预测？
    - V0 太强时，方法创新应该放在哪里？

    ## 第 8 页：想请老师判断什么

    左边图：三条路线：继续方法、转 benchmark、补 biological explanation

    右边短句：

    - 这个“safe transport”角度是否值得继续？
    - 是做方法论文，还是先做 benchmark/protocol？
    - external validation 应该优先补哪些数据？
    - biological explanation 是否需要用 network/module 来增强？
    """)

    write(OUT / "mentor_talk_script.md", """
    # mentor_talk_script.md

    老师，我这段时间主要在看单细胞扰动预测相关的方向。

    我一开始的想法比较大，想直接做一个跨数据集、跨细胞背景的扰动预测模型。但看了一些工作以后，我发现这个方向并不是空白。像 GEARS、CPA、CellOT，还有一些 foundation model 方向，其实都在做扰动响应预测或者泛化。所以我现在不太想硬说自己是在做一个全新的大模型。

    我现在更想把问题收缩一点：不是只问“能不能预测”，而是问“这个扰动效应从一个细胞背景迁移到另一个细胞背景时，到底能不能信”。

    代码里我现在把原始单细胞数据整理成 context × perturbation 的 task。每个 task 的标签不是原始表达量，而是 effect，也就是同一个 context 里 treated mean 减去 control mean。这样就可以把问题变成：已知一些 context 里的 perturbation effect，能不能预测另一个 context 里的 effect。

    我现在也意识到一个很重要的问题，就是 V0 baseline 很强。它其实很简单，就是用同一个 perturbation 在训练集其他 context 下的平均 effect 去预测。如果这个扰动本身比较稳定，这个简单方法就已经很难打败。所以后面不能只和很弱的 baseline 比，必须认真面对 V0。

    另外我现在加了 safety / risk 的逻辑。想法是模型不要每个样本都硬预测，而是给一个 transportability score，也就是这次迁移是否可靠。如果不可靠，就标成 unsafe。这个部分现在还没有完全站住。旧结果里 safety 不是特别稳定，所以我最近在修这里：让 risk score 不只是模型自信，而是结合预测误差、context 相似度、扰动一致性和不同专家模型之间的分歧。

    我现在最想请老师帮我判断的是：这个角度有没有继续做的价值？也就是我们不把它包装成“我发明了一个全新的扰动预测大模型”，而是把它做成一个 cross-context perturbation effect transport 的 reliability / safety 判断组件。后面如果要继续，我觉得最关键的是证明 safe 样本真的比 unsafe 样本更可靠，同时把 split 做得更严谨，比如补 held-out context-perturbation pair 和更独立的 external validation。
    """)

    write(OUT / "SUMMARY_FOR_ME.md", f"""
    # SUMMARY_FOR_ME.md

    ## 我现在能跟老师说什么

    - 我在看单细胞扰动预测，但不准备硬吹“全新大模型”。
    - 我现在把问题收缩成：跨细胞背景迁移 perturbation effect 时，预测到底能不能信。
    - 代码里的 effect 是 `perturbed_mean - control_mean`，按 context + perturbation 构造 task。
    - V0 baseline 很强：主要用同一个 perturbation 在训练集其他 context 的平均 effect。
    - 我已经有 safety / risk / unsafe / risk-coverage 的代码，但结果还没完全站住。

    ## 哪些地方不能说太满

    - 不能说已经解决跨 context perturbation prediction。
    - 不能说稳一区/稳二区。
    - 不能说已经显著超过 GEARS / CPA / CellOT，因为同题比较还不完整。
    - 不能说 safety 已经可靠，因为旧结果里 safe/unsafe 和 risk-coverage 还不够硬。

    ## 老师最可能问什么

    1. 你这个问题和已有 perturbation prediction 有什么区别？
    2. 你的 split 是否真的能证明跨细胞迁移？
    3. V0 这么强，你的方法到底多了什么？
    4. unsafe flag 真的能找出高误差样本吗？
    5. external validation 是否独立？

    ## 下一步最应该补什么

    1. 等当前新代码长跑完成，重新生成 safety_summary。
    2. 补 held-out context-perturbation pair split。
    3. 做 predicted risk vs true RMSE 的稳定相关性分析。
    4. 做 safe vs unsafe 的统计检验。
    5. 补同题 GEARS / CPA / CellOT 或至少诚实说明任务差异。

    ## 当前输出位置

    `{OUT}`
    """)

    print(json.dumps({"out_dir": str(OUT), "figures_dir": str(FIG), "split_rows": int(len(split_summary))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
