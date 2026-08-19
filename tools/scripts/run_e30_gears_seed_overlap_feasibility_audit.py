#!/usr/bin/env python3
"""E30: GEARS seed-overlap feasibility audit.

E26 showed that GEARS-only predicted-effect magnitude can rank some high-error
records.  E29 showed that GEARS–scGPT disagreement is feasible but weak on a
tiny shared Adamson panel.

The next tempting idea is GEARS seed/ensemble uncertainty: if multiple GEARS
runs disagree on the same perturbation task, that disagreement might be a
deployable model-specific risk signal.

E30 checks whether the existing E25 GEARS strict package actually supports that
claim.  It audits task overlap across the three GEARS formal runs, verifies
true-effect consistency for repeated tasks, and computes seed-disagreement
diagnostics only where repeated tasks exist.
"""

from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code/20260426_154505_perturb_transport_final_push"
sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd


E25_DIR = PROJECT_ROOT / "docs/实验结果/E25_gears_strict_prediction_records_20260708"
OUT_DIR = PROJECT_ROOT / "docs/实验结果/E30_gears_seed_overlap_feasibility_audit_20260708"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT)
        return bool(out.decode("utf-8").strip())
    except Exception:
        return True


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() < 2:
        return float("nan")
    xv = xv[mask]
    yv = yv[mask]
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() < 2:
        return float("nan")
    xr = pd.Series(xv[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(yv[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _pairwise_mean_rmse(arrs: list[np.ndarray]) -> float:
    vals = [_rmse(a, b) for a, b in combinations(arrs, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def _seed_ids(sub: pd.DataFrame) -> str:
    return ",".join(str(int(x)) for x in sorted(sub["fold_id"].astype(int).unique()))


def _load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rec = pd.read_csv(E25_DIR / "tables/PREDICTION_RECORDS.csv")
    rec = rec.copy()
    rec["task_gene"] = rec["perturbation"].astype(str).str.replace("+ctrl", "", regex=False)
    rec["task_group"] = rec["dataset_name"].astype(str) + "::" + rec["task_gene"].astype(str)
    with np.load(E25_DIR / "arrays/gears_predicted_effects.npz") as pred_npz:
        pred = {k: np.asarray(pred_npz[k], dtype=np.float32) for k in pred_npz.files}
    with np.load(E25_DIR / "arrays/gears_true_effects.npz") as true_npz:
        true = {k: np.asarray(true_npz[k], dtype=np.float32) for k in true_npz.files}
    return rec, pred, true


def _audit(rec: pd.DataFrame, pred: dict[str, np.ndarray], true: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (dataset, task_gene), sub in rec.groupby(["dataset_name", "task_gene"], sort=True):
        sub = sub.sort_values("fold_id")
        pred_arrs = [pred[str(k)] for k in sub["predicted_effect_key"]]
        true_arrs = [true[str(k)] for k in sub["true_effect_key"]]
        rmses = [_rmse(p, t) for p, t in zip(pred_arrs, true_arrs)]
        true_max_abs_diff = 0.0
        for a, b in combinations(true_arrs, 2):
            true_max_abs_diff = max(true_max_abs_diff, float(np.max(np.abs(a - b))))
        stacked = np.stack(pred_arrs)
        seed_disagreement_rmse = float(np.sqrt(np.mean(np.var(stacked, axis=0)))) if len(pred_arrs) >= 2 else float("nan")
        rows.append(
            {
                "dataset_name": dataset,
                "task_gene": task_gene,
                "task_group": f"{dataset}::{task_gene}",
                "n_records": int(len(sub)),
                "seed_ids": _seed_ids(sub),
                "is_repeat_ge2": bool(len(sub) >= 2),
                "is_repeat_ge3": bool(len(sub) >= 3),
                "gene_panel_id": str(sub["gene_panel_id"].iloc[0]),
                "gene_order_hash": str(sub["gene_order_hash"].iloc[0]),
                "normalization_id": str(sub["normalization_id"].iloc[0]),
                "true_max_abs_diff_across_records": true_max_abs_diff,
                "seed_rmse_mean": float(np.mean(rmses)),
                "seed_rmse_std": float(np.std(rmses, ddof=0)) if len(rmses) >= 2 else 0.0,
                "seed_rmse_min": float(np.min(rmses)),
                "seed_rmse_max": float(np.max(rmses)),
                "seed_pairwise_pred_rmse_mean": _pairwise_mean_rmse(pred_arrs),
                "seed_disagreement_rmse": seed_disagreement_rmse,
                "true_l2_mean": float(np.mean([np.linalg.norm(t) for t in true_arrs])),
                "pred_l2_mean": float(np.mean([np.linalg.norm(p) for p in pred_arrs])),
            }
        )
    coverage = pd.DataFrame(rows)
    dataset_summary = (
        coverage.groupby("dataset_name")
        .agg(
            n_task_groups=("task_group", "count"),
            n_records=("n_records", "sum"),
            singleton_tasks=("is_repeat_ge2", lambda s: int((~s).sum())),
            repeat_ge2_tasks=("is_repeat_ge2", "sum"),
            repeat_ge3_tasks=("is_repeat_ge3", "sum"),
            max_true_diff=("true_max_abs_diff_across_records", "max"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "dataset_name": "overall",
                "n_task_groups": int(coverage.shape[0]),
                "n_records": int(coverage["n_records"].sum()),
                "singleton_tasks": int((~coverage["is_repeat_ge2"]).sum()),
                "repeat_ge2_tasks": int(coverage["is_repeat_ge2"].sum()),
                "repeat_ge3_tasks": int(coverage["is_repeat_ge3"].sum()),
                "max_true_diff": float(coverage["true_max_abs_diff_across_records"].max()),
            }
        ]
    )
    dataset_summary = pd.concat([dataset_summary, overall], ignore_index=True)

    repeated = coverage[coverage["is_repeat_ge2"]].copy()
    summary_rows: list[dict[str, Any]] = []
    for score_col in ["seed_pairwise_pred_rmse_mean", "seed_disagreement_rmse", "pred_l2_mean", "true_l2_mean"]:
        for target_col in ["seed_rmse_mean", "seed_rmse_max"]:
            summary_rows.append(
                {
                    "score": score_col,
                    "target": target_col,
                    "n_repeat_tasks": int(repeated.shape[0]),
                    "spearman": _spearman(repeated[score_col], repeated[target_col]),
                    "pearson": _pearson(repeated[score_col], repeated[target_col]),
                    "score_available_tasks": int(repeated[score_col].notna().sum()),
                }
            )
    risk_summary = pd.DataFrame(summary_rows)

    recommended = coverage.sort_values(["dataset_name", "task_gene"])[
        ["dataset_name", "task_gene", "task_group", "gene_panel_id", "gene_order_hash", "normalization_id"]
    ].copy()
    recommended["recommended_action"] = "rerun_same_task_across_3_seeds_before_claiming_seed_uncertainty"
    return coverage, dataset_summary, repeated, risk_summary, recommended


def _write_report(
    status: dict[str, Any],
    coverage: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    repeated: pd.DataFrame,
    risk_summary: pd.DataFrame,
    recommended: pd.DataFrame,
) -> None:
    md = f"""# E30 GEARS seed-overlap feasibility audit

生成时间：{_now()}

## 先看结论

E30 检查了 E25 strict GEARS 包能否直接支持 seed/ensemble uncertainty。

- E25 records：{status['n_records']}
- unique task groups：{status['n_task_groups']}
- repeat ≥2 tasks：{status['repeat_ge2_tasks']}
- repeat ≥3 tasks：{status['repeat_ge3_tasks']}
- singleton tasks：{status['singleton_tasks']}
- repeated-task true effect max diff：{status['max_true_diff']}

结论：当前 E25 包可以证明 true effect 在重复任务内一致，但不能支撑正式 seed-ensemble uncertainty，因为大多数任务只出现一次。seed disagreement 只能在 {status['repeat_ge2_tasks']} 个重复任务上做 exploratory check。
"""
    (OUT_DIR / "reports/E30_GEARS_SEED_OVERLAP_FEASIBILITY_AUDIT_REPORT.md").write_text(md, encoding="utf-8")

    css = """
body{margin:0;background:#fff;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:42px 28px 76px}h1{font-size:30px;margin:0 0 10px}
h2{font-size:21px;margin:34px 0 12px;border-top:1px solid #e5e7eb;padding-top:24px}p{line-height:1.75;font-size:16px}
.note{border-left:4px solid #315C9B;background:#f8fbff;padding:12px 16px;border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:24px 0}.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;background:#fafafa}
.k{font-size:26px;font-weight:760;color:#111827}.l{font-size:13px;color:#66788a;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 22px}th,td{border-bottom:1px solid #e5e7eb;text-align:left;padding:8px 9px;vertical-align:top}th{background:#f7f7f7}
"""
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E30 GEARS seed-overlap feasibility audit</title><style>{css}</style></head>
<body><main>
<h1>E30 GEARS seed-overlap feasibility audit</h1>
<p class="note">这个报告回答一个很容易被误写的问题：E25 的 3 个 GEARS formal runs 能不能直接当 seed ensemble uncertainty？答案是：目前不能，任务重叠太少。</p>
<div class="cards">
<div class="card"><div class="k">{status['n_records']}</div><div class="l">E25 records</div></div>
<div class="card"><div class="k">{status['n_task_groups']}</div><div class="l">unique tasks</div></div>
<div class="card"><div class="k">{status['repeat_ge2_tasks']}</div><div class="l">repeat ≥2</div></div>
<div class="card"><div class="k">{status['repeat_ge3_tasks']}</div><div class="l">repeat ≥3</div></div>
<div class="card"><div class="k">{status['max_true_diff']:.1g}</div><div class="l">max true diff</div></div>
</div>
<h2>Dataset coverage summary</h2>{dataset_summary.to_html(index=False, escape=False)}
<h2>Repeated-task seed diagnostics</h2>{repeated.to_html(index=False, escape=False)}
<h2>Exploratory risk summary on repeated tasks</h2>{risk_summary.to_html(index=False, escape=False)}
<h2>All task coverage</h2>{coverage.to_html(index=False, escape=False)}
<h2>Recommended rerun manifest</h2><p>如果要正式声称 GEARS seed uncertainty，应按这个任务清单固定任务，并对同一任务重复 3 个 seed。</p>{recommended.head(60).to_html(index=False, escape=False)}
</main></body></html>
"""
    (OUT_DIR / "reports/E30_GEARS_SEED_OVERLAP_FEASIBILITY_AUDIT.html").write_text(page, encoding="utf-8")
    (OUT_DIR / "README_先看这个.md").write_text(
        f"""# E30 GEARS seed-overlap feasibility audit

先看结论：E25 的 GEARS strict 包不能直接当作 seed-ensemble uncertainty benchmark，因为 47 个 unique task groups 中只有 5 个出现至少 2 次，只有 2 个出现 3 次。

可用信息：

- 重复任务内 true effect 最大差异 = {status['max_true_diff']}
- repeated-task exploratory 表：`tables/E30_REPEATED_TASK_SEED_DIAGNOSTICS.csv`
- 全任务覆盖表：`tables/E30_TASK_SEED_COVERAGE.csv`
- 建议重跑清单：`tables/E30_RECOMMENDED_FIXED_SEED_RERUN_MANIFEST.csv`

正确口径：E30 是 feasibility / claim-control audit，不是性能提升结果。
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for rel in ["tables", "reports"]:
        (OUT_DIR / rel).mkdir(parents=True, exist_ok=True)

    rec, pred, true = _load_inputs()
    coverage, dataset_summary, repeated, risk_summary, recommended = _audit(rec, pred, true)
    coverage.to_csv(OUT_DIR / "tables/E30_TASK_SEED_COVERAGE.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "tables/E30_DATASET_SEED_COVERAGE_SUMMARY.csv", index=False)
    repeated.to_csv(OUT_DIR / "tables/E30_REPEATED_TASK_SEED_DIAGNOSTICS.csv", index=False)
    risk_summary.to_csv(OUT_DIR / "tables/E30_REPEATED_TASK_RISK_SUMMARY.csv", index=False)
    recommended.to_csv(OUT_DIR / "tables/E30_RECOMMENDED_FIXED_SEED_RERUN_MANIFEST.csv", index=False)

    overall = dataset_summary[dataset_summary["dataset_name"].eq("overall")].iloc[0]
    status = {
        "status": "ok",
        "generated_at": _now(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "out": os.path.relpath(OUT_DIR, PROJECT_ROOT),
        "source": os.path.relpath(E25_DIR, PROJECT_ROOT),
        "n_records": int(overall["n_records"]),
        "n_task_groups": int(overall["n_task_groups"]),
        "singleton_tasks": int(overall["singleton_tasks"]),
        "repeat_ge2_tasks": int(overall["repeat_ge2_tasks"]),
        "repeat_ge3_tasks": int(overall["repeat_ge3_tasks"]),
        "max_true_diff": float(overall["max_true_diff"]),
        "claim_allowed_now": "E25 repeated tasks have consistent true effects; seed-disagreement can be explored only on sparse repeated tasks.",
        "claim_not_allowed": "Do not claim formal GEARS seed-ensemble uncertainty from E25 because most tasks are singletons.",
    }
    (OUT_DIR / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(status, coverage, dataset_summary, repeated, risk_summary, recommended)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
