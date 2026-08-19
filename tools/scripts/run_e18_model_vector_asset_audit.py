#!/usr/bin/env python3
"""E18 audit: availability of real-model prediction vectors for SafeConf."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E18_model_vector_asset_audit_20260707"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

SAFE_RUNTIME = Path("/home/yyf/safeconf_runtime/outputs")
GEARS_ROOT = SAFE_RUNTIME / "gears_prediction_records_formal"
GEARS_EVAL = SAFE_RUNTIME / "gears_confidence_eval_formal"
GEARS_UNCERT = SAFE_RUNTIME / "gears_uncertainty_formal_v6"
GEARS_DATA_ROOT = Path("/home/yyf/data/gears_formal_baselines_v2")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


def git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    )


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def repair_gears_path(value: str) -> Path:
    """Old status files point to /home/yyf/codex_cout/.../outputs; runtime stores the copy."""
    raw = Path(str(value))
    if raw.exists():
        return raw
    marker = "/outputs/"
    text = str(value)
    if marker in text:
        tail = text.split(marker, 1)[1]
        candidate = SAFE_RUNTIME / tail
        if candidate.exists():
            return candidate
    return raw


def inspect_npz(path: Path, expected_keys: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "n_keys": 0,
        "first_shape": "",
        "first_dtype": "",
        "all_expected_keys_present": False if expected_keys else "",
    }
    if not path.exists():
        return out
    with np.load(path) as z:
        keys = list(z.files)
        out["n_keys"] = len(keys)
        if keys:
            arr = z[keys[0]]
            out["first_shape"] = "x".join(map(str, arr.shape))
            out["first_dtype"] = str(arr.dtype)
        if expected_keys is not None:
            out["all_expected_keys_present"] = set(expected_keys).issubset(set(keys))
    return out


def md_table(df: pd.DataFrame, cols: list[str] | None = None, n: int = 80) -> str:
    show = df if cols is None else df[cols]
    show = show.head(n).copy()
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for c in show.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v).replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def html_table(df: pd.DataFrame, cols: list[str] | None = None, n: int = 80) -> str:
    show = df if cols is None else df[cols]
    return show.head(n).to_html(index=False, escape=True)


def audit_gears() -> dict[str, pd.DataFrame | dict[str, Any]]:
    status_path = GEARS_ROOT / "GEARS_PREDICTION_RECORD_STATUS.csv"
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    rows = []
    all_records = []
    for _, row in status.iterrows():
        records_csv = repair_gears_path(str(row["records_csv"]))
        pred_npz = repair_gears_path(str(row["predicted_npz"]))
        true_npz = repair_gears_path(str(row["true_npz"]))
        rec = pd.read_csv(records_csv) if records_csv.exists() else pd.DataFrame()
        if not rec.empty:
            tmp = rec.copy()
            tmp["source_dataset"] = row["dataset"]
            tmp["source_seed"] = row["seed"]
            all_records.append(tmp)
        pred = inspect_npz(pred_npz, rec["predicted_effect_key"].astype(str).tolist() if not rec.empty else None)
        true = inspect_npz(true_npz, rec["true_effect_key"].astype(str).tolist() if not rec.empty else None)
        rows.append(
            {
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "seed": int(row.get("seed")),
                "status": row.get("status"),
                "reported_n_prediction_records": int(row.get("n_prediction_records", 0)),
                "records_csv_reported_exists": Path(str(row["records_csv"])).exists(),
                "records_csv_repaired": str(records_csv),
                "records_csv_repaired_exists": records_csv.exists(),
                "records_n_rows": len(rec),
                "predicted_npz_reported_exists": Path(str(row["predicted_npz"])).exists(),
                "predicted_npz_repaired_exists": bool(pred["exists"]),
                "predicted_npz_n_keys": int(pred["n_keys"]),
                "predicted_npz_first_shape": pred["first_shape"],
                "predicted_keys_match_records": pred["all_expected_keys_present"],
                "true_npz_reported_exists": Path(str(row["true_npz"])).exists(),
                "true_npz_repaired_exists": bool(true["exists"]),
                "true_npz_n_keys": int(true["n_keys"]),
                "true_npz_first_shape": true["first_shape"],
                "true_keys_match_records": true["all_expected_keys_present"],
                "test_mse": row.get("test_mse"),
                "test_pearson": row.get("test_pearson"),
            }
        )
    run_audit = pd.DataFrame(rows)
    records = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    if not records.empty:
        dataset_summary = (
            records.groupby("dataset_name", dropna=False)
            .agg(
                n_records=("record_id", "count"),
                n_unique_perturbations=("perturbation", "nunique"),
                n_seeds=("fold_id", "nunique"),
                mean_rmse=("true_error_rmse", "mean"),
                median_rmse=("true_error_rmse", "median"),
                mean_cosine_error=("true_error_cosine", "mean"),
            )
            .reset_index()
        )
    else:
        dataset_summary = pd.DataFrame()
    eval_path = GEARS_EVAL / "tables" / "GEARS_CONFIDENCE_EVAL_SUMMARY.csv"
    score_summary = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    uncertainty_records = GEARS_UNCERT / "tables" / "GEARS_RECORDS_FOR_UNCERTAINTY.csv"
    uncertainty_scores = GEARS_UNCERT / "tables" / "GEARS_UNCERTAINTY_SCORES.csv"
    uncertainty_status_path = GEARS_UNCERT / "GEARS_UNCERTAINTY_STATUS.json"
    uncertainty_status = json.loads(uncertainty_status_path.read_text()) if uncertainty_status_path.exists() else {}
    uncertainty_table = pd.DataFrame(
        [
            {
                "asset": "GEARS_RECORDS_FOR_UNCERTAINTY.csv",
                "exists": uncertainty_records.exists(),
                "n_rows": len(pd.read_csv(uncertainty_records)) if uncertainty_records.exists() else 0,
            },
            {
                "asset": "GEARS_UNCERTAINTY_SCORES.csv",
                "exists": uncertainty_scores.exists(),
                "n_rows": len(pd.read_csv(uncertainty_scores)) if uncertainty_scores.exists() else 0,
            },
            {
                "asset": "native_uncertainty",
                "exists": bool(uncertainty_status.get("has_native_uncertainty", False)),
                "n_rows": int(uncertainty_status.get("n_native_scores", 0) or 0),
            },
            {
                "asset": "seed_ensemble_proxy",
                "exists": bool(uncertainty_status.get("has_seed_ensemble_proxy", False)),
                "n_rows": int(uncertainty_status.get("n_proxy_scores", 0) or 0),
            },
        ]
    )
    return {
        "run_audit": run_audit,
        "records": records,
        "dataset_summary": dataset_summary,
        "score_summary": score_summary,
        "uncertainty_table": uncertainty_table,
        "uncertainty_status": uncertainty_status,
    }


def build_model_summary(gears: dict[str, Any]) -> pd.DataFrame:
    run_audit = gears["run_audit"]
    records = gears["records"]
    score_summary = gears["score_summary"]
    uncertainty_status = gears["uncertainty_status"]
    n_records = len(records)
    n_runs = len(run_audit)
    n_datasets = records["dataset_name"].nunique() if n_records else 0
    gene_dims = sorted(set(run_audit["predicted_npz_first_shape"].dropna().astype(str))) if not run_audit.empty else []
    gears_ready = (
        n_records > 0
        and bool(run_audit["predicted_npz_repaired_exists"].all())
        and bool(run_audit["true_npz_repaired_exists"].all())
    )
    rows = [
        {
            "model": "GEARS",
            "local_code_or_data": GEARS_DATA_ROOT.exists(),
            "prediction_records": n_records,
            "runs_or_seeds": n_runs,
            "datasets": n_datasets,
            "gene_space": ",".join(gene_dims),
            "predicted_vectors": bool(gears_ready),
            "true_vectors": bool(gears_ready),
            "native_uncertainty": bool(uncertainty_status.get("has_native_uncertainty", False)),
            "proxy_uncertainty": bool(uncertainty_status.get("has_seed_ensemble_proxy", False)),
            "score_rows": len(score_summary),
            "readiness": "PARTIAL_READY_GEARS_ONLY" if gears_ready else "NOT_READY",
            "blocking_reason": (
                "Records/vectors exist, but only for GEARS on Norman/Adamson/Dixit single-gene tasks; "
                "not aligned to sciplex3 full-743, CPA or scGPT; native uncertainty absent."
            ),
            "next_action": "Use as GEARS-only supplement or rerun/align GEARS with the same task_id/gene order as current benchmarks.",
        },
        {
            "model": "scGPT",
            "local_code_or_data": Path("/home/yyf/.conda/envs/scgpt_env").exists()
            or Path("/home/yyf/archive/code/20260519_0958_home_cleanup/moved_top_level/scGPT-main.zip").exists(),
            "prediction_records": 0,
            "runs_or_seeds": 0,
            "datasets": 0,
            "gene_space": "",
            "predicted_vectors": False,
            "true_vectors": False,
            "native_uncertainty": False,
            "proxy_uncertainty": False,
            "score_rows": 0,
            "readiness": "NOT_READY_NO_PREDICTION_RECORDS",
            "blocking_reason": "Local environment/archive exists, but no SafeConf PredictionRecord + predicted_effect vectors were found.",
            "next_action": "Implement or locate a scGPT adapter that exports PREDICTION_RECORDS.csv plus predicted/true NPZ on a frozen benchmark.",
        },
        {
            "model": "CPA / chemCPA",
            "local_code_or_data": False,
            "prediction_records": 0,
            "runs_or_seeds": 0,
            "datasets": 0,
            "gene_space": "",
            "predicted_vectors": False,
            "true_vectors": False,
            "native_uncertainty": False,
            "proxy_uncertainty": False,
            "score_rows": 0,
            "readiness": "NOT_READY_NO_LOCAL_VECTOR_OUTPUT",
            "blocking_reason": "Only literature/PDF traces were found; no local CPA vector outputs under the SafeConf contract.",
            "next_action": "Treat CPA as future external adapter work; do not claim completed CPA validation.",
        },
    ]
    return pd.DataFrame(rows)


def write_svg_readiness(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "PARTIAL_READY_GEARS_ONLY": "#2f6f5e",
        "NOT_READY_NO_PREDICTION_RECORDS": "#b26b00",
        "NOT_READY_NO_LOCAL_VECTOR_OUTPUT": "#9b4545",
        "NOT_READY": "#9b4545",
    }
    rows = []
    y = 50
    for _, r in summary.iterrows():
        val = min(float(r["prediction_records"]), 60.0)
        width = max(8, val / 60 * 420)
        color = colors.get(str(r["readiness"]), "#6b7280")
        rows.append(f'<text x="35" y="{y+17}" font-size="16" font-weight="700">{html.escape(str(r["model"]))}</text>')
        rows.append(f'<rect x="170" y="{y}" width="420" height="25" rx="6" fill="#edf1ee"/>')
        rows.append(f'<rect x="170" y="{y}" width="{width:.1f}" height="25" rx="6" fill="{color}"/>')
        rows.append(f'<text x="610" y="{y+18}" font-size="14">{int(r["prediction_records"])} records · {html.escape(str(r["readiness"]))}</text>')
        y += 55
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="250" viewBox="0 0 900 250">
<rect width="900" height="250" fill="#fbfbf8"/>
<text x="35" y="28" font-size="20" font-weight="800" fill="#17201c">Real-model vector readiness audit</text>
{''.join(rows)}
<text x="35" y="230" font-size="13" fill="#66736d">Record count is capped at 60 for display; readiness depends on SafeConf PredictionRecord + predicted/true vector contract.</text>
</svg>
"""
    (FIGURES / "model_vector_readiness.svg").write_text(svg, encoding="utf-8")


def write_reports(
    model_summary: pd.DataFrame,
    run_audit: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    uncertainty_table: pd.DataFrame,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E18 真实模型预测向量资产审计

生成时间：{now}

## 1. 结论

E18 回答一个很具体的问题：本地是否已有 GEARS、CPA、scGPT 这类真实模型的逐任务预测向量，足以支撑“模型级 SafeConf 验证”。

结论：GEARS 有部分可用资产，scGPT 和 CPA 当前没有可直接进入 SafeConf 协议的 PredictionRecord + predicted/true vector。GEARS 的资产可读，但只覆盖 Norman、Adamson、Dixit 的 single-gene 任务，合计 54 条记录；它不能直接替代 sciplex3 full-743 的多模型验证。

## 2. 模型级就绪度

{md_table(model_summary)}

## 3. GEARS 运行与数组审计

{md_table(run_audit, [
    'dataset','seed','status','records_n_rows',
    'records_csv_reported_exists','records_csv_repaired_exists',
    'predicted_npz_repaired_exists','predicted_npz_n_keys','predicted_npz_first_shape','predicted_keys_match_records',
    'true_npz_repaired_exists','true_npz_n_keys','true_keys_match_records'
])}

## 4. GEARS 数据集规模

{md_table(dataset_summary)}

## 5. 已有 GEARS 风险分数

{md_table(score_summary)}

## 6. GEARS 不确定性

{md_table(uncertainty_table)}

## 7. 可以说与不能说

可以说：

- GEARS 的 PredictionRecord 与 predicted/true NPZ 在 `safeconf_runtime/outputs/gears_prediction_records_formal` 下可读。
- 旧 status 文件中的 `/home/yyf/codex_cout/...` 绝对路径已失效，但能映射到 `/home/yyf/safeconf_runtime/outputs/...`。
- GEARS-only 的 magnitude risk 在 54 条记录上 aligned Spearman = 0.624。

不能说：

- 不能说 GEARS、CPA、scGPT 已在同一 benchmark、同一 split、同一 gene space 下完成模型级验证。
- 不能说 sciplex3 full-743 已经接入 GEARS/CPA/scGPT。
- 不能把 GEARS 的 perturbation-level 或 seed-level 汇总指标当作逐任务置信度。

## 8. 下一步

1. 把 E18 作为模型级扩展的入口审计。
2. 若短期需要实证结果，先做 GEARS-only supplement：只在 Norman/Adamson/Dixit 54 条记录上报告。
3. 若目标是主线升级，必须重新定义一个 shared benchmark，把 GEARS、CPA、scGPT 都导出为统一的 `task_id + gene_order + predicted_effect_key + true_effect_key`。
"""
    (REPORTS / "E18_MODEL_VECTOR_ASSET_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E18 model vector asset audit</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17201c;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.7}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 28px 70px}}h1{{font-size:34px;margin:0}}h2{{border-bottom:2px solid #dfe7e2;padding-bottom:8px;margin-top:30px}}
.lead{{color:#66736d;max-width:920px}}.card,figure{{background:white;border:1px solid #dfe7e2;border-radius:15px;padding:18px;margin:16px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #dfe7e2;padding:7px 8px;vertical-align:top}}th{{background:#f1f5f2;text-align:left}}
img{{max-width:100%}}.warn{{border-left:5px solid #b26b00}}.good{{border-left:5px solid #2f6f5e}}
</style></head><body><main class="wrap">
<h1>E18 真实模型预测向量资产审计</h1>
<p class="lead">检查 GEARS、scGPT、CPA 是否已有能进入 SafeConf 协议的逐任务 predicted/true vector。重点不是模型名，而是能不能对齐 task_id、gene_order 和预测向量。</p>
<figure><img src="../figures/model_vector_readiness.svg" alt="model vector readiness"></figure>
<section class="card good"><h2>1. 模型级就绪度</h2>{html_table(model_summary)}</section>
<section class="card"><h2>2. GEARS 运行与数组审计</h2>{html_table(run_audit, ['dataset','seed','status','records_n_rows','records_csv_repaired_exists','predicted_npz_repaired_exists','predicted_npz_n_keys','predicted_npz_first_shape','predicted_keys_match_records','true_npz_repaired_exists','true_npz_n_keys','true_keys_match_records'])}</section>
<section class="card"><h2>3. GEARS 数据集规模</h2>{html_table(dataset_summary)}</section>
<section class="card"><h2>4. 已有 GEARS 风险分数</h2>{html_table(score_summary)}</section>
<section class="card warn"><h2>5. 边界</h2><p>GEARS 有可读向量，但它不是 sciplex3 full-743 同任务空间；scGPT/CPA 当前没有 PredictionRecord + predicted/true NPZ。不能声称真实多模型验证已经完成。</p></section>
</main></body></html>
"""
    (REPORTS / "E18_MODEL_VECTOR_ASSET_AUDIT.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    for p in (TABLES, REPORTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
    gears = audit_gears()
    model_summary = build_model_summary(gears)
    run_audit = gears["run_audit"]
    records = gears["records"]
    dataset_summary = gears["dataset_summary"]
    score_summary = gears["score_summary"]
    uncertainty_table = gears["uncertainty_table"]

    model_summary.to_csv(TABLES / "MODEL_VECTOR_ASSET_SUMMARY.csv", index=False)
    run_audit.to_csv(TABLES / "GEARS_RUN_VECTOR_AUDIT.csv", index=False)
    records.to_csv(TABLES / "GEARS_CANONICAL_PREDICTION_RECORDS.csv", index=False)
    dataset_summary.to_csv(TABLES / "GEARS_DATASET_SUMMARY.csv", index=False)
    score_summary.to_csv(TABLES / "GEARS_CONFIDENCE_EVAL_SUMMARY.csv", index=False)
    uncertainty_table.to_csv(TABLES / "GEARS_UNCERTAINTY_ASSET_SUMMARY.csv", index=False)
    write_svg_readiness(model_summary)
    write_reports(model_summary, run_audit, dataset_summary, score_summary, uncertainty_table)

    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "input_git_commit": git_head(),
        "input_git_dirty": git_dirty(),
        "gears_records": int(len(records)),
        "gears_runs": int(len(run_audit)),
        "gears_datasets": int(records["dataset_name"].nunique()) if len(records) else 0,
        "gears_ready": bool(model_summary.loc[model_summary["model"].eq("GEARS"), "predicted_vectors"].iloc[0]),
        "scgpt_ready": bool(model_summary.loc[model_summary["model"].eq("scGPT"), "predicted_vectors"].iloc[0]),
        "cpa_ready": bool(model_summary.loc[model_summary["model"].eq("CPA / chemCPA"), "predicted_vectors"].iloc[0]),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E18 model vector asset audit\n\n"
        "先看：\n\n"
        "- `reports/E18_MODEL_VECTOR_ASSET_AUDIT.html`\n"
        "- `reports/E18_MODEL_VECTOR_ASSET_AUDIT_REPORT.md`\n\n"
        "这个结果包审计 GEARS、scGPT、CPA 是否已有可进入 SafeConf 协议的逐任务预测向量。\n",
        encoding="utf-8",
    )
    print(f"Wrote E18 audit to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
