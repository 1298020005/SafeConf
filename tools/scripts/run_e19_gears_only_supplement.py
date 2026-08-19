#!/usr/bin/env python3
"""E19 package: consolidate existing GEARS-only supplementary evaluations."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "实验结果" / "E19_gears_only_supplement_20260707"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

SOURCES = [
    {
        "source_id": "GEARS_FORMAL_54",
        "label": "Norman/Adamson/Dixit formal",
        "root": Path("/home/yyf/safeconf_runtime/outputs/gears_confidence_eval_formal"),
        "status": "formal_existing",
        "boundary": "GEARS-only; no native uncertainty; not aligned with sciplex3/CPA/scGPT.",
    },
    {
        "source_id": "GEARS_FRANGIEH_62",
        "label": "Frangieh third-predictor run03",
        "root": Path("/home/yyf/safeconf_runtime/outputs/run03_gears_third_predictor_eval_20260607"),
        "status": "supplement_probe_existing",
        "boundary": "GEARS-only Frangieh run; source name contains smoke lineage; use as supplementary signal, not main formal claim.",
    },
]


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


def read_table(source: dict[str, Any], name: str) -> pd.DataFrame:
    path = source["root"] / "tables" / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "source_id", source["source_id"])
    df.insert(1, "source_label", source["label"])
    df.insert(2, "source_status", source["status"])
    return df


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


def collect() -> dict[str, pd.DataFrame]:
    all_records = []
    all_scores = []
    all_eval = []
    all_risk_cov = []
    source_rows = []
    for src in SOURCES:
        records = read_table(src, "GEARS_PREDICTION_RECORDS_COMBINED.csv")
        scores = read_table(src, "GEARS_CONFIDENCE_SCORES.csv")
        eval_df = read_table(src, "GEARS_CONFIDENCE_EVAL_SUMMARY.csv")
        risk_cov = read_table(src, "GEARS_RISK_COVERAGE.csv")
        if not records.empty:
            all_records.append(records)
        if not scores.empty:
            all_scores.append(scores)
        if not eval_df.empty:
            all_eval.append(eval_df)
        if not risk_cov.empty:
            all_risk_cov.append(risk_cov)
        overall = eval_df[eval_df["level"].eq("overall")].copy() if not eval_df.empty else pd.DataFrame()
        best = overall.sort_values("direction_aligned_spearman", ascending=False).head(1)
        source_rows.append(
            {
                "source_id": src["source_id"],
                "source_label": src["label"],
                "source_status": src["status"],
                "exists": src["root"].exists(),
                "n_records": len(records),
                "n_scores": len(scores),
                "n_datasets": records["dataset_name"].nunique() if not records.empty else 0,
                "datasets": ",".join(sorted(records["dataset_name"].dropna().astype(str).unique())) if not records.empty else "",
                "best_score_name": best["score_name"].iloc[0] if len(best) else "",
                "best_aligned_spearman": float(best["direction_aligned_spearman"].iloc[0]) if len(best) else float("nan"),
                "best_n": int(best["n"].iloc[0]) if len(best) else 0,
                "has_uncertainty_confidence": bool((scores["score_name"].eq("gears_uncertainty_confidence")).any()) if not scores.empty else False,
                "boundary": src["boundary"],
            }
        )
    records = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    eval_df = pd.concat(all_eval, ignore_index=True) if all_eval else pd.DataFrame()
    risk_cov = pd.concat(all_risk_cov, ignore_index=True) if all_risk_cov else pd.DataFrame()
    source_summary = pd.DataFrame(source_rows)
    if not records.empty:
        dataset_summary = (
            records.groupby(["source_id", "source_label", "dataset_name"], dropna=False)
            .agg(
                n_records=("record_id", "count"),
                n_unique_perturbations=("perturbation", "nunique"),
                n_seeds=("fold_id", "nunique"),
                mean_rmse=("true_error_rmse", "mean"),
                median_rmse=("true_error_rmse", "median"),
            )
            .reset_index()
        )
    else:
        dataset_summary = pd.DataFrame()
    return {
        "source_summary": source_summary,
        "records": records,
        "scores": scores,
        "eval": eval_df,
        "risk_cov": risk_cov,
        "dataset_summary": dataset_summary,
    }


def write_svg(source_summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    y = 62
    for _, r in source_summary.iterrows():
        width = min(500.0, float(r["best_aligned_spearman"]) * 500.0) if pd.notna(r["best_aligned_spearman"]) else 0
        rows.append(f'<text x="35" y="{y+17}" font-size="15" font-weight="700">{html.escape(str(r["source_id"]))}</text>')
        rows.append(f'<rect x="250" y="{y}" width="500" height="25" rx="6" fill="#edf1ee"/>')
        rows.append(f'<rect x="250" y="{y}" width="{width:.1f}" height="25" rx="6" fill="#2f6f5e"/>')
        rows.append(f'<text x="765" y="{y+18}" font-size="14">ρ={float(r["best_aligned_spearman"]):.3f}, n={int(r["n_records"])}</text>')
        y += 60
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="230" viewBox="0 0 980 230">
<rect width="980" height="230" fill="#fbfbf8"/>
<text x="35" y="30" font-size="20" font-weight="800" fill="#17201c">GEARS-only supplementary signal</text>
{''.join(rows)}
<text x="35" y="205" font-size="13" fill="#66736d">Bars show best overall direction-aligned Spearman in existing GEARS-only outputs. This is not unified GEARS/CPA/scGPT validation.</text>
</svg>
"""
    (FIGURES / "gears_only_supplement_signal.svg").write_text(svg, encoding="utf-8")


def write_reports(tables: dict[str, pd.DataFrame]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    source_summary = tables["source_summary"]
    eval_df = tables["eval"]
    dataset_summary = tables["dataset_summary"]
    risk_cov = tables["risk_cov"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# E19 GEARS-only supplement consolidation

生成时间：{now}

## 1. 结论

E19 汇总本地已有 GEARS-only 结果，目标是形成一个可引用的补充证据包。它不等价于 GEARS、CPA、scGPT 的统一多模型验证。

核心结果：

{md_table(source_summary)}

## 2. Dataset-level summary

{md_table(dataset_summary)}

## 3. Evaluation summary

{md_table(eval_df)}

## 4. Risk coverage

{md_table(risk_cov)}

## 5. 边界

- `GEARS_FORMAL_54` 是 Norman/Adamson/Dixit 的正式旧结果，整体 magnitude risk aligned Spearman = 0.624。
- `GEARS_FRANGIEH_62` 是 Frangieh third-predictor run03，包含 uncertainty confidence，但来源目录带 smoke lineage；可以作为补充探针，不能单独变成主线 formal claim。
- 两者都不是 sciplex3 full-743，也没有 CPA/scGPT 的同任务向量。

## 6. 下一步

1. 若写补充材料，可用 E19 说明 GEARS-only 预测输出可被 SafeConf 风格审计。
2. 若写主线结论，必须另建 unified adapter：同一任务、同一 gene order、同一 split 下同时导出 GEARS、CPA、scGPT。
"""
    (REPORTS / "E19_GEARS_ONLY_SUPPLEMENT_REPORT.md").write_text(report, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>E19 GEARS-only supplement</title>
<style>
body{{margin:0;background:#fbfbf8;color:#17201c;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.7}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 28px 70px}}h1{{font-size:34px;margin:0}}h2{{border-bottom:2px solid #dfe7e2;padding-bottom:8px;margin-top:30px}}
.lead{{color:#66736d;max-width:940px}}.card,figure{{background:white;border:1px solid #dfe7e2;border-radius:15px;padding:18px;margin:16px 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #dfe7e2;padding:7px 8px;vertical-align:top}}th{{background:#f1f5f2;text-align:left}}
img{{max-width:100%}}.warn{{border-left:5px solid #b26b00}}.good{{border-left:5px solid #2f6f5e}}
</style></head><body><main class="wrap">
<h1>E19 GEARS-only supplement</h1>
<p class="lead">整理本地已有 GEARS-only 结果。它可以作为补充证据，不能写成 GEARS/CPA/scGPT 已完成统一模型级验证。</p>
<figure><img src="../figures/gears_only_supplement_signal.svg" alt="GEARS-only signal"></figure>
<section class="card good"><h2>1. Source summary</h2>{html_table(source_summary)}</section>
<section class="card"><h2>2. Dataset summary</h2>{html_table(dataset_summary)}</section>
<section class="card"><h2>3. Evaluation summary</h2>{html_table(eval_df)}</section>
<section class="card warn"><h2>4. Boundary</h2><p>GEARS-only supplement does not provide CPA/scGPT vectors and does not align to sciplex3 full-743.</p></section>
</main></body></html>
"""
    (REPORTS / "E19_GEARS_ONLY_SUPPLEMENT.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    for p in (TABLES, REPORTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
    tables = collect()
    tables["source_summary"].to_csv(TABLES / "GEARS_SUPPLEMENT_SOURCE_SUMMARY.csv", index=False)
    tables["records"].to_csv(TABLES / "GEARS_SUPPLEMENT_PREDICTION_RECORDS.csv", index=False)
    tables["scores"].to_csv(TABLES / "GEARS_SUPPLEMENT_CONFIDENCE_SCORES.csv", index=False)
    tables["eval"].to_csv(TABLES / "GEARS_SUPPLEMENT_EVAL_SUMMARY.csv", index=False)
    tables["risk_cov"].to_csv(TABLES / "GEARS_SUPPLEMENT_RISK_COVERAGE.csv", index=False)
    tables["dataset_summary"].to_csv(TABLES / "GEARS_SUPPLEMENT_DATASET_SUMMARY.csv", index=False)
    write_svg(tables["source_summary"])
    write_reports(tables)
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
        "input_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE
        ).stdout.strip(),
        "n_sources": int(len(tables["source_summary"])),
        "total_records": int(len(tables["records"])),
        "best_source": str(tables["source_summary"].sort_values("best_aligned_spearman", ascending=False)["source_id"].iloc[0]),
        "best_aligned_spearman": float(tables["source_summary"]["best_aligned_spearman"].max()),
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README_先看这个.md").write_text(
        "# E19 GEARS-only supplement\n\n"
        "先看：\n\n"
        "- `reports/E19_GEARS_ONLY_SUPPLEMENT.html`\n"
        "- `reports/E19_GEARS_ONLY_SUPPLEMENT_REPORT.md`\n\n"
        "这个结果包只整理 GEARS-only 既有结果，不代表 CPA/scGPT 已完成统一验证。\n",
        encoding="utf-8",
    )
    print(f"Wrote E19 package to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
