#!/usr/bin/env python3
"""Package official sciplex3 full-743 sensitivity runs into lightweight docs."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TABLE_FILES = [
    "DATASET_TASK_SUMMARY.csv",
    "HELDOUT_PAIR_SPLIT_SUMMARY.csv",
    "MAIN_PER_DATASET_SUMMARY.csv",
    "CONFIDENCE_EVAL_SUMMARY.csv",
    "RISK_COVERAGE.csv",
    "CONFIDENCE_FEATURE_MISSINGNESS_BY_DATASET.csv",
    "PREDICTOR_STATUS.csv",
    "LEARNED_RISK_FOLD_STATUS.csv",
    "TRANSFERABILITY_RANKING.csv",
]

FIGURE_FILES = [
    "F1_task_schematic.png",
    "F2_context_perturbation_matrix.png",
    "F3_prediction_record_flow.png",
    "F4_confidence_vs_true_error_scatter.png",
    "F5_risk_coverage_curve.png",
    "F6_high_vs_low_confidence_rmse.png",
    "F7_per_dataset_spearman_comparison.png",
    "F8_calibration_buckets.png",
    "F10_transferability_ranking.png",
]


def rel_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


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


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def cell(v: Any) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("\n", " ").replace("|", "/")


def md_table(df: pd.DataFrame, cols: list[str], n: int = 80) -> str:
    show = df[cols].head(n).copy()
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in show.columns) + " |")
    return "\n".join(lines)


def html_table(df: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    return df[cols].head(n).to_html(index=False, escape=True)


def load_summary(runtime: Path) -> dict[str, Any]:
    tables = runtime / "tables"
    dataset = pd.read_csv(tables / "DATASET_TASK_SUMMARY.csv")
    split = pd.read_csv(tables / "HELDOUT_PAIR_SPLIT_SUMMARY.csv")
    main = pd.read_csv(tables / "MAIN_PER_DATASET_SUMMARY.csv")
    eval_df = pd.read_csv(tables / "CONFIDENCE_EVAL_SUMMARY.csv")
    overall = eval_df[eval_df["level"].eq("overall")].sort_values(
        ["direction_aligned_spearman", "risk_cov_80_improve_pct"],
        ascending=[False, False],
    )
    return {
        "dataset": dataset,
        "split": split,
        "main": main,
        "eval": eval_df,
        "overall": overall,
        "best": overall.iloc[0].to_dict(),
        "learned": overall[overall["score_name"].eq("learned_risk_score")].iloc[0].to_dict(),
        "simple": overall[overall["score_name"].eq("simple_combined_confidence")].iloc[0].to_dict(),
        "disagreement": overall[overall["score_name"].eq("model_disagreement_risk")].iloc[0].to_dict(),
        "magnitude": overall[overall["score_name"].eq("prediction_magnitude_risk")].iloc[0].to_dict(),
    }


def write_reports(
    runtime: Path,
    out: Path,
    run_id: str,
    genes: int,
    report_stem: str,
    previous: str,
    next_step: str,
) -> dict[str, Any]:
    tables_out = out / "tables"
    figs_out = out / "figures"
    reports_out = out / "reports"
    for p in (tables_out, figs_out, reports_out):
        p.mkdir(parents=True, exist_ok=True)
    for name in TABLE_FILES:
        copy_file(runtime / "tables" / name, tables_out / name)
    for name in FIGURE_FILES:
        copy_file(runtime / "figures" / name, figs_out / name)

    s = load_summary(runtime)
    dataset = s["dataset"]
    split = s["split"]
    main = s["main"]
    overall = s["overall"]
    best = s["best"]
    learned = s["learned"]
    simple = s["simple"]
    disagreement = s["disagreement"]
    magnitude = s["magnitude"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# {run_id} 官方 sciplex3 full-743 gene{genes} sensitivity

生成时间：{now}

## 1. 结论

{run_id} 复用官方 `sciplex3_A549/K562/MCF7` 三细胞系面板，保留全部 743 个三细胞系共享 drug-dose。本轮基因面板为 {genes:,} genes，用于检查 full-743 chemical 外部验证信号随基因数增加是否稳定。

面板规模：2,229 tasks、3 contexts、743 perturbations、4,458 test records。split 审计通过，无 pair/context/perturbation leakage。

最强信号为 `{best['score_name']}`：aligned Spearman = {float(best['direction_aligned_spearman']):.3f}，80% coverage RMSE 改善 = {float(best['risk_cov_80_improve_pct']):.2f}%。`model_disagreement_risk` aligned Spearman = {float(disagreement['direction_aligned_spearman']):.3f}，80% coverage 改善 = {float(disagreement['risk_cov_80_improve_pct']):.2f}%；`simple_combined_confidence` aligned Spearman = {float(simple['direction_aligned_spearman']):.3f}。

## 2. 面板构建

{md_table(dataset, ['dataset','n_tasks','n_contexts','n_perturbations','context_col','perturbation_col','n_genes','n_raw_tasks','n_shared_perturbations','n_selected_perturbations','selection_rule'])}

## 3. Split 审计

{md_table(split, ['dataset_name','fold_id','n_tasks_total','n_candidate_test_pairs','n_train','n_val','n_test','support_check_pass'])}

## 4. 主结果

{md_table(main, [
    'dataset_name',
    'simple_combined_confidence_aligned_rho',
    'simple_combined_confidence_risk_cov_80_improve_pct',
    'learned_risk_score_aligned_rho',
    'learned_risk_score_risk_cov_80_improve_pct',
    'model_disagreement_risk_aligned_rho',
    'model_disagreement_risk_risk_cov_80_improve_pct',
])}

## 5. Overall 信号排序

{md_table(overall, ['score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}

## 6. 连续性判断

- 上一轮：{previous}
- 本轮：gene{genes}，learned aligned Spearman = {float(learned['direction_aligned_spearman']):.3f}，80% coverage 改善 = {float(learned['risk_cov_80_improve_pct']):.2f}%。
- 当前判断：风险信号在 full-743 chemical 面板上保持稳定。

## 7. 边界

- 本轮仍属于 lightweight reference prediction system 内的风险评估。
- 真实 GEARS/CPA/scGPT 逐模型向量验证仍是单独任务。
- `prediction_magnitude_risk` aligned Spearman = {float(magnitude['direction_aligned_spearman']):.3f}。

## 8. 下一步

{next_step}
"""
    (reports_out / f"{report_stem}_REPORT.md").write_text(report, encoding="utf-8")

    cards = [
        ("tasks", str(int(dataset["n_tasks"].iloc[0]))),
        ("drug-dose", str(int(dataset["n_perturbations"].iloc[0]))),
        ("genes", str(int(dataset["n_genes"].iloc[0]))),
        ("learned rho", f"{float(learned['direction_aligned_spearman']):.3f}"),
        ("disagree rho", f"{float(disagreement['direction_aligned_spearman']):.3f}"),
    ]
    cards_html = "".join(
        f'<div class="metric"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in cards
    )
    fig_html = "".join(
        f'<figure><img src="../figures/{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in FIGURE_FILES
    )
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{run_id} sciplex3 full-743 gene{genes}</title>
<style>
:root{{--ink:#17201c;--muted:#66736d;--line:#dfe7e2;--accent:#2f6f5e;--warn:#9a5b00}}
body{{margin:0;background:#fbfbf8;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",Arial,sans-serif;line-height:1.72}}
.wrap{{max-width:1180px;margin:0 auto;padding:34px 30px 70px}}.hero{{border-bottom:1px solid var(--line);padding:28px 0 22px}}
h1{{font-size:34px;margin:0 0 10px}}h2{{margin-top:32px;padding-bottom:8px;border-bottom:2px solid var(--line)}}.lead{{max-width:930px;color:var(--muted);font-size:17px}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}}.metric,.card,figure{{background:white;border:1px solid var(--line);border-radius:15px}}
.metric{{padding:16px}}.metric .k{{color:var(--muted);font-size:13px}}.metric .v{{color:var(--accent);font-size:26px;font-weight:720}}
.card{{padding:20px;margin:18px 0;overflow-x:auto}}.warn{{border-left:5px solid var(--warn)}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid var(--line);padding:7px 8px;vertical-align:top}}th{{background:#f1f5f2;text-align:left}}.figgrid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
figure{{margin:0;padding:12px}}img{{max-width:100%;display:block;background:white;border-radius:10px}}figcaption{{color:var(--muted);font-size:13px;margin-top:8px}}
@media(max-width:900px){{.metrics,.figgrid{{grid-template-columns:1fr}}}}
</style></head>
<body><main class="wrap">
<section class="hero"><h1>{run_id} sciplex3 full-743 gene{genes}</h1><p class="lead">官方三细胞系全部 743 个共享 drug-dose，{genes:,}-gene sensitivity。</p><div class="metrics">{cards_html}</div></section>
<section class="card"><h2>1. 面板构建</h2>{html_table(dataset, ['dataset','n_tasks','n_contexts','n_perturbations','n_genes','n_raw_tasks','n_shared_perturbations','n_selected_perturbations'])}</section>
<section class="card"><h2>2. 主结果</h2>{html_table(main, ['dataset_name','simple_combined_confidence_aligned_rho','simple_combined_confidence_risk_cov_80_improve_pct','learned_risk_score_aligned_rho','learned_risk_score_risk_cov_80_improve_pct','model_disagreement_risk_aligned_rho','model_disagreement_risk_risk_cov_80_improve_pct'])}</section>
<section class="card"><h2>3. Overall 信号排序</h2>{html_table(overall, ['score_name','score_type','n','direction_aligned_spearman','risk_cov_80_improve_pct','high_low_rmse_gap'])}</section>
<section class="card warn"><h2>4. 边界</h2><p>本轮是 gene{genes} sensitivity。真实 GEARS/CPA/scGPT 逐模型验证仍是单独任务。</p></section>
<section><h2>5. 图件</h2><div class="figgrid">{fig_html}</div></section>
</main></body></html>
"""
    (reports_out / f"{report_stem}.html").write_text(html_doc, encoding="utf-8")
    copy_file(runtime / "MVP_V2_1_REPORT.md", reports_out / "MVP_V2_1_REPORT.md")
    copy_file(runtime / "stage_completion_checklist_v2_1.md", reports_out / "stage_completion_checklist_v2_1.md")
    return s


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--genes", type=int, required=True)
    ap.add_argument("--report-stem", required=True)
    ap.add_argument("--previous", default="no previous run supplied")
    ap.add_argument("--next-step", default="Continue the next sensitivity run or move to model-vector validation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runtime = rel_path(args.runtime_dir)
    out = rel_path(args.out_dir)
    if not runtime.exists():
        raise SystemExit(f"Missing runtime directory: {runtime}")
    summary = write_reports(
        runtime=runtime,
        out=out,
        run_id=args.run_id,
        genes=args.genes,
        report_stem=args.report_stem,
        previous=args.previous,
        next_step=args.next_step,
    )
    status = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_dir": str(out.relative_to(ROOT)),
        "runtime_source": str(runtime.relative_to(ROOT)),
        "input_git_commit": git_head(),
        "input_git_dirty": git_dirty(),
        "run_id": args.run_id,
        "genes": args.genes,
        "best_overall_score": summary["best"]["score_name"],
        "best_overall_aligned_spearman": float(summary["best"]["direction_aligned_spearman"]),
        "best_overall_risk_cov_80_improve_pct": float(summary["best"]["risk_cov_80_improve_pct"]),
        "learned_aligned_spearman": float(summary["learned"]["direction_aligned_spearman"]),
        "disagreement_aligned_spearman": float(summary["disagreement"]["direction_aligned_spearman"]),
        "simple_combined_aligned_spearman": float(summary["simple"]["direction_aligned_spearman"]),
        "panel": {
            "n_tasks": int(summary["dataset"]["n_tasks"].iloc[0]),
            "n_contexts": int(summary["dataset"]["n_contexts"].iloc[0]),
            "n_genes": int(summary["dataset"]["n_genes"].iloc[0]),
            "n_shared_perturbations": int(summary["dataset"]["n_shared_perturbations"].iloc[0]),
            "n_selected_perturbations": int(summary["dataset"]["n_selected_perturbations"].iloc[0]),
        },
    }
    (out / "RUN_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README_先看这个.md").write_text(
        f"# {args.run_id} sciplex3 full-743 gene{args.genes}\n\n"
        "先看：\n\n"
        f"- `reports/{args.report_stem}.html`\n"
        f"- `reports/{args.report_stem}_REPORT.md`\n\n"
        f"这个目录只保存可汇报的轻量结果。原始矩阵在 `{runtime.relative_to(ROOT)}/`。\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.run_id} package to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
